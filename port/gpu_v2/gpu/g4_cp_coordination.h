// SPDX-License-Identifier: GPL-2.0-or-later
// Admission-side G4 progress and seal coordination. All methods are called under the same
// short admission mutex. No method records Vulkan commands, waits, or owns a guest pointer.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "video_core/amdgpu/g4_stage1_ledger.h"

namespace iss::g4 {

enum class CpLane : std::uint8_t { Graphics, Asc };

struct CpTick {
    CpLane lane{CpLane::Graphics};
    std::uint64_t tick{};
    friend bool operator==(const CpTick&, const CpTick&) = default;
};

struct QidTick {
    CpLane lane{CpLane::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    std::uint64_t tick{};
    friend bool operator==(const QidTick&, const QidTick&) = default;
};

enum class SealState : std::uint8_t { Pending, Done, Failed, Cancelled };

struct SealRequest {
    explicit SealRequest(QidTick target_) : target{target_} {}
    QidTick target{};
    // A sealed token proves submission, not GPU completion.
    [[nodiscard]] SealState State() const noexcept {
        return state.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t SealedTick() const noexcept {
        return sealed_tick.load(std::memory_order_relaxed);
    }
    void Resolve(SealState result, std::uint64_t tick = 0) noexcept {
        sealed_tick.store(tick, std::memory_order_relaxed);
        state.store(result, std::memory_order_release);
    }

private:
    std::atomic<SealState> state{SealState::Pending};
    std::atomic<std::uint64_t> sealed_tick{};
};

template <std::size_t NumQids, std::size_t MaxFrameMarks = 16>
class CpCoordination {
    static_assert(NumQids > 1 && MaxFrameMarks > 0);

public:
    using Ledger = AmdGpu::G4Stage1Ledger<NumQids>;
    using SubmissionToken = typename Ledger::Token;
    using FrameSnapshot = typename Ledger::MarkSnapshot;
    using Request = std::shared_ptr<SealRequest>;

    struct SealPlan {
        CpLane lane{};
        std::uint64_t through_tick{};
        friend bool operator==(const SealPlan&, const SealPlan&) = default;
    };

    // The caller serializes these with queue admission, completion, and unmap.
    bool Map(std::uint32_t qid, std::uint64_t& generation) noexcept {
        if (!ledger.Map(qid, generation)) return false;
        ResolveStaleRequests();
        PruneDependencies();
        return true;
    }
    bool BeginClose(std::uint32_t qid, std::uint64_t& generation) noexcept {
        return ledger.BeginClose(qid, generation);
    }
    bool Drained(std::uint32_t qid, std::uint64_t generation) const noexcept {
        return ledger.Drained(qid, generation);
    }
    bool FinishClose(std::uint32_t qid, std::uint64_t generation) noexcept {
        if (!ledger.FinishClose(qid, generation)) return false;
        ResolveStaleRequests();
        PruneDependencies();
        return true;
    }
    bool Issue(std::uint32_t qid, SubmissionToken& token) noexcept {
        return ledger.Issue(qid, token);
    }
    bool Complete(SubmissionToken token) noexcept { return ledger.Complete(token); }
    std::uint64_t Generation(std::uint32_t qid) const noexcept {
        return ledger.Generation(qid);
    }
    std::uint64_t Issued(std::uint32_t qid) const noexcept { return ledger.Issued(qid); }
    std::uint64_t Completed(std::uint32_t qid) const noexcept { return ledger.Completed(qid); }

    // Capture and retire in guest frame order. A newer ASC completion cannot release an older
    // mark whose graphics or another qid's prefix is still pending.
    std::optional<FrameSnapshot> CaptureFrame(std::uint64_t id) {
        if (frame_marks.size() == MaxFrameMarks || stopped) return std::nullopt;
        frame_marks.push_back(ledger.Capture(id));
        return frame_marks.back();
    }
    bool FrontFrameReady() const noexcept {
        return !frame_marks.empty() && ledger.Ready(frame_marks.front());
    }
    std::optional<FrameSnapshot> RetireFrontFrame() {
        if (!FrontFrameReady()) return std::nullopt;
        FrameSnapshot mark = frame_marks.front();
        frame_marks.pop_front();
        return mark;
    }
    std::size_t PendingFrames() const noexcept { return frame_marks.size(); }

    // The open tick is the only tick into which a new record may be emitted. A seal plan pins
    // it until the owning scheduler reports a successful submit through FinishSeal.
    bool BeginRecord(CpLane lane, std::uint64_t tick) noexcept {
        auto& owner = owners[Index(lane)];
        if (stopped || owner.record_in_progress || owner.seal_in_flight ||
            tick != owner.open_tick) return false;
        if (BlockedByDependency(CpTick{lane, tick})) return false;
        owner.record_in_progress = true;
        return true;
    }
    bool EndRecord(CpLane lane, std::uint64_t tick) noexcept {
        auto& owner = owners[Index(lane)];
        if (!owner.record_in_progress || tick != owner.open_tick) return false;
        owner.record_in_progress = false;
        return true;
    }

    // Admission enqueues a stable request. A request to one's own lane follows the same rule:
    // while record is pending it cannot be ACKed inline. Waiting is the caller's responsibility
    // and must occur after releasing admission/cache locks.
    Request SealAtLeast(QidTick target) {
        auto request = std::make_shared<SealRequest>(target);
        if (stopped) {
            request->Resolve(SealState::Cancelled);
        } else if (!ValidTarget(target)) {
            request->Resolve(SealState::Failed);
        } else {
            const auto& owner = owners[Index(target.lane)];
            if (target.tick <= owner.submitted_tick) {
                request->Resolve(SealState::Done, owner.submitted_tick);
            } else if (target.tick > owner.open_tick) {
                request->Resolve(SealState::Failed);
            } else {
                pending_seals[Index(target.lane)].push_back(request);
            }
        }
        return request;
    }

    // A guest fault has no ASC qid. Seal the C owner's current global tick,
    // including records from every qid, before a paused G readback runs.
    Request SealGuestAscPrefix() {
        const QidTick target{CpLane::Asc, 0, 0, owners[Index(CpLane::Asc)].open_tick};
        auto request = std::make_shared<SealRequest>(target);
        if (stopped) {
            request->Resolve(SealState::Cancelled);
        } else {
            pending_seals[Index(CpLane::Asc)].push_back(request);
        }
        return request;
    }

    // Called at an owner safe point. The caller must drop the admission mutex and every cache
    // lock before executing the plan's scheduler flush, then call FinishSeal under admission.
    std::optional<SealPlan> PlanSeal(CpLane lane, bool cache_locks_held) noexcept {
        auto& owner = owners[Index(lane)];
        if (stopped || cache_locks_held || owner.record_in_progress || owner.seal_in_flight ||
            pending_seals[Index(lane)].empty() ||
            BlockedByDependency(CpTick{lane, owner.open_tick})) return std::nullopt;
        owner.seal_in_flight = true;
        return SealPlan{lane, owner.open_tick};
    }

    bool FinishSeal(SealPlan plan, bool submitted) noexcept {
        auto& owner = owners[Index(plan.lane)];
        if (!owner.seal_in_flight || owner.open_tick != plan.through_tick) return false;
        owner.seal_in_flight = false;
        if (stopped) {
            ResolveRequests(plan.lane, SealState::Cancelled);
            return false;
        }
        if (!submitted) {
            ResolveRequests(plan.lane, SealState::Failed);
            return false;
        }
        owner.submitted_tick = plan.through_tick;
        ++owner.open_tick;
        ResolveRequests(plan.lane, SealState::Done);
        PruneDependencies();
        return true;
    }

    // The scheduler may submit an open tick for another reason (frame end, queue pressure).
    // Publish that already completed host submission so pending seal requests get the same ACK.
    bool PublishSubmitted(CpLane lane, std::uint64_t through_tick) noexcept {
        auto& owner = owners[Index(lane)];
        if (stopped || owner.record_in_progress || owner.seal_in_flight ||
            through_tick != owner.open_tick ||
            BlockedByDependency(CpTick{lane, through_tick})) return false;
        owner.submitted_tick = through_tick;
        ++owner.open_tick;
        ResolveRequests(lane, SealState::Done);
        PruneDependencies();
        return true;
    }

    // A consumer is not recorded yet. Seal its own earlier prefix first, then link it to the
    // producer. An edge is retained only while both tokens are open. Rejecting a path back to
    // the consumer prevents mutual G<->C waits on the same pair of open submissions.
    bool AddDependency(CpTick consumer, QidTick producer) {
        if (stopped || !ValidTarget(producer) || consumer.tick == 0) return false;
        const auto& consumer_owner = owners[Index(consumer.lane)];
        const auto& producer_owner = owners[Index(producer.lane)];
        if (consumer.tick != consumer_owner.open_tick || consumer_owner.record_in_progress ||
            consumer_owner.seal_in_flight || producer.tick > producer_owner.open_tick)
            return false;
        if (producer.tick <= producer_owner.submitted_tick) return true;
        const CpTick source{producer.lane, producer.tick};
        if (source == consumer || HasPath(source, consumer)) return false;
        if (std::find_if(dependencies.begin(), dependencies.end(), [&](const Edge& edge) {
                return edge.consumer == consumer && edge.producer == source;
            }) == dependencies.end())
            dependencies.push_back(Edge{consumer, source, producer});
        return true;
    }

    // Revalidation can discard a packet before its first emission. Its reserved graph edges
    // must be removed before retrying or routing it through the coordinated slow path.
    void CancelConsumer(CpTick consumer) {
        std::erase_if(dependencies,
                      [&](const Edge& edge) { return edge.consumer == consumer; });
    }

    void Stop() noexcept {
        stopped = true;
        ResolveRequests(CpLane::Graphics, SealState::Cancelled);
        ResolveRequests(CpLane::Asc, SealState::Cancelled);
    }

    std::uint64_t SubmittedTick(CpLane lane) const noexcept {
        return owners[Index(lane)].submitted_tick;
    }
    std::uint64_t OpenTick(CpLane lane) const noexcept { return owners[Index(lane)].open_tick; }
    std::size_t ActiveDependencies() const noexcept { return dependencies.size(); }

private:
    struct OwnerState {
        std::uint64_t open_tick{1};
        std::uint64_t submitted_tick{};
        bool record_in_progress{};
        bool seal_in_flight{};
    };
    struct Edge {
        CpTick consumer{};
        CpTick producer{};
        QidTick producer_token{};
        friend bool operator==(const Edge&, const Edge&) = default;
    };
    static constexpr std::size_t Index(CpLane lane) noexcept {
        return lane == CpLane::Graphics ? 0 : 1;
    }
    bool ValidTarget(QidTick token) const noexcept {
        return token.tick != 0 && token.qid < NumQids && token.generation != 0 &&
               (token.qid == 0 ? token.lane == CpLane::Graphics
                               : token.lane == CpLane::Asc) &&
               ledger.Mapped(token.qid) && ledger.Generation(token.qid) == token.generation;
    }
    static bool IsGuestAscPrefix(QidTick token) noexcept {
        return token.lane == CpLane::Asc && token.qid == 0 && token.generation == 0 &&
               token.tick != 0;
    }
    void ResolveRequests(CpLane lane, SealState failure) noexcept {
        auto& requests = pending_seals[Index(lane)];
        const auto submitted = owners[Index(lane)].submitted_tick;
        std::erase_if(requests, [&](const Request& request) {
            if (!IsGuestAscPrefix(request->target) &&
                (request->target.generation != ledger.Generation(request->target.qid) ||
                 !ledger.Mapped(request->target.qid))) {
                request->Resolve(SealState::Failed);
            } else if (failure != SealState::Done) {
                request->Resolve(failure);
            } else if (request->target.tick <= submitted) {
                request->Resolve(SealState::Done, submitted);
            } else {
                return false;
            }
            return true;
        });
    }
    void ResolveStaleRequests() noexcept {
        for (auto& requests : pending_seals) {
            std::erase_if(requests, [&](const Request& request) {
                if (IsGuestAscPrefix(request->target) ||
                    (request->target.generation == ledger.Generation(request->target.qid) &&
                     ledger.Mapped(request->target.qid))) return false;
                request->Resolve(SealState::Failed);
                return true;
            });
        }
    }
    bool HasPath(CpTick from, CpTick target) const {
        std::vector<CpTick> frontier{from};
        for (std::size_t index = 0; index < frontier.size(); ++index) {
            const CpTick current = frontier[index];
            if (current == target) return true;
            for (const Edge& edge : dependencies) {
                if (edge.consumer == current &&
                    std::find(frontier.begin(), frontier.end(), edge.producer) == frontier.end())
                    frontier.push_back(edge.producer);
            }
        }
        return false;
    }
    bool BlockedByDependency(CpTick consumer) const noexcept {
        return std::any_of(dependencies.begin(), dependencies.end(), [&](const Edge& edge) {
            return edge.consumer == consumer &&
                   edge.producer.tick > owners[Index(edge.producer.lane)].submitted_tick;
        });
    }
    void PruneDependencies() {
        std::erase_if(dependencies, [&](const Edge& edge) {
            return edge.consumer.tick <= owners[Index(edge.consumer.lane)].submitted_tick ||
                   edge.producer.tick <= owners[Index(edge.producer.lane)].submitted_tick ||
                   !ValidTarget(edge.producer_token);
        });
    }

    Ledger ledger{};
    std::deque<FrameSnapshot> frame_marks{};
    std::array<OwnerState, 2> owners{};
    std::array<std::vector<Request>, 2> pending_seals{};
    std::vector<Edge> dependencies{};
    bool stopped{};
};

} // namespace iss::g4
