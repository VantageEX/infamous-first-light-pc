// Owner-local recording selection for the serial qid1 transition. No scheduler
// is submitted here; a future second CP must replace legacy shared scratch.
#pragma once

#include <cstdint>
#include <optional>
#include "g4_qid1_owner.h"

namespace iss::g4 {

template <class Scheduler, class Runtime, class Stream, class BufferQueue>
struct QueueRecordContext {
    const void* rasterizer{};
    CpOwner owner{CpOwner::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    Scheduler* scheduler{};
    Runtime* runtime{};
    Stream* stream{};
    BufferQueue* buffer_queue{};
    bool physical_compute{};

    [[nodiscard]] OwnerContext Identity() const noexcept {
        return OwnerContext{owner, qid, generation, scheduler, runtime, stream,
                            physical_compute ? 1u : 0u};
    }
    [[nodiscard]] bool ValidForSerialRecord() const noexcept {
        return scheduler && runtime && stream && buffer_queue &&
               (owner == CpOwner::Asc1) == physical_compute &&
               MatchesSelected(Identity(), *buffer_queue);
    }
    [[nodiscard]] bool ValidForSplitRecord() const noexcept {
        return qid < 57 && generation != 0 && scheduler && runtime && stream && buffer_queue &&
               (owner == CpOwner::Asc1) == physical_compute &&
               scheduler == buffer_queue->scheduler && runtime == buffer_queue->runtime &&
               stream == buffer_queue->stream_buffer &&
               buffer_queue->index == (physical_compute ? 1u : 0u);
    }
    [[nodiscard]] ProducerToken Writer(std::uint64_t scheduler_tick) const noexcept {
        return ProducerToken{owner, qid, generation, scheduler_tick};
    }
};

// A seal request is a plan, never an ACK. RecordPending must not be sent to a
// scheduler because the final command for that tick has not yet been recorded.
struct SealRequestPlan {
    ProducerGate gate{ProducerGate::Invalid};
    std::optional<ProducerToken> request{};
};

inline SealRequestPlan PlanSealRequest(const ProducerToken& token,
                                       const OwnerProgress& progress) noexcept {
    const auto gate = GateProducer(token, progress);
    return SealRequestPlan{gate, gate == ProducerGate::SealRequired
                                    ? std::optional<ProducerToken>{token} : std::nullopt};
}

// The caller must invoke this after releasing cache/hazard locks. Enqueue is
// owner-directed and may only ACK once recording and submission have finished.
template <class Enqueue>
SealRequestPlan EnqueueSealRequest(const ProducerToken& token,
                                   const OwnerProgress& progress, Enqueue&& enqueue) {
    const auto plan = PlanSealRequest(token, progress);
    if (plan.request) enqueue(*plan.request);
    return plan;
}

} // namespace iss::g4
