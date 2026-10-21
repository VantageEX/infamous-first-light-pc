// Serial qid1 owner adapter and GPU producer-token model. Stage 1 does not start a second CP.
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace iss::g4 {

enum class CpOwner : std::uint8_t { Graphics, Asc1 };

// Explicit recording selection. Generation is zero until Liverpool provides a mapped-qid token.
struct OwnerContext {
    CpOwner owner{CpOwner::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    const void* scheduler{};
    const void* runtime{};
    const void* stream{};
    std::uint32_t queue_index{};
};

template <class QueueContext>
bool MatchesSelected(const OwnerContext& requested, const QueueContext& selected) noexcept {
    if (requested.qid != 1 || requested.queue_index > 1 ||
        (requested.owner == CpOwner::Asc1) != (requested.queue_index == 1)) return false;
    return requested.scheduler == selected.scheduler && requested.runtime == selected.runtime &&
           requested.stream == selected.stream_buffer && requested.queue_index == selected.index;
}

// This adapter asserts the explicit selection, then uses the existing serial BufferCache path.
// It must not be called by a second CP until BufferCache state is partitioned/coherently locked.
template <class BufferCache, class... Args>
auto ObtainForSerialOwner(BufferCache& cache, const OwnerContext& requested, Args&&... args) {
    if (!MatchesSelected(requested, cache.ActiveQueue())) std::abort();
    return cache.ObtainBuffer(std::forward<Args>(args)...);
}

// Scheduler timeline token, deliberately distinct from Liverpool's admission sequence.
struct ProducerToken {
    CpOwner owner{CpOwner::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    std::uint64_t scheduler_tick{};
    bool operator==(const ProducerToken&) const = default;
    bool Valid() const noexcept {
        return qid < 57 && generation != 0 && scheduler_tick != 0 &&
               (owner != CpOwner::Asc1 || qid == 1);
    }
};

struct OwnerProgress {
    CpOwner owner{CpOwner::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    std::uint64_t current_tick{};
    std::uint64_t submitted_tick{};
    std::uint64_t completed_tick{};
    bool record_in_progress{};
};

enum class ProducerGate : std::uint8_t {
    Invalid, StaleGeneration, RecordPending, SealRequired, WaitGpu, Complete
};

inline ProducerGate GateProducer(const ProducerToken& token, const OwnerProgress& progress) noexcept {
    if (!token.Valid() || token.owner != progress.owner || token.qid != progress.qid ||
        progress.current_tick < progress.submitted_tick ||
        progress.submitted_tick < progress.completed_tick ||
        token.scheduler_tick > progress.current_tick) return ProducerGate::Invalid;
    if (token.generation != progress.generation) return ProducerGate::StaleGeneration;
    if (progress.record_in_progress && token.scheduler_tick >= progress.current_tick)
        return ProducerGate::RecordPending;
    if (token.scheduler_tick > progress.submitted_tick) return ProducerGate::SealRequired;
    if (token.scheduler_tick > progress.completed_tick) return ProducerGate::WaitGpu;
    return ProducerGate::Complete;
}

// The exact 8-byte feedback producer and the whole readback window may have different writers.
struct FeedbackSnapshot {
    ProducerToken exact{};
    std::array<ProducerToken, 2> window{}; // graphics and Asc1; zero tick means absent
    std::uint64_t memory_generation{};
    std::uint64_t write_sequence{};
};

inline bool SameWindowWriters(const FeedbackSnapshot& a, const FeedbackSnapshot& b) noexcept {
    return a.window == b.window;
}

// A newer writer or remap must retain dirty state when an older download retires.
inline bool CanRetireDirty(const FeedbackSnapshot& captured,
                           const FeedbackSnapshot& current) noexcept {
    return captured.memory_generation != 0 &&
           captured.memory_generation == current.memory_generation &&
           captured.write_sequence == current.write_sequence &&
           captured.exact == current.exact && SameWindowWriters(captured, current);
}

inline ProducerGate GateFeedback(const FeedbackSnapshot& snapshot,
                                 const OwnerProgress& graphics,
                                 const OwnerProgress& asc1) noexcept {
    if (!snapshot.exact.Valid() || snapshot.memory_generation == 0) return ProducerGate::Invalid;
    if ((snapshot.window[0].scheduler_tick != 0 &&
         snapshot.window[0].owner != CpOwner::Graphics) ||
        (snapshot.window[1].scheduler_tick != 0 &&
         snapshot.window[1].owner != CpOwner::Asc1)) return ProducerGate::Invalid;
    bool exact_in_window = false;
    ProducerGate result = ProducerGate::Complete;
    for (const auto& token : snapshot.window) {
        if (token.scheduler_tick == 0) continue;
        if (token == snapshot.exact) exact_in_window = true;
        const auto& progress = token.owner == CpOwner::Graphics ? graphics : asc1;
        const auto gate = GateProducer(token, progress);
        if (gate == ProducerGate::Invalid || gate == ProducerGate::StaleGeneration) return gate;
        if (gate == ProducerGate::RecordPending) result = gate;
        else if (gate == ProducerGate::SealRequired && result != ProducerGate::RecordPending)
            result = gate;
        else if (gate == ProducerGate::WaitGpu && result == ProducerGate::Complete)
            result = gate;
    }
    return exact_in_window ? result : ProducerGate::Invalid;
}

} // namespace iss::g4
