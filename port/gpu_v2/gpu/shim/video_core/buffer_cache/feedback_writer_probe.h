// Opt-in serial writer observer. Never authorizes a readback or retires dirty state.
#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include "g4_qid1_owner.h"

namespace VideoCore {
class FeedbackWriterProbe {
public:
    using U64 = std::uint64_t;
    struct WriterEvent {
        U64 start{};
        U64 size{};
        U64 sequence{};
        U64 caller_pc{}; // ObtainBuffer caller; symbolization requires the run's module base
        iss::g4::ProducerToken token{};
        bool texel{};
    };
    struct Capture {
        iss::g4::FeedbackSnapshot snapshot{};
        U64 observation_epoch{}; // not a VMA generation
        WriterEvent exact{};
        WriterEvent page{};
        WriterEvent word{};
        WriterEvent window{};
        bool observed{};
    };

    void Publish(U64 start, U64 size, iss::g4::ProducerToken token, U64 sequence,
                 std::uint32_t queue, U64 caller_pc, bool texel) noexcept {
        if (size == 0 || queue > 1) return;
        std::lock_guard lock{mutex};
        bool found = false;
        for (auto& entry : history) {
            if (entry.tick == token.scheduler_tick && entry.queue == queue && entry.tick != 0) {
                if (!(entry.token == token)) entry.token.generation = 0; // mixed identities
                found = true;
                break;
            }
        }
        if (!found) history[next_history++ % history.size()] = {token.scheduler_tick, queue, token};
        const WriterEvent event{start, size, sequence, caller_pc, token, texel};
        for (auto& watch : watches) {
            if (!watch.active) continue;
            if (Overlaps(start, size, watch.address, 8)) {
                watch.exact = token;
                watch.sequence = sequence;
                watch.exact_event = event;
            }
            if (Overlaps(start, size, watch.address & ~U64{4095}, 4096))
                watch.page_event = event;
            if (Overlaps(start, size, watch.address & ~U64{262143}, 262144))
                watch.word_event = event;
            if (watch.window_end > watch.window_start &&
                Overlaps(start, size, watch.window_start, watch.window_end - watch.window_start))
                watch.window_event = event;
        }
    }

    // Guest writes, direct CP writes and unmap invalidations only affect diagnostic certainty.
    void Invalidate(U64 start, U64 size) noexcept {
        if (size == 0) return;
        std::lock_guard lock{mutex};
        for (auto& watch : watches) {
            if (watch.active && Overlaps(start, size, watch.address, 8)) {
                watch.exact = {};
                watch.sequence = 0;
                watch.exact_event = {};
                watch.page_event = {};
                watch.word_event = {};
                watch.window_event = {};
                ++watch.generation;
                if (watch.generation == 0) watch.generation = 1;
            }
        }
    }

    Capture WatchAndCapture(U64 address, U64 size, U64 window_start, U64 window_end,
                            std::array<U64, 2> window_ticks) noexcept {
        if (size != 8) return {};
        std::lock_guard lock{mutex};
        Watch* watch = nullptr;
        for (auto& candidate : watches) {
            if (candidate.active && candidate.address == address) {
                watch = &candidate;
                break;
            }
        }
        if (!watch) {
            watch = &watches[next_watch++ % watches.size()];
            *watch = {.address = address, .generation = 1, .active = true};
        }
        if (watch->window_start != window_start || watch->window_end != window_end) {
            watch->window_start = window_start;
            watch->window_end = window_end;
            watch->window_event = {}; // old window is not comparable
        }
        Capture result{};
        result.observed = true;
        result.snapshot.exact = watch->exact;
        result.observation_epoch = watch->generation;
        // No coherent VMA generation exists here: the safety gate remains closed.
        result.snapshot.memory_generation = 0;
        result.snapshot.write_sequence = watch->sequence;
        result.exact = watch->exact_event;
        result.page = watch->page_event;
        result.word = watch->word_event;
        result.window = watch->window_event;
        for (std::uint32_t q = 0; q < 2; ++q) {
            const U64 tick = window_ticks[q];
            if (tick == 0) continue;
            auto& dst = result.snapshot.window[q];
            dst.owner = q == 0 ? iss::g4::CpOwner::Graphics : iss::g4::CpOwner::Asc1;
            dst.scheduler_tick = tick;
            for (const auto& entry : history) {
                if (entry.queue == q && entry.tick == tick) {
                    dst = entry.token;
                    break;
                }
            }
        }
        return result;
    }

private:
    static bool Overlaps(U64 a, U64 asize, U64 b, U64 bsize) noexcept {
        return a < b + bsize && b < a + asize;
    }
    struct Watch {
        U64 address{};
        U64 generation{};
        U64 sequence{};
        U64 window_start{};
        U64 window_end{};
        iss::g4::ProducerToken exact{};
        WriterEvent exact_event{};
        WriterEvent page_event{};
        WriterEvent word_event{};
        WriterEvent window_event{};
        bool active{};
    };
    struct History {
        U64 tick{};
        std::uint32_t queue{};
        iss::g4::ProducerToken token{};
    };
    std::mutex mutex;
    std::array<Watch, 64> watches{};
    std::array<History, 256> history{};
    U64 next_watch{};
    U64 next_history{};
};
} // namespace VideoCore
