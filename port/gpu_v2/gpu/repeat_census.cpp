// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_REPEAT_CENSUS (repeat_census.h).
#include "repeat_census.h"
#include "common/perf_stats.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace iss::repeat_census {
const bool enabled = [] {
    const char* p = std::getenv("ISS_REPEAT_CENSUS");
    return p && p[0] == '1';
}();

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 SlotBits = 16;
constexpr u32 ReportFrames = 240;
constexpr const char* names[KindCount] = {"obtain_buffer", "stage_buffer", "vertex_range",
                                          "index_range", "texture", "protect",
                                          "fault"};
// The guest's three direct memory mappings (src/memory.c, this game): A, B, C, other.
u32 HeapOf(u64 address) {
    if (address >= 0x80000000ull && address < 0xC8000000ull) return 0;
    if (address >= 0x20000000ull && address < 0x40000000ull) return 1;
    if (address >= 0x200000000ull && address < 0x2B8000000ull) return 2;
    return 3;
}

struct Slot {
    u64 address{}, size{};
    u32 extra{}, count{}, generation{};
};
struct Top {
    u64 address{}, size{};
    u32 extra{}, count{};
};
struct Table {
    std::mutex mutex;
    std::unique_ptr<Slot[]> slots = std::make_unique<Slot[]>(1u << SlotBits);
    u32 generation = 1;
    u64 frame = ~0ull;
    u64 calls{}, unique{}, overflow{};
    // Totals over the report window.
    u64 window_frames{}, window_calls{}, window_unique{};
    std::array<u64, 4> window_heap{};
    std::array<Top, 6> top{}; // most repeated keys of the last finished frame

    void EndFrame() {
        window_calls += calls;
        window_unique += unique;
        ++window_frames;
        // The most repeated keys of this frame (scanned once per report window).
        if (window_frames % ReportFrames == 0) {
            top = {};
            for (u32 i = 0; i < (1u << SlotBits); ++i) {
                const auto& s = slots[i];
                if (s.generation != generation || s.count < 2) continue;
                auto it = std::min_element(top.begin(), top.end(),
                    [](const Top& a, const Top& b) { return a.count < b.count; });
                if (s.count > it->count) *it = {s.address, s.size, s.extra, s.count};
            }
            std::sort(top.begin(), top.end(),
                      [](const Top& a, const Top& b) { return a.count > b.count; });
        }
        calls = unique = 0;
        if (++generation == 0) {
            for (u32 i = 0; i < (1u << SlotBits); ++i) slots[i].generation = 0;
            generation = 1;
        }
    }
};
Table tables[KindCount];

void Report(Kind kind, Table& t) {
    const double frames = double(t.window_frames ? t.window_frames : 1);
    std::fprintf(stderr,
                 "REPEAT_CENSUS kind=%s frames=%llu calls_per_frame=%.1f unique_per_frame=%.1f "
                 "repeat=%.1f%% overflow=%llu ",
                 names[kind], (unsigned long long)t.window_frames, t.window_calls / frames,
                 t.window_unique / frames,
                 t.window_calls ? 100.0 * double(t.window_calls - t.window_unique) /
                                      double(t.window_calls)
                                : 0.0,
                 (unsigned long long)t.overflow);
    std::fprintf(stderr, "heapA=%.1f heapB=%.1f heapC=%.1f other=%.1f top=",
                 t.window_heap[0] / frames, t.window_heap[1] / frames, t.window_heap[2] / frames,
                 t.window_heap[3] / frames);
    for (const auto& top : t.top)
        if (top.count)
            std::fprintf(stderr, "%llx+%llx/%x:%u,", (unsigned long long)top.address,
                         (unsigned long long)top.size, top.extra, top.count);
    std::fputc('\n', stderr);
    t.window_frames = t.window_calls = t.window_unique = 0;
    t.window_heap = {};
}
} // namespace

void Note(Kind kind, u64 address, u64 size, u32 extra) {
    if (!enabled || kind >= KindCount) return;
    const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    auto& t = tables[kind];
    std::scoped_lock lock{t.mutex};
    if (frame != t.frame) {
        if (t.frame != ~0ull) {
            t.EndFrame();
            if (t.window_frames % ReportFrames == 0) Report(kind, t);
        }
        t.frame = frame;
    }
    ++t.calls;
    ++t.window_heap[HeapOf(address)];
    u64 h = (address * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full) ^ extra;
    h ^= h >> 29;
    for (u32 probe = 0; probe < 16; ++probe) {
        auto& s = t.slots[(h + probe) & ((1u << SlotBits) - 1)];
        if (s.generation != t.generation) {
            s = {address, size, extra, 1, t.generation};
            ++t.unique;
            return;
        }
        if (s.address == address && s.size == size && s.extra == extra) {
            ++s.count;
            return;
        }
    }
    ++t.overflow;
    ++t.unique; // counted as distinct: the census stays conservative about repetition
}
} // namespace iss::repeat_census
