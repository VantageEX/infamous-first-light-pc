// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_UNIFIED_MEMORY (unified_memory.h): flag, heap range and statistics.
#include "unified_memory.h"
#include "common/perf_stats.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace iss::unified {
const bool enabled = [] {
    const char* p = std::getenv("ISS_UNIFIED_MEMORY");
    return p && p[0] == '1';
}();

namespace {
using u64 = std::uint64_t;
struct Range {
    u64 base = 0x20000000ull, size = 0x20000000ull;
};
const Range range = [] {
    Range r;
    // ISS_UNIFIED_RANGE=base:size (hexadecimal or decimal).
    if (const char* p = std::getenv("ISS_UNIFIED_RANGE")) {
        char* end{};
        const u64 base = std::strtoull(p, &end, 0);
        if (end && *end == ':') {
            const u64 size = std::strtoull(end + 1, nullptr, 0);
            if (size) r = {base, size};
        }
    }
    if (enabled)
        std::fprintf(stderr, "PORT_TUNING unified_memory base=0x%llx size=0x%llx\n",
                     (unsigned long long)r.base, (unsigned long long)r.size);
    return r;
}();
std::array<std::atomic<u64>, StatCount> stats{};
constexpr const char* names[StatCount] = {
    "reads",         "untouched_reads", "writes",          "image_sources",
    "cp_writes",     "cp_write_bytes",  "cp_writes_touched", "fences_deferred",
    "fences_outside", "fences_flushed", "cp_label_waits", "cp_read_syncs", "cp_read_wait_ms"};
} // namespace

u64 RangeBase() { return range.base; }
u64 RangeSize() { return range.size; }

void Add(Stat stat, u64 value) {
    stats[stat].fetch_add(value, std::memory_order_relaxed);
}

void MaybeReport() {
    if (!enabled) return;
    static u64 last{}, last_frame{};
    const u64 now = Common::Perf::NowNs();
    if (now - last < 5'000'000'000ull) return;
    last = now;
    const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    const double frames = double(frame > last_frame ? frame - last_frame : 1);
    last_frame = frame;
    std::fprintf(stderr, "UNIFIED_STATS frames=%.0f", frames);
    for (unsigned i = 0; i < StatCount; ++i) {
        const u64 value = stats[i].exchange(0, std::memory_order_relaxed);
        if (i == CpReadWaitNs)
            std::fprintf(stderr, " %s/frame=%.3f", names[i], double(value) / 1e6 / frames);
        else
            std::fprintf(stderr, " %s/frame=%.1f", names[i], double(value) / frames);
    }
    std::fputc('\n', stderr);
}
} // namespace iss::unified
