// SPDX-License-Identifier: GPL-2.0-or-later
#include "visual_capture.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include "video_core/renderdoc.h"

namespace iss::visual_capture {
namespace {

constexpr unsigned CaptureCount = 5;
constexpr std::uint64_t IntervalNs = 375'000'000;
enum State : unsigned { Idle, Arming, Active };

std::atomic<unsigned> state{Idle};
std::atomic<std::uint64_t> next_sequence{0};
// Published by Start's release store to state; only the presenting thread
// changes due_tick and next_index while Active.
std::uint64_t sequence_id{};
std::uint64_t start_tick{};
std::uint64_t due_tick{};
unsigned next_index{};

} // namespace

bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_VISUAL_SEQUENCE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool Start(std::uint64_t tick_ns) noexcept {
    if (!Enabled()) return false;
    unsigned expected = Idle;
    if (!state.compare_exchange_strong(expected, Arming, std::memory_order_acq_rel)) {
        std::fprintf(stderr, "VISUAL_SEQUENCE busy tick_ns=%llu\n",
                     static_cast<unsigned long long>(tick_ns));
        return false;
    }
    sequence_id = next_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    start_tick = due_tick = tick_ns;
    next_index = 0;
    std::fprintf(stderr, "VISUAL_SEQUENCE start id=%llu tick_ns=%llu\n",
                 static_cast<unsigned long long>(sequence_id),
                 static_cast<unsigned long long>(tick_ns));
    std::fflush(stderr);
    state.store(Active, std::memory_order_release);
    return true;
}

void OnFrame(std::uint64_t tick_ns, unsigned frame, bool reused) noexcept {
    if (!Enabled() || state.load(std::memory_order_acquire) != Active ||
        tick_ns < due_tick) return;
    VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
    ++next_index;
    std::fprintf(stderr,
                 "VISUAL_SEQUENCE request id=%llu index=%u/%u tick_ns=%llu elapsed_ms=%.3f frame=%u reused=%d\n",
                 static_cast<unsigned long long>(sequence_id), next_index, CaptureCount,
                 static_cast<unsigned long long>(tick_ns),
                 static_cast<double>(tick_ns - start_tick) / 1'000'000.0,
                 frame, static_cast<int>(reused));
    std::fflush(stderr);
    if (next_index == CaptureCount) {
        std::fprintf(stderr, "VISUAL_SEQUENCE requests_complete id=%llu tick_ns=%llu\n",
                     static_cast<unsigned long long>(sequence_id),
                     static_cast<unsigned long long>(tick_ns));
        std::fflush(stderr);
        state.store(Idle, std::memory_order_release);
    } else {
        due_tick = tick_ns + IntervalNs;
    }
}

} // namespace iss::visual_capture
