// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include "common/types.h"

// SHADPS4_FENCE_WATCH=1 (diagnostic, default off): tells a clobbered compute fence from a release the guest never
// issued (notes/freeze-2026-09-27.md). The compute ReleaseMem handler records the last fence write; downloads that
// write guest memory report when they cover that address after it was written; a watchdog thread logs the live
// guest word next to the last written value when no compute release arrived for 2 s.
namespace Common::FenceWatch {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_FENCE_WATCH");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// One fence address the compute queue signals. Written by the CP back thread only; read racily by the watchdog and
/// by download writers (diagnostic: a torn read shows as one odd log line).
struct Slot {
    std::atomic<u64> addr{0};
    std::atomic<u64> value{0};
    std::atomic<u32> size{0};
    std::atomic<u64> written_ns{0};
    std::atomic<u64> releases{0};
};

struct State {
    static constexpr size_t NumSlots = 16;
    /// Frame fences (REFERENCE.md: 0x248000000 area) are pinned: [0x248000000, 0x248000100) by offset / 8.
    static constexpr u64 PinnedBase = 0x248000000ULL;
    static constexpr size_t NumPinned = 32;
    std::array<Slot, NumSlots + NumPinned> slots;
    std::atomic<u64> last_release_ns{0};
    std::atomic<u64> covering_writes{0};
    /// gnmdriver submission gate (sceGnmSubmitDone / ResetSubmissionLock / WaitGpuIdle), mirrored for the watchdog.
    std::atomic<u32> gnm_lock{0};
    std::atomic<u32> gnm_waiters{0};
    std::atomic<u64> gnm_resets{0};
    std::atomic<u64> gnm_races{0};
};

inline State& Get() {
    static State state;
    return state;
}

/// Called by the compute ReleaseMem handler after SignalFence (defined in liverpool.cpp; starts the watchdog).
void NoteRelease(VAddr addr, u64 value, u32 size);

/// Called by guest-memory writers of downloaded data (TryWriteBacking): logs when [addr, addr+size) covers the last
/// fence after it was written.
void NoteGuestWrite(VAddr addr, const void* data, u64 size, const char* who);

} // namespace Common::FenceWatch
