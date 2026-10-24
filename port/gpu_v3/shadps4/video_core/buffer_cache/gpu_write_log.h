// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include "common/types.h"

namespace VideoCore {

/**
 * SHADPS4_FAULT_VETO_EXACT (default 0; 1 = use, 2 = count only): the page ranges of recent
 * GPU write markings, so that a fault job whose region sequence moved can ask whether one of
 * those markings touched its own window.
 *
 * The region sequence (RegionManager::gpu_write_seq) counts every GPU write marking of a
 * 4 MiB region, and the particle regions take hundreds per frame; a job is vetoed for a
 * marking anywhere in its regions, and a veto costs the command processor a synchronous
 * download (flush + GPU wait). The invariant the vote protects is narrower: no GPU write
 * marking may intersect the job's window between its snapshot and its publication (the
 * publication writes the staged bytes into guest memory and clears the window's GPU state).
 *
 * Command processor only: every marking site (RegionManager::ChangeRegionState<GPU, true>,
 * MemoryTracker::TryWrittenFastPath, MemoryTracker::InvalidateGpuWriteSeq), the job snapshot
 * and the vote run there, so the ring has one writer and its reader is the same thread. The
 * head is atomic only so that a stray reader would see a whole value.
 */
class GpuWriteLog {
public:
    enum class Result { Clean, Touched, Overrun };

    static int Mode() noexcept {
        return mode;
    }

    static void Append(u64 begin, u64 end) noexcept {
        const u64 at = head.load(std::memory_order_relaxed);
        entries[at & Mask] = {begin, end};
        head.store(at + 1, std::memory_order_relaxed);
    }

    static u64 Position() noexcept {
        return head.load(std::memory_order_relaxed);
    }

    /// Did a marking logged at or after `since` intersect [begin, end)? Overrun when the ring
    /// no longer holds every entry since then (the caller keeps the region sequence's veto).
    static Result Check(u64 since, u64 begin, u64 end, u64& hit_begin, u64& hit_end) noexcept {
        const u64 now = head.load(std::memory_order_relaxed);
        if (now - since > Capacity) {
            return Result::Overrun;
        }
        for (u64 at = since; at != now; ++at) {
            const Entry& entry = entries[at & Mask];
            if (entry.begin < end && begin < entry.end) {
                hit_begin = entry.begin;
                hit_end = entry.end;
                return Result::Touched;
            }
        }
        return Result::Clean;
    }

private:
    struct Entry {
        u64 begin;
        u64 end;
    };
    // Fire logs a few thousand markings per frame and a job can age a frame or two; an overrun
    // only costs the refinement (FaultRegime exact_overrun counts them).
    static constexpr u64 Capacity = 1ULL << 16;
    static constexpr u64 Mask = Capacity - 1;
    static inline std::array<Entry, Capacity> entries{};
    static inline std::atomic<u64> head{0};
    static inline const int mode = [] {
        const char* env = std::getenv("SHADPS4_FAULT_VETO_EXACT");
        return env ? std::atoi(env) : 0;
    }();
};

} // namespace VideoCore
