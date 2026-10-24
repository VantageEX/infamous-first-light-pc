// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <string>
#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

void SetCurrentThreadRealtime(std::chrono::nanoseconds period_ns);

void SetCurrentThreadPriority(ThreadPriority new_priority);

void SetCurrentThreadName(const char* name);

void SetThreadName(void* thread, const char* name);

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

class AccurateTimer {
    std::chrono::nanoseconds target_interval{};
    std::chrono::nanoseconds total_wait{};

    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit AccurateTimer(std::chrono::nanoseconds target_interval);

    void Start();

    void End();

    /// A new interval from the next tick (the vblank rate changed live); the accumulated
    /// wait / overrun is dropped so the new rate starts clean.
    void SetInterval(std::chrono::nanoseconds interval) {
        target_interval = interval;
        total_wait = {};
    }

    std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }
};

std::string GetCurrentThreadName();

/// SHADPS4_CP_QOS=1: turn off process power throttling (EcoQoS) and raise this thread to
/// THREAD_PRIORITY_HIGHEST; =2 additionally restricts it to the performance cores (highest CPU
/// set efficiency class). Default off. Windows only; call on the thread itself.
void ApplyCommandProcessorPlacement();

/// SHADPS4_MEM_CENSUS: this process's private bytes and working set (0 / 0 where unknown).
void ProcessMemory(u64& private_bytes, u64& working_set);

/// Installed physical memory in bytes (0 where unknown). SHADPS4_SIM_RAM_MB stands in for it.
u64 PhysicalRamBytes();

/// 1.1.1 (GitHub #5): Windows' memory state for an out-of-memory line - free RAM, free commit (RAM +
/// page file) of the commit limit, this process's private bytes. Empty where unknown.
std::string DescribeSystemMemory();

/// SHADPS4_MEM_CENSUS: committed bytes by kind and place (VirtualQuery walk): private / mapped,
/// inside the guest range 0x200000000 - 0x1000000000 or outside it (emulator, driver, heaps).
struct CommitCensus {
    u64 private_guest{}, private_host{}, mapped_guest{}, mapped_host{}, image{};
    u32 host_blocks{};             ///< private allocations outside the guest range
    u64 top_base[8]{}, top_bytes[8]{}; ///< the largest of them (committed bytes per allocation)
    u64 default_heap{}, other_heaps{}; ///< committed bytes: the process heap (malloc / new), others
    u32 heaps{};
};
CommitCensus WalkCommitted();
/// SHADPS4_GUEST_PCORE=<mask> (2026-10-15, notes/gamemain-saturation-2026-10-15.md): bit 0 the game's
/// JobWorker* threads, bit 1 Game:Main get every performance logical processor (minus reserved
/// cores) as selected CPU sets and power throttling off. Called with the thread's handle and name.
void ApplyGuestThreadPlacement(void* handle, const char* name);
/// SHADPS4_POOL_AUTO=1 (2026-10-15): worker pools sized from the logical processors in the process
/// affinity mask (CPU-shape matrix: 6c/12t and 6c/6t gained ~25% with smaller pools). 0 = off.
u32 AutoPoolWorkers(bool lean);
/// SHADPS4_HELPER_PLACE=<n> (2026-10-15): emulator helper threads (lean / A2 workers); 1 = the
/// efficiency cores as selected CPU sets, 2 = THREAD_PRIORITY_BELOW_NORMAL. Call on the thread.
void ApplyHelperThreadPlacement(const char* name);
/// SHADPS4_HOT_CORES=<mask> (default 0): bit 0 gives the Vulkan recording worker, bit 1 the CP
/// front thread a whole performance core each (the two before the one SHADPS4_CP_QOS=3 gives the
/// CP back); the rest of the process is kept off those cores. Measured 2026-10-01 without it: the
/// recorder (80-87% busy) changes logical processor at ~20% of its chunk boundaries and spends
/// 2-15% of them on efficiency cores. Windows only; call on the thread itself.
enum class HotThread { Recorder, Front, GameMain };
void ApplyHotThreadPlacement(HotThread which);
/// SHADPS4_CRIT_PRIO (bit 1 recorders + front, bit 2 Game:Main): above-normal priority for the
/// calling thread; ApplyHotThreadPlacement calls it, recorder lanes past lane 0 call it directly.
void ApplyCriticalThreadPriority(HotThread which);
/// SHADPS4_CP_CORE_STATS=1: sample the logical CPU this thread runs on (call often, it is cheap)
/// and periodically log the share on performance vs efficiency cores. Diagnostic only.
void SampleCommandProcessorCore();
/// SHADPS4_THREAD_STATS=1 (diagnostic, default off): call at a natural point of a hot thread's
/// loop. Every 2^stride_log2-th call notes the logical processor the thread is on; four times a
/// second it times a fixed dependent-operation loop with rdtsc (loop iterations per TSC tick
/// follow the core's real clock, which the invariant TSC does not); every ~10 s the thread logs
/// its processor histogram, migrations, CPU time and the clock probe (median and minimum).
/// Explains run-to-run speed modes: placement and frequency versus extra work.
/// Cost, measured 2026-10-01 on a41 in fire: -3.3% FPS with the flag on (117.75 vs 121.3 / 122.0;
/// the per-record call on the CP back, not the sampled work). Attribution runs only, never in
/// an FPS comparison. Off: one inline test of a global per call site.
extern const bool g_thread_stats;
/// The logical processor the calling thread runs on (cheap), whether a logical processor is on a
/// performance core (hybrid parts; true elsewhere), and the clock probe of SampleThreadPlacement
/// (best of four TSC tick counts of a fixed 4096-step dependent chain, ~2 cycles a step).
u32 CurrentProcessor();
bool IsPerformanceProcessor(u32 cpu);
u32 ClockProbe();
void SampleThreadPlacementSlow(const char* role, unsigned stride_log2);
inline void SampleThreadPlacement(const char* role, unsigned stride_log2 = 6) {
    if (g_thread_stats) [[unlikely]] {
        SampleThreadPlacementSlow(role, stride_log2);
    }
}

/// CPU cycles the calling thread has run (QueryThreadCycleTime; 0 elsewhere). Diagnostics.
u64 CurrentThreadCycles();

} // namespace Common
