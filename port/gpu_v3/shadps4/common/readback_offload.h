// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include "common/logging/log.h"
#include "common/types.h"

/**
 * Flags and diagnostics for the experimental readback wait offload in BufferCache::ReadMemory.
 *
 *   SHADPS4_READBACK_OFFLOAD=1          guest-origin faults wait for the GPU on their own thread
 *   SHADPS4_READBACK_OFFLOAD_VERIFY=1   second GPU readback compared before publication
 *   SHADPS4_READBACK_OFFLOAD_PROFILE=1  per-origin counters, wait ns and a rare log line
 *
 * All default off. With PROFILE=0 no clock is read and no counter is touched. Counters are
 * relaxed atomics because faults arrive on guest threads and on the command processor.
 */
namespace Common::ReadbackOffload {

inline bool EnvFlag(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && std::string_view{env} == "1";
}

// Read once at startup: these are checked on every fault and upload, and a function-local
// static would add a guard check to each call.
inline const bool g_enabled = EnvFlag("SHADPS4_READBACK_OFFLOAD");
inline const bool g_verify_enabled = EnvFlag("SHADPS4_READBACK_OFFLOAD_VERIFY");
inline const bool g_profile_enabled = EnvFlag("SHADPS4_READBACK_OFFLOAD_PROFILE");
inline const bool g_latency_enabled = EnvFlag("SHADPS4_READBACK_LATENCY");

inline bool Enabled() {
    return g_enabled;
}

inline bool VerifyEnabled() {
    return g_verify_enabled;
}

inline bool ProfileEnabled() {
    return g_profile_enabled;
}

/// SHADPS4_READBACK_LATENCY=1: split each guest readback job's latency into CP pickup, prepare,
/// flush (recorder drain + submit), GPU backlog, completion-to-wake and finish. Starts a thread
/// that spins on the timeline semaphore to timestamp every tick; diagnostic only.
inline bool LatencyEnabled() {
    return g_latency_enabled;
}

/// SHADPS4_READBACK_LATENCY: guest RIP of the read fault this thread is resolving (0 unknown);
/// set by the page-fault handler, read by the batch-mix diagnostic.
inline thread_local u64 t_read_fault_rip = 0;
/// SHADPS4_READBACK_LATENCY: set while this thread runs the HLE memset (libc_internal).
inline thread_local bool t_in_hle_memset = false;
/// SHADPS4_READBACK_LATENCY: main-timeline tick of the batch that held the newest graphics-queue
/// label write (EOP / EOS / WRITE_DATA), set on the CP back. A separate async-compute stream
/// could only overtake the unfinished work after it (the game may have seen everything before).
inline std::atomic<u64> g_last_gfx_label_tick{0};

/// SHADPS4_ASC_SPAN_STATS=1 (diagnostic, CP back only): for each guest ASC span (the ASC dispatches
/// between two ASC RELEASE_MEMs), the newest main-timeline batch that accessed (c) or wrote (b) any
/// buffer those dispatches bind, before them. An ASC stream could overtake only the unfinished
/// batches after that. Per VkBuffer granularity (what a first ASC-stream version would track).
inline bool AscSpanStatsEnabled() {
    static const bool enabled = EnvFlag("SHADPS4_ASC_SPAN_STATS");
    return enabled;
}
struct AscSpanState {
    bool in_asc_dispatch = false;
    u64 w_access = 0, w_write = 0, binds = 0, max_buffer_mb = 0;
    // Window counters.
    u64 spans = 0, empty = 0, unfinished_sum = 0, over_access_sum = 0, over_write_sum = 0;
    u64 access_is_current = 0, write_is_current = 0, big_buffer_spans = 0;
    std::array<u64, 5> over_access_hist{}; ///< overtakeable batches 0, 1, 2-3, 4-7, 8+
};
inline AscSpanState& AscSpan() {
    static AscSpanState s;
    return s;
}

/// SHADPS4_READBACK_SPIN_WAIT=N: a guest readback polls the timeline semaphore for up to N us
/// before falling back to vkWaitSemaphores.
inline u64 SpinWaitNs() {
    static const u64 value = [] {
        const char* env = std::getenv("SHADPS4_READBACK_SPIN_WAIT");
        return env ? static_cast<u64>(std::strtoull(env, nullptr, 10)) * 1000ULL : 0ULL;
    }();
    return value;
}

struct LatencyCounters {
    std::atomic<u64> jobs{};
    std::atomic<u64> hop_ns{};      ///< fault -> CP starts PrepareFaultDownload
    std::atomic<u64> prep_ns{};     ///< prepare body before the flush
    std::atomic<u64> flush_ns{};    ///< Scheduler::Flush (recorder drain + vkQueueSubmit)
    std::atomic<u64> wait_ns{};     ///< guest wait for the tick
    std::atomic<u64> finish_ns{};   ///< wake -> FinishFaultDownload returned
    std::atomic<u64> total_ns{};    ///< fault -> finish
    std::atomic<u64> timed_jobs{};  ///< jobs whose tick completion the poller observed
    std::atomic<u64> ahead_ns{};    ///< flush end -> previous tick complete (earlier batches)
    std::atomic<u64> own_ns{};      ///< max(flush end, prev complete) -> own tick complete
    std::atomic<u64> wake_ns{};     ///< own tick complete (poller) -> guest woke
    std::atomic<u64> batch_draws{}; ///< draws in the batch the job's flush submitted
    std::atomic<u64> inflight{};    ///< submitted, incomplete batches ahead at flush
    std::atomic<u64> free_at_wait{};///< tick already complete when the guest started waiting
    std::atomic<u64> side_jobs{};   ///< single-pass jobs on the readback queue
    std::atomic<u64> side_wait_ns{};
    std::atomic<u64> side_total_ns{};
    std::atomic<u64> other_jobs{};  ///< retried jobs, not timed
    std::atomic<u64> next_log_ns{};
};

inline LatencyCounters& Latency() {
    static LatencyCounters counters;
    return counters;
}

/// Which thread took the fault. Anything that is not the command processor counts as Guest.
enum class Origin : u32 {
    CpSelf = 0,
    Guest = 1,
};

/// One synchronous download's submit-and-wait, split. total_ns is what sync_wait_ns has always
/// reported; submit_ns is Scheduler::Flush and gpu_wait_ns is Scheduler::Wait on that tick.
struct SyncTiming {
    u64 total_ns{};
    u64 submit_ns{};
    u64 gpu_wait_ns{};
};

struct Counters {
    std::array<std::atomic<u64>, 2> faults{};
    /// Every synchronous download of that origin that reached the GPU, fallbacks included.
    std::array<std::atomic<u64>, 2> sync_downloads{};
    std::array<std::atomic<u64>, 2> sync_wait_ns{};
    std::array<std::atomic<u64>, 2> sync_submit_ns{};
    std::array<std::atomic<u64>, 2> sync_gpu_wait_ns{};
    std::atomic<u64> jobs{};
    std::atomic<u64> successes{};
    std::atomic<u64> seq_vetoes{};
    std::atomic<u64> list_vetoes{};
    /// Vetoed jobs and capacity misses. Both are also counted in sync_downloads[Guest], and
    /// capacity_wait_ns is likewise a subset of sync_wait_ns[Guest].
    std::atomic<u64> sync_fallbacks{};
    std::atomic<u64> capacity_fallbacks{};
    std::atomic<u64> mapping_fallbacks{};
    std::atomic<u64> capacity_wait_ns{};
    std::atomic<u64> prepare_empty{};
    std::atomic<u64> cp_settles{};
    /// GPU wait the offload moved off the command processor, against cp_settle_wait_ns, the
    /// wait it took back there because something overlapped a job before its owner returned.
    std::atomic<u64> guest_wait_ns{};
    std::atomic<u64> cp_settle_wait_ns{};
    /// Command processor hops an invalidation made to settle jobs before touching the tracker.
    std::atomic<u64> invalidate_settle_hops{};
    /// Invalidations whose GPU data a job had already published, CPU-marked without a download.
    std::atomic<u64> write_marks{};
    /// Snapshots dropped because an image was tiled into the buffer without a tracker marking.
    std::atomic<u64> alias_bumps{};
    std::atomic<u64> verify_checks{};
    /// Jobs vetoed because no independent second readback could be made. Never published.
    std::atomic<u64> verify_skips{};
    std::atomic<u64> verify_mismatches{};
    /// SHADPS4_WRITE_FAULT_OFFLOAD: write-fault jobs admitted (also counted in jobs), and vetoed
    /// write jobs whose fallback was left to an unmap that began over the range meanwhile.
    std::atomic<u64> write_jobs{};
    std::atomic<u64> write_map_skips{};
    std::atomic<u64> next_log_ns{};
};

/// Always on (at most three relaxed increments per fault job): which fault-service regime a
/// run is in (5 vs thousands of sequence vetoes per run were seen on 2026-09-30). Logged
/// cumulatively next to the CpSplit window line.
struct RegimeCounters {
    std::atomic<u64> jobs{}, seq_vetoes{}, sync_fallbacks{};
    /// SHADPS4_FAULT_VETO_EXACT: jobs whose region sequence moved, by what the range log said.
    std::atomic<u64> exact_clean{}, exact_touched{}, exact_overrun{};
};
inline RegimeCounters g_regime;

inline Counters& State() {
    static Counters counters;
    return counters;
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Timestamp for a profiled interval; 0 and no clock read when profiling is off.
inline u64 ProfileNowNs() {
    return ProfileEnabled() ? NowNs() : 0;
}

inline void Add(std::atomic<u64>& counter, u64 value = 1) {
    if (ProfileEnabled()) {
        counter.fetch_add(value, std::memory_order_relaxed);
    }
}

inline void CountFault(Origin origin) {
    Add(State().faults[static_cast<u32>(origin)]);
}

/// A synchronous resolve that reached the GPU, and the ns its submit and its wait took.
/// An empty download returns before either, leaves the timing at 0 and is not counted here.
inline void CountSyncResolve(Origin origin, const SyncTiming& timing) {
    if (timing.total_ns != 0) {
        const u32 index = static_cast<u32>(origin);
        Add(State().sync_downloads[index]);
        Add(State().sync_wait_ns[index], timing.total_ns);
        Add(State().sync_submit_ns[index], timing.submit_ns);
        Add(State().sync_gpu_wait_ns[index], timing.gpu_wait_ns);
    }
}

/// Adds the ns since start_ns, a ProfileNowNs value, to counter.
inline void AddSince(std::atomic<u64>& counter, u64 start_ns) {
    if (ProfileEnabled()) {
        counter.fetch_add(NowNs() - start_ns, std::memory_order_relaxed);
    }
}

inline void MaybeLogLatency() {
    if (!LatencyEnabled()) {
        return;
    }
    constexpr u64 period_ns = 10'000'000'000ULL;
    auto& s = Latency();
    const u64 now = NowNs();
    u64 next = s.next_log_ns.load(std::memory_order_relaxed);
    if (now < next ||
        !s.next_log_ns.compare_exchange_strong(next, now + period_ns, std::memory_order_relaxed)) {
        return;
    }
    const auto get = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    const u64 jobs = std::max<u64>(get(s.jobs), 1);
    const u64 timed = std::max<u64>(get(s.timed_jobs), 1);
    const auto us = [](u64 ns, u64 n) { return double(ns) / double(n) / 1000.0; };
    LOG_INFO(Render,
             "ReadbackLatency: jobs={} side={} hop_us={:.1f} prep_us={:.1f} flush_us={:.1f} "
             "wait_us={:.1f} finish_us={:.1f} total_us={:.1f} timed={} ahead_us={:.1f} "
             "own_us={:.1f} wake_us={:.1f} batch_draws={:.1f} inflight={:.2f} free_at_wait={} "
             "side_wait_us={:.1f} side_total_us={:.1f} other={}",
             get(s.jobs), get(s.side_jobs), us(get(s.hop_ns), jobs), us(get(s.prep_ns), jobs),
             us(get(s.flush_ns), jobs), us(get(s.wait_ns), jobs), us(get(s.finish_ns), jobs),
             us(get(s.total_ns), jobs), get(s.timed_jobs), us(get(s.ahead_ns), timed),
             us(get(s.own_ns), timed), us(get(s.wake_ns), timed),
             double(get(s.batch_draws)) / double(jobs), double(get(s.inflight)) / double(jobs),
             get(s.free_at_wait), us(get(s.side_wait_ns), std::max<u64>(get(s.side_jobs), 1)),
             us(get(s.side_total_ns), std::max<u64>(get(s.side_jobs), 1)), get(s.other_jobs));
}

/// At most one line per ten seconds, from whichever thread wins the exchange.
inline void MaybeLog() {
    if (!ProfileEnabled()) {
        return;
    }
    constexpr u64 period_ns = 10'000'000'000ULL;
    auto& s = State();
    const u64 now = NowNs();
    u64 next = s.next_log_ns.load(std::memory_order_relaxed);
    if (now < next) {
        return;
    }
    if (!s.next_log_ns.compare_exchange_strong(next, now + period_ns,
                                               std::memory_order_relaxed)) {
        return;
    }
    if (next == 0) {
        // First fault only arms the timer.
        return;
    }
    const auto get = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    // Existing fields keep their names and order; new ones are appended. Field names stay
    // [a-z_] so a key=value scan of the line needs no change.
    LOG_INFO(Render,
             "ReadbackOffload: on={} cp_faults={} cp_sync={} cp_sync_wait_ns={} guest_faults={} "
             "guest_sync={} guest_sync_wait_ns={} jobs={} ok={} seq_veto={} list_veto={} "
             "sync_fallback={} cap_fallback={} cap_wait_ns={} empty={} cp_settles={} "
             "guest_wait_ns={} verify={} verify_skip={} verify_bad={} cp_sync_submit_ns={} "
             "cp_sync_gpu_wait_ns={} guest_sync_submit_ns={} guest_sync_gpu_wait_ns={} "
             "cp_settle_wait_ns={} inval_hops={} write_marks={} alias_bumps={} map_fallback={} "
             "write_jobs={} write_map_skip={}",
             Enabled(), get(s.faults[0]), get(s.sync_downloads[0]), get(s.sync_wait_ns[0]),
             get(s.faults[1]), get(s.sync_downloads[1]), get(s.sync_wait_ns[1]), get(s.jobs),
             get(s.successes), get(s.seq_vetoes), get(s.list_vetoes), get(s.sync_fallbacks),
             get(s.capacity_fallbacks), get(s.capacity_wait_ns), get(s.prepare_empty),
             get(s.cp_settles), get(s.guest_wait_ns), get(s.verify_checks), get(s.verify_skips),
             get(s.verify_mismatches), get(s.sync_submit_ns[0]), get(s.sync_gpu_wait_ns[0]),
             get(s.sync_submit_ns[1]), get(s.sync_gpu_wait_ns[1]), get(s.cp_settle_wait_ns),
             get(s.invalidate_settle_hops), get(s.write_marks), get(s.alias_bumps),
             get(s.mapping_fallbacks), get(s.write_jobs), get(s.write_map_skips));
}

} // namespace Common::ReadbackOffload
