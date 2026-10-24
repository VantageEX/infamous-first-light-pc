// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/thread.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

/**
 * SHADPS4_DEFER_READ_PROTECT=1: the read protection the command processor applies when a draw or
 * dispatch binds a buffer for GPU writes (precise readbacks) is handed to a dedicated thread.
 * Measured 2026-09-24 on 386F5167: those VirtualProtect calls were 4.2% of the CP's wall time.
 *
 * The tracker bits change on the CP at once; only the OS protection lags. Guest code may only
 * read GPU results after a signal the CP writes (EOS/EOP/RELEASE_MEM labels, WRITE_DATA,
 * interrupts, GPU idle), and the CP calls Fence() before each of those, which waits until every
 * queued protection has been applied. Watcher counts commute only while no removal is applied
 * before its own queued add: num_read_watchers is a 1-bit field that asserts on underflow ("Not
 * enough watchers", s915), so the CP's read-watch removals fence the queued adds first (q56b,
 * SHADPS4_READ_UNTRACK_FENCE). Other reorderings leave the same final state.
 */
class ProtectQueue {
public:
    static bool Enabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("SHADPS4_DEFER_READ_PROTECT");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    /**
     * SHADPS4_PROTECT_WAKE_STATS=1: counters for the hand-off, logged every 512 presented frames.
     * Measured 2026-09-25 (a19): Push was 0.63% of the CP and the yield loop in Fence 1.17%. The
     * question is whether a waited fence mostly waits for the Protector to wake (then the CP
     * should apply an unclaimed batch itself) or for VirtualProtect (then that would only move the
     * time onto the CP): compare wait per waited fence with apply time per batch, and see how
     * many waits began before the Protector had claimed the batch the fence needs.
     */
    static bool WakeStatsEnabled() {
        const char* env = std::getenv("SHADPS4_PROTECT_WAKE_STATS");
        return env != nullptr && env[0] == '1';
    }

    /**
     * SHADPS4_PROTECT_STEAL, default off (0). Measured 2026-09-25 (a19, SHADPS4_PROTECT_WAKE_STATS):
     * a waited fence took 13.76 us against 4.98 us of Protector apply per batch, and 51.6% of
     * waits began before the Protector had even claimed the batch, so wake latency dominates.
     *   1: Push notifies the condition variable only when the Protector is blocked in it.
     *   2: 1, plus Fence applies a batch the Protector has not yet claimed on the calling thread,
     *      then waits only for the batch the Protector already has in flight.
     */
    static int StealMode() {
        const char* env = std::getenv("SHADPS4_PROTECT_STEAL");
        if (env == nullptr || (env[0] != '1' && env[0] != '2')) {
            return 0;
        }
        return env[0] - '0';
    }

    /**
     * SHADPS4_PROTECT_SPIN_US=N (default 0 = block at once, as before). After a batch the
     * Protector polls `enqueued` for N microseconds before it blocks in the condition variable.
     * Measured 2026-09-30 (a38, back-only CPU samples): with PROTECT_STEAL=2 the CP back applied
     * ~5% (fire) / ~3% (forest) of its time in StealAndApply at label fences, because a blocked
     * Protector wakes later than the next fence arrives (62 ASC labels per frame in fire). A
     * polling Protector claims a push within one pause, so the fence finds the batch claimed and
     * waits only for the calls in flight. Claim order, the chained publication and Fence's
     * contract are unchanged: only when the Protector looks at `pending` changes.
     */
    static u64 SpinNs() {
        const char* env = std::getenv("SHADPS4_PROTECT_SPIN_US");
        if (env == nullptr) {
            return 0;
        }
        const u64 us = std::strtoull(env, nullptr, 10);
        return (us > 100'000 ? 100'000 : us) * 1000;
    }

    static ProtectQueue& Instance() {
        static ProtectQueue queue;
        return queue;
    }

    ProtectQueue() {
        if (spin_ns != 0) {
            LOG_INFO(Render, "ProtectQueue: SHADPS4_PROTECT_SPIN_US={} (Protector polls before "
                             "blocking)", spin_ns / 1000);
        }
        if (steal_mode != 0) {
            LOG_INFO(Render, "ProtectQueue: SHADPS4_PROTECT_STEAL={} ({})", steal_mode,
                     steal_mode == 2 ? "notify only when idle, fence applies unclaimed batches"
                                     : "notify only when idle");
        }
    }

    /// Queues the watcher updates and empties `items`. Command processor only.
    void Push(std::vector<DeferredWatch>& items) {
        if (items.empty()) {
            return;
        }
        bool notify = true;
        {
            std::scoped_lock lk{mutex};
            if (!thread.joinable()) {
                thread = std::jthread([this](std::stop_token stop) { Loop(stop); });
            }
            if (wake_stats) [[unlikely]] {
                CountPush(items.size());
            }
            pending.insert(pending.end(), items.begin(), items.end());
            enqueued.store(enqueued.load(std::memory_order_relaxed) + 1,
                           std::memory_order_release);
            if (steal_mode != 0) [[unlikely]] {
                // `idle` is true only while the Protector is blocked (or about to block) in
                // cv.wait with an empty queue: its wait predicate sets it under this mutex each
                // time it finds `pending` empty. If it is false, the Protector has not yet
                // re-checked the predicate and will see the items appended above, so no wake-up
                // can be lost. Clearing it here makes later pushes skip the notify until the
                // Protector blocks again.
                notify = idle;
                idle = false;
            }
        }
        if (notify) {
            cv.notify_one();
        }
        if (wake_stats) [[unlikely]] {
            stats.push_notifies.fetch_add(notify ? 1 : 0, std::memory_order_relaxed);
        }
        items.clear();
        if (wake_stats) [[unlikely]] {
            MaybeLogStats();
        }
    }

    /// q56b: some protection queued so far is not applied yet.
    bool Pending() const {
        return applied.load(std::memory_order_acquire) < enqueued.load(std::memory_order_acquire);
    }

    /// Waits until every protection queued so far is applied. Cheap when the queue is idle.
    void Fence() {
        if (wake_stats) [[unlikely]] {
            return FenceCounted();
        }
        const u64 target = enqueued.load(std::memory_order_acquire);
        if (applied.load(std::memory_order_acquire) >= target) {
            return;
        }
        if (steal_mode == 2 && StealAndApply()) [[unlikely]] {
            return;
        }
        WaitApplied(target);
    }

private:
    using Clock = std::chrono::steady_clock;

    /**
     * SHADPS4_PROTECT_STEAL=2, from Fence. If `pending` is non-empty, claims it exactly as the
     * Protector would, applies it on this thread, waits for the batch claimed before it, and
     * publishes. Returns false if there was nothing to claim: every push so far has then been
     * claimed by the Protector, and the caller waits for `applied` as before.
     *
     * Why Fence's contract (it returns only after every watch pushed before it is applied, by
     * whichever thread) still holds:
     * - Claiming is serialized by `mutex`: whoever swaps `pending` out takes every item pushed so
     *   far, reads `enqueued` as that batch's sequence number, and records the previous claim in
     *   `claimed_seq`. So batches are disjoint, each has one owner, and claims are totally ordered
     *   by sequence number. The Fence caller's `target` (loaded before the lock) is <= the stolen
     *   batch's number, because every push it counted is either in this batch or claimed earlier.
     * - Publication is chained: a claimer stores `applied = seq` (release) only after it has
     *   finished its own ApplyDeferredWatches and has observed (acquire) `applied >= prev`, the
     *   previous claim's number. By induction `applied` never decreases, and `applied >= s` means
     *   every batch numbered <= s is fully applied, and those VirtualProtect calls happen-before
     *   any thread's acquire load that sees it. That is the same guarantee the single Protector
     *   gave, when claim order and publish order could not differ.
     * - The two batches in flight (Protector's and the stolen one) may be applied concurrently and
     *   out of order. That is allowed for adds, and for removals whose add is already applied (a removal
     *   that overtakes its queued add underflows the 1-bit read count: q56b fences that case), and the
     *   Protector already runs UpdatePageWatchersForRegion concurrently with the CP's own updates
     *   and with fault handlers, under the page manager's locks.
     * - No deadlock: each claimer waits only for the claim immediately before its own, and the
     *   Protector never waits on anything the CP holds. With one CP thread the Protector's `prev`
     *   is always already published (nothing can be pushed while the CP is inside Fence).
     */
    bool StealAndApply() {
        u64 seq;
        u64 prev;
        {
            std::scoped_lock lk{mutex};
            if (pending.empty()) {
                return false;
            }
            steal_work.swap(pending);
            seq = enqueued.load(std::memory_order_relaxed);
            prev = claimed_seq;
            claimed_seq = seq;
            if (wake_stats) [[unlikely]] {
                stats.claimed.store(seq, std::memory_order_release);
            }
        }
        if (wake_stats) [[unlikely]] {
            const auto start = Clock::now();
            const u64 items = steal_work.size();
            ApplyDeferredWatches(steal_work);
            stats.steals.fetch_add(1, std::memory_order_relaxed);
            stats.steal_items.fetch_add(items, std::memory_order_relaxed);
            stats.steal_apply_ns.fetch_add(Ns(start, Clock::now()), std::memory_order_relaxed);
        } else {
            ApplyDeferredWatches(steal_work);
        }
        WaitApplied(prev);
        applied.store(seq, std::memory_order_release);
        return true;
    }

    void WaitApplied(u64 target) const {
        u32 spins = 0;
        while (applied.load(std::memory_order_acquire) < target) {
            if (++spins < 256) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

    void Loop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:Protector");
        std::vector<DeferredWatch> work;
        // SHADPS4_PROTECT_SPIN_US: `enqueued` as of this thread's last look at `pending`, and the
        // start of the current polling window (the end of the last batch).
        u64 seen = 0;
        Clock::time_point spin_from = Clock::now();
        while (!stop.stop_requested()) {
            if (spin_ns != 0) {
                // A push is a release store of `enqueued` under `mutex`. `idle` stays false while
                // polling, so Push does not notify; the predicate below still runs under `mutex`
                // before any block, so no push can be missed when the window ends.
                for (u32 spins = 1; enqueued.load(std::memory_order_acquire) == seen; ++spins) {
                    if ((spins & 63) == 0 &&
                        (stop.stop_requested() || Ns(spin_from, Clock::now()) >= spin_ns)) {
                        break;
                    }
                    _mm_pause();
                }
            }
            u64 seq;
            u64 prev = 0;
            Clock::time_point first_push{};
            {
                std::unique_lock lk{mutex};
                if (spin_ns != 0 && pending.empty() && !stop.stop_requested() &&
                    Ns(spin_from, Clock::now()) < spin_ns) {
                    // A Fence stole the batch (PROTECT_STEAL=2): note it and keep polling.
                    seen = enqueued.load(std::memory_order_relaxed);
                    continue;
                }
                if (wake_stats) [[unlikely]] {
                    sleeping = pending.empty();
                }
                if (steal_mode != 0 && spin_ns != 0) [[unlikely]] {
                    // As below, but a wake-up that finds the batch already stolen by a Fence
                    // returns (enqueued moved past `seen`) so that polling starts again: a
                    // Protector that only ever woke after the fence would otherwise re-block
                    // here forever and never poll (Opus review 2026-09-30). It still blocks
                    // only after setting `idle` under `mutex`.
                    cv.wait(lk, stop, [&] {
                        idle = pending.empty() &&
                               enqueued.load(std::memory_order_relaxed) == seen;
                        return !idle;
                    });
                    if (stop.stop_requested()) {
                        return;
                    }
                    if (pending.empty()) {
                        seen = enqueued.load(std::memory_order_relaxed);
                        spin_from = Clock::now();
                        continue;
                    }
                } else if (steal_mode != 0) [[unlikely]] {
                    // The predicate runs under `mutex` before every block, including after a
                    // wake-up that found the queue already stolen, so `idle` is true whenever
                    // this thread is blocked (see Push).
                    cv.wait(lk, stop, [this] {
                        idle = pending.empty();
                        return !idle;
                    });
                } else {
                    cv.wait(lk, stop, [this] { return !pending.empty(); });
                }
                if (stop.stop_requested()) {
                    return;
                }
                work.swap(pending);
                seq = enqueued.load(std::memory_order_relaxed);
                prev = claimed_seq;
                claimed_seq = seq;
                seen = seq;
                if (wake_stats) [[unlikely]] {
                    sleeping = false;
                    first_push = batch_first_push;
                    stats.claimed.store(seq, std::memory_order_release);
                }
            }
            if (wake_stats) [[unlikely]] {
                const auto start = Clock::now();
                const u64 items = work.size();
                ApplyDeferredWatches(work);
                stats.batches.fetch_add(1, std::memory_order_relaxed);
                stats.batch_items.fetch_add(items, std::memory_order_relaxed);
                stats.pickup_ns.fetch_add(Ns(first_push, start), std::memory_order_relaxed);
                stats.apply_ns.fetch_add(Ns(start, Clock::now()), std::memory_order_relaxed);
            } else {
                ApplyDeferredWatches(work);
            }
            if (steal_mode == 2) [[unlikely]] {
                WaitApplied(prev); // chained publication, see StealAndApply
            }
            applied.store(seq, std::memory_order_release);
            if (spin_ns != 0) {
                spin_from = Clock::now();
            }
        }
    }

    static u64 Ns(Clock::time_point start, Clock::time_point end) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    /// Under `mutex`. `sleeping` says whether this notify will wake a blocked Protector (cut (a)
    /// would skip the others); the first push into an empty queue stamps the batch.
    void CountPush(u64 items) {
        stats.push_calls.fetch_add(1, std::memory_order_relaxed);
        stats.push_items.fetch_add(items, std::memory_order_relaxed);
        stats.push_wakes.fetch_add(sleeping ? 1 : 0, std::memory_order_relaxed);
        sleeping = false; // later pushes before the Protector runs wake nobody
        if (pending.empty()) {
            batch_first_push = Clock::now();
        }
    }

    void FenceCounted() {
        const u64 target = enqueued.load(std::memory_order_acquire);
        stats.fence_calls.fetch_add(1, std::memory_order_relaxed);
        if (applied.load(std::memory_order_acquire) < target) {
            // Unclaimed: the batch this fence needs is still in `pending`, so the wait includes
            // the Protector's wake-up, not just its VirtualProtect calls. With
            // SHADPS4_PROTECT_STEAL=2 the CP applies such a batch itself, and wait_ns includes
            // that apply time, so wait_us/frame stays the CP's whole cost of fencing.
            const bool unclaimed = stats.claimed.load(std::memory_order_acquire) < target;
            const auto start = Clock::now();
            if (steal_mode != 2 || !StealAndApply()) {
                WaitApplied(target);
            }
            stats.fence_waits.fetch_add(1, std::memory_order_relaxed);
            stats.unclaimed_waits.fetch_add(unclaimed ? 1 : 0, std::memory_order_relaxed);
            stats.wait_ns.fetch_add(Ns(start, Clock::now()), std::memory_order_relaxed);
        }
        MaybeLogStats();
    }

    /// Command processor only (Push and Fence callers).
    void MaybeLogStats() const {
        if ((++stats.log_checks & 0xff) != 0) {
            return;
        }
        const u64 frames = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        if (stats.log_frames == 0 || frames < stats.log_frames + 512) {
            if (stats.log_frames == 0 && frames != 0) {
                // Start the first window at the first presented frame, not at boot.
                stats.log_frames = frames;
                stats.Take();
            }
            return;
        }
        const auto s = stats.Take();
        const u64 window = frames - stats.log_frames;
        const double n = static_cast<double>(window);
        stats.log_frames = frames;
        const auto per = [](u64 a, u64 b) { return b != 0 ? static_cast<double>(a) / b : 0.0; };
        const double wait_us = per(s.wait_ns, s.fence_waits) / 1000.0;
        const double apply_us = per(s.apply_ns, s.batches) / 1000.0;
        // Steal columns only with SHADPS4_PROTECT_STEAL set, so a control line keeps its format.
        // `batches` and the columns after it count only the Protector's batches.
        const std::string steal_cols =
            steal_mode == 0
                ? std::string{}
                : fmt::format(" | steal_mode={} steals/frame={:.2f} steal_items/steal={:.1f} "
                              "steal_apply_us/steal={:.2f}",
                              steal_mode, s.steals / n, per(s.steal_items, s.steals),
                              per(s.steal_apply_ns, s.steals) / 1000.0);
        LOG_INFO(Render,
                 "ProtectWake: frames={} fence_calls/frame={:.1f} fence_waits/frame={:.2f} "
                 "wait_us/waited_fence={:.2f} unclaimed_waits={:.1f}% wait_us/frame={:.1f} | "
                 "batches/frame={:.2f} apply_us/batch={:.2f} items/batch={:.1f} "
                 "pickup_us/batch={:.2f} | push_calls/frame={:.1f} push_notifies/frame={:.1f} "
                 "notifies_waking={:.1f}% | wait_over_apply={:.2f}{}",
                 window, s.fence_calls / n, s.fence_waits / n, wait_us,
                 100.0 * per(s.unclaimed_waits, s.fence_waits), s.wait_ns / n / 1000.0,
                 s.batches / n, apply_us, per(s.batch_items, s.batches),
                 per(s.pickup_ns, s.batches) / 1000.0, s.push_calls / n, s.push_notifies / n,
                 100.0 * per(s.push_wakes, s.push_calls), apply_us > 0 ? wait_us / apply_us : 0.0,
                 steal_cols);
    }

    struct WakeStats {
        struct Snapshot {
            u64 fence_calls, fence_waits, unclaimed_waits, wait_ns;
            u64 push_calls, push_items, push_wakes, push_notifies;
            u64 batches, batch_items, apply_ns, pickup_ns;
            u64 steals, steal_items, steal_apply_ns;
        };
        // Written by the CP (fence_*, wait_ns, push_*, steal_*) and the Protector
        // (batch/apply/pickup).
        std::atomic<u64> fence_calls{0}, fence_waits{0}, unclaimed_waits{0}, wait_ns{0};
        std::atomic<u64> push_calls{0}, push_items{0}, push_wakes{0}, push_notifies{0};
        std::atomic<u64> batches{0}, batch_items{0}, apply_ns{0}, pickup_ns{0};
        std::atomic<u64> steals{0}, steal_items{0}, steal_apply_ns{0};
        /// Highest `enqueued` value the Protector has swapped out of `pending`.
        std::atomic<u64> claimed{0};
        u64 log_checks = 0; // CP only
        u64 log_frames = 0; // CP only

        Snapshot Take() {
            const auto take = [](std::atomic<u64>& c) {
                return c.exchange(0, std::memory_order_relaxed);
            };
            return {take(fence_calls),   take(fence_waits),    take(unclaimed_waits),
                    take(wait_ns),       take(push_calls),     take(push_items),
                    take(push_wakes),    take(push_notifies),  take(batches),
                    take(batch_items),   take(apply_ns),       take(pickup_ns),
                    take(steals),        take(steal_items),    take(steal_apply_ns)};
        }
    };

    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<DeferredWatch> pending;
    std::atomic<u64> enqueued{0};
    std::atomic<u64> applied{0};
    const bool wake_stats = WakeStatsEnabled();
    const int steal_mode = StealMode();
    const u64 spin_ns = SpinNs();
    u64 claimed_seq = 0;                  // under `mutex`: `enqueued` value of the latest claim
    bool idle = false;                    // under `mutex`, SHADPS4_PROTECT_STEAL only (see Push)
    std::vector<DeferredWatch> steal_work; // CP only (StealAndApply), empty between fences
    bool sleeping = false;                // under `mutex`, stats only
    Clock::time_point batch_first_push{}; // under `mutex`, stats only
    mutable WakeStats stats;
    std::jthread thread;
};

/// Guest-visible completion signals so far (every ProtectFence call: EOP / EOS / ASC RELEASE_MEM
/// labels and the other fenced writes). Read by SHADPS4_DEFERRED_COPY_CENSUS.
inline std::atomic<u64> g_guest_label_seq{0};

/// Called by the command processor before it makes any guest-visible completion signal.
inline void ProtectFence() {
    g_guest_label_seq.fetch_add(1, std::memory_order_relaxed);
    if (ProtectQueue::Enabled()) {
        ProtectQueue::Instance().Fence();
    }
}

} // namespace VideoCore
