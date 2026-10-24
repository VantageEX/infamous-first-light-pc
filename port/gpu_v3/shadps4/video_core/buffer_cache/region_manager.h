// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <vector>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/types.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/dirty_epoch.h"
#include "video_core/buffer_cache/gpu_write_log.h"
#include "video_core/buffer_cache/fast_settings.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

/// SHADPS4_FAULT_WATCH_OUTSIDE=1: page-watcher updates (the VirtualProtect calls) that a guest
/// write fault's invalidation computes under a region lock, applied after the lock is released.
/// Measured 2026-09-24 on 386F5167: the CP spun on these region locks for ~3.3% of its wall
/// time while fault handlers held them across VirtualProtect. Watcher counts commute as long as no
/// removal is applied before its own add (the 1-bit read count asserts on underflow; q56b), and each
/// update sets protection from the counts under the page manager's own locks, so a reordered
/// update leaves the same final state; at worst the faulting write faults once more.
struct DeferredWatch {
    const PageManager* tracker;
    VAddr addr;
    RegionBits mask;
    bool track;
    bool is_read;
};
inline thread_local std::vector<DeferredWatch>* g_deferred_watch = nullptr;

/// q56b (SHADPS4_READ_UNTRACK_FENCE=1): a read-watch removal applied directly (not deferred) first waits until every
/// read protection queued for the Protector (SHADPS4_DEFER_READ_PROTECT) is applied. The counts are 1-bit per page:
/// a removal that overtakes the queued add of the same page asserts "Not enough watchers" (s867, s915). Defined in
/// buffer_cache.cpp (protect_queue.h includes this header).
bool ReadUntrackFenceOn();
void FenceDeferredReadProtects();

inline void ApplyDeferredWatches(std::vector<DeferredWatch>& list) {
    for (DeferredWatch& w : list) {
        if (w.track) {
            if (w.is_read) {
                w.tracker->UpdatePageWatchersForRegion<true, true>(w.addr, w.mask);
            } else {
                w.tracker->UpdatePageWatchersForRegion<true, false>(w.addr, w.mask);
            }
        } else {
            if (w.is_read) {
                w.tracker->UpdatePageWatchersForRegion<false, true>(w.addr, w.mask);
            } else {
                w.tracker->UpdatePageWatchersForRegion<false, false>(w.addr, w.mask);
            }
        }
    }
    list.clear();
}

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
/**
 * Bumped whenever ANY region transitions to possibly-dirty. This is the invalidation signal that
 * makes a clean-result memo sound: if a range was verified clean at epoch E and the epoch still
 * reads E, nothing anywhere has been marked dirty since, so that range is still clean.
 *
 * Deliberately global and therefore conservative - dirt in an unrelated region invalidates every
 * memo. A per-region epoch would hit more often and is the obvious next step if the hit rate
 * measures poorly, but it costs a range-max lookup per query, which is the thing being avoided.
 */

class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    /**
     * True if this region may have any CPU-modified page. Read WITHOUT the region lock, so it is
     * deliberately conservative: a stale `true` only costs the caller the work it would have done
     * anyway, while a stale `false` would skip a real upload and render stale geometry.
     *
     * False therefore has to imply "the cpu bitset was empty". That holds because the bitset has
     * exactly two mutators, `ChangeRegionState<Type::CPU>` and `ForEachModifiedRange`, both under
     * the region lock, and both maintain this flag: it is raised BEFORE bits are set (so a reader
     * seeing false cannot be racing a set that already happened), and lowered only after observing
     * `None()` on the whole region.
     *
     * Starts true because the constructor does `cpu.Fill()` - a fresh region is entirely dirty.
     */
    bool MayHaveCpuModified() const noexcept {
        return any_cpu.load(std::memory_order_acquire);
    }

    /**
     * Point this region at its bit in the tracker's skippable-region summary. The bit means
     * "this region exists AND its cpu bitset is empty", so it is only ever SET from a state we
     * have just observed clean, and CLEARED before any bit is set - same ordering discipline as
     * any_cpu, one level up. A region with no manager has no bit, which reads as not-skippable,
     * which is the safe default (that page still has to be visited so it can be created).
     */
    void SetSummarySlot(std::atomic<u64>* word, u64 mask) {
        summary_word = word;
        summary_mask = mask;
        // A fresh region is entirely dirty (cpu.Fill()), so it starts not-skippable.
        summary_word->fetch_and(~summary_mask, std::memory_order_release);
    }

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    /// Number of GPU write markings this region has received. Atomic so the CP's lock-free
    /// written-range fast path (SHADPS4_UPLOAD_PRECHECK) can bump it without the region lock.
    u64 GpuWriteSeq() const noexcept {
        return gpu_write_seq.load();
    }

    /**
     * SHADPS4_UPLOAD_PRECHECK, read-only syncs: false proves the range has no CPU-dirty page
     * as far as this caller can be required to see. Read without the lock. CPU bits are only set
     * off the CP (write faults), before the faulting page is unprotected, so any guest write that
     * precedes the submit being processed has its bit visible here; a bit set after this read is
     * the same as a fault landing just after the locked walk released its lock.
     */
    bool CpuDirtyInRangeRelaxed(u64 offset, u64 size) const noexcept {
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        return cpu.AnyInRangeRelaxed(start_page, end_page);
    }

    /**
     * SHADPS4_UPLOAD_PRECHECK, written syncs, CP thread only: true when the locked written path
     * would change nothing but gpu_write_seq - no CPU-dirty page to upload (relaxed read, as
     * above) and every page already GPU-modified. GPU bits are mutated only on the CP, so that
     * read is exact; with them unchanged, `readable` needs no update either (under Precise it is
     * re-synced to ~gpu on every GPU-bit change).
     */
    bool WrittenRangeSettled(u64 offset, u64 size) const noexcept {
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return true;
        }
        return !cpu.AnyInRangeRelaxed(start_page, end_page) && gpu.AllInRange(start_page, end_page);
    }

    /// SHADPS4_GPU_MOD_RELAXED, CP thread only: is any page of the range GPU-modified, read
    /// without the region lock. GPU bits are set and cleared only on the CP (every unmark site
    /// runs in a SendCommand callback, CP-inline or in PopPendingOperations; Opus review
    /// 2026-10-01), so this read is exact. A clear moved off the CP would still be safe; a SET
    /// off the CP would not (=2 cross-checks against the locked read).
    bool GpuModifiedInRangeRelaxed(u64 offset, u64 size) const noexcept {
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        return gpu.AnyInRangeRelaxed(start_page, end_page);
    }

    /// Records a GPU write that sets no tracker bit, so in-flight fault downloads of this
    /// region drop their snapshot. Call under the region lock.
    void BumpGpuWriteSeq() noexcept {
        ++gpu_write_seq;
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        if constexpr (enable) {
            if constexpr (type == Type::CPU) {
                // Raise the summary BEFORE the bits, so a lock-free reader can never observe
                // false while a set that precedes it in program order has already landed.
                // SHADPS4_BACK_CLEAN: log the pages first (under the region lock, which every
                // caller holds), so the back's private "verified clean" bits drop them.
                DirtyLog::Append((cpu_addr >> TRACKER_PAGE_BITS) + start_page,
                                 std::min<size_t>(end_page, NUM_PAGES_PER_REGION) - start_page);
                PublishCpuDirty([&] {
                    MarkDirty();
                    bits.SetRange(start_page, end_page);
                });
                BumpGranules(cpu_addr + start_page * TRACKER_BYTES_PER_PAGE,
                             cpu_addr + std::min<size_t>(end_page, NUM_PAGES_PER_REGION) *
                                            TRACKER_BYTES_PER_PAGE);
            } else {
                // Counts every GPU write marking, including one that finds all bits already
                // set, so an in-flight fault download can tell its snapshot went stale.
                ++gpu_write_seq;
                BumpGranules(cpu_addr + start_page * TRACKER_BYTES_PER_PAGE,
                             cpu_addr + std::min<size_t>(end_page, NUM_PAGES_PER_REGION) *
                                            TRACKER_BYTES_PER_PAGE);
                if (GpuWriteLog::Mode() != 0) {
                    GpuWriteLog::Append(
                        cpu_addr + start_page * TRACKER_BYTES_PER_PAGE,
                        cpu_addr + std::min<size_t>(end_page, NUM_PAGES_PER_REGION) *
                                       TRACKER_BYTES_PER_PAGE);
                }
                bits.SetRange(start_page, end_page);
            }
        } else {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                if (bits.None()) {
                    MarkClean();
                }
            }
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (ReadbacksMode() == GpuReadbacksMode::Precise) {
            UpdateProtection<enable, true>();
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();

        // Nothing modified in the queried range means there is no work in this call at all, and
        // the common case by a wide margin: measured on the per-draw upload path, ~76% of calls
        // find no CPU-modified page. Without this the call still materialises a 1024-bit masked
        // copy, runs UnsetRange over it, and diffs the whole 4 MB region's protection state.
        //
        // Only Type::CPU. The skip is sound because `writeable` is re-synced to `cpu` on every
        // path that mutates `cpu` (ChangeRegionState<CPU> and this function, both under the
        // region lock), so with no bit set in range the UnsetRange is a no-op and
        // UpdateProtection's `cpu ^ writeable` diff is necessarily empty. The GPU side has no
        // such invariant: ChangeRegionState<GPU> only touches `readable` under Precise, while
        // this function updates it whenever readbacks are not Disabled, so under Relaxed `gpu`
        // and `readable` legitimately diverge and the diff must still be taken.
        if constexpr (type == Type::CPU) {
            if (!bits.AnyInRange(start_page, end_page)) {
                return;
            }
        }

        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                if (bits.None()) {
                    MarkClean();
                }
                UpdateProtection<true, false>();
            } else if (ReadbacksMode() != GpuReadbacksMode::Disabled) {
                UpdateProtection<false, true>();
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /// Guest-fault marking, called under the lock BEFORE ChangeRegionState<CPU, true>: pages that
    /// were clean become "recent" for `epoch`. Pages already dirty keep their older status, since
    /// they may hold data written before the current submission. See submit_epoch.h.
    void MarkRecentFromFault(u64 dirty_addr, u64 size, u64 epoch) noexcept {
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page = std::min<size_t>(
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE),
            NUM_PAGES_PER_REGION);
        if (recent_epoch.load(std::memory_order_relaxed) != epoch) {
            recent.Clear();
            recent_epoch.store(epoch, std::memory_order_release);
        }
        for (size_t page = start_page; page < end_page; ++page) {
            if (!cpu.Get(page)) {
                recent.Set(page);
            }
        }
    }

    /// True if pages may be held for `epoch`. Lock-free pre-check for ClearRecent: a page can
    /// only become recent through a guest fault, which stores this before unprotecting it.
    bool MayHaveRecent(u64 epoch) const noexcept {
        return recent_epoch.load(std::memory_order_acquire) == epoch;
    }

    /// Non-fault CPU marking (CP or HLE writes) is never deferred. Call under the lock.
    void ClearRecent(u64 dirty_addr, u64 size) noexcept {
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page = std::min<size_t>(
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE),
            NUM_PAGES_PER_REGION);
        if (end_page > start_page) {
            recent.UnsetRange(start_page, end_page);
        }
    }

    /**
     * ForEachModifiedRange<Type::CPU, true> for read-only syncs under upload deferral: pages
     * first dirtied by a guest fault in `epoch` stay dirty and writable. Returns how many dirty
     * pages were deferred; the caller must not memoise the range as clean if that is nonzero.
     * Protection stays exact: `writeable` is re-synced to `cpu`, so only uploaded pages are
     * re-protected, and contiguous ones in a single call.
     */
    template <typename Func>
    u64 ForEachModifiedRangeDeferred(VAddr query_cpu_range, s64 size, u64 epoch, Func&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return 0;
        }
        if (!cpu.AnyInRange(start_page, end_page)) {
            return 0;
        }
        RegionBits mask(cpu, start_page, end_page);
        u64 deferred = 0;
        if (recent_epoch.load(std::memory_order_relaxed) == epoch &&
            recent.AnyInRange(start_page, end_page)) {
            const RegionBits held = mask & recent;
            for (const auto& [start, end] : held) {
                deferred += end - start;
            }
            mask &= ~recent;
            if (mask.None()) {
                return deferred;
            }
            cpu &= ~mask;
        } else {
            cpu.UnsetRange(start_page, end_page);
        }
        if (cpu.None()) {
            MarkClean();
        }
        UpdateProtection<true, false>();
        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
        return deferred;
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        const RegionBits& bits = GetRegionBits<type>();
        return bits.AnyInRange(start_page, end_page);
    }

    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        // This is the only path out of the tracker that reaches a VirtualProtect syscall. Count
        // and time it: it is the one candidate in the per-draw sync path that can plausibly cost
        // microseconds, and bitset arithmetic cannot.
        // Two users: SHADPS4_DEFER_READ_PROTECT (the CP's read tightens, handed to the Protector)
        // and SHADPS4_FAULT_WATCH_OUTSIDE (a guest write fault's window only runs
        // ChangeRegionState<CPU>, i.e. write-side updates). Do not filter by is_read here: on
        // 2026-09-28 an `&& !is_read` guard silently turned DEFER_READ_PROTECT off (a34).
        if (g_deferred_watch) {
            g_deferred_watch->push_back({tracker, cpu_addr, mask, track, is_read});
            return;
        }
        if constexpr (!track && is_read) {
            if (ReadUntrackFenceOn()) {
                FenceDeferredReadProtects(); // q56b
            }
        }
        const auto t_protect = Common::FaultMonitor::Now();
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
        Common::FaultMonitor::CountProtect<is_read>(Common::FaultMonitor::SinceNs(t_protect));
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    /// Raise the per-region flag and drop the tracker-wide skippable bit. Order matters: the
    /// coarse bit must stop advertising skippability before the fine flag admits dirt, and both
    /// must precede the bits themselves.
    void MarkDirty() {
        if (summary_word) {
            summary_word->fetch_and(~summary_mask, std::memory_order_release);
        }
        any_cpu.store(true, std::memory_order_release);
    }

    /// Called only after observing None() over the whole region, under the region lock.
    void MarkClean() {
        any_cpu.store(false, std::memory_order_release);
        if (summary_word) {
            summary_word->fetch_or(summary_mask, std::memory_order_release);
        }
    }

    std::atomic<u64>* summary_word{nullptr};
    u64 summary_mask{0};
    // Conservative summary of `cpu`, readable without the lock. See MayHaveCpuModified().
    std::atomic<bool> any_cpu{true};
    std::atomic<u64> gpu_write_seq{0};
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
    /// CPU-dirty pages first dirtied by a guest fault during `recent_epoch`. Stale bits for pages
    /// that are no longer dirty are harmless: the deferred walk intersects with `cpu`.
    RegionBits recent;
    std::atomic<u64> recent_epoch{0};
};

} // namespace VideoCore
