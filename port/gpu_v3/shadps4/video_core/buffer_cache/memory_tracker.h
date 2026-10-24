// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <type_traits>
#include <vector>

#include "common/alignment.h"
#include "common/buffer_sample_probe.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/protect_queue.h"
#include "video_core/buffer_cache/region_manager.h"
#include "video_core/buffer_cache/submit_epoch.h"

namespace VideoCore {

// Environment flags read on every upload walk or fault, read once at startup: a function-local
// static costs a guard check per call.
inline const bool g_upload_clean_skip = [] {
    const char* env = std::getenv("SHADPS4_UPLOAD_CLEAN_SKIP");
    return env == nullptr || std::atoi(env) != 0;
}();
inline const bool g_upload_precheck = [] {
    const char* env = std::getenv("SHADPS4_UPLOAD_PRECHECK");
    return env != nullptr && env[0] == '1';
}();
/// SHADPS4_GPU_MOD_RELAXED=1 (=2 verify: also takes the locked read and compares), default off.
inline const int g_gpu_mod_relaxed = [] {
    const char* env = std::getenv("SHADPS4_GPU_MOD_RELAXED");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();
inline const bool g_fault_watch_outside = [] {
    const char* env = std::getenv("SHADPS4_FAULT_WATCH_OUTSIDE");
    return env != nullptr && env[0] == '1';
}();

class MemoryTracker {
public:
    static constexpr size_t MAX_CPU_PAGE_BITS = 40;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - TRACKER_HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;
    using UploadPath = Common::BufferSampleProbe::UploadPath;

public:
    explicit MemoryTracker(PageManager& tracker_) : tracker{&tracker_} {}
    ~MemoryTracker() = default;

    /// SHADPS4_RMEMO_CENSUS: the last BackClean read sync on the back. census_bc: 0 = the call did
    /// not take the BackClean path (the caller resets it), 1 = no unverified page, 2 = walked.
    /// census_check_cyc: rdtsc cycles from entry to the clean / walk decision.
    u8 census_bc{};
    u64 census_check_cyc{};

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                std::scoped_lock lk{manager->lock};
                return manager->template IsRegionModified<Type::CPU>(offset, size);
            });
    }

    /// Returns true if a region has been modified from the GPU
    bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        if (g_gpu_mod_relaxed && Common::FaultMonitor::t_is_cp_thread) {
            // No lock on the CP: see RegionManager::GpuModifiedInRangeRelaxed. The locked read
            // was 1.0% of the CP back in fire (a41 profile), all of it in the spin lock.
            const bool relaxed = IteratePages<false>(
                query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                    return manager->GpuModifiedInRangeRelaxed(offset, size);
                });
            if (g_gpu_mod_relaxed == 2) [[unlikely]] {
                const bool locked = IteratePages<false>(
                    query_cpu_addr, query_size,
                    [](RegionManager* manager, u64 offset, size_t size) {
                        std::scoped_lock lk{manager->lock};
                        return manager->template IsRegionModified<Type::GPU>(offset, size);
                    });
                static u64 checks = 0, mismatches = 0;
                ++checks;
                if (locked != relaxed && ++mismatches <= 16) {
                    LOG_CRITICAL(Render, "GpuModRelaxed mismatch: {:#x}+{:#x} relaxed {} locked {}",
                                 query_cpu_addr, query_size, relaxed, locked);
                }
                if ((checks & 0x3ffffff) == 0) {
                    LOG_INFO(Render, "GpuModRelaxed: checks={} mismatches={}", checks, mismatches);
                }
            }
            return relaxed;
        }
        return IteratePages<false>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                std::scoped_lock lk{manager->lock};
                return manager->template IsRegionModified<Type::GPU>(offset, size);
            });
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                if (g_epoch_defer_upload) {
                                    manager->ClearRecent(manager->GetCpuAddr() + offset, size);
                                }
                                manager->template ChangeRegionState<Type::CPU, true>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// The CP wrote these guest bytes itself without a fault (DMA fast paths): never defer them.
    void ClearRecent(VAddr cpu_addr, u64 size) {
        if (!g_epoch_defer_upload) {
            return;
        }
        const u64 epoch = g_submit_epoch.load(std::memory_order_acquire);
        IteratePages<false>(cpu_addr, size, [epoch](RegionManager* manager, u64 offset,
                                                    size_t size) {
            if (manager->MayHaveRecent(epoch)) {
                std::scoped_lock lk{manager->lock};
                manager->ClearRecent(manager->GetCpuAddr() + offset, size);
            }
        });
    }

    /// Bumped before any GPU-modified bit is cleared (SHADPS4_WRITTEN_MEMO). CP thread only.
    static inline std::atomic<u64> gpu_clear_epoch{0};

    /// True when no CPU-dirty page is recorded in any region the range touches (lock-free).
    bool RangeCpuClean(VAddr cpu_addr, u64 size) const {
        return AllRegionsSkippable(cpu_addr, size);
    }

    /// SHADPS4_BIND_MEMO_STATS: true if any tracker page of the range is CPU-dirty, or a 4 MB
    /// region it touches has no manager yet (counted as dirty). Lock-free and exact per page, for
    /// the reason given at RegionManager::CpuDirtyInRangeRelaxed. CP thread.
    bool CpuDirtyRelaxed(VAddr cpu_addr, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const size_t first = cpu_addr >> TRACKER_HIGHER_PAGE_BITS;
        const size_t last = (cpu_addr + size - 1) >> TRACKER_HIGHER_PAGE_BITS;
        if (last >= NUM_HIGH_PAGES) {
            return true;
        }
        u64 offset = cpu_addr & TRACKER_HIGHER_PAGE_MASK;
        u64 remaining = size;
        for (size_t index = first; index <= last; ++index) {
            const RegionManager* manager = top_tier[index];
            const u64 amount = std::min<u64>(TRACKER_HIGHER_PAGE_SIZE - offset, remaining);
            if (!manager || manager->CpuDirtyInRangeRelaxed(offset, amount)) {
                return true;
            }
            offset = 0;
            remaining -= amount;
        }
        return false;
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ChangeRegionState<Type::GPU, false>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// Removes all protection from a page and ensures GPU data has been flushed if requested
    /// `from_guest_fault`: the call comes from a guest thread's write fault (not the CP, not an
    /// HLE write), so a newly dirtied page may be deferred. See submit_epoch.h.
    /// `block_bytes` (SHADPS4_HEAP_WRITE_BLOCK, 0 = off): when the whole aligned block holding
    /// the range has no GPU-modified page, mark and unprotect the block instead of the range,
    /// deciding under the region lock that also marks it (no race with the back marking a page
    /// GPU-modified between the check and the marking). Returns true when the block was taken.
    bool InvalidateRegion(VAddr cpu_addr, u64 size, bool from_guest_fault, auto&& on_flush,
                          u64 block_bytes = 0) noexcept {
        const bool defer = g_epoch_defer_upload;
        const u64 submit_epoch = defer ? g_submit_epoch.load(std::memory_order_acquire) : 0;
        thread_local std::vector<DeferredWatch> deferred_watches;
        if (block_bytes != 0) {
            // An aligned power-of-two block <= 4 MB lies inside one region manager.
            const VAddr block = Common::AlignDown(cpu_addr, block_bytes);
            RegionManager* manager = block_bytes <= TRACKER_HIGHER_PAGE_SIZE
                                         ? top_tier[block >> TRACKER_HIGHER_PAGE_BITS]
                                         : nullptr;
            if (manager && cpu_addr + size <= block + block_bytes) {
                if (g_fault_watch_outside && from_guest_fault) {
                    g_deferred_watch = &deferred_watches;
                }
                const u64 offset = block & TRACKER_HIGHER_PAGE_MASK;
                bool widened = false;
                {
                    std::scoped_lock lk{manager->lock};
                    if (!(ReadbacksMode() != GpuReadbacksMode::Disabled &&
                          manager->template IsRegionModified<Type::GPU>(offset, block_bytes))) {
                        if (defer) {
                            if (from_guest_fault) {
                                manager->MarkRecentFromFault(block, block_bytes, submit_epoch);
                            } else {
                                manager->ClearRecent(block, block_bytes);
                            }
                        }
                        manager->template ChangeRegionState<Type::CPU, true>(block, block_bytes);
                        widened = true;
                    }
                }
                if (g_deferred_watch) {
                    g_deferred_watch = nullptr;
                    ApplyDeferredWatches(deferred_watches);
                }
                if (widened) {
                    return true;
                }
                // A GPU-modified page in the block: the range alone, on the path below.
            }
        }
        IteratePages<false>(
            cpu_addr, size,
            [&on_flush, defer, from_guest_fault, submit_epoch](RegionManager* manager, u64 offset,
                                                               size_t size) {
                if (g_fault_watch_outside && from_guest_fault) {
                    g_deferred_watch = &deferred_watches;
                }
                const bool should_flush = [&] {
                    // Perform both the GPU modification check and CPU state change with the lock
                    // in case we are racing with GPU thread trying to mark the page as GPU
                    // modified. If we need to flush the flush function is going to perform CPU
                    // state change.
                    std::scoped_lock lk{manager->lock};
                    if (ReadbacksMode() != GpuReadbacksMode::Disabled &&
                        manager->template IsRegionModified<Type::GPU>(offset, size)) {
                        return true;
                    }
                    if (defer) {
                        if (from_guest_fault) {
                            manager->MarkRecentFromFault(manager->GetCpuAddr() + offset, size,
                                                         submit_epoch);
                        } else {
                            manager->ClearRecent(manager->GetCpuAddr() + offset, size);
                        }
                    }
                    manager->template ChangeRegionState<Type::CPU, true>(
                        manager->GetCpuAddr() + offset, size);
                    return false;
                }();
                if (g_deferred_watch) {
                    g_deferred_watch = nullptr;
                    ApplyDeferredWatches(deferred_watches);
                }
                if (should_flush) {
                    on_flush();
                }
            });
        return false;
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    /// SHADPS4_UPLOAD_CLEAN_SKIP=0 disables the pre-lock skip, so the fix can be A/B'd against
    /// itself in one binary. Default on.
    static bool CleanSkipEnabled() {
        return g_upload_clean_skip;
    }

    /**
     * Direct-mapped memo of "this exact range was entirely clean at epoch E".
     *
     * SynchronizeBuffer runs 7.87x per draw - ~49,400 times a frame - and the same bindings repeat
     * across draws, so the same question is asked over and over. 99.6% of the answers are "nothing
     * modified". g_dirty_epoch makes caching that answer sound: it is bumped whenever ANY region is
     * marked dirty, so an unchanged epoch proves no page anywhere became dirty since the check.
     *
     * Direct-mapped and fixed size so a lookup is a hash and one compare - a map would cost more
     * than the scan it replaces. A collision or a stale entry costs only a redundant scan; it can
     * never produce a false "clean", because the epoch is compared on every hit.
     */
    static constexpr size_t CLEAN_MEMO_SIZE = 8192;
    struct CleanMemoEntry {
        VAddr addr;
        u64 size;
        u64 epoch;
    };

    bool MemoSaysClean(VAddr addr, u64 size, u64 epoch) const {
        const auto& e = clean_memo[MemoSlot(addr, size)];
        return e.epoch == epoch && e.addr == addr && e.size == size;
    }

    void MemoRecordClean(VAddr addr, u64 size, u64 epoch) {
        clean_memo[MemoSlot(addr, size)] = {addr, size, epoch};
    }

    static size_t MemoSlot(VAddr addr, u64 size) {
        u64 h = addr * 0x9E3779B97F4A7C15ULL;
        h ^= size * 0xC2B2AE3D27D4EB4FULL;
        h ^= h >> 29;
        return static_cast<size_t>(h % CLEAN_MEMO_SIZE);
    }

    /// SHADPS4_UPLOAD_PRECHECK=1 (default off): skip the region lock when a lock-free read of
    /// the exact range proves the locked work would do nothing. See RegionManager::
    /// CpuDirtyInRangeRelaxed / WrittenRangeSettled for why the reads are sound.
    static bool UploadPrecheckEnabled() {
        return g_upload_precheck;
    }

    /// Precheck counters (CP thread; logged under the flag). "visits" are region visits that
    /// reached the lock before this change could skip them.
    struct PrecheckStats {
        u64 written_calls, written_fast, read_visits, read_skipped;
    };
    static PrecheckStats& Precheck() {
        static thread_local PrecheckStats stats{};
        return stats;
    }

    void MaybeLogPrecheck() {
        auto& st = Precheck();
        if (st.written_calls + st.read_visits < (u64{1} << 22)) {
            return;
        }
        LOG_INFO(Render,
                 "UploadPrecheck: written {} fast {:.1f}% | read region visits {} skipped {:.1f}%",
                 st.written_calls,
                 st.written_calls ? 100.0 * st.written_fast / st.written_calls : 0.0,
                 st.read_visits, st.read_visits ? 100.0 * st.read_skipped / st.read_visits : 0.0);
        st = {};
    }

    /// Written-range fast path: true if every touched region exists and is settled; then bumps
    /// each region's GPU write sequence, which is all the locked path would have changed.
    bool TryWrittenFastPath(VAddr cpu_addr, u64 size) {
        if (size == 0) {
            return false;
        }
        const size_t first = cpu_addr >> TRACKER_HIGHER_PAGE_BITS;
        const size_t last = (cpu_addr + size - 1) >> TRACKER_HIGHER_PAGE_BITS;
        u64 offset = cpu_addr & TRACKER_HIGHER_PAGE_MASK;
        u64 remaining = size;
        for (size_t index = first; index <= last; ++index) {
            const RegionManager* manager = top_tier[index];
            const u64 amount = std::min<u64>(TRACKER_HIGHER_PAGE_SIZE - offset, remaining);
            if (!manager || !manager->WrittenRangeSettled(offset, amount)) {
                return false;
            }
            offset = 0;
            remaining -= amount;
        }
        for (size_t index = first; index <= last; ++index) {
            top_tier[index]->BumpGpuWriteSeq();
        }
        LogGpuWrite(cpu_addr, size);
        return true;
    }

    /// SHADPS4_BACK_CLEAN (see back_clean.h): read-only syncs on the command processor answer
    /// from the private "verified clean" bits; only unverified page runs take the locked walk.
    UploadPath ForEachUploadRangeBackClean(VAddr query_cpu_range, u64 query_size, auto&& func,
                                           auto&& on_upload) {
        const u64 census_c0 = g_rmemo_census ? __rdtsc() : 0;
        auto& bc = back_clean;
        auto& runs = back_clean_runs;
        ++bc.stats.queries;
        if ((bc.stats.queries & (g_back_clean == 2 ? 0x3fffffULL : 0x3ffffffULL)) == 0) {
            const auto& s = bc.stats;
            LOG_INFO(Render,
                     "BackClean: mode={} queries={} clean={} ({:.2f}%) runs={} run_pages={} "
                     "drained={} resets={} raced={} mismatches={}",
                     g_back_clean, s.queries, s.clean, 100.0 * s.clean / s.queries, s.runs,
                     s.run_pages, s.drained, s.resets, s.raced, s.mismatches);
        }
        bc.Sync();
        runs.clear();
        bc.CollectUnverified(query_cpu_range, query_size, runs);
        if (g_back_clean == 2) [[unlikely]] {
            // Verify every verified sub-range of the query (the gaps between the unverified
            // runs) against the tracker's own bits. A dirty bit there had its log entry
            // appended before it was set, so one more drain must drop the page; if the
            // sub-range is still verified after it, the private bits are wrong. Diagnostic
            // read (relaxed), sound on x86.
            const u64 first = query_cpu_range >> TRACKER_PAGE_BITS;
            const u64 end = (query_cpu_range + query_size + TRACKER_BYTES_PER_PAGE - 1) >>
                            TRACKER_PAGE_BITS;
            auto& gaps = back_clean_gaps;
            gaps.clear();
            u64 at = first;
            for (const auto& [first_page, count] : runs) {
                if (first_page > at) {
                    gaps.emplace_back(at, first_page - at);
                }
                at = first_page + count;
            }
            if (end > at) {
                gaps.emplace_back(at, end - at);
            }
            bool resync = false;
            auto& bad = back_clean_bad;
            bad.clear();
            for (const auto& [gap_page, gap_count] : gaps) {
                const VAddr gap_addr = gap_page << TRACKER_PAGE_BITS;
                const u64 gap_size = gap_count << TRACKER_PAGE_BITS;
                if (!CpuDirtyRelaxed(gap_addr, gap_size)) {
                    continue;
                }
                bc.Sync();
                resync = true;
                auto& again = back_clean_again;
                again.clear();
                bc.CollectUnverified(gap_addr, gap_size, again);
                if (again.empty()) {
                    if (++bc.stats.mismatches <= 16) {
                        LOG_CRITICAL(Render,
                                     "BackClean mismatch: {:#x}+{:#x} verified clean, tracker "
                                     "dirty (query {:#x}+{:#x})",
                                     gap_addr, gap_size, query_cpu_range, query_size);
                    }
                    bad.emplace_back(gap_page, gap_count);
                } else {
                    ++bc.stats.raced;
                }
            }
            if (resync) {
                runs.clear();
                bc.CollectUnverified(query_cpu_range, query_size, runs);
                runs.insert(runs.end(), bad.begin(), bad.end());
            }
        }
        if (g_rmemo_census || g_read_sync_fast == 2) [[unlikely]] {
            census_check_cyc = __rdtsc() - census_c0;
            census_bc = runs.empty() ? 1 : 2;
        }
        if (runs.empty()) [[likely]] {
            ++bc.stats.clean;
            on_upload();
            return UploadPath::Memo;
        }
        for (const auto& [first_page, count] : runs) {
            ++bc.stats.runs;
            bc.stats.run_pages += count;
            // Always under the region lock: a fault handler that has logged its pages but not
            // yet set their bits holds it (no lock-free "region is clean" shortcut here).
            IteratePages<true>(first_page << TRACKER_PAGE_BITS, count << TRACKER_PAGE_BITS,
                               [&func](RegionManager* manager, u64 offset, size_t size) {
                                   std::scoped_lock lk{manager->lock};
                                   manager->template ForEachModifiedRange<Type::CPU, true>(
                                       manager->GetCpuAddr() + offset, size, func);
                               });
        }
        // Before on_upload(): nothing may drain the log between a walk and its MarkVerified (a
        // drained entry for a page not yet marked would be lost). The bits are cleared and the
        // ranges recorded at this point, as in the old path.
        for (const auto& [first_page, count] : runs) {
            bc.MarkVerified(first_page, count);
        }
        on_upload();
        return UploadPath::Walk;
    }

    /// SHADPS4_READ_SYNC_FAST: true when ForEachUploadRange(addr, size, is_written = false) would
    /// take the BackClean path and find no unverified page - the case in which it calls no `func`
    /// and changes no tracker or protection state. Same eligibility as that path, BACK_CLEAN=1 only;
    /// BACK_CLEAN=2 is accepted under READ_SYNC_FAST=2, whose full path then checks the private bits
    /// against the tracker (the gap check). Drains the dirty log, as that path does.
    bool ReadSyncClean(VAddr addr, u64 size) {
        const bool bc_ok = g_back_clean == 1 || (g_back_clean == 2 && g_read_sync_fast == 2);
        if (!bc_ok || g_epoch_defer_upload || size == 0 ||
            !Common::FaultMonitor::t_is_cp_thread || ((addr + size) >> MAX_CPU_PAGE_BITS) != 0) {
            return false;
        }
        back_clean.Sync();
        return back_clean.AllVerified(addr, size);
    }

    /// SHADPS4_READ_SYNC_FAST=2: where ReadSyncClean's drain stopped.
    u64 BackCleanDrainedTo() const noexcept {
        return back_clean.DrainedTo();
    }

    /// SHADPS4_NATIVE_COHERENT: keeps the back's private bits drained while no read binding asks.
    void BackCleanSync() {
        if (g_back_clean != 0) {
            back_clean.Sync();
        }
    }

    UploadPath ForEachUploadRange(VAddr query_cpu_range, u64 query_size, bool is_written,
                                  auto&& func, auto&& on_upload) {
        if (g_back_clean != 0 && !is_written && !g_epoch_defer_upload && query_size != 0 &&
            Common::FaultMonitor::t_is_cp_thread &&
            ((query_cpu_range + query_size) >> MAX_CPU_PAGE_BITS) == 0) {
            return ForEachUploadRangeBackClean(query_cpu_range, query_size, func, on_upload);
        }
        // Three phases with very different costs - scan is bitset work, mark can reach a
        // VirtualProtect syscall. Timed separately so a fix is aimed at the one that is expensive.
        const auto t_scan = Common::FaultMonitor::Now();
        // Shape counters: plain locals in the hot loop, one atomic add at the end. See
        // CountUploadShape in fault_monitor.h for why this is counted and not timed.
        u64 shape_managers = 0;
        u64 shape_ranges = 0;
        u64 shape_skipped = 0;
        const auto counting_func = [&func, &shape_ranges](u64 addr, u64 size) {
            ++shape_ranges;
            func(addr, size);
        };
        // Measured 2026-08-19: 99.63% of these calls find zero modified ranges while visiting
        // 11.24 region managers each - ~555,000 lock-and-scan visits per frame to find ~400
        // ranges, 9.5 ms of a 52 ms frame. The AnyInRange early-out from ed9979cb sits INSIDE
        // ForEachModifiedRange, after the lock is already taken, so it cannot remove the lock.
        // The summary flag can, and is read without it - see RegionManager::MayHaveCpuModified.
        //
        // Only on the !is_written path: when is_written the lock is taken here and released in
        // the second IteratePages below, so skipping would unbalance it. That costs almost
        // nothing - is_written is 4.5% of calls.
        const bool clean_skip = CleanSkipEnabled() && !is_written;
        // Coarse fast path: a 42 MB query that is clean everywhere costs one or two word reads
        // instead of 11 pointer chases. Measured 2026-08-19: queries average 41 MB and 99.6% of
        // them find nothing, so this is the common case by a wide margin. !is_written only, for
        // the same lock-pairing reason as below.
        const u64 epoch = g_dirty_epoch.load(std::memory_order_acquire);
        const bool defer = g_epoch_defer_upload && !is_written;
        const u64 submit_epoch = defer ? g_submit_epoch.load(std::memory_order_acquire) : 0;
        u64 deferred_pages = 0;
        if (clean_skip && MemoSaysClean(query_cpu_range, query_size, epoch)) {
            on_upload();
            Common::FaultMonitor::CountUploadShape(0, 0, 0, query_size);
            Common::FaultMonitor::CountUploadMemoHit();
            return UploadPath::Memo;
        }
        if (clean_skip && AllRegionsSkippable(query_cpu_range, query_size)) {
            MemoRecordClean(query_cpu_range, query_size, epoch);
            on_upload();
            Common::FaultMonitor::CountUploadShape(0, 0, 0, query_size);
            Common::FaultMonitor::CountUploadFastPath();
            return UploadPath::Coarse;
        }
        const bool precheck = UploadPrecheckEnabled();
        if (precheck && is_written) {
            ++Precheck().written_calls;
            if (TryWrittenFastPath(query_cpu_range, query_size)) {
                ++Precheck().written_fast;
                MaybeLogPrecheck();
                on_upload();
                Common::FaultMonitor::CountUploadShape(0, 0, 0, query_size);
                return UploadPath::Coarse;
            }
        }
        IteratePages<true>(query_cpu_range, query_size,
                           [&counting_func, &shape_managers, &shape_skipped, &deferred_pages,
                            clean_skip, is_written, defer, precheck,
                            submit_epoch](RegionManager* manager, u64 offset, size_t size) {
                               ++shape_managers;
                               if (clean_skip && !manager->MayHaveCpuModified()) {
                                   ++shape_skipped;
                                   return;
                               }
                               if (precheck && !is_written) {
                                   ++Precheck().read_visits;
                                   if (!manager->CpuDirtyInRangeRelaxed(offset, size)) {
                                       ++Precheck().read_skipped;
                                       ++shape_skipped;
                                       return;
                                   }
                               }
                               manager->lock.lock();
                               if (defer) {
                                   deferred_pages += manager->ForEachModifiedRangeDeferred(
                                       manager->GetCpuAddr() + offset, size, submit_epoch,
                                       counting_func);
                               } else {
                                   manager->template ForEachModifiedRange<Type::CPU, true>(
                                       manager->GetCpuAddr() + offset, size, counting_func);
                               }
                               if (!is_written) {
                                   manager->lock.unlock();
                               }
                           });
        // A walk that found nothing is exactly as memoisable as the fast path, and this is where
        // the cost actually is: up_ranges_per_call is 0.008, so the walking calls almost always
        // find nothing either. A region is flagged dirty when ANY page in its 4 MB is dirty, while
        // the query may not touch that page - so a call can walk 11 regions, take a lock, and
        // still find zero modified ranges in its own range. Sound for the same reason as above:
        // no ranges found, and the epoch read BEFORE the walk still current, means still clean.
        // If dirt landed mid-walk the epoch has moved and the record simply never hits.
        if (clean_skip && shape_ranges == 0 && deferred_pages == 0) {
            MemoRecordClean(query_cpu_range, query_size, epoch);
        }
        if (precheck) {
            MaybeLogPrecheck();
        }
        if (deferred_pages != 0) {
            Common::FaultMonitor::CountDeferredSync(deferred_pages);
        }
        Common::FaultMonitor::CountUploadShape(shape_managers, shape_ranges, shape_skipped,
                                               query_size);
        const auto t_copy = Common::FaultMonitor::Now();
        on_upload();
        const auto t_mark = Common::FaultMonitor::Now();
        if (!is_written) {
            Common::FaultMonitor::CountUploadPhases(Common::FaultMonitor::DiffNs(t_scan, t_copy),
                                                    Common::FaultMonitor::DiffNs(t_copy, t_mark), 0,
                                                    false);
            return UploadPath::Walk;
        }
        // SHADPS4_DEFER_READ_PROTECT: the read protection this marking triggers goes to the
        // protector thread (see ProtectQueue); the bits change here as before.
        thread_local std::vector<DeferredWatch> read_protects;
        const bool defer_read_protect = ProtectQueue::Enabled();
        if (defer_read_protect) {
            g_deferred_watch = &read_protects;
        }
        IteratePages<false>(query_cpu_range, query_size,
                            [&func, is_written](RegionManager* manager, u64 offset, size_t size) {
                                manager->template ChangeRegionState<Type::GPU, true>(
                                    manager->GetCpuAddr() + offset, size);
                                manager->lock.unlock();
                            });
        if (defer_read_protect) {
            g_deferred_watch = nullptr;
            ProtectQueue::Instance().Push(read_protects);
        }
        Common::FaultMonitor::CountUploadPhases(Common::FaultMonitor::DiffNs(t_scan, t_copy),
                                                Common::FaultMonitor::DiffNs(t_copy, t_mark),
                                                Common::FaultMonitor::SinceNs(t_mark), true);
        return UploadPath::Walk;
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <bool clear>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, auto&& func) {
        IteratePages<false>(query_cpu_range, query_size,
                            [&func](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ForEachModifiedRange<Type::GPU, clear>(
                                    manager->GetCpuAddr() + offset, size, func);
                            });
    }

    /// Appends the GPU write sequence of every 4 MB region the range touches, in address order.
    /// A region without a manager reads as 0, the value a new manager starts from, so a region
    /// created and GPU-written later still compares unequal.
    void SnapshotGpuWriteSeq(VAddr cpu_addr, u64 size, std::vector<u64>& out) {
        ForEachGpuWriteSeq(cpu_addr, size, [&out](u64 seq) {
            out.push_back(seq);
            return true;
        });
    }

    /// True if no region of the range received a GPU write marking since the snapshot.
    bool GpuWriteSeqMatches(VAddr cpu_addr, u64 size, const std::vector<u64>& snapshot) {
        size_t index = 0;
        const bool complete = ForEachGpuWriteSeq(cpu_addr, size, [&](u64 seq) {
            return index < snapshot.size() && snapshot[index++] == seq;
        });
        return complete && index == snapshot.size();
    }

    /// Makes every snapshot over the range compare unequal, for a GPU write into a cache
    /// buffer that takes no tracker marking. A region without a manager holds no GPU-dirty
    /// page, so no fault download can have copied from it and it needs no bump.
    void InvalidateGpuWriteSeq(VAddr cpu_addr, u64 size) {
        IteratePages<false>(cpu_addr, size, [](RegionManager* manager, u64, size_t) {
            std::scoped_lock lk{manager->lock};
            manager->BumpGpuWriteSeq();
        });
        LogGpuWrite(cpu_addr, size);
    }

    /// SHADPS4_FAULT_VETO_EXACT: every BumpGpuWriteSeq caller logs its range (whole tracker
    /// pages), as ChangeRegionState<GPU, true> does for its own bump. Command processor only.
    static void LogGpuWrite(VAddr cpu_addr, u64 size) noexcept {
        if (GpuWriteLog::Mode() != 0 && size != 0) {
            GpuWriteLog::Append(cpu_addr & ~(TRACKER_BYTES_PER_PAGE - 1),
                                (cpu_addr + size + TRACKER_BYTES_PER_PAGE - 1) &
                                    ~(TRACKER_BYTES_PER_PAGE - 1));
        }
    }

private:
    /// Calls func with each touched region's sequence until it returns false.
    template <typename Func>
    bool ForEachGpuWriteSeq(VAddr cpu_addr, u64 size, Func&& func) {
        if (size == 0) {
            return true;
        }
        const size_t first = cpu_addr >> TRACKER_HIGHER_PAGE_BITS;
        const size_t last = (cpu_addr + size - 1) >> TRACKER_HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            RegionManager* manager = top_tier[page_index];
            u64 seq = 0;
            if (manager) {
                std::scoped_lock lk{manager->lock};
                seq = manager->GpuWriteSeq();
            }
            if (!func(seq)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief IteratePages Iterates L2 word manager page table.
     * @param cpu_address Start byte cpu address
     * @param size Size in bytes of the region of iterate.
     * @param func Callback for each word manager.
     * @return
     */
    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        RENDERER_TRACE;
        using FuncReturn = typename std::invoke_result<Func, RegionManager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> TRACKER_HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & TRACKER_HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(TRACKER_HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    void CreateRegion(std::size_t page_index) {
        const VAddr base_cpu_addr = page_index << TRACKER_HIGHER_PAGE_BITS;
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                std::construct_at(&last_pool[i], tracker, 0);
                free_managers.push_back(&last_pool[i]);
            }
        }
        // Each manager tracks a 4_MB virtual address space.
        auto* new_manager = free_managers.back();
        new_manager->SetCpuAddress(base_cpu_addr);
        free_managers.pop_back();
        new_manager->SetSummarySlot(&skippable[page_index / 64], 1ULL << (page_index % 64));
        top_tier[page_index] = new_manager;
    }

    /// True if every 4 MB region the query touches is known to exist and be entirely clean, so
    /// the whole scan can be skipped without walking top_tier at all. A cleared bit (no manager,
    /// or a manager that may be dirty) fails the check, which is the safe direction.
    bool AllRegionsSkippable(VAddr cpu_address, size_t size) const {
        const size_t first = cpu_address >> TRACKER_HIGHER_PAGE_BITS;
        const size_t last = (cpu_address + size - 1) >> TRACKER_HIGHER_PAGE_BITS;
        if (last >= NUM_HIGH_PAGES) {
            return false;
        }
        for (size_t word = first / 64; word <= last / 64; ++word) {
            const size_t lo = std::max(first, word * 64) - word * 64;
            const size_t hi = std::min(last, word * 64 + 63) - word * 64;
            const u64 mask =
                (hi - lo == 63) ? ~0ULL : (((1ULL << (hi - lo + 1)) - 1) << lo);
            if ((skippable[word].load(std::memory_order_acquire) & mask) != mask) {
                return false;
            }
        }
        return true;
    }

    PageManager* tracker;
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::array<RegionManager*, NUM_HIGH_PAGES> top_tier{};
    /// One bit per 4 MB region: set means "exists and entirely clean", i.e. skippable. 32 KB for
    /// the whole 40-bit space. Maintained by RegionManager under its lock, read without it.
    std::array<std::atomic<u64>, NUM_HIGH_PAGES / 64> skippable{};
    std::array<CleanMemoEntry, CLEAN_MEMO_SIZE> clean_memo{};
    BackCleanCache back_clean;
    std::vector<BackCleanCache::Run> back_clean_runs;
    std::vector<BackCleanCache::Run> back_clean_gaps, back_clean_bad, back_clean_again; // =2
};

} // namespace VideoCore
