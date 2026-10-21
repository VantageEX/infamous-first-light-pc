// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/perf_stats.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_manager.h"
#include "video_core/buffer_cache/version_snapshot.h"

namespace VideoCore {

/// Content of a page of guest memory before a guest write discarded its GPU data (write merge).
using PageSnapshot = std::array<u8, BYTES_PER_PAGE>;

/// Copies a page of guest memory through its backing, without faulting (see BufferCache).
void CopyGuestPage(VAddr page, u8* dest);

class MemoryTracker {
public:
    static constexpr u64 MAX_CPU_PAGE_BITS = 40;
    static constexpr u64 MAX_CPU_ADDRESS = 1ULL << MAX_CPU_PAGE_BITS;
    static constexpr u64 NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - HIGHER_PAGE_BITS);
    static constexpr u64 MANAGER_POOL_SIZE = 32;

public:
    explicit MemoryTracker(PageManager& tracker_)
        : tracker{&tracker_}, readbacks_mode{EmulatorSettings.GetReadbacksMode()} {
        // perf_tuning.ini, applied before the game starts, takes precedence.
        auto& tuning = Common::Perf::GetTuning();
        if (!tuning.readback_write_discard_from_ini.load(std::memory_order_relaxed)) {
            tuning.readback_write_discard.store(EmulatorSettings.IsReadbacksWriteDiscard(),
                                                std::memory_order_relaxed);
        }
        for (auto& word : region_summary) {
            word.store(~u64{0}, std::memory_order_relaxed);
        }
        for (auto& region : top_tier) {
            region.store(nullptr, std::memory_order_relaxed);
        }
    }
    ~MemoryTracker() = default;

    /// Hint (BufferCache::PrefetchLookup): the entry of the tracking region of the address,
    /// read first by every check of a range.
    void PrefetchRegion(VAddr cpu_addr) const noexcept {
        if (cpu_addr < MAX_CPU_ADDRESS) {
            PrefetchLine(&top_tier[cpu_addr >> HIGHER_PAGE_BITS]);
        }
    }

    /// Returns true if a region has been modified from the GPU
    bool IsRegionGpuModified(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->template IsRegionModified<Type::GPU>(offset, size);
        });
    }

    /// Calls func() with the region of the page locked if the page is neither CPU nor GPU modified
    /// (guest memory and the GPU copy hold the same data): a guest write to it waits meanwhile.
    /// Returns whether func was called.
    template <typename Func>
    bool WithCleanPageLocked(VAddr page, Func&& func) {
        bool called = false;
        IteratePages(page, BYTES_PER_PAGE, [&](RegionManager* manager, u64 offset, u64 size) {
            const auto bounds = manager->GetBounds(offset, size);
            const bool late = manager->Lock(bounds);
            if (!manager->template IsRegionModified<Type::CPU>(offset, size) &&
                !manager->template IsRegionModified<Type::GPU>(offset, size)) {
                func();
                called = true;
            }
            manager->Unlock(bounds, late);
        });
        return called;
    }

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->template IsRegionModified<Type::CPU>(offset, size);
        });
    }

    /// Returns true if the GPU copy of the region is up to date with guest memory: all its pages
    /// were uploaded and not written by the CPU since (regions never uploaded are not clean, hot
    /// pages never are), and none had a write fault in the last `stable_frames` frames.
    bool IsRegionCpuClean(VAddr cpu_addr, u64 size, u32 stable_frames) noexcept {
        if (size == 0 || cpu_addr >= MAX_CPU_ADDRESS || size > MAX_CPU_ADDRESS - cpu_addr) {
            return false;
        }
        u64 page_index = cpu_addr >> HIGHER_PAGE_BITS;
        u64 page_offset = cpu_addr & HIGHER_PAGE_MASK;
        u64 remaining_size = size;
        while (remaining_size > 0) {
            const u64 amount = std::min(HIGHER_PAGE_SIZE - page_offset, remaining_size);
            RegionManager* region = top_tier[page_index].load(std::memory_order_acquire);
            // Hot pages are not write protected: the guest may have written them since their
            // upload of this epoch, a fresh copy is safer for small reads.
            if (!region || region->HasHotPages(page_offset, amount) ||
                region->template IsRegionModified<Type::CPU>(page_offset, amount) ||
                (stable_frames != 0 &&
                 !region->IsRegionStable(page_offset, amount, stable_frames))) {
                return false;
            }
            ++page_index;
            page_offset = 0;
            remaining_size -= amount;
        }
        return true;
    }

    /// Marks clean the hot pages of the range whose content did not change since their last
    /// upload, see RegionManager::SkipUnchangedHotPages.
    template <typename HashFunc>
    void SkipUnchangedHotPages(VAddr cpu_addr, u64 size, HashFunc&& hash_page) {
        IteratePages(cpu_addr, size, [&hash_page](RegionManager* manager, u64 offset, u64 size) {
            manager->SkipUnchangedHotPages(offset, size, hash_page);
        });
    }

    /// After an upload of the range, see RegionManager::VerifyHotPageHashes.
    template <typename HashFunc>
    void VerifyHotPageHashes(VAddr cpu_addr, u64 size, HashFunc&& hash_page) {
        IteratePages(cpu_addr, size, [&hash_page](RegionManager* manager, u64 offset, u64 size) {
            manager->VerifyHotPageHashes(offset, size, hash_page);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr cpu_addr, u64 size, bool is_write) noexcept {
        IteratePages(cpu_addr, size, [is_write](RegionManager* manager, u64 offset, u64 size) {
            if (is_write) {
                manager->template ChangeRegionState<StateOp::Set, StateOp::Clear>(offset, size);
            } else {
                manager->template ChangeRegionState<StateOp::None, StateOp::Clear>(offset, size);
            }
        });
    }

    /// Mark region as modified from the CPU
    void MarkRegionAsCpuModified(VAddr cpu_addr, u64 size) noexcept {
        IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->template ChangeRegionState<StateOp::Set, StateOp::None>(offset, size);
        });
    }

    /**
     * Sync memo (BufferCache): the CPU modification versions of the regions of the range (at
     * most two, 16 MB regions), to know later whether guest writes may have touched the range
     * since. Returns false when a region of the range does not exist yet, or the range spans
     * more than two regions.
     */
    bool GetCpuVersions(VAddr cpu_addr, u64 size, std::array<u64, 2>& versions) noexcept {
        if (size == 0 || cpu_addr >= MAX_CPU_ADDRESS || size > MAX_CPU_ADDRESS - cpu_addr) {
            return false;
        }
        const u64 first = cpu_addr >> HIGHER_PAGE_BITS;
        const u64 last = (cpu_addr + size - 1) >> HIGHER_PAGE_BITS;
        if (last - first > 1) {
            return false;
        }
        for (u64 i = first; i <= last; ++i) {
            RegionManager* region = top_tier[i].load(std::memory_order_acquire);
            if (!region) {
                return false;
            }
            versions[i - first] = region->CpuVersion();
        }
        if (first == last) {
            versions[1] = 0;
        }
        return true;
    }

    /// Conservative large-buffer memo: every region must exist, be tracked and
    /// retain its CPU modification version. Hot pages cannot participate.
    bool CheckLargeCpuVersions(VAddr address, u64 size, std::vector<u64>& versions, bool compare) {
        if (size == 0 || address >= MAX_CPU_ADDRESS || size > MAX_CPU_ADDRESS - address) return false;
        const u64 first = address >> HIGHER_PAGE_BITS;
        const u64 last = (address + size - 1) >> HIGHER_PAGE_BITS;
        return CheckVersionSnapshot(versions, last - first + 1, compare,
            [&](size_t index) -> std::optional<u64> {
                auto* region = top_tier[first + index].load(std::memory_order_acquire);
                if (!region || region->HasHotPages(0, HIGHER_PAGE_SIZE)) return std::nullopt;
                return region->CpuVersion();
            });
    }

    /// Write merge: whether some pages have a snapshot (cheap check before looking them up).
    [[nodiscard]] bool HasPageSnapshots() const noexcept {
        return num_snapshots.load(std::memory_order_acquire) != 0;
    }

    /// Write merge: takes the content the page had before a guest write discarded its GPU data,
    /// if it was kept. The snapshot is forgotten.
    bool TakePageSnapshot(VAddr page, PageSnapshot& out) {
        if (!HasPageSnapshots()) {
            return false;
        }
        std::scoped_lock lk{snapshots_mutex};
        const auto it = snapshots.find(page);
        if (it == snapshots.end()) {
            return false;
        }
        out = it->second->data;
        snapshots.erase(it);
        num_snapshots.store(static_cast<u32>(snapshots.size()), std::memory_order_release);
        return true;
    }

    /// Write merge: forgets the snapshots of pages not uploaded for max_age frames (the GPU data
    /// they protect is then assumed overwritten by the guest, as without merge).
    void PrunePageSnapshots(u32 frame, u32 max_age) {
        if (!HasPageSnapshots()) {
            return;
        }
        std::scoped_lock lk{snapshots_mutex};
        std::erase_if(snapshots, [frame, max_age](const auto& entry) {
            return frame - entry.second->frame > max_age;
        });
        num_snapshots.store(static_cast<u32>(snapshots.size()), std::memory_order_release);
    }

    /// Registers a readback whose data is not written to guest memory yet. Until it is removed,
    /// CPU writes to the range flush instead of discarding the GPU data (write discard mode),
    /// otherwise the pending readback would overwrite them when it completes.
    void AddPendingReadback(VAddr start, VAddr end) {
        std::scoped_lock lk{pending_readbacks_mutex};
        pending_readbacks.emplace_back(start, end);
        num_pending_readbacks.store(static_cast<u32>(pending_readbacks.size()),
                                    std::memory_order_release);
    }

    void RemovePendingReadback(VAddr start, VAddr end) {
        std::scoped_lock lk{pending_readbacks_mutex};
        const auto it = std::ranges::find(pending_readbacks, std::pair{start, end});
        if (it != pending_readbacks.end()) {
            pending_readbacks.erase(it);
        }
        num_pending_readbacks.store(static_cast<u32>(pending_readbacks.size()),
                                    std::memory_order_release);
    }

    bool IsReadbackPending(VAddr cpu_addr, u64 size) {
        if (num_pending_readbacks.load(std::memory_order_acquire) == 0) {
            return false;
        }
        std::scoped_lock lk{pending_readbacks_mutex};
        return std::ranges::any_of(pending_readbacks, [cpu_addr, size](const auto& range) {
            return cpu_addr < range.second && range.first < cpu_addr + size;
        });
    }

    /// Removes all protection from a page and ensures GPU data has been flushed if requested.
    /// Returns true if GPU modified data was discarded instead of flushed (write discard mode).
    /// A write fault (is_fault) also counts towards making the page hot, see RegionManager.
    bool InvalidateRegion(VAddr cpu_addr, u64 size, bool is_fault, bool from_gpu_thread,
                          auto&& on_flush) noexcept {
        const u32 hot_threshold =
            is_fault ? Common::Perf::GetTuning().hot_page_faults.load(std::memory_order_relaxed)
                     : 0;
        if (readbacks_mode == GpuReadbacksMode::Disabled) {
            IteratePages(cpu_addr, size,
                         [is_fault, from_gpu_thread, hot_threshold](RegionManager* manager,
                                                                    u64 offset, u64 size) {
                             const auto bounds = manager->GetBounds(offset, size);
                             const bool late = manager->Lock(bounds);
                             manager->template ChangeRegionState<StateOp::Set, StateOp::None,
                                                                 false>(offset, size);
                             if (is_fault) {
                                 manager->NoteWriteFrame(offset, size);
                                 manager->NoteWriteFault(offset, size, from_gpu_thread,
                                                         hot_threshold);
                             }
                             manager->Unlock(bounds, late);
                         });
            return false;
        }
        bool should_flush = false;
        bool discarded = false;
        const bool discard =
            readbacks_mode == GpuReadbacksMode::Precise &&
            Common::Perf::GetTuning().readback_write_discard.load(std::memory_order_relaxed);
        // Only small invalidations (write faults) keep the content of the pages they discard.
        const bool merge =
            discard && size <= MaxSnapshotPages * BYTES_PER_PAGE &&
            Common::Perf::GetTuning().readback_write_merge.load(std::memory_order_relaxed);
        IteratePages(
            cpu_addr, size,
            [this, &should_flush, &discarded, discard, merge, cpu_addr, is_fault, from_gpu_thread,
             hot_threshold, request_size = size](RegionManager* manager, u64 offset, u64 size) {
                const auto bounds = manager->GetBounds(offset, size);
                const bool late = manager->Lock(bounds);
                if (is_fault) {
                    manager->NoteWriteFrame(offset, size);
                }
                const bool modified = manager->template IsRegionModified<Type::GPU>(offset, size);
                if (!modified) {
                    if (merge && manager->IsGpuOwned(offset, size) &&
                        !manager->template IsRegionModified<Type::CPU>(offset, size)) {
                        // The upload of these pages left bytes the GPU wrote on the GPU: their
                        // content before this write tells what the guest changed.
                        SnapshotPages(manager->GetCpuAddress() + bounds.start_page * BYTES_PER_PAGE,
                                      bounds.end_page - bounds.start_page + 1);
                    }
                    manager->template ChangeRegionState<StateOp::Set, StateOp::None, false>(offset,
                                                                                            size);
                    if (is_fault) {
                        manager->NoteWriteFault(offset, size, from_gpu_thread, hot_threshold);
                    }
                } else if (discard && !IsReadbackPending(cpu_addr, request_size)) {
                    // Checked under the region lock: a readback registers itself before looking
                    // at the pages, so either it sees the discarded state or this sees it.
                    // The CPU is overwriting memory last written by the GPU. Instead of
                    // stalling on a GPU download, mark it CPU modified and forget the GPU
                    // state, as if readbacks were disabled. CPU reads remain precise.
                    if (merge) {
                        // The GPU data stays on the GPU. Before the pages become writable: their
                        // content before the guest writes, to tell the bytes the guest changed
                        // at their next upload.
                        SnapshotPages(manager->GetCpuAddress() + bounds.start_page * BYTES_PER_PAGE,
                                      bounds.end_page - bounds.start_page + 1);
                        manager->SetGpuOwnedLocked(offset, size);
                    }
                    manager->template ChangeRegionState<StateOp::Set, StateOp::Clear, false>(offset,
                                                                                             size);
                    discarded = true;
                } else {
                    if (discard) {
                        // A pending readback would overwrite the CPU write: flush instead.
                        Common::Perf::Add(Common::Perf::ReadbackDiscardsBlocked);
                    }
                    should_flush = true;
                }
                manager->Unlock(bounds, late);
            });
        if (should_flush) {
            on_flush();
        }
        return discarded;
    }

    /// Write fault window (pc-port shim), see RegionManager::ExtendWriteFault. Returns the
    /// number of pages unprotected.
    u32 ExtendWriteFault(VAddr cpu_addr, u64 size) noexcept {
        u32 pages = 0;
        IteratePages(cpu_addr, size, [&pages](RegionManager* manager, u64 offset, u64 size) {
            pages += manager->ExtendWriteFault(offset, size);
        });
        return pages;
    }

    /// Remembers the GPU tick of the command buffer that writes the range (on the given queue:
    /// 0 graphics, 1 compute), and numbers the write (command processor order). Command
    /// processor thread only.
    u64 SetWriteTick(VAddr cpu_addr, u64 size, u64 tick, u32 queue = 0,
                     u64 generation = 0, u64 native_epoch = 0) noexcept {
        const u64 seq = last_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
        IteratePages(cpu_addr, size,
                     [tick, seq, queue, generation, native_epoch](RegionManager* manager, u64 offset, u64 size) {
                         manager->SetWriteTick(offset, size, tick, seq, queue, generation, native_epoch);
                     });
        return seq;
    }

    /// Numbers a GPU write that is tracked elsewhere (ISS_UNIFIED_MEMORY), so the memos keyed on
    /// CurrentWriteSeq see it. Command processor thread only.
    u64 BumpWriteSeq() noexcept {
        return last_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    /// Cross queue dependencies (asc_queue): whether the compute queue ever accessed the range
    /// (regions not created yet were never accessed by the GPU).
    bool ComputeTouched(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->ComputeTouched(offset, size);
        });
    }

    /// Cross queue dependencies (asc_queue): an access of the range by the queue (0 graphics, 1
    /// compute) in the command buffer of this tick, see RegionManager::CrossQueueAccess. Returns
    /// the latest tick of the other queue to wait for, 0 if none. Command processor thread only.
    u64 CrossQueueAccess(VAddr cpu_addr, u64 size, u32 queue, u64 tick, bool is_write,
                         bool& untouched) noexcept {
        u64 dependency = 0;
        IteratePages<true>(cpu_addr, size, [&](RegionManager* manager, u64 offset, u64 size) {
            dependency = std::max(dependency, manager->CrossQueueAccess(offset, size, queue, tick,
                                                                        is_write, untouched));
        });
        return dependency;
    }

    /// Cross queue dependencies (asc_queue): the queue writes the GPU copy of the range (an
    /// upload) in the command buffer of this tick. Command processor thread only.
    void NoteQueueWrite(VAddr cpu_addr, u64 size, u32 queue, u64 tick) noexcept {
        IteratePages<true>(cpu_addr, size,
                           [queue, tick](RegionManager* manager, u64 offset, u64 size) {
                               manager->NoteQueueWrite(offset, size, queue, tick);
                           });
    }

    /// The latest GPU tick of each queue (0 graphics, 1 compute) that wrote the GPU modified
    /// pages of the range, 0 if none.
    std::array<u64, 2> GetWriteTicks(VAddr cpu_addr, u64 size) noexcept {
        std::array<u64, 2> ticks{};
        IteratePages(cpu_addr, size, [&ticks](RegionManager* manager, u64 offset, u64 size) {
            manager->GetWriteTicks(offset, size, ticks);
        });
        return ticks;
    }

    bool GetPageWriteTicks(VAddr cpu_addr, u64 size, std::array<u64, 2>& ticks,
                           u64 generation, u64 native_epoch) noexcept {
        bool complete = true;
        IteratePages(cpu_addr, size, [&](RegionManager* manager, u64 offset, u64 n) {
            if (!manager->GetPageWriteTicks(offset, n, ticks, generation, native_epoch)) complete = false;
        });
        return complete;
    }

    /// The sequence number of the last GPU write recorded (command processor thread only): a
    /// range whose GetWriteSeq is above the number read earlier was written since.
    u64 CurrentWriteSeq() const noexcept {
        return last_write_seq.load(std::memory_order_relaxed);
    }

    /// The sequence number of the latest GPU write of the range (256 KB granularity), 0 if none.
    u64 GetWriteSeq(VAddr cpu_addr, u64 size) noexcept {
        u64 seq = 0;
        IteratePages(cpu_addr, size, [&seq](RegionManager* manager, u64 offset, u64 size) {
            seq = std::max(seq, manager->GetWriteSeq(offset, size));
        });
        return seq;
    }

    /// Returns the latest GPU tick that wrote the GPU modified pages of the range, 0 if none.
    u64 GetWriteTick(VAddr cpu_addr, u64 size) noexcept {
        u64 tick = 0;
        IteratePages(cpu_addr, size, [&tick](RegionManager* manager, u64 offset, u64 size) {
            tick = std::max(tick, manager->GetWriteTick(offset, size));
        });
        return tick;
    }

    /// The command processor wrote the range directly in guest memory.
    void OnGpuThreadWrite(VAddr cpu_addr, u64 size) noexcept {
        IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->OnGpuThreadWrite(offset, size);
        });
    }

    /// Protects the hot pages of every region again, see RegionManager::CoolHotPages.
    /// Returns the number of pages cooled.
    u32 CoolHotPages() {
        const u32 threshold =
            Common::Perf::GetTuning().hot_page_faults.load(std::memory_order_relaxed);
        u32 cooled = 0;
        for (RegionManager* manager : top_tier) {
            if (manager) {
                cooled += manager->CoolHotPages(threshold);
            }
        }
        return cooled;
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified.
    /// func(addr, size, keep_gpu_owned) is called with the region locked; the uploaded pages are
    /// no longer GPU owned (write merge) unless func calls keep_gpu_owned(page) for them.
    void ForEachUploadRange(VAddr cpu_addr, u64 size, bool is_written, auto&& func) {
        if (!is_written &&
            Common::Perf::GetTuning().region_summary.load(std::memory_order_relaxed) &&
            !MayHaveCpuModified(cpu_addr, size)) {
            // No region of the range has CPU modified or hot pages (regions not created yet
            // count as modified): nothing to upload, no region to visit.
            Common::Perf::Add(Common::Perf::RegionSummarySkips);
            return;
        }
        IteratePages<true>(
            cpu_addr, size, [&func, is_written](RegionManager* manager, u64 offset, u64 size) {
                const VAddr base = manager->GetCpuAddress();
                const auto upload = [&func, manager, base](VAddr addr, u64 upload_size) {
                    manager->ClearGpuOwnedLocked(addr - base, upload_size);
                    func(addr, upload_size, [manager, base](VAddr page) {
                        manager->SetGpuOwnedLocked(page - base, BYTES_PER_PAGE);
                    });
                };
                if (is_written) {
                    // Lock free early out for ranges the GPU already owns (buffers written by
                    // every dispatch): nothing to upload, no state or protection to change.
                    if (!manager->template IsRegionModified<Type::CPU>(offset, size) &&
                        manager->IsRegionFullyGpuModified(offset, size)) {
                        Common::Perf::Add(Common::Perf::WrittenRangeSkips);
                        return;
                    }
                    manager->template ForEachModifiedRange<Type::CPU, StateOp::Clear, StateOp::Set>(
                        offset, size, upload);
                } else {
                    // Lock free early out for clean ranges: nothing to upload and no state or
                    // protection to change (hot path, every read only buffer of every draw).
                    if (!manager->template IsRegionModified<Type::CPU>(offset, size)) {
                        return;
                    }
                    manager
                        ->template ForEachModifiedRange<Type::CPU, StateOp::Clear, StateOp::None>(
                            offset, size, upload);
                }
            });
    }

    /// Write merge: whether the range has pages with GPU written bytes guest memory lacks.
    bool IsGpuOwned(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->IsGpuOwned(offset, size);
        });
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <bool clear>
    void ForEachDownloadRange(VAddr cpu_addr, u64 size, auto&& func) {
        constexpr auto gpu_op = clear ? StateOp::Clear : StateOp::None;
        IteratePages(cpu_addr, size, [&func](RegionManager* manager, u64 offset, u64 size) {
            manager->template ForEachModifiedRange<Type::GPU, StateOp::None, gpu_op>(offset, size,
                                                                                     func);
        });
    }

private:
    template <bool create_region_on_fail = false, typename Func>
    bool IteratePages(VAddr cpu_address, u64 size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, RegionManager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        // Only the 40-bit guest address space is tracked. The command processor also writes host
        // memory (video out labels): such addresses must not index past the top level table.
        if (cpu_address >= MAX_CPU_ADDRESS) [[unlikely]] {
            return false;
        }
        u64 remaining_size = std::min(size, MAX_CPU_ADDRESS - cpu_address);
        u64 page_index = cpu_address >> HIGHER_PAGE_BITS;
        u64 page_offset = cpu_address & HIGHER_PAGE_MASK;
        while (remaining_size > 0) {
            const u64 copy_amount = std::min(HIGHER_PAGE_SIZE - page_offset, remaining_size);
            if (auto* region = top_tier[page_index].load(std::memory_order_acquire); region) {
                if constexpr (BOOL_BREAK) {
                    if (func(region, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(region, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                region = CreateRegion(page_index);
                if constexpr (BOOL_BREAK) {
                    if (func(region, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(region, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    RegionManager* CreateRegion(u64 page_index) {
        std::scoped_lock create_lock{region_create_mutex};
        if (auto* existing = top_tier[page_index].load(std::memory_order_acquire)) {
            return existing;
        }
        const VAddr base_cpu_addr = page_index << HIGHER_PAGE_BITS;
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                std::construct_at(&last_pool[i], tracker, 0);
                free_managers.push_back(&last_pool[i]);
            }
        }
        auto* new_manager = free_managers.back();
        new_manager->SetCpuAddress(base_cpu_addr);
        // Its summary bit is set (every page of a new region is CPU modified).
        new_manager->SetSummary(&region_summary[page_index / 64], u64{1} << (page_index % 64));
        free_managers.pop_back();
        top_tier[page_index].store(new_manager, std::memory_order_release);
        return new_manager;
    }

    /// Region summary: whether a region of the range may have CPU modified (or hot) pages,
    /// lock free. Regions not created yet count as modified.
    bool MayHaveCpuModified(VAddr cpu_addr, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        if (cpu_addr >= MAX_CPU_ADDRESS) [[unlikely]] {
            return true;
        }
        const u64 first = cpu_addr >> HIGHER_PAGE_BITS;
        const u64 last = std::min(cpu_addr + size - 1, MAX_CPU_ADDRESS - 1) >> HIGHER_PAGE_BITS;
        for (u64 region = first; region <= last;) {
            const u64 bit = region % 64;
            const u64 count = std::min<u64>(64 - bit, last - region + 1);
            const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
            if ((region_summary[region / 64].load(std::memory_order_acquire) & mask) != 0) {
                return true;
            }
            region += count;
        }
        return false;
    }

    /// Write merge: keeps the content of the pages (unless already kept since their last upload).
    void SnapshotPages(VAddr first_page, u64 num_pages) {
        std::scoped_lock lk{snapshots_mutex};
        const u32 frame = g_tracker_frame.load(std::memory_order_relaxed);
        for (u64 i = 0; i < num_pages; ++i) {
            const VAddr page = first_page + i * BYTES_PER_PAGE;
            if (snapshots.contains(page)) {
                // The oldest content is the one before all guest writes since the upload.
                continue;
            }
            if (snapshots.size() >= MaxSnapshots) {
                Common::Perf::Add(Common::Perf::MergeSnapshotsSkipped);
                continue;
            }
            auto entry = std::make_unique<SnapshotEntry>();
            entry->frame = frame;
            CopyGuestPage(page, entry->data.data());
            snapshots.emplace(page, std::move(entry));
            Common::Perf::Add(Common::Perf::MergeSnapshots);
        }
        num_snapshots.store(static_cast<u32>(snapshots.size()), std::memory_order_release);
    }

    static constexpr u64 MaxSnapshotPages = 4;
    static constexpr size_t MaxSnapshots = 8192;

    struct SnapshotEntry {
        u32 frame;
        PageSnapshot data;
    };

    PageManager* tracker;
    const u32 readbacks_mode;
    std::mutex snapshots_mutex;
    std::unordered_map<VAddr, std::unique_ptr<SnapshotEntry>> snapshots;
    std::atomic<u32> num_snapshots{};
    std::mutex pending_readbacks_mutex;
    std::vector<std::pair<VAddr, VAddr>> pending_readbacks;
    std::atomic<u32> num_pending_readbacks{};
    // Unique sequence IDs for G/C writers; publication with dirty ranges still needs the
    // packet coherence lock before split recording is enabled.
    std::atomic<u64> last_write_seq{};
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::mutex region_create_mutex;
    std::array<std::atomic<RegionManager*>, NUM_HIGH_PAGES> top_tier{};
    /// Region summary (Tuning::region_summary): a bit per region, set while it may have CPU
    /// modified or hot pages (RegionManager::SetSummary), all set at first: regions not created
    /// yet start fully CPU modified.
    std::array<std::atomic<u64>, NUM_HIGH_PAGES / 64> region_summary{};
};

} // namespace VideoCore
