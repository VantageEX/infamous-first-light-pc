// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <bit>
#include <mutex>
#include <utility>

#include "common/adaptive_mutex.h"
#include "common/perf_stats.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"
#include "page_write_provenance.h"

namespace VideoCore {

/// Stage binding memo: incremented whenever any tracked page becomes CPU modified (a guest
/// write fault, an invalidation). Remembered binding outcomes are dropped when it changes.
inline std::atomic<u64> g_cpu_dirty_seq{1};

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = std::mutex;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_},
          readbacks_mode{EmulatorSettings.GetReadbacksMode()} {
        cpu.Fill(~0ULL);
        gpu.Fill(0ULL);
        hot.Fill(0ULL);
        no_hot.Fill(0ULL);
        hash_valid.Fill(0ULL);
        gpu_owned.Fill(0ULL);
        fault_frame.fill(0);
        fault_streak.fill(0);
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    [[nodiscard]] VAddr GetCpuAddress() const noexcept {
        return cpu_addr;
    }

    static constexpr Bounds GetBounds(u64 offset, u64 size) {
        const u64 end_address = offset + size - 1;
        return Bounds{
            .start_word = offset / BYTES_PER_WORD,
            .start_page = (offset) / BYTES_PER_PAGE,
            .end_word = end_address / BYTES_PER_WORD,
            .end_page = (end_address) / BYTES_PER_PAGE,
        };
    }

    static constexpr std::pair<u64, u64> GetMasks(u64 start_page, u64 end_page) {
        const u64 start_mask = ~u64{0} << (start_page & (PAGES_PER_WORD - 1));
        const u64 end_mask = ~u64{0} >> (63 - (end_page & (PAGES_PER_WORD - 1)));
        return std::make_pair(start_mask, end_mask);
    }

    /// Bits first..last (inclusive, last - first < 64) of a summary word.
    static constexpr u64 BitRangeMask(u64 first, u64 last) {
        return (~u64{0} >> (63 - (last - first))) << first;
    }

    /// Region summary (MemoryTracker): the word and bit telling that this region may have CPU
    /// modified or hot pages. Set once, when the region is created.
    void SetSummary(std::atomic<u64>* word, u64 bit) noexcept {
        summary_word = word;
        summary_bit = bit;
    }

    static constexpr void IterateWords(Bounds bounds, auto&& func) {
        const auto [start_word, start_page, end_word, end_page] = bounds;
        const auto [start_mask, end_mask] = GetMasks(start_page, end_page);
        if (start_word == end_word) [[likely]] {
            func(start_word, start_mask & end_mask);
        } else {
            func(start_word, start_mask);
            for (s64 i = start_word + 1; i < end_word; ++i) {
                func(i, ~0ULL);
            }
            func(end_word, end_mask);
        }
    }

    static constexpr void IteratePages(u64 word, auto&& func) {
        u64 offset{};
        while (word != 0) {
            const u64 empty_bits = std::countr_zero(word);
            offset += empty_bits;
            word >>= empty_bits;
            const u64 set_bits = std::countr_one(word);
            func(offset, set_bits);
            word = set_bits < PAGES_PER_WORD ? (word >> set_bits) : 0;
            offset += set_bits;
        }
    }

    template <StateOp cpu_op, StateOp gpu_op, bool locked = true>
    void ChangeRegionState(u64 offset, u64 size) {
        RegionBits write_prot;
        RegionBits read_prot;
        auto bounds = GetBounds(offset, size);
        Bounds watcher_bounds;
        if constexpr (cpu_op == StateOp::Set) {
            // Pages become CPU modified (a guest write): bindings checked clean before are not
            // any more (BufferCache sync memo).
            cpu_version.fetch_add(1, std::memory_order_release);
            g_cpu_dirty_seq.fetch_add(1, std::memory_order_release);
        }
        bool late = false;
        if constexpr (locked) {
            mutex.lock();
            late = tracker->BeginLateProtect();
        }
        RefreshHot();
        if constexpr (cpu_op == StateOp::Set) {
            CpuWordsSetting(BitRangeMask(bounds.start_word, bounds.end_word));
        }
        IterateWords(bounds, [&](u64 index, u64 mask) {
            UpdateStateAndProtection<cpu_op, gpu_op>(write_prot, read_prot, index, mask);
        });
        if constexpr (cpu_op == StateOp::Clear) {
            CpuWordsCleared(bounds.start_word, bounds.end_word);
        }
        const auto write_op = GetPageOp<Type::CPU>(cpu_op);
        const auto read_op = GetPageOp<Type::GPU>(gpu_op);
        const bool update_watchers = write_op != PageOp::None || read_op != PageOp::None;
        if (update_watchers &&
            GetWatcherBounds<cpu_op, gpu_op>(bounds, write_prot, read_prot, watcher_bounds)) {
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, read_prot,
                                                 write_op, read_op);
        }
        if constexpr (locked) {
            mutex.unlock();
            if (late) {
                tracker->EndLateProtect();
            }
        }
    }

    template <Type type, StateOp cpu_op, StateOp gpu_op, bool locked = true>
    void ForEachModifiedRange(u64 offset, s64 size, auto&& func) {
        auto& state = GetRegionBits<type>();
        RegionBits write_prot;
        RegionBits read_prot;
        u64 start_page{};
        u64 end_page{};
        auto bounds = GetBounds(offset, size);
        Bounds watcher_bounds;
        bool late = false;
        if constexpr (locked) {
            mutex.lock();
            late = tracker->BeginLateProtect();
        }
        RefreshHot();
        if constexpr (cpu_op == StateOp::Set) {
            CpuWordsSetting(BitRangeMask(bounds.start_word, bounds.end_word));
        }
        IterateWords(bounds, [&](u64 index, u64 mask) {
            const u64 base_page = index * PAGES_PER_WORD;
            const u64 word = state[index] & mask;
            UpdateStateAndProtection<cpu_op, gpu_op>(write_prot, read_prot, index, mask);
            IteratePages(word, [&](u64 pages_offset, u64 pages_size) {
                if (end_page == base_page + pages_offset) {
                    end_page += pages_size;
                    return;
                }
                if (end_page) {
                    func(cpu_addr + start_page * BYTES_PER_PAGE,
                         (end_page - start_page) * BYTES_PER_PAGE);
                }
                start_page = base_page + pages_offset;
                end_page = start_page + pages_size;
            });
        });
        if constexpr (cpu_op == StateOp::Clear) {
            CpuWordsCleared(bounds.start_word, bounds.end_word);
        }
        if (end_page) {
            func(cpu_addr + start_page * BYTES_PER_PAGE, (end_page - start_page) * BYTES_PER_PAGE);
        }
        const auto write_op = GetPageOp<Type::CPU>(cpu_op);
        const auto read_op = GetPageOp<Type::GPU>(gpu_op);
        const bool update_watchers = write_op != PageOp::None || read_op != PageOp::None;
        if (update_watchers &&
            GetWatcherBounds<cpu_op, gpu_op>(bounds, write_prot, read_prot, watcher_bounds)) {
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, read_prot,
                                                 write_op, read_op);
        }
        if constexpr (locked) {
            mutex.unlock();
            if (late) {
                tracker->EndLateProtect();
            }
        }
    }

    template <Type type>
    bool IsRegionModified(u64 offset, u64 size) noexcept {
        auto& state = GetRegionBits<type>();
        const auto [start_word, start_page, end_word, end_page] = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(start_page, end_page);
        if constexpr (type == Type::CPU) {
            // Hot pages need an upload in every new epoch: until the region is refreshed under
            // its lock, they count as CPU modified.
            if (num_hot.load(std::memory_order_relaxed) != 0 &&
                hot_epoch.load(std::memory_order_acquire) != CurrentUploadEpoch() &&
                AnyBits(hot, start_word, end_word, start_mask, end_mask)) {
                return true;
            }
            // Summary of the words (region_summary): clean words are not read. Huge bindings
            // (a buffer covering the whole guest memory) span every word of many regions.
            if (Common::Perf::GetTuning().region_summary.load(std::memory_order_relaxed) &&
                (cpu_any.load(std::memory_order_acquire) &
                 BitRangeMask(start_word, end_word)) == 0) {
                return false;
            }
        }
        if (start_word == end_word) [[likely]] {
            return state[start_word] & (start_mask & end_mask);
        } else {
            if (state[start_word] & start_mask) {
                return true;
            }
            for (s64 i = start_word + 1; i < end_word; ++i) {
                if (state[i]) {
                    return true;
                }
            }
            return state[end_word] & end_mask;
        }
    }

    /// Returns true if every page of the range is GPU modified (lock free, like IsRegionModified).
    bool IsRegionFullyGpuModified(u64 offset, u64 size) noexcept {
        const auto [start_word, start_page, end_word, end_page] = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(start_page, end_page);
        if (start_word == end_word) {
            const u64 mask = start_mask & end_mask;
            return (gpu[start_word] & mask) == mask;
        }
        if ((gpu[start_word] & start_mask) != start_mask) {
            return false;
        }
        for (u64 i = start_word + 1; i < end_word; ++i) {
            if (gpu[i] != ~u64{0}) {
                return false;
            }
        }
        return (gpu[end_word] & end_mask) == end_mask;
    }

    [[nodiscard]] u64 CpuVersion() const noexcept {
        return cpu_version.load(std::memory_order_acquire);
    }

    /// Returns true if any page of the range is hot (lock free).
    bool HasHotPages(u64 offset, u64 size) noexcept {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        const auto bounds = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(bounds.start_page, bounds.end_page);
        return AnyBits(hot, bounds.start_word, bounds.end_word, start_mask, end_mask);
    }

    /// Remembers the frame of a write fault on the pages of the range (the region is locked).
    void NoteWriteFrame(u64 offset, u64 size) {
        const u16 frame = static_cast<u16>(g_tracker_frame.load(std::memory_order_relaxed));
        const u64 first = offset / BYTES_PER_PAGE;
        const u64 last = (offset + size - 1) / BYTES_PER_PAGE;
        for (u64 page = first; page <= last; ++page) {
            write_frame[page] = frame;
        }
    }

    /// Returns true if no page of the range had a write fault in the last `frames` frames (lock
    /// free, an estimate). Pages written every frame are bound from fresh copies instead of
    /// their GPU copy, whose next upload would need a pipeline barrier after reads in place.
    bool IsRegionStable(u64 offset, u64 size, u32 frames) const noexcept {
        const u16 frame = static_cast<u16>(g_tracker_frame.load(std::memory_order_relaxed));
        const u64 first = offset / BYTES_PER_PAGE;
        const u64 last = (offset + size - 1) / BYTES_PER_PAGE;
        for (u64 page = first; page <= last; ++page) {
            if (static_cast<u16>(frame - write_frame[page]) <= frames) {
                return false;
            }
        }
        return true;
    }

    /// Counts a guest write fault on the pages of the range, which were just marked CPU modified
    /// (the region is locked by the caller). A page faulting in consecutive frames is written by
    /// the guest every frame: it becomes hot, it is no longer write protected and is uploaded
    /// once per upload epoch instead. Pages also written by the command processor never become
    /// hot, as its writes would go unnoticed (constant RAM dumps, labels).
    void NoteWriteFault(u64 offset, u64 size, bool from_gpu_thread, u32 threshold) {
        const u16 frame = static_cast<u16>(g_tracker_frame.load(std::memory_order_relaxed));
        const u64 first = offset / BYTES_PER_PAGE;
        const u64 last = (offset + size - 1) / BYTES_PER_PAGE;
        for (u64 page = first; page <= last; ++page) {
            const u64 index = page / PAGES_PER_WORD;
            const u64 bit = 1ULL << (page % PAGES_PER_WORD);
            if (from_gpu_thread) {
                no_hot[index] |= bit;
                continue;
            }
            if (threshold == 0 || (hot[index] & bit) || (no_hot[index] & bit) ||
                (gpu[index] & bit) || !(cpu[index] & bit)) {
                continue;
            }
            const u16 elapsed = static_cast<u16>(frame - fault_frame[page]);
            u8 streak = fault_streak[page];
            streak = (streak != 0 && elapsed <= 2) ? static_cast<u8>(std::min(streak + 1, 255))
                                                    : u8{1};
            fault_frame[page] = frame;
            fault_streak[page] = streak;
            if (streak >= threshold) {
                // Already CPU modified and unprotected: nothing else to change. Its GPU copy
                // was not uploaded as a hot page yet.
                hot[index] |= bit;
                hash_valid[index] &= ~bit;
                num_hot.fetch_add(1, std::memory_order_relaxed);
                Common::Perf::Add(Common::Perf::HotPagePromotions);
            }
        }
    }

    /// Write fault window (pc-port shim): after a guest write fault, the other pages of the
    /// range that are write protected (CPU clean) and hold no GPU modified data become CPU
    /// modified and unprotected too, so a guest writing on through the window does not fault
    /// on each page. Their next upload copies bytes the GPU copy already has. GPU modified
    /// pages (also those a recorded command writes: bound written ranges are marked when bound),
    /// hot and GPU owned pages keep their state. Returns the number of pages unprotected.
    u32 ExtendWriteFault(u64 offset, u64 size) {
        const auto bounds = GetBounds(offset, size);
        RegionBits write_prot;
        RegionBits read_prot;
        const bool late = Lock(bounds);
        RefreshHot();
        u64 words = 0;
        u32 pages = 0;
        IterateWords(bounds, [&](u64 index, u64 mask) {
            const u64 add = mask & ~(cpu[index] | gpu[index] | hot[index] | gpu_owned[index]);
            write_prot[index] = add;
            if (add != 0) {
                words |= u64{1} << index;
                pages += static_cast<u32>(std::popcount(add));
            }
        });
        if (words != 0) {
            CpuWordsSetting(words);
            IterateWords(bounds, [&](u64 index, u64) { cpu[index] |= write_prot[index]; });
            // After the bits: a sync memo taken from the old version sees them set.
            cpu_version.fetch_add(1, std::memory_order_release);
            g_cpu_dirty_seq.fetch_add(1, std::memory_order_release);
            Bounds watcher_bounds;
            if (GetWatcherBounds<StateOp::Set, StateOp::None>(bounds, write_prot, read_prot,
                                                              watcher_bounds)) {
                tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot,
                                                     read_prot, PageOp::Untrack, PageOp::None);
            }
        }
        Unlock(bounds, late);
        return pages;
    }

    /// Remembers the GPU tick of the command buffer that writes the range, per 256 KB word, and
    /// the sequence number of the write (MemoryTracker); queue: the GPU queue of that command
    /// buffer (0 graphics, 1 compute). Command processor thread only.
    void SetWriteTick(u64 offset, u64 size, u64 tick, u64 seq, u32 queue,
                      u64 generation, u64 native_epoch) {
        std::scoped_lock hazard_lock{hazard_mutex};
        const auto bounds = GetBounds(offset, size);
        if (iss::readback::PageProvenanceEnabled()) {
            if (!page_write_provenance || page_provenance_generation != generation ||
                page_provenance_native_epoch != native_epoch) {
                page_write_provenance =
                    std::make_unique<iss::readback::PageWriteProvenance<NUM_REGION_PAGES>>();
                page_provenance_generation = generation;
                page_provenance_native_epoch = native_epoch;
            }
            page_write_provenance->Publish(bounds.start_page, bounds.end_page, queue, tick);
        }
        for (u64 index = bounds.start_word; index <= bounds.end_word; ++index) {
            write_tick[index] = tick;
            write_seq[index] = seq;
            write_queue[index] = static_cast<u8>(queue);
        }
    }

    /// Cross queue dependencies (asc_queue): whether the compute queue ever accessed a word of
    /// the range (the graphics queue tracks its accesses to those words only).
    bool ComputeTouched(u64 offset, u64 size) const noexcept {
        std::scoped_lock hazard_lock{hazard_mutex};
        const auto bounds = GetBounds(offset, size);
        return (compute_words & BitRangeMask(bounds.start_word, bounds.end_word)) != 0;
    }

    /// Cross queue dependencies (asc_queue): an access of the range by the queue (0 graphics,
    /// 1 compute) in the command buffer of this tick. Returns the latest tick of the other queue
    /// it must wait for (its writes of the range, and its reads when the access is a write), 0
    /// if none; untouched is set when the compute queue accesses a word for the first time (the
    /// reads of the graphics queue there are unknown). The reads are then remembered (the
    /// writes are, by SetWriteTick). Command processor thread only.
    u64 CrossQueueAccess(u64 offset, u64 size, u32 queue, u64 tick, bool is_write,
                         bool& untouched) {
        std::scoped_lock hazard_lock{hazard_mutex};
        // Two levels: an access covering a whole 256 KB word compares with everything the other
        // queue did in the word (queue_*_tick); an access covering part of a word compares with
        // the pages it covers only (plus the other queue's whole word accesses). Word precision
        // alone made neighbouring data of the two queues depend on each other (a compute write
        // ending where graphics reads the indirect arguments made the compute ring wait for the
        // rendering several times per frame).
        const u32 other = queue ^ 1;
        u64 dependency = 0;
        const auto bounds = GetBounds(offset, size);
        u64 mask = BitRangeMask(bounds.start_word, bounds.end_word);
        if (queue == 0) {
            // Graphics: only the words the compute queue ever used are tracked. A first compute
            // use of a word waits for all the graphics work recorded so far, so the graphics
            // ticks of the other words are never read.
            mask &= compute_words;
        }
        while (mask != 0) {
            const u64 index = static_cast<u64>(std::countr_zero(mask));
            mask &= mask - 1;
            const u64 first_page = index * PAGES_PER_WORD;
            const u64 last_page = first_page + PAGES_PER_WORD - 1;
            const u64 p0 = std::max(first_page, bounds.start_page);
            const u64 p1 = std::min(last_page, bounds.end_page);
            if (p0 == first_page && p1 == last_page) {
                dependency = std::max(dependency, queue_write_tick[other][index]);
                if (is_write) {
                    dependency = std::max(dependency, queue_read_tick[other][index]);
                    queue_full_write_tick[queue][index] = tick;
                    queue_write_tick[queue][index] = tick;
                } else {
                    queue_full_read_tick[queue][index] = tick;
                    queue_read_tick[queue][index] = tick;
                }
            } else {
                if (!queue_page_ticks) {
                    queue_page_ticks = std::make_unique<QueuePageTicks>();
                }
                auto& pages = *queue_page_ticks;
                dependency = std::max(dependency, queue_full_write_tick[other][index]);
                if (is_write) {
                    dependency = std::max(dependency, queue_full_read_tick[other][index]);
                }
                for (u64 page = p0; page <= p1; ++page) {
                    dependency = std::max(dependency, pages.write[other][page]);
                    if (is_write) {
                        dependency = std::max(dependency, pages.read[other][page]);
                        pages.write[queue][page] = tick;
                    } else {
                        pages.read[queue][page] = tick;
                    }
                }
                (is_write ? queue_write_tick : queue_read_tick)[queue][index] = tick;
            }
            if (queue == 1) {
                const u64 bit = u64{1} << index;
                untouched |= (compute_words & bit) == 0;
                compute_words |= bit;
            }
        }
        return dependency;
    }

    /// Cross queue dependencies (asc_queue): the queue writes the GPU copy of the range in the
    /// command buffer of this tick (an upload): the other queue waits for it before using it.
    void NoteQueueWrite(u64 offset, u64 size, u32 queue, u64 tick) {
        std::scoped_lock hazard_lock{hazard_mutex};
        const auto bounds = GetBounds(offset, size);
        for (u64 index = bounds.start_word; index <= bounds.end_word; ++index) {
            queue_write_tick[queue][index] = tick;
        }
    }

    /// Returns the latest GPU tick of each queue that wrote the GPU modified pages of the range
    /// (0 if none). Command processor thread only.
    void GetWriteTicks(u64 offset, u64 size, std::array<u64, 2>& ticks) {
        std::unique_lock state_lock{mutex};
        std::scoped_lock hazard_lock{hazard_mutex};
        IterateWords(GetBounds(offset, size), [&](u64 index, u64 mask) {
            if (gpu[index] & mask) {
                u64& tick = ticks[write_queue[index] & 1];
                tick = std::max(tick, write_tick[index]);
            }
        });
    }

    // A dirty page with no recorded producer rejects precision for the whole query.
    // Word-level sequence numbers used for retirement remain unchanged.
    bool GetPageWriteTicks(u64 offset, u64 size, std::array<u64, 2>& ticks,
                           u64 generation, u64 native_epoch) {
        std::unique_lock state_lock{mutex};
        std::scoped_lock hazard_lock{hazard_mutex};
        if (!generation || !native_epoch || generation != page_provenance_generation ||
            native_epoch != page_provenance_native_epoch) return false;
        bool complete = true;
        IterateWords(GetBounds(offset, size), [&](u64 index, u64 mask) {
            u64 dirty = gpu[index] & mask;
            while (dirty) {
                const auto bit = std::countr_zero(dirty);
                dirty &= dirty - 1;
                const auto page = index * PAGES_PER_WORD + bit;
                if (!page_write_provenance || !page_write_provenance->Merge(page, ticks))
                    complete = false;
            }
        });
        return complete;
    }

    /// The sequence number of the latest GPU write of the range (any page of its 256 KB words),
    /// 0 if none. Command processor thread only.
    u64 GetWriteSeq(u64 offset, u64 size) {
        std::scoped_lock hazard_lock{hazard_mutex};
        u64 seq = 0;
        const auto bounds = GetBounds(offset, size);
        for (u64 index = bounds.start_word; index <= bounds.end_word; ++index) {
            seq = std::max(seq, write_seq[index]);
        }
        return seq;
    }

    /// Returns the latest GPU tick that wrote the GPU modified pages of the range, 0 if none.
    /// Command processor thread only.
    u64 GetWriteTick(u64 offset, u64 size) {
        std::unique_lock state_lock{mutex};
        std::scoped_lock hazard_lock{hazard_mutex};
        u64 tick = 0;
        IterateWords(GetBounds(offset, size), [&](u64 index, u64 mask) {
            if (gpu[index] & mask) {
                tick = std::max(tick, write_tick[index]);
            }
        });
        return tick;
    }

    /// Write merge: pages holding bytes the GPU wrote that guest memory lacks (a guest write
    /// discarded their GPU modified state, or an upload left those bytes on the GPU). Lock free.
    bool IsGpuOwned(u64 offset, u64 size) noexcept {
        const auto bounds = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(bounds.start_page, bounds.end_page);
        return AnyBits(gpu_owned, bounds.start_word, bounds.end_word, start_mask, end_mask);
    }

    /// Write merge, called with the region locked.
    void SetGpuOwnedLocked(u64 offset, u64 size) noexcept {
        IterateWords(GetBounds(offset, size), [&](u64 index, u64 mask) { gpu_owned[index] |= mask; });
    }

    /// Write merge, called with the region locked.
    void ClearGpuOwnedLocked(u64 offset, u64 size) noexcept {
        IterateWords(GetBounds(offset, size),
                     [&](u64 index, u64 mask) { gpu_owned[index] &= ~mask; });
    }

    /// The command processor wrote the range directly in guest memory, which does not fault on
    /// hot pages: they are marked CPU modified to be uploaded again, and leave streaming for good.
    void OnGpuThreadWrite(u64 offset, u64 size) {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const auto bounds = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(bounds.start_page, bounds.end_page);
        if (!AnyBits(hot, bounds.start_word, bounds.end_word, start_mask, end_mask)) {
            return;
        }
        std::scoped_lock lk{mutex};
        RefreshHot();
        IterateWords(bounds, [&](u64 index, u64 mask) {
            const u64 leaving = hot[index] & mask;
            if (leaving == 0) {
                return;
            }
            // Unprotected and CPU modified, like any page after a write fault.
            CpuWordsSetting(u64{1} << index);
            cpu[index] |= leaving;
            hot[index] &= ~leaving;
            hash_valid[index] &= ~leaving;
            no_hot[index] |= leaving;
            const auto count = static_cast<u32>(std::popcount(leaving));
            num_hot.fetch_sub(count, std::memory_order_relaxed);
            Common::Perf::Add(Common::Perf::HotPageDemotions, count);
        });
    }

    /// At a new upload epoch every hot page of the region counts as CPU modified, although the
    /// guest may not have written it since its last upload. Before an upload of the range, the
    /// hot pages whose content hash did not change since their last upload are marked clean
    /// instead (their GPU copy is up to date), the others remember the hash of the content about
    /// to be uploaded.
    template <typename HashFunc>
    void SkipUnchangedHotPages(u64 offset, u64 size, HashFunc&& hash_page) {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const auto bounds = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(bounds.start_page, bounds.end_page);
        if (!AnyBits(hot, bounds.start_word, bounds.end_word, start_mask, end_mask)) {
            return;
        }
        u64 skipped = 0;
        u64 uploaded = 0;
        std::scoped_lock lk{mutex};
        RefreshHot();
        IterateWords(bounds, [&](u64 index, u64 mask) {
            u64 candidates = cpu[index] & hot[index] & mask;
            while (candidates != 0) {
                const u64 bit_index = static_cast<u64>(std::countr_zero(candidates));
                const u64 bit = 1ULL << bit_index;
                candidates &= ~bit;
                const u64 page = index * PAGES_PER_WORD + bit_index;
                const u64 hash = hash_page(cpu_addr + page * BYTES_PER_PAGE);
                if ((hash_valid[index] & bit) != 0 && page_hash[page] == hash) {
                    // Hot pages are never write protected: clearing the state is enough.
                    cpu[index] &= ~bit;
                    ++skipped;
                } else {
                    page_hash[page] = hash;
                    hash_valid[index] |= bit;
                    ++uploaded;
                }
            }
            CpuWordsCleared(index, index);
        });
        if (skipped != 0) {
            Common::Perf::Add(Common::Perf::HotPageHashSkips, skipped);
        }
        if (uploaded != 0) {
            Common::Perf::Add(Common::Perf::HotPageHashUploads, uploaded);
        }
    }

    /// After the upload of the range: hot pages whose content changed since their hash was taken
    /// (the guest wrote them during the copy) forget it, so their next upload is not skipped.
    template <typename HashFunc>
    void VerifyHotPageHashes(u64 offset, u64 size, HashFunc&& hash_page) {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const auto bounds = GetBounds(offset, size);
        u64 mismatches = 0;
        std::scoped_lock lk{mutex};
        IterateWords(bounds, [&](u64 index, u64 mask) {
            u64 candidates = hot[index] & hash_valid[index] & mask;
            while (candidates != 0) {
                const u64 bit_index = static_cast<u64>(std::countr_zero(candidates));
                const u64 bit = 1ULL << bit_index;
                candidates &= ~bit;
                const u64 page = index * PAGES_PER_WORD + bit_index;
                if (hash_page(cpu_addr + page * BYTES_PER_PAGE) != page_hash[page]) {
                    hash_valid[index] &= ~bit;
                    ++mismatches;
                }
            }
        });
        if (mismatches != 0) {
            Common::Perf::Add(Common::Perf::HotPageHashMismatches, mismatches);
        }
    }

    /// Protects the hot pages again, to notice the ones the guest no longer writes every frame.
    /// The others become hot again after a single fault. Returns the number of pages cooled.
    u32 CoolHotPages(u32 threshold) {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return 0;
        }
        std::scoped_lock lk{mutex};
        const u16 frame = static_cast<u16>(g_tracker_frame.load(std::memory_order_relaxed));
        const u8 rearm = static_cast<u8>(std::clamp<u32>(threshold, 2, 255) - 1);
        RegionBits write_prot;
        u32 cooled = 0;
        u64 first_word = NUM_REGION_WORDS;
        u64 last_word = 0;
        for (u64 index = 0; index < NUM_REGION_WORDS; ++index) {
            const u64 hot_word = hot[index];
            // Clean hot pages need a write watcher again, dirty ones stay unprotected like any
            // CPU modified page.
            write_prot[index] = hot_word & ~cpu[index];
            if (hot_word == 0) {
                continue;
            }
            hot[index] = 0;
            hash_valid[index] &= ~hot_word;
            cooled += static_cast<u32>(std::popcount(hot_word));
            if (write_prot[index] != 0) {
                first_word = std::min(first_word, index);
                last_word = index;
            }
            IteratePages(hot_word, [&](u64 page_offset, u64 num_pages) {
                for (u64 i = 0; i < num_pages; ++i) {
                    const u64 page = index * PAGES_PER_WORD + page_offset + i;
                    fault_frame[page] = frame;
                    fault_streak[page] = rearm;
                }
            });
        }
        num_hot.store(0, std::memory_order_relaxed);
        if (first_word <= last_word) {
            const Bounds watcher_bounds{
                .start_word = first_word,
                .start_page = static_cast<u64>(std::countr_zero(write_prot[first_word])),
                .end_word = last_word,
                .end_page = PAGES_PER_WORD - std::countl_zero(write_prot[last_word]) - 1,
            };
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, write_prot,
                                                 PageOp::Track, PageOp::None);
        }
        return cooled;
    }

    /// External locking (MemoryTracker::InvalidateRegion): a late protection section too.
    /// Returns whether it was begun, to pass to Unlock.
    bool Lock(const Bounds& bounds) noexcept {
        mutex.lock();
        return tracker->BeginLateProtect();
    }

    void Unlock(const Bounds& bounds, bool late) noexcept {
        mutex.unlock();
        if (late) {
            tracker->EndLateProtect();
        }
    }

private:
    /// At the first use of the region in a new upload epoch, marks the hot pages CPU modified so
    /// they are uploaded again. Called with the region locked.
    void RefreshHot() {
        if (num_hot.load(std::memory_order_relaxed) == 0) {
            return;
        }
        const u64 epoch = CurrentUploadEpoch();
        if (hot_epoch.load(std::memory_order_relaxed) == epoch) {
            return;
        }
        u64 words = 0;
        for (u64 index = 0; index < NUM_REGION_WORDS; ++index) {
            if (hot[index] != 0) {
                words |= u64{1} << index;
            }
        }
        CpuWordsSetting(words);
        for (u64 index = 0; index < NUM_REGION_WORDS; ++index) {
            cpu[index] |= hot[index];
        }
        hot_epoch.store(epoch, std::memory_order_release);
    }

    /// Before words of 'cpu' get bits set (the region is locked): 'words' has a bit per word.
    /// The summaries are set first, so that a reader finding them clear never misses a word
    /// with bits set: clear summary means clean words.
    void CpuWordsSetting(u64 words) noexcept {
        const u64 any = cpu_any.load(std::memory_order_relaxed);
        if ((any | words) == any) {
            return;
        }
        if (any == 0 && summary_word != nullptr) {
            summary_word->fetch_or(summary_bit, std::memory_order_release);
        }
        cpu_any.store(any | words, std::memory_order_release);
    }

    /// After bits of the words first_word..last_word of 'cpu' were cleared (the region is
    /// locked): the summaries of the words now empty are cleared, last.
    void CpuWordsCleared(u64 first_word, u64 last_word) noexcept {
        u64 any = cpu_any.load(std::memory_order_relaxed);
        const u64 before = any;
        for (u64 index = first_word; index <= last_word; ++index) {
            if (cpu[index] == 0) {
                any &= ~(u64{1} << index);
            }
        }
        if (any == before) {
            return;
        }
        cpu_any.store(any, std::memory_order_release);
        // A region with hot pages stays in the summary: its hot pages count as CPU modified at
        // every new epoch, before any word is set.
        if (any == 0 && summary_word != nullptr && num_hot.load(std::memory_order_relaxed) == 0) {
            summary_word->fetch_and(~summary_bit, std::memory_order_release);
        }
    }

    static bool AnyBits(RegionBits& bits, u64 start_word, u64 end_word, u64 start_mask,
                        u64 end_mask) noexcept {
        if (start_word == end_word) {
            return bits[start_word] & (start_mask & end_mask);
        }
        if (bits[start_word] & start_mask) {
            return true;
        }
        for (u64 i = start_word + 1; i < end_word; ++i) {
            if (bits[i]) {
                return true;
            }
        }
        return bits[end_word] & end_mask;
    }

    template <StateOp cpu_op, StateOp gpu_op>
    void UpdateStateAndProtection(RegionBits& write_prot, RegionBits& read_prot, u64 index,
                                  u64 mask) {
        if constexpr (cpu_op != StateOp::None) {
            const u64 prev = cpu[index];
            if constexpr (cpu_op == StateOp::Clear) {
                cpu[index] &= ~mask;
            } else {
                cpu[index] |= mask;
            }
            u64 changed = (cpu[index] ^ prev) & mask;
            if (const u64 hot_word = hot[index]; hot_word != 0) [[unlikely]] {
                if constexpr (gpu_op == StateOp::Set) {
                    // Written by the GPU: the pages leave streaming and get a write watcher
                    // again (their CPU state is now clear), whatever it was before.
                    const u64 leaving = hot_word & mask;
                    if (leaving != 0) {
                        hot[index] = hot_word & ~leaving;
                        hash_valid[index] &= ~leaving;
                        const auto count = static_cast<u32>(std::popcount(leaving));
                        num_hot.fetch_sub(count, std::memory_order_relaxed);
                        Common::Perf::Add(Common::Perf::HotPageDemotions, count);
                    }
                    changed = (changed & ~hot_word) | leaving;
                } else {
                    // Hot pages are never write protected.
                    changed &= ~hot_word;
                }
            }
            write_prot[index] = changed;
        }
        if constexpr (gpu_op != StateOp::None) {
            const u64 prev = gpu[index];
            if constexpr (gpu_op == StateOp::Clear) {
                gpu[index] &= ~mask;
            } else {
                gpu[index] |= mask;
            }
            read_prot[index] = (gpu[index] ^ prev) & mask;
        }
    }

    template <StateOp cpu_op, StateOp gpu_op>
    static bool GetWatcherBounds(const Bounds& bounds, RegionBits& write_prot,
                                 RegionBits& read_prot, Bounds& watcher_bounds) {
        const auto prot = [&](u64 index) {
            u64 word{};
            if constexpr (cpu_op != StateOp::None) {
                word |= write_prot[index];
            }
            if constexpr (gpu_op != StateOp::None) {
                word |= read_prot[index];
            }
            return word;
        };
        u64 start_word = bounds.start_word;
        while (prot(start_word) == 0) {
            if (start_word == bounds.end_word) {
                return false;
            }
            ++start_word;
        }
        u64 end_word = bounds.end_word;
        while (prot(end_word) == 0) {
            --end_word;
        }
        const u64 start_prot = prot(start_word);
        const u64 end_prot = prot(end_word);
        watcher_bounds = Bounds{
            .start_word = start_word,
            .start_page = static_cast<u64>(std::countr_zero(start_prot)),
            .end_word = end_word,
            .end_page = PAGES_PER_WORD - std::countl_zero(end_prot) - 1,
        };
        return true;
    }

    template <Type type>
        requires(std::popcount(std::to_underlying(type)) == 1)
    constexpr PageOp GetPageOp(StateOp state_op) {
        if constexpr (type == Type::CPU) {
            if (state_op == StateOp::Set) {
                return PageOp::Untrack;
            } else if (state_op == StateOp::Clear) {
                return PageOp::Track;
            }
        } else if (type == Type::GPU && readbacks_mode == GpuReadbacksMode::Precise) {
            if (state_op == StateOp::Set) {
                return PageOp::Track;
            } else if (state_op == StateOp::Clear) {
                return PageOp::Untrack;
            }
        }
        return PageOp::None;
    }

    template <Type type>
        requires(std::popcount(std::to_underlying(type)) == 1)
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    PageManager* tracker;
    VAddr cpu_addr{};
    u32 readbacks_mode;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits hot;    ///< Pages written by the guest every frame: unprotected, uploaded per epoch.
    RegionBits no_hot; ///< Pages written by the command processor, never hot.
    RegionBits hash_valid; ///< Hot pages whose page_hash is the content of their GPU copy.
    RegionBits gpu_owned;  ///< Write merge: pages with GPU written bytes guest memory lacks.
    std::atomic<u32> num_hot{};
    std::atomic<u64> hot_epoch{};
    /// Grows whenever pages of the region become CPU modified (lock free read).
    std::atomic<u64> cpu_version{};
    /// Lock free summary of 'cpu': bit i set while word i may have bits set (set before the
    /// word gets any, cleared after it lost its last one). A clear bit means a clean word.
    std::atomic<u64> cpu_any{~u64{0}};
    /// The region's bit in the tracker's region summary, see SetSummary.
    std::atomic<u64>* summary_word{};
    u64 summary_bit{};
    std::array<u64, NUM_REGION_WORDS> write_tick{}; ///< Last GPU tick writing each word.
    std::array<u64, NUM_REGION_WORDS> write_seq{};  ///< Sequence of the last GPU write of each word.
    std::array<u8, NUM_REGION_WORDS> write_queue{}; ///< Queue of write_tick (0 graphics, 1 compute).
    std::unique_ptr<iss::readback::PageWriteProvenance<NUM_REGION_PAGES>> page_write_provenance;
    u64 page_provenance_generation{}, page_provenance_native_epoch{};
    /// Cross queue dependencies (asc_queue): last tick of each queue reading and writing each
    /// word (CrossQueueAccess, NoteQueueWrite), and the words the compute queue ever accessed.
    std::array<std::array<u64, NUM_REGION_WORDS>, 2> queue_read_tick{};
    std::array<std::array<u64, NUM_REGION_WORDS>, 2> queue_write_tick{};
    /// Last tick of each queue accessing a whole word, and per page ticks of the accesses
    /// covering part of a word (allocated on the first such access of the region).
    std::array<std::array<u64, NUM_REGION_WORDS>, 2> queue_full_read_tick{};
    std::array<std::array<u64, NUM_REGION_WORDS>, 2> queue_full_write_tick{};
    struct QueuePageTicks {
        std::array<std::array<u64, NUM_REGION_PAGES>, 2> read{};
        std::array<std::array<u64, NUM_REGION_PAGES>, 2> write{};
    };
    std::unique_ptr<QueuePageTicks> queue_page_ticks;
    u64 compute_words{};
    std::array<u16, NUM_REGION_PAGES> fault_frame{};
    std::array<u8, NUM_REGION_PAGES> fault_streak{};
    std::array<u16, NUM_REGION_PAGES> write_frame{}; ///< Frame of the last write fault.
    std::array<u64, NUM_REGION_PAGES> page_hash{}; ///< See SkipUnchangedHotPages.
    LockType mutex;
    /// Scheduler ticks and first-compute-use metadata shared by G/C. Keep this lock short;
    /// packet resolve/submit and driver waits happen after it is released.
    mutable LockType hazard_mutex;
};

} // namespace VideoCore
