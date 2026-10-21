// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "tlg_toggles.h"
#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
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

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
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
        if constexpr (type == Type::CPU && enable) {
            CountWriteFaults(start_page, end_page);
            NoteCpuBitsSet();
        }
        if constexpr (type == Type::GPU && enable) {
            NoteGpuBitsSet();
        }
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            if constexpr (!enable) {
                NoteCpuBitsCleared();
            }
            UpdateProtection<!enable, false>();
        } else {
            if constexpr (!enable) {
                NoteGpuBitsCleared();
            }
            if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
                UpdateProtection<enable, true>();
            }
        }
    }

    /// tlg: TLG_RELAXED_RANGES=<start>-<end>[,...] (hex guest addresses): in precise readback
    /// mode, the 4 KiB pages of these ranges stay readable when the GPU writes them, as in
    /// relaxed mode. For memory the game rewrites every frame but only the GPU reads: First
    /// Light's particle lists (double buffered) faulted each frame into GPU waits (about 45 to
    /// 25 FPS). Their emitter counters, next to them, must stay precise (without them part of
    /// the dash effects vanish), hence the page granularity.
    static const std::vector<std::pair<VAddr, VAddr>>& RelaxedRanges() {
        static const std::vector<std::pair<VAddr, VAddr>> ranges = [] {
            std::vector<std::pair<VAddr, VAddr>> out;
            const char* env = std::getenv("TLG_RELAXED_RANGES");
            for (const char* p = env; p && *p;) {
                char* end = nullptr;
                const VAddr start = std::strtoull(p, &end, 16);
                if (end == p || *end != '-') {
                    break;
                }
                p = end + 1;
                const VAddr stop = std::strtoull(p, &end, 16);
                if (end == p || stop <= start) {
                    break;
                }
                out.emplace_back(start, stop);
                p = *end == ',' ? end + 1 : end;
            }
            return out;
        }();
        return ranges;
    }

    /// The pages of this region in TLG_RELAXED_RANGES, or null when none (built once).
    const RegionBits* RelaxedPages() {
        if (!relaxed_checked) {
            relaxed_checked = true;
            const VAddr region_end = cpu_addr + TRACKER_HIGHER_PAGE_SIZE;
            for (const auto& [start, stop] : RelaxedRanges()) {
                if (start >= region_end || stop <= cpu_addr) {
                    continue;
                }
                const size_t first = (std::max(start, cpu_addr) - cpu_addr) / TRACKER_BYTES_PER_PAGE;
                const size_t last = Common::DivCeil(std::min(stop, region_end) - cpu_addr,
                                                    TRACKER_BYTES_PER_PAGE);
                relaxed_pages.SetRange(first, last);
                relaxed_any = true;
            }
        }
        return relaxed_any ? &relaxed_pages : nullptr;
    }

    /// tlg: after a guest write fault, also unprotects the other pages of the window that
    /// are protected for CPU writes and hold no GPU-modified data (see MarkFaultWindow).
    void ExtendWriteFault(VAddr window_addr, u64 size) {
        const size_t offset = window_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }
        RegionBits add(~(gpu | writeable), start_page, end_page);
        if (add.None()) {
            return;
        }
        NoteCpuBitsSet();
        cpu |= add;
        UpdateProtection<false, false>();
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
        // tlg: nothing modified in the range means nothing to clear or re-protect (CPU bits
        // always match `writeable` after a change); this runs for every buffer binding, so
        // it is checked before building the range mask.
        if (type == Type::CPU && !TlgToggle::Disabled(TlgToggle::PageTrackingEarlyExit) &&
            (!cpu_any.load(std::memory_order_acquire) || !bits.AnyInRange(start_page, end_page))) {
            return;
        }
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                KeepHotPagesModified(start_page, end_page);
                NoteCpuBitsCleared();
                UpdateProtection<true, false>();
            } else {
                NoteGpuBitsCleared();
                if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
                    UpdateProtection<false, true>();
                }
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
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

        if constexpr (type == Type::CPU) {
            // tlg: the common case, nothing of this region modified since its upload, costs
            // one load instead of a scan of the bitmap (this runs for every buffer of every
            // draw, without the lock: see the note on cpu_any).
            if (!cpu_any.load(std::memory_order_acquire)) {
                return false;
            }
        } else {
            // tlg: the same early-out for the GPU bits: they are empty for most regions
            // between two frames, and this is asked for every guest memcpy/memset into the
            // range (BufferCache::CpuWriteMerge) and for every read fault (see gpu_any).
            if (!gpu_any.load(std::memory_order_acquire)) {
                return false;
            }
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
        // tlg: pages in TLG_RELAXED_RANGES stay readable whatever the GPU wrote.
        RegionBits want_readable = ~gpu;
        if constexpr (is_read) {
            if (const RegionBits* relaxed = RelaxedPages()) {
                want_readable |= *relaxed;
            }
        }
        RegionBits mask = is_read ? (want_readable ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = want_readable;
        } else {
            writeable = cpu;
        }
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
    }

    // tlg: pages the guest writes again and again (per-frame constants, skinning output)
    // cost a protection fault in the writing thread plus an mprotect with TLB shootdowns on
    // every upload. After HotFaults faults a page stays writable and counts as always CPU
    // modified, so it is uploaded on every use instead. The set is rebuilt every HotPeriod.
    // Opt-in (TLG_HOT_PAGES=1): in the test scene the set grew to ~14k pages (56 MB), each
    // re-uploaded on every binding, which cost far more than the faults (33 FPS) and preceded
    // a GPU ring timeout.
    static bool HotPagesEnabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("TLG_HOT_PAGES");
            return env && env[0] == '1';
        }();
        return enabled && !TlgToggle::Disabled(TlgToggle::HotPages);
    }
    // A short period re-protects thousands of pages at once and each faults again before it is
    // hot, a fault storm (a 2 s period gave ~20k faults/s and frame spikes).
    static constexpr u8 HotFaults = 2;
    static constexpr auto HotPeriod = std::chrono::seconds(20);

    void CountWriteFaults(size_t start_page, size_t end_page) {
        const bool hot_enabled = HotPagesEnabled();
        for (size_t page = start_page; page < end_page; ++page) {
            if (writeable.Get(page)) {
                continue; // not protected: no fault
            }
            TlgStats::tracker_faults.fetch_add(1, std::memory_order_relaxed);
            if (!hot_enabled) {
                continue;
            }
            if (write_faults[page] < HotFaults) {
                ++write_faults[page];
                continue;
            }
            if (!hot.Get(page)) {
                if (num_hot == 0) {
                    hot_since = std::chrono::steady_clock::now();
                }
                hot.Set(page);
                ++num_hot;
                TlgStats::hot_pages.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // tlg: cpu_any is true whenever any bit of `cpu` may be set. It is stored before the bits
    // on every path that sets them (all under `lock`), so a reader without the lock that sees
    // a set bit also sees the flag (x86 keeps store order; the lock-free readers rely on the
    // same ordering for the bits themselves, see MemoryTracker::ForEachUploadRange). A reader
    // racing with a clear can only see a stale true, which falls back to the exact scan.
    void NoteCpuBitsSet() {
        if (!cpu_any.load(std::memory_order_relaxed)) {
            cpu_any.store(true, std::memory_order_seq_cst);
        }
    }
    void NoteCpuBitsCleared() {
        cpu_any.store(cpu.Any(), std::memory_order_release);
    }

    // tlg: gpu_any is the mirror of cpu_any for the GPU bits, with the same contract: set
    // before the bits on every path that sets them (all under `lock`, ChangeRegionState and
    // ForEachModifiedRange are the only writers), so a reader without the lock that sees a set
    // bit also sees the flag; a reader racing with a clear sees a stale true and falls back to
    // the exact scan.
    void NoteGpuBitsSet() {
        if (!gpu_any.load(std::memory_order_relaxed)) {
            gpu_any.store(true, std::memory_order_seq_cst);
        }
    }
    void NoteGpuBitsCleared() {
        gpu_any.store(gpu.Any(), std::memory_order_release);
    }

    void KeepHotPagesModified(size_t start_page, size_t end_page) {
        if (num_hot == 0) {
            return;
        }
        if (!HotPagesEnabled() ||
            std::chrono::steady_clock::now() - hot_since > HotPeriod) {
            // Re-protect them (on this upload) and start counting again.
            TlgStats::hot_pages.fetch_sub(num_hot, std::memory_order_relaxed);
            hot.Clear();
            write_faults.fill(0);
            num_hot = 0;
            return;
        }
        NoteCpuBitsSet();
        cpu |= RegionBits(hot, start_page, end_page);
    }

    std::array<u8, NUM_PAGES_PER_REGION> write_faults{};
    RegionBits hot{};
    u32 num_hot = 0;
    std::chrono::steady_clock::time_point hot_since{};

    PageManager* tracker;
    VAddr cpu_addr = 0;
    std::atomic<bool> cpu_any{true}; ///< see NoteCpuBitsSet; the constructor fills `cpu`
    std::atomic<bool> gpu_any{false}; ///< see NoteGpuBitsSet; the constructor clears `gpu`
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
    RegionBits relaxed_pages; ///< TLG_RELAXED_RANGES pages (RelaxedPages)
    bool relaxed_checked = false, relaxed_any = false;
};

} // namespace VideoCore
