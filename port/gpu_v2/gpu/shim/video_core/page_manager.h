// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

struct UffdImpl;
struct SignalImpl;

enum class PageOp : s8 {
    None = 0,
    Track = 1,
    Untrack = -1,
};

class PageManager {
    static constexpr size_t PM_PAGE_BITS = 12;
    static constexpr size_t PM_PAGE_SIZE = 1ULL << PM_PAGE_BITS;

public:
    struct SrtWatchToken {
        VAddr page{};
        u64 generation{};
    };
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    // An SRT lease protects one guest virtual page against CPU writes. Callers must arm
    // every VA alias of every physical byte in their read set before claiming coverage.
    // Tokens from a faulted or remapped page fail Check; Release never removes a newer lease.
    [[nodiscard]] bool ArmSrtWriteWatch(VAddr address, SrtWatchToken& token) const;
    [[nodiscard]] bool CheckSrtWriteWatch(SrtWatchToken token) const;
    void ReleaseSrtWriteWatch(SrtWatchToken token) const;
    void InvalidateSrtWriteWatches(VAddr address, u64 size) const;
    void InvalidateAllSrtWriteWatches() const;
    // Direct host-shadow writes bypass page protection. The write callback executes while
    // no new SRT watch can be armed; do not fault or call back into PageManager here.
    void WithSrtDirectWrite(const std::function<void()>& write) const;
    // Acquire sorted page locks after the packet's coherence/hazard locks, then validate
    // and publish under those locks. The callback must be short and must not fault.
    [[nodiscard]] bool WithSrtWriteWatchesLocked(
        std::span<const SrtWatchToken> tokens,
        const std::function<bool()>& publish) const;

    /// Updates watches in the pages touching the specified region.
    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const;

    /// Updates watches in the pages touching the inclusive bounds using a mask.
    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) const;

    /**
     * Deferred protection: while a vector is set (by the calling thread, null to stop), the write
     * protection this thread adds again to pages (after uploading them) is not applied: their
     * ranges of page numbers [first, end) are added to the vector instead, for
     * RefreshDeferredProtect, and the pages are marked so that any other change of their
     * watchers applies it at once. The guest may write the pages until then without a fault:
     * whoever applies it must read their content only afterwards.
     */
    static void DeferProtect(std::vector<std::pair<u64, u64>>* deferred_pages);

    /// Applies the deferred protections of the pages [first_page, end_page) not applied yet.
    /// Returns the number of pages protected.
    u64 RefreshDeferredProtect(u64 first_page, u64 end_page) const;

    /**
     * Late protection (Tuning::late_protect), for a region lock section: from BeginLateProtect,
     * called with the region locked, the protections UpdatePageWatchersForRegion decides are not
     * applied at once: their pages stay locked, and EndLateProtect, called right after the region
     * lock is released, applies them and unlocks the pages. The system calls no longer run under
     * the region lock (other threads faulting in the region, the command processor), while each
     * page still gets its protections in the order of its watcher changes. Between the two, the
     * calling thread takes no other lock. BeginLateProtect returns false when late protection is
     * off or already begun (nothing to end then).
     */
    bool BeginLateProtect() const;
    void EndLateProtect() const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

private:
    friend struct UffdImpl;
    friend struct SignalImpl;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
