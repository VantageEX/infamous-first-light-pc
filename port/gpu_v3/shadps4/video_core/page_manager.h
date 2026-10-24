// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache//region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace Core {
enum class MemoryPermission : u32;
}

namespace VideoCore {

class PageManager {
    // PAGE_SIZE and PAGE_BITS conflicts with machine/param.h definitions on freebsd!
    // Use the same page size as the tracker.
    static constexpr size_t PM_PAGE_BITS = TRACKER_PAGE_BITS;
    static constexpr size_t PM_PAGE_SIZE = TRACKER_BYTES_PER_PAGE;

    // Keep the lock granularity the same as region granularity. (since each regions has
    // itself a lock)
    static constexpr size_t PAGES_PER_LOCK = NUM_PAGES_PER_REGION;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// A CPU protection change the page manager does not own (guest mprotect, file-map
    /// protects). Serialized with its own protects when SHADPS4_PAGE_OS_SHADOW is on.
    void ExternalProtect(VAddr address, u64 size, Core::MemoryPermission perms) const;

    /// Updates watches in the pages touching the specified region.
    template <bool track>
    void UpdatePageWatchers(VAddr addr, u64 size) const;

    /// Updates watches in the pages touching the specified region using a mask.
    template <bool track, bool is_read = false>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const;

    /// Deferred write protection (SHADPS4_DEFERRED_PROTECT). While a sink is installed on the
    /// calling thread, write-protect requests update watcher counts as usual but append the
    /// affected ranges to the sink instead of calling the OS. ReconcileProtection later applies
    /// each range's *current* watcher state under the same range locks every other protection
    /// change takes, so it is order-independent with concurrent fault-handler unprotects.
    struct ProtectRange {
        VAddr addr;
        u64 size;
    };
    static void SetDeferredProtectSink(std::vector<ProtectRange>* sink);
    void ReconcileProtection(std::span<const ProtectRange> ranges) const;

    /// SHADPS4_LEAN worker pre-check: true if any page of [addr, addr + size) is read-protected
    /// (a read watcher). Lock-free, racy byte reads: a page protected right after this returns
    /// faults, and the worker's fallback counts that.
    [[nodiscard]] bool AnyReadProtected(VAddr addr, u64 size) const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
