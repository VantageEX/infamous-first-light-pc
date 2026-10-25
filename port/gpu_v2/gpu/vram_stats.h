// SPDX-License-Identifier: GPL-2.0-or-later
// VRAM held by the renderer's VMA allocations, by kind: the "other" part of VRAM_BREAKDOWN
// (usage less images and arena) on 8 GB cards was 1.5-1.8 GB nobody could name.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace iss::vram_stats {

/// Buffer memory types (VideoCore::MemoryType order), then images.
enum Kind : int { DeviceLocal, HostUncached, HostCached, Stream, Sparse, Images, KindCount };
constexpr const char* KindNames[KindCount] = {"device_local", "upload", "readback",
                                              "stream", "sparse", "images"};

struct Table {
    std::mutex mutex;
    std::unordered_map<VmaAllocation, std::pair<int, std::uint64_t>> allocations;
    std::array<std::atomic<std::uint64_t>, KindCount> bytes{};
};
inline Table& State() {
    static Table t;
    return t;
}

/// Counts `allocation` under `kind` when it landed in device local memory.
inline void Track(VmaAllocator allocator, VmaAllocation allocation, int kind) {
    if (!allocation) return;
    VkMemoryPropertyFlags flags{};
    vmaGetAllocationMemoryProperties(allocator, allocation, &flags);
    if (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return;
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(allocator, allocation, &info);
    auto& t = State();
    std::scoped_lock lock{t.mutex};
    t.allocations[allocation] = {kind, info.size};
    t.bytes[kind].fetch_add(info.size, std::memory_order_relaxed);
}

/// Before the allocation is destroyed.
inline void Untrack(VmaAllocation allocation) {
    if (!allocation) return;
    auto& t = State();
    std::scoped_lock lock{t.mutex};
    const auto it = t.allocations.find(allocation);
    if (it == t.allocations.end()) return;
    t.bytes[it->second.first].fetch_sub(it->second.second, std::memory_order_relaxed);
    t.allocations.erase(it);
}

inline std::uint64_t Bytes(int kind) {
    return State().bytes[kind].load(std::memory_order_relaxed);
}

} // namespace iss::vram_stats
