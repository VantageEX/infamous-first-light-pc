// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {

class BufferCache;

class FaultManager {
    static constexpr size_t MaxPendingFaults = 8;

public:
    explicit FaultManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          BufferCache& buffer_cache, u32 caching_pagebits, u64 caching_num_pages);

    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return &fault_buffer;
    }

    void ProcessFaultBuffer();
    /// Faulted pages handed to FindBuffer so far (SHADPS4_NATIVE_PT's "NativePt faults" line).
    [[nodiscard]] u64 PagesProcessed() const noexcept {
        return pages_processed.load(std::memory_order_relaxed);
    }

private:
    std::atomic<u64> pages_processed{};
    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    RangeSet fault_ranges;
    u64 caching_pagesize;
    u64 caching_num_pages;
    u64 fault_buffer_size;
    Buffer fault_buffer;
    Buffer download_buffer;
    std::array<u64, MaxPendingFaults> fault_areas{};
    u32 current_area{};
    vk::UniqueDescriptorSetLayout fault_process_desc_layout;
    vk::UniquePipeline fault_process_pipeline;
    vk::UniquePipelineLayout fault_process_pipeline_layout;
};

} // namespace VideoCore
