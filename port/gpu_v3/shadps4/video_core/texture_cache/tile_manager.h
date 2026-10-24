// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace VideoCore {

/// SHADPS4_ASYNC_UPLOAD_RECORD=1: texture uploads queue their pre-copy buffer barrier and detile
/// dispatch on the recording worker instead of taking the command buffer directly, which first
/// drains the worker (measured 2026-09-24: RefreshImage drains were ~1.1% of the CP's time).
inline bool AsyncUploadRecord() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASYNC_UPLOAD_RECORD");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

struct ImageInfo;
struct Image;
class StreamBuffer;

class TileManager {
    static constexpr size_t NUM_BPPS = 5;

public:
    using ScratchBuffer = std::pair<vk::Buffer, VmaAllocation>;
    using Result = std::pair<vk::Buffer, u32>;

    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   vk::Buffer out_buffer, u32 out_offset, u32 copy_size);

    Result DetileImage(vk::Buffer in_buffer, u32 in_offset, const ImageInfo& info);

private:
    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler);
    ScratchBuffer GetScratchBuffer(u32 size);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> detilers{};
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> tilers{};
};

} // namespace VideoCore
