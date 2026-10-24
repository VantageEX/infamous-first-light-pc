// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/**
 * SHADPS4_DRAW_PACKET=1 (=2 verify; default off): one compact recording-worker record per draw
 * for everything the legacy descriptor path records after the bindings are resolved - buffer
 * barriers, push constants, push descriptors, and the draw or dispatch itself when nothing else
 * would be recorded between the descriptors and it (notes/back-split-design-2026-09-28.md,
 * phase 1). The command-processor back copies flat bytes only: the PushData, the resolved
 * buffer/image infos in binding order and the pipeline's binding table (binding number,
 * descriptor count, descriptor type). The worker rebuilds the vk::WriteDescriptorSet array from
 * those and issues the vkCmd* calls in exactly the legacy order.
 *
 * Payload layout, each section aligned to max_align_t like Scheduler::CopyPayload:
 *   DrawPacketHeader | barriers[num_barriers] | table[num_writes] |
 *   buffer_infos[num_buffer_infos] | image_infos[num_image_infos]
 */
struct PacketBinding {
    u16 binding; ///< unified binding number (dstBinding)
    u16 count;   ///< descriptorCount (> 1 only for mip-fallback image arrays)
    u32 type;    ///< VkDescriptorType
};
static_assert(sizeof(PacketBinding) == 8);
static_assert(std::is_trivially_copyable_v<PacketBinding>);

/// Writes per packet the worker rebuilds on its stack; push-descriptor layouts are bounded by
/// the device's maxPushDescriptors (32 on NVIDIA), larger layouts take the legacy path.
constexpr u32 DrawPacketMaxWrites = 64;
/// Same payload cap as Scheduler::RecordArrays / PushDescriptors (chunks are 64 KB).
constexpr size_t DrawPacketMaxBytes = 32768;

enum DrawPacketFlags : u32 {
    DrawPacketDraw = 1u << 0,     ///< bindPipeline(graphics) + draw / drawIndexed
    DrawPacketIndexed = 1u << 1,  ///< with DrawPacketDraw: drawIndexed
    DrawPacketDispatch = 1u << 2, ///< bindPipeline(compute) + dispatch
    /// SHADPS4_DRAW_PACKET_DELTA: the set pushed by the previous graphics packet is still bound
    /// and equal; this packet carries no table and no infos (statistics only on the worker).
    DrawPacketKeepDescriptors = 1u << 3,
};

struct DrawPacketHeader {
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    vk::ShaderStageFlags stage_flags{};
    vk::PipelineBindPoint bind_point{vk::PipelineBindPoint::eGraphics};
    u32 flags{};
    u32 num_barriers{};
    u32 num_writes{};
    u32 num_buffer_infos{};
    u32 num_image_infos{};
    // Draw arguments (DrawPacketDraw).
    u32 count{};
    u32 instances{};
    u32 vertex_offset{};
    u32 instance_offset{};
    // Dispatch arguments (DrawPacketDispatch).
    u32 dim_x{};
    u32 dim_y{};
    u32 dim_z{};
    Shader::PushData push_data{};
};
static_assert(std::is_trivially_copyable_v<DrawPacketHeader>);

/// What the back hands to Scheduler::RecordDrawPacket; the spans are copied into the record.
struct DrawPacketDesc {
    DrawPacketHeader header{};
    std::span<const vk::BufferMemoryBarrier2> barriers{};
    std::span<const PacketBinding> table{};
    std::span<const vk::DescriptorBufferInfo> buffer_infos{};
    std::span<const vk::DescriptorImageInfo> image_infos{};
};

constexpr size_t DrawPacketAlign(size_t bytes) {
    constexpr size_t alignment = alignof(std::max_align_t);
    return (bytes + alignment - 1) & ~(alignment - 1);
}

constexpr size_t DrawPacketPayloadBytes(size_t num_barriers, size_t num_writes,
                                        size_t num_buffer_infos, size_t num_image_infos) {
    return DrawPacketAlign(sizeof(DrawPacketHeader)) +
           DrawPacketAlign(num_barriers * sizeof(vk::BufferMemoryBarrier2)) +
           DrawPacketAlign(num_writes * sizeof(PacketBinding)) +
           DrawPacketAlign(num_buffer_infos * sizeof(vk::DescriptorBufferInfo)) +
           DrawPacketAlign(num_image_infos * sizeof(vk::DescriptorImageInfo));
}

/// Bytes one packet reserves in a worker chunk: the command header (two function pointers, an
/// object pointer and the next offset), the 16-byte callable, then the payload. Estimated from
/// CommandRecordingWorker's layout, not read back from it.
constexpr size_t DrawPacketRecordBytes(size_t num_barriers, size_t num_writes,
                                       size_t num_buffer_infos, size_t num_image_infos) {
    return 32 + 16 +
           DrawPacketPayloadBytes(num_barriers, num_writes, num_buffer_infos, num_image_infos);
}

/// Bytes the legacy path reserves for the same records (estimate from the same chunk layout:
/// 32-byte command header + the captured callable + payload). Barriers: Scheduler::RecordArrays
/// (callable ~48 B + the barrier array). Push constants: Scheduler::Record of a lambda holding
/// layout, stage flags and the 128-byte PushData (~152 B). Descriptors: Scheduler::PushDescriptors
/// (callable ~40 B, 64 B per WriteDescriptorSet, each write's infos padded to 16 B). Draw or
/// dispatch: ~40-byte lambda. The endRendering record that precedes non-empty barriers is the
/// same on both paths and is not counted.
inline size_t DrawPacketLegacyBytes(size_t num_barriers, std::span<const PacketBinding> table,
                                    bool with_tail) {
    size_t bytes = 32 + 152; // push constants
    if (num_barriers != 0) {
        bytes += 32 + 48 + DrawPacketAlign(num_barriers * sizeof(vk::BufferMemoryBarrier2));
    }
    if (!table.empty()) {
        bytes += 32 + 40 + DrawPacketAlign(table.size() * sizeof(vk::WriteDescriptorSet));
        for (const auto& binding : table) {
            // Buffer and image infos are both 24 bytes.
            bytes += DrawPacketAlign(size_t{binding.count} * sizeof(vk::DescriptorImageInfo));
        }
    }
    if (with_tail) {
        bytes += 32 + 40;
    }
    return bytes;
}

/// The sections of a recorded packet, located from its header exactly as RecordDrawPacket
/// laid them out (an empty section occupies no bytes).
struct DrawPacketView {
    const DrawPacketHeader* header{};
    std::span<const vk::BufferMemoryBarrier2> barriers{};
    std::span<const PacketBinding> table{};
    std::span<const vk::DescriptorBufferInfo> buffer_infos{};
    std::span<const vk::DescriptorImageInfo> image_infos{};
};

inline DrawPacketView ViewDrawPacket(const DrawPacketHeader& header) {
    DrawPacketView view{&header};
    const std::byte* cursor =
        reinterpret_cast<const std::byte*>(&header) + DrawPacketAlign(sizeof(DrawPacketHeader));
    view.barriers = std::span<const vk::BufferMemoryBarrier2>(
        reinterpret_cast<const vk::BufferMemoryBarrier2*>(cursor), size_t{header.num_barriers});
    cursor += DrawPacketAlign(header.num_barriers * sizeof(vk::BufferMemoryBarrier2));
    view.table = std::span<const PacketBinding>(reinterpret_cast<const PacketBinding*>(cursor),
                                                size_t{header.num_writes});
    cursor += DrawPacketAlign(header.num_writes * sizeof(PacketBinding));
    view.buffer_infos = std::span<const vk::DescriptorBufferInfo>(
        reinterpret_cast<const vk::DescriptorBufferInfo*>(cursor), size_t{header.num_buffer_infos});
    cursor += DrawPacketAlign(header.num_buffer_infos * sizeof(vk::DescriptorBufferInfo));
    view.image_infos = std::span<const vk::DescriptorImageInfo>(
        reinterpret_cast<const vk::DescriptorImageInfo*>(cursor), size_t{header.num_image_infos});
    return view;
}

constexpr bool IsPacketBufferType(vk::DescriptorType type) {
    return type == vk::DescriptorType::eStorageBuffer || type == vk::DescriptorType::eUniformBuffer;
}

/// Rebuilds the WriteDescriptorSet array of one packet: entry i of the table becomes write i,
/// pointing at the next `count` buffer or image infos in order (the order BindBuffers /
/// BindTextures / EmitTexMemoHit append them, which is the layout's unified binding order).
/// Used by the recording worker, and by the back in verify mode (one source for both).
/// Returns the number of writes, or ~0u when the infos do not add up to the table.
inline u32 BuildDrawPacketWrites(std::span<const PacketBinding> table,
                                 std::span<const vk::DescriptorBufferInfo> buffer_infos,
                                 std::span<const vk::DescriptorImageInfo> image_infos,
                                 std::span<vk::WriteDescriptorSet> out) {
    if (table.size() > out.size()) {
        return ~0u;
    }
    size_t buffer_index = 0;
    size_t image_index = 0;
    for (size_t i = 0; i < table.size(); ++i) {
        const PacketBinding& binding = table[i];
        const auto type = static_cast<vk::DescriptorType>(binding.type);
        vk::WriteDescriptorSet write = {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.binding,
            .dstArrayElement = 0,
            .descriptorCount = binding.count,
            .descriptorType = type,
        };
        if (IsPacketBufferType(type)) {
            if (buffer_index + binding.count > buffer_infos.size()) {
                return ~0u;
            }
            write.pBufferInfo = buffer_infos.data() + buffer_index;
            buffer_index += binding.count;
        } else {
            if (image_index + binding.count > image_infos.size()) {
                return ~0u;
            }
            write.pImageInfo = image_infos.data() + image_index;
            image_index += binding.count;
        }
        out[i] = write;
    }
    if (buffer_index != buffer_infos.size() || image_index != image_infos.size()) {
        return ~0u;
    }
    return static_cast<u32>(table.size());
}

} // namespace Vulkan
