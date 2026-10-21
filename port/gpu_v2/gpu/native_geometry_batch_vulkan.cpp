#include "infamous_renderer/native_geometry_batch_vulkan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "video_core/buffer_cache/buffer.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

namespace iss::native_geometry {
namespace {

constexpr vk::BufferUsageFlags kRequiredStreamUsage =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;
static_assert((VideoCore::AllFlags & kRequiredStreamUsage) == kRequiredStreamUsage);

constexpr std::array<std::uint32_t, 3> kFormatWords{418732, 299948, 193068};

const std::array<vk::Format, 3>& ExpectedFormats() {
    static const auto formats = [] {
        std::array<vk::Format, 3> result{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            AmdGpu::Buffer sharp{};
            sharp.num_format = (kFormatWords[i] >> 12) & 7;
            sharp.data_format = (kFormatWords[i] >> 15) & 15;
            result[i] = Vulkan::LiverpoolToVK::SurfaceFormat(sharp.GetDataFmt(),
                                                              sharp.GetNumberFmt());
        }
        return result;
    }();
    return formats;
}

constexpr std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool ValidLayout(const VulkanRecordContext& context) {
    const auto& formats = ExpectedFormats();
    for (std::uint32_t i = 0; i < 3; ++i) {
        const auto& b = context.bindings[i];
        const auto& a = context.attributes[i];
        if (b.pNext || b.binding != i || b.stride != 8 ||
            b.inputRate != vk::VertexInputRate::eVertex || b.divisor != 1 ||
            a.pNext || a.location != i || a.binding != i ||
            a.format != formats[i] || a.offset != 0) {
            return false;
        }
    }
    return true;
}

struct Offsets {
    std::array<std::size_t, 3> vertices{};
    std::size_t indices{};
    std::size_t commands{};
    std::size_t total{};
};

Offsets GetOffsets(const PackedBatch& batch) {
    Offsets out{};
    std::size_t cursor = 0;
    for (std::uint32_t i = 0; i < 3; ++i) {
        cursor = AlignUp(cursor, 8);
        out.vertices[i] = cursor;
        cursor += batch.vertices[i].size();
    }
    cursor = AlignUp(cursor, 2);
    out.indices = cursor;
    cursor += batch.indices.size();
    cursor = AlignUp(cursor, 4);
    out.commands = cursor;
    cursor += batch.commands.size() * sizeof(DrawIndexedIndirectCommand);
    out.total = cursor;
    return out;
}

bool ValidPackedData(const PackedBatch& batch) {
    if (batch.draw_count < 2 || batch.draw_count > 256 ||
        batch.draw_count != batch.commands.size() ||
        batch.indices.empty() || batch.indices.size() > 4 * 1024 * 1024 ||
        batch.indices.size() % 2) {
        return false;
    }
    const auto stream_bytes = batch.vertices[0].size();
    if (!stream_bytes || stream_bytes > 16 * 1024 * 1024 || stream_bytes % 8 ||
        stream_bytes != batch.vertices[1].size() ||
        stream_bytes != batch.vertices[2].size()) {
        return false;
    }
    for (const auto& command : batch.commands) {
        if (!command.index_count || command.instance_count != 1 ||
            command.first_instance != 0 ||
            std::uint64_t(command.first_index) + command.index_count >
                batch.indices.size() / 2) {
            return false;
        }
    }
    return true;
}

} // namespace

DeviceCaps DeviceCaps::Capture(const Vulkan::Instance& instance) {
    DeviceCaps caps{};
    caps.owner = &instance;
    const auto physical = instance.GetPhysicalDevice();
    caps.multi_draw_indirect = physical.getFeatures().multiDrawIndirect;
    caps.max_draw_indirect_count = physical.getProperties().limits.maxDrawIndirectCount;
    return caps;
}

bool DeviceCaps::Supports(const Vulkan::Instance& instance, std::uint32_t count) const {
    return owner == &instance && multi_draw_indirect && count >= 2 &&
           count <= max_draw_indirect_count;
}

RecordStatus TryUploadAndRecord(const BuildResult& result,
                                const VulkanRecordContext& context) {
    if (!context.opt_in_enabled) return RecordStatus::Disabled;
    if (!result || !ValidPackedData(result.batch)) {
        return RecordStatus::InvalidBatch;
    }
    if (!MayAttemptRecord(result, context.proof) || !context.rendering_scope_active ||
        !context.descriptors_and_dynamic_state_bound) {
        return RecordStatus::MissingProof;
    }
    if (!context.pipeline ||
        context.graphics_pipeline_key_hash != 0x6794975E42E32374ull ||
        !ValidLayout(context)) {
        return RecordStatus::InvalidPipelineOrLayout;
    }
    if (!context.caps.Supports(context.instance, result.batch.draw_count)) {
        return RecordStatus::DeviceUnsupported;
    }

    const Offsets offsets = GetOffsets(result.batch);
    if (!offsets.total || offsets.total > context.stream.SizeBytes() ||
        offsets.total > std::numeric_limits<vk::DeviceSize>::max() ||
        context.stream.mem_type != VideoCore::MemoryType::Stream ||
        !context.stream.Handle()) {
        return RecordStatus::StreamUnavailable;
    }
    // A single reservation ensures a failure cannot leave some data committed
    // while still requiring fallback. Commit uses this stream's scheduler tick
    // watch and flushes non-coherent host memory, as for ordinary vertex uploads.
    const auto [mapped, base] = context.stream.Map(offsets.total, 16, false);
    if (!mapped) return RecordStatus::StreamUnavailable;
    for (std::uint32_t i = 0; i < 3; ++i) {
        std::memcpy(mapped + offsets.vertices[i], result.batch.vertices[i].data(),
                    result.batch.vertices[i].size());
    }
    std::memcpy(mapped + offsets.indices, result.batch.indices.data(),
                result.batch.indices.size());
    std::memcpy(mapped + offsets.commands, result.batch.commands.data(),
                result.batch.commands.size() * sizeof(DrawIndexedIndirectCommand));
    context.stream.Commit();

    const std::array<vk::Buffer, 3> buffers{
        context.stream.Handle(), context.stream.Handle(), context.stream.Handle()};
    const std::array<vk::DeviceSize, 3> vertex_offsets{
        base + offsets.vertices[0], base + offsets.vertices[1], base + offsets.vertices[2]};
    const std::array<vk::DeviceSize, 3> sizes{
        result.batch.vertices[0].size(), result.batch.vertices[1].size(),
        result.batch.vertices[2].size()};
    const std::array<vk::DeviceSize, 3> strides{8, 8, 8};
    const auto cmdbuf = context.scheduler.Recorder();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, context.pipeline);
    if (context.instance.IsVertexInputDynamicState()) {
        cmdbuf.setVertexInputEXT(context.bindings, context.attributes);
        cmdbuf.bindVertexBuffers(0, 3, buffers.data(), vertex_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, 3, buffers.data(), vertex_offsets.data(),
                                  sizes.data(), strides.data());
    }
    cmdbuf.bindIndexBuffer(context.stream.Handle(), base + offsets.indices,
                           vk::IndexType::eUint16);
    cmdbuf.drawIndexedIndirect(context.stream.Handle(), base + offsets.commands,
                               result.batch.draw_count,
                               sizeof(DrawIndexedIndirectCommand));
    return RecordStatus::Recorded;
}

} // namespace iss::native_geometry
