#include "infamous_renderer/native_instanced_batch_vulkan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <optional>

#include "infamous_renderer/native_instanced_feature.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace iss::native_geometry {
namespace {

constexpr vk::BufferUsageFlags kRequiredStreamUsage =
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer;
static_assert((VideoCore::AllFlags & kRequiredStreamUsage) == kRequiredStreamUsage);

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

struct Offsets {
    std::size_t instances{};
    std::size_t vertices{};
    std::size_t indices{};
    std::size_t commands{};
    std::size_t total{};
};

Offsets GetOffsets(const PackedInstancedBatch& batch) {
    Offsets out{};
    out.instances = 0;
    std::size_t cursor = batch.instances.size();
    cursor = AlignUp(cursor, 8);
    out.vertices = cursor;
    cursor += batch.vertices.size();
    cursor = AlignUp(cursor, 2);
    out.indices = cursor;
    cursor += batch.indices.size();
    cursor = AlignUp(cursor, 4);
    out.commands = cursor;
    cursor += batch.commands.size() * sizeof(DrawIndexedIndirectCommand);
    out.total = cursor;
    return out;
}

bool ValidPackedData(const PackedInstancedBatch& batch) {
    if (batch.draw_count < 2 || batch.draw_count > 256 ||
        batch.commands.size() != batch.draw_count ||
        !IsApprovedInstancedKey(batch.graphics_pipeline_key_hash, batch.vertex_hash,
                                batch.vertex_permutation) ||
        batch.vertex_stride == 0 || batch.vertex_stride > 256 ||
        batch.vertices.empty() || batch.vertices.size() > 16 * 1024 * 1024 ||
        batch.vertices.size() % batch.vertex_stride ||
        batch.instances.empty() || batch.instances.size() > 16 * 1024 * 1024 ||
        batch.instances.size() % 72 ||
        batch.indices.empty() || batch.indices.size() > 4 * 1024 * 1024 ||
        batch.indices.size() % 2) return false;
    std::uint64_t next_instance = 0;
    std::uint64_t next_index = 0;
    for (const auto& command : batch.commands) {
        if (!command.index_count || !command.instance_count ||
            command.first_index != next_index ||
            command.first_instance != next_instance ||
            std::uint64_t(command.first_index) + command.index_count > batch.indices.size() / 2 ||
            std::uint64_t(command.first_instance) + command.instance_count >
                batch.instances.size() / 72) return false;
        next_instance += command.instance_count;
        next_index += command.index_count;
    }
    return next_instance == batch.instances.size() / 72 &&
           next_index == batch.indices.size() / 2;
}

std::optional<vk::Format> ExpectedVertexFormat(std::uint32_t format_word) {
    AmdGpu::Buffer sharp{};
    sharp.num_format = (format_word >> 12) & 7;
    sharp.data_format = (format_word >> 15) & 15;
    const auto data = sharp.GetDataFmt();
    const auto number = sharp.GetNumberFmt();
    const auto formats = Vulkan::LiverpoolToVK::SurfaceFormats();
    const auto found = std::ranges::find_if(formats, [&](const auto& item) {
        return item.data_format == data && item.number_format == number;
    });
    if (found == formats.end()) return std::nullopt;
    return found->vk_format;
}

bool ValidLayout(const PackedInstancedBatch& batch,
                 const InstancedVulkanRecordContext& context) {
    const auto& b = context.vertex_binding;
    const auto& a = context.vertex_attribute;
    const auto expected_format = ExpectedVertexFormat(batch.vertex_format_word);
    return expected_format && *expected_format != vk::Format::eUndefined &&
           !b.pNext && b.binding == 0 && b.stride == batch.vertex_stride &&
           b.inputRate == vk::VertexInputRate::eVertex && b.divisor == 1 &&
           !a.pNext && a.location == 0 && a.binding == 0 && a.offset == 0 &&
           a.format == *expected_format;
}

} // namespace

InstancedDeviceCaps InstancedDeviceCaps::Capture(const Vulkan::Instance& instance) {
    InstancedDeviceCaps caps{};
    caps.owner = &instance;
    const auto physical = instance.GetPhysicalDevice();
    const auto features = physical.getFeatures();
    const auto limits = physical.getProperties().limits;
    caps.multi_draw_indirect = features.multiDrawIndirect;
    caps.indirect_first_instance_enabled = features.drawIndirectFirstInstance &&
        IndirectFirstInstanceEnabledFor(instance);
    caps.max_draw_indirect_count = limits.maxDrawIndirectCount;
    caps.max_storage_buffer_range = limits.maxStorageBufferRange;
    caps.storage_alignment = instance.StorageMinAlignment();
    return caps;
}

bool InstancedDeviceCaps::Supports(const Vulkan::Instance& instance,
                                   std::uint32_t draws, std::size_t storage_bytes) const {
    return owner == &instance && multi_draw_indirect && indirect_first_instance_enabled &&
           draws >= 2 && draws <= max_draw_indirect_count &&
           storage_bytes && storage_bytes <= max_storage_buffer_range &&
           storage_alignment && (storage_alignment & (storage_alignment - 1)) == 0;
}

InstancedRecordStatus TryUploadAndRecordInstanced(
    const InstancedBuildResult& result, const InstancedVulkanRecordContext& context) {
    if (!context.opt_in_enabled) return InstancedRecordStatus::Disabled;
    if (!result || !ValidPackedData(result.batch)) return InstancedRecordStatus::InvalidBatch;
    const auto& batch = result.batch;
    if (!context.proof.complete_state_equal_at_record ||
        !context.proof.no_intervening_feedback_or_barrier ||
        !context.rendering_scope_active || !context.dynamic_state_bound ||
        !context.exact_spirv_verified) return InstancedRecordStatus::MissingProof;
    if (std::hash<Vulkan::GraphicsPipelineKey>{}(context.pipeline.GetGraphicsKey()) !=
            batch.graphics_pipeline_key_hash || !ValidLayout(batch, context) ||
        !context.pipeline.IsReady()) return InstancedRecordStatus::InvalidPipelineOrLayout;
    if (!context.caps.Supports(context.instance, batch.draw_count, batch.instances.size())) {
        return InstancedRecordStatus::DeviceUnsupported;
    }
    if (context.descriptor_writes.size() != 2) return InstancedRecordStatus::DescriptorMismatch;
    const auto& ssbo_write = context.descriptor_writes[0];
    if (ssbo_write.dstBinding != 0 || ssbo_write.dstArrayElement != 0 ||
        ssbo_write.descriptorCount != 1 ||
        ssbo_write.descriptorType != vk::DescriptorType::eStorageBuffer ||
        !ssbo_write.pBufferInfo) return InstancedRecordStatus::DescriptorMismatch;

    const Offsets offsets = GetOffsets(batch);
    if (!offsets.total || offsets.total > context.stream.SizeBytes() ||
        context.stream.mem_type != VideoCore::MemoryType::Stream ||
        !context.stream.Handle()) return InstancedRecordStatus::StreamUnavailable;
    // Prepare all allocations before reserving the stream: after Map succeeds,
    // Commit and command recording cannot fail into the individual-draw fallback.
    auto writes = context.descriptor_writes;
    vk::DescriptorBufferInfo packed_ssbo{};
    const auto [mapped, base] = context.stream.Map(offsets.total,
        std::max<std::size_t>(16, context.caps.StorageAlignment()), false);
    if (!mapped) return InstancedRecordStatus::StreamUnavailable;
    std::memcpy(mapped + offsets.instances, batch.instances.data(), batch.instances.size());
    std::memcpy(mapped + offsets.vertices, batch.vertices.data(), batch.vertices.size());
    std::memcpy(mapped + offsets.indices, batch.indices.data(), batch.indices.size());
    std::memcpy(mapped + offsets.commands, batch.commands.data(),
                batch.commands.size() * sizeof(DrawIndexedIndirectCommand));
    context.stream.Commit();

    packed_ssbo = {context.stream.Handle(), base + offsets.instances, batch.instances.size()};
    writes[0].pBufferInfo = &packed_ssbo;
    Shader::PushData push_data = context.push_data;
    push_data.buf_offsets[0] = 0; // Packed SSBO starts exactly at its aligned descriptor offset.
    context.pipeline.BindResources(writes, push_data);
    const auto cmdbuf = context.scheduler.Recorder();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, context.pipeline.Handle());
    if (context.instance.IsVertexInputDynamicState()) {
        const std::array bindings{context.vertex_binding};
        const std::array attributes{context.vertex_attribute};
        cmdbuf.setVertexInputEXT(bindings, attributes);
        const vk::Buffer buffer = context.stream.Handle();
        const vk::DeviceSize offset = base + offsets.vertices;
        cmdbuf.bindVertexBuffers(0, 1, &buffer, &offset);
    } else {
        const vk::Buffer buffer = context.stream.Handle();
        const vk::DeviceSize offset = base + offsets.vertices;
        const vk::DeviceSize size = batch.vertices.size();
        const vk::DeviceSize stride = batch.vertex_stride;
        cmdbuf.bindVertexBuffers2(0, 1, &buffer, &offset, &size, &stride);
    }
    cmdbuf.bindIndexBuffer(context.stream.Handle(), base + offsets.indices,
                           vk::IndexType::eUint16);
    cmdbuf.drawIndexedIndirect(context.stream.Handle(), base + offsets.commands,
                               batch.draw_count, sizeof(DrawIndexedIndirectCommand));
    return InstancedRecordStatus::Recorded;
}

} // namespace iss::native_geometry
