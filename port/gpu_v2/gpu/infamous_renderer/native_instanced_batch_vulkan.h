#pragma once

#include "native_instanced_batch.h"

#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"

namespace Shader { struct PushData; }
namespace VideoCore { struct StreamBuffer; }
namespace Vulkan { class GraphicsPipeline; class Instance; class Scheduler; }

namespace iss::native_geometry {

enum class InstancedRecordStatus {
    Recorded,
    Disabled,
    InvalidBatch,
    MissingProof,
    InvalidPipelineOrLayout,
    DeviceUnsupported,
    DescriptorMismatch,
    StreamUnavailable,
};

class InstancedDeviceCaps {
public:
    static InstancedDeviceCaps Capture(const Vulkan::Instance& instance);
    bool Supports(const Vulkan::Instance& instance, std::uint32_t draws,
                  std::size_t storage_bytes) const;
    vk::DeviceSize StorageAlignment() const { return storage_alignment; }

private:
    const Vulkan::Instance* owner{};
    bool multi_draw_indirect{};
    bool indirect_first_instance_enabled{};
    std::uint32_t max_draw_indirect_count{};
    std::uint32_t max_storage_buffer_range{};
    vk::DeviceSize storage_alignment{};
};

// BindResources(pipeline) has prepared descriptor writes for the first draw,
// but must not yet have committed them. The adapter replaces only VS ssbo_1
// (set=0,binding=0) with the packed 72-byte records, then binds resources and
// records the geometry. Every failure occurs before Vulkan commands are emitted.
struct InstancedVulkanRecordContext {
    const Vulkan::Instance& instance;
    const InstancedDeviceCaps& caps;
    Vulkan::Scheduler& scheduler;
    VideoCore::StreamBuffer& stream;
    const Vulkan::GraphicsPipeline& pipeline;
    const std::vector<vk::WriteDescriptorSet>& descriptor_writes;
    const Shader::PushData& push_data;
    vk::VertexInputBindingDescription2EXT vertex_binding{};
    vk::VertexInputAttributeDescription2EXT vertex_attribute{};
    RecordProof proof{};
    bool opt_in_enabled{};
    bool rendering_scope_active{};
    bool dynamic_state_bound{};
    bool exact_spirv_verified{};
};

InstancedRecordStatus TryUploadAndRecordInstanced(
    const InstancedBuildResult& batch, const InstancedVulkanRecordContext& context);

} // namespace iss::native_geometry
