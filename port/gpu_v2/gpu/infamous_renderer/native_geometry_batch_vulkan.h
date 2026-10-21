#pragma once

#include "native_geometry_batch.h"

#include <array>
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
struct StreamBuffer;
}
namespace Vulkan {
class Instance;
class Scheduler;
}

namespace iss::native_geometry {

enum class RecordStatus {
    Recorded,
    Disabled,
    InvalidBatch,
    MissingProof,
    InvalidPipelineOrLayout,
    DeviceUnsupported,
    StreamUnavailable,
};

// Capture once per Vulkan::Instance, after logical device creation. The
// renderer enables multiDrawIndirect iff the selected physical device has it.
class DeviceCaps {
public:
    static DeviceCaps Capture(const Vulkan::Instance& instance);
    bool Supports(const Vulkan::Instance& instance, std::uint32_t count) const;

private:
    const Vulkan::Instance* owner{};
    bool multi_draw_indirect{};
    std::uint32_t max_draw_indirect_count{};
};

// The caller has already entered the correct rendering scope and bound the
// matching descriptors/push constants/dynamic state. This function only owns
// geometry bindings and the indexed indirect draw.
struct VulkanRecordContext {
    const Vulkan::Instance& instance;
    const DeviceCaps& caps;
    Vulkan::Scheduler& scheduler;
    VideoCore::StreamBuffer& stream;
    vk::Pipeline pipeline{};
    std::uint64_t graphics_pipeline_key_hash{};
    std::array<vk::VertexInputBindingDescription2EXT, 3> bindings{};
    std::array<vk::VertexInputAttributeDescription2EXT, 3> attributes{};
    RecordProof proof{};
    bool opt_in_enabled{};
    bool rendering_scope_active{};
    bool descriptors_and_dynamic_state_bound{};
};

// One Map/Commit in the existing StreamBuffer owns the three vertex streams,
// uint16 indices and indirect commands. Every check runs before any command is
// recorded. A non-Recorded result requires the caller to execute all original
// draws in order. No state or draw is emitted on rejection.
RecordStatus TryUploadAndRecord(const BuildResult& batch,
                                const VulkanRecordContext& context);

} // namespace iss::native_geometry
