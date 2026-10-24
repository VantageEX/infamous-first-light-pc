// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_native_bindings_api.h"

#include <array>
#include <memory>
#include <span>
#include <vector>
#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;
class NativeBindings;
struct DeferredPipeline;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        // SHADPS4_ASYNC_COMPUTE_PIPELINES: the driver compile may still be pending.
        if (deferred) [[unlikely]] {
            WaitDeferred();
        }
        return *pipeline;
    }

    /// SHADPS4_ASYNC_COMPUTE_PIPELINES: the lookup thread found this pipeline; if its boot-preload
    /// compile is still queued, move it ahead of the backlog.
    void PromoteDeferred() const noexcept {
        if (deferred) [[unlikely]] {
            PromoteDeferredSlow();
        }
    }

    /// SHADPS4_PRECOMPILE (fix B): drop a still-queued deferred compile of a pipeline that will
    /// never be used again (RetirePrecompiled). True if a queued compile was dropped.
    bool CancelDeferred() noexcept;

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    /// l4g: a set-0 descriptor set of this (non-push) pipeline from the master-timeline heap,
    /// with `writes` written into it (dstSet is filled here). The back's thread only.
    vk::DescriptorSet CommitSetWrites(std::span<vk::WriteDescriptorSet> writes) const;

    /// Descriptors pushed per dispatch (no descriptor set from the master-timeline heap).
    [[nodiscard]] bool UsesPushDescriptors() const noexcept {
        return uses_push_descriptors && !uses_native_bindings;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::LogicalStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::LogicalStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    using BufferBarriers = boost::container::small_vector<vk::BufferMemoryBarrier2, 16>;

    void BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                       const Shader::PushData& push_data) const;

    /// SHADPS4_DRAW_PACKET (vk_draw_packet.h): one entry per descriptor write of the legacy
    /// layout, in unified binding order - the order BindResources fills set_writes. Built once
    /// with the descriptor set layout; empty for native bindings. Immutable afterwards.
    [[nodiscard]] std::span<const PacketBinding> PacketBindings() const noexcept {
        return packet_bindings;
    }
    /// Buffer infos / image infos (images plus samplers) one draw of this layout resolves.
    [[nodiscard]] u32 PacketBufferInfos() const noexcept {
        return packet_buffer_infos;
    }
    [[nodiscard]] u32 PacketImageInfos() const noexcept {
        return packet_image_infos;
    }

    /// The command-processor side of RecordBufferBarriers (batch-timer counters and the end of
    /// the rendering scope) when the barriers themselves travel in a draw packet (the packet is
    /// recorded before anything else, so the barrier still follows this endRendering directly).
    /// No-op when empty.
    void BeginBufferBarriers(const BufferBarriers& buffer_barriers) const;

    /// True when the pipeline was created with descriptor heap mappings and no legacy layout.
    bool UsesNativeBindings() const noexcept {
        return uses_native_bindings;
    }

    /// Number of unified bindings, i.e. the length of the per-draw heap index array.
    u32 NativeBindingCount() const noexcept {
        return native_binding_count;
    }

    /// Descriptor heap emission: barriers, heap binds when invalidated, then one push-data
    /// block carrying shader push constants plus direct indices or an indirect index address.
    void BindResourcesNative(NativeBindings& native, std::span<const u32> indices,
                             const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const;

    /// SHADPS4_TEX_BIND_MEMO (vk_rasterizer.cpp): texture-binding memo records per logical stage
    /// (one, or SHADPS4_TEX_MEMO_WAYS), owned by the pipeline so a record can never outlive, or be reused for, a different
    /// Shader::Info. Type-erased (the record type is private to vk_rasterizer.cpp); touched only
    /// by the thread that binds resources.
    mutable std::array<std::shared_ptr<void>, Shader::MaxStageTypes> tex_bind_memo{};

protected:
    [[nodiscard]] std::string GetDebugString() const;

    void RecordBufferBarriers(const BufferBarriers& buffer_barriers) const;

    /// Returns once `deferred` has written `pipeline` (vk_pipeline_compile_pool.h).
    void WaitDeferred() const noexcept;
    void PromoteDeferredSlow() const noexcept;

    /// Builds one heap mapping per legacy binding, grouped by the owning shader stage.
    void BuildNativeMappings(std::span<const vk::DescriptorSetLayoutBinding> bindings);

    /// SHADPS4_DRAW_PACKET: the binding table of a legacy (push descriptor) layout.
    void BuildPacketBindings(std::span<const vk::DescriptorSetLayoutBinding> bindings);

    /// Mapping info to chain into the given stage's create info, or null if it binds nothing.
    /// Pointers stay valid for the lifetime of the pipeline object.
    const NativeApi::ShaderDescriptorSetAndBindingMappingInfo* NativeMappingInfo(
        vk::ShaderStageFlagBits stage, const void* next) const;

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    /// SHADPS4_ASYNC_COMPUTE_PIPELINES: set by the constructor when `pipeline` is compiled on the
    /// pool; written before the object is published to another thread, never reset.
    std::unique_ptr<DeferredPipeline> deferred;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
public:
    /// q3 (SHADPS4_NATIVE_SPARSE_LAYOUT): set-0 bindings dropped from this variant's layout
    /// (bindless images / samplers); BindResources drops writes to them.
    u64 sparse_dead_mask{};
protected:
    bool uses_native_bindings{};
    bool uses_direct_push_index{};
    u32 native_binding_count{};
    struct NativeStageMappings {
        vk::ShaderStageFlags stage{};
        std::vector<NativeApi::DescriptorSetAndBindingMapping> mappings;
        mutable NativeApi::ShaderDescriptorSetAndBindingMappingInfo info{};
    };
    std::vector<NativeStageMappings> native_stage_mappings;
    std::vector<PacketBinding> packet_bindings;
    u32 packet_buffer_infos{};
    u32 packet_image_infos{};
};

} // namespace Vulkan
