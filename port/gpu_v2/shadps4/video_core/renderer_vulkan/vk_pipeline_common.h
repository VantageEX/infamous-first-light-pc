// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/pipeline_handle_choice.h"

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

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return vk::Pipeline{replacement.Select(static_cast<VkPipeline>(*pipeline))};
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::SwStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::SwStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    void BindResources(DescriptorWrites& set_writes, const Shader::PushData& push_data) const;
    /// Binds on the given scheduler's command buffer (another queue, asc_queue); only for a
    /// pipeline using push descriptors (descriptor sets are recycled with this pipeline's
    /// scheduler).
    void BindResources(Scheduler& on, DescriptorWrites& set_writes,
                       const Shader::PushData& push_data) const;

    /// Whether the descriptors are pushed (no descriptor set allocated per bind).
    [[nodiscard]] bool UsesPushDescriptors() const noexcept {
        return uses_push_descriptors;
    }

    /// Returns true once the Vulkan pipeline object exists. A pipeline whose creation was
    /// deferred is compiled by a worker thread and must not be bound before that.
    [[nodiscard]] bool IsReady() const noexcept {
        return ready.load(std::memory_order_acquire);
    }

    /// Creates a deferred pipeline, or optimizes an already-ready tiered compute pipeline.
    /// Only one worker may compile a given pipeline; a published initial handle stays alive.
    virtual void Compile() = 0;

protected:
    void PublishReplacement(vk::Pipeline handle) noexcept {
        replacement.Publish(static_cast<VkPipeline>(handle));
    }
    // The initial pipeline remains alive after publishing a replacement: queued GPU
    // commands may still refer to its handle.
    PipelineHandleChoice<VkPipeline> replacement;
    [[nodiscard]] std::string GetDebugString() const;

    void MarkReady() noexcept {
        ready.store(true, std::memory_order_release);
    }

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
    std::atomic<bool> ready{false};
};

} // namespace Vulkan
