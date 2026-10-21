// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "common/perf_stats.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module_,
                                 SerializationSupport& sdata, bool preloading /*=false*/,
                                 bool defer_compile /*=false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_}, module{module_}, vk_pipeline_cache{pipeline_cache} {
    auto& info = stages[int(Shader::SwStage::Compute)];
    tiered = !preloading && !defer_compile &&
             Common::Perf::GetTuning().tiered_compute_pipelines.load(std::memory_order_relaxed);
    info = &info_;
    const auto debug_str = GetDebugString();

    u32 binding{};
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        // During deserialization, we don't have access to the UD to fetch sharp data. To address
        // this properly we need to track shaprs or portion of them in `sdata`, but since we're
        // interested only in "is storage" flag (which is not even effective atm), we can take a
        // shortcut there.
        const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    for (const auto& image : info->images) {
        const u32 num_bindings = image.NumBindings(*info);
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    const auto device = instance.GetDevice();
    auto [descriptor_set_result, descriptor_set] =
        device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(descriptor_set_result == vk::Result::eSuccess,
               "Failed to create compute descriptor set layout: {}",
               vk::to_string(descriptor_set_result));
    desc_layout = std::move(descriptor_set);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create compute pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);

    // Everything that reads the guest state is done above, on the command processor thread.
    // Creating the pipeline object only needs the module and the layout, so it can be deferred
    // to a worker thread (it takes the driver up to a few hundred milliseconds).
    if (tiered) {
        CompileVariant(true);
    } else if (!defer_compile) {
        Compile();
    }
}

ComputePipeline::~ComputePipeline() = default;

void ComputePipeline::Compile() {
    CompileVariant(false);
}

void ComputePipeline::CompileVariant(bool fast) {
    const u64 start_ns = Common::Perf::NowNs();
    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .flags = fast ? vk::PipelineCreateFlagBits::eDisableOptimization : vk::PipelineCreateFlags{},
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    const auto device = instance.GetDevice();
    auto [pipeline_result, pipe] =
        device.createComputePipelineUnique(vk_pipeline_cache, compute_pipeline_ci);
    if (pipeline_result != vk::Result::eSuccess && tiered) {
        if (fast) {
            LOG_WARNING(Render_Vulkan, "Fast compute creation failed ({}); using normal compilation",
                        vk::to_string(pipeline_result));
            tiered = false;
            CompileVariant(false);
            return;
        }
        if (IsReady()) {
            LOG_WARNING(Render_Vulkan, "Compute optimization failed ({}); retaining initial pipeline",
                        vk::to_string(pipeline_result));
            return;
        }
    }
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(pipeline_result));
    SetObjectName(device, *pipe, "Compute Pipeline {} {}", GetDebugString(), fast ? "fast" : "optimized");
    if (tiered && !fast && IsReady()) {
        optimized_pipeline = std::move(pipe);
        PublishReplacement(*optimized_pipeline);
    } else {
        pipeline = std::move(pipe);
        MarkReady();
    }
    if (tiered) {
        LOG_INFO(Render_Vulkan, "Compute tier {} hash={:#x} compile_ms={:.3f}",
                 fast ? "fast" : "optimized", compute_key.value,
                 double(Common::Perf::NowNs() - start_ns) / 1e6);
    }
    Common::Perf::Add(Common::Perf::PipelineCreates);
    Common::Perf::Add(Common::Perf::PipelineCreateNs, Common::Perf::NowNs() - start_ns);
}

} // namespace Vulkan
