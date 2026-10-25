#include "tlg_perf_trace.h"
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>

#include <boost/container/small_vector.hpp>

#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_async_compiler.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "tlg_perf_stats.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading /*=false*/,
                                 const ComputeBuilder* builder /*=nullptr*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    TlgPerfStats::CountPipeline(); // tlg: the overlay's pipeline rate
    auto& info = stages[int(Shader::SwStage::Compute)];
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

    // Runs on the compiler threads too: it only uses what stays fixed after construction and
    // the layout, which the destructor keeps alive until the build is done. Returns no pipeline
    // when one may only come from the driver's cache and it isn't there.
    const auto create = [&instance, module, layout = *pipeline_layout,
                         debug_str, trace_key = TlgPerf::Enabled() ? std::hash<ComputePipelineKey>{}(compute_key) : 0](vk::PipelineCache cache,
                                    vk::PipelineCreateFlags flags) -> vk::UniquePipeline {
        const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
            .requiredSubgroupSize = 64,
        };
        const bool trace = TlgPerf::Enabled(); // attach and read feedback under one decision
        TlgPerf::Span trace_build{TlgPerf::Kind::CompileBegin, TlgPerf::Kind::CompileEnd, trace_key, 'c'};
        vk::PipelineCreationFeedback feedback{};
        const vk::PipelineCreationFeedbackCreateInfo feedback_info{.pPipelineCreationFeedback = &feedback};
        // Pipeline creation feedback is core in Vulkan 1.3, required by this renderer.
        const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
            .pNext = trace ? &feedback_info : nullptr,
            .flags = flags,
            .stage{
                // tlg: the stage must be in requiredSubgroupSizeStages for this to be valid; the
                // profile's subgroup size (vk_pipeline_cache.cpp) is picked to match.
                .pNext = instance.IsRequiredSubgroupSize64ValidFor(
                             vk::ShaderStageFlagBits::eCompute)
                             ? &subgroup_size_ci
                             : nullptr,
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            },
            .layout = layout,
        };
        const auto device = instance.GetDevice();
        auto created = device.createComputePipelineUnique(cache, compute_pipeline_ci);
        for (u32 attempt = 1;
             RetryPipelineCreation(created.result, attempt, "compute", debug_str); ++attempt) {
            created = device.createComputePipelineUnique(cache, compute_pipeline_ci);
        }
        if (trace) {
            TlgPerf::Event e; e.kind = TlgPerf::Kind::PipelineFeedback; e.operation = trace_build.Id(); e.object = trace_key;
            e.b = created.result == vk::Result::eSuccess ? TlgPerf::CacheFeedback(static_cast<unsigned>(feedback.flags)) : 0;
            e.a = e.b ? feedback.duration : 0;
            e.generation = 'c'; TlgPerf::Emit(e);
        }
        if (created.result == vk::Result::ePipelineCompileRequired &&
            (flags & vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired)) {
            return {};
        }
        if (created.result == vk::Result::eSuccess) {
            g_tlg_driver_pipelines_created.fetch_add(1, std::memory_order_relaxed);
        }
        ASSERT_MSG(created.result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
                   vk::to_string(created.result));
        SetObjectName(device, *created.value, "Compute Pipeline {}", debug_str);
        return std::move(created.value);
    };

    // tlg: a dispatch is waiting for a pipeline met in game, and compute work cannot be skipped
    // (5e7995ac skipped it: not taken, for fidelity). Built on the GPU thread, a new one took
    // 60-190 ms in inFAMOUS Second Son and an effect brings a few dozen: freezes of 0.6-3.8 s.
    // TLG_COMPUTE_ASYNC_BUILD=0 builds every one here as before.
    static const bool async_build = [] {
        const char* env = std::getenv("TLG_COMPUTE_ASYNC_BUILD");
        return !(env && env[0] == '0');
    }();
    if (builder && !preloading && async_build &&
        instance.IsPipelineCreationCacheControlSupported()) {
        // One the driver built before (its disk cache) comes at once.
        pipeline = create(pipeline_cache,
                          vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
        if (pipeline) {
            RecordWaveSizes();
            return;
        }
        auto state = std::make_shared<Build>();
        const auto caches = builder->caches;
        const bool deferred_bind = scheduler.RecordsOnThread();
        // With the Vulkan recording thread, the dispatch binds the pipeline there (Handle()
        // waits for it), so the GPU thread goes on and the next new pipelines are built
        // alongside: a burst costs about one build instead of their sum. NVIDIA builds one
        // without optimizations about as slowly, so the optimized one is built directly.
        // Otherwise one is built here without optimizations, several times quicker on some
        // drivers, and the optimized one replaces it (SwapInOptimized).
        const bool queued = builder->compiler->Submit(
            [state, create, caches, deferred_bind, debug_str] {
                const auto start = std::chrono::steady_clock::now();
                auto built = create(*caches[AsyncCompiler::WorkerIndex()], {});
                LOG_INFO(Render_Vulkan, "tlg: compute pipeline {} built on a compiler thread in "
                         "{:.1f} ms{}", debug_str,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start).count(),
                         deferred_bind ? "" : " (replaces the unoptimized one)");
                {
                    std::scoped_lock lock{state->mutex};
                    if (deferred_bind) {
                        state->handle.store(static_cast<VkPipeline>(*built), std::memory_order_release);
                    }
                    state->result = std::move(built);
                    state->done.store(true, std::memory_order_release);
                }
                state->cv.notify_all();
            },
            deferred_bind);
        if (queued) {
            build = std::move(state);
            if (!deferred_bind) {
                pipeline = create(pipeline_cache, vk::PipelineCreateFlagBits::eDisableOptimization);
                build->handle.store(static_cast<VkPipeline>(*pipeline), std::memory_order_release);
                optimized_pending = true;
            }
            return;
        }
        // The compiler queue is full: built here, as before.
    }
    pipeline = create(pipeline_cache, {});
    RecordWaveSizes();
}

ComputePipeline::~ComputePipeline() {
    // A build still queued uses the layout and module.
    if (build) {
        std::unique_lock lock{build->mutex};
        build->cv.wait(lock, [this] { return build->done.load(std::memory_order_acquire); });
    }
}

void ComputePipeline::SwapInOptimized() {
    if (!optimized_pending || !build->done.load(std::memory_order_acquire)) {
        return;
    }
    optimized_pending = false;
    vk::UniquePipeline old = std::move(pipeline);
    pipeline = std::move(build->result);
    build->handle.store(static_cast<VkPipeline>(*pipeline), std::memory_order_release);
    // Commands recorded so far, some perhaps still queued for the recording thread or the draw
    // pipe, may bind the old one: it is destroyed once the GPU is past the command buffer being
    // recorded now and then past the one recorded when that is done.
    scheduler.DeferOperation([scheduler = &scheduler, old = std::move(old)]() mutable {
        scheduler->DeferOperation([old = std::move(old)]() mutable { old.reset(); });
    });
}

} // namespace Vulkan
