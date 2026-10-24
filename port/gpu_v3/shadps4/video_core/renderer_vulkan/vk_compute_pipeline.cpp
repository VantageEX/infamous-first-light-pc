// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/scope_exit.h"
#include "shader_recompiler/info.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/host_shaders/infamous_particle_sort_comp.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
/// q36 (design-game-main D2 round 3): the sort replacement's guard counter {hits, first, last, overflow}, host-visible,
/// read by pointer from the shader. One per process; never freed (lives as long as the pipelines that point at it).
VideoCore::Buffer* SortGuardBuffer(const Instance& instance, Scheduler& scheduler) {
    static VideoCore::Buffer* const buffer = [&]() -> VideoCore::Buffer* {
        if (!instance.GetDevice() || !instance.GetPhysicalDevice()) {
            return nullptr;
        }
        auto* b = new VideoCore::Buffer(instance, scheduler, VideoCore::MemoryUsage::Download, 0,
                                        vk::BufferUsageFlagBits::eStorageBuffer |
                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        256);
        if (b->mapped_data.size() < 16) {
            LOG_WARNING(Render_Vulkan, "q36: sort guard buffer not host-mapped; guard counts off");
            return nullptr;
        }
        std::memset(b->mapped_data.data(), 0, 64); // + the SORT_VERIFY words 4..9
        VideoCore::g_sort_guard_words = reinterpret_cast<volatile u32*>(b->mapped_data.data());
        return b;
    }();
    return buffer;
}
} // namespace

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading /*=false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::LogicalStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();
    const auto device = instance.GetDevice();

    // The game's 64-lane merge loop reads other lanes' SSBO writes between steps.
    // Its translated shader has no barriers. Keep this targeted diagnostic opt-in;
    // changing the pipeline module here also covers shader-cache preloading without
    // replacing/persisting the cached guest SPIR-V or its binding metadata.
    static const bool synchronize_smoke_sort = [] {
        const char* value = std::getenv("SHADPS4_SMOKE_SORT_SYNC");
        return value && std::string_view{value} == "1";
    }();
    vk::ShaderModule synchronized_module{};
    SCOPE_EXIT {
        if (synchronized_module) {
            device.destroyShaderModule(synchronized_module);
        }
    };
    if (synchronize_smoke_sort && info->pgm_hash == 0xa16fa93a) {
        ASSERT_MSG(info->buffers.size() == 5 && info->images.empty() && info->samplers.empty(),
                   "Unexpected particle sort binding contract");
        std::vector<std::string> defines;
        if (auto* guard = SortGuardBuffer(instance, scheduler)) {
            const u64 addr = guard->BufferDeviceAddress(); // q36: invalid-width guard counter
            defines.push_back(fmt::format("SORT_GUARD_LO={}u", static_cast<u32>(addr)));
            defines.push_back(fmt::format("SORT_GUARD_HI={}u", static_cast<u32>(addr >> 32)));
            // SHADPS4_SORT_VERIFY=1 (2026-10-18, the particle flicker hunt): each dispatch checks
            // that its input runs are sorted and its own in-group output (counters in the guard).
            static const bool verify = [] {
                const char* v = std::getenv("SHADPS4_SORT_VERIFY");
                return v && v[0] == '1';
            }();
            if (verify) {
                defines.push_back("SORT_VERIFY=1");
            }
        }
        synchronized_module = Compile(HostShaders::INFAMOUS_PARTICLE_SORT_COMP,
                                      vk::ShaderStageFlagBits::eCompute, device, defines);
        module = synchronized_module;
        SetObjectName(device, module, "Synchronized infamous particle sort");
        LOG_INFO(Render_Vulkan, "Smoke sort synchronization enabled for {:#x}, preloading={}",
                 info->pgm_hash, preloading);
    }

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

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
            .descriptorType = buffer.IsStorage(sharp) ? vk::DescriptorType::eStorageBuffer
                                                      : vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    if (!preloading) {
        sdata.image_binding_counts.clear();
    }
    u32 image_idx{};
    for (const auto& image : info->images) {
        // Image descriptor counts need the user data, which is unavailable while preloading, so
        // they are carried in sdata. Getting them wrong shifts every later binding index.
        const bool have_cached = image_idx < sdata.image_binding_counts.size();
        const u32 num_bindings =
            (preloading && have_cached) ? sdata.image_binding_counts[image_idx]
                                        : image.NumBindings(*info);
        if (!preloading) {
            sdata.image_binding_counts.push_back(num_bindings);
        }
        ++image_idx;
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

    if (instance.IsNativeBindingsActive()) {
        // Descriptor heap model: mappings ride on the shader stage, the pipeline carries the
        // heap flag and there is no descriptor set layout or pipeline layout at all.
        BuildNativeMappings(bindings);
        if (const auto* mapping_info =
                NativeMappingInfo(vk::ShaderStageFlagBits::eCompute, shader_ci.pNext)) {
            shader_ci.pNext = mapping_info;
        }
        const NativeApi::PipelineCreateFlags2CreateInfo flags2_ci = {
            .sType = NativeApi::StructureTypePipelineCreateFlags2CreateInfo,
            .flags = NativeApi::PipelineCreate2DescriptorHeapBit,
        };
        const vk::ComputePipelineCreateInfo native_pipeline_ci = {
            .pNext = &flags2_ci,
            .stage = shader_ci,
            .layout = VK_NULL_HANDLE,
        };
        auto [pipeline_result, pipe] =
            instance.GetDevice().createComputePipelineUnique(pipeline_cache, native_pipeline_ci);
        ASSERT_MSG(pipeline_result == vk::Result::eSuccess,
                   "Failed to create descriptor heap compute pipeline: {}",
                   vk::to_string(pipeline_result));
        pipeline = std::move(pipe);
        SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);
        return;
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    BuildPacketBindings(std::span<const vk::DescriptorSetLayoutBinding>{bindings.data(),
                                                                        bindings.size()});
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
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

    // SHADPS4_ASYNC_COMPUTE_PIPELINES: only the driver compile is deferred; layouts and bindings
    // above exist now. The smoke-sort replacement module dies with this constructor (SCOPE_EXIT),
    // so that one pipeline stays synchronous.
    if (AsyncComputePipelinesEnabled() && !synchronized_module) {
        deferred = std::make_unique<DeferredPipeline>();
        deferred->device = device;
        deferred->cache = pipeline_cache;
        deferred->module = module;
        deferred->layout = *pipeline_layout;
        deferred->subgroup_size_64 = instance.IsSubgroupSize64Supported();
        deferred->preload = preloading;
        deferred->debug_name = fmt::format("Compute Pipeline {}", debug_str);
        deferred->out = &pipeline;
        PipelineCompilePool::Push(deferred.get());
        return;
    }
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .flags = DiagPipelineCreateFlags(),
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    auto [pipeline_result, pipe] =
        instance.GetDevice().createComputePipelineUnique(pipeline_cache, compute_pipeline_ci);
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);
}

ComputePipeline::~ComputePipeline() = default;

} // namespace Vulkan
