#include <cstdio>
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/scope_exit.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_async_compiler.h"
#include "video_core/renderer_vulkan/vk_blend_rewrite.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_lighting_test.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "common/elf_info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "tlg_perf_stats.h"

namespace Vulkan {

using Shader::Backend::SPIRV::AuxShaderType;

// tlg: where pipeline construction time goes (reported when the background warm-up ends).
std::atomic<u64> g_tlg_pipeline_total_ns, g_tlg_pipeline_driver_ns;
std::atomic<u32> g_tlg_pipelines_built;
namespace {
using Clock = std::chrono::steady_clock;
u64 NsSince(Clock::time_point start) {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
} // namespace

static constexpr std::array LogicalStageToStageBit = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

// Required by spec unless VK_EXT_extended_dynamic_state3 is supported.
// In practice, we use dynamic state for all of it.
// tlg: not a constructor local: TLG_GFX_DEFERRED_BIND builds the pipeline after it returned.
constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_info = {};

// tlg: TLG_GFX_DEFERRED_BIND: every structure the create-info of one graphics pipeline points at.
// The constructor fills it; when the pipeline is built on a compiler thread, that thread reads it
// after the constructor returned, where the caller's SerializationSupport is gone.
struct PipelineCreateInfoBlock {
    VertexInputs<vk::VertexInputAttributeDescription> vertex_attributes;
    VertexInputs<vk::VertexInputBindingDescription> vertex_bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    vk::PipelineVertexInputDivisorStateCreateInfo divisor_state{};
    vk::PipelineVertexInputStateCreateInfo vertex_input_info{};
    vk::PipelineInputAssemblyStateCreateInfo input_assembly{};
    vk::PipelineTessellationStateCreateInfo tessellation_state{};
    vk::StructureChain<vk::PipelineRasterizationStateCreateInfo,
                       vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT,
                       vk::PipelineRasterizationDepthClipStateCreateInfoEXT>
        raster_chain;
    vk::PipelineMultisampleStateCreateInfo multisampling{};
    vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control{};
    vk::PipelineViewportStateCreateInfo viewport_info{};
    boost::container::static_vector<vk::DynamicState, 32> dynamic_states;
    vk::PipelineDynamicStateCreateInfo dynamic_info{};
    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
        shader_stages;
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci{};
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats{};
    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples{};
    vk::AttachmentSampleCountInfoAMD mixed_samples{};
    vk::PipelineRenderingCreateInfo pipeline_rendering_ci{};
    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> attachments{};
    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> square_attachments{};
    vk::PipelineColorBlendStateCreateInfo color_blending{};
    vk::PipelineColorBlendStateCreateInfo square_blending{};
    vk::GraphicsPipelineCreateInfo pipeline_info{};
    vk::GraphicsPipelineCreateInfo square_info{};
};

namespace {
// tlg: one graphics pipeline creation, with the driver time in the profile count
// CreateCachedGraphicsPipeline keeps. `flags` carries eFailOnPipelineCompileRequired for the
// probe: the driver then returns no pipeline instead of compiling it here.
vk::ResultValue<vk::UniquePipeline> CreateGraphicsPipelineNow(vk::Device device,
                                                             vk::PipelineCache cache,
                                                             vk::GraphicsPipelineCreateInfo info,
                                                             vk::PipelineCreateFlags flags,
                                                             std::string_view debug_str) {
    info.flags = flags;
    const auto started = Clock::now();
    auto created = device.createGraphicsPipelineUnique(cache, info);
    for (u32 attempt = 1; RetryPipelineCreation(created.result, attempt, "graphics", debug_str);
         ++attempt) {
        created = device.createGraphicsPipelineUnique(cache, info);
    }
    g_tlg_pipeline_driver_ns += NsSince(started);
    if (created.result == vk::Result::eSuccess) {
        g_tlg_driver_pipelines_created.fetch_add(1, std::memory_order_relaxed);
    }
    return created;
}

void NameGraphicsPipeline(vk::Device device, vk::Pipeline pipeline, bool square,
                          std::string_view debug_str) {
    if (square) {
        SetObjectName(device, pipeline, "Graphics Pipeline {} squaring pass", debug_str);
    } else {
        SetObjectName(device, pipeline, "Graphics Pipeline {}", debug_str);
    }
}
} // namespace

bool GraphicsDeferredBindEnabled(const Instance& instance, const Scheduler& scheduler) {
    // tlg: off by default, until it is measured (the driver's cache probe alone must find the
    // pipeline for this to pay off). The once-per-process message is the toggle's startup line.
    static const bool requested = [] {
        const char* v = std::getenv("TLG_GFX_DEFERRED_BIND");
        if (v && *v == '1') {
            std::printf("GPU: a pipeline the driver's cache misses builds on a compiler thread "
                        "(TLG_GFX_DEFERRED_BIND); the draw waits for it at the bind\n");
            return true;
        }
        return false;
    }();
    return requested && scheduler.RecordsOnThread() &&
           instance.IsPipelineCreationCacheControlSupported();
}

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    std::optional<const Shader::Gcn::FetchShaderData> fetch_shader_,
    std::span<const vk::ShaderModule> modules, SerializationSupport& sdata, bool preloading,
    const GraphicsBuilder* builder)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache}, key{key_},
      fetch_shader{std::move(fetch_shader_)} {
    TlgPerfStats::CountPipeline(); // tlg: the overlay's pipeline rate
    const auto started = Clock::now();
    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    BuildDescSetLayout(preloading);
    const auto debug_str = GetDebugString();
    // tlg: TLG_GFX_DEFERRED_BIND=1: a pipeline the driver's cache does not hold is built on a
    // compiler thread and the recording thread waits for it at the bind (Handle()), so a burst
    // of new pipelines costs about one build instead of their sum (as vk_compute_pipeline.cpp
    // does for compute). The create-info then has to outlive this constructor.
    // Wave stats read the pipeline of this object (Pipeline::RecordWaveSizes): left inline.
    const bool defer = GraphicsDeferredBindEnabled(instance, scheduler) && builder && !preloading &&
                       !instance.IsPipelineExecutableInfoEnabled();
    PipelineCreateInfoBlock stack_block;
    std::shared_ptr<PipelineCreateInfoBlock> heap_block;
    PipelineCreateInfoBlock* c = &stack_block;
    if (defer) {
        heap_block = std::make_shared<PipelineCreateInfoBlock>();
        c = heap_block.get();
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = AllGraphicsStageBits,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Graphics PipelineLayout {}", debug_str);

    if (!preloading) {
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        if (!instance.IsVertexInputDynamicState()) {
            const auto& vs_info = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
            GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                            guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1);
        }
    }
    // tlg: the block owns the vertex input data: a compiler thread reads it after this returned.
    c->divisors = sdata.divisors;
    c->vertex_bindings = sdata.vertex_bindings;
    c->vertex_attributes = sdata.vertex_attributes;

    auto& divisor_state = c->divisor_state;
    divisor_state = {
        .vertexBindingDivisorCount = static_cast<u32>(c->divisors.size()),
        .pVertexBindingDivisors = c->divisors.data(),
    };

    auto& vertex_input_info = c->vertex_input_info;
    vertex_input_info = {
        .pNext = c->divisors.empty() ? nullptr : &divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(c->vertex_bindings.size()),
        .pVertexBindingDescriptions = c->vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(c->vertex_attributes.size()),
        .pVertexAttributeDescriptions = c->vertex_attributes.data(),
    };

    const auto topology = LiverpoolToVK::PrimitiveType(key.prim_type);
    // tlg: TLG_PIPELINE_LOG=1 lists every graphics pipeline built: primitive type, targets,
    // vertex formats and shader hashes (finding which pass draws a broken effect).
    static const bool pipeline_log = std::getenv("TLG_PIPELINE_LOG") != nullptr;
    if (pipeline_log) {
        std::string formats;
        for (const auto& attr : sdata.vertex_attributes) {
            formats += fmt::format(" {}:{}", attr.location, vk::to_string(attr.format));
        }
        std::string hashes;
        for (const auto hash : key.stage_hashes) {
            hashes += fmt::format(" {:#x}", hash);
        }
        std::fprintf(stderr, "tlg-pipe: prim=%u poly=%u colors=%u z=%u hashes=[%s ] attrs=[%s ]\n",
                     unsigned(key.prim_type), unsigned(key.polygon_mode), key.num_color_attachments,
                     unsigned(key.z_format), hashes.c_str(), formats.c_str());
    }
    auto& input_assembly = c->input_assembly;
    input_assembly = {
        .topology = topology,
    };

    const bool is_rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == AmdGpu::PrimitiveType::QuadList;
    auto& tessellation_state = c->tessellation_state;
    tessellation_state = vk::PipelineTessellationStateCreateInfo{
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    auto& raster_chain = c->raster_chain;
    raster_chain = {
        vk::PipelineRasterizationStateCreateInfo{
            .depthClampEnable = key.depth_clamp_enable &&
                                (!key.depth_clip_enable || instance.IsDepthClipEnableSupported()),
            .rasterizerDiscardEnable = false,
            .polygonMode = LiverpoolToVK::PolygonMode(key.polygon_mode),
            .lineWidth = 1.0f,
        },
        vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT{
            .provokingVertexMode = key.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First
                                       ? vk::ProvokingVertexModeEXT::eFirstVertex
                                       : vk::ProvokingVertexModeEXT::eLastVertex,
        },
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT{
            .depthClipEnable = key.depth_clip_enable,
        },
    };

    if (!instance.IsProvokingVertexSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>();
    }
    if (!instance.IsDepthClipEnableSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    if (!preloading) {
        const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
        sdata.multisampling = {
            .rasterizationSamples = LiverpoolToVK::NumSamples(
                key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
            // tlg: TLG_LIGHTING_TEST=shading keeps sample shading off (vk_lighting_test.h).
            .sampleShadingEnable = !LightingTest::Current().shading &&
                                   (fs_info.addr_flags.persp_sample_ena ||
                                    fs_info.addr_flags.linear_sample_ena),
        };
    }
    c->multisampling = sdata.multisampling;
    auto& multisampling = c->multisampling;

    auto& clip_control = c->clip_control;
    clip_control = vk::PipelineViewportDepthClipControlCreateInfoEXT{
        .negativeOneToOne = key.clip_space == AmdGpu::ClipSpace::MinusWToW,
    };

    auto& viewport_info = c->viewport_info;
    viewport_info = vk::PipelineViewportStateCreateInfo{
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };

    auto& dynamic_states = c->dynamic_states;
    dynamic_states = {
        vk::DynamicState::eViewportWithCount,  vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eBlendConstants,     vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,  vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilOp,          vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,          vk::DynamicState::eRasterizerDiscardEnable,
        vk::DynamicState::eLineWidth,          vk::DynamicState::ePrimitiveRestartEnable,
    };

    if (instance.IsDepthBoundsSupported()) {
        dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
        dynamic_states.push_back(vk::DynamicState::eDepthBounds);
    }
    if (instance.IsDynamicColorWriteMaskSupported()) {
        dynamic_states.push_back(vk::DynamicState::eColorWriteMaskEXT);
    }
    if (instance.IsVertexInputDynamicState()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputEXT);
    } else if (!sdata.vertex_bindings.empty()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputBindingStride);
    }

    auto& dynamic_info = c->dynamic_info;
    dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    auto& shader_stages = c->shader_stages;
    auto stage = u32(Shader::SwStage::Vertex);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Geometry);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationControl);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto type = is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS;
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(type, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = CompileSPV(sdata.tcs, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationEval);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                AuxShaderType::PassthroughTES, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = CompileSPV(sdata.tes, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Fragment);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (preloading ? !sdata.fragment.empty()
                          : runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs.clip_distance_emulation) {
        // tlg: a preloaded pipeline has no runtime infos; its .key carries the discard shader.
        if (!preloading) {
            const auto& vs = runtime_infos[static_cast<u32>(Shader::SwStage::Vertex)].hw.vs;

            sdata.fragment = Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs.outputs);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = CompileSPV(sdata.fragment, instance.GetDevice()),
            .pName = "main",
        });
    }

    auto& subgroup_size_ci = c->subgroup_size_ci;
    subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .requiredSubgroupSize = 64,
    };
    for (auto& stage : shader_stages) {
        // tlg: a required subgroup size is only valid on the stages the driver lists in
        // requiredSubgroupSizeStages; attaching it elsewhere is a spec violation (the driver may
        // reject the pipeline or clamp the size, either way the shader runs at a size the wave64
        // lowering did not assume).
        stage.pNext = instance.IsRequiredSubgroupSize64ValidFor(stage.stage) ? &subgroup_size_ci
                                                                            : nullptr;
    }

    const auto depth_format =
        instance.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    auto& color_formats = c->color_formats;
    for (s32 i = 0; i < key.num_color_attachments; ++i) {
        const auto& col_buf = key.color_buffers[i];
        const auto format = LiverpoolToVK::SurfaceFormat(col_buf.data_format, col_buf.num_format);
        const auto color_format =
            instance.GetSupportedFormat(format, vk::FormatFeatureFlagBits2::eColorAttachment);
        if (!instance.IsFormatSupported(color_format,
                                        vk::FormatFeatureFlagBits2::eColorAttachment)) {
            LOG_WARNING(Render_Vulkan,
                        "color buffer format {} does not support COLOR_ATTACHMENT_BIT",
                        vk::to_string(color_format));
        }
        color_formats[i] = color_format;
    }

    auto& color_samples = c->color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 num_samples) {
        return num_samples ? LiverpoolToVK::NumSamples(num_samples, instance.GetColorSampleCounts())
                           : vk::SampleCountFlagBits::e1;
    });
    auto& mixed_samples = c->mixed_samples;
    mixed_samples = {
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };

    auto& pipeline_rendering_ci = c->pipeline_rendering_ci;
    pipeline_rendering_ci = {
        .pNext = instance.IsMixedDepthSamplesSupported() ? &mixed_samples : nullptr,
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid
                                     ? depth_format
                                     : vk::Format::eUndefined,
        .stencilAttachmentFormat = key.stencil_format != AmdGpu::DepthBuffer::StencilFormat::Invalid
                                       ? depth_format
                                       : vk::Format::eUndefined,
    };

    auto& attachments = c->attachments;
    // tlg: GNM scales MIN/MAX operands, Vulkan does not (vk_blend_rewrite.h). A self-scaled
    // MIN/MAX is drawn twice: native, then this pipeline squares the destination.
    auto& square_attachments = c->square_attachments;
    bool needs_square_pass = false;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        const auto& control = key.blend_controls[i];
        const auto target_format = key.color_buffers[i].num_format;
        BlendEquation color_equation{control.color_src_factor, control.color_func,
                                     control.color_dst_factor};
        BlendEquation alpha_equation{
            control.separate_alpha_blend ? control.alpha_src_factor : control.color_src_factor,
            control.separate_alpha_blend ? control.alpha_func : control.color_func,
            control.separate_alpha_blend ? control.alpha_dst_factor : control.color_dst_factor};
        const auto color_rewrite =
            RewriteScaledMinMaxBlend(color_equation, target_format, BlendChannel::Color);
        const auto alpha_rewrite =
            RewriteScaledMinMaxBlend(alpha_equation, target_format, BlendChannel::Alpha);
        const bool writes_color = bool(key.write_masks[i] & (vk::ColorComponentFlagBits::eR |
                                                             vk::ColorComponentFlagBits::eG |
                                                             vk::ColorComponentFlagBits::eB));
        const bool writes_alpha = bool(key.write_masks[i] & vk::ColorComponentFlagBits::eA);
        if (control.enable &&
            ((writes_color && color_rewrite == BlendRewriteResult::Unsupported) ||
             (writes_alpha && alpha_rewrite == BlendRewriteResult::Unsupported))) {
            LOG_WARNING(Render_Vulkan,
                        "Scaled MIN/MAX blend on attachment {} has no Vulkan equivalent "
                        "(format {}, color {}/{}/{}, alpha {}/{}/{})",
                        i, u32(target_format), u32(color_equation.src_factor),
                        u32(color_equation.function), u32(color_equation.dst_factor),
                        u32(alpha_equation.src_factor), u32(alpha_equation.function),
                        u32(alpha_equation.dst_factor));
        }
        const bool square_color =
            control.enable && writes_color && color_rewrite == BlendRewriteResult::Squared;
        const bool square_alpha =
            control.enable && writes_alpha && alpha_rewrite == BlendRewriteResult::Squared;
        needs_square_pass |= square_color || square_alpha;

        const auto src_color = LiverpoolToVK::BlendFactor(color_equation.src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(color_equation.dst_factor);

        attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = control.enable,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = LiverpoolToVK::BlendOp(color_equation.function),
            .srcAlphaBlendFactor = LiverpoolToVK::BlendFactor(alpha_equation.src_factor),
            .dstAlphaBlendFactor = LiverpoolToVK::BlendFactor(alpha_equation.dst_factor),
            .alphaBlendOp = LiverpoolToVK::BlendOp(alpha_equation.function),
            .colorWriteMask =
                instance.IsDynamicColorWriteMaskSupported()
                    ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                    : key.write_masks[i],
        };
        square_attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = true,
            .srcColorBlendFactor = vk::BlendFactor::eZero,
            .dstColorBlendFactor = square_color ? vk::BlendFactor::eDstColor : vk::BlendFactor::eOne,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eZero,
            .dstAlphaBlendFactor = square_alpha ? vk::BlendFactor::eDstAlpha : vk::BlendFactor::eOne,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = attachments[i].colorWriteMask,
        };

        // On GCN GPU there is an additional mask which allows to control color components exported
        // from a pixel shader. A situation possible, when the game may mask out the alpha channel,
        // while it is still need to be used in blending ops. For such cases, HW will default alpha
        // to 1 and perform the blending, while shader normally outputs 0 in the last component.
        // Unfortunatelly, Vulkan doesn't provide any control on blend inputs, so below we detecting
        // such cases and override alpha value in order to emulate HW behaviour.
        const auto has_alpha_masked_out =
            (key.cb_shader_mask.GetMask(i) & AmdGpu::ColorBufferMask::ComponentA) == 0;
        const auto has_src_alpha_in_src_blend = src_color == vk::BlendFactor::eSrcAlpha ||
                                                src_color == vk::BlendFactor::eOneMinusSrcAlpha;
        const auto has_src_alpha_in_dst_blend = dst_color == vk::BlendFactor::eSrcAlpha ||
                                                dst_color == vk::BlendFactor::eOneMinusSrcAlpha;
        if (has_alpha_masked_out && has_src_alpha_in_src_blend) {
            attachments[i].srcColorBlendFactor = src_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }
        if (has_alpha_masked_out && has_src_alpha_in_dst_blend) {
            attachments[i].dstColorBlendFactor = dst_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }
    }

    auto& color_blending = c->color_blending;
    color_blending = {
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != AmdGpu::ColorControl::LogicOp::Copy,
        .logicOp = LiverpoolToVK::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    auto& pipeline_info = c->pipeline_info;
    pipeline_info = {
        .pNext = &pipeline_rendering_ci,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &vertex_input_info : nullptr,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = &tessellation_state,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &depth_stencil_info : nullptr,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };

    // tlg: in the game's keyed cache folder, so a new recompiler or "clear cache" drops them too.
    const auto binary_directory = Storage::GameCacheDir() / "pipeline-binaries";
    if (!defer) {
        auto [pipeline_result, pipe] =
            CreateCachedGraphicsPipeline(device, pipeline_cache, pipeline_info, key,
                                         instance.GetPipelineBinaryGlobalKey(), binary_directory);
        ASSERT_MSG(pipeline_result == vk::Result::eSuccess,
                   "Failed to create graphics pipeline: {}", vk::to_string(pipeline_result));
        pipeline = std::move(pipe);
        SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
        RecordWaveSizes();
    } else {
        CreatePipeline(heap_block, pipeline_cache, pipeline_info, pipeline, builder, build,
                       false);
    }

    if (needs_square_pass) {
        auto& square_blending = c->square_blending;
        square_blending = color_blending;
        square_blending.logicOpEnable = false;
        square_blending.pAttachments = square_attachments.data();
        auto& square_info = c->square_info;
        square_info = pipeline_info;
        square_info.pColorBlendState = &square_blending;
        if (!defer) {
            auto [square_result, square] = CreateCachedGraphicsPipeline(
                device, pipeline_cache, square_info, key, instance.GetPipelineBinaryGlobalKey(),
                binary_directory);
            ASSERT_MSG(square_result == vk::Result::eSuccess,
                       "Failed to create the squaring pipeline: {}", vk::to_string(square_result));
            square_pipeline = std::move(square);
            SetObjectName(device, *square_pipeline, "Graphics Pipeline {} squaring pass",
                          debug_str);
        } else {
            CreatePipeline(heap_block, pipeline_cache, square_info, square_pipeline, builder,
                           square_build, true);
        }
    }
    g_tlg_pipeline_total_ns += NsSince(started);
    ++g_tlg_pipelines_built;
}

GraphicsPipeline::~GraphicsPipeline() {
    // A build still queued uses the layout, the descriptor set layout and the create-info block.
    WaitForBuild(build);
    WaitForBuild(square_build);
}

vk::Pipeline GraphicsPipeline::SquarePassHandle() const {
    if (square_build) {
        WaitForBuild(square_build);
        return vk::Pipeline{square_build->handle.load(std::memory_order_acquire)};
    }
    return *square_pipeline;
}

void GraphicsPipeline::WaitForBuild(const std::shared_ptr<Build>& state) {
    if (!state) {
        return;
    }
    std::unique_lock lock{state->mutex};
    state->cv.wait(lock, [&state] { return state->done.load(std::memory_order_acquire); });
}

void GraphicsPipeline::CreatePipeline(const std::shared_ptr<PipelineCreateInfoBlock>& block,
                                      vk::PipelineCache cache,
                                      const vk::GraphicsPipelineCreateInfo& info,
                                      vk::UniquePipeline& out, const GraphicsBuilder* builder,
                                      std::shared_ptr<Build>& state, bool square) {
    const vk::Device device = instance.GetDevice();
    const auto debug_str = GetDebugString();
    // One the driver built in an earlier session (its own cache) comes at once: the probe asks
    // for the pipeline alone, so the caller learns it would have to be compiled here.
    // CreateCachedGraphicsPipeline is not used for these: the pipeline-binary capture it adds is
    // incompatible with the probe's flag (and with a build on another thread).
    auto created =
        CreateGraphicsPipelineNow(device, cache, info,
                                  vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired,
                                  debug_str);
    if (created.result != vk::Result::ePipelineCompileRequired) {
        ASSERT_MSG(created.result == vk::Result::eSuccess,
                   "Failed to create graphics pipeline: {}", vk::to_string(created.result));
        out = std::move(created.value);
        NameGraphicsPipeline(device, *out, square, debug_str);
        return;
    }
    // tlg: built on a compiler thread, in that thread's cache (the driver blob the saver writes
    // keeps it). The recording thread waits for it at the bind, never the GPU thread, so a burst
    // of new pipelines goes on instead of paying for each of them where it is drawn. The pending
    // builds also hold back the piled-up draw submissions (vk_rasterizer.cpp): those wait for the
    // recording thread, which waits for this.
    auto build_state = std::make_shared<Build>();
    const auto caches = builder->caches;
    Pipeline::deferred_binds.fetch_add(1, std::memory_order_relaxed);
    const bool queued = builder->compiler->Submit(
        [this, block, state = build_state, caches, square, debug_str] {
            const auto job_started = Clock::now();
            SCOPE_EXIT {
                g_tlg_pipeline_total_ns += NsSince(job_started);
                Pipeline::deferred_binds.fetch_sub(1, std::memory_order_relaxed);
            };
            const vk::Device device = instance.GetDevice();
            const auto& create_info = square ? block->square_info : block->pipeline_info;
            auto built = CreateGraphicsPipelineNow(device, *caches[AsyncCompiler::WorkerIndex()],
                                                  create_info, {}, debug_str);
            ASSERT_MSG(built.result == vk::Result::eSuccess,
                       "Failed to create graphics pipeline: {}", vk::to_string(built.result));
            NameGraphicsPipeline(device, *built.value, square, debug_str);
            {
                std::scoped_lock lock{state->mutex};
                state->handle.store(static_cast<VkPipeline>(*built.value),
                                    std::memory_order_release);
                state->result = std::move(built.value);
                state->done.store(true, std::memory_order_release);
            }
            state->cv.notify_all();
        },
        true);
    if (queued) {
        state = std::move(build_state);
        return;
    }
    // The compiler queue is full: built here, as before.
    Pipeline::deferred_binds.fetch_sub(1, std::memory_order_relaxed);
    created = CreateGraphicsPipelineNow(device, cache, info, {}, debug_str);
    ASSERT_MSG(created.result == vk::Result::eSuccess, "Failed to create graphics pipeline: {}",
               vk::to_string(created.result));
    out = std::move(created.value);
    NameGraphicsPipeline(device, *out, square, debug_str);
}

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (!fetch_shader || fetch_shader->attributes.empty()) {
        return;
    }
    const auto& vs_info = GetStage(Shader::SwStage::Vertex);
    for (const auto& attrib : fetch_shader->attributes) {
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs_info);
        attributes.push_back(Attribute{
            .location = attrib.semantic,
            .binding = attrib.semantic,
            .format = LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt()),
            .offset = 0,
        });
        bindings.push_back(Binding{
            .binding = attrib.semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
        });
        const u32 divisor = step_rate == InstanceIdType::OverStepRate0
                                ? step_rate_0
                                : (step_rate == InstanceIdType::OverStepRate1 ? step_rate_1 : 1);
        if constexpr (std::is_same_v<Binding, vk::VertexInputBindingDescription2EXT>) {
            bindings.back().divisor = divisor;
        } else if (step_rate != InstanceIdType::None) {
            divisors.push_back(vk::VertexInputBindingDivisorDescriptionEXT{
                .binding = attrib.semantic,
                .divisor = divisor,
            });
        }
        guest_buffers.emplace_back(buffer);
    }
}

// Declare templated GetVertexInputs for necessary types.
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>& attributes,
    VertexInputs<vk::VertexInputBindingDescription>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;

void GraphicsPipeline::BuildDescSetLayout(bool preloading) {
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    u32 binding{};

    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto stage_bit = LogicalStageToStageBit[u32(stage->sw_stage)];
        for (const auto& buffer : stage->buffers) {
            const auto sharp =
                preloading ? AmdGpu::Buffer{}
                           : buffer.GetSharp(*stage); // See for the comment in compute PL creation
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        for (const auto& image : stage->images) {
            const u32 num_bindings = image.NumBindings(*stage);
            bindings.push_back({
                .binding = binding,
                .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                                   : vk::DescriptorType::eSampledImage,
                .descriptorCount = num_bindings,
                .stageFlags = stage_bit,
            });
            binding += num_bindings;
        }
        for (const auto& sampler : stage->samplers) {
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
    }
    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [layout_result, layout] =
        instance.GetDevice().createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics descriptor set layout: {}", vk::to_string(layout_result));
    desc_layout = std::move(layout);
}

} // namespace Vulkan
