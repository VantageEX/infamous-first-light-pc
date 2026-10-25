// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_bindless.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "tlg_perf_stats.h"

namespace Vulkan {

using Shader::Backend::SPIRV::AuxShaderType;

static constexpr std::array LogicalStageToStageBit = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader_,
    std::span<const vk::ShaderModule> modules, SerializationSupport& sdata, bool preloading,
    const DescSetBindings* derived /*= nullptr*/,
    u32 native_records_binding_ /*= NoNativeRecords*/, bool native_bindless_ /*= false*/,
    bool refusable /*= false*/)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache}, key{key_},
      fetch_shader{std::move(fetch_shader_)} {
    TlgPerfStats::CountPipeline(); // tlg: the overlay's pipeline rate
    native_records_binding = native_records_binding_;
    native_bindless = native_bindless_ && Bindless::Init(instance.GetDevice());
    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    // With `derived`, the lookup thread already read everything below that needs an Info's
    // user data (DeriveStageInputs); this constructor may be running on the compiler thread
    // while that thread rewrites the Infos for its next draw.
    ASSERT_MSG(!(derived && preloading), "Derived stage inputs are never preloaded");
    if (derived) {
        BuildDescSetLayout(std::span{derived->data(), derived->size()});
    } else {
        DescSetBindings bindings;
        CollectDescSetBindings(stages, sdata, preloading, bindings);
        if (native_records_binding != NoNativeRecords) {
            // SHADPS4_NATIVE_PASS: the per-draw records the native vertex shader reads.
            const u32 next =
                bindings.empty() ? 0u : bindings.back().binding + bindings.back().descriptorCount;
            ASSERT_MSG(next == native_records_binding, "native records binding {} vs {}", next,
                       native_records_binding);
            bindings.push_back({
                .binding = native_records_binding,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eVertex,
            });
            if (Shader::g_native_ring) {
                // SHADPS4_NATIVE_GPU_DIRECT: the per-batch table (ring index, bases).
                native_table_binding = native_records_binding + 1;
                bindings.push_back({
                    .binding = native_table_binding,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eVertex,
                });
                if (Shader::g_native_pt == 2) {
                    // SHADPS4_NATIVE_PT=2: the compare SSBO (the page table itself is read by
                    // pointer at the table's fallback address: no binding).
                    native_pt_debug_binding = native_records_binding + 2;
                    bindings.push_back({
                        .binding = native_pt_debug_binding,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eVertex,
                    });
                }
            }
        }
        BuildDescSetLayout(std::span{bindings.data(), bindings.size()});
    }
    const auto debug_str = GetDebugString();

    const vk::PushConstantRange push_constants = {
        .stageFlags = AllGraphicsStageBits,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    // Descriptor heap pipelines have neither a descriptor set layout nor a pipeline layout.
    if (!uses_native_bindings) {
        // SHADPS4_NATIVE_BINDLESS: set 1 = the global image / sampler arrays.
        const std::array<vk::DescriptorSetLayout, 2> set_layouts{
            *desc_layout, native_bindless ? *Bindless::G().layout : vk::DescriptorSetLayout{}};
        const vk::PipelineLayoutCreateInfo layout_info = {
            .setLayoutCount = native_bindless ? 2U : 1U,
            .pSetLayouts = set_layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constants,
        };
        auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
        ASSERT_MSG(layout_result == vk::Result::eSuccess,
                   "Failed to create graphics pipeline layout: {}", vk::to_string(layout_result));
        pipeline_layout = std::move(layout);
        SetObjectName(device, *pipeline_layout, "Graphics PipelineLayout {}", debug_str);
    }

    if (!preloading && !derived) {
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        if (!instance.IsVertexInputDynamicState()) {
            const auto& vs_info = runtime_infos[u32(Shader::LogicalStage::Vertex)].vs_info;
            GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                            guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1,
                            Shader::BoundStage{stages[u32(Shader::LogicalStage::Vertex)]});
        }
    }

    const vk::PipelineVertexInputDivisorStateCreateInfo divisor_state = {
        .vertexBindingDivisorCount = static_cast<u32>(sdata.divisors.size()),
        .pVertexBindingDivisors = sdata.divisors.data(),
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .pNext = sdata.divisors.empty() ? nullptr : &divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(sdata.vertex_bindings.size()),
        .pVertexBindingDescriptions = sdata.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(sdata.vertex_attributes.size()),
        .pVertexAttributeDescriptions = sdata.vertex_attributes.data(),
    };

    const auto topology = LiverpoolToVK::PrimitiveType(key.prim_type);
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = topology,
    };

    const bool is_rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == AmdGpu::PrimitiveType::QuadList;
    const vk::PipelineTessellationStateCreateInfo tessellation_state = {
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    vk::StructureChain raster_chain = {
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
        const auto& fs_info = runtime_infos[u32(Shader::LogicalStage::Fragment)].fs_info;
        sdata.multisampling = {
            .rasterizationSamples = LiverpoolToVK::NumSamples(
                key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
            .sampleShadingEnable =
                fs_info.addr_flags.persp_sample_ena || fs_info.addr_flags.linear_sample_ena,
        };
    }

    const vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control = {
        .negativeOneToOne = key.clip_space == AmdGpu::ClipSpace::MinusWToW,
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };

    boost::container::static_vector<vk::DynamicState, 32> dynamic_states = {
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

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
        shader_stages;
    auto stage = u32(Shader::LogicalStage::Vertex);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::LogicalStage::Geometry);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::LogicalStage::TessellationControl);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto type = is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS;
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::LogicalStage::Fragment)].fs_info;
            sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(type, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = CompileSPV(sdata.tcs, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::LogicalStage::TessellationEval);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::LogicalStage::Fragment)].fs_info;
            sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                AuxShaderType::PassthroughTES, fs_info);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = CompileSPV(sdata.tes, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::LogicalStage::Fragment);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (runtime_infos[u32(Shader::LogicalStage::Fragment)].fs_info.clip_distance_emulation) {
        if (!preloading) {
            const auto vs_runtime_info =
                runtime_infos[static_cast<u32>(Shader::LogicalStage::Vertex)].vs_info;

            sdata.fragment =
                Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs_runtime_info.outputs);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = CompileSPV(sdata.fragment, instance.GetDevice()),
            .pName = "main",
        });
    }

    const auto depth_format =
        instance.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats;
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

    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 num_samples) {
        return num_samples ? LiverpoolToVK::NumSamples(num_samples, instance.GetColorSampleCounts())
                           : vk::SampleCountFlagBits::e1;
    });
    const vk::AttachmentSampleCountInfoAMD mixed_samples = {
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };

    const vk::PipelineRenderingCreateInfo pipeline_rendering_ci = {
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

    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> attachments;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        const auto& control = key.blend_controls[i];

        const auto src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
        const auto color_blend = LiverpoolToVK::BlendOp(control.color_func);

        const auto src_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                                   : src_color;
        const auto dst_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                                   : dst_color;
        const auto alpha_blend =
            control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : color_blend;

        const auto color_scaled_min_max =
            (color_blend == vk::BlendOp::eMin || color_blend == vk::BlendOp::eMax) &&
            (src_color != vk::BlendFactor::eOne || dst_color != vk::BlendFactor::eOne);
        const auto alpha_scaled_min_max =
            (alpha_blend == vk::BlendOp::eMin || alpha_blend == vk::BlendOp::eMax) &&
            (src_alpha != vk::BlendFactor::eOne || dst_alpha != vk::BlendFactor::eOne);
        if (color_scaled_min_max || alpha_scaled_min_max) {
            LOG_WARNING(
                Render_Vulkan,
                "Unimplemented use of min/max blend op with blend factor not equal to one.");
        }

        attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = control.enable,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = color_blend,
            .srcAlphaBlendFactor = src_alpha,
            .dstAlphaBlendFactor = dst_alpha,
            .alphaBlendOp = alpha_blend,
            .colorWriteMask =
                instance.IsDynamicColorWriteMaskSupported()
                    ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                    : key.write_masks[i],
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

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != AmdGpu::ColorControl::LogicOp::Copy,
        .logicOp = LiverpoolToVK::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Required by spec unless VK_EXT_extended_dynamic_state3 is supported.
    // In practice, we use dynamic state for all of it.
    constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_info = {};

    // Descriptor mappings go on shader stage creation; auxiliary stages bind nothing.
    const NativeApi::PipelineCreateFlags2CreateInfo flags2_ci = {
        .sType = NativeApi::StructureTypePipelineCreateFlags2CreateInfo,
        .pNext = &pipeline_rendering_ci,
        .flags = NativeApi::PipelineCreate2DescriptorHeapBit,
    };
    if (uses_native_bindings) {
        for (auto& shader_stage : shader_stages) {
            if (const auto* mapping_info =
                    NativeMappingInfo(shader_stage.stage, shader_stage.pNext)) {
                shader_stage.pNext = mapping_info;
            }
        }
    }

    const vk::GraphicsPipelineCreateInfo pipeline_info = {
        .pNext = uses_native_bindings ? static_cast<const void*>(&flags2_ci)
                                      : static_cast<const void*>(&pipeline_rendering_ci),
        .flags = DiagPipelineCreateFlags(),
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &vertex_input_info : nullptr,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = &tessellation_state,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &sdata.multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &depth_stencil_info : nullptr,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = uses_native_bindings ? vk::PipelineLayout{} : *pipeline_layout,
    };

    auto [pipeline_result, pipe] =
        device.createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
    if (pipeline_result != vk::Result::eSuccess && preloading) {
        // A cached pipeline the driver refuses is left out of the preload (IsCreated() false);
        // the draw that needs it compiles it from its own state at play time. 2026-10-19: an
        // RX 7800 XT returned ErrorInitializationFailed in the boot preload and the assert below
        // ended the game on every launch after the first.
        LOG_WARNING(Render_Vulkan,
                    "Cached graphics pipeline {} refused by the driver ({}): left to compile at "
                    "play time",
                    debug_str, vk::to_string(pipeline_result));
        return;
    }
    if (pipeline_result == vk::Result::eSuccess && refusable) {
        // SHADPS4_REFUSE_VARIANT_TEST=N (test, 2026-10-22): every Nth refusable variant is treated
        // as refused by the driver (the created pipeline is dropped), to run the path below on a
        // driver that never refuses.
        static const u32 every = [] {
            const char* v = std::getenv("SHADPS4_REFUSE_VARIANT_TEST");
            return v ? static_cast<u32>(std::strtoul(v, nullptr, 10)) : 0u;
        }();
        static std::atomic<u32> made{0};
        if (every != 0 && made.fetch_add(1, std::memory_order_relaxed) % every == every - 1) {
            pipeline_result = vk::Result::eErrorUnknown;
            pipe.reset();
        }
    }
    if (pipeline_result != vk::Result::eSuccess && refusable) {
        // 1.1.1 (GitHub #25 RX 6800 XT, #31 RX 570): AMD drivers refused some NativeColour
        // variants with ErrorUnknown and this assert ended the game. A variant is optional - the
        // caller keeps the normal pipeline for that draw.
        LOG_WARNING(Render_Vulkan, "Graphics pipeline variant {} refused by the driver ({})",
                    debug_str, vk::to_string(pipeline_result));
        return;
    }
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create graphics pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
}

GraphicsPipeline::~GraphicsPipeline() = default;

// GetVertexInputs on a fetch shader that is not yet owned by a pipeline (DeriveStageInputs).
template <typename Attribute, typename Binding>
static void FetchVertexInputs(const Shader::Gcn::FetchShaderData* fetch_shader,
                              VertexInputs<Attribute>& attributes,
                              VertexInputs<Binding>& bindings,
                              VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
                              VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0,
                              u32 step_rate_1, const Shader::BoundStage& vs) {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (!fetch_shader || fetch_shader->attributes.empty()) {
        return;
    }
    for (const auto& attrib : fetch_shader->attributes) {
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs);
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

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1,
    const Shader::BoundStage& vs) const {
    FetchVertexInputs<Attribute, Binding>(fetch_shader.get(), attributes, bindings, divisors,
                                          guest_buffers, step_rate_0, step_rate_1, vs);
}

// Declare templated GetVertexInputs for necessary types.
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>& attributes,
    VertexInputs<vk::VertexInputBindingDescription>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1,
    const Shader::BoundStage& vs) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1,
    const Shader::BoundStage& vs) const;

u32 GraphicsPipeline::DeriveVertexInputs(AmdGpu::CpSplit::VertexInputsDerived& out,
                                         u32 step_rate_0, u32 step_rate_1,
                                         const Shader::BoundStage& vs) const {
    using namespace AmdGpu::CpSplit;
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    static_assert(MaxVertexInputs == MaxVertexBufferCount);
    out.header = {};
    if (fetch_shader && !fetch_shader->attributes.empty()) {
        // The GetVertexInputs loop (the 2EXT instantiation: the divisor rides in the binding).
        for (const auto& attrib : fetch_shader->attributes) {
            ASSERT_MSG(out.header.count < MaxVertexInputs,
                       "Vertex attribute binding count exceeded limit: {} >= {}",
                       out.header.count, MaxVertexInputs);
            const auto step_rate = attrib.GetStepRate();
            const auto buffer = attrib.GetSharp(vs);
            auto& a = out.attribs[out.header.count++];
            a.desc.vk_format = static_cast<u32>(
                LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt()));
            a.desc.stride = buffer.GetStride();
            a.desc.divisor = step_rate == InstanceIdType::OverStepRate0
                                 ? step_rate_0
                                 : (step_rate == InstanceIdType::OverStepRate1 ? step_rate_1 : 1);
            a.desc.location = attrib.semantic;
            a.desc.instanced = step_rate != InstanceIdType::None;
            a.desc.pad = 0;
            a.base_address = buffer.base_address;
            a.size = buffer.GetSize();
            a.range = VertexInputNoRange;
            a.pad = 0;
        }
    }
    // BindVertexBuffers' range list: the buffers with a base and a size, sorted by base
    // (insertion sort: at most 32, usually a handful), connecting ranges merged.
    std::array<VertexInputRange, MaxVertexInputs> sorted;
    u32 num_sorted = 0;
    for (u32 i = 0; i < out.header.count; ++i) {
        const auto& a = out.attribs[i];
        if (a.base_address == 0 || a.size == 0) {
            continue;
        }
        const VertexInputRange range{a.base_address, a.base_address + a.size};
        u32 pos = num_sorted++;
        while (pos > 0 && sorted[pos - 1].base_address > range.base_address) {
            sorted[pos] = sorted[pos - 1];
            --pos;
        }
        sorted[pos] = range;
    }
    for (u32 i = 0; i < num_sorted; ++i) {
        const auto& range = sorted[i];
        if (i == 0 || out.ranges[out.header.range_count - 1].end_address < range.base_address) {
            out.ranges[out.header.range_count++] = range;
        } else {
            auto& prev = out.ranges[out.header.range_count - 1];
            prev.end_address = std::max(prev.end_address, range.end_address);
        }
    }
    // The merged range each bound buffer falls in (BindVertexBuffers' find_if per buffer).
    for (u32 i = 0; i < out.header.count; ++i) {
        auto& a = out.attribs[i];
        if (a.base_address == 0 || a.size == 0) {
            continue;
        }
        u32 r = 0;
        while (r < out.header.range_count &&
               !(a.base_address >= out.ranges[r].base_address &&
                 a.base_address < out.ranges[r].end_address)) {
            ++r;
        }
        ASSERT(r < out.header.range_count);
        a.range = static_cast<u16>(r);
    }
    return out.Bytes();
}

GraphicsPipeline::DescSetBindings GraphicsPipeline::DeriveStageInputs(
    const Instance& instance, std::span<const Shader::Info*, MaxShaderStages> stages,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    const Shader::Gcn::FetchShaderData* fetch_shader, SerializationSupport& sdata) {
    // The same two reads of the Infos the constructor makes without `derived`, in its order.
    DescSetBindings bindings;
    CollectDescSetBindings(stages, sdata, false, bindings);
    if (!instance.IsVertexInputDynamicState()) {
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        const auto& vs_info = runtime_infos[u32(Shader::LogicalStage::Vertex)].vs_info;
        FetchVertexInputs(fetch_shader, sdata.vertex_attributes, sdata.vertex_bindings,
                          sdata.divisors, guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1,
                          Shader::BoundStage{stages[u32(Shader::LogicalStage::Vertex)]});
    }
    return bindings;
}

void GraphicsPipeline::CollectDescSetBindings(std::span<const Shader::Info* const> stages,
                                              SerializationSupport& sdata, bool preloading,
                                              DescSetBindings& bindings) {
    u32 binding{};
    // Image descriptor counts cannot be recomputed while preloading, so they are carried in
    // sdata. Populated on first creation, replayed on preload.
    u32 image_idx{};
    if (!preloading) {
        sdata.image_binding_counts.clear();
    }

    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto stage_bit = LogicalStageToStageBit[u32(stage->l_stage)];
        for (const auto& buffer : stage->buffers) {
            const auto sharp =
                preloading ? AmdGpu::Buffer{}
                           : buffer.GetSharp(*stage); // See for the comment in compute PL creation
            bindings.push_back({
                .binding = binding++,
                .descriptorType = buffer.IsStorage(sharp) ? vk::DescriptorType::eStorageBuffer
                                                          : vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        for (const auto& image : stage->images) {
            const bool have_cached = image_idx < sdata.image_binding_counts.size();
            if (preloading && !have_cached) {
                // Only reachable from a cache written before this field existed, which the
                // version bump should already have rejected.
                LOG_ERROR(Render_Vulkan,
                          "Preloaded pipeline is missing image binding counts, descriptor set "
                          "layout will be wrong");
            }
            const u32 num_bindings = (preloading && have_cached)
                                         ? sdata.image_binding_counts[image_idx]
                                         : image.NumBindings(*stage);
            if (!preloading) {
                sdata.image_binding_counts.push_back(num_bindings);
            }
            ++image_idx;
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
}

void GraphicsPipeline::BuildDescSetLayout(
    std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    // The unified binding count is the next free index after the last binding.
    const u32 binding =
        bindings.empty() ? 0u : bindings.back().binding + bindings.back().descriptorCount;
    if (instance.IsNativeBindingsActive()) {
        // Same unified binding numbering, expressed as descriptor heap mappings instead.
        BuildNativeMappings(bindings);
        return;
    }
    // l4c: the total descriptor count (a sparse layout - SHADPS4_NATIVE_SPARSE_LAYOUT - has gaps);
    // equal to `binding` for a contiguous one.
    u32 total = 0;
    for (const auto& b : bindings) {
        total += b.descriptorCount;
    }
    uses_push_descriptors = total < instance.MaxPushDescriptors();
    (void)binding;
    BuildPacketBindings(bindings);
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
