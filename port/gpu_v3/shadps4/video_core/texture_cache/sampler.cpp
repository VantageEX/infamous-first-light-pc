// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <algorithm>
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/sampler.h"

namespace VideoCore {

Sampler::Sampler(const Vulkan::Instance& instance, const AmdGpu::Sampler& sampler,
                 const AmdGpu::BorderColorBuffer border_color_base) {
    using namespace Vulkan;
    const bool anisotropy_enable = instance.IsAnisotropicFilteringSupported() &&
                                   (AmdGpu::IsAnisoFilter(sampler.xy_mag_filter) ||
                                    AmdGpu::IsAnisoFilter(sampler.xy_min_filter));
    const float max_anisotropy =
        anisotropy_enable ? std::clamp(sampler.MaxAniso(), 1.0f, instance.MaxSamplerAnisotropy())
                          : 1.0f;
    auto border_color = LiverpoolToVK::BorderColor(sampler.border_color_type);
    if (border_color == vk::BorderColor::eFloatCustomEXT &&
        !instance.IsCustomBorderColorSupported()) {
        LOG_WARNING(Render_Vulkan, "Custom border color is not supported, falling back to black");
        border_color = vk::BorderColor::eFloatOpaqueBlack;
    }

    auto custom_color = [&]() -> std::optional<vk::SamplerCustomBorderColorCreateInfoEXT> {
        if (border_color == vk::BorderColor::eFloatCustomEXT) {
            const auto border_color_index = sampler.border_color_ptr.Value();
            const auto border_color_buffer = border_color_base.Address<std::array<float, 4>*>();
            const auto custom_border_color_array = border_color_buffer[border_color_index];

            const vk::SamplerCustomBorderColorCreateInfoEXT ret{
                .customBorderColor =
                    vk::ClearColorValue{
                        .float32 = custom_border_color_array,
                    },
                .format = vk::Format::eR32G32B32A32Sfloat,
            };
            return ret;
        } else {
            return std::nullopt;
        }
    }();

    // Descriptor heaps take custom border colours by registered index. Never substitute a
    // different colour if registration fails: that would silently alter native-mode rendering.
    auto* native_bindings = instance.IsNativeBindingsActive() ? NativeBindings::Get() : nullptr;
    NativeApi::SamplerCustomBorderColorIndexCreateInfo border_index_ci = {
        .sType = NativeApi::StructureTypeSamplerCustomBorderColorIndexCreateInfo,
    };
    if (native_bindings && custom_color) {
        if (native_bindings->RegisterBorderColor(*custom_color, native.border_index)) {
            native.border_registered = true;
            border_index_ci.index = native.border_index;
            custom_color->pNext = &border_index_ci;
        } else {
            UNREACHABLE_MSG("NativeBindings: custom border color registration failed");
        }
    }

    // SHADPS4_TEX_LOD_BIAS=-1.5 (test, 10-20): added to every sampler's mip bias - DLSS's advice
    // for renders below the output size (log2(render / output)).
    static const float extra_bias = [] {
        const char* v = std::getenv("SHADPS4_TEX_LOD_BIAS");
        return v ? static_cast<float>(std::atof(v)) : 0.0f;
    }();
    const vk::SamplerCreateInfo sampler_ci = {
        .pNext = custom_color ? &*custom_color : nullptr,
        .magFilter = LiverpoolToVK::Filter(sampler.xy_mag_filter),
        .minFilter = LiverpoolToVK::Filter(sampler.xy_min_filter),
        .mipmapMode = LiverpoolToVK::MipFilter(sampler.mip_filter),
        .addressModeU = LiverpoolToVK::ClampMode(sampler.clamp_x),
        .addressModeV = LiverpoolToVK::ClampMode(sampler.clamp_y),
        .addressModeW = LiverpoolToVK::ClampMode(sampler.clamp_z),
        .mipLodBias = std::clamp(sampler.LodBias() + extra_bias, -instance.MaxSamplerLodBias(),
                                 instance.MaxSamplerLodBias()),
        .anisotropyEnable = anisotropy_enable,
        .maxAnisotropy = max_anisotropy,
        .compareEnable = sampler.depth_compare_func != AmdGpu::DepthCompare::Never,
        .compareOp = LiverpoolToVK::DepthCompare(sampler.depth_compare_func),
        .minLod = sampler.MinLod(),
        .maxLod = sampler.MaxLod(),
        .borderColor = border_color,
        .unnormalizedCoordinates = false, // Handled in shader due to Vulkan limitations.
    };
    auto [sampler_result, smplr] = instance.GetDevice().createSamplerUnique(sampler_ci);
    ASSERT_MSG(sampler_result == vk::Result::eSuccess, "Failed to create sampler: {}",
               vk::to_string(sampler_result));
    handle = std::move(smplr);

    if (native_bindings) {
        native.sampler_ci = sampler_ci;
        native.sampler_ci.pNext = nullptr;
        if (custom_color) {
            native.custom_color = *custom_color;
            native.custom_color.pNext = nullptr;
            native.has_custom_color = true;
        }
        native.armed = true;
    }
}

Sampler::~Sampler() = default;

} // namespace VideoCore
