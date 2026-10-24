// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/native_emit.h"
#include <array>
#include <mutex>
#include "common/logging/log.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

/// SHADPS4_NATIVE_BINDLESS (notes/native-path-2026-10-09.md sec. 12): the native colour pass's
/// global descriptor arrays. Set 1 of a bindless colour variant's pipeline layout: binding 0 =
/// sampled images, binding 1 = samplers, update-after-bind, partially bound, updatable while
/// pending (slots are only written while unused). NumSets copies rotate: a copy is cleared only
/// when the GPU has finished every batch that bound it.
namespace Vulkan::Bindless {

constexpr u32 ImageSlots = 32768;
constexpr u32 SamplerSlots = 4096;
constexpr u32 NumSets = 3;

struct Global {
    vk::UniqueDescriptorSetLayout layout;
    vk::UniqueDescriptorPool pool;
    std::array<vk::DescriptorSet, NumSets> sets{};
    bool ok = false;
};

inline Global& G() {
    static Global g;
    return g;
}

/// Creates the layout, pool and sets once (any thread). False: bindless unavailable.
inline bool Init(vk::Device device) {
    static std::once_flag once;
    std::call_once(once, [device] {
        auto& g = G();
        const vk::DescriptorBindingFlags fl = vk::DescriptorBindingFlagBits::eUpdateAfterBind |
                                              vk::DescriptorBindingFlagBits::ePartiallyBound |
                                              vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
        const std::array<vk::DescriptorBindingFlags, 2> flags{fl, fl};
        const vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info{
            .bindingCount = 2,
            .pBindingFlags = flags.data(),
        };
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = ImageSlots,
             .stageFlags = Shader::g_native_bindless_vs
                               ? vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex
                               : vk::ShaderStageFlags{vk::ShaderStageFlagBits::eFragment}},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eSampler,
             .descriptorCount = SamplerSlots,
             .stageFlags = Shader::g_native_bindless_vs
                               ? vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex
                               : vk::ShaderStageFlags{vk::ShaderStageFlagBits::eFragment}},
        }};
        const vk::DescriptorSetLayoutCreateInfo layout_info{
            .pNext = &flags_info,
            .flags = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
            .bindingCount = 2,
            .pBindings = bindings.data(),
        };
        auto [lr, layout] = device.createDescriptorSetLayoutUnique(layout_info);
        if (lr != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "NativeBindless: set layout failed ({})", vk::to_string(lr));
            return;
        }
        const std::array<vk::DescriptorPoolSize, 2> sizes{{
            {vk::DescriptorType::eSampledImage, ImageSlots * NumSets},
            {vk::DescriptorType::eSampler, SamplerSlots * NumSets},
        }};
        const vk::DescriptorPoolCreateInfo pool_info{
            .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
            .maxSets = NumSets,
            .poolSizeCount = 2,
            .pPoolSizes = sizes.data(),
        };
        auto [pr, pool] = device.createDescriptorPoolUnique(pool_info);
        if (pr != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "NativeBindless: pool failed ({})", vk::to_string(pr));
            return;
        }
        std::array<vk::DescriptorSetLayout, NumSets> layouts;
        layouts.fill(*layout);
        const vk::DescriptorSetAllocateInfo alloc_info{
            .descriptorPool = *pool,
            .descriptorSetCount = NumSets,
            .pSetLayouts = layouts.data(),
        };
        auto [ar, sets] = device.allocateDescriptorSets(alloc_info);
        if (ar != vk::Result::eSuccess || sets.size() != NumSets) {
            LOG_ERROR(Render_Vulkan, "NativeBindless: sets failed ({})", vk::to_string(ar));
            return;
        }
        for (u32 i = 0; i < NumSets; ++i) {
            g.sets[i] = sets[i];
        }
        g.layout = std::move(layout);
        g.pool = std::move(pool);
        g.ok = true;
        LOG_INFO(Render_Vulkan, "NativeBindless: {} sets of {} images / {} samplers", NumSets,
                 ImageSlots, SamplerSlots);
    });
    return G().ok;
}

} // namespace Vulkan::Bindless
