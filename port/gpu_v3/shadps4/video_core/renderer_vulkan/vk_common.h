// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Include vulkan-hpp header
#define VK_ENABLE_BETA_EXTENSIONS
#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_SETTERS
#define VULKAN_HPP_HAS_SPACESHIP_OPERATOR
#define VULKAN_HPP_NO_EXCEPTIONS
// Define assert-on-result to nothing to instead return the result for our handling.
#define VULKAN_HPP_ASSERT_ON_RESULT

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#include <vulkan/vulkan.hpp>
#pragma clang diagnostic pop

#include <cstdlib>

namespace Vulkan {

/// SHADPS4_GENERAL_LAYOUTS=1: texture-cache images live in GENERAL instead of per-use layouts, so
/// read-to-read uses never need a layout transition (and its render-pass break). Only meant for
/// devices with VK_KHR_unified_image_layouts' unifiedImageLayouts, where GENERAL is as fast.
/// Presentation and attachment feedback loops keep their dedicated layouts.
inline bool GeneralLayouts() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_GENERAL_LAYOUTS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
inline vk::ImageLayout GuestLayout(vk::ImageLayout layout) {
    if (!GeneralLayouts() || layout == vk::ImageLayout::eUndefined ||
        layout == vk::ImageLayout::ePresentSrcKHR ||
        layout == vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT) {
        return layout;
    }
    return vk::ImageLayout::eGeneral;
}

} // namespace Vulkan

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
