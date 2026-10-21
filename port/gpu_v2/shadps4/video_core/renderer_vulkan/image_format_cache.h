// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <map>
#include <mutex>
#include <utility>
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

// One cache per physical-device Instance. Dimensions are not inputs to this query.
// Extension chains bypass the cache because their pointed-to data affects the result.
class ImageFormatCache {
public:
    using Result = vk::ResultValue<vk::ImageFormatProperties2>;
    template <typename Query>
    Result Get(const vk::PhysicalDeviceImageFormatInfo2& info, Query&& query) const {
        if (info.pNext != nullptr) {
            return query(info);
        }
        const std::array<std::uint32_t, 5> key{
            static_cast<std::uint32_t>(info.format), static_cast<std::uint32_t>(info.type),
            static_cast<std::uint32_t>(info.tiling),
            static_cast<std::uint32_t>(VkImageUsageFlags(info.usage)),
            static_cast<std::uint32_t>(VkImageCreateFlags(info.flags))};
        std::lock_guard lock{mutex};
        if (const auto it = results.find(key); it != results.end()) {
            return it->second;
        }
        const auto result = query(info);
        if (result.result == vk::Result::eSuccess ||
            result.result == vk::Result::eErrorFormatNotSupported) {
            results.emplace(key, result);
        }
        return result;
    }

private:
    mutable std::mutex mutex;
    mutable std::map<std::array<std::uint32_t, 5>, Result> results;
};
} // namespace Vulkan
