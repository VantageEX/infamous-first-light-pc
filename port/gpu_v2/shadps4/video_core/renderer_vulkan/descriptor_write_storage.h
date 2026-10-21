// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <vector>
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
inline void PrepareDescriptorWrites(std::vector<vk::WriteDescriptorSet>& writes,
                                    size_t count, bool reuse) {
    if (!reuse) {
        writes.clear();
    }
    writes.resize(count);
}

inline void WriteBufferDescriptor(vk::WriteDescriptorSet& write, uint32_t binding,
                                  const vk::DescriptorBufferInfo* info) {
    write.dstSet = nullptr;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = vk::DescriptorType::eStorageBuffer;
    write.pBufferInfo = info;
    write.pImageInfo = nullptr;
}

inline void WriteImageDescriptor(vk::WriteDescriptorSet& write, uint32_t binding,
                                 uint32_t count, vk::DescriptorType type,
                                 const vk::DescriptorImageInfo* info) {
    write.dstSet = nullptr;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = count;
    write.descriptorType = type;
    write.pImageInfo = info;
    write.pBufferInfo = nullptr;
}
} // namespace Vulkan
