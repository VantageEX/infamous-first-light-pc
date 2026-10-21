// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <cstdint>
#include <malloc.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "infamous_renderer/feedback_record.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace iss::game::feedback {

// PC-owned, host-coherent Vulkan storage for 48-byte feedback records. This
// does not alias guest memory. A caller must copy the game's initial bytes in,
// keep a slot alive through its GPU tick, and publish results only after that
// tick completes. These rules are deliberately outside this allocation class.
class HostFeedbackBuffer {
public:
    static constexpr std::uint64_t SlotStride = 256;
    static constexpr std::uint64_t RecordBytes = Record::ByteSize;

    static std::optional<HostFeedbackBuffer> Create(const Vulkan::Instance& instance,
                                                     std::uint64_t logical_size) {
        if (!instance.HasExternalMemoryHost() || logical_size < SlotStride ||
            logical_size % SlotStride != 0) {
            return std::nullopt;
        }
        HostFeedbackBuffer result;
        const auto device = instance.GetDevice();
        const auto properties = instance.GetPhysicalDevice().getProperties2<
            vk::PhysicalDeviceProperties2, vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
        result.alignment = std::max<std::uint64_t>(
            4096, properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
                      .minImportedHostPointerAlignment);
        const vk::ExternalMemoryBufferCreateInfo external{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT};
        const vk::BufferCreateInfo info{
            .pNext = &external,
            .size = logical_size,
            .usage = vk::BufferUsageFlagBits::eTransferSrc |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive};
        auto created = device.createBufferUnique(info);
        if (created.result != vk::Result::eSuccess) {
            throw std::runtime_error("create host feedback buffer failed");
        }
        result.buffer = std::move(created.value);
        const auto requirements = device.getBufferMemoryRequirements(*result.buffer);
        const auto allocation_size =
            ((requirements.size + result.alignment - 1) / result.alignment) * result.alignment;
        result.allocation.reset(_aligned_malloc(allocation_size, result.alignment));
        if (!result.allocation) throw std::bad_alloc();
        const auto host_properties = device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, result.allocation.get());
        if (host_properties.result != vk::Result::eSuccess) {
            throw std::runtime_error("host feedback pointer properties failed");
        }
        const auto memory_properties = instance.GetPhysicalDevice().getMemoryProperties();
        const auto compatible =
            host_properties.value.memoryTypeBits & requirements.memoryTypeBits;
        const auto needed = vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent;
        auto type = memory_properties.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((compatible & (1u << i)) && (flags & needed) == needed) {
                type = i;
                break;
            }
        }
        if (type == memory_properties.memoryTypeCount) return std::nullopt;
        const vk::ImportMemoryHostPointerInfoEXT imported{
            .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            .pHostPointer = result.allocation.get()};
        auto allocated = device.allocateMemoryUnique(vk::MemoryAllocateInfo{
            .pNext = &imported, .allocationSize = allocation_size, .memoryTypeIndex = type});
        if (allocated.result != vk::Result::eSuccess) {
            throw std::runtime_error("allocate host feedback memory failed");
        }
        result.memory = std::move(allocated.value);
        if (device.bindBufferMemory(*result.buffer, *result.memory, 0) != vk::Result::eSuccess) {
            throw std::runtime_error("bind host feedback memory failed");
        }
        result.logical_size = logical_size;
        result.memory_type = type;
        return result;
    }

    HostFeedbackBuffer(HostFeedbackBuffer&&) noexcept = default;
    HostFeedbackBuffer& operator=(HostFeedbackBuffer&&) noexcept = default;
    HostFeedbackBuffer(const HostFeedbackBuffer&) = delete;
    HostFeedbackBuffer& operator=(const HostFeedbackBuffer&) = delete;

    [[nodiscard]] vk::Buffer Handle() const noexcept { return *buffer; }
    [[nodiscard]] std::uint8_t* Data() const noexcept {
        return static_cast<std::uint8_t*>(allocation.get());
    }
    [[nodiscard]] std::uint64_t Size() const noexcept { return logical_size; }
    [[nodiscard]] std::uint64_t Capacity() const noexcept { return logical_size / SlotStride; }
    [[nodiscard]] std::uint64_t Alignment() const noexcept { return alignment; }
    [[nodiscard]] std::uint32_t MemoryTypeIndex() const noexcept { return memory_type; }
    [[nodiscard]] vk::DescriptorBufferInfo Descriptor(std::uint64_t slot) const {
        if (slot >= Capacity()) throw std::out_of_range("feedback slot");
        return vk::DescriptorBufferInfo{.buffer = *buffer,
                                        .offset = slot * SlotStride,
                                        .range = RecordBytes};
    }

private:
    HostFeedbackBuffer() = default;
    // Reverse destruction order: buffer, imported memory, then host allocation.
    std::unique_ptr<void, void (*)(void*)> allocation{nullptr, _aligned_free};
    vk::UniqueDeviceMemory memory{};
    vk::UniqueBuffer buffer{};
    std::uint64_t logical_size{};
    std::uint64_t alignment{};
    std::uint32_t memory_type{};
};

} // namespace iss::game::feedback
