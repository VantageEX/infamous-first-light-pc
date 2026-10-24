// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

// Narrow private C ABI for VK_EXT_descriptor_heap, transcribed from the installed
// SDK 1.4.357.0 vulkan_core.h. The project's Vulkan headers may predate the extension, so
// everything lives under private names and never collides with a newer header.
namespace Vulkan::NativeApi {

inline constexpr const char* DescriptorHeapExtensionName = "VK_EXT_descriptor_heap";
inline constexpr const char* Maintenance5ExtensionName = "VK_KHR_maintenance5";
inline constexpr const char* ShaderUntypedPointersExtensionName = "VK_KHR_shader_untyped_pointers";

inline constexpr auto StructureTypePipelineCreateFlags2CreateInfo =
    static_cast<VkStructureType>(1000470005);
inline constexpr auto StructureTypeImageDescriptorInfo = static_cast<VkStructureType>(1000135001);
inline constexpr auto StructureTypeResourceDescriptorInfo =
    static_cast<VkStructureType>(1000135002);
inline constexpr auto StructureTypeBindHeapInfo = static_cast<VkStructureType>(1000135003);
inline constexpr auto StructureTypePushDataInfo = static_cast<VkStructureType>(1000135004);
inline constexpr auto StructureTypeDescriptorSetAndBindingMapping =
    static_cast<VkStructureType>(1000135005);
inline constexpr auto StructureTypeShaderDescriptorSetAndBindingMappingInfo =
    static_cast<VkStructureType>(1000135006);
inline constexpr auto StructureTypePhysicalDeviceDescriptorHeapProperties =
    static_cast<VkStructureType>(1000135008);
inline constexpr auto StructureTypePhysicalDeviceDescriptorHeapFeatures =
    static_cast<VkStructureType>(1000135009);
inline constexpr auto StructureTypeSamplerCustomBorderColorIndexCreateInfo =
    static_cast<VkStructureType>(1000135011);

inline constexpr VkBufferUsageFlags BufferUsageDescriptorHeapBit = 0x10000000;
inline constexpr u64 PipelineCreate2DescriptorHeapBit = 0x1000000000ULL;

enum DescriptorMappingSource : s32 {
    MappingSourceHeapWithConstantOffset = 0,
    MappingSourceHeapWithPushIndex = 1,
    MappingSourceHeapWithIndirectIndex = 2,
    MappingSourceHeapWithIndirectIndexArray = 3,
    MappingSourceResourceHeapData = 4,
    MappingSourcePushData = 5,
    MappingSourcePushAddress = 6,
    MappingSourceIndirectAddressValue = 7,
    MappingSourceMaxEnum = 0x7FFFFFFF,
};

inline constexpr u32 SpirvResourceTypeAll = 0x7FFFFFFF;

struct HostAddressRange {
    void* address;
    size_t size;
};

struct HostAddressRangeConst {
    const void* address;
    size_t size;
};

// VkDeviceAddressRangeKHR. Not part of the supplied excerpt; layout assumed.
struct DeviceAddressRange {
    VkDeviceAddress address;
    VkDeviceSize size;
};

struct ImageDescriptorInfo {
    VkStructureType sType;
    const void* pNext;
    const VkImageViewCreateInfo* pView;
    VkImageLayout layout;
};

union ResourceDescriptorData {
    const ImageDescriptorInfo* pImage;
    const void* pTexelBuffer;
    const DeviceAddressRange* pAddressRange;
    const void* pTensorARM;
};

struct ResourceDescriptorInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorType type;
    ResourceDescriptorData data;
};

struct BindHeapInfo {
    VkStructureType sType;
    const void* pNext;
    DeviceAddressRange heapRange;
    VkDeviceSize reservedRangeOffset;
    VkDeviceSize reservedRangeSize;
};

struct PushDataInfo {
    VkStructureType sType;
    const void* pNext;
    u32 offset;
    HostAddressRangeConst data;
};

struct MappingSourceConstantOffset {
    u32 heapOffset;
    u32 heapArrayStride;
    const VkSamplerCreateInfo* pEmbeddedSampler;
    u32 samplerHeapOffset;
    u32 samplerHeapArrayStride;
};

struct MappingSourcePushIndex {
    u32 heapOffset;
    u32 pushOffset;
    u32 heapIndexStride;
    u32 heapArrayStride;
    const VkSamplerCreateInfo* pEmbeddedSampler;
    VkBool32 useCombinedImageSamplerIndex;
    u32 samplerHeapOffset;
    u32 samplerPushOffset;
    u32 samplerHeapIndexStride;
    u32 samplerHeapArrayStride;
};

struct MappingSourceIndirectIndex {
    u32 heapOffset;
    u32 pushOffset;
    u32 addressOffset;
    u32 heapIndexStride;
    u32 heapArrayStride;
    const VkSamplerCreateInfo* pEmbeddedSampler;
    VkBool32 useCombinedImageSamplerIndex;
    u32 samplerHeapOffset;
    u32 samplerPushOffset;
    u32 samplerAddressOffset;
    u32 samplerHeapIndexStride;
    u32 samplerHeapArrayStride;
};

struct MappingSourceIndirectIndexArray {
    u32 heapOffset;
    u32 pushOffset;
    u32 addressOffset;
    u32 heapIndexStride;
    const VkSamplerCreateInfo* pEmbeddedSampler;
    VkBool32 useCombinedImageSamplerIndex;
    u32 samplerHeapOffset;
    u32 samplerPushOffset;
    u32 samplerAddressOffset;
    u32 samplerHeapIndexStride;
};

struct MappingSourceHeapData {
    u32 heapOffset;
    u32 pushOffset;
};

struct MappingSourceIndirectAddress {
    u32 pushOffset;
    u32 addressOffset;
};

// The shader-record members of the SDK union are omitted; none is larger than indirectIndex,
// so the union keeps the SDK size and alignment.
union MappingSourceData {
    MappingSourceConstantOffset constantOffset;
    MappingSourcePushIndex pushIndex;
    MappingSourceIndirectIndex indirectIndex;
    MappingSourceIndirectIndexArray indirectIndexArray;
    MappingSourceHeapData heapData;
    u32 pushDataOffset;
    u32 pushAddressOffset;
    MappingSourceIndirectAddress indirectAddress;
};

struct DescriptorSetAndBindingMapping {
    VkStructureType sType;
    const void* pNext;
    u32 descriptorSet;
    u32 firstBinding;
    u32 bindingCount;
    u32 resourceMask;
    DescriptorMappingSource source;
    MappingSourceData sourceData;
};

struct ShaderDescriptorSetAndBindingMappingInfo {
    VkStructureType sType;
    const void* pNext;
    u32 mappingCount;
    const DescriptorSetAndBindingMapping* pMappings;
};

struct PipelineCreateFlags2CreateInfo {
    VkStructureType sType;
    const void* pNext;
    u64 flags;
};

struct PhysicalDeviceDescriptorHeapFeatures {
    VkStructureType sType;
    void* pNext;
    VkBool32 descriptorHeap;
    VkBool32 descriptorHeapCaptureReplay;
};

struct PhysicalDeviceDescriptorHeapProperties {
    VkStructureType sType;
    void* pNext;
    VkDeviceSize samplerHeapAlignment;
    VkDeviceSize resourceHeapAlignment;
    VkDeviceSize maxSamplerHeapSize;
    VkDeviceSize maxResourceHeapSize;
    VkDeviceSize minSamplerHeapReservedRange;
    VkDeviceSize minSamplerHeapReservedRangeWithEmbedded;
    VkDeviceSize minResourceHeapReservedRange;
    VkDeviceSize samplerDescriptorSize;
    VkDeviceSize imageDescriptorSize;
    VkDeviceSize bufferDescriptorSize;
    VkDeviceSize samplerDescriptorAlignment;
    VkDeviceSize imageDescriptorAlignment;
    VkDeviceSize bufferDescriptorAlignment;
    VkDeviceSize maxPushDataSize;
    size_t imageCaptureReplayOpaqueDataSize;
    u32 maxDescriptorHeapEmbeddedSamplers;
    u32 samplerYcbcrConversionCount;
    VkBool32 sparseDescriptorHeaps;
    VkBool32 protectedDescriptorHeaps;
};

struct SamplerCustomBorderColorIndexCreateInfo {
    VkStructureType sType;
    const void* pNext;
    u32 index;
};

using PfnWriteSamplerDescriptors = VkResult(VKAPI_PTR*)(VkDevice, u32, const VkSamplerCreateInfo*,
                                                        const HostAddressRange*);
using PfnWriteResourceDescriptors = VkResult(VKAPI_PTR*)(VkDevice, u32,
                                                         const ResourceDescriptorInfo*,
                                                         const HostAddressRange*);
using PfnCmdBindHeap = void(VKAPI_PTR*)(VkCommandBuffer, const BindHeapInfo*);
using PfnCmdPushData = void(VKAPI_PTR*)(VkCommandBuffer, const PushDataInfo*);
using PfnRegisterCustomBorderColor =
    VkResult(VKAPI_PTR*)(VkDevice, const VkSamplerCustomBorderColorCreateInfoEXT*, VkBool32, u32*);
using PfnUnregisterCustomBorderColor = void(VKAPI_PTR*)(VkDevice, u32);

struct Functions {
    PfnWriteSamplerDescriptors write_sampler_descriptors{};
    PfnWriteResourceDescriptors write_resource_descriptors{};
    PfnCmdBindHeap cmd_bind_sampler_heap{};
    PfnCmdBindHeap cmd_bind_resource_heap{};
    PfnCmdPushData cmd_push_data{};
    PfnRegisterCustomBorderColor register_custom_border_color{};
    PfnUnregisterCustomBorderColor unregister_custom_border_color{};

    /// Border-colour entry points are optional; they are only needed with custom colours.
    bool IsComplete() const {
        return write_sampler_descriptors && write_resource_descriptors && cmd_bind_sampler_heap &&
               cmd_bind_resource_heap && cmd_push_data;
    }
};

} // namespace Vulkan::NativeApi

namespace Vulkan {

/// The shaders' push-constant block keeps offset 0; the per-draw index array address follows
/// at a fixed offset so the pipeline mappings never depend on sizeof(Shader::PushData).
inline constexpr u32 NativePushIndexAddressOffset = 128;
inline constexpr u32 NativePushBlockSize = NativePushIndexAddressOffset + sizeof(u64);

/// Upper bound on unified bindings per pipeline: Shader::NUM_BUFFERS (40) buffer infos plus
/// Shader::NUM_IMAGES (64) image and sampler infos, which is what the rasterizer can hold.
inline constexpr u32 NativeMaxBindingsPerDraw = 128;
inline constexpr u32 NativeDirectPushBindingCapacity = 32;
inline constexpr u32 NativeDirectPushBlockCapacity =
    NativePushIndexAddressOffset + NativeDirectPushBindingCapacity * sizeof(u32);

inline constexpr u32 NativeNullStorageBufferSlot = 0;
inline constexpr u32 NativeNullUniformBufferSlot = 1;
inline constexpr u32 NativeBufferStaticSlots = 8;
inline constexpr u32 NativeBufferBlockSlots = 4096;
inline constexpr u32 NativeBufferBlocks = 16;

inline constexpr u32 NativeNullSampledImageSlot = 0;
inline constexpr u32 NativeNullStorageImageSlot = 1;
inline constexpr u32 NativeImageStaticSlots = 2;

/// Heap geometry, fixed from queried device properties before any guest pipeline exists.
/// Guest pipelines bake these offsets and strides into their descriptor mappings.
struct NativeBindingLayout {
    u64 max_push_data_size{};
    u64 resource_heap_size{};
    u64 resource_reserved_size{};
    u64 resource_heap_alignment{};
    u32 image_base{};
    u32 image_stride{};
    u32 image_descriptor_size{};
    u32 image_slots{};
    u32 buffer_base{};
    u32 buffer_stride{};
    u32 buffer_descriptor_size{};
    u32 buffer_slots{};

    u64 sampler_heap_size{};
    u64 sampler_reserved_size{};
    u64 sampler_heap_alignment{};
    u32 sampler_base{};
    u32 sampler_stride{};
    u32 sampler_descriptor_size{};
    u32 sampler_slots{};
};

} // namespace Vulkan
