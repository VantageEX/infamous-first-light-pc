// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_native_bindings.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {

std::atomic<NativeBindings*> g_native_bindings{nullptr};

// Worst case one draw can consume; Shader::NUM_BUFFERS is 40.
constexpr u32 MaxBuffersPerDraw = 64;

} // Anonymous namespace

NativeImageViewState::~NativeImageViewState() {
    Release();
}

NativeImageViewState::NativeImageViewState(NativeImageViewState&& other) noexcept {
    *this = std::move(other);
}

NativeImageViewState& NativeImageViewState::operator=(NativeImageViewState&& other) noexcept {
    if (this != &other) {
        Release();
        view_ci = other.view_ci;
        usage_ci = other.usage_ci;
        min_lod_ci = other.min_lod_ci;
        has_min_lod = other.has_min_lod;
        armed = other.armed;
        num_entries = other.num_entries;
        entries = other.entries;
        other.num_entries = 0;
        other.armed = false;
    }
    return *this;
}

void NativeImageViewState::Release() noexcept {
    if (num_entries == 0) {
        return;
    }
    if (auto* native = NativeBindings::Get()) {
        for (u32 i = 0; i < num_entries; ++i) {
            native->RetireImageSlot(entries[i].slot, entries[i].epoch);
        }
    }
    num_entries = 0;
}

NativeSamplerState::~NativeSamplerState() {
    Release();
}

NativeSamplerState::NativeSamplerState(NativeSamplerState&& other) noexcept {
    *this = std::move(other);
}

NativeSamplerState& NativeSamplerState::operator=(NativeSamplerState&& other) noexcept {
    if (this != &other) {
        Release();
        sampler_ci = other.sampler_ci;
        custom_color = other.custom_color;
        has_custom_color = other.has_custom_color;
        border_registered = other.border_registered;
        armed = other.armed;
        has_slot = other.has_slot;
        border_index = other.border_index;
        slot = other.slot;
        epoch = other.epoch;
        other.border_registered = false;
        other.has_slot = false;
        other.armed = false;
    }
    return *this;
}

void NativeSamplerState::Release() noexcept {
    if (!has_slot && !border_registered) {
        return;
    }
    if (auto* native = NativeBindings::Get()) {
        native->RetireSamplerSlot(slot, epoch, has_slot, border_registered, border_index);
    }
    has_slot = false;
    border_registered = false;
}

NativeBindings* NativeBindings::Get() noexcept {
    return g_native_bindings.load(std::memory_order_acquire);
}

std::unique_ptr<NativeBindings> NativeBindings::Create(const Instance& instance,
                                                       Scheduler& scheduler) {
    if (!instance.IsNativeBindingsActive()) {
        return nullptr;
    }
    std::unique_ptr<NativeBindings> native{new NativeBindings(instance, scheduler)};
    if (!native->Initialize()) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: heap setup failed, using legacy bindings");
        instance.DeactivateNativeBindings();
        return nullptr;
    }
    g_native_bindings.store(native.get(), std::memory_order_release);
    return native;
}

NativeBindings::NativeBindings(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_}, api{instance_.GetNativeApi()},
      layout{instance_.GetNativeBindingLayout()} {}

NativeBindings::~NativeBindings() {
    g_native_bindings.store(nullptr, std::memory_order_release);
    // Descriptor memory must outlive every GPU access to it.
    static_cast<void>(instance.GetDevice().waitIdle());
    if (api.unregister_custom_border_color) {
        for (const auto& border : retired_borders) {
            api.unregister_custom_border_color(instance.GetDevice(), border.index);
        }
    }
    DestroyBuffer(index_buffer);
    DestroyBuffer(sampler_heap);
    DestroyBuffer(resource_heap);
}

bool NativeBindings::CreateBuffer(OwnedBuffer& out, u64 size, VkBufferUsageFlags usage,
                                  const char* name) {
    const VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    // Host-coherent is required so descriptor writes never need an explicit flush.
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    const VkResult result = vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &handle,
                                            &out.allocation, &alloc_info);
    if (result != VK_SUCCESS || !alloc_info.pMappedData) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: failed to allocate {} ({:#x} bytes): {}", name,
                    size, vk::to_string(vk::Result{result}));
        if (result == VK_SUCCESS) {
            vmaDestroyBuffer(instance.GetAllocator(), handle, out.allocation);
            out.allocation = {};
        }
        return false;
    }
    out.buffer = vk::Buffer{handle};
    out.mapped = static_cast<u8*>(alloc_info.pMappedData);
    out.size = size;
    out.address = instance.GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo{
        .buffer = out.buffer,
    });
    SetObjectName(instance.GetDevice(), out.buffer, "NativeBindings {}", name);
    return out.address != 0;
}

void NativeBindings::DestroyBuffer(OwnedBuffer& buffer) {
    if (buffer.buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), buffer.buffer, buffer.allocation);
        buffer = {};
    }
}

bool NativeBindings::Initialize() {
    if (!CreateBuffer(resource_heap, layout.resource_heap_size,
                      NativeApi::BufferUsageDescriptorHeapBit, "resource heap") ||
        !CreateBuffer(sampler_heap, layout.sampler_heap_size,
                      NativeApi::BufferUsageDescriptorHeapBit, "sampler heap") ||
        !CreateBuffer(index_buffer, u64(IndexBlockBytes) * IndexBlocks,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      "index ring")) {
        return false;
    }
    if (resource_heap.address % layout.resource_heap_alignment != 0 ||
        sampler_heap.address % layout.sampler_heap_alignment != 0) {
        LOG_WARNING(Render_Vulkan, "NativeBindings: heap addresses {:#x}/{:#x} are misaligned",
                    resource_heap.address, sampler_heap.address);
        return false;
    }

    // Reserved ranges sit at the start of each heap and are never written by us.
    resource_bind_info = {
        .sType = NativeApi::StructureTypeBindHeapInfo,
        .heapRange = {resource_heap.address, layout.resource_heap_size},
        .reservedRangeOffset = 0,
        .reservedRangeSize = layout.resource_reserved_size,
    };
    sampler_bind_info = {
        .sType = NativeApi::StructureTypeBindHeapInfo,
        .heapRange = {sampler_heap.address, layout.sampler_heap_size},
        .reservedRangeOffset = 0,
        .reservedRangeSize = layout.sampler_reserved_size,
    };

    buffer_ring.block_units = NativeBufferBlockSlots;
    buffer_ring.ticks.assign(NativeBufferBlocks, 0);
    index_ring.block_units = IndexBlockBytes / IndexUnit;
    index_ring.ticks.assign(IndexBlocks, 0);
    buffer_keys.assign(size_t(NativeBufferBlocks) * NativeBufferBlockSlots, BufferKey{});
    buffer_cache.fill(~0u);

    image_pool.first = NativeImageStaticSlots;
    image_pool.capacity = layout.image_slots;
    image_pool.next_unused = image_pool.first;
    sampler_pool.first = 0;
    sampler_pool.capacity = layout.sampler_slots;
    sampler_pool.next_unused = sampler_pool.first;

    const char* verify_env = std::getenv("SHADPS4_NATIVE_BINDINGS_VERIFY");
    verify = verify_env && std::string_view{verify_env} == "1";
    if (verify) {
        buffer_shadow.assign(buffer_keys.size(), BufferShadow{});
        image_shadow.assign(layout.image_slots, ImageShadow{});
        sampler_shadow.assign(layout.sampler_slots, VkSampler{});
        LOG_INFO(Render_Vulkan, "NativeBindings: verifier enabled");
    }

    WriteNullDescriptors();
    next_report = ReportInterval;
    return true;
}

void NativeBindings::WriteNullDescriptors() {
    // Null resources are written through the extension, never as invented zero bytes.
    const auto write_null = [&](VkDescriptorType type, u8* dst, u32 size) {
        const NativeApi::ResourceDescriptorInfo info = {
            .sType = NativeApi::StructureTypeResourceDescriptorInfo,
            .type = type,
            .data = {.pImage = nullptr},
        };
        const NativeApi::HostAddressRange range = {dst, size};
        const VkResult result =
            api.write_resource_descriptors(instance.GetDevice(), 1, &info, &range);
        ASSERT_MSG(result == VK_SUCCESS, "Failed to write null descriptor: {}",
                   vk::to_string(vk::Result{result}));
    };
    write_null(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, BufferSlotPtr(NativeNullStorageBufferSlot),
               layout.buffer_descriptor_size);
    write_null(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, BufferSlotPtr(NativeNullUniformBufferSlot),
               layout.buffer_descriptor_size);
    write_null(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ImageSlotPtr(NativeNullSampledImageSlot),
               layout.image_descriptor_size);
    write_null(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ImageSlotPtr(NativeNullStorageImageSlot),
               layout.image_descriptor_size);
}

void NativeBindings::EnsureRingSpace(BlockRing& ring, u32 units) {
    if (ring.cursor + units <= ring.block_units) {
        return;
    }
    const u32 next = (ring.current + 1) % u32(ring.ticks.size());
    const u64 tick = ring.ticks[next];
    ++counters.ring_wraps;
    if (tick >= scheduler.CurrentTick()) {
        // The whole ring was consumed by the unsubmitted command buffer. We are between
        // draws, so submitting here is as safe as any other guest-requested flush.
        scheduler.Flush();
        ++counters.ring_flushes;
    }
    if (!scheduler.IsFree(tick)) {
        scheduler.Wait(tick);
        ++counters.ring_waits;
    }
    ring.current = next;
    ring.cursor = 0;
}

void NativeBindings::BeginDraw() {
    EnsureRingSpace(buffer_ring, MaxBuffersPerDraw);
    EnsureRingSpace(index_ring, u32((MaxIndicesPerDraw * sizeof(u32) + IndexUnit - 1) / IndexUnit));

    const bool low = AvailableSlots(image_pool) < ReservedPersistentSlots ||
                     AvailableSlots(sampler_pool) < ReservedPersistentSlots;
    if (low) {
        RecycleRetired();
        if (AvailableSlots(image_pool) < ReservedPersistentSlots ||
            AvailableSlots(sampler_pool) < ReservedPersistentSlots) {
            Rollover();
        }
    } else if (retire_pending.load(std::memory_order_relaxed) &&
               ((counters.draws + counters.dispatches) & 0x3ff) == 0) {
        RecycleRetired();
    }
}

u32 NativeBindings::AcquireSlot(SlotPool& pool) {
    if (!pool.free_list.empty()) {
        const u32 slot = pool.free_list.back();
        pool.free_list.pop_back();
        return slot;
    }
    ASSERT_MSG(pool.next_unused < pool.capacity,
               "NativeBindings: persistent slots exhausted inside a draw");
    return pool.next_unused++;
}

void NativeBindings::RecycleRetired() {
    std::scoped_lock lock{retire_mutex};
    const auto recycle = [&](SlotPool& pool) {
        while (!pool.retired.empty() && scheduler.IsFree(pool.retired.front().tick)) {
            pool.free_list.push_back(pool.retired.front().slot);
            pool.retired.pop_front();
        }
    };
    recycle(image_pool);
    recycle(sampler_pool);
    while (!retired_borders.empty() && scheduler.IsFree(retired_borders.front().tick)) {
        if (api.unregister_custom_border_color) {
            api.unregister_custom_border_color(instance.GetDevice(), retired_borders.front().index);
        }
        retired_borders.pop_front();
    }
    retire_pending.store(!image_pool.retired.empty() || !sampler_pool.retired.empty() ||
                             !retired_borders.empty(),
                         std::memory_order_relaxed);
}

void NativeBindings::Rollover() {
    // Conservative retirement: wait for every submission, then invalidate every object-owned
    // slot at once. Owners notice the epoch change and re-encode on their next use.
    scheduler.Finish();
    std::scoped_lock lock{retire_mutex};
    ++epoch;
    for (SlotPool* pool : {&image_pool, &sampler_pool}) {
        pool->free_list.clear();
        pool->retired.clear();
        pool->next_unused = pool->first;
    }
    if (api.unregister_custom_border_color) {
        for (const auto& border : retired_borders) {
            api.unregister_custom_border_color(instance.GetDevice(), border.index);
        }
    }
    retired_borders.clear();
    retire_pending.store(false, std::memory_order_relaxed);
    ++counters.rollovers;
    LOG_WARNING(Render_Vulkan, "NativeBindings: persistent slot rollover, epoch {}", epoch);
}

void NativeBindings::RetireImageSlot(u32 slot, u32 slot_epoch) noexcept {
    std::scoped_lock lock{retire_mutex};
    if (slot_epoch != epoch) {
        return;
    }
    image_pool.retired.push_back({slot, scheduler.CurrentTick()});
    retire_pending.store(true, std::memory_order_relaxed);
}

void NativeBindings::RetireSamplerSlot(u32 slot, u32 slot_epoch, bool has_slot,
                                       bool border_registered, u32 border_index) noexcept {
    std::scoped_lock lock{retire_mutex};
    const u64 tick = scheduler.CurrentTick();
    if (has_slot && slot_epoch == epoch) {
        sampler_pool.retired.push_back({slot, tick});
    }
    if (border_registered) {
        retired_borders.push_back({border_index, tick});
    }
    retire_pending.store(true, std::memory_order_relaxed);
}

void NativeBindings::WriteBufferDescriptor(u8* dst, u64 address, u64 range, bool storage) {
    const NativeApi::DeviceAddressRange address_range = {address, range};
    const NativeApi::ResourceDescriptorInfo info = {
        .sType = NativeApi::StructureTypeResourceDescriptorInfo,
        .type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .data = {.pAddressRange = &address_range},
    };
    const NativeApi::HostAddressRange dst_range = {dst, layout.buffer_descriptor_size};
    const VkResult result =
        api.write_resource_descriptors(instance.GetDevice(), 1, &info, &dst_range);
    ASSERT_MSG(result == VK_SUCCESS, "Failed to write buffer descriptor: {}",
               vk::to_string(vk::Result{result}));
}

void NativeBindings::WriteImageDescriptor(u8* dst, const NativeImageViewState& state,
                                          vk::ImageLayout image_layout, bool storage) {
    // Rebuild the creation chain from the owner's copies; the owner may have moved.
    vk::ImageViewMinLodCreateInfoEXT min_lod_ci = state.min_lod_ci;
    min_lod_ci.pNext = nullptr;
    vk::ImageViewUsageCreateInfo usage_ci = state.usage_ci;
    usage_ci.pNext = state.has_min_lod ? &min_lod_ci : nullptr;
    vk::ImageViewCreateInfo view_ci = state.view_ci;
    view_ci.pNext = &usage_ci;

    const NativeApi::ImageDescriptorInfo image_info = {
        .sType = NativeApi::StructureTypeImageDescriptorInfo,
        .pView = reinterpret_cast<const VkImageViewCreateInfo*>(&view_ci),
        .layout = static_cast<VkImageLayout>(image_layout),
    };
    const NativeApi::ResourceDescriptorInfo info = {
        .sType = NativeApi::StructureTypeResourceDescriptorInfo,
        .type = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .data = {.pImage = &image_info},
    };
    const NativeApi::HostAddressRange dst_range = {dst, layout.image_descriptor_size};
    const VkResult result =
        api.write_resource_descriptors(instance.GetDevice(), 1, &info, &dst_range);
    ASSERT_MSG(result == VK_SUCCESS, "Failed to write image descriptor: {}",
               vk::to_string(vk::Result{result}));
}

void NativeBindings::WriteSamplerDescriptor(u8* dst, const NativeSamplerState& state) {
    NativeApi::SamplerCustomBorderColorIndexCreateInfo index_ci = {
        .sType = NativeApi::StructureTypeSamplerCustomBorderColorIndexCreateInfo,
        .index = state.border_index,
    };
    vk::SamplerCustomBorderColorCreateInfoEXT custom_color = state.custom_color;
    custom_color.pNext = state.border_registered ? &index_ci : nullptr;
    vk::SamplerCreateInfo sampler_ci = state.sampler_ci;
    sampler_ci.pNext = state.has_custom_color ? &custom_color : nullptr;

    const NativeApi::HostAddressRange dst_range = {dst, layout.sampler_descriptor_size};
    const VkResult result = api.write_sampler_descriptors(
        instance.GetDevice(), 1, reinterpret_cast<const VkSamplerCreateInfo*>(&sampler_ci),
        &dst_range);
    ASSERT_MSG(result == VK_SUCCESS, "Failed to write sampler descriptor: {}",
               vk::to_string(vk::Result{result}));
}

bool NativeBindings::RegisterBorderColor(const vk::SamplerCustomBorderColorCreateInfoEXT& color,
                                         u32& index) {
    if (!api.register_custom_border_color) {
        return false;
    }
    vk::SamplerCustomBorderColorCreateInfoEXT local = color;
    local.pNext = nullptr;
    const VkResult result = api.register_custom_border_color(
        instance.GetDevice(),
        reinterpret_cast<const VkSamplerCustomBorderColorCreateInfoEXT*>(&local), VK_FALSE, &index);
    return result == VK_SUCCESS;
}

u32 NativeBindings::BufferIndex(vk::Buffer handle, u64 device_address, u64 offset, u64 range,
                                bool storage) {
    if (!handle) {
        return storage ? NativeNullStorageBufferSlot : NativeNullUniformBufferSlot;
    }
    const BufferKey key = {device_address + offset, range, storage};
    u64 hash = key.address ^ (key.range * 0x9E3779B97F4A7C15ULL) ^ key.storage;
    hash ^= hash >> 29;
    hash *= 0xBF58476D1CE4E5B9ULL;
    u32& cached = buffer_cache[(hash >> 32) & (BufferCacheSize - 1)];

    // Reuse is honoured only inside the current block, so block completion ticks stay in
    // ring order and a wrap never has to wait on a recently touched block.
    const u32 block_first = buffer_ring.current * buffer_ring.block_units;
    if (cached - block_first < buffer_ring.cursor && buffer_keys[cached] == key &&
        (!verify || (buffer_shadow[cached].handle == static_cast<VkBuffer>(handle) &&
                     buffer_shadow[cached].offset == offset))) {
        ++counters.buffers_reused;
        return NativeBufferStaticSlots + cached;
    }

    ASSERT_MSG(buffer_ring.cursor < buffer_ring.block_units,
               "NativeBindings: buffer ring block overrun inside a draw");
    const u32 slot = block_first + buffer_ring.cursor++;
    WriteBufferDescriptor(BufferSlotPtr(NativeBufferStaticSlots + slot), key.address, key.range,
                          storage);
    buffer_keys[slot] = key;
    cached = slot;
    if (verify) {
        buffer_shadow[slot] = {static_cast<VkBuffer>(handle), offset};
    }
    ++counters.buffers_encoded;
    counters.descriptor_bytes += layout.buffer_descriptor_size;
    return NativeBufferStaticSlots + slot;
}

u32 NativeBindings::ImageIndex(NativeImageViewState& state, vk::ImageView handle,
                               vk::ImageLayout image_layout, bool storage) {
    ASSERT_MSG(state.armed, "NativeBindings: image view has no retained create info");
    for (u32 i = 0; i < state.num_entries;) {
        const auto& entry = state.entries[i];
        if (entry.epoch != epoch) {
            // Invalidated by a rollover; the slot itself was already reclaimed.
            state.entries[i] = state.entries[--state.num_entries];
            continue;
        }
        if (entry.layout == image_layout && entry.storage == storage) {
            ++counters.images_reused;
            return entry.slot;
        }
        ++i;
    }

    u32 entry_index = state.num_entries;
    if (entry_index == state.entries.size()) {
        entry_index = 0;
        RetireImageSlot(state.entries[0].slot, state.entries[0].epoch);
    } else {
        ++state.num_entries;
    }
    const u32 slot = AcquireSlot(image_pool);
    WriteImageDescriptor(ImageSlotPtr(slot), state, image_layout, storage);
    state.entries[entry_index] = {slot, epoch, image_layout, storage};
    if (verify) {
        image_shadow[slot] = {static_cast<VkImageView>(handle), image_layout, storage};
    }
    ++counters.images_encoded;
    counters.descriptor_bytes += layout.image_descriptor_size;
    return slot;
}

u32 NativeBindings::SamplerIndex(NativeSamplerState& state, vk::Sampler handle) {
    ASSERT_MSG(state.armed, "NativeBindings: sampler has no retained create info");
    if (state.has_slot && state.epoch == epoch) {
        ++counters.samplers_reused;
        return state.slot;
    }
    const u32 slot = AcquireSlot(sampler_pool);
    WriteSamplerDescriptor(SamplerSlotPtr(slot), state);
    state.slot = slot;
    state.epoch = epoch;
    state.has_slot = true;
    if (verify) {
        sampler_shadow[slot] = static_cast<VkSampler>(handle);
    }
    ++counters.samplers_encoded;
    counters.descriptor_bytes += layout.sampler_descriptor_size;
    return slot;
}

u64 NativeBindings::CommitIndices(std::span<const u32> indices, bool is_compute) {
    ASSERT(indices.size() <= MaxIndicesPerDraw);
    const u32 units = u32((indices.size_bytes() + IndexUnit - 1) / IndexUnit);
    ASSERT_MSG(index_ring.cursor + units <= index_ring.block_units,
               "NativeBindings: index ring block overrun inside a draw");
    const u64 offset =
        (u64(index_ring.current) * index_ring.block_units + index_ring.cursor) * IndexUnit;
    if (!indices.empty()) {
        std::memcpy(index_buffer.mapped + offset, indices.data(), indices.size_bytes());
    }
    index_ring.cursor += units;

    // Both current blocks are referenced by the command buffer being recorded.
    const u64 tick = scheduler.CurrentTick();
    index_ring.ticks[index_ring.current] = tick;
    buffer_ring.ticks[buffer_ring.current] = tick;
    ++counters.indirect_index_bindings;

    counters.index_bytes += indices.size_bytes();
    ++(is_compute ? counters.dispatches : counters.draws);
    if (counters.draws + counters.dispatches >= next_report) {
        ReportCounters();
    }
    return index_buffer.address + offset;
}

void NativeBindings::CommitDirect(bool is_compute) {
    buffer_ring.ticks[buffer_ring.current] = scheduler.CurrentTick();
    ++counters.direct_push_bindings;
    ++(is_compute ? counters.dispatches : counters.draws);
    if (counters.draws + counters.dispatches >= next_report) {
        ReportCounters();
    }
}

void NativeBindings::BindHeapsIfNeeded() {
    if (scheduler.NativeHeapsBound()) {
        return;
    }
    // Both heap binds are valid inside rendering; no render-pass break is needed.
    scheduler.Record([bind_sampler = api.cmd_bind_sampler_heap,
                      bind_resource = api.cmd_bind_resource_heap, sampler_info = sampler_bind_info,
                      resource_info = resource_bind_info](vk::CommandBuffer cmdbuf) {
        bind_sampler(cmdbuf, &sampler_info);
        bind_resource(cmdbuf, &resource_info);
    });
    scheduler.MarkNativeHeapsBound();
    ++counters.heap_binds;
}

void NativeBindings::VerifyFailure(const char* what, u32 binding, u32 element) const {
    LOG_CRITICAL(Render_Vulkan, "NativeBindings verify: {} mismatch at binding {} element {}",
                 what, binding, element);
    UNREACHABLE_MSG("NativeBindings verification failed");
}

void NativeBindings::Verify(std::span<const vk::WriteDescriptorSet> writes,
                            std::span<const u32> indices) {
    std::array<u8, 256> scratch{};
    for (const auto& write : writes) {
        for (u32 e = 0; e < write.descriptorCount; ++e) {
            const u32 binding = write.dstBinding + e;
            if (binding >= indices.size()) {
                VerifyFailure("binding count", binding, e);
            }
            const u32 slot = indices[binding];
            switch (write.descriptorType) {
            case vk::DescriptorType::eStorageBuffer:
            case vk::DescriptorType::eUniformBuffer: {
                const bool storage = write.descriptorType == vk::DescriptorType::eStorageBuffer;
                const auto& info = write.pBufferInfo[e];
                if (!info.buffer) {
                    if (slot != (storage ? NativeNullStorageBufferSlot
                                         : NativeNullUniformBufferSlot)) {
                        VerifyFailure("null buffer", binding, e);
                    }
                    break;
                }
                if (slot < NativeBufferStaticSlots ||
                    slot - NativeBufferStaticSlots >= buffer_keys.size()) {
                    VerifyFailure("buffer slot range", binding, e);
                }
                const u32 ring_slot = slot - NativeBufferStaticSlots;
                const auto& key = buffer_keys[ring_slot];
                const auto& shadow = buffer_shadow[ring_slot];
                // The driver's own address for the legacy handle is the independent oracle.
                const u64 expected =
                    instance.GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo{
                        .buffer = info.buffer,
                    }) +
                    info.offset;
                if (shadow.handle != static_cast<VkBuffer>(info.buffer) ||
                    shadow.offset != info.offset || key.range != info.range ||
                    key.storage != u32(storage) || key.address != expected) {
                    VerifyFailure("buffer payload", binding, e);
                }
                if (layout.buffer_descriptor_size <= scratch.size()) {
                    WriteBufferDescriptor(scratch.data(), key.address, key.range, storage);
                    if (std::memcmp(scratch.data(), BufferSlotPtr(slot),
                                    layout.buffer_descriptor_size) != 0) {
                        VerifyFailure("buffer heap bytes", binding, e);
                    }
                }
                break;
            }
            case vk::DescriptorType::eSampledImage:
            case vk::DescriptorType::eStorageImage: {
                const bool storage = write.descriptorType == vk::DescriptorType::eStorageImage;
                const auto& info = write.pImageInfo[e];
                if (!info.imageView) {
                    if (slot != NullImageIndex(storage)) {
                        VerifyFailure("null image", binding, e);
                    }
                    break;
                }
                if (slot < NativeImageStaticSlots || slot >= image_shadow.size()) {
                    VerifyFailure("image slot range", binding, e);
                }
                const auto& shadow = image_shadow[slot];
                if (shadow.handle != static_cast<VkImageView>(info.imageView) ||
                    shadow.layout != info.imageLayout || shadow.storage != storage) {
                    VerifyFailure("image payload", binding, e);
                }
                break;
            }
            case vk::DescriptorType::eSampler: {
                const auto& info = write.pImageInfo[e];
                if (slot >= sampler_shadow.size() ||
                    sampler_shadow[slot] != static_cast<VkSampler>(info.sampler)) {
                    VerifyFailure("sampler payload", binding, e);
                }
                break;
            }
            default:
                VerifyFailure("descriptor type", binding, e);
            }
            ++counters.verified;
        }
    }
}

void NativeBindings::ReportCounters() {
    next_report += ReportInterval;
    LOG_INFO(Render_Vulkan,
             "NativeBindings: native draws={} dispatches={} legacy=0 | buffers encoded={} "
             "reused={} | images encoded={} reused={} | samplers encoded={} reused={} | "
             "direct_push={} indirect_index={} | ring wraps={} flushes={} waits={} rollovers={} "
             "heap_binds={} | index_bytes={} descriptor_bytes={} verified={}",
             counters.draws, counters.dispatches, counters.buffers_encoded,
             counters.buffers_reused, counters.images_encoded, counters.images_reused,
             counters.samplers_encoded, counters.samplers_reused, counters.direct_push_bindings,
             counters.indirect_index_bindings, counters.ring_wraps, counters.ring_flushes,
             counters.ring_waits, counters.rollovers, counters.heap_binds, counters.index_bytes,
             counters.descriptor_bytes, counters.verified);
}

} // namespace Vulkan
