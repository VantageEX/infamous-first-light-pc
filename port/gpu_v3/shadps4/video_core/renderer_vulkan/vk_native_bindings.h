// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_native_bindings_api.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance;
class Scheduler;

/// Persistent heap descriptors owned by one VideoCore::ImageView. The create-info chain is
/// kept because the heap encodes views from their creation parameters, not from a handle.
struct NativeImageViewState {
    struct Entry {
        u32 slot;
        u32 epoch;
        vk::ImageLayout layout;
        bool storage;
    };

    NativeImageViewState() = default;
    ~NativeImageViewState();
    NativeImageViewState(const NativeImageViewState&) = delete;
    NativeImageViewState& operator=(const NativeImageViewState&) = delete;
    NativeImageViewState(NativeImageViewState&& other) noexcept;
    NativeImageViewState& operator=(NativeImageViewState&& other) noexcept;

    void Release() noexcept;

    vk::ImageViewCreateInfo view_ci{};
    vk::ImageViewUsageCreateInfo usage_ci{};
    vk::ImageViewMinLodCreateInfoEXT min_lod_ci{};
    bool has_min_lod{};
    bool armed{};
    u32 num_entries{};
    std::array<Entry, 4> entries{};
};

/// Persistent heap descriptor owned by one VideoCore::Sampler.
struct NativeSamplerState {
    NativeSamplerState() = default;
    ~NativeSamplerState();
    NativeSamplerState(const NativeSamplerState&) = delete;
    NativeSamplerState& operator=(const NativeSamplerState&) = delete;
    NativeSamplerState(NativeSamplerState&& other) noexcept;
    NativeSamplerState& operator=(NativeSamplerState&& other) noexcept;

    void Release() noexcept;

    vk::SamplerCreateInfo sampler_ci{};
    vk::SamplerCustomBorderColorCreateInfoEXT custom_color{};
    bool has_custom_color{};
    bool border_registered{};
    bool armed{};
    bool has_slot{};
    u32 border_index{};
    u32 slot{};
    u32 epoch{};
};

/// Opt-in VK_EXT_descriptor_heap backend for guest graphics and compute bindings.
/// All methods except the Retire* entry points run on the command-processor thread.
class NativeBindings {
public:
    /// Returns null, after deactivating the mode on the instance, when the heaps cannot be
    /// set up. Must run before any guest pipeline, buffer, image view or sampler is created.
    static std::unique_ptr<NativeBindings> Create(const Instance& instance, Scheduler& scheduler);
    static NativeBindings* Get() noexcept;

    ~NativeBindings();

    NativeBindings(const NativeBindings&) = delete;
    NativeBindings& operator=(const NativeBindings&) = delete;

    /// The only place that may flush or wait. Call between draws, never while a draw's
    /// bindings are partially assembled.
    void BeginDraw();

    /// Encodes or reuses a bounded buffer descriptor for exactly the resolved legacy payload.
    u32 BufferIndex(vk::Buffer handle, u64 device_address, u64 offset, u64 range, bool storage);
    u32 ImageIndex(NativeImageViewState& state, vk::ImageView handle, vk::ImageLayout layout,
                   bool storage);
    static u32 NullImageIndex(bool storage) {
        return storage ? NativeNullStorageImageSlot : NativeNullSampledImageSlot;
    }
    u32 SamplerIndex(NativeSamplerState& state, vk::Sampler handle);

    /// Registers a custom border colour for a sampler being created. Returns false when the
    /// driver refuses, in which case the caller falls back exactly as the legacy path does.
    bool RegisterBorderColor(const vk::SamplerCustomBorderColorCreateInfoEXT& color, u32& index);

    /// Copies the draw's index array into GPU-visible memory owned until GPU completion and
    /// returns its device address.
    u64 CommitIndices(std::span<const u32> indices, bool is_compute);

    /// Marks a direct push-index draw without consuming the GPU index ring. Buffer descriptors
    /// still belong to the current submission block and must stamp its completion tick.
    void CommitDirect(bool is_compute);

    /// Records the heap binds when the scheduler reports them invalid.
    void BindHeapsIfNeeded();

    /// Compares legacy writes built from the same resolved snapshot with the native slots.
    void Verify(std::span<const vk::WriteDescriptorSet> writes, std::span<const u32> indices);

    bool IsVerifying() const {
        return verify;
    }

    /// Thread-safe, tick-deferred release of object-owned slots.
    void RetireImageSlot(u32 slot, u32 epoch) noexcept;
    void RetireSamplerSlot(u32 slot, u32 epoch, bool has_slot, bool border_registered,
                           u32 border_index) noexcept;

private:
    NativeBindings(const Instance& instance, Scheduler& scheduler);
    bool Initialize();

    struct OwnedBuffer {
        vk::Buffer buffer{};
        VmaAllocation allocation{};
        u8* mapped{};
        u64 address{};
        u64 size{};
    };
    bool CreateBuffer(OwnedBuffer& out, u64 size, VkBufferUsageFlags usage, const char* name);
    void DestroyBuffer(OwnedBuffer& buffer);

    /// Ring of fixed blocks; a block is reused only after its completion tick is free.
    struct BlockRing {
        u32 block_units{};
        u32 current{};
        u32 cursor{};
        std::vector<u64> ticks;
    };
    void EnsureRingSpace(BlockRing& ring, u32 units);

    /// Persistent slots with tick-deferred recycling and conservative epoch rollover.
    struct SlotPool {
        u32 first{};
        u32 capacity{};
        u32 next_unused{};
        std::vector<u32> free_list;
        struct Retired {
            u32 slot;
            u64 tick;
        };
        std::deque<Retired> retired;
    };
    u32 AvailableSlots(const SlotPool& pool) const {
        return u32(pool.free_list.size()) + (pool.capacity - pool.next_unused);
    }
    u32 AcquireSlot(SlotPool& pool);
    void RecycleRetired();
    void Rollover();
    void WriteNullDescriptors();

    void WriteBufferDescriptor(u8* dst, u64 address, u64 range, bool storage);
    void WriteImageDescriptor(u8* dst, const NativeImageViewState& state, vk::ImageLayout layout,
                              bool storage);
    void WriteSamplerDescriptor(u8* dst, const NativeSamplerState& state);

    u8* BufferSlotPtr(u32 slot) const {
        return resource_heap.mapped + layout.buffer_base + u64(slot) * layout.buffer_stride;
    }
    u8* ImageSlotPtr(u32 slot) const {
        return resource_heap.mapped + layout.image_base + u64(slot) * layout.image_stride;
    }
    u8* SamplerSlotPtr(u32 slot) const {
        return sampler_heap.mapped + layout.sampler_base + u64(slot) * layout.sampler_stride;
    }

    void ReportCounters();
    [[noreturn]] void VerifyFailure(const char* what, u32 binding, u32 element) const;

private:
    struct BufferKey {
        u64 address;
        u64 range;
        u32 storage;
        bool operator==(const BufferKey&) const = default;
    };
    struct BufferShadow {
        VkBuffer handle;
        u64 offset;
    };
    struct ImageShadow {
        VkImageView handle;
        vk::ImageLayout layout;
        bool storage;
    };

    static constexpr u32 BufferCacheSize = 8192;
    static constexpr u32 IndexBlockBytes = 256 * 1024;
    static constexpr u32 IndexBlocks = 16;
    static constexpr u32 IndexUnit = 16;
    static constexpr u32 MaxIndicesPerDraw = NativeMaxBindingsPerDraw;
    static constexpr u32 ReservedPersistentSlots = 128;
    static constexpr u64 ReportInterval = 200000;

    const Instance& instance;
    Scheduler& scheduler;
    const NativeApi::Functions& api;
    const NativeBindingLayout& layout;
    bool verify{};

    OwnedBuffer resource_heap;
    OwnedBuffer sampler_heap;
    OwnedBuffer index_buffer;
    NativeApi::BindHeapInfo resource_bind_info{};
    NativeApi::BindHeapInfo sampler_bind_info{};

    BlockRing buffer_ring;
    BlockRing index_ring;
    std::vector<BufferKey> buffer_keys;
    std::array<u32, BufferCacheSize> buffer_cache{};

    SlotPool image_pool;
    SlotPool sampler_pool;
    u32 epoch{1};

    std::mutex retire_mutex;
    std::atomic<bool> retire_pending{false};
    struct RetiredBorder {
        u32 index;
        u64 tick;
    };
    std::deque<RetiredBorder> retired_borders;

    std::vector<BufferShadow> buffer_shadow;
    std::vector<ImageShadow> image_shadow;
    std::vector<VkSampler> sampler_shadow;

    struct Counters {
        u64 draws{};
        u64 dispatches{};
        u64 buffers_encoded{};
        u64 buffers_reused{};
        u64 images_encoded{};
        u64 images_reused{};
        u64 samplers_encoded{};
        u64 samplers_reused{};
        u64 ring_wraps{};
        u64 ring_flushes{};
        u64 ring_waits{};
        u64 rollovers{};
        u64 direct_push_bindings{};
        u64 indirect_index_bindings{};
        u64 heap_binds{};
        u64 index_bytes{};
        u64 descriptor_bytes{};
        u64 verified{};
    } counters;
    u64 next_report{};
};

} // namespace Vulkan
