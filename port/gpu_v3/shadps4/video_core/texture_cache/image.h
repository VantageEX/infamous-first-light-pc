// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/texture_cache/tex_table.h"
#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/types.h"
#include "video_core/amdgpu/mtb_conflict.h"
#include "video_core/buffer_cache/dirty_epoch.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

#include <deque>
#include <optional>
#include <source_location>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Registered = 1 << 6,  ///< True when the image is registered
    Picked = 1 << 7,      ///< Temporary flag to mark the image as picked
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

struct UniqueImage {
    explicit UniqueImage() = default;
    explicit UniqueImage(vk::Device device, VmaAllocator allocator)
        : device{device}, allocator{allocator} {}
    ~UniqueImage();

    UniqueImage(const UniqueImage&) = delete;
    UniqueImage& operator=(const UniqueImage&) = delete;

    UniqueImage(UniqueImage&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{std::move(other.image_ci)} {}
    UniqueImage& operator=(UniqueImage&& other) {
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image_ci = std::move(other.image_ci);
        return *this;
    }

    void Create(const vk::ImageCreateInfo& image_ci);

    void Destroy();

    operator vk::Image() const {
        return image;
    }

    operator bool() const {
        return image;
    }

public:
    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
};

class BlitHelper;

struct Image {
    Image(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler, BlitHelper& blit_helper,
          Common::SlotVector<ImageView, Common::StableSlotCapacity>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        const auto image_addr = info.guest_address;
        const auto image_end = info.guest_address + info.guest_size;
        return image_addr < overlap_end && overlap_cpu_addr < image_end;
    }

    vk::Image GetImage() const {
        return backing->image.image;
    }

    bool IsTracked() {
        return track_addr != 0 && track_addr_end != 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & (ImageFlagBits::Dirty));
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        const bool changed = depth_id != depth_image_id || depth_uid != depth_image_uid;
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
        if (changed) {
            VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
            // SHADPS4_DRAW_REPEAT: a full bind would now redirect this image (review 2026-10-02).
            g_image_epoch.fetch_add(1, std::memory_order_release);
        }
    }

    void DisassociateDepth() {
        const bool changed = static_cast<bool>(depth_id) || depth_uid != 0;
        depth_id = {};
        depth_uid = {};
        if (changed) {
            g_image_epoch.fetch_add(1, std::memory_order_release);
        }
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);

    using Barriers = boost::container::small_vector<vk::ImageMemoryBarrier2, 32>;
    Barriers GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                         vk::PipelineStageFlags2 dst_stage,
                         std::optional<SubresourceRange> subres_range, bool raw_layout = false);
    void Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                 std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf = {},
                 std::source_location loc = std::source_location::current());
    void Upload(std::span<const vk::BufferImageCopy> upload_copies, vk::Buffer buffer, u64 offset,
                u64 barrier_size = 0);
    void Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                  u64 offset, u64 download_size);

    void CopyImage(Image& src_image);
    void CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset);
    void CopyMip(Image& src_image, u32 mip, u32 slice);

    void Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                 const VideoCore::SubresourceRange& mrt1_range);
    void Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range);

    void SetBackingSamples(u32 num_samples, bool copy_backing = true);
    /// SHADPS4_WATER_REFLECTION_SCALE (video_core/water_scale.h): switches to a backing of the
    /// same sample count at `scale` x the guest width / height (contents not carried over: the
    /// game rewrites these targets every frame). Single-level, single-layer images only.
    void SetBackingScale(u32 scale);
    /// The current backing's scale (1 = the guest extent).
    [[nodiscard]] u32 BackingScale() const noexcept {
        return backing ? backing->scale : 1u;
    }

public:
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    BlitHelper* blit_helper;
    Common::SlotVector<ImageView, Common::StableSlotCapacity>* slot_image_views;
    ImageInfo info;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    ImageFlagBits flags = ImageFlagBits::Dirty;
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;
    ImageId depth_id{};
    u64 depth_uid{};
    // SHADPS4_STENCIL_PLANE_MERGE: written as a storage image since its last merge into a depth.
    bool stencil_plane_written{};

    // Resource state tracking
    vk::ImageUsageFlags usage_flags;
    vk::FormatFeatureFlags2 format_features;
    struct State {
        vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
        vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;

        bool operator==(const State&) const = default;
    };
    // Piecewise-constant map of subresource index (mip * layers + layer) -> State, kept as a
    // coalesced list of contiguous [begin, end) runs. Empty == "not promoted": the whole image
    // shares `state`. Replaces a flat one-entry-per-subresource vector; see image.cpp GetBarriers.
    struct StateRun {
        u32 begin;
        u32 end;
        State state;
    };
    struct BackingImage {
        UniqueImage image;
        State state;
        boost::container::small_vector<StateRun, 8> subresource_runs;
        boost::container::small_vector<ImageViewInfo, 4> image_view_infos;
        boost::container::small_vector<ImageViewId, 4> image_view_ids;
        u32 num_samples;
        u32 scale{1}; ///< SHADPS4_WATER_REFLECTION_SCALE: extent = scale x the guest extent
        /// Bumped by GetBarriers whenever it emits a barrier or changes `state` or the runs
        /// (SHADPS4_TEX_TRANSIT_SKIP: an unchanged serial proves a repeated Transit is a no-op).
        u64 state_serial{1};
        /// SHADPS4_MTB_CONFLICT (diagnostic): batch of the last state change / draw touch.
        AmdGpu::MtbConflict::Stamp cf{};
    };
    std::deque<BackingImage> backing_images;
    BackingImage* backing{};
    boost::container::static_vector<u64, 16> mip_hashes{};
    u64 image_uid{};
    /// SHADPS4_TEX_MEMO_LOCAL_GEN: bumped whenever an image sharing a page with this one is
    /// registered or unregistered (RegisterImage / UnregisterImage). Relaxed atomic_ref access.
    u64 overlap_epoch{};
    /// q8: the texture-table generation this image was inserted at (0: never; atomic_ref).
    u64 tab_gen{};
    u32 tab_id{}; ///< q10: its ImageId index, stored with tab_gen at insert (per-image kills)
    u64 lru_id{};
    u64 tick_accessed_last{};
    /// The GC tick of the last use (TextureCache::TouchImage, atomic_ref store). Kept here, not
    /// by moving the LRU list item: draws touch images from several recording threads while the
    /// back registers / frees them, and the unlocked list moves corrupted the list (10-20
    /// live-resize hang s1513 / double unregister s1514-s1515).
    u64 gc_touch{};
    u64 hash{};
    /// SHADPS4_WATER_REFLECTION_SCALE: 1 = the reflection's own targets (scaled for good),
    /// 2 = a scratch target scaled only while the game's blur fs_0x6d29fb0f renders into it.
    u8 water_scale_mode{};

    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};

    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
};

} // namespace VideoCore
