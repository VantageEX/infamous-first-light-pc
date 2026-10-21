// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class BlitHelper;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;

class Runtime {
public:
    explicit Runtime(const Instance& instance, Scheduler& scheduler);
    ~Runtime() = default;

    const Instance& GetInstance() const {
        return instance;
    }

    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }

    void TickFrame();

    void CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const vk::BufferCopy> copies);

    void FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value);

    void InlineData(VideoCore::Buffer* dst, u64 offset, u32 value);

    /// Writes guest occlusion query results (num_pairs begin/end counter pairs): every counter
    /// valid, the end counters of the first pairs receiving the sample counts of the queries.
    void WriteOcclusionResults(const VideoCore::Buffer* dst, u64 offset, u32 num_pairs,
                               vk::QueryPool pool, std::span<const u32> queries);

    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> subres_range = {});

    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> upload_copies);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> download_copies);

    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                             const VideoCore::Buffer* buffer, u64 offset);
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);

    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);

    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& sub_range);

    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& clear_value);

    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    void AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                      vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access);

    bool IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                          bool check_read_access = false);

    void FlushBarriers();

    /// ISS_UNIFIED_MEMORY: the imported heap's buffer, whose reads are not recorded as ranges
    /// (a write to it is then always ordered after a barrier, see IsBufferAccessed).
    const VideoCore::Buffer* untracked_reads{};

    /// Stage binding memo: changes whenever a recorded buffer write, an image layout or access
    /// transition or a backing replacement could make a remembered binding outcome stale.
    [[nodiscard]] u64 HazardSeq() const noexcept {
        return hazard_seq;
    }
    /// Changes at every barrier flush (the recorded read ranges are forgotten then).
    [[nodiscard]] u64 FlushSeq() const noexcept {
        return flush_seq;
    }

private:
    void AssertCpRecordingLane() const noexcept;
    void MakeCurrent(const VideoCore::Buffer* handle);

private:
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    /// Read ranges are only queried by a later write (Overlaps), far less often than added
    /// (every bound buffer of every draw). ISS_LAZY_READ_RANGES=1: appended unsorted and merged
    /// into the sorted list on the next query; the set and every Overlaps answer are unchanged.
    struct LazyReadList {
        IntervalList<Interval> merged;
        std::vector<Interval> pending;
        static bool Lazy();
        void Add(Interval range) {
            if (Lazy()) {
                if (range.start < range.end) pending.push_back(range);
            } else {
                merged.Add(range);
            }
        }
        bool Overlaps(u64 start, u64 end) {
            Merge();
            return merged.Overlaps(start, end);
        }
        void Clear() {
            merged.Clear();
            pending.clear();
        }
        void Merge();
    };
    struct BufferBarriers {
        const VideoCore::Buffer* handle;
        using AccessList = IntervalList<Interval>;
        LazyReadList read_ranges;
        AccessList write_ranges;
    };
    BufferBarriers* resource{};
    std::vector<BufferBarriers> resources;
    /// MakeCurrent: index + 1 of the entry last used for a handle hashing to the slot (checked
    /// against the entry's handle, so any later reuse of the entry only costs a miss).
    std::array<u32, 64> resource_slots{};
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
    u64 hazard_seq{1};
    u64 flush_seq{1};
};

} // namespace Vulkan
