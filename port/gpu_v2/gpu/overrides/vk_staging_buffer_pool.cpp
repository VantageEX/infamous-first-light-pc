// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <mutex>

#include "common/alignment.h"
#include "common/perf_stats.h"
#include "common/assert.h"
#include "common/types.h"
#include "core/devtools/widget/gpu_memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace Vulkan {

using VideoCore::MemoryType;

namespace {

constexpr u64 RING_IDLE_FRAMES = 300;
constexpr u64 LARGE_IDLE_FRAMES = 120;
constexpr u64 LARGE_MIN_GRANULARITY = 64_KB;
constexpr u64 BLOCK_SIZE = 16_MB;

u64 RoundAllocationSize(u64 size) {
    const u64 step = std::max(std::bit_floor(size) / 4, LARGE_MIN_GRANULARITY);
    return Common::AlignUp(size, step);
}

// ISS_STAGING_CLASSES (default on, =0 off): dedicated staging buffers in power of two sizes, a
// request taking any free one up to twice its size, kept LARGE_KEEP_FRAMES once idle within
// LARGE_KEEP_BYTES per memory type. The streaming uploads (16-256 MB, every size) found no
// cached buffer of their size and created one each time (vmaCreateBuffer of fresh dedicated
// memory: 9% of the command processor while running through the city, tex-chasing-run2
// 2026-10-18), the idle ones being freed again after 2 s.
static bool StagingClasses() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_STAGING_CLASSES");
        return !value || value[0] != '0';
    }();
    return enabled;
}
[[maybe_unused]] constexpr u64 LARGE_KEEP_FRAMES = 3600;
// ISS_STAGING_LARGE_KEEP_MB (default 2048): idle large buffers kept per memory type. At 1 GB the
// city went over it (1062 MB), trimmed 8 buffers and allocated again: 32 dedicated allocations,
// 150 ms of the command processor in a 140 s run (cpprof3 2026-10-18).
static u64 LargeKeepBytes() {
    static const u64 bytes = [] {
        const char* value = std::getenv("ISS_STAGING_LARGE_KEEP_MB");
        return (value ? u64(std::strtoull(value, nullptr, 10)) : u64(2048)) << 20;
    }();
    return bytes;
}
static std::atomic<u64> g_large_hits{}, g_large_allocs{}, g_large_alloc_bytes{}, g_large_alloc_ns{},
    g_large_trimmed{};
static void NoteLargeAlloc() {
    const u64 n = g_large_allocs.load(std::memory_order_relaxed);
    if (n == 0 || (n & 15) != 0) {
        return;
    }
    std::fprintf(stderr, "STAGING_LARGE hits=%llu allocs=%llu alloc_mb=%.0f alloc_ms=%.1f trimmed=%llu classes=%d\n",
                 (unsigned long long)g_large_hits.load(), (unsigned long long)n,
                 double(g_large_alloc_bytes.load()) / 1048576.0,
                 double(g_large_alloc_ns.load()) / 1e6, (unsigned long long)g_large_trimmed.load(),
                 int(StagingClasses()));
}

// ISS_STAGING_RING_KEEP (default 8, 0 = the old trimming): the blocks of a ring kept however long
// they stay idle, and the others kept RING_KEEP_IDLE_FRAMES. A streaming burst needs a few extra
// 16 MB blocks; freed after 4 s idle, the next burst created them again (dedicated vmaCreateBuffer
// on the command processor: 9% of its time inside the city's 40-70 ms hitches, cpprof2
// 2026-10-18).
static size_t RingKeepBlocks() {
    static const size_t keep = [] {
        const char* value = std::getenv("ISS_STAGING_RING_KEEP");
        return value ? size_t(std::strtoul(value, nullptr, 10)) : size_t(8);
    }();
    return keep;
}
constexpr u64 RING_KEEP_IDLE_FRAMES = 4500;

} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(u64 size, MemoryType type, u64 alignment, bool deferred,
                                            bool unsynchronized) {
    Ring& ring = rings[u32(type)];
    if (deferred || size > BLOCK_SIZE) {
        return RequestLarge(size, type, deferred, unsynchronized);
    }
    return RequestFromRing(ring, size, alignment, type, unsynchronized);
}

void StagingBufferPool::PrewarmLarge(u64 size, MemoryType type, u32 count) {
    if (size <= BLOCK_SIZE || count == 0) return;
    std::scoped_lock lk{large_mutex};
    auto& cache = large_caches[u32(type)];
    const u64 allocation_size = RoundAllocationSize(size);
    for (u32 i = 0; i < count; ++i) {
        cache.emplace_back(LargeBuffer{
            .buffer = std::make_unique<VideoCore::Buffer>(
                instance, 0, allocation_size, type, "StagingBufferPool:Prewarmed"),
            .tick = 0,
            .last_used_frame = frame,
            .held = false,
            .prewarmed = true,
        });
    }
}

StagingBufferRef StagingBufferPool::RequestFromRing(Ring& ring, u64 size, u64 alignment,
                                                    MemoryType type, bool unsynchronized) {
    Block* best{};
    u64 best_offset{};
    for (size_t i = 0; i < ring.blocks.size(); ++i) {
        const size_t index = (ring.current + i) % ring.blocks.size();
        Block* block = &ring.blocks[index];
        if (const auto offset = block->buffer->Reserve(size, alignment, false)) {
            ring.current = index;
            best = block;
            best_offset = *offset;
            break;
        }
    }

    if (!best) {
        if (!ring.blocks.empty() && unsynchronized) {
            best = &ring.blocks[(ring.current + 1) % ring.blocks.size()];
            best_offset = 0;
        } else {
            // A ring's first use also creates its spare blocks (RingKeepBlocks): the streaming
            // bursts find them instead of allocating on the command processor mid-frame.
            const size_t spares = ring.blocks.empty() ? std::min<size_t>(RingKeepBlocks(), 4) : 0;
            for (size_t i = 0; i < spares; ++i)
                ring.blocks.emplace_back(Block{
                    .buffer = std::make_unique<VideoCore::StreamBuffer>(instance, scheduler, type,
                                                                        BLOCK_SIZE),
                    .last_used_frame = frame,
                });
            best = &ring.blocks.emplace_back(Block{
                .buffer = std::make_unique<VideoCore::StreamBuffer>(instance, scheduler, type,
                                                                    BLOCK_SIZE),
                .last_used_frame = frame,
            });
            static std::atomic<u32> created{0};
            const u32 n = created.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n <= 64)
                std::fprintf(stderr, "STAGING_RING type=%u blocks=%zu frame=%llu (created %u)\n",
                             u32(type), ring.blocks.size(), (unsigned long long)frame, n);
            ring.current = ring.blocks.size() - 1;
            best_offset = *best->buffer->Reserve(size, alignment, false);
        }
    }

    auto& buffer = *best->buffer;
    best->last_used_frame = frame;

    StagingBufferRef ref{
        .buffer = &buffer,
        .offset = best_offset,
        .size = size,
        .type = type,
    };
    if (!buffer.mapped_data.empty()) {
        ref.mapped = buffer.mapped_data.data() + best_offset;
    }
    return ref;
}

StagingBufferRef StagingBufferPool::RequestLarge(u64 size, MemoryType type, bool deferred,
                                                 bool unsynchronized) {
    std::scoped_lock lk{large_mutex};
    auto& cache = large_caches[u32(type)];
    const bool classes = StagingClasses();
    const u64 max_size = classes ? std::max<u64>(2 * size, std::bit_ceil(size))
                                 : std::max(size + size / 4, RoundAllocationSize(size));

    LargeBuffer* best{};
    for (LargeBuffer& entry : cache) {
        const u64 entry_size = entry.buffer->SizeBytes();
        if (entry.held || entry_size < size || entry_size > max_size ||
            (entry.tick != 0 && !scheduler.IsFree(entry.tick) && !unsynchronized)) {
            continue;
        }
        if (!best || entry_size < best->buffer->SizeBytes()) {
            best = &entry;
        }
    }

    if (!best) {
        const u64 allocation_size = classes ? std::max<u64>(std::bit_ceil(size), LARGE_MIN_GRANULARITY)
                                            : RoundAllocationSize(size);
        const u64 alloc_start = Common::Perf::NowNs();
        best = &cache.emplace_back(LargeBuffer{
            .buffer = std::make_unique<VideoCore::Buffer>(instance, 0, allocation_size, type,
                                                          "StagingBufferPool:Dedicated"),
            .tick = 0,
            .last_used_frame = frame,
            .held = false,
            .prewarmed = false,
        });
        g_large_alloc_ns.fetch_add(Common::Perf::NowNs() - alloc_start, std::memory_order_relaxed);
        g_large_alloc_bytes.fetch_add(allocation_size, std::memory_order_relaxed);
        g_large_allocs.fetch_add(1, std::memory_order_relaxed);
        NoteLargeAlloc();
    } else {
        g_large_hits.fetch_add(1, std::memory_order_relaxed);
    }

    best->tick = scheduler.CurrentTick();
    best->last_used_frame = frame;
    best->held = deferred;
    best->prewarmed = false;

    VideoCore::Buffer& buffer = *best->buffer;
    StagingBufferRef ref{
        .buffer = &buffer,
        .offset = 0,
        .size = size,
        .type = type,
    };
    if (!buffer.mapped_data.empty()) {
        ref.mapped = buffer.mapped_data.data();
    }
    return ref;
}

void StagingBufferPool::FreeDeferred(const StagingBufferRef& ref) {
    // Also called from the scheduler's priority thread (asynchronous image downloads).
    std::scoped_lock lk{large_mutex};
    auto& cache = large_caches[u32(ref.type)];
    const auto it = std::ranges::find_if(
        cache, [&](const LargeBuffer& entry) { return entry.buffer.get() == ref.buffer; });
    ASSERT_MSG(it != cache.end() && it->held, "FreeDeferred on an allocation that is not held");
    it->held = false;
    it->tick = scheduler.CurrentTick();
    it->last_used_frame = frame;
}

void StagingBufferPool::TickFrame() {
    ++frame;
    for (Ring& ring : rings) {
        TrimRing(ring);
    }
    for (auto& cache : large_caches) {
        TrimLarge(cache);
    }
    PublishStats();
}


void StagingBufferPool::TrimRing(Ring& ring) {
    auto& blocks = ring.blocks;
    const size_t keep = RingKeepBlocks();
    const u64 idle_frames = keep ? RING_KEEP_IDLE_FRAMES : RING_IDLE_FRAMES;
    for (size_t i = blocks.size(); i-- > 0;) {
        if (blocks.size() <= keep) break;
        const Block& block = blocks[i];
        if (frame - block.last_used_frame < idle_frames ||
            !scheduler.IsFree(block.buffer->LastTick())) {
            continue;
        }
        blocks.erase(blocks.begin() + i);
        if (i < ring.current) {
            --ring.current;
        } else if (i == ring.current) {
            ring.current = 0;
        }
    }
    if (ring.current >= blocks.size()) {
        ring.current = 0;
    }
}

void StagingBufferPool::TrimLarge(std::vector<LargeBuffer>& cache) {
    std::scoped_lock lk{large_mutex};
    if (!StagingClasses()) {
        std::erase_if(cache, [this](const LargeBuffer& entry) {
            return !entry.held && !entry.prewarmed &&
                   frame - entry.last_used_frame >= LARGE_IDLE_FRAMES &&
                   scheduler.IsFree(entry.tick);
        });
        return;
    }
    // Idle buffers stay for a while (the next streaming burst reuses them); beyond the budget
    // the ones idle longest go first.
    u64 bytes = 0;
    for (const LargeBuffer& entry : cache) {
        bytes += entry.buffer->SizeBytes();
    }
    const auto removable = [this](const LargeBuffer& entry) {
        return !entry.held && !entry.prewarmed && scheduler.IsFree(entry.tick) &&
               frame - entry.last_used_frame >= LARGE_IDLE_FRAMES;
    };
    while (true) {
        LargeBuffer* oldest = nullptr;
        for (LargeBuffer& entry : cache) {
            if (removable(entry) && (!oldest || entry.last_used_frame < oldest->last_used_frame)) {
                oldest = &entry;
            }
        }
        // Within the budget nothing is freed: freeing a large dedicated buffer costs the command
        // processor milliseconds mid-frame (7% of its time inside the remaining city hitches,
        // jobfix1 2026-10-18), and the next streaming burst would allocate it again.
        if (!oldest || bytes <= LargeKeepBytes()) {
            break;
        }
        bytes -= oldest->buffer->SizeBytes();
        cache.erase(cache.begin() + (oldest - cache.data()));
        g_large_trimmed.fetch_add(1, std::memory_order_relaxed);
    }
}

void StagingBufferPool::PublishStats() const {
    using namespace Core::Devtools::Widget;
    if (!GpuMemoryViewer::IsEnabled()) {
        return;
    }
    static constexpr std::array<const char*, NUM_TYPES> names = {"DeviceLocal", "HostUncached",
                                                                 "HostCached"};
    GpuMemoryGroup group{.name = "Staging"};
    for (size_t i = 0; i < NUM_TYPES; ++i) {
        u64 bytes = 0;
        for (const Block& block : rings[i].blocks) {
            bytes += block.buffer->SizeBytes();
        }
        std::scoped_lock lk{large_mutex};
        for (const LargeBuffer& entry : large_caches[i]) {
            bytes += entry.buffer->SizeBytes();
        }
        if (bytes != 0) {
            group.rows.push_back({.name = names[i], .bytes = bytes});
        }
    }
    GpuMemoryViewer::Publish(std::move(group));
}

} // namespace Vulkan
