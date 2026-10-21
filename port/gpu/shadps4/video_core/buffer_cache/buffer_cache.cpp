#include "tlg_perf_trace.h"
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstring>
#include <chrono>
#include <array>
#include <map>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <magic_enum/magic_enum.hpp>
#include "tlg_copy.h"
#include "tlg_toggles.h"
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/unified_memory.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
// Memory is made resident in aligned granules of this size. Sparse binds get slower the more is
// already bound on some drivers, so it pays to bind a little ahead in few larger operations
// rather than a block at a time as the game streams data in.
static constexpr u64 RESIDENCY_GRANULE_SIZE = 16_MB;
static constexpr u64 RESIDENCY_CHUNK_SIZE = 256_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    // tlg: TLG_MAX_BUFFER_MB lowers the driver's buffer size limit as the arenas see it
    // (diagnostics: reproduces AMD on Windows 2048, RADV 4095 on any GPU).
    u64 max_buffer = instance.GetMaxBufferSize();
    if (const char* forced = std::getenv("TLG_MAX_BUFFER_MB")) {
        if (const u64 mb = std::strtoull(forced, nullptr, 10)) {
            max_buffer = std::min<u64>(max_buffer, mb << 20);
        }
    }
    // tlg: with a limit under two full pages (AMD on Windows 2 GiB, RADV ~4 GiB, Intel), pages
    // are the smallest size and an arena joins as many as the limit allows: one inFAMOUS request
    // of ~2.4 GiB (the cutscene before the graffiti) crossed three 1 GiB pages and stopped the
    // game ("cannot span more than two VA arena pages", iss_pc #18).
    if ((u64{2} << arena_page_bits) > max_buffer) {
        arena_page_bits = MIN_ARENA_PAGE_BITS;
    }
    arena_page_size = u64{1} << arena_page_bits;
    max_arena_pages = std::clamp<u64>(max_buffer >> arena_page_bits, 2, 16);
    num_arena_pages = u64{1} << (ADDRESS_SPACE_BITS - arena_page_bits);
    std::printf("GPU: buffer arenas of %llu MiB, up to %llu pages (driver buffer limit %llu MiB)\n",
                static_cast<unsigned long long>(arena_page_size >> 20),
                static_cast<unsigned long long>(max_arena_pages),
                static_cast<unsigned long long>(max_buffer >> 20));
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = arena_page_size,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = arena_page_size / block_size;
    blocks_per_arena_page_shift = arena_page_bits - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * num_arena_pages) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * num_arena_pages);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() {
    if (unified_buffer) {
        unified_buffer.reset();
        instance.GetDevice().freeMemory(unified_memory);
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (IsUnified(device_addr, size)) {
        return; // the GPU reads the guest's bytes: nothing to upload again
    }
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

// tlg: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in the test scene, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (flushed after the copy).
void BufferCache::RunGuestCopy(const TlgCopy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
    auto* buffer = reinterpret_cast<Buffer*>(item.extra);
    buffer->Flush(dst - buffer->mapped_data.data(), item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
void BufferCache::SmallGuestCopy(const TlgCopy::Item& item) {
    if (scheduler.IsRecordingDeferred() && !TlgToggle::Disabled(TlgToggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
        return;
    }
    TlgCopy::QueueCopy(item);
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("TLG_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= TRACKER_BYTES_PER_PAGE || TlgToggle::Disabled(TlgToggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

// tlg: TLG_READBACK_STATS=1: every 5 s, how long CPU read/write faults that need GPU data spend
// waiting for the GPU command thread to reach them (queue) and inside the download (flush).
namespace {
struct ReadbackStats {
    std::atomic<u64> count{0}, queue_ns{0}, flush_ns{0}, finish_ns{0}, bytes{0};
    /// Deferred readbacks: recorded, joined, joined without the GPU command thread (applied on
    /// the faulting thread, TLG_READBACK_LOCAL_JOIN), applied by guest threads, stale (fell back
    /// to the download), guest time from the request to the copy applied, bytes copied.
    std::atomic<u64> deferred{0}, joined{0}, local_joined{0}, applied{0}, stale{0}, deferred_ns{0},
        deferred_bytes{0};
    /// Eager copies: pages copied, used by a fault, gone stale unread, skipped for backoff.
    std::atomic<u64> eager_copied{0}, eager_used{0}, eager_stale{0}, eager_skipped{0};
    /// Waits for a readback copy tick (WaitReadbackTick): the scheduler's own waits do not
    /// count them.
    std::atomic<u64> guest_waits{0}, guest_wait_ns{0};
    /// Lagged reads (TLG_LAGGED_READS): faults served without a wait, copies written back.
    std::atomic<u64> lagged{0}, lagged_recorded{0}, lagged_applied{0}, nowait_writes{0};
    std::atomic<s64> window_start{0};
};
ReadbackStats& Readbacks() {
    static ReadbackStats stats;
    return stats;
}
bool ReadbackStatsEnabled() {
    static const bool enabled = [] { const char* e = std::getenv("TLG_READBACK_STATS"); return e && *e == '1'; }();
    return enabled;
}
s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
// tlg: TLG_READBACK_STATS also lists the 1 MiB regions the deferred readbacks come from (the
// game memory that keeps faulting GPU data back to the CPU), most bytes first.
std::mutex readback_sites_mutex;
std::unordered_map<u64, std::pair<u64, u64>> readback_sites; // region -> {readbacks, bytes}
void NoteReadbackSite(VAddr address, u64 bytes) {
    std::scoped_lock lock{readback_sites_mutex};
    auto& site = readback_sites[address >> 20];
    ++site.first;
    site.second += bytes;
}
void ReportReadbackSites() {
    std::vector<std::pair<u64, std::pair<u64, u64>>> sites;
    {
        std::scoped_lock lock{readback_sites_mutex};
        sites.assign(readback_sites.begin(), readback_sites.end());
        readback_sites.clear();
    }
    std::sort(sites.begin(), sites.end(),
              [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    for (size_t i = 0; i < sites.size() && i < 12; ++i) {
        std::printf("  readback region %#llx: %llu readbacks, %.2f MB\n",
                    static_cast<unsigned long long>(sites[i].first << 20),
                    static_cast<unsigned long long>(sites[i].second.first),
                    sites[i].second.second / 1048576.0);
    }
}
void ReportReadbacks() {
    auto& r = Readbacks();
    const s64 now = NowNs();
    s64 start = r.window_start.load();
    if (!start) {
        r.window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 5'000'000'000 || !r.window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const u64 n = r.count.exchange(0);
    std::printf("Readback stats (5 s): %llu downloads, queue %.1f ms, flush %.1f ms (GPU finish %.1f ms), %.1f MB\n",
                static_cast<unsigned long long>(n), r.queue_ns.exchange(0) / 1e6, r.flush_ns.exchange(0) / 1e6,
                r.finish_ns.exchange(0) / 1e6, r.bytes.exchange(0) / 1048576.0);
    std::printf("  deferred: %llu recorded, %llu joined (%llu without the GPU thread), %llu applied, "
                "%llu stale (downloaded), guest wait %.1f ms, %.1f MB; eager: %llu pages copied, "
                "%llu used, %llu stale unread, %llu skipped (backoff); lagged: %llu reads, "
                "%llu copies recorded, %llu written; no-wait writes: %llu\n",
                static_cast<unsigned long long>(r.deferred.exchange(0)),
                static_cast<unsigned long long>(r.joined.exchange(0)),
                static_cast<unsigned long long>(r.local_joined.exchange(0)),
                static_cast<unsigned long long>(r.applied.exchange(0)),
                static_cast<unsigned long long>(r.stale.exchange(0)), r.deferred_ns.exchange(0) / 1e6,
                r.deferred_bytes.exchange(0) / 1048576.0,
                static_cast<unsigned long long>(r.eager_copied.exchange(0)),
                static_cast<unsigned long long>(r.eager_used.exchange(0)),
                static_cast<unsigned long long>(r.eager_stale.exchange(0)),
                static_cast<unsigned long long>(r.eager_skipped.exchange(0)),
                static_cast<unsigned long long>(r.lagged.exchange(0)),
                static_cast<unsigned long long>(r.lagged_recorded.exchange(0)),
                static_cast<unsigned long long>(r.lagged_applied.exchange(0)),
                static_cast<unsigned long long>(r.nowait_writes.exchange(0)));
    std::printf("  waits for a readback copy: %llu, %.1f ms\n",
                static_cast<unsigned long long>(r.guest_waits.exchange(0)),
                r.guest_wait_ns.exchange(0) / 1e6);
    ReportReadbackSites();
    std::fflush(stdout);
}
} // namespace

// tlg: the waits for a readback copy tick (eager and deferred copies) go straight to the work
// semaphore, so the scheduler's own tick_wait_ns misses them; they are counted here instead.
// Scheduler::Wait also submits when the tick has not been (the copies are always submitted when
// they are recorded, so this is the bounded fallback that keeps the guest from waiting forever).
void BufferCache::WaitReadbackTick(u64 tick) {
    const s64 started = ReadbackStatsEnabled() ? NowNs() : 0;
    // tlg: the semaphore, not scheduler.Wait: this runs on guest threads (faults, write merge),
    // which must never submit (the tick was flushed when the copy was recorded).
    scheduler.GetWorkSemaphore()->Wait(tick);
    if (started) {
        auto& r = Readbacks();
        r.guest_waits.fetch_add(1);
        r.guest_wait_ns.fetch_add(NowNs() - started);
    }
}

std::tuple<const Buffer*, VAddr, VAddr> BufferCache::GetReadbackWindow(VAddr device_addr,
                                                                       u64 size) {
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);

    // GPU-modified ranges come as many small scattered islands,
    // so the download is widened to a window around the request.
    // tlg: TLG_DOWNLOAD_WINDOW_KB sizes it (default 512). Each download waits for the GPU;
    // a game that writes records across megabytes the GPU also updates (inFAMOUS particles)
    // otherwise faults and waits once per window, many times per frame.
    static const u64 WindowSize = [] {
        const char* e = std::getenv("TLG_DOWNLOAD_WINDOW_KB");
        const u64 kb = e && *e ? std::strtoull(e, nullptr, 10) : 512;
        return std::max<u64>(kb, 64) * 1024;
    }();
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end = std::min<VAddr>(
        std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    return {arena, window_start, window_end};
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    TlgPerf::Span trace_read{TlgPerf::Kind::WaitBegin, TlgPerf::Kind::WaitEnd,
                            TlgPerf::Label("buffer_readback"), is_write ? 1ULL : 0ULL};
    const auto trace_operation = trace_read.Id();
    const auto trace_origin = TlgPerf::Enabled() ? TlgPerf::Hash(device_addr) : 0;
    if (trace_operation) {
        TlgPerf::Event e; e.kind=TlgPerf::Kind::ResourceRead; e.operation=trace_operation;
        e.object=trace_origin; e.a=size; e.generation=is_write; TlgPerf::Emit(e);
    }
    const s64 queued_at = ReadbackStatsEnabled() ? NowNs() : 0;
    const auto flush_request = [this, device_addr, size, is_write, queued_at,trace_operation,trace_origin] {
        if (trace_operation) {
            TlgPerf::Event e; e.kind=TlgPerf::Kind::ReadbackApply;e.operation=trace_operation;
            e.object=trace_origin;e.a=size;e.generation=is_write;TlgPerf::Emit(e);
        }
        const s64 started_at = queued_at ? NowNs() : 0;
        const auto [arena, window_start, window_end] = GetReadbackWindow(device_addr, size);
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        if (queued_at) {
            auto& r = Readbacks();
            r.count.fetch_add(1);
            r.queue_ns.fetch_add(started_at - queued_at);
            r.flush_ns.fetch_add(NowNs() - started_at);
            ReportReadbacks();
        }
    };
    if (assume_locks) {
        // A GPU-side thread touched the memory itself: it has to wait for the GPU.
        flush_request();
    } else if (DeferredReadbackEnabled() && !TlgToggle::Disabled(TlgToggle::DeferredReadback)) {
        // tlg: the GPU command thread only records the copy and goes on with the game's
        // commands; this thread waits for the copy, as it would for the GPU on a console.
        std::shared_ptr<DeferredReadback> readback;
        if (LocalReadbackJoin()) {
            // Join an already-recorded copy without sending another GPU command.
            readback = PendingDeferredReadback(device_addr, size);
            if (readback && queued_at) {
                Readbacks().local_joined.fetch_add(1);
            }
        }
        if (!readback) {
            liverpool->SendCommand<true, true>([&] {
                if (trace_operation) {
                    TlgPerf::Event e; e.kind=TlgPerf::Kind::ReadbackApply;e.operation=trace_operation;
                    e.object=trace_origin;e.a=size;e.generation=is_write;TlgPerf::Emit(e);
                }
                readback = StartDeferredReadback(device_addr, size);
                if (!readback && is_write) {
                    // Nothing to copy back, as when DownloadMemory finds nothing.
                    memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                }
            });
        }
        if (readback && !FinishDeferredReadback(*readback, true)) {
            // The GPU (or a CPU write without a fault) changed the memory after the copy was
            // recorded: download it the old way, which first recovers the stale copy's ranges.
            if (queued_at) {
                Readbacks().stale.fetch_add(1);
            }
            liverpool->SendCommand<true, true>(flush_request);
        } else if (readback) {
            if (is_write) {
                // After the data is back; see MarkRegionAsCpuModifiedUnlessGpu.
                memory_tracker->MarkRegionAsCpuModifiedUnlessGpu(device_addr, size);
            }
            if (queued_at) {
                Readbacks().deferred_ns.fetch_add(NowNs() - queued_at);
                Readbacks().applied.fetch_add(1);
                ReportReadbacks();
            }
        }
    } else {
        liverpool->SendCommand<true, true>(flush_request);
    }
}

void BufferCache::ReleaseReadProtection(VAddr device_addr, u64 size) {
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

bool BufferCache::LaggedRead(VAddr device_addr, u64 size) {
    if (!DeferredReadbackEnabled() || TlgToggle::Disabled(TlgToggle::DeferredReadback)) {
        return false;
    }
    const VAddr page = Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE);
    const u64 bytes = Common::AlignUp(device_addr + size, TRACKER_BYTES_PER_PAGE) - page;
    // The guest reads what memory holds now; the copy queued below lands there for the next
    // read. In order with the game's commands, so it holds the GPU work submitted before it.
    memory_tracker->UnmarkRegionAsGpuModified(page, bytes);
    liverpool->SendCommand<false, true>([this, page, bytes] { StartLaggedReadback(page, bytes); });
    if (ReadbackStatsEnabled()) {
        Readbacks().lagged.fetch_add(1);
    }
    return true;
}

bool BufferCache::NoWaitWrite(VAddr device_addr) {
    if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise) {
        return false;
    }
    const VAddr page = Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE);
    // Copies in flight over the page must not land on the bytes about to be written.
    MarkDeferredStale(page, TRACKER_BYTES_PER_PAGE, true);
    MarkEagerStale(page, TRACKER_BYTES_PER_PAGE);
    memory_tracker->MarkRegionAsCpuModifiedNoReadback(page, TRACKER_BYTES_PER_PAGE);
    // The page is the CPU's now: what the GPU wrote there before is dropped, in order with the
    // game's commands (GPU writes recorded after this mark it again). The game reuses these
    // pages frame after frame (a ring), so the old GPU bytes are dead.
    liverpool->SendCommand<false, true>(
        [this, page] { gpu_modified_ranges.Subtract(page, TRACKER_BYTES_PER_PAGE); });
    if (ReadbackStatsEnabled()) {
        Readbacks().nowait_writes.fetch_add(1);
    }
    return true;
}

void BufferCache::StartLaggedReadback(VAddr page, u64 size) {
    PruneDeferredReadbacks();
    {
        // A copy in flight over this page: the next read takes its result instead.
        std::scoped_lock lock{deferred_mutex};
        for (const auto& readback : deferred_readbacks) {
            if (!readback->Done() && readback->start < page + size && page < readback->end) {
                return;
            }
        }
    }
    const auto [arena, window_start, window_end] = GetReadbackWindow(page, size);
    if (eager_count.load(std::memory_order_acquire)) {
        std::scoped_lock lock{eager_mutex};
        ApplyEagerSubtractsLocked();
    }
    FlushMergedWrites(arena, page, size);
    auto readback = std::make_shared<DeferredReadback>();
    readback->lagged = true;
    readback->arena_base = arena->cpu_addr;
    readback->start = page;
    readback->end = page + size;
    // Not CollectDownloads: it only looks at pages the tracker still marks GPU modified, and
    // LaggedRead already unmarked these. gpu_modified_ranges says what the GPU wrote.
    u64 total_size_bytes = 0;
    gpu_modified_ranges.ForEachInRange(page, size, [&](VAddr start, VAddr end) {
        readback->copies.push_back(vk::BufferCopy{
            .srcOffset = start - arena->cpu_addr,
            .dstOffset = total_size_bytes,
            .size = end - start,
        });
        total_size_bytes += Common::AlignUp(end - start, 64);
    });
    gpu_modified_ranges.Subtract(page, size);
    if (total_size_bytes == 0) {
        return;
    }
    readback->staging =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
    for (auto& copy : readback->copies) {
        copy.dstOffset += readback->staging.offset;
    }
    runtime.CopyBuffer(arena, readback->staging.buffer, readback->copies);
    readback->tick = scheduler.CurrentTick();
    scheduler.Flush();
    if (ReadbackStatsEnabled()) {
        Readbacks().lagged_recorded.fetch_add(1);
    }
    std::scoped_lock lock{deferred_mutex};
    deferred_readbacks.push_back(std::move(readback));
    deferred_count.store(static_cast<u32>(deferred_readbacks.size()), std::memory_order_release);
}

bool BufferCache::ApplyLaggedReadback(DeferredReadback& readback, bool wait) {
    if (wait) {
        scheduler.GetWorkSemaphore()->Wait(readback.tick);
    } else if (!scheduler.IsFree(readback.tick)) {
        return false;
    }
    std::scoped_lock lock{readback.mutex};
    if (readback.Done()) {
        return true;
    }
    readback.staging.Invalidate();
    const u8* const mapped = readback.staging.mapped - readback.staging.offset;
    for (const auto& copy : readback.copies) {
        const VAddr copy_start = readback.arena_base + copy.srcOffset;
        const VAddr copy_end = copy_start + copy.size;
        for (VAddr page = Common::AlignDown(copy_start, TRACKER_BYTES_PER_PAGE); page < copy_end;
             page += TRACKER_BYTES_PER_PAGE) {
            // A CPU write since the copy was recorded is newer than it.
            if (memory_tracker->IsRegionCpuModified(page, TRACKER_BYTES_PER_PAGE)) {
                continue;
            }
            const VAddr lo = std::max<VAddr>(page, copy_start);
            const VAddr hi = std::min<VAddr>(page + TRACKER_BYTES_PER_PAGE, copy_end);
            memory->TryWriteBacking(reinterpret_cast<void*>(lo),
                                    mapped + copy.dstOffset + (lo - copy_start), hi - lo);
        }
    }
    readback.applied.store(true, std::memory_order_release);
    if (ReadbackStatsEnabled()) {
        Readbacks().lagged_applied.fetch_add(1);
    }
    return true;
}

u64 BufferCache::CollectDownloads(const Buffer* arena, VAddr device_addr, u64 size,
                                  boost::container::small_vector<vk::BufferCopy, 8>& copies) {
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
    });
    return total_size_bytes;
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    // Copies back in flight over this memory took their ranges out of gpu_modified_ranges:
    // applied, or put back when stale, before those are looked at.
    SettleDeferredReadbacks(device_addr, device_addr + size);
    if (eager_count.load(std::memory_order_acquire)) {
        std::scoped_lock lock{eager_mutex};
        ApplyEagerSubtractsLocked();
    }
    FlushMergedWrites(arena, device_addr, size); // merged CPU bytes precede the download
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const u64 total_size_bytes = CollectDownloads(arena, device_addr, size, copies);
    if (total_size_bytes == 0) {
        return;
    }
    const VAddr arena_base = arena->cpu_addr;
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    const s64 finish_at = ReadbackStatsEnabled() ? NowNs() : 0;
    scheduler.Finish();
    if (finish_at) {
        Readbacks().finish_ns.fetch_add(NowNs() - finish_at);
        Readbacks().bytes.fetch_add(total_size_bytes);
    }

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

namespace {
// tlg: TLG_BUFFER_STATS=1 — how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = TlgStats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("TLG_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace

namespace {
// tlg: TLG_WRITTEN_BINDINGS=1: every 5 s, the buffer ranges bound as written by shaders (the
// whole range is marked GPU modified, whatever the shader stores), most bound first, to tell
// which binding makes the pages CPU write faults wait on (TLG_FAULT_PROFILE) GPU modified.
void NoteWrittenBinding(VAddr addr, u32 size) {
    static const bool enabled = [] { const char* e = std::getenv("TLG_WRITTEN_BINDINGS"); return e && *e == '1'; }();
    if (!enabled) {
        return;
    }
    // TLG_WRITTEN_BINDINGS_RANGE=<lo>-<hi> (hex): only the bindings overlapping that range.
    static const std::pair<VAddr, VAddr> range = [] {
        const char* e = std::getenv("TLG_WRITTEN_BINDINGS_RANGE");
        char* end = nullptr;
        const VAddr lo = e ? std::strtoull(e, &end, 16) : 0;
        const VAddr hi = end && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : ~VAddr{0};
        return std::pair{lo, hi};
    }();
    if (addr + size <= range.first || range.second <= addr) {
        return;
    }
    static std::mutex mutex;
    static std::map<std::tuple<VAddr, u32, const char*, u64>, u64> bindings;
    static auto window = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++bindings[{addr, size, written_binding_source.kind, written_binding_source.hash}];
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5)) {
        return;
    }
    window = now;
    std::vector<std::pair<std::tuple<VAddr, u32, const char*, u64>, u64>> top(bindings.begin(), bindings.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::printf("Written bindings (5 s, %zu ranges; kind, shader):", top.size());
    for (size_t i = 0; i < top.size() && i < 16; ++i) {
        const auto& [b_addr, b_size, b_kind, b_hash] = top[i].first;
        std::printf(" [%#llx+%#x x%llu %s %#llx]", static_cast<unsigned long long>(b_addr), b_size,
                    static_cast<unsigned long long>(top[i].second), b_kind,
                    static_cast<unsigned long long>(b_hash));
    }
    std::printf("\n");
    std::fflush(stdout);
    bindings.clear();
}
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    if (Unified::Enabled()) [[unlikely]] {
        if (unified_end == 0 && !unified_failed) {
            EnableUnified();
        }
        if (IsUnified(device_addr, size)) {
            texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferObtain);
            return ObtainUnified(device_addr, size, is_written);
        }
    }
    if (is_written) {
        NoteGpuWriteForEager(device_addr, size);
        NoteWrittenBinding(device_addr, size);
    }
    texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferObtain);
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // tlg: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !TlgToggle::Disabled(TlgToggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, instance.UniformMinAlignment())) {
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    const u64 uploaded_before = TlgStats::buffer_upload_bytes.load(std::memory_order_relaxed);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            TlgStats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

namespace {
/// tlg: ObtainBufferForImage calls in progress on this thread. The texture cache calls it with
/// its mutex held (UpdateImage, ExpandImage), so no texture can be freed from there.
thread_local u32 image_refresh_depth = 0;
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    ++image_refresh_depth;
    struct DepthGuard {
        ~DepthGuard() {
            --image_refresh_depth;
        }
    } depth_guard;
    texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferForImage);
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    if (!TlgToggle::Disabled(TlgToggle::DeferredUploads)) {
        // tlg: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const TlgCopy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                TlgCopy::Async([item] { item.run(item); });
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (IsUnified(addr, size)) [[unlikely]] {
        return UnifiedTouched(addr, size);
    }
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::EnableUnified() {
    const auto fail = [this](const char* why) {
        unified_failed = true;
        std::printf("Unified memory off: %s\n", why);
        return false;
    };
    if (!instance.HasExternalMemoryHost()) {
        return fail("VK_EXT_external_memory_host not available");
    }
    const auto& range = Unified::GetRange();
    const VAddr base = range.base;
    const u64 size = range.size;
    void* host = memory->BackingView(base, size);
    if (!host) {
        return false; // not mapped yet: the next binding tries again
    }
    if ((base & (block_size - 1)) != 0 || (size & (block_size - 1)) != 0) {
        return fail("the range is not aligned to sparse blocks");
    }
    bool resident = false;
    resident_ranges.ForEachInRange(base >> block_shift, (base + size) >> block_shift,
                                   [&](const Backing&) { resident = true; });
    if (resident) {
        return fail("the arena already backs part of the range");
    }
    const auto device = instance.GetDevice();
    const auto properties =
        instance.GetPhysicalDevice()
            .getProperties2<vk::PhysicalDeviceProperties2,
                            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
    const u64 alignment = std::max<u64>(
        properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
            .minImportedHostPointerAlignment,
        4_KB);
    if ((reinterpret_cast<u64>(host) & (alignment - 1)) != 0 || (size & (alignment - 1)) != 0) {
        return fail("the backing is not aligned for import");
    }
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT};
    const vk::BufferCreateInfo buffer_ci{
        .pNext = &external,
        .size = size,
        .usage = AllFlags,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    vk::Buffer handle{};
    if (device.createBuffer(&buffer_ci, nullptr, &handle) != vk::Result::eSuccess) {
        return fail("createBuffer");
    }
    const auto requirements = device.getBufferMemoryRequirements(handle);
    vk::MemoryHostPointerPropertiesEXT host_properties{};
    if (device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, host, &host_properties) !=
        vk::Result::eSuccess) {
        device.destroyBuffer(handle);
        return fail("getMemoryHostPointerPropertiesEXT");
    }
    const auto& memory_properties = instance.GetMemoryProperties();
    const u32 compatible = host_properties.memoryTypeBits & requirements.memoryTypeBits;
    u32 type = memory_properties.memoryTypeCount;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const auto flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((compatible >> i & 1) && (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
            type = i;
            break;
        }
    }
    if (type == memory_properties.memoryTypeCount) {
        device.destroyBuffer(handle);
        return fail("no host coherent memory type for the import");
    }
    const vk::MemoryAllocateFlagsInfo allocate_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
    const vk::ImportMemoryHostPointerInfoEXT import_info{
        .pNext = &allocate_flags,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        .pHostPointer = host,
    };
    const vk::MemoryAllocateInfo alloc_info{
        .pNext = &import_info,
        .allocationSize = size,
        .memoryTypeIndex = type,
    };
    if (device.allocateMemory(&alloc_info, nullptr, &unified_memory) != vk::Result::eSuccess) {
        device.destroyBuffer(handle);
        return fail("allocateMemory (import)");
    }
    if (device.bindBufferMemory(handle, unified_memory, 0) != vk::Result::eSuccess) {
        device.destroyBuffer(handle);
        device.freeMemory(unified_memory);
        unified_memory = vk::DeviceMemory{};
        return fail("bindBufferMemory");
    }
    // A Buffer around the imported handle: created as a small sparse buffer (no memory), whose
    // handle is replaced. Its destructor destroys the handle only.
    unified_buffer = std::make_unique<Buffer>(instance, base, 64_KB, MemoryType::Sparse,
                                              "Unified guest memory");
    unified_buffer->buffer.Destroy();
    unified_buffer->buffer.buffer = handle;
    unified_buffer->buffer.bda_addr =
        device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = handle});
    unified_buffer->cpu_addr = base;
    unified_buffer->size_bytes = size;
    unified_buffer->mem_type = MemoryType::HostCached;
    unified_buffer->is_coherent = true;
    const u64 num_chunks = size >> UNIFIED_CHUNK_SHIFT;
    unified_chunk_tick = std::make_unique<std::atomic<u64>[]>(num_chunks);

    // Shaders that read guest memory through device addresses: the range's blocks resolve into
    // the imported buffer (EnsureResident never binds them).
    const u64 first_block = base >> block_shift;
    const u64 num_blocks = size >> block_shift;
    const auto staging =
        staging_pool.Request(num_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    auto* entries = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    for (u64 i = 0; i < num_blocks; ++i) {
        entries[i] = unified_buffer->buffer.bda_addr + (i << block_shift);
    }
    staging.Flush();
    const vk::BufferCopy copy{
        .srcOffset = staging.offset,
        .dstOffset = first_block * sizeof(vk::DeviceAddress),
        .size = num_blocks * sizeof(vk::DeviceAddress),
    };
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), std::span{&copy, 1});
    unified_base = base;
    unified_end = base + size;
    std::printf("Unified memory on: guest 0x%llx size 0x%llx host %p memory type %u (bits 0x%x)\n",
                static_cast<unsigned long long>(base), static_cast<unsigned long long>(size),
                host, type, host_properties.memoryTypeBits);
    return true;
}

std::pair<const Buffer*, u64> BufferCache::ObtainUnified(VAddr device_addr, u32 size,
                                                         bool is_written) {
    if (is_written) {
        const u64 tick = scheduler.CurrentTick();
        const u64 first = (device_addr - unified_base) >> UNIFIED_CHUNK_SHIFT;
        const u64 last = (device_addr + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
        for (u64 c = first; c <= last; ++c) {
            unified_chunk_tick[c].store(tick, std::memory_order_release);
        }
        Unified::Add(Unified::Writes);
    } else {
        Unified::Add(Unified::Reads);
    }
    return {unified_buffer.get(), device_addr - unified_base};
}

bool BufferCache::UnifiedTouched(VAddr addr, u64 size) {
    // Only GPU writes not known to be complete: once done, the guest's bytes are current.
    const u64 first = (addr - unified_base) >> UNIFIED_CHUNK_SHIFT;
    const u64 last = (addr + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
    for (u64 c = first; c <= last; ++c) {
        const u64 tick = unified_chunk_tick[c].load(std::memory_order_acquire);
        if (tick != 0 && !scheduler.IsFree(tick)) {
            return true;
        }
    }
    return false;
}

void BufferCache::SyncUnifiedRead(VAddr addr, u64 size) {
    if (!IsUnified(addr, size)) {
        return;
    }
    const u64 first = (addr - unified_base) >> UNIFIED_CHUNK_SHIFT;
    const u64 last = (addr + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
    u64 tick = 0;
    for (u64 c = first; c <= last; ++c) {
        tick = std::max(tick, unified_chunk_tick[c].load(std::memory_order_acquire));
    }
    if (tick == 0 || scheduler.IsFree(tick)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    if (tick >= scheduler.CurrentTick()) {
        // Recorded in the command buffer being built: visible to the host when it completes.
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            const vk::MemoryBarrier2 barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
            });
        });
        scheduler.Flush();
    }
    scheduler.Wait(tick);
    Unified::Add(Unified::CpReadSyncs);
    Unified::Add(Unified::CpReadWaitUs,
                 std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start).count());
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager->ProcessFaultBuffer();
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (arena_page_bits - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    // Arenas cover contiguous runs of pages: when both ends are in the same one, so is the rest.
    if (const Buffer* arena = address_space[first_page]; arena && arena == address_space[last_page]) {
        return arena;
    }
    // tlg: the pages the new arena covers: the request plus every arena it touches (an arena may
    // join several pages, up to the driver's buffer limit).
    u64 lo = first_page, hi = last_page;
    for (u64 page = first_page; page <= last_page; ++page) {
        if (const Buffer* arena = address_space[page]) {
            lo = std::min<u64>(lo, arena->cpu_addr >> arena_page_bits);
            hi = std::max<u64>(hi, ((arena->cpu_addr + arena->size_bytes) >> arena_page_bits) - 1);
        }
    }
    const u64 num_pages = hi - lo + 1;
    // Over the driver's limit: the old code built such arenas too (two merged arenas could
    // exceed it) and drivers accepted them, so warn instead of stopping.
    if (num_pages > max_arena_pages) {
        LOG_WARNING(Render, "Arena of {} pages of {} MiB exceeds the driver buffer limit "
                    "(request {:#x}+{} MiB)", num_pages, arena_page_size >> 20,
                    first_block << block_shift,
                    ((last_block - first_block + 1) << block_shift) >> 20);
    }

    bool any_arena = false;
    for (u64 page = lo; page <= hi; ++page) {
        any_arena |= address_space[page] != nullptr;
    }
    const u64 base_addr = lo << arena_page_bits;
    const u64 total_size = num_pages << arena_page_bits;
    auto* new_arena = &arenas.emplace_back(instance, base_addr, total_size, MemoryType::Sparse);
    if (any_arena) {
        LOG_WARNING(Render, "Migrating arena ({} pages of {} MiB)", num_pages, arena_page_size >> 20);
        const u64 base_block = base_addr >> block_shift;
        const u64 end_block = (base_addr + total_size) >> block_shift;
        auto* bind = BindsForArena(new_arena);
        resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
            const u64 start = std::max(base_block, backing.start);
            const u64 end = std::min(end_block, backing.end);
            bind->binds.push_back(vk::SparseMemoryBind{
                .resourceOffset = (start - base_block) << block_shift,
                .size = (end - start) << block_shift,
                .memory = backing.memory,
                .memoryOffset = (backing.offset + start - backing.start) << block_shift,
            });
        });
    }
    for (u64 page = lo; page <= hi; ++page) {
        address_space[page] = new_arena;
    }
    return new_arena;
}

// tlg: from shadPS4-Infamous-FixBuild 4a206865 and 0c3fada0. Memory used to become resident a
// block at a time, each range its own driver allocation and sparse bind; granules and chunks
// make far fewer. When device memory runs out, old textures are freed and it is tried once more.
void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    if (unified_end != 0) [[unlikely]] {
        // tlg: the imported range is never bound in the arena; the parts around it are.
        const u64 unified_first = unified_base >> block_shift;
        const u64 unified_last = (unified_end >> block_shift) - 1;
        if (first_block <= unified_last && unified_first <= last_block) {
            if (first_block < unified_first) {
                EnsureResident(arena, first_block, unified_first - 1);
            }
            if (last_block > unified_last) {
                EnsureResident(arena, unified_last + 1, last_block);
            }
            return;
        }
    }
    // Granules never cross an arena page (1 GiB or more), so the widened range stays within the
    // arena.
    const u64 granule_blocks = std::max<u64>(RESIDENCY_GRANULE_SIZE >> block_shift, 1);
    const u64 bind_first = Common::AlignDown(first_block, granule_blocks);
    const u64 bind_end = Common::AlignUp(last_block + 1, granule_blocks);

    for (u32 attempt = 0;; ++attempt) {
        u32 resident_blocks{};
        IntervalList bind_ranges;
        resident_ranges.ForEachGap(bind_first, bind_end, [&](u64 start, u64 end) {
            resident_blocks += end - start;
            bind_ranges.Add({start, end});
        });

        if (bind_ranges.Empty()) {
            return;
        }
        TlgStats::Timer timer{TlgStats::t_resident};

        // All memory is allocated before anything is recorded or bound: making room for it
        // submits, which sends the pending binds and runs pending operations (and so can make
        // some of these ranges resident: the gaps are looked up again after it).
        ResidencyAttempt state{.chunk = residency_chunk, .chunk_used = residency_chunk_used};
        boost::container::small_vector<std::pair<vk::DeviceMemory, u64>, 8> memories;
        for (const auto& range : bind_ranges) {
            const auto memory = AllocateResidency((range.end - range.start) << block_shift, state);
            if (!memory) {
                break;
            }
            memories.push_back(*memory);
        }
        if (memories.size() != bind_ranges.Size()) {
            UndoResidency(state);
            const u64 bytes = u64{resident_blocks} << block_shift;
            ASSERT_MSG(attempt == 0,
                       "Out of device memory making {} MB resident, after freeing textures",
                       bytes >> 20);
            ReleaseMemoryForResidency(bytes);
            continue;
        }

        boost::container::small_vector<vk::BufferCopy, 8> copies;
        const auto staging = staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress),
                                                  MemoryType::HostUncached);

        ArenaBinds* binds = BindsForArena(arena);
        auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
        u64 offset = staging.offset;
        size_t index = 0;
        for (const auto& range : bind_ranges) {
            const u64 range_size = (range.end - range.start) << block_shift;
            const auto [device_memory, memory_offset] = memories[index++];

            Backing backing;
            backing.start = range.start;
            backing.end = range.end;
            backing.memory = device_memory;
            // In blocks, as Backing::CanMergeWith and SubRange count them.
            backing.offset = memory_offset >> block_shift;
            resident_ranges.Add(backing);

            LOG_DEBUG(Render, "Making range start={}, end={} resident", backing.start,
                      backing.end);

            const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
                .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
                .size = range_size,
                .memory = device_memory,
                .memoryOffset = memory_offset,
            });

            for (u32 block = 0; block < bind.size; block += block_size) {
                *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
            }
            const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
            copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
            offset += copy_size;
        }

        staging.Flush();
        runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
        return;
    }
}

std::optional<std::pair<vk::DeviceMemory, u64>> BufferCache::AllocateResidency(
    u64 size, ResidencyAttempt& attempt) {
    // Null when the device is out of memory; any other failure ends the game as before.
    const auto allocate = [&](u64 allocation_size) -> vk::DeviceMemory {
        const vk::MemoryAllocateInfo alloc_info = {
            .allocationSize = allocation_size,
            .memoryTypeIndex = arena_memory_type_index,
        };
        auto result = instance.GetDevice().allocateMemory(alloc_info);
        if (result.result == vk::Result::eErrorOutOfDeviceMemory) {
            return {};
        }
        const auto memory = Vulkan::Check(std::move(result));
        attempt.allocated.push_back(memory);
        ++residency_allocations;
        return memory;
    };
    using Allocation = std::pair<vk::DeviceMemory, u64>;
    // Ranges keep coming as the game streams data in. Carving them out of large chunks keeps the
    // driver from allocating memory each time, which is slow and limited to a few thousand
    // allocations per process on Windows. Large ranges get their own.
    if (size > RESIDENCY_CHUNK_SIZE / 4) {
        const auto memory = allocate(size);
        return memory ? std::optional{Allocation{memory, 0}} : std::nullopt;
    }
    if (!residency_chunk || residency_chunk_used + size > RESIDENCY_CHUNK_SIZE) {
        const auto chunk = allocate(RESIDENCY_CHUNK_SIZE);
        if (!chunk) {
            // A whole chunk may not fit when memory is short, while the range itself still does.
            static u32 warned = 0;
            if (warned < 16 || warned % 64 == 0) {
                LOG_WARNING(Render, "No room for a {} MB residency chunk, allocating {} MB instead",
                            RESIDENCY_CHUNK_SIZE >> 20, size >> 20);
            }
            ++warned;
            const auto memory = allocate(size);
            return memory ? std::optional{Allocation{memory, 0}} : std::nullopt;
        }
        residency_chunk = chunk;
        residency_chunk_used = 0;
    }
    const u64 offset = residency_chunk_used;
    residency_chunk_used += size;
    return Allocation{residency_chunk, offset};
}

void BufferCache::UndoResidency(const ResidencyAttempt& attempt) {
    // Nothing was bound to this memory yet: chunks taken since are freed with the rest.
    for (const vk::DeviceMemory memory : attempt.allocated) {
        instance.GetDevice().freeMemory(memory);
    }
    residency_allocations -= attempt.allocated.size();
    residency_chunk = attempt.chunk;
    residency_chunk_used = attempt.chunk_used;
}

void BufferCache::ReleaseMemoryForResidency(u64 bytes) {
    // The device can run out of memory when the game makes gigabytes resident at once, which the
    // texture collector, running once per submission, can't keep up with.
    LOG_WARNING(Render, "Out of device memory making {} MB resident, freeing textures",
                bytes >> 20);
    // The texture binding helper may be looking images up for the draw being prepared on this
    // thread: join it before images go (the same hook the buffer cache uses before it changes
    // image state).
    runtime.BeforeImageAccess();
    if (image_refresh_depth == 0) {
        texture_cache.ReleaseMemoryForAllocation();
    } else {
        // Refreshing an image, under the texture cache's lock: only the deletions already
        // pending can help.
        LOG_WARNING(Render, "Out of device memory while refreshing an image: only waiting for "
                            "the GPU to release memory");
    }
    // The images' memory is freed by pending operations once the GPU is done with them. This
    // thread owns the caches (stage B while draw packets are in flight, stage A otherwise:
    // the scheduler's owner hook drains the draw pipe first), as for DownloadMemory's Finish.
    // Inside a pending operation (the fault buffer's) the nested pop does nothing and the
    // deletions run after it, so the second attempt may still fail there.
    scheduler.Finish();
    scheduler.PopPendingOperations();
}

namespace {
constexpr u64 EagerPageSize = 4096;
constexpr u64 EagerBufferSize = 32ull << 20; // 8192 pages
constexpr u32 EagerIdleFences = 2400;        // ~40 s at 60 fences/s: forgotten
// tlg: a page whose copies keep going stale unread (the GPU writes it again before the game
// reads it) is copied at fewer fences, twice as few each time, up to 1 in 8, until a copy of it
// is read again. In the fork three quarters of such copies were wasted (9a5224bf).
constexpr u8 EagerMaxBackoff = 7;
} // namespace

bool BufferCache::EagerEnabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("TLG_EAGER_READBACK");
        return e && *e == '1' && EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise;
    }();
    return enabled;
}

// tlg: TLG_LAGGED_RANGES=<start>-<end>[,...] (hex guest addresses), precise readback mode: the
// GPU-written pages there the CPU reads back (First Light's particle emitter counters, written
// by the GPU and read by the game every frame) are served from the eager copy of an earlier
// fence instead of waiting for the GPU, so the values are a frame or so late. Without this each
// read waited for the GPU to finish the frame (about 25 FPS); relaxed mode never brings them
// back and part of the dash effects vanish.
bool BufferCache::LaggedPage(VAddr page) {
    static const std::vector<std::pair<VAddr, VAddr>> ranges = [] {
        std::vector<std::pair<VAddr, VAddr>> out;
        if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise) {
            return out;
        }
        const char* env = std::getenv("TLG_LAGGED_RANGES");
        for (const char* p = env; p && *p;) {
            char* end = nullptr;
            const VAddr start = std::strtoull(p, &end, 16);
            if (end == p || *end != '-') {
                break;
            }
            p = end + 1;
            const VAddr stop = std::strtoull(p, &end, 16);
            if (end == p || stop <= start) {
                break;
            }
            out.emplace_back(start, stop);
            p = *end == ',' ? end + 1 : end;
        }
        if (!out.empty()) {
            std::printf("Readbacks: %zu lagged range(s) (TLG_LAGGED_RANGES)\n", out.size());
        }
        return out;
    }();
    for (const auto& [start, stop] : ranges) {
        if (page < stop && start < page + EagerPageSize) {
            return true;
        }
    }
    return false;
}

void BufferCache::NoteReadback(VAddr addr, u64 size) {
    const bool eager = EagerEnabled();
    std::scoped_lock lock{eager_mutex};
    const auto take_slot = [&](u64& offset) {
        if (!eager_free_offsets.empty()) {
            offset = eager_free_offsets.back();
            eager_free_offsets.pop_back();
            return true;
        }
        if (eager_next_offset + EagerPageSize <= EagerBufferSize) {
            offset = eager_next_offset;
            eager_next_offset += EagerPageSize;
            return true;
        }
        return false;
    };
    for (VAddr page = Common::AlignDown(addr, EagerPageSize); page < addr + size; page += EagerPageSize) {
        const bool lagged = LaggedPage(page);
        if (!eager && !lagged) {
            continue;
        }
        auto [it, inserted] = eager_pages.try_emplace(page);
        if (!inserted) {
            it->second.idle = 0;
            it->second.skip = 0; // a copy at the next fence would have spared this fault
            continue;
        }
        u64 offset, alt_offset = 0;
        if (!take_slot(offset)) {
            eager_pages.erase(it);
            break;
        }
        if (lagged && !take_slot(alt_offset)) {
            eager_free_offsets.push_back(offset);
            eager_pages.erase(it);
            break;
        }
        it->second = EagerPage{offset, 0, true, false, 0, 0, 0};
        it->second.lagged = lagged;
        it->second.alt_offset = alt_offset;
    }
    eager_count.store(static_cast<u32>(eager_pages.size()), std::memory_order_release);
}

void BufferCache::ApplyEagerSubtractsLocked() {
    for (const VAddr page : eager_subtracts) {
        gpu_modified_ranges.Subtract(page, EagerPageSize);
    }
    eager_subtracts.clear();
}

void BufferCache::MarkEagerStale(VAddr addr, u64 size) {
    if (!eager_count.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lock{eager_mutex};
    for (auto it = eager_pages.lower_bound(Common::AlignDown(addr, EagerPageSize));
         it != eager_pages.end() && it->first < addr + size; ++it) {
        it->second.dirty = true;
        it->second.valid = false;
        it->second.prev_valid = false; // neither copy has the CPU's bytes
    }
}

void BufferCache::NoteGpuWriteForEager(VAddr addr, u64 size) {
    // Before the memory is marked GPU modified (SynchronizeMemory), see ApplyGpuCopyIf.
    MarkDeferredStale(addr, size, false);
    if (!eager_count.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lock{eager_mutex};
    ApplyEagerSubtractsLocked();
    for (auto it = eager_pages.lower_bound(Common::AlignDown(addr, EagerPageSize));
         it != eager_pages.end() && it->first < addr + size; ++it) {
        auto& page = it->second;
        if (page.lagged) {
            // Its copies stay usable (a frame late is the point); the next fence copies it again.
            page.dirty = true;
            continue;
        }
        if (page.valid) {
            // Copied at a fence and never read before the GPU wrote it again.
            page.backoff = std::min<u8>(static_cast<u8>(page.backoff * 2 + 1), EagerMaxBackoff);
            page.skip = page.backoff;
            if (ReadbackStatsEnabled()) {
                Readbacks().eager_stale.fetch_add(1);
            }
        }
        page.dirty = true;
        page.valid = false;
    }
}

void BufferCache::EagerCopyAtFence() {
    if (deferred_count.load(std::memory_order_acquire)) {
        PruneDeferredReadbacks();
    }
    if (!eager_count.load(std::memory_order_acquire)) {
        return;
    }
    if (!eager_buffer) {
        eager_buffer = std::make_unique<Buffer>(instance, 0, EagerBufferSize, MemoryType::HostCached,
                                                "tlg eager readback");
    }
    std::scoped_lock lock{eager_mutex};
    ApplyEagerSubtractsLocked();
    const Buffer* arena = nullptr;
    boost::container::small_vector<vk::BufferCopy, 64> copies;
    boost::container::small_vector<EagerPage*, 64> copied;
    u64 skipped = 0;
    const auto flush_copies = [&] {
        if (!copies.empty()) {
            runtime.CopyBuffer(arena, eager_buffer.get(), copies);
            copies.clear();
        }
    };
    for (auto it = eager_pages.begin(); it != eager_pages.end();) {
        auto& page = it->second;
        if (++page.idle > EagerIdleFences) {
            eager_free_offsets.push_back(page.offset);
            if (page.lagged) {
                eager_free_offsets.push_back(page.alt_offset);
            }
            it = eager_pages.erase(it);
            continue;
        }
        if (page.lagged) {
            // Always copied when the GPU wrote it, into the slot the newest finished copy is not in.
            if (page.dirty && memory_tracker->IsRegionGpuModified(it->first, EagerPageSize)) {
                if (page.valid) {
                    page.prev_offset = page.offset;
                    page.prev_tick = page.tick;
                    page.prev_valid = true;
                    std::swap(page.offset, page.alt_offset);
                }
                const u64 block = it->first >> block_shift;
                const Buffer* page_arena = GetArena(block, (it->first + EagerPageSize - 1) >> block_shift);
                if (page_arena != arena) {
                    flush_copies();
                    arena = page_arena;
                }
                copies.push_back(vk::BufferCopy{.srcOffset = it->first - arena->cpu_addr,
                                                .dstOffset = page.offset,
                                                .size = EagerPageSize});
                copied.push_back(&page);
            }
            ++it;
            continue;
        }
        if (page.dirty && page.skip > 0 &&
            memory_tracker->IsRegionGpuModified(it->first, EagerPageSize)) {
            --page.skip;
            ++skipped;
        } else if (page.dirty && memory_tracker->IsRegionGpuModified(it->first, EagerPageSize)) {
            const u64 block = it->first >> block_shift;
            const Buffer* page_arena = GetArena(block, (it->first + EagerPageSize - 1) >> block_shift);
            if (page_arena != arena) {
                flush_copies();
                arena = page_arena;
            }
            copies.push_back(vk::BufferCopy{.srcOffset = it->first - arena->cpu_addr,
                                            .dstOffset = page.offset,
                                            .size = EagerPageSize});
            copied.push_back(&page);
        }
        ++it;
    }
    eager_count.store(static_cast<u32>(eager_pages.size()), std::memory_order_release);
    flush_copies();
    if (ReadbackStatsEnabled()) {
        Readbacks().eager_copied.fetch_add(copied.size());
        Readbacks().eager_skipped.fetch_add(skipped);
        ReportReadbacks();
    }
    if (copied.empty()) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    scheduler.Flush();
    for (EagerPage* page : copied) {
        page->tick = tick;
        page->dirty = false;
        page->valid = true;
    }
}

bool BufferCache::TryEagerRead(VAddr addr, u64 size) {
    if (!eager_count.load(std::memory_order_acquire)) {
        return false;
    }
    const VAddr page_addr = Common::AlignDown(addr, EagerPageSize);
    if (addr + size > page_addr + EagerPageSize || !eager_buffer) {
        return false;
    }
    u64 offset, tick;
    {
        std::scoped_lock lock{eager_mutex};
        const auto it = eager_pages.find(page_addr);
        if (it == eager_pages.end() || !it->second.valid ||
            !memory_tracker->IsRegionGpuModified(page_addr, EagerPageSize)) {
            return false;
        }
        if (merged_count.load(std::memory_order_acquire)) {
            std::scoped_lock merge_lock{merge_mutex};
            const auto m = merged_writes.upper_bound(page_addr + EagerPageSize - 1);
            if (m != merged_writes.begin() && std::prev(m)->second > page_addr) {
                return false; // CPU bytes not yet on the GPU: the copy would lose them
            }
        }
        it->second.idle = 0;
        offset = it->second.offset;
        tick = it->second.tick;
        if (it->second.lagged && !scheduler.IsFree(tick) && it->second.prev_valid &&
            scheduler.IsFree(it->second.prev_tick)) {
            // The newest copy is still on its way: the one before, finished, without waiting.
            offset = it->second.prev_offset;
            tick = it->second.prev_tick;
        }
    }
    WaitReadbackTick(tick);
    std::scoped_lock lock{eager_mutex};
    const auto it = eager_pages.find(page_addr);
    if (it == eager_pages.end()) {
        return false;
    }
    const bool lagged = it->second.lagged;
    if (lagged ? !((it->second.valid && it->second.tick == tick && it->second.offset == offset) ||
                   (it->second.prev_valid && it->second.prev_tick == tick &&
                    it->second.prev_offset == offset))
               : (!it->second.valid || it->second.tick != tick)) {
        return false; // the GPU wrote it again meanwhile (or the CPU, for a lagged page)
    }
    eager_buffer->Invalidate(offset, EagerPageSize);
    // tlg: only while the page is still GPU modified, checked under the region lock: a deferred
    // readback or a download may have brought it back meanwhile, and the guest may have
    // written it since without a fault.
    // (Unmapped memory has no backing to write, as for DownloadMemory.)
    const bool applied = memory_tracker->ApplyGpuCopyIf(
        page_addr, EagerPageSize, [] { return true; }, [&](VAddr start, VAddr end) {
            memory->TryWriteBacking(reinterpret_cast<void*>(start),
                                    eager_buffer->mapped_data.data() + offset + (start - page_addr),
                                    end - start);
        });
    if (!applied) {
        return false;
    }
    eager_subtracts.push_back(page_addr);
    if (!lagged) {
        it->second.valid = false;
    }
    it->second.backoff = 0;
    it->second.skip = 0;
    if (ReadbackStatsEnabled()) {
        Readbacks().eager_used.fetch_add(1);
    }
    return true;
}

bool BufferCache::DeferredReadbackEnabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("TLG_DEFERRED_READBACK");
        return e && *e == '1' && EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise;
    }();
    return enabled;
}

bool BufferCache::LocalReadbackJoin() {
    static const bool enabled = [] {
        const char* env = std::getenv("TLG_READBACK_LOCAL_JOIN");
        const bool on = env && env[0] == '1';
        if (on) {
            // In last_run.log whatever the GPU log level, so a tester's log shows the mode.
            std::printf("GPU: local readback join on (TLG_READBACK_LOCAL_JOIN=1)\n");
        }
        return on;
    }();
    return enabled;
}

std::shared_ptr<BufferCache::DeferredReadback> BufferCache::StartDeferredReadback(VAddr device_addr,
                                                                                  u64 size) {
    PruneDeferredReadbacks();
    const auto [arena, window_start, window_end] = GetReadbackWindow(device_addr, size);
    {
        // Another guest thread is already waiting for this memory: wait for the same copy.
        std::scoped_lock lock{deferred_mutex};
        for (const auto& readback : deferred_readbacks) {
            if (!readback->Done() && !readback->stale && !readback->lagged &&
                readback->start <= window_start &&
                window_end <= readback->end) {
                if (ReadbackStatsEnabled()) {
                    Readbacks().joined.fetch_add(1);
                }
                return readback;
            }
        }
    }
    // Overlapping copies are settled first, so one doesn't unmark memory the other hasn't
    // written back yet, and their ranges are where CollectDownloads looks for them.
    SettleDeferredReadbacks(window_start, window_end);
    if (eager_count.load(std::memory_order_acquire)) {
        std::scoped_lock lock{eager_mutex};
        ApplyEagerSubtractsLocked();
    }
    FlushMergedWrites(arena, window_start, window_end - window_start);

    auto readback = std::make_shared<DeferredReadback>();
    readback->arena_base = arena->cpu_addr;
    readback->start = window_start;
    readback->end = window_end;
    const u64 total_size_bytes =
        CollectDownloads(arena, window_start, window_end - window_start, readback->copies);
    if (total_size_bytes == 0) {
        return nullptr;
    }
    // Held until pruned: the guest thread reads it after the GPU is done with it.
    readback->staging =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
    for (auto& copy : readback->copies) {
        copy.dstOffset += readback->staging.offset;
    }
    runtime.CopyBuffer(arena, readback->staging.buffer, readback->copies);
    readback->tick = scheduler.CurrentTick();
    scheduler.Flush();
    {
        std::scoped_lock lock{deferred_mutex};
        deferred_readbacks.push_back(readback);
        deferred_count.store(static_cast<u32>(deferred_readbacks.size()),
                             std::memory_order_release);
    }
    if (ReadbackStatsEnabled()) {
        Readbacks().deferred.fetch_add(1);
        Readbacks().deferred_bytes.fetch_add(total_size_bytes);
        NoteReadbackSite(window_start, total_size_bytes);
    }
    return readback;
}

std::shared_ptr<BufferCache::DeferredReadback> BufferCache::PendingDeferredReadback(
    VAddr device_addr, u64 size) {
    if (!deferred_count.load(std::memory_order_acquire)) {
        return nullptr;
    }
    const auto window = GetReadbackWindow(device_addr, size);
    const VAddr window_start = std::get<1>(window);
    const VAddr window_end = std::get<2>(window);
    std::scoped_lock lock{deferred_mutex};
    for (const auto& readback : deferred_readbacks) {
        // The same test StartDeferredReadback joins with on the GPU command thread.
        if (!readback->Done() && !readback->stale && readback->start <= window_start &&
            window_end <= readback->end) {
            return readback;
        }
    }
    return nullptr;
}

bool BufferCache::FinishDeferredReadback(DeferredReadback& readback, bool wait) {
    if (wait) {
        WaitReadbackTick(readback.tick);
    }
    std::scoped_lock lock{readback.mutex};
    if (readback.applied.load(std::memory_order_acquire)) {
        return true;
    }
    if (readback.recovered.load(std::memory_order_acquire) || readback.stale.load()) {
        return false;
    }
    readback.staging.Invalidate();
    const u8* const mapped = readback.staging.mapped - readback.staging.offset;
    const bool applied = memory_tracker->ApplyGpuCopyIf(
        readback.start, readback.end - readback.start, [&] { return !readback.stale.load(); },
        [&](VAddr start, VAddr end) {
            for (const auto& copy : readback.copies) {
                const VAddr copy_start = readback.arena_base + copy.srcOffset;
                const VAddr lo = std::max<VAddr>(copy_start, start);
                const VAddr hi = std::min<VAddr>(copy_start + copy.size, end);
                if (lo < hi) {
                    memory->TryWriteBacking(reinterpret_cast<void*>(lo),
                                            mapped + copy.dstOffset + (lo - copy_start), hi - lo);
                }
            }
        });
    if (!applied) {
        return false;
    }
    readback.applied.store(true, std::memory_order_release);
    return true;
}

void BufferCache::RecoverDeferredReadback(DeferredReadback& readback) {
    std::scoped_lock lock{readback.mutex};
    if (readback.Done()) {
        return;
    }
    // Its ranges left gpu_modified_ranges when the copy was recorded. The copy won't be used:
    // they are downloaded again, with whatever the GPU wrote since.
    for (const auto& copy : readback.copies) {
        gpu_modified_ranges.Add(readback.arena_base + copy.srcOffset, copy.size);
    }
    readback.recovered.store(true, std::memory_order_release);
}

void BufferCache::SettleDeferredReadbacks(VAddr start, VAddr end) {
    if (!deferred_count.load(std::memory_order_acquire)) {
        return;
    }
    boost::container::small_vector<std::shared_ptr<DeferredReadback>, 8> overlapping;
    {
        std::scoped_lock lock{deferred_mutex};
        for (const auto& readback : deferred_readbacks) {
            if (!readback->Done() && readback->start < end && start < readback->end) {
                overlapping.push_back(readback);
            }
        }
    }
    for (const auto& readback : overlapping) {
        if (readback->lagged) {
            // Its ranges left gpu_modified_ranges: its bytes go to memory before a newer copy.
            ApplyLaggedReadback(*readback, true);
            continue;
        }
        // A stale copy won't be applied: no need to wait for it (2d2511fe).
        if (readback->stale.load() || !FinishDeferredReadback(*readback, true)) {
            RecoverDeferredReadback(*readback);
        }
    }
}

void BufferCache::PruneDeferredReadbacks() {
    if (!deferred_count.load(std::memory_order_acquire)) {
        return;
    }
    boost::container::small_vector<std::shared_ptr<DeferredReadback>, 8> stale, lagged;
    {
        std::scoped_lock lock{deferred_mutex};
        for (const auto& readback : deferred_readbacks) {
            if (readback->Done()) {
                continue;
            }
            if (readback->lagged) {
                lagged.push_back(readback);
            } else if (readback->stale.load()) {
                stale.push_back(readback);
            }
        }
    }
    for (const auto& readback : stale) {
        RecoverDeferredReadback(*readback);
    }
    for (const auto& readback : lagged) {
        ApplyLaggedReadback(*readback, false); // once the GPU is done with it
    }
    std::scoped_lock lock{deferred_mutex};
    std::erase_if(deferred_readbacks, [this](const std::shared_ptr<DeferredReadback>& readback) {
        if (!readback->Done()) {
            return false;
        }
        // Freed for reuse once the current tick completes; a guest thread still holding the
        // object only looks at its flags (applied or recovered) from now on.
        staging_pool.FreeDeferred(readback->staging);
        return true;
    });
    deferred_count.store(static_cast<u32>(deferred_readbacks.size()), std::memory_order_release);
}

void BufferCache::MarkDeferredStale(VAddr addr, u64 size, bool cpu) {
    if (!deferred_count.load(std::memory_order_acquire)) {
        return;
    }
    boost::container::small_vector<std::shared_ptr<DeferredReadback>, 4> marked;
    {
        std::scoped_lock lock{deferred_mutex};
        for (const auto& readback : deferred_readbacks) {
            if (!readback->Done() && readback->start < addr + size && addr < readback->end) {
                readback->stale.store(true);
                if (cpu) {
                    marked.push_back(readback);
                }
            }
        }
    }
    // An apply already past its stale check finishes before the guest writes.
    for (const auto& readback : marked) {
        std::scoped_lock lock{readback->mutex};
    }
}

bool BufferCache::CpuWriteMerge(VAddr dst, const void* src, int fill, u64 size) {
    static const bool enabled = [] {
        const char* e = std::getenv("TLG_WRITE_MERGE");
        return e && *e == '1' && EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise;
    }();
    if (!enabled || size < 16 || !memory_tracker->IsRegionGpuModified(dst, size)) {
        return false;
    }
    // The copies of these pages made earlier (eager, deferred) lack the bytes about to be
    // written: drop them before the write, waiting for an apply in progress, so none lands
    // over the new bytes (guest thread: only marks them, gpu_modified_ranges belongs to the
    // GPU-side threads).
    MarkEagerStale(dst, size);
    MarkDeferredStale(dst, size, true);
    if (src) {
        if (!memory->TryWriteBacking(reinterpret_cast<void*>(dst), src, size)) {
            return false;
        }
    } else if (!memory->FillBacking(reinterpret_cast<void*>(dst), fill, size)) {
        // Part of the range has no backing: the guest writes it itself. The runs written through
        // the backing hold the same fill byte, so a partial fill is not lost.
        return false;
    }
    std::scoped_lock lock{merge_mutex};
    VAddr start = dst, end = dst + size;
    auto it = merged_writes.upper_bound(start);
    if (it != merged_writes.begin() && std::prev(it)->second >= start) {
        --it;
        start = it->first;
        end = std::max(end, it->second);
        it = merged_writes.erase(it);
    }
    while (it != merged_writes.end() && it->first <= end) {
        end = std::max(end, it->second);
        it = merged_writes.erase(it);
    }
    merged_writes.emplace(start, end);
    merged_count.store(merged_writes.size(), std::memory_order_release);
    return true;
}

void BufferCache::FlushMergedWrites(const Buffer* arena, VAddr device_addr, u64 size) {
    if (!merged_count.load(std::memory_order_acquire)) {
        return;
    }
    const VAddr lo = std::max<VAddr>(device_addr, arena->cpu_addr);
    const VAddr hi = std::min<VAddr>(device_addr + size, arena->cpu_addr + arena->size_bytes);
    if (lo >= hi) {
        return;
    }
    boost::container::small_vector<std::pair<VAddr, VAddr>, 16> ranges;
    {
        std::scoped_lock lock{merge_mutex};
        auto it = merged_writes.upper_bound(lo);
        if (it != merged_writes.begin() && std::prev(it)->second > lo) {
            --it;
        }
        while (it != merged_writes.end() && it->first < hi) {
            const VAddr a = it->first, b = it->second;
            const VAddr s = std::max(a, lo), e = std::min(b, hi);
            ranges.emplace_back(s, e);
            it = merged_writes.erase(it);
            if (a < s) merged_writes.emplace(a, s);
            if (e < b) it = merged_writes.emplace(e, b).first, ++it;
        }
        merged_count.store(merged_writes.size(), std::memory_order_release);
    }
    if (ranges.empty()) {
        return;
    }
    for (const auto& [s, e] : ranges) {
        NoteGpuWriteForEager(s, e - s); // the GPU copy changes: an older eager copy is stale
    }
    u64 total = 0;
    for (const auto& [s, e] : ranges) {
        total += Common::AlignUp(e - s, 16);
    }
    const auto staging = staging_pool.Request(total, MemoryType::HostUncached);
    boost::container::small_vector<vk::BufferCopy, 16> copies;
    u64 offset = 0;
    for (const auto& [s, e] : ranges) {
        memory->CopySparseMemory(s, staging.mapped + offset, e - s);
        copies.push_back(vk::BufferCopy{.srcOffset = staging.offset + offset,
                                        .dstOffset = s - arena->cpu_addr,
                                        .size = e - s});
        offset += Common::AlignUp(e - s, 16);
    }
    staging.Flush();
    runtime.CopyBuffer(staging.buffer, arena, copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    FlushMergedWrites(arena, device_addr, size);
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
        },
        [&] { src_buffer = UploadCopies(arena, copies, total_size_bytes); });

    if (src_buffer) {
        runtime.CopyBuffer(src_buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    // Resolve before launching host copies: a copy-thread fault could deadlock submission.
    for (const auto& copy : copies) {
        texture_cache.ResolveReadbacks(copy.dstOffset, copy.size, true, ReadbackSource::BufferUpload);
    }
    TlgStats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    if (total_size_bytes > MAX_UPLOAD_PART_BYTES) {
        // Recorded here, part by part: nothing left for the caller to copy.
        UploadInParts(arena, copies);
        return nullptr;
    }
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    // tlg: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!TlgToggle::Disabled(TlgToggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dstOffset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.srcOffset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!TlgToggle::Disabled(TlgToggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            TlgCopy::Async([group, memory = memory, staging] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
                staging.Flush();
            });
        };
        for (auto& copy : copies) {
            group->push_back({copy.dstOffset, staging.mapped + copy.srcOffset, copy.size});
            group_bytes += copy.size;
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dstOffset, staging.mapped + copies[i].srcOffset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        TlgCopy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.srcOffset += staging.offset;
        copy.dstOffset -= arena->cpu_addr;
    }
    staging.Flush();
    return staging.buffer;
}

void BufferCache::UploadInParts(const Buffer* arena, std::span<const vk::BufferCopy> ranges) {
    u64 total_bytes = 0;
    for (const auto& range : ranges) {
        total_bytes += range.size;
    }
    LOG_WARNING(Render, "Uploading {} MB in parts of {} MB", total_bytes >> 20,
                MAX_UPLOAD_PART_BYTES >> 20);
    // Guest memory is copied on this thread (helped by the copy threads), not deferred: the
    // staging memory is reused as soon as the GPU has copied the part.
    struct HostCopy {
        VAddr source;
        u8* destination;
        u64 size;
    };
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    std::vector<HostCopy> host_copies;
    Vulkan::StagingBufferRef staging{};
    u64 part_bytes = 0;
    u32 parts = 0;
    const auto upload_part = [&] {
        TlgCopy::ParallelFor(host_copies.size(), [&](std::size_t i) {
            memory->CopySparseMemory(host_copies[i].source, host_copies[i].destination,
                                     host_copies[i].size);
        });
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
        copies.clear();
        host_copies.clear();
        part_bytes = 0;
        ++parts;
    };
    // Pieces of a few MB spread a part over the copy threads.
    constexpr u64 PieceBytes = 4_MB;
    for (const auto& range : ranges) {
        for (u64 done = 0; done < range.size;) {
            if (part_bytes == 0) {
                if (parts != 0) {
                    // The staging memory is free again once the GPU has copied the last part.
                    // This thread owns the caches, as for DownloadMemory's Finish.
                    scheduler.Finish();
                }
                staging = staging_pool.Request(MAX_UPLOAD_PART_BYTES, MemoryType::HostUncached);
            }
            const u64 size = std::min(range.size - done, MAX_UPLOAD_PART_BYTES - part_bytes);
            const VAddr addr = range.dstOffset + done;
            copies.push_back(vk::BufferCopy{.srcOffset = staging.offset + part_bytes,
                                            .dstOffset = addr - arena->cpu_addr,
                                            .size = size});
            for (u64 piece = 0; piece < size; piece += PieceBytes) {
                host_copies.push_back({addr + piece, staging.mapped + part_bytes + piece,
                                       std::min(PieceBytes, size - piece)});
            }
            part_bytes += size;
            done += size;
            if (part_bytes == MAX_UPLOAD_PART_BYTES) {
                upload_part();
            }
        }
    }
    if (part_bytes != 0) {
        upload_part();
    }
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    // tlg: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !TlgToggle::Disabled(TlgToggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        miss = {device_addr, size, generation};
        return false;
    }
    // tlg: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    if (image.info.props.is_bc_decoded) {
        return false; // tlg: TLG_BC_TINY: decoded texels, not the guest's blocks
    }
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    // tlg: the submit callback runs inside SubmitExecution, which with TLG_SPLIT_SUBMIT_LOCK=1
    // holds the queue lock only for its own submit.
    auto queue_lock = Vulkan::Scheduler::LockQueueInSubmit();
    const auto bind_start = std::chrono::steady_clock::now();
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    const auto elapsed = std::chrono::steady_clock::now() - bind_start;
    queue_lock = std::unique_lock<std::mutex>{};
    // Sparse binds are reported to get slower the more memory is bound on NVIDIA drivers. Log the
    // slow ones (the first few, then every 32nd) with how much is resident, to confirm the cost.
    if (elapsed >= std::chrono::milliseconds{2}) {
        static u32 slow_binds = 0;
        if (slow_binds < 16 || slow_binds % 32 == 0) {
            u64 bound_blocks{};
            for (const auto& range : resident_ranges) {
                bound_blocks += range.end - range.start;
            }
            LOG_WARNING(Render,
                        "Sparse bind took {:.1f} ms, {} MB resident in {} allocations "
                        "(slow bind #{})",
                        std::chrono::duration<double, std::milli>(elapsed).count(),
                        (bound_blocks << block_shift) >> 20, residency_allocations,
                        slow_binds + 1);
        }
        ++slow_binds;
    }

    pending_binds.clear();
}

} // namespace VideoCore
