#include <atomic>
#include "deep_profile.h"
#include "repeat_census.h"
#include "compute_phase.h"
#include "runtime_timeline.h"
#include "unified_memory.h"
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include "parallel.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <intrin.h>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "common/memory_patcher.h"
#include "common/perf_stats.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"
#include "video_core/buffer_cache/upload_parts.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace iss {void ProbeNativeFeedback(const Vulkan::Instance&,Vulkan::Scheduler&);}
namespace iss::asc_family {extern u32 family;extern bool readback;extern u32 readback_family;extern thread_local u32 pool_family;}
namespace iss::asc_stats {
// Cross-queue flush causes (asc_queue). Count and CP time of each forced submit.
// Same enum in vk_rasterizer.cpp (printed there).
enum Cause:u32 {WaitGraphicsSubmitted,ComputeSubmit,GdsFlushGraphics,GdsFlushCompute,BdaFlush,
                RangeFlushCompute,RangeFlushGraphics,FirstTouchFlushGraphics,ReadbackWriterFlush,Num};
std::atomic<u64> count[Num]{},ns[Num]{};
struct Timer {Cause c;u64 start{Common::Perf::NowNs()};~Timer(){count[c].fetch_add(1,std::memory_order_relaxed);ns[c].fetch_add(Common::Perf::NowNs()-start,std::memory_order_relaxed);}};
// Sources of cross-queue waits (CP thread only): which ranges make a queue wait for the other.
struct WaitKey {u64 page;u32 size;u8 queue,kind,untouched;bool operator==(const WaitKey&) const=default;};
struct WaitKeyHash {size_t operator()(const WaitKey& k) const noexcept {return std::hash<u64>{}(k.page*31+k.size*7+k.queue*3+k.kind+k.untouched*5);}};
std::unordered_map<WaitKey,u64,WaitKeyHash> sources;
std::array<std::array<u64,4>,2> age{}; // per waiting queue: other tick age 0,1,2-3,4+
bool in_upload{};
void NoteWait(u64 addr,u64 size,u32 queue,bool written,bool untouched,u64 dependency,u64 other_current){
    const u8 kind=in_upload?2:written?1:0;
    ++sources[{addr>>12,u32(std::min<u64>(size,~0u)),u8(queue),kind,u8(untouched)}];
    const u64 a=other_current>dependency?other_current-dependency:0;
    ++age[queue][a==0?0:a==1?1:a<4?2:3];
}
void Print(){
    std::vector<std::pair<WaitKey,u64>> v(sources.begin(),sources.end());
    std::sort(v.begin(),v.end(),[](auto& a,auto& b){return a.second>b.second;});
    std::fprintf(stderr,"ASC_WAIT_AGE compute(0,1,2-3,4+)=%llu,%llu,%llu,%llu gfx=%llu,%llu,%llu,%llu distinct=%zu\n",
        age[1][0],age[1][1],age[1][2],age[1][3],age[0][0],age[0][1],age[0][2],age[0][3],v.size());
    static constexpr const char* kinds[]{"read","write","upload"};
    for(size_t i=0;i<std::min<size_t>(v.size(),24);++i){const auto& [k,n]=v[i];
        std::fprintf(stderr,"ASC_WAIT_SRC q=%s %s%s addr=%llx size=%u n=%llu\n",k.queue?"compute":"gfx",kinds[k.kind],k.untouched?" first":"",k.page<<12,k.size,n);}
}
}
namespace iss::asc_stats_api {void Print(){iss::asc_stats::Print();}}
// The command buffer segment the command processor reads (liverpool.cpp).
u64 g_iss_dcb_lo{}, g_iss_dcb_hi{};

namespace VideoCore {

namespace {
// ISS_PARALLEL_UPLOAD_COPY (default on, =0 off): the guest copy of a large upload part split in
// 4 MB chunks over the parallel algorithms' threads. The world's first frame after a load binds a
// 2.6 GB texel buffer: 747 ms of copy on the command processor alone (load1 2026-10-18).
void ParallelGuestCopy(Core::MemoryManager* memory, VAddr source, u8* dest, u64 bytes) {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_PARALLEL_UPLOAD_COPY");
        return !p || p[0] != '0';
    }();
    constexpr u64 Chunk = 4ull << 20;
    if (!enabled || bytes < 2 * Chunk) {
        memory->CopySparseMemory(source, dest, bytes);
        return;
    }
    std::vector<u64> chunks((bytes + Chunk - 1) / Chunk);
    for (u64 i = 0; i < chunks.size(); ++i) chunks[i] = i * Chunk;
    ISS_PAR_FOR_EACH(chunks.begin(), chunks.end(), [&](u64 offset) {
        memory->CopySparseMemory(source + offset, dest + offset, std::min(Chunk, bytes - offset));
    });
}

bool SmallReadMemoEnabled() {
    static const bool enabled = [] {
        const char* flag = std::getenv("ISS_SMALL_READ_MEMO");
        return flag && flag[0] == '1';
    }();
    return enabled;
}
// Cache only alias metadata, never content or GPU completion. Native remaps and
// GPU mapping changes invalidate every cached answer through their generations.
bool UniqueReadbackBacking(Core::MemoryManager& memory, VAddr address, u64 size,
                           u64 generation, u64 epoch) {
    struct Entry {const void* owner{};u64 address{},size{},generation{},epoch{};bool unique{};};
    static thread_local std::array<Entry,64> entries{};
    static thread_local std::size_t next{};
    for(const auto& e:entries) if(e.owner==&memory && e.address==address && e.size==size &&
        e.generation==generation && e.epoch==epoch)return e.unique;
    const auto aliases=memory.EnumerateBackingAliases(address,size);
    const bool unique=aliases && aliases->size()==1 && aliases->front().guest_address==address &&
        aliases->front().size==size && memory.MappingEpoch()==epoch;
    entries[next++%entries.size()]={&memory,address,size,generation,epoch,unique};
    return unique;
}
// Diagnostic only. No guest memory contents are logged. The fixed cap keeps a long run bounded
// even when the guest faults on the same feedback page every frame.
constexpr u64 ReadbackDiagMaxBytes = 32_MB;
constexpr u64 ReadbackDiagMaxRows = 200'000;
struct ReadbackDiagEvent {
    const char* event{};
    u64 rip{};
    u64 address{};
    u64 size{};
    u64 src{};
    u64 dst{};
    u64 window_start{};
    u64 window_end{};
    u64 request_gfx_tick{};
    u64 request_compute_tick{};
    u64 window_gfx_tick{};
    u64 window_compute_tick{};
    u32 request_state{}; // bit 0 done, 1 submitted, 2 current
    u32 window_state{};
    u32 queue{};
    u64 wait_tick{};
    u32 copies{};
    u64 copy_bytes{};
    u64 duration_ns{};
    u32 flags{}; // event-specific, documented in the report
};

bool G4FeedbackProbeEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_G4_FEEDBACK_PROBE");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool ReadbackDiagEnabled() {
    static const bool enabled = [] {
        const char* flag = std::getenv("ISS_READBACK_DIAG");
        const char* path = std::getenv("ISS_READBACK_DIAG_LOG");
        return flag && flag[0] == '1' && path && path[0] != '\0';
    }();
    if (!enabled) return false;
    static const u64 first_frame = [] {
        const char* p = std::getenv("ISS_READBACK_DIAG_START_FRAME");
        return p ? std::strtoull(p, nullptr, 0) : 0;
    }();
    static const u64 frame_count = [] {
        const char* p = std::getenv("ISS_READBACK_DIAG_FRAMES");
        return p ? std::strtoull(p, nullptr, 0) : ~u64{0};
    }();
    const auto frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    return frame >= first_frame && frame - first_frame < frame_count;
}

void LogReadbackDiag(const ReadbackDiagEvent& e) {
    if (!ReadbackDiagEnabled()) [[likely]] {
        return;
    }
    static std::mutex mutex;
    static std::FILE* file = [] {
        std::FILE* out = std::fopen(std::getenv("ISS_READBACK_DIAG_LOG"), "wb");
        if (out) {
            std::setvbuf(out, nullptr, _IOFBF, 1 << 20);
            std::fputs("event,ns,frame,thread,rip,address,size,src,dst,window_start,window_end,"
                       "request_gfx_tick,request_compute_tick,window_gfx_tick,window_compute_tick,"
                       "request_state,window_state,queue,wait_tick,copies,copy_bytes,duration_ns,flags\n", out);
        }
        return out;
    }();
    if (!file) {
        return;
    }
    static u64 bytes = 0;
    static u64 rows = 0;
    static bool truncated = false;
    const u64 now = Common::Perf::NowNs();
    const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    const u64 thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    char line[512];
    const int result = std::snprintf(
        line, sizeof(line),
        "%s,%llu,%llu,%llu,0x%llx,0x%llx,%llu,0x%llx,0x%llx,0x%llx,0x%llx,"
        "%llu,%llu,%llu,%llu,%u,%u,%u,%llu,%u,%llu,%llu,%u\n",
        e.event, static_cast<unsigned long long>(now), static_cast<unsigned long long>(frame),
        static_cast<unsigned long long>(thread), static_cast<unsigned long long>(e.rip),
        static_cast<unsigned long long>(e.address), static_cast<unsigned long long>(e.size),
        static_cast<unsigned long long>(e.src), static_cast<unsigned long long>(e.dst),
        static_cast<unsigned long long>(e.window_start),
        static_cast<unsigned long long>(e.window_end),
        static_cast<unsigned long long>(e.request_gfx_tick),
        static_cast<unsigned long long>(e.request_compute_tick),
        static_cast<unsigned long long>(e.window_gfx_tick),
        static_cast<unsigned long long>(e.window_compute_tick), e.request_state, e.window_state,
        e.queue, static_cast<unsigned long long>(e.wait_tick), e.copies,
        static_cast<unsigned long long>(e.copy_bytes),
        static_cast<unsigned long long>(e.duration_ns), e.flags);
    if (result <= 0 || static_cast<size_t>(result) >= sizeof(line)) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (rows >= ReadbackDiagMaxRows || bytes + result > ReadbackDiagMaxBytes) {
        if (!truncated) {
            std::fputs("truncated\n", file);
            std::fflush(file);
            truncated = true;
        }
        return;
    }
    std::fwrite(line, 1, result, file);
    bytes += result;
    ++rows;
    if ((rows & 4095) == 0) {
        std::fflush(file);
    }
}

u32 ReadbackDiagState(bool done, bool submitted, bool current) {
    return (done ? 1U : 0U) | (submitted ? 2U : 0U) | (current ? 4U : 0U);
}
} // namespace

thread_local BufferCache::ThreadQueueSelection BufferCache::thread_queue_selection{};

void BufferCache::NoteMappingChange(VAddr address, u64 size) noexcept {
    if (size == 0) return;
    {
        std::scoped_lock lock{coherence_mutex};
        mapping_generation.fetch_add(1, std::memory_order_acq_rel);
        const u64 first = address >> 24;
        const u64 last = size - 1 > std::numeric_limits<u64>::max() - address
                             ? std::numeric_limits<u64>::max() >> 24
                             : (address + size - 1) >> 24;
        for (auto& [region, version] : prepared_read_mapping_versions) {
            if (region >= first && region <= last) ++version;
        }
        if (cp_split_enabled) packet_hazards.InvalidateUnpublished();
    }
    if (G4FeedbackProbeEnabled()) feedback_writer_probe.Invalidate(address, size);
}

bool BufferCache::PlanGuestAccess(VAddr address, u64 size, bool write,
                                  u64 prepared_mapping_generation,
                                  std::vector<iss::g4::PacketResourceAccess>& out) const {
    if (size == 0 || address > std::numeric_limits<u64>::max() - size ||
        prepared_mapping_generation != MappingGeneration()) return false;
    std::vector<iss::g4::PacketResourceAccess> prepared;
    prepared.reserve(4);
    while (size != 0) {
        // An unusual fragmented binding stays on the coordinated graphics path. Do not grow
        // packet preparation linearly with a multi-gigabyte sparse descriptor.
        if (prepared.size() == 128) return false;
        const auto span = memory->DescribeBacking(address, size);
        if (!span || span->size == 0) return false;
        prepared.push_back({iss::g4::GuestBackingKey, span->physical_start, span->size, write});
        address += span->size;
        size -= span->size;
    }
    if (prepared_mapping_generation != MappingGeneration()) return false;
    out.insert(out.end(), prepared.begin(), prepared.end());
    return true;
}

std::optional<BufferCache::PreparedGuestBufferBinding>
BufferCache::PrepareGuestBufferBinding(VAddr address, u32 size, bool write) const {
    if (!cp_split_enabled || size == 0 ||
        address > std::numeric_limits<VAddr>::max() - size) return std::nullopt;
    const u64 mapgen = MappingGeneration();
    const bool gpu_owned = memory_tracker->IsRegionGpuModified(address, size);
    const auto cpu_ready = [&] {
        return gpu_owned
                   ? Common::Perf::GetTuning().hot_page_faults.load(std::memory_order_relaxed) == 0 &&
                         !memory_tracker->IsRegionCpuModified(address, size)
                   : memory_tracker->IsRegionCpuClean(address, size, 0);
    };
    std::array<u64, 2> versions{};
    if (!cpu_ready() ||
        !memory_tracker->GetCpuVersions(address, size, versions)) return std::nullopt;
    const auto writer_ticks = gpu_owned ? memory_tracker->GetWriteTicks(address, size)
                                        : std::array<u64, 2>{};
    const u64 writer_seq = gpu_owned ? memory_tracker->GetWriteSeq(address, size) : 0;
    if (gpu_owned && (writer_seq == 0 || (writer_ticks[0] == 0 && writer_ticks[1] == 0)))
        return std::nullopt; // no exact producer to seal
    const PublishedArena arena = PeekPublishedArena(address, size);
    if (!arena || mapgen != MappingGeneration() || !cpu_ready()) return std::nullopt;
    std::array<u64, 2> after{};
    if (!memory_tracker->GetCpuVersions(address, size, after) || after != versions)
        return std::nullopt;
    if (gpu_owned && (memory_tracker->GetWriteTicks(address, size) != writer_ticks ||
                      memory_tracker->GetWriteSeq(address, size) != writer_seq))
        return std::nullopt;
    return PreparedGuestBufferBinding{arena.arena, arena.offset, size, address, write,
                                      arena.residency_tick, arena.graphics_bda_tick,
                                      mapgen, versions, writer_ticks, writer_seq, gpu_owned};
}

bool BufferCache::RevalidateGuestBufferBinding(const PreparedGuestBufferBinding& plan) const {
    if (!plan.buffer || !plan.size || plan.size > std::numeric_limits<u32>::max() ||
        plan.mapping_generation != MappingGeneration()) return false;
    if (plan.gpu_owned) {
        if (Common::Perf::GetTuning().hot_page_faults.load(std::memory_order_relaxed) != 0 ||
            !memory_tracker->IsRegionGpuModified(plan.guest_address, plan.size) ||
            memory_tracker->IsRegionCpuModified(plan.guest_address, plan.size) ||
            memory_tracker->GetWriteTicks(plan.guest_address, plan.size) != plan.writer_ticks ||
            memory_tracker->GetWriteSeq(plan.guest_address, plan.size) !=
                plan.writer_write_seq) return false;
    } else if (!memory_tracker->IsRegionCpuClean(plan.guest_address, plan.size, 0)) {
        return false;
    }
    std::array<u64, 2> versions{};
    if (!memory_tracker->GetCpuVersions(plan.guest_address, plan.size, versions) ||
        versions != plan.cpu_versions) return false;
    const PublishedArena arena = PeekPublishedArena(plan.guest_address,
                                                    static_cast<u32>(plan.size));
    return arena && arena.arena == plan.buffer && arena.offset == plan.offset &&
           arena.residency_tick == plan.residency_tick &&
           arena.graphics_bda_tick == plan.graphics_bda_tick &&
           plan.mapping_generation == MappingGeneration();
}

bool BufferCache::PublishPacketWithBufferWrites(
    u64 lease_id, iss::g4::QidTick token, bool guest_inputs_stable,
    std::span<const PreparedGuestBufferBinding> bindings,
    std::span<const iss::g4::NegativeImageSnapshot> negative_images) {
    const u32 queue = token.lane == iss::g4::CpLane::Graphics ? 0 : 1;
    if (!cp_split_enabled || ActiveQueuePtr()->index != queue ||
        Sched().CurrentTick() != token.tick) return false;
    std::scoped_lock lock{coherence_mutex};
    if (std::ranges::any_of(bindings, [&](const auto& binding) {
            return !binding.buffer || !binding.size ||
                   binding.mapping_generation != MappingGeneration();
        })) return false;
    for (const auto& binding : bindings) {
        if (!binding.write) continue;
        const bool covered = std::ranges::any_of(negative_images, [&](const auto& snapshot) {
            return snapshot.result == iss::g4::NegativeImageResult::NoRegisteredPage &&
                   snapshot.address == binding.guest_address &&
                   snapshot.size == binding.size;
        });
        if (!covered) return false;
    }
    const auto publish = [&]() -> bool {
        if (!packet_hazards.Publish(lease_id, token, MappingGeneration(),
                                    guest_inputs_stable)) return false;
        // The lease's token and dirty bytes become visible to readback as one update.
        for (const auto& binding : bindings) {
            if (!binding.write) continue;
            gpu_modified_ranges.Add(binding.guest_address, binding.size);
            const u64 seq = memory_tracker->SetWriteTick(
                binding.guest_address, binding.size, token.tick, queue,
                MappingGeneration(), memory->MappingEpoch());
            ++coherence_write_generation;
            readback_writers.Publish({token, binding.guest_address, binding.size,
                                      MappingGeneration(), seq});
        }
        return true;
    };
    // Lock order is coherence -> texture registry writer. A newly registered image
    // cannot invalidate a negative proof between this check and writer publication.
    return negative_images.empty()
               ? publish()
               : texture_cache.WithNegativeSnapshotsLocked(negative_images, publish);
}

void BufferCache::InvalidatePreparedPacketAccess(VAddr address, u64 size) {
    if (!cp_split_enabled || size == 0) return;
    {
        std::scoped_lock lock{coherence_mutex};
        readback_writers.Invalidate(address, size);
    }
    std::vector<iss::g4::PacketResourceAccess> changed;
    const u64 mapgen = MappingGeneration();
    if (!PlanGuestAccess(address, size, true, mapgen, changed)) {
        packet_hazards.InvalidateUnpublished();
        return;
    }
    for (const auto& access : changed)
        packet_hazards.InvalidateCpuWrite(access.backing_key, access.offset, access.size);
}

std::optional<iss::g4::ReadbackWriterSnapshot> BufferCache::CaptureG4ReadbackWriters(
    VAddr address, u64 size) const {
    if (!cp_split_enabled || size == 0 || address > std::numeric_limits<VAddr>::max() - size)
        return std::nullopt;
    std::scoped_lock lock{coherence_mutex};
    const auto ticks = memory_tracker->GetWriteTicks(address, size);
    if (ticks[0] == 0 && ticks[1] == 0 && gpu_modified_ranges.Intersects(address, size))
        return std::nullopt;
    return readback_writers.Capture(address, size, MappingGeneration(),
                                    memory_tracker->GetWriteSeq(address, size),
                                    ticks);
}

BufferCache::G4ReadbackPreflight BufferCache::EnqueueG4ReadbackPreflight(
    iss::g4::ReadbackWriterSnapshot snapshot) {
    if (!cp_split_enabled || !liverpool) return {};
    auto result = std::make_shared<std::atomic<G4ReadbackPreflightState>>(
        G4ReadbackPreflightState::Pending);
    auto control = liverpool->EnqueueGraphicsControl([this, snapshot, result] {
        // G safe point: the control owns all inputs and never waits for C or records
        // Vulkan. The cache lock is released before checking scheduler progress.
        if (!RevalidateG4ReadbackWriters(snapshot)) {
            result->store(G4ReadbackPreflightState::Stale, std::memory_order_release);
            return;
        }
        if (!UseReadbackQueue()) {
            result->store(G4ReadbackPreflightState::Unsupported, std::memory_order_release);
            return;
        }
        const bool graphics_submitted = snapshot.writer_ticks[0] <= scheduler.SubmittedTick();
        const bool asc_submitted = snapshot.writer_ticks[1] == 0 ||
                                   (compute_queue && snapshot.writer_ticks[1] <=
                                                         compute_queue->scheduler->SubmittedTick());
        result->store(graphics_submitted && asc_submitted
                          ? G4ReadbackPreflightState::Submitted
                          : G4ReadbackPreflightState::NeedsSeal,
                      std::memory_order_release);
    });
    if (!control) return {};
    return {std::move(control), std::move(result)};
}

BufferCache::G4FeedbackCopy BufferCache::EnqueueG4FeedbackCopy(
    iss::g4::ReadbackWriterSnapshot snapshot) {
    if (!cp_split_enabled || !liverpool || snapshot.size != 8) return {};
    auto result = std::make_shared<G4FeedbackCopyData>();
    auto control = liverpool->EnqueueGraphicsControl([this, snapshot, result] {
        const auto finish = [&](G4FeedbackCopyState state) {
            result->state.store(state, std::memory_order_release);
        };
        if (!RevalidateG4ReadbackWriters(snapshot)) {
            finish(G4FeedbackCopyState::Stale);
            return;
        }
        if (!UseReadbackQueue() || (snapshot.writer_ticks[0] == 0 &&
                                    snapshot.writer_ticks[1] == 0)) {
            finish(G4FeedbackCopyState::Unsupported);
            return;
        }
        // A control never asks C to flush or waits for an open producer tick.
        if (snapshot.writer_ticks[0] > scheduler.SubmittedTick() ||
            (snapshot.writer_ticks[1] != 0 &&
             (!compute_queue || snapshot.writer_ticks[1] >
                                    compute_queue->scheduler->SubmittedTick()))) {
            finish(G4FeedbackCopyState::NeedsSeal);
            return;
        }
        const PublishedArena arena = PeekPublishedArena(snapshot.address, 8);
        if (!arena) {
            finish(G4FeedbackCopyState::NeedsSeal);
            return;
        }
        std::array<u64, 2> cpu_versions{};
        if (!memory_tracker->IsRegionCpuClean(snapshot.address, 8, 0) ||
            !memory_tracker->GetCpuVersions(snapshot.address, 8, cpu_versions)) {
            finish(G4FeedbackCopyState::Stale);
            return;
        }
        // Staging and driver calls happen with no coherence, region, or residency lock.
        const auto staging = staging_pool.Request(8, MemoryType::HostCached);
        const vk::BufferCopy copy{arena.offset, staging.offset, 8};
        const ReadbackQueue::Job job{arena.arena->Handle(), staging.buffer->Handle(),
                                     std::span{&copy, 1}};
        std::array<ReadbackQueue::Wait, 3> waits{};
        size_t wait_count{};
        if (arena.residency_tick != 0)
            waits[wait_count++] = {memory_semaphore.Handle(), arena.residency_tick};
        if (snapshot.writer_ticks[0] != 0)
            waits[wait_count++] = {scheduler.GetWorkSemaphore()->Handle(),
                                   snapshot.writer_ticks[0]};
        if (snapshot.writer_ticks[1] != 0)
            waits[wait_count++] = {compute_queue->scheduler->GetWorkSemaphore()->Handle(),
                                   snapshot.writer_ticks[1]};
        const u64 readback_tick = readback_queue->Submit(std::span{&job, 1},
                                                          std::span{waits.data(), wait_count});
        // Every waited producer has already been handed to the driver. This wait
        // cannot require G or C to advance PM4 or submit another command buffer.
        readback_queue->Semaphore().Wait(readback_tick);
        staging.Invalidate();
        std::array<u8, 8> bytes{};
        std::memcpy(bytes.data(), staging.mapped, bytes.size());
        std::array<u64, 2> after{};
        if (!RevalidateG4ReadbackWriters(snapshot) ||
            !memory_tracker->IsRegionCpuClean(snapshot.address, 8, 0) ||
            !memory_tracker->GetCpuVersions(snapshot.address, 8, after) ||
            after != cpu_versions) {
            finish(G4FeedbackCopyState::Stale);
            return;
        }
        result->bytes = bytes;
        finish(G4FeedbackCopyState::Copied);
    });
    if (!control) return {};
    return {std::move(control), std::move(result)};
}

bool BufferCache::RevalidateG4ReadbackWriters(
    const iss::g4::ReadbackWriterSnapshot& snapshot) const {
    if (!cp_split_enabled) return false;
    std::scoped_lock lock{coherence_mutex};
    return readback_writers.StillCurrent(snapshot, MappingGeneration(),
                                         memory_tracker->GetWriteSeq(snapshot.address,
                                                                     snapshot.size),
                                         memory_tracker->GetWriteTicks(snapshot.address,
                                                                       snapshot.size));
}

std::vector<std::pair<VAddr, VAddr>> BufferCache::SnapshotGpuRanges(VAddr address,
                                                                     u64 size,
                                                                     u64* write_generation) const {
    std::vector<std::pair<VAddr, VAddr>> ranges;
    std::scoped_lock lock{coherence_mutex};
    if (write_generation) *write_generation = coherence_write_generation;
    gpu_modified_ranges.ForEachInRange(address, size,
                                       [&](VAddr first, VAddr last) {
                                           ranges.emplace_back(first, last);
                                       });
    return ranges;
}

bool BufferCache::GpuRangesIntersect(VAddr address, u64 size) const {
    std::scoped_lock lock{coherence_mutex};
    return gpu_modified_ranges.Intersects(address, size);
}

void BufferCache::SubtractGpuRange(VAddr address, u64 size) {
    std::scoped_lock lock{coherence_mutex};
    gpu_modified_ranges.Subtract(address, size);
    ++gpu_ranges_subtracts;
}

void BufferCache::SubtractGpuRangeIfUnchanged(VAddr address, u64 size,
                                              u64 write_generation) {
    std::scoped_lock lock{coherence_mutex};
    // A concurrent producer may have dirtied bytes after the readback plan was built.
    // Keep those bytes dirty so a later request can fetch the new GPU result.
    if (coherence_write_generation == write_generation) {
        gpu_modified_ranges.Subtract(address, size);
        ++gpu_ranges_subtracts;
    }
}

static std::atomic<u64> g_prefetch_scan_skips{}, g_publish_repeat_skips{},
    g_request_page_readbacks{};

u64 BufferCache::PublishGpuWrite(VAddr address, u64 size, u64 tick, u32 queue) {
    std::scoped_lock lock{coherence_mutex};
    // ISS_PUBLISH_STATS=1 (diagnosis): how often a write repeats a range already published in the
    // same submission (tick and queue), as the compute rings' many small dispatches may.
    static const bool publish_stats = [] {
        const char* p = std::getenv("ISS_PUBLISH_STATS");
        return p && p[0] == '1';
    }();
    if (publish_stats) {
        struct Entry { VAddr address; u64 size, tick; u32 queue; };
        static std::array<Entry, 64> recent{};
        static u32 next = 0;
        static u64 calls = 0, repeats = 0, contained = 0, bytes = 0;
        ++calls;
        bytes += size;
        for (const auto& e : recent) {
            if (e.tick == tick && e.queue == queue && e.address <= address &&
                address + size <= e.address + e.size) {
                ++contained;
                if (e.address == address && e.size == size) ++repeats;
                break;
            }
        }
        recent[next++ & 63] = {address, size, tick, queue};
        if ((calls & ((1u << 17) - 1)) == 0)
            std::fprintf(stderr, "PUBLISH_STATS calls=%llu same_range=%llu contained=%llu avg_bytes=%llu\n",
                         (unsigned long long)calls, (unsigned long long)repeats,
                         (unsigned long long)contained, (unsigned long long)(bytes / calls));
    }
    // ISS_PUBLISH_REPEAT_SKIP (default on, =0 off): the same range published again in the same
    // submission, with nothing subtracted from the set since, is already in it: the interval set
    // insert (3.7% of the command processor in the forest, 37% of the 200k publishes a second
    // being such repeats) is skipped. The write tick and the generation are updated as before.
    static const bool repeat_skip = [] {
        const char* p = std::getenv("ISS_PUBLISH_REPEAT_SKIP");
        return !p || p[0] != '0';
    }();
    auto& slot = publish_memo[((address >> 12) ^ (size >> 8)) & (publish_memo.size() - 1)];
    if (repeat_skip && slot.address == address && slot.size == size && slot.tick == tick &&
        slot.queue == queue && slot.subtracts == gpu_ranges_subtracts) {
        g_publish_repeat_skips.fetch_add(1, std::memory_order_relaxed);
    } else {
        gpu_modified_ranges.Add(address, size);
        slot = {address, size, tick, queue, gpu_ranges_subtracts};
    }
    const u64 sequence = memory_tracker->SetWriteTick(address, size, tick, queue,
        MappingGeneration(), memory->MappingEpoch());
    ++coherence_write_generation;
    return sequence;
}

// The rasterizer calls this at the actual DMA copy site, after choosing its queue and tick.
// It is deliberately separate from ObtainBuffer's generic "is_written" mark.
void RecordReadbackDma(VAddr dst, VAddr src, u64 size, u64 tick, u32 queue) {
    LogReadbackDiag({.event = "dma_copy", .address = dst, .size = size, .src = src,
                     .dst = dst, .queue = queue, .wait_tick = tick});
}

// Write fault window statistics (see InvalidateMemory), printed by TickFrame.
static std::atomic<u64> fault_window_faults{0};
static std::atomic<u64> fault_window_pages{0};
static std::atomic<u64> fault_streak_extends{0};

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

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
      staging_pool{runtime_.GetStagingPool()},
      graphics_queue{&scheduler_, &runtime_, nullptr, 0}, active{&graphics_queue},
      liverpool{liverpool_}, memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    graphics_queue.stream_buffer = &stream_buffer;
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
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
    resident_bits.resize(((u64{1} << (ADDRESS_SPACE_BITS - block_shift)) + 63) / 64);
    resident_full.resize((resident_bits.size() + 63) / 64);
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    if (instance.HasReadbackQueue()) {
        // The readback queue may belong to the compute-only family (iss::asc_family).
        if (iss::asc_family::readback) {
            iss::asc_family::pool_family = iss::asc_family::family;
        } else if (iss::asc_family::readback_family != ~0u) {
            iss::asc_family::pool_family = iss::asc_family::readback_family;
        }
        readback_queue = std::make_unique<ReadbackQueue>(instance);
        iss::asc_family::pool_family = ~0u;
    }
    upload_worker = std::make_unique<UploadWorker>(tracker);
    scheduler.SetSubmitDependency([this](u64 seq) { upload_worker->Wait(seq); });
    if (const char* prewarm = std::getenv("ISS_PREWARM_LARGE_UPLOAD");
        prewarm && prewarm[0] == '1') {
        const u64 start_ns = Common::Perf::NowNs();
        staging_pool.PrewarmLarge(256_MB, MemoryType::HostUncached, 2);
        LOG_INFO(Render_Vulkan, "Prewarmed two 256 MiB upload buffers in {:.3f} ms",
                 (Common::Perf::NowNs() - start_ns) / 1'000'000.0);
    }
}

BufferCache::~BufferCache() {
    scheduler.SetSubmitDependency({});
}

UploadWorker::UploadWorker(PageManager& page_manager_) : page_manager{page_manager_} {
    thread = std::jthread([this](std::stop_token stoken) { Run(stoken); });
}

UploadWorker::~UploadWorker() {
    thread.request_stop();
    queue_cv.notify_all();
    if (thread.joinable()) {
        thread.join();
    }
}

namespace {
/// The sleeping worker is woken up once this many jobs are queued (each wake up is a system
/// call for the command processor); otherwise it looks for jobs every millisecond, and a
/// submission waiting for jobs does them itself.
constexpr size_t UploadWorkerWakeJobs = 8;
} // Anonymous namespace

u64 UploadWorker::Enqueue(Job&& job) {
    u64 seq;
    bool wake;
    {
        std::scoped_lock lk{queue_mutex};
        queue.push_back(std::move(job));
        seq = enqueued.fetch_add(1, std::memory_order_acq_rel) + 1;
        wake = worker_sleeping && queue.size() >= UploadWorkerWakeJobs;
    }
    if (wake) {
        queue_cv.notify_one();
    }
    return seq;
}

void UploadWorker::Wait(u64 seq) {
    if (completed.load(std::memory_order_acquire) >= seq) {
        return;
    }
    // Jobs the worker took are done once it releases exec_mutex: do those still queued.
    std::scoped_lock exec_lock{exec_mutex};
    if (completed.load(std::memory_order_acquire) < seq) {
        ExecuteQueued();
    }
}

void UploadWorker::Run(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuUploadWorker");
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{queue_mutex};
            worker_sleeping = true;
            queue_cv.wait_for(lk, stoken, std::chrono::milliseconds{1},
                              [this] { return queue.size() >= UploadWorkerWakeJobs; });
            worker_sleeping = false;
            if (queue.empty()) {
                continue;
            }
        }
        std::scoped_lock exec_lock{exec_mutex};
        ExecuteQueued();
    }
}

void UploadWorker::ExecuteQueued() {
    u64 last_seq;
    {
        std::scoped_lock lk{queue_mutex};
        if (queue.empty()) {
            return;
        }
        batch.swap(queue);
        last_seq = enqueued.load(std::memory_order_acquire);
    }
    const u64 start_ns = Common::Perf::NowNs();
    // First the protections of all the jobs, merged (fewer system calls)...
    ranges.clear();
    for (const Job& job : batch) {
        ranges.insert(ranges.end(), job.pages.begin(), job.pages.end());
    }
    std::sort(ranges.begin(), ranges.end());
    u64 num_pages = 0;
    for (size_t i = 0; i < ranges.size();) {
        const u64 first = ranges[i].first;
        u64 end = ranges[i].second;
        for (++i; i < ranges.size() && ranges[i].first <= end; ++i) {
            end = std::max(end, ranges[i].second);
        }
        num_pages += page_manager.RefreshDeferredProtect(first, end);
    }
    // ... then the copies: a guest write to these pages from now on faults.
    auto* const memory = Core::Memory::Instance();
    u64 num_bytes = 0;
    for (const Job& job : batch) {
        for (const Copy& copy : job.copies) {
            memory->CopySparseMemory(copy.src, copy.dst, copy.size);
            num_bytes += copy.size;
        }
        if (job.flush_buffer) {
            job.flush_buffer->Flush(job.flush_offset, job.flush_size);
        }
    }
    Common::Perf::Add(Common::Perf::AsyncUploadJobs, batch.size());
    Common::Perf::Add(Common::Perf::AsyncUploadBytes, num_bytes);
    Common::Perf::Add(Common::Perf::AsyncUploadPages, num_pages);
    batch.clear();
    completed.store(last_seq, std::memory_order_release);
    Common::Perf::Add(Common::Perf::AsyncUploadBusyNs, Common::Perf::NowNs() - start_ns);
}

BufferCache::ReadbackQueue::ReadbackQueue(const Vulkan::Instance& instance_)
    : instance{instance_}, semaphore{instance_}, command_pool{instance_, &semaphore} {}

BufferCache::ReadbackQueue::~ReadbackQueue() {
    // The command buffers and download buffers must not be in use anymore.
    std::ignore = instance.GetReadbackQueue().waitIdle();
}

u64 BufferCache::ReadbackQueue::Submit(std::span<const Job> jobs, std::span<const Wait> waits) {
    // Committed with the tick the submission below signals.
    const vk::CommandBuffer cmdbuf = command_pool.Commit();
    Vulkan::Check(cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    for (const Job& job : jobs) {
        cmdbuf.copyBuffer(job.src, job.dst, job.copies);
    }
    // Makes the copied data visible to host reads once the semaphore is signaled.
    const vk::MemoryBarrier2 to_host = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &to_host,
    });
    Vulkan::Check(cmdbuf.end());

    const u64 signal_tick = semaphore.NextTick();
    const vk::Semaphore signal_semaphore = semaphore.Handle();
    // The waits make the writes of the submissions up to their ticks available and visible to
    // the copies.
    std::array<vk::Semaphore, 4> wait_semaphores;
    std::array<u64, 4> wait_ticks;
    std::array<vk::PipelineStageFlags, 4> wait_stages;
    u32 num_waits = 0;
    for (const Wait& wait : waits) {
        if (wait.tick == 0 || num_waits == wait_semaphores.size()) {
            continue;
        }
        wait_semaphores[num_waits] = wait.semaphore;
        wait_ticks[num_waits] = wait.tick;
        wait_stages[num_waits++] = vk::PipelineStageFlagBits::eTransfer;
    }
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = num_waits,
        .pWaitSemaphoreValues = wait_ticks.data(),
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_tick,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = num_waits,
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    const auto result = instance.GetReadbackQueue().submit(submit_info);
    if (result == vk::Result::eErrorDeviceLost) {
        Vulkan::GpuCrashDiag::ReportDeviceLost(instance, "readback submit");
    }
    ASSERT_MSG(result == vk::Result::eSuccess, "Readback submit failed: {}",
               vk::to_string(result));
    Common::Perf::Add(Common::Perf::ReadbackQueueSubmits);
    return signal_tick;
}

bool BufferCache::UseReadbackQueue() const {
    return readback_queue &&
           Common::Perf::GetTuning().readback_queue.load(std::memory_order_relaxed);
}

void BufferCache::AssertGraphicsReadbackOwner() const {
    if (cp_split_enabled && liverpool && liverpool->IsCurrentCpThread()) {
        ASSERT_MSG(liverpool->CurrentQueue() == 0 && ActiveQueuePtr()->index == 0,
                   "ASC attempted direct G-owned readback; use safe-point control");
    }
}

Vulkan::StagingBufferPool& BufferCache::Staging() const noexcept {
    return ActiveQueuePtr()->runtime->GetStagingPool();
}

void BufferCache::SetCrossQueueTracking(QueueContext* compute) {
    if (compute_queue != nullptr && compute_queue != compute) {
        compute_queue->scheduler->SetSubmitDependency({});
    }
    compute_queue = compute;
    if (compute != nullptr) {
        compute->scheduler->SetSubmitDependency([this](u64 seq) { upload_worker->Wait(seq); });
    }
}

void BufferCache::WaitArenaBinds(Vulkan::SubmitInfo& info) {
    AddResidencyWait(info, PublishedResidencyTick());
}

void BufferCache::AddResidencyWait(Vulkan::SubmitInfo& info, u64 required_tick) {
    if (required_tick == 0 || memory_semaphore.IsFree(required_tick)) {
        return;
    }
    for (u32 i = 0; i < info.num_wait_semas; ++i) {
        if (info.wait_semas[i] == memory_semaphore.Handle()) {
            info.wait_ticks[i] = std::max(info.wait_ticks[i], required_tick);
            return;
        }
    }
    info.AddWait(memory_semaphore.Handle(), required_tick);
}

BufferCache::WindowWriters BufferCache::WindowWritersOf(VAddr start, u64 size, bool* precise) {
    AssertGraphicsReadbackOwner();
    WindowWriters writers;
    if(precise)*precise=false;
    const auto epoch=memory->MappingEpoch();
    const auto generation=MappingGeneration();
    iss::g4::NegativeImageSnapshot negative{};
    bool may_use_pages=false;
    if(iss::readback::PageProvenanceEnabled() && !cp_split_enabled && !compute_queue) {
        negative=texture_cache.ProbeNoRegisteredImage(start,size);
        may_use_pages=negative.result==iss::g4::NegativeImageResult::NoRegisteredPage &&
            UniqueReadbackBacking(*memory,start,size,generation,epoch);
    }
    {
        std::scoped_lock lock{coherence_mutex};
        const bool valid=may_use_pages &&
            memory_tracker->GetPageWriteTicks(start,size,writers.ticks,generation,epoch) &&
            MappingGeneration()==generation && memory->MappingEpoch()==epoch &&
            texture_cache.NegativeImageSnapshotCurrent(negative);
        if (!valid) {
            writers.ticks = memory_tracker->GetWriteTicks(start, size);
        }
        if(precise)*precise=valid;
        if(iss::readback::PageProvenanceEnabled()) {
            static u64 queries{}, accepted{};
            ++queries;accepted+=valid;
            if((queries&8191)==1)std::fprintf(stderr,
                "NATIVE_READBACK_PROVENANCE queries=%llu precise=%llu\n",
                static_cast<unsigned long long>(queries),static_cast<unsigned long long>(accepted));
        }
    }
    if (compute_queue == nullptr) {
        writers.ticks[1] = 0;
    }
    for (u32 queue = 0; queue < 2; ++queue) {
        const u64 tick = writers.ticks[queue];
        if (tick == 0) {
            continue;
        }
        Vulkan::Scheduler& sched = queue == 0 ? scheduler : *compute_queue->scheduler;
        writers.submitted &= tick <= sched.SubmittedTick();
        writers.done &= sched.IsFree(tick);
        writers.current |= tick >= sched.CurrentTick();
    }
    return writers;
}

boost::container::static_vector<BufferCache::ReadbackQueue::Wait, 3> BufferCache::ReadbackWaits(
    const WindowWriters& writers, u64 residency_tick) {
    boost::container::static_vector<ReadbackQueue::Wait, 3> waits;
    if (residency_tick != 0) {
        waits.push_back(ReadbackQueue::Wait{memory_semaphore.Handle(), residency_tick});
    }
    if (writers.ticks[0] != 0) {
        waits.push_back(ReadbackQueue::Wait{scheduler.GetWorkSemaphore()->Handle(), writers.ticks[0]});
    }
    if (writers.ticks[1] != 0 && compute_queue != nullptr) {
        waits.push_back(ReadbackQueue::Wait{compute_queue->scheduler->GetWorkSemaphore()->Handle(),
                                            writers.ticks[1]});
    }
    return waits;
}

void BufferCache::WaitComputeWriter(const WindowWriters& writers) {
    AssertGraphicsReadbackOwner();
    const u64 tick = writers.ticks[1];
    if (tick == 0 || compute_queue == nullptr || compute_queue->scheduler->IsFree(tick)) {
        return;
    }
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive()) {
        ASSERT_MSG(liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() == 0 &&
                       ActiveQueuePtr()->index == 0 &&
                       liverpool->AscOwnerPausedForGraphics() &&
                       tick <= compute_queue->scheduler->SubmittedTick(),
                   "G4 readback cannot submit an unsealed C window writer");
    } else if (tick >= compute_queue->scheduler->CurrentTick()) {
        iss::asc_stats::Timer t{iss::asc_stats::ReadbackWriterFlush};
        FlushComputeQueue();
    }
    scheduler.AddCrossWait(compute_queue->scheduler->GetWorkSemaphore()->Handle(), tick);
    Common::Perf::Add(Common::Perf::CrossWaitsGraphics);
}

void BufferCache::FinishAllQueues() {
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive()) {
        ASSERT_MSG(liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() == 0 &&
                       ActiveQueuePtr()->index == 0 &&
                       liverpool->AscOwnerPausedForGraphics(),
                   "G4 readback requires acknowledged C owner pause");
        if (compute_queue) {
            auto* c = compute_queue->scheduler;
            ASSERT_MSG(!c->HasWork() && c->SubmittedTick() < c->CurrentTick(),
                       "G4 readback cannot remotely flush C's open session");
            const u64 submitted = c->SubmittedTick();
            if (submitted != 0) c->Wait(submitted); // completion, never a remote Flush
        }
        graphics_queue.scheduler->Finish(); // G owns this scheduler
        return;
    }
    if (compute_queue != nullptr && compute_queue != ActiveQueuePtr()) {
        compute_queue->scheduler->Finish();
    }
    if (ActiveQueuePtr() != &graphics_queue) {
        graphics_queue.scheduler->Finish();
    }
    Sched().Finish();
}

void BufferCache::FlushComputeQueue() {
    if (compute_queue == nullptr || !compute_queue->scheduler->HasWork()) {
        return;
    }
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive())
        ASSERT_MSG(liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() > 0 &&
                       ActiveQueuePtr()->index == 1,
                   "G4 remote C scheduler Flush is forbidden");
    // A wait on a graphics tick is submitted only once that tick is (threaded recording submits
    // a flushed session a little later).
    const u64 graphics_tick = compute_queue->scheduler->CrossWaitTick(
        graphics_queue.scheduler->GetWorkSemaphore()->Handle());
    if (graphics_tick != 0) {
        iss::asc_stats::Timer t{iss::asc_stats::WaitGraphicsSubmitted};
        graphics_queue.scheduler->WaitSubmitted(graphics_tick);
    }
    iss::asc_stats::Timer t{iss::asc_stats::ComputeSubmit};
    compute_queue->scheduler->Flush();
    Common::Perf::Add(Common::Perf::ComputeQueueSubmits);
}

void BufferCache::NoteGdsAccess(bool is_written) {
    if (compute_queue == nullptr) {
        return;
    }
    const u32 queue = ActiveQueuePtr()->index;
    const u32 other = queue ^ 1;
    u64 dependency = gds_write_tick[other];
    if (is_written) {
        dependency = std::max(dependency, gds_read_tick[other]);
    }
    if (dependency != 0) {
        QueueContext& other_queue = other == 0 ? graphics_queue : *compute_queue;
        if (!other_queue.scheduler->IsFree(dependency)) {
            if (dependency >= other_queue.scheduler->CurrentTick()) {
                // The session being recorded on the other queue: submitted first (never a wait
                // on a tick that is not).
                iss::asc_stats::Timer t{other == 0 ? iss::asc_stats::GdsFlushGraphics
                                                   : iss::asc_stats::GdsFlushCompute};
                if (other == 0) {
                    graphics_queue.scheduler->Flush();
                } else {
                    FlushComputeQueue();
                }
                Common::Perf::Add(Common::Perf::CrossQueueFlushes);
            }
            Sched().AddCrossWait(other_queue.scheduler->GetWorkSemaphore()->Handle(), dependency);
            Common::Perf::Add(queue == 0 ? Common::Perf::CrossWaitsGraphics
                                         : Common::Perf::CrossWaitsCompute);
        }
    }
    (is_written ? gds_write_tick : gds_read_tick)[queue] = Sched().CurrentTick();
}

void BufferCache::NoteBdaAccess() {
    if (compute_queue == nullptr) {
        return;
    }
    const u32 other = ActiveQueuePtr()->index ^ 1;
    const u64 tick = bda_write_tick[other];
    QueueContext& other_queue = other == 0 ? graphics_queue : *compute_queue;
    if (tick == 0 || other_queue.scheduler->IsFree(tick)) {
        return;
    }
    if (tick >= other_queue.scheduler->CurrentTick()) {
        iss::asc_stats::Timer t{iss::asc_stats::BdaFlush};
        if (other == 0) {
            graphics_queue.scheduler->Flush();
        } else {
            FlushComputeQueue();
        }
        Common::Perf::Add(Common::Perf::CrossQueueFlushes);
    }
    Sched().AddCrossWait(other_queue.scheduler->GetWorkSemaphore()->Handle(), tick);
    Common::Perf::Add(ActiveQueuePtr()->index == 0 ? Common::Perf::CrossWaitsGraphics
                                         : Common::Perf::CrossWaitsCompute);
}

void BufferCache::NoteQueueAccess(VAddr device_addr, u64 size, bool is_written) {
    if (compute_queue == nullptr) {
        return;
    }
    const u32 queue = ActiveQueuePtr()->index;
    if (queue == 0 && !memory_tracker->ComputeTouched(device_addr, size)) {
        // The compute queue never used these words: nothing to wait for, and no need to
        // remember this access (the compute queue waits for all the graphics work recorded so
        // far when it first uses a word).
        return;
    }
    bool untouched = false;
    u64 dependency = memory_tracker->CrossQueueAccess(device_addr, size, queue,
                                                      Sched().CurrentTick(), is_written, untouched);
    QueueContext& other_queue = queue == 0 ? *compute_queue : graphics_queue;
    if (queue == 1 && untouched) {
        // Graphics reads of these words are unknown: everything recorded so far on the graphics
        // queue comes first.
        dependency = std::max(dependency, graphics_queue.scheduler->CurrentTick());
        Common::Perf::Add(Common::Perf::CrossQueueFirstTouches);
    }
    if (dependency == 0 || other_queue.scheduler->IsFree(dependency)) {
        return;
    }
    if (dependency >= other_queue.scheduler->CurrentTick()) {
        // The session being recorded on the other queue: submitted first (never a wait on a
        // tick that is not).
        iss::asc_stats::Timer t{queue == 0 ? iss::asc_stats::RangeFlushCompute
                                : untouched ? iss::asc_stats::FirstTouchFlushGraphics
                                            : iss::asc_stats::RangeFlushGraphics};
        if (queue == 1) {
            graphics_queue.scheduler->Flush();
        } else {
            FlushComputeQueue();
        }
        Common::Perf::Add(Common::Perf::CrossQueueFlushes);
    }
    iss::asc_stats::NoteWait(device_addr, size, queue, is_written, untouched, dependency,
                             other_queue.scheduler->CurrentTick());
    Sched().AddCrossWait(other_queue.scheduler->GetWorkSemaphore()->Handle(), dependency);
    Common::Perf::Add(queue == 0 ? Common::Perf::CrossWaitsGraphics
                                 : Common::Perf::CrossWaitsCompute);
}

bool BufferCache::IsReadbackDone(bool on_readback_queue, u64 tick) {
    if (on_readback_queue) {
        return readback_queue->IsFree(tick);
    }
    return scheduler.IsFree(tick);
}

u64 BufferCache::GpuWriteSeq() const noexcept {
    return memory_tracker->CurrentWriteSeq();
}

void NextUploadEpoch() noexcept {
    g_upload_epoch.fetch_add(1, std::memory_order_acq_rel);
    Common::Perf::Add(Common::Perf::UploadEpochs);
}

void BufferCache::TickFrame() {
    if (static u32 frames = 0; ++frames % 600 == 0) {
        std::fprintf(stderr,
                     "FOREST_SKIPS prefetch_scans=%llu publish_repeats=%llu request_pages=%llu\n",
                     (unsigned long long)g_prefetch_scan_skips.load(),
                     (unsigned long long)g_publish_repeat_skips.load(),
                     (unsigned long long)g_request_page_readbacks.load());
    }
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
    // Write the readbacks the GPU has finished, their pages become readable without a fault.
    RetireReadbacks();
    if (cp_split_enabled) {
        auto* gfx_done = scheduler.GetWorkSemaphore();
        gfx_done->Refresh();
        packet_hazards.RetireCompleted(iss::g4::CpLane::Graphics,
                                        gfx_done->KnownGpuTick());
        if (compute_queue) {
            auto* asc_done = compute_queue->scheduler->GetWorkSemaphore();
            asc_done->Refresh();
            packet_hazards.RetireCompleted(iss::g4::CpLane::Asc,
                                            asc_done->KnownGpuTick());
        }
    }

    // Hot pages are protected again from time to time, to notice those the guest stopped
    // writing every frame (the others become hot again after one fault).
    const u32 frame = g_tracker_frame.fetch_add(1, std::memory_order_relaxed) + 1;
    if (SmallReadMemoEnabled() && (frame % 600) == 0) {
        const auto& stats = small_read_memo.Statistics();
        std::fprintf(stderr,
            "SMALL_READ_MEMO checks=%llu matches=%llu hits=%llu hazards=%llu stores=%llu unstable=%llu\n",
            static_cast<unsigned long long>(stats.checks),
            static_cast<unsigned long long>(stats.matches),
            static_cast<unsigned long long>(stats.hits),
            static_cast<unsigned long long>(stats.hazards),
            static_cast<unsigned long long>(stats.stores),
            static_cast<unsigned long long>(stats.unstable));
    }
    if (frame % 1800 == 0 && iss::phase::Enabled()) {
        std::fprintf(stderr, "COMPUTE_PHASES barriers_skipped=%llu run_flushes=%llu\n",
                     static_cast<unsigned long long>(iss::phase::g_skipped.exchange(0)),
                     static_cast<unsigned long long>(iss::phase::g_flushed.exchange(0)));
    }
    if (frame % 1800 == 0 && fault_window_faults.load(std::memory_order_relaxed) != 0) {
        std::fprintf(stderr, "FAULT_WINDOW faults=%llu streak_extends=%llu pages_unprotected=%llu\n",
                     static_cast<unsigned long long>(fault_window_faults.exchange(0)),
                     static_cast<unsigned long long>(fault_streak_extends.exchange(0)),
                     static_cast<unsigned long long>(fault_window_pages.exchange(0)));
    }
    // Write merge: pages not uploaded for a few seconds forget their content before the guest
    // write (their GPU bytes are then overwritten at their next upload, as without merge).
    if (frame % 64 == 0) {
        memory_tracker->PrunePageSnapshots(frame, 240);
    }
    const u32 cooldown =
        Common::Perf::GetTuning().hot_page_cooldown_frames.load(std::memory_order_relaxed);
    if (cooldown != 0 && frame % cooldown == 0) {
        const u32 cooled = memory_tracker->CoolHotPages();
        Common::Perf::Add(Common::Perf::HotPageDemotions, cooled);
    }
}

void BufferCache::OnGpuThreadWrite(VAddr device_addr, u64 size) {
    if (iss::unified::enabled && IsUnified(device_addr, size)) [[unlikely]] {
        iss::unified::Add(iss::unified::CpWrites);
        iss::unified::Add(iss::unified::CpWriteBytes, size);
        if (UnifiedTouched(device_addr, size)) iss::unified::Add(iss::unified::CpWritesTouched);
    }
    if (G4FeedbackProbeEnabled()) feedback_writer_probe.Invalidate(device_addr, size);
    InvalidatePreparedPacketAccess(device_addr, size);
    NoteHostWrite();
    memory_tracker->OnGpuThreadWrite(device_addr, size);
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (G4FeedbackProbeEnabled()) feedback_writer_probe.Invalidate(device_addr, size);
    // Guest write faults arrive before the write. Invalidate physical aliases
    // before RegionManager can invoke a readback callback.
    InvalidatePreparedPacketAccess(device_addr, size);
    // Guest write faults invalidate at most 8 bytes, unmaps and other invalidations whole
    // ranges. Faults of the command processor thread come with assume_locks.
    const bool is_fault = size <= 8;
    const bool discarded = memory_tracker->InvalidateRegion(
        device_addr, size, is_fault, assume_locks, [this, device_addr, size, assume_locks] {
            ReadMemory(device_addr, size, true, assume_locks);
        });
    if (is_fault && !assume_locks) {
        // Write fault window (ISS_FAULT_WINDOW_KB, power of two, 0 off): the guest fills its
        // per-frame buffers (the particle arena) sequentially, one fault per 4 KB page. The clean
        // pages around the fault that hold no GPU data are unprotected with it.
        static const u64 window = [] {
            const char* setting = std::getenv("ISS_FAULT_WINDOW_KB");
            const u64 kb = setting ? std::strtoull(setting, nullptr, 10) : 0;
            const u64 bytes = kb >= 8 ? std::bit_floor(std::min<u64>(kb, 1024)) * 1024 : 0;
            std::fprintf(stderr, "PORT_TUNING fault_window_bytes=%llu\n",
                         static_cast<unsigned long long>(bytes));
            return bytes;
        }();
        if (window != 0) {
            const u32 pages =
                memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
            fault_window_faults.fetch_add(1, std::memory_order_relaxed);
            fault_window_pages.fetch_add(pages, std::memory_order_relaxed);
        }
        // ISS_FAULT_STREAK_KB (default 256, 0 off): a guest thread writing on through pages it
        // faults on one after the other (a memcpy over memory the GPU wrote or owns, which the
        // window above leaves alone) gets the pages ahead invalidated now, exactly as their
        // faults would (snapshots for the merge, discards, CPU modified, unprotected), 4 pages
        // at a time as a fault does. Every ~2 s the game copies ~11 MB at 0x2093xxxxx-0x209Fxxxxx:
        // ~2870 faults, a 45 ms frame (the "mini frame skips", 2026-10-17). Pages with a
        // readback pending are left protected: they fault as before.
        static const u64 streak_bytes = [] {
            const char* setting = std::getenv("ISS_FAULT_STREAK_KB");
            const u64 kb = setting ? std::strtoull(setting, nullptr, 10) : 256;
            return std::min<u64>(kb, 4096) * 1024;
        }();
        if (streak_bytes != 0) {
            thread_local u64 last_page = 0, next_page = 0, streak = 0;
            const u64 page = device_addr / BYTES_PER_PAGE;
            if (page == last_page + 1 || (next_page != 0 && page == next_page)) ++streak;
            else if (page != last_page) streak = 0;
            last_page = page;
            if (streak >= 3) {
                const VAddr ahead = (page + 1) * BYTES_PER_PAGE;
                constexpr u64 chunk = 4 * BYTES_PER_PAGE;
                for (u64 off = 0; off < streak_bytes; off += chunk) {
                    memory_tracker->InvalidateRegion(ahead + off, chunk, true, false, [] {});
                }
                InvalidatePreparedPacketAccess(ahead, streak_bytes);
                next_page = page + 1 + streak_bytes / BYTES_PER_PAGE;
                fault_streak_extends.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (discarded) {
        Common::Perf::Add(Common::Perf::ReadbackDiscards);
        if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
            // Frame x-ray: the GPU data of the page is forgotten (write discard), or kept on
            // the GPU (write merge). Any thread.
            LOG_INFO(Render_Vulkan, "DUMPX discard {:x} size {} {}", device_addr, size,
                     assume_locks ? "gpu-thread" : "guest");
        }
    }
}

std::pair<VAddr, VAddr> BufferCache::GetReadbackWindow(const Buffer* arena, VAddr device_addr,
                                                       u64 size) {
    const u64 window_size = memory->PreferredReadbackWindow(device_addr,
        Common::Perf::GetTuning().readback_window_bytes.load(std::memory_order_relaxed));
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, window_size), arena->cpu_addr);
    // Page aligned: downloads and protection work on whole pages.
    const VAddr window_end = std::min<VAddr>(
        Common::AlignUp(std::max<VAddr>(window_start + window_size, device_addr + size),
                        BYTES_PER_PAGE),
        arena_end);
    // Preserve broad amortized downloads normally. For a small CPU fault only,
    // a completed requested page need not wait for an unrelated open producer
    // elsewhere in the default window. Download the WHOLE page's dirty bytes;
    // existing sequence checks, pending-copy ordering and protections still apply.
    if (iss::readback::PageProvenanceEnabled() && !cp_split_enabled && !compute_queue &&
        size && size <= 8 && UseReadbackQueue() && window_end-window_start>BYTES_PER_PAGE) {
        const auto page_start=std::max<VAddr>(Common::AlignDown(device_addr,BYTES_PER_PAGE),arena->cpu_addr);
        const auto page_end=std::min<VAddr>(Common::AlignUp(device_addr+size,BYTES_PER_PAGE),arena_end);
        // An older broad copy must retire first; another narrow copy cannot
        // shortcut that ordering and would only add work.
        const bool pending=std::ranges::any_of(pending_readbacks,[&](const auto& copy) {
            return copy.window_start<page_end && page_start<copy.window_end;
        });
        bool precise{};
        const auto page=WindowWritersOf(page_start,page_end-page_start,&precise);
        if (!pending && precise && (page.ticks[0]||page.ticks[1])) {
            if (page.done && !WindowWritersOf(window_start,window_end-window_start).done) {
                static u64 selected{};
                if ((++selected & 1023)==1)
                    std::fprintf(stderr,"NATIVE_READBACK_PAGE selections=%llu bytes=%llu\n",
                        static_cast<unsigned long long>(selected),
                        static_cast<unsigned long long>(page_end-page_start));
                return {page_start,page_end};
            }
        }
    }
    return {window_start, window_end};
}

bool BufferCache::IsRelaxedSite(u64 guest_rip) const {
    if (guest_rip == 0) {
        return false;
    }
    const auto& tuning = Common::Perf::GetTuning();
    if (tuning.relaxed_all.load(std::memory_order_relaxed)) {
        return true;
    }
    const u32 count = tuning.num_relaxed_sites.load(std::memory_order_acquire);
    const u64 base = MemoryPatcher::g_eboot_address;
    if (count == 0 || base == 0 || guest_rip < base) {
        return false;
    }
    const u64 block = (guest_rip - base) & ~u64{0xFF};
    for (u32 i = 0; i < count; ++i) {
        if (tuning.relaxed_sites[i].load(std::memory_order_relaxed) == block) {
            return true;
        }
    }
    return false;
}

void BufferCache::BeginRelaxedReadback(VAddr device_addr, u64 size) {
    AssertGraphicsReadbackOwner();
    // Readbacks the GPU has finished are written first: the page may be up to date already.
    CompleteReadbacks(false);
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    if (!memory_tracker->IsRegionGpuModified(page_start, page_end - page_start)) {
        return;
    }
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    const auto [window_start, window_end] = GetReadbackWindow(arena, device_addr, size);
    // Downloaded again at the next fences while the guest keeps reading it.
    NoteHotWindow(window_start, window_end);

    // The GPU data of the window no pending readback covers is downloaded now, as for a read
    // that waits (nobody waits for it).
    PendingReadback readback{
        .arena_base = arena->cpu_addr,
        .window_start = window_start,
        .window_end = window_end,
    };
    const WindowWriters writers = WindowWritersOf(window_start, window_end - window_start);
    const bool use_queue = UseReadbackQueue() && writers.submitted;
    if (use_queue ? CollectReadback(arena, readback) : RecordReadback(arena, readback)) {
        if (use_queue) {
            const ReadbackQueue::Job job{
                .src = readback.arena_handle,
                .dst = readback.download.buffer->Handle(),
                .copies = readback.copies,
            };
            readback.tick = readback_queue->Submit(std::span{&job, 1}, ReadbackWaits(writers));
            readback.on_readback_queue = true;
        } else {
            WaitComputeWriter(writers);
            readback.tick = scheduler.CurrentTick();
            scheduler.MarkReadbackSession();
            scheduler.Flush();
        }
        pending_readbacks.push_back(std::move(readback));
        Common::Perf::Add(Common::Perf::ReadbackAsync);
    } else {
        memory_tracker->RemovePendingReadback(window_start, window_end);
    }
    // Every readback in flight for these pages gives them its data when it lands, unless the
    // guest or the GPU changed them meanwhile.
    for (VAddr page = page_start; page < page_end; page += BYTES_PER_PAGE) {
        for (auto& pending : pending_readbacks) {
            if (pending.window_start <= page && page < pending.window_end &&
                std::ranges::find(pending.relaxed_pages, page) == pending.relaxed_pages.end()) {
                pending.relaxed_pages.push_back(page);
            }
        }
    }
    // Readable at once: the guest reads what guest memory holds (the previous download).
    memory_tracker->UnmarkRegionAsGpuModified(page_start, page_end - page_start, false);
    Common::Perf::Add(Common::Perf::ReadbackRelaxed);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks,
                             u64 guest_rip) {
    AssertGraphicsReadbackOwner();
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive())
        ASSERT_MSG(liverpool->AscOwnerPausedForGraphics(),
                   "G4 readback entered without an acknowledged C owner pause");
    const u64 request_ns = Common::Perf::NowNs();
    static std::atomic<u64> trace_sequence{};
    const auto trace_id=iss::timeline::enabled?trace_sequence.fetch_add(1,std::memory_order_relaxed)+1:0;
    if(trace_id)iss::timeline::Emit("readback_begin",reinterpret_cast<u64>(this),trace_id,
        device_addr,size,guest_rip,(is_write?1u:0u)|(assume_locks?2u:0u));
    if (ReadbackDiagEnabled()) {
        LogReadbackDiag({.event = "request_start", .rip = guest_rip, .address = device_addr,
                         .size = size, .flags = (is_write ? 1U : 0U) |
                                                  (assume_locks ? 2U : 0U)});
    }
    SCOPE_EXIT {
        if(trace_id)iss::timeline::Emit("readback_end",reinterpret_cast<u64>(this),trace_id,
            device_addr,size,guest_rip,0,Common::Perf::NowNs()-request_ns);
        if (ReadbackDiagEnabled()) {
            LogReadbackDiag({.event = "request_end", .rip = guest_rip,
                             .address = device_addr, .size = size,
                             .duration_ns = Common::Perf::NowNs() - request_ns,
                             .flags = (is_write ? 1U : 0U) | (assume_locks ? 2U : 0U)});
        }
    };
    // Particle trace (Tuning::trace_reads): what the guest gets when it reads GPU written memory
    // in the traced range, logged once the read is complete.
    const auto& trace_tuning = Common::Perf::GetTuning();
    const u64 trace_lo = trace_tuning.trace_reads_lo.load(std::memory_order_relaxed);
    const bool traced = trace_lo != 0 && device_addr + size > trace_lo &&
                        device_addr < trace_tuning.trace_reads_hi.load(std::memory_order_relaxed);
    bool trace_retry = false;
    bool trace_nothing = false;
    SCOPE_EXIT {
        if (!traced) [[likely]] {
            return;
        }
        std::array<u32, 4> words{};
        const u64 n = std::min<u64>(size, sizeof(words));
        memory->CopySparseMemory(device_addr, reinterpret_cast<u8*>(words.data()), n);
        std::string bytes;
        for (u64 i = 0; i < (n + 3) / 4; ++i) {
            bytes += fmt::format("{}{:08x}", i ? "," : "", words[i]);
        }
        LOG_INFO(Render_Vulkan, "PTRACE read f{} {:x}+{} {} {} {} wait {}us{}{}",
                 Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed), device_addr,
                 size, is_write ? "write" : "read", assume_locks ? "gpu-thread" : "guest", bytes,
                 (Common::Perf::NowNs() - request_ns) / 1000, trace_retry ? " retry" : "",
                 trace_nothing ? " nothing" : "");
    };
    if (Common::Perf::FrameDumpActive().load(std::memory_order_relaxed)) [[unlikely]] {
        // Frame x-ray: a guest access to GPU written memory (precise readback). Any thread.
        LOG_INFO(Render_Vulkan, "DUMPX readback {:x} size {} {} {}", device_addr, size,
                 is_write ? "write" : "read", assume_locks ? "gpu-thread" : "guest");
    }
    Common::Perf::ScopedTimer stall_timer{is_write ? Common::Perf::ReadbackWrites
                                                   : Common::Perf::ReadbackReads,
                                          Common::Perf::ReadbackStallNs};
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive()) {
        ASSERT_MSG(assume_locks && liverpool->IsCurrentCpThread() &&
                       liverpool->CurrentQueue() == 0 && ActiveQueuePtr()->index == 0,
                   "G4 readback must run directly in the paused G control");
        ReadbackG4Window(device_addr, size, is_write);
        return;
    }
    if (!assume_locks && !is_write && IsRelaxedSite(guest_rip)) {
        // readback_relaxed_sites: this code reads GPU results without waiting for the GPU (the
        // previous ones, the new ones land in guest memory once the GPU is done).
        liverpool->SendCommand<true>([&] {
            Common::Perf::Add(Common::Perf::ReadbackQueueNs, Common::Perf::NowNs() - request_ns);
            LogReadbackDiag({.event = "guest_enqueue_done", .rip = guest_rip,
                             .address = device_addr, .size = size,
                             .duration_ns = Common::Perf::NowNs() - request_ns,
                             .flags = 4U});
            BeginRelaxedReadback(device_addr, size);
        });
        return;
    }
    if (!assume_locks &&
        Common::Perf::GetTuning().async_readbacks.load(std::memory_order_relaxed)) {
        // The command processor records and submits the download, then goes on with its work
        // while this (guest) thread waits for the GPU. Previously the command processor itself
        // waited for the GPU to become idle, hundreds of times per second in busy scenes.
        u64 wait_tick = 0;
        Vulkan::Semaphore* wait_semaphore = nullptr;
        liverpool->SendCommand<true>([&] {
            Common::Perf::Add(Common::Perf::ReadbackQueueNs, Common::Perf::NowNs() - request_ns);
            LogReadbackDiag({.event = "guest_enqueue_done", .rip = guest_rip,
                             .address = device_addr, .size = size,
                             .duration_ns = Common::Perf::NowNs() - request_ns});
            wait_tick = BeginReadback(device_addr, size, is_write, wait_semaphore);
        });
        if (wait_tick == 0) {
            trace_nothing = true;
            return;
        }
        WaitReadback(device_addr, wait_semaphore, wait_tick);
        for (u32 round = 0;; ++round) {
            // An earlier readback of the window (another queue) may still be in flight: its data
            // must land first, then this one's. The guest thread waits for it (a few times at
            // most), not the command processor.
            Vulkan::Semaphore* next_semaphore = nullptr;
            u64 next_tick = 0;
            const bool may_wait = round < 4;
            liverpool->SendCommand<true>([&] {
                CompleteReadbacks(false);
                if (!memory_tracker->IsRegionGpuModified(device_addr, size)) {
                    if (is_write) {
                        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                    }
                    return;
                }
                if (const PendingReadback* pending =
                        may_wait ? FirstUnfinishedReadback(device_addr, size) : nullptr) {
                    next_semaphore = pending->on_readback_queue ? &readback_queue->Semaphore()
                                                                : scheduler.GetWorkSemaphore();
                    next_tick = pending->tick;
                    return;
                }
                // The GPU wrote this memory again while the download was in flight. Download it
                // synchronously so the request completes instead of faulting again.
                trace_retry = true;
                Common::Perf::Add(Common::Perf::ReadbackAsyncRetries);
                ReadbackSync(device_addr, size, is_write);
            });
            if (next_tick == 0) {
                break;
            }
            WaitReadback(device_addr, next_semaphore, next_tick);
        }
        return;
    }
    const auto flush_request = [this, device_addr, size, is_write, request_ns, guest_rip] {
        // Time spent waiting for the command processor thread to pick up the request.
        Common::Perf::Add(Common::Perf::ReadbackQueueNs, Common::Perf::NowNs() - request_ns);
        LogReadbackDiag({.event = "guest_enqueue_done", .rip = guest_rip,
                         .address = device_addr, .size = size,
                         .duration_ns = Common::Perf::NowNs() - request_ns,
                         .flags = 8U});
        ReadbackSync(device_addr, size, is_write);
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::WaitReadback(VAddr device_addr, Vulkan::Semaphore* semaphore, u64 tick) {
    const u64 wait_start = Common::Perf::NowNs();
    iss::timeline::Emit("readback_wait_begin",reinterpret_cast<u64>(semaphore),tick,device_addr);
    LogReadbackDiag({.event = "wait_start", .address = device_addr,
                     .wait_tick = tick, .flags = readback_queue &&
                         semaphore == &readback_queue->Semaphore() ? 1U : 0U});
    // A readback takes milliseconds: one still waiting after seconds is reported once (hang
    // diagnostics), then the wait goes on.
    if (!semaphore->WaitFor(tick, 2'000'000'000ULL)) {
        const bool on_queue = readback_queue && semaphore == &readback_queue->Semaphore();
        LOG_WARNING(Render_Vulkan,
                    "Readback of {:#x} still waiting after 2 s: tick {} of the {} queue (done "
                    "{}), graphics submitted {} recorded {} done {}",
                    device_addr, tick, on_queue ? "readback" : "graphics",
                    semaphore->KnownGpuTick(), scheduler.SubmittedTick(), scheduler.CurrentTick(),
                    scheduler.GetWorkSemaphore()->KnownGpuTick());
        semaphore->Wait(tick);
    }
    Common::Perf::Add(Common::Perf::ReadbackAsyncWaitNs, Common::Perf::NowNs() - wait_start);
    iss::timeline::Emit("readback_wait_end",reinterpret_cast<u64>(semaphore),tick,device_addr,
        semaphore->KnownGpuTick(),0,0,Common::Perf::NowNs()-wait_start);
    LogReadbackDiag({.event = "wait_end", .address = device_addr,
                     .wait_tick = tick, .duration_ns = Common::Perf::NowNs() - wait_start,
                     .flags = readback_queue &&
                         semaphore == &readback_queue->Semaphore() ? 1U : 0U});
}

const BufferCache::PendingReadback* BufferCache::FirstUnfinishedReadback(VAddr device_addr,
                                                                         u64 size) {
    const auto overlaps = [](const PendingReadback& readback, VAddr start, VAddr end) {
        return readback.window_start < end && start < readback.window_end;
    };
    for (size_t i = 0; i < pending_readbacks.size(); ++i) {
        const PendingReadback& readback = pending_readbacks[i];
        if (!overlaps(readback, device_addr, device_addr + size)) {
            continue;
        }
        if (!IsReadbackDone(readback.on_readback_queue, readback.tick)) {
            return &readback;
        }
        // Done but not written yet: an earlier readback of its window is still in flight.
        for (size_t j = 0; j < i; ++j) {
            const PendingReadback& earlier = pending_readbacks[j];
            if (overlaps(earlier, readback.window_start, readback.window_end) &&
                !IsReadbackDone(earlier.on_readback_queue, earlier.tick)) {
                return &earlier;
            }
        }
    }
    return nullptr;
}

void BufferCache::ReadbackSync(VAddr device_addr, u64 size, bool is_write) {
    AssertGraphicsReadbackOwner();
    // Pending asynchronous readbacks keep their pages protected until they are written.
    CompleteReadbacks(true);
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    const auto [window_start, window_end] = GetReadbackWindow(arena, device_addr, size);
    // CPU writes to the window must not discard GPU data the download below writes back.
    memory_tracker->AddPendingReadback(window_start, window_end);
    DownloadMemory(arena, window_start, window_end - window_start);
    memory_tracker->RemovePendingReadback(window_start, window_end);
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::ReadbackG4Window(VAddr device_addr, u64 size, bool is_write) {
    ASSERT_MSG(cp_split_enabled && liverpool && liverpool->CpSplitActive() &&
                   liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() == 0 &&
                   ActiveQueuePtr()->index == 0 && liverpool->AscOwnerPausedForGraphics(),
               "G4 full-window readback requires acknowledged C owner pause");
    // The C owner has sealed its exact prefix before acknowledging the pause. G can
    // submit its own copy, but must never advance C's open scheduler session.
    constexpr u32 MaxReadbackRounds = 8;
    for (u32 round = 0; round < MaxReadbackRounds; ++round) {
        CompleteReadbacks(false);
        if (!memory_tracker->IsRegionGpuModified(device_addr, size)) {
            if (is_write) memory_tracker->MarkRegionAsCpuModified(device_addr, size);
            return;
        }
        Vulkan::Semaphore* wait_semaphore = nullptr;
        u64 wait_tick = 0;
        if (const PendingReadback* pending = FirstUnfinishedReadback(device_addr, size)) {
            wait_semaphore = pending->on_readback_queue ? &readback_queue->Semaphore()
                                                         : scheduler.GetWorkSemaphore();
            wait_tick = pending->tick;
        } else {
            wait_tick = BeginReadback(device_addr, size, is_write, wait_semaphore);
        }
        ASSERT_MSG(wait_tick != 0 && wait_semaphore,
                   "G4 full-window readback has dirty bytes without a GPU copy");
        WaitReadback(device_addr, wait_semaphore, wait_tick);
    }
    CompleteReadbacks(false);
    ASSERT_MSG(!memory_tracker->IsRegionGpuModified(device_addr, size),
               "G4 full-window readback did not retire its exact GPU writer");
    if (is_write) memory_tracker->MarkRegionAsCpuModified(device_addr, size);
}

bool BufferCache::RecordReadback(const Buffer* arena, PendingReadback& readback) {
    if (!CollectReadback(arena, readback)) {
        return false;
    }
    runtime.CopyBuffer(arena, readback.download.buffer, readback.copies);
    return true;
}

bool BufferCache::KeepRanges() const {
    return Common::Perf::GetTuning().readback_keep_ranges.load(std::memory_order_relaxed);
}

bool BufferCache::CollectReadback(const Buffer* arena, PendingReadback& readback) {
    AssertGraphicsReadbackOwner();
    // Registered before looking at the pages: from now on CPU writes to the window flush instead
    // of discarding the GPU data, which this readback would overwrite when it completes.
    memory_tracker->AddPendingReadback(readback.window_start, readback.window_end);
    readback.arena_handle = arena->Handle();
    readback.record_seq = memory_tracker->CurrentWriteSeq();
    // readback_keep_ranges: the GPU modified ranges stay known until the download has landed in
    // guest memory (FinishReadback), so the GPU copy keeps being read and, with write merge,
    // kept, meanwhile. Otherwise they are forgotten now, so a later download skips them.
    const bool keep_ranges = KeepRanges();

    u64 total_size_bytes = 0;
    const auto add_copy = [&](VAddr start, u64 size) {
        readback.copies.push_back(vk::BufferCopy{
            .srcOffset = start - readback.arena_base,
            .dstOffset = total_size_bytes,
            .size = size,
        });
        // Align up to avoid cache conflicts
        total_size_bytes += Common::AlignUp(size, 64ULL);
    };
    std::vector<std::pair<VAddr, u64>> download_ranges;
    memory_tracker->ForEachDownloadRange<false>(
        readback.window_start, readback.window_end - readback.window_start,
        [&](u64 address, u64 range_size) {
            download_ranges.emplace_back(address, range_size);
        });
    // RegionManager releases its page lock before the coherence snapshot below.
    for (const auto& [address, range_size] : download_ranges) {
        u64 write_generation{};
        for (const auto& [start, end] :
             SnapshotGpuRanges(address, range_size, &write_generation)) {
            if (keep_ranges) {
                // Bytes a pending download already copies are not copied again.
                pending_download_ranges.ForEachNotInRange(start, end - start, add_copy);
            } else {
                add_copy(start, end - start);
            }
        }
        if (!keep_ranges)
            SubtractGpuRangeIfUnchanged(address, range_size, write_generation);
    }
    if (readback.copies.empty()) {
        return false;
    }
    if (keep_ranges) {
        for (const auto& copy : readback.copies) {
            pending_download_ranges.Add(readback.arena_base + copy.srcOffset, copy.size);
        }
    }
    // Held until the readback completes. Power of two sizes so the buffers get reused.
    const u64 request_size = std::max<u64>(std::bit_ceil(total_size_bytes), 64_KB);
    readback.download = staging_pool.Request(request_size, MemoryType::HostCached, 0, true);
    for (auto& copy : readback.copies) {
        copy.dstOffset += readback.download.offset;
    }
    Common::Perf::Add(Common::Perf::ReadbackBytes, total_size_bytes);
    return true;
}

void BufferCache::NoteHotWindow(VAddr start, VAddr end) {
    const u64 now = Common::Perf::NowNs();
    for (auto& window : hot_windows) {
        if (window.start == start && window.end == end) {
            window.last_read_ns = now;
            return;
        }
    }
    if (hot_windows.size() == hot_windows.capacity()) {
        hot_windows.erase(std::ranges::min_element(hot_windows, {}, &HotWindow::last_read_ns));
    }
    hot_windows.push_back(HotWindow{
        .start = start,
        .end = end,
        .last_read_ns = now,
        .prefetch_tick = 0,
        .prefetch_seq = 0,
        .prefetch_on_queue = false,
    });
}

u64 BufferCache::BeginReadback(VAddr device_addr, u64 size, bool is_write,
                               Vulkan::Semaphore*& wait_semaphore) {
    AssertGraphicsReadbackOwner();
    // Readbacks the GPU has finished are written first, the requested data may be among them.
    CompleteReadbacks(false);

    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    auto [window_start, window_end] = GetReadbackWindow(arena, device_addr, size);
    if (!is_write && size <= BYTES_PER_PAGE) {
        // Guest reads (faults), not writes nor large invalidations such as unmaps.
        NoteHotWindow(window_start, window_end);
    }
    // ISS_READBACK_REQUEST_PAGES (default on, =0 off): when the GPU is still writing other parts of the window but
    // the requested pages' writers are done, only those pages are downloaded (their data is
    // complete): the guest does not wait for writes it does not read. With the compute rings
    // native the next frame's particle writes are in flight sooner, and the engine's job threads
    // writing 8 bytes next to them waited ~2.3 ms each for them (rb-q-on 2026-10-18). Without the
    // native rings too: Aftermath 4.06 -> 0.76 ms a frame of readback waits, Parting 4.93 -> 1.39
    // (a2-*, same images).
    static const bool request_pages = [] {
        const char* p = std::getenv("ISS_READBACK_REQUEST_PAGES");
        return !p || p[0] != '0';
    }();
    if (request_pages && !(cp_split_enabled && liverpool && liverpool->CpSplitActive())) {
        const VAddr page_start = std::max<VAddr>(window_start,
                                                 Common::AlignDown(device_addr, BYTES_PER_PAGE));
        const VAddr page_end = std::min<VAddr>(
            window_end, Common::AlignUp(device_addr + size, BYTES_PER_PAGE));
        if (page_start < page_end && (page_start > window_start || page_end < window_end) &&
            !WindowWritersOf(window_start, window_end - window_start).done &&
            WindowWritersOf(page_start, page_end - page_start).done) {
            window_start = page_start;
            window_end = page_end;
            g_request_page_readbacks.fetch_add(1, std::memory_order_relaxed);
        }
    }

    PendingReadback readback{
        .arena_base = arena->cpu_addr,
        .window_start = window_start,
        .window_end = window_end,
    };
    // The command buffers that last wrote the GPU modified pages of the window, and where they
    // are now (statistics: what the guest waits for).
    WindowWriters writers = WindowWritersOf(window_start, window_end - window_start);
    // asc_queue: data written by the compute session being recorded (the guest reading its
    // compute results, e.g. particle feedback). Submitting that session now lets the copy run on
    // the readback queue and wait for the compute work only; recorded by the graphics command
    // buffer it would wait for all the rendering recorded before it. Not when the graphics queue
    // also wrote the window in commands not submitted yet.
    const bool g4_readback = cp_split_enabled && liverpool && liverpool->CpSplitActive();
    if (g4_readback && writers.ticks[1] != 0) {
        ASSERT_MSG(compute_queue && liverpool->AscOwnerPausedForGraphics() &&
                       writers.ticks[1] <= compute_queue->scheduler->SubmittedTick(),
                   "G4 full-window readback has an unsealed C writer");
    }
    if (!g4_readback && compute_queue != nullptr && UseReadbackQueue() && !writers.submitted &&
        writers.ticks[1] != 0 && writers.ticks[0] <= scheduler.SubmittedTick()) {
        static const bool compute_first = [] {
            const char* p = std::getenv("ISS_ASC_READBACK");
            return !p || p[0] != '0';
        }();
        if (compute_first) {
            FlushComputeQueue();
            compute_queue->scheduler->WaitSubmitted(writers.ticks[1]);
            writers = WindowWritersOf(window_start, window_end - window_start);
            Common::Perf::Add(Common::Perf::CrossQueueFlushes);
        }
    }
    // Observation only. The first visit has no exact producer history and remains invalid.
    if (G4FeedbackProbeEnabled() && !is_write && size == 8) {
        const auto capture = feedback_writer_probe.WatchAndCapture(
            device_addr, size, window_start, window_end, writers.ticks);
        static const u64 start_frame = [] {
            const char* p = std::getenv("ISS_G4_FEEDBACK_START_FRAME");
            return p ? std::strtoull(p, nullptr, 10) : 0;
        }();
        const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
        static std::atomic<u32> rows{};
        if (capture.observed && frame >= start_frame &&
            rows.fetch_add(1, std::memory_order_relaxed) < 4096) {
            const auto& snap = capture.snapshot;
            std::fprintf(stderr,
                "G4_FEEDBACK_OBS ns=%llu addr=0x%llx exact_qid=%u exact_gen=%llu "
                "exact_tick=%llu window_gfx_tick=%llu window_asc_tick=%llu "
                "obs_epoch=%llu exact_seq=%llu tracker_seq=%llu current=%u submitted=%u\n",
                static_cast<unsigned long long>(Common::Perf::NowNs()),
                static_cast<unsigned long long>(device_addr), snap.exact.qid,
                static_cast<unsigned long long>(snap.exact.generation),
                static_cast<unsigned long long>(snap.exact.scheduler_tick),
                static_cast<unsigned long long>(snap.window[0].scheduler_tick),
                static_cast<unsigned long long>(snap.window[1].scheduler_tick),
                static_cast<unsigned long long>(capture.observation_epoch),
                static_cast<unsigned long long>(snap.write_sequence),
                static_cast<unsigned long long>(memory_tracker->GetWriteSeq(device_addr, size)),
                static_cast<unsigned>(writers.current), static_cast<unsigned>(writers.submitted));
            const auto& e = capture.exact;
            const auto& p = capture.page;
            const auto& w = capture.word;
            const auto& v = capture.window;
            // Each tuple is sequence/tick/range start/range size/ObtainBuffer caller PC.
            std::fprintf(stderr,
                "G4_FEEDBACK_CAUSE frame=%llu addr=0x%llx "
                "exact=%llu/%llu/0x%llx/%llu/0x%llx "
                "page=%llu/%llu/0x%llx/%llu/0x%llx "
                "word=%llu/%llu/0x%llx/%llu/0x%llx "
                "window=%llu/%llu/0x%llx/%llu/0x%llx\n",
                static_cast<unsigned long long>(frame),
                static_cast<unsigned long long>(device_addr),
                static_cast<unsigned long long>(e.sequence),
                static_cast<unsigned long long>(e.token.scheduler_tick),
                static_cast<unsigned long long>(e.start),
                static_cast<unsigned long long>(e.size),
                static_cast<unsigned long long>(e.caller_pc),
                static_cast<unsigned long long>(p.sequence),
                static_cast<unsigned long long>(p.token.scheduler_tick),
                static_cast<unsigned long long>(p.start),
                static_cast<unsigned long long>(p.size),
                static_cast<unsigned long long>(p.caller_pc),
                static_cast<unsigned long long>(w.sequence),
                static_cast<unsigned long long>(w.token.scheduler_tick),
                static_cast<unsigned long long>(w.start),
                static_cast<unsigned long long>(w.size),
                static_cast<unsigned long long>(w.caller_pc),
                static_cast<unsigned long long>(v.sequence),
                static_cast<unsigned long long>(v.token.scheduler_tick),
                static_cast<unsigned long long>(v.start),
                static_cast<unsigned long long>(v.size),
                static_cast<unsigned long long>(v.caller_pc));
        }
    }
    WindowWriters request_writers{};
    if (ReadbackDiagEnabled()) {
        request_writers = WindowWritersOf(device_addr, size);
        LogReadbackDiag({.event = "begin_readback", .address = device_addr, .size = size,
                         .window_start = window_start, .window_end = window_end,
                         .request_gfx_tick = request_writers.ticks[0],
                         .request_compute_tick = request_writers.ticks[1],
                         .window_gfx_tick = writers.ticks[0],
                         .window_compute_tick = writers.ticks[1],
                         .request_state = ReadbackDiagState(request_writers.done,
                                                            request_writers.submitted,
                                                            request_writers.current),
                         .window_state = ReadbackDiagState(writers.done, writers.submitted,
                                                           writers.current),
                         .queue = ActiveQueuePtr()->index, .flags = is_write ? 1U : 0U});
    }
    if (!is_write && size <= BYTES_PER_PAGE) {
        static const bool trace=[] {const char *p=std::getenv("ISS_FEEDBACK_TRACE");return p&&p[0]=='1';}();
        if(trace){
            const auto precise=WindowWritersOf(device_addr,size);
            static std::FILE *file=[] {const char *p=std::getenv("ISS_DEPENDENCY_LOG");auto *f=p?std::fopen(p,"w"):nullptr;if(f)std::fputs("time_ns,address,size,window_start,window_end,window_gfx_tick,window_compute_tick,request_gfx_tick,request_compute_tick,window_done,window_submitted,window_current,request_done,request_submitted,request_current\n",f);return f;}();
            if(file){std::fprintf(file,"%llu,0x%llx,%llu,0x%llx,0x%llx,%llu,%llu,%llu,%llu,%d,%d,%d,%d,%d,%d\n",(unsigned long long)Common::Perf::NowNs(),(unsigned long long)device_addr,(unsigned long long)size,(unsigned long long)window_start,(unsigned long long)window_end,(unsigned long long)writers.ticks[0],(unsigned long long)writers.ticks[1],(unsigned long long)precise.ticks[0],(unsigned long long)precise.ticks[1],writers.done,writers.submitted,writers.current,precise.done,precise.submitted,precise.current);std::fflush(file);}
        }
        if (writers.done) {
            Common::Perf::Add(Common::Perf::ReadbackProducerDone);
        } else if (writers.submitted) {
            Common::Perf::Add(Common::Perf::ReadbackProducerSubmitted);
        } else if (!writers.current) {
            Common::Perf::Add(Common::Perf::ReadbackProducerQueued);
        } else {
            Common::Perf::Add(Common::Perf::ReadbackProducerCurrent);
        }
    }
    // readback_queue: the copy runs on the readback queue, waiting only for the command buffers
    // that wrote the data, once they are submitted to the driver. Data written by commands not
    // submitted yet (being recorded, or handed to the recording thread) is copied by the graphics
    // command buffer: the readback queue never waits for a signal that is not submitted yet
    // (with threaded recording, such a wait froze the game).
    const bool queue_enabled = UseReadbackQueue();
    bool use_queue = queue_enabled && writers.submitted;
    // Pages of the window downloaded by earlier readbacks stay protected until those complete:
    // wait for the last overlapping one (earlier ones of another queue may complete after it,
    // the guest thread then waits for them too, see ReadMemory).
    const PendingReadback* overlapping = nullptr;
    for (const auto& pending : pending_readbacks) {
        if (pending.window_start < window_end && window_start < pending.window_end) {
            overlapping = &pending;
        }
    }
    bool recorded = false;
    u64 residency_tick = 0;
    if (KeepRanges() && overlapping &&
        overlapping->record_seq >=
            memory_tracker->GetWriteSeq(window_start, window_end - window_start)) {
        // readback_keep_ranges: the ranges are still known, a new download would repeat the
        // pending one, which was recorded after the last GPU write of the window.
        memory_tracker->AddPendingReadback(window_start, window_end);
    } else {
        recorded = CollectReadback(arena, readback);
        if (recorded && g4_readback && use_queue) {
            // A separate Vulkan queue also needs the sparse arena's submitted bind
            // timeline. If any copied span lacks published residency, record on G.
            for (const auto& copy : readback.copies) {
                if (copy.size > std::numeric_limits<u32>::max()) {
                    use_queue = false;
                    break;
                }
                const PublishedArena published = PeekPublishedArena(
                    readback.arena_base + copy.srcOffset, static_cast<u32>(copy.size));
                if (!published || published.arena != arena) {
                    use_queue = false;
                    break;
                }
                residency_tick = std::max(residency_tick, published.residency_tick);
            }
        }
        if (recorded && !use_queue) {
            runtime.CopyBuffer(arena, readback.download.buffer, readback.copies);
        }
    }
    if (!recorded) {
        // Nothing left to download in the window.
        memory_tracker->RemovePendingReadback(window_start, window_end);
        if (!overlapping) {
            LogReadbackDiag({.event = "readback_nothing", .address = device_addr,
                             .size = size, .window_start = window_start,
                             .window_end = window_end, .queue = ActiveQueuePtr()->index});
            // Same as a synchronous readback.
            if (is_write) {
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
            }
            return 0;
        }
        if (overlapping->prefetch) {
            Common::Perf::Add(Common::Perf::ReadbackPrefetchHits);
        }
        LogReadbackDiag({.event = "readback_reuse", .address = device_addr,
                         .size = size, .window_start = window_start,
                         .window_end = window_end, .queue = ActiveQueuePtr()->index,
                         .wait_tick = overlapping->tick,
                         .flags = (overlapping->prefetch ? 1U : 0U) |
                                  (overlapping->on_readback_queue ? 2U : 0U)});
        wait_semaphore = overlapping->on_readback_queue ? &readback_queue->Semaphore()
                                                        : scheduler.GetWorkSemaphore();
        return overlapping->tick;
    }
    const u64 submit_start = Common::Perf::NowNs();
    if (queue_enabled) {
        Common::Perf::Add(use_queue ? Common::Perf::ReadbackQueueReads
                                    : Common::Perf::ReadbackQueueFallbacks);
    }
    if (use_queue) {
        const ReadbackQueue::Job job{
            .src = readback.arena_handle,
            .dst = readback.download.buffer->Handle(),
            .copies = readback.copies,
        };
        readback.tick = readback_queue->Submit(std::span{&job, 1},
                                               ReadbackWaits(writers, residency_tick));
        readback.on_readback_queue = true;
        wait_semaphore = &readback_queue->Semaphore();
    } else {
        WaitComputeWriter(writers);
        readback.tick = scheduler.CurrentTick();
        scheduler.MarkReadbackSession();
        scheduler.Flush();
        wait_semaphore = scheduler.GetWorkSemaphore();
    }
    Common::Perf::Add(Common::Perf::ReadbackSyncNs, Common::Perf::NowNs() - submit_start);
    const u64 wait_tick = readback.tick;
    if(iss::timeline::enabled) {
        iss::timeline::Emit("readback_copy",reinterpret_cast<u64>(this),wait_tick,
            device_addr,reinterpret_cast<u64>(wait_semaphore),window_start,window_end,
            Common::Perf::NowNs()-submit_start);
        iss::timeline::Emit("readback_dependency",reinterpret_cast<u64>(this),wait_tick,
            reinterpret_cast<u64>(&scheduler),writers.ticks[0],writers.ticks[1],use_queue);
    }
    if (ReadbackDiagEnabled()) {
        u64 copy_bytes = 0;
        for (const auto& copy : readback.copies) {
            copy_bytes += copy.size;
        }
        LogReadbackDiag({.event = "readback_record", .address = device_addr,
                         .size = size, .window_start = window_start,
                         .window_end = window_end,
                         .request_gfx_tick = request_writers.ticks[0],
                         .request_compute_tick = request_writers.ticks[1],
                         .window_gfx_tick = writers.ticks[0],
                         .window_compute_tick = writers.ticks[1],
                         .request_state = ReadbackDiagState(request_writers.done,
                                                            request_writers.submitted,
                                                            request_writers.current),
                         .window_state = ReadbackDiagState(writers.done, writers.submitted,
                                                           writers.current),
                         .queue = ActiveQueuePtr()->index, .wait_tick = wait_tick,
                         .copies = static_cast<u32>(readback.copies.size()),
                         .copy_bytes = copy_bytes,
                         .duration_ns = Common::Perf::NowNs() - submit_start,
                         .flags = (use_queue ? 1U : 0U) |
                                  (overlapping ? 2U : 0U) |
                                  (is_write ? 4U : 0U)});
    }
    pending_readbacks.push_back(std::move(readback));
    Common::Perf::Add(Common::Perf::ReadbackAsync);
    return wait_tick;
}

void BufferCache::RetireReadbacks() {
    AssertGraphicsReadbackOwner();
    if (pending_readbacks.empty()) {
        return;
    }
    // Cheap check of the known GPU progress first (called between the packets of the command
    // processor); the readback queue is queried when it has readbacks and none is known done.
    bool any_done = false;
    bool any_on_queue = false;
    for (const auto& readback : pending_readbacks) {
        if (readback.on_readback_queue) {
            any_on_queue = true;
            any_done = readback_queue->Semaphore().IsFree(readback.tick);
        } else {
            any_done = scheduler.GetWorkSemaphore()->IsFree(readback.tick);
        }
        if (any_done) {
            break;
        }
    }
    if (!any_done && any_on_queue) {
        readback_queue->Semaphore().Refresh();
        any_done = std::ranges::any_of(pending_readbacks, [this](const PendingReadback& readback) {
            return readback.on_readback_queue && readback_queue->Semaphore().IsFree(readback.tick);
        });
    }
    if (any_done) {
        CompleteReadbacks(false);
    }
}

void BufferCache::PrefetchReadbacks() {
    AssertGraphicsReadbackOwner();
    // A fence-time hint has no acknowledged C pause. Guest faults use the
    // full-window path after the C owner seals its prefix instead.
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive()) return;
    {
        // async_written_uploads: the upload worker applies the read protection of the ranges
        // the GPU writes. Called at fences (OnFence), before the label lets the guest read them:
        // those protections are in place first, so its reads fault and wait for the GPU data.
        const auto& tuning = Common::Perf::GetTuning();
        if (upload_worker && tuning.async_uploads.load(std::memory_order_relaxed) &&
            tuning.async_written_uploads.load(std::memory_order_relaxed)) {
            upload_worker->WaitIdle();
        }
    }
    if (hot_windows.empty()) {
        return;
    }
    // ISS_PREFETCH_PER_SUBMIT=1 (default off): the windows are scanned again only once a
    // submission went out since the last scan. The compute rings fence thousands of times a
    // second (the forest: ~110k scans for ~11k prefetches recorded, 5.6% of the command
    // processor), and without a new submission no new GPU result can be ready; a guest read of
    // data recorded since then faults and takes the precise path as before.
    // Measured 2026-10-18: CP -0.6 ms in the forest, but the guest waits for readbacks longer
    // where the prefetch matters (Aftermath 1.4 -> 4.8 ms a frame, Parting 3.1 -> 4.5): off.
    static const bool per_submit = [] {
        const char* p = std::getenv("ISS_PREFETCH_PER_SUBMIT");
        return p && p[0] == '1';
    }();
    if (per_submit) {
        const u64 current = scheduler.CurrentTick();
        if (current == prefetch_scan_tick) {
            g_prefetch_scan_skips.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        prefetch_scan_tick = current;
    }
    const auto& tuning = Common::Perf::GetTuning();
    // Only precise readbacks keep the pages protected until the data is written.
    if (!tuning.async_readbacks.load(std::memory_order_relaxed) ||
        !tuning.readback_prefetch.load(std::memory_order_relaxed) ||
        EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise) {
        return;
    }
    RetireReadbacks();

    // Windows the guest read back recently are likely read again after this fence: download what
    // the GPU wrote there now, in one submission. When the copies complete before the guest
    // reads, the pages are written and unprotected without any fault.
    constexpr u64 HotWindowLifetimeNs = 1'000'000'000ULL;
    constexpr u64 MaxPrefetchBytes = 8_MB;
    const u64 now = Common::Perf::NowNs();
    const u64 tick = scheduler.CurrentTick();
    // With the readback queue, a window whose writer is already submitted to the driver is
    // copied there, waiting only for that writer. The others (written by commands not submitted
    // yet) are copied by the command buffer being recorded, which is then submitted.
    const bool queue_enabled = UseReadbackQueue();
    const size_t first_queued = pending_readbacks.size();
    std::array<u64, 2> queue_wait_ticks{};
    bool recorded_on_queue = false;
    bool recorded_on_graphics = false;
    u64 prefetched_bytes = 0;
    u32 skipped_inflight = 0;
    u32 skipped_clean = 0;
    u32 skipped_sequence = 0;
    for (auto it = hot_windows.begin(); it != hot_windows.end();) {
        if (prefetched_bytes >= MaxPrefetchBytes) {
            break;
        }
        if (now - it->last_read_ns > HotWindowLifetimeNs) {
            it = hot_windows.erase(it);
            continue;
        }
        HotWindow& window = *it;
        ++it;
        const VAddr start = window.start;
        const VAddr end = window.end;
        // One prefetch in flight per window, and only when the GPU wrote something there since.
        if (window.prefetch_tick != 0 &&
            !(window.prefetch_on_queue
                  ? readback_queue->IsFree(window.prefetch_tick)
                  : scheduler.GetWorkSemaphore()->IsFree(window.prefetch_tick))) {
            ++skipped_inflight;
            continue;
        }
        if (!GpuRangesIntersect(start, end - start) ||
            !memory_tracker->IsRegionGpuModified(start, end - start)) {
            ++skipped_clean;
            continue;
        }
        const u64 window_write_seq = memory_tracker->GetWriteSeq(start, end - start);
        if (KeepRanges() && window_write_seq <= window.prefetch_seq) {
            // readback_keep_ranges: the ranges stay known after a download, the writes since
            // the last prefetch tell whether there is anything new to download.
            ++skipped_sequence;
            continue;
        }
        const WindowWriters writers = WindowWritersOf(start, end - start);
        const bool on_queue = queue_enabled && writers.submitted;
        LogReadbackDiag({.event = "prefetch_candidate", .address = start,
                         .size = end - start, .window_start = start, .window_end = end,
                         .window_gfx_tick = writers.ticks[0],
                         .window_compute_tick = writers.ticks[1],
                         .window_state = ReadbackDiagState(writers.done, writers.submitted,
                                                           writers.current),
                         .queue = ActiveQueuePtr()->index, .flags = on_queue ? 1U : 0U});
        const auto* arena = GetArena(start >> block_shift, (end - 1) >> block_shift);
        PendingReadback readback{
            .tick = tick,
            .arena_base = arena->cpu_addr,
            .window_start = start,
            .window_end = end,
            .prefetch = true,
            .on_readback_queue = on_queue,
        };
        if (!(on_queue ? CollectReadback(arena, readback) : RecordReadback(arena, readback))) {
            memory_tracker->RemovePendingReadback(start, end);
            LogReadbackDiag({.event = "prefetch_nothing", .address = start,
                             .size = end - start, .window_start = start,
                             .window_end = end, .queue = ActiveQueuePtr()->index});
            continue;
        }
        prefetched_bytes += readback.download.size;
        window.prefetch_tick = tick;
        window.prefetch_seq = window_write_seq;
        window.prefetch_on_queue = on_queue;
        if (on_queue) {
            queue_wait_ticks[0] = std::max(queue_wait_ticks[0], writers.ticks[0]);
            queue_wait_ticks[1] = std::max(queue_wait_ticks[1], writers.ticks[1]);
            recorded_on_queue = true;
        } else {
            WaitComputeWriter(writers);
            recorded_on_graphics = true;
        }
        pending_readbacks.push_back(std::move(readback));
        Common::Perf::Add(Common::Perf::ReadbackPrefetches);
    }
    if (recorded_on_graphics) {
        scheduler.MarkReadbackSession();
        scheduler.Flush();
        Common::Perf::Add(Common::Perf::ReadbackPrefetchFlushes);
    }
    if (ReadbackDiagEnabled()) {
        for (size_t i = first_queued; i < pending_readbacks.size(); ++i) {
            const auto& readback = pending_readbacks[i];
            if (readback.on_readback_queue) {
                continue; // Its final tick comes from the queue submit below.
            }
            u64 copy_bytes = 0;
            for (const auto& copy : readback.copies) {
                copy_bytes += copy.size;
            }
            LogReadbackDiag({.event = "prefetch_record", .address = readback.window_start,
                             .size = readback.window_end - readback.window_start,
                             .window_start = readback.window_start,
                             .window_end = readback.window_end,
                             .queue = ActiveQueuePtr()->index, .wait_tick = readback.tick,
                             .copies = static_cast<u32>(readback.copies.size()),
                             .copy_bytes = copy_bytes});
        }
        LogReadbackDiag({.event = "prefetch_summary",
                         .address = static_cast<u64>(hot_windows.size()),
                         .request_state = skipped_inflight,
                         .window_state = skipped_clean, .queue = skipped_sequence,
                         .copies = static_cast<u32>(pending_readbacks.size() - first_queued),
                         .copy_bytes = prefetched_bytes,
                         .flags = (recorded_on_graphics ? 1U : 0U) |
                                  (recorded_on_queue ? 2U : 0U)});
    }
    if (!recorded_on_queue) {
        return;
    }
    // One submission for the windows of the readback queue, after the last graphics submission
    // that wrote them (all submitted to the driver already).
    boost::container::small_vector<ReadbackQueue::Job, 16> jobs;
    for (size_t i = first_queued; i < pending_readbacks.size(); ++i) {
        const auto& readback = pending_readbacks[i];
        if (!readback.on_readback_queue) {
            continue;
        }
        jobs.push_back(ReadbackQueue::Job{
            .src = readback.arena_handle,
            .dst = readback.download.buffer->Handle(),
            .copies = readback.copies,
        });
    }
    WindowWriters queue_writers;
    queue_writers.ticks = queue_wait_ticks;
    const u64 queue_tick = readback_queue->Submit(jobs, ReadbackWaits(queue_writers));
    for (size_t i = first_queued; i < pending_readbacks.size(); ++i) {
        auto& readback = pending_readbacks[i];
        if (!readback.on_readback_queue) {
            continue;
        }
        readback.tick = queue_tick;
        for (auto& window : hot_windows) {
            if (window.start == readback.window_start && window.end == readback.window_end) {
                window.prefetch_tick = queue_tick;
            }
        }
        if (ReadbackDiagEnabled()) {
            u64 copy_bytes = 0;
            for (const auto& copy : readback.copies) {
                copy_bytes += copy.size;
            }
            LogReadbackDiag({.event = "prefetch_record", .address = readback.window_start,
                             .size = readback.window_end - readback.window_start,
                             .window_start = readback.window_start,
                             .window_end = readback.window_end,
                             .queue = ActiveQueuePtr()->index, .wait_tick = queue_tick,
                             .copies = static_cast<u32>(readback.copies.size()),
                             .copy_bytes = copy_bytes, .flags = 1U});
        }
    }
}

void BufferCache::CompleteReadbacks(bool wait) {
    AssertGraphicsReadbackOwner();
    if (wait) {
        // All of them, in order.
        while (!pending_readbacks.empty()) {
            PendingReadback& readback = pending_readbacks.front();
            if (!IsReadbackDone(readback.on_readback_queue, readback.tick)) {
                const u64 sync_start = Common::Perf::NowNs();
                if (readback.on_readback_queue) {
                    readback_queue->Semaphore().Wait(readback.tick);
                } else {
                    scheduler.Wait(readback.tick);
                }
                Common::Perf::Add(Common::Perf::ReadbackSyncNs,
                                  Common::Perf::NowNs() - sync_start);
            }
            FinishReadback(0);
            pending_readbacks.pop_front();
        }
        return;
    }
    if (pending_readbacks.empty()) {
        return;
    }
    // The readbacks the GPU has finished, in any order (the graphics queue and the readback queue
    // progress independently), except after an unfinished earlier readback of the same window:
    // the later download has the newer data and must land last. The GPU progress of each queue
    // is queried once.
    scheduler.GetWorkSemaphore()->Refresh();
    if (readback_queue) {
        readback_queue->Semaphore().Refresh();
    }
    const auto done = [this](const PendingReadback& readback) {
        return readback.on_readback_queue ? readback_queue->Semaphore().IsFree(readback.tick)
                                          : scheduler.GetWorkSemaphore()->IsFree(readback.tick);
    };
    for (size_t i = 0; i < pending_readbacks.size();) {
        const PendingReadback& readback = pending_readbacks[i];
        bool blocked = false;
        for (size_t j = 0; j < i && !blocked; ++j) {
            const PendingReadback& earlier = pending_readbacks[j];
            blocked = earlier.window_start < readback.window_end &&
                      readback.window_start < earlier.window_end;
        }
        if (blocked || !done(readback)) {
            ++i;
            continue;
        }
        if (i != 0) {
            Common::Perf::Add(Common::Perf::ReadbackOutOfOrder);
        }
        FinishReadback(i);
        pending_readbacks.erase(pending_readbacks.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

void BufferCache::FinishReadback(size_t index) {
    AssertGraphicsReadbackOwner();
    PendingReadback& readback = pending_readbacks[index];
    const u64 finish_start = Common::Perf::NowNs();
    if (KeepRanges()) {
        for (const auto& copy : readback.copies) {
            pending_download_ranges.Subtract(readback.arena_base + copy.srcOffset, copy.size);
        }
    }
    // Pages of the window still GPU modified. While the readback is pending, CPU writes do not
    // discard GPU data, so these include every page it downloaded.
    boost::container::small_vector<std::pair<VAddr, VAddr>, 16> pages;
    memory_tracker->ForEachDownloadRange<false>(
        readback.window_start, readback.window_end - readback.window_start,
        [&](u64 address, u64 range_size) { pages.emplace_back(address, address + range_size); });

    if (!readback.copies.empty()) {
        readback.download.Invalidate();
        for (const auto& copy : readback.copies) {
            const VAddr copy_start = readback.arena_base + copy.srcOffset;
            const VAddr copy_end = copy_start + copy.size;
            const u8* src = readback.download.mapped + (copy.dstOffset - readback.download.offset);
            for (const auto& [start, end] : pages) {
                const VAddr lo = std::max(copy_start, start);
                const VAddr hi = std::min(copy_end, end);
                if (lo < hi) {
                    memory->TryWriteBacking(std::bit_cast<u8*>(lo), src + (lo - copy_start),
                                            hi - lo);
                }
            }
        }
        // Guest memory changed under copies taken before (after the writes: a copy that saw the
        // new count saw the new data).
        NoteHostWrite();
    }

    // The pages are now up to date in guest memory, except those written again by the GPU since
    // the download was recorded, or downloaded by a later readback: they stay GPU modified.
    const bool keep_ranges = KeepRanges();
    const auto keep_page = [&](VAddr page) {
        if (keep_ranges ? memory_tracker->GetWriteSeq(page, BYTES_PER_PAGE) > readback.record_seq
                        : GpuRangesIntersect(page, BYTES_PER_PAGE)) {
            return true;
        }
        for (size_t i = index + 1; i < pending_readbacks.size(); ++i) {
            const auto& later = pending_readbacks[i];
            if (page >= later.window_end || page + BYTES_PER_PAGE <= later.window_start) {
                continue;
            }
            for (const auto& copy : later.copies) {
                const VAddr start = later.arena_base + copy.srcOffset;
                if (page < start + copy.size && start < page + BYTES_PER_PAGE) {
                    return true;
                }
            }
        }
        return false;
    };
    const auto unmark = [&](VAddr run_start, VAddr run_end) {
        memory_tracker->UnmarkRegionAsGpuModified(run_start, run_end - run_start, false);
        if (keep_ranges) {
            // Guest memory has these bytes now.
            SubtractGpuRange(run_start, run_end - run_start);
        }
    };
    for (const auto& [start, end] : pages) {
        VAddr run_start = start;
        for (VAddr page = start; page < end; page += BYTES_PER_PAGE) {
            if (keep_page(page)) {
                if (run_start < page) {
                    unmark(run_start, page);
                }
                run_start = page + BYTES_PER_PAGE;
            }
        }
        if (run_start < end) {
            unmark(run_start, end);
        }
    }
    // readback_relaxed_sites: pages a guest read was let through meanwhile (no longer GPU
    // modified) get the new data too, unless the guest or the GPU changed them since.
    if (!readback.relaxed_pages.empty() && !readback.copies.empty()) {
        bool refreshed_any = false;
        for (const VAddr page : readback.relaxed_pages) {
            const bool refreshed = memory_tracker->WithCleanPageLocked(page, [&] {
                for (const auto& copy : readback.copies) {
                    const VAddr copy_start = readback.arena_base + copy.srcOffset;
                    const VAddr lo = std::max(copy_start, page);
                    const VAddr hi = std::min(copy_start + copy.size, page + BYTES_PER_PAGE);
                    if (lo < hi) {
                        const u8* src =
                            readback.download.mapped + (copy.dstOffset - readback.download.offset);
                        memory->TryWriteBacking(std::bit_cast<u8*>(lo), src + (lo - copy_start),
                                                hi - lo);
                    }
                }
            });
            refreshed_any |= refreshed;
            Common::Perf::Add(refreshed ? Common::Perf::ReadbackRelaxedRefreshes
                                        : Common::Perf::ReadbackRelaxedSkips);
        }
        if (refreshed_any) {
            NoteHostWrite();
        }
    }
    // A write request marks its range CPU modified once its own wait is over (ReadMemory), only
    // if the range is no longer GPU modified: a page cannot be readable-protected but writable.
    memory_tracker->RemovePendingReadback(readback.window_start, readback.window_end);
    if (readback.download.buffer) {
        staging_pool.FreeDeferred(readback.download);
    }
    Common::Perf::Add(Common::Perf::ReadbackSyncNs, Common::Perf::NowNs() - finish_start);
    if (ReadbackDiagEnabled()) {
        u64 copy_bytes = 0;
        for (const auto& copy : readback.copies) {
            copy_bytes += copy.size;
        }
        LogReadbackDiag({.event = "readback_finish", .address = readback.window_start,
                         .size = readback.window_end - readback.window_start,
                         .window_start = readback.window_start,
                         .window_end = readback.window_end,
                         .queue = ActiveQueuePtr()->index, .wait_tick = readback.tick,
                         .copies = static_cast<u32>(readback.copies.size()),
                         .copy_bytes = copy_bytes,
                         .duration_ns = Common::Perf::NowNs() - finish_start,
                         .flags = (readback.on_readback_queue ? 1U : 0U) |
                                  (readback.prefetch ? 2U : 0U)});
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    AssertGraphicsReadbackOwner();
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
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
    std::vector<std::pair<VAddr, u64>> download_ranges;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 count) {
        download_ranges.emplace_back(address, count);
    });
    for (const auto& [address, count] : download_ranges) {
        u64 write_generation{};
        for (const auto& [first, last] :
             SnapshotGpuRanges(address, count, &write_generation))
            add_download(first, last);
        SubtractGpuRangeIfUnchanged(address, count, write_generation);
    }
    if (total_size_bytes == 0) {
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    {
        // Submits all recorded GPU work and waits for the GPU to become idle.
        const u64 sync_start = Common::Perf::NowNs();
        scheduler.MarkReadbackSession();
        FinishAllQueues();
        Common::Perf::Add(Common::Perf::ReadbackSyncNs, Common::Perf::NowNs() - sync_start);
        Common::Perf::Add(Common::Perf::ReadbackBytes, total_size_bytes);
    }

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    NoteHostWrite();
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
}

void BufferCache::PrefetchLookup(VAddr device_addr, u64 size) const noexcept {
    const auto& stream_copies = ComputeScratchActive() ? compute_scratch.stream_copies
                                                      : this->stream_copies;
    const auto& sync_memos = ComputeScratchActive() ? compute_scratch.sync_memos
                                                   : this->sync_memos;
    // The keys of ObtainBuffer (the size it is given is 32 bits).
    const u64 key = device_addr ^ (static_cast<u64>(static_cast<u32>(size)) << 40);
    const void* entry;
    size_t entry_size;
    if (size <= STREAM_THRESHOLD) {
        entry = &stream_copies[(key * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
        entry_size = sizeof(StreamCopy);
    } else {
        entry = &sync_memos[((key ^ 0x5bd1e995ULL) * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
        entry_size = sizeof(SyncMemo);
    }
    PrefetchLine(entry);
    PrefetchLine(static_cast<const u8*>(entry) + entry_size - 1);
    memory_tracker->PrefetchRegion(device_addr);
}

void BufferCache::PrefetchStreamSource(VAddr device_addr, u64 size) noexcept {
    const auto& stream_copies = ComputeScratchActive() ? compute_scratch.stream_copies
                                                      : this->stream_copies;
    if (size == 0 || size > STREAM_THRESHOLD) {
        return;
    }
    const auto& tuning = Common::Perf::GetTuning();
    // As ObtainBuffer: a copy reused, or pages bound in place, read nothing.
    if (tuning.stream_dedup.load(std::memory_order_relaxed)) {
        const u64 key = device_addr ^ (size << 40);
        const auto& reuse = stream_copies[(key * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
        if (reuse.address == device_addr && reuse.size == size &&
            reuse.tick == scheduler.CurrentTick() && reuse.epoch == CurrentUploadEpoch() &&
            reuse.host_write_gen == CurrentHostWriteGen()) {
            return;
        }
    }
    if (tuning.stream_static.load(std::memory_order_relaxed) &&
        memory_tracker->IsRegionCpuClean(
            device_addr, size, tuning.stream_static_frames.load(std::memory_order_relaxed))) {
        return;
    }
    constexpr u64 MaxLines = 16;
    const VAddr first = device_addr & ~VAddr{63};
    const u64 lines = std::min<u64>((device_addr + size - first + 63) / 64, MaxLines);
    for (u64 i = 0; i < lines; ++i) {
        PrefetchLine(reinterpret_cast<const void*>(first + i * 64));
    }
}

// ISS_OBTAIN_CENSUS=1: ObtainBuffer outcomes on the command processor thread (calls, cycles,
// bytes, and how many addresses lie in the command buffer segment being processed).
namespace {
bool ObtainCensusEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_OBTAIN_CENSUS");
        return value && value[0] == '1';
    }();
    return enabled;
}
enum ObtainOutcome : u32 { OSmallMemo, OStreamReuse, OArenaClean, OStreamCopy, OLarge, ONum };
struct ObtainCensus {
    std::array<u64, ONum> calls{}, cycles{}, bytes{}, in_dcb{};
    std::array<u64, ONum> written{};
    u64 total{};
};
ObtainCensus g_obtain_census;
struct ObtainTimer {
    bool on;
    u64 start;
    VAddr address;
    u32 size;
    bool is_written;
    u32 outcome = OLarge;
    ObtainTimer(VAddr a, u32 s, bool w, bool enabled)
        : on{enabled},
          start{on ? __rdtsc() : 0}, address{a}, size{s}, is_written{w} {}
    ~ObtainTimer() {
        if (!on) {
            return;
        }
        auto& c = g_obtain_census;
        ++c.calls[outcome];
        c.cycles[outcome] += __rdtsc() - start;
        c.bytes[outcome] += size;
        c.in_dcb[outcome] += address >= g_iss_dcb_lo && address < g_iss_dcb_hi;
        c.written[outcome] += is_written;
        if ((++c.total & ((1u << 20) - 1)) != 0) {
            return;
        }
        static const char* const names[] = {"small_memo", "stream_reuse", "arena_clean",
                                            "stream_copy", "large"};
        for (u32 i = 0; i < ONum; ++i) {
            std::fprintf(stderr,
                         "OBTAIN_CENSUS %s calls=%llu cycles/call=%.0f bytes/call=%.0f "
                         "in_dcb=%llu written=%llu\n",
                         names[i], (unsigned long long)c.calls[i],
                         c.calls[i] ? double(c.cycles[i]) / c.calls[i] : 0.0,
                         c.calls[i] ? double(c.bytes[i]) / c.calls[i] : 0.0,
                         (unsigned long long)c.in_dcb[i], (unsigned long long)c.written[i]);
        }
    }
};
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
                                                        bool* known_unwritten) {
    if (iss::repeat_census::enabled) [[unlikely]]
        iss::repeat_census::Note(iss::repeat_census::ObtainBuffer, device_addr, size, is_written);
    if (iss::unified::enabled) [[unlikely]] {
        if (unified_end == 0 && !unified_failed) {
            EnableUnified();
        }
        if (IsUnified(device_addr, size)) {
            return ObtainUnified(device_addr, size, is_written, known_unwritten);
        }
    }
    ObtainTimer obtain_timer{device_addr, size, is_written,
                             ObtainCensusEnabled() && liverpool && liverpool->IsCurrentCpThread()};
    auto& stream_copies = ComputeScratchActive() ? compute_scratch.stream_copies
                                                : this->stream_copies;
    auto& stream_clean_sample = ComputeScratchActive() ? compute_scratch.stream_clean_sample
                                                      : this->stream_clean_sample;
    auto& sync_memos = ComputeScratchActive() ? compute_scratch.sync_memos : this->sync_memos;
    auto& large_read_memos = ComputeScratchActive() ? compute_scratch.large_read_memos
                                                   : this->large_read_memos;
    iss::diagnostics::Scope phase_scope(iss::diagnostics::ObtainBuffer);
    if (known_unwritten) *known_unwritten = false;
    const u64 writer_caller_pc = G4FeedbackProbeEnabled() && is_written
                                     ? reinterpret_cast<u64>(_ReturnAddress()) : 0;
    // Reuse the ordinary CPU-clean small arena binding across ALL shader
    // families. Keep current hazards live; never return a remembered "no write"
    // result after a later upload/draw touched the arena in this same tick.
    const auto& read_tuning = Common::Perf::GetTuning();
    const bool memo_small_read = SmallReadMemoEnabled() && !is_written &&
        !is_texel_buffer && device_addr && size && size <= STREAM_THRESHOLD &&
        !cp_split_enabled && !compute_queue && ActiveQueuePtr()->index == 0 &&
        liverpool && liverpool->IsCurrentCpThread() &&
        read_tuning.stream_static.load(std::memory_order_relaxed) &&
        read_tuning.hot_page_faults.load(std::memory_order_relaxed) == 0 &&
        !read_tuning.readback_write_merge.load(std::memory_order_relaxed);
    const auto capture_small_read = [&]() -> std::optional<iss::buffers::SmallReadStamp> {
        iss::buffers::SmallReadStamp stamp{
            .address = device_addr, .size = size, .mapping_generation = MappingGeneration(),
            .arena_generation = arena_binding_generation.load(std::memory_order_acquire),
            .gpu_write_sequence = memory_tracker->CurrentWriteSeq(),
            .upload_epoch = CurrentUploadEpoch(), .host_write_generation = CurrentHostWriteGen(),
            .scheduler_tick = Sched().CurrentTick(), .queue = ActiveQueuePtr()->index,
            .stable_frames = read_tuning.stream_static_frames.load(std::memory_order_relaxed),
        };
        if (!memory_tracker->GetCpuVersions(device_addr, size, stamp.cpu_versions))
            return std::nullopt;
        return stamp;
    };
    const auto small_read_before = memo_small_read ? capture_small_read() : std::nullopt;
    if (small_read_before) {
        if (const auto* cached = small_read_memo.Find(*small_read_before)) {
            if (!Rt().IsBufferAccessed(cached->handle, cached->offset, size)) {
                small_read_memo.Hit();
                obtain_timer.outcome = OSmallMemo;
                Common::Perf::Add(Common::Perf::StreamArenaBinds);
                if (known_unwritten) *known_unwritten = true;
                return {cached->handle, cached->offset};
            }
            small_read_memo.Hazard();
        }
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !HasGpuData(device_addr, size)) {
        const auto& tuning = Common::Perf::GetTuning();
        const u32 stable_frames = tuning.stream_static_frames.load(std::memory_order_relaxed);
        StreamCopy* reuse = nullptr;
        u64 epoch = 0;
        u64 host_write_gen = 0;
        if (tuning.stream_dedup.load(std::memory_order_relaxed)) {
            // A copy of this range made in this command buffer and epoch is reused first: the
            // cheapest answer (checking the GPU copy below reads the page tracking tables).
            // Read before copying: a change while copying makes the entry unusable.
            epoch = CurrentUploadEpoch();
            host_write_gen = CurrentHostWriteGen();
            const u64 key = device_addr ^ (static_cast<u64>(size) << 40);
            reuse = &stream_copies[(key * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
            if (reuse->address == device_addr && reuse->size == size &&
                reuse->queue == ActiveQueuePtr()->index && reuse->tick == Sched().CurrentTick() &&
                reuse->epoch == epoch && reuse->host_write_gen == host_write_gen) {
                Common::Perf::Add(Common::Perf::StreamReuses);
                Common::Perf::Add(Common::Perf::StreamReuseBytes, size);
                obtain_timer.outcome = OStreamReuse;
                if (known_unwritten) *known_unwritten = true;
                return {&Stream(), reuse->offset};
            }
        }
        // asc_queue: the compute queue copies small CPU data instead of binding the shared GPU
        // copy in place. In place it would inherit, through the 256 KB tracking words, the
        // graphics uploads of neighbouring data and wait for the graphics queue (measured:
        // ~800 such waits per frame serialize the compute ring behind rendering). The guest
        // bytes are current here (no GPU data), so the copy has no cross queue dependency.
        static const bool compute_copies = [] {
            const char* p = std::getenv("ISS_ASC_STREAM_COPY");
            return !p || p[0] != '0';
        }();
        if (tuning.stream_static.load(std::memory_order_relaxed) &&
            !(compute_copies && OnComputeQueue())) {
            if (!is_texel_buffer &&
                memory_tracker->IsRegionCpuClean(device_addr, size, stable_frames)) {
                // The GPU copy of these pages is up to date: no copy or upload is needed to read
                // them there (the binding is then tracked for barriers like any arena range).
                const u64 first_block = device_addr >> block_shift;
                const u64 last_block = (device_addr + size - 1) >> block_shift;
                const auto* arena = GetArena(first_block, last_block);
                const u64 arena_offset = arena->Offset(device_addr);
                // Unless the range was written since the last barrier (an upload): reading it in
                // place would need a full pipeline barrier, a copy needs none.
                if (!Rt().IsBufferAccessed(arena, arena_offset, size)) {
                    Common::Perf::Add(Common::Perf::StreamArenaBinds);
                    obtain_timer.outcome = OArenaClean;
                    EnsureResident(arena, first_block, last_block);
                    NoteQueueAccess(device_addr, size, false);
                    if (small_read_before) {
                        if (const auto after = capture_small_read())
                            small_read_memo.Remember(*small_read_before, *after, arena, arena_offset);
                    }
                    if (known_unwritten) *known_unwritten = true;
                    return {arena, arena_offset};
                }
                Common::Perf::Add(Common::Perf::StreamBarrierAvoided);
            }
        } else if ((++stream_clean_sample & 15) == 0 && !is_texel_buffer &&
                   memory_tracker->IsRegionCpuClean(device_addr, size, stable_frames)) {
            // Statistics only (one binding in 16): copies stream_static would avoid.
            Common::Perf::Add(Common::Perf::StreamCleanCopies, 16);
        }
        StreamBuffer& stream = Stream();
        const auto [data, offset] = stream.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream.Commit();
        Common::Perf::Add(Common::Perf::StreamCopies);
        Common::Perf::Add(Common::Perf::StreamCopyBytes, size);
        obtain_timer.outcome = OStreamCopy;
        if (reuse) {
            // The copy stays valid until the command buffer it was committed in completes.
            *reuse = StreamCopy{
                .address = device_addr,
                .tick = Sched().CurrentTick(),
                .epoch = epoch,
                .host_write_gen = host_write_gen,
                .offset = offset,
                .size = size,
                .queue = ActiveQueuePtr()->index,
            };
        }
        if (known_unwritten) *known_unwritten = true;
        return {&stream, offset};
    }
    if (const u32 probe = Common::Perf::GetTuning().probe_host_import.load(
            std::memory_order_relaxed);
        probe != probed_host_import) [[unlikely]] {
        probed_host_import = probe;
        if (probe != 0) {
            ProbeHostImport(device_addr);
        }
    }
    // Sync memo: a read only range bound again while no guest write touched its regions since
    // it was last found up to date needs no synchronization (large vertex and index buffers
    // bound by most draws; the checks walk the page tracking tables of the whole range).
    const auto& tuning = Common::Perf::GetTuning();
    const bool memo_candidate = !is_written && !is_texel_buffer &&
                                tuning.sync_memo.load(std::memory_order_relaxed) &&
                                tuning.hot_page_faults.load(std::memory_order_relaxed) == 0;
    SyncMemo* memo = nullptr;
    std::array<u64, 2> versions{};
    bool have_versions = false;
    if (memo_candidate) {
        const u64 key = device_addr ^ (static_cast<u64>(size) << 40) ^ 0x5bd1e995ULL;
        memo = &sync_memos[(key * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
        have_versions = memory_tracker->GetCpuVersions(device_addr, size, versions);
        if (have_versions && memo->address == device_addr && memo->size == size &&
            memo->versions == versions) {
            Common::Perf::Add(Common::Perf::SyncMemoHits);
            NoteQueueAccess(device_addr, size, false);
            return {memo->arena, memo->arena->Offset(device_addr)};
        }
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    const bool large_read_memo_enabled = tuning.large_read_memo.load(std::memory_order_relaxed);
    LargeReadMemo* large = nullptr;
    u64 epoch = 0;
    u64 host_gen = 0;
    bool large_hit = false;
    bool have_large_versions = false;
    if (large_read_memo_enabled && !is_written && size >= 64_MB &&
        tuning.hot_page_faults.load(std::memory_order_relaxed) == 0 &&
        !tuning.readback_write_merge.load(std::memory_order_relaxed)) {
        epoch = CurrentUploadEpoch();
        host_gen = CurrentHostWriteGen();
        Common::Perf::Add(Common::Perf::LargeReadMemoChecks);
        const u64 key = (device_addr >> 12) ^ (u64{size} * 0x9E3779B97F4A7C15ULL);
        large = &large_read_memos[(key * 0x9E3779B97F4A7C15ULL) >> 60];
        large_hit = large->valid && large->arena == arena && large->address == device_addr &&
                    large->size == size && large->epoch == epoch && large->host_write_gen == host_gen &&
                    large->queue == ActiveQueuePtr()->index &&
                    memory_tracker->CheckLargeCpuVersions(device_addr, size, large->versions, true);
        if (!large_hit) {
            large->valid = false;
            have_large_versions = memory_tracker->CheckLargeCpuVersions(device_addr, size,
                                                                         large->versions, false);
        }
    }
    if (large_hit) {
        Common::Perf::Add(Common::Perf::LargeReadMemoHits);
        // GPU writes to an aliased image remain visible even if the CPU copy
        // has not changed. Only the CPU dirty-page scan/upload is bypassed.
        if (is_texel_buffer && !OnComputeQueue()) {
            SynchronizeMemoryFromImage(arena, device_addr, size);
        }
    } else {
        SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
        if (large && have_large_versions) {
            large->arena = arena;
            large->address = device_addr;
            large->size = size;
            large->epoch = epoch;
            large->host_write_gen = host_gen;
            large->queue = ActiveQueuePtr()->index;
            large->valid = true;
        }
    }
    // Cross queue dependencies before the write tick of this access is recorded.
    NoteQueueAccess(device_addr, size, is_written);
    if (is_written) {
        // Publish dirty bytes and their producer tick as one logical metadata update.
        const u64 write_seq = PublishGpuWrite(device_addr, size, Sched().CurrentTick(),
                                              ActiveQueuePtr()->index);
        if (G4FeedbackProbeEnabled()) {
            const auto identity = serial_producer_identity;
            const bool matched = identity.generation != 0 && identity.qid < 57 &&
                ((ActiveQueuePtr()->index == 0 && identity.owner == iss::g4::CpOwner::Graphics) ||
                 (ActiveQueuePtr()->index == 1 && identity.owner == iss::g4::CpOwner::Asc1 &&
                  identity.qid == 1));
            feedback_writer_probe.Publish(device_addr, size,
                iss::g4::ProducerToken{identity.owner, identity.qid,
                                       matched ? identity.generation : 0,
                                       Sched().CurrentTick()}, write_seq, ActiveQueuePtr()->index,
                writer_caller_pc, is_texel_buffer);
        }
    } else if (memo && have_versions) {
        // Versions read before the synchronization: a write during it changes them.
        Common::Perf::Add(Common::Perf::SyncMemoMisses);
        *memo = SyncMemo{
            .address = device_addr,
            .arena = arena,
            .versions = versions,
            .size = size,
        };
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::SyncReadInPlace(VAddr device_addr, u32 size) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, device_addr, size, false, false);
    NoteQueueAccess(device_addr, size, false);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsUnified(device_addr, size)) {
        // The GPU copies the image from the guest's bytes in place (in order with its writes).
        iss::unified::Add(iss::unified::ImageSources);
        return ObtainBuffer(device_addr, size, false);
    }
    if (HasGpuData(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuClean(VAddr address, u64 size) const noexcept {
    return memory_tracker->IsRegionCpuClean(address, size, 0);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

std::optional<BufferCache::PreparedReadToken> BufferCache::CapturePreparedRead(
    VAddr address, u32 size, const Buffer* buffer, u64 offset) {
    if (!address || !size || address > std::numeric_limits<u64>::max() - size ||
        !buffer || buffer == &Stream() || cp_split_enabled || compute_queue ||
        ActiveQueuePtr()->index != 0 || !liverpool || !liverpool->IsCurrentCpThread())
        return std::nullopt;
    const auto& tuning = Common::Perf::GetTuning();
    if (tuning.hot_page_faults.load(std::memory_order_relaxed) != 0 ||
        tuning.readback_write_merge.load(std::memory_order_relaxed)) return std::nullopt;
    PreparedReadToken token{.address = address, .size = size, .buffer = buffer, .offset = offset};
    const auto mapping_snapshot = [&](std::array<u64, 2>& versions) {
        const u64 first = address >> 24, last = (address + size - 1) >> 24;
        if (last - first > 1) return false;
        std::scoped_lock lock{coherence_mutex};
        for (u64 region = first; region <= last; ++region) {
            auto it = std::ranges::find(prepared_read_mapping_versions, region,
                                       &std::pair<u64, u64>::first);
            if (it == prepared_read_mapping_versions.end()) {
                prepared_read_mapping_versions.emplace_back(region, 1);
                it = std::prev(prepared_read_mapping_versions.end());
            }
            versions[region - first] = it->second;
        }
        return true;
    };
    if (!mapping_snapshot(token.mapping_versions) ||
        !memory_tracker->GetCpuVersions(address, size, token.cpu_versions)) return std::nullopt;
    token.writer_sequence = memory_tracker->GetWriteSeq(address, size);
    if (memory_tracker->IsRegionCpuModified(address, size) || HasGpuData(address, size) ||
        memory->ClampRangeSize(address, size) != size) return std::nullopt;
    const auto [current, current_offset] = PeekArena(address, size);
    if (current != buffer || current_offset != offset ||
        Rt().IsBufferAccessed(buffer, offset, size, false)) return std::nullopt;
    if (size <= STREAM_THRESHOLD) {
        if (tuning.stream_dedup.load(std::memory_order_relaxed)) {
            const u64 key = address ^ (static_cast<u64>(size) << 40);
            const auto& copy = stream_copies[(key * 0x9E3779B97F4A7C15ULL) >> (64 - 12)];
            if (copy.address == address && copy.size == size && copy.queue == 0 &&
                copy.tick == Sched().CurrentTick() && copy.epoch == CurrentUploadEpoch() &&
                copy.host_write_gen == CurrentHostWriteGen()) return std::nullopt;
        }
        if (!tuning.stream_static.load(std::memory_order_relaxed) ||
            !memory_tracker->IsRegionCpuClean(address, size,
                tuning.stream_static_frames.load(std::memory_order_relaxed))) return std::nullopt;
    }
    std::array<u64, 2> cpu_after{}, mapping_after{};
    if (!memory_tracker->GetCpuVersions(address, size, cpu_after) ||
        cpu_after != token.cpu_versions || !mapping_snapshot(mapping_after) ||
        mapping_after != token.mapping_versions ||
        memory_tracker->GetWriteSeq(address, size) != token.writer_sequence) return std::nullopt;
    return token;
}

bool BufferCache::ValidatePreparedRead(const PreparedReadToken& token) {
    const auto current = CapturePreparedRead(token.address, token.size, token.buffer, token.offset);
    return current && current->cpu_versions == token.cpu_versions &&
           current->mapping_versions == token.mapping_versions &&
           current->writer_sequence == token.writer_sequence;
}

bool BufferCache::CommitPreparedRead(const PreparedReadToken& token) {
    if (!ValidatePreparedRead(token)) return false;
    NoteQueueAccess(token.address, token.size, false);
    return true;
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (IsUnified(addr, size)) {
        return UnifiedTouched(addr, size);
    }
    return memory_tracker->IsRegionGpuModified(addr, size);
}

std::pair<const Buffer*, u64> BufferCache::PeekArena(VAddr device_addr, u32 size) {
    if (size == 0 || device_addr == 0 || device_addr > std::numeric_limits<u64>::max() - size) {
        return {nullptr, 0};
    }
    std::scoped_lock lock{residency_mutex};
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    if (!IsResident(first_block, last_block)) {
        return {nullptr, 0};
    }
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    const auto* arena = address_space[first_page];
    if (!arena || address_space[last_page] != arena) {
        return {nullptr, 0};
    }
    return {arena, arena->Offset(device_addr)};
}

BufferCache::PublishedArena BufferCache::PeekPublishedArena(VAddr device_addr,
                                                            u32 size) const {
    if (size == 0 || device_addr == 0 || device_addr > std::numeric_limits<u64>::max() - size)
        return {};
    std::scoped_lock lock{residency_mutex};
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    if (!IsResident(first_block, last_block)) return {};
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    const auto* arena = address_space[first_page];
    if (!arena || address_space[last_page] != arena) return {};
    const auto required_tick = residency_publication.RequiredSubmittedTick(arena);
    if (!required_tick) return {};
    return {arena, arena->Offset(device_addr), *required_tick,
            published_bda_g_tick.load(std::memory_order_acquire)};
}

std::optional<BufferCache::AscArenaPreflight> BufferCache::PrepareArenaForAsc(
    VAddr device_addr, u32 size) {
    if (!cp_split_enabled || !liverpool || !liverpool->CpSplitActive() ||
        !liverpool->IsCurrentCpThread() || liverpool->CurrentQueue() != 0 ||
        ActiveQueuePtr()->index != 0 || size == 0 || device_addr == 0 ||
        device_addr > std::numeric_limits<VAddr>::max() - size)
        return std::nullopt;
    if (!liverpool->AscOwnerPausedForGraphics()) return std::nullopt;
    if (compute_queue) {
        auto* c = compute_queue->scheduler;
        const u64 submitted = c->SubmittedTick();
        if (c->HasWork() || submitted >= c->CurrentTick() ||
            (submitted != 0 && !c->IsFree(submitted))) return std::nullopt;
    }
    const u64 mapping_before = MappingGeneration();
    // This path is for CPU-owned bytes only. A GPU producer requires a readback
    // transaction, which must never be silently replaced by a CPU upload.
    if (HasGpuData(device_addr, size)) return std::nullopt;
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    if ((first_block >> blocks_per_arena_page_shift) !=
        (last_block >> blocks_per_arena_page_shift)) return std::nullopt;
    // C's acknowledged pause and completed prefix above keep every ASC qid from
    // changing the arena while this G-only safe-point prepares the upload.
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, device_addr, size, false, false);
    if (mapping_before != MappingGeneration() || HasGpuData(device_addr, size))
        return std::nullopt;
    return AscArenaPreflight{Sched().CurrentTick(), mapping_before};
}

std::string BufferCache::DescribePages(VAddr device_addr, u32 size) {
    if (size == 0 || device_addr == 0) {
        return "none";
    }
    const bool merge =
        Common::Perf::GetTuning().readback_write_merge.load(std::memory_order_relaxed);
    return fmt::format("cpu{} gpu{} ranges{} owned{} pending{} tick{}",
                       memory_tracker->IsRegionCpuModified(device_addr, size) ? 1 : 0,
                       memory_tracker->IsRegionGpuModified(device_addr, size) ? 1 : 0,
                       GpuRangesIntersect(device_addr, size) ? 1 : 0,
                       merge && memory_tracker->IsGpuOwned(device_addr, size) ? 1 : 0,
                       memory_tracker->IsReadbackPending(device_addr, size) ? 1 : 0,
                       memory_tracker->GetWriteTick(device_addr, size));
}

bool BufferCache::HasGpuData(VAddr addr, size_t size) {
    if (IsUnified(addr, size)) {
        // GPU written guest bytes are not on the CPU before the GPU runs: the callers stay
        // on their GPU paths (and native memos treat the range as written).
        return UnifiedTouched(addr, size);
    }
    if (memory_tracker->IsRegionGpuModified(addr, size)) {
        return true;
    }
    // Write merge: bytes the GPU wrote in pages whose GPU state a guest write discarded are only
    // on the GPU (whether or not their ranges are still known: a page stays owned until an
    // upload finds nothing of the GPU's to keep in it).
    return Common::Perf::GetTuning().readback_write_merge.load(std::memory_order_relaxed) &&
           memory_tracker->IsGpuOwned(addr, size);
}

void CopyGuestPage(VAddr page, u8* dest) {
    Core::Memory::Instance()->CopySparseMemory(page, dest, BYTES_PER_PAGE);
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    ASSERT_MSG(!cp_split_enabled || !liverpool || !liverpool->CpSplitActive() ||
                   (ActiveQueuePtr()->index == 0 && liverpool &&
                    liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() == 0),
               "ASC sparse arena creation requires G safe-point RPC");
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    // The command processor is the only thread that creates or migrates arenas (readbacks run
    // there through SendCommand): without a split worker it reads the table without the lock.
    if (!cp_split_enabled && liverpool && liverpool->IsCurrentCpThread()) {
        const auto* arena = address_space[first_page];
        if (arena && address_space[last_page] == arena) {
            return arena;
        }
    }
    std::unique_lock lock{residency_mutex};
    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            lock.unlock();
            Buffer prepared{instance, base_block << block_shift,
                            num_pages << ARENA_PAGE_BITS, MemoryType::Sparse};
            lock.lock();
            ASSERT_MSG(!address_space[first_page] && !address_space[last_page],
                       "Sparse arena table changed during G-only creation");
            const auto* new_arena = &arenas.emplace_back(std::move(prepared));
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
            arena_binding_generation.fetch_add(1, std::memory_order_release);
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    lock.unlock();
    Buffer prepared{instance, first_addr, total_size, MemoryType::Sparse};
    lock.lock();
    ASSERT_MSG(address_space[first_page] == first_arena &&
                   address_space[last_page] == last_arena,
               "Sparse arena table changed during G-only migration");
    auto* new_arena = &arenas.emplace_back(std::move(prepared));
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        NoteArenaBindLocked(new_arena);
        auto* bind = BindsForArena(new_arena);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    arena_binding_generation.fetch_add(1, std::memory_order_release);
    return new_arena;
}

void BufferCache::NoteArenaBindLocked(const Buffer* arena) {
    residency_publication.NoteBind(arena, [this] { return memory_semaphore.NextTick(); });
}

static constexpr u64 BitsMask(u64 first_bit, u64 count) {
    return (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << first_bit;
}

bool BufferCache::IsResident(u64 first_block, u64 last_block) const {
    const u64 first_word = first_block / 64;
    const u64 last_word = last_block / 64;
    if (first_word == last_word) {
        const u64 mask = BitsMask(first_block % 64, last_block - first_block + 1);
        return (resident_bits[first_word] & mask) == mask;
    }
    const u64 first_mask = BitsMask(first_block % 64, 64 - first_block % 64);
    if ((resident_bits[first_word] & first_mask) != first_mask) {
        return false;
    }
    const u64 last_mask = BitsMask(0, last_block % 64 + 1);
    if ((resident_bits[last_word] & last_mask) != last_mask) {
        return false;
    }
    // The words in between must be fully resident: a bit per word (bindings covering the whole
    // guest memory span tens of thousands of blocks).
    for (u64 word = first_word + 1; word < last_word;) {
        const u64 bit = word % 64;
        const u64 count = std::min<u64>(64 - bit, last_word - word);
        const u64 mask = BitsMask(bit, count);
        if ((resident_full[word / 64] & mask) != mask) {
            return false;
        }
        word += count;
    }
    return true;
}

void BufferCache::MarkResident(u64 first_block, u64 last_block) {
    for (u64 block = first_block; block <= last_block;) {
        const u64 bit = block % 64;
        const u64 count = std::min<u64>(64 - bit, last_block - block + 1);
        const u64 word = block / 64;
        resident_bits[word] |= BitsMask(bit, count);
        if (resident_bits[word] == ~u64{0}) {
            resident_full[word / 64] |= u64{1} << (word % 64);
        }
        block += count;
    }
}

namespace {
// iss: the device memory of new resident ranges comes from 64 MB chunks handed out in order
// (residency only grows and is never freed), and the next chunk is allocated on its own thread
// ahead of need: the command processor's vkAllocateMemory calls cost ~5-20 ms each when the
// game reached new memory (frames of 27-49 ms in the city, 2026-10-19). Ranges of half a chunk
// or more get their own allocation as before. ISS_RESIDENCY_POOL=0 off.
struct ResidencyPool {
    static constexpr u64 ChunkSize = 64ull << 20;
    std::mutex mutex;
    vk::DeviceMemory chunk{};
    u64 chunk_used{};
    vk::DeviceMemory spare{};
    bool refilling{};
    u64 chunks{}, sync_allocations{};
};
ResidencyPool residency_pool;

std::pair<vk::DeviceMemory, u64> AllocateResidency(vk::Device device, u32 type_index, u64 size) {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_RESIDENCY_POOL");
        return !p || p[0] != '0';
    }();
    const auto allocate = [device, type_index](u64 bytes) {
        return Vulkan::Check(device.allocateMemory(vk::MemoryAllocateInfo{
            .allocationSize = bytes,
            .memoryTypeIndex = type_index,
        }));
    };
    if (!enabled || size > ResidencyPool::ChunkSize / 2) {
        return {allocate(size), 0};
    }
    auto& pool = residency_pool;
    std::unique_lock lock{pool.mutex};
    if (!pool.chunk || pool.chunk_used + size > ResidencyPool::ChunkSize) {
        if (pool.spare) {
            pool.chunk = std::exchange(pool.spare, vk::DeviceMemory{});
        } else {
            ++pool.sync_allocations;
            pool.chunk = allocate(ResidencyPool::ChunkSize);
        }
        pool.chunk_used = 0;
        ++pool.chunks;
        std::fprintf(stderr, "RESIDENCY_POOL chunk=%llu synchronous=%llu\n",
                     (unsigned long long)pool.chunks, (unsigned long long)pool.sync_allocations);
        if (!pool.refilling) {
            pool.refilling = true;
            std::thread([allocate] {
                const auto memory = allocate(ResidencyPool::ChunkSize);
                std::scoped_lock refill_lock{residency_pool.mutex};
                residency_pool.spare = memory;
                residency_pool.refilling = false;
            }).detach();
        }
    }
    const u64 offset = pool.chunk_used;
    pool.chunk_used += size;
    return {pool.chunk, offset};
}
} // namespace

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    ASSERT_MSG(!cp_split_enabled || !liverpool || !liverpool->CpSplitActive() ||
                   (ActiveQueuePtr()->index == 0 && liverpool &&
                    liverpool->IsCurrentCpThread() && liverpool->CurrentQueue() == 0),
               "ASC sparse residency requires G safe-point RPC");
    u32 resident_blocks{};
    IntervalList bind_ranges;
    // Residency only grows, and only the command processor makes ranges resident.
    if (!cp_split_enabled && liverpool && liverpool->IsCurrentCpThread() &&
        IsResident(first_block, last_block)) {
        return;
    }
    {
        std::scoped_lock lock{residency_mutex};
        if (IsResident(first_block, last_block)) return;
        resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
            resident_blocks += end - start;
            bind_ranges.Add({start, end});
        });
    }

    if (bind_ranges.Empty()) {
        return;
    }

    const auto [device_memory, memory_base] = AllocateResidency(
        instance.GetDevice(), arena_memory_type_index, resident_blocks << block_shift);

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        Staging().Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    const auto arena_bda = arena->BufferDeviceAddress();

    struct PreparedBind {
        Backing backing;
        vk::SparseMemoryBind bind;
    };
    std::vector<PreparedBind> prepared;
    u64 memory_offset = memory_base;
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const vk::SparseMemoryBind bind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        };
        prepared.push_back({backing, bind});
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena_bda + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    {
        std::scoped_lock lock{residency_mutex};
        // G is the only residency preparer in split mode. Guest callbacks never create
        // sparse backing, so the gaps cannot change while Vulkan allocation runs.
        ASSERT_MSG(!IsResident(first_block, last_block),
                   "Sparse backing changed during G-only preparation");
        auto* binds = BindsForArena(arena);
        for (const auto& item : prepared) {
            resident_ranges.Add(item.backing);
            NoteArenaBindLocked(arena);
            binds->binds.push_back(item.bind);
        }
        MarkResident(first_block, last_block);
    }

    staging.Flush();
    // The page table is read by the shaders of both queues: a copy on the active queue, waited
    // for by the other one when it reads it next (NoteGdsAccess tracks it like the GDS).
    Rt().CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    if (cp_split_enabled) {
        // G is the only sparse preparer in split mode. The BDA write is separate
        // from bindSparse and must be sealed on the G scheduler before C consumes it.
        published_bda_g_tick.store(Sched().CurrentTick(), std::memory_order_release);
    }
    if (compute_queue != nullptr) {
        bda_write_tick[ActiveQueuePtr()->index] = Sched().CurrentTick();
    }
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    auto& deferred_pages = ComputeScratchActive() ? compute_scratch.deferred_pages
                                                  : this->deferred_pages;
    iss::diagnostics::Scope phase_scope(iss::diagnostics::SyncMemory);
    const auto& tuning = Common::Perf::GetTuning();
    const bool hash_hot_pages = tuning.hot_page_faults.load(std::memory_order_relaxed) != 0 &&
                                tuning.hot_page_hash.load(std::memory_order_relaxed);
    const auto hash_page = [this](VAddr page_addr) {
        return memory->HashSparseMemory(page_addr, BYTES_PER_PAGE);
    };
    if (hash_hot_pages) {
        // Hot pages unchanged since their last upload need none in this epoch.
        memory_tracker->SkipUnchangedHotPages(device_addr, size, hash_page);
    }
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const auto add_copy = [&](VAddr addr, u64 copy_size) {
        copies.emplace_back(total_size_bytes, addr, copy_size);
        total_size_bytes += copy_size;
    };
    // Write merge: pages whose GPU data a guest write discarded only upload the bytes outside
    // those the GPU wrote, and those the guest changed since (compared with the content they
    // had before that write).
    const bool merge = tuning.readback_write_merge.load(std::memory_order_relaxed) &&
                       memory_tracker->HasPageSnapshots();
    const std::vector<std::pair<VAddr, VAddr>> gpu_ranges =
        merge ? SnapshotGpuRanges(device_addr, size) : std::vector<std::pair<VAddr, VAddr>>{};
    boost::container::small_vector<std::pair<VAddr, VAddr>, 4> guest_owned;
    // Upload worker: the pages uploaded here are protected again by the worker, which copies
    // them only afterwards (no write merge or hot pages). Ranges the GPU writes too
    // (async_written_uploads): their read protection is applied by the worker as well, before
    // the submission of the commands writing them, so before the guest can wait for them.
    const bool async_upload =
        !merge && !hash_hot_pages && tuning.hot_page_faults.load(std::memory_order_relaxed) == 0 &&
        tuning.async_uploads.load(std::memory_order_relaxed) &&
        (!is_written || tuning.async_written_uploads.load(std::memory_order_relaxed));
    if (async_upload) {
        deferred_pages.clear();
        PageManager::DeferProtect(&deferred_pages);
    }
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written, [&](u64 addr, u64 size, auto&& keep_gpu_owned) {
            if (!merge) {
                add_copy(addr, size);
                return;
            }
            VAddr run_start = addr;
            const VAddr end = addr + size;
            PageSnapshot before;
            for (VAddr page = addr; page < end; page += BYTES_PER_PAGE) {
                if (!memory_tracker->TakePageSnapshot(page, before)) {
                    continue;
                }
                boost::container::small_vector<std::pair<VAddr, VAddr>, 8> gpu_bytes;
                auto it = std::lower_bound(gpu_ranges.begin(), gpu_ranges.end(), page,
                                           [](const auto& range, VAddr address) {
                                               return range.second <= address;
                                           });
                for (; it != gpu_ranges.end() && it->first < page + BYTES_PER_PAGE; ++it)
                    gpu_bytes.emplace_back(std::max(it->first, page),
                                           std::min(it->second, page + BYTES_PER_PAGE));
                if (gpu_bytes.empty()) {
                    continue;
                }
                if (run_start < page) {
                    add_copy(run_start, page - run_start);
                }
                run_start = page + BYTES_PER_PAGE;
                Common::Perf::Add(Common::Perf::MergePages);
                PageSnapshot now;
                CopyGuestPage(page, now.data());
                VAddr cursor = page;
                u64 kept = 0;
                for (const auto [lo, hi] : gpu_bytes) {
                    if (cursor < lo) {
                        add_copy(cursor, lo - cursor);
                    }
                    // Inside bytes the GPU wrote: only those the guest changed are uploaded.
                    u64 i = lo - page;
                    const u64 last = hi - page;
                    while (i < last) {
                        const u64 same_start = i;
                        while (i < last && now[i] == before[i]) {
                            ++i;
                        }
                        kept += i - same_start;
                        const u64 changed_start = i;
                        while (i < last && now[i] != before[i]) {
                            ++i;
                        }
                        if (i != changed_start) {
                            add_copy(page + changed_start, i - changed_start);
                            guest_owned.emplace_back(page + changed_start, page + i);
                            Common::Perf::Add(Common::Perf::MergeGuestBytes, i - changed_start);
                        }
                    }
                    cursor = hi;
                }
                if (cursor < page + BYTES_PER_PAGE) {
                    add_copy(cursor, page + BYTES_PER_PAGE - cursor);
                }
                if (kept != 0) {
                    Common::Perf::Add(Common::Perf::MergeKeptBytes, kept);
                    keep_gpu_owned(page);
                }
            }
            if (run_start < end) {
                add_copy(run_start, end - run_start);
            }
        });
    if (async_upload) {
        PageManager::DeferProtect(nullptr);
    }
    for (const auto [lo, hi] : guest_owned) {
        // The guest wrote these bytes after the GPU: guest memory has their current value.
        SubtractGpuRange(lo, hi - lo);
    }
    constexpr u64 MaxUploadPartBytes = 256_MB;
    if (total_size_bytes > MaxUploadPartBytes && !hash_hot_pages && !merge) {
        Common::Perf::ScopedTimer upload_timer{Common::Perf::BufferUploads,
                                               Common::Perf::BufferUploadNs};
        Common::Perf::Add(Common::Perf::BufferUploadBytes, total_size_bytes);
        // Deferred pages must be protected before the command processor reads
        // them. Drain preceding worker jobs before copying from guest memory.
        if (async_upload) {
            UploadWorker::Job protect;
            protect.pages = std::move(deferred_pages);
            upload_worker->Wait(upload_worker->Enqueue(std::move(protect)));
        }
        decltype(Staging().Request(MaxUploadPartBytes, MemoryType::HostUncached)) staging{};
        boost::container::small_vector<vk::BufferCopy, 4> part_copies;
        // Keep at most two dedicated staging buffers in flight. Flush each completed part so
        // its GPU transfer can overlap the CPU copy of the next one. A buffer becomes reusable
        // only after the tick that reads it has completed; waiting on that tick also covers
        // threaded command recording, where Flush can return before the actual submission.
        constexpr size_t UploadSlots = 2;
        std::array<u64, UploadSlots> slot_ticks{};
        const u64 num_parts = (total_size_bytes + MaxUploadPartBytes - 1) / MaxUploadPartBytes;
        u64 part_index = 0;
        u64 reuse_wait_ns = 0;
        u64 staging_request_ns = 0;
        u64 guest_copy_ns = 0;
        u64 staging_flush_ns = 0;
        u64 gpu_record_ns = 0;
        const u64 parts_start_ns = Common::Perf::NowNs();
        ForEachUploadPart(copies, MaxUploadPartBytes, [&] {
            const size_t slot = part_index % UploadSlots;
            if (slot_ticks[slot] != 0) {
                const u64 wait_start = Common::Perf::NowNs();
                Sched().Wait(slot_ticks[slot]);
                reuse_wait_ns += Common::Perf::NowNs() - wait_start;
            }
            const u64 request_start = Common::Perf::NowNs();
            staging = Staging().Request(MaxUploadPartBytes, MemoryType::HostUncached);
            staging_request_ns += Common::Perf::NowNs() - request_start;
        }, [&](VAddr address, u64 offset, u64 bytes) {
            const u64 copy_start = Common::Perf::NowNs();
            ParallelGuestCopy(memory, address, staging.mapped + offset, bytes);
            guest_copy_ns += Common::Perf::NowNs() - copy_start;
            part_copies.emplace_back(staging.offset + offset, address - arena->cpu_addr, bytes);
        }, [&](u64 bytes) {
            const u64 flush_start = Common::Perf::NowNs();
            staging.Flush();
            staging_flush_ns += Common::Perf::NowNs() - flush_start;
            const u64 record_start = Common::Perf::NowNs();
            Rt().CopyBuffer(staging.buffer, arena, part_copies);
            gpu_record_ns += Common::Perf::NowNs() - record_start;
            // Only these destination ranges are written by this part. Tracking the entire
            // guest binding once per part repeatedly scans gigabytes of untouched memory and
            // adds unnecessary cross-queue dependencies.
            for (const auto& copy : part_copies) {
                NoteQueueUpload(arena->cpu_addr + copy.dstOffset, copy.size);
            }
            part_copies.clear();
            if (++part_index < num_parts) {
                slot_ticks[(part_index - 1) % UploadSlots] = Sched().CurrentTick();
                Sched().Flush();
            }
        });
        LOG_INFO(Render_Vulkan,
                 "Large upload: guest={:x}+{:x} copied={} ranges={} parts={} "
                 "written={} texel={} elapsed_ms={:.3f} reuse_wait_ms={:.3f} "
                 "staging_ms={:.3f} guest_copy_ms={:.3f} flush_ms={:.3f} record_ms={:.3f}",
                 device_addr, size, total_size_bytes, copies.size(), part_index, is_written,
                 is_texel_buffer, (Common::Perf::NowNs() - parts_start_ns) / 1'000'000.0,
                 reuse_wait_ns / 1'000'000.0,
                 staging_request_ns / 1'000'000.0, guest_copy_ns / 1'000'000.0,
                 staging_flush_ns / 1'000'000.0, gpu_record_ns / 1'000'000.0);
    } else if (async_upload && (!copies.empty() || !deferred_pages.empty())) {
        Common::Perf::ScopedTimer upload_timer{Common::Perf::BufferUploads,
                                               Common::Perf::BufferUploadNs};
        Common::Perf::Add(Common::Perf::BufferUploadBytes, total_size_bytes);
        UploadWorker::Job job;
        job.pages = deferred_pages;
        if (!copies.empty()) {
            const auto staging = Staging().Request(total_size_bytes, MemoryType::HostUncached);
            job.copies.reserve(copies.size());
            for (auto& copy : copies) {
                job.copies.push_back(UploadWorker::Copy{
                    .src = copy.dstOffset,
                    .dst = staging.mapped + copy.srcOffset,
                    .size = copy.size,
                });
                copy.srcOffset += staging.offset;
                copy.dstOffset -= arena->cpu_addr;
            }
            job.flush_buffer = staging.buffer;
            job.flush_offset = staging.offset;
            job.flush_size = staging.size;
            // The worker writes the staging memory before the submission of these copies.
            Rt().CopyBuffer(staging.buffer, arena, copies);
            NoteQueueUpload(device_addr, size);
        }
        Sched().RequireBeforeSubmit(upload_worker->Enqueue(std::move(job)));
    } else if (!copies.empty()) {
        Common::Perf::ScopedTimer upload_timer{Common::Perf::BufferUploads,
                                               Common::Perf::BufferUploadNs};
        Common::Perf::Add(Common::Perf::BufferUploadBytes, total_size_bytes);
        const auto staging = Staging().Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            if (hash_hot_pages) {
                // The hashes must describe what was uploaded.
                memory_tracker->VerifyHotPageHashes(copy.dstOffset, copy.size, hash_page);
            }
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        Rt().CopyBuffer(staging.buffer, arena, copies);
        NoteQueueUpload(device_addr, size);
    }
    if (is_texel_buffer && !is_written && !OnComputeQueue()) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

void BufferCache::NoteQueueUpload(VAddr device_addr, u64 size) {
    // A GPU write of the range: the other queue must be done reading and writing it (the guest
    // changed the data once its own fences told it the GPU was done; those fences are signaled
    // when the commands are translated, before the GPU runs them).
    iss::asc_stats::in_upload = true;
    NoteQueueAccess(device_addr, size, true);
    iss::asc_stats::in_upload = false;
}

void BufferCache::ProbeHostImport(VAddr address) {
    iss::ProbeNativeFeedback(instance,scheduler);
    if (!instance.HasExternalMemoryHost()) {
        LOG_INFO(Render_Vulkan, "Host import probe: VK_EXT_external_memory_host not available");
        return;
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
    const u64 size = Common::AlignUp(64_KB, alignment);
    const VAddr base = Common::AlignDown(address, alignment);
    // The guest mapping of the memory (protected pages included), and its backing (the same
    // physical memory, never protected), which a GPU copy in place would use.
    u8* backing = nullptr;
    if (const auto* entry = memory->GetAddressSpace().BackingPages().find(base >> 14);
        entry && *entry) {
        backing = *entry + (base & (16_KB - 1));
    }
    const auto memory_properties = instance.GetPhysicalDevice().getMemoryProperties();
    const auto probe = [&](const char* what, void* pointer) {
        if (pointer == nullptr) {
            LOG_INFO(Render_Vulkan, "Host import probe {}: no address", what);
            return;
        }
        vk::MemoryHostPointerPropertiesEXT host_properties{};
        const vk::Result result = device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, pointer, &host_properties);
        LOG_INFO(Render_Vulkan,
                 "Host import probe {}: {} at {} (alignment {:#x}): properties {}, memory types "
                 "{:#x}",
                 what, size, fmt::ptr(pointer), alignment, vk::to_string(result),
                 host_properties.memoryTypeBits);
        if (result != vk::Result::eSuccess || host_properties.memoryTypeBits == 0) {
            return;
        }
        u32 type_index = std::countr_zero(host_properties.memoryTypeBits);
        for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((host_properties.memoryTypeBits >> i & 1) != 0 &&
                (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
                type_index = i;
                break;
            }
        }
        const vk::ImportMemoryHostPointerInfoEXT import_info = {
            .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            .pHostPointer = pointer,
        };
        const vk::MemoryAllocateInfo alloc_info = {
            .pNext = &import_info,
            .allocationSize = size,
            .memoryTypeIndex = type_index,
        };
        vk::DeviceMemory imported{};
        const vk::Result alloc_result = device.allocateMemory(&alloc_info, nullptr, &imported);
        LOG_INFO(Render_Vulkan, "Host import probe {}: import into memory type {} ({}): {}",
                 what, type_index,
                 vk::to_string(memory_properties.memoryTypes[type_index].propertyFlags),
                 vk::to_string(alloc_result));
        if (alloc_result == vk::Result::eSuccess) {
            device.freeMemory(imported);
        }
    };
    probe("guest mapping", reinterpret_cast<void*>(base));
    probe("backing", backing);
}

// ISS_UNIFIED_MEMORY (unified_memory.h). The heap is imported on the command processor thread
// at the first binding after it is mapped (the game maps its heaps at boot, before it draws),
// so no binding of it went through the arena before: its blocks are marked resident without
// sparse backing and their BDA page table entries point into the imported buffer.
bool BufferCache::EnableUnified() {
    if (!liverpool || !liverpool->IsCurrentCpThread()) {
        return false;
    }
    const auto fail = [this](const char* why) {
        unified_failed = true;
        std::fprintf(stderr, "UNIFIED_MEMORY off: %s\n", why);
        return false;
    };
    if (cp_split_enabled || compute_queue != nullptr) {
        return fail("needs the graphics queue only (ISS_CP_SPLIT=0, asc_queue=0)");
    }
    if (!instance.HasExternalMemoryHost()) {
        return fail("VK_EXT_external_memory_host not available");
    }
    const VAddr base = iss::unified::RangeBase();
    const u64 size = iss::unified::RangeSize();
    const auto span = memory->DescribeBacking(base, size);
    if (!span) {
        return false; // not mapped yet: the next binding tries again
    }
    if (span->size != size) {
        return fail("the range is not one contiguous mapping");
    }
    if ((base & (block_size - 1)) != 0 || (size & (block_size - 1)) != 0) {
        return fail("the range is not aligned to sparse blocks");
    }
    {
        std::scoped_lock lock{residency_mutex};
        bool resident = false;
        resident_ranges.ForEachInRange(base >> block_shift, (base + size) >> block_shift,
                                       [&](const Backing&) { resident = true; });
        if (resident) {
            return fail("the arena already backs part of the range");
        }
    }
    void* host = reinterpret_cast<void*>(span->physical_start);
    const auto device = instance.GetDevice();
    const auto properties =
        instance.GetPhysicalDevice()
            .getProperties2<vk::PhysicalDeviceProperties2,
                            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
    const u64 alignment = std::max<u64>(
        properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
            .minImportedHostPointerAlignment,
        4_KB);
    if ((span->physical_start & (alignment - 1)) != 0 || (size & (alignment - 1)) != 0) {
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
                                              "Unified guest heap");
    unified_buffer->buffer.Destroy();
    unified_buffer->buffer.buffer = handle;
    unified_buffer->buffer.bda_addr = device.getBufferAddress(vk::BufferDeviceAddressInfo{
        .buffer = handle});
    unified_buffer->cpu_addr = base;
    unified_buffer->size_bytes = size;
    unified_buffer->mem_type = MemoryType::HostCached;
    unified_buffer->is_coherent = true;
    unified_chunk_tick.assign(size >> UNIFIED_CHUNK_SHIFT, 0);

    // Shaders that read guest memory through device addresses: the heap's blocks resolve into
    // the imported buffer, and are resident (never backed in the arena).
    const u64 first_block = base >> block_shift;
    const u64 last_block = (base + size - 1) >> block_shift;
    const u64 num_blocks = last_block - first_block + 1;
    const auto staging =
        Staging().Request(num_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
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
    Rt().CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), std::span{&copy, 1});
    {
        std::scoped_lock lock{residency_mutex};
        MarkResident(first_block, last_block);
    }
    arena_binding_generation.fetch_add(1, std::memory_order_release);
    Rt().untracked_reads = unified_buffer.get();
    unified_base = base;
    unified_end = base + size;
    std::fprintf(stderr,
                 "UNIFIED_MEMORY on: guest 0x%llx size 0x%llx host %p memory type %u (bits 0x%x)\n",
                 (unsigned long long)base, (unsigned long long)size, host, type,
                 host_properties.memoryTypeBits);
    return true;
}

std::pair<const Buffer*, u64> BufferCache::ObtainUnified(VAddr device_addr, u32 size,
                                                         bool is_written, bool* known_unwritten) {
    const u64 first = (device_addr - unified_base) >> UNIFIED_CHUNK_SHIFT;
    const u64 last = (device_addr + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
    if (is_written) {
        const u64 tick = Sched().CurrentTick();
        for (u64 c = first; c <= last; ++c) {
            unified_chunk_tick[c] = tick;
        }
        // Memos keyed on the GPU write sequence (stage memo stamps) see the write.
        memory_tracker->BumpWriteSeq();
        iss::unified::Add(iss::unified::Writes);
    } else {
        const bool touched = UnifiedTouched(device_addr, size);
        // Bytes only the CPU writes: no GPU write to order the read after (reads are still
        // recorded after the draw, for the barriers of later GPU writes).
        if (!touched && known_unwritten) {
            *known_unwritten = true;
        }
        iss::unified::Add(touched ? iss::unified::Reads : iss::unified::UntouchedReads);
    }
    return {unified_buffer.get(), device_addr - unified_base};
}

bool BufferCache::UnifiedTouched(VAddr address, u64 size) const noexcept {
    // Only GPU writes not known to be complete: once done, the guest's bytes are current.
    const u64 known = Sched().GetWorkSemaphore()->KnownGpuTick();
    const u64 first = (address - unified_base) >> UNIFIED_CHUNK_SHIFT;
    const u64 last = (address + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
    for (u64 c = first; c <= last; ++c) {
        if (unified_chunk_tick[c] > known) {
            return true;
        }
    }
    return false;
}

void BufferCache::UnifiedHostBarrier() {
    auto& sched = Sched();
    sched.EndRendering();
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    sched.Recorder().pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

void BufferCache::SyncUnifiedRead(VAddr address, u64 size) {
    if (!IsUnified(address, size)) {
        return;
    }
    const u64 first = (address - unified_base) >> UNIFIED_CHUNK_SHIFT;
    const u64 last = (address + size - 1 - unified_base) >> UNIFIED_CHUNK_SHIFT;
    u64 tick = 0;
    for (u64 c = first; c <= last; ++c) {
        tick = std::max(tick, unified_chunk_tick[c]);
    }
    auto& sched = Sched();
    if (tick == 0 || sched.IsFree(tick)) {
        return;
    }
    const u64 start = Common::Perf::NowNs();
    if (tick >= sched.CurrentTick()) {
        UnifiedHostBarrier();
    }
    sched.Wait(tick);
    iss::unified::Add(iss::unified::CpReadSyncs);
    iss::unified::Add(iss::unified::CpReadWaitNs, Common::Perf::NowNs() - start);
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
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
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

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info, bool graphics_submit) {
    // Scheduler submit callbacks may run on a recorder thread, not the CP. In split mode
    // only G drains sparse binds; C waits on the already published residency ticket.
    if (cp_split_enabled && liverpool && liverpool->CpSplitActive() && !graphics_submit)
        return;
    std::vector<ArenaBinds> ready_binds;
    u64 signal_tick{};
    {
        std::scoped_lock lock{residency_mutex};
        if (pending_binds.empty()) return;
        ASSERT_MSG(residency_publication.PendingTick() != 0,
                   "Sparse batch has no reserved timeline tick");
        ready_binds.swap(pending_binds);
        signal_tick = residency_publication.TakePendingTick();
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(ready_binds.size());

    for (const auto& binds : ready_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

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

    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        Vulkan::GpuCrashDiag::ReportDeviceLost(instance, "sparse binding");
    }
    ASSERT_MSG(submit_result == vk::Result::eSuccess,
               "Sparse bind submit failed: {}", vk::to_string(submit_result));
    residency_publication.PublishSubmitted(signal_tick);
    AddResidencyWait(info, signal_tick);
}

} // namespace VideoCore

