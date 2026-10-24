// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "common/assert.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/amdgpu/mtb_conflict.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/water_scale.h"

#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"

namespace Vulkan {

namespace {
// Per-draw flags, read once at startup instead of through function-local statics (each of
// those adds a guard check per call).
const int g_skip_empty_dynstate = [] {
    const char* value = std::getenv("SHADPS4_SKIP_EMPTY_DYNSTATE");
    return value ? value[0] - '0' : 0;
}();
const bool g_pop_fastpath = [] {
    const char* value = std::getenv("SHADPS4_POP_FASTPATH");
    return value && value[0] == '1';
}();
} // namespace

std::mutex Scheduler::submit_mutex;
std::atomic<VkSemaphore> Scheduler::async_timeline{VK_NULL_HANDLE};
static bool g_maintenance7 = false; // p2: set once in the constructor
std::atomic<u64> Scheduler::async_submitted_tick{0};
std::atomic<void (*)(u64)> Scheduler::submitted_hook{nullptr};

Scheduler::Scheduler(const Instance& instance, bool allow_recording_worker)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
    g_maintenance7 = instance.IsMaintenance7Supported();
    if (pending_ops_lock) {
        LOG_INFO(Render_Vulkan, "SHADPS4_PENDING_OPS_LOCK=1: deferred operations queue locked "
                                "(guest-thread DeferOperation; upstream #4844 / #4846)");
    }
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    const char* enabled = std::getenv("SHADPS4_ASYNC_RECORDING");
    if (allow_recording_worker && enabled && std::string_view{enabled} == "1") {
        const char* timing = std::getenv("SHADPS4_ASYNC_RECORDING_TIMING");
        recording_measure_timing = timing && std::string_view{timing} == "1";
        const char* async = std::getenv("SHADPS4_ASYNC_SUBMIT");
        async_submit = async && std::string_view{async} == "1";
        // SHADPS4_PAR_REC lanes need the worker to submit (a submission ends a batch); the ASC
        // stream (two command buffers switched on the worker) and the batch timer (one query
        // pool) assume one recorder.
        par_rec = ParRecRequested();
        if (par_rec != 0) {
            const char* why = !async_submit                  ? "SHADPS4_ASYNC_SUBMIT is off"
                              : instance.HasAscQueue()       ? "the ASC stream is on"
                              : GpuBatchTimer::Enabled()     ? "SHADPS4_GPU_BATCH_TIMING is on"
                                                             : nullptr;
            if (why) {
                LOG_WARNING(Render_Vulkan, "ParRec: {} lanes requested, off because {}", par_rec,
                            why);
                par_rec = 0;
            }
        }
        g_par_rec_lanes = par_rec;
        for (u32 i = 0; i < par_rec; ++i) {
            rec_lanes[i].pool = std::make_unique<CommandPool>(instance, &master_semaphore);
        }
        recording_worker = std::make_unique<Common::CommandRecordingWorker<>>(
            recording_measure_timing, par_rec,
            [this](std::size_t lane) {
                t_rec_lane = &rec_lanes[lane];
                Common::SetCurrentThreadName(
                    lane == 0 ? "shadPS4:VkRecorder"
                              : (lane == 1 ? "shadPS4:VkRecorder1"
                                           : (lane == 2 ? "shadPS4:VkRecorder2"
                                                        : "shadPS4:VkRecorder3")));
                if (lane == 0) {
                    Common::ApplyHotThreadPlacement(Common::HotThread::Recorder);
                } else {
                    Common::ApplyCriticalThreadPriority(Common::HotThread::Recorder);
                }
            },
            [this] {
                rec_submitted.fetch_or(RecAbortBit, std::memory_order_acq_rel);
                rec_submitted.notify_all();
            });
        if (par_rec == 0) {
            recording_worker->Record([] {
                Common::SetCurrentThreadName("shadPS4:VkRecorder");
                Common::ApplyHotThreadPlacement(Common::HotThread::Recorder);
            });
        }
        Common::CommandRecordingWorker<>::SetChunkHook(
            [] { Common::SampleThreadPlacement("recorder", 2); });
        LOG_INFO(Render_Vulkan, "AsyncRecording: enabled, ordered primary command worker, {} chunks",
                 recording_worker->ActiveChunks());
        if (par_rec != 0) {
            LOG_INFO(Render_Vulkan,
                     "ParRec: {} recording lanes (batch k on lane k % {}, submitted in batch order)",
                     par_rec, par_rec);
        }
        if (MtbP1Requested()) {
            // P1d (notes/parrec-build-2026-10-08.md sec. 14): the back installs its translating
            // context later (InstallMtbContext); these gates are fixed before any other thread
            // reads them.
            // The ASC stream records stream switches and segment submissions straight into
            // the worker (Opus review 10-08 night): they would overtake the sink.
            if (async_submit && !instance.HasAscQueue()) {
                g_mtb_p1 = true;
                record_ctx_installed = true;
                MasterSemaphore::reservations_enabled = true;
                LOG_INFO(Render_Vulkan, "MtbP1: on (worker rasterizer, record sink, reserved tick "
                                        "on the back; serial)");
            } else {
                LOG_WARNING(Render_Vulkan, "MtbP1: off because {}",
                            async_submit ? "the ASC stream is on" : "SHADPS4_ASYNC_SUBMIT is off");
            }
        }
        if (async_submit) {
            async_timeline.store(static_cast<VkSemaphore>(master_semaphore.Handle()));
            LOG_INFO(Render_Vulkan, "AsyncSubmit: the recording worker ends and submits batches");
            if (GpuBatchTimer::Enabled()) {
                batch_timer = std::make_unique<GpuBatchTimer>(instance, master_semaphore);
            }
        }
    }
    if (async_submit && instance.HasAscQueue()) {
        asc_enabled = true;
        asc_semaphore = std::make_unique<MasterSemaphore>(instance);
        // Recycled by the ASC timeline: an ASC command buffer must not be reset while the compute
        // queue still runs it (keyed by the main semaphore, this only held through I1; the
        // SHADPS4_ASC_NO_I1 probe froze, 2026-09-28).
        asc_pool = std::make_unique<CommandPool>(instance, asc_semaphore.get(),
                                                 instance.GetReadbackQueueFamilyIndex());
        if (const char* t = std::getenv("SHADPS4_ASC_TIMING"); t && t[0] == '1') {
            asc_timer = std::make_unique<AscTimer>();
            asc_timer->pool = instance.GetDevice()
                                  .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                      .queryType = vk::QueryType::eTimestamp,
                                      .queryCount = AscTimer::Slots * 2,
                                  })
                                  .value;
            asc_timer->period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
            asc_timer->window = std::chrono::steady_clock::now();
        }
        LOG_INFO(Render_Vulkan, "AscStream: enabled");
    }
    if (async_submit && !asc_enabled && instance.HasAscLaneQueue()) {
        lane_enabled = true;
        asc_semaphore = std::make_unique<MasterSemaphore>(instance);
        asc_pool = std::make_unique<CommandPool>(instance, asc_semaphore.get(),
                                                 instance.GetReadbackQueueFamilyIndex());
        if (const char* t = std::getenv("SHADPS4_ASC_TIMING"); t && t[0] == '1') {
            // q16: the lane's segments timed like the ASC stream's.
            asc_timer = std::make_unique<AscTimer>();
            const u32 fam = instance.GetReadbackQueueFamilyIndex();
            const auto families = instance.GetPhysicalDevice().getQueueFamilyProperties();
            LOG_INFO(Render_Vulkan, "G0: lane segment timing on family {} (timestampValidBits {})", fam,
                     fam < families.size() ? families[fam].timestampValidBits : 0u);
            asc_timer->pool = instance.GetDevice()
                                  .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                      .queryType = vk::QueryType::eTimestamp,
                                      .queryCount = AscTimer::Slots * 2,
                                  })
                                  .value;
            asc_timer->period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
            asc_timer->window = std::chrono::steady_clock::now();
        }
        if (const char* e = std::getenv("SHADPS4_LANE_RECORDER"); e && e[0] == '1') {
            lane_rec = true; // q12
            lane_rec_thread = std::thread([this] { LaneRecLoop(); });
            LOG_INFO(Render_Vulkan, "AscLane: q12 lane recorder thread on (recording and submits off the back)");
        }
        if (const char* e = std::getenv("SHADPS4_LANE_DEFER_UPLOAD"); e && (e[0] == '2' || e[0] == '3') && !lane_rec) {
            lane_up_on = true; // q36
            lane_up_thread = std::thread([this] { LaneUploadLoop(); });
            LOG_INFO(Render_Vulkan, "q36: SHADPS4_LANE_DEFER_UPLOAD={} (lane uploads: protect + copy and the lane "
                     "submit on the LaneUpload FIFO; recording stays on the back)", e[0]);
        }
        lane_wait_census = VideoCore::g_lane_wait_census && VideoCore::g_lane_seg_census;
        if (lane_wait_census) {
            LOG_INFO(Render_Vulkan, "q59: SHADPS4_LANE_WAIT_CENSUS=1 (frame label {:#x})", VideoCore::g_frame_label_addr);
        }
        LOG_INFO(Render_Vulkan,
                 "AscLane: on (early ASC records inline on the back, queue 1 of family {}, no I1, "
                 "{} recording lanes)",
                 instance.GetReadbackQueueFamilyIndex(), par_rec);
    }
    if (const char* e = std::getenv("SHADPS4_CHAIN_TIMING"); e && e[0] == '1' && !lane_enabled) {
        // q24 (design-game-main D3): the particle chain's early-service runs in the main batch, timed.
        chain_timer = std::make_unique<ChainTimer>();
        chain_timer->pool = instance.GetDevice()
                                .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                    .queryType = vk::QueryType::eTimestamp,
                                    .queryCount = ChainTimer::Slots * 3,
                                })
                                .value;
        chain_timer->period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
        chain_timer->window = std::chrono::steady_clock::now();
        LOG_INFO(Render_Vulkan, "q24: SHADPS4_CHAIN_TIMING=1 (early ASC service runs timed in the main batch)");
    }
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
    if (lane_up_thread.joinable()) { // q36: every queued protect / copy / submit runs before the pools go
        LaneUploadDrain();
        {
            std::scoped_lock lk{lane_up_mu};
            lane_up_stop = true;
        }
        lane_up_cv.notify_all();
        lane_up_thread.join();
    }
    if (lane_rec_thread.joinable()) { // q12
        {
            std::scoped_lock lk{lane_rec_mu};
            lane_rec_stop = true;
        }
        lane_rec_cv.notify_all();
        lane_rec_thread.join();
    }
    mtb.reset(); // its sink returns worker chunks
    recording_worker.reset();
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

bool Scheduler::TransferCoalesceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_TRANSFER_COALESCE");
        return value && value[0] == '1';
    }();
    return enabled;
}

namespace {
struct TransferCoalesceStats {
    u64 ops = 0, groups = 0, tt_barriers = 0, overflow = 0;
};
// Per recorder thread (SHADPS4_PAR_REC lanes record at once); log-only counters.
thread_local TransferCoalesceStats g_tcs;

bool RangesOverlap(const Scheduler::TransferRange& a, const Scheduler::TransferRange& b) {
    return a.buffer == b.buffer && a.lo < b.hi && b.lo < a.hi;
}
} // namespace

void Scheduler::BeginTransferOnWorker(vk::CommandBuffer cmdbuf, const TransferRange& read,
                                      const TransferRange& write) {
    TransferGroup& group = RecorderTransferGroup();
    ++g_tcs.ops;
    // q17 (b): DIAGNOSTIC, UNSAFE - skip the group-open barrier (timing only).
    static const bool open_barrier = [] {
        const char* e = std::getenv("SHADPS4_TRANSFER_OPEN_BARRIER");
        const bool on = !(e && e[0] == '0');
        if (!on) {
            LOG_WARNING(Render_Vulkan, "q17: SHADPS4_TRANSFER_OPEN_BARRIER=0 - UNSAFE timing diagnostic");
        }
        return on;
    }();
    if (!group.open && !open_barrier) {
        group.open = true;
        group.num_reads = 0;
        group.num_writes = 0;
        ++g_tcs.groups;
    } else if (!group.open) {
        // Everything recorded before the group -> the group's transfers (was: one such barrier
        // per op, scoped to its buffer; global is a superset).
        const vk::MemoryBarrier2 pre = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask =
                vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &pre,
        });
        group.open = true;
        group.num_reads = 0;
        group.num_writes = 0;
        ++g_tcs.groups;
    } else {
        bool hazard = group.num_reads == group.reads.size() ||
                      group.num_writes == group.writes.size();
        g_tcs.overflow += hazard;
        for (u32 i = 0; i < group.num_writes && !hazard; ++i) {
            hazard = (write.buffer && RangesOverlap(write, group.writes[i])) ||
                     (read.buffer && RangesOverlap(read, group.writes[i]));
        }
        for (u32 i = 0; i < group.num_reads && !hazard; ++i) {
            hazard = write.buffer && RangesOverlap(write, group.reads[i]);
        }
        if (hazard) {
            // Transfers of one group run unordered: order this op after the earlier ones.
            const vk::MemoryBarrier2 tt = {
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask =
                    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &tt,
            });
            group.num_reads = 0;
            group.num_writes = 0;
            ++g_tcs.tt_barriers;
        }
    }
    if (read.buffer) {
        group.reads[group.num_reads++] = read;
    }
    if (write.buffer) {
        group.writes[group.num_writes++] = write;
    }
    if ((g_tcs.ops & 0x3ffff) == 0) {
        LOG_INFO(Render_Vulkan, "TransferCoalesce: ops={} groups={} tt_barriers={} overflow={}",
                 g_tcs.ops, g_tcs.groups, g_tcs.tt_barriers, g_tcs.overflow);
    }
}

void Scheduler::CloseTransferGroupSlow(TransferGroup& group, vk::CommandBuffer cmdbuf) {
    group.open = false;
    // The group's transfers -> everything recorded after it (was: one such barrier per op).
    const vk::MemoryBarrier2 post = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &post,
    });
}

void Scheduler::DrainRecording() {
    if (recording_worker) {
        PublishActiveSink();
        recording_worker->Drain();
    }
}

bool Scheduler::InstallMtbContext(const bool reserve_tick) {
    if (!g_mtb_p1 || !recording_worker) {
        return false;
    }
    if (mtb) {
        return t_record_ctx == &mtb->rc;
    }
    // Everything recorded so far went through rc_main: submit it, so the context starts a batch
    // (state invalidated, no pass open) - every later batch starts the same way.
    Flush();
    mtb = std::make_unique<MtbContext>();
    mtb->sink = NewRecordSink();
    mtb->sink->SetEagerPublish(true);
    // RecordSeq continues (a memo's end_seq from before must not match a later sink value).
    mtb->sink->StartSeq(RecordSeq() + 1);
    mtb->rc.owner = this;
    mtb->rc.sink = mtb->sink.get();
    mtb->rc.stream_seq = rc_main.stream_seq;
    mtb->rc.dynamic_state.Invalidate();
    mtb->rc.native_heaps_bound = false;
    t_record_ctx = &mtb->rc;
    AmdGpu::MtbConflict::g_batch = master_semaphore.CurrentTickRaw();
    // P1d: the back translates and reserves; P3: the committer carries the real tick and only a
    // translating worker reserves (at each run).
    if (reserve_tick) {
        master_semaphore.Reserve(master_semaphore.CurrentTickRaw());
    }
    LOG_INFO(Render_Vulkan, "MtbP1: translating context installed at tick {} (reserved: {})",
             master_semaphore.CurrentTickRaw(), reserve_tick);
    return true;
}

void Scheduler::MtbBackExit() {
    if (!mtb || t_record_ctx != &mtb->rc) {
        return;
    }
    // Shutdown runs on other threads through rc_main: close the back's pass and hand over the
    // sink's tail here (Opus review 10-08 night).
    EndRendering();
    PublishActiveSink();
    recording_worker->Publish();
}

bool Scheduler::AttachMtbContext() {
    if (!mtb) {
        return false;
    }
    t_record_ctx = &mtb->rc;
    return true;
}

void Scheduler::MtbTickAdvanced(const u64 signal_value) {
    RecordContext* const ctx = t_record_ctx;
    if (!ctx || ctx->owner != this) {
        return;
    }
    // The batch just submitted was translated with the tick its NextTick returned (checked and
    // moved on the thread that holds a reservation: the translating one).
    if (const u64 reserved = master_semaphore.ReservedTick(); reserved != 0) {
        if (reserved != signal_value) {
            ++mtb_res_mismatch;
        }
        master_semaphore.Reserve(master_semaphore.CurrentTickRaw());
    }
    ctx->sink->MarkBoundary(); // the submission was recorded outside the sink
    if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
        AmdGpu::MtbConflict::EndBatch(master_semaphore.CurrentTickRaw());
    }
    ++mtb_batches;
    if (++mtb_log_batches >= 16384) {
        LogMtbP1();
    }
}

void Scheduler::LogMtbP1() {
    mtb_log_batches = 0;
    LOG_INFO(Render_Vulkan,
             "MtbP1: batches={} res_mismatch={} breaks={} eager_publishes={} sink_space_waits={}",
             mtb_batches, mtb_res_mismatch, mtb_breaks, mtb ? mtb->sink->EagerPublishes() : 0,
             recording_worker->SinkSpaceWaits());
}

void Scheduler::LogRecordingWindow() {
    if (!recording_worker) {
        return;
    }
    static Common::CommandRecordingWorker<>::Counters prev{};
    const auto now = recording_worker->GetCounters();
    LOG_INFO(Render_Vulkan,
             "AsyncRecordingWindow: timing={} commands={} chunks={} bytes={} worker_ms={:.1f} "
             "producer_drain_wait_ms={:.1f} producer_space_wait_ms={:.1f} active_chunks={}",
             recording_measure_timing, now.commands - prev.commands, now.chunks - prev.chunks,
             now.bytes - prev.bytes, double(now.worker_ns - prev.worker_ns) / 1e6,
             double(now.drain_ns - prev.drain_ns) / 1e6,
             double(now.space_wait_ns - prev.space_wait_ns) / 1e6, recording_worker->ActiveChunks());
    prev = now;
}

void Scheduler::CommitDynamicState() {
    if (!IsAsyncRecording()) {
        Rc().dynamic_state.Commit(instance, current_cmdbuf);
        return;
    }
    // SHADPS4_SKIP_EMPTY_DYNSTATE=1: do not record a ~0.8 KB DynamicState snapshot when Commit
    // would emit nothing. Commit clears exactly the bits it emits, so it emits nothing iff the
    // postponed set computed below equals the current dirty set (padding bits only make the
    // comparison conservative). =2 also logs the skip rate every 2^20 calls.
    const int skip_empty = g_skip_empty_dynstate;
    auto& state = Rc().dynamic_state;
    const auto old_dirty = state.dirty_state;
    // Keep the conditional dirtiness which Commit deliberately postpones until
    // its corresponding test is enabled. All values in the packet are owned.
    const DynamicDirtyBits kept = PostponedDynamicDirty();
    if (skip_empty > 0) {
        const bool empty = std::memcmp(&kept, &old_dirty, sizeof(kept)) == 0;
        if (skip_empty == 2) {
            static u64 calls = 0, skipped = 0;
            skipped += empty;
            if ((++calls & ((1u << 20) - 1)) == 0) {
                LOG_INFO(Render_Vulkan, "SKIP_EMPTY_DYNSTATE: skipped {} of {} commits ({:.1f}%)",
                         skipped, calls, 100.0 * skipped / calls);
            }
        }
        if (empty) {
            return;
        }
    }
    Record([snapshot = state, instance = &instance](vk::CommandBuffer cmdbuf) mutable {
        snapshot.Commit(*instance, cmdbuf);
    });
    state.dirty_state = kept;
}

Scheduler::DynamicDirtyBits Scheduler::PostponedDynamicDirty() const {
    const auto& state = Rc().dynamic_state;
    const auto old_dirty = state.dirty_state;
    DynamicDirtyBits kept{};
    kept.depth_compare_op = old_dirty.depth_compare_op && !state.depth_test_enabled;
    kept.depth_bounds = old_dirty.depth_bounds && !state.depth_bounds_test_enabled;
    kept.depth_bias = old_dirty.depth_bias && !state.depth_bias_enabled;
    if (!state.stencil_test_enabled) {
        kept.stencil_front_ops = old_dirty.stencil_front_ops;
        kept.stencil_back_ops = old_dirty.stencil_back_ops;
        kept.stencil_front_reference = old_dirty.stencil_front_reference;
        kept.stencil_back_reference = old_dirty.stencil_back_reference;
        kept.stencil_front_write_mask = old_dirty.stencil_front_write_mask;
        kept.stencil_back_write_mask = old_dirty.stencil_back_write_mask;
        kept.stencil_front_compare_mask = old_dirty.stencil_front_compare_mask;
        kept.stencil_back_compare_mask = old_dirty.stencil_back_compare_mask;
    }
    kept.feedback_loop_enabled =
        old_dirty.feedback_loop_enabled && !instance.IsAttachmentFeedbackLoopLayoutSupported();
    return kept;
}

bool Scheduler::CommitDynamicStateWouldRecord() const {
    if (!IsAsyncRecording() || g_skip_empty_dynstate <= 0) {
        return true;
    }
    // CommitDynamicState's own skip test, on the same inputs.
    const DynamicDirtyBits kept = PostponedDynamicDirty();
    return std::memcmp(&kept, &Rc().dynamic_state.dirty_state, sizeof(kept)) != 0;
}

void Scheduler::RecordDrawPacket(const DrawPacketDesc& desc) {
    ASSERT_MSG(IsAsyncRecording(), "Draw packets need the recording worker");
    ++Rc().stream_seq[active_stream];
    if (desc.header.num_writes != 0) {
        Rc().native_heaps_bound = false; // as PushDescriptors
    }
    std::span<const PacketBinding> table = desc.table;
    std::span<const vk::DescriptorBufferInfo> buffer_infos = desc.buffer_infos;
    std::span<const vk::DescriptorImageInfo> image_infos = desc.image_infos;
    bool keep_descriptors = false;
    // SHADPS4_DRAW_PACKET_DELTA: see scripts/patch/draw-packet-delta-2026-10-02.py. Graphics
    // packets only; this thread recorded everything since the previous packet, and every other
    // graphics descriptor push sets packet_state_unknown.
    static const bool packet_delta = [] {
        const char* value = std::getenv("SHADPS4_DRAW_PACKET_DELTA");
        return value && value[0] == '1';
    }();
    if (packet_delta && desc.header.bind_point == vk::PipelineBindPoint::eGraphics &&
        !table.empty()) {
        auto& ps = Rc().packet_push;
        ++ps.packets;
        const VkPipelineLayout layout = desc.header.layout;
        if (Rc().dynamic_state.packet_state_unknown) {
            ++ps.unknown;
        } else if (ps.layout != layout) {
            ++ps.layout_changed;
        } else {
            static_assert(sizeof(PacketBinding) == sizeof(u64));
            bool same = ps.table.size() == table.size() &&
                        ps.buffer_infos.size() == buffer_infos.size() &&
                        ps.image_infos.size() == image_infos.size() &&
                        std::memcmp(ps.table.data(), table.data(), table.size_bytes()) == 0;
            for (size_t i = 0; same && i < buffer_infos.size(); ++i) {
                const auto& a = ps.buffer_infos[i];
                const auto& b = buffer_infos[i];
                same = a.buffer == b.buffer && a.offset == b.offset && a.range == b.range;
            }
            for (size_t i = 0; same && i < image_infos.size(); ++i) {
                const auto& a = ps.image_infos[i];
                const auto& b = image_infos[i];
                same = a.sampler == b.sampler && a.imageView == b.imageView &&
                       a.imageLayout == b.imageLayout;
            }
            keep_descriptors = same;
            ps.differs += !same;
        }
        if (keep_descriptors) {
            ++ps.kept;
            table = {};
            buffer_infos = {};
            image_infos = {};
        } else {
            ps.layout = layout;
            ps.table.resize(table.size());
            std::memcpy(ps.table.data(), table.data(), table.size_bytes());
            ps.buffer_infos.assign(buffer_infos.begin(), buffer_infos.end());
            ps.image_infos.assign(image_infos.begin(), image_infos.end());
            Rc().dynamic_state.packet_state_unknown = false;
        }
        if ((ps.packets & 0x1fffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "DrawPacketDelta: packets={} kept={} ({:.1f}%) unknown={} "
                     "layout_changed={} differs={}",
                     ps.packets, ps.kept, 100.0 * ps.kept / ps.packets, ps.unknown,
                     ps.layout_changed, ps.differs);
            ps.packets = ps.kept = ps.unknown = ps.layout_changed = ps.differs = 0;
        }
    }
    const size_t bytes = DrawPacketPayloadBytes(desc.barriers.size(), table.size(),
                                                buffer_infos.size(), image_infos.size());
    ASSERT_MSG(bytes <= DrawPacketMaxBytes, "Draw packet of {} bytes", bytes);
    WorkerRecordWithPayload(bytes, [&](std::span<std::byte> storage) {
        std::byte* cursor = storage.data();
        auto* header = reinterpret_cast<DrawPacketHeader*>(cursor);
        std::memcpy(header, &desc.header, sizeof(DrawPacketHeader));
        if (keep_descriptors) {
            header->flags |= DrawPacketKeepDescriptors;
            header->num_writes = 0;
            header->num_buffer_infos = 0;
            header->num_image_infos = 0;
        }
        cursor += DrawPacketAlign(sizeof(DrawPacketHeader));
        // The same layout ViewDrawPacket reads back (an empty span advances nothing).
        CopyPayload(cursor, desc.barriers);
        CopyPayload(cursor, table);
        CopyPayload(cursor, buffer_infos);
        CopyPayload(cursor, image_infos);
        return [this, header] { ExecuteDrawPacket(*header); };
    });
}

/// SHADPS4_DRAW_PACKET_STATS=1 (recorder thread only): how much of a packet repeats the packet
/// before it in the same command buffer. A census: it copies the packet's tables.
static void DrawPacketRecorderStats(vk::CommandBuffer cmdbuf, const DrawPacketHeader& header,
                                    const DrawPacketView& view) {
    struct State {
        VkCommandBuffer cmdbuf{};
        VkPipelineLayout layout{};
        VkPipeline pipeline{};
        std::vector<PacketBinding> table;
        std::vector<vk::DescriptorBufferInfo> buffers;
        std::vector<vk::DescriptorImageInfo> images;
        u64 packets{}, same_layout{}, same_pipeline{}, writes{}, writes_in_same{}, writes_equal{};
        u64 all_equal{}, push_data_equal{};
        std::array<u8, sizeof(header.push_data)> push{};
    };
    static State st;
    ++st.packets;
    st.writes += header.num_writes;
    const bool table_same =
        st.cmdbuf == static_cast<VkCommandBuffer>(cmdbuf) &&
        st.layout == static_cast<VkPipelineLayout>(header.layout) &&
        st.table.size() == view.table.size() &&
        (view.table.empty() || std::memcmp(st.table.data(), view.table.data(),
                                           view.table.size_bytes()) == 0) &&
        st.buffers.size() == view.buffer_infos.size() &&
        st.images.size() == view.image_infos.size();
    st.same_pipeline += st.cmdbuf == static_cast<VkCommandBuffer>(cmdbuf) &&
                        st.pipeline == static_cast<VkPipeline>(header.pipeline);
    if (table_same) {
        ++st.same_layout;
        st.writes_in_same += header.num_writes;
        size_t bi = 0, ii = 0;
        u32 equal = 0;
        for (const PacketBinding& binding : view.table) {
            const auto type = static_cast<vk::DescriptorType>(binding.type);
            bool same = true;
            if (IsPacketBufferType(type)) {
                for (u32 k = 0; k < binding.count; ++k, ++bi) {
                    const auto& a = st.buffers[bi];
                    const auto& b = view.buffer_infos[bi];
                    same &= a.buffer == b.buffer && a.offset == b.offset && a.range == b.range;
                }
            } else {
                for (u32 k = 0; k < binding.count; ++k, ++ii) {
                    const auto& a = st.images[ii];
                    const auto& b = view.image_infos[ii];
                    same &= a.sampler == b.sampler && a.imageView == b.imageView &&
                            a.imageLayout == b.imageLayout;
                }
            }
            equal += same;
        }
        st.writes_equal += equal;
        st.all_equal += equal == header.num_writes;
        st.push_data_equal +=
            std::memcmp(st.push.data(), &header.push_data, sizeof(header.push_data)) == 0;
    }
    st.cmdbuf = static_cast<VkCommandBuffer>(cmdbuf);
    st.layout = static_cast<VkPipelineLayout>(header.layout);
    st.pipeline = static_cast<VkPipeline>(header.pipeline);
    st.table.assign(view.table.begin(), view.table.end());
    st.buffers.assign(view.buffer_infos.begin(), view.buffer_infos.end());
    st.images.assign(view.image_infos.begin(), view.image_infos.end());
    std::memcpy(st.push.data(), &header.push_data, sizeof(header.push_data));
    if ((st.packets & 0x1fffff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "DrawPacketRec: packets={} writes/packet={:.2f} same_layout={:.1f}% "
                 "same_pipeline={:.1f}% | in same-layout packets: writes equal={:.1f}% "
                 "all writes equal={:.1f}% push constants equal={:.1f}%",
                 st.packets, static_cast<double>(st.writes) / st.packets,
                 100.0 * st.same_layout / st.packets, 100.0 * st.same_pipeline / st.packets,
                 st.writes_in_same ? 100.0 * st.writes_equal / st.writes_in_same : 0.0,
                 st.same_layout ? 100.0 * st.all_equal / st.same_layout : 0.0,
                 st.same_layout ? 100.0 * st.push_data_equal / st.same_layout : 0.0);
        st.packets = st.same_layout = st.same_pipeline = st.writes = st.writes_in_same =
            st.writes_equal = st.all_equal = st.push_data_equal = 0;
    }
}

void Scheduler::ExecuteDrawPacket(const DrawPacketHeader& header) {
    // Recording-worker thread. Order = the legacy records: barrier, push constants, push
    // descriptors, then (when folded) bind + draw / dispatch.
    const vk::CommandBuffer cmdbuf = RecorderCmdbuf();
    CloseTransferGroup(cmdbuf);
    const DrawPacketView view = ViewDrawPacket(header);
    if (header.num_barriers != 0) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = header.num_barriers,
            .pBufferMemoryBarriers = view.barriers.data(),
        });
    }
    cmdbuf.pushConstants(header.layout, header.stage_flags, 0u, sizeof(header.push_data),
                         &header.push_data);
    static const bool packet_stats = [] {
        const char* value = std::getenv("SHADPS4_DRAW_PACKET_STATS");
        return value && value[0] == '1';
    }();
    if (packet_stats) [[unlikely]] {
        DrawPacketRecorderStats(cmdbuf, header, view);
    }
    if (header.num_writes != 0) {
        // Not value-initialized: BuildDrawPacketWrites assigns every write it returns (the
        // constructors of 64 writes per packet were 5.6% of the recorder, 2026-10-02).
        std::aligned_storage_t<sizeof(vk::WriteDescriptorSet) * DrawPacketMaxWrites,
                               alignof(vk::WriteDescriptorSet)>
            write_storage;
        const std::span<vk::WriteDescriptorSet> writes{
            reinterpret_cast<vk::WriteDescriptorSet*>(&write_storage), DrawPacketMaxWrites};
        const u32 count =
            BuildDrawPacketWrites(view.table, view.buffer_infos, view.image_infos, writes);
        ASSERT_MSG(count == header.num_writes, "Draw packet rebuilt {} of {} writes", count,
                   header.num_writes);
        cmdbuf.pushDescriptorSetKHR(header.bind_point, header.layout, 0,
                                    std::span<const vk::WriteDescriptorSet>{writes.data(), count});
    }
    if (header.flags & DrawPacketDraw) {
        if (!GraphicsBindTracker::Enabled() ||
            GraphicsBindTracker::NeedBind(cmdbuf, header.pipeline)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, header.pipeline);
        }
        if (header.flags & DrawPacketIndexed) {
            cmdbuf.drawIndexed(header.count, header.instances, 0, s32(header.vertex_offset),
                               header.instance_offset);
        } else {
            cmdbuf.draw(header.count, header.instances, header.vertex_offset,
                        header.instance_offset);
        }
    } else if (header.flags & DrawPacketDispatch) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, header.pipeline);
        cmdbuf.dispatch(header.dim_x, header.dim_y, header.dim_z);
    }
}

void Scheduler::PushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                                std::span<const vk::WriteDescriptorSet> writes) {
    Rc().native_heaps_bound = false;
    Rc().dynamic_state.packet_state_unknown = true; // a push outside the draw packets
    if (LaneRecQueues()) [[unlikely]] {
        // q12: deep copy (buffer / image infos); anything else drains and pushes inline.
        bool plain = true;
        size_t nb = 0, ni = 0;
        for (const auto& w : writes) {
            plain = plain && !w.pNext && !w.pTexelBufferView;
            nb += w.pBufferInfo ? w.descriptorCount : 0;
            ni += w.pImageInfo ? w.descriptorCount : 0;
        }
        if (plain) {
            std::vector<vk::WriteDescriptorSet> ws(writes.begin(), writes.end());
            std::vector<vk::DescriptorBufferInfo> bi;
            std::vector<vk::DescriptorImageInfo> ii;
            bi.reserve(nb);
            ii.reserve(ni);
            for (auto& w : ws) {
                if (w.pBufferInfo) {
                    const size_t at = bi.size();
                    bi.insert(bi.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
                    w.pBufferInfo = reinterpret_cast<const vk::DescriptorBufferInfo*>(at + 1);
                }
                if (w.pImageInfo) {
                    const size_t at = ii.size();
                    ii.insert(ii.end(), w.pImageInfo, w.pImageInfo + w.descriptorCount);
                    w.pImageInfo = reinterpret_cast<const vk::DescriptorImageInfo*>(at + 1);
                }
            }
            LaneRecPush([this, bind_point, layout, ws = std::move(ws), bi = std::move(bi),
                         ii = std::move(ii)]() mutable {
                for (auto& w : ws) { // indices + 1 -> pointers into the job's own copies
                    if (w.pBufferInfo) {
                        w.pBufferInfo = bi.data() + (reinterpret_cast<size_t>(w.pBufferInfo) - 1);
                    }
                    if (w.pImageInfo) {
                        w.pImageInfo = ii.data() + (reinterpret_cast<size_t>(w.pImageInfo) - 1);
                    }
                }
                CloseTransferGroup(asc_cmdbuf);
                asc_cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, ws);
            });
            return;
        }
        LaneRecDrainForBorrow();
        InlineCmdbuf().pushDescriptorSetKHR(bind_point, layout, 0, writes);
        return;
    }
    if (!IsAsyncRecording()) {
        InlineCmdbuf().pushDescriptorSetKHR(bind_point, layout, 0, writes);
        return;
    }
    size_t bytes = AlignedPayloadSize(writes.size_bytes());
    for (const auto& write : writes) {
        // Extension chains are not produced by the guest pipeline path. Keep a
        // scoped synchronous fallback for future callers with other descriptors.
        if (write.pNext) {
            auto access = AcquireCommandBuffer();
            access.Handle().pushDescriptorSetKHR(bind_point, layout, 0, writes);
            return;
        }
        if (write.pBufferInfo) {
            bytes += AlignedPayloadSize(sizeof(vk::DescriptorBufferInfo) * write.descriptorCount);
        }
        if (write.pImageInfo) {
            bytes += AlignedPayloadSize(sizeof(vk::DescriptorImageInfo) * write.descriptorCount);
        }
        if (write.pTexelBufferView) {
            bytes += AlignedPayloadSize(sizeof(vk::BufferView) * write.descriptorCount);
        }
    }
    if (bytes > 32768) {
        auto access = AcquireCommandBuffer();
        access.Handle().pushDescriptorSetKHR(bind_point, layout, 0, writes);
        return;
    }
    WorkerRecordWithPayload(bytes, [&](std::span<std::byte> storage) {
        auto* cursor = storage.data();
        auto* copies = reinterpret_cast<vk::WriteDescriptorSet*>(cursor);
        std::memcpy(copies, writes.data(), writes.size_bytes());
        cursor += AlignedPayloadSize(writes.size_bytes());
        for (size_t i = 0; i < writes.size(); ++i) {
            auto& write = copies[i];
            if (write.pBufferInfo) {
                write.pBufferInfo =
                    CopyPayload(cursor, std::span{write.pBufferInfo, write.descriptorCount}).data();
            }
            if (write.pImageInfo) {
                write.pImageInfo =
                    CopyPayload(cursor, std::span{write.pImageInfo, write.descriptorCount}).data();
            }
            if (write.pTexelBufferView) {
                write.pTexelBufferView =
                    CopyPayload(cursor, std::span{write.pTexelBufferView, write.descriptorCount})
                        .data();
            }
        }
        return [this, bind_point, layout, copies, count = writes.size()] {
            RecorderCmdbuf().pushDescriptorSetKHR(
                bind_point, layout, 0, std::span<const vk::WriteDescriptorSet>{copies, count});
        };
    });
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (Rc().is_rendering && Rc().render_state == new_state) {
        return;
    }
    if (batch_timer && !Rc().is_rendering && Rc().last_end_site != ~0u &&
        Rc().render_state == new_state) {
        GpuBreakSites().Resumed(Rc().last_end_site);
    }
    EndRendering();
    Rc().is_rendering = true;
    if (batch_timer) {
        GpuBreaks().render_begins.fetch_add(1, std::memory_order_relaxed);
        Rc().last_end_site = ~0u;
    }
    Rc().render_state = new_state;
    {
        // SHADPS4_WATER_REFLECTION_SCALE: the pass's viewports / scissors follow its scale.
        Rc().dynamic_state.SetRenderScale(new_state.scale, new_state.skip != 0);
    }

    Record([render_state = new_state](vk::CommandBuffer cmdbuf) {
        std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
        for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
            const auto& cb = render_state.color_attachments[i];
            color_attachments[i] = vk::RenderingAttachmentInfo{
                .imageView = cb.image_view,
                .imageLayout = cb.image_layout,
                .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue =
                    vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
            };
        }

        const auto& db = render_state.depth_stencil_attachment;
        const vk::RenderingAttachmentInfo depth_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.depthStencil =
                                             vk::ClearDepthStencilValue{
                                                 .depth = std::bit_cast<float>(db.clear_value[0])}},
        };
        const vk::RenderingAttachmentInfo stencil_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{.depthStencil =
                                   vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
        };

        // p2 (SHADPS4_A2_SEC + maintenance7): worker secondaries and inline draws in one pass.
        static const bool sec_pass = [] {
            const char* e = std::getenv("SHADPS4_A2_SEC");
            return e && e[0] == '1';
        }();
        const vk::RenderingInfo rendering_info = {
            .flags = sec_pass && g_maintenance7
                         ? static_cast<vk::RenderingFlags>(
                               VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT |
                               VK_RENDERING_CONTENTS_INLINE_BIT_KHR)
                         : vk::RenderingFlags{},
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {render_state.width, render_state.height},
                },
            .layerCount = render_state.num_layers,
            .colorAttachmentCount = render_state.num_color_attachments,
            .pColorAttachments = color_attachments.data(),
            .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
            .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
        };

        cmdbuf.beginRendering(rendering_info);
    });
}

void Scheduler::EndRendering(std::source_location loc) {
    if (t_lane_mode) [[unlikely]] {
        return; // the main batch's render pass stays open; the lane never renders
    }
    if (!Rc().is_rendering) {
        return;
    }
    Rc().is_rendering = false;
    if (batch_timer) {
        Rc().last_end_site = GpuBreakSites().Site(loc);
        GpuBreakSites().End(Rc().last_end_site);
    }
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
}

thread_local BackCycleStats t_back_cycles{};

bool BackCyclesEnabled() {
    static const bool on = [] {
        const char* e = std::getenv("SHADPS4_BACK_CYCLES");
        return e && e[0] == '1';
    }();
    return on;
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    if (BackCyclesEnabled()) [[unlikely]] {
        const u64 c0 = __rdtsc();
        SubmitExecution(info);
        ++t_back_cycles.flush_calls;
        t_back_cycles.flush_cyc += __rdtsc() - c0;
        return;
    }
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    WaitMain(tick);
    // R0: the ASC segments submitted before that main flush (a no-op while I1 holds).
    if (const u64 value = AscAt(tick); value != 0) {
        asc_semaphore->Wait(value);
    }
}

u64 Scheduler::AscMainWaitValue(bool no_i1) {
    // SHADPS4_ASC_I1 (notes/asc-i1-relax-design-2026-09-28.md): unset/1 -> I1, the last ASC value
    // submitted; 2 -> I1 plus counters of the precise value; 0 -> the precise value only: the
    // newest unretired ASC segment an access recorded in this batch depends on (RAW/WAR/WAW),
    // always <= the value just submitted by SubmitAscSegment, so never an unsubmitted value.
    static const char mode = [] {
        const char* value = std::getenv("SHADPS4_ASC_I1");
        return value ? value[0] : '1';
    }();
    const u64 last = asc_last_submitted.load(std::memory_order_relaxed);
    const u64 precise = std::min(VideoCore::g_main_wait_asc, last);
    VideoCore::g_main_wait_asc = 0;
    // q23: SHADPS4_ASC_KNOWN_LAZY=1 (lane only) - the counter query at most once per ~100 us; a stale (lower)
    // g_asc_known is conservative for every reader.
    static const bool known_lazy = [] {
        const char* e = std::getenv("SHADPS4_ASC_KNOWN_LAZY");
        const bool on = e && e[0] == '1';
        if (on) {
            LOG_INFO(Render_Vulkan, "q23: SHADPS4_ASC_KNOWN_LAZY=1 (lane: ASC semaphore refresh at most every ~100 us)");
        }
        return on;
    }();
    static u64 q23_calls = 0, q23_refresh = 0, q23_cyc = 0, q23_last = 0;
    ++q23_calls;
    const u64 q23_now = __rdtsc();
    if (!(known_lazy && lane_enabled) || q23_now - q23_last > 350'000) {
        q23_last = q23_now;
        ++q23_refresh;
        asc_semaphore->Refresh();
        q23_cyc += __rdtsc() - q23_now;
    }
    if (lane_enabled && (q23_calls & 0xffff) == 0) {
        LOG_INFO(Render_Vulkan, "q23 AscMainWaitValue: calls={} refreshes={} refresh cycles/call={:.0f} (lazy {})",
                 q23_calls, q23_refresh, q23_refresh ? double(q23_cyc) / double(q23_refresh) : 0.0, known_lazy);
    }
    VideoCore::g_asc_known = asc_semaphore->KnownGpuTick();
    // The batch being flushed took the tick NextTick() is about to return; accesses from now on
    // belong to the next one (+1). Only the falsifier reads it.
    VideoCore::g_asc_open_tick = master_semaphore.CurrentTick() + 1;
    if (mode == '0' || mode == '2') {
        static u64 flushes = 0, i1_waits = 0, precise_waits = 0;
        ++flushes;
        i1_waits += last > asc_waited_by_main && last > VideoCore::g_asc_known;
        precise_waits += precise > asc_waited_by_main && precise > VideoCore::g_asc_known;
        if ((flushes & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "AscDeps: mode={} main_flushes={} i1_unfinished_waits={} precise_waits={} "
                     "slow_checks={} hook_miss={} whole_buffers={} raises whole/entry={}/{} "
                     "segment_wait_distance[none/1/2-3/4+]={}/{}/{}/{} top:{}",
                     mode, flushes, i1_waits, precise_waits, VideoCore::g_asc_dep_slow,
                     VideoCore::g_asc_hook_miss, VideoCore::g_asc_whole_buffers,
                     VideoCore::g_asc_raise_whole, VideoCore::g_asc_raise_entry, asc_wait_dist[0],
                     asc_wait_dist[1], asc_wait_dist[2], asc_wait_dist[3], [] {
                         std::string top;
                         if (!VideoCore::g_asc_raise_by_base) {
                             return top;
                         }
                         std::vector<std::pair<u64, u64>> v(VideoCore::g_asc_raise_by_base->begin(),
                                                            VideoCore::g_asc_raise_by_base->end());
                         std::sort(v.begin(), v.end(),
                                   [](const auto& a, const auto& b) { return a.second > b.second; });
                         for (size_t k = 0; k < std::min<size_t>(v.size(), 5); ++k) {
                             top += fmt::format(" {:#x}={}", v[k].first, v[k].second);
                         }
                         return top;
                     }());
        }
    }
    if (no_i1) {
        return 0;
    }
    return (mode == '0' || lane_enabled) ? precise : last; // the lane: never I1
}

void Scheduler::WaitMain(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    const bool back_cyc = BackCyclesEnabled();
    const u64 c0 = back_cyc ? __rdtsc() : 0;
    const bool blocked = back_cyc && !master_semaphore.IsFree(tick);
    const auto start = Common::FaultMonitor::Now();
    master_semaphore.Wait(tick);
    Common::FaultMonitor::CountGpuWait(Common::FaultMonitor::SinceNs(start));
    if (back_cyc) [[unlikely]] {
        ++t_back_cycles.wait_calls;
        t_back_cycles.wait_blocked += blocked;
        t_back_cycles.wait_cyc += __rdtsc() - c0;
    }
}

void Scheduler::PopPendingOperations() {
    if (pending_ops_lock) {
        // r19a: pop under pending_ops_mutex, run each callback outside it (a callback may defer
        // more; a guest DeferOperation never waits for a callback). Same order, same retire test.
        // Poppers are serialized by priority_pending_ops_mutex and pushers only append, so the
        // front read under the lock stays the front while the retire test (a driver call in
        // AscRetired) runs outside it (review).
        if (pending_ops_count.load(std::memory_order_acquire) == 0) {
            if (!g_pop_fastpath) {
                // As the unlocked path: the GPU tick is refreshed on every call (review: other
                // readers of KnownGpuTick rely on it between submits).
                std::unique_lock lk(priority_pending_ops_mutex);
                master_semaphore.Refresh();
            }
            return;
        }
        std::unique_lock lk(priority_pending_ops_mutex);
        master_semaphore.Refresh();
        bool lane_synced = false;
        for (;;) {
            u64 front_tick = 0;
            {
                std::scoped_lock ql{pending_ops_mutex};
                if (pending_ops.empty()) {
                    break;
                }
                front_tick = pending_ops.front().gpu_tick;
            }
            if (!master_semaphore.IsFree(front_tick) || !AscRetired(front_tick)) {
                break;
            }
            Common::UniqueFunction<void> callback;
            {
                std::scoped_lock ql{pending_ops_mutex};
                callback = std::move(pending_ops.front().callback);
                pending_ops.pop();
                pending_ops_count.fetch_sub(1, std::memory_order_relaxed);
            }
            if (lane_rec && !lane_synced) {
                LaneRecSync(); // q12d: queued lane closures name objects these operations destroy
                lane_synced = true;
            }
            callback();
        }
        return;
    }
    // SHADPS4_POP_FASTPATH=1: called on every draw and dispatch; skip the lock and the driver
    // counter query when nothing is queued, and poll the counter only every 16th call while the
    // oldest operation is not yet known to be free. Measured 2026-09-24 on 386F5167: ~1.6% of the
    // CP's wall time. Without SHADPS4_PENDING_OPS_LOCK a guest-thread DeferOperation races this.
    const bool fast = g_pop_fastpath;
    if (fast) {
        if (pending_ops.empty()) {
            return;
        }
        static thread_local u32 calls = 0;
        if (!master_semaphore.IsFree(pending_ops.front().gpu_tick) && (++calls & 15) != 0) {
            return;
        }
    }
    std::unique_lock lk(priority_pending_ops_mutex);
    master_semaphore.Refresh();
    // R0: an operation retires when its main tick and the ASC segments before it are done.
    bool lane_synced = false;
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick) &&
           AscRetired(pending_ops.front().gpu_tick)) {
        if (lane_rec && !lane_synced) {
            LaneRecSync(); // q12d: queued lane closures name objects these operations destroy
            lane_synced = true;
        }
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    DrainRecording();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    if (par_rec != 0) {
        // Every lane always holds one begun command buffer (startup; after a synchronous
        // submission, the lane that submitted). The open batch's is borrowed while drained.
        for (u32 i = 0; i < par_rec; ++i) {
            RecLane& lane = rec_lanes[i];
            if (!lane.cmdbuf) {
                lane.cmdbuf = lane.pool->CommitInFlight(lane.cmdbuf_slot);
                Check(lane.cmdbuf.begin(begin_info));
            }
        }
        current_cmdbuf = OpenLane().cmdbuf;
        GraphicsBindTracker::Invalidate();
        Rc().dynamic_state.Invalidate();
        Rc().native_heaps_bound = false;
        return;
    }

    main_cmdbuf = command_pool.Commit();
    Check(main_cmdbuf.begin(begin_info));
    current_cmdbuf = worker_stream == 1 ? asc_cmdbuf : main_cmdbuf;
    GraphicsBindTracker::Invalidate();

    // Invalidate dynamic state so it gets applied to the new command buffer.
    Rc().dynamic_state.Invalidate();
    Rc().native_heaps_bound = false;

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecutionAsync(SubmitInfo& info) {
    // SHADPS4_ASYNC_SUBMIT=1: the command processor only picks the tick; the recording worker
    // ends the command buffer, submits it and begins the next one after every command recorded
    // before this point. Measured 2026-09-24: the CP spent ~4.2% of its wall time draining the
    // worker for submits/borrows and ~1.6% inside vkQueueSubmit.
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    // P1d: the batch's translated commands (incl. that EndRendering) precede its submission. With
    // lanes the submission itself goes into the sink below (one chunk run per batch).
    if (par_rec == 0) {
        PublishActiveSink();
    }
    if (asc_enabled) {
        // I1: the ASC work recorded before this batch is submitted first and waited on.
        SubmitAscSegment();
        // SHADPS4_ASC_NO_I1=1: DIAGNOSTIC ONLY, UNSAFE (data races, resource reuse): skip the
        // wait, to measure the upper bound of relaxing I1 (notes/asc-i1-relax-design-2026-09-28.md).
        static const bool no_i1 = [] {
            const char* value = std::getenv("SHADPS4_ASC_NO_I1");
            return value && value[0] == '1';
        }();
        const u64 need = AscMainWaitValue(no_i1);
        if (need > asc_waited_by_main) {
            // Wait masks are per index (index 1 = colour output): only index 0 or 2 is AllCommands.
            ASSERT(info.num_wait_semas != 1);
            info.AddWait(asc_semaphore->Handle(), need);
            asc_waited_by_main = need;
        }
    } else if (lane_enabled) {
        LaneCutForFlush(info);
    }
    const u64 signal_value = AdvanceTick();
    if (lane_wait_census) [[unlikely]] {
        batch_close_tsc[signal_value & 4095] = {signal_value, __rdtsc()}; // q59
    }
    NoteAscAtFlush(signal_value);
    info.AddSignal(master_semaphore.Handle(), signal_value);
    // SHADPS4_SUBMIT_LOCK_SKIP=1: take the global submit lock for the overlay's texture uploads
    // only when one is queued. The recorder holds this lock across vkQueueSubmit, the readback
    // prebatch queue and the presenter take it too, and a30 stacks showed the command processor
    // blocked here 3.4% (fire) / 6.9% (forest) of its wall time with nothing to upload
    // (runs threads-stack-*-a30-20260928-a).
    static const bool submit_lock_skip = [] {
        const char* value = std::getenv("SHADPS4_SUBMIT_LOCK_SKIP");
        return value && value[0] == '1';
    }();
    if (!submit_lock_skip || ImGui::Core::TextureManager::HasPendingUploads()) {
        std::scoped_lock lk{submit_mutex};
        ImGui::Core::TextureManager::Submit();
    }
    const u32 draws = Rc().batch_draws;
    const u32 dispatches = Rc().batch_dispatches;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;
    if (par_rec != 0) {
        // SHADPS4_PAR_REC: this batch's lane ends, submits (in batch order) and replaces its own
        // command buffer; the batch end routes everything recorded after it to the next lane.
        const u64 batch = recording_worker->ProducerBatch();
        auto submit = [this, info, signal_value, batch] {
            SubmitLaneOnWorker(info, signal_value, batch);
        };
        if (RecordSink* const sink = ActiveSink()) [[unlikely]] {
            // P1d: the submission closes the batch's own sink chunks, so a batch does not cost
            // the lanes an extra producer chunk (s264 m1: +50% chunks per lane).
            sink->Record(std::move(submit));
            recording_worker->PublishSink(*sink);
        } else {
            recording_worker->Record(std::move(submit));
        }
        recording_worker->Publish(true);
        Rc().dynamic_state.Invalidate();
        Rc().native_heaps_bound = false;
        master_semaphore.Refresh();
        PopPendingOperations();
        if (++rec_log_batches >= 16384) {
            LogParRec();
        }
        return;
    }
    recording_worker->Record([this, info, signal_value, draws, dispatches]() mutable {
        if (transfer_group[0].open) {
            CloseTransferGroupSlow(transfer_group[0], main_cmdbuf);
        }
        if (batch_timer) {
            bool asc_wait = false;
            for (u32 i = 0; i < info.num_wait_semas; ++i) {
                asc_wait |= asc_enabled && info.wait_semas[i] == asc_semaphore->Handle();
            }
            batch_timer->End(main_cmdbuf, signal_value, draws, dispatches, asc_wait);
        }
        Check(main_cmdbuf.end());
        static constexpr std::array<vk::PipelineStageFlags, 3> wait_stage_masks = {
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eAllCommands,
        };
        const vk::TimelineSemaphoreSubmitInfo timeline_si = {
            .waitSemaphoreValueCount = info.num_wait_semas,
            .pWaitSemaphoreValues = info.wait_ticks.data(),
            .signalSemaphoreValueCount = info.num_signal_semas,
            .pSignalSemaphoreValues = info.signal_ticks.data(),
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_si,
            .waitSemaphoreCount = info.num_wait_semas,
            .pWaitSemaphores = info.wait_semas.data(),
            .pWaitDstStageMask = wait_stage_masks.data(),
            .commandBufferCount = 1U,
            .pCommandBuffers = &main_cmdbuf,
            .signalSemaphoreCount = info.num_signal_semas,
            .pSignalSemaphores = info.signal_semas.data(),
        };
        // SHADPS4_ASC_I1_STATS=1 (diagnostic, notes/asc-i1-relax-design-2026-09-28.md step 0b): at
        // submit time, is the ASC value this batch waits on (I1) still unfinished, and is the main
        // queue then already idle (previous batch done) so the wait certainly stalls it?
        static const bool i1_stats = [] {
            const char* value = std::getenv("SHADPS4_ASC_I1_STATS");
            return value && value[0] == '1';
        }();
        if (i1_stats && asc_enabled) [[unlikely]] {
            static u64 submits = 0, asc_waits = 0, pending = 0, pending_main_idle = 0,
                       backlog_sum = 0;
            ++submits;
            for (u32 i = 0; i < info.num_wait_semas; ++i) {
                if (info.wait_semas[i] != asc_semaphore->Handle()) {
                    continue;
                }
                ++asc_waits;
                asc_semaphore->Refresh();
                const u64 known = asc_semaphore->KnownGpuTick();
                if (known < info.wait_ticks[i]) {
                    ++pending;
                    backlog_sum += info.wait_ticks[i] - known;
                    master_semaphore.Refresh();
                    if (master_semaphore.IsFree(signal_value - 1)) {
                        ++pending_main_idle;
                    }
                }
            }
            if (submits % 10000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "AscI1Stats: main_submits={} with_asc_wait={} asc_unfinished_at_submit={} "
                         "({:.1f}%) of_which_main_idle={} mean_asc_backlog={:.2f}",
                         submits, asc_waits, pending, 100.0 * double(pending) / double(submits),
                         pending_main_idle,
                         pending ? double(backlog_sum) / double(pending) : 0.0);
            }
        }
        {
            // SHADPS4_DEBUG_SUBMIT_STALL_MS (diagnostic): one artificial recorder stall, see
            // scripts/patch/debug-submit-stall-2026-10-02.py.
            static const u32 stall_ms = [] {
                const char* value = std::getenv("SHADPS4_DEBUG_SUBMIT_STALL_MS");
                return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
            }();
            if (stall_ms != 0) [[unlikely]] {
                static const u32 at_s = [] {
                    const char* value = std::getenv("SHADPS4_DEBUG_SUBMIT_STALL_AT_S");
                    return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 10u;
                }();
                static const auto first_submit = std::chrono::steady_clock::now();
                static bool stalled = false;
                if (!stalled && std::chrono::steady_clock::now() - first_submit >=
                                    std::chrono::seconds(at_s)) {
                    stalled = true;
                    LOG_WARNING(Render_Vulkan,
                                "DebugSubmitStall: the recorder sleeps {} ms before submitting "
                                "tick {}",
                                stall_ms, signal_value);
                    std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
                    LOG_WARNING(Render_Vulkan, "DebugSubmitStall: awake, submitting");
                }
            }
            std::scoped_lock lk{submit_mutex};
            const auto result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
            if (result == vk::Result::eErrorDeviceLost) {
                instance.LogDeviceFault();
            }
            ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during submit");
            async_submitted_tick.store(signal_value, std::memory_order_release);
        }
        // SHADPS4_ARENA_PREFETCH_RETRY: outside submit_mutex (the hook may post to the CP thread).
        if (const auto hook = submitted_hook.load(std::memory_order_acquire)) [[unlikely]] {
            hook(signal_value);
        }
        main_cmdbuf = command_pool.Commit();
        Check(main_cmdbuf.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        }));
        if (worker_stream == 0) {
            current_cmdbuf = main_cmdbuf;
        }
        GraphicsBindTracker::Invalidate();
        if (batch_timer) {
            batch_timer->Begin(main_cmdbuf);
        }
    });
    // A partial chunk is otherwise only handed over when full or drained; a waiter on this tick
    // would then wait for commands nobody records.
    recording_worker->Publish();
    // What AllocateWorkerCommandBuffers resets on this thread for a new command buffer.
    Rc().dynamic_state.Invalidate();
    Rc().native_heaps_bound = false;
    master_semaphore.Refresh();
    PopPendingOperations();
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    // SHADPS4_ASC_LANE: a main flush from inside lane mode (a nested service, ring space, a
    // deferred cut) records and submits main work: leave lane mode around it.
    if (!t_lane_mode) [[likely]] {
        SubmitExecutionImpl(info);
        return;
    }
    const u64 carry = asc_wait_main;
    LaneSwitch(0);
    if (synchronous_scopes != 0) {
        DrainRecording();
        if (par_rec != 0) {
            current_cmdbuf = OpenLane().cmdbuf;
        }
    }
    SubmitExecutionImpl(info);
    LaneSwitch(1);
    LaneNeedMain(carry, 4);
}

void Scheduler::SubmitExecutionImpl(SubmitInfo& info) {
    if (async_submit && synchronous_scopes == 0) {
        SubmitExecutionAsync(info);
        return;
    }
    if (async_submit) {
        // Inside a borrow scope. The worker may still hold queued submissions, which take
        // submit_mutex: drain it before taking the lock below, or both wait forever.
        DrainRecording();
    }
    // A batch on this queue must not wait for a draw-timeline value the async worker has not
    // submitted yet: it would block the queue ahead of its own signal.
    if (const VkSemaphore async = async_timeline.load(); async != VK_NULL_HANDLE) {
        for (u32 i = 0; i < info.num_wait_semas; ++i) {
            if (static_cast<VkSemaphore>(info.wait_semas[i]) == async) {
                while (async_submitted_tick.load(std::memory_order_acquire) < info.wait_ticks[i]) {
                    std::this_thread::yield();
                }
            }
        }
    }
    if (par_rec != 0) {
        SubmitExecutionLaneSync(info);
        return;
    }
    std::scoped_lock lk{submit_mutex};
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    DrainRecording();
    if (asc_enabled) {
        SubmitAscSegment(); // Drained: runs on this thread.
        const u64 need = AscMainWaitValue(false);
        if (need > asc_waited_by_main) {
            info.AddWait(asc_semaphore->Handle(), need);
            asc_waited_by_main = need;
        }
    } else if (lane_enabled) {
        LaneCutForFlush(info);
    }
    if (recording_worker && ++recording_submits % 1000 == 0) {
        const auto counts = recording_worker->GetCounters();
        LOG_INFO(
            Render_Vulkan, "AsyncRecording: commands={} chunks={} bytes={} borrows={} submits={}",
            counts.commands, counts.chunks, counts.bytes, synchronous_borrows, recording_submits);
        if (recording_measure_timing) {
            LOG_INFO(Render_Vulkan, "AsyncRecordingTime: worker_ns={} drain_ns={} space_wait_ns={}",
                     counts.worker_ns, counts.drain_ns, counts.space_wait_ns);
            for (const auto& [caller, calls] : recording_borrow_counts) {
                LOG_INFO(Render_Vulkan, "AsyncBorrow: calls={} caller={}", calls, caller);
            }
        }
    }
    const u64 signal_value = AdvanceTick();
    if (lane_wait_census) [[unlikely]] {
        batch_close_tsc[signal_value & 4095] = {signal_value, __rdtsc()}; // q59
    }
    NoteAscAtFlush(signal_value);
    last_batch_draws = Rc().batch_draws;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    if (transfer_group[0].open) {
        CloseTransferGroupSlow(transfer_group[0], main_cmdbuf);
    }
    Check(main_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 3> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eAllCommands,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &main_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault();
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    if (async_submit) {
        // A synchronous submission of the async timeline (inside a borrow scope) counts too.
        async_submitted_tick.store(signal_value, std::memory_order_release);
    }

    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
}

bool Scheduler::WaitRecTurn(u64 batch, bool submit) {
    u64 value = rec_submitted.load(std::memory_order_acquire);
    if (value == batch) {
        return true;
    }
    using Clock = std::chrono::steady_clock;
    const auto start = recording_measure_timing ? Clock::now() : Clock::time_point{};
    for (u32 spins = 0; value != batch; ++spins) {
        if (value & RecAbortBit) {
            return false;
        }
        ASSERT_MSG(value < batch, "ParRec: batch {} waits, {} batches already submitted", batch,
                   value);
        if (spins < 64) {
#if defined(_M_X64) || defined(__x86_64__)
            _mm_pause();
#endif
        } else {
            rec_submitted.wait(value, std::memory_order_acquire);
        }
        value = rec_submitted.load(std::memory_order_acquire);
    }
    if (RecLane* const lane = t_rec_lane) {
        (submit ? lane->submit_waits : lane->ordered_waits).fetch_add(1, std::memory_order_relaxed);
        if (recording_measure_timing) {
            lane->wait_ns.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     Clock::now() - start)
                                     .count()),
                std::memory_order_relaxed);
        }
    }
    return true;
}

void Scheduler::SubmitLaneCmdbufLocked(RecLane& lane, const SubmitInfo& info, u64 signal_value) {
    static constexpr std::array<vk::PipelineStageFlags, 3> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eAllCommands,
    };
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &lane.cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };
    const auto result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault();
    }
    ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    async_submitted_tick.store(signal_value, std::memory_order_release);
    // Reusable once the GPU signals this submission (it was tagged in flight until now).
    lane.pool->Retire(lane.cmdbuf_slot, signal_value);
    lane.submits.fetch_add(1, std::memory_order_relaxed);
}

void Scheduler::SubmitLaneOnWorker(const SubmitInfo& info, u64 signal_value, u64 batch) {
    RecLane* const lane_ptr = t_rec_lane;
    ASSERT_MSG(lane_ptr != nullptr, "ParRec: a lane submission ran off a lane thread");
    RecLane& lane = *lane_ptr;
    if (lane.transfer_group.open) {
        CloseTransferGroupSlow(lane.transfer_group, lane.cmdbuf);
    }
    Check(lane.cmdbuf.end());
    // Queue order == timeline order == batch order: wait until every earlier batch is submitted.
    if (!WaitRecTurn(batch, true)) {
        return;
    }
    {
        std::scoped_lock lk{submit_mutex};
        SubmitLaneCmdbufLocked(lane, info, signal_value);
    }
    rec_submitted.fetch_add(1, std::memory_order_acq_rel);
    rec_submitted.notify_all();
    // SHADPS4_ARENA_PREFETCH_RETRY: outside submit_mutex (the hook may post to the CP thread).
    if (const auto hook = submitted_hook.load(std::memory_order_acquire)) [[unlikely]] {
        hook(signal_value);
    }
    // The lane's next batch is batch + lanes; its signal value is unknown until then.
    lane.cmdbuf = lane.pool->CommitInFlight(lane.cmdbuf_slot);
    Check(lane.cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    GraphicsBindTracker::Invalidate();
}

void Scheduler::SubmitExecutionLaneSync(SubmitInfo& info) {
    // Inside a borrow scope: SubmitExecution drained every lane, nothing new was published.
    std::scoped_lock lk{submit_mutex};
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    DrainRecording();
    if (lane_enabled) {
        LaneCutForFlush(info);
    }
    const u64 signal_value = AdvanceTick();
    if (lane_wait_census) [[unlikely]] {
        batch_close_tsc[signal_value & 4095] = {signal_value, __rdtsc()}; // q59
    }
    NoteAscAtFlush(signal_value);
    last_batch_draws = Rc().batch_draws;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;
    const u64 batch = recording_worker->ProducerBatch();
    ASSERT_MSG(rec_submitted.load(std::memory_order_acquire) == batch,
               "ParRec: synchronous submission of batch {} with {} submitted", batch,
               rec_submitted.load());
    RecLane& lane = OpenLane();
    if (lane.transfer_group.open) {
        CloseTransferGroupSlow(lane.transfer_group, lane.cmdbuf);
    }
    Check(lane.cmdbuf.end());
    info.AddSignal(master_semaphore.Handle(), signal_value);
    ImGui::Core::TextureManager::Submit();
    SubmitLaneCmdbufLocked(lane, info, signal_value);
    rec_submitted.fetch_add(1, std::memory_order_acq_rel);
    rec_submitted.notify_all();
    lane.cmdbuf = vk::CommandBuffer{};
    recording_worker->Publish(true); // what is recorded from now on belongs to the next batch
    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers(); // begins this lane's next buffer, borrows the next lane's
    PopPendingOperations();
}

void Scheduler::LogParRec() {
    rec_log_batches = 0;
    using Clock = std::chrono::steady_clock;
    struct Prev {
        u64 submits, submit_waits, ordered_waits, wait_ns, chunk_ns, chunks;
    };
    static std::array<Prev, MaxRecLanes> prev{};
    static u64 prev_borrows = 0;
    static Clock::time_point prev_time = Clock::now();
    const auto now = Clock::now();
    std::string lanes;
    for (u32 i = 0; i < par_rec; ++i) {
        const RecLane& lane = rec_lanes[i];
        const Prev cur{lane.submits.load(std::memory_order_relaxed),
                       lane.submit_waits.load(std::memory_order_relaxed),
                       lane.ordered_waits.load(std::memory_order_relaxed),
                       lane.wait_ns.load(std::memory_order_relaxed),
                       recording_worker->LaneWorkerNs(i), recording_worker->LaneChunks(i)};
        const Prev& p = prev[i];
        const double chunk_ms = double(cur.chunk_ns - p.chunk_ns) / 1e6;
        const double wait_ms = double(cur.wait_ns - p.wait_ns) / 1e6;
        lanes += fmt::format(" | lane{} submits={} chunks={} submit_waits={} ordered_waits={} "
                             "chunk_ms={:.1f} wait_ms={:.1f} run_ms={:.1f}",
                             i, cur.submits - p.submits, cur.chunks - p.chunks,
                             cur.submit_waits - p.submit_waits, cur.ordered_waits - p.ordered_waits,
                             chunk_ms, wait_ms, chunk_ms - wait_ms);
        prev[i] = cur;
    }
    LOG_INFO(Render_Vulkan, "ParRec: lanes={} window_ms={:.1f} borrows={} timing={}{}", par_rec,
             std::chrono::duration<double, std::milli>(now - prev_time).count(),
             synchronous_borrows - prev_borrows, recording_measure_timing, lanes);
    prev_borrows = synchronous_borrows;
    prev_time = now;
}

void Scheduler::LaneUploadPush(Common::UniqueFunction<void>&& job) {
    // q36d (review 4): done / pushed reasoning needs one pushing thread.
    static const std::thread::id pusher = std::this_thread::get_id();
    ASSERT_MSG(std::this_thread::get_id() == pusher, "q36: LaneUploadPush from a second thread");
    bool wake;
    {
        std::scoped_lock lk{lane_up_mu};
        lane_up_q.emplace_back(__rdtsc(), std::move(job));
        wake = lane_up_sleeping;
    }
    lane_up_pushed.fetch_add(1, std::memory_order_release);
    ++lane_up_jobs;
    if (wake) {
        lane_up_cv.notify_one();
    }
    const u64 now = __rdtsc();
    if (now - lane_up_log_tsc > 17'500'000'000ULL) [[unlikely]] {
        const u64 done = lane_up_done.load(std::memory_order_acquire);
        LOG_INFO(Render_Vulkan,
                 "q36 LaneUpload (~5 s): jobs {} (submits {}, inline submits {}), queued now {}, lag mean {:.1f} us "
                 "max {:.1f} us, FIFO busy {:.1f} ms",
                 lane_up_jobs, lane_up_submits, lane_up_inline, lane_up_pushed.load() - done,
                 done ? double(lane_up_lag_sum.exchange(0)) / 3500.0 / double(std::max<u64>(lane_up_jobs, 1)) : 0.0,
                 double(lane_up_max_lag.exchange(0)) / 3500.0, double(lane_up_busy_cyc.exchange(0)) / 3.5e6);
        lane_up_log_tsc = now;
        lane_up_jobs = lane_up_submits = lane_up_inline = 0;
    }
}

void Scheduler::LaneUploadDefer(Common::UniqueFunction<void>&& job) {
    static const bool batch = [] {
        const char* e = std::getenv("SHADPS4_LANE_UP_BATCH");
        return e && e[0] == '1';
    }();
    if (!batch) {
        LaneUploadPush(std::move(job));
        return;
    }
    lane_up_batch.push_back(std::move(job));
    if (lane_up_batch.size() >= 32) {
        LaneUploadFlushBatch();
    }
}

void Scheduler::LaneUploadFlushBatch() {
    if (lane_up_batch.empty()) {
        return;
    }
    std::vector<Common::UniqueFunction<void>> jobs;
    jobs.swap(lane_up_batch);
    lane_up_batch.reserve(32);
    LaneUploadPush([jobs = std::move(jobs)]() mutable {
        for (auto& j : jobs) {
            j(); // q39: in push order (protect, then copy, per upload)
        }
    });
}

void Scheduler::LaneUploadDrain() {
    LaneUploadFlushBatch(); // q39
    std::unique_lock lk{lane_up_mu};
    if (lane_up_sleeping && !lane_up_q.empty()) {
        lane_up_cv.notify_one();
    }
    lane_up_idle_cv.wait(lk, [this] {
        return lane_up_done.load(std::memory_order_acquire) == lane_up_pushed.load(std::memory_order_acquire);
    });
}

void Scheduler::LaneUploadLoop() {
    Common::SetCurrentThreadName("shadPS4:LaneUpload");
    std::unique_lock lk{lane_up_mu};
    for (;;) {
        if (lane_up_q.empty()) {
            lane_up_idle_cv.notify_all();
            if (lane_up_stop) {
                return;
            }
            lane_up_sleeping = true;
            lane_up_cv.wait(lk, [this] { return !lane_up_q.empty() || lane_up_stop; });
            lane_up_sleeping = false;
            continue;
        }
        auto [tsc, job] = std::move(lane_up_q.front());
        lane_up_q.pop_front();
        lk.unlock();
        const u64 t0 = __rdtsc();
        const u64 lag = t0 - tsc;
        lane_up_lag_sum.fetch_add(lag, std::memory_order_relaxed);
        if (lag > lane_up_max_lag.load(std::memory_order_relaxed)) {
            lane_up_max_lag.store(lag, std::memory_order_relaxed);
        }
        job();
        lane_up_busy_cyc.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
        lk.lock();
        lane_up_done.fetch_add(1, std::memory_order_release);
    }
}

void Scheduler::BeginAscCmdbufOnWorker() {
    std::unique_lock pool_lk{asc_pool_mu, std::defer_lock}; // q36: vs the FIFO's submit of an older slot
    if (lane_up_on) {
        pool_lk.lock();
    }
    asc_cmdbuf = asc_pool->Commit();
    Check(asc_cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    if (asc_timer) {
        AscTimerBegin();
    }
}

void Scheduler::ChainTimerBegin() {
    auto& t = *chain_timer;
    ChainTimerCollect();
    if (t.write_seq - t.read_seq >= ChainTimer::Slots) {
        t.began = false; // ring full: this run is not timed
        ++t.skipped;
        return;
    }
    t.open_slot = static_cast<u32>(t.write_seq % ChainTimer::Slots);
    t.begin_tick = master_semaphore.CurrentTick();
    EndRendering(); // the query reset must be outside a render pass (the dispatch ends it anyway)
    Record([pool = *t.pool, q = t.open_slot * 3](vk::CommandBuffer cmdbuf) {
        cmdbuf.resetQueryPool(pool, q, 3);
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, pool, q);
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pool, q + 1);
    });
    t.began = true;
}

void Scheduler::ChainTimerEnd(u32 ndisp) {
    auto& t = *chain_timer;
    if (!t.began) {
        return;
    }
    t.began = false;
    Record([pool = *t.pool, q = t.open_slot * 3](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pool, q + 2);
    });
    const u64 end_tick = master_semaphore.CurrentTick();
    t.tick[t.open_slot] = end_tick;
    t.ndisp[t.open_slot] = ndisp;
    t.crossed[t.open_slot] = end_tick != t.begin_tick ? 1 : 0;
    ++t.write_seq;
}

void Scheduler::ChainTimerCollect() {
    auto& t = *chain_timer;
    while (t.read_seq < t.write_seq) {
        const u32 slot = static_cast<u32>(t.read_seq % ChainTimer::Slots);
        if (master_semaphore.KnownGpuTick() < t.tick[slot]) {
            master_semaphore.Refresh();
            if (master_semaphore.KnownGpuTick() < t.tick[slot]) {
                break;
            }
        }
        std::array<u64, 3> ts{};
        const auto result = instance.GetDevice().getQueryPoolResults(
            *t.pool, slot * 3, 3, sizeof(ts), ts.data(), sizeof(u64), vk::QueryResultFlagBits::e64);
        ++t.read_seq;
        if (result != vk::Result::eSuccess || ts[2] < ts[1] || ts[1] < ts[0]) {
            ++t.bad;
            continue;
        }
        const u64 span = u64(double(ts[2] - ts[1]) * t.period_ns);
        t.span_ns += span;
        t.lag_ns += u64(double(ts[1] - ts[0]) * t.period_ns);
        t.max_span = std::max(t.max_span, span);
        t.disp += t.ndisp[slot];
        t.crossed_runs += t.crossed[slot];
        ++t.runs;
        ++t.hist[span < 10'000 ? 0 : span < 50'000 ? 1 : span < 200'000 ? 2 : span < 1'000'000 ? 3
                                                                              : span < 5'000'000 ? 4 : 5];
        if (t.prev_end != 0 && ts[1] >= t.prev_end) {
            t.gap_ns += u64(double(ts[1] - t.prev_end) * t.period_ns);
            ++t.gaps;
        }
        t.prev_end = ts[2];
    }
    const auto now = std::chrono::steady_clock::now();
    const double wall_ns =
        double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t.window).count());
    if (wall_ns >= 10e9) {
        const double s = wall_ns / 1e9;
        LOG_INFO(Render_Vulkan,
                 "q24 ChainTiming (per s over {:.1f} s): runs {:.0f} dispatches {:.0f} | GPU span {:.1f} ms/s, "
                 "{:.1f} us per run, {:.2f} us per dispatch, max {:.1f} us | lag behind earlier work {:.1f} ms/s "
                 "({:.1f} us per run) | gap between runs {:.1f} us mean | span <10us/<50/<200/<1ms/<5ms/more "
                 "{}/{}/{}/{}/{}/{} | crossed a batch {} | skipped {} bad {}",
                 s, double(t.runs) / s, double(t.disp) / s, double(t.span_ns) / 1e6 / s,
                 t.runs ? double(t.span_ns) / double(t.runs) / 1e3 : 0.0,
                 t.disp ? double(t.span_ns) / double(t.disp) / 1e3 : 0.0, double(t.max_span) / 1e3,
                 double(t.lag_ns) / 1e6 / s, t.runs ? double(t.lag_ns) / double(t.runs) / 1e3 : 0.0,
                 t.gaps ? double(t.gap_ns) / double(t.gaps) / 1e3 : 0.0, t.hist[0], t.hist[1], t.hist[2],
                 t.hist[3], t.hist[4], t.hist[5], t.crossed_runs, t.skipped, t.bad);
        t.runs = t.disp = t.span_ns = t.lag_ns = t.gap_ns = t.gaps = t.crossed_runs = t.max_span = 0;
        t.skipped = t.bad = 0;
        t.hist.fill(0);
        t.window = now;
    }
}

void Scheduler::AscTimerBegin() {
    auto& t = *asc_timer;
    AscTimerCollect();
    if (t.write_seq - t.read_seq >= AscTimer::Slots) {
        t.began = false; // ring full: skip this segment
        return;
    }
    t.open_slot = static_cast<u32>(t.write_seq % AscTimer::Slots);
    asc_cmdbuf.resetQueryPool(*t.pool, t.open_slot * 2, 2);
    asc_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *t.pool, t.open_slot * 2);
    t.began = true;
}

void Scheduler::AscTimerEnd(u64 value, u64 wait_main) {
    auto& t = *asc_timer;
    const std::array<u32, 3> cur = t.cur; // G0: work noted since the previous segment ended
    t.cur = {};
    if (!t.began) {
        return;
    }
    asc_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *t.pool,
                              t.open_slot * 2 + 1);
    t.value[t.open_slot] = value;
    for (u32 k = 0; k < 3; ++k) {
        t.work[t.open_slot][k] = static_cast<u16>(std::min<u32>(cur[k], 0xffff));
    }
    bool unfinished = false;
    if (wait_main != 0) {
        master_semaphore.Refresh();
        unfinished = master_semaphore.KnownGpuTick() < wait_main;
    }
    t.waits_unfinished[t.open_slot] = unfinished ? 1 : 0;
    ++t.write_seq;
    t.began = false;
}

void Scheduler::AscTimerCollect() {
    auto& t = *asc_timer;
    while (t.read_seq < t.write_seq) {
        const u32 slot = static_cast<u32>(t.read_seq % AscTimer::Slots);
        if (asc_semaphore->KnownGpuTick() < t.value[slot]) {
            asc_semaphore->Refresh();
            if (asc_semaphore->KnownGpuTick() < t.value[slot]) {
                break;
            }
        }
        std::array<u64, 2> ts{};
        const auto result = instance.GetDevice().getQueryPoolResults(
            *t.pool, slot * 2, 2, sizeof(ts), ts.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        ++t.read_seq;
        if (result == vk::Result::eSuccess && ts[1] >= ts[0]) {
            const u64 ns = u64(double(ts[1] - ts[0]) * t.period_ns);
            t.busy_ns += ns;
            t.max_ns = std::max(t.max_ns, ns);
            ++t.segments;
            // G0: class by content; lane idle before this segment (same device timestamp domain).
            const auto& w = t.work[slot];
            const u32 cls = (w[0] | w[1] | w[2]) == 0 ? 4
                            : (w[1] | w[2]) == 0  ? 0
                            : (w[0] | w[2]) == 0  ? 1
                            : (w[0] | w[1]) == 0  ? 2
                                                  : 3;
            t.cls_ns[cls] += ns;
            ++t.cls_n[cls];
            for (u32 k = 0; k < 3; ++k) {
                t.work_n[k] += w[k];
            }
            if (cls == 0) {
                t.disp_only_disp += w[0];
            }
            const u64 start = u64(double(ts[0]) * t.period_ns);
            const u64 end = u64(double(ts[1]) * t.period_ns);
            if (t.prev_end != 0) {
                if (start < t.prev_end) {
                    ++t.overlaps;
                } else if (start - t.prev_end < 50'000'000) {
                    if (t.waits_unfinished[slot]) {
                        t.gap_wait_ns += start - t.prev_end;
                        ++t.gap_wait_n;
                    } else {
                        t.gap_other_ns += start - t.prev_end;
                        ++t.gap_other_n;
                    }
                }
            }
            t.prev_end = end;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double wall_ns =
        double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t.window).count());
    if (wall_ns >= 10e9) {
        LOG_INFO(Render_Vulkan,
                 "AscTiming: segments={} busy_ms={:.1f} busy_pct={:.1f} mean_us={:.1f} max_us={:.1f} "
                 "(per {:.1f} s window; ASC queue, segment start to end)",
                 t.segments, double(t.busy_ns) / 1e6, 100.0 * double(t.busy_ns) / wall_ns,
                 t.segments ? double(t.busy_ns) / double(t.segments) / 1000.0 : 0.0,
                 double(t.max_ns) / 1000.0, wall_ns / 1e9);
        const double ws = wall_ns / 1e9;
        LOG_INFO(Render_Vulkan,
                 "AscTimingG0: per s: segs disp/up/copy/mixed/empty {:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f} busy_ms "
                 "{:.1f}/{:.1f}/{:.1f}/{:.1f}/{:.1f} | dispatches {:.0f} uploads {:.0f} fault copies {:.0f} | "
                 "us per dispatch in dispatch-only segs {:.2f} | lane idle gaps ms: before a segment waiting on "
                 "unfinished main {:.1f} ({:.0f}), other {:.1f} ({:.0f}) | overlaps {}",
                 t.cls_n[0] / ws, t.cls_n[1] / ws, t.cls_n[2] / ws, t.cls_n[3] / ws, t.cls_n[4] / ws,
                 t.cls_ns[0] / 1e6 / ws, t.cls_ns[1] / 1e6 / ws, t.cls_ns[2] / 1e6 / ws, t.cls_ns[3] / 1e6 / ws,
                 t.cls_ns[4] / 1e6 / ws, t.work_n[0] / ws, t.work_n[1] / ws, t.work_n[2] / ws,
                 t.disp_only_disp ? double(t.cls_ns[0]) / double(t.disp_only_disp) / 1000.0 : 0.0,
                 t.gap_wait_ns / 1e6 / ws, t.gap_wait_n / ws, t.gap_other_ns / 1e6 / ws, t.gap_other_n / ws,
                 t.overlaps);
        t.cls_ns = {};
        t.cls_n = {};
        t.work_n = {};
        t.disp_only_disp = t.overlaps = t.gap_wait_ns = t.gap_wait_n = t.gap_other_ns = t.gap_other_n = 0;
        t.busy_ns = t.segments = t.max_ns = 0;
        t.window = now;
    }
}

void Scheduler::LaneRecPush(Common::UniqueFunction<void>&& job) {
    bool wake;
    {
        std::scoped_lock lk{lane_rec_mu};
        lane_rec_q.push_back(std::move(job));
        ++lane_rec_pushed;
        wake = lane_rec_sleeping;
    }
    ++lane_rec_jobs;
    if (wake) {
        lane_rec_cv.notify_one();
    }
}

void Scheduler::LaneRecDrain() {
    std::unique_lock lk{lane_rec_mu};
    if (lane_rec_sleeping && !lane_rec_q.empty()) {
        lane_rec_cv.notify_one();
    }
    lane_rec_idle_cv.wait(lk, [this] { return lane_rec_q.empty() && !lane_rec_busy; });
}

void Scheduler::LaneRecDrainForBorrow() {
    ++lane_rec_drains;
    LaneRecDrain();
    // The worker is idle (and stays so: only this thread pushes): the back records directly.
    if (!asc_cmdbuf) {
        BeginAscCmdbufOnWorker();
    }
}

void Scheduler::LaneRecLoop() {
    Common::SetCurrentThreadName("shadPS4:LaneRecorder");
    t_lane_mode = true; // InlineCmdbuf / RecorderTransferGroup pick the lane's on this thread
    std::unique_lock lk{lane_rec_mu};
    for (;;) {
        if (lane_rec_q.empty()) {
            lane_rec_busy = false;
            lane_rec_idle_cv.notify_all();
            if (lane_rec_stop) {
                return;
            }
            lane_rec_sleeping = true;
            lane_rec_cv.wait(lk, [this] { return !lane_rec_q.empty() || lane_rec_stop; });
            lane_rec_sleeping = false;
            continue;
        }
        auto job = std::move(lane_rec_q.front());
        lane_rec_q.pop_front();
        lane_rec_busy = true;
        lk.unlock();
        if (!asc_cmdbuf) {
            BeginAscCmdbufOnWorker(); // a submit job begins its own if none is open
        }
        job();
        lk.lock();
        ++lane_rec_done;
        if (lane_rec_waiters != 0) {
            lane_rec_idle_cv.notify_all(); // q12d
        }
    }
}

void Scheduler::LaneRecSync() {
    if (!lane_rec) {
        return;
    }
    // q12e: calls / calls that waited / wait cycles since the last log (any thread: under the lock).
    static u64 s_calls = 0, s_waited = 0, s_cyc = 0, s_log = 0;
    std::unique_lock lk{lane_rec_mu};
    ++s_calls;
    const u64 now0 = __rdtsc();
    if (now0 - s_log > 17'500'000'000ULL) {
        if (s_log != 0) {
            const double secs = double(now0 - s_log) / 3.5e9;
            LOG_INFO(Render_Vulkan,
                     "q12e LaneRecSync (before deferred operations): {:.0f} calls/s, {:.0f} waited/s, "
                     "wait {:.2f} ms/s",
                     double(s_calls) / secs, double(s_waited) / secs, double(s_cyc) / 3.5e6 / secs);
        }
        s_log = now0;
        s_calls = s_waited = s_cyc = 0;
    }
    const u64 target = lane_rec_pushed;
    if (lane_rec_done >= target) {
        return;
    }
    ++s_waited;
    ++lane_rec_waiters;
    if (lane_rec_sleeping) {
        lane_rec_cv.notify_one();
    }
    lane_rec_idle_cv.wait(lk, [this, target] { return lane_rec_done >= target; });
    --lane_rec_waiters;
    s_cyc += __rdtsc() - now0;
}

void Scheduler::LaneSwitch(u32 stream) {
    if (stream == active_stream) {
        return;
    }
    if (stream == 1) {
        if (!asc_dirty) {
            open_asc_value = asc_semaphore->NextTick();
            asc_dirty = true;
            asc_wait_main = 0;
            asc_wait_site = 0;
        }
        if (!lane_rec) {
            if (!asc_cmdbuf) {
                BeginAscCmdbufOnWorker(); // the back's: lane mode records inline
            }
        } else if (synchronous_scopes != 0) {
            LaneRecDrainForBorrow(); // q12c: inside a borrow lane records run inline: drain + begin
        }
    }
    active_stream = stream;
    t_lane_mode = stream == 1;
    Rc().native_heaps_bound = false;
    VideoCore::g_record_stream = stream;
    if (stream == 1) {
        // p1e: inline lane records do not pass the worker; move RecordSeq() ("nothing recorded
        // since") so draw repeat / native joins / dispatch runs see them.
        ++synchronous_borrows;
    }
}

void Scheduler::LaneCopyCensus(u64 copy_value) {
    // q33: what holds this lane fault copy - the unfinished lane segments queued ahead of it and their main waits.
    asc_semaphore->Refresh();
    const u64 lane_known = asc_semaphore->KnownGpuTick();
    master_semaphore.Refresh();
    const u64 main_known = master_semaphore.KnownGpuTick();
    u64 max_wait = 0, ahead = 0, wid = 0;
    u32 site = 0;
    for (const auto& r : lane_seg_ring) {
        if (r.value > lane_known && r.value <= copy_value) {
            ++ahead;
            if (r.wait > max_wait) {
                max_wait = r.wait;
                site = r.site;
                wid = r.wid;
            }
        }
    }
    ++lane_copy_n;
    lane_copy_ahead += ahead;
    if (max_wait > main_known) {
        ++lane_copy_blocked;
        lane_copy_behind += max_wait - main_known;
        ++lane_copy_site_c[std::min<u32>(site, 323)];
        ++lane_copy_wid_c[(u64(site) << 48) ^ (wid & 0xffffffffffffULL)]; // q34
        if (site == 256 + 15) {
            ++lane_copy_h5[(wid >> 56) & 15]; // q35 H5
        }
    }
    const u64 now = __rdtsc();
    if (now - lane_copy_log_tsc < 17'500'000'000ULL) {
        return;
    }
    lane_copy_log_tsc = now;
    std::vector<std::pair<u64, u32>> top;
    for (u32 i = 0; i < lane_copy_site_c.size(); ++i) {
        if (lane_copy_site_c[i] != 0) {
            top.emplace_back(lane_copy_site_c[i], i);
        }
    }
    std::sort(top.begin(), top.end(), std::greater<>());
    std::string s;
    for (size_t k = 0; k < std::min<size_t>(top.size(), 10); ++k) {
        const u32 c = top[k].second;
        s += c < 256 ? fmt::format(" [raw k{} {}]", c, top[k].first)
             : c < 320 ? fmt::format(" [war s{} {}]", c - 256, top[k].first)
                       : fmt::format(" [{} {}]", c == 320 ? "label" : c == 321 ? "gds" : c == 322 ? "carry" : "other",
                                     top[k].first);
    }
    LOG_INFO(Render_Vulkan,
             "q33 LaneCopyCensus (~5 s): lane fault copies {} held behind an unfinished main batch {} (mean {:.2f} "
             "batches behind) | lane segments queued ahead {:.1f} per copy | holding site:{}",
             lane_copy_n, lane_copy_blocked,
             lane_copy_blocked ? double(lane_copy_behind) / double(lane_copy_blocked) : 0.0,
             lane_copy_n ? double(lane_copy_ahead) / double(lane_copy_n) : 0.0, s);
    {
        // q34: holding (site, draw pipeline) pairs and the s15 stamp sizes over probes.
        std::vector<std::pair<u64, u64>> w(lane_copy_wid_c.begin(), lane_copy_wid_c.end());
        std::sort(w.begin(), w.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string ws;
        for (size_t k = 0; k < std::min<size_t>(w.size(), 8); ++k) {
            ws += fmt::format(" [s{} {:012x} {}]", (w[k].first >> 48) >= 256 ? (w[k].first >> 48) - 256 : (w[k].first >> 48),
                              w[k].first & 0xffffffffffffULL, w[k].second);
        }
        const auto& h = VideoCore::g_s15_size_hist;
        LOG_INFO(Render_Vulkan,
                 "q34 holding draws:{} | s15 stamps over a probe by size <64K/256K/1M/4M/12M/32M/128M/more "
                 "{}/{}/{}/{}/{}/{}/{}/{}",
                 ws, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        lane_copy_wid_c.clear();
        VideoCore::g_s15_size_hist.fill(0);
        // q35 H5: class 0 unknown (not indirect) / 1 same particle group as the reader's args / 2 other; flips 0..3+.
        const auto h5 = [](const std::array<u64, 16>& a) {
            std::string s;
            for (u32 c = 0; c < 3; ++c) {
                s += fmt::format(" {}:", c == 0 ? "unk" : c == 1 ? "same" : "other");
                for (u32 f = 0; f < 4; ++f) {
                    s += fmt::format("{}{}", f ? "/" : "", a[c | (f << 2)]);
                }
            }
            return s;
        };
        LOG_INFO(Render_Vulkan, "q35 H5 s15 by group and flips 0/1/2/3+: held copies{} | waiting segments{} | q38 window "
                 "reads lowered by the extent check {}",
                 h5(lane_copy_h5), h5(lane_seg_h5), VideoCore::g_lane_ext_saved);
        VideoCore::g_lane_ext_saved = 0;
        LOG_INFO(Render_Vulkan, "q39 LaneProducerSuper (~5 s): 4 MiB blocks skipped {} | verify compares {} mismatches {} "
                 "(mode {})", VideoCore::g_lc_super_c[0], VideoCore::g_lc_super_c[1], VideoCore::g_lc_super_c[2],
                 VideoCore::g_lane_producer_super);
        VideoCore::g_lc_super_c[0] = VideoCore::g_lc_super_c[1] = 0;
        lane_copy_h5.fill(0);
        lane_seg_h5.fill(0);
    }
    lane_copy_n = lane_copy_blocked = lane_copy_behind = lane_copy_ahead = 0;
    lane_copy_site_c.fill(0);
}

void Scheduler::LaneWaitCapture(u32 src, u32 site) {
    // q59: the access behind the wait just raised (set by the LaneProbeWrite / LaneMainProducerFull call before it).
    if (src == 2 && site >= 256 && site < 320) {
        asc_wait_exact = VideoCore::g_lane_war_exact;
        asc_wait_tsc = VideoCore::g_lane_war_tsc;
    } else if (src == 0 || src == 1 || src == 5 || src == 6) {
        asc_wait_exact = VideoCore::g_lc_hit_exact;
        asc_wait_tsc = VideoCore::g_lc_hit_tsc;
    } else {
        asc_wait_exact = src == 2 ? 10 : src == 3 ? 11 : 12; // label fallback, gds, other
        asc_wait_tsc = 0;
    }
}

void Scheduler::LaneWaitResolve() {
    // q59: waiting segments whose batch has closed: where the access sat in its batch (record time).
    const u64 open = master_semaphore.CurrentTick();
    while (lane_wait_r < lane_wait_w) {
        const LaneWaitRec w = lane_wait_pend[lane_wait_r % lane_wait_pend.size()];
        if (w.t >= open) {
            break;
        }
        ++lane_wait_r;
        ++lw_resolved;
        ++lw_cls[std::min<u32>(w.exact, 15)];
        ++lw_frame[std::min<u32>(w.frame, 3)];
        const bool real = w.exact == 1 || w.exact == 8;
        lw_real += real ? 1 : 0;
        u32 pos = 5; // unknown
        const auto& ce = batch_close_tsc[w.t & 4095];
        if (w.tsc != 0 && ce.first == w.t && ce.second >= w.tsc) {
            u64 bt = w.t; // the batch that recorded the access: close(bt - 1) < tsc <= close(bt)
            for (u32 k = 0; k < 8 && bt > 1; ++k) {
                const auto& pe = batch_close_tsc[(bt - 1) & 4095];
                if (pe.first != bt - 1 || pe.second < w.tsc) {
                    break;
                }
                --bt;
            }
            const auto& pe = batch_close_tsc[(bt - 1) & 4095];
            const auto& be = batch_close_tsc[bt & 4095];
            if (bt != w.t) {
                pos = 4;
            } else if (pe.first == bt - 1 && be.first == bt && be.second > pe.second) {
                const double f = double(w.tsc - std::min(w.tsc, pe.second)) / double(be.second - pe.second);
                pos = f < 0.25 ? 0 : f < 0.5 ? 1 : f < 0.75 ? 2 : 3;
            }
            lw_left_cyc += ce.second - w.tsc;
            ++lw_left_n;
        }
        ++lw_pos[pos];
        lw_pos_real[pos] += real ? 1 : 0;
        lw_real_cur_late += (real && w.frame == 2 && pos == 3) ? 1 : 0;
        const u64 key = (u64{w.site} << 32) | (w.wid & 0xffffffffULL);
        if (lw_pipe.size() < 4096 || lw_pipe.contains(key)) {
            ++lw_pipe[key];
        }
    }
}

void Scheduler::LaneSegCensus(u64 wait_main, u64 value) {
    if (lane_wait_census) [[unlikely]] {
        LaneWaitResolve(); // q59
    }
    // q26 E0: which site set each segment's final main wait, and whether that batch is still unfinished.
    lane_seg_ring[lane_seg_ring_pos++ % lane_seg_ring.size()] = {value, wait_main, asc_wait_site, asc_wait_wid}; // q33/q34
    ++lane_seg_n;
    master_semaphore.Refresh();
    const u64 known = master_semaphore.KnownGpuTick();
    if (wait_main > known) {
        ++lane_seg_unf;
        lane_seg_behind += wait_main - known;
        ++lane_site_c[std::min<u32>(asc_wait_site, 323)];
        if (asc_wait_site == 256 + 15) {
            ++lane_seg_h5[(asc_wait_wid >> 56) & 15]; // q35 H5
        }
        if (lane_wait_census) [[unlikely]] {
            // q59: frame age of the waited batch vs the newest frame-end label batches the back recorded.
            u8 frame = 3;
            const u64 n = VideoCore::g_frame_end_n;
            if (n >= 2) {
                const u64 fe1 = VideoCore::g_frame_end_ticks[(n - 1) & 3];
                const u64 fe2 = VideoCore::g_frame_end_ticks[(n - 2) & 3];
                frame = wait_main <= fe2 ? 0 : wait_main <= fe1 ? 1 : 2;
            }
            if (lane_wait_w - lane_wait_r < lane_wait_pend.size()) {
                lane_wait_pend[lane_wait_w++ % lane_wait_pend.size()] = {wait_main, asc_wait_tsc, asc_wait_wid,
                                                                          asc_wait_site, asc_wait_exact, frame};
            } else {
                ++lw_dropped;
            }
        }
    }
    const u64 now = __rdtsc();
    if (now - lane_seg_log_tsc < 17'500'000'000ULL) {
        return;
    }
    lane_seg_log_tsc = now;
    std::vector<std::pair<u64, u32>> top;
    for (u32 i = 0; i < lane_site_c.size(); ++i) {
        if (lane_site_c[i] != 0) {
            top.emplace_back(lane_site_c[i], i);
        }
    }
    std::sort(top.begin(), top.end(), std::greater<>());
    std::string s;
    for (size_t k = 0; k < std::min<size_t>(top.size(), 10); ++k) {
        const u32 c = top[k].second;
        s += c < 256 ? fmt::format(" [raw k{} {}]", c, top[k].first)
             : c < 320 ? fmt::format(" [war s{} {}]", c - 256, top[k].first)
                       : fmt::format(" [{} {}]", c == 320 ? "label" : c == 321 ? "gds" : c == 322 ? "carry" : "other",
                                     top[k].first);
    }
    std::string ra;
    for (u32 i = 0; i < VideoCore::g_lprobe_all_c.size(); ++i) {
        if (VideoCore::g_lprobe_all_c[i] != 0) {
            ra += fmt::format(" s{}={}", i, VideoCore::g_lprobe_all_c[i]);
        }
    }
    LOG_INFO(Render_Vulkan,
             "q26 LaneSegCensus (~5 s): segments {} waiting on an unfinished main batch {} (mean {:.2f} batches "
             "behind) | by site:{} | probe ReadAll by site:{} | probes {}",
             lane_seg_n, lane_seg_unf, lane_seg_unf ? double(lane_seg_behind) / double(lane_seg_unf) : 0.0, s, ra,
             VideoCore::g_lprobe_n.load(std::memory_order_relaxed));
    if (lane_wait_census) {
        std::vector<std::pair<u64, u64>> tp;
        tp.reserve(lw_pipe.size());
        for (const auto& [k, v] : lw_pipe) {
            tp.emplace_back(v, k);
        }
        std::sort(tp.begin(), tp.end(), std::greater<>());
        std::string ps;
        for (size_t k = 0; k < std::min<size_t>(tp.size(), 10); ++k) {
            const u32 site = static_cast<u32>(tp[k].second >> 32);
            const u32 wid = static_cast<u32>(tp[k].second);
            ps += site < 256   ? fmt::format(" [raw k{} {:08x} {}]", site, wid, tp[k].first)
                  : site < 320 ? fmt::format(" [war s{} {:08x} {}]", site - 256, wid, tp[k].first)
                               : fmt::format(" [{} {}]", site == 320 ? "label" : site == 321 ? "gds" : "other",
                                             tp[k].first);
        }
        const auto& q = lw_cls;
        LOG_INFO(Render_Vulkan,
                 "q59 LaneWaitCensus (~5 s): waiting segments resolved {} (dropped {}) | frame age vs the back's "
                 "flips ({} seen): >=2 frames old {} previous frame {} current frame {} unknown {} | overlap: war exact "
                 "{} / war window, no byte overlap {} / war older-tick floor {} / war ReadAll floor {} / war new or "
                 "re-based probe {} / war list overflow {} / war grown probe {} / raw note {} / raw coarse {} / label "
                 "fallback {} / gds {} / other {} / unknown {} | access position in its batch (record time) <25% {} "
                 "25-50% {} 50-75% {} >=75% {} earlier batch {} unknown {} | record time from the access to the waited "
                 "batch's close {:.3f} ms mean | real (byte / note overlap) {} of which current frame and >=75% into "
                 "its batch {} | real by position <25% {} 25-50% {} 50-75% {} >=75% {} earlier batch {} unknown {} | "
                 "label {:#x} writes {} | top (site, pipeline):{}",
                 lw_resolved, lw_dropped, VideoCore::g_frame_end_n, lw_frame[0], lw_frame[1], lw_frame[2], lw_frame[3],
                 q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[0], lw_pos[0], lw_pos[1],
                 lw_pos[2], lw_pos[3], lw_pos[4], lw_pos[5],
                 lw_left_n ? double(lw_left_cyc) / double(lw_left_n) / 3.5e6 : 0.0, lw_real, lw_real_cur_late,
                 lw_pos_real[0], lw_pos_real[1], lw_pos_real[2], lw_pos_real[3], lw_pos_real[4], lw_pos_real[5],
                 VideoCore::g_frame_label_addr, VideoCore::g_frame_label_hits, ps);
        lw_cls.fill(0);
        lw_frame.fill(0);
        lw_pos.fill(0);
        lw_pos_real.fill(0);
        lw_left_cyc = lw_left_n = lw_real = lw_real_cur_late = lw_resolved = lw_dropped = 0;
        lw_pipe.clear();
    }
    lane_seg_n = lane_seg_unf = lane_seg_behind = 0;
    lane_site_c.fill(0);
    VideoCore::g_lprobe_all_c.fill(0);
}

void Scheduler::LaneSubmit() {
    VideoCore::LbScope lb_scope{8}; // q11a
    ++VideoCore::g_lb[9];
    const u64 value = open_asc_value;
    const u64 wait_main = asc_wait_main;
    if (VideoCore::g_lane_seg_census) [[unlikely]] {
        LaneSegCensus(wait_main, value);
    }
    asc_dirty = false;
    if (async_submit &&
        wait_main > async_submitted_tick.load(std::memory_order_acquire)) [[unlikely]] {
        ++lane_c[3]; // wait-before-signal on the main timeline (never a cycle: see LaneCutForFlush)
    }
    // q12b: queued only outside a borrow; inside one (lane or main borrow -> nested flush) the
    // recorder is drained and the segment is submitted inline.
    const bool queued = lane_rec && synchronous_scopes == 0;
    // q12f (diagnostic): SHADPS4_LANE_REC_WAIT=1 the recorder waits for each segment on the GPU,
    // =2 the back waits until the recorder has submitted it.
    static const int rec_wait = [] {
        const char* e = std::getenv("SHADPS4_LANE_REC_WAIT");
        const int m = e ? std::atoi(e) : 0;
        if (m != 0) {
            LOG_INFO(Render_Vulkan, "q12f: SHADPS4_LANE_REC_WAIT={}", m);
        }
        return m;
    }();
    if (queued) {
        // q12: the lane recorder ends and submits it after the segment's queued commands.
        if (rec_wait == 1) {
            LaneRecPush([this, value, wait_main] {
                AscSubmitOnWorker(value, wait_main);
                asc_semaphore->Wait(value);
            });
        } else {
            LaneRecPush([this, value, wait_main] { AscSubmitOnWorker(value, wait_main); });
        }
        if (rec_wait == 2) {
            LaneRecDrain();
        }
    } else {
        if (lane_rec) {
            LaneRecDrain(); // the worker idle: the back owns asc_cmdbuf now
        }
        AscSubmitOnWorker(value, wait_main); // inline: the back owns asc_cmdbuf in lane mode
    }
    asc_last_submitted.store(value, std::memory_order_release);
    ++lane_c[0];
    if (t_lane_mode) {
        // Still recording lane work: continue in a new segment with the same main wait (the rest
        // of the record depends on the same main batches).
        open_asc_value = asc_semaphore->NextTick();
        asc_dirty = true;
        asc_wait_main = wait_main;
        if (!queued) {
            BeginAscCmdbufOnWorker(); // q12b: with the submit queued the recorder begins it
        }
    }
}

void Scheduler::LaneCutForFlush(SubmitInfo& info) {
    // The batch being flushed is CurrentTick(). A segment that does not wait on it is submitted
    // first (its value <= AscAt(this tick): R0), then this batch's lane wait is chosen, then a
    // segment deferred behind this batch is submitted - after the wait is chosen, so this batch
    // never waits on a segment that waits on it.
    const u64 tick = master_semaphore.CurrentTick();
    const bool deferred = asc_dirty && asc_wait_main >= tick;
    if (asc_dirty && !deferred) {
        LaneSubmit();
    }
    // Rule B: the game consumes ASC output after an ASC label; main waits on the newest segment
    // that ended at one (plus explicit raises in g_main_wait_asc).
    VideoCore::g_main_wait_asc = std::max(VideoCore::g_main_wait_asc, lane_label_value);
    const u64 need = AscMainWaitValue(false);
    if (need > asc_waited_by_main) {
        ASSERT(info.num_wait_semas != 1);
        info.AddWait(asc_semaphore->Handle(), need);
        asc_waited_by_main = need;
        ++lane_c[11];
    }
    if (deferred) {
        ++lane_c[1];
        LaneSubmit();
    }
}

u64 Scheduler::LaneCutNow() {
    if (!lane_enabled || !asc_dirty) {
        return 0;
    }
    const u64 value = open_asc_value;
    if (asc_wait_main >= master_semaphore.CurrentTick()) {
        ++lane_c[2];
        Flush(); // submits the open main batch, then this segment (LaneCutForFlush)
        return value;
    }
    LaneSubmit();
    return value;
}

void Scheduler::LaneLabelCut() {
    ++lane_c[4];
    // p1e: mark only; the segment is cut at the next main flush (or by a fault download).
    lane_label_value = asc_dirty ? open_asc_value : asc_last_submitted.load();
}

void Scheduler::LaneFallbackMain() {
    ++lane_c[6];
    if (!lane_enabled) {
        return;
    }
    u64 newest = asc_last_submitted.load();
    if (asc_dirty) {
        if (asc_wait_main >= master_semaphore.CurrentTick()) {
            newest = LaneCutNow(); // deferred: main must flush first (rare)
        } else {
            newest = open_asc_value; // p1e: pin it - submitted before this batch, which waits it
            lane_pinned = open_asc_value;
        }
    }
    VideoCore::g_main_wait_asc = std::max(VideoCore::g_main_wait_asc, newest);
}

void Scheduler::SwitchStream(u32 stream) {
    if (lane_enabled) {
        LaneSwitch(stream);
        return;
    }
    if (!asc_enabled || stream == active_stream) {
        return;
    }
    if (active_stream == 0) {
        // A render pass lives in the main command buffer only.
        EndRendering();
    }
    if (stream == 1 && !asc_dirty) {
        open_asc_value = asc_semaphore->NextTick();
        asc_dirty = true;
        asc_wait_main = 0;
    }
    active_stream = stream;
    Rc().native_heaps_bound = false;
    VideoCore::g_record_stream = stream;
    auto apply = [this, stream] {
        if (stream == 1 && !asc_cmdbuf) {
            BeginAscCmdbufOnWorker();
        }
        worker_stream = stream;
        current_cmdbuf = stream == 1 ? asc_cmdbuf : main_cmdbuf;
    };
    if (IsAsyncRecording()) {
        recording_worker->Record(std::move(apply));
    } else {
        apply();
    }
}

u64 Scheduler::SubmitAscSegment() {
    if (lane_enabled) {
        return LaneCutNow();
    }
    if (!asc_enabled || !asc_dirty) {
        return 0;
    }
    const u64 value = open_asc_value;
    const u64 wait_main = asc_wait_main;
    {
        // I2 coupling: how far back is the main batch this segment waits (0 = none).
        const u64 now = master_semaphore.CurrentTick();
        const u64 d = wait_main == 0 ? 0 : now - std::min(now, wait_main);
        ++asc_wait_dist[wait_main == 0 ? 0 : d <= 1 ? 1 : d <= 3 ? 2 : 3];
    }
    asc_dirty = false;
    asc_last_submitted = value;
    auto submit = [this, value, wait_main] { AscSubmitOnWorker(value, wait_main); };
    if (IsAsyncRecording()) {
        recording_worker->Record(std::move(submit));
        recording_worker->Publish();
    } else {
        submit();
    }
    if (active_stream == 1) {
        // Still recording ASC work: continue in a new segment. It keeps the main wait: the rest
        // of the dispatch being recorded depends on the same main batches (Opus review
        // 2026-09-28: a mid-pass-2 flush otherwise left it unordered after them).
        open_asc_value = asc_semaphore->NextTick();
        asc_dirty = true;
        asc_wait_main = wait_main;
    }
    return value;
}

void Scheduler::AscSubmitOnWorker(u64 value, u64 wait_main) {
    if (!asc_cmdbuf) {
        BeginAscCmdbufOnWorker(); // An empty segment still signals its value.
    }
    if (transfer_group[1].open) {
        CloseTransferGroupSlow(transfer_group[1], asc_cmdbuf);
    }
    if (asc_timer) {
        AscTimerEnd(value, wait_main);
    }
    Check(asc_cmdbuf.end());
    if (lane_up_on) {
        LaneUploadFlushBatch(); // q39: this segment's collected CPU halves go first (I2)
        // q36: behind every queued upload CPU half (I2) and any queued submit (one queue user, in order). With the
        // FIFO idle (done == pushed, acquire) every pushed fill already happened before this inline submit.
        if (lane_up_done.load(std::memory_order_acquire) != lane_up_pushed.load(std::memory_order_relaxed)) {
            ++lane_up_submits;
            LaneUploadPush([this, cmdbuf = asc_cmdbuf, value, wait_main] { AscQueueSubmit(cmdbuf, value, wait_main); });
        } else {
            ++lane_up_inline;
            AscQueueSubmit(asc_cmdbuf, value, wait_main);
        }
    } else {
        AscQueueSubmit(asc_cmdbuf, value, wait_main);
    }
    asc_cmdbuf = nullptr;
    if (worker_stream == 1) {
        BeginAscCmdbufOnWorker();
        current_cmdbuf = asc_cmdbuf;
    }
}

void Scheduler::AscQueueSubmit(vk::CommandBuffer cmdbuf, u64 value, u64 wait_main) {
    std::unique_lock pool_lk{asc_pool_mu, std::defer_lock}; // q36
    if (lane_up_on) {
        pool_lk.lock();
    }
    const vk::Semaphore wait_sema = master_semaphore.Handle();
    const vk::Semaphore signal_sema = asc_semaphore->Handle();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
    const u32 wait_count = wait_main != 0 ? 1u : 0u;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = wait_count,
        .pWaitSemaphoreValues = &wait_main,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &value,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = wait_count,
        .pWaitSemaphores = &wait_sema,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1U,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_sema,
    };
    const auto result = instance.GetAscQueue().submit(submit_info);
    if (result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault(); // q12g
    }
    ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
               "Device lost during ASC submit (q12f: value {} wait_main {} main gpu {} asc gpu {} "
               "recorder thread {})",
               value, wait_main, master_semaphore.KnownGpuTick(), asc_semaphore->KnownGpuTick(),
               t_lane_mode && lane_rec);
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (const u64 value = AscAt(op.gpu_tick); value != 0) {
            asc_semaphore->Wait(value); // R0
        }
        if (stoken.stop_requested()) {
            break;
        }

        LaneRecSync(); // q12d
        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (render_skip && dirty_state.scissors) [[unlikely]] {
        // SHADPS4_WATER_REFLECTION_NO_BLUR: the pass rasterizes nothing.
        dirty_state.scissors = false;
        Scissors none = scissors;
        for (auto& r : none) {
            r = vk::Rect2D{.offset = {0, 0}, .extent = {0, 0}};
        }
        cmdbuf.setScissorWithCount(none);
    }
    if (render_scale > 1) [[unlikely]] {
        // SHADPS4_WATER_REFLECTION_SCALE: a pass on scaled backings. Window coordinates scale
        // linearly, so the viewport rectangle and the scissors scale with the target.
        const float s = static_cast<float>(render_scale);
        if (dirty_state.viewports) {
            dirty_state.viewports = false;
            Viewports scaled = viewports;
            const float max_w = static_cast<float>(instance.GetMaxViewportWidth());
            const float max_h = static_cast<float>(instance.GetMaxViewportHeight());
            constexpr float bound = 32767.0f; // viewportBoundsRange upper end (NVIDIA)
            for (auto& v : scaled) {
                v.x *= s;
                v.y *= s;
                v.width *= s;
                v.height *= s;
                if (v.width > max_w || v.height > max_h || v.x + v.width > bound ||
                    v.y + v.height > bound) {
                    // The clip-disabled 16K window: never expected in the reflection chain.
                    VideoCore::WaterScale::g_counters.clip_disabled.fetch_add(
                        1, std::memory_order_relaxed);
                    v.width = std::min({v.width, max_w, bound - v.x});
                    v.height = std::min({v.height, max_h, bound - v.y});
                }
            }
            cmdbuf.setViewportWithCount(scaled);
        }
        if (dirty_state.scissors) {
            dirty_state.scissors = false;
            Scissors scaled = scissors;
            for (auto& r : scaled) {
                r.offset.x = static_cast<s32>(std::min<s64>(s64(r.offset.x) * render_scale, 0x3fffffff));
                r.offset.y = static_cast<s32>(std::min<s64>(s64(r.offset.y) * render_scale, 0x3fffffff));
                r.extent.width = static_cast<u32>(std::min<u64>(u64(r.extent.width) * render_scale, 0x3fffffff));
                r.extent.height = static_cast<u32>(std::min<u64>(u64(r.extent.height) * render_scale, 0x3fffffff));
            }
            cmdbuf.setScissorWithCount(scaled);
        }
    }
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
        // Not dynamic in the graphics pipelines: only the next bind overrides it, so the next
        // draw must bind even if its pipeline is already bound (SHADPS4_SKIP_REBIND).
        GraphicsBindTracker::Invalidate();
    }
}

} // namespace Vulkan
