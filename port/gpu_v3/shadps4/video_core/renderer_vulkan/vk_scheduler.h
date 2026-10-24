// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <source_location>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <queue>
#include <deque>
#include <vector>

#include "common/command_recording_worker.h"
#include "common/logging/log.h"

#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

/// SHADPS4_BACK_CYCLES=1 (2026-10-09): what the back thread's records spend outside
/// translation - GPU waits, submits, stream-buffer wraps. Per thread; the back logs its own.
struct BackCycleStats {
    u64 wait_calls{}, wait_blocked{}, wait_cyc{};
    u64 flush_calls{}, flush_cyc{};
    u64 stream_wraps{}, stream_waits{}, stream_wait_cyc{};
};
extern thread_local BackCycleStats t_back_cycles;
bool BackCyclesEnabled();

class Instance;
struct DrawPacketDesc;
struct DrawPacketHeader;

struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value;
    union {
        u32 is_clear;
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;
    /// SHADPS4_WATER_REFLECTION_SCALE: the attachments' backing scale (0 / 1 = the guest
    /// extent); width / height above are then the scaled extent and Scheduler::BeginRendering
    /// scales the pass's viewports and scissors at commit (DynamicState::render_scale).
    u16 scale;
    /// SHADPS4_WATER_REFLECTION_NO_BLUR: 1 = the pass draws nothing (zero scissors at commit).
    u16 skip;
    u16 reserved[2]; ///< keeps the struct free of padding (memcmp equality)

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 3> wait_semas;
    std::array<u64, 3> wait_ticks;
    std::array<vk::Semaphore, 3> signal_semas;
    std::array<u64, 3> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas++] = tick;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
    } dirty_state{};

    /// CP split FRONT_DERIVE: the command buffer's setVertexInputEXT state is unknown. Set by
    /// Invalidate, cleared by BufferCache::BindVertexBuffers when it records one; deliberately
    /// not a dirty_state bit (Scheduler::CommitDynamicState rebuilds those from scratch, and
    /// Commit does not record this state). While clear, a draw whose description id equals the
    /// one recorded last skips its setVertexInputEXT.
    bool vertex_input_unknown{true};
    /// SHADPS4_DRAW_PACKET_DELTA: the descriptors pushed by the last graphics draw packet may
    /// no longer be bound (new command buffer, a blit or post-process pass, a legacy descriptor
    /// push). Set by Invalidate and by the legacy push paths, cleared by RecordDrawPacket.
    bool packet_state_unknown{true};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};
    /// SHADPS4_WATER_REFLECTION_SCALE: the current pass's RenderState::scale (set by
    /// Scheduler::BeginRendering, which marks viewports / scissors dirty when it changes); Commit
    /// records the viewports and scissors multiplied by it. 1 = unchanged.
    u32 render_scale{1};
    bool render_skip{}; ///< SHADPS4_WATER_REFLECTION_NO_BLUR: zero scissors for this pass

    /// SHADPS4_WATER_REFLECTION_SCALE: set by Rasterizer::BeginRendering / RenderStateMemoBegin
    /// BEFORE the draw's dynamic state is committed (the commit precedes Scheduler::
    /// BeginRendering; review 2026-10-14 BUG 1), and by Scheduler::BeginRendering as a backstop.
    void SetRenderScale(u32 scale, bool skip = false) {
        scale = scale ? scale : 1u;
        if (render_scale != scale) {
            render_scale = scale;
            dirty_state.viewports = true;
            dirty_state.scissors = true;
        }
        if (render_skip != skip) {
            render_skip = skip;
            dirty_state.scissors = true;
        }
    }

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
        vertex_input_unknown = true;
        packet_state_unknown = true;
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        if (depth_bounds_min != min || depth_bounds_max != max) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }
};

/// SHADPS4_PAR_REC=<lanes> (unset / 0: one ordered recording worker): submission batches are
/// recorded on `lanes` threads (1-4), batch k on lane k % lanes, submitted in batch order
/// (notes/multithreaded-back-design-2026-10-08.md sec. 11). The scheduler falls back to 0 where
/// lanes are not supported (see Scheduler::Scheduler).
inline u32 ParRecRequested() {
    static const u32 lanes = [] {
        const char* value = std::getenv("SHADPS4_PAR_REC");
        const unsigned long n = value ? std::strtoul(value, nullptr, 10) : 0ul;
        return static_cast<u32>(n > 4 ? 4 : n);
    }();
    return lanes;
}
/// Lanes the draw scheduler actually runs (0 = the ordered worker), written once by its
/// constructor (after the fallbacks) before any draw.
inline u32 g_par_rec_lanes = 0;

/// SHADPS4_MTB_P1=1 (multi-threaded back P1d, notes/parrec-build-2026-10-08.md sec. 14): the
/// back translates split draws / dispatches through a worker rasterizer instance, into a
/// translating RecordContext with a record sink and a reserved tick - the P3 code path, run on
/// the back thread. Needs the recording worker with SHADPS4_ASYNC_SUBMIT (refused otherwise).
inline bool MtbP1Requested() {
    static const bool on = [] {
        const char* value = std::getenv("SHADPS4_MTB_P1");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}
/// The draw scheduler accepted SHADPS4_MTB_P1 (written once by its constructor).
inline bool g_mtb_p1 = false;

/// SHADPS4_SKIP_REBIND=1: the draw paths skip vkCmdBindPipeline(graphics) when the same pipeline
/// is already bound in the same command buffer. Updated in recording order (the recording worker,
/// or the back after a drain). Every command-buffer begin and every other graphics bind into a
/// scheduler command buffer (blit helper, post-process pass) invalidates it, so a missed case
/// can only cost an extra bind. Off with SHADPS4_PAR_REC >= 2: one shared slot is wrong when
/// several lanes bind at once.
struct GraphicsBindTracker {
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_SKIP_REBIND");
            return value != nullptr && value[0] == '1' && g_par_rec_lanes < 2;
        }();
        return enabled;
    }
    static void Invalidate() {
        last_pipeline.store(VkPipeline{}, std::memory_order_relaxed);
    }
    /// True if the draw must bind `pipeline` into `cmdbuf`; records it as bound.
    static bool NeedBind(vk::CommandBuffer cmdbuf, vk::Pipeline pipeline) {
        const VkCommandBuffer cb = cmdbuf;
        const VkPipeline pl = pipeline;
        if ((++binds & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "SkipRebind: {} graphics binds, {} skipped ({:.1f}%)", binds,
                     skipped, 100.0 * double(skipped) / double(binds));
        }
        if (last_cmdbuf.load(std::memory_order_relaxed) == cb &&
            last_pipeline.load(std::memory_order_relaxed) == pl) {
            ++skipped;
            return false;
        }
        last_cmdbuf.store(cb, std::memory_order_relaxed);
        last_pipeline.store(pl, std::memory_order_relaxed);
        return true;
    }
    static inline std::atomic<VkCommandBuffer> last_cmdbuf{};
    static inline std::atomic<VkPipeline> last_pipeline{};
    static inline u64 binds{};   ///< Recording order only (no concurrent writers).
    static inline u64 skipped{};
};

class Scheduler {
public:
    explicit Scheduler(const Instance& instance, bool allow_recording_worker = false);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);
    /// Main timeline only (see IsFreeMain).
    void WaitMain(u64 tick);
    /// The ASC value the batch being flushed must wait (I1 or precise; SHADPS4_ASC_I1).
    u64 AscMainWaitValue(bool no_i1);

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope. `loc` attributes the break under SHADPS4_GPU_BATCH_TIMING.
    void EndRendering(std::source_location loc = std::source_location::current());

    /// The translating context has a render pass open (diagnostic: MTB census).
    [[nodiscard]] bool IsRenderingNow() const noexcept {
        return Rc().is_rendering;
    }

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return Rc().render_state;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return Rc().dynamic_state;
    }

    // A raw handle may only be used while this scope is alive. Nested recordings run
    // inline, so helpers retaining a handle cannot race the recording worker.
    class CommandBufferAccess {
    public:
        explicit CommandBufferAccess(Scheduler& scheduler, std::source_location location)
            : scheduler{scheduler} {
            if (scheduler.synchronous_scopes == 0 && !t_lane_mode) {
                scheduler.DrainRecording();
                if (scheduler.par_rec != 0) {
                    // Every lane is idle: borrow the open batch's command buffer.
                    scheduler.current_cmdbuf = scheduler.OpenLane().cmdbuf;
                }
            }
            if (scheduler.synchronous_scopes == 0 && t_lane_mode && scheduler.lane_rec) [[unlikely]] {
                scheduler.LaneRecDrainForBorrow(); // q12: the lane recorder idle, asc_cmdbuf begun
            }
            // The worker is idle: a coalesced transfer group left open by it is closed before the
            // borrower records directly (SHADPS4_TRANSFER_COALESCE).
            scheduler.CloseTransferGroup(scheduler.InlineCmdbuf());
            // A borrower may bind legacy descriptor state behind the scheduler's back.
            RecordContext& rc = scheduler.Rc();
            rc.native_heaps_bound = false;
            ++scheduler.synchronous_scopes;
            ++scheduler.synchronous_borrows;
            ++rc.stream_seq[scheduler.active_stream];
            if (rc.sink) [[unlikely]] {
                rc.sink->MarkBoundary(); // recorded outside the sink (DrainRecording published it)
            }
            if (scheduler.recording_measure_timing) {
                ++scheduler.recording_borrow_counts[location.function_name()];
            }
        }
        ~CommandBufferAccess() {
            scheduler.Rc().native_heaps_bound = false;
            --scheduler.synchronous_scopes;
        }
        CommandBufferAccess(const CommandBufferAccess&) = delete;
        CommandBufferAccess& operator=(const CommandBufferAccess&) = delete;
        vk::CommandBuffer Handle() const {
            return scheduler.InlineCmdbuf();
        }

    private:
        Scheduler& scheduler;
    };

    [[nodiscard]] CommandBufferAccess AcquireCommandBuffer(
        std::source_location location = std::source_location::current()) {
        return CommandBufferAccess{*this, location};
    }

    void DrainRecording();
    /// Hands the recording worker everything recorded so far without waiting.
    void PublishRecording() {
        if (recording_worker) {
            PublishActiveSink();
            recording_worker->Publish();
        }
    }
    /// SHADPS4_MTB_P1: installs the translating context on the calling thread (the back), once:
    /// submits what was recorded so far, then every later command of this thread goes through
    /// the context's sink (P1d, serial). False when the mode is off.
    bool InstallMtbContext(bool reserve_tick = true);
    /// SHADPS4_MTB_P3 (parked translation worker): the calling thread shares the installed
    /// context (it records only while the installing thread is parked on it). False if none.
    bool AttachMtbContext();
    /// SHADPS4_MTB_P1: the back is leaving its loop - end its pass and publish its sink.
    void MtbBackExit();
    /// SHADPS4_MTB_P3: a translating thread reserves the open batch's tick for its next run.
    void ReserveTickForRun() noexcept {
        master_semaphore.Reserve(master_semaphore.CurrentTickRaw());
    }
    /// SHADPS4_MTB_P1_BREAK (pricing option b): a render pass ended before a packet.
    void NoteMtbBreak() noexcept {
        ++mtb_breaks;
    }
    void CommitDynamicState();
    /// SHADPS4_DRAW_PACKET: true when CommitDynamicState would record a command right now (the
    /// resource packet of a draw must be recorded before it to keep the legacy command order).
    [[nodiscard]] bool CommitDynamicStateWouldRecord() const;
    /// SHADPS4_DRAW_PACKET: true when BeginRendering(state) would record nothing.
    [[nodiscard]] bool RenderingMatches(const RenderState& state) const noexcept {
        const RecordContext& rc = Rc();
        return rc.is_rendering && rc.render_state == state;
    }
    /// SHADPS4_DRAW_PACKET (vk_draw_packet.h): records one worker command carrying the draw's
    /// buffer barriers, push constants, push descriptors and optionally the draw / dispatch, as
    /// flat bytes; the worker rebuilds the WriteDescriptorSets. The caller has already done the
    /// command-processor side of the barriers (Pipeline::BeginBufferBarriers) and guarantees
    /// IsAsyncRecording() and DrawPacketPayloadBytes(..) <= DrawPacketMaxBytes.
    void RecordDrawPacket(const DrawPacketDesc& desc);
    /// Diagnostic (SHADPS4_ASC_RING_LATENCY + SHADPS4_ASYNC_RECORDING_TIMING=1): logs the recording
    /// worker's counters since the previous call. Command-processor thread (the producer) only.
    void LogRecordingWindow();
    /// SHADPS4_BACK_CYCLES: the producer's cumulative waits on the recording worker (ns; needs
    /// SHADPS4_ASYNC_RECORDING_TIMING=1): {free chunk space, drains}. Producer thread only.
    std::pair<u64, u64> ProducerWaitNs() const {
        if (!recording_worker) {
            return {0, 0};
        }
        const auto c = recording_worker->GetCounters();
        return {c.space_wait_ns, c.drain_ns};
    }
    void PushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                         std::span<const vk::WriteDescriptorSet> writes);

    /// Multi-threaded back P1c: commands a translating thread records into chunks of the
    /// recording worker, queued in stream order when the command processor publishes them.
    using RecordSink = Common::CommandRecordingWorker<>::Sink;
    /// A new sink on the recording worker (null without one).
    [[nodiscard]] std::unique_ptr<RecordSink> NewRecordSink() {
        return recording_worker ? std::make_unique<RecordSink>(*recording_worker) : nullptr;
    }
    /// Command processor: queues everything recorded so far, then the sink's commands.
    void PublishRecordSink(RecordSink& sink) {
        recording_worker->PublishSink(sink);
    }

    bool IsAsyncRecording() const {
        return recording_worker && synchronous_scopes == 0 && !t_lane_mode;
    }
    /// q17 (SHADPS4_LANE_COALESCE=1): an inline transfer record on the lane (back, outside a borrow) keeps the
    /// lane's transfer group open; the next non-transfer record, a borrow or the segment's submit closes it.
    bool LaneCoalesce() const noexcept {
        static const bool on = [] {
            const char* e = std::getenv("SHADPS4_LANE_COALESCE");
            return e && e[0] == '1';
        }();
        return on && t_lane_mode && synchronous_scopes == 0 && !lane_rec;
    }
    /// The command buffer an inline record writes into: the lane's in lane mode.
    vk::CommandBuffer InlineCmdbuf() const noexcept {
        return t_lane_mode ? asc_cmdbuf : current_cmdbuf;
    }
    /// Command-processor thread: moves whenever anything is recorded (worker commands) or the
    /// command buffer is borrowed directly. Equal values = nothing recorded in between.
    u64 RecordSeq() const {
        // A translating context with a sink counts its own commands (P1c).
        if (const RecordSink* const sink = Rc().sink) [[unlikely]] {
            return sink->Seq();
        }
        return (recording_worker ? recording_worker->AcceptedCommandsProducer() : 0) +
               synchronous_borrows;
    }

    /// Descriptor heap binds are invalidated by legacy descriptor state and do not survive a
    /// command buffer. Tracked in recording order on the command-processor thread.
    bool NativeHeapsBound() const {
        return Rc().native_heaps_bound && synchronous_scopes == 0;
    }
    void MarkNativeHeapsBound() {
        Rc().native_heaps_bound = true;
    }
    void InvalidateNativeHeaps() {
        Rc().native_heaps_bound = false;
    }

    /// IsTransfer: the command records only buffer copies that call BeginTransferOnWorker
    /// (SHADPS4_TRANSFER_COALESCE); every other command first closes an open transfer group.
    template <bool IsTransfer = false, typename F>
    void Record(F&& command) {
        ++Rc().stream_seq[active_stream];
        if (LaneRecQueues()) [[unlikely]] {
            // q12: the lane recorder records it (closes the lane's transfer group around it).
            LaneRecPush([this, command = std::forward<F>(command)]() mutable {
                const vk::CommandBuffer cmdbuf = asc_cmdbuf;
                CloseTransferGroup(cmdbuf);
                command(cmdbuf);
                CloseTransferGroup(cmdbuf);
            });
            return;
        }
        if (!IsAsyncRecording()) {
            const vk::CommandBuffer cmdbuf = InlineCmdbuf();
            if constexpr (IsTransfer) {
                if (LaneCoalesce()) {
                    command(cmdbuf); // q17: the lane's group stays open across uploads
                    return;
                }
            }
            CloseTransferGroup(cmdbuf);
            command(cmdbuf);
            CloseTransferGroup(cmdbuf); // no group survives a synchronous record
            return;
        }
        // The handle is read when the worker runs the command: with SHADPS4_ASYNC_SUBMIT the
        // worker itself ends, submits and replaces the command buffer, in recording order.
        WorkerRecord([this, command = std::forward<F>(command)]() mutable {
            const vk::CommandBuffer cmdbuf = RecorderCmdbuf();
            if constexpr (!IsTransfer) {
                CloseTransferGroup(cmdbuf);
            }
            command(cmdbuf);
        });
    }

    /// SHADPS4_PAR_REC: Record / RecordArrays for a command that touches state shared across
    /// batches when it runs (guest-memory reads and page protection of the deferred uploads, a
    /// readback-queue submission): it waits until every earlier batch has been submitted (and so
    /// fully recorded), which restores the single recorder's order for it. Without lanes, or
    /// recorded inline (every lane drained), it is plain Record / RecordArrays.
    template <bool IsTransfer = false, typename F>
    void RecordOrdered(F&& command) {
        if (par_rec == 0 || !IsAsyncRecording()) {
            Record<IsTransfer>(std::forward<F>(command));
            return;
        }
        const u64 batch = recording_worker->ProducerBatch();
        Record<IsTransfer>([this, batch, command = std::forward<F>(command)](
                               vk::CommandBuffer cmdbuf) mutable {
            if (WaitRecTurn(batch, false)) {
                command(cmdbuf);
            }
        });
    }
    template <bool IsTransfer = false, typename F, typename... T>
    void RecordArraysOrdered(F&& command, std::span<const T>... arrays) {
        if (par_rec == 0 || !IsAsyncRecording()) {
            RecordArrays<IsTransfer>(std::forward<F>(command), arrays...);
            return;
        }
        const u64 batch = recording_worker->ProducerBatch();
        RecordArrays<IsTransfer>(
            [this, batch, command = std::forward<F>(command)](vk::CommandBuffer cmdbuf,
                                                              auto... spans) mutable {
                if (WaitRecTurn(batch, false)) {
                    command(cmdbuf, spans...);
                }
            },
            arrays...);
    }

    // Clone transient arrays into the same unpublished chunk as their callback.
    // Factories never execute guest-memory reads or reenter the scheduler.
    template <bool IsTransfer = false, typename F, typename... T>
    void RecordArrays(F&& command, std::span<const T>... arrays) {
        static_assert((std::is_trivially_copyable_v<T> && ...));
        ++Rc().stream_seq[active_stream];
        if (LaneRecQueues()) [[unlikely]] {
            // q12: the arrays are copied into the job.
            LaneRecPush([this, command = std::forward<F>(command),
                         copies = std::tuple{std::vector<T>(arrays.begin(), arrays.end())...}]() mutable {
                const vk::CommandBuffer cmdbuf = asc_cmdbuf;
                CloseTransferGroup(cmdbuf);
                std::apply([&](auto&... v) { command(cmdbuf, std::span<const T>{v}...); }, copies);
                CloseTransferGroup(cmdbuf);
            });
            return;
        }
        if (!IsAsyncRecording()) {
            const vk::CommandBuffer cmdbuf = InlineCmdbuf();
            if constexpr (IsTransfer) {
                if (LaneCoalesce()) {
                    command(cmdbuf, arrays...); // q17
                    return;
                }
            }
            CloseTransferGroup(cmdbuf);
            command(cmdbuf, arrays...);
            CloseTransferGroup(cmdbuf);
            return;
        }
        const size_t bytes = (size_t{} + ... + AlignedPayloadSize(arrays.size_bytes()));
        if (bytes > 32768) {
            auto access = AcquireCommandBuffer(); // closes an open group
            command(access.Handle(), arrays...);
            CloseTransferGroup(access.Handle());
            return;
        }
        WorkerRecordWithPayload(bytes, [&](std::span<std::byte> storage) {
            auto* cursor = storage.data();
            auto copies = std::tuple{CopyPayload(cursor, arrays)...};
            return [this, command = std::forward<F>(command), copies]() mutable {
                const vk::CommandBuffer cmdbuf = RecorderCmdbuf();
                if constexpr (!IsTransfer) {
                    CloseTransferGroup(cmdbuf);
                }
                std::apply([&](auto... spans) { command(cmdbuf, spans...); }, copies);
            };
        });
    }

    /// SHADPS4_TRANSFER_COALESCE=1 (notes/transfer-serialization-2026-09-28.md): consecutive
    /// transfer ops (uploads, guest buffer copies) share one barrier pair instead of one pair
    /// each; each pair drains the GPU (~4.5 us per tiny particle job in fire). Runs where the
    /// command records (worker, or the CP thread when synchronous), in recording order.
    struct TransferRange {
        VkBuffer buffer;
        u64 lo;
        u64 hi;
    };
    static bool TransferCoalesceEnabled();
    /// Called by a transfer command before its copies: opens the group with a global
    /// AllCommands -> Transfer barrier, or adds a Transfer -> Transfer barrier when this op
    /// overlaps an op already in the group (RAW / WAR / WAW between transfers).
    void BeginTransferOnWorker(vk::CommandBuffer cmdbuf, const TransferRange& read,
                               const TransferRange& write);
    /// Emits the group's Transfer -> AllCommands barrier (before any other command, a borrow,
    /// or the end of the command buffer).
    void CloseTransferGroup(vk::CommandBuffer cmdbuf) {
        TransferGroup& group = RecorderTransferGroup();
        if (group.open) [[unlikely]] {
            CloseTransferGroupSlow(group, cmdbuf);
        }
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU. With the ASC stream this is the
    /// dual-timeline retirement (R0, notes/asc-i1-relax-design-2026-09-28.md): main tick `tick`
    /// complete AND every ASC segment submitted before that main flush complete. While I1 holds,
    /// main already waited those segments, so the extra check only refreshes the ASC counter.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (!IsFreeMain(tick)) {
            return false;
        }
        return AscRetired(tick);
    }

    /// Main timeline only (fault jobs whose copy ran on main: they must not inherit ASC waits).
    [[nodiscard]] bool IsFreeMain(u64 tick) noexcept {
        if (master_semaphore.IsFree(tick)) {
            return true;
        }
        master_semaphore.Refresh();
        return master_semaphore.IsFree(tick);
    }

    /// The last ASC value submitted before main tick `tick` was flushed (0: none / ASC off).
    /// Evicted entries answer the newest submitted value (conservative).
    [[nodiscard]] u64 AscAt(u64 tick) const noexcept {
        if ((!asc_enabled && !lane_enabled) || tick == 0) {
            return 0;
        }
        const u64 i = tick % AscAtSlots;
        const u64 tag = asc_at_tag[i].load(std::memory_order_acquire);
        if (tag == tick) {
            const u64 value = asc_at_val[i].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (asc_at_tag[i].load(std::memory_order_relaxed) == tick) {
                return value;
            }
        }
        if (tag > tick) {
            // Evicted (>= 4096 flushes ago): the newest evicted value is >= the true value and
            // already old; the newest submitted value would stall on current ASC work (Opus review:
            // StreamBuffer ring wraps waited on it, 17% of the CP back).
            ++asc_at_evicted_hits;
            return asc_at_evicted.load(std::memory_order_acquire);
        }
        ++asc_at_fallback_hits; // not flushed yet (callers check main first) or being written
        return asc_last_submitted.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool AscRetired(u64 tick) noexcept {
        const u64 value = AscAt(tick);
        if (value == 0 || asc_semaphore->IsFree(value)) {
            return true;
        }
        asc_semaphore->Refresh();
        ++asc_retired_refreshes;
        const bool free = asc_semaphore->IsFree(value);
        asc_retired_blocked += !free; // R0 falsifier: ~0 while I1 holds
        return free;
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return &master_semaphore;
    }

    /// Draws recorded since the last submission, and in the last submitted batch (diagnostic).
    void NoteDraw() noexcept {
        RecordContext& rc = Rc();
        ++rc.batch_draws;
        ++rc.batch_ops;
    }
    /// Draws plus dispatches recorded since the last submission of any kind (SHADPS4_FLUSH_EVERY_OPS);
    /// NoteDraw counts direct draws, NoteOp dispatches and indirect draws.
    void NoteOp() noexcept {
        ++Rc().batch_ops;
    }
    /// A compute dispatch: NoteOp plus the per-batch dispatch count (SHADPS4_GPU_BATCH_TIMING).
    void NoteDispatch() noexcept {
        if (active_stream != 0) {
            NoteAscWork(0); // G0
            return; // ASC stream: its segments are cut at guest ASC RELEASE_MEMs.
        }
        RecordContext& rc = Rc();
        ++rc.batch_ops;
        ++rc.batch_dispatches;
    }
    /// G0 (SHADPS4_ASC_TIMING=1, back thread): what the open ASC / lane segment holds, for the per-segment
    /// GPU time split (0 dispatch, 1 upload, 2 fault copy). Counted only while stream 1 is active.
    void NoteAscWork(u32 kind) noexcept {
        if (asc_timer && active_stream != 0) [[unlikely]] {
            ++asc_timer->cur[kind];
        }
    }

    /**
     * SHADPS4_ASC_STREAM=1 (with SHADPS4_ASYNC_SUBMIT=1, SHADPS4_READBACK_QUEUE=3): eligible guest
     * async-compute dispatches are recorded into a second command stream submitted on a
     * compute-only queue (design notes/asc-stream-design-2026-09-28.md). Stream 0 = main,
     * 1 = ASC. Streams switch in recording order (the worker applies a marker), so everything
     * recorded in between lands in that stream's command buffer.
     * I1: every main submission waits on the ASC value last submitted before it, so a main tick
     * completing still implies all work recorded before it completed (tick lifetimes hold).
     * I2: an ASC segment waits on the newest main batch its bindings depend on (AddAscWaitMain);
     * only values already flushed on the CP are ever waited on (no cycle).
     */
    bool HasAscStream() const noexcept {
        return asc_enabled;
    }
    /// SHADPS4_ASC_LANE=1 (sec. 21, p1_asc_lane.py): the back records its early ASC records into a
    /// lane command buffer on queue 1 of the compute-only family (own timeline: the ASC semaphore).
    bool HasLane() const noexcept {
        return lane_enabled;
    }
    /// q27: lane-mode recording goes to the q12 lane recorder thread right now (public view of LaneRecQueues).
    bool LaneRecordsQueued() const noexcept {
        return LaneRecQueues();
    }
    /// q36 (D2 R2): SHADPS4_LANE_DEFER_UPLOAD=2/3 - lane uploads' CPU halves and the lane submits on the LaneUpload FIFO.
    bool LaneUploadOn() const noexcept {
        return lane_up_on;
    }
    /// Back thread only: queue a lane upload's CPU half (protect, then copy); the segment's submit follows it.
    void LaneUploadPush(Common::UniqueFunction<void>&& job);
    /// Blocks until every queued job ran (back thread; teardown).
    void LaneUploadDrain();
    /// q39: SHADPS4_LANE_UP_BATCH=1 - collect upload CPU halves; pushed as one job at the next submit (or every 32).
    void LaneUploadDefer(Common::UniqueFunction<void>&& job);
    void LaneUploadFlushBatch();
    /// q24: SHADPS4_CHAIN_TIMING=1 - GPU timestamps around each early ASC service run in the main batch (lane off).
    bool ChainTiming() const noexcept {
        return chain_timer != nullptr;
    }
    void ChainTimerBegin();
    void ChainTimerEnd(u32 ndisp);
    /// Back thread: recording into the lane (SwitchStream(1) with the lane on).
    static inline thread_local bool t_lane_mode = false;
    /// Rule A: the open lane segment must run after main tick `tick` (deferred if it is the open
    /// batch: submitted right after that batch's flush).
    /// p1d: `src` 0 exact producer, 1 coarse, 2 WAR (gfx label), 3 GDS, 4 carry; `cap`: never
    /// defer (an open-batch tick becomes the newest submitted batch).
    void LaneNeedMain(u64 tick, u32 src = 0, bool cap = false, u32 site = ~0u, u64 wid = 0) {
        if (tick == 0 || !asc_dirty) {
            return;
        }
        if (site == ~0u) {
            // q26 E0: compact site code (callers with better information pass it).
            site = src == 2 ? (cap ? 320u : 256u) : src == 3 ? 321u : src == 4 ? 322u : 255u;
        }
        const u64 open = master_semaphore.CurrentTick();
        if (cap && tick >= open) {
            ++lane_c[15];
            tick = open - 1;
        }
        if (tick >= open && lane_pinned == open_asc_value && asc_wait_main < open) {
            // p1e: main waits on this segment (a record kept on main): it cannot also wait on
            // the open batch - submit it now and continue in a new segment.
            ++lane_c[14];
            LaneSubmit();
            if (!asc_dirty) {
                return;
            }
        }
        if (tick > asc_wait_main) {
            const u32 prev_site = asc_wait_site; // q59
            const u64 prev_wid = asc_wait_wid;
            asc_wait_main = tick;
            asc_wait_site = site; // q26 E0
            asc_wait_wid = wid;   // q34
            if (lane_wait_census) [[unlikely]] {
                if (src == 4) {
                    asc_wait_site = prev_site; // q59: a carried wait keeps its cause
                    asc_wait_wid = prev_wid;
                } else {
                    LaneWaitCapture(src, site);
                }
            }
            const bool on_open = tick >= open;
            ++lane_c[on_open ? 10 : 9];
            if (on_open) {
                ++lane_open_src[std::min<u32>(src, 6)];
            }
        }
    }
    std::array<u64, 7> lane_open_src{};
    /// p1e: the open lane segment main already waits on (a record kept on main), 0 if none.
    u64 lane_pinned = 0;
    /// Submits the open lane segment now (a deferred one: flushes main first). Returns its value.
    u64 LaneCutNow();
    /// An ASC label was written: the open segment ends there; main batches flushed later wait it.
    void LaneLabelCut();
    /// An early record runs on main instead of the lane: main must see the lane's work so far.
    void LaneFallbackMain();
    /// A deferred open segment is submitted (flushing main) before a fault download is prepared.
    void LaneFlushIfDeferred() {
        if (lane_enabled && asc_dirty && asc_wait_main >= master_semaphore.CurrentTick()) {
            ++lane_c[13];
            LaneCutNow();
        }
    }
    void LaneCount(u32 i) noexcept {
        ++lane_c[i];
    }
    /// The open lane segment's main wait (carried across a nested leave / re-enter of lane mode).
    u64 LaneWaitMain() const noexcept {
        return asc_dirty ? asc_wait_main : 0;
    }
    /// Submits the open lane segment if it is not deferred (no main flush); else counts (14).
    void LaneSubmitIfOpen() {
        if (!lane_enabled || !asc_dirty) {
            return;
        }
        if (asc_wait_main >= master_semaphore.CurrentTick()) {
            ++lane_c[14];
            return;
        }
        LaneSubmit();
    }
    /// Counters (back thread): 0 submits, 1 deferred submits, 2 main flushes for a deferred cut,
    /// 3 submits waiting on an unsubmitted main tick, 4 label cuts, 5 lane dispatches, 6 early
    /// records kept on main, 7 lane packets, 8 main packets, 9 rule-A waits (in flight), 10 rule-A
    /// waits on the open batch, 11 main flushes waiting on the lane, 12 lane-routed dispatches
    /// that never reached BindBuffers, 13 fault-download pre-cuts, 14 pinned segments cut by an
    /// open-batch dependency (p1e), 15 WAR / GDS waits capped to the newest submitted batch (p1d).
    std::array<u64, 16> lane_c{};
    u32 ActiveStream() const noexcept {
        return active_stream;
    }
    void SwitchStream(u32 stream);
    void AddAscWaitMain(u64 main_tick) noexcept {
        if (main_tick > asc_wait_main) {
            asc_wait_site = 323; // q26 E0: other
        }
        asc_wait_main = std::max(asc_wait_main, main_tick);
    }
    /// Submits the open ASC segment (if any) and returns its timeline value, else 0.
    u64 SubmitAscSegment();
    /// Value the open ASC segment will signal (0 if none is open).
    u64 OpenAscValue() const noexcept {
        return asc_dirty ? open_asc_value : 0;
    }
    MasterSemaphore* GetAscSemaphore() noexcept {
        return asc_semaphore.get();
    }
    /// Commands recorded into one stream (switch markers excluded).
    u64 RecordSeqOf(u32 stream) const noexcept {
        return Rc().stream_seq[stream];
    }
    /// SHADPS4_GPU_BATCH_TIMING timer, or null (diagnostic).
    [[nodiscard]] class GpuBatchTimer* BatchTimer() const noexcept {
        return batch_timer.get();
    }
    [[nodiscard]] u32 BatchOps() const noexcept {
        return Rc().batch_ops;
    }
    /// Submissions made only to shorten GPU batches (SHADPS4_FLUSH_EVERY_OPS / _DRAWS, the
    /// readback early flush), counted by their callers just before Flush(). The texture cache
    /// subtracts them from its age clock (TextureCache::AgeTick): its overlap eviction counts
    /// submissions since an image's last use, and these multiply the submissions per frame.
    void NoteCadenceFlush() noexcept {
        cadence_flushes.fetch_add(1, std::memory_order_relaxed);
    }
    /// Read off the command processor thread too (TextureCache::FindImage from the video-out
    /// surface registration), hence atomic.
    [[nodiscard]] u64 CadenceFlushes() const noexcept {
        return cadence_flushes.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u32 LastBatchDraws() const noexcept {
        return last_batch_draws;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    /// SHADPS4_PENDING_OPS_LOCK=1 (r19a; upstream #4844 / #4846): a guest thread reaches this too
    /// (sceKernelMunmap -> Rasterizer::UnmapMemory -> TextureCache::FreeImage -> DeleteImage)
    /// while the command processor pops the queue, so the queue is guarded by its own mutex.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        if (pending_ops_lock) {
            std::scoped_lock lk{pending_ops_mutex};
            pending_ops.emplace(std::move(func), CurrentTick());
            pending_ops_count.fetch_add(1, std::memory_order_release);
            return;
        }
        pending_ops.emplace(std::move(func), CurrentTick());
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops.emplace(std::move(func), CurrentTick());
        }
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;

    /// SHADPS4_ASYNC_SUBMIT: timeline of the scheduler whose submissions the recording worker
    /// makes, and the highest value the worker has actually handed to vkQueueSubmit. A
    /// synchronous submission waiting on that timeline first waits for the value to be
    /// submitted, so no batch on the queue waits for a signal queued behind it.
    static std::atomic<VkSemaphore> async_timeline;
    static std::atomic<u64> async_submitted_tick;
    /// SHADPS4_ARENA_PREFETCH_RETRY (installed by the buffer cache; null: off): called with the
    /// submitted value after the recording worker's and a par-rec lane worker's submit, outside
    /// submit_mutex. Not called for the CP thread's synchronous submits.
    static std::atomic<void (*)(u64)> submitted_hook;

private:
    static constexpr size_t AlignedPayloadSize(size_t bytes) {
        constexpr size_t alignment = alignof(std::max_align_t);
        return (bytes + alignment - 1) & ~(alignment - 1);
    }
    template <typename T>
    static std::span<const T> CopyPayload(std::byte*& cursor, std::span<const T> source) {
        static_assert(alignof(T) <= alignof(std::max_align_t));
        const auto* result = reinterpret_cast<const T*>(cursor);
        if (!source.empty()) {
            std::memcpy(cursor, source.data(), source.size_bytes());
            cursor += AlignedPayloadSize(source.size_bytes());
        }
        return {result, source.size()};
    }

    void AllocateWorkerCommandBuffers();

    /// The dirty bits CommitDynamicState leaves set because their test is disabled (the half of
    /// its decision that CommitDynamicStateWouldRecord shares).
    using DynamicDirtyBits = decltype(DynamicState::dirty_state);
    [[nodiscard]] DynamicDirtyBits PostponedDynamicDirty() const;

    /// Recording-worker side of RecordDrawPacket.
    void ExecuteDrawPacket(const DrawPacketHeader& header);

    void SubmitExecution(SubmitInfo& info);
    void SubmitExecutionImpl(SubmitInfo& info);
    /// SHADPS4_ASC_LANE: enters / leaves lane mode (back thread).
    void LaneSwitch(u32 stream);
    /// SHADPS4_ASC_LANE: submits the open lane segment inline on the back.
    void LaneSubmit();
    /// SHADPS4_ASC_LANE: the lane cut and the main wait of a main flush (before AdvanceTick).
    void LaneCutForFlush(SubmitInfo& info);
    /// The newest lane segment that ended at an ASC label (rule B).
    u64 lane_label_value = 0;

    void PriorityPendingOpsThread(std::stop_token stoken);

private:
    const Instance& instance;
    std::unique_ptr<Common::CommandRecordingWorker<>> recording_worker;
    bool recording_measure_timing = false;
    std::unordered_map<std::string_view, u64> recording_borrow_counts;
    u32 synchronous_scopes = 0;
    u64 synchronous_borrows = 0;
    u64 recording_submits = 0;
    bool async_submit = false;
    std::unique_ptr<class GpuBatchTimer> batch_timer;
    void SubmitExecutionAsync(SubmitInfo& info);
    // SHADPS4_ASC_STREAM state. Worker side (or the CP while drained): main_cmdbuf, asc_cmdbuf,
    // worker_stream. CP side: the rest.
    bool asc_enabled = false;
    bool lane_enabled = false;
    std::unique_ptr<MasterSemaphore> asc_semaphore;
    std::unique_ptr<CommandPool> asc_pool;
    static constexpr u64 AscAtSlots = 4096;
    std::array<std::atomic<u64>, AscAtSlots> asc_at_tag{};
    std::array<std::atomic<u64>, AscAtSlots> asc_at_val{};
    std::atomic<u64> asc_at_evicted{0};
    mutable std::atomic<u64> asc_at_evicted_hits{0};
    mutable std::atomic<u64> asc_at_fallback_hits{0};
    std::atomic<u64> asc_retired_refreshes{0};
    std::atomic<u64> asc_retired_blocked{0};
    void NoteAscAtFlush(u64 main_tick) noexcept {
        if (!asc_enabled && !lane_enabled) {
            return;
        }
        const u64 i = main_tick % AscAtSlots;
        if (asc_at_tag[i].load(std::memory_order_relaxed) != 0) {
            // Slots are overwritten in tick order, so the evicted values are monotonic.
            asc_at_evicted.store(asc_at_val[i].load(std::memory_order_relaxed),
                                 std::memory_order_release);
        }
        asc_at_tag[i].store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        asc_at_val[i].store(asc_last_submitted.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
        asc_at_tag[i].store(main_tick, std::memory_order_release);
        if ((main_tick & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "AscR0: flushes={} evicted_hits={} fallback_hits={} retired_refreshes={} "
                     "retired_blocked={}",
                     main_tick, asc_at_evicted_hits.load(), asc_at_fallback_hits.load(),
                     asc_retired_refreshes.load(), asc_retired_blocked.load());
        }
    }
    vk::CommandBuffer main_cmdbuf;
    vk::CommandBuffer asc_cmdbuf;
    // q12 (SHADPS4_LANE_RECORDER=1): the lane recorder. Back: pushes jobs; worker: runs them in order (owns
    // asc_cmdbuf / asc_pool / transfer_group[1] / the ASC queue while it runs).
    bool lane_rec = false;
    std::mutex lane_rec_mu;
    std::condition_variable lane_rec_cv, lane_rec_idle_cv;
    std::deque<Common::UniqueFunction<void>> lane_rec_q;
    bool lane_rec_busy = false;
    bool lane_rec_sleeping = false;
    bool lane_rec_stop = false;
    std::thread lane_rec_thread;
    u64 lane_rec_jobs = 0, lane_rec_drains = 0;
    u64 lane_rec_pushed = 0, lane_rec_done = 0; // q12d: under lane_rec_mu
    u32 lane_rec_waiters = 0;                   // q12d: LaneRecSync callers waiting
    /// q12d: blocks until every job queued before the call has run (any thread but the recorder).
    void LaneRecSync();
    bool LaneRecQueues() const noexcept {
        return lane_rec && t_lane_mode && synchronous_scopes == 0;
    }
    void LaneRecPush(Common::UniqueFunction<void>&& job);
    void LaneRecDrain();
    // q36: the LaneUpload FIFO (see LaneUploadPush). pushed / done: only the back pushes; done == pushed = idle.
    bool lane_up_on = false;
    bool lane_up_stop = false;         // under lane_up_mu
    bool lane_up_sleeping = false;     // under lane_up_mu
    std::mutex lane_up_mu;
    std::mutex asc_pool_mu; // q36: asc_pool Commit / begin (back) vs the FIFO's vkQueueSubmit
    std::condition_variable lane_up_cv, lane_up_idle_cv;
    std::deque<std::pair<u64, Common::UniqueFunction<void>>> lane_up_q; // {push tsc, job}
    std::atomic<u64> lane_up_pushed{0}, lane_up_done{0};
    std::vector<Common::UniqueFunction<void>> lane_up_batch; // q39 (back thread)
    std::atomic<u64> lane_up_max_lag{0}, lane_up_lag_sum{0}, lane_up_busy_cyc{0};
    u64 lane_up_jobs = 0, lane_up_submits = 0, lane_up_inline = 0, lane_up_log_tsc = 0;
    std::thread lane_up_thread;
    void LaneUploadLoop();
    void AscQueueSubmit(vk::CommandBuffer cmdbuf, u64 value, u64 wait_main);
    void LaneRecDrainForBorrow();
    void LaneRecLoop();
    u32 worker_stream = 0;
    u32 active_stream = 0;
    bool asc_dirty = false;
    u64 open_asc_value = 0;
    u64 asc_wait_main = 0;
    u32 asc_wait_site = 0; // q26 E0: the site that set asc_wait_main
    u64 asc_wait_wid = 0;  // q34: its stamping draw pipeline id (WAR s15)
    // q59 SHADPS4_LANE_WAIT_CENSUS: overlap class and record tsc of the access behind asc_wait_main; when each main
    // batch closed (record time); waiting segments kept until their batch closes; tallies.
    bool lane_wait_census = false;
    u8 asc_wait_exact = 0;
    u64 asc_wait_tsc = 0;
    std::array<std::pair<u64, u64>, 4096> batch_close_tsc{};
    struct LaneWaitRec {
        u64 t, tsc, wid;
        u32 site;
        u8 exact, frame;
    };
    std::array<LaneWaitRec, 2048> lane_wait_pend{};
    u64 lane_wait_w = 0, lane_wait_r = 0;
    std::array<u64, 16> lw_cls{};
    std::array<u64, 4> lw_frame{};
    std::array<u64, 6> lw_pos{}, lw_pos_real{};
    u64 lw_left_cyc = 0, lw_left_n = 0, lw_real = 0, lw_real_cur_late = 0, lw_resolved = 0, lw_dropped = 0;
    std::unordered_map<u64, u64> lw_pipe;
    void LaneWaitCapture(u32 src, u32 site);
    void LaneWaitResolve();
    std::unordered_map<u64, u64> lane_copy_wid_c; // q34
    std::array<u64, 16> lane_copy_h5{}, lane_seg_h5{}; // q35 H5: held s15 copies / waiting s15 segments by (class, flips)
    std::array<u64, 324> lane_site_c{};
    u64 lane_seg_n = 0, lane_seg_unf = 0, lane_seg_behind = 0, lane_seg_log_tsc = 0;
    void LaneSegCensus(u64 wait_main, u64 value);
    // q33: submitted lane segments (value, final main wait, site) and the lane fault copies' census.
    struct LaneSegRec {
        u64 value = 0, wait = 0;
        u32 site = 0;
        u64 wid = 0; // q34
    };
    std::array<LaneSegRec, 128> lane_seg_ring{};
    u64 lane_seg_ring_pos = 0;
    std::array<u64, 324> lane_copy_site_c{};
    u64 lane_copy_n = 0, lane_copy_blocked = 0, lane_copy_behind = 0, lane_copy_ahead = 0, lane_copy_log_tsc = 0;

public:
    void LaneCopyCensus(u64 copy_value);

private:
    std::atomic<u64> asc_last_submitted{0};
    std::array<u64, 4> asc_wait_dist{};
    /// SHADPS4_ASC_TIMING=1 (diagnostic): GPU time of each ASC segment on the compute queue.
    struct AscTimer {
        static constexpr u32 Slots = 4096;
        vk::UniqueQueryPool pool;
        std::array<u64, Slots> value{};
        u64 write_seq = 0, read_seq = 0;
        u32 open_slot = 0;
        bool began = false;
        double period_ns = 1.0;
        u64 busy_ns = 0, segments = 0, max_ns = 0;
        std::chrono::steady_clock::time_point window{};
        // G0 (plan-gpu-bound-verified-2026-10-12): the lane's GPU time by segment content, and the lane's idle
        // gaps split by whether the next segment waited on a main batch that was unfinished at its submit.
        std::array<u32, 3> cur{};                     // work noted into the open segment (NoteAscWork)
        std::array<std::array<u16, 3>, Slots> work{}; // per timed segment: dispatches, uploads, fault copies
        std::array<u8, Slots> waits_unfinished{};
        std::array<u64, 5> cls_ns{}, cls_n{}; // dispatch-only, upload-only, copy-only, mixed, empty
        std::array<u64, 3> work_n{};
        u64 disp_only_disp = 0, prev_end = 0, overlaps = 0;
        u64 gap_wait_ns = 0, gap_wait_n = 0, gap_other_ns = 0, gap_other_n = 0;
    };
    std::unique_ptr<AscTimer> asc_timer;
    struct ChainTimer { // q24
        static constexpr u32 Slots = 8192;
        vk::UniqueQueryPool pool; // 3 queries per slot
        std::array<u64, Slots> tick{};
        std::array<u32, Slots> ndisp{};
        std::array<u8, Slots> crossed{};
        u64 write_seq = 0, read_seq = 0, begin_tick = 0, prev_end = 0;
        u32 open_slot = 0;
        bool began = false;
        double period_ns = 1.0;
        u64 runs = 0, disp = 0, span_ns = 0, lag_ns = 0, gap_ns = 0, gaps = 0, crossed_runs = 0, max_span = 0,
            skipped = 0, bad = 0;
        std::array<u64, 6> hist{};
        std::chrono::steady_clock::time_point window{};
    };
    std::unique_ptr<ChainTimer> chain_timer;
    void ChainTimerCollect();
    void AscTimerBegin();
    void AscTimerEnd(u64 value, u64 wait_main = 0);
    void AscTimerCollect();
    u64 asc_waited_by_main = 0;
    struct TransferGroup {
        bool open = false;
        u32 num_reads = 0;
        u32 num_writes = 0;
        // 64: with 16 entries, 175k of 187k transfer->transfer barriers in fire were list overflows.
        std::array<TransferRange, 64> reads{};
        std::array<TransferRange, 64> writes{};
    };
    /// Per worker stream (0 = main, 1 = ASC command buffer); recording-order state.
    std::array<TransferGroup, 2> transfer_group{};
    void CloseTransferGroupSlow(TransferGroup& group, vk::CommandBuffer cmdbuf);
    void AscSubmitOnWorker(u64 value, u64 wait_main);
    void BeginAscCmdbufOnWorker();

    /// SHADPS4_PAR_REC: what one lane thread owns. The command processor touches a lane only
    /// while every lane is drained (borrow scopes, synchronous submission, startup).
    struct RecLane {
        vk::CommandBuffer cmdbuf{}; ///< Begun; receives the lane's current or next batch.
        std::size_t cmdbuf_slot{};  ///< Its pool slot (tagged in flight until submitted).
        std::unique_ptr<CommandPool> pool;
        TransferGroup transfer_group{};
        // Diagnostics (written by the lane, read racily by the ParRec log line).
        std::atomic<u64> submits{0};
        std::atomic<u64> submit_waits{0};  ///< Submissions that waited for an earlier batch.
        std::atomic<u64> ordered_waits{0}; ///< Ordered commands that waited likewise.
        std::atomic<u64> wait_ns{0};       ///< Time in both waits (ASYNC_RECORDING_TIMING=1).
    };
    static constexpr u32 MaxRecLanes = 4;
    /// rec_submitted: batches handed to vkQueueSubmit, in batch order; this bit = a recording
    /// command failed (wakes and releases every waiting lane).
    static constexpr u64 RecAbortBit = 1ULL << 63;
    static inline thread_local RecLane* t_rec_lane = nullptr;
    u32 par_rec = 0;
    std::array<RecLane, MaxRecLanes> rec_lanes{};
    std::atomic<u64> rec_submitted{0};
    u64 rec_log_batches = 0; ///< Command processor: batches since the last ParRec line.

    /// Command processor: the lane of the batch being recorded.
    RecLane& OpenLane() noexcept {
        return rec_lanes[recording_worker->ProducerBatch() % par_rec];
    }
    /// The command buffer a recorded command writes into: its lane's, else the ordered worker's
    /// (or, recorded inline, the borrowed one).
    vk::CommandBuffer RecorderCmdbuf() const noexcept {
        if (par_rec != 0) [[unlikely]] {
            if (RecLane* const lane = t_rec_lane) {
                return lane->cmdbuf;
            }
        }
        return current_cmdbuf;
    }
    /// The transfer group of the command buffer being recorded (SHADPS4_TRANSFER_COALESCE).
    TransferGroup& RecorderTransferGroup() noexcept {
        if (t_lane_mode) [[unlikely]] {
            return transfer_group[1]; // the lane's (back thread only)
        }
        if (par_rec != 0) [[unlikely]] {
            RecLane* const lane = t_rec_lane;
            return lane ? lane->transfer_group : OpenLane().transfer_group;
        }
        return transfer_group[worker_stream];
    }
    /// Lane side: blocks until every batch before `batch` is submitted. False = aborted.
    bool WaitRecTurn(u64 batch, bool submit);
    /// Lane side: ends the lane's command buffer, submits it in batch order, begins the next.
    void SubmitLaneOnWorker(const SubmitInfo& info, u64 signal_value, u64 batch);
    /// vkQueueSubmit of the lane's ended command buffer; the caller holds submit_mutex.
    void SubmitLaneCmdbufLocked(RecLane& lane, const SubmitInfo& info, u64 signal_value);
    /// SubmitExecution inside a borrow scope (every lane drained), on the command processor.
    void SubmitExecutionLaneSync(SubmitInfo& info);
    void LogParRec();

    u32 last_batch_draws = 0;
    std::atomic<u64> cadence_flushes{0};
    MasterSemaphore master_semaphore;
    CommandPool command_pool;
    /// SHADPS4_DRAW_PACKET_DELTA (producer thread only): what the last graphics draw packet that
    /// pushed descriptors pushed, and with which layout.
    struct PacketPushState {
        VkPipelineLayout layout{};
        std::vector<u64> table; ///< the PacketBinding entries (8 bytes each), as words
        std::vector<vk::DescriptorBufferInfo> buffer_infos;
        std::vector<vk::DescriptorImageInfo> image_infos;
        u64 packets{}, kept{}, unknown{}, layout_changed{}, differs{};
    };
    /// Translation-time recording state (multi-threaded back P1b, notes/parrec-build-2026-10-08.md
    /// sec. 6): what the thread translating draws decides about the command buffer it records
    /// into. The scheduler's own serves the command processor; a translating worker installs its
    /// own with SetThreadRecordContext (none do yet).
    struct RecordContext {
        DynamicState dynamic_state;
        PacketPushState packet_push;
        RenderState render_state{};
        bool is_rendering = false;
        u32 last_end_site = ~0u;
        bool native_heaps_bound = false;
        u32 batch_draws = 0;
        u32 batch_dispatches = 0;
        u32 batch_ops = 0;
        std::array<u64, 2> stream_seq{};
        /// P1c: where this context's recorded commands go (null = the recording worker).
        RecordSink* sink = nullptr;
        /// The scheduler the context belongs to: another scheduler used on the same thread (the
        /// presenter's flip / present schedulers on the back) keeps its own rc_main.
        const Scheduler* owner = nullptr;
    };
    RecordContext rc_main;
    /// SHADPS4_MTB_P1 (P1d): the back's translating context, installed once by
    /// InstallMtbContext. Destroyed before the recording worker (its sink owns worker chunks).
    struct MtbContext {
        RecordContext rc;
        std::unique_ptr<RecordSink> sink;
    };
    std::unique_ptr<MtbContext> mtb;
    u64 mtb_res_mismatch = 0;
    u64 mtb_breaks = 0;
    u64 mtb_batches = 0;
    u64 mtb_log_batches = 0;
    void LogMtbP1();
    /// The calling thread's translating sink (null without one).
    RecordSink* ActiveSink() noexcept {
        return record_ctx_installed ? Rc().sink : nullptr;
    }
    /// Committer: hands the translating context's unpublished commands to the recorder, before
    /// anything this thread records outside the sink (submission lambda, borrow, drain).
    void PublishActiveSink() {
        if (record_ctx_installed) [[unlikely]] {
            if (RecordSink* const sink = Rc().sink; sink && !sink->Empty()) {
                recording_worker->PublishSink(*sink);
            }
        }
    }
    /// master_semaphore.NextTick(); with a translating context on this thread also checks that
    /// its reservation was the batch's tick, reserves the next one and marks the sink boundary.
    u64 AdvanceTick() {
        const u64 signal_value = master_semaphore.NextTick();
        if (record_ctx_installed) [[unlikely]] {
            MtbTickAdvanced(signal_value);
        }
        return signal_value;
    }
    void MtbTickAdvanced(u64 signal_value);
    /// The recording worker, or the translating context's sink.
    template <typename F>
    void WorkerRecord(F&& command) {
        if (RecordSink* const sink = Rc().sink) [[unlikely]] {
            sink->Record(std::forward<F>(command));
            return;
        }
        recording_worker->Record(std::forward<F>(command));
    }
    template <typename Factory>
    void WorkerRecordWithPayload(size_t bytes, Factory&& factory) {
        if (RecordSink* const sink = Rc().sink) [[unlikely]] {
            sink->RecordWithPayload(bytes, std::forward<Factory>(factory));
            return;
        }
        recording_worker->RecordWithPayload(bytes, std::forward<Factory>(factory));
    }
    static inline thread_local RecordContext* t_record_ctx = nullptr;
    /// Set at startup by the draw scheduler when SHADPS4_MTB_P1 is accepted (before any other
    /// thread reads it): without it Rc() costs one predictable branch on the back (the
    /// thread_local read was ~15-30 per draw; s259, sec. 12).
    static inline bool record_ctx_installed = false;
    RecordContext& Rc() noexcept {
        if (!record_ctx_installed) [[likely]] {
            return rc_main;
        }
        RecordContext* const ctx = t_record_ctx;
        return ctx && ctx->owner == this ? *ctx : rc_main;
    }
    const RecordContext& Rc() const noexcept {
        if (!record_ctx_installed) [[likely]] {
            return rc_main;
        }
        const RecordContext* const ctx = t_record_ctx;
        return ctx && ctx->owner == this ? *ctx : rc_main;
    }
    vk::CommandBuffer current_cmdbuf;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    /// r19a SHADPS4_PENDING_OPS_LOCK: guards pending_ops (never held while a callback runs);
    /// the count lets the per-draw empty check stay lock-free.
    std::mutex pending_ops_mutex;
    std::atomic<u64> pending_ops_count{0};
    /// SHADPS4_PENDING_OPS_LOCK=1 (read once; checked on every draw).
    static inline const bool pending_ops_lock = [] {
        const char* e = std::getenv("SHADPS4_PENDING_OPS_LOCK");
        return e != nullptr && e[0] == '1';
    }();
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    tracy::VkCtxScope* profiler_scope{};
};

} // namespace Vulkan
