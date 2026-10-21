// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <span>

#include "common/interval_set.h"
#include "common/assert.h"
#include "common/unique_function.h"
#include "g4_record_ownership.h"
#include "descriptor_delta.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_command_stream.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "vulkan/vulkan.hpp"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

class Instance;
class CommandRecorder;

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

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 8> wait_semas;
    std::array<u64, 8> wait_ticks;
    std::array<vk::PipelineStageFlags, 8> wait_stages;
    std::array<vk::Semaphore, 4> signal_semas;
    std::array<u64, 4> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;
    /// Work of another thread the command buffers depend on (see Scheduler::RequireBeforeSubmit).
    u64 upload_dependency{};

    /// The submission waits for the semaphore (tick, for a timeline semaphore) before the given
    /// pipeline stages (all of them by default).
    void AddWait(vk::Semaphore semaphore, u64 tick = 1,
                 vk::PipelineStageFlags stages = vk::PipelineStageFlagBits::eAllCommands) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas] = tick;
        wait_stages[num_wait_semas++] = stages;
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

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const CommandRecorder& cmdbuf);

    /// Pipelines bound, constants pushed and vertex input set last in the command buffer, kept by
    /// every CommandRecorder of the scheduler. With skip_redundant_binds (perf_tuning.ini), the
    /// same ones are not recorded again. Forgotten with the rest of the dynamic state (new command
    /// buffer, blits) and when the command buffer is used directly (Scheduler::CommandBuffer).
    struct BindCache {
        static constexpr u32 MaxPushBytes = 128;
        static constexpr u32 MaxVertexInputs = 32;

        struct Push {
            vk::PipelineLayout layout{};
            vk::ShaderStageFlags stages{};
            u32 offset{};
            u32 size{}; ///< 0: unknown
            std::array<std::byte, MaxPushBytes> data{};
        };

        /// Descriptors pushed for set 0 of a bind point, the contents of their writes copied.
        struct Descriptors {
            static constexpr u32 MaxWrites = 64;
            static constexpr u32 MaxInfos = 128;
            struct Write {
                u32 binding;
                u32 array_element;
                u32 count;
                vk::DescriptorType type;
            };
            struct Info {
                u64 a; ///< sampler, buffer or texel buffer view
                u64 b; ///< image view or offset
                u64 c; ///< image layout or range
                bool operator==(const Info&) const = default;
            };
            vk::PipelineLayout layout{};
            u32 num_writes{~0U}; ///< ~0U: unknown
            u32 num_infos{};
            std::array<Write, MaxWrites> writes;
            std::array<Info, MaxInfos> infos;
        };

        std::array<vk::Pipeline, 2> pipelines{}; ///< graphics, compute
        std::array<Push, 2> pushes{};             ///< graphics stages, compute stage
        std::array<Descriptors, 2> descriptors{}; ///< graphics, compute
        std::array<iss::DescriptorDelta<Descriptors::MaxWrites>, 2> descriptor_deltas;
        bool vertex_input_known{};
        u32 num_vertex_bindings{};
        u32 num_vertex_attributes{};
        std::array<vk::VertexInputBindingDescription2EXT, MaxVertexInputs> vertex_bindings{};
        std::array<vk::VertexInputAttributeDescription2EXT, MaxVertexInputs> vertex_attributes{};

        /// Changes whenever a pipeline, push constants, descriptors or a vertex input are recorded
        /// or the kept state is forgotten: unchanged, the command buffer still has what was
        /// recorded last (fast draws of the rasterizer).
        u64 epoch{1};

        void Forget() noexcept {
            ++epoch;
            pipelines = {};
            pushes[0].size = 0;
            pushes[1].size = 0;
            descriptors[0].num_writes = ~0U;
            descriptors[1].num_writes = ~0U;
            descriptor_deltas[0].Reset();
            descriptor_deltas[1].Reset();
            vertex_input_known = false;
        }

        /// Descriptor sets bound otherwise than by a push: the pushed ones are no longer known.
        void ForgetDescriptors(vk::PipelineBindPoint bind_point) noexcept {
            ++epoch;
            descriptors[bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0].num_writes = ~0U;
            descriptor_deltas[bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0].Reset();
        }

        /// Notes a pipeline bind; true when it can be left out (already bound, skipping on).
        bool SkipPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) noexcept;

        /// Notes constants pushed; true when they can be left out (same layout, stages, range
        /// and values as the last push of these stages, skipping on).
        bool SkipPush(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset, u32 size,
                      const void* values) noexcept;

        /// Notes descriptors pushed; true when they can be left out (set 0 of the bind point was
        /// pushed last with the same layout and the same writes, skipping on).
        bool SkipDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, u32 set,
                             std::span<const vk::WriteDescriptorSet> writes) noexcept;
        std::optional<u32> SelectDescriptorDeltas(vk::PipelineBindPoint bind_point,
            vk::PipelineLayout layout, u32 set, std::span<const vk::WriteDescriptorSet> writes,
            std::span<u8> selected) noexcept;

        /// Notes a vertex input set; true when it can be left out (same as the last one, with the
        /// same graphics pipeline bound since, skipping on).
        /// Fast draw verification: whether these are the kept descriptors / constants.
        bool MatchesDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                                std::span<const vk::WriteDescriptorSet> writes) const noexcept;
        bool MatchesPush(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset,
                         u32 size, const void* values) const noexcept;
        bool SkipVertexInput(std::span<const vk::VertexInputBindingDescription2EXT> bindings,
                             std::span<const vk::VertexInputAttributeDescription2EXT> attributes) noexcept;
    };
    BindCache binds{};

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
        binds.Forget();
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

using SessionFunc = Common::UniqueFunction<void>;
using SubmitFunc = Common::UniqueFunction<void, SubmitInfo&>;

/// Notified when rendering scopes begin and end, for work that cannot span them (Vulkan
/// occlusion queries begun inside a rendering scope must end in it).
class RenderingObserver {
public:
    virtual ~RenderingObserver() = default;
    /// The rendering scope has just begun.
    virtual void OnBeginRendering(const CommandRecorder& cmdbuf) = 0;
    /// The rendering scope is about to end.
    virtual void OnEndRendering(const CommandRecorder& cmdbuf) = 0;
};

class Scheduler {
public:
    /// queue: the queue the command buffers are submitted to (the graphics queue by default,
    /// or another queue of the graphics family); name: for the log.
    explicit Scheduler(const Instance& instance, vk::Queue queue = {},
                       const char* name = "graphics");
    ~Scheduler();

    // Bind before a split worker records. Graphics is the default; the ACE
    // scheduler is explicitly bound to Asc when it is created.
    void SetCpRecordingLane(iss::g4::CpLane lane) noexcept { recording_lane = lane; }
    void AssertCpRecordingLane() const noexcept {
        ASSERT_MSG(iss::g4::CurrentRecordingOwnerMatches(recording_lane),
                   "G4 frontend recorded on another owner's Scheduler");
    }

    /// The queue the sessions are submitted to.
    [[nodiscard]] vk::Queue Queue() const noexcept {
        return queue;
    }

    /// The commands of the current session depend on work of another queue: the submission of
    /// the session waits for this tick of that queue's timeline semaphore (at most 3 queues,
    /// one wait per semaphore, the highest tick). Scheduler thread only.
    void AddCrossWait(vk::Semaphore semaphore, u64 tick);

    /// Whether the current session has a cross queue wait on this semaphore above the tick.
    [[nodiscard]] u64 CrossWaitTick(vk::Semaphore semaphore) const noexcept;

    /// Waits until the command buffers signaling this tick have been handed to the driver
    /// (threaded recording submits them later than Flush). Returns at once when they have.
    void WaitSubmitted(u64 tick);

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

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();
    /// A callback from an already-submitted session could become runnable between draws.
    /// Native groups may only defer callbacks belonging to the still-unsubmitted current tick.
    [[nodiscard]] bool HasPriorPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope.
    void EndRendering();

    /// Starts a rendering scope whose contents are secondary command buffers
    /// (ISS_NATIVE_RECORDER): ended before the next inline command.
    void BeginSecondaryRendering(const RenderState& new_state);

    /// Starts a new session.
    void BeginSession();

    /// Returns the current command buffer used for uploads.
    vk::CommandBuffer UploadCommandBuffer();

    /// Sets a function to be called on every session finalization.
    void SetSessionCallback(SessionFunc&& on_session) {
        this->on_session = std::move(on_session);
    }

    /// Sets a function to be called on every scheduler submission.
    /// Called first when a session is submitted, on the submitting thread, before its cross
    /// queue waits are taken and with no lock held (ISS_ASC_NATIVE_QUEUE: the compute work
    /// recorded so far is submitted and waited for).
    void SetBeforeSubmitCallback(std::function<void()> callback) {
        on_before_submit = std::move(callback);
    }

    void SetSubmitCallback(SubmitFunc&& on_submit) {
        this->on_submit = std::move(on_submit);
    }

    /// Called after queue.submit succeeds and SubmittedTick has been published.
    /// May run on the recorder worker. The callback must only publish metadata;
    /// it must not record commands or call back into this Scheduler.
    void SetPostSubmitCallback(std::function<void(u64)> callback) {
        on_post_submit = std::move(callback);
    }

    /// The commands recorded so far read data another thread produces (staging memory the upload
    /// worker fills, work number seq): the submission sending them waits for it first.
    void RequireBeforeSubmit(u64 seq) noexcept {
        AssertCpRecordingLane();
        u64 current = submit_dependency.load(std::memory_order_relaxed);
        while (current < seq && !submit_dependency.compare_exchange_weak(
                                    current, seq, std::memory_order_relaxed)) {
        }
    }

    /// Sets the function waiting for the work given to RequireBeforeSubmit (called under
    /// submit_mutex, before each submission depending on it).
    void SetSubmitDependency(std::function<void(u64)>&& wait) {
        std::scoped_lock lk{submit_mutex};
        submit_dependency_wait = std::move(wait);
    }

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Whether a rendering scope is open in the current command buffer.
    [[nodiscard]] bool IsRendering() const noexcept {
        return is_rendering;
    }

    /// Sets the object notified of rendering scope changes (nullptr: none).
    void SetRenderingObserver(RenderingObserver* observer) noexcept {
        rendering_observer = observer;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Records the commands of the session: directly in the command buffer, or through the
    /// recording thread (threaded recording). Use this rather than CommandBuffer().
    [[nodiscard]] CommandRecorder Recorder() noexcept;

    /// Returns the current command buffer, to record in it directly. With threaded recording,
    /// waits for the recording thread to record what it was given, and the rest of the session
    /// is recorded directly by the calling thread (in order with the recorded commands).
    vk::CommandBuffer CommandBuffer();

    /// Lets this scheduler record its commands on a thread of its own when threaded_recording
    /// is set in perf_tuning.ini (from its next session). Only for a scheduler used by a single
    /// thread (the command processor).
    void AllowThreadedRecording();

    /// The command buffer to record in directly, or a null handle when the commands go to the
    /// recording thread (Stream()). For CommandRecorder.
    [[nodiscard]] vk::CommandBuffer DirectCommandBuffer() const noexcept {
        AssertCpRecordingLane();
        session_used = true;
        return direct_cmdbuf;
    }

    /// Whether commands were recorded in the current session (through CommandRecorder,
    /// CommandBuffer() or Stream()).
    [[nodiscard]] bool HasWork() const noexcept {
        return session_used;
    }

    /// The commands for the recording thread, handed to it in batches. For CommandRecorder.
    CommandStream& Stream();

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return work_semaphore.CurrentTick();
    }

    /// gpu_profile (perf_tuning.ini, command processor scheduler only): the current session
    /// carries a GPU readback (its GPU time is counted apart).
    void MarkReadbackSession() noexcept {
        session_has_readback = true;
    }

    /// gpu_profile = 2: GPU timestamps around a compute dispatch, counted for its shader.
    void ProfileDispatchBegin(u64 shader_hash);
    void ProfileDispatchEnd();

    /// gpu_profile: reads the timestamps of the submissions the GPU has finished (command
    /// processor thread, once per guest frame).
    void CollectProfile();

    /// The highest tick of the command buffers handed to the driver so far (submitted, not
    /// necessarily executed). With threaded recording, a flushed session is submitted later by
    /// the recording thread: another queue must not wait for a tick above this one (its signal
    /// would not be submitted yet).
    [[nodiscard]] u64 SubmittedTick() const noexcept {
        return submitted_tick.load(std::memory_order_acquire);
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (work_semaphore.IsFree(tick)) {
            return true;
        }
        work_semaphore.Refresh();
        return work_semaphore.IsFree(tick);
    }

    /// Returns the scheduler timeline semaphore.
    [[nodiscard]] Semaphore* GetWorkSemaphore() noexcept {
        return &work_semaphore;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        std::unique_lock lk(pending_ops_mutex);
        pending_ops.emplace(std::move(func), CurrentTick());
        num_pending_ops.fetch_add(1, std::memory_order_release);
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

    /// Defers operations until the gpu has reached each one's tick (ISS_UNIFIED_MEMORY: the
    /// fences of a command buffer handed over at once). Empties ops.
    void DeferPriorityOperationsAt(
        std::vector<std::pair<u64, Common::UniqueFunction<void>>>& ops) {
        if (ops.empty()) return;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            for (auto& [tick, func] : ops) priority_pending_ops.emplace(std::move(func), tick);
        }
        ops.clear();
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;

private:
    void EndSession();

    void SubmitExecution(SubmitInfo& info);

    void PriorityPendingOpsThread(std::stop_token stoken);

    /// Submits command buffers to the queue, signaling the scheduler timeline.
    void SubmitToQueue(std::span<const vk::CommandBuffer> cmd_buffers, SubmitInfo& info);

    /// Adds the cross queue waits of the session to the submission and forgets them.
    void TakeCrossWaits(SubmitInfo& info);

    /// Threaded recording: the recording thread records the command streams it is given in the
    /// command buffer of their session, and submits the sessions.
    void RecorderThread(std::stop_token stoken);
    /// Hands the commands recorded so far to the recording thread.
    void DispatchStream();
    /// Waits until the recording thread has recorded and submitted all it was given.
    void SyncRecorder();
    /// The command buffer of the session recorded by the recording thread (begun when needed).
    /// Recording thread, or command processor while the recording thread is idle.
    vk::CommandBuffer RecorderCommandBuffer();

    struct RecorderJob {
        std::unique_ptr<CommandStream> stream;
        bool submit{};
        SubmitInfo info{};
    };

private:
    const Instance& instance;
    iss::g4::CpLane recording_lane{iss::g4::CpLane::Graphics};
    vk::Queue queue;
    const char* name;
    Semaphore work_semaphore;
    CommandPool command_pool;
    /// Cross queue waits of the current session (AddCrossWait).
    struct CrossWait {
        vk::Semaphore semaphore;
        u64 tick;
    };
    boost::container::static_vector<CrossWait, 3> cross_waits;
    std::mutex submitted_mutex;
    std::condition_variable submitted_cv;
    DynamicState dynamic_state;
    SessionFunc on_session{};
    SubmitFunc on_submit{};
    std::function<void()> on_before_submit{};
    std::function<void(u64)> on_post_submit{};
    struct Session {
        vk::CommandBuffer upload{};
        vk::CommandBuffer primary{};
    };
    std::vector<Session> sessions;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::recursive_mutex pending_ops_mutex;
    /// Lock free count of pending_ops: nothing to pop (and no semaphore query) when zero.
    std::atomic<u32> num_pending_ops{};
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    RenderState render_state;
    vk::RenderingFlags rendering_flags{}; ///< of the open rendering scope
    void BeginRenderingScope(const RenderState& new_state);
    bool is_rendering = false;
    RenderingObserver* rendering_observer{};
    tracy::VkCtxScope* profiler_scope{};

    /// Command buffer recorded directly by the scheduler thread in this session, null while the
    /// session is recorded by the recording thread.
    vk::CommandBuffer direct_cmdbuf{};
    mutable bool session_used{}; ///< See HasWork.
    bool threading_allowed{};
    bool stream_session{};     ///< This session started recorded by the recording thread.
    bool recorder_used{};      ///< The recording thread may still use the command pool.
    std::unique_ptr<CommandStream> stream; ///< Commands not handed to the recording thread yet.
    std::mutex recorder_mutex;
    std::condition_variable_any recorder_cv;
    std::condition_variable_any recorder_idle_cv;
    std::deque<RecorderJob> recorder_jobs;
    std::vector<std::unique_ptr<CommandStream>> free_streams;
    bool recorder_busy{};
    bool recorder_waiting{}; ///< The recording thread sleeps until it is notified.
    std::atomic<u32> recorder_queued{}; ///< Jobs given to the recording thread, not started yet.
    vk::CommandBuffer recorder_cmdbuf{}; ///< Command buffer of the recorded session.
    std::jthread recorder_thread;
    std::atomic<u64> submit_dependency{}; ///< See RequireBeforeSubmit.
    std::atomic<u64> submitted_tick{};    ///< See SubmittedTick.

    /// gpu_profile state (vk_scheduler.cpp), created when first enabled.
    struct GpuProfile;
    std::unique_ptr<GpuProfile> profile;
    bool session_has_readback{};
    /// Records the timestamp opening the session (gpu_profile).
    void ProfileSessionBegin();
    /// Records the timestamp closing the session, before its submission with this tick.
    void ProfileSessionEnd();
    void ProfileSessionSubmitted(u64 tick);
    std::function<void(u64)> submit_dependency_wait;
};

/**
 * Records Vulkan commands for a scheduler, with the names and arguments of the vk::CommandBuffer
 * methods it replaces: directly in its command buffer, or in the command stream of its recording
 * thread (threaded recording), where the data the arguments point to is copied.
 */
class CommandRecorder {
public:
    explicit CommandRecorder(Scheduler& scheduler_) noexcept : scheduler{&scheduler_} {}
    explicit CommandRecorder(vk::CommandBuffer cmdbuf) noexcept : fixed{cmdbuf} {}

    /// A bound state of the command buffer changes (BindCache::epoch).
    void NoteBind() const noexcept {
        if (scheduler) {
            ++scheduler->GetDynamicState().binds.epoch;
        }
    }

    /// Records a command: func(vk::CommandBuffer) runs now, or on the recording thread. The
    /// function must own copies of what it uses.
    template <typename Func>
    void Record(Func&& func) const {
        if (const vk::CommandBuffer cmdbuf = Direct()) {
            func(cmdbuf);
        } else {
            scheduler->Stream().Record(std::forward<Func>(func));
        }
    }

    void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
        if (scheduler && scheduler->GetDynamicState().binds.SkipPipeline(bind_point, pipeline)) {
            return;
        }
        NoteBind();
        Record([=](vk::CommandBuffer c) { c.bindPipeline(bind_point, pipeline); });
    }

    void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
        Record([=](vk::CommandBuffer c) { c.bindIndexBuffer(buffer, offset, type); });
    }

    void bindVertexBuffers(u32 first, u32 count, const vk::Buffer* buffers,
                           const vk::DeviceSize* offsets) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.bindVertexBuffers(first, count, buffers, offsets);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const auto* b = s.Copy(buffers, count);
        const auto* o = s.Copy(offsets, count);
        s.Record([=](vk::CommandBuffer c) { c.bindVertexBuffers(first, count, b, o); });
    }

    void bindVertexBuffers2(u32 first, u32 count, const vk::Buffer* buffers,
                            const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                            const vk::DeviceSize* strides) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const auto* b = s.Copy(buffers, count);
        const auto* o = s.Copy(offsets, count);
        const auto* z = s.Copy(sizes, count);
        const auto* t = s.Copy(strides, count);
        s.Record([=](vk::CommandBuffer c) { c.bindVertexBuffers2(first, count, b, o, z, t); });
    }

    void setVertexInputEXT(
        vk::ArrayProxy<const vk::VertexInputBindingDescription2EXT> const& bindings,
        vk::ArrayProxy<const vk::VertexInputAttributeDescription2EXT> const& attributes) const {
        if (scheduler && scheduler->GetDynamicState().binds.SkipVertexInput(
                             {bindings.data(), bindings.size()},
                             {attributes.data(), attributes.size()})) {
            return;
        }
        NoteBind();
        if (const vk::CommandBuffer c = Direct()) {
            c.setVertexInputEXT(bindings, attributes);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 num_bindings = bindings.size();
        const u32 num_attributes = attributes.size();
        const auto* b = s.Copy(bindings.data(), num_bindings);
        const auto* a = s.Copy(attributes.data(), num_attributes);
        s.Record([=](vk::CommandBuffer c) {
            c.setVertexInputEXT(num_bindings, b, num_attributes, a);
        });
    }

    void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset,
                       u32 size, const void* values) const {
        if (scheduler &&
            scheduler->GetDynamicState().binds.SkipPush(layout, stages, offset, size, values)) {
            return;
        }
        NoteBind();
        if (const vk::CommandBuffer c = Direct()) {
            c.pushConstants(layout, stages, offset, size, values);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const auto* data = s.Copy(static_cast<const std::byte*>(values), size);
        s.Record([=](vk::CommandBuffer c) { c.pushConstants(layout, stages, offset, size, data); });
    }

    void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                              u32 set,
                              vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
        std::span<const vk::WriteDescriptorSet> active{writes.data(), writes.size()};
        boost::container::static_vector<vk::WriteDescriptorSet,
            DynamicState::BindCache::Descriptors::MaxWrites> filtered;
        if (scheduler) {
            auto& cache = scheduler->GetDynamicState().binds;
            std::array<u8, DynamicState::BindCache::Descriptors::MaxWrites> selected;
            if (const auto count = cache.SelectDescriptorDeltas(bind_point, layout, set,
                                                               active, selected)) {
                if (*count == 0) return;
                if (*count != active.size()) {
                    for (u32 i = 0; i < *count; ++i) filtered.push_back(active[selected[i]]);
                    active = {filtered.data(), filtered.size()};
                }
            } else if (cache.SkipDescriptors(bind_point, layout, set, active)) {
                return;
            }
        }
        NoteBind();
        if (const vk::CommandBuffer c = Direct()) {
            c.pushDescriptorSetKHR(bind_point, layout, set,
                                   static_cast<u32>(active.size()), active.data());
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = static_cast<u32>(active.size());
        vk::WriteDescriptorSet* const copies = s.Copy(active.data(), count);
        for (u32 i = 0; i < count; ++i) {
            CopyDescriptorWrite(s, copies[i]);
        }
        s.Record([=](vk::CommandBuffer c) {
            c.pushDescriptorSetKHR(bind_point, layout, set, count, copies);
        });
    }

    void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                            u32 first_set, vk::ArrayProxy<const vk::DescriptorSet> const& sets,
                            vk::ArrayProxy<const u32> const& dynamic_offsets) const {
        if (scheduler) {
            scheduler->GetDynamicState().binds.ForgetDescriptors(bind_point);
        }
        if (const vk::CommandBuffer c = Direct()) {
            c.bindDescriptorSets(bind_point, layout, first_set, sets, dynamic_offsets);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 num_sets = sets.size();
        const u32 num_offsets = dynamic_offsets.size();
        const auto* d = s.Copy(sets.data(), num_sets);
        const auto* o = s.Copy(dynamic_offsets.data(), num_offsets);
        s.Record([=](vk::CommandBuffer c) {
            c.bindDescriptorSets(bind_point, layout, first_set, num_sets, d, num_offsets, o);
        });
    }

    void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) const {
        Record([=](vk::CommandBuffer c) {
            c.draw(vertex_count, instance_count, first_vertex, first_instance);
        });
    }

    void drawIndexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset,
                     u32 first_instance) const {
        Record([=](vk::CommandBuffer c) {
            c.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
        });
    }

    void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 draw_count,
                      u32 stride) const {
        Record([=](vk::CommandBuffer c) { c.drawIndirect(buffer, offset, draw_count, stride); });
    }

    void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 draw_count,
                             u32 stride) const {
        Record([=](vk::CommandBuffer c) {
            c.drawIndexedIndirect(buffer, offset, draw_count, stride);
        });
    }

    void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
                           vk::DeviceSize count_offset, u32 max_draw_count, u32 stride) const {
        Record([=](vk::CommandBuffer c) {
            c.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count,
                                stride);
        });
    }

    void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset,
                                  vk::Buffer count_buffer, vk::DeviceSize count_offset,
                                  u32 max_draw_count, u32 stride) const {
        Record([=](vk::CommandBuffer c) {
            c.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count,
                                       stride);
        });
    }

    void dispatch(u32 x, u32 y, u32 z) const {
        Record([=](vk::CommandBuffer c) { c.dispatch(x, y, z); });
    }

    void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
        Record([=](vk::CommandBuffer c) { c.dispatchIndirect(buffer, offset); });
    }

    void pipelineBarrier2(const vk::DependencyInfo& info) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.pipelineBarrier2(info);
            return;
        }
        CommandStream& s = scheduler->Stream();
        vk::DependencyInfo copy = info;
        copy.pMemoryBarriers = s.Copy(info.pMemoryBarriers, info.memoryBarrierCount);
        copy.pBufferMemoryBarriers =
            s.Copy(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount);
        copy.pImageMemoryBarriers = s.Copy(info.pImageMemoryBarriers, info.imageMemoryBarrierCount);
        s.Record([copy](vk::CommandBuffer c) { c.pipelineBarrier2(copy); });
    }

    void beginRendering(const vk::RenderingInfo& info) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.beginRendering(info);
            return;
        }
        CommandStream& s = scheduler->Stream();
        vk::RenderingInfo copy = info;
        copy.pColorAttachments = s.Copy(info.pColorAttachments, info.colorAttachmentCount);
        copy.pDepthAttachment = s.Copy(info.pDepthAttachment, info.pDepthAttachment ? 1 : 0);
        copy.pStencilAttachment = s.Copy(info.pStencilAttachment, info.pStencilAttachment ? 1 : 0);
        s.Record([copy](vk::CommandBuffer c) { c.beginRendering(copy); });
    }

    void endRendering() const {
        Record([](vk::CommandBuffer c) { c.endRendering(); });
    }

    void copyBuffer(vk::Buffer src, vk::Buffer dst,
                    vk::ArrayProxy<const vk::BufferCopy> const& regions) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.copyBuffer(src, dst, regions);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = regions.size();
        const auto* r = s.Copy(regions.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.copyBuffer(src, dst, count, r); });
    }

    void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout,
                           vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.copyBufferToImage(src, dst, layout, regions);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = regions.size();
        const auto* r = s.Copy(regions.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.copyBufferToImage(src, dst, layout, count, r); });
    }

    void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst,
                           vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.copyImageToBuffer(src, layout, dst, regions);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = regions.size();
        const auto* r = s.Copy(regions.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.copyImageToBuffer(src, layout, dst, count, r); });
    }

    void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                   vk::ImageLayout dst_layout,
                   vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.copyImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = regions.size();
        const auto* r = s.Copy(regions.data(), count);
        s.Record([=](vk::CommandBuffer c) {
            c.copyImage(src, src_layout, dst, dst_layout, count, r);
        });
    }

    void resolveImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                      vk::ImageLayout dst_layout,
                      vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.resolveImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = regions.size();
        const auto* r = s.Copy(regions.data(), count);
        s.Record([=](vk::CommandBuffer c) {
            c.resolveImage(src, src_layout, dst, dst_layout, count, r);
        });
    }

    void clearColorImage(vk::Image image, vk::ImageLayout layout,
                         const vk::ClearColorValue& color,
                         vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.clearColorImage(image, layout, color, ranges);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = ranges.size();
        const auto* r = s.Copy(ranges.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.clearColorImage(image, layout, &color, count, r); });
    }

    void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                    u32 data) const {
        Record([=](vk::CommandBuffer c) { c.fillBuffer(buffer, offset, size, data); });
    }

    void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
                      const void* data) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.updateBuffer(buffer, offset, size, data);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const auto* d = s.Copy(static_cast<const std::byte*>(data), size);
        s.Record([=](vk::CommandBuffer c) { c.updateBuffer(buffer, offset, size, d); });
    }

    void beginQuery(vk::QueryPool pool, u32 query, vk::QueryControlFlags flags) const {
        Record([=](vk::CommandBuffer c) { c.beginQuery(pool, query, flags); });
    }

    void endQuery(vk::QueryPool pool, u32 query) const {
        Record([=](vk::CommandBuffer c) { c.endQuery(pool, query); });
    }

    void copyQueryPoolResults(vk::QueryPool pool, u32 first, u32 count, vk::Buffer dst,
                              vk::DeviceSize offset, vk::DeviceSize stride,
                              vk::QueryResultFlags flags) const {
        Record([=](vk::CommandBuffer c) {
            c.copyQueryPoolResults(pool, first, count, dst, offset, stride, flags);
        });
    }

    void setViewportWithCount(vk::ArrayProxy<const vk::Viewport> const& viewports) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.setViewportWithCount(viewports);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = viewports.size();
        const auto* v = s.Copy(viewports.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.setViewportWithCount(count, v); });
    }

    void setScissorWithCount(vk::ArrayProxy<const vk::Rect2D> const& scissors) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.setScissorWithCount(scissors);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = scissors.size();
        const auto* r = s.Copy(scissors.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.setScissorWithCount(count, r); });
    }

    void setDepthTestEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setDepthTestEnable(enable); });
    }

    void setDepthWriteEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setDepthWriteEnable(enable); });
    }

    void setDepthCompareOp(vk::CompareOp op) const {
        Record([=](vk::CommandBuffer c) { c.setDepthCompareOp(op); });
    }

    void setDepthBoundsTestEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(enable); });
    }

    void setDepthBounds(float min, float max) const {
        Record([=](vk::CommandBuffer c) { c.setDepthBounds(min, max); });
    }

    void setDepthBiasEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setDepthBiasEnable(enable); });
    }

    void setDepthBias(float constant, float clamp, float slope) const {
        Record([=](vk::CommandBuffer c) { c.setDepthBias(constant, clamp, slope); });
    }

    void setStencilTestEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setStencilTestEnable(enable); });
    }

    void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
                      vk::StencilOp depth_fail, vk::CompareOp compare) const {
        Record([=](vk::CommandBuffer c) { c.setStencilOp(faces, fail, pass, depth_fail, compare); });
    }

    void setStencilReference(vk::StencilFaceFlags faces, u32 reference) const {
        Record([=](vk::CommandBuffer c) { c.setStencilReference(faces, reference); });
    }

    void setStencilWriteMask(vk::StencilFaceFlags faces, u32 mask) const {
        Record([=](vk::CommandBuffer c) { c.setStencilWriteMask(faces, mask); });
    }

    void setStencilCompareMask(vk::StencilFaceFlags faces, u32 mask) const {
        Record([=](vk::CommandBuffer c) { c.setStencilCompareMask(faces, mask); });
    }

    void setPrimitiveRestartEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setPrimitiveRestartEnable(enable); });
    }

    void setRasterizerDiscardEnable(vk::Bool32 enable) const {
        Record([=](vk::CommandBuffer c) { c.setRasterizerDiscardEnable(enable); });
    }

    void setCullMode(vk::CullModeFlags mode) const {
        Record([=](vk::CommandBuffer c) { c.setCullMode(mode); });
    }

    void setFrontFace(vk::FrontFace face) const {
        Record([=](vk::CommandBuffer c) { c.setFrontFace(face); });
    }

    void setBlendConstants(const float constants[4]) const {
        const std::array<float, 4> values{constants[0], constants[1], constants[2], constants[3]};
        Record([=](vk::CommandBuffer c) { c.setBlendConstants(values.data()); });
    }

    void setColorWriteMaskEXT(u32 first,
                              vk::ArrayProxy<const vk::ColorComponentFlags> const& masks) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.setColorWriteMaskEXT(first, masks);
            return;
        }
        CommandStream& s = scheduler->Stream();
        const u32 count = masks.size();
        const auto* m = s.Copy(masks.data(), count);
        s.Record([=](vk::CommandBuffer c) { c.setColorWriteMaskEXT(first, count, m); });
    }

    void setLineWidth(float width) const {
        Record([=](vk::CommandBuffer c) { c.setLineWidth(width); });
    }

    void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
        Record([=](vk::CommandBuffer c) { c.setAttachmentFeedbackLoopEnableEXT(aspects); });
    }

    void beginDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.beginDebugUtilsLabelEXT(label);
            return;
        }
        CommandStream& s = scheduler->Stream();
        vk::DebugUtilsLabelEXT copy = label;
        copy.pLabelName = s.CopyString(label.pLabelName);
        s.Record([copy](vk::CommandBuffer c) { c.beginDebugUtilsLabelEXT(copy); });
    }

    void insertDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (const vk::CommandBuffer c = Direct()) {
            c.insertDebugUtilsLabelEXT(label);
            return;
        }
        CommandStream& s = scheduler->Stream();
        vk::DebugUtilsLabelEXT copy = label;
        copy.pLabelName = s.CopyString(label.pLabelName);
        s.Record([copy](vk::CommandBuffer c) { c.insertDebugUtilsLabelEXT(copy); });
    }

    void endDebugUtilsLabelEXT() const {
        Record([](vk::CommandBuffer c) { c.endDebugUtilsLabelEXT(); });
    }

private:
    /// The command buffer to record in now, or null to record into the stream.
    [[nodiscard]] vk::CommandBuffer Direct() const noexcept {
        return scheduler ? scheduler->DirectCommandBuffer() : fixed;
    }

    static void CopyDescriptorWrite(CommandStream& s, vk::WriteDescriptorSet& write) {
        const u32 count = write.descriptorCount;
        switch (write.descriptorType) {
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eInputAttachment:
            write.pImageInfo = s.Copy(write.pImageInfo, count);
            write.pBufferInfo = nullptr;
            write.pTexelBufferView = nullptr;
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer:
            write.pTexelBufferView = s.Copy(write.pTexelBufferView, count);
            write.pImageInfo = nullptr;
            write.pBufferInfo = nullptr;
            break;
        default:
            write.pBufferInfo = s.Copy(write.pBufferInfo, count);
            write.pImageInfo = nullptr;
            write.pTexelBufferView = nullptr;
            break;
        }
    }

    Scheduler* scheduler{};
    vk::CommandBuffer fixed{};
};

inline CommandRecorder Scheduler::Recorder() noexcept {
    AssertCpRecordingLane();
    return CommandRecorder{*this};
}

/// GPU crash diagnostics. When enabled (perf_tuning.ini gpu_checkpoints = 1), each draw and
/// dispatch leaves a checkpoint in the command buffer (VK_NV_device_diagnostic_checkpoints). When
/// the device is lost, the last checkpoints the GPU reached are logged with the work they mark,
/// together with the device fault information (VK_EXT_device_fault).
namespace GpuCrashDiag {

enum class Kind : u32 {
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
    Dispatch,
    DispatchIndirect,
    Detile,
    Tile,
    NativeSegment, ///< GPU black box: a native pass segment executed by the command processor.
    NativeDraw,    ///< GPU black box: a draw inside a native segment.
};

struct Entry {
    u64 seq;
    u64 tick;
    Kind kind;
    u32 count;     ///< Vertices/indices, or the maximum count of an indirect draw.
    u32 instances; ///< Instances, or the indirect stride.
    u32 dims[3];   ///< Render area or dispatch size (indirect: arguments read on the CPU).
    u64 hashes[2]; ///< Vertex and fragment shader, or compute shader.
    u64 address;   ///< Indirect arguments or image guest address.
};

/// GPU black box (gpu_blackbox.h): what a marked command draws into.
struct Context {
    u64 color[8];        ///< Color target guest addresses (0: none).
    u32 color_format[8]; ///< CB_COLORn_INFO.format.
    u64 depth;           ///< Depth target guest address (0: none).
    u32 blend_mask;      ///< Bit per color target with blending on.
    u32 path;            ///< 0 command processor, 1 native segment, 2 draw in a native segment.
    u64 tex_uid[8];      ///< Texture cache image uids the command samples (0: none).
    u64 tex_addr[8];     ///< Their guest addresses.
    u64 rt_uid[8];       ///< Image uids of the color targets.
    u64 depth_uid;       ///< Image uid of the depth target.
};

/// GPU black box: on (ISS_GPU_BLACKBOX, default) and the marker buffer exists.
bool Blackbox();

/// GPU black box: marks a command of the scheduler `owner` (its queue's order).
void MarkBox(const void* owner, const CommandRecorder& cmdbuf, u64 tick, const Entry& entry,
             const Context& ctx);

/// GPU black box: a new native segment number (20 bits) for MarkNativeDraw.
u32 NewNativeSegment();

/// GPU black box: marks a draw inside a native segment's secondary command buffer (any thread).
void MarkNativeDraw(VkCommandBuffer cmd, u32 segment, u32 draw, const Entry& entry,
                    const Context& ctx);

/// GPU black box: the scheduler `owner` submitted everything it marked so far.
void NoteSubmit(const void* owner, const char* name);

/// GPU black box: writes gpu-crash.txt/.json (device lost), once.
void BlackboxCrash(const char* where);

/// Returns true when checkpoints are enabled and supported.
bool Enabled();

/// Records a checkpoint for the given work in the command buffer.
void Mark(const CommandRecorder& cmdbuf, u64 tick, const Entry& entry);

/// Logs the last checkpoints reached by the GPU and the device fault information, once.
void ReportDeviceLost(const Instance& instance, const char* where);

} // namespace GpuCrashDiag

} // namespace Vulkan
