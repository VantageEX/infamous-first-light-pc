// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <coroutine>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <thread>
#include <utility>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/scout.h"
#include "video_core/amdgpu/g4_stage1_ledger.h"
#include "g4_cp_coordination.h"
#include "g4_asc_owner_pause.h"
#include "g4_fallback_packet.h"
#include "g4_dma_fallback.h"
#include "g4_graphics_control.h"
#include "g4_scheduler_timeline_bridge.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

struct PM4CmdWaitRegMem;

struct Liverpool {
    struct CommandCompletion {
        std::binary_semaphore sem{0};
        // 0 pending, 1 executing, 2 done, 3 cancelled, 4 failed.
        std::atomic<u32> state{};
    };
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    std::unique_ptr<Scout> scout;
    std::shared_ptr<ScoutSubmission> scout_submission;
    const ScoutDraw* current_scout_draw{};
    u64 gfx_submit_seq{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    bool SubmitAsc(u32 gnm_vqid, std::span<const u32> acb, u64 generation);

    /// End of a guest frame. Also measures how many guest frames the command processor is
    /// behind, and optionally waits for it to catch up (see Tuning::max_gpu_lag_frames).
    void SubmitDone() noexcept;

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0 || cp_stopping; });
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    [[nodiscard]] bool CpSplitActive() const noexcept {
        return cp_split_active.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsCurrentCpThread() const noexcept {
        return thread_context.owner == this;
    }

    using CpCoordination = iss::g4::CpCoordination<NumTotalQueues>;
    using CpLane = iss::g4::CpLane;
    using CpTick = iss::g4::CpTick;
    using QidTick = iss::g4::QidTick;
    using SealRequest = CpCoordination::Request;
    // These calls only publish short-lived metadata. A packet must be fully prepared before
    // BeginCpRecord, and no cache lock may be held while waiting for a seal request.
    bool BeginCpRecord(CpLane lane, u64 tick);
    bool EndCpRecord(CpLane lane, u64 tick, bool allow_submits = false);
    bool AddCpDependency(CpTick consumer, QidTick producer);
    void CancelCpConsumer(CpTick consumer);
    bool PublishCpSubmitted(CpLane lane, u64 through_tick);
    [[nodiscard]] u64 CpOpenTick(CpLane lane);
    [[nodiscard]] u64 CpGraphicsGeneration();
    SealRequest SealAtLeast(QidTick target);
    SealRequest SealDependency(CpTick consumer, QidTick producer);
    iss::g4::SealState WaitCpSeal(const SealRequest& request, CpLane caller);

    using FallbackRequest = iss::g4::FallbackMailbox::Request;
    // Installed before CP startup. The callback runs only at a G frontend safe point,
    // records an already prepared packet in G's pipeline partition and returns a proof;
    // it must not read guest memory or wait while holding a cache lock.
    using GraphicsFallbackRecorder = std::function<std::optional<iss::g4::GraphicsRecordingProof>(
        const iss::g4::FallbackPacket&, const QidTick& sealed_asc_prefix)>;
    bool BindGraphicsFallbackRecorder(GraphicsFallbackRecorder recorder);
    FallbackRequest EnqueueAscFallback(iss::g4::FallbackPacket packet,
                                       const SealRequest& sealed_asc_prefix);

    // A C packet may ask G to perform a prepared frontend operation at a safe point.
    // C polls Status() and yields its coroutine; G never runs this in ProcessCommands.
    using GraphicsControlHandle = iss::g4::GraphicsControlMailbox::Request;
    GraphicsControlHandle EnqueueGraphicsControl(std::function<void()> operation);
    GraphicsControlHandle EnqueueGuestGraphicsControl(std::function<void()> operation);
    GraphicsControlHandle EnqueuePausedGraphicsControl(std::function<void()> operation,
                                                       u64 sealed_tick);
    GraphicsControlHandle EnqueuePausedGuestGraphicsControl(std::function<void()> operation,
                                                            u64 sealed_tick);
    // Guest-origin readback: atomically pin C's open prefix and pause all ASC
    // qids. The G control runs only after C has submitted that prefix and ACKed
    // an idle scheduler; the caller waits for the control's terminal status.
    GraphicsControlHandle EnqueueSealingPausedGuestGraphicsControl(
        std::function<void()> operation);
    bool PauseAscForGraphicsControl(const GraphicsControlHandle& request,u64 sealed_tick);
    bool PauseAscForGuestGraphicsControl(const GraphicsControlHandle& request,u64 sealed_tick);
    [[nodiscard]] bool AscOwnerPausedForGraphics();

    /// Starts the scout thread (Tuning::scout), which reads the graphics submissions ahead of
    /// the command processor and precomputes the draws for the client.
    void StartScout(ScoutClient& client);

    /// Stops it (before the client goes away).
    void StopScout() {
        scout.reset();
    }

    const Scout* GetScout() const noexcept {
        return scout.get();
    }

    /// The scout entry of the draw the command processor is processing, null when none.
    const ScoutDraw* CurrentScoutDraw() const noexcept {
        return current_scout_draw;
    }

    /// The scout entry of the draw packet after the current one, when already published (its
    /// lookups can be prefetched a draw ahead), null when none.
    const ScoutDraw* PeekNextScoutDraw() const noexcept {
        return scout_submission ? scout_submission->Peek() : nullptr;
    }

    /// Context, config and uconfig register writes that changed a value so far (command
    /// processor thread).
    /// ISS_DRAW_PIPE_DEFER: a fence packet (graphics EOS/EOP, compute RELEASE_MEM) the command
    /// processor queued behind the draws before it instead of draining: run by the draw pipe
    /// worker in order, as the command processor runs it without the pipe.
    void RunDeferredFence(u32 opcode, const u32* words, u32 pipe_id);
    u64 ContextRegChanges() const noexcept {
        return context_reg_changes;
    }
    /// Fast draw statistics: the register dword offsets whose value changed since the
    /// rasterizer last cleared the list (only filled while note_changed_regs is set).
    bool note_changed_regs{};
    u32 num_changed_regs{};
    std::array<u32, 32> changed_regs{};
    void NoteChangedRegs(u32 reg_addr, const u32* old_values, const u32* new_values, u32 count) {
        for (u32 i = 0; i < count; ++i) {
            if (old_values[i] != new_values[i]) {
                if (num_changed_regs < changed_regs.size()) {
                    changed_regs[num_changed_regs] = reg_addr + i;
                }
                ++num_changed_regs;
            }
        }
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            auto completion = std::make_shared<CommandCompletion>();
            {
                std::scoped_lock lk{submit_mutex};
                ASSERT_MSG(!cp_stopping, "CP command submitted during shutdown");
                command_queue.emplace([completion,
                                       function = std::forward<decltype(func)>(func)]() mutable {
                    u32 pending = 0;
                    if (!completion->state.compare_exchange_strong(pending, 1u)) return;
                    try {
                        function();
                    } catch (...) {
                        completion->state.store(4u, std::memory_order_release);
                        completion->sem.release();
                        throw;
                    }
                    completion->state.store(2u, std::memory_order_release);
                    completion->sem.release();
                });
                command_waiters.push_back(completion);
                ++num_commands;
                submit_cv.notify_all();
            }
            completion->sem.acquire();
            {
                std::scoped_lock lk{submit_mutex};
                std::erase(command_waiters, completion);
            }
            ASSERT_MSG(completion->state.load(std::memory_order_acquire) == 2u,
                       "CP command cancelled or failed during shutdown");
        } else {
            std::scoped_lock lk{submit_mutex};
            ASSERT_MSG(!cp_stopping, "CP command submitted during shutdown");
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_all();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    /// The graphics queue's compute registers (ISS_NATIVE_PASSES submit mode seed).
    const ComputeProgram& GfxCsState() const noexcept {
        return mapped_queues[GfxQueueId].cs_state;
    }

    inline ComputeProgram& GetCsRegs() {
        const s32 qid = CurrentQueue();
        ASSERT_MSG(qid >= 0 && qid < NumTotalQueues, "No CP queue selected for CS registers");
        return mapped_queues[qid].cs_state;
    }

    /// The queue whose commands the command processor is processing: 0 graphics, 1+ compute.
    [[nodiscard]] s32 CurrentQueue() const noexcept {
        return thread_context.owner == this ? thread_context.qid : curr_qid;
    }
    /// CP thread only: generation of the coroutine currently being resumed.
    /// Remains valid while its slot is closing; zero in callbacks/idle.
    [[nodiscard]] u64 CurrentSubmissionGeneration() const noexcept {
        return thread_context.owner == this ? thread_context.token.generation
                                            : current_submission_token.generation;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

    struct AscQueueSnapshot {
        VAddr map_addr{};
        u32 ring_size_dw{};
        u64 generation{};
    };
    // The gnmdriver holds m_submit_lock while mapping, describing, or closing a
    // slot. Close rejects new admission; Finish waits for coroutine lifetime.
    std::optional<u32> MapAsc(VAddr map_addr, u32* read_addr, u32 ring_size_dw, u32 pipe_id);
    std::optional<AscQueueSnapshot> DescribeAsc(u32 gnm_vqid);
    std::optional<u64> BeginUnmapAsc(u32 gnm_vqid);
    bool WaitAscDrained(u32 gnm_vqid, u64 generation);
    bool FinishUnmapAsc(u32 gnm_vqid, u64 generation);

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid.load(std::memory_order_acquire);
    }
    bool IsCpThreadId(u32 tid) const noexcept {
        return tid != 0 &&
               (tid == gpu_tid.load(std::memory_order_acquire) ||
                (CpSplitActive() && tid == asc_gpu_tid.load(std::memory_order_acquire)));
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb, u64 scout_seq = 0);
    Task ProcessGraphicsGpuCall(std::function<void()> action);
    Task ProcessGraphicsHostWrite(VAddr address, u64 size, std::function<void()> action,
                                  std::function<bool()> ready = {});
    Task ProcessAscHostWrite(u32 vqid, VAddr address, u64 size,
                             std::function<void()> action,
                             std::function<bool()> ready = {});
    // Called only after a complete owning fallback packet exists, before Vulkan emission.
    Task ProcessPreparedAscFallback(iss::g4::FallbackPacket packet, u32 vqid);
    Task ProcessSplitAscDispatch(u32 vqid, ComputeProgram cs,
                                 std::optional<std::pair<VAddr, u32>> indirect);
    Task ProcessAscDmaFallback(u32 vqid, iss::g4::DmaFallbackPacket packet);
    void BeginScoutDraw(const void* packet);
    void EndScoutDraw();
    Task ProcessCeUpdate(std::span<const u32> ccb);
    /// submit_ns: when the guest submitted the work (top level submissions only, for statistics).
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid, u64 submit_ns = 0);

    void ProcessCommands();
    void Process(std::stop_token stoken);
    bool OnlyGraphicsSubmitsPending();

    /// Remembers values the command processor wrote at small addresses (labels, fences), so a
    /// wait satisfied by one of them is not mistaken for a guest CPU signal.
    void RecordCpLabel(VAddr address, const void* data, u32 num_bytes);
    /// Same, for a command processor write of guest memory: copies taken before may be stale.
    void NoteCpWrite(VAddr address, const void* data, u32 num_bytes);
    /// Writes emulated depth block sample counters for an occlusion counter dump: they grow at
    /// each dump, so every query sees samples visible.
    void WriteEmulatedOcclusionCounters(VAddr address);
    /// Whether the value at the address is the last one the command processor wrote there.
    bool IsWrittenByCp(VAddr address) const;
    /// Starts a new upload epoch when the satisfied wait may have been released by the guest CPU
    /// (which writes the data the following commands read before the value they wait for).
    void OnWaitRegMemSatisfied(const PM4CmdWaitRegMem& wait);
    /// A wait for the results of an occlusion query counted by the GPU is satisfied at once when
    /// the value waited for does not depend on the samples counted (valid bits): the commands that
    /// follow run after the GPU wrote the results, reading them back would wait for the GPU.
    bool SkipOcclusionWait(const PM4CmdWaitRegMem& wait);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        struct Submission {
            Task::Handle handle{};
            G4Stage1Ledger<NumTotalQueues>::Token token{};
        };
        std::queue<Submission> submits{};
        ComputeProgram cs_state{};
        u64 packet_sequence{}; // C worker only; reset after drained remap
        // ASC register writes and register waits belong to the queue being translated.
        // Graphics Regs is never written by the ASC worker.
        Regs regs{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    std::atomic<u32> num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::jthread asc_process_thread{};
    const bool cp_split_requested = [] {
        const char* flag = std::getenv("ISS_CP_SPLIT");
        return flag && flag[0] == '1';
    }();
    std::atomic<bool> cp_split_decided{};
    std::atomic<bool> cp_split_active{};
    std::atomic<bool> cp_stopping{};
    std::atomic<u32> num_submits{};
    /// Asynchronous compute submissions not fully processed yet. The PS4 runs them alongside the
    /// graphics queue; here the graphics queue can give way to them (Tuning::asc_yield_packets).
    std::atomic<u32> asc_submits{};
    /// Submissions queued by the guest and fully processed by the command processor (monotonic),
    /// and the number queued at the end of each recent guest frame.
    std::atomic<u64> submits_issued{};
    std::atomic<u64> submits_completed{};
    // Canonical per-qid/generation progress for split frame waits. The global completed
    // counter remains diagnostic because C may complete a newer submission before G.
    G4Stage1Ledger<NumTotalQueues> g4_ledger{};
    CpCoordination cp_coordination{};
    iss::g4::SchedulerTimelineBridge<CpCoordination> cp_timeline{cp_coordination};
    iss::g4::FallbackMailbox fallback_mailbox{};
    GraphicsFallbackRecorder graphics_fallback_recorder{}; // set before split startup
    std::atomic<u32> fallback_pending{};
    iss::g4::GraphicsControlMailbox graphics_controls{};
    iss::g4::AscOwnerPause asc_owner_pause{}; // submit_mutex
    GraphicsControlHandle guest_pause_control{}; // submit_mutex, until terminal
    SealRequest guest_pause_seal{}; // submit_mutex, cancellation proof
    // Retain and wake pending seal controls independently of PM4 submissions.
    // Protected by submit_mutex.
    std::array<std::vector<SealRequest>, 2> cp_seal_requests{};
    std::array<std::atomic<bool>, 2> cp_control_pending{};
    static constexpr size_t NumFrameMarks = 16;
    std::array<u64, NumFrameMarks> frame_marks{};
    std::array<G4Stage1Ledger<NumTotalQueues>::MarkSnapshot, NumFrameMarks> g4_frame_marks{};
    u64 g4_shadow_global_ready_gap{};
    u64 frame_index{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    // Serializes scout sequence and coroutine preparation without holding
    // submit_mutex or queue.m_access during Scout::Submit allocation.
    std::mutex gfx_admission_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::vector<std::shared_ptr<CommandCompletion>> command_waiters{};
    std::thread::id gpu_id;
#ifdef __linux__
    std::atomic<u32> gpu_tid{};
    std::atomic<u32> asc_gpu_tid{};
#endif
    s32 curr_qid{-1};
    G4Stage1Ledger<NumTotalQueues>::Token current_submission_token{};
    // Set while a coroutine is resumed. Downstream shader/cache lookups use this queue's
    // registers even when a future graphics and ASC worker run at the same time.
    struct CpThreadContext {
        const Liverpool* owner{};
        s32 qid{-1};
        G4Stage1Ledger<NumTotalQueues>::Token token{};
    };
    static thread_local CpThreadContext thread_context;
    /// Packets and commands processed by the command processor thread, used to tell work from
    /// spinning on unsatisfied waits. Only accessed by that thread, counted at every packet: on
    /// a cache line of its own, away from the fields guest threads write when they submit
    /// (counting there waited for that line to come back from their cores).
    alignas(64) u64 processed_packets{};
    /// Graphics packets processed since the graphics queue last gave way to compute work. Only
    /// accessed by the command processor thread.
    u32 gfx_packets_since_yield{};
    /// Context, config and uconfig register writes that changed a value (and state clears), by
    /// the command processor: draws with the count of the draw before have the same render
    /// state. Only accessed by that thread.
    u64 context_reg_changes{};
    /// Time the command processor thread slept inside packets (flip label waits).
    u64 blocked_ns{};
    struct CpLabel {
        VAddr address{};
        u32 value{};
    };
    static constexpr size_t NumCpLabels = 1024;
    /// Last values written by either command processor owner, indexed by dword address.
    /// The small table is protected independently from guest memory and cache locks.
    std::array<CpLabel, NumCpLabels> cp_labels{};
    mutable std::mutex cp_labels_mutex{};
    // The ASC worker has its own accounting; graphics busy/spin sampling must not observe
    // ASC packet increments as graphics work.
    u64 asc_processed_packets{};

    void ProcessAsc(std::stop_token stoken);
    void DecideCpSplit();
    bool GraphicsSubmitsPending();
    void RefreshCpControlPendingLocked(CpLane lane);
    void ServiceCpSeals(CpLane lane);
    void ServiceGraphicsFallbacks();
    void ServiceGraphicsControls();
    void AdvanceAscReadPointer(AscQueueInfo& queue, u32 dwords);
};

} // namespace AmdGpu
