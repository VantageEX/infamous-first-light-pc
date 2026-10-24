// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <coroutine>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>
#include <queue>
#include <array>
#include <deque>

#include "common/assert.h"
#include "common/fault_monitor.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

union PM4Header;
struct PM4CmdReleaseMem;
enum class PM4ItOpcode : u32;

struct Liverpool {
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
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};
    // Bumped on every write into regs.reg_array (all GFX SET_*_REG handlers + ClearState). Tried as
    // a cheap "did any register change since the last draw" signal for memoizing GetGraphicsPipeline
    // (RefreshGraphicsKey is 0.955 us/draw = 96% of its cost). MEASURED 2026-08-10: gen_unchanged is
    // 0.0% - the guest re-emits a register before every single draw (per-draw SH user_data), so this
    // coarse signal never fires even though 85.7% of draws rebuild an identical key. A working memo
    // needs key-relevant-range dirty tracking; see notes/perf-forest.md. Behaviorally inert - only
    // the pipeline probe reads it. Kept as a documented negative result.
    u64 regs_generation{0};
    // Same idea but bumped ONLY on context/uconfig/config regs + ClearState, i.e. excluding the SH
    // per-draw user_data churn that made regs_generation useless. If THIS one is often unchanged
    // across draws, a pipeline-key dirty flag on the context regs is viable; the probe's mismatch
    // count then says whether excluding SH program-address writes is safe. Measurement only.
    u64 regs_generation_ctx{0};
    // SHADPS4_VIEWPORT_MEMO: bumped when a SET_CONTEXT_REG changes any byte, and on CLEAR_STATE.
    // Only maintained while the flag is on (the value compare costs a memcmp per packet).
    u64 context_change_epoch{1};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        // POST_TASK_DONE: a submission the front finished is idle only once the back retired it.
        submit_cv.wait(lk, [this] { return num_submits == 0 && pending_retire == 0; });
    }

    bool IsGpuIdle() const {
        return num_submits == 0 && pending_retire == 0;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// False until Process() has recorded its thread id (diagnostic checks skip startup).
    bool GpuThreadStarted() const {
        return gpu_id != std::thread::id{};
    }

    /// True on the command processor thread, where SendCommand runs its callback inline.
    bool IsGpuThread() const {
        // SHADPS4_MTB_P3: the parked-mode translation worker acts for the back during a run.
        return std::this_thread::get_id() == gpu_id || Common::FaultMonitor::t_acting_cp;
    }

    /// Runs `func` on the CP thread. With SHADPS4_CP_SPLIT that is the back thread, which owns
    /// the caches and the scheduler; it runs queued commands between records and while idle.
    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (IsGpuThread()) { // incl. the SHADPS4_MTB_P3 worker while the back is parked on it
            return func();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            // q13: queue vs run split of Game:Main's waits (census only).
            const bool sc_on =
                Common::FaultMonitor::t_is_game_main && Common::FaultMonitor::GameMainCensus();
            const u64 sc_enq = sc_on ? Common::FaultMonitor::CensusTicks() : 0;
            u64 sc_t1 = 0, sc_t2 = 0;
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func, &sc_t1, &sc_t2, sc_on] {
                    if (sc_on) {
                        sc_t1 = Common::FaultMonitor::CensusTicks();
                    }
                    func();
                    if (sc_on) {
                        sc_t2 = Common::FaultMonitor::CensusTicks();
                    }
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            if (split_enabled) [[unlikely]] {
                split->WakeBackIfParked();
            }
            {
                Common::FaultMonitor::GameMainWaitScope gm_wait{10 + Common::FaultMonitor::t_sendcmd_kind}; // f2
                sem.acquire();
            }
            if (sc_on) {
                const u32 k = Common::FaultMonitor::t_sendcmd_kind & 3;
                Common::FaultMonitor::g_sc_queue_ticks[k].fetch_add(sc_t1 - sc_enq, std::memory_order_relaxed);
                Common::FaultMonitor::g_sc_run_ticks[k].fetch_add(sc_t2 - sc_t1, std::memory_order_relaxed);
            }
        } else {
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace(std::move(func));
                ++num_commands;
                submit_cv.notify_one();
            }
            if (split_enabled) [[unlikely]] {
                split->WakeBackIfParked();
            }
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    // CP front/back split (notes/cp-split-inventory-2026-09-25.md): every register, render-target
    // hint and compute-register read made by the back path (the rasterizer after the pipeline
    // lookup, and the buffer cache) goes through these, so the back can later read its own mirror.
    // Without SHADPS4_CP_SPLIT they return the front's state. With it, the back's mirror
    // (cp_split.h), and while a gfx dispatch record executes, its ComputeProgram snapshot.
    // The pipeline cache and FilterDraw's classifier keep reading regs (front).
    const Regs& BackRegs() const {
        return *back_regs;
    }
    const CbDbExtent& BackCbExtent(u32 cb) const {
        return back_cb_extent[cb];
    }
    const CbDbExtent& BackDbExtent() const {
        return *back_db_extent;
    }
    /// SHADPS4_NATIVE_COLOUR open gate (back thread, inside the draw record being executed):
    /// 1 = the next published draw record (past register-only records) uses `pipeline`, 0 = it
    /// does not or something heavier comes first, -1 = not published yet / unknown.
    int NativePeekNextDraw(const void* pipeline);

    u64 BackContextEpoch() const {
        return *back_ctx_epoch;
    }
    const ComputeProgram& BackCsRegs() const {
        if (back_cs) {
            return *back_cs;
        }
        // Without POST_ASC, ASC dispatches run through Call records, which are synchronous at
        // every depth, and read the front's mapped_queues[curr_qid] here while the front is
        // parked. With POST_ASC they are Dispatch records and never get here (back_cs is set).
        // With run-ahead any other caller (a posted draw reaching the SharedMemory binding)
        // would race the front, so it asserts instead.
        ASSERT_MSG(!split_run_ahead || back_in_call,
                   "CpSplit: BackCsRegs outside a dispatch record or Call record");
        return mapped_queues[curr_qid].cs_state;
    }

    /// CP split run-ahead: a fault taken on the front thread (a pipeline lookup or the PM4 parser
    /// reading a GPU-modified page) must see what the draws it already posted write, so it drains
    /// the back before the download (inventory 4.6). Called by the rasterizer's fault entry
    /// points before any lock is taken; a no-op on every other thread and without run-ahead.
    void DrainForFrontFault(VAddr addr = 0, u64 size = 0, bool read = true) {
        if (split_run_ahead && CpSplit::t_is_front) [[unlikely]] {
            NoteFrontFault(addr, size, read);
            split->Sync(CpSplit::DrainReason::FrontFault);
        }
    }
    /// SHADPS4_CP_SPLIT_TASK_GUARD census (l3n): what the front's faults read.
    void NoteFrontFault(VAddr addr, u64 size, bool read);
    /// SHADPS4_CP_SPLIT_WAIT_BARRIER (l3n2): posts a Call that records Rasterizer::WaitBarrier at
    /// this stream position. `kind` 0 = POST_ORDER forward, 1 = FENCE_FORWARD skip (counters).
    void PostWaitBarrier(u32 kind);
    /// SHADPS4_CP_SPLIT_RPTR_BACKING (l3o): advances an ASC queue's read pointer by `dwords`.
    struct AscQueueInfo;
    void AdvanceAscReadPtr(AscQueueInfo& queue, u32 dwords);

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
        u32 rptr_last{~0u}; ///< SHADPS4_CP_SPLIT_RPTR_BACKING: the value the front wrote last
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

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
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb);
    /// Set by the IndirectBuffer packet just before it first resumes the nested task.
    bool starting_indirect_gfx{false};
    Task ProcessCeUpdate(std::span<const u32> ccb, bool is_indirect = false);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void Process(std::stop_token stoken);
    /// The submission loop: today's Process body. Runs on the CP thread, or with
    /// SHADPS4_CP_SPLIT on the front thread.
    void ProcessLoop(std::stop_token stoken);

    // ---- CP front/back split (cp_split.h). All no-ops or unused when split_enabled is false.
    /// TASK_GUARD: registers a Process* coroutine's own command-buffer span (which it shrinks as
    /// it parses, so the registry always sees the unparsed rest) for the life of the coroutine
    /// body, and checks the whole span against outstanding posted writes at entry. Front only;
    /// does nothing unless split_task_guard.
    class SplitSpanScope {
    public:
        SplitSpanScope(Liverpool& liverpool, const std::span<const u32>* span,
                       CpSplit::DrainReason entry_reason);
        ~SplitSpanScope();
        SplitSpanScope(const SplitSpanScope&) = delete;
        SplitSpanScope& operator=(const SplitSpanScope&) = delete;

    private:
        Liverpool* lp{};
        s32 slot{-1};
    };
    /// TASK_GUARD: drains if an outstanding posted write overlaps `span`.
    void SplitGuardSpan(std::span<const u32> span, CpSplit::DrainReason reason);
    /// TASK_GUARD: drains if an outstanding posted write overlaps the unparsed rest of any active
    /// command buffer (every registered span), before ProcessLoop resumes a task.
    void SplitGuardActive();
    /// True if [lo, hi) overlaps the unparsed rest of any active command buffer (every registered
    /// span), or if some level is untracked (conservative).
    bool SplitSpansOverlap(VAddr lo, VAddr hi) const;
    void FrontProcess(std::stop_token stoken);
    void BackLoop(std::stop_token stoken);
    void ExecuteRecord(const CpSplit::RecordHeader& record);
    void ExecuteRecordImpl(const CpSplit::RecordHeader& record);
    /// SHADPS4_A1 (architecture A step A1): ExecuteRecordImpl plus the per-DCB census.
    void A1Execute(const CpSplit::RecordHeader& record);
    /// SHADPS4_A2 (architecture A step A2): the back's record execution with worker stretches.
    void A2Execute(const CpSplit::RecordHeader& record);
    void A2WorkerLoop(std::stop_token stoken, u32 index);
    void A2Shutdown();
    /// `draw_payload`: the SplitDrawRecord to run (default: the record's own payload; a
    /// Segment record passes its embedded member-0 record).
    void ExecuteDrawRecord(const CpSplit::RecordHeader& record, const void* draw_payload = nullptr);
    /// SHADPS4_NATIVE_SEGX: a Segment record (evaluate, or execute).
    void ExecuteSegmentRecord(const CpSplit::RecordHeader& record);
    const CpSplit::RecordHeader* back_draw_record{}; ///< SHADPS4_NATIVE_COLOUR peek
    void ExecuteDispatchRecord(const CpSplit::RecordHeader& record);
    /// SHADPS4_MTB_P1_BREAK=1 (P1d pricing): ends an open render pass before a packet record.
    void MtbBreakAtPacket(const PM4Header* header);
    /// `cs` (dispatches): compared with mapped_queues[cs_queue].cs_state.
    void VerifyMirror(const char* what, u64 seq, const ComputeProgram* cs,
                      u32 cs_queue = 0) const;
    /// Run-ahead VERIFY: compares the mirror with the register windows a posted draw or
    /// dispatch carries.
    void VerifySnapshot(const u8* snapshot, u64 seq) const;
    /// The back-side part of the gfx packets that have one (fences, DMA, WRITE_DATA, markers,
    /// flip IRQ, ZPASS_DONE, PFP_SYNC_ME, MEM_SEMAPHORE signal/decrement). The inline path calls
    /// it directly; the split path copies the packet into a Packet record.
    void ExecuteGfxPacket(const PM4Header* header, bool guest_markers_enabled);
    /// Inline: ExecuteGfxPacket now. Split: a synchronous Packet record.
    void RunGfxPacket(const PM4Header* header, bool guest_markers_enabled,
                      CpSplit::DrainReason reason);
    /// Gfx DMA_DATA: posted without waiting under run-ahead + POST_DMA unless it writes into
    /// `rest` (the unparsed part of the current command buffer); otherwise RunGfxPacket.
    void RunGfxDma(const PM4Header* header, bool guest_markers_enabled,
                   std::span<const u32> rest);
    /// POST_ASC: an ASC DMA_DATA / WRITE_DATA as a Packet record (ExecuteGfxPacket has the
    /// same body). Posted, unless [dst, dst + bytes) overlaps `rest` (the unparsed part of the
    /// current ASC command buffer): then synchronous with `guard_reason`.
    void RunAscPacket(const PM4Header* header, std::span<const u32> rest, VAddr dst, u64 bytes,
                      CpSplit::DrainReason guard_reason);
    /// POST_FENCE: a fence packet (gfx EOP / EOS, or ASC RELEASE_MEM with `flags` =
    /// AscReleaseMem | pipe << 8) posted as a Packet record, its label noted in the write log.
    /// Synchronous with `sync_reason` if the label lies in `rest`.
    void RunFence(const PM4Header* header, std::span<const u32> rest, VAddr label, u16 flags,
                  CpSplit::DrainReason sync_reason);
    /// ASC RELEASE_MEM body (back): download images, fence value, interrupt on `pipe_id`.
    void ExecuteAscReleaseMem(const PM4CmdReleaseMem* release_mem, u32 pipe_id);
    /// Inline: fn() now. Split: fn() on the back while the front waits (a drain at any depth).
    template <typename F>
    void OnBack(F&& fn, CpSplit::DrainReason reason = CpSplit::DrainReason::Call);
    /// Journals regs.reg_array[word_offset, +count) to the back's mirror (split only).
    void JournalRegs(u32 word_offset, u32 count);
    template <typename T>
    void JournalField(const T& field) {
        static_assert(sizeof(T) % sizeof(u32) == 0);
        if (split_enabled) [[unlikely]] {
            const auto byte_offset =
                reinterpret_cast<const u8*>(&field) - reinterpret_cast<const u8*>(&regs);
            JournalRegs(u32(byte_offset / sizeof(u32)), u32(sizeof(T) / sizeof(u32)));
        }
    }
    /// Journals the CB/DB extent a SET_CONTEXT_REG at reg_addr may have changed (split only).
    void JournalExtent(u32 reg_addr);
    /// VideoCore::BumpSubmitEpoch, or in split mode an EpochBump record (runs in stream order).
    void BumpEpoch(PM4ItOpcode opcode);
    bool NativeRecMember(const PM4Header* header, const char* name, bool host_marker,
                         const CpSplit::DrawArgs& args);
    void NativeRecHead(const PM4Header* header, const void* pipeline, bool plain_draw,
                       u32 vi_bytes, const u8* record_body, u32 stage_count);
    /// SHADPS4_LEAN (L1 shadow): chain verify + scan + worker hand-off at a top-level DCB start.
    void LeanFrontBegin(std::span<const u32> dcb, std::span<const u32> ccb, bool guest_markers);
    /// SHADPS4_LEAN=2 (front): queue the head span, apply it to the live state, emit DcbRef.
    struct LeanL2Begin {
        u32 cut{};       ///< 0: not taken (the front parses the whole DCB)
        u32 run_bumps{}; ///< run-ending packets in the head span
    };
    LeanL2Begin LeanFrontBeginL2(std::span<const u32> dcb, std::span<const u32> ccb,
                                 bool guest_markers);
    /// SHADPS4_LEAN=2 (back): run a DcbRef's job channel to its LeanEnd.
    void LeanBackFollow(const CpSplit::DcbRefPayload& ref);
    bool lean_following{}; ///< back: inside LeanBackFollow (NativePeekNextDraw: unknown)
    /// m2 (SHADPS4_ASC_SERVICE & 2): the BackLoop's early ASC service, callable inside follows.
    void (*early_service_fn)(void*) = nullptr;
    void* early_service_ctx = nullptr;
    /// Starts the lean worker threads (part b).
    void LeanStart();
    void LeanWorkerLoop(std::stop_token stoken, u32 index);
    void EmitDraw(const PM4Header* header, const char* name, bool host_marker,
                  const CpSplit::DrawArgs& args);
    /// `queue`: 0 = gfx, vqid + 1 = ASC (the mapped_queues index whose cs_state is copied).
    void EmitDispatch(const PM4Header* header, const char* name, bool host_marker,
                      bool is_indirect, VAddr address, u32 offset, u32 size, u32 queue);
    /// Publishes and waits for everything emitted so far. Called where the front reads what the
    /// back produces (inventory section 4.6). At depth 0 it only flushes register records.
    void FrontDrain(CpSplit::DrainReason reason) {
        if (split_enabled) [[unlikely]] {
            split->Sync(reason);
        }
    }

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available

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
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    /// POST_TASK_DONE: submissions the front finished whose Retire record the back has not run.
    std::atomic<u32> pending_retire{};
    std::atomic<u32> num_commands{};
    /// SHADPS4_ASC_YIELD: bumped by SubmitAsc; the graphics task yields when it changed.
    std::atomic<u64> asc_submit_seq{};
    u64 gfx_seen_asc_seq{};
    /// ASC submissions not yet finished (SubmitAsc ++, the round-robin's pop --), and the number
    /// the front finished (front only): after one finishes with more queued, graphics yields again.
    std::atomic<u32> asc_outstanding{};
    u64 asc_done_seq{};
    u64 gfx_seen_asc_done{};
    /// SHADPS4_ASC_START_STATS: submit times of queued top-level ASC buffers per mapped queue
    /// (under that queue's m_access), consumed when the buffer's task first runs.
    /// Low bit set: the queue already held an unfinished submission when this one arrived.
    std::array<std::deque<u64>, NumTotalQueues> asc_submit_ns{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    s32 curr_qid{-1};

    // CP front/back split. Set in the constructor before any thread starts, never changed.
    bool split_enabled{false};
    bool split_verify{false};
    bool split_run_ahead{false}; ///< SHADPS4_CP_SPLIT=2: draws (and POST_DMA DMAs) are posted.
    bool split_post_dma{false};  ///< Run-ahead + SHADPS4_CP_SPLIT_POST_DMA=1.
    u32 split_post_asc{0};       ///< Run-ahead + POST_ASC: CpSplit::PostAscBits, else 0.
    bool split_post_task_done{false}; ///< Run-ahead + SHADPS4_CP_SPLIT_POST_TASK_DONE=1.
    u32 split_task_guard{0}; ///< POST_TASK_DONE + SHADPS4_CP_SPLIT_TASK_GUARD (1-3), else 0.
    bool split_post_fence{false}; ///< TASK_GUARD + SHADPS4_CP_SPLIT_POST_FENCE=1.
    bool split_disp_guard{true};  ///< SHADPS4_CP_SPLIT_DISP_GUARD (default 1).
    /// TASK_GUARD (front): spans of the active Process* levels of every queue (SplitSpanScope);
    /// null = free. A level that finds no free slot counts in split_span_untracked, and resume
    /// checks drain conservatively while it is non-zero.
    static constexpr u32 SplitSpanSlots = 64;
    std::array<const std::span<const u32>*, SplitSpanSlots> split_spans{};
    u32 split_span_hwm{0}; ///< Slots [split_span_hwm, SplitSpanSlots) are free.
    u32 split_span_untracked{0};
    u32 split_verify_every{1};   ///< Run-ahead VERIFY: one synchronous draw in this many.
    u64 split_verify_draws{0};   ///< Front: draws emitted with VERIFY on.
    u64 split_verify_dispatches{0}; ///< Front: posting-eligible dispatches with VERIFY on.
    u64 split_relookup_checks{0};   ///< Back, VERIFY: synced dispatches re-looked-up.
    u64 split_relookup_diffs{0};    ///< Back, VERIFY: ... whose re-lookup disagreed.
    u64 split_draw_relookup_checks{0}; ///< Back, run-ahead VERIFY: synced draws re-looked-up.
    u64 split_draw_relookup_diffs{0};  ///< Back, run-ahead VERIFY: ... whose re-lookup disagreed.
    /// SHADPS4_CP_SPLIT_FRONT_DERIVE (0 / 1 / 2). Front: the derivation's output and the previous
    /// draw's description part, which numbers descriptions (VertexInputsPayload::desc_id).
    int split_front_derive{0};
    std::unique_ptr<CpSplit::VertexInputsDerived> front_vi;
    std::array<CpSplit::VertexInputDesc, CpSplit::MaxVertexInputs> front_vi_prev{};
    u32 front_vi_prev_count{~0u}; ///< No previous description yet: the first draw bumps the id.
    u64 front_vi_desc_id{0};
    bool back_in_call{false};    ///< Back: inside a Call record (the front is parked).
    std::unique_ptr<CpSplit::Channel> split;
    std::unique_ptr<CpSplit::BackMirror> mirror;
    std::jthread front_thread{}; ///< Started and joined by the back (process_thread).
    /// SHADPS4_MTB_P3=1 (multi-threaded back P3.0, notes/parrec-build-2026-10-08.md sec. 14d):
    /// a translation thread runs runs of Draw / Dispatch / RegWrite / Extent / ClearState records
    /// while the back is parked on it (one hand-off per run).
    struct MtbHandoff {
        std::atomic<u32> state{0}; ///< 0 idle, 1 posted, 2 done, 3 stop
        std::vector<const CpSplit::RecordHeader*> run;
        size_t executed{0}; ///< Records of `run` the worker ran (it stops early for commands).
        u64 runs{0}, records{0}, draws{0}, log_runs{0}, spin_yields{0};
    };
    MtbHandoff mtb_handoff;
    std::jthread mtb_thread{};
    void MtbWorkerLoop(std::stop_token stoken);
    /// Back: collects the run starting at `record` (read-ahead), runs it on the worker, releases
    /// its records. Returns false when `record` does not start a run.
    bool MtbRunAhead(const CpSplit::RecordHeader* record);
    // What the Back* accessors read: the front's own state when the split is off.
    const Regs* back_regs{&regs};
    const CbDbExtent* back_cb_extent{last_cb_extent.data()};
    const CbDbExtent* back_db_extent{&last_db_extent};
    const u64* back_ctx_epoch{&context_change_epoch};
    const ComputeProgram* back_cs{};
};

} // namespace AmdGpu
