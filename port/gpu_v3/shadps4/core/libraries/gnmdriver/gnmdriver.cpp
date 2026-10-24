// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <immintrin.h>
#include <atomic>
#include <array>
#include <cstdlib>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include "gnm_error.h"
#include "gnmdriver.h"

#include "common/fence_watch.h"
#include "common/scope_exit.h"
#include "common/fault_monitor.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/slot_vector.h"
#include "core/address_space.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/gnmdriver/gnm_error.h"
#include "core/libraries/gnmdriver/gnmdriver_init.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/libs.h"
#include "core/libraries/videoout/video_out.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/native_rec.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern Frontend::WindowSDL* g_window;
std::unique_ptr<Vulkan::Presenter> presenter;
std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace VideoCore {
void ArenaReadPrefetchFrameEnd(); // buffer_cache.cpp (q48)
}

namespace Libraries::GnmDriver {

using namespace AmdGpu;

enum GnmEventType : u64 {
    Compute0RelMem = 0x00,
    Compute1RelMem = 0x01,
    Compute2RelMem = 0x02,
    Compute3RelMem = 0x03,
    Compute4RelMem = 0x04,
    Compute5RelMem = 0x05,
    Compute6RelMem = 0x06,
    GfxEop = 0x40
};

enum ShaderStages : u32 {
    Cs,
    Ps,
    Vs,
    Gs,
    Es,
    Hs,
    Ls,

    Max
};

static constexpr std::array indirect_sgpr_offsets{0u, 0u, 0x4cu, 0u, 0xccu, 0u, 0x14cu};

// Gates use of what appear to be the neo-mode init sequences but with the older
// IA_MULTI_VGT_PARAM register address. No idea what this is for as the ioctl
// that controls it is still a mystery, but leaving the sequences in gated behind
// this flag in case we need it in the future.
static constexpr bool UseNeoCompatSequences = false;

// In case if `submitDone` is issued we need to block submissions until GPU idle
static u32 submission_lock{};
std::condition_variable cv_lock{};
std::mutex m_submission{};
static u64 frames_submitted{};      // frame counter
static bool send_init_packet{true}; // initialize HW state before first game's submit in a frame
static s32 sdk_version{0};

static u32 asc_next_offs_dw[Liverpool::NumComputeRings];

// This address is initialized in sceGnmGetTheTessellationFactorRingBufferBaseAddress
static VAddr tessellation_factors_ring_addr = -1;
static constexpr u32 tessellation_offchip_buffer_size = 0x800000u;

/// SHADPS4_SUBMIT_LOCK_FIX=1: sceGnmSubmitDone checks the GPU and sets submission_lock under m_submission. Without
/// it the GpuIdle reset can land between the check and the set; the lock then stays set with the GPU idle, no later
/// reset comes, and every WaitGpuIdle (DingDong, submits) blocks forever (notes/freeze-2026-09-27.md).
static bool SubmitLockFix() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_SUBMIT_LOCK_FIX");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}
static std::atomic<u64> submission_resets{0};

static void ResetSubmissionLock(Platform::InterruptId irq) {
    std::unique_lock lock{m_submission};
    submission_lock = 0;
    submission_resets.fetch_add(1, std::memory_order_relaxed);
    if (Common::FenceWatch::Enabled()) {
        Common::FenceWatch::Get().gnm_lock.store(0, std::memory_order_relaxed);
        Common::FenceWatch::Get().gnm_resets.fetch_add(1, std::memory_order_relaxed);
    }
    cv_lock.notify_all();
}

static void WaitGpuIdle() {
    HLE_TRACE;
    std::unique_lock lock{m_submission};
    if (Common::FenceWatch::Enabled() && submission_lock != 0) [[unlikely]] {
        auto& fw = Common::FenceWatch::Get();
        fw.gnm_waiters.fetch_add(1, std::memory_order_relaxed);
        cv_lock.wait(lock, [] { return submission_lock == 0; });
        fw.gnm_waiters.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    cv_lock.wait(lock, [] { return submission_lock == 0; });
}

// Write a special ending NOP packet with N DWs data block
static inline u32* WriteTrailingNop(u32* cmdbuf, u32 data_block_size) {
    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
    nop->header = PM4Type3Header{PM4ItOpcode::Nop, data_block_size - 1};
    nop->data_block[0] = 0u; // only one out of `data_block_size` is initialized
    return cmdbuf + data_block_size + 1 /* header */;
}

// Write a special ending NOP packet with N DWs data block
template <u32 data_block_size>
static inline u32* WriteTrailingNop(u32* cmdbuf) {
    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
    nop->header = PM4Type3Header{PM4ItOpcode::Nop, data_block_size - 1};
    nop->data_block[0] = 0u; // only one out of `data_block_size` is initialized
    return cmdbuf + data_block_size + 1 /* header */;
}

static inline u32* ClearContextState(u32* cmdbuf) {
    static constexpr std::array ClearStateSequence{
        0xc0012800u, 0x80000000u, 0x80000000u, 0xc0001200u, 0u, 0xc0055800u,
        0x2ec47fc0u, 0xffffffffu, 0u,          0u,          0u, 10u,
    };
    static_assert(ClearStateSequence.size() == 0xc);

    std::memcpy(cmdbuf, ClearStateSequence.data(), ClearStateSequence.size() * 4);
    return cmdbuf + ClearStateSequence.size();
}

static inline bool IsValidEventType(Platform::InterruptId id) {
    return (static_cast<u32>(id) >= static_cast<u32>(Platform::InterruptId::Compute0RelMem) &&
            static_cast<u32>(id) <= static_cast<u32>(Platform::InterruptId::Compute6RelMem)) ||
           static_cast<u32>(id) == static_cast<u32>(Platform::InterruptId::GfxEop);
}

s32 PS4_SYSV_ABI sceGnmAddEqEvent(OrbisKernelEqueue eq, u64 id, void* udata) {
    LOG_TRACE(Lib_GnmDriver, "called");

    auto equeue = GetEqueue(eq);
    if (!equeue) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    EqueueEvent kernel_event{};
    kernel_event.event.ident = id;
    kernel_event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    kernel_event.event.flags = OrbisKernelEvent::Flags::Add;
    kernel_event.event.fflags = 0;
    kernel_event.event.data = id;
    kernel_event.event.udata = udata;

    equeue->AddEvent(kernel_event);

    Platform::IrqC::Instance()->Register(
        static_cast<Platform::InterruptId>(id),
        [=](Platform::InterruptId irq) {
            ASSERT_MSG(irq == static_cast<Platform::InterruptId>(id), "An unexpected IRQ occured");

            // We need to convert IRQ# to event id
            if (!IsValidEventType(irq))
                return;

            // Event data is expected to be an event type as per sceGnmGetEqEventType.
            equeue->TriggerEvent(static_cast<GnmEventType>(id),
                                 OrbisKernelEvent::Filter::GraphicsCore,
                                 reinterpret_cast<void*>(id));
        },
        equeue);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmAreSubmitsAllowed() {
    LOG_TRACE(Lib_GnmDriver, "called");
    return submission_lock == 0;
}

int PS4_SYSV_ABI sceGnmBeginWorkload(u32 workload_stream, u64* workload) {
    if (workload) {
        *workload = (-(u32)(workload_stream < 0x10) & 1);
        return 0xf < workload_stream;
    }
    return 3;
}

s32 PS4_SYSV_ABI sceGnmComputeWaitOnAddress(u32* cmdbuf, u32 size, uintptr_t addr, u32 mask,
                                            u32 cmp_func, u32 ref) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 0xe)) {
        cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, 3);
        cmdbuf = WriteBody(cmdbuf, 0u);
        cmdbuf += 2;

        const u32 is_mem = addr > 0xffffu;
        const u32 addr_mask = is_mem ? 0xfffffffcu : 0xffffu;
        auto* wait_reg_mem = reinterpret_cast<PM4CmdWaitRegMem*>(cmdbuf);
        wait_reg_mem->header = PM4Type3Header{PM4ItOpcode::WaitRegMem, 5};
        wait_reg_mem->raw = (is_mem << 4u) | (cmp_func & 7u);
        wait_reg_mem->poll_addr_lo_raw = u32(addr & addr_mask);
        wait_reg_mem->poll_addr_hi = u32(addr >> 32u);
        wait_reg_mem->ref = ref;
        wait_reg_mem->mask = mask;
        wait_reg_mem->poll_interval = 10u;

        WriteTrailingNop<2>(cmdbuf + 7);
        return ORBIS_OK;
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmComputeWaitSemaphore() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmCreateWorkloadStream(u64 param1, u32* workload_stream) {
    if (param1 != 0 && workload_stream) {
        *workload_stream = 1;
        return 0;
    }
    return 3;
}

int PS4_SYSV_ABI sceGnmDebuggerGetAddressWatch() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerHaltWavefront() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerReadGds() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerReadSqIndirectRegister() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerResumeWavefront() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerResumeWavefrontCreation() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerSetAddressWatch() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerWriteGds() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerWriteSqIndirectRegister() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebugHardwareStatus() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDeleteEqEvent(OrbisKernelEqueue eq, u64 id) {
    LOG_TRACE(Lib_GnmDriver, "called");

    auto equeue = GetEqueue(eq);
    if (!equeue) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    equeue->RemoveEvent(id, OrbisKernelEvent::Filter::GraphicsCore);

    Platform::IrqC::Instance()->Unregister(static_cast<Platform::InterruptId>(id), equeue);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDestroyWorkloadStream() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

void PS4_SYSV_ABI sceGnmDingDong(u32 gnm_vqid, u32 next_offs_dw) {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "vqid {}, offset_dw {}", gnm_vqid, next_offs_dw);

    if (gnm_vqid == 0) {
        return;
    }

    {
        Common::FaultMonitor::GnmWaitScope gm_wait{9};
        WaitGpuIdle();
    }

    if (DebugState.ShouldPauseInSubmit()) {
        DebugState.PauseGuestThreads();
    }

    auto vqid = gnm_vqid - 1;
    auto& asc_queue = liverpool->asc_queues[{vqid}];

    auto& offs_dw = asc_next_offs_dw[vqid];

    if (next_offs_dw == offs_dw) {
        return;
    }

    if (next_offs_dw < offs_dw && next_offs_dw != 0) {
        // For cases if a submission is split at the end of the ring buffer, we need to submit it in
        // two parts to handle the wrap
        liverpool->SubmitAsc(gnm_vqid, {reinterpret_cast<const u32*>(asc_queue.map_addr) + offs_dw,
                                        asc_queue.ring_size_dw - offs_dw});
        offs_dw = 0;
    }

    const auto* acb_ptr = reinterpret_cast<const u32*>(asc_queue.map_addr) + offs_dw;
    const auto acb_size_dw = (next_offs_dw ? next_offs_dw : asc_queue.ring_size_dw) - offs_dw;
    const std::span acb_span{acb_ptr, acb_size_dw};

    asc_next_offs_dw[vqid] = next_offs_dw;

    if (DebugState.DumpingCurrentFrame()) {
        static auto last_frame_num = -1LL;
        static u32 seq_num{};
        if (last_frame_num == frames_submitted) {
            ++seq_num;
        } else {
            last_frame_num = frames_submitted;
            seq_num = 0u;
        }

        // Up to this point, all ACB submissions have been stored in a secondary command buffer.
        // Dumping them using the current ring pointer would result in files containing only the
        // `IndirectBuffer` command. To access the actual command stream, we need to unwrap the IB.
        auto acb = acb_span;
        auto base_addr = reinterpret_cast<uintptr_t>(acb_ptr);
        const auto* indirect_buffer =
            reinterpret_cast<const PM4CmdIndirectBuffer*>(acb_span.data());
        if (indirect_buffer->header.opcode == PM4ItOpcode::IndirectBuffer) {
            base_addr = reinterpret_cast<uintptr_t>(indirect_buffer->Address<const u32>());
            acb = {reinterpret_cast<const u32*>(base_addr), indirect_buffer->ib_size};
        }

        using namespace DebugStateType;

        DebugState.PushQueueDump({
            .type = QueueType::acb,
            .submit_num = seq_num,
            .num2 = gnm_vqid,
            .data = {acb.begin(), acb.end()},
            .base_addr = base_addr,
        });
    }
    liverpool->SubmitAsc(gnm_vqid, acb_span);
}

void PS4_SYSV_ABI sceGnmDingDongForWorkload(u32 gnm_vqid, u32 next_offs_dw, u64 workload_id) {
    LOG_DEBUG(Lib_GnmDriver, "called, redirecting to sceGnmDingDong");
    sceGnmDingDong(gnm_vqid, next_offs_dw);
}

int PS4_SYSV_ABI sceGnmDisableMipStatsReport() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDispatchDirect(u32* cmdbuf, u32 size, u32 threads_x, u32 threads_y,
                                      u32 threads_z, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && ((s32)(threads_x | threads_y | threads_z) > -1)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DispatchDirect>(cmdbuf, 4, PM4ShaderType::ShaderCompute,
                                                          predicate);
        cmdbuf = WriteBody(cmdbuf, threads_x, threads_y, threads_z);
        cmdbuf[0] = (flags & 0x18) + 1; // ordered append mode

        WriteTrailingNop<3>(cmdbuf + 1);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 7)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DispatchIndirect>(cmdbuf, 2, PM4ShaderType::ShaderCompute,
                                                            predicate);
        cmdbuf[0] = data_offset;
        cmdbuf[1] = (flags & 0x18) + 1; // ordered append mode

        WriteTrailingNop<3>(cmdbuf + 2);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirectOnMec(u32* cmdbuf, u32 size, VAddr args, u32 modifier) {
    if (cmdbuf != nullptr && size == 8 && args != 0 && ((args & 3u) == 0)) {
        cmdbuf[0] = 0xc0021602 | (modifier & 1u);
        *(VAddr*)(&cmdbuf[1]) = args;
        cmdbuf[3] = (modifier & 0x18) | 1u;
        cmdbuf[4] = 0xc0021000;
        cmdbuf[5] = 0;
        return ORBIS_OK;
    }
    return ORBIS_FAIL;
}

u32 PS4_SYSV_ABI sceGnmDispatchInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x216u,
        0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE0
    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x217u,
        0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE1

    if (sceKernelIsNeoMode()) {
        cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
            cmdbuf, 0x219u,
            0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE2
        cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
            cmdbuf, 0x21au,
            0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE3
    }

    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x215u, 0x170u); // COMPUTE_RESOURCE_LIMITS

    cmdbuf = WriteHeader<PM4ItOpcode::AcquireMem>(cmdbuf, 6);
    cmdbuf = WriteBody(cmdbuf, 0x28000000u, 0u, 0u, 0u, 0u, 0xau);

    cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, sceKernelIsNeoMode() ? 0xe9 : 0xef);
    cmdbuf = WriteBody(cmdbuf, 0u);
    return HwInitPacketSize;
}

namespace {
// SHADPS4_DRAW_THREAD_CENSUS=1 (2026-10-09, notes/native-hook-study-2026-10-09.md): which guest
// threads call sceGnmDrawIndex (the game's per-draw entry, 93% of draws) and how evenly - is the
// game's 6-core command-buffer recording visible at this HLE point?
const bool g_draw_thread_census = [] {
    const char* v = std::getenv("SHADPS4_DRAW_THREAD_CENSUS");
    return v && v[0] == '1';
}();
struct DrawThreadSlot {
    std::atomic<u64> draws{0};
    std::atomic<u32> os_tid{0};
};
std::array<DrawThreadSlot, 64> g_draw_threads{};
std::atomic<u32> g_draw_thread_slots{0};
std::atomic<u64> g_draw_total{0};
thread_local int t_draw_slot = -1;

void DrawThreadCensusNote() {
    if (t_draw_slot < 0) {
        const u32 slot = g_draw_thread_slots.fetch_add(1, std::memory_order_relaxed);
        t_draw_slot = slot < g_draw_threads.size() ? int(slot) : int(g_draw_threads.size() - 1);
        g_draw_threads[t_draw_slot].os_tid.store(static_cast<u32>(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff),
                                                 std::memory_order_relaxed);
    }
    g_draw_threads[t_draw_slot].draws.fetch_add(1, std::memory_order_relaxed);
    if ((g_draw_total.fetch_add(1, std::memory_order_relaxed) & 0xfffff) == 0xfffff) {
        std::string line;
        const u32 n = std::min<u32>(g_draw_thread_slots.load(), u32(g_draw_threads.size()));
        u64 sum = 0;
        for (u32 i = 0; i < n; ++i) {
            sum += g_draw_threads[i].draws.load(std::memory_order_relaxed);
        }
        for (u32 i = 0; i < n; ++i) {
            const u64 d = g_draw_threads[i].draws.exchange(0, std::memory_order_relaxed);
            line += fmt::format(" tid{}={:.1f}%", g_draw_threads[i].os_tid.load(),
                                sum ? 100.0 * double(d) / double(sum) : 0.0);
        }
        LOG_INFO(Lib_GnmDriver, "DrawThreadCensus: {} draws, {} threads:{}", sum, n, line);
    }
}
// SHADPS4_DRAW_TAG=1 (2026-10-09 evening, notes/native-path-2026-10-09.md): the game's job
// threads stamp every DrawIndex with (thread slot, sub-run id) in the trailing NOP. The sub-run id
// moves at every Set/Update Vs/Ps shader call on that thread (a draw-class object change), so
// consecutive draws with one id are the game's own run: same shaders, one job, one command buffer.
// The front copies the tag into the draw record; the back groups its per-draw cost by run.
const bool g_draw_tag = [] {
    const char* v = std::getenv("SHADPS4_DRAW_TAG");
    return v && (v[0] == '1' || v[0] == '2');
}();
std::atomic<u32> g_draw_tag_slots{0};
thread_local u32 t_draw_tag_slot = 0; // 1..15, 0 = not assigned yet
thread_local u32 t_draw_tag_run = 0;
thread_local u32 t_draw_tag_pgm[2] = {};  // current VS / PS program address (>> 8)
thread_local u64 t_draw_tag_hash[2] = {}; // and its shader hash (binary info)
std::atomic<u64> g_draw_tag_calls{0}, g_draw_tag_addr_changes{0}, g_draw_tag_hash_changes{0};

// s373 / s374: the game calls Set/Update for every draw-class object, and binds the same shader
// code from different addresses; a run ends only when the shader HASH changes (the emulator keys
// pipelines on it).
/// The shader hash from the binary info after the code, or 0 when not found (Opus review 3a:
/// AmdGpu::SearchBinaryInfo is UNREACHABLE on a miss - not on a game thread).
u64 ShaderHashNoAssert(u32 pgm) {
    if (pgm == 0) {
        return 0;
    }
    const auto* code = reinterpret_cast<const u32*>(u64{pgm} << 8);
    for (u32 i = 0; i < 0x4000; ++i) {
        const auto* info = std::bit_cast<const AmdGpu::BinaryInfo*>(code + i);
        if (info->Valid()) {
            return info->shader_hash;
        }
    }
    return 0;
}

/// r5 (2026-10-14): SHADPS4_SHADER_HASH_MEMO=1 - a per-thread memo of ShaderHashNoAssert. The game
/// binds per-object shader copies: the program address changed on 99.6% of Set/Update calls (s971r,
/// rooftop: 25.3 M of 25.4 M), and each change scanned the code dword by dword (up to 64 KB) for the
/// binary-info footer, on the job threads and Game:Main. An entry keeps the address, the footer's
/// dword offset, the hash and the code's first 4 dwords; a hit re-checks both (the footer still
/// valid with that hash, the code start unchanged), so new code at a reused address is rescanned.
const bool g_shader_hash_memo = [] {
    const char* v = std::getenv("SHADPS4_SHADER_HASH_MEMO");
    return v && v[0] == '1';
}();
struct ShaderHashMemoEntry {
    u32 pgm;
    u32 off;
    u64 hash;
    std::array<u32, 4> head;
};
u64 ShaderHashMemo(u32 pgm) {
    if (!g_shader_hash_memo || pgm == 0) {
        return ShaderHashNoAssert(pgm);
    }
    static thread_local std::array<ShaderHashMemoEntry, 1024> memo{};
    auto& e = memo[(pgm * 0x9E3779B1u) >> 22];
    const auto* code = reinterpret_cast<const u32*>(u64{pgm} << 8);
    if (e.pgm == pgm && e.off >= 4 && std::memcmp(code, e.head.data(), sizeof(e.head)) == 0) {
        const auto* info = std::bit_cast<const AmdGpu::BinaryInfo*>(code + e.off);
        if (info->Valid() && info->shader_hash == e.hash) {
            return e.hash;
        }
    }
    for (u32 i = 0; i < 0x4000; ++i) {
        const auto* info = std::bit_cast<const AmdGpu::BinaryInfo*>(code + i);
        if (info->Valid()) {
            e.pgm = pgm;
            e.off = i;
            e.hash = info->shader_hash;
            std::memcpy(e.head.data(), code, sizeof(e.head));
            return e.hash;
        }
    }
    e.pgm = 0;
    return 0;
}

/// q47 (SHADPS4_NR_SHARDED=2): the three census counters per thread, folded into the shared ones
/// every 4,096 calls (they were locked adds on one shared line by every job thread, per call).
const bool g_draw_tag_local = [] {
    const char* v = std::getenv("SHADPS4_NR_SHARDED");
    return v && v[0] == '2';
}();
thread_local u64 t_dt_calls = 0, t_dt_addr = 0, t_dt_hash = 0;

inline void DrawTagShaderChange(u32 stage, const u32* regs) {
    const u32 pgm = regs && regs[1] == 0 ? regs[0] : 0;
    u64 calls = 0;
    if (g_draw_tag_local) {
        if ((++t_dt_calls & 0xfff) == 0) {
            calls = g_draw_tag_calls.fetch_add(t_dt_calls, std::memory_order_relaxed) + t_dt_calls;
            g_draw_tag_addr_changes.fetch_add(t_dt_addr, std::memory_order_relaxed);
            g_draw_tag_hash_changes.fetch_add(t_dt_hash, std::memory_order_relaxed);
            t_dt_calls = t_dt_addr = t_dt_hash = 0;
        }
    } else {
        calls = g_draw_tag_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    if (pgm != t_draw_tag_pgm[stage]) {
        t_draw_tag_pgm[stage] = pgm;
        if (g_draw_tag_local) {
            ++t_dt_addr;
        } else {
            g_draw_tag_addr_changes.fetch_add(1, std::memory_order_relaxed);
        }
        const u64 hash = ShaderHashMemo(pgm);
        if (hash != t_draw_tag_hash[stage]) {
            t_draw_tag_hash[stage] = hash;
            if (g_draw_tag_local) {
                ++t_dt_hash;
            } else {
                g_draw_tag_hash_changes.fetch_add(1, std::memory_order_relaxed);
            }
            ++t_draw_tag_run;
        }
    }
    if (g_draw_tag_local ? calls != 0 && (calls >> 18) != ((calls - 4096) >> 18)
                         : (calls & 0x3ffff) == 0) {
        LOG_INFO(Lib_GnmDriver, "DrawTagShader: {} Set/Update calls, program address changed {}, "
                                "shader hash changed {}",
                 calls, g_draw_tag_addr_changes.load(), g_draw_tag_hash_changes.load());
    }
}

// SHADPS4_DRAW_TAG=2 (2026-10-09 evening, notes/native-path-2026-10-09.md sec. 2): what the game
// writes between consecutive draws on one job thread. A thread-local cursor sits at the end of
// the last HLE call that reserved packets (DrawIndex, Set/Update Vs/Ps: the game reserves `size`
// dwords and continues after them); at each DrawIndex the packets from the cursor to `cmdbuf`
// are walked (exact landing required) and classified.
const bool g_draw_gap = [] {
    const char* v = std::getenv("SHADPS4_DRAW_TAG");
    return v && v[0] == '2';
}();
struct GapStats {
    std::array<u64, 256> op{};
    std::array<u64, 0x400> ctx{};
    std::array<u64, 0x100> uc{};
    std::array<u64, 16> s{}; // see GapFlushLog
};
struct GapGlobal {
    std::array<std::atomic<u64>, 256> op{};
    std::array<std::atomic<u64>, 0x400> ctx{};
    std::array<std::atomic<u64>, 0x100> uc{};
    std::array<std::atomic<u64>, 16> s{};
    std::atomic<u64> total{};
};
GapGlobal g_gap;
thread_local GapStats t_gap;
thread_local const u32* t_gap_cursor = nullptr;
thread_local u32 t_gap_need[2] = {};  // user-SGPR window of the bound VS / PS (RSRC2 count)
thread_local u32 t_gap_since[2] = {}; // ud registers written since the shader was set
enum GapStat : u32 {
    GsDraws,
    GsOk,
    GsNoCursor,
    GsRange,
    GsBadType,
    GsNoLand,
    GsDwords,
    GsPackets,
    GsShOther,
    GsNopDw,
    GsFullGapVs,
    GsFullGapPs,
    GsFullSinceVs,
    GsFullSincePs,
    GsFullBoth,
    GsEmptyNeed,
};

void GapFlushLog() {
    auto& t = t_gap;
    for (u32 i = 0; i < 256; ++i) {
        if (t.op[i]) {
            g_gap.op[i].fetch_add(t.op[i], std::memory_order_relaxed);
        }
    }
    for (u32 i = 0; i < 0x400; ++i) {
        if (t.ctx[i]) {
            g_gap.ctx[i].fetch_add(t.ctx[i], std::memory_order_relaxed);
        }
    }
    for (u32 i = 0; i < 0x100; ++i) {
        if (t.uc[i]) {
            g_gap.uc[i].fetch_add(t.uc[i], std::memory_order_relaxed);
        }
    }
    for (u32 i = 0; i < 16; ++i) {
        g_gap.s[i].fetch_add(t.s[i], std::memory_order_relaxed);
    }
    const u64 n_draws = t.s[GsDraws];
    const u64 before = g_gap.total.fetch_add(n_draws, std::memory_order_relaxed);
    t = {};
    if ((before >> 20) == ((before + n_draws) >> 20)) {
        return;
    }
    std::array<u64, 16> v{};
    for (u32 i = 0; i < 16; ++i) {
        v[i] = g_gap.s[i].exchange(0, std::memory_order_relaxed);
    }
    const double ok = double(std::max<u64>(v[GsOk], 1));
    const auto pct = [](u64 x, u64 n) { return n ? 100.0 * double(x) / double(n) : 0.0; };
    std::string ops, ctxs, ucs;
    for (u32 i = 0; i < 256; ++i) {
        if (const u64 n = g_gap.op[i].exchange(0, std::memory_order_relaxed)) {
            ops += fmt::format(" {:#x}:{:.3f}", i, double(n) / ok);
        }
    }
    std::vector<std::pair<u64, u32>> top;
    for (u32 i = 0; i < 0x400; ++i) {
        if (const u64 n = g_gap.ctx[i].exchange(0, std::memory_order_relaxed)) {
            top.emplace_back(n, i);
        }
    }
    std::sort(top.rbegin(), top.rend());
    for (size_t k = 0; k < std::min<size_t>(top.size(), 24); ++k) {
        ctxs += fmt::format(" {:#x}:{:.4f}", top[k].second, double(top[k].first) / ok);
    }
    for (u32 i = 0; i < 0x100; ++i) {
        if (const u64 n = g_gap.uc[i].exchange(0, std::memory_order_relaxed)) {
            ucs += fmt::format(" {:#x}:{:.3f}", i, double(n) / ok);
        }
    }
    LOG_INFO(Lib_GnmDriver,
             "DrawGap: {} draws, walked {:.2f}% (no cursor {:.2f}% out of range {:.2f}% bad "
             "type {:.2f}% no landing {:.2f}%) | per walked gap: {:.1f} dwords {:.2f} packets "
             "{:.2f} other-SH regs {:.1f} NOP dwords | ud window rewritten in this gap: VS "
             "{:.1f}% PS {:.1f}%; since the shader set: VS {:.1f}% PS {:.1f}% both {:.1f}% "
             "(empty VS need {:.1f}%)",
             v[GsDraws], pct(v[GsOk], v[GsDraws]), pct(v[GsNoCursor], v[GsDraws]),
             pct(v[GsRange], v[GsDraws]), pct(v[GsBadType], v[GsDraws]),
             pct(v[GsNoLand], v[GsDraws]), double(v[GsDwords]) / ok, double(v[GsPackets]) / ok,
             double(v[GsShOther]) / ok, double(v[GsNopDw]) / ok, pct(v[GsFullGapVs], v[GsOk]),
             pct(v[GsFullGapPs], v[GsOk]), pct(v[GsFullSinceVs], v[GsOk]),
             pct(v[GsFullSincePs], v[GsOk]), pct(v[GsFullBoth], v[GsOk]),
             pct(v[GsEmptyNeed], v[GsOk]));
    LOG_INFO(Lib_GnmDriver, "DrawGap opcodes per walked gap:{}", ops);
    LOG_INFO(Lib_GnmDriver, "DrawGap context regs per walked gap (top 24):{}", ctxs);
    LOG_INFO(Lib_GnmDriver, "DrawGap uconfig regs per walked gap:{}", ucs);
}

void GapShaderSet(u32 stage, const u32* regs, const u32* cmdbuf, u32 size) {
    t_gap_cursor = cmdbuf + size;
    const u32 rsrc2 = regs ? regs[3] : 0;
    const u32 n = std::min<u32>((rsrc2 >> 1) & 0x1f, 16);
    t_gap_need[stage] = n == 16 ? 0xffffu : ((1u << n) - 1);
    t_gap_since[stage] = 0;
}

void GapDraw(const u32* cmdbuf) {
    auto& t = t_gap;
    ++t.s[GsDraws];
    const u32* p = t_gap_cursor;
    t_gap_cursor = cmdbuf + 10;
    const auto done = [&] {
        if (t.s[GsDraws] >= 8192) {
            GapFlushLog();
        }
    };
    if (!p) {
        ++t.s[GsNoCursor];
        return done();
    }
    if (cmdbuf < p || cmdbuf - p > 16384) {
        ++t.s[GsRange];
        return done();
    }
    u32 vs = 0, ps = 0, packets = 0, sh_other = 0, nop_dw = 0;
    std::array<u16, 64> ops{};
    u32 nops = 0;
    std::array<u16, 64> ctx{};
    u32 nctx = 0;
    std::array<u16, 16> uc{};
    u32 nuc = 0;
    const u32* q = p;
    while (q < cmdbuf) {
        const u32 h = *q;
        const u32 type = h >> 30;
        if (type == 2) {
            ++q;
            continue;
        }
        if (type != 3) {
            ++t.s[GsBadType];
            return done();
        }
        const u32 n = ((h >> 16) & 0x3fff) + 1;
        const u32 op = (h >> 8) & 0xff;
        if (q + 1 + n > cmdbuf) {
            break;
        }
        ++packets;
        if (nops < ops.size()) {
            ops[nops++] = static_cast<u16>(op);
        }
        if (op == static_cast<u32>(PM4ItOpcode::SetShReg)) {
            const u32 off = q[1] & 0xffff;
            for (u32 r = off; r < off + n - 1; ++r) {
                if (r >= 0x4c && r < 0x5c) {
                    vs |= 1u << (r - 0x4c);
                } else if (r >= 0x0c && r < 0x1c) {
                    ps |= 1u << (r - 0x0c);
                } else {
                    ++sh_other;
                }
            }
        } else if (op == static_cast<u32>(PM4ItOpcode::SetContextReg)) {
            const u32 off = q[1] & 0xffff;
            for (u32 r = off; r < off + n - 1 && nctx < ctx.size(); ++r) {
                ctx[nctx++] = static_cast<u16>(r & 0x3ff);
            }
        } else if (op == static_cast<u32>(PM4ItOpcode::SetUconfigReg)) {
            if (nuc < uc.size()) {
                uc[nuc++] = static_cast<u16>(q[1] & 0xff);
            }
        } else if (op == static_cast<u32>(PM4ItOpcode::Nop)) {
            nop_dw += n;
        }
        q += 1 + n;
    }
    if (q != cmdbuf) {
        ++t.s[GsNoLand];
        return done();
    }
    ++t.s[GsOk];
    t.s[GsDwords] += u64(cmdbuf - p);
    t.s[GsPackets] += packets;
    t.s[GsShOther] += sh_other;
    t.s[GsNopDw] += nop_dw;
    for (u32 i = 0; i < nops; ++i) {
        ++t.op[ops[i]];
    }
    for (u32 i = 0; i < nctx; ++i) {
        ++t.ctx[ctx[i]];
    }
    for (u32 i = 0; i < nuc; ++i) {
        ++t.uc[uc[i]];
    }
    t_gap_since[0] |= vs;
    t_gap_since[1] |= ps;
    t.s[GsFullGapVs] += (vs & t_gap_need[0]) == t_gap_need[0];
    t.s[GsFullGapPs] += (ps & t_gap_need[1]) == t_gap_need[1];
    const bool fvs = (t_gap_since[0] & t_gap_need[0]) == t_gap_need[0];
    const bool fps = (t_gap_since[1] & t_gap_need[1]) == t_gap_need[1];
    t.s[GsFullSinceVs] += fvs;
    t.s[GsFullSincePs] += fps;
    t.s[GsFullBoth] += fvs && fps;
    t.s[GsEmptyNeed] += t_gap_need[0] == 0;
    done();
}

// SHADPS4_NATIVE_REC (native_rec.h): the job thread's side. A cursor at the end of the last HLE
// call that reserved packets; at every DrawIndex the packets the game wrote since are walked
// (exact landing) into a per-thread mirror of the VS / PS user-data registers and the per-draw
// state, and the draw either continues the thread's open run or starts a new one.
namespace NR = AmdGpu::NativeRec;
struct RecThread {
    const u32* cursor{};
    std::array<u32, 16> vs{}, ps{};
    u32 vs_known{}, ps_known{};
    u32 index_type{}, prim{}, instances{1};
    bool it_known{}, prim_known{}, inst_known{};
    bool set_called{}; ///< a Set (not Update) shader call since the last draw: ends the run
    bool wl{true};     ///< every packet since the last draw was on the run whitelist
    u32 gap{};         ///< dwords walked since the last draw
    NR::Record* ring{};
    u32 next{};
    NR::Record* head{};
    u32 head_idx{};
    u32 run_id{};
    u32 run_index_type{}, run_prim{};
    // Stage a2.
    u32 vs_nuser{16}, ps_nuser{16}; ///< RSRC2 user-SGPR counts of the bound VS / PS
    u8* payload{};
    u64 payload_pos{};
    AmdGpu::CpSplit::VertexInputsDerived vi_scratch{};
    bool run_derived{};
    u64 spec_key{}; ///< NativeRecSpecKey of VS + PS at the last derived draw
    u64 run_fetch{};
    u64 run_vi_key{};
    // SHADPS4_NATIVE_GPU_DIRECT: the GPU record ring, the run-union ring and the open run's entry.
    u32* gpu{};
    NR::RunUnion* unions{};
    u32 union_next{};
    u32 run_union{~0u};
    // Coverage (sec. 11): the inherited registers of the last derive and of the open run.
    u32 inh_vs{}, inh_ps{};
    u32 run_inh_vs{}, run_inh_ps{};
};
thread_local RecThread t_rec;
/// Stage a2 job-side derive outcomes: 0 ok, 1 user data unknown, 2 VS not in the table yet,
/// 3 no pipeline for (VS, PS, fetch shader); 4 run ended by the vertex layout.
/// SHADPS4_NR_SHARDED=1 (2026-10-13): the per-draw census counters of the NativeRec path were
/// global atomics bumped by every job thread on every draw (s838 JobWorker1 profile: two of them
/// were ~38% of NativeRec's samples - cache-line ping-pong). Each thread now counts in its own
/// shard; readers sum the shards. Off: one shared shard, as before.
inline const bool g_nr_sharded = [] {
    const char* v = std::getenv("SHADPS4_NR_SHARDED");
    return v && (v[0] == '1' || v[0] == '2');
}();
/// SHADPS4_NR_SHARDED=2 (q47, s854 JobWorker1 profile): a shard is written only by its own
/// thread, so it counts with a plain load + store instead of a locked add. The locked add at the
/// end of NativeRecGpu (g_q15_why) was 2.4% of the job thread's wall time - a full barrier that
/// waits for the GPU record's write-combined stores to drain. Readers report deltas.
inline const bool g_nr_owner_inc = [] {
    const char* v = std::getenv("SHADPS4_NR_SHARDED");
    return v && v[0] == '2';
}();
/// r5 (2026-10-14): SHADPS4_NR_QUIET=1 - the NativeRec census counters (pure diagnostics, logged
/// every 2^20 draws) are not counted at all: ShardedCounters::Ref::fetch_add was still 2.9% of the
/// job threads' render-span samples (s949 / s971r: a TLS lookup + a call per counter, ~4-6 per draw).
inline const bool g_nr_quiet = [] {
    const char* v = std::getenv("SHADPS4_NR_QUIET");
    return v && v[0] == '1';
}();
template <size_t N, int Tag>
struct ShardedCounters {
    struct alignas(64) Shard {
        std::array<std::atomic<u64>, N> v{};
    };
    static inline std::mutex mu;
    static inline std::vector<Shard*> shards;
    static inline Shard shared;
    static inline std::array<u64, N> reported{}; ///< owner-inc mode: sums already reported
    static Shard& Local() {
        if (!g_nr_sharded) {
            return shared;
        }
        static thread_local Shard* t = nullptr;
        if (!t) [[unlikely]] {
            t = new Shard();
            std::scoped_lock lk{mu};
            shards.push_back(t);
        }
        return *t;
    }
    struct Ref {
        size_t i;
        void fetch_add(u64 x, std::memory_order o = std::memory_order_relaxed) {
            if (g_nr_quiet) {
                return;
            }
            auto& c = Local().v[i];
            if (g_nr_owner_inc) {
                c.store(c.load(std::memory_order_relaxed) + x, std::memory_order_relaxed);
            } else {
                c.fetch_add(x, o);
            }
        }
        u64 load(std::memory_order o = std::memory_order_relaxed) const {
            u64 s = shared.v[i].load(o);
            std::scoped_lock lk{mu};
            for (Shard* sh : shards) {
                s += sh->v[i].load(o);
            }
            return g_nr_owner_inc ? s - reported[i] : s;
        }
        u64 exchange(u64, std::memory_order o = std::memory_order_relaxed) {
            u64 s = shared.v[i].exchange(0, o);
            std::scoped_lock lk{mu};
            if (g_nr_owner_inc) {
                u64 sum = 0;
                for (Shard* sh : shards) {
                    sum += sh->v[i].load(std::memory_order_relaxed);
                }
                const u64 d = sum - reported[i];
                reported[i] = sum;
                return s + d;
            }
            for (Shard* sh : shards) {
                s += sh->v[i].exchange(0, o);
            }
            return s;
        }
    };
    Ref operator[](size_t i) const {
        return Ref{i};
    }
    static constexpr size_t size() {
        return N;
    }
};
ShardedCounters<5, 1> g_nr_d{};
// Diagnostic: which packet put a gap off the whitelist (opcode; 256 = other SH register, 257 = a
// marker / flip NOP, 258 = other uconfig, 259 = other DMA).
ShardedCounters<260, 2> g_nr_off{};
std::atomic<u32> g_nr_last_nop{0};

/// Walks the packets the game wrote from the cursor to `to` into the mirror. False (and every
/// known bit cleared) when the walk does not land exactly on `to`.
/// SHADPS4_JOB_CENSUS (2026-10-12): cycles of one NativeRecDraw / NativeRecShader call, by thread.
struct NativeRecTimer {
    u64 t0;
    NativeRecTimer() : t0(Common::FaultMonitor::JobCensus() ? __rdtsc() : 0) {}
    ~NativeRecTimer() {
        if (t0) {
            Common::FaultMonitor::NoteNativeRecCycles(__rdtsc() - t0);
        }
    }
};
bool NativeRecWalk(const u32* to) {
    auto& r = t_rec;
    const u32* p = r.cursor;
    bool landed = false;
    // s378 / Opus review 3c: a job's own gaps are ~30 dwords; a walk over 1,024 crossed into
    // another command buffer's packets - fail it before reading them.
    if (p && to >= p && to - p <= 1024) {
        const u32* q = p;
        while (q < to) {
            const u32 h = *q;
            const u32 type = h >> 30;
            if (type == 2) {
                ++q;
                continue;
            }
            if (type != 3) {
                break;
            }
            const u32 n = ((h >> 16) & 0x3fff) + 1;
            if (q + 1 + n > to) {
                break;
            }
            switch (static_cast<PM4ItOpcode>((h >> 8) & 0xff)) {
            case PM4ItOpcode::SetShReg: {
                const u32 off = q[1] & 0xffff;
                for (u32 j = 0; j + 1 < n; ++j) {
                    const u32 reg = off + j;
                    if (reg >= 0x4c && reg < 0x5c) {
                        r.vs[reg - 0x4c] = q[2 + j];
                        r.vs_known |= 1u << (reg - 0x4c);
                    } else if (reg >= 0x0c && reg < 0x1c) {
                        r.ps[reg - 0x0c] = q[2 + j];
                        r.ps_known |= 1u << (reg - 0x0c);
                    } else {
                        r.wl = false;
                        g_nr_off[256].fetch_add(1, std::memory_order_relaxed);
                    }
                }
                break;
            }
            case PM4ItOpcode::Nop: {
                // The game's own 0x6875xxxx hints (sharp follows, s379: 0x6875000d before most
                // draws) are plain; flip payloads and debug markers have back effects.
                const u32 pl = q[1];
                const bool marker = pl == 0x68750001u || pl == 0x68750002u || pl == 0x68750003u ||
                                    pl == 0x6875000eu || (pl >= 0x68750776u && pl <= 0x68750781u);
                if (marker) {
                    r.wl = false;
                    g_nr_off[257].fetch_add(1, std::memory_order_relaxed);
                    g_nr_last_nop.store(pl, std::memory_order_relaxed);
                }
                break;
            }
            case PM4ItOpcode::IndexType:
                r.index_type = q[1];
                r.it_known = true;
                break;
            case PM4ItOpcode::NumInstances:
                r.instances = q[1];
                r.inst_known = true;
                break;
            case PM4ItOpcode::SetUconfigReg:
                if ((q[1] & 0xffff) == 0x242 && n >= 2) { // VGT_PRIMITIVE_TYPE
                    r.prim = q[2];
                    r.prim_known = true;
                } else {
                    r.wl = false;
                    g_nr_off[258].fetch_add(1, std::memory_order_relaxed);
                }
                break;
            case PM4ItOpcode::DmaData:
                if (n < 6 || q[4] != 0x3022Cu) {
                    r.wl = false; // only the L2 code prefetch (a no-op for the emulator)
                    g_nr_off[259].fetch_add(1, std::memory_order_relaxed);
                    r.vs_known = r.ps_known = 0; // may target registers (review 5)
                }
                break;
            case PM4ItOpcode::LoadShReg:
            case PM4ItOpcode::SetShRegOffset:
            case PM4ItOpcode::WriteData:
            case PM4ItOpcode::CopyData:
            case PM4ItOpcode::IndirectBuffer:
                // Opus review 5: these can write SH registers the walk does not decode.
                r.vs_known = r.ps_known = 0;
                r.wl = false;
                g_nr_off[(h >> 8) & 0xff].fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                r.wl = false;
                g_nr_off[(h >> 8) & 0xff].fetch_add(1, std::memory_order_relaxed);
                break;
            }
            q += 1 + n;
        }
        landed = q == to;
        r.gap += static_cast<u32>(to - p);
    }
    if (!landed) {
        r.vs_known = r.ps_known = 0;
        r.it_known = r.prim_known = r.inst_known = false;
        r.wl = false;
    }
    return landed;
}

/// Opus review 3b: a shader setter the run logic does not decode (non-350 PS, embedded, ES / GS /
/// HS / LS): the stage's hash becomes 0 (no derive, the run ends).
void DrawTagOtherShader(const u32* cmdbuf, u32 size, u32 stage);

void NativeRecShader(const u32* cmdbuf, u32 size, bool is_set, u32 stage, const u32* regs) {
    const NativeRecTimer timer;
    NativeRecWalk(cmdbuf);
    t_rec.cursor = cmdbuf + size;
    t_rec.set_called = t_rec.set_called || is_set;
    const u32 nuser = regs ? std::min<u32>((regs[3] >> 1) & 0x1f, 16) : 0;
    (stage == 0 ? t_rec.vs_nuser : t_rec.ps_nuser) = nuser;
}

/// The fields of the bound descriptors that Shader::StageSpecialization keys a permutation on
/// (specialization.h), hashed from the job thread's own BoundStage. Equal keys + equal shaders +
/// equal context = the same pipeline (s382: 2.3% of derived members changed pipeline without it).
u64 NativeRecSpecKey(const Shader::BoundStage& bs, const Shader::Gcn::FetchShaderData* fetch) {
    u64 h = 0xcbf29ce484222325ull;
    const auto mix = [&h](u64 v) { h = (h ^ v) * 0x100000001B3ull; };
    const auto mix_bytes = [&](const auto& x) {
        u64 v = 0;
        std::memcpy(&v, &x, std::min<size_t>(sizeof(x), sizeof(v)));
        mix(v);
    };
    const Shader::Info& info = *bs.info;
    if (fetch && info.stage == Shader::Stage::Vertex) {
        for (const auto& attrib : fetch->attributes) {
            const AmdGpu::Buffer sharp = attrib.GetSharp(bs);
            mix(u64(AmdGpu::GetNumberClass(sharp.GetNumberFmt())));
            mix_bytes(sharp.DstSelect());
        }
    }
    for (const auto& desc : info.buffers) {
        const AmdGpu::Buffer sharp = desc.GetSharp(bs);
        if (!sharp) {
            mix(0x51);
            continue;
        }
        mix(u64(sharp.GetStride()) << 8 | u64(sharp.swizzle_enable) << 1 | sharp.add_tid_enable);
        if (desc.is_formatted) {
            mix(u64(sharp.GetDataFmt()) << 8 | u64(sharp.GetNumberFmt()));
            mix_bytes(sharp.DstSelect());
            mix(u64(sharp.GetNumberConversion()));
        }
        if (sharp.swizzle_enable) {
            mix(u64(sharp.index_stride) << 4 | sharp.element_size);
        }
    }
    for (const auto& desc : info.images) {
        const AmdGpu::Image sharp = desc.GetSharp(bs);
        if (!sharp) {
            mix(0x52);
            continue;
        }
        mix(u64(sharp.GetViewType(desc.is_array)) << 8 |
            u64(AmdGpu::IsInteger(sharp.GetNumberFmt())) << 4 | u64(sharp.IsCube()) << 2 |
            u64(sharp.IsBlockCoded1D()) << 1 |
            u64(!desc.is_written && sharp.GetNumberFmt() == AmdGpu::NumberFormat::Srgb));
        if (desc.is_written) {
            mix_bytes(sharp.DstSelect());
        }
        mix(u64(sharp.GetNumberConversion()));
        mix(desc.NumBindings(bs));
    }
    for (const auto& desc : info.fmasks) {
        const AmdGpu::Image sharp = desc.GetSharp(bs);
        mix(u64(sharp.width) << 32 | sharp.height);
    }
    for (const auto& desc : info.samplers) {
        const AmdGpu::Sampler sharp = desc.GetSharp(bs);
        mix(u64(sharp.force_unnormalized) << 1 | sharp.force_degamma);
    }
    return h;
}

/// Stage a2: derive what the front's FRONT_DERIVE + stage copy produce (flattened user data of
/// VS / PS, the vertex inputs) on the job thread, through the program table the front publishes.
/// Writes into the thread's payload ring; fills the record's derive fields (derived = 0: none).
ShardedCounters<4, 3> g_nr_need{};
ShardedCounters<32, 4> g_nr_need_reg{};

void NativeRecDerive(NR::Record& rec) {
    auto& r = t_rec;
    rec.derived = 0;
    rec.pipeline = nullptr;
    rec.fetch_addr = 0;
    rec.inh_vs = 0;
    rec.inh_ps = 0;
    r.inh_vs = 0;
    r.inh_ps = 0;
    const u32 cnt_vs = r.vs_nuser >= 16 ? 0xffffu : ((1u << r.vs_nuser) - 1);
    const u32 cnt_ps = r.ps_nuser >= 16 ? 0xffffu : ((1u << r.ps_nuser) - 1);
    const u64 vs_hash = t_draw_tag_hash[0];
    const u64 ps_hash = t_draw_tag_hash[1];
    const u32 fb = NR::VsFetchBase(vs_hash);
    if (fb == 0) {
        g_nr_d[2].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    u64 fetch = 0;
    if (fb != 0xffffffffu) {
        const u32 base = fb - 1;
        if (base + 1 >= 16 || (r.vs_known >> base & 3) != 3) {
            g_nr_d[1].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        fetch = (u64(r.vs[base + 1]) << 32 | r.vs[base]) & 0xFFFFFFFFFFFFull;
    }
    u64 needs = ~u64{0};
    const auto* gp =
        static_cast<const Vulkan::GraphicsPipeline*>(NR::TableFind(vs_hash, ps_hash, fetch, &needs));
    if (!gp) {
        g_nr_d[3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Coverage (sec. 11): every strict register (walker roots, fetch SGPRs) must be known; the
    // other needed ones may be inherited (constant over the run; the back takes them from the
    // batch opener and checks them at every member). With an unknown read mask: every register
    // below the RSRC2 count, strict.
    const u32 nv = static_cast<u32>(needs), np = static_cast<u32>(needs >> 32);
    const u32 need_vs = nv == 0xffffffffu ? cnt_vs : (nv & cnt_vs);
    const u32 need_ps = np == 0xffffffffu ? cnt_ps : (np & cnt_ps);
    const u32 strict_vs = nv == 0xffffffffu ? need_vs : ((nv >> 16) & cnt_vs);
    const u32 strict_ps = np == 0xffffffffu ? need_ps : ((np >> 16) & cnt_ps);
    r.inh_vs = NR::g_inherit_on ? need_vs & ~r.vs_known : 0;
    r.inh_ps = NR::g_inherit_on ? need_ps & ~r.ps_known : 0;
    if ((r.vs_known & strict_vs) != strict_vs || (r.ps_known & strict_ps) != strict_ps ||
        (!NR::g_inherit_on &&
         ((r.vs_known & need_vs) != need_vs || (r.ps_known & need_ps) != need_ps))) {
        g_nr_d[1].fetch_add(1, std::memory_order_relaxed);
        // Diagnostic: masks unknown (VS / PS) vs known-but-missing, and which registers.
        g_nr_need[nv == 0xffffffffu ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        g_nr_need[np == 0xffffffffu ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
        const u32 miss_vs = need_vs & ~r.vs_known, miss_ps = need_ps & ~r.ps_known;
        for (u32 i = 0; i < 16; ++i) {
            if ((miss_vs >> i) & 1) {
                g_nr_need_reg[i].fetch_add(1, std::memory_order_relaxed);
            }
            if ((miss_ps >> i) & 1) {
                g_nr_need_reg[16 + i].fetch_add(1, std::memory_order_relaxed);
            }
        }
        return;
    }
    const auto stages = gp->GetStages();
    const Shader::Info* vsi = stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
    const Shader::Info* psi = stages[static_cast<u32>(Shader::LogicalStage::Fragment)];
    if (!vsi || vsi->stage != Shader::Stage::Vertex) {
        g_nr_d[3].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 nvs = vsi->srt_info.flattened_bufsize_dw;
    const u32 nps = psi ? psi->srt_info.flattened_bufsize_dw : 0u;
    // Room for both flat buffers and a full vertex-input block, 8-aligned, never across the end.
    const u64 worst = (u64(nvs + nps) * 4 + 7) / 8 * 8 + sizeof(AmdGpu::CpSplit::VertexInputsDerived);
    u64 pos = r.payload_pos;
    if ((pos & (NR::PayloadSize - 1)) + worst > NR::PayloadSize) {
        pos = (pos + NR::PayloadSize) & ~(NR::PayloadSize - 1);
    }
    // Opus review 2: claim the bytes before writing them (seqlock order: claim, release fence,
    // write); a reader re-checks the claim after its copy.
    NR::g_payload_claim[t_draw_tag_slot].store(pos + worst, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    u8* const base = r.payload + (pos & (NR::PayloadSize - 1));
    u32* flat_vs = reinterpret_cast<u32*>(base);
    u32* flat_ps = flat_vs + nvs;
    // As Info::RefreshFlatBuf: the user data first, then the SRT walker.
    std::memset(flat_vs, 0, size_t(nvs + nps) * 4);
    std::memcpy(flat_vs, r.vs.data(), std::min<u32>(nvs, 16) * 4);
    if (vsi->srt_info.walker_func && nvs != 0) {
        vsi->srt_info.walker_func(r.vs.data(), flat_vs);
    }
    if (psi && nps != 0) {
        std::memcpy(flat_ps, r.ps.data(), std::min<u32>(nps, 16) * 4);
        if (psi->srt_info.walker_func) {
            psi->srt_info.walker_func(r.ps.data(), flat_ps);
        }
    }
    u8* vi_out = base + (u64(nvs + nps) * 4 + 7) / 8 * 8;
    Shader::BoundStage vs{};
    vs.info = vsi;
    vs.user_data = std::span<const u32>{r.vs.data(), 16};
    vs.flattened_ud_buf = std::span<const u32>{flat_vs, nvs};
    vs.pgm_base = VAddr{t_draw_tag_pgm[0]} << 8;
    auto& vi = r.vi_scratch;
    gp->DeriveVertexInputs(vi, 0, 0, vs);
    vi.Write(vi_out);
    r.spec_key = NativeRecSpecKey(vs, gp->GetFetchShader());
    if (psi) {
        Shader::BoundStage ps{};
        ps.info = psi;
        ps.user_data = std::span<const u32>{r.ps.data(), 16};
        ps.flattened_ud_buf = std::span<const u32>{flat_ps, nps};
        ps.pgm_base = VAddr{t_draw_tag_pgm[1]} << 8;
        r.spec_key = r.spec_key * 31 + NativeRecSpecKey(ps, nullptr);
    }
    rec.flat_vs_dw = nvs;
    rec.flat_ps_dw = nps;
    rec.vi_bytes = vi.Bytes();
    rec.payload_pos = pos;
    rec.pipeline = gp;
    rec.fetch_addr = fetch;
    rec.derived = 1;
    rec.inh_vs = r.inh_vs;
    rec.inh_ps = r.inh_ps;
    r.payload_pos = pos + (u64(vi_out - base) + rec.vi_bytes + 7) / 8 * 8;
    g_nr_d[0].fetch_add(1, std::memory_order_relaxed);
}


/// SHADPS4_NATIVE_GPU_DIRECT (notes/native-path-2026-10-09.md sec. 10): the GPU record of a
/// derived draw (what NativeOpen / NativeJoin would write, with guest addresses instead of buffer
/// offsets) and the run's binding ranges. Called after the run decision (rec.run_k / head_idx set).
static_assert(NR::GpuRecDwords == Shader::NativeRecordDwords);
static_assert(NR::GpuRecHiOffset == Shader::NativeRecordHiOffset);

/// q3b: indirect records by outcome (0 total, 1 not derived, 2 index type unknown, 3 no union,
/// 4 killed, 5 ok).
ShardedCounters<6, 5> g_nr_ind{};
/// q18 B: SHADPS4_NR_ITYPE_LATE=1 (see q18_pt_vertex.py).
const bool g_nr_itype_late = [] {
    const char* e = std::getenv("SHADPS4_NR_ITYPE_LATE");
    return e && e[0] == '1';
}();
thread_local u32 t_nr_gpu_why = 0; // q3b: NativeRecGpu's exit (1-4; 0 = ok / not reached)
/// q15: every record's NativeRecGpu exit (see q15_nr_gpu_why.py): 0 ok, 1-3 as q3b, 4 kill other, 5 stage,
/// 6 slots, 7 unmapped V#.
ShardedCounters<8, 6> g_q15_why{};
thread_local u32 t_nr_kill = 0;
thread_local bool t_nr_indirect = false; // q4: the record being written is an indirect draw's

void NativeRecGpu(NR::Record& rec, u32 idx, u32 slot) {
    auto& r = t_rec;
    t_nr_gpu_why = 0;
    t_nr_kill = 0;
    rec.gpu_ok = 0;
    SCOPE_EXIT {
        const u32 w = rec.gpu_ok ? 0u : t_nr_gpu_why == 4 && t_nr_kill != 0 ? t_nr_kill : t_nr_gpu_why;
        g_q15_why[w & 7].fetch_add(1, std::memory_order_relaxed);
    };
    rec.union_idx = ~0u;
    rec.push_inh = 0;
    rec.inh_slots = 0;
    if (!r.gpu) [[unlikely]] {
        if (const auto alloc = NR::g_gpu_alloc.load(std::memory_order_acquire)) {
            r.gpu = alloc(slot); // ring mode: mapped GPU memory
        }
        if (!r.gpu) {
            r.gpu = new u32[size_t(NR::RingSize) * NR::GpuRecDwords]();
        }
        r.unions = new NR::RunUnion[NR::UnionMask + 1]();
        NR::g_gpu_recs[slot].store(r.gpu, std::memory_order_release);
        NR::g_unions[slot].store(r.unions, std::memory_order_release);
    }
    // q4: an indirect record needs no known index type (the translator reads the mirror's).
    const bool itype_late = g_nr_itype_late && !r.it_known && !t_nr_indirect; // q18 B
    if (!rec.derived || (!r.it_known && !t_nr_indirect && !itype_late)) {
        t_nr_gpu_why = !rec.derived ? 1 : 2;
        if (rec.run_k == 0) {
            r.run_union = ~0u;
        } else if (r.run_union != ~0u) {
            r.unions[r.run_union & NR::UnionMask].stamp.store(0, std::memory_order_release);
            r.run_union = ~0u;
        }
        return;
    }
    const auto* gp = static_cast<const Vulkan::GraphicsPipeline*>(rec.pipeline);
    const auto stages = gp->GetStages();
    constexpr u32 vs_i = static_cast<u32>(Shader::LogicalStage::Vertex);
    constexpr u32 ps_i = static_cast<u32>(Shader::LogicalStage::Fragment);
    if (rec.run_k == 0) {
        // A new run: claim a union entry (stamp 0 while it is reset).
        r.run_union = r.union_next++;
        auto& nu = r.unions[r.run_union & NR::UnionMask];
        nu.stamp.store(0, std::memory_order_relaxed);
        nu.n.store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        nu.guest_mask = 0;
        nu.eq_mask = ~u64{0};
        nu.vcount = 0;
        nu.vlo.fill(~u64{0});
        nu.vhi.fill(0);
        nu.ilo = ~u64{0};
        nu.ihi = 0;
        nu.stamp.store(idx + 1, std::memory_order_release);
    }
    if (r.run_union == ~0u) {
        t_nr_gpu_why = 3;
        return;
    }
    auto& u = r.unions[r.run_union & NR::UnionMask];
    if (u.stamp.load(std::memory_order_relaxed) != rec.head_idx + 1) {
        r.run_union = ~0u;
        t_nr_gpu_why = 3;
        return;
    }
    // A draw of the run without a GPU record: the union no longer covers the run (no later draw
    // may publish a count past it).
    const auto kill = [&] {
        t_nr_gpu_why = 4;
        u.stamp.store(0, std::memory_order_release);
        r.run_union = ~0u;
    };
    std::array<u32, NR::GpuRecDwords> g{};
    const u8* base = r.payload + (rec.payload_pos & (NR::PayloadSize - 1));
    const u32* flat_vs = reinterpret_cast<const u32*>(base);
    const u32* flat_ps = flat_vs + rec.flat_vs_dw;
    Shader::PushData push{};
    Shader::Backend::Bindings bnd{};
    u32 bi = 0;
    auto* memory = Core::Memory::Instance();
    std::array<u32, 512> key;
    u32 kn = 0;
    bool key_ok = true;
    bool key_inh = false;
    const auto add_key = [&](const auto& sharp) {
        constexpr u32 words = static_cast<u32>(sizeof(sharp) / sizeof(u32));
        if (kn + words > key.size()) {
            key_ok = false;
            return;
        }
        std::memcpy(&key[kn], &sharp, sizeof(sharp));
        kn += words;
    };
    for (u32 i = 0; i < stages.size(); ++i) {
        const Shader::Info* info = stages[i];
        if (!info) {
            continue;
        }
        if (i != vs_i && i != ps_i) {
            t_nr_kill = 5;
            return kill();
        }
        Shader::BoundStage bs{};
        bs.info = info;
        bs.user_data = std::span<const u32>{i == vs_i ? r.vs.data() : r.ps.data(), 16};
        bs.flattened_ud_buf = std::span<const u32>{i == vs_i ? flat_vs : flat_ps,
                                                   i == vs_i ? rec.flat_vs_dw : rec.flat_ps_dw};
        bs.pgm_base = VAddr{i == vs_i ? rec.vs_pgm : rec.ps_pgm} << 8;
        const u32 inh = i == vs_i ? rec.inh_vs : rec.inh_ps;
        {
            // Push-layout slots of inherited registers (PushUd's order: ud_mask ascending).
            u32 p = bnd.user_data;
            for (u32 mask = info->ud_mask.mask; mask != 0; mask &= mask - 1) {
                const u32 reg = static_cast<u32>(std::countr_zero(mask));
                if (((inh >> reg) & 1) && p < 32) {
                    rec.push_inh |= 1u << p;
                }
                ++p;
            }
        }
        bs.PushUd(bnd, push);
        for (const auto& desc : info->buffers) {
            if (bi >= NR::UnionSlots) {
                t_nr_kill = 6;
                return kill();
            }
            if (desc.buffer_type == Shader::BufferType::Guest && !desc.inline_cbuf &&
                desc.sharp_idx < 16 && ((inh >> desc.sharp_idx) & 0xfu) != 0) {
                // The V# sits in inherited registers: the opener's binding (the back's table).
                rec.inh_slots |= u64{1} << bi;
                ++bi;
                continue;
            }
            if (desc.buffer_type == Shader::BufferType::Guest) {
                const auto vsharp = desc.GetSharp(bs);
                const u64 addr = vsharp.base_address;
                const u64 size = (addr == 0 || vsharp.GetSize() == 0)
                                     ? 0
                                     : memory->TryClampRangeSize(addr, vsharp.GetSize());
                if (size == 0 && addr != 0 && vsharp.GetSize() != 0) {
                    t_nr_kill = 7;
                    return kill(); // Opus G2 #6: unmapped huge V# (no assert on a game thread)
                }
                g[Shader::NativeRecordOffOffset + bi] = static_cast<u32>(addr);
                g[Shader::NativeRecordSizeOffset + bi] =
                    static_cast<u32>(std::min<u64>(size, 0xffffffffull));
                // SHADPS4_NATIVE_PT: bits 32..39 of the guest address, one byte per binding.
                g[Shader::NativeRecordHiOffset + bi / 4] |=
                    static_cast<u32>((addr >> 32) & 0xffu) << ((bi % 4) * 8);
                const u64 bit = u64{1} << bi;
                if (!(u.guest_mask & bit)) {
                    u.lo[bi] = addr;
                    u.hi[bi] = addr + size;
                    u.first_addr[bi] = addr;
                    u.first_size[bi] = static_cast<u32>(std::min<u64>(size, 0xffffffffull));
                    u.guest_mask |= bit;
                } else {
                    u.lo[bi] = std::min(u.lo[bi], addr);
                    u.hi[bi] = std::max<u64>(u.hi[bi], addr + size);
                    if (addr != u.first_addr[bi] ||
                        std::min<u64>(size, 0xffffffffull) != u.first_size[bi]) {
                        u.eq_mask &= ~bit;
                    }
                }
            }
            ++bi;
        }
        // A T# / S# word read straight from an inherited register: 0 in the key (ImgInhBit).
        // Only those words - the known words still tell materials apart (advisor: s418 showed
        // half-inherited 4-register groups). The back compares members with the opener's job
        // key; the inherited registers themselves are compared with the opener's at every
        // confirm (sec. 11).
        const auto add_key_inh = [&](const auto& sharp, u32 idx, u32 reg_words) {
            const u32 at = kn;
            add_key(sharp);
            if (!key_ok || idx >= 16) {
                return;
            }
            for (u32 w = 0; w < reg_words && idx + w < 16; ++w) {
                if ((inh >> (idx + w)) & 1) {
                    key[at + w] = 0;
                    key_inh = true;
                }
            }
        };
        for (const auto& desc : info->images) {
            add_key_inh(desc.GetSharp(bs), desc.sharp_idx, desc.is_r128 ? 4u : 8u);
        }
        for (const auto& desc : info->samplers) {
            add_key_inh(desc.GetSharp(bs), desc.sharp_idx, desc.is_inline_sampler ? 0u : 4u);
        }
    }
    rec.img_n = key_ok ? (kn | (key_inh ? NR::ImgInhBit : 0u)) : ~0u;
    NR::ImageKeyHash(key.data(), kn, rec.img_h1, rec.img_h2);
    std::memcpy(g.data(), push.ud_regs.data(), sizeof(u32) * Shader::NUM_USER_DATA_REGS);
    const auto& vi = r.vi_scratch;
    const u32 vcount = std::min<u32>(vi.header.count, 16);
    rec.vcount = vcount;
    for (u32 a = 0; a < vcount; ++a) {
        const auto& at = vi.attribs[a];
        const bool null = at.range == AmdGpu::CpSplit::VertexInputNoRange;
        const u64 vaddr = null ? 0 : at.base_address;
        rec.vaddr[a] = vaddr;
        if (!null) {
            const u32 loc = at.desc.location;
            if (loc < Shader::NativeMaxPulledLocations) {
                g[Shader::NativeRecordVoffOffset + loc] = static_cast<u32>(vaddr);
                g[Shader::NativeRecordVsizeOffset + loc] = 0xffffffffu;
                g[Shader::NativeRecordVhiOffset + loc / 4] |=
                    static_cast<u32>((vaddr >> 32) & 0xffu) << ((loc % 4) * 8);
            }
            u.vlo[a] = std::min(u.vlo[a], vaddr);
            u.vhi[a] = std::max<u64>(u.vhi[a], vaddr + at.size);
        }
    }
    u.vcount = std::max(u.vcount, vcount);
    const u32 index_size = itype_late || (r.index_type & 3) == 1 ? 4u : 2u; // q18 B: unknown -> upper bound
    u.ilo = std::min(u.ilo, rec.index_addr);
    u.ihi = std::max<u64>(u.ihi, rec.index_addr + u64(rec.index_count) * index_size);
    const auto* fetch = gp->GetFetchShader();
    rec.vo_ud = fetch && fetch->vertex_offset_sgpr >= 0 && fetch->vertex_offset_sgpr < 16
                    ? r.vs[fetch->vertex_offset_sgpr]
                    : 0;
    rec.io_ud = fetch && fetch->instance_offset_sgpr >= 0 && fetch->instance_offset_sgpr < 16
                    ? r.vs[fetch->instance_offset_sgpr]
                    : 0;
    u32* const gpu_rec = r.gpu + size_t(idx & (NR::RingSize - 1)) * NR::GpuRecDwords;
    std::memcpy(gpu_rec, g.data(), sizeof(u32) * NR::GpuRecJobDwords);
    // The guest-address hi bytes (SHADPS4_NATIVE_PT), always; 128-159 stay the back's.
    std::memcpy(gpu_rec + NR::GpuRecHiOffset, g.data() + NR::GpuRecHiOffset,
                sizeof(u32) * (NR::GpuRecDwords - NR::GpuRecHiOffset));
    _mm_sfence(); // Opus G2 #5: write-combined ring memory before the union count
    rec.union_idx = r.run_union;
    rec.gpu_ok = itype_late ? 2 : 1; // q18 B: 2 = only the A2 translator may use it
    u.n.store(rec.run_k + 1, std::memory_order_release);
}

ShardedCounters<12, 7> g_nr_why{};

/// Returns the ring index written for this draw.
u32 NativeRecDraw(const u32* cmdbuf, u64 index_addr, u32 index_count, u32 tag,
                  u32 packet_dwords = 10, bool indirect = false) {
    const NativeRecTimer timer;
    auto& r = t_rec;
    if ((tag >> 28) > NR::MaxRecordSlot) [[unlikely]] {
        return ~0u; // a shared overflow slot: no records (Opus review 6)
    }
    if (!r.ring) [[unlikely]] {
        // Payload and ring published together, payload first (the front loads the ring first).
        r.payload = new u8[NR::PayloadSize];
        NR::g_payloads[tag >> 28].store(r.payload, std::memory_order_release);
        r.ring = new NR::Record[NR::RingSize]();
        NR::g_rings[tag >> 28].store(r.ring, std::memory_order_release);
    }
    const bool landed = NativeRecWalk(cmdbuf);
    const bool wl = landed && r.wl;
    const u32 gap = landed ? r.gap : 0;
    r.wl = true;
    r.gap = 0;
    r.cursor = cmdbuf + packet_dwords;
    const u32 run_id = tag & 0x0fffffffu;
    const u32 idx = r.next++;
    if (NR::g_gpu_mode != 0) {
        NR::g_job_next[tag >> 28].store(r.next, std::memory_order_relaxed);
    }
    NR::Record& rec = r.ring[idx & (NR::RingSize - 1)];
    rec.stamp.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    NativeRecDerive(rec);
    // The vertex layout (fetch shader, attribute formats / strides) is part of the pipeline: a
    // run ends where it changes (s380: 2.9% of members changed pipeline under one shader pair).
    u64 vi_key = 0;
    if (rec.derived) {
        const auto& vi = r.vi_scratch;
        vi_key = (rec.fetch_addr ^ r.spec_key) * 0x9E3779B97F4A7C15ull + vi.header.count;
        for (u32 a = 0; a < vi.header.count; ++a) {
            const auto& d = vi.attribs[a].desc;
            vi_key = (vi_key ^ (u64(d.vk_format) << 32 | d.stride)) * 0x100000001B3ull;
            vi_key = (vi_key ^ (u64(d.location) << 8 | d.instanced)) * 0x100000001B3ull;
        }
    }
    // A run continues only through draws the job thread derived (the front skips members in
    // stage b), with the head's vertex layout.
    const bool layout_same = rec.derived && r.run_derived && vi_key == r.run_vi_key &&
                             rec.inh_vs == r.run_inh_vs && rec.inh_ps == r.run_inh_ps;
    if (r.head && rec.derived && r.run_derived && !layout_same) {
        g_nr_d[4].fetch_add(1, std::memory_order_relaxed);
    }
    const bool cont = !indirect && layout_same && r.head && wl && !r.set_called && run_id == r.run_id &&
                      r.head->count.load(std::memory_order_relaxed) < NR::MaxRun && r.it_known && r.index_type == r.run_index_type &&
                      r.prim_known && r.prim == r.run_prim &&
                      r.head->stamp.load(std::memory_order_relaxed) == r.head_idx + 1;
    {
        // Why a draw did not continue the open run (first failing condition), logged per 2^20.
        const u32 why = cont                                ? 0
                        : !r.head                           ? 1
                        : !landed                           ? 2
                        : !wl                               ? 3
                        : r.set_called                      ? 4
                        : run_id != r.run_id                ? 5
                        : r.head->count.load(std::memory_order_relaxed) >= NR::MaxRun ? 6
                        : !r.it_known                       ? 7
                        : r.index_type != r.run_index_type  ? 8
                        : !r.prim_known                     ? 9
                        : r.prim != r.run_prim              ? 10
                                                            : 11;
        g_nr_why[why].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<u64> total{0};
        thread_local u64 t_total = 0;
        const u64 tick = g_nr_sharded ? t_total++ : total.fetch_add(1, std::memory_order_relaxed);
        if ((tick & 0xfffff) == 0xfffff) {
            std::string line;
            for (u32 i = 0; i < g_nr_why.size(); ++i) {
                line += fmt::format(" {}={}", i, g_nr_why[i].exchange(0));
            }
            std::string off;
            for (u32 i = 0; i < g_nr_off.size(); ++i) {
                if (const u64 c = g_nr_off[i].exchange(0)) {
                    off += fmt::format(" {:#x}={}", i, c);
                }
            }
            LOG_INFO(Lib_GnmDriver,
                     "q15 all records' gpu exit: ok {} not derived {} index type unknown {} no union {} "
                     "kill other {} kill stage {} kill slots {} kill unmapped V# {}",
                     g_q15_why[0].exchange(0), g_q15_why[1].exchange(0), g_q15_why[2].exchange(0),
                     g_q15_why[3].exchange(0), g_q15_why[4].exchange(0), g_q15_why[5].exchange(0),
                     g_q15_why[6].exchange(0), g_q15_why[7].exchange(0));
            LOG_INFO(Lib_GnmDriver,
                     "q3b indirect records: total {} not derived {} index type unknown {} no union "
                     "{} killed {} ok {}",
                     g_nr_ind[0].load(), g_nr_ind[1].load(), g_nr_ind[2].load(),
                     g_nr_ind[3].load(), g_nr_ind[4].load(), g_nr_ind[5].load());
            LOG_INFO(Lib_GnmDriver,
                     "NativeRec job-side derive: ok={} user-data unknown={} VS not in table={} "
                     "no pipeline={} | runs ended by the vertex layout={}",
                     g_nr_d[0].exchange(0), g_nr_d[1].exchange(0), g_nr_d[2].exchange(0),
                     g_nr_d[3].exchange(0), g_nr_d[4].exchange(0));
            {
                std::string regs;
                for (u32 i = 0; i < 32; ++i) {
                    if (const u64 c = g_nr_need_reg[i].exchange(0)) {
                        regs += fmt::format(" {}{}={}", i < 16 ? "vs" : "ps", i & 15, c);
                    }
                }
                LOG_INFO(Lib_GnmDriver,
                         "NativeRec user-data unknown: VS mask unknown={} known={} | PS mask "
                         "unknown={} known={} | missing needed registers:{}",
                         g_nr_need[0].exchange(0), g_nr_need[1].exchange(0),
                         g_nr_need[2].exchange(0), g_nr_need[3].exchange(0), regs);
            }
            LOG_INFO(Lib_GnmDriver, "NativeRec off-whitelist packets (256 SH 257 NOP 258 uconfig 259 DMA):{} last NOP payload {:#x}",
                     off, g_nr_last_nop.load());
            LOG_INFO(Lib_GnmDriver,
                     "NativeRec job side: 0=continued 1=no head 2=walk failed 3=packet off the "
                     "whitelist 4=Set shader call 5=shader hash 6=max run 7/8 index type 9/10 "
                     "primitive 11=head overwritten:{}",
                     line);
        }
    }
    r.set_called = false;
    rec.tag = tag;
    rec.cmd = cmdbuf;
    rec.index_addr = index_addr;
    rec.index_count = index_count;
    rec.instances = r.instances;
    rec.inst_known = r.inst_known ? 1u : 0u;
    rec.vs_known = r.vs_known;
    rec.ps_known = r.ps_known;
    rec.gap_dwords = gap;
    rec.ud_vs = r.vs;
    rec.ud_ps = r.ps;
    rec.vs_pgm = t_draw_tag_pgm[0];
    rec.ps_pgm = t_draw_tag_pgm[1];
    if (cont) {
        rec.run_k = r.head->count.load(std::memory_order_relaxed);
        rec.head_idx = r.head_idx;
        rec.count.store(0, std::memory_order_relaxed);
        rec.end.store(nullptr, std::memory_order_relaxed);
        r.head->end.store(cmdbuf + packet_dwords, std::memory_order_relaxed);
        r.head->count.store(rec.run_k + 1, std::memory_order_release);
    } else {
        rec.run_k = 0;
        rec.head_idx = idx;
        rec.count.store(1, std::memory_order_relaxed);
        rec.end.store(cmdbuf + packet_dwords, std::memory_order_relaxed);
        r.head = &rec;
        r.head_idx = idx;
        r.run_id = run_id;
        r.run_index_type = r.index_type;
        r.run_prim = r.prim;
        r.run_derived = rec.derived != 0;
        r.run_vi_key = vi_key;
        r.run_inh_vs = rec.inh_vs;
        r.run_inh_ps = rec.inh_ps;
    }
    if (NR::g_gpu_mode != 0) [[unlikely]] {
        NativeRecGpu(rec, idx, tag >> 28);
    }
    rec.stamp.store(idx + 1, std::memory_order_release);
    return idx;
}

void DrawTagOtherShader(const u32* cmdbuf, u32 size, u32 stage) {
    if (!g_draw_tag && !NR::g_mode) {
        return;
    }
    for (u32 s = 0; s < 2; ++s) {
        if (s == stage || stage > 1) {
            t_draw_tag_pgm[s] = 0;
            t_draw_tag_hash[s] = 0;
        }
    }
    ++t_draw_tag_run;
    if (NR::g_mode && cmdbuf) {
        NativeRecShader(cmdbuf, size, true, stage > 1 ? 0 : stage, nullptr);
    }
}

/// Opus review 5: a default-state call starts a command buffer's state over: forget the mirror.
void NativeRecResetState() {
    if (!NR::g_mode) {
        return;
    }
    auto& r = t_rec;
    r.cursor = nullptr;
    r.vs_known = r.ps_known = 0;
    r.it_known = r.prim_known = r.inst_known = false;
    r.head = nullptr;
}

inline u32 DrawTagValue() {
    if (t_draw_tag_slot == 0) [[unlikely]] {
        t_draw_tag_slot = std::min<u32>(g_draw_tag_slots.fetch_add(1) + 1, 15u);
    }
    return (t_draw_tag_slot << 28) | (t_draw_tag_run & 0x0fffffffu);
}
} // namespace

s32 PS4_SYSV_ABI sceGnmDrawIndex(u32* cmdbuf, u32 size, u32 index_count, uintptr_t index_addr,
                                 u32 flags, u32 type) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (g_draw_thread_census) [[unlikely]] {
        DrawThreadCensusNote();
    }
    if (g_draw_gap && cmdbuf && size == 10) [[unlikely]] {
        GapDraw(cmdbuf);
    }

    if (cmdbuf && (size == 10) && (index_addr != 0) && (index_addr & 1) == 0 &&
        (flags & 0x1ffffffe) == 0) { // no predication will be set in the packet
        auto* draw_index = reinterpret_cast<PM4CmdDrawIndex2*>(cmdbuf);
        draw_index->header =
            PM4Type3Header{PM4ItOpcode::DrawIndex2, 4, PM4ShaderType::ShaderGraphics};
        draw_index->max_size = index_count;
        draw_index->index_base_lo = u32(index_addr);
        draw_index->index_base_hi = u32(index_addr >> 32);
        draw_index->index_count = index_count;
        draw_index->draw_initiator = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        WriteTrailingNop<3>(cmdbuf + 6);
        if (NR::g_mode) [[unlikely]] {
            const u32 tag = DrawTagValue();
            const u32 idx = NativeRecDraw(cmdbuf, index_addr, index_count, tag);
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf + 6);
            nop->data_block[0] = idx == ~0u ? 0u : NR::Magic;
            nop->data_block[1] = tag;
            nop->data_block[2] = idx;
        } else if (g_draw_tag) [[unlikely]] {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf + 6);
            nop->data_block[0] = PM4CmdNop::PayloadType::NativeDrawTag;
            nop->data_block[1] = DrawTagValue();
            nop->data_block[2] = 0;
        }
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexAuto(u32* cmdbuf, u32 size, u32 index_count, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 7) &&
        (flags & 0x1ffffffe) == 0) { // no predication will be set in the packet
        cmdbuf = WritePacket<PM4ItOpcode::DrawIndexAuto>(
            cmdbuf, PM4ShaderType::ShaderGraphics, index_count,
            sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u);
        WriteTrailingNop<3>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

static inline bool predicate_on(u32 flags) {
    return (flags & 1) != 0;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                         u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                         u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && (shader_stage < ShaderStages::Max) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        u32* const packet = cmdbuf;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirect>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0u;

        cmdbuf += 4;
        WriteTrailingNop<3>(cmdbuf);
        // q2 (SHADPS4_A2_INDIRECT=1): a job record for the indirect draw (index address / count
        // live in INDEX_BASE / the GPU args), tagged in the trailing NOP as DrawIndex does. It
        // neither continues a run nor lets the next draw continue it.
        static const bool a2_indirect = [] {
            const char* e = std::getenv("SHADPS4_A2_INDIRECT");
            return e && e[0] == '1';
        }();
        if (NR::g_mode && a2_indirect && !predicate_on(flags)) [[unlikely]] {
            const u32 tag = DrawTagValue();
            t_nr_indirect = true;
            const u32 idx = NativeRecDraw(packet, 0, 0, tag, 9, true);
            t_nr_indirect = false;
            t_rec.wl = false;
            if (idx != ~0u) {
                g_nr_ind[0].fetch_add(1, std::memory_order_relaxed);
                g_nr_ind[t_nr_gpu_why == 0 ? 5 : t_nr_gpu_why].fetch_add(1,
                                                                        std::memory_order_relaxed);
            }
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->data_block[0] = idx == ~0u ? 0u : NR::Magic;
            nop->data_block[1] = tag;
            nop->data_block[2] = idx;
        }
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectCountMulti(u32* cmdbuf, u32 size, u32 data_offset,
                                                   u32 max_count, u64 count_addr, u32 shader_stage,
                                                   u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                                   u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if ((!sceKernelIsNeoMode() || !UseNeoCompatSequences) && cmdbuf && (size == 16) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u) &&
        (shader_stage == ShaderStages::Vs || shader_stage == ShaderStages::Es ||
         shader_stage == ShaderStages::Ls)) {

        cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, 2);
        cmdbuf = WriteBody(cmdbuf, 0u);
        cmdbuf += 1;

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirectCountMulti>(
            cmdbuf, 9, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = (count_addr != 0 ? 1u : 0u) << 0x1e;
        cmdbuf[4] = max_count;
        *(u64*)(&cmdbuf[5]) = count_addr;
        cmdbuf[7] = sizeof(DrawIndexedIndirectArgs);
        cmdbuf[8] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        cmdbuf += 9;
        WriteTrailingNop<2>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndexIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                              u32 shader_stage, u32 vertex_sgpr_offset,
                                              u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 11) && (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u) &&
        (shader_stage == ShaderStages::Vs || shader_stage == ShaderStages::Es ||
         shader_stage == ShaderStages::Ls)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirectMulti>(
            cmdbuf, 6, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = max_count;
        cmdbuf[4] = sizeof(DrawIndexedIndirectArgs);
        cmdbuf[5] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        cmdbuf += 6;
        WriteTrailingNop<3>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndexMultiInstanced() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    UNREACHABLE();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexOffset(u32* cmdbuf, u32 size, u32 index_offset, u32 index_count,
                                       u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexOffset2>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);
        cmdbuf = WriteBody(cmdbuf, index_count, index_offset, index_count,
                           sceKernelIsNeoMode() ? flags & 0xe0000000u : 0u);

        WriteTrailingNop<3>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                    u32 vertex_sgpr_offset, u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && (shader_stage < ShaderStages::Max) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndirect>(cmdbuf, 4, PM4ShaderType::ShaderGraphics,
                                                        predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u; // auto index

        cmdbuf += 4;
        WriteTrailingNop<3>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndirectCountMulti() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    UNREACHABLE();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                         u32 shader_stage, u32 vertex_sgpr_offset,
                                         u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && size == 11 && shader_stage < ShaderStages::Max && vertex_sgpr_offset < 0x10 &&
        instance_sgpr_offset < 0x10) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndirectMulti>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];
        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = max_count;
        cmdbuf[4] = sizeof(DrawIndirectArgs);
        cmdbuf[5] = sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u; // auto index

        cmdbuf += 6;
        WriteTrailingNop<3>(cmdbuf);
        return ORBIS_OK;
    }
    return -1;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    const auto& SetupContext = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        std::memcpy(cmdbuf, &InitSequence[2], (InitSequence.size() - 2) * 4);
        cmdbuf += InitSequence.size() - 2;

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    return SetupContext(cmdbuf, size, true);
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState175(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
    cmdbuf = ClearContextState(cmdbuf);
    std::memcpy(cmdbuf, &InitSequence175[2], (InitSequence175.size() - 2) * 4);
    cmdbuf += InitSequence175.size() - 2;

    const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
    WriteTrailingNop(cmdbuf, cmdbuf_left);

    return HwInitPacketSize;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState200(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    const auto& SetupContext200 = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        if (sceKernelIsNeoMode()) {
            if (!UseNeoCompatSequences) {
                std::memcpy(cmdbuf, &InitSequence200Neo[2], (InitSequence200Neo.size() - 2) * 4);
                cmdbuf += InitSequence200Neo.size() - 2;
            } else {
                std::memcpy(cmdbuf, &InitSequence200NeoCompat[2],
                            (InitSequence200NeoCompat.size() - 2) * 4);
                cmdbuf += InitSequence200NeoCompat.size() - 2;
            }
        } else {
            std::memcpy(cmdbuf, &InitSequence200[2], (InitSequence200.size() - 2) * 4);
            cmdbuf += InitSequence200.size() - 2;
        }

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    return SetupContext200(cmdbuf, size, true);
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState350(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    const auto& SetupContext350 = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        if (sceKernelIsNeoMode()) {
            if (!UseNeoCompatSequences) {
                std::memcpy(cmdbuf, &InitSequence350Neo[2], (InitSequence350Neo.size() - 2) * 4);
                cmdbuf += InitSequence350Neo.size() - 2;
            } else {
                std::memcpy(cmdbuf, &InitSequence350NeoCompat[2],
                            (InitSequence350NeoCompat.size() - 2) * 4);
                cmdbuf += InitSequence350NeoCompat.size() - 2;
            }
        } else {
            std::memcpy(cmdbuf, &InitSequence350[2], (InitSequence350.size() - 2) * 4);
            cmdbuf += InitSequence350.size() - 2;
        }

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    return SetupContext350(cmdbuf, size, true);
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    constexpr auto CtxInitPacketSize = 0x20u;
    if (size != CtxInitPacketSize) {
        return 0;
    }

    if (sceKernelIsNeoMode()) {
        std::memcpy(cmdbuf, CtxInitSequenceNeo.data(), CtxInitSequenceNeo.size() * 4);
    } else {
        std::memcpy(cmdbuf, CtxInitSequence.data(), CtxInitSequence.size() * 4);
    }
    return CtxInitPacketSize;
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState400(u32* cmdbuf, u32 size) {
    NativeRecResetState();
    LOG_TRACE(Lib_GnmDriver, "called");

    constexpr auto CtxInitPacketSize = 0x100u;
    if (size != CtxInitPacketSize) {
        return 0;
    }

    if (sceKernelIsNeoMode()) {
        if (!UseNeoCompatSequences) {
            std::memcpy(cmdbuf, CtxInitSequence400Neo.data(), CtxInitSequence400Neo.size() * 4);
        } else {
            std::memcpy(cmdbuf, CtxInitSequence400NeoCompat.data(),
                        CtxInitSequence400NeoCompat.size() * 4);
        }
    } else {
        std::memcpy(cmdbuf, CtxInitSequence400.data(), CtxInitSequence400.size() * 4);
    }
    return CtxInitPacketSize;
}

int PS4_SYSV_ABI sceGnmDrawOpaqueAuto() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmDriverCaptureInProgress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterface() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForGpuDebugger() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForGpuException() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForHDRScopes() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForReplay() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForResourceRegistration() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForValidation() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

int PS4_SYSV_ABI sceGnmDriverInternalVirtualQuery() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmDriverTraceInProgress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI sceGnmDriverTriggerCapture() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_RAZOR_NOT_LOADED;
}

int PS4_SYSV_ABI sceGnmEndWorkload(u64 workload) {
    if (workload != 0) {
        return (0xf < ((workload >> 0x38) & 0xff)) * 2;
    }
    return 2;
}

s32 PS4_SYSV_ABI sceGnmFindResourcesPublic() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

void PS4_SYSV_ABI sceGnmFlushGarlic() {
    LOG_TRACE(Lib_GnmDriver, "(STUBBED) called");
}

int PS4_SYSV_ABI sceGnmGetCoredumpAddress() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetCoredumpMode() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetCoredumpProtectionFaultTimestamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetDbgGcHandle() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return -1;
}

int PS4_SYSV_ABI sceGnmGetDebugTimestamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetEqEventType(const OrbisKernelEvent* ev) {
    LOG_TRACE(Lib_GnmDriver, "called");
    return sceKernelGetEventData(ev);
}

int PS4_SYSV_ABI sceGnmGetEqTimeStamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetGpuBlockStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

u32 PS4_SYSV_ABI sceGnmGetGpuCoreClockFrequency() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // On console this uses an ioctl check, but we assume it is equal to just checking for neo mode.
    return sceKernelIsNeoMode() ? 911'000'000 : 800'000'000;
}

int PS4_SYSV_ABI sceGnmGetGpuInfoStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetLastWaitedAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetNumTcaUnits() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetOffChipTessellationBufferSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    return tessellation_offchip_buffer_size;
}

int PS4_SYSV_ABI sceGnmGetOwnerName() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetPhysicalCounterFromVirtualized() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

u32 PS4_SYSV_ABI sceGnmGetProtectionFaultTimeStamp() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0;
}

int PS4_SYSV_ABI sceGnmGetResourceBaseAddressAndSizeInBytes() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceName() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceShaderGuid() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceType() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetShaderProgramBaseAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetShaderStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

VAddr PS4_SYSV_ABI sceGnmGetTheTessellationFactorRingBufferBaseAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (tessellation_factors_ring_addr == -1) {
        auto* memory = Core::Memory::Instance();
        auto& address_space = memory->GetAddressSpace();
        tessellation_factors_ring_addr = address_space.SystemReservedVirtualBase() +
                                         address_space.SystemReservedVirtualSize() - 0x10000000;
    }

    return tessellation_factors_ring_addr;
}

void PS4_SYSV_ABI sceGnmGpuPaDebugEnter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

void PS4_SYSV_ABI sceGnmGpuPaDebugLeave() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

s32 PS4_SYSV_ABI sceGnmInsertDingDongMarker(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf == nullptr || size != 4) {
        return -1;
    }
    WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0u, 0u, 0u);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertPopMarker(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 6)) {
        cmdbuf =
            WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics,
                                          PM4CmdNop::PayloadType::DebugMarkerPop, 0u, 0u, 0u, 0u);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmInsertPushColorMarker(u32* cmdbuf, u32 size, const char* marker, u32 color) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 0xc) >> 2) + ((len + 0x10) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugColorMarkerPush;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            *reinterpret_cast<u32*>(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len + 8) =
                color;
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len + 8 + sizeof(u32),
                        0, packet_size * 4 - marker_len - 8 - sizeof(u32));
            return ORBIS_OK;
        }
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmInsertPushMarker(u32* cmdbuf, u32 size, const char* marker) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 8) >> 2) + ((len + 0xc) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugMarkerPush;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len, 0,
                        packet_size * 4 - marker_len);
            return ORBIS_OK;
        }
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmInsertSetColorMarker() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertSetMarker(u32* cmdbuf, u32 size, const char* marker) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 8) >> 2) + ((len + 0xc) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugSetMarker;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len, 0,
                        packet_size * 4 - marker_len);
            return ORBIS_OK;
        }
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmInsertThreadTraceMarker() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertWaitFlipDone(u32* cmdbuf, u32 size, s32 vo_handle, u32 buf_idx) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size != 7) {
        return -1;
    }

    uintptr_t label_addr{};
    ASSERT_MSG(VideoOut::sceVideoOutGetBufferLabelAddress(vo_handle, &label_addr) == 16,
               "sceVideoOutGetBufferLabelAddress call failed");

    auto* wait_reg_mem = reinterpret_cast<PM4CmdWaitRegMem*>(cmdbuf);
    wait_reg_mem->header = PM4Type3Header{PM4ItOpcode::WaitRegMem, 5};
    wait_reg_mem->raw = 0x13u;
    *reinterpret_cast<uintptr_t*>(&wait_reg_mem->poll_addr_lo) =
        (label_addr + buf_idx * sizeof(uintptr_t)) & ~0x3ull;
    wait_reg_mem->ref = 0u;
    wait_reg_mem->mask = 0xffff'ffffu;
    wait_reg_mem->poll_interval = 10u;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmIsCoredumpValid() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmIsUserPaEnabled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI sceGnmLogicalCuIndexToPhysicalCuIndex() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmLogicalCuMaskToPhysicalCuMask(s64, s32 logical_cu_mask) {
    LOG_INFO(Lib_GnmDriver, "called, logical_cu_mask: {}", logical_cu_mask);
    return logical_cu_mask;
}

int PS4_SYSV_ABI sceGnmLogicalTcaUnitToPhysical() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmMapComputeQueue(u32 pipe_id, u32 queue_id, VAddr ring_base_addr,
                                       u32 ring_size_dw, u32* read_ptr_addr) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (pipe_id >= Liverpool::NumComputePipes) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_PIPE_ID;
    }

    if (queue_id >= Liverpool::NumQueuesPerPipe) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_QUEUE_ID;
    }

    if (VAddr(ring_base_addr) % 256 != 0) { // alignment check
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_RING_BASE_ADDR;
    }

    if (!std::has_single_bit(ring_size_dw)) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_RING_SIZE;
    }

    if (VAddr(read_ptr_addr) % 4 != 0) { // alignment check
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_READ_PTR_ADDR;
    }

    const auto vqid =
        liverpool->asc_queues.insert(VAddr(ring_base_addr), read_ptr_addr, ring_size_dw, pipe_id);
    // We need to offset index as `dingDong` assumes it to be from the range [1..64]
    const auto gnm_vqid = vqid.index + 1;
    LOG_INFO(Lib_GnmDriver, "ASC pipe {} queue {} mapped to vqueue {}", pipe_id, queue_id,
             gnm_vqid);

    const auto& queue = liverpool->asc_queues[vqid];
    *queue.read_addr = 0u;

    return gnm_vqid;
}

int PS4_SYSV_ABI sceGnmMapComputeQueueWithPriority(u32 pipe_id, u32 queue_id, VAddr ring_base_addr,
                                                   u32 ring_size_dw, u32* read_ptr_addr,
                                                   u32 pipePriority) {
    LOG_TRACE(Lib_GnmDriver, "called");

    (void)pipePriority;
    return sceGnmMapComputeQueue(pipe_id, queue_id, ring_base_addr, ring_size_dw, read_ptr_addr);
}

int PS4_SYSV_ABI sceGnmPaDisableFlipCallbacks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmPaEnableFlipCallbacks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmPaHeartbeat() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmQueryResourceRegistrationUserMemoryRequirements() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRaiseUserExceptionEvent() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRegisterGdsResource() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

void PS4_SYSV_ABI sceGnmRegisterGnmLiveCallbackConfig() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

s32 PS4_SYSV_ABI sceGnmRegisterOwner(void* handle, const char* name) {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmRegisterResource(void* res_handle, void* owner_handle, const void* addr,
                                        size_t size, const char* name, int res_type,
                                        u64 user_data) {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRequestFlipAndSubmitDone() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRequestFlipAndSubmitDoneForWorkload() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRequestMipStatsReportAndReset() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmResetVgtControl(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (cmdbuf == nullptr || size != 3) {
        return -1;
    }
    if (sceKernelIsNeoMode()) {
        if (!UseNeoCompatSequences) {
            PM4CmdSetData::SetUconfigReg(cmdbuf, 0x40000258u, 0x6d007fu); // IA_MULTI_VGT_PARAM
        } else {
            PM4CmdSetData::SetContextReg(cmdbuf, 0x100002aau, 0xd00ffu); // IA_MULTI_VGT_PARAM
        }
    } else {
        PM4CmdSetData::SetContextReg(cmdbuf, 0x2aau, 0xffu); // IA_MULTI_VGT_PARAM
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSdmaClose() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaConstFill() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyLinear() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyTiled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyWindow() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaFlush() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaGetMinCmdSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaOpen() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmSetCsShader(u32* cmdbuf, u32 size, const u32* cs_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x18) {
        return -1;
    }
    if (!cs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer in shader registers.");
        return -1;
    }
    if (cs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x20cu, cs_regs[0],
                                     0u); // COMPUTE_PGM_LO/COMPUTE_PGM_HI
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x212u, cs_regs[2],
                                     cs_regs[3]); // COMPUTE_PGM_RSRC1/COMPUTE_PGM_RSRC2
    cmdbuf = PM4CmdSetData::SetShReg(
        cmdbuf, 0x207u, cs_regs[4], cs_regs[5],
        cs_regs[6]); // COMPUTE_NUM_THREAD_X/COMPUTE_NUM_THREAD_Y/COMPUTE_NUM_THREAD_Z

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetCsShaderWithModifier(u32* cmdbuf, u32 size, const u32* cs_regs,
                                               u32 modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x18) {
        return -1;
    }
    if (!cs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer in shader registers.");
        return -1;
    }
    if ((modifier & 0xfffffc3fu) != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask.");
        return -1;
    }
    if (cs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x20cu, cs_regs[0],
                                     0u); // COMPUTE_PGM_LO/COMPUTE_PGM_HI
    const u32 rsrc1 = modifier == 0 ? cs_regs[2] : (cs_regs[2] & 0xfffffc3fu) | modifier;
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x212u, rsrc1,
                                     cs_regs[3]); // COMPUTE_PGM_RSRC1/COMPUTE_PGM_RSRC2
    cmdbuf = PM4CmdSetData::SetShReg(
        cmdbuf, 0x207u, cs_regs[4], cs_regs[5],
        cs_regs[6]); // COMPUTE_NUM_THREAD_X/COMPUTE_NUM_THREAD_Y/COMPUTE_NUM_THREAD_Z

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEmbeddedPsShader(u32* cmdbuf, u32 size, u32 shader_id,
                                           u32 shader_modifier) {
    DrawTagOtherShader(cmdbuf, size, 1);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (shader_id > 1) {
        LOG_ERROR(Lib_GnmDriver, "Unknown shader id {}", shader_id);
        return ORBIS_GNM_ERROR_FAILURE;
    }

    // clang-format off
    constexpr static std::array ps0_code alignas(256) = {
        0xbeeb03ffu, 0x00000003u, // s_mov_b32     vcc_hi, $0x00000003
        0x7e000280u,              // v_mov_b32     v0, 0
        0x5e000100u,              // v_cvt_pkrtz_f16_f32 v0, v0, v0
        0xbf800000u,              // s_nop
        0xf8001c0fu, 0x00000000u, // exp           mrt0, v0, v0 compr vm done
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00002043u, 0u, 0xb0a45b2bu, 0x1d39766du, 0x72044b7bu, 0x0000000fu,
        // PS regs
        0x0fe000f0u, 0u, 0xc0000u, 4u, 0u, 4u, 2u, 2u, 0u, 0u, 0x10u, 0xfu, 0xcu, 0u, 0u, 0u,
    };

    const auto shader0_addr = uintptr_t(ps0_code.data()); // Original address is 0xfe000f00
    const static u32 ps0_regs[] = {
        u32(shader0_addr >> 8), u32(shader0_addr >> 40), 0xc0000u, 4u, 0u, 4u, 2u, 2u, 0u, 0u, 0x10u, 0xfu, 0xcu};

    constexpr static std::array ps1_code alignas(256) = {
        0xbeeb03ffu, 0x00000003u, // s_mov_b32     vcc_hi, $0x00000003
        0x7e040280u,              // v_mov_b32     v2, 0 
        0xf8001803u, 0x02020202u, // exp           mrt0, v2, v2, off, off vm done
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00001841u, 0x04080002u, 0x98b9cb94u, 0u, 0x6f130734u, 0x0000000fu,
        // PS regs
        0x0fe000f2u, 0u, 0x2000u, 0u, 0u, 2u, 2u, 2u, 0u, 0u, 0x10u, 3u, 0xcu,
    };

    const auto shader1_addr = uintptr_t(ps1_code.data()); // Original address is 0xfe000f20
    const static u32 ps1_regs[] = {
        u32(shader1_addr >> 8), u32(shader1_addr >> 40), 0x2000u, 0u, 0u, 2u, 2u, 2u, 0u, 0u, 0x10u, 3u, 0xcu};
    // clang-format on

    const auto ps_regs = shader_id == 0 ? ps0_regs : ps1_regs;

    // Normally the driver will do a call to `sceGnmSetPsShader350()`, but this function has
    // a check for zero in the upper part of shader address. In our case, the address is a
    // pointer to a stack memory, so the check will likely fail. To workaround it we will
    // repeat set shader functionality here as it is trivial.
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                     ps_regs[1]); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                     ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c4u, ps_regs[4],
                                          ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                          ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

    WriteTrailingNop<11>(cmdbuf);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEmbeddedVsShader(u32* cmdbuf, u32 size, u32 shader_id,
                                           u32 shader_modifier) {
    DrawTagOtherShader(cmdbuf, size, 0);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (shader_id != 0) {
        LOG_ERROR(Lib_GnmDriver, "Unknown shader id {}", shader_id);
        return ORBIS_GNM_ERROR_FAILURE;
    }

    // A fullscreen triangle with one uv set
    // clang-format off
    constexpr static std::array shader_code alignas(256) = {
        0xbeeb03ffu, 0x00000007u, // s_mov_b32     vcc_hi, $0x00000007
        0x36020081u,              // v_and_b32     v1, 1, v0
        0x34020281u,              // v_lshlrev_b32 v1, 1, v1
        0x360000c2u,              // v_and_b32     v0, -2, v0
        0x4a0202c1u,              // v_add_i32     v1, vcc, -1, v1
        0x4a0000c1u,              // v_add_i32     v0, vcc, -1, v0
        0x7e020b01u,              // v_cvt_f32_i32 v1, v1
        0x7e000b00U,              // v_cvt_f32_i32 v0, v0
        0x7e040280u,              // v_mov_b32     v2, 0
        0x7e0602f2u,              // v_mov_b32     v3, 1.0
        0xf80008cfu, 0x03020001u, // exp           pos0, v1, v0, v2, v3 done
        0xf800020fu, 0x03030303u, // exp           param0, v3, v3, v3, v3
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00004047u, 0u, 0x47f8c29fu, 0x9b2da5cfu, 0xff7c5b7du, 0x00000017u,
        // VS regs
        0x0fe000f1u, 0u, 0x000c0000u, 4u, 0u, 4u, 0u, 7u,
    };
    // clang-format on

    const auto shader_addr = uintptr_t(shader_code.data()); // Original address is 0xfe000f10
    const static u32 vs_regs[] = {
        u32(shader_addr >> 8), u32(shader_addr >> 40), 0xc0000u, 4, 0, 4, 0, 7};

    // Normally the driver will do a call to `sceGnmSetVsShader()`, but this function has
    // a check for zero in the upper part of shader address. In our case, the address is a
    // pointer to a stack memory, so the check will likely fail. To workaround it we will
    // repeat set shader functionality here as it is trivial.
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], vs_regs[1]); // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, vs_regs[2],
                                     vs_regs[3]); // SPI_SHADER_PGM_RSRC1_VS/SPI_SHADER_PGM_RSRC2_VS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x207u, vs_regs[6]); // PA_CL_VS_OUT_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b1u, vs_regs[4]); // SPI_VS_OUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c3u, vs_regs[5]); // SPI_SHADER_POS_FORMAT

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEsShader(u32* cmdbuf, u32 size, const u32* es_regs, u32 shader_modifier) {
    DrawTagOtherShader(cmdbuf, size, 2);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size < 0x14) {
        return -1;
    }

    if (!es_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (es_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? es_regs[2] : ((es_regs[2] & 0xfcfffc3f) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0xc8u, es_regs[0], 0u);  // SPI_SHADER_PGM_LO_ES
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0xcau, var, es_regs[3]); // SPI_SHADER_PGM_RSRC1_ES

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetGsRingSizes() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    DrawTagOtherShader(cmdbuf, size, 2);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size < 0x1d) {
        return -1;
    }

    if (!gs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (gs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x88u, gs_regs[0], 0u); // SPI_SHADER_PGM_LO_GS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x8au, gs_regs[2],
                                     gs_regs[3]); // SPI_SHADER_PGM_RSRC1_GS/SPI_SHADER_PGM_RSRC2_GS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2e5u, gs_regs[4]); // VGT_STRMOUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x29bu, gs_regs[5]); // VGT_GS_OUT_PRIM_TYPE
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2e4u, gs_regs[6]); // VGT_GS_INSTANCE_CNT

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 param4) {
    DrawTagOtherShader(cmdbuf, size, 2);
    LOG_TRACE(Lib_GnmDriver, "called");
    if (!cmdbuf || size < 0x1E) {
        return -1;
    }

    if (!hs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (hs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x108u, hs_regs[0], 0u); // SPI_SHADER_PGM_LO_HS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x10au, hs_regs[2],
                                     hs_regs[3]); // SPI_SHADER_PGM_RSRC1_HS/SPI_SHADER_PGM_RSRC2_HS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x286u,
                                          hs_regs[5],                  // VGT_HOS_MAX_TESS_LEVEL
                                          hs_regs[6]);                 // VGT_HOS_MIN_TESS_LEVEL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2dbu, hs_regs[4]); // VGT_TF_PARAM
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2d6u, param4);     // VGT_LS_HS_CONFIG

    // right padding?
    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetLsShader(u32* cmdbuf, u32 size, const u32* ls_regs, u32 shader_modifier) {
    DrawTagOtherShader(cmdbuf, size, 2);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size < 0x17) {
        return -1;
    }

    if (!ls_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    const auto modifier_mask = ((shader_modifier & 0xfffffc3f) == 0) ? 0xfffffc3f : 0xfcfffc3f;
    if (shader_modifier & modifier_mask) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (ls_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? ls_regs[2] : ((ls_regs[2] & modifier_mask) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x148u, ls_regs[0], 0u);  // SPI_SHADER_PGM_LO_LS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x14bu, ls_regs[3]);      // SPI_SHADER_PGM_RSRC2_LS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x14au, var, ls_regs[3]); // SPI_SHADER_PGM_RSRC1_LS

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetPsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    DrawTagOtherShader(cmdbuf, size, 1);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, 0u); // DB_SHADER_CONTROL

        WriteTrailingNop<0x20>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = PM4CmdSetData::SetContextReg(
            cmdbuf, 0x1c4u, ps_regs[4], ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                              ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetPsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (g_draw_tag || NR::g_mode) [[unlikely]] {
        DrawTagShaderChange(1, ps_regs);
        if (g_draw_gap) {
            GapShaderSet(1, ps_regs, cmdbuf, size);
        }
        if (NR::g_mode) {
            NativeRecShader(cmdbuf, size, true, 1, ps_regs);
        }
    }

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, 0u);  // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, 0xfu); // CB_SHADER_MASK

        WriteTrailingNop<0x1d>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = PM4CmdSetData::SetContextReg(
            cmdbuf, 0x1c4u, ps_regs[4], ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                              ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetResourceRegistrationUserMemory() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetResourceUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetSpiEnableSqCounters() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetSpiEnableSqCountersForUnitInstance() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetupMipStatsReport() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetVgtControl(u32* cmdbuf, u32 size, u32 prim_group_sz_minus_one,
                                     u32 partial_vs_wave_mode, u32 wd_switch_only_on_eop_mode) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size != 3 || (prim_group_sz_minus_one >= 0x100) ||
        ((wd_switch_only_on_eop_mode | partial_vs_wave_mode) >= 2)) {
        return -1;
    }

    if (sceKernelIsNeoMode()) {
        const u32 wd_switch_on_eop = u32(wd_switch_only_on_eop_mode != 0) << 0x14;
        const u32 switch_on_eoi = u32(wd_switch_only_on_eop_mode == 0) << 0x13;
        const u32 reg_value =
            wd_switch_only_on_eop_mode != 0
                ? (partial_vs_wave_mode & 1) << 0x10 | prim_group_sz_minus_one | wd_switch_on_eop |
                      switch_on_eoi | 0x40000u
                : prim_group_sz_minus_one & 0x1cffffu | wd_switch_on_eop | switch_on_eoi | 0x50000u;
        if (!UseNeoCompatSequences) {
            PM4CmdSetData::SetUconfigReg(cmdbuf, 0x40000258u,
                                         reg_value | 0x600000u); // IA_MULTI_VGT_PARAM
        } else {
            PM4CmdSetData::SetContextReg(cmdbuf, 0x100002aau, reg_value); // IA_MULTI_VGT_PARAM
        }
    } else {
        const u32 reg_value =
            ((partial_vs_wave_mode & 1) << 0x10) | (prim_group_sz_minus_one & 0xffffu);
        PM4CmdSetData::SetContextReg(cmdbuf, 0x2aau, reg_value); // IA_MULTI_VGT_PARAM
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (g_draw_tag || NR::g_mode) [[unlikely]] {
        DrawTagShaderChange(0, vs_regs);
        if (g_draw_gap) {
            GapShaderSet(0, vs_regs, cmdbuf, size);
        }
        if (NR::g_mode) {
            NativeRecShader(cmdbuf, size, true, 0, vs_regs);
        }
    }

    if (!cmdbuf || size <= 0x1c) {
        return -1;
    }

    if (!vs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (vs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var = shader_modifier == 0 ? vs_regs[2] : (vs_regs[2] & 0xfcfffc3f) | shader_modifier;
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], 0u);   // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, var, vs_regs[3]);  // SPI_SHADER_PGM_RSRC1_VS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x207u, vs_regs[6]); // PA_CL_VS_OUT_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b1u, vs_regs[4]); // SPI_VS_OUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c3u, vs_regs[5]); // SPI_SHADER_POS_FORMAT

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetWaveLimitMultiplier() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetWaveLimitMultipliers() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSpmEndSpm() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmInit() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmInit2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetDelay() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetMuxRam() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetMuxRam2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSelectCounter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSpmSelects() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSpmSelects2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmStartSpm() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttFini() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttFinishTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetBcInfo() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetGpuClocks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetHiWater() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetTraceCounter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetTraceWptr() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWrapCounts() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWrapCounts2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWritebackLabels() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttInit() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectMode() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectTarget() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectTokens() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetCuPerfMask() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetDceEventWrite() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetHiWater() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetTraceBuffer2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetTraceBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetUserdataTimer() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttStartTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttStopTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSwitchTraceBuffer() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSwitchTraceBuffer2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttWaitForEvent() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

static inline s32 PatchFlipRequest(u32* cmdbuf, u32 size, u32 vo_handle, u32 buf_idx, u32 flip_mode,
                                   s64 flip_arg, void* unk) {
    // check for `prepareFlip` packet
    cmdbuf += size - 64;
    ASSERT_MSG(cmdbuf[0] == 0xc03e1000, "Can't find `prepareFlip` packet");

    std::array<u32, 7> backup{};
    std::memcpy(backup.data(), cmdbuf, backup.size() * sizeof(decltype(backup)::value_type));

    ASSERT_MSG(((backup[2] & 3) == 0u) || (backup[1] != PM4CmdNop::PayloadType::PrepareFlipLabel),
               "Invalid flip packet");
    ASSERT_MSG(buf_idx != 0xffff'ffffu, "Invalid VO buffer index");

    const s32 flip_result = VideoOut::sceVideoOutSubmitEopFlip(vo_handle, buf_idx, flip_mode,
                                                               flip_arg, nullptr /*unk*/);
    if (flip_result != 0) {
        if (flip_result == 0x80290012) {
            LOG_ERROR(Lib_GnmDriver, "Flip queue is full");
            return 0x80d11081;
        } else {
            LOG_ERROR(Lib_GnmDriver, "Flip request failed");
            return flip_result;
        }
    }

    uintptr_t label_addr{};
    ASSERT_MSG(VideoOut::sceVideoOutGetBufferLabelAddress(vo_handle, &label_addr) == 16,
               "sceVideoOutGetBufferLabelAddress call failed");

    // Write event to lock the VO surface
    auto* write_lock = reinterpret_cast<PM4CmdWriteData*>(cmdbuf);
    write_lock->header = PM4Type3Header{PM4ItOpcode::WriteData, 3};
    write_lock->raw = 0x500u;
    const auto addr = (label_addr + buf_idx * sizeof(label_addr)) & ~0x3ull;
    write_lock->Address<uintptr_t>(addr);
    write_lock->data[0] = 1;

    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf + 5);

    if (backup[1] == PM4CmdNop::PayloadType::PrepareFlip) {
        nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x39};
        nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;
    } else {
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipLabel) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x34};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            // Write event to update label
            auto* write_label = reinterpret_cast<PM4CmdWriteData*>(cmdbuf + 0x3b);
            write_label->header = PM4Type3Header{PM4ItOpcode::WriteData, 3};
            write_label->raw = 0x500u;
            write_label->dst_addr_lo = backup[2] & 0xffff'fffcu;
            write_label->dst_addr_hi = backup[3];
            write_label->data[0] = backup[4];
        }
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipInterruptLabel) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x33};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            auto* write_eop = reinterpret_cast<PM4CmdEventWriteEop*>(cmdbuf + 0x3a);
            write_eop->header = PM4Type3Header{PM4ItOpcode::EventWriteEop, 4};
            write_eop->event_control = (backup[5] & 0x3f) + 0x500u + (backup[6] & 0x3f) * 0x1000;
            write_eop->address_lo = backup[2] & 0xffff'fffcu;
            write_eop->data_control = (backup[3] & 0xffffu) | 0x2200'0000u;
            write_eop->data_lo = backup[4];
            write_eop->data_hi = 0u;
        }
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipInterrupt) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x33};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            auto* write_eop = reinterpret_cast<PM4CmdEventWriteEop*>(cmdbuf + 0x3a);
            write_eop->header = PM4Type3Header{PM4ItOpcode::EventWriteEop, 4};
            write_eop->event_control = (backup[5] & 0x3f) + 0x500u + (backup[6] & 0x3f) * 0x1000;
            write_eop->address_lo = 0u;
            write_eop->data_control = 0x100'0000u;
            write_eop->data_lo = 0u;
            write_eop->data_hi = 0u;
        }
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffers(u32 count, u32* dcb_gpu_addrs[],
                                                   u32* dcb_sizes_in_bytes, u32* ccb_gpu_addrs[],
                                                   u32* ccb_sizes_in_bytes, u32 vo_handle,
                                                   u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    return sceGnmSubmitAndFlipCommandBuffersForWorkload(
        count, count, dcb_gpu_addrs, dcb_sizes_in_bytes, ccb_gpu_addrs, ccb_sizes_in_bytes,
        vo_handle, buf_idx, flip_mode, flip_arg);
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    u32 workload, u32 count, u32* dcb_gpu_addrs[], u32* dcb_sizes_in_bytes, u32* ccb_gpu_addrs[],
    u32* ccb_sizes_in_bytes, u32 vo_handle, u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    LOG_DEBUG(Lib_GnmDriver, "called [buf = {}]", buf_idx);

    auto* cmdbuf = dcb_gpu_addrs[count - 1];
    const auto size_dw = dcb_sizes_in_bytes[count - 1] / 4;

    const s32 patch_result =
        PatchFlipRequest(cmdbuf, size_dw, vo_handle, buf_idx, flip_mode, flip_arg, nullptr /*unk*/);
    if (patch_result != ORBIS_OK) {
        return patch_result;
    }

    return sceGnmSubmitCommandBuffers(count, const_cast<const u32**>(dcb_gpu_addrs),
                                      dcb_sizes_in_bytes, const_cast<const u32**>(ccb_gpu_addrs),
                                      ccb_sizes_in_bytes);
}

// SHADPS4_HOUSE_CENSUS=1 (also on with SHADPS4_HOUSE_SPLIT / SHADPS4_HOUSE_CHUNK; 2026-10-14): DCBs
// submitted per frame and the game's recording-context counter (CUSA00004: pool 0x800f1bcd0 +
// 0xdf0ef0, read at every submit, max per frame; the pool holds at most 254 contexts, no bounds
// check in the game). A frame ends at sceGnmSubmitDone. Logged every 600 frames.
namespace {
struct HouseCensus {
    std::mutex m;
    u32 dcbs = 0, ctx_max = 0, frames = 0, dcb_peak = 0, ctx_peak = 0;
    u64 dcb_sum = 0, ctx_sum = 0;
};
HouseCensus* HouseCensusOn() {
    static HouseCensus* hc = [] {
        const char* a = std::getenv("SHADPS4_HOUSE_SPLIT");
        const char* b = std::getenv("SHADPS4_HOUSE_CHUNK");
        const char* c = std::getenv("SHADPS4_HOUSE_CENSUS");
        const bool on = (a && (a[0] == '1' || a[0] == '2')) || (b && b[0] != '\0' && b[0] != '0') || (c && c[0] == '1');
        return on && MemoryPatcher::g_eboot_address != 0 &&
                       MemoryPatcher::g_game_serial == "CUSA00004"
                   ? new HouseCensus
                   : nullptr;
    }();
    return hc;
}
u32 HouseContextCounter() {
    return *reinterpret_cast<const volatile u32*>(MemoryPatcher::g_eboot_address + 0x1d0cbc0);
}
void HouseCensusSubmit(u32 count) {
    if (auto* hc = HouseCensusOn()) {
        const u32 ctx = HouseContextCounter();
        std::scoped_lock lk{hc->m};
        hc->dcbs += count;
        hc->ctx_max = std::max(hc->ctx_max, ctx);
    }
}
void HouseCensusFrame() {
    if (auto* hc = HouseCensusOn()) {
        const u32 ctx = HouseContextCounter();
        std::scoped_lock lk{hc->m};
        hc->ctx_max = std::max(hc->ctx_max, ctx);
        hc->dcb_sum += hc->dcbs;
        hc->ctx_sum += hc->ctx_max;
        hc->dcb_peak = std::max(hc->dcb_peak, hc->dcbs);
        hc->ctx_peak = std::max(hc->ctx_peak, hc->ctx_max);
        hc->dcbs = 0;
        hc->ctx_max = 0;
        if (++hc->frames == 600) {
            LOG_INFO(Lib_GnmDriver,
                     "HouseCensus: 600 frames | DCBs per frame mean {:.1f} max {} | game recording "
                     "contexts per frame (max of submit-time reads) mean {:.1f} max {} (pool limit 254)",
                     double(hc->dcb_sum) / 600.0, hc->dcb_peak, double(hc->ctx_sum) / 600.0,
                     hc->ctx_peak);
            hc->frames = 0;
            hc->dcb_sum = hc->ctx_sum = 0;
            hc->dcb_peak = hc->ctx_peak = 0;
        }
    }
}
} // namespace

int PS4_SYSV_ABI sceGnmSubmitCommandBuffersForWorkload(u32 workload, u32 count,
                                                       const u32* dcb_gpu_addrs[],
                                                       u32* dcb_sizes_in_bytes,
                                                       const u32* ccb_gpu_addrs[],
                                                       u32* ccb_sizes_in_bytes) {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "called");
    HouseCensusSubmit(count);

    if (!dcb_gpu_addrs || !dcb_sizes_in_bytes) {
        LOG_ERROR(Lib_GnmDriver, "dcbGpuAddrs and dcbSizesInBytes must not be NULL");
        return 0x80d11000;
    }

    for (u32 i = 0; i < count; i++) {
        if (dcb_sizes_in_bytes[i] == 0) {
            LOG_ERROR(Lib_GnmDriver, "Submitting a null DCB {}", i);
            return 0x80d11000;
        }
        if (dcb_sizes_in_bytes[i] > 0x3ffffc) {
            LOG_ERROR(Lib_GnmDriver, "dcbSizesInBytes[{}] ({}) is limited to (2*20)-1 DWORDS", i,
                      dcb_sizes_in_bytes[i]);
            return 0x80d11000;
        }
        if (ccb_sizes_in_bytes && ccb_sizes_in_bytes[i] > 0x3ffffc) {
            LOG_ERROR(Lib_GnmDriver, "ccbSizesInBytes[{}] ({}) is limited to (2*20)-1 DWORDS", i,
                      ccb_sizes_in_bytes[i]);
            return 0x80d11000;
        }
    }

    {
        Common::FaultMonitor::GnmWaitScope gm_wait{8};
        WaitGpuIdle();
    }

    if (DebugState.ShouldPauseInSubmit()) {
        DebugState.PauseGuestThreads();
    }

    if (send_init_packet) {
        if (sdk_version < Common::ElfInfo::FW_200) {
            liverpool->SubmitGfx(InitSequence, {});
        } else if (sdk_version < Common::ElfInfo::FW_400) {
            if (sceKernelIsNeoMode()) {
                if (!UseNeoCompatSequences) {
                    liverpool->SubmitGfx(InitSequence200Neo, {});
                } else {
                    liverpool->SubmitGfx(InitSequence200NeoCompat, {});
                }
            } else {
                liverpool->SubmitGfx(InitSequence200, {});
            }
        } else {
            if (sceKernelIsNeoMode()) {
                if (!UseNeoCompatSequences) {
                    liverpool->SubmitGfx(InitSequence350Neo, {});
                } else {
                    liverpool->SubmitGfx(InitSequence350NeoCompat, {});
                }
            } else {
                liverpool->SubmitGfx(InitSequence350, {});
            }
        }
        send_init_packet = false;
    }

    for (auto cbpair = 0u; cbpair < count; ++cbpair) {
        const auto* ccb = ccb_gpu_addrs ? ccb_gpu_addrs[cbpair] : nullptr;
        const auto ccb_size_in_bytes = ccb_sizes_in_bytes ? ccb_sizes_in_bytes[cbpair] : 0;

        const auto dcb_size_dw = dcb_sizes_in_bytes[cbpair] >> 2;
        const auto ccb_size_dw = ccb_size_in_bytes >> 2;

        const auto& dcb_span = std::span{dcb_gpu_addrs[cbpair], dcb_size_dw};
        const auto& ccb_span = std::span{ccb, ccb_size_dw};

        if (DebugState.DumpingCurrentFrame()) {
            static auto last_frame_num = -1LL;
            static u32 seq_num{};
            if (last_frame_num == frames_submitted && cbpair == 0) {
                ++seq_num;
            } else {
                last_frame_num = frames_submitted;
                seq_num = 0u;
            }

            using DebugStateType::QueueType;

            DebugState.PushQueueDump({
                .type = QueueType::dcb,
                .submit_num = seq_num,
                .num2 = cbpair,
                .data = {dcb_span.begin(), dcb_span.end()},
                .base_addr = reinterpret_cast<uintptr_t>(dcb_gpu_addrs[cbpair]),
            });
            DebugState.PushQueueDump({
                .type = QueueType::ccb,
                .submit_num = seq_num,
                .num2 = cbpair,
                .data = {ccb_span.begin(), ccb_span.end()},
                .base_addr = reinterpret_cast<uintptr_t>(ccb),
            });
        }
        liverpool->SubmitGfx(dcb_span, ccb_span);
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSubmitCommandBuffers(u32 count, const u32* dcb_gpu_addrs[],
                                            u32* dcb_sizes_in_bytes, const u32* ccb_gpu_addrs[],
                                            u32* ccb_sizes_in_bytes) {
    return sceGnmSubmitCommandBuffersForWorkload(count, count, dcb_gpu_addrs, dcb_sizes_in_bytes,
                                                 ccb_gpu_addrs, ccb_sizes_in_bytes);
}

int PS4_SYSV_ABI sceGnmSubmitDone() {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "called");
    HouseCensusFrame();
    if (Common::FaultMonitor::t_is_game_main) {
        // q48 SHADPS4_ARENA_READ_PREFETCH: the frame's particle chain is recorded (wait #2 passed).
        VideoCore::ArenaReadPrefetchFrameEnd();
    }
    {
        Common::FaultMonitor::GnmWaitScope gm_wait{7};
        WaitGpuIdle();
    }
    // 1.1 live screen percentage: a queued size goes into the game's mode switch here, between
    // two frames on Game:Main (memory_patcher.cpp OnSubmitDone).
    if (Common::FaultMonitor::t_is_game_main) {
        MemoryPatcher::OnSubmitDone();
    }
    if (SubmitLockFix()) {
        std::unique_lock lock{m_submission};
        if (!liverpool->IsGpuIdle()) {
            submission_lock = true;
        }
    } else {
        const u64 resets_before = submission_resets.load(std::memory_order_acquire);
        if (!liverpool->IsGpuIdle()) {
            submission_lock = true;
            if (submission_resets.load(std::memory_order_acquire) != resets_before) [[unlikely]] {
                // The GpuIdle reset ran between the idle check and the set: the lock may now stay set.
                LOG_WARNING(Lib_GnmDriver,
                            "SubmitLockRace: GpuIdle reset landed between the idle check and the lock "
                            "set (resets {} -> {}); lock now {}, gpu idle {} (1 and true = stuck)",
                            resets_before, submission_resets.load(), submission_lock,
                            liverpool->IsGpuIdle());
                if (Common::FenceWatch::Enabled()) {
                    Common::FenceWatch::Get().gnm_races.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    if (Common::FenceWatch::Enabled()) {
        Common::FenceWatch::Get().gnm_lock.store(submission_lock, std::memory_order_relaxed);
    }
    liverpool->SubmitDone();
    send_init_packet = true;
    ++frames_submitted;
    DebugState.IncGnmFrameNum();
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUnmapComputeQueue(u32 vqid) {
    liverpool->asc_queues.erase(Common::SlotId{vqid - 1});
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUnregisterAllResourcesForOwner() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmUnregisterOwnerAndResources() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmUnregisterResource() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmUpdateGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size < 0x1d) {
        return -1;
    }

    if (!gs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (gs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x88u, gs_regs[0], 0u); // SPI_SHADER_PGM_LO_GS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x8au, gs_regs[2],
                                     gs_regs[3]); // SPI_SHADER_PGM_RSRC1_GS/SPI_SHADER_PGM_RSRC2_GS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02e5u,
                                           gs_regs[4]);
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e029bu,
                                           gs_regs[5]);
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02e4u,
                                           gs_regs[6]);

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUpdateHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x1c) {
        return -1;
    }

    if (!hs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (hs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x108u, hs_regs[0],
                                     0u); // SPI_SHADER_PGM_LO_HS/SPI_SHADER_PGM_HI_HS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x10au, hs_regs[2],
                                     hs_regs[3]); // SPI_SHADER_PGM_RSRC1_HS/SPI_SHADER_PGM_RSRC1_LS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(
        cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0286u, hs_regs[5],
        hs_regs[6]); // VGT_HOS_MAX_TESS_LEVEL/VGT_HOS_MIN_TESS_LEVEL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02dbu,
                                           hs_regs[4]); // VGT_TF_PARAM update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02d6u,
                                           ls_hs_config); // VGT_LS_HS_CONFIG update

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdatePsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    DrawTagOtherShader(cmdbuf, size, 1);
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               0u); // DB_SHADER_CONTROL update
        WriteTrailingNop<0x20>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c4u, ps_regs[4],
            ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b3u, ps_regs[6],
            ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b6u,
                                               ps_regs[8]); // SPI_PS_IN_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b8u,
                                               ps_regs[9]); // SPI_BARYC_CNTL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               ps_regs[10]); // DB_SHADER_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e008fu,
                                               ps_regs[11]); // CB_SHADER_MASK update

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdatePsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (g_draw_tag || NR::g_mode) [[unlikely]] {
        DrawTagShaderChange(1, ps_regs);
        if (g_draw_gap) {
            GapShaderSet(1, ps_regs, cmdbuf, size);
        }
        if (NR::g_mode) {
            NativeRecShader(cmdbuf, size, ps_regs == nullptr, 1, ps_regs);
        }
    }

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               0u);                 // DB_SHADER_CONTROL update
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, 0xfu); // CB_SHADER_MASK

        WriteTrailingNop<0x1d>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c4u, ps_regs[4],
            ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b3u, ps_regs[6],
            ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b6u,
                                               ps_regs[8]); // SPI_PS_IN_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b8u,
                                               ps_regs[9]); // SPI_BARYC_CNTL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               ps_regs[10]); // DB_SHADER_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e008fu,
                                               ps_regs[11]); // CB_SHADER_MASK update

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdateVsShader(u32* cmdbuf, u32 size, const u32* vs_regs,
                                      u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (g_draw_tag || NR::g_mode) [[unlikely]] {
        DrawTagShaderChange(0, vs_regs);
        if (g_draw_gap) {
            GapShaderSet(0, vs_regs, cmdbuf, size);
        }
        if (NR::g_mode) {
            NativeRecShader(cmdbuf, size, vs_regs == nullptr, 0, vs_regs);
        }
    }

    if (!cmdbuf || size <= 0x1c) {
        return -1;
    }

    if (!vs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (vs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? vs_regs[2] : ((vs_regs[2] & 0xfcfffc3f) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], 0u);  // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, var, vs_regs[3]); // SPI_SHADER_PGM_RSRC1_VS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0207u,
                                           vs_regs[6]); // PA_CL_VS_OUT_CNTL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b1u,
                                           vs_regs[4]); // PA_CL_VS_OUT_CNTL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c3u,
                                           vs_regs[5]); // PA_CL_VS_OUT_CNTL update

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmValidateCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDisableDiagnostics() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDisableDiagnostics2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDispatchCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDrawCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetDiagnosticInfo() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetDiagnostics() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetVersion() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0;
}

bool PS4_SYSV_ABI sceGnmValidateOnSubmitEnabled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI sceGnmValidateResetState() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidationRegisterMemoryCheckCallback() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceRazorCaptureCommandBuffersOnlyImmediate() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureCommandBuffersOnlySinceLastFlip() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureImmediate() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureSinceLastFlip() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

bool PS4_SYSV_ABI sceRazorIsLoaded() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI Func_063D065A2D6359C3() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_0CABACAFB258429D() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_150CF336FC2E99A3() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_17CA687F9EE52D49() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_1870B89F759C6B45() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_26F9029EF68A955E() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_301E3DBBAB092DB0() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_30BAFE172AF17FEF() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_3E6A3E8203D95317() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_40FEEF0C6534C434() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_416B9079DE4CBACE() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_4774D83BB4DDBF9A() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_50678F1CCEEB9A00() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_54A2EC5FA4C62413() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_5A9C52C83138AE6B() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_5D22193A31EA1142() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_725A36DEBB60948D() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_8021A502FA61B9BB() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_9D002FE0FA40F0E6() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_9D297F36A7028B71() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_A2D7EC7A7BCF79B3() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_AA12A3CB8990854A() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_ADC8DDC005020BC6() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_B0A8688B679CB42D() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_B489020B5157A5FF() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_BADE7B4C199140DD() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D1511B9DCFFB3DD9() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D53446649B02E58E() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D8B6E8E28E1EF0A3() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D93D733A19DD7454() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_DE995443BC2A8317() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_DF6E9528150C23FF() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_ECB4C6BA41FE3350() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDebugModuleReset() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebugReset() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI Func_C4C328B7CF3B4171() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalCommand(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (sdk_version >= Common::ElfInfo::FW_400) {
        return sceGnmDrawInitToDefaultContextState400(cmdbuf, size);
    }
    return sceGnmDrawInitToDefaultContextState(cmdbuf, size);
}

int PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (sdk_version >= Common::ElfInfo::FW_400) {
        return 0x100;
    }
    return 0x20;
}

int PS4_SYSV_ABI sceGnmFindResources() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceRegistrationBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRegisterOwnerForSystem() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI Func_1C43886B16EE5530() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_81037019ECCD0E01() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_BFB41C057478F0BF() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_E51D44DB8151238C() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_F916890425496553() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LOG_INFO(Lib_GnmDriver, "Initializing presenter");
    liverpool = std::make_unique<AmdGpu::Liverpool>();
    presenter = std::make_unique<Vulkan::Presenter>(*g_window, liverpool.get());

    const s32 result = sceKernelGetCompiledSdkVersion(&sdk_version);
    if (result != ORBIS_OK) {
        sdk_version = 0;
    }

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        liverpool->ReserveCopyBufferSpace();
    }

    Platform::IrqC::Instance()->Register(Platform::InterruptId::GpuIdle, ResetSubmissionLock,
                                         nullptr);

    LIB_FUNCTION("b0xyllnVY-I", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmAddEqEvent);
    LIB_FUNCTION("b08AgtPlHPg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmAreSubmitsAllowed);
    LIB_FUNCTION("ihxrbsoSKWc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmBeginWorkload);
    LIB_FUNCTION("ffrNQOshows", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmComputeWaitOnAddress);
    LIB_FUNCTION("EJapNl2+pgU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmComputeWaitSemaphore);
    LIB_FUNCTION("5udAm+6boVg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmCreateWorkloadStream);
    LIB_FUNCTION("jwCEzr7uEP4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerGetAddressWatch);
    LIB_FUNCTION("PNf0G7gvFHQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerHaltWavefront);
    LIB_FUNCTION("nO-tMnaxJiE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebuggerReadGds);
    LIB_FUNCTION("t0HIQWnvK9E", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerReadSqIndirectRegister);
    LIB_FUNCTION("HsLtF4jKe48", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerResumeWavefront);
    LIB_FUNCTION("JRKSSV0YzwA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerResumeWavefrontCreation);
    LIB_FUNCTION("jpTMyYB8UBI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerSetAddressWatch);
    LIB_FUNCTION("MJG69Q7ti+s", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebuggerWriteGds);
    LIB_FUNCTION("PaFw9w6f808", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerWriteSqIndirectRegister);
    LIB_FUNCTION("qpGITzPE+Zc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebugHardwareStatus);
    LIB_FUNCTION("PVT+fuoS9gU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDeleteEqEvent);
    LIB_FUNCTION("UtObDRQiGbs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDestroyWorkloadStream);
    LIB_FUNCTION("bX5IbRvECXk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDingDong);
    LIB_FUNCTION("byXlqupd8cE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDingDongForWorkload);
    LIB_FUNCTION("HHo1BAljZO8", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDisableMipStatsReport);
    LIB_FUNCTION("0BzLGljcwBo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDispatchDirect);
    LIB_FUNCTION("Z43vKp5k7r0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDispatchIndirect);
    LIB_FUNCTION("wED4ZXCFJT0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDispatchIndirectOnMec);
    LIB_FUNCTION("nF6bFRUBRAU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDispatchInitDefaultHardwareState);
    LIB_FUNCTION("HlTPoZ-oY7Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndex);
    LIB_FUNCTION("GGsn7jMTxw4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexAuto);
    LIB_FUNCTION("ED9-Fjr8Ta4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexIndirect);
    LIB_FUNCTION("thbPcG7E7qk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexIndirectCountMulti);
    LIB_FUNCTION("5q95ravnueg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexIndirectMulti);
    LIB_FUNCTION("jHdPvIzlpKc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexMultiInstanced);
    LIB_FUNCTION("oYM+YzfCm2Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexOffset);
    LIB_FUNCTION("4v+otIIdjqg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndirect);
    LIB_FUNCTION("cUCo8OvArrw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndirectCountMulti);
    LIB_FUNCTION("f5QQLp9rzGk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndirectMulti);
    LIB_FUNCTION("Idffwf3yh8s", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState);
    LIB_FUNCTION("QhnyReteJ1M", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState175);
    LIB_FUNCTION("0H2vBYbTLHI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState200);
    LIB_FUNCTION("yb2cRhagD1I", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState350);
    LIB_FUNCTION("8lH54sfjfmU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextState);
    LIB_FUNCTION("im2ZuItabu4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextState400);
    LIB_FUNCTION("stDSYW2SBVM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawOpaqueAuto);
    LIB_FUNCTION("TLV4mswiZ4A", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverCaptureInProgress);
    LIB_FUNCTION("ODEeJ1GfDtE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterface);
    LIB_FUNCTION("4LSXsEKPTsE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForGpuDebugger);
    LIB_FUNCTION("MpncRjHNYRE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForGpuException);
    LIB_FUNCTION("EwjWGcIOgeM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForHDRScopes);
    LIB_FUNCTION("3EXdrVC7WFk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForReplay);
    LIB_FUNCTION("P9iKqxAGeck", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForResourceRegistration);
    LIB_FUNCTION("t-vIc5cTEzg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForValidation);
    LIB_FUNCTION("BvvO8Up88Zc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalVirtualQuery);
    LIB_FUNCTION("R6z1xM3pW-w", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverTraceInProgress);
    LIB_FUNCTION("d88anrgNoKY", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverTriggerCapture);
    LIB_FUNCTION("Fa3x75OOLRA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmEndWorkload);
    LIB_FUNCTION("4Mv9OXypBG8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmFindResourcesPublic);
    LIB_FUNCTION("iBt3Oe00Kvc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmFlushGarlic);
    LIB_FUNCTION("GviyYfFQIkc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetCoredumpAddress);
    LIB_FUNCTION("meiO-5ZCVIE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetCoredumpMode);
    LIB_FUNCTION("O-7nHKgcNSQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetCoredumpProtectionFaultTimestamp);
    LIB_FUNCTION("bSJFzejYrJI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetDbgGcHandle);
    LIB_FUNCTION("pd4C7da6sEg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetDebugTimestamp);
    LIB_FUNCTION("UoYY0DWMC0U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetEqEventType);
    LIB_FUNCTION("H7-fgvEutM0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetEqTimeStamp);
    LIB_FUNCTION("oL4hGI1PMpw", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetGpuBlockStatus);
    LIB_FUNCTION("Fwvh++m9IQI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetGpuCoreClockFrequency);
    LIB_FUNCTION("tZCSL5ulnB4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetGpuInfoStatus);
    LIB_FUNCTION("iFirFzgYsvw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetLastWaitedAddress);
    LIB_FUNCTION("KnldROUkWJY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetNumTcaUnits);
    LIB_FUNCTION("FFVZcCu3zWU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetOffChipTessellationBufferSize);
    LIB_FUNCTION("QJjPjlmPAL0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetOwnerName);
    LIB_FUNCTION("dewXw5roLs0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetPhysicalCounterFromVirtualized);
    LIB_FUNCTION("fzJdEihTFV4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetProtectionFaultTimeStamp);
    LIB_FUNCTION("4PKnYXOhcx4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetResourceBaseAddressAndSizeInBytes);
    LIB_FUNCTION("O0S96YnD04U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceName);
    LIB_FUNCTION("UBv7FkVfzcQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetResourceShaderGuid);
    LIB_FUNCTION("bdqdvIkLPIU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceType);
    LIB_FUNCTION("UoBuWAhKk7U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceUserData);
    LIB_FUNCTION("nEyFbYUloIM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetShaderProgramBaseAddress);
    LIB_FUNCTION("k7iGTvDQPLQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetShaderStatus);
    LIB_FUNCTION("ln33zjBrfjk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetTheTessellationFactorRingBufferBaseAddress);
    LIB_FUNCTION("QLdG7G-PBZo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGpuPaDebugEnter);
    LIB_FUNCTION("tVEdZe3wlbY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGpuPaDebugLeave);
    LIB_FUNCTION("NfvOrNzy6sk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertDingDongMarker);
    LIB_FUNCTION("7qZVNgEu+SY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertPopMarker);
    LIB_FUNCTION("aPIZJTXC+cU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertPushColorMarker);
    LIB_FUNCTION("W1Etj-jlW7Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertPushMarker);
    LIB_FUNCTION("aj3L-iaFmyk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertSetColorMarker);
    LIB_FUNCTION("jiItzS6+22g", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertSetMarker);
    LIB_FUNCTION("URDgJcXhQOs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertThreadTraceMarker);
    LIB_FUNCTION("1qXLHIpROPE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertWaitFlipDone);
    LIB_FUNCTION("HRyNHoAjb6E", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmIsCoredumpValid);
    LIB_FUNCTION("jg33rEKLfVs", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmIsUserPaEnabled);
    LIB_FUNCTION("26PM5Mzl8zc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalCuIndexToPhysicalCuIndex);
    LIB_FUNCTION("RU74kek-N0c", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalCuMaskToPhysicalCuMask);
    LIB_FUNCTION("Kl0Z3LH07QI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalTcaUnitToPhysical);
    LIB_FUNCTION("29oKvKXzEZo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmMapComputeQueue);
    LIB_FUNCTION("A+uGq+3KFtQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmMapComputeQueueWithPriority);
    LIB_FUNCTION("+N+wrSYBLIw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmPaDisableFlipCallbacks);
    LIB_FUNCTION("8WDA9RiXLaw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmPaEnableFlipCallbacks);
    LIB_FUNCTION("tNuT48mApTc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmPaHeartbeat);
    LIB_FUNCTION("6IMbpR7nTzA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmQueryResourceRegistrationUserMemoryRequirements);
    LIB_FUNCTION("+rJnw2e9O+0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRaiseUserExceptionEvent);
    LIB_FUNCTION("9Mv61HaMhfA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterGdsResource);
    LIB_FUNCTION("t7-VbMosbR4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRegisterGnmLiveCallbackConfig);
    LIB_FUNCTION("ZFqKFl23aMc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterOwner);
    LIB_FUNCTION("nvEwfYAImTs", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterResource);
    LIB_FUNCTION("gObODli-OH8", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestFlipAndSubmitDone);
    LIB_FUNCTION("6YRHhh5mHCs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestFlipAndSubmitDoneForWorkload);
    LIB_FUNCTION("f85orjx7qts", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestMipStatsReportAndReset);
    LIB_FUNCTION("MYRtYhojKdA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmResetVgtControl);
    LIB_FUNCTION("hS0MKPRdNr0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaClose);
    LIB_FUNCTION("31G6PB2oRYQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaConstFill);
    LIB_FUNCTION("Lg2isla2XeQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyLinear);
    LIB_FUNCTION("-Se2FY+UTsI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyTiled);
    LIB_FUNCTION("OlFgKnBsALE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyWindow);
    LIB_FUNCTION("LQQN0SwQv8c", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaFlush);
    LIB_FUNCTION("suUlSjWr7CE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaGetMinCmdSize);
    LIB_FUNCTION("5AtqyMgO7fM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaOpen);
    LIB_FUNCTION("KXltnCwEJHQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetCsShader);
    LIB_FUNCTION("Kx-h-nWQJ8A", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetCsShaderWithModifier);
    LIB_FUNCTION("X9Omw9dwv5M", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEmbeddedPsShader);
    LIB_FUNCTION("+AFvOEXrKJk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEmbeddedVsShader);
    LIB_FUNCTION("FUHG8sQ3R58", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEsShader);
    LIB_FUNCTION("jtkqXpAOY6w", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetGsRingSizes);
    LIB_FUNCTION("UJwNuMBcUAk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetGsShader);
    LIB_FUNCTION("VJNjFtqiF5w", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetHsShader);
    LIB_FUNCTION("vckdzbQ46SI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetLsShader);
    LIB_FUNCTION("bQVd5YzCal0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetPsShader);
    LIB_FUNCTION("5uFKckiJYRM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetPsShader350);
    LIB_FUNCTION("q-qhDxP67Hg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetResourceRegistrationUserMemory);
    LIB_FUNCTION("K3BKBBYKUSE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetResourceUserData);
    LIB_FUNCTION("0O3xxFaiObw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetSpiEnableSqCounters);
    LIB_FUNCTION("lN7Gk-p9u78", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetSpiEnableSqCountersForUnitInstance);
    LIB_FUNCTION("+xuDhxlWRPg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetupMipStatsReport);
    LIB_FUNCTION("cFCp0NX8wf0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetVgtControl);
    LIB_FUNCTION("gAhCn6UiU4Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetVsShader);
    LIB_FUNCTION("y+iI2lkX+qI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetWaveLimitMultiplier);
    LIB_FUNCTION("XiyzNZ9J4nQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetWaveLimitMultipliers);
    LIB_FUNCTION("kkn+iy-mhyg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmEndSpm);
    LIB_FUNCTION("aqhuK2Mj4X4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmInit);
    LIB_FUNCTION("KHpZ9hJo1c0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmInit2);
    LIB_FUNCTION("QEsMC+M3yjE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetDelay);
    LIB_FUNCTION("hljMAxTLNF0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetMuxRam);
    LIB_FUNCTION("bioGsp74SLM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetMuxRam2);
    LIB_FUNCTION("cMWWYeqQQlM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSelectCounter);
    LIB_FUNCTION("-zJi8Vb4Du4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSpmSelects);
    LIB_FUNCTION("xTsOqp-1bE4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSpmSelects2);
    LIB_FUNCTION("AmmYLcJGTl0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmStartSpm);
    LIB_FUNCTION("UHDiSFDxNao", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttFini);
    LIB_FUNCTION("a3tLC56vwug", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttFinishTrace);
    LIB_FUNCTION("L-owl1dSKKg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetBcInfo);
    LIB_FUNCTION("LQtzqghKQm4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetGpuClocks);
    LIB_FUNCTION("wYN5mmv6Ya8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetHiWater);
    LIB_FUNCTION("9X4SkENMS0M", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetStatus);
    LIB_FUNCTION("lbMccQM2iqc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetTraceCounter);
    LIB_FUNCTION("DYAC6JUeZvM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetTraceWptr);
    LIB_FUNCTION("pS2tjBxzJr4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetWrapCounts);
    LIB_FUNCTION("rXV8az6X+fM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetWrapCounts2);
    LIB_FUNCTION("ARS+TNLopyk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttGetWritebackLabels);
    LIB_FUNCTION("X6yCBYPP7HA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttInit);
    LIB_FUNCTION("2IJhUyK8moE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectMode);
    LIB_FUNCTION("QA5h6Gh3r60", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectTarget);
    LIB_FUNCTION("F5XJY1XHa3Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectTokens);
    LIB_FUNCTION("wJtaTpNZfH4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetCuPerfMask);
    LIB_FUNCTION("kY4dsQh+SH4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSetDceEventWrite);
    LIB_FUNCTION("7XRH1CIfNpI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetHiWater);
    LIB_FUNCTION("05YzC2r3hHo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetTraceBuffer2);
    LIB_FUNCTION("ASUric-2EnI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetTraceBuffers);
    LIB_FUNCTION("gPxYzPp2wlo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetUserData);
    LIB_FUNCTION("d-YcZX7SIQA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSetUserdataTimer);
    LIB_FUNCTION("ru8cb4he6O8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttStartTrace);
    LIB_FUNCTION("gVuGo1nBnG8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttStopTrace);
    LIB_FUNCTION("OpyolX6RwS0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSwitchTraceBuffer);
    LIB_FUNCTION("dl5u5eGBgNk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSwitchTraceBuffer2);
    LIB_FUNCTION("QLzOwOF0t+A", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttWaitForEvent);
    LIB_FUNCTION("xbxNatawohc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitAndFlipCommandBuffers);
    LIB_FUNCTION("Ga6r7H6Y0RI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitAndFlipCommandBuffersForWorkload);
    LIB_FUNCTION("zwY0YV91TTI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitCommandBuffers);
    LIB_FUNCTION("jRcI8VcgTz4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitCommandBuffersForWorkload);
    LIB_FUNCTION("yvZ73uQUqrk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSubmitDone);
    LIB_FUNCTION("ArSg-TGinhk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUnmapComputeQueue);
    LIB_FUNCTION("yhFCnaz5daw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmUnregisterAllResourcesForOwner);
    LIB_FUNCTION("fhKwCVVj9nk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmUnregisterOwnerAndResources);
    LIB_FUNCTION("k8EXkhIP+lM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUnregisterResource);
    LIB_FUNCTION("nLM2i2+65hA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateGsShader);
    LIB_FUNCTION("GNlx+y7xPdE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateHsShader);
    LIB_FUNCTION("4MgRw-bVNQU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdatePsShader);
    LIB_FUNCTION("mLVL7N7BVBg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdatePsShader350);
    LIB_FUNCTION("V31V01UiScY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateVsShader);
    LIB_FUNCTION("iCO804ZgzdA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateCommandBuffers);
    LIB_FUNCTION("SXw4dZEkgpA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDisableDiagnostics);
    LIB_FUNCTION("BgM3t3LvcNk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDisableDiagnostics2);
    LIB_FUNCTION("qGP74T5OWJc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDispatchCommandBuffers);
    LIB_FUNCTION("hsZPf1lON7E", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDrawCommandBuffers);
    LIB_FUNCTION("RX7XCNSaL6I", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateGetDiagnosticInfo);
    LIB_FUNCTION("5SHGNwLXBV4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateGetDiagnostics);
    LIB_FUNCTION("HzMN7ANqYEc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmValidateGetVersion);
    LIB_FUNCTION("rTIV11nMQuM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateOnSubmitEnabled);
    LIB_FUNCTION("MBMa6EFu4Ko", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmValidateResetState);
    LIB_FUNCTION("Q7t4VEYLafI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidationRegisterMemoryCheckCallback);
    LIB_FUNCTION("xeTLfxVIQO4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureCommandBuffersOnlyImmediate);
    LIB_FUNCTION("9thMn+uB1is", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureCommandBuffersOnlySinceLastFlip);
    LIB_FUNCTION("u9YKpRRHe-M", "libSceGnmDriver", 1, "libSceGnmDriver", sceRazorCaptureImmediate);
    LIB_FUNCTION("4UFagYlfuAM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureSinceLastFlip);
    LIB_FUNCTION("f33OrruQYbM", "libSceGnmDriver", 1, "libSceGnmDriver", sceRazorIsLoaded);
    LIB_FUNCTION("Bj0GWi1jWcM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_063D065A2D6359C3);
    LIB_FUNCTION("DKusr7JYQp0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_0CABACAFB258429D);
    LIB_FUNCTION("FQzzNvwumaM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_150CF336FC2E99A3);
    LIB_FUNCTION("F8pof57lLUk", "libSceGnmDriver", 1, "libSceGnmDriver", Func_17CA687F9EE52D49);
    LIB_FUNCTION("GHC4n3Wca0U", "libSceGnmDriver", 1, "libSceGnmDriver", Func_1870B89F759C6B45);
    LIB_FUNCTION("JvkCnvaKlV4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_26F9029EF68A955E);
    LIB_FUNCTION("MB49u6sJLbA", "libSceGnmDriver", 1, "libSceGnmDriver", Func_301E3DBBAB092DB0);
    LIB_FUNCTION("MLr+Fyrxf+8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_30BAFE172AF17FEF);
    LIB_FUNCTION("Pmo+ggPZUxc", "libSceGnmDriver", 1, "libSceGnmDriver", Func_3E6A3E8203D95317);
    LIB_FUNCTION("QP7vDGU0xDQ", "libSceGnmDriver", 1, "libSceGnmDriver", Func_40FEEF0C6534C434);
    LIB_FUNCTION("QWuQed5Mus4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_416B9079DE4CBACE);
    LIB_FUNCTION("R3TYO7Tdv5o", "libSceGnmDriver", 1, "libSceGnmDriver", Func_4774D83BB4DDBF9A);
    LIB_FUNCTION("UGePHM7rmgA", "libSceGnmDriver", 1, "libSceGnmDriver", Func_50678F1CCEEB9A00);
    LIB_FUNCTION("VKLsX6TGJBM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_54A2EC5FA4C62413);
    LIB_FUNCTION("WpxSyDE4rms", "libSceGnmDriver", 1, "libSceGnmDriver", Func_5A9C52C83138AE6B);
    LIB_FUNCTION("XSIZOjHqEUI", "libSceGnmDriver", 1, "libSceGnmDriver", Func_5D22193A31EA1142);
    LIB_FUNCTION("clo23rtglI0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_725A36DEBB60948D);
    LIB_FUNCTION("gCGlAvphubs", "libSceGnmDriver", 1, "libSceGnmDriver", Func_8021A502FA61B9BB);
    LIB_FUNCTION("nQAv4PpA8OY", "libSceGnmDriver", 1, "libSceGnmDriver", Func_9D002FE0FA40F0E6);
    LIB_FUNCTION("nSl-NqcCi3E", "libSceGnmDriver", 1, "libSceGnmDriver", Func_9D297F36A7028B71);
    LIB_FUNCTION("otfsenvPebM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_A2D7EC7A7BCF79B3);
    LIB_FUNCTION("qhKjy4mQhUo", "libSceGnmDriver", 1, "libSceGnmDriver", Func_AA12A3CB8990854A);
    LIB_FUNCTION("rcjdwAUCC8Y", "libSceGnmDriver", 1, "libSceGnmDriver", Func_ADC8DDC005020BC6);
    LIB_FUNCTION("sKhoi2ectC0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_B0A8688B679CB42D);
    LIB_FUNCTION("tIkCC1FXpf8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_B489020B5157A5FF);
    LIB_FUNCTION("ut57TBmRQN0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_BADE7B4C199140DD);
    LIB_FUNCTION("0VEbnc-7Pdk", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D1511B9DCFFB3DD9);
    LIB_FUNCTION("1TRGZJsC5Y4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D53446649B02E58E);
    LIB_FUNCTION("2Lbo4o4e8KM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D8B6E8E28E1EF0A3);
    LIB_FUNCTION("2T1zOhnddFQ", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D93D733A19DD7454);
    LIB_FUNCTION("3plUQ7wqgxc", "libSceGnmDriver", 1, "libSceGnmDriver", Func_DE995443BC2A8317);
    LIB_FUNCTION("326VKBUMI-8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_DF6E9528150C23FF);
    LIB_FUNCTION("7LTGukH+M1A", "libSceGnmDriver", 1, "libSceGnmDriver", Func_ECB4C6BA41FE3350);
    LIB_FUNCTION("dqPBvjFVpTA", "libSceGnmDebugModuleReset", 1, "libSceGnmDriver",
                 sceGnmDebugModuleReset);
    LIB_FUNCTION("RNPAItiMLIg", "libSceGnmDebugReset", 1, "libSceGnmDriver", sceGnmDebugReset);
    LIB_FUNCTION("xMMot887QXE", "libSceGnmDebugReset", 1, "libSceGnmDriver", Func_C4C328B7CF3B4171);
    LIB_FUNCTION("pF1HQjbmQJ0", "libSceGnmDriverCompat", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextStateInternalCommand);
    LIB_FUNCTION("jajhf-Gi3AI", "libSceGnmDriverCompat", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextStateInternalSize);
    LIB_FUNCTION("vbcR4Ken6AA", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmFindResources);
    LIB_FUNCTION("eLQbNsKeTkU", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmGetResourceRegistrationBuffers);
    LIB_FUNCTION("j6mSQs3UgaY", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmRegisterOwnerForSystem);
    LIB_FUNCTION("HEOIaxbuVTA", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_1C43886B16EE5530);
    LIB_FUNCTION("gQNwGezNDgE", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_81037019ECCD0E01);
    LIB_FUNCTION("v7QcBXR48L8", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_BFB41C057478F0BF);
    LIB_FUNCTION("5R1E24FRI4w", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_E51D44DB8151238C);
    LIB_FUNCTION("+RaJBCVJZVM", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_F916890425496553);
    LIB_FUNCTION("Fwvh++m9IQI", "libSceGnmGetGpuCoreClockFrequency", 1, "libSceGnmDriver",
                 sceGnmGetGpuCoreClockFrequency);
    LIB_FUNCTION("R3TYO7Tdv5o", "libSceGnmWaitFreeSubmit", 1, "libSceGnmDriver",
                 Func_4774D83BB4DDBF9A);
    LIB_FUNCTION("ut57TBmRQN0", "libSceGnmWaitFreeSubmit", 1, "libSceGnmDriver",
                 Func_BADE7B4C199140DD);
};

} // namespace Libraries::GnmDriver
