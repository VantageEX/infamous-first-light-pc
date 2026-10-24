// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/logging/log.h"

// The front's own helpers the architecture-A worker parser (lean_parse.*) must call so it behaves
// exactly as ProcessGraphics (notes/architecture-a-2026-10-10.md sec. 16). Defined in liverpool.cpp
// as thin wrappers over the file-local functions.

#include <array>
#include <atomic>
#include <span>

#include "video_core/amdgpu/pm4_cmds.h"

namespace Shader {
struct Info;
}
namespace Vulkan {
class GraphicsPipeline;
}
namespace AmdGpu {
/// NativeRec stage a2: publish (VS hash, PS hash, fetch shader) -> pipeline for the job threads'
/// derive (any thread; the tables are multi-writer). `vs_user_data` empty: the VS Info's own.
const Shader::Info* NativeRecPublishProgram(const Vulkan::GraphicsPipeline* pipeline,
                                            std::span<const u32> vs_user_data);
/// SetContextReg bumps the context epoch on a real value change only when this is true.
bool LeanViewportMemoEnabled();
/// A NOP whose body is back work (flip IRQ, guest markers): ProcessGraphics runs it through RunGfxPacket.
bool LeanNopHasBackEffect(const PM4Header* header, bool guest_markers_enabled);
/// After every packet ProcessGraphics calls BumpEpoch(opcode) unless this is true.
bool LeanKeepsSubmitEpoch(PM4ItOpcode opcode);

/// SHADPS4_LEAN: set on the lean worker threads (Rasterizer::ReadMemory / InvalidateMemory count
/// their faults instead of treating them as the front's).
inline thread_local bool t_lean_worker = false;
/// SHADPS4_LEAN_PARSE: 0 off, 1 L1 shadow compare, 2 L2 (workers parse the head spans for real).
/// (Not SHADPS4_LEAN: that is Common::Lean, the diagnostic skip, =1 in every arm.)
inline int LeanMode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_LEAN_PARSE");
        return v ? std::atoi(v) : 0;
    }();
    return mode;
}
/// Set by a fault on this lean worker; the parser stops its span before emitting (FaultFallback).
inline thread_local bool t_lean_fault = false;
struct LeanFaultStats {
    std::atomic<u64> fallback{};            ///< faults on lean workers (must read 0)
    std::array<std::atomic<u64>, 4> first{}; ///< their first addresses
    std::atomic<u32> n_first{};
    /// Pre-check stops: 0 SRT page read-protected, 1 SRT address unmapped, 2 V# page
    /// read-protected, 3 V# address unmapped, 4 walker code not understood, 5 flat buffer overflow.
    std::array<std::atomic<u64>, 8> precheck{};
    /// Shared-lock takes of the pipeline cache by lean workers, and the draws they were for
    /// (watch item: contention with the front's exclusive lock).
    std::atomic<u64> lock_takes{}, lock_draws{};
    /// PipelineMiss sub-reasons (Vulkan::GraphicsLookupScratch::Miss).
    std::array<std::atomic<u64>, 8> miss{};
};
inline LeanFaultStats g_lean_fault_stats;
/// SHADPS4_LEAN=2 (worker threads): the fault hook's wait for the back, the worker's context, and
/// whether the worker holds the pipeline cache's lock right now (then the hook must not wait).
inline thread_local void (*t_lean_l2_wait)(void*) = nullptr;
inline thread_local void* t_lean_l2_ctx = nullptr;
inline thread_local bool t_lean_in_lock = false;
/// The front holds the cache's exclusive lock (L2 PrepareDrawFront): a fault here is counted.
inline thread_local bool t_front_in_lock = false;
/// Back thread (L2): what to run while it waits for the cache's lock (ProcessCommands).
inline thread_local void (*t_lock_service)(void*) = nullptr;
inline thread_local void* t_lock_service_ctx = nullptr;
struct LeanL2Stats {
    std::atomic<u64> fault_waits{}, faults_in_lock{}, build_null{}, native_preps{}, colour_preps{},
        touches{}, wait_ticks{}, builds{}, build_refresh_fail{}, front_faults_in_lock{},
        regs_only_runs{}, dropped_draws{}, lock_service_waits{};
    std::atomic<u64> first_in_lock{};
};
inline LeanL2Stats g_lean_l2;
/// Lean worker phases (TSC ticks, summed over the workers), cache hits, memo verify.
struct LeanPhaseStats {
    std::atomic<u64> claim_wait{}, seed{}, parse{}, prewalk{}, lookup{}, derive{}, emit{}, draws{};
    std::atomic<u64> emit_commit{}, emit_table{}, table_skips{};
    std::atomic<u64> hdr_hit{}, hdr_miss{}, fetch_hit{}, fetch_miss{};
    /// The prewalk split (ticks) and its lock takes (counts): mapped-interval lookups and
    /// ProgramInfoFor calls (both only on a per-worker cache miss).
    std::atomic<u64> pw_header{}, pw_info{}, pw_walk{}, pw_fetch{}, map_lock{}, info_lock{};
    std::atomic<u64> memo_verify{}, memo_verify_bad{}, memo_ineligible{};
    std::array<std::atomic<u64>, 8> memo_hit{}, memo_miss{}; ///< per worker
};
inline LeanPhaseStats g_lean_phase;
inline void LeanNoteWorkerFault(u64 addr) {
    if (t_lean_l2_wait) [[likely]] {
        if (t_lean_in_lock) {
            if (g_lean_l2.faults_in_lock.fetch_add(1, std::memory_order_relaxed) == 0) {
                LOG_CRITICAL(Render, "Lean L2: worker fault while holding the pipeline cache lock "
                                     "at {:#x} (completes without the back wait)", addr);
            }
            u64 none = 0;
            g_lean_l2.first_in_lock.compare_exchange_strong(none, addr);
            return; // no wait: the back may be waiting for the lock this worker holds
        }
        g_lean_l2.fault_waits.fetch_add(1, std::memory_order_relaxed);
        t_lean_l2_wait(t_lean_l2_ctx); // publish, then wait for the back up to this position
        return;
    }
    auto& s = g_lean_fault_stats;
    s.fallback.fetch_add(1, std::memory_order_relaxed);
    const u32 i = s.n_first.fetch_add(1, std::memory_order_relaxed);
    if (i < s.first.size()) {
        s.first[i].store(addr, std::memory_order_relaxed);
    }
    t_lean_fault = true;
}
} // namespace AmdGpu
