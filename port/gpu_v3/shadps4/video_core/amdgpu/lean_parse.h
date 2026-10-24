// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Architecture A, piece 2 (notes/NATIVE-RENDERER.md sec. 9; notes/architecture-a-2026-10-10.md
// sec. 12-16): the lean worker parser. A worker thread parses the START of one of the game's
// top-level graphics command buffers (PM4 type-3 packets) and emits, into its own CpSplit
// channel, exactly the records Liverpool::ProcessGraphics would emit for those packets in split
// mode (register words, CB/DB extents, ClearState, epoch bumps, the no-op DMA packet record and
// posted draw records). It stops BEFORE the first packet it does not support or cannot reproduce
// exactly; that packet and everything after it is parsed by the existing front from the chained
// state. A stop therefore leaves the ChainState and the channel exactly as the front's state and
// records would be just before that packet.

#include <array>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_shader.h"

namespace Shader {
struct Info;
}

namespace Vulkan {
class Rasterizer;
class PipelineCache;
struct GraphicsLookupScratch;
} // namespace Vulkan

namespace AmdGpu::Lean {

/// The front state the supported packets read or write, carried from one span to the next
/// (worker n publishes it for worker n + 1). Each field mirrors one Liverpool member:
struct ChainState {
    std::unique_ptr<Regs> regs;                              ///< Liverpool::regs (full file)
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};   ///< Liverpool::last_cb_extent
    CbDbExtent db_extent{};                                  ///< Liverpool::last_db_extent
    u64 ctx_epoch{1};                                        ///< Liverpool::context_change_epoch
    ComputeProgram cs_state{};                               ///< mapped_queues[Gfx].cs_state
    VAddr indirect_args_addr{};                              ///< Liverpool::indirect_args_addr
    /// FRONT_DERIVE vertex-input description id (Liverpool::front_vi_desc_id). The caller seeds
    /// it with a range disjoint from the front's and never moves it backwards; the parser bumps
    /// it when a draw's description differs from the previous one THIS CHAIN derived
    /// (Liverpool::front_vi_prev / _count below, carried across spans like the front carries its
    /// own across command buffers; a fresh chain starts with "no previous": the first derived
    /// draw bumps). A repeated id can never name a different description: both counters bump
    /// monotonically in disjoint ranges.
    u64 vi_desc_id{0};
    std::array<CpSplit::VertexInputDesc, CpSplit::MaxVertexInputs> vi_prev{};
    u32 vi_prev_count{~0u};
};

/// What the parser needs of the front's configuration (set once per worker from Liverpool's
/// members of the same name and the renderer). Flags that are inline globals (VideoCore::
/// g_native_pass / g_native_colour / g_epoch_defer_upload / g_native_coherent, NativeRec::g_mode /
/// g_segb, SplitHazard::Enabled) are read directly by the parser.
struct Env {
    bool l2{}; ///< SHADPS4_LEAN=2: resolve runtime cases in place (no runtime stops)
    Vulkan::Rasterizer* rasterizer{};
    Vulkan::PipelineCache* pipeline_cache{};
    bool split_enabled{};      ///< Liverpool::split_enabled (must be true)
    bool split_run_ahead{};    ///< Liverpool::split_run_ahead (must be true: depth 0 syncs draws)
    bool split_verify{};       ///< Liverpool::split_verify (must be false)
    bool split_post_dma{};     ///< Liverpool::split_post_dma (else the no-op DMA syncs: stop)
    u32 split_task_guard{};    ///< Liverpool::split_task_guard (NotePostedWrite is front-only)
    int split_front_derive{};  ///< Liverpool::split_front_derive
    bool host_markers{};       ///< ProcessGraphics' host_markers_enabled
    bool guest_markers{};      ///< ProcessGraphics' guest_markers_enabled
    /// liverpool.cpp file-local diagnostics the parser cannot see: SHADPS4_DRAW_TAG=1
    /// (g_draw_tag_front writes SplitDrawRecord::pad). The dispatch-run diagnostics are read from
    /// their environment variables by the parser.
    bool draw_tag_front{};
};

/// Why Parse stopped (one per distinct reason; the stop packet is at SpanResult::end_dw).
enum class Stop : u32 {
    None,                 ///< internal: no stop
    Cut,                  ///< the next packet starts at or crosses cut_dw
    End,                  ///< the command buffer ended (end_dw == dcb.size())
    Unsupported,          ///< a packet the parser does not mirror (type 0 / 1, other opcodes, a
                          ///< non-no-op DMA_DATA, a register write outside the register file,
                          ///< an SH write that would overflow cs_state)
    Overrun,              ///< a packet, or a read past it the front makes (NOP hints, draw tag
                          ///< words), exceeds the command buffer: the front logs / asserts
    HintMissing,          ///< a CB/DB extent NOP hint word is not 0xc0001000 (front asserts)
    BadSetBase,           ///< SET_BASE with a base other than the indirect patch table (asserts)
    NopBackEffect,        ///< a NOP with back work (flip IRQ, guest markers): RunGfxPacket syncs
    DmaSync,              ///< no-op DMA without POST_DMA: the front syncs on it
    DmaGuard,             ///< no-op DMA overlapping the rest of the buffer (sync) or with
                          ///< TASK_GUARD on (the posted-write log is front-only)
    PipelineMiss,         ///< LookupGraphics returned null (the front may build / compile)
    NativeUnchecked,      ///< NATIVE_PASS: the front would run PrepareNativeVariant
    ColourVariantMissing, ///< NATIVE_COLOUR: the front would decide / claim a colour variant
    Config,               ///< the whole parser is off for this configuration (VERIFY, depth 0,
                          ///< SEGB, DRAW_TAG, dispatch-run diagnostics, hazard stats, no
                          ///< rasterizer / pipeline cache / split)
    Fault,                ///< a guest read the pre-check found read-protected / unmapped (or a
                          ///< walker it could not interpret): the front parses from here
    FaultFallback,        ///< a fault happened anyway (racing protect; counted, must be 0)
    NonVsDraw,            ///< a draw packet with a stage setup other than plain Vs (static)
    Count,
};

const char* StopName(Stop stop);

/// What StaticCheck needs besides the packet and the registers.
struct StaticEnv {
    bool guest_markers{};
    bool split_post_dma{};
};
/// Every stop decidable from the packet words of `dcb` at `off` and the registers BEFORE the
/// packet (L2: the chain's cut and the worker's static stops are this one function). Returns
/// Stop::None for a packet the worker parses (type-2 padding included); `opcode` = its opcode.
Stop StaticCheck(std::span<const u32> dcb, u32 off, const Regs& regs, const StaticEnv& env,
                 u32& opcode);
/// True for the stops StaticCheck can return (a worker stop with one of these = parity break).
bool IsStaticStop(Stop stop);

struct SpanResult {
    u32 packets{};     ///< packets fully parsed (type-2 padding included)
    u32 draws{};       ///< draw records emitted
    u32 end_dw{};      ///< dword offset of the first packet NOT parsed (the front resumes here)
    Stop stop{};
    u32 stop_opcode{}; ///< PM4ItOpcode of the stop packet (0 for Cut / End)
    u32 dropped{};     ///< register-only mode: draws not emitted
    u32 run_bumps{};   ///< GUEST_DISPATCH_ORDER: parsed packets that end a dispatch run
                       ///< (not RunNeutral); L2 hand-back: one fresh gfx run if > 0
};

/// One per worker thread: owns the pipeline-lookup scratch and the vertex-input scratch.
class Parser {
public:
    explicit Parser(const Env& env);
    ~Parser();

    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    /// Parses packets from dcb[0] while the packet's dword offset is below cut_dw, applying them
    /// to `st` and emitting the front's records into `out` (the worker's own channel; this thread
    /// is its only producer). Never waits on the back except through the channel's own ring
    /// space / run-ahead depth, as the front does. Returns where and why it stopped.
    SpanResult Parse(std::span<const u32> dcb, u32 cut_dw, ChainState& st, CpSplit::Channel& out);

    const Env& GetEnv() const noexcept {
        return env;
    }
    /// L2 early stop: the next Parse starts at `start_dw` and emits register words, extents and
    /// ClearState only (draws dropped and counted) - the back's mirror reaches the cut state.
    void SetRegsOnly(bool on, u32 start_dw) {
        regs_only = on;
        start = on ? start_dw : 0;
    }
    /// L2 front: touch the pages the front's own lookup of this draw reads (shader headers, SRT
    /// tables, fetch code, fetch V#s), so its faults happen before it takes the cache's lock.
    void Pretouch(const Regs& regs);
    /// Adds this parser's phase ticks / cache counts to g_lean_phase (worker slot `w`), clears.
    void FlushPhases(u32 w);
    /// Rasterizer::IsMapped without its lock (per-worker interval cache, map generation).
    bool MappedFast(VAddr addr, u64 size) const;

private:
    /// SHADPS4_LEAN: shader headers + checked SRT walks into the scratch, outside the cache's
    /// lock (sec. 16). False: a pre-check stop (Stop::Fault).
    bool Prewalk(const Regs& regs);
    /// The checked interpreter of a generated SRT walker (RefreshFlatBuf's walk).
    bool CheckedWalk(const Shader::Info& info, const u32* user_data, std::vector<u32>& flat);
    /// IsMapped + AnyReadProtected for one guest read; `kind` 0 SRT, 2 V#.
    bool ReadOk(VAddr addr, u64 size, u32 kind) const;
    /// L2, after a build: the captured Fragment / Vertex slots' per-draw fields from the worker's
    /// registers (user data, pgm_base, checked SRT walk). False: a walk read failed.
    bool RefreshCaptured(const Regs& regs);
    /// The shader header search (AmdGpu::SearchBinaryInfo) page-checked; kind 6 / 7.
    bool HeaderOk(VAddr code, const AmdGpu::BinaryInfo*& found) const;
    /// Gcn::ParseFetchShader's reads page-checked: the fetch code (decoded to S_SETPC_B64) and
    /// the V# of every attribute (Info::ReadUdReg); kinds 6 / 7 (code), 2 / 3 (V#).
    bool FetchOk(const Shader::Info& info, std::span<const u32> user_data) const;
    struct WalkOp {
        u8 op;
        s32 disp;
        u64 imm;
    };
    std::unordered_map<const void*, std::vector<WalkOp>> walk_plans; ///< per walker code
    StaticEnv static_env{};
    struct Caches;
    std::unique_ptr<Caches> caches;
    bool regs_only{};
    u32 start{};
    struct DrawPlan;
    Stop DecideDraw(const PM4Header* header, std::span<const u32> dcb, u32 off,
                    const char* name, const CpSplit::DrawArgs& args, const ChainState& st,
                    DrawPlan& plan);
    void EmitDraw(const PM4Header* header, const DrawPlan& plan, ChainState& st,
                  CpSplit::Channel& out);

    Env env;
    Stop config_stop{Stop::None}; ///< Stop::Config when the configuration is not mirrorable
    bool viewport_memo{};         ///< LeanViewportMemoEnabled()
    bool epoch_records{};         ///< BumpEpoch emits EpochBump records (defer-upload / coherent)
    std::unique_ptr<Vulkan::GraphicsLookupScratch> scratch;
    std::unique_ptr<CpSplit::VertexInputsDerived> vi;
};

} // namespace AmdGpu::Lean
