// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Architecture A, piece 2: the lean worker parser (lean_parse.h). Every handler below is a
// line-for-line mirror of the matching case of Liverpool::ProcessGraphics (liverpool.cpp, the
// type-3 switch) and of Liverpool::EmitDraw / Rasterizer::PrepareDrawFront in split run-ahead
// mode, minus the front's diagnostics. Where the front's path would wait on the back, assert,
// or touch front-only state, the parser stops BEFORE the packet instead (Stop::*), leaving the
// chain state and the channel exactly as the front's would be before that packet.
//
// ---- Thread safety: everything non-local this code reads while the front keeps running -----
//  * ChainState `st`: owned by the caller, handed from the worker of command buffer n to the
//    worker of n + 1 in stream order; one thread at a time. Its register file is a private copy
//    (the front's `regs` is never read). ASC SET_SH_REG writes land in the FRONT's register file
//    (ProcessCompute, liverpool.cpp ~8470), not here: the chain owner must carry them, measured 0
//    while a gfx DCB is open in the city (architecture-a sec. 14, s510).
//  * CpSplit::Channel `out`: the worker's own channel; this thread is its single producer (the
//    channel's producer-side state is unsynchronised by contract). Reserve waits for ring space
//    and PostRunAhead for run-ahead depth exactly as the front's channel does; nothing else waits.
//  * The command buffer: guest memory the guest may not reuse until the submission retires, read
//    by the front the same way. Reads past a packet that the front makes (CB/DB extent NOP hints,
//    the DRAW_INDEX_2 trailing job tag) are bounds-checked against the span here (Stop::Overrun).
//  * Vulkan::PipelineCache::LookupGraphics (piece 1): the ONLY pipeline-cache entry point used.
//    It holds the cache's shared lock for the lookup, reads program_cache / graphics_pipelines
//    and each program's immutable Info half, and writes only the caller's scratch (private Info
//    views, user data and flat buffer copies). Never LookupOrBuildGraphics / PrepareNative* -
//    those write the shared Infos and member state and are left to the front (Stop::PipelineMiss,
//    Stop::NativeUnchecked, Stop::ColourVariantMissing).
//  * GraphicsPipeline fields: GetFetchShader / GetStages / key are immutable after construction.
//    `native_checked` and `native_colour_static` are plain `mutable` bytes the front writes ONCE
//    under the cache's exclusive lock (false -> true, 0 -> 1 or 2); this thread reads them without
//    the lock. A stale read only makes the parser stop where the front would already have moved
//    on (conservative); it can never make it proceed where the front would write. The colour
//    slot keys are atomics (acquire).
//  * Guest memory behind user data: LookupGraphics walks SRT tables (flat buffer) and
//    DeriveVertexInputs reads one V# per fetch attribute through the scratch's user data copy -
//    at parse time, exactly as the front's lookup and FRONT_DERIVE do today (the run-ahead note in
//    RunGfxDma: measured 0 hazards against DMA CPU-path writes, nested tables unmeasured). OPEN
//    RISK for the engineer: a read fault taken here hits the fault handlers, which key their drain
//    on CpSplit::t_is_front; a thread that is neither front nor back is new to them.
//  * Shader::Info immutable half (ud_mask, buffers, images, srt_info, ...): read by the lookup
//    and the derivation; never written after a program is published.
//  * Rasterizer::ClassifyDraw / SkipEmptyDrawRegs / NativeColourVariantReady: pure functions of
//    their arguments plus read-once environment statics.
//  * Globals: VideoCore::g_native_pass / g_native_colour / g_epoch_defer_upload /
//    g_native_coherent, NativeRec::g_mode / g_segb, SplitHazard::Enabled(): inline const or
//    static-init read-once values. LeanViewportMemoEnabled (static bool), LeanKeepsSubmitEpoch
//    (constexpr), LeanNopHasBackEffect (pure over the packet).
//  * NOT touched: any Liverpool member (regs, last_cb/db_extent, context_change_epoch,
//    front_vi_*, cs_state, indirect_args_addr, the posted-write log / span registry), DebugState,
//    NativeRec run state (g_nr_run) and its verify (NativeRecFront / Head / Member), SegB,
//    SplitHazard, the front's counters (regs_generation*, EMPTY_DRAWS, DrawTagFront, FrontCost,
//    DispatchSync) and the program prefetch hint (PrefetchProgramsOnWrite).

#include <algorithm>
#include <bit>
#include <cstddef>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "common/assert.h"
#include "common/decoder.h"
#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/prepared_graphics_state.h"
#include "shader_recompiler/frontend/decode.h"
#include "video_core/amdgpu/lean_front_api.h"
#include "video_core/amdgpu/lean_parse.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/amdgpu/split_records.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/submit_epoch.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu::Lean {

namespace {

using Vulkan::Rasterizer;

/// GetStages() of a graphics pipeline spans the logical stages before Compute.
constexpr u32 GraphicsStageCount = Shader::MaxStageTypes - 1;

/// The NOP header the guest places after a CB / DB register block (PM4 NOP, count 0).
constexpr u32 ExtentHintHeader = 0xc0001000u;

bool EnvFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

/// SHADPS4_LEAN_PERDRAW=1 (2026-10-15; also read by liverpool.cpp and vk_a2.cpp): the worker's
/// lookup memo becomes 4-way set-associative (256 sets, LRU; one render state may keep several
/// specializations) and its program-Info and fetch-shader caches grow 8x - fewer LookupGraphics /
/// ProgramInfoFor calls (shared lock) and fetch-shader decodes per draw. Every hit keeps today's
/// validation (memo: full state + epoch + both guards; Info: the hash; fetch: the code words).
const bool g_lean_perdraw = [] { // bit 1: the bigger worker memos (the value is a decimal bit mask)
    const char* v = std::getenv("SHADPS4_LEAN_PERDRAW");
    return v != nullptr && ((std::strtoul(v, nullptr, 10) & 1) != 0);
}();

/// Liverpool::JournalField: the field's word offset inside the register file.
template <typename T>
u32 FieldWordOffset(const Regs& regs, const T& field) {
    const auto bytes = reinterpret_cast<const u8*>(&field) - reinterpret_cast<const u8*>(&regs);
    return static_cast<u32>(bytes / sizeof(u32));
}

template <typename T>
void JournalField(CpSplit::Channel& out, const Regs& regs, const T& field) {
    static_assert(sizeof(T) % sizeof(u32) == 0);
    const u32 off = FieldWordOffset(regs, field);
    out.EmitRegWrite(off, &regs.reg_array[off], static_cast<u32>(sizeof(T) / sizeof(u32)));
}

/// Liverpool::JournalExtent (split mode).
void JournalExtent(CpSplit::Channel& out, const ChainState& st, u32 reg_addr) {
    using CR = Liverpool;
    switch (reg_addr) {
    case CR::CbColor0Base:
    case CR::CbColor1Base:
    case CR::CbColor2Base:
    case CR::CbColor3Base:
    case CR::CbColor4Base:
    case CR::CbColor5Base:
    case CR::CbColor6Base:
    case CR::CbColor7Base: {
        const u32 cb = (reg_addr - CR::CbColor0Base) / (CR::CbColor1Base - CR::CbColor0Base);
        out.EmitExtent(cb, st.cb_extent[cb].raw);
        break;
    }
    case CR::CbColor0Cmask:
    case CR::CbColor1Cmask:
    case CR::CbColor2Cmask:
    case CR::CbColor3Cmask:
    case CR::CbColor4Cmask:
    case CR::CbColor5Cmask:
    case CR::CbColor6Cmask:
    case CR::CbColor7Cmask: {
        const u32 cb = (reg_addr - CR::CbColor0Cmask) / (CR::CbColor1Cmask - CR::CbColor0Cmask);
        out.EmitExtent(cb, st.cb_extent[cb].raw);
        break;
    }
    case CR::DbZInfo:
        out.EmitExtent(NUM_COLOR_BUFFERS, st.db_extent.raw);
        break;
    default:
        break;
    }
}

constexpr const char* StopNames[] = {
    "none",          "cut",           "end",       "unsupported",    "overrun",
    "hint-missing",  "bad-set-base",  "nop-back",  "dma-sync",       "dma-guard",
    "pipeline-miss", "native-unchecked", "colour-variant-missing", "config",
    "fault",         "fault-fallback", "non-vs-draw",
};
static_assert(std::size(StopNames) == static_cast<size_t>(Stop::Count));

} // namespace

const char* StopName(Stop stop) {
    const auto i = static_cast<size_t>(stop);
    return i < std::size(StopNames) ? StopNames[i] : "?";
}

/// What DecideDraw settles before anything is emitted (EmitDraw then writes it).
struct Parser::DrawPlan {
    Rasterizer::SplitDraw draw{};
    const char* name{};
    bool has_stages{};
    bool vi_same{}; ///< FRONT_DERIVE: the description equals st.vi_prev (no desc-id bump)
    u32 bytes{};
    u32 vi_bytes{};
    u32 job_tag{};
    u32 job_idx{};
};

struct Parser::Caches {
    // Page state, this draw only (Prewalk clears it).
    std::array<u64, 64> ok_pages{};
    u32 n_ok{};
    // GPU-mapped intervals, valid while Rasterizer::MappedGen() equals span_gen (any map / unmap
    // clears them). Mapping only - the read-protection state stays per page per draw (ok_pages).
    struct Span {
        VAddr lo{}, hi{};
    };
    std::array<Span, 16> spans{};
    u32 n_spans{}, next_span{};
    u64 span_gen{~u64{0}};
    // Program Infos by shader hash (non-null ProgramInfoFor results; graphics programs are never
    // erased from the program cache and a retired one stays alive in retired_programs).
    // The three tables below are vectors sized once in the constructor (powers of two; indexed
    // with size() - 1 as the mask): 256 / 64 / 128 entries, SHADPS4_LEAN_PERDRAW 2048 / 512 / 1024.
    struct InfoEnt {
        u64 hash{};
        const Shader::Info* info{};
    };
    std::vector<InfoEnt> info;
    struct Hdr {
        VAddr code{};
        u32 off{};
        std::array<u8, sizeof(AmdGpu::BinaryInfo)> copy{};
    };
    std::array<Hdr, 256> hdr{};
    struct Fetch {
        VAddr code{};
        std::vector<u32> words;
        std::vector<std::pair<u32, u32>> attrs; ///< (V# pointer sgpr, dword offset)
    };
    std::vector<Fetch> fetch;
    struct Memo {
        bool valid{};
        Vulkan::PreparedGraphicsState state{};
        u64 epoch{};
        const Vulkan::GraphicsPipeline* pipeline{};
        const Shader::Gcn::FetchShaderData* fetch{};
        Shader::PreparedSpecialization fs_guard, vs_guard;
        std::unique_ptr<Shader::Info> fs_info, vs_info; ///< private copies (MatchesInputs)
        const Shader::Info* fs_shared{};
        const Shader::Info* vs_shared{};
        vk::ShaderModule fs_module{}, vs_module{};
        u64 last_use{}; ///< SHADPS4_LEAN_PERDRAW: memo_clock at the last hit or store (LRU way)
    };
    /// Direct-mapped (128), or SHADPS4_LEAN_PERDRAW: MemoSets sets of MemoWays ways.
    static constexpr u32 MemoSets = 256, MemoWays = 4;
    std::vector<Memo> memo;
    u64 memo_clock{};
    u64 memo_hits_total{};
    // Phase ticks / counts since the last FlushPhases.
    u64 prewalk{}, lookup{}, derive{}, emit{}, draws{};
    u64 emit_commit{}, emit_table{}, table_skips{};
    // Program-table publishes this worker already made: (pipeline, fetch shader address).
    struct PubMemo {
        const void* pipeline{};
        u64 fetch{};
    };
    std::array<PubMemo, 256> pub_memo{};
    u64 pw_header{}, pw_info{}, pw_walk{}, pw_fetch{}, map_lock{}, info_lock{};
    u64 memo_hit{}, memo_miss{}, hdr_hit{}, hdr_miss{}, fetch_hit{}, fetch_miss{}, inelig{};
};

void Parser::FlushPhases(u32 w) {
    auto& c = *caches;
    auto& g = g_lean_phase;
    g.prewalk += std::exchange(c.prewalk, 0);
    g.lookup += std::exchange(c.lookup, 0);
    g.derive += std::exchange(c.derive, 0);
    g.emit += std::exchange(c.emit, 0);
    g.emit_commit += std::exchange(c.emit_commit, 0);
    g.emit_table += std::exchange(c.emit_table, 0);
    g.table_skips += std::exchange(c.table_skips, 0);
    g.draws += std::exchange(c.draws, 0);
    g.memo_hit[w & 7] += std::exchange(c.memo_hit, 0);
    g.memo_miss[w & 7] += std::exchange(c.memo_miss, 0);
    g.memo_ineligible += std::exchange(c.inelig, 0);
    g.hdr_hit += std::exchange(c.hdr_hit, 0);
    g.hdr_miss += std::exchange(c.hdr_miss, 0);
    g.fetch_hit += std::exchange(c.fetch_hit, 0);
    g.fetch_miss += std::exchange(c.fetch_miss, 0);
    g.pw_header += std::exchange(c.pw_header, 0);
    g.pw_info += std::exchange(c.pw_info, 0);
    g.pw_walk += std::exchange(c.pw_walk, 0);
    g.pw_fetch += std::exchange(c.pw_fetch, 0);
    g.map_lock += c.map_lock;
    g.info_lock += c.info_lock;
    g_lean_fault_stats.lock_takes.fetch_add(std::exchange(c.info_lock, 0), std::memory_order_relaxed);
    c.map_lock = 0;
}

bool Parser::MappedFast(VAddr addr, u64 size) const {
    // = Rasterizer::IsMapped(addr, size) without its lock: the containing interval is cached
    // per worker and dropped whenever the map generation moves.
    auto& c = *caches;
    const u64 gen = Vulkan::Rasterizer::MappedGen();
    if (gen != c.span_gen) {
        c.n_spans = 0;
        c.next_span = 0;
        c.span_gen = gen;
    }
    if (addr > ~u64{0} - size) {
        return false;
    }
    const VAddr end = addr + size;
    for (u32 i = 0; i < c.n_spans; ++i) {
        if (addr >= c.spans[i].lo && end <= c.spans[i].hi) {
            return true;
        }
    }
    ++c.map_lock;
    VAddr lo = 0, hi = 0;
    if (!env.rasterizer->MappedInterval(addr, lo, hi) || end > hi) {
        return false;
    }
    c.spans[c.next_span] = {lo, hi};
    c.next_span = (c.next_span + 1) % c.spans.size();
    c.n_spans = std::min<u32>(c.n_spans + 1, static_cast<u32>(c.spans.size()));
    return true;
}

Parser::Parser(const Env& env_) : env{env_} {
    caches = std::make_unique<Caches>();
    caches->info.resize(g_lean_perdraw ? 2048 : 256);
    caches->fetch.resize(g_lean_perdraw ? 512 : 64);
    caches->memo.resize(g_lean_perdraw ? size_t{Caches::MemoSets} * Caches::MemoWays : 128);
    static_assert(std::has_single_bit(Caches::MemoSets));
    // The whole parser is off when the front would take a path this file does not mirror:
    //  - split_verify: EmitDraw's periodic synchronous VERIFY draw and the snapshot block;
    //  - !split_run_ahead (depth 0): every draw record syncs;
    //  - NativeRec::g_segb: SegB::NotePacket on every packet and the segment builder in EmitDraw;
    //  - SHADPS4_DRAW_TAG: g_draw_tag_front writes SplitDrawRecord::pad and front counters;
    //  - SHADPS4_DISPATCH_SYNC_STATS: g_dispatch_sync.Note keeps front statistics per packet.
    //    (SHADPS4_GUEST_DISPATCH_ORDER is supported: the span counts its run-ending packets in
    //    SpanResult::run_bumps; runs are only compared for equality, so one fresh run at the
    //    hand-back is equivalent.)
    //  - SHADPS4_SPLIT_HAZARD_STATS: SplitHazard::NoteDraw keeps front-global page statistics;
    //  - no rasterizer / pipeline cache / split: ProcessGraphics draws inline, no records.
    const bool ok = env.rasterizer && env.pipeline_cache && env.split_enabled &&
                    env.split_run_ahead && !env.split_verify && NativeRec::g_segb == 0 &&
                    !env.draw_tag_front && !EnvFlag("SHADPS4_DISPATCH_SYNC_STATS") && !SplitHazard::Enabled();
    config_stop = ok ? Stop::None : Stop::Config;
    static_env = StaticEnv{.guest_markers = env.guest_markers, .split_post_dma = env.split_post_dma};
    viewport_memo = LeanViewportMemoEnabled();
    // Liverpool::BumpEpoch in split mode emits a record only under these flags.
    epoch_records = VideoCore::g_epoch_defer_upload || VideoCore::g_native_coherent != 0;
    scratch = std::make_unique<Vulkan::GraphicsLookupScratch>();
    vi = std::make_unique<CpSplit::VertexInputsDerived>();
}

Parser::~Parser() = default;

Stop Parser::DecideDraw(const PM4Header* header, std::span<const u32> dcb, u32 off,
                        const char* name, const CpSplit::DrawArgs& args, const ChainState& st,
                        DrawPlan& plan) {
    // EmitDraw up to (not including) the record, with PrepareDrawFront inlined; the registers
    // already hold this draw packet's writes (as the front's do when EmitDraw runs). Nothing here
    // writes `st` or the channel: a Stop leaves both as they were, and the caller restores the
    // draw packet's register words.
    const Regs& regs = *st.regs;
    plan = {};
    plan.name = name;
    plan.draw.args = args;

    if (regs_only) {
        return Stop::None; // the caller journals the draw packet's registers and drops the draw
    }
    // Rasterizer::PrepareDrawFront.
    plan.draw.draw_class = Rasterizer::ClassifyDraw(regs);
    Vulkan::Dlss::NoteDraw(regs);
    plan.draw.dlss_phase = Vulkan::Dlss::t_phase;
    if (plan.draw.draw_class == Rasterizer::DrawClass::Draw) {
        if (!args.is_indirect) {
            // SplitHazard::NoteDraw: front-global statistics, off by configuration (Config stop).
            plan.draw.skip = Rasterizer::SkipEmptyDrawRegs(regs);
        }
        if (!plan.draw.skip) {
            // Piece 1: read-only, shared-locked, never compiles. Null for every reason the
            // front's GetGraphicsPipeline would return null for AND for every miss it would
            // compile / build on: the front must run that draw.
            // Guest reads first, outside the cache's lock, each pre-checked (sec. 16).
            t_lean_fault = false;
            auto& cc = *caches;
            const u64 tp0 = __rdtsc();
            if (!Prewalk(regs)) {
                scratch->prewalked = false;
                cc.prewalk += __rdtsc() - tp0;
                return Stop::Fault;
            }
            const u64 tp1 = __rdtsc();
            cc.prewalk += tp1 - tp0;
            // The lookup memo on the front's key (PreparedGraphicsState: render state + shader
            // hashes + settings, no user data) + the specialization guards of the selection.
            constexpr u32 fsi = static_cast<u32>(Shader::LogicalStage::Fragment);
            constexpr u32 vsi = static_cast<u32>(Shader::LogicalStage::Vertex);
            Vulkan::PreparedGraphicsState pstate;
            const bool memo_ok = env.rasterizer && scratch->pre_params[fsi].has_value() &&
                                 scratch->pre_params[vsi].has_value() &&
                                 !scratch->pre_missing[fsi] && !scratch->pre_missing[vsi] &&
                                 env.rasterizer->LeanVertexInputDynamic() &&
                                 pstate.Capture(regs, scratch->pre_params[fsi]->hash,
                                                scratch->pre_params[vsi]->hash);
            Caches::Memo* memo = nullptr;
            bool verify_hit = false;
            if (memo_ok) {
                // Refresh a memo's private Infos with THIS draw, then its guards.
                const auto matches = [&](Caches::Memo& m) {
                    const auto refresh = [&](Shader::Info& info, u32 i, const auto& sh) {
                        info.user_data = sh.user_data;
                        info.flattened_ud_buf = scratch->pre_flat[i];
                        info.pgm_base = scratch->pre_params[i]->Base();
                    };
                    refresh(*m.fs_info, fsi, regs.ps_program);
                    refresh(*m.vs_info, vsi, regs.vs_program);
                    return m.fs_guard.MatchesInputs(*m.fs_info) &&
                           m.vs_guard.MatchesInputs(*m.vs_info);
                };
                bool hit = false;
                if (g_lean_perdraw) {
                    // The set of this state: a way hits under exactly the direct-mapped rule
                    // (valid, same epoch, same state, both guards). On a miss `memo` is the
                    // way the store below fills: a dead one (invalid / older epoch), else the
                    // least recently used.
                    const u64 epoch = env.pipeline_cache->PreparedEpoch();
                    Caches::Memo* set = &cc.memo[(pstate.Hash() & (Caches::MemoSets - 1)) *
                                                 Caches::MemoWays];
                    Caches::Memo* victim = set;
                    for (u32 w = 0; w < Caches::MemoWays; ++w) {
                        Caches::Memo& m = set[w];
                        const bool live = m.valid && m.epoch == epoch;
                        if (live && m.state == pstate && matches(m)) {
                            memo = &m;
                            hit = true;
                            break;
                        }
                        const bool victim_live = victim->valid && victim->epoch == epoch;
                        if (!live ? victim_live : (victim_live && m.last_use < victim->last_use)) {
                            victim = &m;
                        }
                    }
                    if (hit) {
                        memo->last_use = ++cc.memo_clock;
                    } else {
                        memo = victim;
                    }
                } else {
                    memo = &cc.memo[pstate.Hash() & (cc.memo.size() - 1)];
                    hit = memo->valid && memo->state == pstate &&
                          memo->epoch == env.pipeline_cache->PreparedEpoch() && matches(*memo);
                }
                if (hit) {
                    ++cc.memo_hit;
                    verify_hit = (++cc.memo_hits_total & 63) == 0;
                    if (!verify_hit) {
                        plan.draw.pipeline = memo->pipeline;
                        scratch->Reset();
                        const auto fill = [&](u32 i, const Shader::Info* shared,
                                              const Shader::Info& priv, vk::ShaderModule mod) {
                            auto& slot = scratch->stages[i];
                            slot.info = shared;
                            slot.pgm_base = priv.pgm_base;
                            slot.user_data.assign(priv.user_data.begin(), priv.user_data.end());
                            slot.flattened_ud_buf = priv.flattened_ud_buf;
                            slot.module = mod;
                        };
                        fill(fsi, memo->fs_shared, *memo->fs_info, memo->fs_module);
                        fill(vsi, memo->vs_shared, *memo->vs_info, memo->vs_module);
                        scratch->fetch_shader = memo->fetch;
                        scratch->pipeline = memo->pipeline;
                        scratch->prewalked = false;
                        goto looked_up;
                    }
                }
                ++cc.memo_miss;
            } else {
                ++cc.inelig;
            }
            scratch->capture_guards = memo_ok;
            g_lean_fault_stats.lock_takes.fetch_add(1, std::memory_order_relaxed);
            g_lean_fault_stats.lock_draws.fetch_add(1, std::memory_order_relaxed);
            if (env.l2) {
                // L2: the read-only lookup first; a miss builds (the front's own lookup over the
                // worker's registers, exclusive; its guest reads were pre-checked / faulted in by
                // the prewalk), then the captured per-draw fields are rebuilt from the registers.
                t_lean_in_lock = true;
                plan.draw.pipeline = env.pipeline_cache->LookupGraphics(regs, *scratch);
                if (!plan.draw.pipeline) {
                    plan.draw.pipeline = env.pipeline_cache->LookupOrBuildGraphics(regs, *scratch);
                    t_lean_in_lock = false;
                    g_lean_l2.builds.fetch_add(1, std::memory_order_relaxed);
                    if (plan.draw.pipeline && !RefreshCaptured(regs)) {
                        g_lean_l2.build_refresh_fail.fetch_add(1, std::memory_order_relaxed);
                        scratch->prewalked = false;
                        return Stop::Fault;
                    }
                }
                t_lean_in_lock = false;
            } else {
                plan.draw.pipeline = env.pipeline_cache->LookupGraphics(regs, *scratch);
            }
            scratch->prewalked = false;
            scratch->capture_guards = false;
            if (verify_hit && memo) {
                // 1 in 64 memo hits: the real lookup above, compared field by field.
                ++g_lean_phase.memo_verify;
                bool same = plan.draw.pipeline == memo->pipeline &&
                            scratch->fetch_shader == memo->fetch;
                const auto cmp = [&](u32 i, const Shader::Info* shared, const Shader::Info& priv,
                                     vk::ShaderModule mod) {
                    const auto& slot = scratch->stages[i];
                    return slot.info == shared && slot.pgm_base == priv.pgm_base &&
                           std::equal(slot.user_data.begin(), slot.user_data.end(),
                                      priv.user_data.begin(), priv.user_data.end()) &&
                           slot.flattened_ud_buf == priv.flattened_ud_buf && slot.module == mod;
                };
                same = same && cmp(fsi, memo->fs_shared, *memo->fs_info, memo->fs_module) &&
                       cmp(vsi, memo->vs_shared, *memo->vs_info, memo->vs_module);
                if (!same && g_lean_phase.memo_verify_bad.fetch_add(1) < 8) {
                    LOG_WARNING(Render, "Lean memo VERIFY mismatch: pipeline memo {} real {} fetch {} / {}",
                                fmt::ptr(memo->pipeline), fmt::ptr(plan.draw.pipeline),
                                fmt::ptr(memo->fetch), fmt::ptr(scratch->fetch_shader));
                }
            } else if (memo && plan.draw.pipeline && !t_lean_fault &&
                       scratch->guard_views[fsi] && scratch->guard_views[vsi] &&
                       scratch->guards[fsi].valid && scratch->guards[vsi].valid &&
                       scratch->stages[fsi].info && scratch->stages[vsi].info) {
                // Memoize a successful read-only lookup (a build path captures no guards).
                memo->valid = true;
                memo->state = pstate;
                memo->epoch = env.pipeline_cache->PreparedEpoch();
                memo->pipeline = plan.draw.pipeline;
                memo->fetch = scratch->fetch_shader;
                memo->fs_guard = scratch->guards[fsi];
                memo->vs_guard = scratch->guards[vsi];
                if (g_lean_perdraw && memo->fs_info && memo->vs_info) {
                    // Copy-assign into the evicted entry's private Infos (their vectors keep
                    // their capacity), as GraphicsLookupScratch::View refreshes a view.
                    *memo->fs_info = *scratch->guard_views[fsi];
                    *memo->vs_info = *scratch->guard_views[vsi];
                } else {
                    memo->fs_info = std::make_unique<Shader::Info>(*scratch->guard_views[fsi]);
                    memo->vs_info = std::make_unique<Shader::Info>(*scratch->guard_views[vsi]);
                }
                if (g_lean_perdraw) {
                    memo->last_use = ++cc.memo_clock;
                }
                memo->fs_shared = scratch->stages[fsi].info;
                memo->vs_shared = scratch->stages[vsi].info;
                memo->fs_module = scratch->stages[fsi].module;
                memo->vs_module = scratch->stages[vsi].module;
            }
            scratch->guard_views = {};
            for (auto& g : scratch->guards) {
                g.valid = false;
            }
        looked_up:
            cc.lookup += __rdtsc() - tp1;
            if (t_lean_fault) {
                t_lean_fault = false;
                return Stop::FaultFallback;
            }
            if (!plan.draw.pipeline) {
                if (!env.l2) {
                    g_lean_fault_stats.miss[static_cast<u32>(scratch->miss) & 7].fetch_add(
                        1, std::memory_order_relaxed);
                    return Stop::PipelineMiss;
                }
                // The front's GetGraphicsPipeline returned null too: a draw with no stages.
                g_lean_l2.build_null.fetch_add(1, std::memory_order_relaxed);
            } else if (VideoCore::g_native_pass != 0 && !plan.draw.pipeline->native_checked) {
                if (!env.l2) {
                    return Stop::NativeUnchecked; // the front would run PrepareNativeVariant
                }
                t_lean_in_lock = true;
                env.pipeline_cache->PrepareNativeVariantRefreshed(plan.draw.pipeline, regs);
                t_lean_in_lock = false;
                g_lean_l2.native_preps.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (!plan.draw.skip && plan.draw.draw_class == Rasterizer::DrawClass::Draw &&
        Rasterizer::NoParticlesSkip(plan.draw.pipeline)) [[unlikely]] {
        plan.draw.skip = true; // q40
    }
    plan.has_stages = plan.draw.pipeline &&
                      plan.draw.draw_class == Rasterizer::DrawClass::Draw && !plan.draw.skip;

    plan.bytes = sizeof(SplitDrawRecord);
    if (plan.has_stages) {
        // pipeline->GetStages() in logical-stage order = the scratch's bound stages (piece 1's
        // LookupMatchesFront checks the Info pointers slot by slot).
        for (u32 i = 0; i < GraphicsStageCount; ++i) {
            const auto& slot = scratch->stages[i];
            if (slot.info) {
                plan.bytes += StageBytes(static_cast<u32>(slot.user_data.size()),
                                         static_cast<u32>(slot.flattened_ud_buf.size()));
            }
        }
    }

    // FRONT_DERIVE: the derivation from the scratch's vertex stage (the front builds its
    // BoundStage from the shared Info the lookup just refreshed; same words). The V# reads go
    // through the scratch's user data copy. Not for indirect draws (as EmitDraw).
    const u64 td0 = __rdtsc();
    SCOPE_EXIT {
        caches->derive += __rdtsc() - td0;
    };
    // q1: SHADPS4_DERIVE_INDIRECT=1 derives indirect draws too (the back ignores the block for
    // them; the colour-variant preparation below then runs for their pipelines).
    static const bool derive_indirect = EnvFlag("SHADPS4_DERIVE_INDIRECT");
    if (env.split_front_derive != 0 && plan.has_stages && (!args.is_indirect || derive_indirect)) {
        const auto& vslot = scratch->stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
        Shader::BoundStage vs{};
        vs.info = vslot.info;
        vs.user_data = vslot.user_data;
        vs.flattened_ud_buf = vslot.flattened_ud_buf;
        vs.pgm_base = vslot.pgm_base;
        if (const auto* fetch = scratch->fetch_shader) {
            // One V# per fetch attribute, read through a user-data pointer (Info::ReadUdReg).
            for (const auto& attrib : fetch->attributes) {
                if (attrib.sgpr_base == Shader::IR::NumScalarRegs) {
                    continue; // from the user data itself (no guest read)
                }
                if (attrib.sgpr_base + 1u >= vslot.user_data.size()) {
                    g_lean_fault_stats.precheck[3].fetch_add(1, std::memory_order_relaxed);
                    return Stop::Fault; // ReadUdReg would build a pointer past the user data
                }
                const u64 base = (u64{vslot.user_data[attrib.sgpr_base]} |
                                  (u64{vslot.user_data[attrib.sgpr_base + 1u]} << 32)) &
                                 0xFFFFFFFFFFFFull;
                if (!ReadOk(base + u64{attrib.dword_offset} * 4, sizeof(AmdGpu::Buffer), 2)) {
                    return Stop::Fault;
                }
            }
        }
        t_lean_fault = false;
        plan.vi_bytes = plan.draw.pipeline->DeriveVertexInputs(
            *vi, regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);
        if (t_lean_fault) {
            t_lean_fault = false;
            return Stop::FaultFallback;
        }
        plan.bytes += plan.vi_bytes;
        // The desc-id rule's comparison (applied in EmitDraw, after the decision).
        bool same = vi->header.count == st.vi_prev_count;
        for (u32 i = 0; same && i < vi->header.count; ++i) {
            same = std::memcmp(&vi->attribs[i].desc, &st.vi_prev[i],
                               sizeof(CpSplit::VertexInputDesc)) == 0;
        }
        plan.vi_same = same;
        if (VideoCore::g_native_colour) [[unlikely]] {
            // PrepareNativeColourFront: only when it would return without writing anything.
            if (!Rasterizer::NativeColourVariantReady(plan.draw.pipeline, *vi)) {
                if (!env.l2) {
                    return Stop::ColourVariantMissing;
                }
                // PrepareNativeColourFront: decides / queues the variant (compiled elsewhere).
                t_lean_in_lock = true;
                env.pipeline_cache->PrepareNativeColourRefreshed(plan.draw.pipeline, *vi, regs);
                t_lean_in_lock = false;
                g_lean_l2.colour_preps.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // NativeRec >= 2: EmitDraw reads the DRAW_INDEX_2 trailing NOP (the next packet) for the
    // job record of this draw. The front's member path / run state (NativeRecMember, NativeRecHead,
    // g_nr_run) is front-global and NOT reproduced: nr_tag / nr_idx / job_member stay 0.
    if (NativeRec::g_mode >= 2 && header->type3.opcode == PM4ItOpcode::DrawIndex2) {
        const u32* w = reinterpret_cast<const u32*>(header);
        if (off + 8 > dcb.size()) {
            return Stop::Overrun;
        }
        if (w[6] == 0xc0021000u && w[7] == NativeRec::Magic) {
            if (off + 10 > dcb.size()) {
                return Stop::Overrun;
            }
            plan.job_tag = w[8];
            plan.job_idx = w[9];
        }
    } else if (NativeRec::g_mode >= 2 &&
               header->type3.opcode == PM4ItOpcode::DrawIndexIndirect && off + 9 <= dcb.size()) {
        // q2: DRAW_INDEX_INDIRECT (5 dwords) + the HLE's tagged NOP (SHADPS4_A2_INDIRECT).
        const u32* w = reinterpret_cast<const u32*>(header);
        if (w[5] == 0xc0021000u && w[6] == NativeRec::Magic) {
            plan.job_tag = w[7];
            plan.job_idx = w[8];
        }
    }
    return Stop::None;
}

void Parser::EmitDraw(const PM4Header* header, const DrawPlan& plan, ChainState& st,
                      CpSplit::Channel& out) {
    // The record part of Liverpool::EmitDraw for a posted, non-verify, non-member draw.
    auto* rec = out.Reserve<SplitDrawRecord>(CpSplit::RecordType::Draw, plan.bytes);
    rec->draw = plan.draw;
    rec->ctx_epoch = st.ctx_epoch;
    rec->marker = env.host_markers ? plan.name : nullptr;
    rec->cmd_address = header;
    rec->stage_count = 0;
    rec->verify = 0;
    rec->vi_bytes = plan.vi_bytes;
    rec->pad = 0;
    rec->nr_tag = 0;
    rec->nr_idx = 0;
    rec->job_tag = plan.job_tag;
    rec->job_idx = plan.job_idx;
    rec->job_member = 0;
    rec->job_pad = 0;
    rec->segb_id = 0;
    rec->segb_member = 0;
    rec->segb_index_addr = 0;
    rec->segb_index_count = 0;
    rec->segb_instances = 0;
    auto* p = reinterpret_cast<u8*>(rec + 1);
    if (plan.has_stages) {
        for (u32 i = 0; i < GraphicsStageCount; ++i) {
            const auto& slot = scratch->stages[i];
            if (slot.info) {
                p = WriteStage(p, slot.info, slot.pgm_base, slot.user_data, slot.flattened_ud_buf,
                               i);
                ++rec->stage_count;
            }
        }
    }
    if (plan.vi_bytes != 0) {
        // Number the description part: the back skips the setVertexInputEXT of a repeated id.
        if (!plan.vi_same) {
            ++st.vi_desc_id;
            st.vi_prev_count = vi->header.count;
            for (u32 i = 0; i < vi->header.count; ++i) {
                st.vi_prev[i] = vi->attribs[i].desc;
            }
        }
        vi->header.desc_id = st.vi_desc_id;
        p = vi->Write(p);
    }
    const u64 tc0 = __rdtsc();
    out.Commit(); // L3: the draw hook (walk + TakeDraw) runs inside
    const u64 tc1 = __rdtsc();
    caches->emit_commit += tc1 - tc0;
    // NativeRec stage a2, as the front's NativeRecFront: the job threads derive their records
    // from this table (in L2 the front sees only the tails).
    if (NativeRec::g_mode >= 2 && plan.has_stages && plan.job_tag != 0) {
        const auto& vud = scratch->stages[static_cast<u32>(Shader::LogicalStage::Vertex)].user_data;
        const Shader::Info* vsi =
            plan.draw.pipeline->GetStages()[static_cast<u32>(Shader::LogicalStage::Vertex)];
        u64 fetch = 0;
        if (vsi && vsi->has_fetch_shader) {
            const u32 b = vsi->fetch_shader_sgpr_base;
            if (b + 1 < vud.size()) {
                fetch = (u64(vud[b + 1]) << 32 | vud[b]) & 0xFFFFFFFFFFFFull;
            }
        }
        const u64 h = (reinterpret_cast<u64>(plan.draw.pipeline) >> 4) ^ (fetch * 0x9E3779B97F4A7C15ull);
        auto& m = caches->pub_memo[(h ^ (h >> 29)) & 255];
        if (m.pipeline == plan.draw.pipeline && m.fetch == fetch) {
            ++caches->table_skips;
        } else {
            (void)NativeRecPublishProgram(plan.draw.pipeline, vud);
            m = {plan.draw.pipeline, fetch};
        }
        caches->emit_table += __rdtsc() - tc1;
    }
    out.PostRunAhead(CpSplit::PostKind::Draw);
    out.MaybeLog();
}

bool IsStaticStop(Stop stop) {
    switch (stop) {
    case Stop::Unsupported:
    case Stop::Overrun:
    case Stop::HintMissing:
    case Stop::BadSetBase:
    case Stop::NopBackEffect:
    case Stop::DmaSync:
    case Stop::DmaGuard:
    case Stop::NonVsDraw:
        return true;
    default:
        return false;
    }
}

Stop StaticCheck(std::span<const u32> dcb, u32 off, const Regs& regs, const StaticEnv& env,
                 u32& opcode) {
    // The parser's per-packet checks that do not depend on runtime state, in its order.
    using CR = Liverpool;
    opcode = 0;
    const u32 n = static_cast<u32>(dcb.size());
    const u32 num_regs = static_cast<u32>(regs.reg_array.size());
    const u32* w = dcb.data() + off;
    const auto* header = reinterpret_cast<const PM4Header*>(w);
    if (header->type == 2) {
        return Stop::None;
    }
    if (header->type != 3) {
        return Stop::Unsupported; // type 0 / 1: the front asserts
    }
    const u32 count = header->type3.NumWords();
    const u32 dwords = count + 1;
    opcode = static_cast<u32>(header->type3.opcode.Value());
    if (count == 0) {
        return Stop::Unsupported;
    }
    if (off + dwords > n) {
        return Stop::Overrun;
    }
    switch (static_cast<PM4ItOpcode>(opcode)) {
    case PM4ItOpcode::Nop:
        return LeanNopHasBackEffect(header, env.guest_markers) ? Stop::NopBackEffect : Stop::None;
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::NumInstances:
        return Stop::None;
    case PM4ItOpcode::SetConfigReg:
        return Regs::ConfigRegWordOffset + (w[1] & 0xffff) + (count - 1) > num_regs
                   ? Stop::Unsupported
                   : Stop::None;
    case PM4ItOpcode::SetUconfigReg:
        return Regs::UconfigRegWordOffset + (w[1] & 0xffff) + (count - 1) > num_regs
                   ? Stop::Unsupported
                   : Stop::None;
    case PM4ItOpcode::SetContextReg: {
        const u32 reg_addr = Regs::ContextRegWordOffset + (w[1] & 0xffff);
        if (reg_addr + (count - 1) > num_regs) {
            return Stop::Unsupported;
        }
        u32 hint = ~0u;
        switch (reg_addr) {
        case CR::CbColor0Base:
        case CR::CbColor1Base:
        case CR::CbColor2Base:
        case CR::CbColor3Base:
        case CR::CbColor4Base:
        case CR::CbColor5Base:
        case CR::CbColor6Base:
        case CR::CbColor7Base: {
            const u32 nop_offset = header->type3.count.Value();
            if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                hint = nop_offset;
            }
            break;
        }
        case CR::CbColor0Cmask:
        case CR::CbColor1Cmask:
        case CR::CbColor2Cmask:
        case CR::CbColor3Cmask:
        case CR::CbColor4Cmask:
        case CR::CbColor5Cmask:
        case CR::CbColor6Cmask:
        case CR::CbColor7Cmask:
            if (header->type3.count == 0x04) {
                hint = 0x04;
            }
            break;
        case CR::DbZInfo:
            if (header->type3.count == 8) {
                hint = 20;
            }
            break;
        default:
            break;
        }
        if (hint != ~0u) {
            if (off + 2 + hint + 2 > n) {
                return Stop::Overrun;
            }
            if (w[2 + hint] != ExtentHintHeader) {
                return Stop::HintMissing;
            }
        }
        return Stop::None;
    }
    case PM4ItOpcode::SetShReg: {
        const u32 reg_off = w[1] & 0xffff;
        const size_t set_size = size_t{count - 1} * sizeof(u32);
        if (reg_off >= 0x200 && reg_off <= (0x200 + sizeof(ComputeProgram) / 4)) {
            const size_t byte_off = size_t{reg_off - 0x200} * sizeof(u32);
            return set_size > sizeof(ComputeProgram) || byte_off + set_size > sizeof(ComputeProgram)
                       ? Stop::Unsupported
                       : Stop::None;
        }
        return Regs::ShRegWordOffset + reg_off + (count - 1) > num_regs ? Stop::Unsupported
                                                                        : Stop::None;
    }
    case PM4ItOpcode::SetBase:
        return reinterpret_cast<const PM4CmdSetBase*>(header)->base_index.Value() !=
                       PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable
                   ? Stop::BadSetBase
                   : Stop::None;
    case PM4ItOpcode::DmaData: {
        const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
        if (dma->dst_addr_lo != 0x3022C) {
            return Stop::Unsupported;
        }
        if (!env.split_post_dma) {
            return Stop::DmaSync;
        }
        if (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) {
            const VAddr dst = dma->DstAddress<VAddr>();
            const VAddr dst_end = dst + dma->NumBytes();
            const auto lo = reinterpret_cast<VAddr>(w);
            const VAddr hi = lo + size_t{n - off} * sizeof(u32);
            if (dst < hi && lo < dst_end) {
                return Stop::DmaGuard;
            }
        }
        return Stop::None;
    }
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndexIndirect:
        if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
            return Stop::NonVsDraw;
        }
        if (NativeRec::g_mode >= 2 && static_cast<PM4ItOpcode>(opcode) == PM4ItOpcode::DrawIndex2) {
            if (off + 8 > n) {
                return Stop::Overrun;
            }
            if (w[6] == 0xc0021000u && w[7] == NativeRec::Magic && off + 10 > n) {
                return Stop::Overrun;
            }
        }
        return Stop::None;
    default:
        return Stop::Unsupported;
    }
}

SpanResult Parser::Parse(std::span<const u32> dcb, u32 cut_dw, ChainState& st,
                         CpSplit::Channel& out) {
    SpanResult r{};
    const auto stop_at = [&](Stop stop, u32 off, u32 opcode) {
        r.stop = stop;
        r.end_dw = off;
        r.stop_opcode = opcode;
        return r;
    };
    if (config_stop != Stop::None) {
        return stop_at(config_stop, 0, 0);
    }
    ASSERT(st.regs);
    Regs& regs = *st.regs;
    const u32 n = static_cast<u32>(dcb.size());
    const u32 limit = std::min(cut_dw, n);
    const u32 num_regs = static_cast<u32>(regs.reg_array.size());
    u32 off = start;

    for (;;) {
        if (off >= n) {
            return stop_at(Stop::End, n, 0);
        }
        if (off >= limit) {
            return stop_at(Stop::Cut, off, 0);
        }
        const u32* w = dcb.data() + off;
        const auto* header = reinterpret_cast<const PM4Header*>(w);
        if (header->type == 2) {
            // Padding: NextPacket(dcb, 1) and `continue` (no epoch rule).
            ++off;
            ++r.packets;
            continue;
        }
        if (header->type != 3) {
            return stop_at(Stop::Unsupported, off, 0); // type 0 / 1: the front asserts
        }
        const u32 count = header->type3.NumWords();
        const u32 dwords = count + 1;
        const PM4ItOpcode opcode = header->type3.opcode;
        const u32 opcode_u = static_cast<u32>(opcode);
        if (count == 0) {
            // count field 0x3fff: NumWords wraps to 0 and `count - 1` below would wrap too (the
            // front has the same hole); leave such a packet to the front.
            return stop_at(Stop::Unsupported, off, opcode_u);
        }
        if (off + dwords > n) {
            return stop_at(Stop::Overrun, off, opcode_u); // NextPacket's empty-return case
        }
        if (off + dwords > limit) {
            return stop_at(Stop::Cut, off, opcode_u);
        }
        {
            // The shared static predicate first (the chain cut with the same function).
            u32 sop = 0;
            const Stop s = StaticCheck(dcb, off, regs, static_env, sop);
            if (s != Stop::None) {
                return stop_at(s, off, opcode_u);
            }
        }

        Stop stop = Stop::None;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // Flip IRQ / guest markers are back work (RunGfxPacket: a synchronous Packet record);
            // every other NOP (extent hints, job tags, sharp-in-udata notes) is nothing.
            if (LeanNopHasBackEffect(header, env.guest_markers)) {
                stop = Stop::NopBackEffect;
            }
            break;
        }
        case PM4ItOpcode::ContextControl:
        case PM4ItOpcode::AcquireMem: {
            break; // the front ignores both
        }
        case PM4ItOpcode::ClearState: {
            regs.SetDefaults();
            ++st.ctx_epoch;
            out.EmitClearState();
            break;
        }
        case PM4ItOpcode::SetConfigReg: {
            const u32 reg_addr = Regs::ConfigRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, size_t{words} * sizeof(u32));
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            break;
        }
        case PM4ItOpcode::SetContextReg: {
            using CR = Liverpool;
            const u32 reg_addr = Regs::ContextRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            // Decide first: the CB / DB extent hint the front reads PAST this packet
            // (payload[k] = w[2 + k] with payload = header + 2): a NOP header at k, the raw
            // extent at k + 1. The front ASSERTs the header word; the parser stops instead.
            u32 hint = ~0u;
            switch (reg_addr) {
            case CR::CbColor0Base:
            case CR::CbColor1Base:
            case CR::CbColor2Base:
            case CR::CbColor3Base:
            case CR::CbColor4Base:
            case CR::CbColor5Base:
            case CR::CbColor6Base:
            case CR::CbColor7Base: {
                const u32 nop_offset = header->type3.count.Value();
                if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                    hint = nop_offset;
                }
                break;
            }
            case CR::CbColor0Cmask:
            case CR::CbColor1Cmask:
            case CR::CbColor2Cmask:
            case CR::CbColor3Cmask:
            case CR::CbColor4Cmask:
            case CR::CbColor5Cmask:
            case CR::CbColor6Cmask:
            case CR::CbColor7Cmask: {
                if (header->type3.count == 0x04) {
                    hint = 0x04;
                }
                break;
            }
            case CR::DbZInfo: {
                if (header->type3.count == 8) {
                    hint = 20;
                }
                break;
            }
            default:
                break;
            }
            if (hint != ~0u) {
                if (off + 2 + hint + 2 > n) {
                    stop = Stop::Overrun;
                    break;
                }
                if (w[2 + hint] != ExtentHintHeader) {
                    stop = Stop::HintMissing;
                    break;
                }
            }
            // Apply, in the front's order: epoch on a real change, the words, the journal, the
            // extent, the extent record.
            const size_t ctx_sz = size_t{words} * sizeof(u32);
            if (viewport_memo && std::memcmp(&regs.reg_array[reg_addr], w + 2, ctx_sz) != 0) {
                ++st.ctx_epoch;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, ctx_sz);
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            switch (reg_addr) {
            case CR::CbColor0Base:
            case CR::CbColor1Base:
            case CR::CbColor2Base:
            case CR::CbColor3Base:
            case CR::CbColor4Base:
            case CR::CbColor5Base:
            case CR::CbColor6Base:
            case CR::CbColor7Base: {
                const u32 cb = (reg_addr - CR::CbColor0Base) / (CR::CbColor1Base - CR::CbColor0Base);
                st.cb_extent[cb].raw = hint != ~0u ? w[2 + hint + 1] : 0u;
                break;
            }
            case CR::CbColor0Cmask:
            case CR::CbColor1Cmask:
            case CR::CbColor2Cmask:
            case CR::CbColor3Cmask:
            case CR::CbColor4Cmask:
            case CR::CbColor5Cmask:
            case CR::CbColor6Cmask:
            case CR::CbColor7Cmask: {
                if (hint != ~0u) { // else untouched (the front leaves the extent as it was)
                    const u32 cb =
                        (reg_addr - CR::CbColor0Cmask) / (CR::CbColor1Cmask - CR::CbColor0Cmask);
                    st.cb_extent[cb].raw = w[2 + hint + 1];
                }
                break;
            }
            case CR::DbZInfo: {
                st.db_extent.raw = hint != ~0u ? w[2 + hint + 1] : 0u;
                break;
            }
            default:
                break;
            }
            JournalExtent(out, st, reg_addr);
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const u32 reg_off = w[1] & 0xffff;
            const size_t set_size = size_t{count - 1} * sizeof(u32);
            if (reg_off >= 0x200 && reg_off <= (0x200 + sizeof(ComputeProgram) / 4)) {
                // The gfx queue's compute program (dispatch state): no journal, no register
                // words. The front ASSERTs set_size <= sizeof(ComputeProgram) and does not check
                // the end: both become a stop here (the second would write past cs_state).
                const size_t byte_off = size_t{reg_off - 0x200} * sizeof(u32);
                if (set_size > sizeof(ComputeProgram) ||
                    byte_off + set_size > sizeof(ComputeProgram)) {
                    stop = Stop::Unsupported;
                    break;
                }
                std::memcpy(reinterpret_cast<u8*>(&st.cs_state) + byte_off, w + 2, set_size);
            } else {
                const u32 reg_addr = Regs::ShRegWordOffset + reg_off;
                if (reg_addr + (count - 1) > num_regs) {
                    stop = Stop::Unsupported;
                    break;
                }
                std::memcpy(&regs.reg_array[reg_addr], w + 2, set_size);
                out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], count - 1);
                // PrefetchProgramsOnWrite (SHADPS4_PGM_PREFETCH): a cache prefetch hint for the
                // front's own next lookup; nothing observable, skipped.
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg: {
            const u32 reg_addr = Regs::UconfigRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, size_t{words} * sizeof(u32));
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            break;
        }
        case PM4ItOpcode::IndexType: {
            const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
            regs.index_buffer_type.raw = index_type->raw;
            JournalField(out, regs, regs.index_buffer_type);
            break;
        }
        case PM4ItOpcode::NumInstances: {
            const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
            regs.num_instances.num_instances = num_instances->num_instances;
            JournalField(out, regs, regs.num_instances);
            break;
        }
        case PM4ItOpcode::IndexBase: {
            const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
            regs.index_base_address.base_addr_lo = index_base->addr_lo;
            regs.index_base_address.base_addr_hi = index_base->addr_hi;
            JournalField(out, regs, regs.index_base_address);
            break;
        }
        case PM4ItOpcode::IndexBufferSize: {
            const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
            regs.num_indices = index_size->num_indices;
            JournalField(out, regs, regs.num_indices);
            regs.max_index_size = index_size->num_indices; // r4: VGT_DMA_MAX_SIZE (upstream #5278)
            JournalField(out, regs, regs.max_index_size);
            break;
        }
        case PM4ItOpcode::SetBase: {
            const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
            if (set_base->base_index.Value() !=
                PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable) {
                stop = Stop::BadSetBase; // the front ASSERTs
                break;
            }
            // = set_base->Address<u64>() (no journal: a front-only value for indirect draws).
            st.indirect_args_addr =
                VAddr{set_base->address0} | (VAddr{set_base->address1 & 0xffff} << 32u);
            break;
        }
        case PM4ItOpcode::DmaData: {
            // Only the no-op L2-prefetch DMA (destination 0x3022C; liverpool.cpp A2NoopDma): the
            // back executes its Packet record as a no-op. Front path = RunGfxDma under POST_DMA:
            // a verbatim Packet record, posted. Everything that makes the front wait or log is a
            // stop.
            const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
            if (dma->dst_addr_lo != 0x3022C) {
                stop = Stop::Unsupported;
                break;
            }
            if (!env.split_post_dma) {
                stop = Stop::DmaSync; // RunGfxPacket: synchronous at every depth
                break;
            }
            if (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) {
                // RunGfxDma's guard against the rest of THIS command buffer (the front's `dcb`
                // span starts at this packet and runs to the buffer's end).
                const VAddr dst = dma->DstAddress<VAddr>();
                const VAddr dst_end = dst + dma->NumBytes();
                const auto lo = reinterpret_cast<VAddr>(w);
                const VAddr hi = lo + size_t{n - off} * sizeof(u32);
                if (dst < hi && lo < dst_end) {
                    stop = Stop::DmaGuard; // the front syncs (PacketDmaGuard)
                    break;
                }
            }
            // TASK_GUARD: the front does not note the no-op DMA in its posted-write log (the
            // back never runs it; RunGfxDma), so there is nothing to mirror.
            void* copy = out.Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                     env.guest_markers ? u16{CpSplit::GuestMarkers} : u16{0});
            std::memcpy(copy, header, dwords * sizeof(u32));
            out.Commit();
            out.PostRunAhead(CpSplit::PostKind::GfxPacket);
            break;
        }
        case PM4ItOpcode::DrawIndex2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            const u32 old_max = regs.max_index_size;
            const auto old_base = regs.index_base_address;
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.max_index_size = d->max_size;
            regs.index_base_address.base_addr_lo = d->index_base_lo;
            regs.index_base_address.base_addr_hi = d->index_base_hi;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndex2", {.is_indexed = true}, st, plan);
            if (stop != Stop::None) {
                regs.max_index_size = old_max;
                regs.index_base_address = old_base;
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.max_index_size);
            JournalField(out, regs, regs.index_base_address);
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            // DebugState.PushRegsDump (debugger register dumps): not mirrored.
            if (regs_only) {
                ++r.dropped;
            } else {
                const u64 te0 = __rdtsc();
                EmitDraw(header, plan, st, out);
                caches->emit += __rdtsc() - te0;
                ++caches->draws;
                ++r.draws;
            }
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            const u32 old_max = regs.max_index_size;
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.max_index_size = d->max_size;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndexOffset2",
                              {.is_indexed = true, .index_offset = d->index_offset}, st, plan);
            if (stop != Stop::None) {
                regs.max_index_size = old_max;
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.max_index_size);
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            if (regs_only) {
                ++r.dropped;
            } else {
                const u64 te0 = __rdtsc();
                EmitDraw(header, plan, st, out);
                caches->emit += __rdtsc() - te0;
                ++caches->draws;
                ++r.draws;
            }
            break;
        }
        case PM4ItOpcode::DrawIndexIndirect: {
            // As ProcessGraphics: EmitDraw with the indirect args at SET_BASE's address, no
            // register writes.
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndexIndirect",
                              {.is_indexed = true,
                               .is_indirect = true,
                               .arg_address = st.indirect_args_addr,
                               .offset = d->data_offset,
                               .stride = u32(sizeof(DrawIndexedIndirectArgs)),
                               .max_count = 1},
                              st, plan);
            if (stop != Stop::None) {
                break;
            }
            if (regs_only) {
                ++r.dropped;
            } else {
                const u64 te0 = __rdtsc();
                EmitDraw(header, plan, st, out);
                caches->emit += __rdtsc() - te0;
                ++caches->draws;
                ++r.draws;
            }
            break;
        }
        case PM4ItOpcode::DrawIndexAuto: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndexAuto", {}, st, plan);
            if (stop != Stop::None) {
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            if (regs_only) {
                ++r.dropped;
            } else {
                const u64 te0 = __rdtsc();
                EmitDraw(header, plan, st, out);
                caches->emit += __rdtsc() - te0;
                ++caches->draws;
                ++r.draws;
            }
            break;
        }
        default:
            stop = Stop::Unsupported;
            break;
        }
        if (stop != Stop::None) {
            return stop_at(stop, off, opcode_u);
        }
        // After every packet: Liverpool::BumpEpoch(opcode) unless the opcode keeps the window.
        if (!LeanKeepsSubmitEpoch(opcode) && epoch_records) {
            out.EmitEpochBump(opcode_u);
        }
        off += dwords;
        ++r.packets;
        switch (static_cast<PM4ItOpcode>(opcode_u)) { // DispatchSyncStats::RunNeutral
        case PM4ItOpcode::Nop:
        case PM4ItOpcode::SetShReg:
        case PM4ItOpcode::SetContextReg:
        case PM4ItOpcode::SetUconfigReg:
        case PM4ItOpcode::SetConfigReg:
        case PM4ItOpcode::IndirectBuffer:
        case PM4ItOpcode::IndirectBufferConst:
        case PM4ItOpcode::DispatchDirect:
        case PM4ItOpcode::DispatchIndirect:
            break;
        default:
            ++r.run_bumps;
            break;
        }
    }
}

bool Parser::ReadOk(VAddr addr, u64 size, u32 kind) const {
    auto& st = g_lean_fault_stats;
    auto& c = *caches;
    if (size == 0) {
        return true;
    }
    // Every page of the range already checked in THIS draw: nothing to do (never across draws).
    const u64 p0 = addr >> 12, p1 = (addr + size - 1) >> 12;
    bool all = addr >= 0x10000;
    for (u64 p = p0; all && p <= p1; ++p) {
        bool hit = false;
        for (u32 i = 0; i < c.n_ok && !hit; ++i) {
            hit = c.ok_pages[i] == p;
        }
        all = hit;
    }
    if (all) {
        return true;
    }
    const auto note_ok = [&] {
        for (u64 p = p0; p <= p1 && c.n_ok < c.ok_pages.size(); ++p) {
            c.ok_pages[c.n_ok++] = p;
        }
    };
    if (addr < 0x10000 || !MappedFast(addr, size)) {
        st.precheck[kind + 1].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (env.rasterizer->LeanPageManager().AnyReadProtected(addr, size)) {
        if (env.l2) {
            // L2: fault it in HERE, outside every lock (the hook waits for the back first).
            g_lean_l2.touches.fetch_add(1, std::memory_order_relaxed);
            for (VAddr p = addr & ~VAddr{0xfff}; p < addr + size; p += 0x1000) {
                const VAddr at = std::max(p, addr);
                (void)*reinterpret_cast<const volatile u8*>(at);
            }
            note_ok();
            return true;
        }
        st.precheck[kind].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    note_ok();
    return true;
}

bool Parser::HeaderOk(VAddr code, const AmdGpu::BinaryInfo*& found) const {
    found = nullptr;
    // AmdGpu::SearchBinaryInfo's reads, each page checked before it is touched.
    if (code >= (u64{1} << 40)) {
        g_lean_fault_stats.precheck[7].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto& hc = caches->hdr[(code >> 8) % caches->hdr.size()];
    if (hc.code == code) {
        // Validated hit: the same header bytes (signature, hash, length) at the same place.
        const VAddr info = code + hc.off;
        if (ReadOk(info, sizeof(AmdGpu::BinaryInfo), 6) &&
            std::memcmp(reinterpret_cast<const void*>(info), hc.copy.data(), hc.copy.size()) == 0) {
            found = reinterpret_cast<const AmdGpu::BinaryInfo*>(info);
            ++caches->hdr_hit;
            return true;
        }
    }
    ++caches->hdr_miss;
    const auto remember = [&](const AmdGpu::BinaryInfo* f) {
        hc.code = code;
        hc.off = static_cast<u32>(reinterpret_cast<VAddr>(f) - code);
        std::memcpy(hc.copy.data(), f, hc.copy.size());
    };
    if (!ReadOk(code, 8, 6)) {
        return false;
    }
    const auto* w = reinterpret_cast<const u32*>(code);
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (w[0] == token_mov_vcchi) {
        const VAddr info = code + (u64{w[1]} + 1) * 8;
        // The code itself too: a worker's build compiles it (reads [code, info)).
        if (!ReadOk(code, info + sizeof(AmdGpu::BinaryInfo) - code, 6)) {
            return false;
        }
        if (reinterpret_cast<const AmdGpu::BinaryInfo*>(info)->Valid()) {
            found = reinterpret_cast<const AmdGpu::BinaryInfo*>(info);
            remember(found);
            return true;
        }
    }
    constexpr u32 search_limit = 0x4000;
    VAddr checked_end = code; // [code, checked_end) verified
    for (u32 i = 0; i < search_limit; ++i) {
        const VAddr at = code + u64{i} * 4;
        while (at + sizeof(AmdGpu::BinaryInfo) > checked_end) {
            const VAddr page_end = (checked_end & ~VAddr{0xfff}) + 0x1000;
            if (!ReadOk(checked_end, page_end - checked_end, 6)) {
                return false;
            }
            checked_end = page_end;
        }
        if (reinterpret_cast<const AmdGpu::BinaryInfo*>(at)->Valid()) {
            found = reinterpret_cast<const AmdGpu::BinaryInfo*>(at);
            remember(found);
            return true;
        }
    }
    g_lean_fault_stats.precheck[7].fetch_add(1, std::memory_order_relaxed); // no header: the front asserts
    return false;
}

bool Parser::FetchOk(const Shader::Info& info, std::span<const u32> ud) const {
    // Gcn::GetFetchShaderCode (unmasked pointer) + DecodeFetchShader + the attributes' V#s.
    auto& st = g_lean_fault_stats;
    const auto bad = [&] {
        st.precheck[7].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    const u32 sb = info.fetch_shader_sgpr_base;
    if (sb + 1u >= ud.size()) {
        return bad();
    }
    const VAddr code = u64{ud[sb]} | (u64{ud[sb + 1u]} << 32);
    if (code >= (u64{1} << 40)) {
        return bad();
    }
    auto& cc = *caches;
    const auto check_vsharp = [&](u32 sgpr, u32 dword_offset) {
        if (sgpr == Shader::IR::NumScalarRegs) {
            return true; // ReadUdReg reads the user data itself (host)
        }
        if (sgpr + 1u >= ud.size()) {
            return bad();
        }
        const VAddr base = (u64{ud[sgpr]} | (u64{ud[sgpr + 1u]} << 32)) & 0xFFFFFFFFFFFFull;
        return ReadOk(base + u64{dword_offset} * 4, sizeof(AmdGpu::Buffer), 2);
    };
    auto& fc = cc.fetch[(code >> 4) & (cc.fetch.size() - 1)];
    if (fc.code == code && !fc.words.empty() && ReadOk(code, fc.words.size() * 4, 6) &&
        std::memcmp(reinterpret_cast<const void*>(code), fc.words.data(), fc.words.size() * 4) == 0) {
        // Validated hit: the same instruction words (incl. the terminating S_SETPC_B64).
        ++cc.fetch_hit;
        for (const auto& [sgpr, off] : fc.attrs) {
            if (!check_vsharp(sgpr, off)) {
                return false;
            }
        }
        return true;
    }
    ++cc.fetch_miss;
    std::vector<std::pair<u32, u32>> attrs;
    struct Load {
        u32 dword_offset;
        u32 base_sgpr;
    };
    std::array<Load, 104> loads{};
    Shader::Gcn::GcnDecodeContext decoder;
    VAddr checked_end = code; // [code, checked_end) verified
    VAddr pos = code;
    for (u32 n = 0;; ++n) {
        if (n == 1024) {
            return bad();
        }
        while (pos + 16 > checked_end) { // an instruction is at most 12 bytes
            const VAddr page_end = (checked_end & ~VAddr{0xfff}) + 0x1000;
            if (!ReadOk(checked_end, page_end - checked_end, 6)) {
                return false;
            }
            checked_end = page_end;
        }
        Shader::Gcn::GcnCodeSlice slice(reinterpret_cast<const u32*>(pos),
                                        reinterpret_cast<const u32*>(checked_end));
        const auto inst = decoder.decodeInstruction(slice);
        if (inst.length == 0) {
            return bad();
        }
        pos += inst.length;
        if (inst.opcode == Shader::Gcn::Opcode::S_SETPC_B64) {
            fc.code = code;
            fc.words.assign(reinterpret_cast<const u32*>(code), reinterpret_cast<const u32*>(pos));
            fc.attrs = std::move(attrs);
            return true;
        }
        if (inst.inst_class == Shader::Gcn::InstClass::ScalarMemRd) {
            const u32 d = inst.dst[0].code;
            if (d >= loads.size()) {
                return bad();
            }
            loads[d] = Load{inst.control.smrd.offset, inst.src[0].code * 2u};
            continue;
        }
        if (inst.inst_class == Shader::Gcn::InstClass::VectorMemBufFmt) {
            const u32 b = inst.src[2].code * 4u;
            if (b >= loads.size()) {
                return bad();
            }
            const u32 sgpr = loads[b].base_sgpr;
            attrs.emplace_back(sgpr, loads[b].dword_offset);
            if (!check_vsharp(sgpr, loads[b].dword_offset)) {
                return false;
            }
        }
    }
}

void Parser::Pretouch(const Regs& regs) {
    (void)Prewalk(regs);
    scratch->prewalked = false;
}

bool Parser::RefreshCaptured(const Regs& regs) {
    auto& sc = *scratch;
    const std::array<std::pair<Shader::Stage, Shader::LogicalStage>, 2> stages{{
        {Shader::Stage::Fragment, Shader::LogicalStage::Fragment},
        {Shader::Stage::Vertex, Shader::LogicalStage::Vertex},
    }};
    for (const auto& [in, out] : stages) {
        auto& slot = sc.stages[static_cast<u32>(out)];
        if (!slot.info) {
            continue;
        }
        const auto* pgm = regs.ProgramForStage(static_cast<u32>(in));
        if (!pgm) {
            return false;
        }
        // = GetProgram's refresh: info.user_data = params.user_data, pgm_base = params.Base().
        slot.user_data.assign(pgm->user_data.begin(), pgm->user_data.end());
        const auto& pre = sc.pre_params[static_cast<u32>(out)];
        slot.pgm_base = pre.has_value() ? pre->Base() : pgm->Address<VAddr>();
        if (!CheckedWalk(*slot.info, slot.user_data.data(), slot.flattened_ud_buf)) {
            return false;
        }
    }
    return true;
}

bool Parser::Prewalk(const Regs& regs) {
    // ResolveStageReadOnly's guest reads for the bound Fragment / Vertex stages, done here with no
    // lock held: GetParams (the shader header) and RefreshFlatBuf (user data + the SRT walk).
    auto& sc = *scratch;
    sc.prewalked = false;
    caches->n_ok = 0; // page state: this draw only
    if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs) {
        return true; // LookupGraphicsShared returns null (UnsupportedStages) before any read
    }
    const std::array<std::pair<Shader::Stage, Shader::LogicalStage>, 2> stages{{
        {Shader::Stage::Fragment, Shader::LogicalStage::Fragment},
        {Shader::Stage::Vertex, Shader::LogicalStage::Vertex},
    }};
    for (const auto& [in, out] : stages) {
        const u32 in_idx = static_cast<u32>(in);
        const u32 out_idx = static_cast<u32>(out);
        sc.pre_params[out_idx].reset();
        sc.pre_flat[out_idx].clear();
        sc.pre_missing[out_idx] = false;
        if (!regs.stage_enable.IsStageEnabled(in_idx)) {
            continue;
        }
        const auto* pgm = regs.ProgramForStage(in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            continue;
        }
        auto& cc = *caches;
        const AmdGpu::BinaryInfo* bin = nullptr;
        const u64 t0 = __rdtsc();
        const bool hdr_ok = HeaderOk(pgm->Address<VAddr>(), bin);
        const u64 t1 = __rdtsc();
        cc.pw_header += t1 - t0;
        if (!hdr_ok) {
            return false;
        }
        // AmdGpu::GetParams from the checked header (no second search, no memo).
        const Shader::ShaderParams params{
            .user_data = pgm->user_data,
            .code = std::span<const u32>{pgm->Address<u32*>(), bin->length / sizeof(u32)},
            .hash = bin->shader_hash,
        };
        sc.pre_params[out_idx] = params;
        auto& ie = cc.info[(params.hash ^ (params.hash >> 23)) & (cc.info.size() - 1)];
        const Shader::Info* info = ie.info != nullptr && ie.hash == params.hash ? ie.info : nullptr;
        if (!info) {
            ++cc.info_lock;
            info = env.pipeline_cache->ProgramInfoFor(params.hash);
            if (info) {
                ie = {params.hash, info};
            }
        }
        const u64 t2 = __rdtsc();
        cc.pw_info += t2 - t1;
        if (!info) {
            sc.pre_missing[out_idx] = true; // the lookup misses on it (no flat buffer built)
            continue;
        }
        const bool walk_ok = CheckedWalk(*info, params.user_data.data(), sc.pre_flat[out_idx]);
        const u64 t3 = __rdtsc();
        cc.pw_walk += t3 - t2;
        if (!walk_ok) {
            return false;
        }
        if (out == Shader::LogicalStage::Vertex && info->has_fetch_shader) {
            const bool fetch_ok = FetchOk(*info, params.user_data);
            cc.pw_fetch += __rdtsc() - t3;
            if (!fetch_ok) {
                return false;
            }
        }
    }
    sc.prewalked = true;
    return true;
}

bool Parser::CheckedWalk(const Shader::Info& info, const u32* user_data, std::vector<u32>& flat) {
    // RefreshFlatBuf: flat = user data, then the generated walker copies the SRT dwords. The
    // walker is interpreted (it only uses rdi / r10 / rsi and the stack) so every guest read is
    // pre-checked first. The fault handler's patches (xor rdi,rdi / xor r10d,r10d + nops) read 0.
    auto& st = g_lean_fault_stats;
    flat.assign(info.srt_info.flattened_bufsize_dw, 0u);
    std::memcpy(flat.data(), user_data,
                std::min<size_t>(flat.size(), Shader::ShaderParams::NumShaderUserData) * 4);
    const auto* code = reinterpret_cast<const u8*>(info.srt_info.walker_func);
    if (!code) {
        return true;
    }
    enum : u8 { PushRdi, PopRdi, LoadRdi, MovR10Imm, AndRdiR10, LoadR10, StoreR10, ZeroRdi, ZeroR10, Ret };
    auto& plan = walk_plans[code];
    if (plan.empty()) {
        const size_t size = info.srt_info.walker_func_size;
        size_t pos = 0;
        bool ok = false;
        while (pos < size) {
            ZydisDecodedInstruction ins;
            ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
            if (!ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                    ins, ops, const_cast<u8*>(code + pos), 15))) {
                break;
            }
            const auto reg0 = ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER ? ops[0].reg.value
                                                                         : ZYDIS_REGISTER_NONE;
            const auto reg1 = ops[1].type == ZYDIS_OPERAND_TYPE_REGISTER ? ops[1].reg.value
                                                                         : ZYDIS_REGISTER_NONE;
            const bool mem1 = ops[1].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                              ops[1].mem.base == ZYDIS_REGISTER_RDI &&
                              ops[1].mem.index == ZYDIS_REGISTER_NONE;
            const bool mem0 = ops[0].type == ZYDIS_OPERAND_TYPE_MEMORY &&
                              ops[0].mem.base == ZYDIS_REGISTER_RSI &&
                              ops[0].mem.index == ZYDIS_REGISTER_NONE;
            const s32 d1 = mem1 ? static_cast<s32>(ops[1].mem.disp.value) : 0;
            const s32 d0 = mem0 ? static_cast<s32>(ops[0].mem.disp.value) : 0;
            WalkOp w{0xff, 0, 0};
            switch (ins.mnemonic) {
            case ZYDIS_MNEMONIC_PUSH:
                if (reg0 == ZYDIS_REGISTER_RDI) w = {PushRdi, 0, 0};
                break;
            case ZYDIS_MNEMONIC_POP:
                if (reg0 == ZYDIS_REGISTER_RDI) w = {PopRdi, 0, 0};
                break;
            case ZYDIS_MNEMONIC_MOV:
                if (reg0 == ZYDIS_REGISTER_RDI && mem1) {
                    w = {LoadRdi, d1, 0};
                } else if (reg0 == ZYDIS_REGISTER_R10 &&
                           ops[1].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                    w = {MovR10Imm, 0, ops[1].imm.value.u};
                } else if (reg0 == ZYDIS_REGISTER_R10D && mem1) {
                    w = {LoadR10, d1, 0};
                } else if (mem0 && reg1 == ZYDIS_REGISTER_R10D) {
                    w = {StoreR10, d0, 0};
                }
                break;
            case ZYDIS_MNEMONIC_AND:
                if (reg0 == ZYDIS_REGISTER_RDI && reg1 == ZYDIS_REGISTER_R10) w = {AndRdiR10, 0, 0};
                break;
            case ZYDIS_MNEMONIC_XOR:
                if (reg0 == ZYDIS_REGISTER_RDI && reg1 == ZYDIS_REGISTER_RDI) w = {ZeroRdi, 0, 0};
                if (reg0 == ZYDIS_REGISTER_R10D && reg1 == ZYDIS_REGISTER_R10D) w = {ZeroR10, 0, 0};
                break;
            case ZYDIS_MNEMONIC_NOP:
                w = {0xfe, 0, 0}; // skipped
                break;
            case ZYDIS_MNEMONIC_RET:
                w = {Ret, 0, 0};
                break;
            default:
                break;
            }
            if (w.op == 0xff) {
                break; // not understood
            }
            if (w.op != 0xfe) {
                plan.push_back(w);
            }
            pos += ins.length;
            if (w.op == Ret) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            plan.clear();
            st.precheck[4].fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }
    // Run it: rdi starts at the host user-data array (no check); every load through a loaded
    // pointer is a guest read.
    struct Rdi {
        u64 v;
        bool guest;
    };
    Rdi rdi{reinterpret_cast<u64>(user_data), false};
    u64 r10 = 0;
    std::array<Rdi, 64> stack{};
    u32 sp = 0;
    for (const auto& w : plan) {
        switch (w.op) {
        case PushRdi:
            if (sp == stack.size()) {
                st.precheck[4].fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            stack[sp++] = rdi;
            break;
        case PopRdi:
            if (sp == 0) {
                st.precheck[4].fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            rdi = stack[--sp];
            break;
        case LoadRdi: {
            const u64 a = rdi.v + static_cast<s64>(w.disp);
            if (rdi.guest && !ReadOk(a, 8, 0)) {
                return false;
            }
            u64 v;
            std::memcpy(&v, reinterpret_cast<const void*>(a), 8);
            rdi = {v, true};
            break;
        }
        case MovR10Imm:
            r10 = w.imm;
            break;
        case AndRdiR10:
            rdi.v &= r10;
            break;
        case LoadR10: {
            const u64 a = rdi.v + static_cast<s64>(w.disp);
            if (rdi.guest && !ReadOk(a, 4, 0)) {
                return false;
            }
            u32 v;
            std::memcpy(&v, reinterpret_cast<const void*>(a), 4);
            r10 = v;
            break;
        }
        case StoreR10: {
            const u64 i = static_cast<u64>(w.disp) / 4;
            if (w.disp < 0 || i >= flat.size()) {
                st.precheck[5].fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            flat[i] = static_cast<u32>(r10);
            break;
        }
        case ZeroRdi:
            rdi = {0, true};
            break;
        case ZeroR10:
            r10 = 0;
            break;
        case Ret:
            return true;
        default:
            break;
        }
    }
    return true;
}

} // namespace AmdGpu::Lean
