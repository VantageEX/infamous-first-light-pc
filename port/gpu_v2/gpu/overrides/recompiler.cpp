// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/perf_stats.h"
#include <string>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <array>
#include "common/logging/classes.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

void EmitControlFlowGraph(IR::Program& program, Pools& pools, Gcn::CFG& cfg,
                          RuntimeInfo& runtime_info, const Profile& profile) {
    Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        auto* ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    translator.EmitPrologue(program.blocks.front());
    ASSERT_MSG(!program.info.translation_failed, "Shader translation has failed");
    for (auto& block : cfg) {
        auto* ir_block = block.ir_block;
        if (block.branch_true) {
            auto* true_block = block.branch_true->ir_block;
            ir_block->AddBranch(true_block);
        }
        if (block.branch_false) {
            auto* false_block = block.branch_false->ir_block;
            ir_block->AddBranch(false_block);
        }
    }
    program.post_order_blocks = Shader::IR::PostOrder(program.blocks.front());
}

// ISS_SHADER_PASSES=1: milliseconds per step of TranslateProgram (averaged, printed every 64).
namespace {
struct PassTimes {
    static constexpr u32 Max = 32;
    std::array<std::atomic<u64>, Max> ns{};
    std::array<const char*, Max> names{};
    std::atomic<u64> programs{};
};
PassTimes g_pass_times;
bool PassTimingEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_SHADER_PASSES");
        return value && value[0] == '1';
    }();
    return enabled;
}
struct PassClock {
    bool on = PassTimingEnabled();
    u64 last = on ? Common::Perf::NowNs() : 0;
    u32 index = 0;
    void Mark(const char* name) {
        if (!on) {
            return;
        }
        const u64 now = Common::Perf::NowNs();
        if (index < PassTimes::Max) {
            g_pass_times.names[index] = name;
            g_pass_times.ns[index].fetch_add(now - last, std::memory_order_relaxed);
        }
        ++index;
        last = now;
    }
    ~PassClock() {
        if (!on) {
            return;
        }
        const u64 n = g_pass_times.programs.fetch_add(1) + 1;
        if ((n & 63) != 0) {
            return;
        }
        std::string line = fmt::format("SHADER_PASSES n={}", n);
        for (u32 i = 0; i < PassTimes::Max && g_pass_times.names[i]; ++i) {
            line += fmt::format(" {}={:.3f}", g_pass_times.names[i],
                                double(g_pass_times.ns[i].load()) / 1e6 / double(n));
        }
        std::fputs(line.c_str(), stderr);
        std::fputc(10, stderr);
    }
};
} // namespace

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    PassClock clock;
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }
    clock.Mark("decode");

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, runtime_info, profile);
    clock.Mark("cfg_emit");

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    Shader::InjectClipDistanceAttributes(program, runtime_info);
    clock.Mark("clipdist");

    // Run optimization passes on unstructured graph
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    clock.Mark("ssa_cp1");
    Shader::Optimization::ReadLaneEliminationPass(program);
    clock.Mark("readlane");
    if (info.sw_stage == SwStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.sw_stage == SwStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    clock.Mark("ring");
    Shader::IR::DumpProgram(program, info, "pre-res-discover.");
    auto resources = Shader::Optimization::ResourceDiscoverPass(program, profile);
    clock.Mark("res_discover");
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    clock.Mark("flatten_ud");
    Shader::IR::DumpProgram(program, info, "pre-res-patch.");
    Shader::Optimization::ResourcePatchingPass(program.info, resources, profile);
    clock.Mark("res_patch");
    Shader::Optimization::LowerBufferFormatToRaw(program);
    clock.Mark("lower_buf");
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    clock.Mark("shared_mem");
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    clock.Mark("clip_planes");
    Shader::Optimization::PhiSimplificationPass(program);
    Shader::Optimization::InverseBallotEliminationPass(program);
    clock.Mark("phi_ballot");
    Shader::IR::DumpProgram(program, info, "pre-lower-phi.");

    // Prepare for structurization by clearing flow graph and lowering phis
    for (auto* ir_block : program.blocks) {
        ir_block->imm_predecessors.clear();
        ir_block->imm_successors.clear();
        ir_block->ssa_state.Reset();
    }
    Shader::Optimization::LowerPhisToRegsPass(program);
    clock.Mark("lower_phis");

    // Structurize control flow graph and create program.
    program.syntax_list = Shader::Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front().data.block);
    clock.Mark("structurize");

    // Run optimization passes on structured graph
    Shader::Optimization::SsaRepairPass(program);
    Shader::Optimization::SsaRewritePass(program);
    clock.Mark("ssa2");
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::WaveMemoryBarrierPass(program, runtime_info, profile);
    clock.Mark("barriers");
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::LowerWave64BallotPass(program, runtime_info, profile);
    Shader::Optimization::LowerHardwareIntrinsics(program);
    clock.Mark("dce_wave_intrin");
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::CollectShaderInfoPass(program, profile);
    clock.Mark("cp2_dce_collect");
    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
