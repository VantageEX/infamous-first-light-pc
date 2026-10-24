// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <magic_enum/magic_enum.hpp>
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
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

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
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

    // With dump_shaders: the decoded GCN instruction list, <stage>_<hash>.gcn.txt, so recompiler
    // problems (e.g. UndefU1 condition flags in the IR) can be traced to instructions.
    if (EmulatorSettings.IsDumpShaders()) {
        using namespace Common::FS;
        const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
        std::filesystem::create_directories(dump_dir);
        const auto name = fmt::format("{}_{:#018x}.gcn.txt", info.stage, info.pgm_hash);
        const IOFile file{dump_dir / name, FileAccessMode::Create, FileType::TextFile};
        const auto operand = [](const Gcn::InstOperand& op) {
            return fmt::format("{}:{:#x}", magic_enum::enum_name(op.field), op.code);
        };
        u32 pc = 0;
        for (const auto& inst : program.ins_list) {
            std::string line = fmt::format("{:05x} {:<28} {}", pc, magic_enum::enum_name(inst.opcode),
                                           magic_enum::enum_name(inst.encoding));
            for (u32 i = 0; i < inst.dst_count; ++i) {
                line += fmt::format(" d{}={}", i, operand(inst.dst[i]));
            }
            for (u32 i = 0; i < inst.src_count; ++i) {
                line += fmt::format(" s{}={}", i, operand(inst.src[i]));
            }
            file.WriteString(line + "\n");
            pc += inst.length;
        }
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};

    // Structurize control flow graph and create program.
    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    // On NVIDIA GPUs HW interpolation of clip distance values seems broken, and we need to emulate
    // it with expensive discard in PS.
    Shader::InjectClipDistanceAttributes(program, runtime_info);

    // Run optimization passes
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    if (info.l_stage == LogicalStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.l_stage == LogicalStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::Optimization::ReadLaneEliminationPass(program);
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    Shader::Optimization::ResourceTrackingPass(program, profile);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::LowerUserClipPlanes(program, runtime_info);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
