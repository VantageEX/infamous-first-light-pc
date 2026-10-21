// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_set>
#include <vector>
#include "common/logging/log.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsWriteShared(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    if (opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>();
    }
    return opcode == IR::Opcode::WriteSharedU16 || opcode == IR::Opcode::WriteSharedU32 ||
           opcode == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
// Returns the number of barriers inserted.
static u32 EmitBarrierInBlock(IR::Block* block) {
    u32 num_barriers{};
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
                ++num_barriers;
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
                ++num_barriers;
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
        ++num_barriers;
    }
    return num_barriers;
}

using NodeSet = std::unordered_set<const IR::Block*>;

static void EmitBarrierAtBlockStart(IR::Block* block) {
    auto insert_point = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
    IR::IREmitter ir{*block, insert_point};
    ir.Barrier();
}

static bool IsDivergent(const IR::U1& cond) {
    return IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
               if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                   inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
                   return true;
               }
               return std::nullopt;
           }) == true;
}

// Values that can differ between the invocations of a workgroup spanning several host
// subgroups: invocation ids, subgroup operations (only uniform within one subgroup) and atomics
// (each invocation receives its own previous value).
static bool IsVaryingSource(const IR::Inst* inst) {
    const IR::Opcode opcode = inst->GetOpcode();
    if ((opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) ||
        (opcode >= IR::Opcode::BufferAtomicIAdd32 &&
         opcode <= IR::Opcode::BufferAtomicFCmpSwap32) ||
        (opcode >= IR::Opcode::ImageAtomicIAdd32 && opcode <= IR::Opcode::ImageAtomicCmpSwap32)) {
        return true;
    }
    switch (opcode) {
    case IR::Opcode::GetAttributeU32: {
        const IR::Attribute attribute = inst->Arg(0).Attribute();
        return attribute == IR::Attribute::LocalInvocationId ||
               attribute == IR::Attribute::LocalInvocationIndex ||
               attribute == IR::Attribute::GlobalInvocationId ||
               attribute == IR::Attribute::SubgroupLtMask;
    }
    case IR::Opcode::LaneId:
    case IR::Opcode::Ballot:
    case IR::Opcode::BallotFindLsb:
    case IR::Opcode::ReadLane:
    case IR::Opcode::ReadFirstLane:
    case IR::Opcode::WriteLane:
    case IR::Opcode::QuadBroadcast:
    case IR::Opcode::MaskedBitCount32:
    case IR::Opcode::DataAppend:
    case IR::Opcode::DataConsume:
        return true;
    default:
        return false;
    }
}

static bool IsDivergentStrict(const IR::U1& cond) {
    return IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
               if (IsVaryingSource(inst)) {
                   return true;
               }
               return std::nullopt;
           }) == true;
}

static bool IsDivergent(const IR::U1& cond, bool strict) {
    return strict ? IsDivergentStrict(cond) : IsDivergent(cond);
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static u32 EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                   NodeSet& divergence_end, u32& divergence_depth, bool strict) {
    const IR::U1 cond = data.if_node.cond;
    u32 num_barriers{};
    if (IsDivergent(cond, strict)) {
        if (divergence_depth == 0) {
            EmitBarrierAtBlockStart(data.if_node.merge);
            ++num_barriers;
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
    return num_barriers;
}

// A barrier inside a loop is invalid when different invocations leave on different iterations.
// Mark such loops so their shared-memory synchronization can be deferred to the merge block.
static NodeSet FindDivergentLoops(const IR::AbstractSyntaxList& syntax_list, bool strict) {
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (IsDivergent(node.data.repeat.cond, strict)) {
                divergent_loops.emplace(node.data.repeat.merge);
            }
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (IsDivergent(node.data.break_node.cond, strict)) {
                divergent_loops.emplace(node.data.break_node.merge);
            }
            break;
        default:
            break;
        }
    }
    return divergent_loops;
}

static constexpr u32 GcnSubgroupSize = 64;

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    if (shared_memory_size == 0 || !profile.needs_lds_barriers) {
        return;
    }
    // The compiler can only omit barriers when the local workgroup size is the same as the HW
    // subgroup: its 64 invocations run in lockstep on GCN, not on hosts with smaller subgroups.
    const bool single_wave = threadgroup_size == GcnSubgroupSize;
    // Workgroups of several GCN waves synchronize their waves with barriers, but code may still
    // rely on the 64 invocations of each wave running in lockstep to share data in LDS without
    // barriers (sorting and reduction steps shorter than a wave). Hosts with smaller subgroups
    // split each wave, so those steps race: synchronize the uniform parts of such shaders too.
    const bool multi_wave = profile.lds_barriers_multi_wave &&
                            threadgroup_size > GcnSubgroupSize &&
                            threadgroup_size > profile.subgroup_size;
    if (!single_wave && !multi_wave) {
        return;
    }
    // Conditions only uniform within a subgroup are divergent in multi wave workgroups.
    const bool strict = multi_wave;
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    u32 num_barriers{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program.syntax_list, strict);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            num_barriers +=
                EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth, strict);
            continue;
        }
        if (node.type == Type::Loop && divergent_loops.contains(node.data.loop.merge)) {
            ++divergence_depth;
            continue;
        }
        if (node.type == Type::Repeat && divergent_loops.contains(node.data.repeat.merge)) {
            ASSERT(divergence_depth > 0);
            --divergence_depth;
            if (divergence_depth == 0) {
                EmitBarrierAtBlockStart(node.data.repeat.merge);
                ++num_barriers;
            }
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0) {
            num_barriers += EmitBarrierInBlock(node.data.block);
        }
    }
    if (multi_wave && num_barriers != 0) {
        LOG_INFO(Render_Recompiler,
                 "Compute shader {:#x}: {} LDS barriers for {} invocations ({} bytes of LDS) "
                 "sharing data within GCN waves",
                 program.info.pgm_hash, num_barriers, threadgroup_size, shared_memory_size);
    }
}

static bool IsLoadBuffer(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    return opcode >= IR::Opcode::LoadBufferU8 && opcode <= IR::Opcode::LoadBufferFormatF32;
}

static bool IsStoreBuffer(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    return opcode >= IR::Opcode::StoreBufferU8 && opcode <= IR::Opcode::StoreBufferFormatF32;
}

// Instructions whose order with the barriers of a workgroup matters: memory accesses and
// barriers. Code after a conditional block without them cannot observe which invocations ran it.
static bool HasOrderedInst(const IR::Block& block) {
    for (const IR::Inst& inst : block.Instructions()) {
        const IR::Opcode opcode = inst.GetOpcode();
        if (IsLoadBuffer(inst) || IsStoreBuffer(inst) || IsLoadShared(inst) ||
            IsWriteShared(inst) || opcode == IR::Opcode::Barrier ||
            opcode == IR::Opcode::WorkgroupMemoryBarrier ||
            opcode == IR::Opcode::DeviceMemoryBarrier ||
            (opcode >= IR::Opcode::SharedAtomicIAdd32 &&
             opcode <= IR::Opcode::SharedAtomicCmpSwap64) ||
            (opcode >= IR::Opcode::BufferAtomicIAdd32 &&
             opcode <= IR::Opcode::BufferAtomicFCmpSwap32) ||
            (opcode >= IR::Opcode::ImageAtomicIAdd32 &&
             opcode <= IR::Opcode::ImageAtomicCmpSwap32) ||
            opcode == IR::Opcode::ImageWrite || opcode == IR::Opcode::DataAppend ||
            opcode == IR::Opcode::DataConsume) {
            return true;
        }
    }
    return false;
}

// GCN waves run their 64 invocations in lockstep: within a wave, code can exchange data through
// buffer memory without barriers, ordering its stores before the loads of the next step with
// s_waitcnt only (for example sorting passes merging runs no longer than a wave in a loop).
// Hosts with smaller subgroups split each wave into parts that run independently, and the
// compiler may move the loads of a step before the stores of the previous one: the steps race.
// Workgroups spanning several host subgroups are synchronized after the buffer stores of their
// uniform loops that also load buffer memory. Invocations that skipped such a loop with a
// condition that leads to the end of the shader do not take part in the barriers (on hosts
// that treat exited invocations as arrived, which is the case of the drivers that need this).
void WaveMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile) {
    if (!profile.wave_memory_barriers || program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    if (threadgroup_size <= profile.subgroup_size) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    const IR::AbstractSyntaxList& list = program.syntax_list;
    const size_t num_nodes = list.size();

    // Matching structure nodes: the EndIf of each If, the Repeat of each Loop (by merge block).
    std::vector<size_t> end_of(num_nodes, num_nodes);
    {
        std::vector<size_t> open;
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = list[i];
            if (node.type == Type::If || node.type == Type::Loop) {
                open.push_back(i);
            } else if (node.type == Type::EndIf || node.type == Type::Repeat) {
                if (open.empty()) {
                    return;
                }
                const size_t start = open.back();
                open.pop_back();
                const auto& start_node = list[start];
                const bool matches =
                    node.type == Type::EndIf
                        ? start_node.type == Type::If &&
                              start_node.data.if_node.merge == node.data.end_if.merge
                        : start_node.type == Type::Loop &&
                              start_node.data.loop.merge == node.data.repeat.merge;
                if (!matches) {
                    // Unexpected nesting: leave the shader as it is.
                    return;
                }
                end_of[start] = i;
            }
        }
        if (!open.empty()) {
            return;
        }
    }
    // An If is a tail if when nothing after its EndIf accesses memory or synchronizes: the
    // invocations that do not take it only leave the shader.
    std::vector<bool> inert_suffix(num_nodes + 1, true);
    for (size_t i = num_nodes; i-- > 0;) {
        const auto& node = list[i];
        bool inert = inert_suffix[i + 1];
        switch (node.type) {
        case Type::Block:
            inert = inert && !HasOrderedInst(*node.data.block);
            break;
        case Type::EndIf:
        case Type::Return:
        case Type::Unreachable:
            break;
        default:
            inert = false;
            break;
        }
        inert_suffix[i] = inert;
    }
    const auto loads_in_range = [&](size_t first, size_t last) {
        for (size_t i = first; i <= last && i < num_nodes; ++i) {
            if (list[i].type != Type::Block) {
                continue;
            }
            for (const IR::Inst& inst : list[i].data.block->Instructions()) {
                if (IsLoadBuffer(inst)) {
                    return true;
                }
            }
        }
        return false;
    };
    const NodeSet divergent_loops = FindDivergentLoops(list, true);

    struct Scope {
        size_t end;
        bool divergent;    ///< Invocations of the workgroup may take different paths in it.
        bool uniform_loop; ///< A loop all invocations iterate together.
        bool loop_loads;   ///< The loop body loads buffer memory.
    };
    std::vector<Scope> scopes;
    u32 num_barriers{};
    for (size_t i = 0; i < num_nodes; ++i) {
        while (!scopes.empty() && scopes.back().end < i) {
            scopes.pop_back();
        }
        const auto& node = list[i];
        if (node.type == Type::If) {
            const size_t end = end_of[i];
            const bool tail = end < num_nodes && inert_suffix[end + 1];
            const bool divergent = !tail && IsDivergentStrict(node.data.if_node.cond);
            scopes.push_back({end, divergent, false, false});
            continue;
        }
        if (node.type == Type::Loop) {
            const size_t end = end_of[i];
            const bool divergent = divergent_loops.contains(node.data.loop.merge);
            scopes.push_back({end, divergent, !divergent, !divergent && loads_in_range(i, end)});
            continue;
        }
        if (node.type != Type::Block) {
            continue;
        }
        bool divergent = false;
        bool in_loop_with_loads = false;
        for (const auto& scope : scopes) {
            divergent |= scope.divergent;
        }
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->uniform_loop) {
                in_loop_with_loads = it->loop_loads;
                break;
            }
        }
        if (divergent || !in_loop_with_loads) {
            continue;
        }
        IR::Block* const block = node.data.block;
        // A barrier after each store followed by a load in the block, and after the last store.
        bool pending_store = false;
        for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
            if (IsLoadBuffer(*it) && pending_store) {
                IR::IREmitter ir{*block, it};
                ir.WorkgroupMemoryBarrier();
                ir.Barrier();
                ++num_barriers;
                pending_store = false;
            }
            if (IsStoreBuffer(*it)) {
                pending_store = true;
            }
        }
        if (pending_store) {
            auto last_store = block->Instructions().end();
            for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
                if (IsStoreBuffer(*it)) {
                    last_store = it;
                }
            }
            IR::IREmitter ir{*block, std::next(last_store)};
            ir.WorkgroupMemoryBarrier();
            ir.Barrier();
            ++num_barriers;
        }
    }
    if (num_barriers != 0) {
        LOG_INFO(Render_Recompiler,
                 "Compute shader {:#x}: {} buffer memory barriers for {} invocations exchanging "
                 "data within GCN waves",
                 program.info.pgm_hash, num_barriers, threadgroup_size);
    }
}

} // namespace Shader::Optimization
