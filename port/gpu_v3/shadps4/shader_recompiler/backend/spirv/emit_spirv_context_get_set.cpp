// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/native_emit.h"
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/ir/patch.h"
#include "shader_recompiler/runtime_info.h"

#include <magic_enum/magic_enum.hpp>

namespace Shader::Backend::SPIRV {

using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

static std::pair<Id, bool> OutputAttrComponentType(EmitContext& ctx, IR::Attribute attr) {
    if (IR::IsParam(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& info{ctx.output_params.at(index)};
        return {info.component_type, info.is_integer};
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        return {info.component_type, info.is_integer};
    }
    switch (attr) {
    case IR::Attribute::Position0:
    case IR::Attribute::ClipDistance:
    case IR::Attribute::CullDistance:
    case IR::Attribute::Depth:
    case IR::Attribute::PointSize:
        return {ctx.F32[1], false};
    case IR::Attribute::RenderTargetIndex:
    case IR::Attribute::ViewportIndex:
    case IR::Attribute::SampleMask:
        return {ctx.U32[1], true};
    case IR::Attribute::StencilRef:
        return {ctx.S32[1], true};
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

Id EmitGetUserData(EmitContext& ctx, IR::ScalarReg reg) {
    const u32 index = ctx.binding.user_data + ctx.info.ud_mask.Index(reg);
    if (ctx.native) {
        // SHADPS4_NATIVE_PASS: this draw's registers, from the records SSBO.
        Id ud_reg{ctx.NativeRecordLoad(Shader::NativeRecordUdOffset + index)};
        if (ctx.native_ring) {
            // Coverage (sec. 11): an inherited register comes from the batch table.
            const Id bit{ctx.OpBitwiseAnd(
                ctx.U32[1],
                ctx.OpShiftRightLogical(ctx.U32[1],
                                        ctx.NativeTableLoad(Shader::NativeTablePushInh),
                                        ctx.ConstU32(index)),
                ctx.ConstU32(1u))};
            ud_reg = ctx.OpSelect(ctx.U32[1], ctx.OpINotEqual(ctx.U1[1], bit, ctx.u32_zero_value),
                                  ctx.NativeTableLoad(Shader::NativeTableUd + index), ud_reg);
        }
        ctx.Name(ud_reg, fmt::format("ud_{}", u32(reg)));
        return ud_reg;
    }
    const u32 half = PushData::UdRegsIndex + (index >> 2);
    const Id ud_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]),
                                      ctx.push_data_block, ctx.ConstU32(half),
                                      ctx.ConstU32(index & 3))};
    const Id ud_reg{ctx.OpLoad(ctx.U32[1], ud_ptr)};
    ctx.Name(ud_reg, fmt::format("ud_{}", u32(reg)));
    return ud_reg;
}

Id EmitReadConst(EmitContext& ctx, IR::Inst* inst, Id addr, Id offset) {
    const u32 flatbuf_off_dw = inst->Flags<u32>();
    // info.uses_dma is false whenever the setting is off, and also for a permutation pinned to
    // a program without DMA (read_const is only defined when uses_dma is set).
    if (!ctx.info.uses_dma) {
        return ctx.EmitFlatbufferLoad(ctx.ConstU32(flatbuf_off_dw));
    }
    // We can only provide a fallback for immediate offsets.
    if (flatbuf_off_dw == 0) {
        return ctx.OpFunctionCall(ctx.U32[1], ctx.read_const_dynamic, addr, offset);
    } else {
        return ctx.OpFunctionCall(ctx.U32[1], ctx.read_const, addr, offset,
                                  ctx.ConstU32(flatbuf_off_dw));
    }
}

// SHADPS4_NATIVE_PT: a Guest binding of a native variant reads through its device address
// (DefineBufferProperties translated the draw's guest address once) instead of the descriptor.
static bool NativePtLoads(EmitContext& ctx, u32 handle);
static Id NativePtLoadDword(EmitContext& ctx, u32 handle, Id byte_off, u32 base_align, Id* shift);

Id EmitReadConstBuffer(EmitContext& ctx, u32 handle, Id index) {
    const auto& buffer = ctx.buffers[handle];
    const Id rel_index = index;
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        index = ctx.OpIAdd(ctx.U32[1], index, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    // SHADPS4_NATIVE_PASS: whole-buffer descriptors; the guest V# bound (Opus code review #3).
    const bool bounded = ctx.native && handle < ctx.native_size_dw.size() &&
                         Sirit::ValidId(ctx.native_size_dw[handle]);
    // SHADPS4_NATIVE_PT (brick C v3 item 2): the scalar load through the page table like the
    // vector loads (EmitLoadBufferB32xN): a segment's members address their own buffers.
    const bool pt = bounded && NativePtLoads(ctx, handle);
    Id result{};
    if (!pt || ctx.native_pt_compare) {
        const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index)};
        result = ctx.OpLoad(ctx.U32[1], ptr);
    }
    if (bounded) {
        const Id in_bounds = ctx.OpULessThan(ctx.U1[1], rel_index, ctx.native_size_dw[handle]);
        if (Sirit::ValidId(result)) {
            result = ctx.OpSelect(ctx.U32[1], in_bounds, result, ctx.u32_zero_value);
        }
        if (pt) {
            const Id ok = ctx.OpLogicalAnd(ctx.U1[1], in_bounds,
                                           ctx.OpLogicalNot(ctx.U1[1], ctx.native_pt_miss[handle]));
            const Id byte_off = ctx.OpSelect(
                ctx.U32[1], ok, ctx.OpShiftLeftLogical(ctx.U32[1], rel_index, ctx.ConstU32(2u)),
                ctx.u32_zero_value);
            const Id bits = ctx.OpSelect(ctx.U32[1], ok,
                                         NativePtLoadDword(ctx, handle, byte_off, 4, nullptr),
                                         ctx.u32_zero_value);
            if (ctx.native_pt_compare) {
                ctx.NativePtMismatch(handle, rel_index, bits, result, 32, true);
            }
            result = bits;
        }
    }
    return result;
}

Id EmitGetAttribute(EmitContext& ctx, IR::Attribute attr, u32 comp, u32 index) {
    if (IR::IsParam(attr)) {
        const u32 param_index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& param{ctx.input_params.at(param_index)};
        if (ctx.native_pull && Sirit::ValidId(ctx.native_attr_value[param_index])) {
            // SHADPS4_NATIVE_COLOUR: the attribute was pulled in the prologue.
            const Id v = ctx.OpCompositeExtract(param.component_type,
                                                ctx.native_attr_value[param_index], comp);
            return param.is_integer ? ctx.OpBitcast(ctx.F32[1], v) : v;
        }
        const Id value = [&] {
            if (param.is_array) {
                ASSERT(param.num_components > 1);
                if (param.is_loaded) {
                    return ctx.OpCompositeExtract(param.component_type, param.id_array[index],
                                                  comp);
                } else {
                    return ctx.OpLoad(param.component_type,
                                      ctx.OpAccessChain(param.pointer_type, param.id,
                                                        ctx.ConstU32(index), ctx.ConstU32(comp)));
                }
            } else {
                ASSERT(!param.is_loaded);
                if (param.num_components > 1) {
                    return ctx.OpLoad(
                        param.component_type,
                        ctx.OpAccessChain(param.pointer_type, param.id, ctx.ConstU32(comp)));
                } else {
                    return ctx.OpLoad(param.component_type, param.id);
                }
            }
        }();
        return param.is_integer ? ctx.OpBitcast(ctx.F32[1], value) : value;
    }
    if (IR::IsBarycentricCoord(attr) && ctx.profile.supports_fragment_shader_barycentric) {
        ++comp;
    }
    switch (attr) {
    case IR::Attribute::Position0:
        ASSERT(ctx.l_stage == LogicalStage::Geometry);
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.gl_in, ctx.ConstU32(index),
                                            ctx.ConstU32(0U), ctx.ConstU32(comp)));
    case IR::Attribute::FragCoord:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.frag_coord, ctx.ConstU32(comp)));
    case IR::Attribute::TessellationEvaluationPointU:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.u32_zero_value));
    case IR::Attribute::TessellationEvaluationPointV:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.ConstU32(1U)));
    case IR::Attribute::BaryCoordSmooth:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordSmoothCentroid:
        return ctx.OpLoad(
            ctx.F32[1],
            ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_centroid, ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordSmoothSample:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_sample,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordNoPersp:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_nopersp,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordNoPerspSample:
        return ctx.OpLoad(
            ctx.F32[1],
            ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_nopersp_sample, ctx.ConstU32(comp)));
    default:
        UNREACHABLE_MSG("Read attribute {}", attr);
    }
}

Id EmitGetAttributeU32(EmitContext& ctx, IR::Attribute attr, u32 comp) {
    switch (attr) {
    case IR::Attribute::VertexId:
        return ctx.OpLoad(ctx.U32[1], ctx.vertex_index);
    case IR::Attribute::InstanceId:
        return ctx.OpLoad(ctx.U32[1], ctx.instance_id);
    case IR::Attribute::WorkgroupIndex:
        return ctx.workgroup_index_id;
    case IR::Attribute::WorkgroupId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), comp);
    case IR::Attribute::LocalInvocationId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id),
                                      comp);
    case IR::Attribute::IsFrontFace:
        return ctx.OpSelect(ctx.U32[1], ctx.OpLoad(ctx.U1[1], ctx.front_facing), ctx.u32_one_value,
                            ctx.u32_zero_value);
    case IR::Attribute::SampleIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.sample_index);
    case IR::Attribute::RenderTargetIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.output_layer);
    case IR::Attribute::PrimitiveId:
        return ctx.OpLoad(ctx.U32[1], ctx.primitive_id);
    case IR::Attribute::InvocationId:
        ASSERT(ctx.info.l_stage == LogicalStage::Geometry ||
               ctx.info.l_stage == LogicalStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
    case IR::Attribute::PatchVertices:
        ASSERT(ctx.info.l_stage == LogicalStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.patch_vertices);
    case IR::Attribute::PackedHullInvocationInfo: {
        ASSERT(ctx.info.l_stage == LogicalStage::TessellationControl);
        // [0:8]: patch id within VGT
        // [8:12]: output control point id
        // But 0:8 should be treated as 0 for attribute addressing purposes
        if (ctx.runtime_info.hs_info.IsPassthrough()) {
            // Gcn shader would run with 1 thread, but we need to run a thread for
            // each output control point.
            // If Gcn shader uses this value, we should make sure all threads in the
            // Vulkan shader use 0
            return ctx.ConstU32(0u);
        } else {
            const Id invocation_id = ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
            return ctx.OpShiftLeftLogical(ctx.U32[1], invocation_id, ctx.ConstU32(8u));
        }
    }
    default:
        UNREACHABLE_MSG("Read U32 attribute {}", attr);
    }
}

void EmitSetAttribute(EmitContext& ctx, IR::Attribute attr, Id value, u32 element) {
    const auto op_store = [&](Id pointer) {
        const auto [component_type, is_integer] = OutputAttrComponentType(ctx, attr);
        if (is_integer) {
            ctx.OpStore(pointer, ctx.OpBitcast(component_type, value));
        } else {
            ctx.OpStore(pointer, value);
        }
    };
    if (IR::IsParam(attr)) {
        const u32 attr_index{u32(attr) - u32(IR::Attribute::Param0)};
        if (ctx.stage == Stage::Local) {
            const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
            return op_store(ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                                              ctx.ConstU32(attr_index), ctx.ConstU32(element)));
        } else {
            const auto& info{ctx.output_params.at(attr_index)};
            ASSERT(info.num_components > 0);
            if (info.num_components == 1) {
                return op_store(info.id);
            } else {
                return op_store(
                    ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
            }
        }
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        if (info.num_components == 1) {
            return op_store(info.id);
        } else {
            return op_store(ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
        }
    }
    switch (attr) {
    case IR::Attribute::Position0:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.output_position, ctx.ConstU32(element)));
    case IR::Attribute::ClipDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.clip_distances, ctx.ConstU32(element)));
    case IR::Attribute::CullDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.cull_distances, ctx.ConstU32(element)));
    case IR::Attribute::PointSize:
        return op_store(ctx.output_point_size);
    case IR::Attribute::RenderTargetIndex:
        return op_store(ctx.output_layer);
    case IR::Attribute::ViewportIndex:
        return op_store(ctx.output_viewport_index);
    case IR::Attribute::Depth:
        return op_store(ctx.frag_depth);
    case IR::Attribute::SampleMask:
        return op_store(ctx.OpAccessChain(ctx.output_u32, ctx.sample_mask, ctx.u32_zero_value));
    case IR::Attribute::StencilRef:
        if (ctx.profile.supports_shader_stencil_export) {
            return op_store(ctx.stencil_ref);
        }
        return;
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

Id EmitGetTessGenericAttribute(EmitContext& ctx, Id vertex_index, Id attr_index, Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Input, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.input_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

Id EmitReadTcsGenericOuputAttribute(EmitContext& ctx, Id vertex_index, Id attr_index,
                                    Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.output_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

void EmitSetTcsGenericAttribute(EmitContext& ctx, Id value, Id attr_index, Id comp_index) {
    // Implied vertex index is invocation_id
    const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    Id pointer =
        ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                          ctx.OpLoad(ctx.U32[1], ctx.invocation_id), attr_index, comp_index);
    ctx.OpStore(pointer, value);
}

Id EmitGetPatch(EmitContext& ctx, IR::Patch patch) {
    const u32 index{IR::GenericPatchIndex(patch)};
    const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
    const Id type{ctx.l_stage == LogicalStage::TessellationControl ? ctx.output_f32
                                                                   : ctx.input_f32};
    const Id pointer{ctx.OpAccessChain(type, ctx.patches.at(index), element)};
    return ctx.OpLoad(ctx.F32[1], pointer);
}

void EmitSetPatch(EmitContext& ctx, IR::Patch patch, Id value) {
    const Id pointer{[&] {
        if (IR::IsGeneric(patch)) {
            const u32 index{IR::GenericPatchIndex(patch)};
            const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
            return ctx.OpAccessChain(ctx.output_f32, ctx.patches.at(index), element);
        }
        switch (patch) {
        case IR::Patch::TessellationLodLeft:
        case IR::Patch::TessellationLodRight:
        case IR::Patch::TessellationLodTop:
        case IR::Patch::TessellationLodBottom: {
            const u32 index{static_cast<u32>(patch) - u32(IR::Patch::TessellationLodLeft)};
            const Id index_id{ctx.ConstU32(index)};
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_outer, index_id);
        }
        case IR::Patch::TessellationLodInteriorU:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner,
                                     ctx.u32_zero_value);
        case IR::Patch::TessellationLodInteriorV:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner, ctx.ConstU32(1u));
        default:
            UNREACHABLE_MSG("Patch {}", u32(patch));
        }
    }()};
    ctx.OpStore(pointer, value);
}

static bool NativePtLoads(EmitContext& ctx, u32 handle) {
    return ctx.native_pt && handle < ctx.native_dev_base.size() &&
           Sirit::ValidId(ctx.native_dev_base[handle]);
}

// The aligned dword holding byte `byte_off` of the binding (0 when out of bounds or the page is
// missing: the pointer then stays inside mapped memory and the caller zeroes the result), from
// the binding's device address rounded down to `base_align` bytes - what the descriptor path's
// element index does to the binding's offset (dword index for 32-bit loads, halfword for
// 16-bit). `shift` (optional): the bit position of the byte inside the dword.
static Id NativePtLoadDword(EmitContext& ctx, u32 handle, Id byte_off, u32 base_align,
                            Id* shift) {
    const Id base = ctx.OpBitwiseAnd(ctx.U64, ctx.native_dev_base[handle],
                                     ctx.Constant(ctx.U64, ~u64{base_align - 1}));
    const Id addr = ctx.OpIAdd(ctx.U64, base, ctx.OpUConvert(ctx.U64, byte_off));
    const Id dword_addr = ctx.OpBitwiseAnd(ctx.U64, addr, ctx.Constant(ctx.U64, ~u64{3}));
    if (shift) {
        *shift = ctx.OpShiftLeftLogical(
            ctx.U32[1],
            ctx.OpBitwiseAnd(ctx.U32[1], ctx.OpUConvert(ctx.U32[1], addr), ctx.ConstU32(3u)),
            ctx.ConstU32(3u));
    }
    const Id ptr = ctx.OpConvertUToPtr(ctx.physical_pointer_type_u32, dword_addr);
    return ctx.OpLoad(ctx.U32[1], ptr, spv::MemoryAccessMask::Aligned, 4u);
}

template <u32 N, PointerType alias>
static Id EmitLoadBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    const Id rel_address = address;
    if (const Id offset = spv_buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);
    // SHADPS4_NATIVE_PASS: whole-buffer descriptors; the guest V# bound is applied here.
    const Id size_dw = ctx.native && handle < ctx.native_size_dw.size() ? ctx.native_size_dw[handle]
                                                                        : Id{};
    const bool pt = NativePtLoads(ctx, handle);

    boost::container::static_vector<Id, N> ids;
    for (u32 i = 0; i < N; i++) {
        const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
        Id result_i{};
        if (!pt || ctx.native_pt_compare) {
            const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
            result_i = ctx.OpLoad(data_types[1], ptr_i);
        }
        if (Sirit::ValidId(size_dw)) {
            const Id rel_i =
                i == 0 ? rel_address : ctx.OpIAdd(ctx.U32[1], rel_address, ctx.ConstU32(i));
            const Id in_bounds = ctx.OpULessThan(ctx.U1[1], rel_i, size_dw);
            const Id zero = alias == PointerType::U32 ? ctx.u32_zero_value : ctx.f32_zero_value;
            if (Sirit::ValidId(result_i)) {
                result_i = ctx.OpSelect(data_types[1], in_bounds, result_i, zero);
            }
            if (pt) {
                // SHADPS4_NATIVE_PT: the same dword through the page table.
                const Id ok = ctx.OpLogicalAnd(ctx.U1[1], in_bounds,
                                               ctx.OpLogicalNot(ctx.U1[1], ctx.native_pt_miss[handle]));
                const Id byte_off = ctx.OpSelect(
                    ctx.U32[1], ok, ctx.OpShiftLeftLogical(ctx.U32[1], rel_i, ctx.ConstU32(2u)),
                    ctx.u32_zero_value);
                const Id bits = ctx.OpSelect(ctx.U32[1], ok,
                                             NativePtLoadDword(ctx, handle, byte_off, 4, nullptr),
                                             ctx.u32_zero_value);
                if (ctx.native_pt_compare) {
                    const Id desc_bits = alias == PointerType::U32
                                             ? result_i
                                             : ctx.OpBitcast(ctx.U32[1], result_i);
                    ctx.NativePtMismatch(handle, rel_i, bits, desc_bits, 32);
                }
                result_i = alias == PointerType::U32 ? bits : ctx.OpBitcast(ctx.F32[1], bits);
            }
        }
        ids.push_back(result_i);
    }

    const Id result = N == 1 ? ids[0] : ctx.OpCompositeConstruct(data_types[N], ids);
    return result;
}

// SHADPS4_NATIVE_COLOUR: whole-buffer descriptors; the guest V# bound for 8 / 16-bit loads
// (`shift` = log2 of the load size in bytes), as EmitLoadBufferB32xN does for 32-bit loads; in
// bytes, so the last 1-3 bytes of a binding (and bindings under 4 bytes) stay readable
// (s306-s309: a dword-rounded bound zeroed them - speckles on the hero's jeans).
static Id NativeNarrowInBounds(EmitContext& ctx, u32 handle, Id rel_address, u32 shift) {
    if (!ctx.native || handle >= ctx.native_size_bytes.size() ||
        !Sirit::ValidId(ctx.native_size_bytes[handle])) {
        return Id{};
    }
    const Id size_units = shift == 0 ? ctx.native_size_bytes[handle]
                                     : ctx.OpShiftRightLogical(ctx.U32[1],
                                                               ctx.native_size_bytes[handle],
                                                               ctx.ConstU32(shift));
    return ctx.OpULessThan(ctx.U1[1], rel_address, size_units);
}

// SHADPS4_NATIVE_PT: an 8 / 16-bit load (`shift` = log2 of its size) through the page table:
// the containing aligned dword, the byte / halfword extracted (no 8 / 16-bit physical pointer
// capability needed). `desc` (compare mode): the descriptor load's value, zero-extended.
static Id NativePtLoadNarrow(EmitContext& ctx, u32 handle, Id rel_address, Id in_bounds,
                             u32 shift, Id desc) {
    const Id ok = ctx.OpLogicalAnd(ctx.U1[1], in_bounds,
                                   ctx.OpLogicalNot(ctx.U1[1], ctx.native_pt_miss[handle]));
    const Id rel_bytes =
        shift == 0 ? rel_address
                   : ctx.OpShiftLeftLogical(ctx.U32[1], rel_address, ctx.ConstU32(shift));
    const Id byte_off = ctx.OpSelect(ctx.U32[1], ok, rel_bytes, ctx.u32_zero_value);
    Id bit_pos{};
    const Id dword = NativePtLoadDword(ctx, handle, byte_off, 1u << shift, &bit_pos);
    const Id bits = ctx.OpSelect(
        ctx.U32[1], ok, ctx.OpBitFieldUExtract(ctx.U32[1], dword, bit_pos, ctx.ConstU32(8u << shift)),
        ctx.u32_zero_value);
    if (ctx.native_pt_compare) {
        ctx.NativePtMismatch(handle, rel_address, bits, desc, 8u << shift);
    }
    return bits;
}

Id EmitLoadBufferU8(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    const Id in_bounds = NativeNarrowInBounds(ctx, handle, address, 0);
    const Id rel_address = address;
    if (const Id offset = spv_buffer.Offset(PointerSize::B8); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const bool pt = NativePtLoads(ctx, handle) && Sirit::ValidId(in_bounds);
    Id result{};
    if (!pt || ctx.native_pt_compare) {
        const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U8);
        const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
        result = ctx.OpLoad(ctx.U8, ptr);
        if (Sirit::ValidId(in_bounds)) {
            result = ctx.OpSelect(ctx.U8, in_bounds, result, ctx.Constant(ctx.U8, 0U));
        }
    }
    if (pt) {
        const Id desc = Sirit::ValidId(result) ? ctx.OpUConvert(ctx.U32[1], result) : Id{};
        result = ctx.OpUConvert(ctx.U8,
                                NativePtLoadNarrow(ctx, handle, rel_address, in_bounds, 0, desc));
    }
    return result;
}

Id EmitLoadBufferU16(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    const Id in_bounds = NativeNarrowInBounds(ctx, handle, address, 1);
    const Id rel_address = address;
    if (const Id offset = spv_buffer.Offset(PointerSize::B16); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const bool pt = NativePtLoads(ctx, handle) && Sirit::ValidId(in_bounds);
    Id result{};
    if (!pt || ctx.native_pt_compare) {
        const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U16);
        const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
        result = ctx.OpLoad(ctx.U16, ptr);
        if (Sirit::ValidId(in_bounds)) {
            result = ctx.OpSelect(ctx.U16, in_bounds, result, ctx.Constant(ctx.U16, 0U));
        }
    }
    if (pt) {
        const Id desc = Sirit::ValidId(result) ? ctx.OpUConvert(ctx.U32[1], result) : Id{};
        result = ctx.OpUConvert(ctx.U16,
                                NativePtLoadNarrow(ctx, handle, rel_address, in_bounds, 1, desc));
    }
    return result;
}

Id EmitLoadBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U64);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u64_zero_value, address)};
    const Id result{ctx.OpLoad(ctx.U64, ptr)};
    return result;
}

Id EmitLoadBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

template <u32 N, PointerType alias>
static void EmitStoreBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                                 Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);

    for (u32 i = 0; i < N; i++) {
        const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
        const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
        const Id value_i = N == 1 ? value : ctx.OpCompositeExtract(data_types[1], value, i);
        ctx.OpStore(ptr_i, value_i);
    }
}

void EmitStoreBufferU8(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B8); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U8);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    ctx.OpStore(ptr, value);
}

void EmitStoreBufferU16(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B16); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U16);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    ctx.OpStore(ptr, value);
}

void EmitStoreBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U64);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u64_zero_value, address)};
    ctx.OpStore(ptr, value);
}

void EmitStoreBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

void EmitGetThreadBitScalarReg(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetThreadBitScalarReg(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetMaskLaneVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetMaskLaneVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

} // namespace Shader::Backend::SPIRV
