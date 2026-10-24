// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/div_ceil.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/native_emit.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/buffer_cache/buffer_cache.h"

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include <numbers>
#include <string_view>

namespace Shader::Backend::SPIRV {
namespace {

std::string_view StageName(Stage stage) {
    switch (stage) {
    case Stage::Vertex:
        return "vs";
    case Stage::Local:
        return "ls";
    case Stage::Export:
        return "es";
    case Stage::Hull:
        return "hs";
    case Stage::Geometry:
        return "gs";
    case Stage::Fragment:
        return "fs";
    case Stage::Compute:
        return "cs";
    }
    UNREACHABLE_MSG("Invalid hw stage {}", u32(stage));
}

static constexpr u32 NumVertices(AmdGpu::PrimitiveType type) {
    switch (type) {
    case AmdGpu::PrimitiveType::PointList:
        return 1u;
    case AmdGpu::PrimitiveType::LineList:
    case AmdGpu::PrimitiveType::LineStrip:
        return 2u;
    case AmdGpu::PrimitiveType::TriangleList:
    case AmdGpu::PrimitiveType::TriangleStrip:
    case AmdGpu::PrimitiveType::RectList:
        return 3u;
    case AmdGpu::PrimitiveType::AdjTriangleList:
        return 6u;
    case AmdGpu::PrimitiveType::AdjLineList:
        return 4u;
    default:
        UNREACHABLE();
    }
}

template <typename... Args>
void Name(EmitContext& ctx, Id object, std::string_view format_str, Args&&... args) {
    ctx.Name(object, fmt::format(fmt::runtime(format_str), StageName(ctx.stage),
                                 std::forward<Args>(args)...)
                         .c_str());
}

} // Anonymous namespace

EmitContext::EmitContext(const Profile& profile_, const RuntimeInfo& runtime_info_, Info& info_,
                         Bindings& binding_)
    : Sirit::Module(profile_.supported_spirv), info{info_}, runtime_info{runtime_info_},
      profile{profile_}, stage{info.stage}, l_stage{info.l_stage}, binding{binding_} {
    native = Shader::t_native_emit.active &&
             (info.l_stage == LogicalStage::Vertex ||
              (Shader::t_native_emit.colour && info.l_stage == LogicalStage::Fragment));
    native_colour = native && Shader::t_native_emit.colour;
    native_pull = native && Shader::t_native_emit.pull && info.l_stage == LogicalStage::Vertex;
    native_ring = native && Shader::t_native_emit.ring;
    native_bindless = native_colour && Shader::t_native_emit.bindless &&
                      (info.l_stage == LogicalStage::Fragment ||
                       (info.l_stage == LogicalStage::Vertex && Shader::t_native_emit.bindless_vs));
    bindless_smp_base =
        info.l_stage == LogicalStage::Vertex ? Shader::t_native_emit.vs_smp_base : 0u; // q3
    native_pt = native_ring && Shader::t_native_emit.pt;
    native_pt_compare = native_pt && Shader::t_native_emit.pt_compare;
    native_seg = native_pt && Shader::t_native_emit.seg;
    native_seg_compare = native_seg && Shader::t_native_emit.seg_compare;
    if (info.uses_dma || native_pt) {
        SetMemoryModel(spv::AddressingModel::PhysicalStorageBuffer64, spv::MemoryModel::GLSL450);
    } else {
        SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
    }
    String(fmt::format("{:#x}", info.pgm_hash));

    AddCapability(spv::Capability::Shader);
    DefineArithmeticTypes();
    DefineInterfaces();
    DefineSharedMemory();
    DefineBuffers();
    DefineImagesAndSamplers();
    DefineFunctions();
}

EmitContext::~EmitContext() = default;

Id EmitContext::Def(const IR::Value& value) {
    if (!value.IsImmediate()) {
        return value.InstRecursive()->Definition<Id>();
    }
    switch (value.Type()) {
    case IR::Type::Void:
        return Id{};
    case IR::Type::U1:
        return value.U1() ? true_value : false_value;
    case IR::Type::U32:
        return ConstU32(value.U32());
    case IR::Type::U64:
        return Constant(U64, value.U64());
    case IR::Type::F32:
        return ConstF32(value.F32());
    case IR::Type::F64:
        return Constant(F64[1], value.F64());
    case IR::Type::StringLiteral:
        return String(value.StringLiteral());
    default:
        UNREACHABLE_MSG("Immediate type {}", value.Type());
    }
}

void EmitContext::DefineArithmeticTypes() {
    void_id = Name(TypeVoid(), "void_id");
    U1[1] = Name(TypeBool(), "bool_id");
    U8 = Name(TypeUInt(8), "u8_id");
    S8 = Name(TypeSInt(8), "i8_id");
    U16 = Name(TypeUInt(16), "u16_id");
    S16 = Name(TypeSInt(16), "i16_id");
    if (info.uses_fp16) {
        F16[1] = Name(TypeFloat(16), "f16_id");
    }
    if (info.uses_fp64) {
        F64[1] = Name(TypeFloat(64), "f64_id");
    }
    F32[1] = Name(TypeFloat(32), "f32_id");
    S32[1] = Name(TypeSInt(32), "i32_id");
    U32[1] = Name(TypeUInt(32), "u32_id");
    U64 = Name(TypeUInt(64), "u64_id");

    for (u32 i = 2; i <= 4; i++) {
        if (info.uses_fp16) {
            F16[i] = Name(TypeVector(F16[1], i), fmt::format("f16vec{}_id", i));
        }
        if (info.uses_fp64) {
            F64[i] = Name(TypeVector(F64[1], i), fmt::format("f64vec{}_id", i));
        }
        F32[i] = Name(TypeVector(F32[1], i), fmt::format("f32vec{}_id", i));
        S32[i] = Name(TypeVector(S32[1], i), fmt::format("i32vec{}_id", i));
        U32[i] = Name(TypeVector(U32[1], i), fmt::format("u32vec{}_id", i));
        U1[i] = Name(TypeVector(U1[1], i), fmt::format("bvec{}_id", i));
    }

    true_value = ConstantTrue(U1[1]);
    false_value = ConstantFalse(U1[1]);
    u8_one_value = Constant(U8, 1U);
    u8_zero_value = Constant(U8, 0U);
    u16_zero_value = Constant(U16, 0U);
    u32_one_value = ConstU32(1U);
    u32_zero_value = ConstU32(0U);
    f32_zero_value = ConstF32(0.0f);
    u64_one_value = Constant(U64, 1ULL);
    u64_zero_value = Constant(U64, 0ULL);

    pi_x2 = ConstF32(2.0f * float{std::numbers::pi});

    input_f32 = Name(TypePointer(spv::StorageClass::Input, F32[1]), "input_f32");
    input_u32 = Name(TypePointer(spv::StorageClass::Input, U32[1]), "input_u32");
    input_s32 = Name(TypePointer(spv::StorageClass::Input, S32[1]), "input_s32");

    output_f32 = Name(TypePointer(spv::StorageClass::Output, F32[1]), "output_f32");
    output_u32 = Name(TypePointer(spv::StorageClass::Output, U32[1]), "output_u32");
    output_s32 = Name(TypePointer(spv::StorageClass::Output, S32[1]), "output_s32");

    full_result_i32x2 = Name(TypeStruct(S32[1], S32[1]), "full_result_i32x2");
    full_result_u32x2 = Name(TypeStruct(U32[1], U32[1]), "full_result_u32x2");
    frexp_result_f32 = Name(TypeStruct(F32[1], S32[1]), "frexp_result_f32");
    if (info.uses_fp64) {
        frexp_result_f64 = Name(TypeStruct(F64[1], S32[1]), "frexp_result_f64");
    }
    if (info.uses_dma || native_pt) {
        physical_pointer_type_u32 = TypePointer(spv::StorageClass::PhysicalStorageBuffer, U32[1]);
    }
    if (native_pt) {
        physical_pointer_type_u64 = TypePointer(spv::StorageClass::PhysicalStorageBuffer, U64);
    }
}

void EmitContext::DefineInterfaces() {
    DefinePushDataBlock();
    DefineInputs();
    DefineOutputs();
}

const VectorIds& GetAttributeType(EmitContext& ctx, AmdGpu::NumberFormat fmt) {
    switch (GetNumberClass(fmt)) {
    case AmdGpu::NumberClass::Float:
        return ctx.F32;
    case AmdGpu::NumberClass::Sint:
        return ctx.S32;
    case AmdGpu::NumberClass::Uint:
        return ctx.U32;
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid attribute type {}", fmt);
}

EmitContext::SpirvAttribute EmitContext::GetAttributeInfo(AmdGpu::NumberFormat fmt, Id id,
                                                          u32 num_components, bool output,
                                                          bool loaded, bool array) {
    switch (GetNumberClass(fmt)) {
    case AmdGpu::NumberClass::Float:
        return {id, output ? output_f32 : input_f32, F32[1], num_components, false, loaded, array};
    case AmdGpu::NumberClass::Uint:
        return {id, output ? output_u32 : input_u32, U32[1], num_components, true, loaded, array};
    case AmdGpu::NumberClass::Sint:
        return {id, output ? output_s32 : input_s32, S32[1], num_components, true, loaded, array};
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid attribute type {}", fmt);
}

Id EmitContext::GetBufferSize(const u32 sharp_idx) {
    // Can this be done with memory access? Like we do now with ReadConst
    const auto& srt_flatbuf = buffers[flatbuf_index];
    ASSERT(srt_flatbuf.buffer_type == BufferType::Flatbuf);
    const auto [id, pointer_type] = srt_flatbuf.Alias(PointerType::U32);

    const auto rsrc1{
        OpLoad(U32[1], OpAccessChain(pointer_type, id, u32_zero_value, ConstU32(sharp_idx + 1)))};
    const auto rsrc2{
        OpLoad(U32[1], OpAccessChain(pointer_type, id, u32_zero_value, ConstU32(sharp_idx + 2)))};

    const auto stride{OpBitFieldUExtract(U32[1], rsrc1, ConstU32(16u), ConstU32(14u))};
    const auto num_records{rsrc2};

    const auto stride_zero{OpIEqual(U1[1], stride, u32_zero_value)};
    const auto stride_size{OpIMul(U32[1], num_records, stride)};
    return OpSelect(U32[1], stride_zero, num_records, stride_size);
}

void EmitContext::DefineBufferProperties() {
    if (native) {
        Id draw;
        if (l_stage == LogicalStage::Fragment) {
            draw = OpLoad(U32[1], native_draw_in);
        } else {
            draw = OpLoad(U32[1], draw_index);
            if (native_colour) {
                OpStore(native_draw_out, draw);
            }
        }
        if (native_seg) {
            // SHADPS4_NATIVE_SEG: the record by pointer; the SSBO index below only for =2.
            NativeSegRecordPointer(draw);
        }
        if (native_ring && (!native_seg || native_seg_compare)) {
            // SHADPS4_NATIVE_GPU_DIRECT: entry (first + draw) & mask of the records buffer.
            draw = OpBitwiseAnd(U32[1],
                                OpIAdd(U32[1], NativeTableLoad(Shader::NativeTableFirst), draw),
                                NativeTableLoad(Shader::NativeTableMask));
        }
        native_rec_base = OpIMul(U32[1], draw, ConstU32(Shader::NativeRecordDwords));
        Name(native_rec_base, "native_rec_base");
        native_draw = draw;
        if (native_pull && Shader::t_native_emit.compare) {
            const Id vindex{OpLoad(U32[1], vertex_index)};
            for (u32 loc = 0; loc < IR::NumParams; ++loc) {
                if (Shader::t_native_emit.attrs[loc].valid) {
                    NativeCompareAttribute(loc, vindex, draw);
                }
            }
        } else if (native_pull) {
            const Id vindex{OpLoad(U32[1], vertex_index)};
            for (u32 loc = 0; loc < IR::NumParams; ++loc) {
                if (Shader::t_native_emit.attrs[loc].valid &&
                    !((Shader::t_native_emit.keep_inputs >> loc) & 1)) {
                    native_attr_value[loc] = NativePullAttribute(loc, vindex);
                }
            }
        }
    }
    if (!profile.needs_buffer_offsets && !native) {
        return;
    }
    for (u32 i = 0; i < buffers.size(); i++) {
        auto& buffer = buffers[i];
        const auto& desc = info.buffers[i];
        const u32 binding = buffer.binding;
        if (buffer.buffer_type != BufferType::Guest) {
            continue;
        }

        Id buf_offset;
        if (native) {
            // The full byte offset of this draw's binding inside the bound VkBuffer.
            const Id rec_word{NativeRecordLoad(Shader::NativeRecordOffOffset + binding)};
            buf_offset = rec_word;
            native_size_bytes[i] = NativeRecordLoad(Shader::NativeRecordSizeOffset + binding);
            if (native_ring) {
                // The record holds the guest address (low 32 bits); the table the buffer's base.
                buf_offset = OpISub(U32[1], buf_offset,
                                    NativeTableLoad(Shader::NativeTableBase + binding));
                // Coverage (sec. 11): an inherited binding's size comes from the table.
                const Id word{NativeTableLoad(Shader::NativeTableInhSlots + binding / 32)};
                const Id bit{OpBitwiseAnd(
                    U32[1], OpShiftRightLogical(U32[1], word, ConstU32(binding % 32)),
                    ConstU32(1u))};
                const Id inherited{OpINotEqual(U1[1], bit, u32_zero_value)};
                native_size_bytes[i] =
                    OpSelect(U32[1], inherited,
                             NativeTableLoad(Shader::NativeTableSize + binding),
                             native_size_bytes[i]);
                if (native_pt) {
                    NativePtTranslate(i, binding, rec_word, inherited);
                }
            }
            native_size_dw[i] = OpShiftRightLogical(U32[1], native_size_bytes[i], ConstU32(2U));
        } else {
            const u32 half = PushData::BufOffsetIndex + (binding >> 4);
            const u32 comp = (binding & 0xf) >> 2;
            const u32 offset = (binding & 0x3) << 3;
            const Id ptr{OpAccessChain(TypePointer(spv::StorageClass::PushConstant, U32[1]),
                                       push_data_block, ConstU32(half), ConstU32(comp))};
            const Id value{OpLoad(U32[1], ptr)};
            buf_offset = OpBitFieldUExtract(U32[1], value, ConstU32(offset), ConstU32(8U));
        }
        Name(buf_offset, fmt::format("buf{}_off", binding));
        buffer.Offset(PointerSize::B8) = buf_offset;

        if (True(desc.used_types & IR::Type::U16)) {
            const Id buf_word_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(1U))};
            Name(buf_word_offset, fmt::format("buf{}_word_off", binding));
            buffer.Offset(PointerSize::B16) = buf_word_offset;
        }
        if (True(desc.used_types & IR::Type::U32)) {
            const Id buf_dword_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(2U))};
            Name(buf_dword_offset, fmt::format("buf{}_dword_off", binding));
            buffer.Offset(PointerSize::B32) = buf_dword_offset;
        }
        if (True(desc.used_types & IR::Type::U64)) {
            const Id buf_qword_offset{OpShiftRightLogical(U32[1], buf_offset, ConstU32(3U))};
            Name(buf_qword_offset, fmt::format("buf{}_qword_off", binding));
            buffer.Offset(PointerSize::B64) = buf_qword_offset;
        }
    }
}

void EmitContext::DefineAmdPerVertexAttribs() {
    if (!profile.supports_amd_shader_explicit_vertex_parameter) {
        return;
    }
    for (s32 i = 0; i < runtime_info.fs_info.num_inputs; i++) {
        const auto& input = runtime_info.fs_info.inputs[i];
        if (input.IsDefault() || info.fs_interpolation[i].primary != Qualifier::PerVertex) {
            continue;
        }
        auto& param = input_params[i];
        const Id pointer = param.id;
        param.id_array[0] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(0U));
        param.id_array[1] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(1U));
        param.id_array[2] =
            OpInterpolateAtVertexAMD(F32[param.num_components], pointer, ConstU32(2U));
        param.is_loaded = true;
    }
}

void EmitContext::DefineWorkgroupIndex() {
    const Id workgroup_id_val{OpLoad(U32[3], workgroup_id)};
    const Id workgroup_x{OpCompositeExtract(U32[1], workgroup_id_val, 0)};
    const Id workgroup_y{OpCompositeExtract(U32[1], workgroup_id_val, 1)};
    const Id workgroup_z{OpCompositeExtract(U32[1], workgroup_id_val, 2)};
    const Id num_workgroups{OpLoad(U32[3], num_workgroups_id)};
    const Id num_workgroups_x{OpCompositeExtract(U32[1], num_workgroups, 0)};
    const Id num_workgroups_y{OpCompositeExtract(U32[1], num_workgroups, 1)};
    workgroup_index_id =
        OpIAdd(U32[1], OpIAdd(U32[1], workgroup_x, OpIMul(U32[1], workgroup_y, num_workgroups_x)),
               OpIMul(U32[1], workgroup_z, OpIMul(U32[1], num_workgroups_x, num_workgroups_y)));
    Name(workgroup_index_id, "workgroup_index");
}

void EmitContext::DefineInputs() {
    if (info.uses_lane_id) {
        subgroup_local_invocation_id = DefineVariable(
            U32[1], spv::BuiltIn::SubgroupLocalInvocationId, spv::StorageClass::Input);
        Decorate(subgroup_local_invocation_id, spv::Decoration::Flat);
    }
    switch (l_stage) {
    case LogicalStage::Vertex: {
        vertex_index = DefineVariable(U32[1], spv::BuiltIn::VertexIndex, spv::StorageClass::Input);
        base_vertex = DefineVariable(U32[1], spv::BuiltIn::BaseVertex, spv::StorageClass::Input);
        if (native) {
            draw_index = DefineVariable(U32[1], spv::BuiltIn::DrawIndex, spv::StorageClass::Input);
        }
        instance_id = DefineVariable(U32[1], spv::BuiltIn::InstanceIndex, spv::StorageClass::Input);

        const auto fetch_shader = Gcn::ParseFetchShader(info);
        if (!fetch_shader) {
            break;
        }
        for (const auto& attrib : fetch_shader->attributes) {
            ASSERT(attrib.semantic < IR::NumParams);
            const auto sharp = attrib.GetSharp(info);
            if (native_pull && !((Shader::t_native_emit.keep_inputs >> attrib.semantic) & 1)) {
                // SHADPS4_NATIVE_COLOUR: no vertex input; NativePullAttribute loads it.
                ASSERT_MSG(Shader::t_native_emit.attrs[attrib.semantic].valid,
                           "native pull: attribute {} has no format", attrib.semantic);
                input_params[attrib.semantic] =
                    GetAttributeInfo(sharp.GetNumberFmt(), Id{}, 4, false);
                continue;
            }
            const Id type{GetAttributeType(*this, sharp.GetNumberFmt())[4]};
            Id id{DefineInput(type, attrib.semantic)};
            if (attrib.GetStepRate() != Gcn::VertexAttribute::InstanceIdType::None) {
                Name(id, fmt::format("vs_instance_attr{}", attrib.semantic));
            } else {
                Name(id, fmt::format("vs_in_attr{}", attrib.semantic));
            }
            input_params[attrib.semantic] = GetAttributeInfo(sharp.GetNumberFmt(), id, 4, false);
        }
        break;
    }
    case LogicalStage::Fragment: {
        if (native_colour) {
            native_draw_in = DefineInput(U32[1], Shader::t_native_emit.draw_location);
            Decorate(native_draw_in, spv::Decoration::Flat);
            Name(native_draw_in, "native_draw_in");
        }
        if (info.loads.GetAny(IR::Attribute::FragCoord)) {
            frag_coord = DefineVariable(F32[4], spv::BuiltIn::FragCoord, spv::StorageClass::Input);
        }
        if (info.loads.Get(IR::Attribute::IsFrontFace)) {
            front_facing =
                DefineVariable(U1[1], spv::BuiltIn::FrontFacing, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::RenderTargetIndex)) {
            output_layer = DefineVariable(U32[1], spv::BuiltIn::Layer, spv::StorageClass::Input);
            Decorate(output_layer, spv::Decoration::Flat);
        }
        if (info.loads.Get(IR::Attribute::SampleIndex)) {
            sample_index = DefineVariable(U32[1], spv::BuiltIn::SampleId, spv::StorageClass::Input);
            Decorate(sample_index, spv::Decoration::Flat);
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmooth)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth = DefineVariable(F32[2], spv::BuiltIn::BaryCoordSmoothAMD,
                                                   spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmoothCentroid)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth_centroid = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordSmoothCentroidAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth_centroid =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_smooth_centroid, spv::Decoration::Centroid);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordSmoothSample)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_smooth_sample = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordSmoothSampleAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_smooth_sample =
                    DefineVariable(F32[3], spv::BuiltIn::BaryCoordKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_smooth_sample, spv::Decoration::Sample);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordNoPersp)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_nopersp = DefineVariable(F32[2], spv::BuiltIn::BaryCoordNoPerspAMD,
                                                    spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_nopersp = DefineVariable(F32[3], spv::BuiltIn::BaryCoordNoPerspKHR,
                                                    spv::StorageClass::Input);
            }
        }
        if (info.loads.GetAny(IR::Attribute::BaryCoordNoPerspSample)) {
            if (profile.supports_amd_shader_explicit_vertex_parameter) {
                bary_coord_nopersp_sample = DefineVariable(
                    F32[2], spv::BuiltIn::BaryCoordNoPerspSampleAMD, spv::StorageClass::Input);
            } else if (profile.supports_fragment_shader_barycentric) {
                bary_coord_nopersp_sample = DefineVariable(
                    F32[3], spv::BuiltIn::BaryCoordNoPerspKHR, spv::StorageClass::Input);
                // Decorate(bary_coord_nopersp_sample, spv::Decoration::Sample);
            }
        }

        const bool has_clip_distance_inputs = runtime_info.fs_info.clip_distance_emulation;
        // Clip distances attribute vector is the last in inputs array
        const auto num_inputs =
            runtime_info.fs_info.num_inputs - (has_clip_distance_inputs ? 1 : 0);

        for (s32 i = 0; i < num_inputs; i++) {
            const auto& input = runtime_info.fs_info.inputs[i];
            if (input.IsDefault()) {
                continue;
            }
            const IR::Attribute param = IR::Attribute::Param0 + i;
            const u32 num_components = info.loads.NumComponents(param);
            const auto [primary, auxiliary] = info.fs_interpolation[i];
            const Id type = F32[num_components];
            const Id attr_id = [&] {
                const auto bind_location = input.param_index + (has_clip_distance_inputs ? 1 : 0);
                if (primary == Qualifier::PerVertex &&
                    profile.supports_fragment_shader_barycentric) {
                    return Name(DefineInput(TypeArray(type, ConstU32(3U)), bind_location),
                                fmt::format("fs_in_attr{}_p", i));
                }
                return Name(DefineInput(type, bind_location), fmt::format("fs_in_attr{}", i));
            }();
            if (primary == Qualifier::PerVertex) {
                Decorate(attr_id, profile.supports_amd_shader_explicit_vertex_parameter
                                      ? spv::Decoration::ExplicitInterpAMD
                                      : spv::Decoration::PerVertexKHR);
            } else if (primary != Qualifier::Smooth) {
                Decorate(attr_id, primary == Qualifier::Flat ? spv::Decoration::Flat
                                                             : spv::Decoration::NoPerspective);
            }
            if (auxiliary != Qualifier::None) {
                Decorate(attr_id, auxiliary == Qualifier::Centroid ? spv::Decoration::Centroid
                                                                   : spv::Decoration::Sample);
            }
            input_params[i] = GetAttributeInfo(AmdGpu::NumberFormat::Float, attr_id, num_components,
                                               false, false, primary == Qualifier::PerVertex);
        }

        if (has_clip_distance_inputs) {
            const auto type = F32[MaxEmulatedClipDistances];
            const auto attr_id = Name(DefineInput(type, 0), fmt::format("cldist_attr{}", 0));
            input_params[num_inputs] = GetAttributeInfo(AmdGpu::NumberFormat::Float, attr_id,
                                                        MaxEmulatedClipDistances, false);
        }
        break;
    }
    case LogicalStage::Compute:
        if (info.loads.GetAny(IR::Attribute::WorkgroupIndex) ||
            info.loads.GetAny(IR::Attribute::WorkgroupId)) {
            workgroup_id =
                DefineVariable(U32[3], spv::BuiltIn::WorkgroupId, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::WorkgroupIndex)) {
            num_workgroups_id =
                DefineVariable(U32[3], spv::BuiltIn::NumWorkgroups, spv::StorageClass::Input);
        }
        if (info.loads.GetAny(IR::Attribute::LocalInvocationId)) {
            local_invocation_id =
                DefineVariable(U32[3], spv::BuiltIn::LocalInvocationId, spv::StorageClass::Input);
        }
        break;
    case LogicalStage::Geometry: {
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);
        const auto gl_per_vertex =
            Name(TypeStruct(F32[4], F32[1], TypeArray(F32[1], ConstU32(1u))), "gl_PerVertex");
        MemberName(gl_per_vertex, 0, "gl_Position");
        MemberName(gl_per_vertex, 1, "gl_PointSize");
        MemberName(gl_per_vertex, 2, "gl_ClipDistance");
        MemberDecorate(gl_per_vertex, 0, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::Position));
        MemberDecorate(gl_per_vertex, 1, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::PointSize));
        MemberDecorate(gl_per_vertex, 2, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::ClipDistance));
        Decorate(gl_per_vertex, spv::Decoration::Block);
        const auto num_verts_in = NumVertices(runtime_info.gs_info.in_primitive);
        const auto vertices_in = TypeArray(gl_per_vertex, ConstU32(num_verts_in));
        gl_in = Name(DefineVar(vertices_in, spv::StorageClass::Input), "gl_in");
        interfaces.push_back(gl_in);

        const auto num_params = runtime_info.gs_info.in_vertex_data_size / 4 - 1u;
        for (int param_id = 0; param_id < num_params; ++param_id) {
            const Id type{TypeArray(F32[4], ConstU32(num_verts_in))};
            const Id id{DefineInput(type, param_id)};
            Name(id, fmt::format("gs_in_attr{}", param_id));
            input_params[param_id] =
                GetAttributeInfo(AmdGpu::NumberFormat::Float, id, 4, false, false, true);
        }
        break;
    }
    case LogicalStage::TessellationControl: {
        invocation_id =
            DefineVariable(U32[1], spv::BuiltIn::InvocationId, spv::StorageClass::Input);
        patch_vertices =
            DefineVariable(U32[1], spv::BuiltIn::PatchVertices, spv::StorageClass::Input);
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);

        const u32 num_attrs = Common::AlignUp(runtime_info.hs_info.ls_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(per_vertex_type, ConstU32(32u))};
            input_attr_array = DefineInput(patch_array_type, 0);
            Name(input_attr_array, "in_attrs");
        }
        break;
    }
    case LogicalStage::TessellationEval: {
        tess_coord = DefineInput(F32[3], std::nullopt, spv::BuiltIn::TessCoord);
        primitive_id = DefineVariable(U32[1], spv::BuiltIn::PrimitiveId, spv::StorageClass::Input);

        const u32 num_attrs =
            Common::AlignUp(runtime_info.hs_es_vs_info.hs_output_cp_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(per_vertex_type, ConstU32(32u))};
            input_attr_array = DefineInput(patch_array_type, 0);
            Name(input_attr_array, "in_attrs");
        }

        const u32 patch_base_location = num_attrs;
        for (size_t index = 0; index < 30; ++index) {
            if (!(info.uses_patches & (1U << index))) {
                continue;
            }
            const Id id{DefineInput(F32[4], patch_base_location + index)};
            Decorate(id, spv::Decoration::Patch);
            Name(id, fmt::format("patch_in{}", index));
            patches[index] = id;
        }
        break;
    }
    default:
        break;
    }
}

void EmitContext::DefineVertexBlock() {
    const std::array<Id, 8> zero{f32_zero_value, f32_zero_value, f32_zero_value, f32_zero_value,
                                 f32_zero_value, f32_zero_value, f32_zero_value, f32_zero_value};
    output_position = DefineVariable(F32[4], spv::BuiltIn::Position, spv::StorageClass::Output);
    const bool needs_clip_distance_emulation = l_stage == LogicalStage::Vertex &&
                                               stage == Stage::Vertex &&
                                               profile.needs_clip_distance_emulation;
    const auto has_clip_distance_outputs = info.stores.GetAny(IR::Attribute::ClipDistance);
    if (has_clip_distance_outputs && !needs_clip_distance_emulation) {
        const Id type{TypeArray(F32[1], ConstU32(8U))};
        const Id initializer{ConstantComposite(type, zero)};
        clip_distances = DefineVariable(type, spv::BuiltIn::ClipDistance, spv::StorageClass::Output,
                                        initializer);
    }
    if (info.stores.GetAny(IR::Attribute::CullDistance)) {
        const Id type{TypeArray(F32[1], ConstU32(8U))};
        const Id initializer{ConstantComposite(type, zero)};
        cull_distances = DefineVariable(type, spv::BuiltIn::CullDistance, spv::StorageClass::Output,
                                        initializer);
    }
    if (info.stores.GetAny(IR::Attribute::PointSize)) {
        output_point_size =
            DefineVariable(F32[1], spv::BuiltIn::PointSize, spv::StorageClass::Output);
    }
    if (info.stores.GetAny(IR::Attribute::RenderTargetIndex)) {
        output_layer = DefineVariable(U32[1], spv::BuiltIn::Layer, spv::StorageClass::Output);
    }
    if (info.stores.GetAny(IR::Attribute::ViewportIndex)) {
        output_viewport_index =
            DefineVariable(U32[1], spv::BuiltIn::ViewportIndex, spv::StorageClass::Output);
    }
}

void EmitContext::DefineOutputs() {
    switch (l_stage) {
    case LogicalStage::Vertex: {
        DefineVertexBlock();
        if (stage == Shader::Stage::Local) {
            const u32 num_attrs = Common::AlignUp(runtime_info.ls_info.ls_stride, 16) >> 4;
            if (num_attrs > 0) {
                const Id type{TypeArray(F32[4], ConstU32(num_attrs))};
                output_attr_array = DefineOutput(type, 0);
                Name(output_attr_array, "out_attrs");
            }
        } else {
            const bool needs_clip_distance_emulation =
                stage == Stage::Vertex && profile.needs_clip_distance_emulation &&
                info.stores.GetAny(IR::Attribute::ClipDistance);
            u32 num_attrs = 0u;
            for (u32 i = 0; i < IR::NumParams; i++) {
                const IR::Attribute param{IR::Attribute::Param0 + i};
                if (!info.stores.GetAny(param)) {
                    continue;
                }
                const u32 num_components = info.stores.NumComponents(param);
                const Id id{
                    DefineOutput(F32[num_components], i + (needs_clip_distance_emulation ? 1 : 0))};
                Name(id, fmt::format("out_attr{}", i));
                output_params[i] =
                    GetAttributeInfo(AmdGpu::NumberFormat::Float, id, num_components, true);
                ++num_attrs;
            }

            if (native_colour) {
                native_draw_out = DefineOutput(U32[1], Shader::t_native_emit.draw_location);
                Decorate(native_draw_out, spv::Decoration::Flat);
                Name(native_draw_out, "native_draw_out");
            }
            if (needs_clip_distance_emulation) {
                clip_distances = Id{DefineOutput(F32[MaxEmulatedClipDistances], 0)};
                output_params[num_attrs] = GetAttributeInfo(
                    AmdGpu::NumberFormat::Float, clip_distances, MaxEmulatedClipDistances, true);
                Name(clip_distances, fmt::format("cldist_attr{}", 0));
            }
        }
        break;
    }
    case LogicalStage::TessellationControl: {
        if (info.stores_tess_level_outer) {
            const Id type{TypeArray(F32[1], ConstU32(4U))};
            output_tess_level_outer =
                DefineOutput(type, std::nullopt, spv::BuiltIn::TessLevelOuter);
            Decorate(output_tess_level_outer, spv::Decoration::Patch);
        }
        if (info.stores_tess_level_inner) {
            const Id type{TypeArray(F32[1], ConstU32(2U))};
            output_tess_level_inner =
                DefineOutput(type, std::nullopt, spv::BuiltIn::TessLevelInner);
            Decorate(output_tess_level_inner, spv::Decoration::Patch);
        }

        const u32 num_attrs =
            Common::AlignUp(runtime_info.hs_es_vs_info.hs_output_cp_stride, 16) >> 4;
        if (num_attrs > 0) {
            const Id per_vertex_type{TypeArray(F32[4], ConstU32(num_attrs))};
            // The input vertex count isn't statically known, so make length 32 (what
            // glslang does)
            const Id patch_array_type{TypeArray(
                per_vertex_type, ConstU32(runtime_info.hs_info.NumOutputControlPoints()))};
            output_attr_array = DefineOutput(patch_array_type, 0);
            Name(output_attr_array, "out_attrs");
        }

        const u32 patch_base_location = num_attrs;
        for (size_t index = 0; index < 30; ++index) {
            if (!(info.uses_patches & (1U << index))) {
                continue;
            }
            const Id id{DefineOutput(F32[4], patch_base_location + index)};
            Decorate(id, spv::Decoration::Patch);
            Name(id, fmt::format("patch_out{}", index));
            patches[index] = id;
        }
        break;
    }
    case LogicalStage::TessellationEval: {
        DefineVertexBlock();
        for (u32 i = 0; i < IR::NumParams; i++) {
            const IR::Attribute param{IR::Attribute::Param0 + i};
            if (!info.stores.GetAny(param)) {
                continue;
            }
            const u32 num_components = info.stores.NumComponents(param);
            const Id id{DefineOutput(F32[num_components], i)};
            Name(id, fmt::format("out_attr{}", i));
            output_params[i] =
                GetAttributeInfo(AmdGpu::NumberFormat::Float, id, num_components, true);
        }
        break;
    }
    case LogicalStage::Fragment: {
        if (info.stores.Get(IR::Attribute::Depth)) {
            frag_depth = DefineVariable(F32[1], spv::BuiltIn::FragDepth, spv::StorageClass::Output);
        }
        if (info.stores.Get(IR::Attribute::SampleMask)) {
            sample_mask = DefineVariable(TypeArray(U32[1], u32_one_value), spv::BuiltIn::SampleMask,
                                         spv::StorageClass::Output);
        }
        if (info.stores.Get(IR::Attribute::StencilRef) && profile.supports_shader_stencil_export) {
            stencil_ref =
                DefineVariable(S32[1], spv::BuiltIn::FragStencilRefEXT, spv::StorageClass::Output);
        }
        u32 num_render_targets = 0;
        for (u32 i = 0; i < IR::NumRenderTargets; i++) {
            const IR::Attribute mrt{IR::Attribute::RenderTarget0 + i};
            if (!info.stores.GetAny(mrt)) {
                continue;
            }
            const u32 num_components = info.stores.NumComponents(mrt);
            const AmdGpu::NumberFormat num_format{runtime_info.fs_info.color_buffers[i].num_format};
            const Id type{GetAttributeType(*this, num_format)[num_components]};
            Id id;
            if (runtime_info.fs_info.dual_source_blending) {
                id = DefineOutput(type, 0);
                Decorate(id, spv::Decoration::Index, i);
            } else {
                id = DefineOutput(type, i);
            }
            Name(id, fmt::format("frag_color{}", i));
            frag_outputs[i] = GetAttributeInfo(num_format, id, num_components, true);
            ++num_render_targets;
        }
        // Dual source blending allows at most 2 render targets, one for each source.
        // Fewer targets are allowed but the missing blending source values will be
        // undefined.
        ASSERT_MSG(!runtime_info.fs_info.dual_source_blending || num_render_targets <= 2,
                   "Dual source blending enabled, there must be at most two MRT exports");
        break;
    }
    case LogicalStage::Geometry: {
        DefineVertexBlock();
        for (u32 attr_id = 0; attr_id < info.gs_copy_data.num_attrs; attr_id++) {
            const Id id{DefineOutput(F32[4], attr_id)};
            Name(id, fmt::format("out_attr{}", attr_id));
            output_params[attr_id] = GetAttributeInfo(AmdGpu::NumberFormat::Float, id, 4, true);
        }
        break;
    }
    case LogicalStage::Compute:
        break;
    default:
        UNREACHABLE();
    }
}

void EmitContext::DefinePushDataBlock() {
    // Create push constants block for instance steps rates
    const Id struct_type{Name(TypeStruct(F32[1], F32[1], F32[1], F32[1], U32[4], U32[4], U32[4],
                                         U32[4], U32[4], U32[4], U32[2]),
                              "AuxData")};
    Decorate(struct_type, spv::Decoration::Block);
    MemberName(struct_type, PushData::XOffsetIndex, "xoffset");
    MemberName(struct_type, PushData::YOffsetIndex, "yoffset");
    MemberName(struct_type, PushData::XScaleIndex, "xscale");
    MemberName(struct_type, PushData::YScaleIndex, "yscale");
    MemberName(struct_type, PushData::UdRegsIndex + 0, "ud_regs0");
    MemberName(struct_type, PushData::UdRegsIndex + 1, "ud_regs1");
    MemberName(struct_type, PushData::UdRegsIndex + 2, "ud_regs2");
    MemberName(struct_type, PushData::UdRegsIndex + 3, "ud_regs3");
    MemberName(struct_type, PushData::BufOffsetIndex + 0, "buf_offsets0");
    MemberName(struct_type, PushData::BufOffsetIndex + 1, "buf_offsets1");
    MemberName(struct_type, PushData::BufOffsetIndex + 2, "buf_offsets2");
    MemberDecorate(struct_type, PushData::XOffsetIndex, spv::Decoration::Offset, 0U);
    MemberDecorate(struct_type, PushData::YOffsetIndex, spv::Decoration::Offset, 4U);
    MemberDecorate(struct_type, PushData::XScaleIndex, spv::Decoration::Offset, 8U);
    MemberDecorate(struct_type, PushData::YScaleIndex, spv::Decoration::Offset, 12U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 0, spv::Decoration::Offset, 16U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 1, spv::Decoration::Offset, 32U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 2, spv::Decoration::Offset, 48U);
    MemberDecorate(struct_type, PushData::UdRegsIndex + 3, spv::Decoration::Offset, 64U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 0, spv::Decoration::Offset, 80U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 1, spv::Decoration::Offset, 96U);
    MemberDecorate(struct_type, PushData::BufOffsetIndex + 2, spv::Decoration::Offset, 112U);
    push_data_block = DefineVar(struct_type, spv::StorageClass::PushConstant);
    Name(push_data_block, "push_data");
    interfaces.push_back(push_data_block);
}

EmitContext::BufferSpv EmitContext::DefineBuffer(bool is_storage, bool is_written, u32 elem_shift,
                                                 BufferType buffer_type, Id data_type, bool is_coherent) {
    // Define array type.
    const Id max_num_items = ConstU32(u32(profile.max_ubo_size) >> elem_shift);
    const Id record_array_type{is_storage ? TypeRuntimeArray(data_type)
                                          : TypeArray(data_type, max_num_items)};
    // Define block struct type. Don't perform decorations twice on the same Id.
    const Id struct_type{TypeStruct(record_array_type)};
    if (std::ranges::find(buf_type_ids, record_array_type.value, &Id::value) ==
        buf_type_ids.end()) {
        Decorate(record_array_type, spv::Decoration::ArrayStride, 1 << elem_shift);
        Decorate(struct_type, spv::Decoration::Block);
        MemberName(struct_type, 0, "data");
        MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
        buf_type_ids.push_back(record_array_type);
    }
    // Define buffer binding interface.
    const auto storage_class =
        is_storage ? spv::StorageClass::StorageBuffer : spv::StorageClass::Uniform;
    const Id struct_pointer_type{TypePointer(storage_class, struct_type)};
    const Id pointer_type = TypePointer(storage_class, data_type);
    const Id id{AddGlobalVariable(struct_pointer_type, storage_class)};
    Decorate(id, spv::Decoration::Binding, binding.unified);
    Decorate(id, spv::Decoration::DescriptorSet, 0U);
    if (is_storage && !is_written) {
        Decorate(id, spv::Decoration::NonWritable);
    } else if (is_storage && is_coherent) {
        // Port of upstream #4896: the guest V# MTYPE marks this buffer as shared between
        // invocations; Coherent forbids caching its accesses across barriers.
        Decorate(id, spv::Decoration::Coherent);
    }
    switch (buffer_type) {
    case BufferType::GdsBuffer:
        Name(id, "gds_buffer");
        break;
    case BufferType::Flatbuf:
        Name(id, "srt_flatbuf");
        break;
    case BufferType::ClipPlanes:
        Name(id, "clip_planes");
        break;
    case BufferType::BdaPagetable:
        Name(id, "bda_pagetable");
        break;
    case BufferType::FaultBuffer:
        Name(id, "fault_buffer");
        break;
    case BufferType::SharedMemory:
        Name(id, "ssbo_shmem");
        break;
    default:
        Name(id, fmt::format("{}_{}", is_storage ? "ssbo" : "ubo", binding.buffer));
        break;
    }
    interfaces.push_back(id);
    return {id, pointer_type};
};

void EmitContext::DefineBuffers() {
    for (const auto& desc : info.buffers) {
        const auto buf_sharp = desc.GetSharp(info);
        const bool is_storage = desc.IsStorage(buf_sharp);
        // SHADPS4_SPIRV_COHERENT=1 (default off, disables the pipeline cache for the run).
        const bool is_coherent = Shader::SpirvCoherentEnabled() &&
                                 (desc.buffer_type == BufferType::SharedMemory || buf_sharp.mtype == 3);

        // Set indexes for special buffers.
        if (desc.buffer_type == BufferType::Flatbuf) {
            flatbuf_index = buffers.size();
        } else if (desc.buffer_type == BufferType::BdaPagetable) {
            bda_pagetable_index = buffers.size();
        } else if (desc.buffer_type == BufferType::FaultBuffer) {
            fault_buffer_index = buffers.size();
        }

        // Define aliases depending on the shader usage.
        auto& spv_buffer = buffers.emplace_back(binding.buffer++, desc.buffer_type);
        if (True(desc.used_types & IR::Type::U64)) {
            spv_buffer.Alias(PointerType::U64) =
                DefineBuffer(is_storage, desc.is_written, 3, desc.buffer_type, U64,
                             is_coherent);
        }
        if (True(desc.used_types & IR::Type::U32)) {
            spv_buffer.Alias(PointerType::U32) =
                DefineBuffer(is_storage, desc.is_written, 2, desc.buffer_type, U32[1],
                             is_coherent);
        }
        if (True(desc.used_types & IR::Type::F32)) {
            spv_buffer.Alias(PointerType::F32) =
                DefineBuffer(is_storage, desc.is_written, 2, desc.buffer_type, F32[1],
                             is_coherent);
        }
        if (True(desc.used_types & IR::Type::U16)) {
            spv_buffer.Alias(PointerType::U16) =
                DefineBuffer(is_storage, desc.is_written, 1, desc.buffer_type, U16,
                             is_coherent);
        }
        if (True(desc.used_types & IR::Type::U8)) {
            spv_buffer.Alias(PointerType::U8) =
                DefineBuffer(is_storage, desc.is_written, 0, desc.buffer_type, U8,
                             is_coherent);
        }
        ++binding.unified;
    }
    if (native) {
        // The records SSBO sits after every binding of every stage (records_binding = the base
        // pipeline's binding count); binding.unified is not advanced for it.
        const Id array_type{TypeRuntimeArray(U32[1])};
        const Id struct_type{TypeStruct(array_type)};
        if (std::ranges::find(buf_type_ids, array_type.value, &Id::value) == buf_type_ids.end()) {
            Decorate(array_type, spv::Decoration::ArrayStride, 4U);
            Decorate(struct_type, spv::Decoration::Block);
            MemberName(struct_type, 0, "data");
            MemberDecorate(struct_type, 0, spv::Decoration::Offset, 0U);
            buf_type_ids.push_back(array_type);
        }
        const Id struct_pointer_type{TypePointer(spv::StorageClass::StorageBuffer, struct_type)};
        native_records_ptr = TypePointer(spv::StorageClass::StorageBuffer, U32[1]);
        native_records = AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
        Decorate(native_records, spv::Decoration::Binding, Shader::t_native_emit.records_binding);
        Decorate(native_records, spv::Decoration::DescriptorSet, 0U);
        Decorate(native_records, spv::Decoration::NonWritable);
        Name(native_records, "native_records");
        interfaces.push_back(native_records);
        if (native_pull) {
            native_vbuf = AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
            Decorate(native_vbuf, spv::Decoration::Binding, Shader::t_native_emit.vertex_binding);
            Decorate(native_vbuf, spv::Decoration::DescriptorSet, 0U);
            Decorate(native_vbuf, spv::Decoration::NonWritable);
            Name(native_vbuf, "native_vbuf");
            interfaces.push_back(native_vbuf);
            if (Shader::t_native_emit.vbuf2) {
                // q53: the second vertex SSBO (same block type).
                native_vbuf2 = AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
                Decorate(native_vbuf2, spv::Decoration::Binding, Shader::t_native_emit.vertex2_binding);
                Decorate(native_vbuf2, spv::Decoration::DescriptorSet, 0U);
                Decorate(native_vbuf2, spv::Decoration::NonWritable);
                Name(native_vbuf2, "native_vbuf2");
                interfaces.push_back(native_vbuf2);
            }
        }
        if (native_ring) {
            native_table = AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
            Decorate(native_table, spv::Decoration::Binding, Shader::t_native_emit.table_binding);
            Decorate(native_table, spv::Decoration::DescriptorSet, 0U);
            Decorate(native_table, spv::Decoration::NonWritable);
            Name(native_table, "native_table");
            interfaces.push_back(native_table);
        }
        if (native_pull && Shader::t_native_emit.compare) {
            native_dbg = AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
            Decorate(native_dbg, spv::Decoration::Binding, Shader::t_native_emit.debug_binding);
            Decorate(native_dbg, spv::Decoration::DescriptorSet, 0U);
            Name(native_dbg, "native_dbg");
            interfaces.push_back(native_dbg);
        }
        // SHADPS4_NATIVE_PT: the page table itself is read by pointer (NativePtTranslate), no
        // binding; =2 the compare SSBO.
        if (native_pt_compare) {
            native_pt_dbg =
                AddGlobalVariable(struct_pointer_type, spv::StorageClass::StorageBuffer);
            Decorate(native_pt_dbg, spv::Decoration::Binding,
                     Shader::t_native_emit.pt_debug_binding);
            Decorate(native_pt_dbg, spv::Decoration::DescriptorSet, 0U);
            Name(native_pt_dbg, "native_pt_dbg");
            interfaces.push_back(native_pt_dbg);
        }
    }
}

Id EmitContext::PtClampPage(Id page) {
    const u64 table_pages = VideoCore::BufferCache::BdaNumPages();
    return OpSelect(U64, OpULessThan(U1[1], page, Constant(U64, table_pages)), page,
                    Constant(U64, u64{0}));
}

Id EmitContext::PtAbove(Id last_page) {
    // Also at the full 40-bit table: guest + size - 1 can cross 2^40 (review 10-20, item 4).
    return OpUGreaterThanEqual(U1[1], last_page,
                               Constant(U64, VideoCore::BufferCache::BdaNumPages()));
}

void EmitContext::NativePtTranslate(u32 handle, u32 binding, Id record_word, Id inherited) {
    // SHADPS4_NATIVE_PT (sec. 18): this draw's full guest address of the binding - the record's
    // offset word (ring: the job thread's guest address, low 32 bits) and hi byte, an inherited
    // slot's from the table - through the buffer cache's page table (a u64 device address per
    // 16 KB guest page, 0 = not cached) to the device address the binding's loads start at. A
    // miss maps to the table's fallback (the page table buffer itself: always mapped memory, so
    // no pointer ever leaves mapped memory) and every load of the binding then reads 0. The hi
    // byte bounds the address below 2^40; with a smaller table (SHADPS4_PT_BITS) a page above it
    // is clamped to page 0 and counts as a miss, so the pointer never leaves the table.
    const auto hi_byte = [&](Id word) {
        return OpBitFieldUExtract(U32[1], word, ConstU32((binding % 4) * 8), ConstU32(8u));
    };
    const Id lo{OpSelect(U32[1], inherited, NativeTableLoad(Shader::NativeTablePtLo + binding),
                         record_word)};
    const Id hi{OpSelect(U32[1], inherited,
                         hi_byte(NativeTableLoad(Shader::NativeTablePtHi + binding / 4)),
                         hi_byte(NativeRecordLoad(Shader::NativeRecordHiOffset + binding / 4)))};
    const auto u64_of = [&](Id lo32, Id hi32) {
        return OpBitwiseOr(U64, OpUConvert(U64, lo32),
                           OpShiftLeftLogical(U64, OpUConvert(U64, hi32), ConstU32(32u)));
    };
    const Id guest{u64_of(lo, hi)};
    Name(guest, fmt::format("buf{}_guest", binding));
    // The table's fallback pair is the page table buffer's own device address: the entry is
    // read by pointer at fallback + page * 8 (no set-0 binding for the page table).
    const Id fallback{u64_of(NativeTableLoad(Shader::NativeTablePtFallback),
                             NativeTableLoad(Shader::NativeTablePtFallback + 1))};
    const Id page_raw{OpShiftRightLogical(U64, guest,
                                          ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS))};
    // last_page >= page, so checking the binding's last page covers both.
    const Id page{PtClampPage(page_raw)};
    const Id entry_addr{OpIAdd(U64, fallback, OpShiftLeftLogical(U64, page, ConstU32(3u)))};
    const Id entry{OpLoad(U64, OpConvertUToPtr(physical_pointer_type_u64, entry_addr),
                          spv::MemoryAccessMask::Aligned, 8u)};
    // Contiguity (brick C v2): the binding's last byte's page must map to entry + distance (one
    // cached buffer, contiguous) and be cached; else the binding is a miss (loads read 0) and
    // its pages are flagged in the fault buffer (count 0 otherwise: the helper's loop runs no
    // iteration), so ProcessFaultBuffer creates one buffer over the range. One extra
    // page-table load per binding per invocation.
    const Id size{OpUMax(U32[1], native_size_bytes[handle], ConstU32(1u))};
    const Id last_byte{OpIAdd(U64, guest, OpUConvert(U64, OpISub(U32[1], size, ConstU32(1u))))};
    const Id last_page_raw{OpShiftRightLogical(
        U64, last_byte, ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS))};
    const Id above{PtAbove(last_page_raw)};
    const Id last_page{PtClampPage(last_page_raw)};
    const Id last_addr{OpIAdd(U64, fallback, OpShiftLeftLogical(U64, last_page, ConstU32(3u)))};
    const Id entry_last{OpLoad(U64, OpConvertUToPtr(physical_pointer_type_u64, last_addr),
                               spv::MemoryAccessMask::Aligned, 8u)};
    const Id distance{OpShiftLeftLogical(U64, OpISub(U64, last_page, page),
                                         ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS))};
    const Id contiguous{OpLogicalAnd(U1[1], OpIEqual(U1[1], entry_last, OpIAdd(U64, entry, distance)),
                                     OpINotEqual(U1[1], entry_last, u64_zero_value))};
    const Id miss{OpLogicalOr(U1[1],
                              OpLogicalOr(U1[1], OpIEqual(U1[1], entry, u64_zero_value),
                                          OpLogicalNot(U1[1], contiguous)),
                              above)};
    const Id npages{OpUConvert(U32[1], OpIAdd(U64, OpISub(U64, last_page, page), Constant(U64, u64{1})))};
    // Above the table: a miss, but not flagged (no buffer could ever be found there).
    const Id flag{OpLogicalAnd(U1[1],
                               OpLogicalAnd(U1[1], miss, OpINotEqual(U1[1], guest, u64_zero_value)),
                               OpLogicalNot(U1[1], above))};
    OpFunctionCall(void_id, native_pt_fault, NativeTableLoad(Shader::NativeTableFaultAddr),
                   NativeTableLoad(Shader::NativeTableFaultAddr + 1), OpUConvert(U32[1], page),
                   OpSelect(U32[1], flag, npages, u32_zero_value));
    const Id in_page{
        OpBitwiseAnd(U64, guest, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1))};
    native_dev_base[handle] = OpSelect(U64, miss, fallback, OpIAdd(U64, entry, in_page));
    Name(native_dev_base[handle], fmt::format("buf{}_dev", binding));
    native_pt_miss[handle] = miss;
    native_guest_lo[handle] = lo;
    native_guest_hi[handle] = hi;
}

void EmitContext::NativePtMismatch(u32 handle, Id rel_index, Id pt_bits, Id desc_bits, u32 width,
                                   bool scalar) {
    // SHADPS4_NATIVE_PT=2: the page-table load against the descriptor load of the same element
    // (both as bits, both 0 past the bound); count every mismatch, keep the first 64 (slot 64 =
    // discard), as NativeCompareAttribute does.
    // The atomic and the stores run only on a mismatch, inside a helper function (an atomic per
    // load on one counter took the city to ~1 FPS, s459; a branch here would split the IR's block).
    // Tag bits: 1 vs / 2 ps, bit 2 = a scalar (s_buffer_load) load (brick C v3 item 2).
    const u32 tag = (l_stage == LogicalStage::Fragment ? 2u : 1u) | (scalar ? 4u : 0u) | (width << 8);
    OpFunctionCall(void_id, native_pt_report, ConstU32(buffers[handle].binding), rel_index, pt_bits,
                   desc_bits, native_draw, native_guest_lo[handle], native_guest_hi[handle],
                   ConstU32(tag));
}

Id EmitContext::DefineNativePtReport(std::string_view name, u32 counter_dword,
                                     u32 entries_dword) {
    // void(a, b, x, y, c, d, e, tag): when x != y, count at the compare SSBO's counter_dword and
    // store the 8 words as entry (count < 64 ? count : 64) from entries_dword (NATIVE_PT=2 at
    // 0 / 16, NATIVE_SEG=2 at 1 / 2048).
    const Id func_type{TypeFunction(void_id, U32[1], U32[1], U32[1], U32[1], U32[1], U32[1],
                                    U32[1], U32[1])};
    const Id func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
    std::array<Id, 8> p{};
    for (auto& param : p) {
        param = OpFunctionParameter(U32[1]);
    }
    Name(func, name);
    AddLabel();
    const Id report_label{OpLabel()};
    const Id merge_label{OpLabel()};
    const Id bad{OpINotEqual(U1[1], p[2], p[3])};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(bad, report_label, merge_label);
    AddLabel(report_label);
    const Id ptr0{OpAccessChain(native_records_ptr, native_pt_dbg, u32_zero_value,
                                ConstU32(counter_dword))};
    const Id old{OpAtomicIAdd(U32[1], ptr0, ConstU32(1u), u32_zero_value, ConstU32(1u))};
    const Id slot{OpSelect(U32[1], OpULessThan(U1[1], old, ConstU32(64u)), old, ConstU32(64u))};
    const Id base{
        OpIAdd(U32[1], ConstU32(entries_dword), OpIMul(U32[1], slot, ConstU32(16u)))};
    for (u32 i = 0; i < p.size(); ++i) {
        OpStore(OpAccessChain(native_records_ptr, native_pt_dbg, u32_zero_value,
                              OpIAdd(U32[1], base, ConstU32(i))),
                p[i]);
    }
    OpBranch(merge_label);
    AddLabel(merge_label);
    OpReturn();
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineNativePtFault() {
    // void(fault_lo, fault_hi, first_page, count): the fault bit of pages first .. first +
    // min(count, 256) - 1 in the fault buffer at the 64-bit address (bit page % 32 of word
    // page / 32; a plain read-or-write, as DefineGetBdaPointer's single-page fault).
    const Id func_type{TypeFunction(void_id, U32[1], U32[1], U32[1], U32[1])};
    const Id func{OpFunction(void_id, spv::FunctionControlMask::MaskNone, func_type)};
    const Id lo{OpFunctionParameter(U32[1])};
    const Id hi{OpFunctionParameter(U32[1])};
    const Id first{OpFunctionParameter(U32[1])};
    const Id count{OpFunctionParameter(U32[1])};
    Name(func, "native_pt_fault");
    AddLabel();
    const Id i_var{AddLocalVariable(TypePointer(spv::StorageClass::Function, U32[1]),
                                    spv::StorageClass::Function)};
    OpStore(i_var, u32_zero_value);
    const Id limit{OpUMin(U32[1], count, ConstU32(256u))};
    const Id base{OpBitwiseOr(U64, OpUConvert(U64, lo),
                              OpShiftLeftLogical(U64, OpUConvert(U64, hi), ConstU32(32u)))};
    const Id header{OpLabel()};
    const Id cond{OpLabel()};
    const Id body{OpLabel()};
    const Id cont{OpLabel()};
    const Id merge{OpLabel()};
    OpBranch(header);
    AddLabel(header);
    OpLoopMerge(merge, cont, spv::LoopControlMask::MaskNone);
    OpBranch(cond);
    AddLabel(cond);
    const Id iv{OpLoad(U32[1], i_var)};
    OpBranchConditional(OpULessThan(U1[1], iv, limit), body, merge);
    AddLabel(body);
    const Id p{OpIAdd(U32[1], first, iv)};
    const Id word{OpShiftLeftLogical(U32[1], OpShiftRightLogical(U32[1], p, ConstU32(5u)),
                                     ConstU32(2u))};
    const Id ptr{OpConvertUToPtr(physical_pointer_type_u32, OpIAdd(U64, base, OpUConvert(U64, word)))};
    const Id value{OpLoad(U32[1], ptr, spv::MemoryAccessMask::Aligned, 4u)};
    const Id bit{OpShiftLeftLogical(U32[1], u32_one_value, OpBitwiseAnd(U32[1], p, ConstU32(31u)))};
    OpStore(ptr, OpBitwiseOr(U32[1], value, bit), spv::MemoryAccessMask::Aligned, 4u);
    OpBranch(cont);
    AddLabel(cont);
    OpStore(i_var, OpIAdd(U32[1], iv, u32_one_value));
    OpBranch(header);
    AddLabel(merge);
    OpReturn();
    OpFunctionEnd();
    return func;
}

void EmitContext::NativeSegRecordPointer(Id draw) {
    // SHADPS4_NATIVE_SEG (sec. 20 brick A): the batch's list entry of this draw (by pointer, at
    // the table's list address) names the record: slot in bits 24..31 (a job thread's ring, or
    // NativeSegLegacySlot = the batch's own records; clamped so a bad entry stays inside the
    // table), ring index in the low bits (masked by the ring size). The slot's device address
    // is in the table. Once per invocation (VS: DrawIndex, PS: the flat draw varying).
    const auto u64_of = [&](Id lo32, Id hi32) {
        return OpBitwiseOr(U64, OpUConvert(U64, lo32),
                           OpShiftLeftLogical(U64, OpUConvert(U64, hi32), ConstU32(32u)));
    };
    const auto load_ptr = [&](Id addr) {
        return OpLoad(U32[1], OpConvertUToPtr(physical_pointer_type_u32, addr),
                      spv::MemoryAccessMask::Aligned, 4u);
    };
    const Id list{u64_of(NativeTableLoad(Shader::NativeTableSegList),
                         NativeTableLoad(Shader::NativeTableSegList + 1))};
    const Id entry{load_ptr(OpIAdd(
        U64, list, OpUConvert(U64, OpShiftLeftLogical(U32[1], draw, ConstU32(2u)))))};
    Name(entry, "native_seg_entry");
    const Id slot{OpUMin(U32[1],
                         OpBitwiseAnd(U32[1], OpShiftRightLogical(U32[1], entry, ConstU32(24u)),
                                      ConstU32(Shader::NativeSegSlotMask)),
                         ConstU32(Shader::NativeSegSlots - 1))};
    const Id word{OpIAdd(U32[1], ConstU32(Shader::NativeTableSlotAddr),
                         OpShiftLeftLogical(U32[1], slot, ConstU32(1u)))};
    const Id base{u64_of(NativeTableLoad(word), NativeTableLoad(OpIAdd(U32[1], word, ConstU32(1u))))};
    const Id ridx{OpBitwiseAnd(U32[1], entry, ConstU32(Shader::NativeSegRingMask))};
    native_rec_ptr = OpIAdd(
        U64, base,
        OpUConvert(U64, OpIMul(U32[1], ridx, ConstU32(Shader::NativeRecordDwords * 4))));
    Name(native_rec_ptr, "native_rec_ptr");
    native_seg_entry = entry;
    // Brick C v3 item 4: the slot words - the segment's table row of this draw's material
    // (entry bits 29..31) when the batch has a table, else the record's own slot words.
    const Id mat_addr{u64_of(NativeTableLoad(Shader::NativeTableSegMat),
                             NativeTableLoad(Shader::NativeTableSegMat + 1))};
    const Id mat{OpShiftRightLogical(U32[1], entry, ConstU32(Shader::NativeSegMatShift))};
    const Id mat_ptr{OpIAdd(
        U64, mat_addr,
        OpUConvert(U64, OpIMul(U32[1], mat, ConstU32(Shader::NativeSegMatWords * 4))))};
    const Id rec_slots{
        OpIAdd(U64, native_rec_ptr, Constant(U64, u64{Shader::NativeRecordImgOffset} * 4))};
    native_seg_slot_ptr = OpSelect(U64, OpINotEqual(U1[1], mat_addr, u64_zero_value), mat_ptr,
                                   rec_slots);
    Name(native_seg_slot_ptr, "native_seg_slot_ptr");
}

Id EmitContext::NativeSegSlotLoad(u32 word) {
    const Id addr{OpIAdd(U64, native_seg_slot_ptr, Constant(U64, u64{word} * 4))};
    return OpLoad(U32[1], OpConvertUToPtr(physical_pointer_type_u32, addr),
                  spv::MemoryAccessMask::Aligned, 4u);
}

Id EmitContext::NativeVbufLoad(Id dword_index, Id buffer) {
    return OpLoad(U32[1], OpAccessChain(native_records_ptr, Sirit::ValidId(buffer) ? buffer : native_vbuf,
                                        u32_zero_value, dword_index));
}

void EmitContext::NativeCompareAttribute(u32 loc, Id vindex, Id draw) {
    // Diagnostic: the pulled value vs the real vertex input (both exist in compare mode).
    const auto& a = Shader::t_native_emit.attrs[loc];
    const auto& param = input_params[loc];
    const Id pulled = NativePullAttribute(loc, vindex);
    const Id vec_type = !param.is_integer ? F32[4]
                        : param.component_type.value == U32[1].value ? U32[4]
                                                                      : S32[4];
    const Id real = OpLoad(vec_type, param.id);
    Id bad = false_value;
    std::array<Id, 4> pb{}, rb{};
    for (u32 i = 0; i < 4; ++i) {
        const Id p = OpCompositeExtract(param.component_type, pulled, i);
        const Id r = OpCompositeExtract(param.component_type, real, i);
        Id diff;
        if (param.is_integer) {
            diff = OpINotEqual(U1[1], p, r);
            pb[i] = param.component_type.value == U32[1].value ? p : OpBitcast(U32[1], p);
            rb[i] = param.component_type.value == U32[1].value ? r : OpBitcast(U32[1], r);
        } else {
            // Bit-exact: a depth-equal pass after a generic depth prepass needs identical
            // positions, so 1-ulp differences matter.
            pb[i] = OpBitcast(U32[1], p);
            rb[i] = OpBitcast(U32[1], r);
            diff = OpINotEqual(U1[1], pb[i], rb[i]);
        }
        bad = OpLogicalOr(U1[1], bad, diff);
    }
    const Id ptr0 = OpAccessChain(native_records_ptr, native_dbg, u32_zero_value, u32_zero_value);
    const Id old = OpAtomicIAdd(U32[1], ptr0, ConstU32(1u), u32_zero_value,
                                OpSelect(U32[1], bad, ConstU32(1u), u32_zero_value));
    const Id slot = OpSelect(U32[1], OpLogicalAnd(U1[1], bad, OpULessThan(U1[1], old, ConstU32(64u))),
                             old, ConstU32(64u));
    const Id base = OpIAdd(U32[1], ConstU32(16u), OpIMul(U32[1], slot, ConstU32(16u)));
    const std::array<Id, 16> words{ConstU32(loc), vindex, draw,
                                   NativeRecordLoad(Shader::NativeRecordVoffOffset + loc),
                                   pb[0], pb[1], pb[2], pb[3], rb[0], rb[1], rb[2], rb[3],
                                   ConstU32(Shader::t_native_emit.vs_hash), ConstU32(u32(a.kind)),
                                   ConstU32(u32(a.bits)), ConstU32(u32(a.count))};
    for (u32 i = 0; i < 16; ++i) {
        OpStore(OpAccessChain(native_records_ptr, native_dbg, u32_zero_value,
                              OpIAdd(U32[1], base, ConstU32(i))),
                words[i]);
    }
}

Id EmitContext::NativePullAttribute(u32 loc, Id vindex) {
    // What the Vulkan vertex input would fetch for location `loc` (binding = location, offset 0,
    // per-vertex rate): `stride` bytes per vertex from the draw's stream offset, decoded by the
    // format into the input variable's type, missing components (0, 0, 0, 1). Reads past the
    // stream's bound size give (0, 0, 0, 1). Stream offsets / strides need not be 4-byte aligned
    // (s301: 43% of textured draws are not): two dwords per dword, funnel-shifted.
    using Shader::NativeVtxKind;
    const auto& a = Shader::t_native_emit.attrs[loc];
    const auto& param = input_params[loc];
    const bool packed = a.bits == 0; // A2B10G10R10: R 0-9, G 10-19, B 20-29, A 30-31
    const u32 bytes = packed ? 4u : u32(a.bits) / 8 * a.count;
    const u32 ndw = (bytes + 3) / 4;
    Id voff{NativeRecordLoad(Shader::NativeRecordVoffOffset + loc)};
    if (native_ring) {
        voff = OpISub(U32[1], voff, NativeTableLoad(Shader::NativeTableVbase + loc));
    }
    const Id vsize{NativeRecordLoad(Shader::NativeRecordVsizeOffset + loc)};
    const Id rel{OpIMul(U32[1], vindex, ConstU32(a.stride))};
    const Id end{OpIAdd(U32[1], rel, ConstU32(bytes))};
    const Id in_bounds{OpLogicalAnd(U1[1], OpULessThanEqual(U1[1], end, vsize),
                                    OpUGreaterThanEqual(U1[1], end, rel))};
    // Out of bounds: read the stream's first bytes (a valid address) and discard them.
    const Id addr{OpSelect(U32[1], in_bounds, OpIAdd(U32[1], voff, rel), voff)};
    const Id dw0{OpShiftRightLogical(U32[1], addr, ConstU32(2u))};
    const Id sh{
        OpShiftLeftLogical(U32[1], OpBitwiseAnd(U32[1], addr, ConstU32(3u)), ConstU32(3u))};
    const Id aligned{OpIEqual(U1[1], sh, u32_zero_value)};
    const Id inv{OpISub(U32[1], ConstU32(32u), sh)};
    std::array<Id, 4> d{};
    const auto vbuf_dwords_from = [&](Id buffer) {
        std::array<Id, 4> r{};
        Id lo{NativeVbufLoad(dw0, buffer)};
        for (u32 i = 0; i < ndw; ++i) {
            const Id hi{NativeVbufLoad(OpIAdd(U32[1], dw0, ConstU32(i + 1)), buffer)};
            const Id funnel{OpBitwiseOr(U32[1], OpShiftRightLogical(U32[1], lo, sh),
                                        OpShiftLeftLogical(U32[1], hi, inv))};
            r[i] = OpSelect(U32[1], aligned, lo, funnel);
            lo = hi;
        }
        return r;
    };
    const auto vbuf_dwords = [&] {
        if (!Sirit::ValidId(native_vbuf2)) {
            return vbuf_dwords_from(native_vbuf);
        }
        // q53: bit 16 + loc of the push word picks the second vertex SSBO (a uniform branch: the
        // push constants are the same for every invocation); the same offsets and decode.
        const Id mask_ptr{OpAccessChain(TypePointer(spv::StorageClass::PushConstant, U32[1]),
                                        push_data_block, ConstU32(Shader::PushData::BufOffsetIndex),
                                        u32_zero_value)};
        const Id use_2{OpINotEqual(U1[1],
                                   OpBitwiseAnd(U32[1], OpLoad(U32[1], mask_ptr), ConstU32(1u << (16 + loc))),
                                   u32_zero_value)};
        const Id l_2{OpLabel()};
        const Id l_1{OpLabel()};
        const Id l_merge{OpLabel()};
        OpSelectionMerge(l_merge, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(use_2, l_2, l_1);
        AddLabel(l_2);
        const auto r2 = vbuf_dwords_from(native_vbuf2);
        OpBranch(l_merge);
        AddLabel(l_1);
        const auto r1 = vbuf_dwords_from(native_vbuf);
        OpBranch(l_merge);
        AddLabel(l_merge);
        std::array<Id, 4> r{};
        for (u32 i = 0; i < ndw; ++i) {
            r[i] = OpPhi(U32[1], r2[i], l_2, r1[i], l_1);
        }
        return r;
    };
    if (native_pt && native_ring && Shader::g_pt_vertex != 0) {
        // q18: this location's stream may sit outside the bound vertex buffer (push mask bit): read it through
        // the page table from its full guest address, as Guest bindings (NativePtTranslate).
        const auto pt_dwords = [&] {
            const auto u64_of = [&](Id lo32, Id hi32) {
                return OpBitwiseOr(U64, OpUConvert(U64, lo32),
                                   OpShiftLeftLogical(U64, OpUConvert(U64, hi32), ConstU32(32u)));
            };
            const Id glo{NativeRecordLoad(Shader::NativeRecordVoffOffset + loc)};
            const Id ghi{OpBitFieldUExtract(U32[1], NativeRecordLoad(Shader::NativeRecordVhiOffset + loc / 4),
                                            ConstU32((loc % 4) * 8), ConstU32(8u))};
            const Id guest{u64_of(glo, ghi)};
            const Id a{OpIAdd(U64, guest,
                              OpUConvert(U64, OpSelect(U32[1], in_bounds, rel, u32_zero_value)))};
            const Id a4{OpBitwiseAnd(U64, a, Constant(U64, ~u64{3}))};
            // q18b: the attribute's real extent (the funnel's padding dword is not part of it).
            const Id last{OpIAdd(U64, a, Constant(U64, u64{bytes} - 1))};
            const Id fallback{u64_of(NativeTableLoad(Shader::NativeTablePtFallback),
                                     NativeTableLoad(Shader::NativeTablePtFallback + 1))};
            // SHADPS4_PT_BITS: clamped like NativePtTranslate (review 10-20, item 1).
            const Id page{PtClampPage(
                OpShiftRightLogical(U64, a4, ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS)))};
            const Id last_page_raw{
                OpShiftRightLogical(U64, last, ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS))};
            const Id above{PtAbove(last_page_raw)};
            const Id last_page{PtClampPage(last_page_raw)};
            const auto entry_at = [&](Id pg) {
                const Id ea{OpIAdd(U64, fallback, OpShiftLeftLogical(U64, pg, ConstU32(3u)))};
                return OpLoad(U64, OpConvertUToPtr(physical_pointer_type_u64, ea),
                              spv::MemoryAccessMask::Aligned, 8u);
            };
            const Id entry{entry_at(page)};
            const Id entry_last{entry_at(last_page)};
            const Id distance{OpShiftLeftLogical(U64, OpISub(U64, last_page, page),
                                                 ConstU32(VideoCore::BufferCache::CACHING_PAGEBITS))};
            const Id contiguous{OpLogicalAnd(U1[1], OpIEqual(U1[1], entry_last, OpIAdd(U64, entry, distance)),
                                             OpINotEqual(U1[1], entry_last, u64_zero_value))};
            const Id miss{OpLogicalOr(U1[1],
                                      OpLogicalOr(U1[1], OpIEqual(U1[1], entry, u64_zero_value),
                                                  OpLogicalNot(U1[1], contiguous)),
                                      above)};
            const Id npages{
                OpUConvert(U32[1], OpIAdd(U64, OpISub(U64, last_page, page), Constant(U64, u64{1})))};
            const Id flag{OpLogicalAnd(
                U1[1], OpLogicalAnd(U1[1], miss, OpINotEqual(U1[1], guest, u64_zero_value)),
                OpLogicalNot(U1[1], above))};
            OpFunctionCall(void_id, native_pt_fault, NativeTableLoad(Shader::NativeTableFaultAddr),
                           NativeTableLoad(Shader::NativeTableFaultAddr + 1), OpUConvert(U32[1], page),
                           OpSelect(U32[1], flag, npages, u32_zero_value));
            const Id in_page{OpBitwiseAnd(U64, a4, Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1))};
            const Id dev{OpSelect(U64, miss, fallback, OpIAdd(U64, entry, in_page))};
            const Id psh{OpShiftLeftLogical(U32[1], OpBitwiseAnd(U32[1], OpUConvert(U32[1], a), ConstU32(3u)),
                                            ConstU32(3u))};
            const Id paligned{OpIEqual(U1[1], psh, u32_zero_value)};
            const Id pinv{OpISub(U32[1], ConstU32(32u), psh)};
            std::array<Id, 5> w{};
            for (u32 j = 0; j <= ndw; ++j) {
                Id daddr{OpIAdd(U64, dev, Constant(U64, u64{j} * 4))};
                Id zero_it{miss};
                if (j == ndw) {
                    // q18b: the padding dword only inside the checked span (else a safe address, value 0).
                    const Id inside{OpULessThanEqual(U1[1], OpIAdd(U64, a4, Constant(U64, u64{j} * 4)), last)};
                    daddr = OpSelect(U64, inside, daddr, dev);
                    zero_it = OpLogicalOr(U1[1], miss, OpLogicalNot(U1[1], inside));
                }
                const Id ptr{OpConvertUToPtr(physical_pointer_type_u32, daddr)};
                w[j] = OpSelect(U32[1], zero_it, u32_zero_value,
                                OpLoad(U32[1], ptr, spv::MemoryAccessMask::Aligned, 4u));
            }
            std::array<Id, 4> r{};
            for (u32 i = 0; i < ndw; ++i) {
                const Id funnel{OpBitwiseOr(U32[1], OpShiftRightLogical(U32[1], w[i], psh),
                                            OpShiftLeftLogical(U32[1], w[i + 1], pinv))};
                r[i] = OpSelect(U32[1], paligned, w[i], funnel);
            }
            return r;
        };
        if (Shader::g_pt_vertex == 2) {
            d = pt_dwords(); // q18 bit-exact gate: every location through the page table
        } else {
        const Id mask_ptr{OpAccessChain(TypePointer(spv::StorageClass::PushConstant, U32[1]),
                                        push_data_block, ConstU32(Shader::PushData::BufOffsetIndex),
                                        u32_zero_value)};
        const Id use_pt{OpINotEqual(U1[1], OpBitwiseAnd(U32[1], OpLoad(U32[1], mask_ptr), ConstU32(1u << loc)),
                                    u32_zero_value)};
        const Id l_pt{OpLabel()};
        const Id l_vb{OpLabel()};
        const Id l_merge{OpLabel()};
        OpSelectionMerge(l_merge, spv::SelectionControlMask::MaskNone);
        OpBranchConditional(use_pt, l_pt, l_vb);
        AddLabel(l_pt);
        const auto dp = pt_dwords();
        OpBranch(l_merge);
        AddLabel(l_vb);
        const auto dv = vbuf_dwords();
        // q53: vbuf_dwords may end in its own merge block; the phi names this one.
        const Id l_vb_end{OpLabel()};
        OpBranch(l_vb_end);
        AddLabel(l_vb_end);
        OpBranch(l_merge);
        AddLabel(l_merge);
        for (u32 i = 0; i < ndw; ++i) {
            d[i] = OpPhi(U32[1], dp[i], l_pt, dv[i], l_vb_end);
        }
        }
    } else {
        d = vbuf_dwords();
    }
    const bool is_int = param.is_integer;
    const Id comp_type = param.component_type;
    const bool is_sint = is_int && comp_type.value == S32[1].value;
    const bool is_uint = is_int && comp_type.value == U32[1].value;
    const Id one = is_int ? (is_sint ? ConstS32(1) : ConstU32(1u)) : ConstF32(1.f);
    const Id zero = is_int ? (is_sint ? ConstS32(0) : u32_zero_value) : f32_zero_value;
    const auto from_float = [&](Id v) -> Id { return is_int ? OpBitcast(comp_type, v) : v; };
    const auto from_bits = [&](Id v) -> Id {
        return is_uint ? v : OpBitcast(comp_type, v);
    };
    // Vulkan's fixed-function conversion is the correctly rounded c / (2^b - 1) (s320: the
    // GLSL unpack's c * (1 / 255) was 1 ulp off; a depth-equal colour pass after a generic depth
    // prepass then failed on scattered pixels - speckles). Markstein: q0 = c * r, remainder by
    // fma, one fma correction; checked against every 8 / 10 / 16-bit input (0 mismatches).
    const auto exact_div = [&](Id cf, u32 d) -> Id {
        const Id r = ConstF32(1.f / float(d));
        const Id q0 = OpFMul(F32[1], cf, r);
        Decorate(q0, spv::Decoration::NoContraction);
        const Id e = OpFma(F32[1], OpFNegate(F32[1], q0), ConstF32(float(d)), cf);
        return OpFma(F32[1], e, r, q0);
    };
    const auto unorm = [&](Id u, u32 w) {
        return exact_div(OpConvertUToF(F32[1], u), (1u << w) - 1);
    };
    const auto snorm = [&](Id s_u32, u32 w) {
        const Id sf = OpConvertSToF(F32[1], OpBitcast(S32[1], s_u32));
        return OpFMax(F32[1], exact_div(sf, (1u << (w - 1)) - 1), ConstF32(-1.f));
    };
    std::array<Id, 4> c{zero, zero, zero, one};
    for (u32 i = 0; packed && i < a.count; ++i) {
        const u32 w = i < 3 ? 10u : 2u;
        const Id pos = ConstU32(i * 10u);
        const Id width = ConstU32(w);
        const Id u = OpBitFieldUExtract(U32[1], d[0], pos, width);
        const Id s = OpBitcast(S32[1], OpBitFieldSExtract(U32[1], d[0], pos, width));
        switch (a.kind) {
        case NativeVtxKind::Unorm:
            c[i] = from_float(unorm(u, w));
            break;
        case NativeVtxKind::Snorm:
            c[i] = from_float(snorm(OpBitcast(U32[1], s), w));
            break;
        case NativeVtxKind::Uint16:
            c[i] = from_bits(u);
            break;
        case NativeVtxKind::Sint16:
            c[i] = from_bits(OpBitcast(U32[1], s));
            break;
        case NativeVtxKind::Uscaled:
            c[i] = from_float(OpConvertUToF(F32[1], u));
            break;
        case NativeVtxKind::Sscaled:
            c[i] = from_float(OpConvertSToF(F32[1], s));
            break;
        default:
            UNREACHABLE_MSG("native pull: packed kind {}", u32(a.kind));
        }
    }
    for (u32 i = 0; !packed && i < a.count; ++i) {
        const u32 bit = i * a.bits;
        const Id word = d[bit / 32];
        const Id pos = ConstU32(bit % 32);
        const Id width = ConstU32(u32(a.bits));
        switch (a.kind) {
        case NativeVtxKind::Float:
            c[i] = from_float(OpBitcast(F32[1], word));
            break;
        case NativeVtxKind::Uint:
        case NativeVtxKind::Sint:
            c[i] = from_bits(word);
            break;
        case NativeVtxKind::Half:
            c[i] = from_float(
                OpCompositeExtract(F32[1], OpUnpackHalf2x16(F32[2], word), (bit % 32) / 16));
            break;
        case NativeVtxKind::Unorm:
            c[i] = from_float(unorm(OpBitFieldUExtract(U32[1], word, pos, width), a.bits));
            break;
        case NativeVtxKind::Snorm:
            c[i] = from_float(snorm(OpBitFieldSExtract(U32[1], word, pos, width), a.bits));
            break;
        case NativeVtxKind::Uint16:
            c[i] = from_bits(OpBitFieldUExtract(U32[1], word, pos, width));
            break;
        case NativeVtxKind::Sint16:
            c[i] = from_bits(OpBitFieldSExtract(U32[1], word, pos, width));
            break;
        case NativeVtxKind::Uscaled:
            c[i] = from_float(OpConvertUToF(F32[1], OpBitFieldUExtract(U32[1], word, pos, width)));
            break;
        case NativeVtxKind::Sscaled:
            c[i] = from_float(OpConvertSToF(
                F32[1], OpBitcast(S32[1], OpBitFieldSExtract(U32[1], word, pos, width))));
            break;
        default:
            UNREACHABLE_MSG("native pull: kind {}", u32(a.kind));
        }
    }
    // Out of bounds (robustBufferAccess2): components the format has read 0, missing ones
    // keep the (0, 0, 1) fill (Opus code review #2).
    const u32 present = packed ? 4u : u32(a.count);
    for (u32 i = 0; i < 4; ++i) {
        c[i] = OpSelect(comp_type, in_bounds, c[i], (i == 3 && i >= present) ? one : zero);
    }
    const Id vec_type = !is_int ? F32[4] : is_uint ? U32[4] : S32[4];
    const Id value{OpCompositeConstruct(vec_type, c[0], c[1], c[2], c[3])};
    Name(value, fmt::format("native_attr{}", loc));
    return value;
}

Id EmitContext::NativeTableLoad(u32 dword) {
    return OpLoad(U32[1],
                  OpAccessChain(native_records_ptr, native_table, u32_zero_value, ConstU32(dword)));
}

Id EmitContext::NativeTableLoad(Id dword) {
    return OpLoad(U32[1], OpAccessChain(native_records_ptr, native_table, u32_zero_value, dword));
}

Id EmitContext::LoadImage(const TextureDefinition& texture) {
    if (!texture.bindless) {
        return OpLoad(texture.image_type, texture.id);
    }
    const Id slot{native_seg ? NativeSegSlotLoad(texture.bindless_index)
                             : NativeRecordLoad(Shader::NativeRecordImgOffset +
                                                texture.bindless_index)};
    const Id ptr{OpAccessChain(TypePointer(spv::StorageClass::UniformConstant, texture.image_type),
                               texture.id, slot)};
    const Id value{OpLoad(texture.image_type, ptr)};
    return value;
}

Id EmitContext::LoadSampler(u32 index) {
    if (!native_bindless) {
        return OpLoad(sampler_type, samplers[index]);
    }
    const Id slot{native_seg
                      ? NativeSegSlotLoad(Shader::g_bl_img_max + bindless_smp_base + index)
                      : NativeRecordLoad(Shader::NativeRecordImgOffset + Shader::g_bl_img_max +
                                         bindless_smp_base + index)}; // q54: samplers after g_bl_img_max images
    const Id ptr{OpAccessChain(sampler_pointer_type, samplers[index], slot)};
    const Id value{OpLoad(sampler_type, ptr)};
    return value;
}

Id EmitContext::NativeRecordLoad(u32 dword) {
    if (native_seg) {
        // SHADPS4_NATIVE_SEG: the word through the record pointer; =2 the SSBO word too, a
        // mismatch reported through the branching helper (the pointer's value is what the draw
        // uses, so a mismatch is also visible on screen).
        const Id addr{OpIAdd(U64, native_rec_ptr, Constant(U64, u64{dword} * 4))};
        const Id value{OpLoad(U32[1], OpConvertUToPtr(physical_pointer_type_u32, addr),
                              spv::MemoryAccessMask::Aligned, 4u)};
        if (native_seg_compare) {
            const Id index{OpIAdd(U32[1], native_rec_base, ConstU32(dword))};
            const Id ssbo{OpLoad(
                U32[1], OpAccessChain(native_records_ptr, native_records, u32_zero_value, index))};
            const u32 tag = (l_stage == LogicalStage::Fragment ? 2u : 1u) | (dword << 8);
            OpFunctionCall(void_id, native_seg_report, ConstU32(dword), native_seg_entry, value,
                           ssbo, native_draw, OpUConvert(U32[1], native_rec_ptr),
                           OpUConvert(U32[1], OpShiftRightLogical(U64, native_rec_ptr,
                                                                  ConstU32(32u))),
                           ConstU32(tag));
        }
        return value;
    }
    const Id index{OpIAdd(U32[1], native_rec_base, ConstU32(dword))};
    return OpLoad(U32[1], OpAccessChain(native_records_ptr, native_records, u32_zero_value, index));
}

spv::ImageFormat GetFormat(const AmdGpu::Image& image) {
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::R32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Sint) {
        return spv::ImageFormat::R32i;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R32f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rg32f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rg32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rgba32ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::R16ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rg16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Snorm) {
        return spv::ImageFormat::Rg16Snorm;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rg8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16_16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rgba16f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format16_16_16_16 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rgba16;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::R8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Unorm) {
        return spv::ImageFormat::Rgba8;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Uint) {
        return spv::ImageFormat::Rgba8ui;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format10_11_11 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::R11fG11fB10f;
    }
    if (image.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32 &&
        image.GetNumberFmt() == AmdGpu::NumberFormat::Float) {
        return spv::ImageFormat::Rgba32f;
    }
    UNREACHABLE_MSG("Unknown storage format data_format={}, num_format={}", image.GetDataFmt(),
                    image.GetNumberFmt());
}

Id ImageType(EmitContext& ctx, const ImageResource& desc, Id sampled_type) {
    const auto image = desc.GetSharp(ctx.info);
    const auto format = desc.is_atomic ? GetFormat(image) : spv::ImageFormat::Unknown;
    const auto type = image.GetHostViewType(desc.is_array);
    const u32 sampled = desc.is_written ? 2 : 1;
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, false, false, sampled, format);
    case AmdGpu::ImageType::Color1DArray:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim1D, false, true, false, sampled, format);
    case AmdGpu::ImageType::Color2D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, false, false, sampled, format);
    case AmdGpu::ImageType::Color2DArray:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, true, false, sampled, format);
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim2D, false, false, true, sampled, format);
    case AmdGpu::ImageType::Color3D:
        return ctx.TypeImage(sampled_type, spv::Dim::Dim3D, false, false, false, sampled, format);
    default:
        break;
    }
    UNREACHABLE_MSG("Invalid texture type {}", type);
}

void EmitContext::DefineImagesAndSamplers() {
    if (native_bindless) {
        // SHADPS4_NATIVE_BINDLESS (sec. 12): every image a runtime array of its own image type
        // aliasing set 1 binding 0 (one sampled-image array), every sampler one of samplers at
        // binding 1; indexed by the record's slot words. binding.unified advances as before (set 0 unchanged).
        AddCapability(spv::Capability::RuntimeDescriptorArray);
        // No NonUniform (s433: the decoration is a per-fetch loop on NVIDIA): the slot comes
        // from the record of this draw, and DrawIndex is dynamically uniform within a draw.
        if (profile.supported_spirv < 0x00010500) {
            AddExtension("SPV_EXT_descriptor_indexing");
        }
        u32 index = info.l_stage == LogicalStage::Vertex ? Shader::t_native_emit.vs_img_base
                                                         : 0u; // q3
        for (const auto& image_desc : info.images) {
            const auto sharp = image_desc.GetSharp(info);
            const auto nfmt = sharp.GetNumberFmt();
            const VectorIds& data_types = GetAttributeType(*this, nfmt);
            const Id image_type{ImageType(*this, image_desc, data_types[1])};
            const Id pointer_type{
                TypePointer(spv::StorageClass::UniformConstant, TypeRuntimeArray(image_type))};
            const Id id{AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant)};
            Decorate(id, spv::Decoration::Binding, 0U);
            Decorate(id, spv::Decoration::DescriptorSet, 1U);
            binding.unified += image_desc.NumBindings(info);
            Name(id, fmt::format("{}_bimg{}", stage, index));
            images.push_back({
                .data_types = &data_types,
                .id = id,
                .sampled_type = TypeSampledImage(image_type),
                .image_type = image_type,
                .view_type = sharp.GetHostViewType(image_desc.is_array),
                .is_integer = AmdGpu::IsInteger(nfmt),
                .is_storage = false,
                .mip_fallback_mode = image_desc.mip_fallback_mode,
                .bindless = true,
                .bindless_index = index++,
            });
            interfaces.push_back(id);
        }
        if (info.samplers.empty()) {
            return;
        }
        sampler_type = TypeSampler();
        sampler_pointer_type = TypePointer(spv::StorageClass::UniformConstant, sampler_type);
        const Id array_ptr{
            TypePointer(spv::StorageClass::UniformConstant, TypeRuntimeArray(sampler_type))};
        for (size_t j = 0; j < info.samplers.size(); ++j) {
            const Id id{AddGlobalVariable(array_ptr, spv::StorageClass::UniformConstant)};
            Decorate(id, spv::Decoration::Binding, 1U);
            Decorate(id, spv::Decoration::DescriptorSet, 1U);
            ++binding.unified;
            Name(id, fmt::format("{}_bsamp{}", stage, j));
            samplers.push_back(id);
            interfaces.push_back(id);
        }
        return;
    }
    for (const auto& image_desc : info.images) {
        const auto sharp = image_desc.GetSharp(info);
        const auto nfmt = sharp.GetNumberFmt();
        const bool is_integer = AmdGpu::IsInteger(nfmt);
        const bool is_storage = image_desc.is_written;
        const MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const VectorIds& data_types = GetAttributeType(*this, nfmt);
        const Id sampled_type = data_types[1];
        const Id image_type{ImageType(*this, image_desc, sampled_type)};

        const u32 num_bindings = image_desc.NumBindings(info);
        Id pointee_type = image_type;
        if (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex) {
            pointee_type = TypeArray(pointee_type, ConstU32(num_bindings));
        }

        const Id pointer_type{TypePointer(spv::StorageClass::UniformConstant, pointee_type)};
        const Id id{AddGlobalVariable(pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding.unified);
        binding.unified += num_bindings;
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        // TODO better naming for resources (flattened sharp_idx is not informative)
        Name(id, fmt::format("{}_{}{}", stage, "img", image_desc.sharp_idx));
        images.push_back({
            .data_types = &data_types,
            .id = id,
            .sampled_type = is_storage ? sampled_type : TypeSampledImage(image_type),
            .image_type = image_type,
            .view_type = sharp.GetHostViewType(image_desc.is_array),
            .is_integer = is_integer,
            .is_storage = is_storage,
            .mip_fallback_mode = mip_fallback_mode,
        });
        interfaces.push_back(id);
    }
    if (std::ranges::any_of(info.images, &ImageResource::is_atomic)) {
        image_u32 = TypePointer(spv::StorageClass::Image, U32[1]);
        image_f32 = TypePointer(spv::StorageClass::Image, F32[1]);
    }
    if (info.samplers.empty()) {
        return;
    }
    sampler_type = TypeSampler();
    sampler_pointer_type = TypePointer(spv::StorageClass::UniformConstant, sampler_type);
    for (const auto& samp_desc : info.samplers) {
        const Id id{AddGlobalVariable(sampler_pointer_type, spv::StorageClass::UniformConstant)};
        Decorate(id, spv::Decoration::Binding, binding.unified++);
        Decorate(id, spv::Decoration::DescriptorSet, 0U);
        const auto sharp_desc =
            samp_desc.is_inline_sampler
                ? fmt::format("inline:{:#x}:{:#x}", samp_desc.inline_sampler.raw0,
                              samp_desc.inline_sampler.raw1)
                : fmt::format("sgpr:{}", samp_desc.sharp_idx);
        Name(id, fmt::format("{}_{}{}", stage, "samp", sharp_desc));
        samplers.push_back(id);
        interfaces.push_back(id);
    }
}

void EmitContext::DefineSharedMemory() {
    const auto num_types = std::popcount(static_cast<u32>(info.shared_types));
    if (num_types == 0) {
        return;
    }
    ASSERT(info.stage == Stage::Compute);
    const u32 shared_memory_size = runtime_info.cs_info.shared_memory_size;

    const auto make_type = [&](IR::Type type, Id element_type, u32 element_size,
                               std::string_view name) {
        if (False(info.shared_types & type)) {
            // Skip unused shared memory types.
            return std::make_tuple(Id{}, Id{}, Id{});
        }

        // A dispatch may use DS instructions with LDS_SIZE 0 (the accesses fall outside the
        // allocation on hardware). A zero-length OpTypeArray is invalid SPIR-V (VUID 08737); AMD's
        // RDNA1 driver divides by the length and crashes in vkCreateComputePipelines (GitHub #5).
        const u32 num_elements{std::max(1u, Common::DivCeil(shared_memory_size, element_size))};
        const Id array_type{TypeArray(element_type, ConstU32(num_elements))};

        const auto mem_type = [&] {
            if (num_types > 1) {
                const Id struct_type{TypeStruct(array_type)};
                Decorate(struct_type, spv::Decoration::Block);
                MemberDecorate(struct_type, 0u, spv::Decoration::Offset, 0u);
                return struct_type;
            } else {
                return array_type;
            }
        }();

        const Id pointer = TypePointer(spv::StorageClass::Workgroup, mem_type);
        const Id element_pointer = TypePointer(spv::StorageClass::Workgroup, element_type);
        const Id variable = AddGlobalVariable(pointer, spv::StorageClass::Workgroup);
        Name(variable, name);
        interfaces.push_back(variable);

        if (num_types > 1) {
            Decorate(array_type, spv::Decoration::ArrayStride, element_size);
            Decorate(variable, spv::Decoration::Aliased);
        }

        return std::make_tuple(variable, element_pointer, pointer);
    };
    std::tie(shared_memory_u16, shared_u16, shared_memory_u16_type) =
        make_type(IR::Type::U16, U16, 2u, "shared_mem_u16");
    std::tie(shared_memory_u32, shared_u32, shared_memory_u32_type) =
        make_type(IR::Type::U32, U32[1], 4u, "shared_mem_u32");
    std::tie(shared_memory_u64, shared_u64, shared_memory_u64_type) =
        make_type(IR::Type::U64, U64, 8u, "shared_mem_u64");
}

Id EmitContext::DefineFloat32ToUfloatM5(u32 mantissa_bits, const std::string_view name) {
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/util/format_r11g11b10f.h
    const auto func_type{TypeFunction(U32[1], F32[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto value{OpFunctionParameter(F32[1])};
    Name(func, name);
    AddLabel();

    const auto raw_value{OpBitcast(U32[1], value)};
    const auto exponent{
        OpBitcast(S32[1], OpBitFieldSExtract(U32[1], raw_value, ConstU32(23U), ConstU32(8U)))};
    const auto sign{OpBitFieldUExtract(U32[1], raw_value, ConstU32(31U), ConstU32(1U))};

    const auto is_zero{OpLogicalOr(U1[1], OpIEqual(U1[1], raw_value, ConstU32(0U)),
                                   OpIEqual(U1[1], sign, ConstU32(1U)))};
    const auto is_nan{OpIsNan(U1[1], value)};
    const auto is_inf{OpIsInf(U1[1], value)};
    const auto is_denorm{OpSLessThanEqual(U1[1], exponent, ConstS32(-15))};

    const auto denorm_mantissa{OpConvertFToU(
        U32[1],
        OpRoundEven(F32[1], OpFMul(F32[1], value,
                                   ConstF32(static_cast<float>(1 << (mantissa_bits + 14))))))};
    const auto denorm_overflow{
        OpINotEqual(U1[1], OpShiftRightLogical(U32[1], denorm_mantissa, ConstU32(mantissa_bits)),
                    ConstU32(0U))};
    const auto denorm{
        OpSelect(U32[1], denorm_overflow, ConstU32(1U << mantissa_bits), denorm_mantissa)};

    const auto norm_mantissa{OpConvertFToU(
        U32[1],
        OpRoundEven(F32[1],
                    OpLdexp(F32[1], value,
                            OpISub(S32[1], ConstS32(static_cast<int>(mantissa_bits)), exponent))))};
    const auto norm_overflow{
        OpUGreaterThanEqual(U1[1], norm_mantissa, ConstU32(2U << mantissa_bits))};
    const auto norm_final_mantissa{OpBitwiseAnd(
        U32[1],
        OpSelect(U32[1], norm_overflow, OpShiftRightLogical(U32[1], norm_mantissa, ConstU32(1U)),
                 norm_mantissa),
        ConstU32((1U << mantissa_bits) - 1))};
    const auto norm_final_exponent{OpBitcast(
        U32[1],
        OpIAdd(S32[1],
               OpSelect(S32[1], norm_overflow, OpIAdd(S32[1], exponent, ConstS32(1)), exponent),
               ConstS32(15)))};
    const auto norm{OpBitFieldInsert(U32[1], norm_final_mantissa, norm_final_exponent,
                                     ConstU32(mantissa_bits), ConstU32(5U))};

    const auto result{OpSelect(U32[1], is_zero, ConstU32(0U),
                               OpSelect(U32[1], is_nan, ConstU32(31u << mantissa_bits | 1U),
                                        OpSelect(U32[1], is_inf, ConstU32(31U << mantissa_bits),
                                                 OpSelect(U32[1], is_denorm, denorm, norm))))};

    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineUfloatM5ToFloat32(u32 mantissa_bits, const std::string_view name) {
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/util/format_r11g11b10f.h
    const auto func_type{TypeFunction(F32[1], U32[1])};
    const auto func{OpFunction(F32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto value{OpFunctionParameter(U32[1])};
    Name(func, name);
    AddLabel();

    const Id exponent{OpBitFieldUExtract(U32[1], value, ConstU32(mantissa_bits), ConstU32(5U))};
    const Id mantissa{OpBitFieldUExtract(U32[1], value, ConstU32(0U), ConstU32(mantissa_bits))};
    const Id mantissa_f{OpConvertUToF(F32[1], mantissa)};
    const Id a{OpSelect(F32[1], OpINotEqual(U1[1], mantissa, u32_zero_value),
                        OpFMul(F32[1], ConstF32(1.f / (1 << (14 + mantissa_bits))), mantissa_f),
                        f32_zero_value)};
    const Id b{OpBitcast(F32[1], OpBitwiseOr(U32[1], mantissa, ConstU32(0x7f800000U)))};
    const Id exponent_c{OpISub(U32[1], exponent, ConstU32(15U))};
    const Id scale_a{
        OpFDiv(F32[1], ConstF32(1.f),
               OpConvertUToF(F32[1], OpShiftLeftLogical(U32[1], u32_one_value,
                                                        OpSNegate(U32[1], exponent_c))))};
    const Id scale_b{OpConvertUToF(F32[1], OpShiftLeftLogical(U32[1], u32_one_value, exponent_c))};
    const Id scale{
        OpSelect(F32[1], OpSLessThan(U1[1], exponent_c, u32_zero_value), scale_a, scale_b)};
    const Id c{OpFMul(F32[1], scale,
                      OpFAdd(F32[1], ConstF32(1.f),
                             OpFDiv(F32[1], mantissa_f, ConstF32(f32(1 << mantissa_bits)))))};
    const Id result{OpSelect(F32[1], OpIEqual(U1[1], exponent, u32_zero_value), a,
                             OpSelect(F32[1], OpIEqual(U1[1], exponent, ConstU32(31U)), b, c))};
    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineGetBdaPointer() {
    const auto caching_pagebits{
        Constant(U64, static_cast<u64>(VideoCore::BufferCache::CACHING_PAGEBITS))};
    const auto caching_pagemask{Constant(U64, VideoCore::BufferCache::CACHING_PAGESIZE - 1)};

    const auto func_type{TypeFunction(U64, U64)};
    const auto func{OpFunction(U64, spv::FunctionControlMask::MaskNone, func_type)};
    const auto address{OpFunctionParameter(U64)};
    Name(func, "get_bda_pointer");
    AddLabel();

    const auto fault_label{OpLabel()};
    const auto available_label{OpLabel()};
    const auto merge_label{OpLabel()};

    // Get page BDA
    const auto page{OpShiftRightLogical(U64, address, caching_pagebits)};
    const auto page32{OpUConvert(U32[1], page)};
    const auto& bda_buffer{buffers[bda_pagetable_index]};
    const auto [bda_buffer_id, bda_pointer_type] = bda_buffer.Alias(PointerType::U64);
    const auto bda_ptr{OpAccessChain(bda_pointer_type, bda_buffer_id, u32_zero_value, page32)};
    const auto bda{OpLoad(U64, bda_ptr)};

    // Check if page is GPU cached
    const auto is_fault{OpIEqual(U1[1], bda, u64_zero_value)};
    OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    OpBranchConditional(is_fault, fault_label, available_label);

    // First time acces, mark as fault
    AddLabel(fault_label);
    const auto& fault_buffer{buffers[fault_buffer_index]};
    const auto [fault_buffer_id, fault_pointer_type] = fault_buffer.Alias(PointerType::U32);
    const auto page_div32{OpShiftRightLogical(U32[1], page32, ConstU32(5U))};
    const auto page_mod32{OpBitwiseAnd(U32[1], page32, ConstU32(31U))};
    const auto page_mask{OpShiftLeftLogical(U32[1], u32_one_value, page_mod32)};
    const auto fault_ptr{
        OpAccessChain(fault_pointer_type, fault_buffer_id, u32_zero_value, page_div32)};
    const auto fault_value{OpLoad(U32[1], fault_ptr)};
    const auto fault_value_masked{OpBitwiseOr(U32[1], fault_value, page_mask)};
    OpStore(fault_ptr, fault_value_masked);

    // Return null pointer
    const auto fallback_result{u64_zero_value};
    OpBranch(merge_label);

    // Value is available, compute address
    AddLabel(available_label);
    const auto offset_in_bda{OpBitwiseAnd(U64, address, caching_pagemask)};
    const auto addr{OpIAdd(U64, bda, offset_in_bda)};
    OpBranch(merge_label);

    // Merge
    AddLabel(merge_label);
    const auto result{OpPhi(U64, addr, available_label, fallback_result, fault_label)};
    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

Id EmitContext::DefineReadConst(bool dynamic) {
    const auto func_type{!dynamic ? TypeFunction(U32[1], U32[2], U32[1], U32[1])
                                  : TypeFunction(U32[1], U32[2], U32[1])};
    const auto func{OpFunction(U32[1], spv::FunctionControlMask::MaskNone, func_type)};
    const auto base{OpFunctionParameter(U32[2])};
    const auto offset{OpFunctionParameter(U32[1])};
    const auto flatbuf_offset{!dynamic ? OpFunctionParameter(U32[1]) : Id{}};
    Name(func, dynamic ? "read_const_dynamic" : "read_const");
    AddLabel();

    const auto base_lo{OpUConvert(U64, OpCompositeExtract(U32[1], base, 0))};
    const auto base_hi{OpUConvert(U64, OpCompositeExtract(U32[1], base, 1))};
    const auto base_shift{OpShiftLeftLogical(U64, base_hi, ConstU32(32U))};
    const auto base_addr{OpBitwiseOr(U64, base_lo, base_shift)};
    const auto offset_bytes{OpShiftLeftLogical(U32[1], offset, ConstU32(2U))};
    const auto addr{OpIAdd(U64, base_addr, OpUConvert(U64, offset_bytes))};

    const auto result = EmitDwordMemoryRead(addr, [&]() {
        if (dynamic) {
            return u32_zero_value;
        } else {
            return EmitFlatbufferLoad(flatbuf_offset);
        }
    });

    OpReturnValue(result);
    OpFunctionEnd();
    return func;
}

void EmitContext::DefineFunctions() {
    if (native_pt) {
        native_pt_fault = DefineNativePtFault(); // called from the prologue (DefineMain)
    }
    if (info.uses_pack_10_11_11) {
        f32_to_uf11 = DefineFloat32ToUfloatM5(6, "f32_to_uf11");
        f32_to_uf10 = DefineFloat32ToUfloatM5(5, "f32_to_uf10");
    }
    if (info.uses_unpack_10_11_11) {
        uf11_to_f32 = DefineUfloatM5ToFloat32(6, "uf11_to_f32");
        uf10_to_f32 = DefineUfloatM5ToFloat32(5, "uf10_to_f32");
    }
    if (info.uses_dma) {
        get_bda_pointer = DefineGetBdaPointer();
    }
    if (native_pt_compare && Sirit::ValidId(native_pt_dbg)) {
        native_pt_report = DefineNativePtReport("native_pt_report", 0, 16);
    }
    if (native_seg_compare && Sirit::ValidId(native_pt_dbg)) {
        native_seg_report = DefineNativePtReport("native_seg_report", 1, 2048);
    }

    if (True(info.readconst_types & Info::ReadConstType::Immediate)) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses immediate ReadConst", info.pgm_hash);
        read_const = DefineReadConst(false);
    }
    if (True(info.readconst_types & Info::ReadConstType::Dynamic)) {
        LOG_DEBUG(Render_Recompiler, "Shader {:#x} uses dynamic ReadConst", info.pgm_hash);
        read_const_dynamic = DefineReadConst(true);
    }
}

} // namespace Shader::Backend::SPIRV
