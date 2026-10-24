// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <vector>
#include "common/types.h"
#include "shader_recompiler/info.h"

namespace Serialization {
struct Archive;
}

namespace Shader::Gcn {

/// SHADPS4_CACHE_REBUILD (2026-10-16): V#s standing in for the guest vertex-buffer table while a
/// vertex shader is re-translated at WarmUp (the table is not in memory yet), keyed by
/// (sgpr_base << 8) | dword_offset; made from the cached VS attribute specialization (number
/// class, swizzle). Per thread (thread_local), set around one translation.
inline thread_local const std::vector<std::pair<u32, AmdGpu::Buffer>>* g_synthetic_vs_sharps =
    nullptr;

struct VertexAttribute {
    enum InstanceIdType : u8 {
        None = 0,
        OverStepRate0 = 1,
        OverStepRate1 = 2,
        Plain = 3,
    };

    u8 semantic;      ///< Semantic index of the attribute
    u8 dest_vgpr;     ///< Destination VGPR to load first component.
    u8 num_elements;  ///< Number of components to load
    u8 sgpr_base;     ///< SGPR that contains the pointer to the list of vertex V#
    u8 dword_offset;  ///< The dword offset of the V# that describes this attribute.
    u8 instance_data; ///< Indicates that the buffer will be accessed in instance rate
    u8 inst_offset;   ///< Instruction offset applied on the formatted buffer loads
    u8 data_format{}; ///< Data format override when typed buffer loads are used
    u8 num_format{};  ///< Number format override when typed buffer loads are used

    InstanceIdType GetStepRate() const {
        return static_cast<InstanceIdType>(instance_data);
    }

    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer{};
        bool synthetic = false;
        if (g_synthetic_vs_sharps) [[unlikely]] {
            const u32 key = (u32{sgpr_base} << 8) | dword_offset;
            for (const auto& [k, v] : *g_synthetic_vs_sharps) {
                if (k == key) {
                    buffer = v;
                    synthetic = true;
                    break;
                }
            }
        }
        if (!synthetic) {
            buffer = info.template ReadUdReg<AmdGpu::Buffer>(sgpr_base, dword_offset);
        }
        buffer.base_address += inst_offset;
        if (data_format) {
            buffer.data_format = data_format;
            buffer.num_format = num_format;
        }
        return buffer;
    }

    bool operator==(const VertexAttribute& other) const {
        return semantic == other.semantic && dest_vgpr == other.dest_vgpr &&
               num_elements == other.num_elements && sgpr_base == other.sgpr_base &&
               dword_offset == other.dword_offset && instance_data == other.instance_data;
    }
};

struct FetchShaderData {
    u32 size = 0;
    std::vector<VertexAttribute> attributes;
    s8 vertex_offset_sgpr = -1;   ///< SGPR of vertex offset from VADDR
    s8 instance_offset_sgpr = -1; ///< SGPR of instance offset from VADDR

    bool operator==(const FetchShaderData& other) const {
        return attributes == other.attributes && vertex_offset_sgpr == other.vertex_offset_sgpr &&
               instance_offset_sgpr == other.instance_offset_sgpr;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& buffer);
};

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base);

/// SHADPS4_CACHE_REBUILD (2026-10-16): while set, ParseFetchShader returns this instead of
/// decoding guest memory - a vertex shader re-translated at WarmUp from the game's files has no
/// fetch subroutine in memory yet; its cached FetchShaderData stands in (per thread).
inline thread_local const FetchShaderData* g_synthetic_fetch = nullptr;

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info);

} // namespace Shader::Gcn
