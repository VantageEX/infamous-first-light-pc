// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/code_match.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {

/**
 * s_load_dwordx4 s[8:11], s[2:3], 0x00
 * s_load_dwordx4 s[12:15], s[2:3], 0x04
 * s_load_dwordx4 s[16:19], s[2:3], 0x08
 * s_waitcnt     lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
 * buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen
 * buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen
 * s_waitcnt     0
 * s_setpc_b64   s[0:1]

 * s_load_dwordx4  s[4:7], s[2:3], 0x0
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
 * s_load_dwordx4  s[4:7], s[2:3], 0x8
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[8:11], v0, s[4:7], 0 idxen
 * s_waitcnt       vmcnt(0) & expcnt(0) & lgkmcnt(0)
 * s_setpc_b64     s[0:1]

 * A normal fetch shader looks like the above, the instructions are generated
 * using input semantics on cpu side. Load instructions can either be separate or interleaved
 * We take the reverse way, extract the original input semantics from these instructions.
 **/

static bool IsTypedBufferLoad(const Gcn::GcnInst& inst) {
    return inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_X ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XY ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZ ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZW;
}

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base) {
    const u32* code;
    std::memcpy(&code, &info.user_data[sgpr_base], sizeof(code));
    return code;
}

static FetchShaderData DecodeFetchShader(const u32* code) {
    FetchShaderData data{};
    GcnCodeSlice code_slice(code, code + std::numeric_limits<u32>::max());
    GcnDecodeContext decoder;

    struct VsharpLoad {
        u32 dword_offset{};
        u32 base_sgpr{};
    };
    std::array<VsharpLoad, 104> loads{};

    u32 semantic_index = 0;
    while (!code_slice.atEnd()) {
        const auto inst = decoder.decodeInstruction(code_slice);
        data.size += inst.length;

        if (inst.opcode == Opcode::S_SETPC_B64) {
            break;
        }

        if (inst.inst_class == InstClass::ScalarMemRd) {
            loads[inst.dst[0].code] = VsharpLoad{inst.control.smrd.offset, inst.src[0].code * 2};
            continue;
        }

        if (inst.opcode == Opcode::V_ADD_I32) {
            const auto vgpr = inst.dst[0].code;
            const auto sgpr = s8(inst.src[0].code);
            switch (vgpr) {
            case 0: // V0 is always the vertex offset
                data.vertex_offset_sgpr = sgpr;
                break;
            case 3: // V3 is always the instance offset
                data.instance_offset_sgpr = sgpr;
                break;
            default:
                UNREACHABLE();
            }
        }

        if (inst.inst_class == InstClass::VectorMemBufFmt) {
            // SRSRC is in units of 4 SPGRs while SBASE is in pairs of SGPRs
            const u32 base_sgpr = inst.src[2].code * 4;
            auto& attrib = data.attributes.emplace_back();
            attrib.semantic = semantic_index++;
            attrib.dest_vgpr = inst.src[1].code;
            attrib.num_elements = inst.control.mubuf.count;
            attrib.sgpr_base = loads[base_sgpr].base_sgpr;
            attrib.dword_offset = loads[base_sgpr].dword_offset;
            attrib.inst_offset = inst.control.mtbuf.offset;
            attrib.instance_data = inst.src[0].code;
            if (IsTypedBufferLoad(inst)) {
                attrib.data_format = inst.control.mtbuf.dfmt;
                attrib.num_format = inst.control.mtbuf.nfmt;
            }
        }
    }

    return data;
}

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info) {
    if (!info.has_fetch_shader) {
        return std::nullopt;
    }
    if (g_synthetic_fetch) [[unlikely]] {
        return *g_synthetic_fetch;
    }
    const auto* code = GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_FETCH_CACHE");
        return !value || value[0] != '0';
    }();
    static const bool verify = [] {
        const char* value = std::getenv("SHADPS4_FETCH_VERIFY");
        return value && value[0] == '1';
    }();
    if (!enabled) {
        return DecodeFetchShader(code);
    }

    // Parsing depends only on the instruction bytes, not the current vertex buffers,
    // user data or instance divisors. Those are still read by StageSpecialization.
    // Address selects a bounded slot; every hit validates the full instruction stream,
    // including its terminating instruction. No guest write tracking is assumed.
    struct Entry {
        const u32* address{};
        std::vector<u32> words;
        FetchShaderData data;
    };
    static thread_local std::array<Entry, 256> entries;
    const auto address = reinterpret_cast<uintptr_t>(code);
    auto& entry = entries[((address >> 2) ^ (address >> 12) ^ (address >> 20)) & 255];
    const bool hit = entry.address == code && !entry.words.empty() &&
                     FetchShaderCodeMatches(code, entry.words);
    if (!hit) {
        auto decoded = DecodeFetchShader(code);
        // Keep per-thread storage bounded even for unusual generated programs.
        if (decoded.size > 4096) {
            return decoded;
        }
        entry.address = code;
        entry.words.assign(code, code + decoded.size / sizeof(u32));
        entry.data = std::move(decoded);
    }
    if (verify) {
        const auto reference = DecodeFetchShader(code);
        const auto& cached = entry.data;
        // Existing equality intentionally ignores several fields used by other paths.
        ASSERT(cached.size == reference.size &&
               cached.vertex_offset_sgpr == reference.vertex_offset_sgpr &&
               cached.instance_offset_sgpr == reference.instance_offset_sgpr &&
               cached.attributes.size() == reference.attributes.size());
        for (size_t i = 0; i < cached.attributes.size(); ++i) {
            const auto& a = cached.attributes[i];
            const auto& b = reference.attributes[i];
            ASSERT(a == b && a.inst_offset == b.inst_offset && a.data_format == b.data_format &&
                   a.num_format == b.num_format);
        }
        static thread_local u64 calls = 0, hits = 0;
        hits += hit;
        if (++calls % 100000 == 0) {
            LOG_INFO(Render, "FetchCacheVerify: calls={} hits={} hit_pct={:.2f}", calls, hits,
                     100.0 * hits / calls);
        }
    }
    return entry.data;
}

} // namespace Shader::Gcn
