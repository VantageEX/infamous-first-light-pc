// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

#include <bit>
#include "shader_recompiler/frontend/code_match.h"
#include "shader_recompiler/specialization.h"

namespace Shader {

// Inputs to StageSpecialization, after Info::RefreshFlatBuf. This is a guard on a
// previously resolved permutation, not a snapshot of bindings or resource contents.
// Keep these dependencies in step with StageSpecialization's constructor. In
// particular, addresses may change while their specialization properties do not.
struct PreparedSpecialization {
    static constexpr size_t MaxFetchBytes = 4096;
    static constexpr size_t MaxFetchAttributes = 32;

    bool valid{};
    size_t permutation{};
    Backend::Bindings start{};
    std::array<u8, sizeof(RuntimeInfo)> runtime{};
    std::vector<u64> resource_inputs;
    std::vector<u32> fetch_words;
    std::optional<Gcn::FetchShaderData> fetch;

    static bool Eligible(const Info& info, Stage stage, LogicalStage logical) {
        // Tessellation reads another guest constant buffer, and geometry reads a
        // copy program. They retain the original path until those dependencies
        // have their own contract. Cover both ordinary graphics stages here.
        // SHADPS4_PREPARED_PROGRAMS_COMPUTE=1: compute has neither a fetch shader nor a
        // tessellation constant buffer; its RuntimeInfo (workgroup setup) is compared whole.
        static const bool compute = [] {
            const char* value = std::getenv("SHADPS4_PREPARED_PROGRAMS_COMPUTE");
            return value && value[0] == '1';
        }();
        return info.stage == stage && info.l_stage == logical &&
               ((stage == Stage::Fragment && logical == LogicalStage::Fragment) ||
                (stage == Stage::Vertex && logical == LogicalStage::Vertex) ||
                (compute && stage == Stage::Compute && logical == LogicalStage::Compute));
    }

    static u64 BufferInput(const BufferResource& desc, const Info& info) {
        const auto sharp = desc.GetSharp(info);
        if (!sharp) {
            return 0;
        }
        const auto raw = std::bit_cast<std::array<u32, 4>>(sharp);
        // Raw format/swizzle bits are stronger than their decoded equivalence.
        // IsStorage is evaluated live, including any future size-based decision.
        return (u64(raw[3]) << 32) | (1ULL << 31) | sharp.stride |
               (u64(sharp.swizzle_enable) << 14) | (u64(desc.IsStorage(sharp)) << 15) |
               (u64(desc.is_formatted) << 16);
    }

    static u64 ImageInput(const ImageResource& desc, const Info& info) {
        const auto sharp = desc.GetSharp(info);
        if (!sharp) {
            return 0;
        }
        const auto raw = std::bit_cast<std::array<u64, 4>>(sharp);
        // data/number formats; dst select; base/last mip; guest type. These
        // determine every image specialization field, including block-1D and
        // dynamic mip binding count. Dimensions and addresses do not enter it.
        return (1ULL << 63) | ((raw[0] >> 52) & 0x3ffULL) |
               (((raw[1] >> 32) & 0xf00fffffULL) << 16);
    }

    bool ForEachInput(const Info& info, auto&& consume) const {
        if (info.stage == Stage::Vertex && fetch) {
            for (const auto& desc : fetch->attributes) {
                const auto sharp = desc.GetSharp(info);
                const auto raw = std::bit_cast<std::array<u32, 4>>(sharp);
                // Number class and remapped dst select depend only on these
                // format/swizzle bits. Instance divisors are in RuntimeInfo.
                if (!consume(sharp ? ((1ULL << 63) | (raw[3] & 0x7ffffU)) : 0)) {
                    return false;
                }
            }
        }
        for (const auto& desc : info.buffers) {
            if (!consume(BufferInput(desc, info))) {
                return false;
            }
        }
        for (const auto& desc : info.images) {
            if (!consume(ImageInput(desc, info))) {
                return false;
            }
        }
        for (const auto& desc : info.fmasks) {
            const auto sharp = desc.GetSharp(info);
            const u64 key = sharp ? ((1ULL << 63) | sharp.width | (u64(sharp.height) << 14)) : 0;
            if (!consume(key)) {
                return false;
            }
        }
        for (const auto& desc : info.samplers) {
            const auto sharp = desc.GetSharp(info);
            if (!consume(u64(sharp.force_unnormalized) | (u64(sharp.force_degamma) << 1))) {
                return false;
            }
        }
        return true;
    }

    bool Matches(const Info& info, const RuntimeInfo& current, Backend::Bindings binding) const {
        if (!valid || start != binding ||
            std::memcmp(runtime.data(), &current, sizeof(current)) != 0) {
            return false;
        }
        return MatchesInputs(info);
    }

    // The guest-visible half of Matches: fetch code and resource specialization inputs,
    // after Info::RefreshFlatBuf. Only for a caller that separately guarantees the
    // captured RuntimeInfo and incoming Bindings (prepared graphics pipelines derive
    // both from an exact register snapshot plus the program pair).
    bool MatchesInputs(const Info& info) const {
        if (!valid || info.has_fetch_shader != fetch.has_value()) {
            return false;
        }
        if (fetch && !Gcn::FetchShaderCodeMatches(
                         Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base), fetch_words)) {
            return false;
        }
        size_t index{};
        return ForEachInput(info,
                            [&](u64 key) {
                                return index < resource_inputs.size() &&
                                       resource_inputs[index++] == key;
                            }) &&
               index == resource_inputs.size();
    }

    void Capture(const Info& info, const RuntimeInfo& current, Backend::Bindings binding,
                 const StageSpecialization& incoming, size_t selected) {
        valid = false;
        const auto& parsed = incoming.fetch_shader_data;
        if (parsed &&
            (parsed->size == 0 || parsed->size > MaxFetchBytes || parsed->size % sizeof(u32) != 0 ||
             parsed->attributes.size() > MaxFetchAttributes)) {
            return;
        }
        fetch = parsed; // The incoming decode, not the chosen module's weaker equality.
        fetch_words.clear();
        if (fetch) {
            const auto* code = Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
            fetch_words.assign(code, code + fetch->size / sizeof(u32));
        }
        resource_inputs.clear();
        ForEachInput(info, [&](u64 key) {
            resource_inputs.push_back(key);
            return true;
        });
        std::memcpy(runtime.data(), &current, sizeof(current));
        start = binding;
        permutation = selected;
        valid = true;
    }
};

} // namespace Shader
