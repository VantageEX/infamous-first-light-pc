// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The CP split's draw record and stage blocks (moved verbatim from liverpool.cpp, 2026-10-10, so the
// architecture-A worker parser builds the same records; notes/architecture-a-2026-10-10.md sec. 16).

#include <array>
#include <cstring>
#include <span>

#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

/// A draw: the front's classification and lookup plus the per-stage data copied right after the
/// lookup. Followed by stage_count stage blocks (WriteStage).
struct SplitDrawRecord {
    Vulkan::Rasterizer::SplitDraw draw;
    u64 ctx_epoch;           ///< Liverpool::context_change_epoch at the draw (viewport memo).
    const char* marker;      ///< Host marker packet name (static string), never dereferenced
                             ///< on the front's behalf; null = no marker.
    const void* cmd_address; ///< For the host marker text only (formatted, never read).
    u32 stage_count;
    u32 verify;              ///< DrawVerifyFull / DrawVerifySnapshot (VERIFY only), else 0.
    u32 vi_bytes;            ///< FRONT_DERIVE: bytes of the VertexInputsPayload block after the
                             ///< stage blocks (0: none).
    u32 pad;
    u32 nr_tag;              ///< SHADPS4_NATIVE_REC: the job-thread run this draw heads (0: none)
    u32 nr_idx;
    u32 job_tag;             ///< SHADPS4_NATIVE_REC >= 2: this draw's OWN job record (0: none),
    u32 job_idx;             ///< heads, members and refused members alike (REC_PREP, sec. 9)
    u32 job_member;          ///< 1: posted by the member path (NativeRecMember)
    u32 job_pad;
    // SHADPS4_NATIVE_SEGB (brick B): the front segment builder's prediction for this draw
    // (segb_id 0: outside every segment); the member's (slot, ring index) are job_tag / job_idx.
    u32 segb_id;          ///< 1-based segment number
    u32 segb_member;      ///< position in the segment (0 = the opener)
    u64 segb_index_addr;  ///< predicted index guest address (= index union lo + firstIndex * size)
    u32 segb_index_count; ///< predicted indexCount
    u32 segb_instances;   ///< predicted instanceCount
};
static_assert(sizeof(SplitDrawRecord) % 8 == 0);

inline u32 StageBytes(const Shader::Info& info) {
    const size_t bytes = sizeof(CpSplit::StagePayload) +
                         (info.user_data.size() + info.flattened_ud_buf.size()) * sizeof(u32);
    return static_cast<u32>((bytes + 7) & ~size_t{7});
}

/// The same block size from the per-draw word counts alone (architecture A worker parser: the
/// words live in a GraphicsLookupScratch stage, not in the shared Info).
inline u32 StageBytes(u32 ud_dw, u32 flat_dw) {
    const size_t bytes = sizeof(CpSplit::StagePayload) + (size_t{ud_dw} + flat_dw) * sizeof(u32);
    return static_cast<u32>((bytes + 7) & ~size_t{7});
}

/// WriteStage from caller-owned per-draw data (Vulkan::GraphicsLookupScratch::Stage after
/// PipelineCache::LookupGraphics): `info` is the SHARED Shader::Info pointer the back reads the
/// immutable part of; the words are exactly what the front's lookup would have left in that Info
/// for the same registers. Same bytes, same layout as WriteStage(out, info, logical_stage).
inline u8* WriteStage(u8* out, const Shader::Info* info, VAddr pgm_base,
                      std::span<const u32> user_data, std::span<const u32> flattened_ud_buf,
                      u32 logical_stage) {
    auto* stage = reinterpret_cast<CpSplit::StagePayload*>(out);
    stage->info = info;
    stage->pgm_base = pgm_base;
    stage->logical_stage = logical_stage;
    stage->ud_dw = static_cast<u32>(user_data.size());
    stage->flat_dw = static_cast<u32>(flattened_ud_buf.size());
    stage->pad = 0;
    auto* dwords = reinterpret_cast<u32*>(stage + 1);
    if (!user_data.empty()) {
        std::memcpy(dwords, user_data.data(), user_data.size_bytes());
    }
    if (!flattened_ud_buf.empty()) {
        std::memcpy(dwords + stage->ud_dw, flattened_ud_buf.data(),
                    flattened_ud_buf.size_bytes());
    }
    return out + StageBytes(stage->ud_dw, stage->flat_dw);
}

/// Copies what the back reads of a stage's per-draw data (Shader::BoundStage): user data,
/// flat buffer and pgm_base, as the lookup just left them in the shared Info.
inline u8* WriteStage(u8* out, const Shader::Info& info, u32 logical_stage) {
    auto* stage = reinterpret_cast<CpSplit::StagePayload*>(out);
    stage->info = &info;
    stage->pgm_base = info.pgm_base;
    stage->logical_stage = logical_stage;
    stage->ud_dw = static_cast<u32>(info.user_data.size());
    stage->flat_dw = static_cast<u32>(info.flattened_ud_buf.size());
    stage->pad = 0;
    auto* dwords = reinterpret_cast<u32*>(stage + 1);
    if (!info.user_data.empty()) {
        std::memcpy(dwords, info.user_data.data(), info.user_data.size_bytes());
    }
    if (!info.flattened_ud_buf.empty()) {
        std::memcpy(dwords + stage->ud_dw, info.flattened_ud_buf.data(),
                    info.flattened_ud_buf.size() * sizeof(u32));
    }
    return out + StageBytes(info);
}

/// The back's BoundStages, pointing into the record (valid until Channel::Done). Returns the
/// end of the last stage block.
inline const u8* ReadStages(const u8* in, u32 count,
                     std::array<Shader::BoundStage, Shader::MaxStageTypes>& out) {
    for (u32 i = 0; i < count; ++i) {
        const auto* stage = reinterpret_cast<const CpSplit::StagePayload*>(in);
        const auto* dwords = reinterpret_cast<const u32*>(stage + 1);
        ASSERT(stage->logical_stage < Shader::MaxStageTypes);
        auto& bound = out[stage->logical_stage];
        bound.info = static_cast<const Shader::Info*>(stage->info);
        bound.user_data = std::span<const u32>{dwords, stage->ud_dw};
        bound.flattened_ud_buf = std::span<const u32>{dwords + stage->ud_dw, stage->flat_dw};
        bound.pgm_base = stage->pgm_base;
        const size_t bytes = sizeof(CpSplit::StagePayload) +
                             (size_t{stage->ud_dw} + stage->flat_dw) * sizeof(u32);
        in += (bytes + 7) & ~size_t{7};
    }
    return in;
}

} // namespace AmdGpu
