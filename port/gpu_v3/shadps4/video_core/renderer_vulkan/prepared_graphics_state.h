// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <type_traits>
#include "common/types.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {

// Exact raw snapshot of every register PipelineCache::RefreshGraphicsKey,
// BuildRuntimeInfo(Vertex/Fragment) and RefreshGraphicsStages read for an ordinary
// VS+PS draw, plus both program hashes (the original path's own code identity).
// Whole raw registers are kept where the original decodes only some fields, so an
// unrelated bit can only cause a miss. Addresses enter only as the null tests the
// original performs. Keep in step with those three functions.
struct PreparedGraphicsState {
    static constexpr u32 MaxPsInputs = 32;

    u64 fs_hash;
    u64 vs_hash;
    u32 stage_enable;
    u32 primitive_type;
    u32 clipper_control;
    u32 polygon_control;
    u32 depth_render_override;
    u32 color_control;
    u32 color_shader_mask;
    u32 color_target_mask;
    u32 color_export_format;
    u32 depth_shader_control;
    u32 vs_output_control;
    u32 step_rate_0;
    u32 step_rate_1;
    u32 ps_input_ena;
    u32 ps_input_addr;
    u32 num_interp;
    u32 z_export_format;
    u32 z_info;
    u32 stencil_info;
    u32 depth_presence; // bit0 z_read_base != 0, bit1 stencil_read_base != 0
    std::array<u32, 2> fs_settings;
    std::array<u32, 2> vs_settings;
    u32 color_presence; // bit per color buffer: base_address && info.format
    u32 reserved;
    std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> blend_control;
    std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> color_info;   // zero unless present
    std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> color_attrib; // zero unless present
    std::array<u32, MaxPsInputs> ps_inputs;                  // first num_interp, rest zero

    template <typename T>
    static u32 Raw(const T& reg) noexcept {
        static_assert(sizeof(T) == sizeof(u32) && std::is_trivially_copyable_v<T>);
        u32 value;
        std::memcpy(&value, &reg, sizeof(value));
        return value;
    }

    // False for anything other than the plain VS+PS configuration.
    bool Capture(const AmdGpu::Regs& regs, u64 fs_hash_, u64 vs_hash_) noexcept {
        if (regs.stage_enable.raw != AmdGpu::ShaderStageEnable::VgtStages::Vs ||
            regs.num_interp > MaxPsInputs) {
            return false;
        }
        std::memset(this, 0, sizeof(*this));
        fs_hash = fs_hash_;
        vs_hash = vs_hash_;
        stage_enable = Raw(regs.stage_enable);
        primitive_type = Raw(regs.primitive_type);
        clipper_control = Raw(regs.clipper_control);
        polygon_control = Raw(regs.polygon_control);
        depth_render_override = Raw(regs.depth_render_override);
        color_control = Raw(regs.color_control);
        color_shader_mask = regs.color_shader_mask.raw;
        color_target_mask = regs.color_target_mask.raw;
        color_export_format = regs.color_export_format.raw;
        depth_shader_control = Raw(regs.depth_shader_control);
        vs_output_control = Raw(regs.vs_output_control);
        step_rate_0 = regs.vgt_instance_step_rate_0;
        step_rate_1 = regs.vgt_instance_step_rate_1;
        ps_input_ena = Raw(regs.ps_input_ena);
        ps_input_addr = Raw(regs.ps_input_addr);
        num_interp = regs.num_interp;
        z_export_format = Raw(regs.z_export_format);
        z_info = Raw(regs.depth_buffer.z_info);
        stencil_info = Raw(regs.depth_buffer.stencil_info);
        depth_presence = u32(regs.depth_buffer.z_read_base != 0) |
                         (u32(regs.depth_buffer.stencil_read_base != 0) << 1);
        fs_settings = std::bit_cast<std::array<u32, 2>>(regs.ps_program.settings);
        vs_settings = std::bit_cast<std::array<u32, 2>>(regs.vs_program.settings);
        for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
            blend_control[cb] = Raw(regs.blend_control[cb]);
            const auto& col_buf = regs.color_buffers[cb];
            if (col_buf) {
                color_presence |= 1u << cb;
                color_info[cb] = col_buf.info.raw;
                color_attrib[cb] = col_buf.attrib.raw;
            }
        }
        for (u32 i = 0; i < num_interp; ++i) {
            ps_inputs[i] = Raw(regs.ps_inputs[i]);
        }
        return true;
    }

    bool operator==(const PreparedGraphicsState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(*this)) == 0;
    }

    // Table indexing only; a hit always compares the full snapshot.
    u64 Hash() const noexcept {
        u64 h = fs_hash ^ (vs_hash * 0x9E3779B97F4A7C15ULL);
        h ^= u64(color_info[0]) | (u64(blend_control[0]) << 32);
        h ^= (u64(color_target_mask) << 17) ^ (u64(depth_shader_control) << 49) ^
             (u64(primitive_type) << 7) ^ u64(z_info);
        h *= 0xD6E8FEB86659FD93ULL;
        return h ^ (h >> 32);
    }
};
static_assert(std::is_trivial_v<PreparedGraphicsState>);
static_assert(std::has_unique_object_representations_v<PreparedGraphicsState>);

} // namespace Vulkan
