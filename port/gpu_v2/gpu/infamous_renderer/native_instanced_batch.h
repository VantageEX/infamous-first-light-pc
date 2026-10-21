#pragma once

#include "native_geometry_batch.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace iss::native_geometry {

// Exact approved-pack keys whose VS SPIR-V has one vertex input, reads its only
// guest SSBO as 18 dwords per InstanceIndex, and does not access clip planes.
bool IsApprovedInstancedKey(std::uint64_t graphics_key_hash,
                            std::uint32_t vertex_hash, std::uint32_t permutation);

struct InstancedDraw {
    std::uint32_t vertex_hash{};
    std::uint32_t fragment_hash{};
    std::uint64_t graphics_pipeline_key_hash{};
    std::uint32_t vertex_permutation{};
    bool exact_spirv_verified{};

    std::uint32_t queue{};
    std::uint32_t primitive_type{};
    std::uint32_t index_type{};
    std::uint32_t index_offset{};
    std::int32_t vertex_offset{};
    std::uint32_t first_instance{};
    std::uint32_t instance_count{};
    std::uint32_t binding_count{};
    std::uint32_t attribute_count{};

    Stream vertex{};        // Effective one-attribute V# after GetSharp.
    Bytes instance_bytes{}; // Complete 72-byte records for this draw's SSBO0.
    std::uint64_t instance_guest_base{};
    bool instance_cpu_owned{};
    bool instance_mapped{};
    bool instance_snapshot_stable{};

    Bytes index_bytes{};    // Starts at the effective uint16 index address.
    std::uint32_t index_count{};
    bool index_cpu_owned{};
    bool index_mapped{};
    bool index_snapshot_stable{};
    StateSnapshot state{};  // Excludes only V# and SSBO0 ranges being replaced.
};

struct PackedInstancedBatch {
    std::vector<std::byte> vertices{};
    std::vector<std::byte> instances{};
    std::vector<std::byte> indices{};
    std::vector<DrawIndexedIndirectCommand> commands{};
    std::uint32_t vertex_stride{};
    std::uint32_t vertex_control_word{};
    std::uint32_t vertex_format_word{};
    std::uint64_t graphics_pipeline_key_hash{};
    std::uint32_t vertex_hash{};
    std::uint32_t vertex_permutation{};
    std::uint32_t draw_count{};
};

struct InstancedBuildResult {
    Reject reject{Reject::None};
    PackedInstancedBatch batch{};
    explicit operator bool() const { return reject == Reject::None; }
};

// All-or-nothing geometry transformation. Nonzero firstInstance is emitted for
// draws after the first; never record without drawIndirectFirstInstance enabled.
InstancedBuildResult BuildInstanced(std::span<const InstancedDraw> draws);

} // namespace iss::native_geometry
