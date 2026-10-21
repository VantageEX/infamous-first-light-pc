#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace iss::native_geometry {

// Byte-for-byte layout of VkDrawIndexedIndirectCommand. The renderer may copy these
// commands to a Vulkan indirect buffer only after proving the record conditions below.
struct DrawIndexedIndirectCommand {
    std::uint32_t index_count;
    std::uint32_t instance_count;
    std::uint32_t first_index;
    std::int32_t vertex_offset;
    std::uint32_t first_instance;
};
static_assert(sizeof(DrawIndexedIndirectCommand) == 20);
static_assert(offsetof(DrawIndexedIndirectCommand, index_count) == 0);
static_assert(offsetof(DrawIndexedIndirectCommand, instance_count) == 4);
static_assert(offsetof(DrawIndexedIndirectCommand, first_index) == 8);
static_assert(offsetof(DrawIndexedIndirectCommand, vertex_offset) == 12);
static_assert(offsetof(DrawIndexedIndirectCommand, first_instance) == 16);

struct Bytes {
    const std::byte* data{};
    std::size_t size{};
};

struct Stream {
    Bytes bytes{};
    std::uint64_t guest_base{};
    std::uint32_t descriptor_size{};
    std::uint32_t stride{};
    std::uint32_t descriptor_control_word{}; // V# raw word 1.
    std::uint32_t format_word{}; // V# raw word 3, including data/number format.
    std::uint32_t sgpr_base{};
    std::uint32_t dword_offset{};
    std::uint32_t instance_data{};
    bool cpu_owned{};            // No GPU-produced bytes in the descriptor range.
    bool mapped{};               // Entire descriptor range is mapped.
    bool snapshot_stable{};      // Bytes copied under the caller's generation guard.
};

// Canonical, complete state is supplied by the rasterizer. It must cover pipeline
// key, dynamic state, attachments, descriptors/material and feedback dependencies,
// excluding only the three V# addresses and the bound index range being packed.
// Byte equality is used; a hash alone is not accepted as a state proof.
struct StateSnapshot {
    Bytes canonical{};
    bool complete{};
    bool immutable_through_record{};
};

struct Draw {
    std::uint32_t vertex_hash{};
    std::uint32_t fragment_hash{};
    std::uint64_t graphics_pipeline_key_hash{};
    std::uint32_t vertex_permutation{};
    std::uint32_t queue{};
    std::uint32_t primitive_type{};
    std::uint32_t index_type{};
    std::uint32_t instance_count{};
    std::uint32_t first_instance{};
    std::uint32_t binding_count{};
    std::uint32_t attribute_count{};
    bool fetch_inputs_only{}; // Fetch pointer is not also read by shader resources.
    bool vertex_index_unobserved{}; // Exact VS SPIR-V has no load of VertexIndex.
    std::uint32_t index_offset{}; // Guest packet offset, required zero for this contract.
    std::int32_t vertex_offset{};
    std::array<Stream, 3> streams{};
    Bytes index_bytes{};         // Starts at the effective bound index address.
    std::uint32_t index_count{};
    bool index_cpu_owned{};
    bool index_mapped{};
    bool index_snapshot_stable{};
    StateSnapshot state{};
};

enum class Reject {
    None,
    EmptyOrTooManyDraws,
    UnsupportedShaderOrLayout,
    UnverifiedState,
    DifferentState,
    UnstableSource,
    InvalidRange,
    OutOfBoundsIndex,
    OutputOverflow,
};

struct PackedBatch {
    std::array<std::vector<std::byte>, 3> vertices{};
    std::vector<std::byte> indices{};
    std::vector<DrawIndexedIndirectCommand> commands{};
    std::uint32_t draw_count{};
};

struct BuildResult {
    Reject reject{Reject::None};
    PackedBatch batch{};
    explicit operator bool() const { return reject == Reject::None; }
};

// All-or-nothing geometry transformation. Does not record GPU commands or alter
// guest state. Each draw keeps its order and index values; all three destination
// streams start its vertices at one common logical vertex offset.
BuildResult Build(std::span<const Draw> draws);

// Pre-upload requirements. TryUploadAndRecord handles the single StreamBuffer
// Map/Commit and checks Vulkan device capabilities before emitting commands.
struct RecordProof {
    bool complete_state_equal_at_record{};
    bool no_intervening_feedback_or_barrier{};
};
bool MayAttemptRecord(const BuildResult& result, const RecordProof& proof);

} // namespace iss::native_geometry
