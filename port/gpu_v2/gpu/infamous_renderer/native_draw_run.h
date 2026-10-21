// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <vector>

namespace iss::native_geometry {
struct IndexedDrawInput {
    std::uint64_t fetch_descriptors{};
    std::uint64_t index_address{};
    std::uint32_t index_count{}, max_index_size{}, initiator{};
    std::size_t packet_word{};
};
struct RunInputs {
    std::span<const std::uint32_t> registers;
    std::size_t shader_register_base{}, uconfig_register_base{}, fetch_pointer_word{};
    std::uint32_t index_type{}, instance_register{};
    IndexedDrawInput first;
};
struct IndexedRun {
    std::vector<IndexedDrawInput> draws;
    // Trailing setters/padding are deliberately left for the ordinary interpreter.
    std::size_t consumed_words{}, packet_count{};
};

// Look ahead only inside this DCB. Accept neutral packets and writes equal to the current
// state, except the two words selecting the vertex-fetch descriptor list. Stop before any
// different material/state, indirect buffer, draw kind, synchronization or unknown packet.
// This function neither mutates registers nor reads a guest address stored in a packet.
IndexedRun CollectIndexedRun(std::span<const std::uint32_t> following, const RunInputs&,
                             std::size_t max_draws = 64, std::size_t max_packets = 4096);
// Keep complete accepted draws only; setters preceding the next draw stay for
// the ordinary decoder. Recounts the original packet budget of the prefix.
IndexedRun PrefixIndexedRun(const IndexedRun&, std::span<const std::uint32_t> following,
                            std::size_t draws);
// Exact referenced vertex prefix for zero-base uint16 indexed fetch. Reject
// any original out-of-bounds fetch and any format wider than its stride.
std::optional<std::uint64_t> UsedVertexPrefix(std::span<const std::uint16_t> indices,
    std::uint32_t stride,std::uint32_t fetch_bytes,std::uint64_t declared_size);
struct MemoryRead { std::uint64_t address{}, size{}; };
struct SnapshotRange { std::uint64_t address{}, size{}, offset{}; };
struct ReadSnapshotPlan {
    std::vector<SnapshotRange> ranges;
    std::vector<std::uint64_t> offsets; // One offset per input read, in original order.
    std::uint64_t size{};
};
// Merge overlapping/adjacent reads only. Gaps are never read. This plans immutable CPU
// storage; mapping, ownership and generation are still checked by the actual reader.
std::optional<ReadSnapshotPlan> PlanReadSnapshot(std::span<const MemoryRead> reads,
                                                std::uint64_t capacity = 16 * 1024 * 1024);
}
