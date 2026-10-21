#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace iss::native_geometry {

struct InstancedPacket {
    std::uint64_t fetch_descriptors{};
    std::array<std::uint32_t, 4> ssbo_raw{};
    std::uint64_t index_address{};
    std::uint32_t index_count{};
    std::uint32_t max_index_size{};
    std::uint32_t initiator{};
    std::uint32_t instances{};
    std::size_t packet_word{};
};

struct InstancedRunInputs {
    std::span<const std::uint32_t> registers;
    std::size_t shader_register_base{};
    std::size_t uconfig_register_base{};
    std::size_t fetch_pointer_word{};   // VS userdata[2].
    std::size_t ssbo_descriptor_word{}; // VS userdata[4].
    std::uint32_t index_type{};
    InstancedPacket first{};
};

struct InstancedRun {
    std::vector<InstancedPacket> draws{};
    std::size_t consumed_words{};
    std::size_t packet_count{};
};

// Bounded lookahead in the current DCB only. Fetch descriptor pointer, SSBO0
// descriptor and NumInstances may change; every other setter must write the
// original value. DMA, synchronization, markers, another draw kind and unknown
// packets end the run before they execute. Does not mutate guest registers.
InstancedRun CollectInstancedRun(std::span<const std::uint32_t> following,
                                 const InstancedRunInputs& inputs,
                                 std::size_t max_draws = 64,
                                 std::size_t max_packets = 4096);

} // namespace iss::native_geometry
