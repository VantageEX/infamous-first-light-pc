#include "infamous_renderer/native_instanced_run.h"

#include <algorithm>

namespace iss::native_geometry {

InstancedRun CollectInstancedRun(std::span<const std::uint32_t> words,
                                 const InstancedRunInputs& in,
                                 std::size_t max_draws, std::size_t max_packets) {
    InstancedRun out{};
    if (max_draws < 2 || max_draws > 256 || !max_packets || in.index_type != 0 ||
        !in.first.fetch_descriptors || !in.first.index_address ||
        (in.first.index_address & 1) || !in.first.index_count ||
        in.first.index_count > in.first.max_index_size || in.first.initiator != 0 ||
        !in.first.instances || in.first.instances > 256 ||
        in.fetch_pointer_word >= in.registers.size() ||
        in.registers.size() - in.fetch_pointer_word < 2 ||
        in.ssbo_descriptor_word >= in.registers.size() ||
        in.registers.size() - in.ssbo_descriptor_word < 4 ||
        in.ssbo_descriptor_word <= in.fetch_pointer_word + 1) return out;
    out.draws.reserve(std::min<std::size_t>(max_draws, 64));
    out.draws.push_back(in.first);
    auto fetch = in.first.fetch_descriptors;
    auto ssbo = in.first.ssbo_raw;
    auto instances = in.first.instances;
    std::size_t cursor = 0, packets = 0;
    while (cursor < words.size() && cursor < 4096 && out.draws.size() < max_draws &&
           packets < max_packets) {
        const std::uint32_t header = words[cursor];
        const std::uint32_t type = header >> 30;
        if (type == 2) { ++cursor; ++packets; continue; }
        if (type != 3 || (header & 0xff)) break;
        const std::size_t count = (((header >> 16) & 0x3fff) + 1) & 0x3fff;
        const std::size_t length = count + 1;
        if (!count || length > words.size() - cursor || length > 4096 - cursor) break;
        const auto packet = words.subspan(cursor, length);
        const auto opcode = (header >> 8) & 0xff;
        bool accepted = false, draw = false;
        switch (opcode) {
        case 0x10: // Neutral NOP only: empty, or the game's 0x6875000d payload, which the
                   // graphics CP ignores (no PM4CmdNop::PayloadType, not an ISS marker).
            accepted = packet[1] == 0 || packet[1] == 0x6875000du;
            break;
        case 0x76: // SET_SH_REG
        case 0x79: { // SET_UCONFIG_REG
            if (length < 3 || (packet[1] & 0xffff0000)) break;
            const auto base = opcode == 0x76 ? in.shader_register_base : in.uconfig_register_base;
            if (base > in.registers.size() || packet[1] > in.registers.size() - base) break;
            const auto first = base + packet[1];
            if (length - 2 > in.registers.size() - first) break;
            auto next_fetch = fetch;
            auto next_ssbo = ssbo;
            accepted = true;
            for (std::size_t i = 2; i < length; ++i) {
                const auto reg = first + i - 2;
                if (opcode == 0x76 && reg == in.fetch_pointer_word)
                    next_fetch = (next_fetch & 0xffffffff00000000ull) | packet[i];
                else if (opcode == 0x76 && reg == in.fetch_pointer_word + 1)
                    next_fetch = (next_fetch & 0xffffffffull) | (std::uint64_t(packet[i]) << 32);
                else if (opcode == 0x76 && reg >= in.ssbo_descriptor_word &&
                         reg < in.ssbo_descriptor_word + 4)
                    next_ssbo[reg - in.ssbo_descriptor_word] = packet[i];
                else if (packet[i] != in.registers[reg]) { accepted = false; break; }
            }
            if (accepted) { fetch = next_fetch; ssbo = next_ssbo; }
            break;
        }
        case 0x2a: // INDEX_TYPE
            accepted = length == 2 && packet[1] == in.index_type;
            break;
        case 0x2f: // NUM_INSTANCES
            accepted = length == 2 && packet[1] > 0 && packet[1] <= 256;
            if (accepted) instances = packet[1];
            break;
        case 0x27: { // DRAW_INDEX_2
            if (length != 6 || !packet[4] || packet[4] > packet[1] || packet[5] != 0) break;
            const auto address = std::uint64_t(packet[2]) | (std::uint64_t(packet[3]) << 32);
            if (!address || (address & 1) || !fetch || !ssbo[0]) break;
            out.draws.push_back({fetch, ssbo, address, packet[4], packet[1], packet[5],
                                 instances, cursor});
            accepted = draw = true;
            break;
        }
        default: break;
        }
        if (!accepted) break;
        cursor += length;
        ++packets;
        if (draw) { out.consumed_words = cursor; out.packet_count = packets; }
    }
    if (out.draws.size() < 2) return {};
    return out;
}

} // namespace iss::native_geometry
