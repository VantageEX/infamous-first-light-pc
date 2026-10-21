// SPDX-License-Identifier: GPL-2.0-or-later
#include "infamous_renderer/native_draw_run.h"
#include <algorithm>

namespace iss::native_geometry {
std::optional<std::uint64_t> UsedVertexPrefix(std::span<const std::uint16_t> indices,
    std::uint32_t stride,std::uint32_t fetch_bytes,std::uint64_t declared_size) {
    if(indices.empty() || !stride || !fetch_bytes || fetch_bytes>stride || !declared_size)return {};
    const auto maximum=*std::max_element(indices.begin(),indices.end());
    const auto bytes=(std::uint64_t(maximum)+1)*stride;
    if(bytes>declared_size)return {};
    return bytes;
}
IndexedRun PrefixIndexedRun(const IndexedRun& run,std::span<const std::uint32_t> words,
                            std::size_t draws) {
    if(draws<2 || draws>run.draws.size())return {};
    const auto last=run.draws[draws-1].packet_word;
    if(last>words.size() || words.size()-last<6)return {};
    const auto end=last+6;
    std::size_t cursor=0,packets=0;
    while(cursor<end) {
        const auto header=words[cursor],type=header>>30;
        const std::size_t length=type==2?1:type==3?((((header>>16)&0x3fff)+1)&0x3fff)+1:0;
        if(!length || length>end-cursor)return {};
        cursor+=length;++packets;
    }
    IndexedRun out;out.draws.assign(run.draws.begin(),run.draws.begin()+draws);
    out.consumed_words=end;out.packet_count=packets;return out;
}
std::optional<ReadSnapshotPlan> PlanReadSnapshot(std::span<const MemoryRead> reads,
                                                std::uint64_t capacity) {
    if (reads.empty()) return std::nullopt;
    auto sorted = std::vector<MemoryRead>(reads.begin(), reads.end());
    for (const auto& r : sorted)
        if (!r.address || !r.size || r.address > UINT64_MAX - r.size) return std::nullopt;
    std::ranges::sort(sorted, {}, &MemoryRead::address);
    ReadSnapshotPlan result;
    result.ranges.reserve(sorted.size());
    for (const auto& r : sorted) {
        if (!result.ranges.empty() && r.address <= result.ranges.back().address + result.ranges.back().size) {
            auto& last = result.ranges.back();
            last.size = std::max(last.address + last.size, r.address + r.size) - last.address;
        } else result.ranges.push_back({r.address, r.size, 0});
    }
    for (auto& r : result.ranges) {
        if (r.size > capacity - result.size) return std::nullopt;
        r.offset = result.size; result.size += r.size;
    }
    result.offsets.reserve(reads.size());
    for (const auto& r : reads) {
        auto it = std::upper_bound(result.ranges.begin(), result.ranges.end(), r.address,
            [](std::uint64_t address, const SnapshotRange& range) { return address < range.address; });
        if (it == result.ranges.begin()) return std::nullopt;
        --it;
        if (r.address + r.size > it->address + it->size) return std::nullopt;
        result.offsets.push_back(it->offset + r.address - it->address);
    }
    return result;
}

IndexedRun CollectIndexedRun(std::span<const std::uint32_t> words, const RunInputs& in,
                             std::size_t max_draws, std::size_t max_packets) {
    IndexedRun out;
    if (max_draws < 2 || max_draws > 256 || in.index_type != 0 ||
        !in.first.index_count || in.first.index_count > in.first.max_index_size ||
        in.first.initiator != 0 || !in.first.index_address || (in.first.index_address & 1) ||
        in.fetch_pointer_word >= in.registers.size() ||
        in.registers.size() - in.fetch_pointer_word < 2) return out;
    out.draws.reserve(std::min<std::size_t>(max_draws, 64));
    out.draws.push_back(in.first);
    std::uint64_t fetch = in.first.fetch_descriptors;
    std::size_t cursor = 0, packets = 0;
    while (cursor < words.size() && cursor < 4096 && out.draws.size() < max_draws &&
           packets < max_packets) {
        const auto header = words[cursor];
        const auto type = header >> 30;
        if (type == 2) { ++cursor; ++packets; continue; }
        if (type != 3 || (header & 0xff)) break; // No predication/compute/reserved flags.
        const std::size_t count = (((header >> 16) & 0x3fff) + 1) & 0x3fff;
        const std::size_t length = count + 1;
        if (!count || length > words.size() - cursor || length > 4096 - cursor) break;
        const auto packet = words.subspan(cursor, length);
        const auto opcode = (header >> 8) & 0xff;
        bool accepted = false, draw = false;
        switch (opcode) {
        case 0x10: // NOP payload type 0: the remaining payload is intentionally ignored by GNM.
            accepted = packet[1] == 0;
            break;
        case 0x76: // SET_SH_REG
        case 0x79: { // SET_UCONFIG_REG
            if (length < 3 || (packet[1] & 0xffff0000)) break;
            // Liverpool routes the compute SH range into a separate cs_state, not reg_array.
            // Even an apparently identical write there must stay in the ordinary decoder.
            if (opcode == 0x76 && (packet[1] >= 0x200 || length - 2 > 0x200 - packet[1])) break;
            const auto base = opcode == 0x76 ? in.shader_register_base : in.uconfig_register_base;
            if (base > in.registers.size() || packet[1] > in.registers.size() - base) break;
            const auto first = base + packet[1];
            if (length - 2 > in.registers.size() - first) break;
            auto next_fetch = fetch;
            accepted = true;
            for (std::size_t i = 2; i < length; ++i) {
                const auto reg = first + i - 2;
                if (opcode == 0x76 && reg == in.fetch_pointer_word)
                    next_fetch = (next_fetch & 0xffffffff00000000ULL) | packet[i];
                else if (opcode == 0x76 && reg == in.fetch_pointer_word + 1)
                    next_fetch = (next_fetch & 0xffffffffULL) | (std::uint64_t{packet[i]} << 32);
                else if (packet[i] != in.registers[reg]) { accepted = false; break; }
            }
            if (accepted) fetch = next_fetch;
            break;
        }
        case 0x2a: accepted = length == 2 && packet[1] == in.index_type; break;
        case 0x2f: accepted = length == 2 && packet[1] == in.instance_register; break;
        case 0x27: { // DRAW_INDEX_2
            if (length != 6 || !packet[4] || packet[4] > packet[1] || packet[5] != 0) break;
            const auto address = std::uint64_t{packet[2]} | (std::uint64_t{packet[3]} << 32);
            if (!address || (address & 1) || !fetch) break;
            out.draws.push_back({fetch, address, packet[4], packet[1], packet[5], cursor});
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
}
