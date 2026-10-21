// SPDX-License-Identifier: GPL-2.0-or-later
// Owning scalar description of an ASC DMA packet handed to the G frontend.
#pragma once

#include <cstdint>

namespace iss::g4 {

enum class DmaFallbackKind : std::uint8_t { Fill, Copy };

struct DmaFallbackPacket {
    std::uint32_t guest_qid{};
    std::uint64_t guest_generation{};
    std::uint64_t submission_sequence{};
    std::uint64_t packet_sequence{};
    DmaFallbackKind kind{DmaFallbackKind::Fill};
    std::uint64_t dst{};
    std::uint64_t src{};
    std::uint32_t size{};
    std::uint32_t value{};
    bool dst_gds{};
    bool src_gds{};
    bool record_label{};

    [[nodiscard]] bool Valid() const noexcept {
        return guest_qid > 0 && guest_qid < 57 && guest_generation != 0 &&
               submission_sequence != 0 && packet_sequence != 0 && size != 0 &&
               !(dst_gds && src_gds) &&
               (kind == DmaFallbackKind::Copy ? !record_label
                                              : (!src_gds && record_label != dst_gds));
    }
};

enum class DmaFallbackStatus : std::uint8_t { Busy, Ready, Failed };
struct DmaFallbackResult {
    DmaFallbackStatus status{DmaFallbackStatus::Failed};
    std::uint64_t graphics_tick{};
};

} // namespace iss::g4
