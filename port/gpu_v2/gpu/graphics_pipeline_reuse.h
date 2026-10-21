// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace iss::pipeline_reuse {

struct Stamp {
    std::uint64_t context{}, upload_epoch{}, host_write{}, recording_tick{};
    std::uint32_t stages{}, vertex_offset{}, instance_offset{};
    bool operator==(const Stamp&) const = default;
};

struct StageIdentity {
    const void* program{};
    std::uint64_t module{}, shader_hash{};
    std::uint32_t permutation{}, code_dwords{};
    bool operator==(const StageIdentity&) const = default;
};

inline bool SameBytes(std::span<const std::byte> a, std::span<const std::byte> b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

// A resolved scout entry is the existing specialization proof. The reuse path must
// additionally retain the concrete program/module, register header and runtime inputs.
// Resource addresses may change: the caller publishes THIS entry's flat words only
// after both stages pass, never the words from the draw that seeded the cache.
inline bool StageMatches(const StageIdentity& saved, const StageIdentity& current,
                         std::span<const std::byte> saved_registers,
                         std::span<const std::byte> live_registers,
                         std::span<const std::byte> saved_runtime,
                         std::span<const std::byte> current_runtime,
                         std::span<const std::byte> live_user_data,
                         std::span<const std::byte> scouted_user_data) {
    return saved.program != nullptr && saved.module != 0 && saved == current &&
           SameBytes(saved_registers, live_registers) &&
           SameBytes(saved_runtime, current_runtime) &&
           SameBytes(live_user_data, scouted_user_data);
}

} // namespace iss::pipeline_reuse
