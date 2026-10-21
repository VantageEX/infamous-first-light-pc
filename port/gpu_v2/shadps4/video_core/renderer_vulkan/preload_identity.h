// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <utility>

namespace Vulkan {
// Identity is the persisted key and the complete metadata bytes, never the
// asymmetric runtime "compatible specialization" predicate.
class PreloadIdentity {
public:
    bool Matches(uint64_t key, const std::vector<uint8_t>& metadata) const {
        const auto it = records.find(key);
        return it != records.end() && it->second == metadata;
    }
    void Remember(uint64_t key, std::vector<uint8_t> metadata) {
        records.emplace(key, std::move(metadata));
    }
    void Clear() { records.clear(); }
private:
    std::unordered_map<uint64_t, std::vector<uint8_t>> records;
};
} // namespace Vulkan
