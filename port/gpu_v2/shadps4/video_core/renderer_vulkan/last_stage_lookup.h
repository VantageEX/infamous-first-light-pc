// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Vulkan {
// Command-processor owned. This caches an index, never a pointer into a vector
// that can grow, and rechecks the caller's compatibility predicate on every hit.
class LastStageLookup {
public:
    template <typename Validate>
    std::optional<uint32_t> Find(uint64_t key, size_t count, Validate&& validate) const {
        if (valid && key == saved_key && index < count && validate(index)) return index;
        return std::nullopt;
    }
    void Remember(uint64_t key, uint32_t value) {
        saved_key = key;
        index = value;
        valid = true;
    }
private:
    uint64_t saved_key{};
    uint32_t index{};
    bool valid{};
};
}
