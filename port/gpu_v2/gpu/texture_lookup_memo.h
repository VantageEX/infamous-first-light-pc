// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace iss::texture_lookup {
// CP-owned value cache. The caller puts every lifetime/revision dependency in Key.
// A collision can only evict. Values are identities or copied metadata, never texels.
template <typename Key, typename Value, std::size_t Capacity>
class Memo {
    static_assert(Capacity && (Capacity & (Capacity - 1)) == 0);
    struct Entry { Key key{}; Value value{}; bool valid{}; };
    std::array<Entry, Capacity> entries{};
    static std::size_t Slot(std::uint64_t hint) {
        return std::size_t((hint ^ (hint >> 17)) * 0x9e3779b97f4a7c15ULL) & (Capacity - 1);
    }
public:
    const Value* Find(std::uint64_t hint, const Key& key) const {
        const auto& entry = entries[Slot(hint)];
        return entry.valid && entry.key == key ? &entry.value : nullptr;
    }
    void Store(std::uint64_t hint, const Key& key, const Value& value) {
        entries[Slot(hint)] = {key, value, true};
    }
};
}
