// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace VideoCore {
template <class Read>
bool CheckVersionSnapshot(std::vector<uint64_t>& snapshot, size_t count, bool compare, Read read) {
    if (compare && snapshot.size() != count) return false;
    if (!compare) snapshot.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const auto version = read(i);
        if (!version) return false;
        if (compare) {
            if (snapshot[i] != *version) return false;
        } else {
            snapshot[i] = *version;
        }
    }
    return count != 0;
}
}
