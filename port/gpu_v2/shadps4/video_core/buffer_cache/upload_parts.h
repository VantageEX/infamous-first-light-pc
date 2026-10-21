// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace VideoCore {
template <class Ranges, class Begin, class Copy, class End>
void ForEachUploadPart(const Ranges& ranges, uint64_t limit, Begin begin, Copy copy, End end) {
    assert(limit != 0);
    uint64_t used = 0;
    for (const auto& range : ranges) {
        for (uint64_t done = 0; done < range.size;) {
            if (used == 0) begin();
            const uint64_t count = std::min<uint64_t>(range.size - done, limit - used);
            copy(range.dstOffset + done, used, count);
            used += count;
            done += count;
            if (used == limit) { end(used); used = 0; }
        }
    }
    if (used != 0) end(used);
}
}
