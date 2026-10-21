// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>
#include <utility>

namespace iss::g4 {

// with_mapping_lock invokes its callback while a shared native mapping lease
// is held. Predicate and write run in that same lease, never after it.
template <class WithMappingLock, class Predicate, class Write>
bool ConditionalBackingWrite(WithMappingLock&& with_mapping_lock,
                             std::uint64_t expected_epoch, Predicate&& predicate,
                             Write&& write) {
    if (expected_epoch == 0) return false;
    return std::forward<WithMappingLock>(with_mapping_lock)([&](std::uint64_t epoch) {
        if (epoch != expected_epoch || !std::forward<Predicate>(predicate)()) return false;
        std::forward<Write>(write)();
        return true;
    });
}

} // namespace iss::g4
