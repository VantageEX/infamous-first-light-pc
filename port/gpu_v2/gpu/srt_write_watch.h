// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>

namespace iss::g4 {

// Used only under the PageManager page lock. The generation prevents a lease
// released after a fault/remap from removing a newly armed lease of that page.
class SrtWriteWatchState {
public:
    [[nodiscard]] bool Arm(std::uint8_t total_write_watchers,
                           std::uint64_t& generation_out) noexcept {
        if (count == 255 || total_write_watchers == 255) return false;
        ++count;
        generation_out = generation;
        return true;
    }

    [[nodiscard]] bool Check(std::uint64_t token_generation) const noexcept {
        return token_generation != 0 && token_generation == generation && count != 0;
    }

    [[nodiscard]] bool Active() const noexcept { return count != 0; }

    [[nodiscard]] bool Release(std::uint64_t token_generation) noexcept {
        if (!Check(token_generation)) return false;
        --count;
        return true;
    }

    [[nodiscard]] std::uint8_t Invalidate() noexcept {
        const auto removed = count;
        if (removed != 0) {
            count = 0;
            if (++generation == 0) generation = 1;
        }
        return removed;
    }

private:
    std::uint8_t count{};
    std::uint64_t generation{1};
};

} // namespace iss::g4
