// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace iss {
// A snapshot of single-element, non-overlapping push-descriptor writes. It
// caches descriptor values, never resource contents, validity or barriers.
struct DescriptorValue {
    std::uint32_t binding, array_element, count, type;
    std::uint64_t a, b, c;
    bool SameShape(const DescriptorValue& other) const noexcept {
        return binding == other.binding && array_element == other.array_element &&
               count == other.count && type == other.type;
    }
    bool SameValue(const DescriptorValue& other) const noexcept {
        return a == other.a && b == other.b && c == other.c;
    }
};

template<std::size_t Capacity> class DescriptorDelta {
public:
    void Reset() noexcept { valid = false; }

    // nullopt means unsupported input: caller emits its full original batch.
    // A layout/shape change emits every write, including immutable samplers.
    std::optional<std::size_t> Select(std::uint64_t layout,
                                      std::span<const DescriptorValue> input,
                                      std::span<std::uint8_t> selected) noexcept {
        static_assert(Capacity <= 256);
        if (!layout || input.size() > Capacity || selected.size() < input.size()) {
            Reset(); return std::nullopt;
        }
        for (std::size_t i = 0; i < input.size(); ++i) {
            // This excludes arrays/spanning writes and duplicate/overlapping
            // bindings. Partial filtering must not omit a later overwriter.
            if (input[i].count != 1 || input[i].array_element != 0 ||
                (i && input[i-1].binding >= input[i].binding)) {
                Reset(); return std::nullopt;
            }
        }
        bool same_shape = valid && previous_layout == layout && count == input.size();
        if (same_shape)
            for (std::size_t i = 0; i < input.size(); ++i)
                same_shape &= previous[i].SameShape(input[i]);
        std::size_t changed = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (!same_shape || !previous[i].SameValue(input[i]))
                selected[changed++] = static_cast<std::uint8_t>(i);
            previous[i] = input[i];
        }
        count = input.size(); previous_layout = layout; valid = true;
        return changed;
    }
private:
    std::array<DescriptorValue, Capacity> previous;
    std::uint64_t previous_layout{};
    std::size_t count{};
    bool valid{};
};
} // namespace iss
