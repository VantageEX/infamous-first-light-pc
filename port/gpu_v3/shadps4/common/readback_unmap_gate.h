// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace Common {

/// Serializes readback admission with guest-memory unmap/remap transitions.
///
/// A blocked interval remains blocked until the owner explicitly calls Unblock. In particular,
/// an unmap that has no matching remap leaves its interval as a tombstone rather than allowing a
/// late readback to publish into the removed mapping. The owner calls Unblock from MapMemory.
class ReadbackUnmapGate {
public:
    ReadbackUnmapGate() = default;
    ReadbackUnmapGate(const ReadbackUnmapGate&) = delete;
    ReadbackUnmapGate& operator=(const ReadbackUnmapGate&) = delete;

    /// Blocks [address, address + size). Overflow is conservatively clamped to UINT64_MAX.
    void Block(std::uint64_t address, std::uint64_t size) {
        const auto range = NormalizeForMutation(address, size);
        if (!range) {
            return;
        }

        std::scoped_lock lock{mutex};
        blocked.push_back(*range);
        std::ranges::sort(blocked, {}, &Range::begin);

        std::vector<Range> merged;
        merged.reserve(blocked.size());
        for (const Range current : blocked) {
            if (merged.empty() || !Touches(merged.back(), current)) {
                merged.push_back(current);
                continue;
            }
            merged.back().end = std::max(merged.back().end, current.end);
        }
        blocked.swap(merged);
    }

    /// Unblocks [address, address + size), splitting existing intervals as necessary.
    /// Overflow is conservatively clamped to UINT64_MAX. An empty range is a no-op.
    void Unblock(std::uint64_t address, std::uint64_t size) {
        const auto range = NormalizeForMutation(address, size);
        if (!range) {
            return;
        }

        std::scoped_lock lock{mutex};
        std::vector<Range> remaining;
        remaining.reserve(blocked.size() + 1);
        for (const Range current : blocked) {
            if (!Overlaps(current, *range)) {
                remaining.push_back(current);
                continue;
            }
            if (current.begin < range->begin) {
                remaining.push_back({current.begin, range->begin - 1});
            }
            if (range->end < current.end) {
                remaining.push_back({range->end + 1, current.end});
            }
        }
        blocked.swap(remaining);
    }

    /// Invokes callback while holding the gate mutex only when the whole range is unblocked.
    /// Empty or overflowing admission ranges are rejected conservatively.
    template <typename Callback>
    [[nodiscard]] bool TryAdmit(std::uint64_t address, std::uint64_t size,
                                Callback&& callback) {
        const auto range = NormalizeForAdmission(address, size);
        if (!range) {
            return false;
        }

        std::scoped_lock lock{mutex};
        for (const Range current : blocked) {
            if (Overlaps(current, *range)) {
                return false;
            }
        }
        std::invoke(std::forward<Callback>(callback));
        return true;
    }

private:
    struct Range {
        std::uint64_t begin;
        std::uint64_t end;
    };

    // Store inclusive endpoints so UINT64_MAX can be represented without an end+1 sentinel.
    static std::optional<Range> NormalizeForMutation(std::uint64_t address,
                                                     std::uint64_t size) {
        if (size == 0) {
            return std::nullopt;
        }
        const std::uint64_t max_size = std::numeric_limits<std::uint64_t>::max() - address;
        return Range{address, size - 1 > max_size ? std::numeric_limits<std::uint64_t>::max()
                                                  : address + size - 1};
    }

    static std::optional<Range> NormalizeForAdmission(std::uint64_t address,
                                                      std::uint64_t size) {
        if (size == 0 || size - 1 > std::numeric_limits<std::uint64_t>::max() - address) {
            return std::nullopt;
        }
        return Range{address, address + size - 1};
    }

    static bool Overlaps(const Range lhs, const Range rhs) {
        return lhs.begin <= rhs.end && rhs.begin <= lhs.end;
    }

    static bool Touches(const Range lhs, const Range rhs) {
        // Block sorts by begin before calling this helper, so lhs is the left interval.
        return Overlaps(lhs, rhs) ||
               (lhs.end != std::numeric_limits<std::uint64_t>::max() && lhs.end + 1 >= rhs.begin);
    }

    std::mutex mutex;
    std::vector<Range> blocked;
};

} // namespace Common
