// SPDX-License-Identifier: GPL-2.0-or-later
// Logical sparse-bind publication. Caller serializes writers and registry reads;
// the submitted watermark is atomic for submit-side inspection.
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

namespace iss::g4 {

template <class ArenaKey>
class SparseResidencyPublication {
public:
    template <class ReserveTick>
    void NoteBind(ArenaKey arena, ReserveTick&& reserve_tick) {
        if (pending_tick == 0) pending_tick = std::forward<ReserveTick>(reserve_tick)();
        required_ticks[arena] = pending_tick;
    }

    [[nodiscard]] std::uint64_t TakePendingTick() noexcept {
        return std::exchange(pending_tick, 0);
    }
    [[nodiscard]] std::uint64_t PendingTick() const noexcept { return pending_tick; }

    void PublishSubmitted(std::uint64_t tick) noexcept {
        submitted_tick.store(tick, std::memory_order_release);
    }
    [[nodiscard]] std::uint64_t SubmittedTick() const noexcept {
        return submitted_tick.load(std::memory_order_acquire);
    }

    // A newly planned bind hides even an older submitted bind for the same arena.
    [[nodiscard]] std::optional<std::uint64_t> RequiredSubmittedTick(
        ArenaKey arena) const noexcept {
        const auto it = required_ticks.find(arena);
        if (it == required_ticks.end() || it->second == 0 ||
            it->second > SubmittedTick()) return std::nullopt;
        return it->second;
    }

private:
    std::uint64_t pending_tick{};
    std::unordered_map<ArenaKey, std::uint64_t> required_ticks;
    std::atomic<std::uint64_t> submitted_tick{};
};

} // namespace iss::g4
