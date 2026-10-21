// SPDX-License-Identifier: GPL-2.0-or-later
// Maps real Scheduler timeline submissions to CpCoordination open ticks.
// Every call is serialized by Liverpool::submit_mutex. A post-submit callback
// may arrive while an explicit seal is in flight; reconciliation then waits
// until FinishSeal has released the open tick.
#pragma once

#include <array>
#include <cstdint>

#include "g4_cp_coordination.h"

namespace iss::g4 {

template <class Coordination>
class SchedulerTimelineBridge {
public:
    explicit SchedulerTimelineBridge(Coordination& coordination_) noexcept
        : coordination{coordination_} {}

    // `through_tick` must be observed only after Scheduler submitted its timeline
    // signal to the driver, never from the pre-submit callback.
    bool ObserveSubmitted(CpLane lane, std::uint64_t through_tick) noexcept {
        if (through_tick == 0) return false;
        auto& observed = submitted[Index(lane)];
        if (through_tick > observed) observed = through_tick;
        return Reconcile(lane);
    }

    // False means the owner still has a pinned record/seal or unresolved dependency.
    // The caller retries after EndRecord/FinishSeal at a safe point.
    bool Reconcile(CpLane lane) noexcept {
        const auto target = submitted[Index(lane)];
        while (coordination.OpenTick(lane) <= target) {
            const auto open = coordination.OpenTick(lane);
            if (!coordination.PublishSubmitted(lane, open)) return false;
        }
        return true;
    }

    [[nodiscard]] std::uint64_t ObservedTick(CpLane lane) const noexcept {
        return submitted[Index(lane)];
    }

private:
    static constexpr std::size_t Index(CpLane lane) noexcept {
        return lane == CpLane::Graphics ? 0 : 1;
    }
    Coordination& coordination;
    std::array<std::uint64_t, 2> submitted{};
};

} // namespace iss::g4
