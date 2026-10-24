// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <condition_variable>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;
class MasterSemaphore;

/// A thread's reserved tick on one MasterSemaphore (see MasterSemaphore::Reserve).
struct TickReservation {
    const MasterSemaphore* owner{};
    u64 tick{};
};

class MasterSemaphore {
public:
    explicit MasterSemaphore(const Instance& instance_);
    ~MasterSemaphore();

    /// SHADPS4_MTB_CENSUS=1 (diagnostic): CurrentTick() reads per thread (multi-threaded back
    /// P1d: tick stamps taken while a draw is translated, notes/parrec-build-2026-10-08.md 6a).
    static inline const bool count_tick_reads = [] {
        const char* env = std::getenv("SHADPS4_MTB_CENSUS");
        return env && env[0] == '1';
    }();
    static inline thread_local u64 t_tick_reads = 0;

    /// Multi-threaded back P1d (SHADPS4_MTB_P1, notes/parrec-build-2026-10-08.md sec. 14): a
    /// translating thread stamps lifetimes with the tick of the batch it translates, which the
    /// committer reserves for it; CurrentTick() of the owning semaphore returns it on that thread.
    /// Serial P1d: always equal to the real current tick (checked at every submission).
    /// Set once at startup when a mode that reserves ticks is on; until then one branch.
    static inline bool reservations_enabled = false;
    static inline thread_local TickReservation t_reservation{};
    void Reserve(u64 tick) noexcept {
        t_reservation = TickReservation{this, tick};
    }
    /// The calling thread's reservation on this semaphore (0: none).
    [[nodiscard]] u64 ReservedTick() const noexcept {
        return t_reservation.owner == this ? t_reservation.tick : 0;
    }

    [[nodiscard]] u64 CurrentTick() const noexcept {
        if (count_tick_reads) [[unlikely]] {
            ++t_tick_reads;
        }
        if (reservations_enabled) [[unlikely]] {
            const TickReservation& r = t_reservation;
            if (r.owner == this) {
                return r.tick;
            }
        }
        return current_tick.load(std::memory_order_acquire);
    }
    /// CurrentTick without the census count.
    [[nodiscard]] u64 CurrentTickRaw() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Polls the counter for up to spin_ns before falling back to Wait.
    void SpinWait(u64 tick, u64 spin_ns);

    /// Reads the semaphore counter without touching the known tick.
    [[nodiscard]] u64 QueryCounter() const;

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
};

} // namespace Vulkan
