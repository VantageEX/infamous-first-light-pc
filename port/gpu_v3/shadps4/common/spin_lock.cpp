// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/spin_lock.h"

#if _MSC_VER
#include <intrin.h>
#if _M_AMD64
#define __x86_64__ 1
#endif
#if _M_ARM64
#define __aarch64__ 1
#endif
#else
#if __x86_64__
#include <xmmintrin.h>
#endif
#endif

namespace {

void ThreadPause() {
#if __x86_64__
    _mm_pause();
#elif __aarch64__ && _MSC_VER
    __yield();
#elif __aarch64__
    asm("yield");
#endif
}

} // Anonymous namespace

#include <cstdlib>

namespace {

/// SHADPS4_SPINLOCK_PARK=N (default off; "1" = 512): a waiter spins N pauses, then parks on the
/// lock word (std::atomic::wait = WaitOnAddress) until the holder's unlock wakes it. Measured
/// 2026-10-15 (4c/8t, s1166): guest job threads spent 25% of their wall time in lock() on region
/// locks; a preempted holder left the waiters spinning whole quanta on the cores it needed.
/// Read on first use and cached in a plain atomic (review 10-15: a function-local static's guard
/// on ~100k unlocks per frame is not free). Every thread computes the same value, so the race to
/// fill it is harmless, and no lock can be taken in one mode and released in the other.
std::atomic<unsigned> g_park_spins{~0u};

unsigned ReadParkSpins() {
    const char* value = std::getenv("SHADPS4_SPINLOCK_PARK");
    unsigned spins = 0;
    if (value != nullptr && value[0] != '0' && value[0] != '\0') {
        const unsigned long n = std::strtoul(value, nullptr, 10);
        spins = n <= 1 ? 512u : static_cast<unsigned>(n > 1u << 20 ? 1u << 20 : n);
    }
    g_park_spins.store(spins, std::memory_order_relaxed);
    return spins;
}

inline unsigned ParkSpins() {
    const unsigned spins = g_park_spins.load(std::memory_order_relaxed);
    return spins != ~0u ? spins : ReadParkSpins();
}

} // Anonymous namespace

namespace Common {

void SpinLock::lock() {
    unsigned expected = 0;
    if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                      std::memory_order_relaxed)) [[likely]] {
        return;
    }
    if (const unsigned spins = ParkSpins(); spins != 0) {
        LockPark(spins);
        return;
    }
    while (state.exchange(1, std::memory_order_acquire) != 0) {
        ThreadPause();
    }
}

void SpinLock::LockPark(unsigned spins) {
    for (unsigned i = 0; i < spins; ++i) {
        ThreadPause();
        unsigned expected = 0;
        if (state.load(std::memory_order_relaxed) == 0 &&
            state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
            return;
        }
    }
    // Drepper, "Futexes Are Tricky", mutex 3: whoever takes the lock from here marks it 2, so the
    // unlock that follows wakes one parked waiter (a spurious wake only re-checks).
    while (state.exchange(2, std::memory_order_acquire) != 0) {
        state.wait(2, std::memory_order_relaxed);
    }
}

void SpinLock::unlock() {
    if (ParkSpins() != 0) {
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
        return;
    }
    state.store(0, std::memory_order_release);
}

bool SpinLock::try_lock() {
    unsigned expected = 0;
    return state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                         std::memory_order_relaxed);
}

} // namespace Common
