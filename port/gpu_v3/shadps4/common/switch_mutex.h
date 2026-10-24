// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <mutex>

#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#endif

#include "common/types.h"

namespace Common {

/// Defined in core/libraries/kernel/sync/mutex.cpp (keeps <windows.h> out of this header).
void FutexWait(std::atomic<u32>* address, u32 value);
void FutexWakeOne(std::atomic<u32>* address);
/// Win32 SRWLOCK on a pointer-sized, zero-initialised slot (SRWLOCK_INIT is all zero bits).
void SrwLockExclusive(void** lock);
bool SrwTryLockExclusive(void** lock);
void SrwUnlockExclusive(void** lock);
/// Makes every other thread of the process execute a serializing instruction before returning
/// (Win32 FlushProcessWriteBuffers). The slow side of an asymmetric Dekker handshake: the fast
/// side then needs only plain stores and loads. Returns false where unsupported.
bool FlushProcessWriteBuffersAll();
/// Futex mode: the owning thread locked again (see SwitchMutex). Logs the first few events.
void ReportMutexReentry(const char* name, void* outer_caller, void* inner_caller);

/// Win32 thread id from the TEB (what GetCurrentThreadId returns), without a call.
inline u32 CurrentThreadIdFast() {
#ifdef _WIN64
    return __readgsdword(0x48);
#else
    return 0;
#endif
}

/**
 * A mutex that is either std::mutex, an inline futex lock (flag value 1: 0 free, 1 held, 2 held
 * with waiters; waiters sleep on WaitOnAddress) or a bare Win32 SRW lock (flag value 2: the lock
 * std::mutex wraps, minus the runtime's _Mtx_lock bookkeeping), chosen once per process.
 *
 * Re-entry: MSVC's std::mutex records its owner thread and lets that thread lock again (count
 * bumped, no SRW acquire), so code that re-enters - e.g. a memory-fault handler taken while this
 * lock is held - works under the default mode but self-deadlocks on a plain futex or SRW lock.
 * The futex mode therefore keeps the same owner/count semantics and reports each re-entry.
 * The futex path costs one atomic operation for an uncontended lock or unlock instead of a call
 * through the runtime into the SRW lock. BasicLockable, so scoped_lock/unique_lock work; not for
 * use with condition variables.
 */
template <const char* FlagName>
class SwitchMutex {
public:
    // Read once at startup into a static data member: Mode() runs two or three times per lock and
    // unlock, and a function-local static would add a guard check to each. Only TextureCache
    // members use this class, and none is locked before main().
    static int Mode() {
        return mode;
    }
    static bool Fast() {
        return Mode() == 1;
    }

    void lock() {
        if (Mode() == 2) {
            SrwLockExclusive(&srw);
            return;
        }
        if (!Fast()) {
            slow.lock();
            return;
        }
        const u32 self = CurrentThreadIdFast();
        u32 expected = 0;
        if (!state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed)) {
            // Only this thread ever stores its own id here, so reading it back proves ownership.
            if (owner.load(std::memory_order_relaxed) == self) {
                ++count;
                ReportMutexReentry(FlagName, outer_caller, __builtin_return_address(0));
                return;
            }
            LockContended();
        }
        owner.store(self, std::memory_order_relaxed);
        outer_caller = __builtin_return_address(0);
        count = 1;
    }

    bool try_lock() {
        if (Mode() == 2) {
            return SrwTryLockExclusive(&srw);
        }
        if (!Fast()) {
            return slow.try_lock();
        }
        const u32 self = CurrentThreadIdFast();
        u32 expected = 0;
        if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
            owner.store(self, std::memory_order_relaxed);
            count = 1;
            return true;
        }
        if (owner.load(std::memory_order_relaxed) == self) {
            ++count;
            return true;
        }
        return false;
    }

    void unlock() {
        if (Mode() == 2) {
            SrwUnlockExclusive(&srw);
            return;
        }
        if (!Fast()) {
            slow.unlock();
            return;
        }
        if (--count != 0) {
            return;
        }
        owner.store(0, std::memory_order_relaxed);
        if (state.exchange(0, std::memory_order_release) == 2) {
            FutexWakeOne(&state);
        }
    }

private:
    void LockContended() {
        for (int i = 0; i < 128; ++i) {
            _mm_pause();
            u32 expected = 0;
            if (state.load(std::memory_order_relaxed) == 0 &&
                state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return;
            }
        }
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            FutexWait(&state, 2);
        }
    }

    static inline const int mode = [] {
#ifdef _WIN32
        const char* value = std::getenv(FlagName);
        return value == nullptr ? 0 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
#else
        return 0;
#endif
    }();

    std::atomic<u32> state{0};
    std::atomic<u32> owner{0}; // Futex mode: owning thread id, 0 when free.
    u32 count = 0;             // Futex mode: lock depth of the owner.
    void* outer_caller = nullptr;
    void* srw = nullptr;
    std::mutex slow;
};

inline constexpr char FastTexMutexFlag[] = "SHADPS4_FAST_TEX_MUTEX";

} // namespace Common
