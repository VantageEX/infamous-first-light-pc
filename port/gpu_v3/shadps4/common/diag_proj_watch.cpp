// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SHADPS4_FIND_PROJ=1 (diagnostic, 2026-10-20, ROADMAP 1.1 part C - dynamic aspect ratio): finds
// the game's camera projection matrix and the code that writes it.
//  1. SHADPS4_FIND_PROJ_AT_S seconds after start (default 70), scans committed readable guest
//     memory (0x200000000 - 0x1000000000) for 4x4 float matrices whose [0] and [5] are positive with
//     [5] / [0] within 1% of 16:9 and [1..4] zero (a perspective projection; [5] / [0] = aspect),
//     and logs up to 32 of them.
//  2. Puts a hardware write watch (DR0, 4 bytes) on each hit's [0] in turn, in every thread of the
//     process, for SHADPS4_FIND_PROJ_WATCH_S seconds each (default 2), and logs the distinct code
//     addresses that wrote it (the instruction after the store).
// Not for players: it suspends every thread briefly to set the debug registers.

#include "common/diag_proj_watch.h"

#ifdef _WIN32
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <vector>
#include <windows.h>
#include <tlhelp32.h>
#include "common/logging/log.h"
#include "common/thread.h"

namespace Common {

namespace {
constexpr size_t MaxRips = 64;
std::array<std::atomic<u64>, MaxRips> g_rips{};
std::atomic<u32> g_rip_hits{0};
std::atomic<bool> g_watching{false};

LONG CALLBACK WatchHandler(PEXCEPTION_POINTERS info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP ||
        !g_watching.load(std::memory_order_relaxed)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if ((info->ContextRecord->Dr6 & 1) == 0) {
        return EXCEPTION_CONTINUE_SEARCH; // not our DR0
    }
    g_rip_hits.fetch_add(1, std::memory_order_relaxed);
    const u64 rip = info->ContextRecord->Rip;
    for (auto& slot : g_rips) {
        u64 cur = slot.load(std::memory_order_relaxed);
        if (cur == rip) {
            break;
        }
        if (cur == 0 && slot.compare_exchange_strong(cur, rip, std::memory_order_relaxed)) {
            break;
        }
    }
    info->ContextRecord->Dr6 = 0;
    return EXCEPTION_CONTINUE_EXECUTION;
}

/// DR0 = address (write, 4 bytes) or off, in every other thread of this process.
void SetWatchAllThreads(u64 address, bool on) {
    const DWORD pid = GetCurrentProcessId();
    const DWORD self = GetCurrentThreadId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return;
    }
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    u32 set = 0;
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) {
            continue;
        }
        HANDLE t = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
                              FALSE, te.th32ThreadID);
        if (!t) {
            continue;
        }
        if (SuspendThread(t) != DWORD(-1)) {
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (GetThreadContext(t, &ctx)) {
                ctx.Dr0 = on ? address : 0;
                // L0 enable, R/W0 = 01 (write), LEN0 = 11 (4 bytes).
                ctx.Dr7 = on ? (ctx.Dr7 & ~0xF0003ull) | 0x1ull | (0x1ull << 16) | (0x3ull << 18)
                             : (ctx.Dr7 & ~0xF0003ull);
                ctx.Dr6 = 0;
                if (SetThreadContext(t, &ctx)) {
                    ++set;
                }
            }
            ResumeThread(t);
        }
        CloseHandle(t);
    }
    CloseHandle(snap);
    LOG_INFO(Common, "FindProj: write watch {} at {:#x} in {} threads", on ? "on" : "off", address,
             set);
}

bool LooksLikeProjection(const float* m) {
    if (!(m[0] > 0.2f && m[0] < 8.0f && m[5] > 0.2f && m[5] < 8.0f)) {
        return false;
    }
    for (int i = 1; i <= 4; ++i) {
        if (std::fabs(m[i]) > 1e-6f) {
            return false;
        }
    }
    const float ratio = m[5] / m[0];
    // Rows / columns of a perspective matrix: [10] and [11] / [14] carry depth terms (non-zero).
    return std::fabs(ratio - 16.0f / 9.0f) < 0.0178f && (m[11] != 0.0f || m[14] != 0.0f);
}
} // namespace

void StartProjectionFinder() {
    const char* env = std::getenv("SHADPS4_FIND_PROJ");
    if (!env || env[0] != '1') {
        return;
    }
    const auto env_int = [](const char* name, int def) {
        const char* v = std::getenv(name);
        return v ? std::atoi(v) : def;
    };
    const int at_s = env_int("SHADPS4_FIND_PROJ_AT_S", 70);
    const int watch_s = env_int("SHADPS4_FIND_PROJ_WATCH_S", 2);
    std::thread([at_s, watch_s] {
        SetCurrentThreadName("FindProj");
        std::this_thread::sleep_for(std::chrono::seconds(at_s));
        std::vector<u64> hits;
        u64 scanned = 0;
        for (u64 a = 0x200000000ull; a < 0x1000000000ull && hits.size() < 32;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof(mbi)) == 0) {
                break;
            }
            const u64 base = reinterpret_cast<u64>(mbi.BaseAddress);
            const u64 end = base + mbi.RegionSize;
            const DWORD p = mbi.Protect & 0xff;
            const bool readable = mbi.State == MEM_COMMIT && !(mbi.Protect & PAGE_GUARD) &&
                                  (p == PAGE_READONLY || p == PAGE_READWRITE ||
                                   p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE);
            if (readable) {
                const float* f = reinterpret_cast<const float*>(base);
                const size_t n = (end - base) / 4;
                for (size_t i = 0; i + 16 <= n && hits.size() < 32; i += 4) {
                    if (LooksLikeProjection(f + i)) {
                        hits.push_back(base + i * 4);
                        LOG_INFO(Common,
                                 "FindProj: {:#x} m0 {:.5f} m5 {:.5f} (ratio {:.5f}) m10 {:.5f} "
                                 "m11 {:.5f} m14 {:.5f} m15 {:.5f}",
                                 base + i * 4, f[i], f[i + 5], f[i + 5] / f[i], f[i + 10],
                                 f[i + 11], f[i + 14], f[i + 15]);
                    }
                }
                scanned += end - base;
            }
            a = end;
        }
        LOG_INFO(Common, "FindProj: scanned {} MB, {} candidates", scanned >> 20, hits.size());
        if (hits.empty() || watch_s <= 0) {
            return;
        }
        AddVectoredExceptionHandler(1, WatchHandler);
        for (const u64 target : hits) {
            for (auto& slot : g_rips) {
                slot.store(0, std::memory_order_relaxed);
            }
            g_rip_hits.store(0, std::memory_order_relaxed);
            g_watching.store(true, std::memory_order_relaxed);
            SetWatchAllThreads(target, true);
            std::this_thread::sleep_for(std::chrono::seconds(watch_s));
            SetWatchAllThreads(target, false);
            g_watching.store(false, std::memory_order_relaxed);
            LOG_INFO(Common, "FindProj: {:#x} written {} times", target,
                     g_rip_hits.load(std::memory_order_relaxed));
            for (const auto& slot : g_rips) {
                if (const u64 rip = slot.load(std::memory_order_relaxed)) {
                    LOG_INFO(Common, "FindProj:   writer (next instruction) {:#x}", rip);
                }
            }
        }
    }).detach();
}

} // namespace Common

#else
namespace Common {
void StartProjectionFinder() {}
} // namespace Common
#endif
