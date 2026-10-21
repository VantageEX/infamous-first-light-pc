// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// Global pause guard. While the game is paused (eboot counter at +0x1D1F7FC: sub_27DB10
// increments it, sub_27E210 decrements it; it is also 1 in the start menus) and for
// ISS_PAUSE_GUARD_HOLD_MS (3000) after it, every native path is off and the command processor
// draws everything as the original engine does. Native work executed in the pause hung the GPU
// (TDR), see out/experimental/20261017-native-passes-v2/CORRECAO-HANG-PAUSE.md.
// ISS_PAUSE_GUARD=0 turns it off (diagnostics only: it hangs the GPU).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "gpu_blackbox.h"

namespace iss::pause_guard {

inline constexpr std::uint64_t CounterOffset = 0x1D1F7FC;
inline constexpr std::uint64_t CounterSite = 0x27DB27; // lea r14, [rip+counter]
inline const volatile std::uint32_t* counter{};
inline std::int64_t hold_ms = 3000;
inline std::atomic<std::int64_t> last_paused_ms{};
inline std::atomic<bool> active{};
inline std::atomic<std::uint64_t> engaged{};

inline std::int64_t NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

inline void Install(std::uint64_t image_base) {
    if (const char* v = std::getenv("ISS_PAUSE_GUARD"); v && v[0] == '0') {
        std::fprintf(stderr, "PAUSE_GUARD off (ISS_PAUSE_GUARD=0): native paths run in the pause\n");
        return;
    }
    if (const char* v = std::getenv("ISS_PAUSE_GUARD_HOLD_MS")) hold_ms = std::strtoll(v, nullptr, 10);
    // The counter is only trusted when the code that increments it (sub_27DB10 of the US
    // CUSA00223 1.07 eboot: lea r14,[rip+disp32] at +0x27DB27) points at it. Another eboot
    // (base 1.00, other region) would read garbage, keep the guard engaged forever and turn
    // every native path off, so the guard stays off there instead.
    const auto* site = reinterpret_cast<const unsigned char*>(image_base + CounterSite);
    std::int32_t disp{};
    __builtin_memcpy(&disp, site + 3, 4);
    if (site[0] != 0x4c || site[1] != 0x8d || site[2] != 0x35 ||
        CounterSite + 7 + static_cast<std::int64_t>(disp) != CounterOffset) {
        std::fprintf(stderr, "V2_SITE pause_counter not found (eboot is not US 1.07): pause guard off\n");
        return;
    }
    counter = reinterpret_cast<const volatile std::uint32_t*>(image_base + CounterOffset);
    std::fprintf(stderr, "PAUSE_GUARD on: counter eboot+0x%llx = %u, hold %lld ms\n",
                 (unsigned long long)CounterOffset, *counter, (long long)hold_ms);
}

// True while every native path must stay off. One volatile load when the game is running.
inline bool Blocked() {
    if (!counter) return false;
    if (*counter == 0 && !active.load(std::memory_order_relaxed)) return false;
    const std::int64_t now = NowMs();
    if (*counter != 0) {
        last_paused_ms.store(now, std::memory_order_relaxed);
        if (!active.exchange(true, std::memory_order_relaxed)) {
            engaged.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "PAUSE_GUARD engaged: native paths off\n");
            iss::blackbox::Event(iss::blackbox::General, "PAUSE: entrou (nativo desligado)");
        }
        return true;
    }
    if (now - last_paused_ms.load(std::memory_order_relaxed) < hold_ms) return true;
    if (active.exchange(false, std::memory_order_relaxed)) {
        std::fprintf(stderr, "PAUSE_GUARD released: native paths on\n");
        iss::blackbox::Event(iss::blackbox::General, "PAUSE: saiu ha 3 s (nativo religado)");
    }
    return false;
}

} // namespace iss::pause_guard
