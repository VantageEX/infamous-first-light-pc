// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3's stutter report (TLG_V3_STUTTER=1, on by default; 0 off): every LOG_* site of the
// fork counts its calls, logged or not (one relaxed increment), and each presented frame that took
// over twice the recent average and over 33 ms is written to the game log with what happened in
// it: the write faults and the busiest log sites (pipeline compiles, cache misses, readbacks...).
#pragma once

#include <atomic>
#include <cstdint>

namespace V3::Stutter {

struct Site {
    const char* file;
    int line;
    const char* format;
    std::atomic<uint64_t> hits{0};
    uint64_t last{0}; // the hits at the previous frame (present thread only)
    std::atomic<bool> registered{false};
    void Hit() {
        hits.fetch_add(1, std::memory_order_relaxed);
        if (!registered.load(std::memory_order_relaxed) && !registered.exchange(true)) Register(this);
    }
    static void Register(Site* site);
};

/// Every write fault the renderer took (gpu_host.cpp tlggpu_handle_fault).
extern std::atomic<uint64_t> g_faults;
/// A guest fault's address, for the report's busiest MiB of the frame (tlggpu_handle_fault).
void RecordFault(uintptr_t address);

/// The presenter, once per presented game frame (glue/overrides/vk_presenter.cpp).
void OnPresent();

} // namespace V3::Stutter
