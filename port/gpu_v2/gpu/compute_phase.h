// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <atomic>
#include <cstdint>

// Compute phases (ISS_COMPUTE_PHASES=1).
//
// Experimental hypothesis, NOT a proven independence rule. Session 07 mapping found a
// producer/consumer pair inside 0x3F49E0: shader E902349D writes element+0xC and 03E4B72E
// reads that field, with no explicit 0x1694E0 call between them. Some particle phases contain
// multiple dispatches per object. Register writes/NOP alone do not prove independence.
// Keep this opt-in experiment disabled in normal use until resource-specific hazards and
// queue ownership are proven. See evidence/mapping/SESSAO-07.md.
// Observed baseline: about 1350 barriers/frame for about 1430 dispatches; broad shared
// writable descriptor ranges can cause conservative barriers, but that does not make every
// barrier redundant.
//
// A run: consecutive compute queue dispatches, separated only by register writes (SET_SH_REG,
// NOP), starting at a barrier flush made for one of them. Inside a run, a dispatch whose only
// hazards are with the earlier dispatches of the run needs no barrier. Any other packet of the
// compute queue (release, wait, acquire, DMA, write data, indirect dispatch...) ends the run;
// any buffer access recorded by something else (uploads, copies, draws) or any pending image
// barrier makes the next hazard flush as before. The command processor thread only.
namespace iss::phase {

inline std::atomic<bool> g_enabled{false};
/// A compute queue packet other than register writes since the last dispatch of the run.
inline std::atomic<bool> g_boundary{true};
/// Buffer accesses are being recorded for the bindings of a dispatch of the open run.
inline std::atomic<bool> g_run_access{false};
/// Since the last barrier flush of the owner runtime, an access not made by the open run.
inline std::atomic<bool> g_foreign{true};
/// The run is open: it started with a flush of the owner runtime.
inline std::atomic<bool> g_open{false};
/// The runtime of the command processor's graphics scheduler (the only one tracked).
inline std::atomic<const void*> g_owner{nullptr};
inline std::atomic<std::uint64_t> g_skipped{0};
inline std::atomic<std::uint64_t> g_flushed{0};

inline bool Enabled() noexcept {
    return g_enabled.load(std::memory_order_relaxed);
}

} // namespace iss::phase
