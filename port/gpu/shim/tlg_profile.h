// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the profiling session's sensors (docs/profiling.md). With TLG_SESSION_DIR set, a thread
// samples the machine every TLG_SESSION_SAMPLE_MS (default 20 ms) and writes
// <dir>/sensors.csv, one "time_ns,name,value" row per reading: the clock and load of each core,
// the GPU's load, clocks, power and temperature (Linux sysfs: the Steam Deck), the CPU's
// temperature, the battery, and every 100 ms for each thread of the process (by name) the share of
// that time it ran and, on Linux, the share it waited for a free core (schedstat). The overlay's
// numbers (frame rate, VRAM, RAM) come along every 500 ms. time_ns is std::chrono::steady_clock,
// the clock of the renderer v2 timeline (ISS_TIMELINE_LOG). In the profiling build (TLG_PROFILE)
// every reading is also a Tracy plot.

#pragma once

namespace TlgProfile {

/// Starts the sampler once when TLG_SESSION_DIR is set.
void Start();

} // namespace TlgProfile
