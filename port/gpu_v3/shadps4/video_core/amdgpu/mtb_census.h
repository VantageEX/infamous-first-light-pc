// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <vector>
#include "common/fault_monitor.h"
#include "common/types.h"

namespace AmdGpu::MtbCensus {

/// SHADPS4_MTB_CENSUS=1 (diagnostic, default off; notes/multithreaded-back-design-2026-10-08.md P0):
/// cuts the back's record stream into virtual segments (at sequence-point records and every
/// SHADPS4_MTB_SEGMENT draws/dispatches, default 128) and reports how much of the back's time is
/// in draws a lookup-only worker could translate (before the segment's first serial-only event).
inline const bool g_enabled = [] {
    const char* env = std::getenv("SHADPS4_MTB_CENSUS");
    return env && env[0] == '1';
}();

enum Reason : u32 {
    None = 0,
    BufferCreate,
    BufferDelete,
    BufferUpload,
    ImageRegister,
    ImageUnregister,
    ImageUpload,
    StreamAlloc,
    OcclusionSlot,
    ImageInvalidate,
    Count,
};

/// The serial-only event seen during the current draw / dispatch (first one wins). CP thread.
inline u32 g_draw_event = None;

/// Hooks: called where the serial path does something a lookup-only worker may not.
inline void Note(Reason reason) {
    if (g_enabled && g_draw_event == None && ::Common::FaultMonitor::t_is_cp_thread) {
        g_draw_event = reason;
    }
}

/// Census v3 (Fable review 2026-10-08 Q4.1): the address ranges a serial-only event of an EARLIER draw of the open
/// segment touched (uploaded buffer range, uploaded / registered / GPU-invalidated image), and whether the current
/// draw uses one of them (it would have to wait for that hole's commit). CP thread.
struct Range {
    u64 lo;
    u64 hi;
};
inline std::vector<Range> g_seg_events;   ///< events of earlier draws of the open segment
inline std::vector<Range> g_draw_events;  ///< events of the current draw (merged at its end)
inline bool g_draw_dependent = false;
inline u64 g_invalidate_hits = 0;

inline void NoteEventRange(Reason reason, u64 addr, u64 size) {
    if (!g_enabled || !::Common::FaultMonitor::t_is_cp_thread) {
        return;
    }
    Note(reason);
    g_draw_events.push_back({addr, addr + size});
}

inline void NoteUse(u64 addr, u64 size) {
    if (!g_enabled || g_seg_events.empty() || g_draw_dependent ||
        !::Common::FaultMonitor::t_is_cp_thread) {
        return;
    }
    const u64 hi = addr + size;
    for (const Range& r : g_seg_events) {
        if (addr < r.hi && r.lo < hi) {
            g_draw_dependent = true;
            return;
        }
    }
}

} // namespace AmdGpu::MtbCensus
