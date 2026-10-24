// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include "common/logging/log.h"
#include "common/types.h"

namespace AmdGpu::MtbConflict {

/// SHADPS4_MTB_CONFLICT=1 (diagnostic, default off; notes/parrec-build-2026-10-08.md sec. 14h):
/// the discard / fix-up rate of a two-translator back (the back translating batch j-1 while a
/// worker translates batch j against the state at the end of j-2). Every image backing, buffer
/// and metadata surface carries the batch that last changed its cross-batch state (image layout /
/// access, buffer hazard fields, meta clear bits); the first touch of each object by a draw /
/// dispatch in batch j counts a conflict when batch j-1 changed it (the worker could not have
/// seen it) or a packet / Call earlier in batch j did (an in-batch sequence point the worker
/// passed). Serial: stamps and counts only, behaviour unchanged. Needs SHADPS4_MTB_P1=1 (batch
/// boundaries come from the translating context).
inline const bool g_enabled = [] {
    const char* env = std::getenv("SHADPS4_MTB_CONFLICT");
    return env && env[0] == '1';
}();

enum Class : u32 { Image = 0, Buffer = 1, Meta = 2, Count = 3 };

/// The batch being translated (the draw scheduler's open tick). Back thread.
inline u64 g_batch = 1;
/// A split draw / dispatch record is being translated (else a packet, Call or command).
inline bool g_in_draw = false;

struct Stamp {
    u64 change{};       ///< batch of the last change
    u64 touch{};        ///< batch of the last draw touch
    bool change_pkt{};  ///< that change came from a packet / Call / command, not a draw
};

struct Window {
    u64 batches{}, any{};
    std::array<u64, Count> touches{}, conf_prev{}, conf_pkt{}, batches_prev{}, batches_pkt{};
    std::array<u64, Count> conf_mid{}, batches_mid{}; ///< v2: packet change after a draw touch
    std::array<u64, Count> pkt_changes{};             ///< v2 sanity: every packet-made change
};
inline Window g_win{};
inline u32 g_batch_flags = 0;

/// Before a draw's access to the object.
inline void Touch(Stamp& s, Class c) {
    if (!g_in_draw || s.touch == g_batch) {
        return;
    }
    s.touch = g_batch;
    ++g_win.touches[c];
    if (s.change != 0 && s.change + 1 == g_batch) {
        ++g_win.conf_prev[c];
        g_batch_flags |= 1u << c;
    } else if (s.change == g_batch && s.change_pkt) {
        ++g_win.conf_pkt[c];
        g_batch_flags |= 1u << (c + 8);
    }
}

/// After the object's cross-batch state changed.
inline void Change(Stamp& s, Class c) {
    // v2: a packet / Call changes an object a draw of this batch already touched - a worker that
    // tracks it locally would need a mid-batch join.
    if (!g_in_draw) {
        ++g_win.pkt_changes[c];
        if (s.touch == g_batch) {
            ++g_win.conf_mid[c];
            g_batch_flags |= 1u << (c + 16);
        }
    }
    s.change = g_batch;
    s.change_pkt = !g_in_draw;
}

/// At a submission: the batch that ended, and the one that starts.
inline void EndBatch(u64 next_batch) {
    auto& w = g_win;
    ++w.batches;
    w.any += g_batch_flags != 0;
    for (u32 c = 0; c < Count; ++c) {
        w.batches_prev[c] += (g_batch_flags >> c) & 1;
        w.batches_pkt[c] += (g_batch_flags >> (c + 8)) & 1;
        w.batches_mid[c] += (g_batch_flags >> (c + 16)) & 1;
    }
    g_batch_flags = 0;
    g_batch = next_batch;
    if (w.batches < 16384) {
        return;
    }
    const double b = double(w.batches);
    LOG_INFO(Render, "MtbConflict: batches={} any={:.1f}% | batches with a conflict from batch j-1 "
                     "image/buffer/meta={:.1f}/{:.1f}/{:.1f}% from an in-batch packet "
                     "{:.1f}/{:.1f}/{:.1f}% | first touches per batch {:.1f}/{:.1f}/{:.1f} | "
                     "conflicting objects per batch prev {:.2f}/{:.2f}/{:.2f} pkt {:.2f}/{:.2f}/{:.2f}",
             w.batches, 100.0 * w.any / b, 100.0 * w.batches_prev[0] / b,
             100.0 * w.batches_prev[1] / b, 100.0 * w.batches_prev[2] / b,
             100.0 * w.batches_pkt[0] / b, 100.0 * w.batches_pkt[1] / b,
             100.0 * w.batches_pkt[2] / b, w.touches[0] / b, w.touches[1] / b, w.touches[2] / b,
             w.conf_prev[0] / b, w.conf_prev[1] / b, w.conf_prev[2] / b, w.conf_pkt[0] / b,
             w.conf_pkt[1] / b, w.conf_pkt[2] / b);
    LOG_INFO(Render, "MtbConflict v2: batches with a packet change after a draw touch "
                     "image/buffer/meta={:.1f}/{:.1f}/{:.1f}% objects per batch {:.2f}/{:.2f}/{:.2f} | "
                     "all packet-made changes per batch {:.2f}/{:.2f}/{:.2f}",
             100.0 * w.batches_mid[0] / b, 100.0 * w.batches_mid[1] / b,
             100.0 * w.batches_mid[2] / b, w.conf_mid[0] / b, w.conf_mid[1] / b,
             w.conf_mid[2] / b, w.pkt_changes[0] / b, w.pkt_changes[1] / b, w.pkt_changes[2] / b);
    w = Window{};
}

} // namespace AmdGpu::MtbConflict
