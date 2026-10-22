/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Which scePadSetVibration calls reach the controller (runtime_pad.c, pad_vibration).
 *
 * The game can set the vibration every frame. Sent as is, that is 30-60 rumble commands a
 * second, enough to take down SteamOS's InputPlumber on a OneXPlayer Apex (issue #505: the whole
 * controller stops for a minute while draining a generator with R2). So a call is sent only when:
 *   - it is the first one, or it stops the motors (a stop is never held back);
 *   - the strength changed and RUMBLE_GATE_MIN_US passed since the last command;
 *   - the strength is the same, the motors run, and RUMBLE_GATE_REFRESH_US passed: the port asks
 *     SDL for 1000 ms of rumble, so a steady rumble is renewed before it runs out.
 * A change that comes sooner than RUMBLE_GATE_MIN_US waits for the game's next call.
 * Kept free of SDL and the runtime so tests/test_rumble_gate.c checks it alone. */
#ifndef ISS_RUMBLE_GATE_H
#define ISS_RUMBLE_GATE_H
#include <stdint.h>

#define RUMBLE_GATE_MIN_US 100000u     /* at most 10 changes a second */
#define RUMBLE_GATE_REFRESH_US 500000u /* half of the 1000 ms asked of SDL */

typedef struct { uint8_t low, high, sent; uint64_t sent_us; } RumbleGate;

/* 1 when (low, high) should be sent to the controller now; the gate then remembers it. */
static inline int rumble_gate(RumbleGate *g, uint8_t low, uint8_t high, uint64_t now_us) {
    const int same=g->sent && g->low==low && g->high==high;
    const uint64_t since=now_us-g->sent_us;
    int send;
    if (!g->sent) send=1;
    else if (same) send=(low || high) && since>=RUMBLE_GATE_REFRESH_US;
    else send=(!low && !high) || since>=RUMBLE_GATE_MIN_US;
    if (send) { g->low=low; g->high=high; g->sent=1; g->sent_us=now_us; }
    return send;
}
#endif
