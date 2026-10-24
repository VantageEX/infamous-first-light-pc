// SPDX-License-Identifier: GPL-2.0-or-later
// This port (renderer v3): the fork names some of the game's allocations by address (the
// per-frame heap from 0x248000000, its end-of-frame label, the control block at 0x270000000), as
// its own emulator places them. This port's runtime places the same allocations 0xE00000000
// higher (its user range starts at 0x1000000000, the emulator's at 0x200000000): measured, the
// game's write faults gather at 0x1048000000-0x1059000000 here. GuestAddr() moves an address of
// the fork's layout to this one.
#pragma once

#include "common/types.h"

namespace VideoCore {
constexpr u64 GuestLayoutShift = 0xE00000000ULL;
constexpr u64 GuestAddr(u64 fork_address) {
    return fork_address + GuestLayoutShift;
}
} // namespace VideoCore
