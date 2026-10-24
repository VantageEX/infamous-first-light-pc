// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common {

/// v1.0.2 (PERF-PLAN 14.5): writes the current thread's call stack to stderr as "module+offset"
/// lines (up to 24 frames, `skip` frames above the caller left out), so a player's fatal.txt says
/// where an assert or fault came from; llvm-symbolizer resolves the offsets with the release PDB.
/// Windows only (a no-op elsewhere).
void PrintStackToStderr(unsigned skip);

} // namespace Common
