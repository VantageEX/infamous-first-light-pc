// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common {

/// SHADPS4_FIND_PROJ=1 (diagnostic): logs the game's 16:9 projection matrices and the code that
/// writes the first one (diag_proj_watch.cpp). Off by default.
void StartProjectionFinder();

} // namespace Common
