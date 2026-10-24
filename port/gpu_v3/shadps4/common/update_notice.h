// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Common {

/// The fork's release tag this build belongs to; the release checklist bumps it (ROADMAP).
inline constexpr const char* ReleaseTag = "second-son-v1.1.1";

/// 1.1 (LATER-PLAN 17): one background request to GitHub's latest release of axylpup/shadPS4 at
/// every start, shown on the boot screen (GetUpdateLine). Nothing is downloaded.
/// SHADPS4_UPDATE_CHECK=0 turns it off.
void StartUpdateNotice();

/// 1.1 (user 10-21): the boot screen's version line ("Latest version: 1.2 - New update out!"),
/// read by every boot frame while the check runs on its own thread.
struct UpdateLine {
    std::string text;
    std::string detail; ///< second line (where to download), only when newer
    bool newer = false;
};
UpdateLine GetUpdateLine();

} // namespace Common
