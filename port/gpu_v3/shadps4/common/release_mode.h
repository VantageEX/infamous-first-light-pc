// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <string>

// Release mode (2026-10-17): a folder holding shadPS4.exe + shadps4-flags.txt starts by a
// double-click, with no launcher script and no console window.
// - shadps4-flags.txt (next to the EXE): NAME=VALUE lines, '#' comments. Each variable not already
//   in the environment is set before the emulator's static initializers run (several flags are read
//   by namespace-scope initializers), so the file acts like a launcher's environment; a launcher's
//   own variables still win. The working directory becomes the EXE's folder (portable user\).
// - With no arguments the game path comes from shadps4-game.txt next to the EXE; the first launch
//   asks for the game folder (the one holding eboot.bin) and saves it there.
namespace Common::ReleaseMode {

/// True when shadps4-flags.txt was found next to the EXE.
bool Active();
/// Variables taken from the file / lines skipped because the environment already set them.
int FlagsApplied();
int FlagsOverridden();
/// Variables from shadps4-flags-user.txt (read first; never shipped) / files copied from the
/// defaults folder because they were missing (2026-10-19: an update over an old folder keeps them).
int UserFlagsApplied();
int DefaultsCopied();
std::string FlagsFilePath();

/// Hides the console window when this process is its only user (started by a double-click).
void HideOwnConsole();

/// The game's eboot.bin: saved choice, else a folder picker (saved on success). nullopt = cancelled.
std::optional<std::filesystem::path> GamePath();

/// 1.1.1 (GitHub #5): a player-facing box before an out-of-memory assert closes the game - what did
/// not fit and what usually helps (other programs, the page file, free space on C:). Windows only;
/// once per process; blocks the calling thread until closed.
void ShowOutOfMemoryNotice(unsigned long long bytes);

} // namespace Common::ReleaseMode
