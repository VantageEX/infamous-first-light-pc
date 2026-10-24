// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// SHADPS4_MOUSE_AIM=1 (inFAMOUS Second Son, CUSA00004; default off): true 1:1 mouse aim.
// The gameplay camera update (eboot 0x14ddd0) turns the right stick into pan / tilt speeds and
// stores the integrated angles at camera+0x330 (one store, eboot 0x14ebce; the next frame starts
// from them: FUN_80014d6f0). A code cave at that store adds the mouse's yaw / pitch, so mouse
// motion bypasses the stick curve, sensitivity, acceleration and aim assist while a pad's stick
// keeps the game's feel. notes/study-input-kbm-2026-10-14.md sec. 6 (design A).
// Mouse aim is on while the mouse is captured with F7 (mouse-to-joystick mode).
// SHADPS4_MOUSE_AIM_SENS = degrees per 100 px (default 5.12); SHADPS4_MOUSE_AIM_FLIP = x, y or xy.
namespace Core::MouseAim {

bool Enabled();

/// Patches the eboot (base address) once all modules are loaded, before the game's entry.
void InstallCameraHook(VAddr eboot_base, u64 eboot_size);

/// The hook is in place (the game's code matched).
bool Installed();

/// Mouse timer (33 ms): notices when the camera stops / resumes updating; parked motion is
/// cleared while it does not run (menus, cut-scenes).
void Tick();

/// Main thread: relative mouse motion in pixels while mouse aim is on.
void OnMouseMotion(float xrel, float yrel);

/// - / = keys: multiplies the sensitivity; returns the new degrees per 100 px.
float ScaleSensitivity(float factor);

} // namespace Core::MouseAim
