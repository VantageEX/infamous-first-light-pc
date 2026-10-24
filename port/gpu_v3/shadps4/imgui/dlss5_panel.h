// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The DLSS 5 panel (Page Up; video_core/renderer_vulkan/vk_dlss5.h): a small window over the game
// so the picture stays visible while the model is tuned. Changes apply on the next frame and are
// saved next to the model (dlss5-settings.ini).
namespace Dlss5Panel {

/// Page Up: opens the panel, or closes it when open (window thread).
void Toggle();
/// The window thread frees the mouse while it is open, as for the F3 menu.
bool IsOpen();

} // namespace Dlss5Panel
