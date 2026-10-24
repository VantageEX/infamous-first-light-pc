// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <SDL3/SDL.h>

#include "imgui/imgui_layer.h"
#include "settings_dialog_imgui.h"

namespace ImGuiEmuSettings {

class SettingsLayer final : public ImGui::Layer {
public:
    SettingsLayer();
    ~SettingsLayer() override;

    void Draw() override;

private:
    void Finish();
    SettingsWindow settingsWindow = SettingsWindow(true);
};

void OpenInGameSettingsDialog();
/// F3: opens the dialog, or closes it when open.
void ToggleInGameSettingsDialog();
/// The window thread frees the mouse while it is open and gives it back afterwards.
bool IsInGameSettingsDialogOpen();

}; // namespace  ImGuiEmuSettings
