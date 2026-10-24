// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <imgui.h>

#include "imgui/imgui_std.h"
#include "settings_dialog_layer.h"

namespace ImGuiEmuSettings {

static bool running = false;
static std::atomic<bool> g_open{false}; // read by the window thread (mouse release / restore)
static std::optional<SettingsLayer> settingsLayer = std::nullopt;

SettingsLayer::SettingsLayer() {
    AddLayer(this);
    settingsWindow.Prepare();
}

SettingsLayer::~SettingsLayer() {}

void SettingsLayer::Finish() {
    g_open.store(false, std::memory_order_release);
    RemoveLayer(this);
    settingsLayer.reset();
}

void SettingsLayer::Draw() {
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[IMGUI_FONT_SETTINGS_WINDOW]);
    settingsWindow.DrawSettings(&running, nullptr);
    ImGui::PopFont();

    if (!running) {
        Finish();
    }
}

void OpenInGameSettingsDialog() {
    if (settingsLayer.has_value()) {
        return;
    }

    settingsLayer.emplace();
    running = true;
    g_open.store(true, std::memory_order_release);
}

void ToggleInGameSettingsDialog() {
    if (g_open.load(std::memory_order_acquire)) {
        running = false; // the layer finishes on its next draw
    } else {
        OpenInGameSettingsDialog();
    }
}

bool IsInGameSettingsDialogOpen() {
    return g_open.load(std::memory_order_acquire);
}

} // namespace ImGuiEmuSettings
