// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <vector>

#include "common/types.h"
#include "imgui/imgui_texture.h"

namespace vk {
class CommandBuffer;
}

namespace ImGui::Core::TextureManager {

struct Inner;

void StartWorker();

void StopWorker();

void DecodePngTexture(std::vector<u8> data, Inner* core);

void DecodePngFile(std::filesystem::path path, Inner* core);

void Submit();

/// True when an upload is queued (may be briefly stale): lets a caller skip taking the queue lock.
bool HasPendingUploads();

/// SHADPS4_SWAPCHAIN_SAFE_WAIT=1: the presenter calls this after submitting each frame; texture
/// destroys wait for two of these instead of two scheduler submits.
void NotePresented();

}; // namespace ImGui::Core::TextureManager