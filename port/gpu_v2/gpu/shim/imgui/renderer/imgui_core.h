// Adapted from bloodborne_pc@4d4d20812c0af23a11c8b8d40f5e06a9054925ef, GPL-2.0-or-later.
#pragma once
#include <imgui.h>
#include "video_core/renderer_vulkan/vk_common.h"
namespace Frontend { class WindowSDL; }
namespace Vulkan { class Instance; }
namespace ImGui::Core {
void Initialize(const ::Vulkan::Instance&, const Frontend::WindowSDL&, u32, vk::Format);
void OnSurfaceFormatChange(vk::Format);
void Shutdown(vk::Device);
ImGuiID NewFrame(bool reused=false);
void Render(const vk::CommandBuffer&,const vk::ImageView&,const vk::Extent2D&);
bool MustKeepDrawing(); // gpu/imgui_display.cpp: true while the save dialog is up
} // namespace ImGui::Core
