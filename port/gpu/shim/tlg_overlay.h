// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: in-game settings menu (Dear ImGui), drawn by the presenter into the swapchain image
// after the game frame, at display resolution. Insert (keyboard) or R3+L2 (gamepad) opens it;
// while it is open the game gets no pad/keyboard input. Settings live in tlg_settings.h.

#pragma once

#include <string>
#include <vector>
#include "common/types.h"
#include "tlg_png.h"
#include "video_core/renderer_vulkan/vk_common.h"

union SDL_Event;
struct SDL_Window;

namespace Vulkan {
class Instance;
}

namespace TlgOverlay {

/// Present thread, once: the ImGui context and its Vulkan backend.
void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count);

/// Window thread, for every SDL event: true when the menu consumed it.
bool HandleEvent(const SDL_Event& event);
/// Turns SDL text input on while the menu edits a value (window thread, once per poll).
void UpdateTextInput(SDL_Window* window);

/// Start-up loading screen (pipeline cache warm-up): `total` 0 shows no count.
void SetLoading(bool active, u32 done, u32 total);
/// The loading screen says the shaders are being compiled (first-run compilation), with the time left.
void SetLoadingCompile(bool compile);
/// Background cache warm-up: a small corner note while the game runs; `total` 0 hides it.
void SetBackgroundLoading(u32 done, u32 total);

/// Whether anything is drawn this frame (menu open or FPS counter on).
bool Visible();

/// Present thread: draws into `view` (layout ColorAttachmentOptimal).
void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent);

/// Present thread, before Render: the game's frame of this present, sampled as `view` in
/// `layout`, drawn at `where` on the swapchain image. The menus blur it behind their glass (the
/// launcher's frosted panels); a present without it draws them on a plain dark glass.
void SetBackdrop(vk::ImageView view, vk::ImageLayout layout, vk::Rect2D where);
/// Before the presenter destroys a frame's view: drops the overlay's descriptor for it.
void ForgetBackdrop(vk::ImageView view);

/// A short message at the top of the screen for `seconds` (any thread), e.g. an input mode change.
void Notice(const std::string& text, float seconds);

/// The menu is open or a text entry is shown: the game's input is held neutral.
bool CapturesInput();

/// Window thread: the game's text dialog (sceImeDialog). Shown as a box on screen until
/// `active` is false; typing goes to the window (sdl_window), the box only displays it.
void SetTextEntry(bool active, const std::string& prompt, const std::string& text);

/// The game's save dialogs (sceSaveDataDialog). kind: 1 save, 2 load, 3 delete (0: a save made
/// without a dialog). Rows carry the game's text for each save; the dates, sizes and every other
/// word are the port's, in the menu language (pt_BR / en_US).
struct SaveRow {
    std::string title, subtitle, detail;
    s64 mtime = 0;  // 0: unknown
    u64 bytes = 0;  // 0: unknown
    bool is_new = false;
};
/// The list: arrows/D-pad or the mouse choose, Enter/Cross or a click confirm, Esc/Circle or a
/// right click cancel. `pictures`: one per row (empty = none). PollSaveList: -1 while open,
/// -2 cancelled, else the chosen index.
void BeginSaveList(int kind, std::vector<SaveRow> rows, std::vector<TlgPng::Image> pictures, int focus);
int PollSaveList();
/// A message the player answers (see tlggpu_save_message_begin). PollSaveMessage: -1 while
/// open, 1 yes/OK, 2 no/cancel.
void BeginSaveMessage(int kind, int code, u64 value, const std::string& text, int buttons,
                      const SaveRow* entry, TlgPng::Image picture);
int PollSaveMessage();
/// The game closed its dialog first: the list or message goes away unanswered.
void CloseSaveDialog();
/// "Saving..." panel: active 1 on (percent < 0: no bar; text null keeps the last), 0 off,
/// 2 on for a moment (dialogs the port answers at once).
void SaveProgress(int kind, int active, int percent, const char* text);
/// The graffiti guide (sensor mode without a gyro) in the middle of the screen, closed with a
/// button; `force` false shows it only the first time (remembered in tlg.ini).
void ShowGraffitiGuide(bool force);
/// Touchpad mode of the mouse (src/runtime_pad.c): a small touchpad with the virtual finger in
/// the bottom-left corner while `active` (x, y: 0..1 over the pad). Any thread.
void SetMousePad(bool active, float x, float y, bool touching, bool clicking);
/// "Game saved" in the bottom-right corner.
void SaveToast(int kind);

} // namespace TlgOverlay
