// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include <string>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_dlss.h"

// DLSS 5 Neural Rendering on top of SHADPS4_DLSS=1 (LATER-PLAN 9; private build, 2026-10-18).
// The same deal as DLSS 4.5: the player's own files are looked for, and anything missing or any
// failure leaves the frame exactly as DLSS 4.5 alone makes it. Looked for, first match wins:
// SHADPS4_DLSS5_DIR, dlss5\ next to the EXE, and ..\..\dlss5 (a test bundle in artifacts\).
// The folder holds the model nvngx_dlssnr.dll (not public yet; one DLL holds the networks the
// panel calls A and B) and the forwarder nvngx.dll_dlssnr.dll built from
// github.com/Skynizz/optiscaler-dlss5 (GPL-3, dlss5\forwarder-src, our fixes in its patch file):
// the model only accepts calls from a module whose path contains "nvngx.dll", and the forwarder
// is that module. SHADPS4_DLSS5=0 never loads them.
//
// Gameplay (Rasterizer::DlssResolve): the game's 1080p tonemapped frame (no HUD yet; red / blue
// swapped in the game's images, put right for the model) -> the model, 1-3 passes (each pass its
// own feature on the previous pass's answer) -> composed back onto the game's frame (light as a
// ratio, colour by a strength, saturation, a split compare) -> DLSS 4.5 -> the HUD on top.
// Video cutscenes (game/art/movies/cutscene, H.264, no depth or motion): frames without the T2x
// resolve while the game's video decoder runs go through the model at the presenter instead,
// spatially (no history), and the post pass shows the result.
// Keys: End = DLSS 5 on / off, Page Up = the DLSS 5 panel (Home stays DLSS 4.5 on / off).
namespace Vulkan {
class Instance;
}

namespace Vulkan::Dlss5 {

/// What the panel edits. The model reads its tuning (style ... mv_mode) once, when a feature is
/// built, so a change there rebuilds it; everything from `video` on applies on the next frame.
struct Settings {
    bool enabled = true;          ///< End
    int style = 0;                ///< 0 standard, 1 natural, 2 cinematic (the model's own profiles)
    float intensity = 1.0f;       ///< overall strength of the model's edit
    float local_structure = 1.0f; ///< fine detail / surface structure
    float local_tone = 1.0f;      ///< local light and shade
    bool skin_follows = true;     ///< skin uses the local-structure value (the model's default)
    float skin_structure = 1.0f;  ///< used when skin_follows is off
    bool auto_mask = true;        ///< the model decides where it may edit
    int preset = 0;              ///< network: 0 = the DLL default, 1 = A (first model), 2 = B (newer)
    bool ui_correction = false;   ///< off: the frame it gets has no HUD in it
    int mv_mode = 0;              ///< motion vectors: 0 pixels (default), 1 normalised, 2 none
    // Not read by the model (live):
    bool video = false;       ///< also on video cutscenes (off by default since 10-18: without
                              ///< depth or motion it mostly acts like colour correction)
    float video_depth = 0.5f; ///< the flat depth a video frame is given (reversed Z, 0 = far)
    bool video_fresh = false; ///< a new history on every video frame (round 2's way; flickered)
    int passes = 1;           ///< 1-3 model passes (layering)
    float transfer = 1.0f;    ///< how much of the model's light change reaches the frame (0-1.5)
    float colour = 1.0f;      ///< how much of the model's colour change reaches it (0-1)
    float saturation = 1.0f;  ///< of the result (0.5-1.5)
    float max_ratio = 8.0f;   ///< the most a pixel's brightness may change (x or 1 / x)
    bool compare = false;     ///< split screen: the game's frame left of `split`
    float split = 0.5f;
};

enum class State : int {
    Off,      ///< SHADPS4_DLSS != 1 or SHADPS4_DLSS5=0
    NotFound, ///< no folder with both DLLs
    Starting, ///< found; waiting for the Vulkan device
    Ready,
    Failed, ///< start-up failed (see message); DLSS 4.5 runs alone
};

struct Status {
    State state = State::Off;
    std::string message;
    std::string folder;
    float gpu_ms_avg = 0.0f; ///< DLSS 5's GPU time per gameplay frame (all passes), last ~2 s
    float gpu_ms_max = 0.0f;
    u64 frames = 0; ///< gameplay frames the model ran on
    u32 width = 0;
    u32 height = 0;
    bool running = false; ///< ran on the last DLSS frame
    int passes_running = 0;
    u64 video_frames = 0;   ///< video cutscene frames the model ran on
    bool video_running = false;
    u32 vram_used_mb = 0;   ///< the GPU's device-local memory in use (whole process), last check
    u32 vram_budget_mb = 0; ///< what the driver allows before it starts paging
    u32 features_alive = 0; ///< model copies alive (each pass of each lane, plus retiring ones)
};

/// SHADPS4_DLSS=1, SHADPS4_DLSS5 not "0" and both DLLs found (decided once).
bool Wanted();
/// Device extensions the model uses (Streamline's DLSS list normally has them already).
const std::vector<const char*>& DeviceExtensions();
/// vk_instance, after the device exists: loads the forwarder and the model, checks the parameter
/// block layout, NVSDK_NGX_VULKAN_Init_Ext. Any failure: State::Failed, DLSS 4.5 alone.
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);
/// The device came up without the needed extensions (or the DLSS ones were refused).
void Disable(const char* why);
/// Instance destruction, GPU idle, before Streamline goes.
void Shutdown();

Status GetStatus();
Settings GetSettings();
/// Any thread (the panel): takes effect on the next frame and is saved to dlss5-settings.ini.
/// A tuning change rebuilds the features; `rebuild` forces that (Defaults after a refusal).
/// `save` = false while a live slider is dragged (written to the file when it is let go).
void SetSettings(const Settings& settings, bool rebuild = false, bool save = true);
Settings DefaultSettings();
/// End key: returns the new on / off state.
bool ToggleEnabled();

/// Rasterizer::DlssResolve: bumped once per evaluated T2x replacement, so the presenter can tell
/// a game-rendered frame from a video frame.
inline std::atomic<u64> g_resolves{0};

/// Rasterizer::DlssResolve, before Dlss::Evaluate, on the same command buffer: the model on
/// `color` (the game's frame, in `color.layout`, left in that layout); returns the image DLSS 4.5
/// should take as its colour input. Nothing = DLSS 5 off or it failed this frame (DLSS 4.5 takes
/// `color` as before).
std::optional<Dlss::EvalImage> Apply(const Instance& instance, vk::CommandBuffer cmdbuf,
                                     const Dlss::EvalImage& color, const Dlss::EvalImage& motion,
                                     const Dlss::EvalImage& depth, bool reset);

/// Presenter, per flip: DLSS 5 is on, video cutscenes are wanted, the game's video decoder ran in
/// the last quarter second and no T2x replacement ran since the previous flip.
bool VideoFrame(u64 resolves_at_last_flip);
/// Presenter, on its command buffer: the model on the display buffer (`frame_view`, shader-read
/// layout, `linear` = an sRGB view). Returns a linear RGBA16F view of the result at the same
/// size for the post pass, or nothing (show the frame as it is).
std::optional<vk::ImageView> ApplyVideo(const Instance& instance, vk::CommandBuffer cmdbuf,
                                        vk::ImageView frame_view, u32 width, u32 height,
                                        bool linear);

} // namespace Vulkan::Dlss5
