// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#include "common/types.h"

// SHADPS4_WATER_REFLECTION_SCALE=2|4 (game-specific, inFAMOUS Second Son; 2026-10-14): render the
// planar water reflection (480x270 RGBA16F colour + its own 480x270 D32 depth) at 2x or 4x per
// axis - supersampling against its crawling edges (notes/water-capture-analysis-2026-09-25.md).
// Keyed on image descriptions, not addresses (the reflection's memory also holds a 960x540 image
// later in the frame): the depth is the frame's only 480x270 D32 target; the colour is a 480x270
// RGBA16F target the first time it is bound TOGETHER with that depth (five other 480x270
// RGBA16F images of the bloom / particle chains are never bound with it and stay as they are).
// Such an image keeps its guest ImageInfo (cache lookups, overlaps, tracking unchanged) and gets
// a second Vulkan image (a backing, like the MSAA sample backings) at scale x the extent; its
// render area, viewports and scissors are scaled in the pass (RenderState::scale ->
// DynamicState::render_scale at commit). Every shader of the chain samples with normalized
// coordinates (no texel fetches, size queries or FragCoord), so the game's fog, blur offsets
// (constant-buffer uvs) and the water lookup keep their screen-space meaning. The blur's
// intermediate (a shared 480x270 scratch image) stays 1x in v1.
namespace VideoCore::WaterScale {

inline const u32 g_scale = [] {
    const char* v = std::getenv("SHADPS4_WATER_REFLECTION_SCALE");
    if (!v) {
        return 1u;
    }
    const u32 s = static_cast<u32>(std::atoi(v));
    // 2..8: 8 = 3840x2160, twice the 1080p screen per axis, so the water's bilinear lookup
    // averages 2x2 reflection samples per screen pixel (supersampling at screen resolution).
    return (s >= 2 && s <= 8) ? s : 1u;
}();

/// SHADPS4_WATER_REFLECTION_NO_BLUR=1: the reflection's own two blur passes (the game's blur
/// shader rendering right after a reflection pass, or into the reflection itself) draw nothing,
/// so the water samples the unblurred (fogged) reflection. The bloom chain's uses of the same
/// shader are untouched. Works at any scale (1 = the game's size).
inline const bool g_no_blur = [] {
    const char* v = std::getenv("SHADPS4_WATER_REFLECTION_NO_BLUR");
    return v && v[0] == '1';
}();
/// The reflection-identification logic runs when either feature is on.
inline const bool g_active = g_scale > 1 || g_no_blur;

inline constexpr u32 kWidth = 480;
inline constexpr u32 kHeight = 270;
/// The game's separable blur of the reflection (09-25 capture: 36982 -> scratch 25763 -> 36982).
/// v2: its scratch target (a 480x270 RGBA16F image the bloom chain shares) gets the scaled backing
/// while this shader renders into it, and the 1x backing back in any other pass, so the blur
/// keeps the reflection's resolution instead of squeezing it through 480x270.
inline constexpr u64 kBlurFs = 0x6d29fb0f;

/// Guest addresses of the images given a scaled backing (colour and depth), published by the
/// command-processor back thread at the switch; A2 workers read them (relaxed) to leave draws
/// into those targets to the back, whose BeginRendering scales the pass.
inline std::array<std::atomic<u64>, 8> g_addrs{};

inline void Publish(u64 addr) {
    for (auto& a : g_addrs) {
        u64 cur = a.load(std::memory_order_relaxed);
        if (cur == addr) {
            return;
        }
        if (cur == 0 && a.compare_exchange_strong(cur, addr, std::memory_order_relaxed)) {
            return;
        }
        if (cur == addr) {
            return;
        }
    }
}

inline bool IsPublished(u64 addr) {
    if (addr == 0) {
        return false;
    }
    for (const auto& a : g_addrs) {
        const u64 cur = a.load(std::memory_order_relaxed);
        if (cur == addr) {
            return true;
        }
        if (cur == 0) {
            return false;
        }
    }
    return false;
}

/// Counters (relaxed; logged on every switch and every 2^14 scaled passes).
struct Counters {
    std::atomic<u64> depth_switches{};
    std::atomic<u64> colour_switches{};
    std::atomic<u64> scaled_passes{};
    std::atomic<u64> mixed_passes{};     ///< attachments with different scales (kept at 1x)
    std::atomic<u64> worker_refusals{};  ///< A2 draws left to the back for a scaled target
    std::atomic<u64> downloads{};        ///< downloads of a scaled image (expected 0)
    std::atomic<u64> copies{};           ///< image copies from / into a scaled image
    std::atomic<u64> clip_disabled{};    ///< scaled viewports wider than the device limit
    std::atomic<u64> scratch_switches{}; ///< v2: blur scratch backing switches (2 per frame)
    std::atomic<u64> skipped_blurs{};    ///< NO_BLUR: reflection blur passes drawn as nothing
};
inline Counters g_counters{};

} // namespace VideoCore::WaterScale
