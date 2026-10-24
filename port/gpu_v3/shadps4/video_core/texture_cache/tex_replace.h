// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "common/types.h"

namespace VideoCore {

struct ImageInfo;

/// SHADPS4_TEX_REPLACE=1 (2026-10-18, LATER-PLAN 11): user-made pictures copied over parts of a game
/// texture right after the emulator uploads it - e.g. keyboard / mouse icons over the PS4 button glyphs.
/// Only the listed rectangles are replaced; the rest of the texture stays the game's own data.
/// Manifest: <exe dir>/tex_replace/replacements.txt (folder: SHADPS4_TEX_REPLACE_DIR), one line per
/// picture: `<texture key, 16 hex digits> <texture width> <texture height> <x> <y> <png> [noflip]`.
/// The key is XXH3-64 of the texture's guest bytes; x / y = top-left corner in the upright picture.
/// By default the picture is stored upside down in the texture (rows y -> height - y - h), as the
/// game's UI atlas is (inferred from the font table, scratch/glyphs-20261018); `noflip` keeps rows.
namespace TexReplace {

struct Patch {
    u32 x;
    u32 y; ///< first texture row in memory order
    u32 width;
    u32 height;
    std::vector<u8> rgba; ///< rows in memory order, 4 bytes per pixel
};

struct Target {
    u64 key;
    u32 width;
    u32 height;
    std::vector<Patch> patches;
};

/// True when the flag is set and the manifest gave at least one usable picture.
bool Enabled();

/// The target for this image, or nullptr. Cheap unless the image is a single-mip 2D 32-bit
/// texture with the width / height of some target; then hashes its guest bytes.
const Target* Find(const ImageInfo& info);

/// Whether the image's Vulkan format stores blue first (the pictures are RGBA).
bool SwapRedBlue(const ImageInfo& info);

/// Device switch (SHADPS4_TEX_REPLACE_AUTO, on unless set to 0): the pictures show while the
/// keyboard / mouse was used last; a pad button or a stick / trigger past half way brings the game's
/// own glyphs back, a key or mouse click the pictures. A switch invalidates the matched textures'
/// guest memory as a CPU write would (any thread may call it), so the next use re-uploads them.
bool AutoSwitch();
void NoteInputDevice(bool gamepad);
bool PicturesActive();

} // namespace TexReplace
} // namespace VideoCore
