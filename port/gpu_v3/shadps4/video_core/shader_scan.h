// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace VideoCore {

/// Where one shader's code sits inside the game's own files (2026-10-16,
/// notes/study-shader-stutter-2026-10-16.md sec. 4 step 1). Offsets only: no code is copied.
struct ShaderLocation {
    u32 file{};        ///< index into ShaderIndex::files
    u64 offset{};      ///< file offset of the first code dword (the 0xBEEB03FF token)
    u32 code_length{}; ///< bytes (BinaryInfo::length)
    u32 type{};        ///< BinaryInfo::type: 0 pixel, 1 vertex, 4 compute
    u32 crc32{};       ///< BinaryInfo::crc32 (a seed stamp checks it, with code_length)
};

struct ShaderIndex {
    std::vector<std::filesystem::path> files;
    std::unordered_map<u64, ShaderLocation> shaders; ///< by BinaryInfo::shader_hash
};

/// Finds every shader binary in the game folder: an "OrbShdr" BinaryInfo record whose code start
/// is the nearest 0xBEEB03FF token before it with (code[1] + 1) * 8 == record - code (the rule
/// of SearchBinaryInfo). Reads every file once on several threads; the result is kept in
/// `index_file` with the folder's file list (relative path + size per file) and read back while
/// the list is the same. Write times are not compared (v3, 2026-10-16), so an index shipped with a
/// seed fits any copy of the same game version; ReadShaderCode checks every shader's own record.
/// v1.0.2: when only some files differ (a player's dump with extra or other files), the shaders of
/// the unchanged files are kept and only the new / changed files are read. `progress(done, total)`
/// (bytes) is called on the calling thread about every 100 ms while files are read.
ShaderIndex BuildShaderIndex(const std::filesystem::path& game_folder,
                             const std::filesystem::path& index_file,
                             const std::function<void(u64, u64)>& progress = {});

/// HeapCompact of the process heap (Windows; no-op elsewhere): coalesces the free blocks a burst of
/// boot threads left behind (2026-10-16: after the 16-thread rebuild pre-pass, freeing the A2
/// recorder's SubBatch vectors took 60% of VkRecorder's time and the rooftop ran at 31-45 FPS).
/// Returns the largest free block in bytes (0 on failure or elsewhere).
size_t CompactProcessHeap();

/// xxh3 of the running emulator EXE file (0 if unreadable): with the scm description, the seed
/// stamp's build id (uncommitted builds share one description).
u64 ExecutableHash();

/// The shader's code read from its game file; empty on any read error, or when `expected_hash` is
/// set and the OrbShdr record after the code does not carry that hash with the location's crc32 and
/// code length (the index may come from another machine's copy of the game).
std::vector<u32> ReadShaderCode(const ShaderIndex& index, const ShaderLocation& location,
                                u64 expected_hash = 0);

} // namespace VideoCore
