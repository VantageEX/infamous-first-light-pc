// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include "common/types.h"

// When the game last decoded a video frame (Second Son's cutscene movies, art/movies/cutscene,
// and the in-world TV feeds): DLSS 5 (vk_dlss5.h) treats a frame without the game's T2x pass
// while this is recent as a video cutscene frame.
namespace Libraries::Videodec {

/// steady_clock milliseconds of the last decode call (any decoder).
inline std::atomic<s64> g_last_decode_ms{0};

inline void NoteDecode() {
    g_last_decode_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count(),
                           std::memory_order_relaxed);
}

} // namespace Libraries::Videodec
