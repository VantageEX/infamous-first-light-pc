// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace VideoCore {

// Global conservative invalidation for clean-range memos. Only publish a new
// epoch AFTER the dirty summaries and bits have been updated under their lock.
inline std::atomic<std::uint64_t> g_dirty_epoch{1};

// SHADPS4_DRAW_REPEAT: moves after every change of an image's dirty flags or tracked range (texture
// cache; guest fault handlers included). Unchanged = no cached image went stale or was re-tracked.
inline std::atomic<std::uint64_t> g_image_epoch{1};

// Change clock per 2 MB guest granule (2026-10-09, dcb note sec. 8): every CPU-dirty and GPU-write
// marking stores a fresh clock value into each granule it touches, so "nothing in [lo, hi) changed
// since clock T" is a few loads (granule gen <= T). Addresses at or above 1 TB share the last slot.
inline constexpr std::uint32_t GranuleBits = 21;
inline constexpr std::size_t NumGranules = std::size_t{1} << (40 - GranuleBits);
inline std::atomic<std::uint64_t> g_granule_clock{1};
/// Only SHADPS4_STATIC_KEY_CENSUS and SHADPS4_NATIVE_GPU_DIRECT=1 (the ring batches' span
/// validity cache) read the clock (Opus review 2026-10-09 item 6: off = no cost).
inline const bool g_granule_clock_on = [] {
    const char* v = std::getenv("SHADPS4_STATIC_KEY_CENSUS");
    const char* g = std::getenv("SHADPS4_NATIVE_GPU_DIRECT");
    return (v && v[0] == '1') || (g && (g[0] == '1' || g[0] == '3') && g[1] == 0);
}();
inline std::atomic<std::uint64_t> g_granule_gen[NumGranules]{};

inline std::size_t GranuleIndex(std::uint64_t addr) noexcept {
    const std::size_t g = static_cast<std::size_t>(addr >> GranuleBits);
    return g < NumGranules ? g : NumGranules - 1;
}

inline void BumpGranules(std::uint64_t lo, std::uint64_t hi) noexcept {
    if (!g_granule_clock_on || hi <= lo) {
        return;
    }
    const std::uint64_t v = g_granule_clock.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::size_t g0 = GranuleIndex(lo);
    const std::size_t g1 = GranuleIndex(hi - 1);
    for (std::size_t g = g0; g <= g1; ++g) {
        g_granule_gen[g].store(v, std::memory_order_release);
    }
}

/// True when no granule of [lo, hi) was marked after clock value `t`.
inline bool GranulesUnchangedSince(std::uint64_t lo, std::uint64_t hi, std::uint64_t t) noexcept {
    if (hi <= lo) {
        return true;
    }
    const std::size_t g0 = GranuleIndex(lo);
    const std::size_t g1 = GranuleIndex(hi - 1);
    for (std::size_t g = g0; g <= g1; ++g) {
        if (g_granule_gen[g].load(std::memory_order_acquire) > t) {
            return false;
        }
    }
    return true;
}

template <typename Mutation>
void PublishCpuDirty(Mutation&& mutation) noexcept {
    mutation();
    g_dirty_epoch.fetch_add(1, std::memory_order_release);
}

} // namespace VideoCore
