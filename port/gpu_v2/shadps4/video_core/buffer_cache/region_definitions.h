// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <utility>

#include "common/perf_stats.h"
#include "common/types.h"
#if !defined(__GNUC__) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace VideoCore {

/// Hint: brings the cache line of the address into the caches of the calling thread, so that a
/// read soon after does not wait for memory (lookups whose keys are known early).
inline void PrefetchLine(const void* address) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address);
#elif defined(_M_X64) || defined(_M_IX86)
    _mm_prefetch(static_cast<const char*>(address), _MM_HINT_T0);
#endif
}

/// Upload epochs. A new epoch starts whenever the guest CPU may have written memory the
/// following GPU commands read: a new submission, a wait satisfied by a value the guest wrote
/// (not by the command processor). Hot pages (written by the guest every frame) are not write
/// protected and are uploaded at most once per epoch instead of faulting on every guest write.
inline std::atomic<u64> g_upload_epoch{1};

/// Guest frames (submit done), used to recognize the pages the guest writes every frame.
inline std::atomic<u32> g_tracker_frame{1};

/// Incremented whenever the emulator itself writes guest memory (command processor writes and
/// labels, readback and image write backs): copies of guest memory taken before may be stale.
inline std::atomic<u64> g_host_write_gen{1};

inline u64 CurrentUploadEpoch() noexcept {
    return g_upload_epoch.load(std::memory_order_acquire);
}

void NextUploadEpoch() noexcept;

inline u64 CurrentHostWriteGen() noexcept {
    return g_host_write_gen.load(std::memory_order_acquire);
}

inline void NoteHostWrite() noexcept {
    g_host_write_gen.fetch_add(1, std::memory_order_acq_rel);
    Common::Perf::Add(Common::Perf::HostWrites);
}

constexpr u64 PAGES_PER_WORD = 64;
constexpr u64 BYTES_PER_PAGE = 4_KB;
constexpr u64 BYTES_PER_WORD = PAGES_PER_WORD * BYTES_PER_PAGE;

constexpr u64 HIGHER_PAGE_BITS = 24;
constexpr u64 HIGHER_PAGE_SIZE = 1ULL << HIGHER_PAGE_BITS;
constexpr u64 HIGHER_PAGE_MASK = HIGHER_PAGE_SIZE - 1ULL;

constexpr u64 NUM_REGION_PAGES = HIGHER_PAGE_SIZE / BYTES_PER_PAGE;
constexpr u64 NUM_REGION_WORDS = HIGHER_PAGE_SIZE / BYTES_PER_WORD;

enum class Type : u8 {
    CPU = 1 << 0,
    GPU = 1 << 1,
};

enum class StateOp : u8 {
    None = 0,
    Set = 1,
    Clear = 2,
};

constexpr bool operator&(Type a, Type b) noexcept {
    return std::to_underlying(a) & std::to_underlying(b);
}

constexpr Type operator|(Type a, Type b) noexcept {
    return static_cast<Type>(std::to_underlying(a) | std::to_underlying(b));
}

struct Bounds {
    u64 start_word;
    u64 start_page;
    u64 end_word;
    u64 end_page;
};

struct RegionBits {
    constexpr void Fill(u64 value) {
        data.fill(value);
    }

    constexpr bool GetPage(u64 page) const {
        return data[page / PAGES_PER_WORD] & (1ULL << (page % PAGES_PER_WORD));
    }

    constexpr u64& operator[](u64 index) {
        return data[index];
    }

private:
    alignas(64) std::array<u64, NUM_REGION_WORDS> data;
};

} // namespace VideoCore
