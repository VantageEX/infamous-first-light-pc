// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore {

/**
 * The byte-exact set of GPU-modified guest ranges (BufferCache::gpu_modified_ranges).
 *
 * SHADPS4_GPU_RANGES_FLAT, read once:
 *   0 / unset: the boost::icl interval set (RangeSet), as before.
 *   1: intervals kept per 4 MiB region (the memory tracker's region size) in small sorted
 *      vectors, reached through a direct-indexed table. Measured 2026-09-30 (a38, back-thread CPU
 *      samples): the icl set was 6.0% of the back in fire and 3.1% in forest, almost all in Add
 *      (one per GPU-written binding: tree searches over ~3,800 nodes, node joins through a
 *      mutex-guarded pool allocator).
 *   2: both, compared: every ForEachInRange result must be identical (first mismatches are
 *      logged as Critical), and Contains answers are compared.
 *
 * Same set semantics as icl's right-open joining interval_set: Add is a union and touching
 * intervals join, Subtract is a difference, ForEachInRange visits every interval that overlaps
 * the query, clipped to it, in address order. An interval that crosses a region boundary is
 * stored as one piece per region; ForEachInRange joins pieces that touch, so callers never see
 * the split. Not thread-safe, like the set it replaces.
 */
class GpuRangeSet {
public:
    static constexpr u64 RegionBits = 22;
    static constexpr u64 RegionSize = 1ULL << RegionBits;
    static constexpr u64 AddressBits = 40;
    static constexpr u64 NumRegions = 1ULL << (AddressBits - RegionBits);

    GpuRangeSet() {
        const char* env = std::getenv("SHADPS4_GPU_RANGES_FLAT");
        mode = env && (env[0] == '1' || env[0] == '2') ? env[0] - '0' : 0;
        if (mode != 0) {
            regions = std::make_unique<std::unique_ptr<Region>[]>(NumRegions);
            LOG_INFO(Render, "GpuRangeSet: SHADPS4_GPU_RANGES_FLAT={} ({})", mode,
                     mode == 2 ? "flat set, verified against boost::icl" : "flat set");
        }
    }

    void Add(VAddr base, size_t size) {
        if (mode != 1) {
            icl.Add(base, size);
        }
        if (mode == 0 || size == 0) {
            return;
        }
        ForEachRegion(base, size, [this](u64 index, u32 lo, u32 hi) {
            auto& slot = regions[index];
            if (!slot) {
                slot = std::make_unique<Region>();
            }
            AddPiece(slot->pieces, lo, hi);
        });
    }

    void Subtract(VAddr base, size_t size) {
        if (mode != 1) {
            icl.Subtract(base, size);
        }
        if (mode == 0 || size == 0) {
            return;
        }
        ForEachRegion(base, size, [this](u64 index, u32 lo, u32 hi) {
            if (auto& slot = regions[index]) {
                SubtractPiece(slot->pieces, lo, hi);
            }
        });
    }

    bool Contains(VAddr base, size_t size) const {
        if (mode == 0) {
            return icl.Contains(base, size);
        }
        if (size == 0) {
            return true; // as icl: the empty interval is contained in every set
        }
        // Contained iff one joined interval covers the whole query.
        bool contained = false;
        if (size != 0) {
            u32 visits = 0;
            ForEachFlat(base, size, [&](VAddr lo, VAddr hi) {
                contained = ++visits == 1 && lo == base && hi == base + size;
            });
        }
        if (mode == 2) [[unlikely]] {
            // Verify behaves as mode 0: the icl answer is returned (Opus review 2026-09-30).
            const bool reference = icl.Contains(base, size);
            if (contained != reference) {
                Mismatch("Contains", base, size);
            }
            return reference;
        }
        return contained;
    }

    template <typename Func>
    void ForEachInRange(VAddr base, size_t size, Func&& func) const {
        if (mode == 0) {
            icl.ForEachInRange(base, size, func);
            return;
        }
        if (mode == 2) [[unlikely]] {
            boost::container::small_vector<std::pair<VAddr, VAddr>, 32> flat, reference;
            ForEachFlat(base, size, [&](VAddr lo, VAddr hi) { flat.emplace_back(lo, hi); });
            icl.ForEachInRange(base, size,
                               [&](VAddr lo, VAddr hi) { reference.emplace_back(lo, hi); });
            ++checks;
            if (size != 0 && flat != reference) {
                Mismatch("ForEachInRange", base, size);
            }
            if ((checks & 0xfffff) == 0) {
                LOG_INFO(Render, "GpuRangeSet VERIFY: {} queries compared, {} mismatches, {} pieces",
                         checks, mismatches, fragments);
            }
            for (const auto& [lo, hi] : reference) {
                func(lo, hi);
            }
            return;
        }
        ForEachFlat(base, size, func);
    }

    /// Number of stored intervals (flat modes: pieces, so one more per region boundary crossed).
    size_t Fragments() const {
        return mode == 1 ? fragments : icl.m_ranges_set.iterative_size();
    }

private:
    struct Piece {
        u32 lo;
        u32 hi; ///< [lo, hi) inside the region, lo < hi <= RegionSize.
    };
    /// Sorted, disjoint and not touching each other.
    using Pieces = boost::container::small_vector<Piece, 6>;
    struct Region {
        Pieces pieces;
    };

    /// Calls func(region index, lo, hi) for the part of [base, base + size) in each region.
    template <typename Func>
    static void ForEachRegion(VAddr base, size_t size, Func&& func) {
        const VAddr end = base + size;
        ASSERT_MSG(end <= (1ULL << AddressBits) && end >= base, "GpuRangeSet: range {:#x}+{:#x}",
                   base, size);
        for (VAddr addr = base; addr < end;) {
            const u64 index = addr >> RegionBits;
            const VAddr region_base = index << RegionBits;
            const VAddr part_end = std::min<VAddr>(end, region_base + RegionSize);
            func(index, static_cast<u32>(addr - region_base),
                 static_cast<u32>(part_end - region_base));
            addr = part_end;
        }
    }

    /// First piece with hi >= x (overlapping or touching a range that starts at x, or behind it).
    template <typename P>
    static auto FirstEndingAt(P& pieces, u32 x) {
        return std::partition_point(pieces.begin(), pieces.end(),
                                    [x](const Piece& piece) { return piece.hi < x; });
    }
    /// First piece with hi > x (overlapping a range that starts at x, or behind it).
    template <typename P>
    static auto FirstEndingAfter(P& pieces, u32 x) {
        return std::partition_point(pieces.begin(), pieces.end(),
                                    [x](const Piece& piece) { return piece.hi <= x; });
    }

    void AddPiece(Pieces& pieces, u32 lo, u32 hi) {
        // First piece that ends at or after lo: it overlaps or touches the new range, or lies
        // behind it.
        auto first = FirstEndingAt(pieces, lo);
        const auto end = pieces.end();
        if (first == end || first->lo > hi) {
            pieces.insert(first, Piece{lo, hi});
            ++fragments;
            return;
        }
        if (first->lo <= lo && hi <= first->hi) {
            return; // already contained: the common case
        }
        first->lo = std::min(first->lo, lo);
        first->hi = std::max(first->hi, hi);
        // Swallow the following pieces the grown one now overlaps or touches.
        auto last = first + 1;
        while (last != end && last->lo <= first->hi) {
            first->hi = std::max(first->hi, last->hi);
            ++last;
        }
        fragments -= static_cast<size_t>(last - (first + 1));
        pieces.erase(first + 1, last);
    }

    void SubtractPiece(Pieces& pieces, u32 lo, u32 hi) {
        auto it = FirstEndingAfter(pieces, lo);
        while (it != pieces.end() && it->lo < hi) {
            if (it->lo < lo && it->hi > hi) {
                // The hole is strictly inside: split.
                const Piece right{hi, it->hi};
                it->hi = lo;
                pieces.insert(it + 1, right);
                ++fragments;
                return;
            }
            if (it->lo < lo) {
                it->hi = lo; // keep the left part
                ++it;
            } else if (it->hi > hi) {
                it->lo = hi; // keep the right part
                return;
            } else {
                it = pieces.erase(it);
                --fragments;
            }
        }
    }

    template <typename Func>
    void ForEachFlat(VAddr base, size_t size, Func&& func) const {
        if (size == 0 || fragments == 0) {
            return;
        }
        // The interval being assembled: pieces of neighbouring regions that touch are one
        // interval of the set.
        VAddr run_lo = 0;
        VAddr run_hi = 0;
        bool open = false;
        ForEachRegion(base, size, [&](u64 index, u32 lo, u32 hi) {
            const auto& slot = regions[index];
            if (!slot) {
                return;
            }
            const VAddr region_base = index << RegionBits;
            const auto& pieces = slot->pieces;
            for (auto it = FirstEndingAfter(pieces, lo); it != pieces.end() && it->lo < hi; ++it) {
                const Piece& piece = *it;
                const VAddr piece_lo = region_base + std::max(piece.lo, lo);
                const VAddr piece_hi = region_base + std::min(piece.hi, hi);
                if (open && piece_lo == run_hi) {
                    run_hi = piece_hi;
                    continue;
                }
                if (open) {
                    func(run_lo, run_hi);
                }
                run_lo = piece_lo;
                run_hi = piece_hi;
                open = true;
            }
        });
        if (open) {
            func(run_lo, run_hi);
        }
    }

    void Mismatch(const char* what, VAddr base, size_t size) const {
        if (++mismatches <= 16) {
            LOG_CRITICAL(Render, "GpuRangeSet VERIFY: {} differs from boost::icl for {:#x}+{:#x}",
                         what, base, size);
        }
    }

    int mode = 0;
    RangeSet icl;
    std::unique_ptr<std::unique_ptr<Region>[]> regions;
    size_t fragments = 0;
    mutable u64 checks = 0;
    mutable u64 mismatches = 0;
};

} // namespace VideoCore
