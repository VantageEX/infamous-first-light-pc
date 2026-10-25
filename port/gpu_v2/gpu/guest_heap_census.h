// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_HEAP_CENSUS=1 (diagnosis, off by default): how much of the game's GPU heaps is allocated.
//
// The game takes its GPU memory from the kernel once (512 MB onion, ~3.3 GB garlic) and carves
// it with its own allocator (US 1.07: blocks in an address-ordered red-black tree; insert
// 0x58b60, release of a block 0x5a790; ghidra notes in CONHECIMENTO_PORTS_PS4.md). Renderer
// v2's arena keeps every range the GPU used resident for good, so compare it with what the game
// holds: hooks on both routines count the bytes of live blocks per heap.
#pragma once

#include <cstdint>

namespace iss::heap_census {

struct Heap {
    std::uint64_t live_bytes;  ///< Blocks and slabs allocated now.
    std::uint64_t peak_bytes;  ///< The most live_bytes has been.
    std::uint64_t lowest;      ///< Lowest and highest address a block reached (the span used).
    std::uint64_t highest;
    std::uint64_t allocs, frees;
};

/// Installs the hooks (US 1.07 only: the entry bytes are checked). Before guest threads start.
void Install(std::uint64_t image_base);
/// Heap 0 and 1 in the order they were first seen; false when not installed.
bool Read(Heap out[2]);

} // namespace iss::heap_census
