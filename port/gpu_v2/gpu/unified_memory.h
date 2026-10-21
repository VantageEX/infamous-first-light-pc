// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_UNIFIED_MEMORY=1: the guest's dynamic heap is shared with the GPU as on the PS4.
//
// The game maps three direct memory heaps (src/memory.c): A at 0x80000000, B at 0x20000000
// (512 MB, the per-frame data the CPU writes: command buffers, constants, rings) and C at
// 0x200000000 (resources). With this flag heap B (ISS_UNIFIED_RANGE=base:size to change it) is
// imported host memory (its backing view, which is never protected) bound as one Vulkan buffer
// (BufferCache::EnableUnified): the GPU reads and writes the guest's own bytes in place.
// Bindings inside it skip uploads, stream copies, CPU write tracking (page protection) and
// readbacks.
//
// What keeps it correct, as on the hardware:
// - fences: the command processor writes the EOP/EOS/RELEASE_MEM labels that land in the heap
//   when the GPU work before them has completed (Rasterizer::DeferUnifiedFence), not when it
//   reads the packet, so the guest reuses its memory only after the GPU is done with it;
// - command processor writes into the heap (WRITE_DATA, DUMP_CONST_RAM, DMA fills and copies
//   of memory the GPU did not write) stay immediate as before: the command processor reads
//   what they write when it records later draws (user data tables). They are counted;
// - the command processor's own reads (WAIT_REG_MEM, COND_EXEC, semaphores) wait for the GPU
//   writes recorded into the heap before them (BufferCache::SyncUnifiedRead);
// - GPU writes are made visible to the host before each deferred fence (a host barrier).
// Requires the graphics queue only (no ISS_CP_SPLIT, no asc_queue, no draw pipe).
#pragma once
#include <cstdint>

namespace iss::unified {
extern const bool enabled;

enum Stat : unsigned {
    Reads,             // bindings read in place
    UntouchedReads,    // of which in memory the GPU never wrote (no hazard check)
    Writes,            // bindings the GPU writes
    ImageSources,      // texture uploads copied from the heap by the GPU
    CpWrites,          // immediate command processor writes into the heap
    CpWriteBytes,      // their bytes
    CpWritesTouched,   // of which into memory the GPU wrote
    FencesDeferred,    // labels written at GPU completion
    FencesOutside,     // labels outside the heap (written as before)
    FencesFlushed,     // deferred fences that submitted the work at once
    CpLabelWaits,      // command processor waits satisfied by a pending deferred label
    CpReadSyncs,       // command processor reads that waited for GPU writes
    CpReadWaitNs,      // time they waited
    StatCount
};
void Add(Stat stat, std::uint64_t value = 1);
/// UNIFIED_STATS on stderr every 5 seconds (command processor thread).
void MaybeReport();
/// The heap (ISS_UNIFIED_RANGE, default B).
std::uint64_t RangeBase();
std::uint64_t RangeSize();
} // namespace iss::unified
