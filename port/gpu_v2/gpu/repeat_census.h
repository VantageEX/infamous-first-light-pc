// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_REPEAT_CENSUS=1: how often the same resource is prepared again within one guest frame.
// Each kind counts its calls and its distinct keys per frame (buffer ranges, image descriptors,
// page protections); every 240 frames REPEAT_CENSUS reports the averages and the most repeated
// keys of the last frame, and how the calls split over the guest heaps (A 0x80000000,
// B 0x20000000 (dynamic), C 0x200000000 (resources)). Diagnostic only: a mutex per kind.
#pragma once
#include <cstdint>

namespace iss::repeat_census {
enum Kind : unsigned {
    ObtainBuffer, // BufferCache::ObtainBuffer, every caller
    StageBuffer,  // a shader stage's buffer descriptor (BindBuffers)
    VertexRange,  // a merged vertex buffer range (BindVertexBuffers)
    IndexRange,   // BindIndexBuffer
    Texture,      // an image descriptor (BindTextures)
    Protect,      // a page protection change (PageManager::Protect)
    Fault,        // a guest access violation handed to the GPU library (page, write flag)
    KindCount
};
extern const bool enabled;
void Note(Kind kind, std::uint64_t address, std::uint64_t size, std::uint32_t extra = 0);
} // namespace iss::repeat_census
