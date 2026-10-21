// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include <span>
#include <vector>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);
PFN_SrtWalker RegisterCaptureWalkerCode(const u8* ptr, size_t size,
                                        std::span<const u32> relocations);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs
    // Optional capture-only variant. Every transitive guest load calls ReadCapturedSrt.
    // Relocations rewrite its helper addresses after a shader-cache load (ASLR).
    PFN_SrtWalker capture_walker_func{};
    size_t capture_walker_func_size{};
    u32 capture_schema{};
    std::vector<u32> capture_relocations{};

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
