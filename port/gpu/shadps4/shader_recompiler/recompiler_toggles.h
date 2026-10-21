// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: recompiler fixes ported from the Shadow of the Colossus native port
// (github.com/AjncSystem/ShadowOfTheColossus-PC, MIT/GPL-2.0, commits 6f0b3e56, debb0c49,
// e3227814 and its vendored shared_memory_barrier_pass.cpp at 70bb4385) and the env switches
// that turn them on or off for A/B tests. Each switch is read once. The ISS_/GR2_ names reach
// here as TLG_ (the probe aliases the port prefix at start-up).

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include "common/elf_info.h"

namespace Shader::Toggles {

namespace Detail {
/// TLG_<name>=0/1 when set, otherwise the default (only evaluated then).
inline bool EnvFlag(const char* name, bool (*fallback)()) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) != 0 : fallback();
}

inline bool AlwaysOn() {
    return true;
}

/// inFAMOUS Second Son (CUSA00223 EU/US disc, CUSA00004 US digital) and First Light (CUSA00897,
/// CUSA00575; same engine and particle shaders): the LDS barrier fixes are aimed at the GPU
/// particle smoke (iss_pc #7) and stay off for the other games until tested.
inline bool InfamousDefault() {
    const std::string_view serial = Common::ElfInfo::Instance().GameSerial();
    return serial == "CUSA00223" || serial == "CUSA00004" || serial == "CUSA00897" ||
           serial == "CUSA00575";
}
} // namespace Detail

/// Workgroups smaller than one GCN wave (<64 threads) get LDS barriers like a 64-thread one.
/// TLG_LDS_SMALL=0/1; default on for inFAMOUS only.
inline bool SmallGroupLdsBarriers() {
    static const bool enabled = Detail::EnvFlag("TLG_LDS_SMALL", &Detail::InfamousDefault);
    return enabled;
}

/// Workgroups larger than one wave (any size above 64, a multiple of 64 or not: 96, 192, 256,
/// ...) get barriers where the guest relies on lockstep inside one wave. On 32-wide hardware
/// every 64-lane guest wave spans two host subgroups, so these are workgroup barriers; only
/// when each guest wave fits in one host subgroup (64-wide hardware) is a subgroup barrier
/// enough. TLG_LDS_MULTI=0/1; default on for inFAMOUS only.
inline bool MultiWaveLdsBarriers() {
    static const bool enabled = Detail::EnvFlag("TLG_LDS_MULTI", &Detail::InfamousDefault);
    return enabled;
}

/// The LDS barrier pass treats a ReadLane with a constant lane in uniform flow as uniform when
/// it decides whether a branch is divergent (one-wave workgroups only). From Pink-shadPS4
/// (luizgustavs) via SotC. TLG_LDS_READLANE=0/1; default on for inFAMOUS only.
inline bool LdsUniformReadLane() {
    static const bool enabled = Detail::EnvFlag("TLG_LDS_READLANE", &Detail::InfamousDefault);
    return enabled;
}

/// S_WQM_B64 writes the source mask grown to whole quads (it was a no-op). TLG_WQM=0 restores
/// the no-op; default on for every game (correctness fix).
inline bool WholeQuadMode() {
    static const bool enabled = Detail::EnvFlag("TLG_WQM", &Detail::AlwaysOn);
    return enabled;
}

/// Branch divergence (LDS barrier pass and wave64 lowering) also counts the lane-varying
/// sources other than LocalInvocationId: LocalInvocationIndex, LaneId, the lane masks behind
/// v_mbcnt (SubgroupLtMask) and the values returned by atomics. Without it the append pattern's
/// "if (mbcnt == 0) atomic" looked uniform, so barriers (the wave64 exchanges included) landed
/// inside a branch only one lane takes, which Vulkan leaves undefined. TLG_LANE_DIVERGENCE=0
/// restores the old test; default on for inFAMOUS only.
inline bool LaneDivergence() {
    static const bool enabled = Detail::EnvFlag("TLG_LANE_DIVERGENCE", &Detail::InfamousDefault);
    return enabled;
}

/// Lowers v_readfirstlane in every compute shader wider than a 32-lane host wave (not only in
/// those that also use ballots or v_mbcnt): the two host halves of a wave64 swap the value
/// through LDS so both read the first active lane of the whole wave, in uniform flow, unless the
/// value is provably the same in every lane. TLG_RFL_ALL=0/1; default off (experiment: adds two
/// workgroup barriers per lowered instruction).
inline bool ReadFirstLaneAll() {
    static const bool enabled = Detail::EnvFlag("TLG_RFL_ALL", [] { return false; });
    return enabled;
}

/// Bits for the shader binary cache version: shaders recompiled under other switches must not
/// be reused (the cache key only hashes the recompiler sources).
inline std::uint32_t ShaderBinarySalt() {
    return (SmallGroupLdsBarriers() ? 1u : 0u) | (MultiWaveLdsBarriers() ? 2u : 0u) |
           (LdsUniformReadLane() ? 4u : 0u) | (WholeQuadMode() ? 0u : 8u) |
           (LaneDivergence() ? 16u : 0u) | (ReadFirstLaneAll() ? 32u : 0u);
}

/// One line per switch with its state (shown once, when the GPU starts). Takes the logger as a
/// callback so this header stays free of the logging includes.
template <typename Log>
inline void LogStates(Log&& log) {
    log("TLG_LDS_SMALL", SmallGroupLdsBarriers(), "LDS barriers in workgroups under 64 threads");
    log("TLG_LDS_MULTI", MultiWaveLdsBarriers(),
        "LDS barriers in workgroups over 64 threads (workgroup scope on 32-wide GPUs)");
    log("TLG_LDS_READLANE", LdsUniformReadLane(), "uniform constant-lane ReadLane in LDS barriers");
    log("TLG_WQM", WholeQuadMode(), "S_WQM_B64 grows the mask to quads");
    log("TLG_LANE_DIVERGENCE", LaneDivergence(),
        "mbcnt masks, lane ids and atomic results make a branch divergent");
    log("TLG_RFL_ALL", ReadFirstLaneAll(), "wave64 ReadFirstLane lowered in every wide compute shader");
}

} // namespace Shader::Toggles
