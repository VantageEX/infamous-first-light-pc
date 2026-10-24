// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

namespace Shader {

// Default-off switches that change the emitted SPIR-V. The on-disk pipeline cache is keyed by
// guest program hash only, so while any of them is on the cache is skipped for the whole run
// (PipelineCache::WarmUp), otherwise cached binaries from the other setting would be reused.

/// SHADPS4_SPIRV_COHERENT=1: port of upstream #4896 - written storage buffers whose V# MTYPE is 3
/// (shared between invocations), and shared memory lowered to buffers, are decorated Coherent.
inline bool SpirvCoherentEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_SPIRV_COHERENT");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_MSAA_SAMPLE_WRAP=1: port of upstream c1d5b311 (#4889) - GCN wraps out-of-range MSAA
/// sample indices on image loads; Vulkan returns undefined values, so emit ms %= sample count.
inline bool MsaaSampleWrapEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_MSAA_SAMPLE_WRAP");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_PATCH_SHADERS=1: apply user/shader/patch/*.spv like GPU.patch_shaders, without editing
/// the config. Patches only apply when a module is compiled, so the cache is skipped as well.
inline bool PatchShadersEnv() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_PATCH_SHADERS");
        return value && value[0] == '1';
    }();
    return enabled;
}

inline bool SpirvVariantActive() {
    return SpirvCoherentEnabled() || MsaaSampleWrapEnabled() || PatchShadersEnv();
}

} // namespace Shader
