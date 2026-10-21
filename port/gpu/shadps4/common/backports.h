// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// tlg: switches and counters for the fixes backported from other PS4 ports and from shadPS4
// upstream (iss_pc catalog of 2026-10-15). One entry per backport, so a bad scene can be traced
// to the exact change: every switch is logged once at start-up ("Backports: BPnn ..."), every
// hit is counted and the non-zero counters are printed with the frame statistics. The ISS_/GR2_
// names reach here as TLG_ (the probe aliases the port prefix at start-up).
//
//   TLG_BPnn=0/1      turns one backport off/on (default in the table below)
//   (the in-game menu switches the ones marked live in kLive; the rest need a restart)
//   TLG_BACKPORTS=0   turns every backport with a default of "on" off (the baseline of the
//                     release before them); individual TLG_BPnn still win over it.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Common::Backports {

enum class Id : unsigned {
    IndirectFrameLeak,    // BP01 shadPS4 #5258
    ImageFromRangeFirst,  // BP02 shadPS4 145932bf (#5240)
    ExecEmptyBranch,      // BP03 shadPS4 c34ca21f
    ClampOpsPlain,        // BP04 shadPS4 1c2b8f97 / #5346
    SaturateNan,          // BP05 shadPS4 #5257
    MaximalReconvergence, // BP06 SotC 8d84cf7
    AsyncSkippable,       // BP07 bbport 10baf9d
    PipelineCacheBlob,    // BP08 Mrsuss b972798 + 0xCydral bdb9769
    Image1DAs2D,          // BP09 CTR GetLostIn d26bcb2
    VertexIndexBarriers,  // BP10 shadPS4 #5262
    FirstInstanceDouble,  // BP11 shadPS4 #5348
    ShadPS4Pr5264,        // BP12 shadPS4 #5264
    Count,
};

struct Entry {
    const char* id;     // "BP01"
    const char* env;    // TLG_BP01
    bool default_on;
    bool shader;        // changes the generated shaders (part of the shader cache key)
    const char* what;
    const char* source;
};

inline constexpr std::array<Entry, static_cast<std::size_t>(Id::Count)> kEntries{{
    {"BP01", "TLG_BP01", true, false, "free the frame of nested indirect buffers", "shadPS4 #5258"},
    {"BP02", "TLG_BP02", true, false, "FindImageFromRange scans the first page only", "shadPS4 145932bf #5240"},
    {"BP03", "TLG_BP03", true, true, "keep EXEC after an empty divergent branch", "shadPS4 c34ca21f"},
    {"BP04", "TLG_BP04", true, true, "V_RCP/RSQ/LOG_CLAMP and V_RSQ_LEGACY really clamp to +-FLT_MAX", "shadPS4 1c2b8f97 #5346"},
    {"BP05", "TLG_BP05", true, true, "saturate maps NaN to 0 like DX10_CLAMP (NClamp)", "shadPS4 #5257"},
    {"BP06", "TLG_BP06", true, true, "VK_KHR_shader_maximal_reconvergence in compute", "SotC 8d84cf7"},
    {"BP07", "TLG_BP07", false, false, "async compile never skips full-screen/indirect draws", "bbport 10baf9d"},
    {"BP08", "TLG_BP08", true, false, "persist the driver VkPipelineCache blob", "Mrsuss b972798, 0xCydral bdb9769"},
    {"BP09", "TLG_BP09", false, true, "every 1D image as 2D with height 1", "CTR d26bcb2"},
    {"BP10", "TLG_BP10", true, false, "barriers for vertex/index/indirect buffers", "shadPS4 #5262"},
    {"BP11", "TLG_BP11", true, true, "firstInstance counted once in indirect draws", "shadPS4 #5348"},
    {"BP12", "TLG_BP12", false, true, "EarlyFragmentTests when DEPTH_BEFORE_SHADER is set", "shadPS4 7d28d279 #5264"},
}};

namespace Detail {
inline bool Master() {
    static const bool on = [] {
        const char* v = std::getenv("TLG_BACKPORTS");
        return !(v && *v) || std::atoi(v) != 0;
    }();
    return on;
}

inline std::array<std::atomic<std::uint64_t>, kEntries.size()>& Hits() {
    static std::array<std::atomic<std::uint64_t>, kEntries.size()> hits{};
    return hits;
}

inline bool Eval(const Entry& e) {
    const char* v = std::getenv(e.env);
    if (v && *v) {
        return std::atoi(v) != 0;
    }
    return e.default_on && Master();
}
} // namespace Detail

/// Backports whose path is chosen at every use, so the in-game menu can switch them while the
/// game runs. The others change generated shaders, the device or the cache files and only take
/// effect on the next start.
inline constexpr std::array<bool, kEntries.size()> kLive{{
    true,  // BP01 IndirectFrameLeak
    true,  // BP02 ImageFromRangeFirst
    false, // BP03 ExecEmptyBranch (shader)
    false, // BP04 ClampOpsPlain (shader)
    false, // BP05 SaturateNan (shader)
    false, // BP06 MaximalReconvergence (device)
    true,  // BP07 AsyncSkippable
    false, // BP08 PipelineCacheBlob (loaded at start)
    false, // BP09 Image1DAs2D (image creation, shader)
    true,  // BP10 VertexIndexBarriers
    false, // BP11 FirstInstanceDouble (shader)
    false, // BP12 (EarlyFragmentTests, shader)
}};

/// A backport that changes generated shaders (it is part of the shader cache key) can never be
/// switched while the game runs: the pipelines already built keep the old code.
constexpr bool LiveMatchesShader() {
    for (std::size_t i = 0; i < kEntries.size(); ++i) {
        if (kLive[i] && kEntries[i].shader) {
            return false;
        }
    }
    return true;
}
static_assert(LiveMatchesShader(), "a shader-changing backport cannot be live");

namespace Detail {
/// 0 = off, 1 = on, set from the environment on first use and by SetLive afterwards.
inline std::array<std::atomic<std::int8_t>, kEntries.size()>& States() {
    static std::array<std::atomic<std::int8_t>, kEntries.size()> states{};
    static const bool initialized = [] {
        for (std::size_t i = 0; i < states.size(); ++i) {
            states[i].store(Eval(kEntries[i]) ? 1 : 0, std::memory_order_relaxed);
        }
        return true;
    }();
    (void)initialized;
    return states;
}
} // namespace Detail

/// TLG_OCCLUSION_MODE (ISS_ alias): how the ZPASS_DONE occlusion counter is answered. 0 = the old fake,
/// ever-growing counter, 1 = both values equal, 2 = end far above begin, 3 = zero counts (default),
/// 4 = like 0 but TLG_OCCLUSION_STEP pixels per render backend and query (First Light: modes 1-3
/// write a backend's end over the next one's begin, and 3 gives its light glare a 0/0).
inline int OcclusionMode() {
    static const int mode = [] {
        const char* v = std::getenv("TLG_OCCLUSION_MODE");
        return v && *v ? std::atoi(v) : 3;
    }();
    return mode;
}

/// TLG_OCCLUSION_STEP: pixels each render backend counts per query in occlusion mode 4 (default 1024).
inline unsigned long long OcclusionStep() {
    static const unsigned long long step = [] {
        const char* v = std::getenv("TLG_OCCLUSION_STEP");
        const long long n = v && *v ? std::atoll(v) : 1024;
        return static_cast<unsigned long long>(n > 0 ? n : 1024);
    }();
    return step;
}

/// State of one backport.
inline bool On(Id id) {
    return Detail::States()[static_cast<std::size_t>(id)].load(std::memory_order_relaxed) != 0;
}

/// Applies a choice stored in tlg.ini (bpNN=0/1, read by TlgSettings::Load at start-up), unless the
/// entry's environment variable is set: the environment (the launcher, a script, a .bat) wins.
/// Only meant for start-up, before the shaders and the device are built.
inline void ApplyConfig(std::size_t index, bool on) {
    if (index >= kEntries.size()) {
        return;
    }
    const char* v = std::getenv(kEntries[index].env);
    if (v && *v) {
        return;
    }
    Detail::States()[index].store(on ? 1 : 0, std::memory_order_relaxed);
}

/// Whether the in-game menu may switch this backport without a restart.
inline bool IsLive(Id id) {
    return kLive[static_cast<std::size_t>(id)];
}

/// Switches a live backport; returns false (and changes nothing) for the ones that need a restart.
inline bool SetLive(Id id, bool on) {
    if (!IsLive(id)) {
        return false;
    }
    Detail::States()[static_cast<std::size_t>(id)].store(on ? 1 : 0, std::memory_order_relaxed);
    std::printf("Backports: %s switched to %d in game\n", kEntries[static_cast<std::size_t>(id)].id,
                on ? 1 : 0);
    return true;
}

/// Counts one use of the backport (cheap; call where the changed path actually runs).
inline void Hit(Id id) {
    Detail::Hits()[static_cast<std::size_t>(id)].fetch_add(1, std::memory_order_relaxed);
}

/// Hits counted so far for one backport (the in-game menu shows it).
inline std::uint64_t Hits(Id id) {
    return Detail::Hits()[static_cast<std::size_t>(id)].load(std::memory_order_relaxed);
}

/// Bits for the shader binary cache version: shaders recompiled under other states must not be
/// reused (the cache key only hashes the recompiler sources). Fits in 8 bits.
inline std::uint32_t ShaderSalt() {
    std::uint32_t salt = 0;
    std::uint32_t bit = 1;
    for (std::size_t i = 0; i < kEntries.size(); ++i) {
        if (kEntries[i].shader) {
            salt |= On(static_cast<Id>(i)) ? bit : 0u;
            bit <<= 1;
        }
    }
    return salt;
}

/// One line per backport, once at start-up (the run log keeps stdout).
inline void LogStates() {
    std::printf("Backports: occlusion mode %d (TLG_OCCLUSION_MODE; 0 = old fake counter, 3 = default)\n",
                OcclusionMode());
    for (std::size_t i = 0; i < kEntries.size(); ++i) {
        const Entry& e = kEntries[i];
        std::printf("Backports: %s %s=%d (%s; %s)\n", e.id, e.env,
                    On(static_cast<Id>(i)) ? 1 : 0, e.what, e.source);
    }
}

/// With the frame statistics: the counters that moved since the last call, as totals.
inline void LogHits() {
    static std::array<std::uint64_t, kEntries.size()> last{};
    bool header = false;
    for (std::size_t i = 0; i < kEntries.size(); ++i) {
        const std::uint64_t now = Detail::Hits()[i].load(std::memory_order_relaxed);
        if (now != last[i]) {
            std::printf("%s%s %llu", header ? ", " : "Backport hits (total): ", kEntries[i].id,
                        static_cast<unsigned long long>(now));
            header = true;
            last[i] = now;
        }
    }
    if (header) {
        std::printf("\n");
    }
}

} // namespace Common::Backports
