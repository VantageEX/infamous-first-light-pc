// SPDX-License-Identifier: GPL-2.0-or-later
// Passive phase-0 census. Included only by liverpool.cpp, outside AmdGpu.
#include <array>
#include <unordered_set>
#include <cstring>
#include <cstdlib>
#include "common/rdtsc.h"
namespace Vulkan { extern std::atomic<u64> g_iss_frame; }
namespace iss::hybrid_profile {
static bool Enabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_HYBRID_FAMILY_PROFILE");
        return p && p[0] == '1';
    }();
    return enabled;
}
// Any coroutine switch contaminates an elapsed TSC interval: never count it as CP work.
static thread_local u64 switches;
struct KeyHash {
    size_t operator()(const std::array<u64, 3>& k) const {
        return k[0] * 0x9e3779b97f4a7c15ULL ^ k[1] * 0xc2b2ae3d27d4eb4fULL ^ k[2];
    }
};
struct Family {
    u64 draws{}, ticks{}, contaminated{}, elapsed_ticks{}, repeat_tm{}, repeat_tmg{}, unknown{};
    u64 indexed{}, automatic{}, indirect{}, empty{}, overwritten{}, unfinished{};
    std::unordered_set<std::array<u64, 3>, KeyHash> tm, tmg;
    std::unordered_set<u64> materials, geometries;
};
struct Census {
    u64 frame = ~u64{0}, missing{}, abandoned{};
    u64 start_ns = Common::Perf::NowNs(), start_tsc = Common::FencedRDTSC();
    std::array<Family, 4> families;
    void Frame() {
        const auto now = Vulkan::g_iss_frame.load(std::memory_order_relaxed);
        if (now == frame) return;
        if (frame != ~u64{0}) {
            const auto ns = Common::Perf::NowNs();
            const double hz = double(Common::FencedRDTSC() - start_tsc) * 1e9 / double(ns - start_ns);
            for (unsigned k = 0; k < 4; ++k) {
                const auto& f = families[k];
                std::fprintf(stderr, "HYBRID_FAMILY frame=%llu kind=%u draws=%llu ticks=%llu ms=%.6f contaminated=%llu elapsed_ticks=%llu repeat_tm=%llu repeat_tmg=%llu materials=%zu geometries=%zu unknown_material=%llu missing=%llu abandoned=%llu hz=%.0f ns=%llu identities=%u\n",
                    frame, k, f.draws, f.ticks, f.ticks * 1000.0 / hz, f.contaminated,
                    f.elapsed_ticks, f.repeat_tm, f.repeat_tmg, f.materials.size(),
                    f.geometries.size(), f.unknown, missing, abandoned, hz, ns,
                    unsigned((frame & 15) == 0));
                std::fprintf(stderr, "HYBRID_COVERAGE frame=%llu kind=%u indexed=%llu automatic=%llu indirect=%llu empty=%llu overwritten=%llu unfinished=%llu\n",
                    frame, k, f.indexed, f.automatic, f.indirect, f.empty, f.overwritten,
                    f.unfinished);
            }
        }
        frame = now;
        families = {};
        missing = abandoned = 0;
    }
};
static Census& Stats() { static Census s; return s; }
// Lives in the graphics coroutine: pending markers never leak into another DCB.
struct Segment {
    bool enabled = Enabled(), pending{};
    u32 kind{};
    u64 technique{}, geometry{}, material{}, material_technique{}, bound_material{};
    u64 start{}, switch_start{};
    explicit Segment() { if (enabled) Stats().Frame(); }
    ~Segment() {
        if (enabled && pending) { ++Stats().abandoned; ++Stats().families[kind].unfinished; }
    }
    void Marker(const u32* p) {
        if (!enabled) return;
        u64 v[4]; std::memcpy(v, p + 1, sizeof(v));
        const u32 k = p[0] & 255;
        if (k == 4) {
            material_technique = v[0]; bound_material = v[1];
            if (pending && kind != 1) material = technique == v[0] ? v[1] : 0;
            return;
        }
        if (pending) { ++Stats().abandoned; ++Stats().families[kind].overwritten; }
        pending = k >= 1 && k <= 3;
        kind = pending ? k : 0;
        technique = v[0]; geometry = v[1];
        material = k == 1 ? v[2] : (material_technique == technique ? bound_material : 0);
        switch_start = switches;
        start = Common::FencedRDTSC();
    }
    void BeginDraw() {
        if (!enabled) return;
        if (!pending) { start = Common::FencedRDTSC(); switch_start = switches; kind = 0; }
    }
    void EndDraw(u32 index_count = ~u32{0}, u32 packet_type = 0) {
        if (!enabled) return;
        const u64 end = Common::FencedRDTSC();
        auto& s = Stats();
        auto& f = s.families[kind];
        ++f.draws;
        if (packet_type == 0) ++f.indexed;
        else if (packet_type == 1) ++f.automatic;
        else ++f.indirect;
        if (packet_type != 2 && index_count == 0) ++f.empty;
        f.elapsed_ticks += end - start;
        if (switches == switch_start) f.ticks += end - start;
        else ++f.contaminated;
        // Exact identities in one of every 16 frames; timing and counts on every frame.
        // Avoid thousands of diagnostic hash-node allocations on the timing-only frames.
        if (pending && (s.frame & 15) == 0) {
            f.geometries.insert(geometry);
            if (material) {
                f.materials.insert(material);
                f.repeat_tm += !f.tm.insert({technique, material, 0}).second;
                f.repeat_tmg += !f.tmg.insert({technique, material, geometry}).second;
            } else ++f.unknown;
        } else if (!pending) ++s.missing;
        pending = false;
        s.Frame();
    }
};
}
