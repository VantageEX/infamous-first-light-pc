// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_CP_PATH_CYCLES=1 (diagnosis): exclusive time of the rasterizer entry points, by path, on
// the thread that runs them. Every few seconds: milliseconds per wall second (the share of a
// core) and microseconds per call of each path; the rest of the command processor's busy time
// is packet processing and waits outside these entry points.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <intrin.h>

namespace iss::cp_path {

enum Path : std::uint32_t {
    ShadowNative, GbufferNative, ShadowSuffix, Draw, DrawIndirect, Dispatch,
    // Functions the paths share (their time leaves the path that called them).
    RecorderExecute, MaterialBind, MaterialFind, PrepareRender, BeginRender, BindRes, BindTex,
    BindVertex, BindIndex, Dynamic, Reset,
    // Phases of NativeRecorderExecute (Switch).
    RecChain, RecTargets, RecBarriers, RecDraws, RecReads, RecSubmit, RecAccess,
    // Phases of NativeComputeExecute (its scope is RecSubmit: the checks before the first).
    CsTextures, CsReads, CsWrites, CsBarriers, CsRecord, CsAccess, PathCount
};

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_CP_PATH_CYCLES");
        return value && value[0] == '1';
    }();
    return enabled;
}

struct Totals {
    std::array<std::atomic<std::uint64_t>, PathCount> cycles{}, calls{};
    std::atomic<std::uint64_t> events{};
    std::uint64_t start_cycles{};
    std::chrono::steady_clock::time_point start_time{};
    bool started{};
};
inline Totals& State() {
    static Totals totals;
    return totals;
}

inline void Report() {
    auto& t = State();
    const auto now = std::chrono::steady_clock::now();
    const std::uint64_t cycles = __rdtsc();
    if (!t.started) {
        t.started = true;
        t.start_time = now;
        t.start_cycles = cycles;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - t.start_time).count();
    if (seconds < 5.0) return;
    const double hz = double(cycles - t.start_cycles) / seconds;
    static const char* const names[PathCount] = {"shadow_native", "gbuffer_native",
                                                  "shadow_suffix", "draw", "draw_indirect",
                                                  "dispatch", "recorder_execute",
                                                  "material_bind", "material_find",
                                                  "prepare_render", "begin_render", "bind_res",
                                                  "bind_tex", "bind_vertex", "bind_index",
                                                  "dynamic", "reset", "rec_chain",
                                                  "rec_targets", "rec_transit", "rec_draws",
                                                  "rec_reads", "rec_submit", "rec_access",
                                                  "cs_textures", "cs_reads", "cs_writes",
                                                  "cs_barriers", "cs_record", "cs_access"};
    char line[4096];
    int n = std::snprintf(line, sizeof(line), "CP_PATH_CYCLES seconds=%.1f", seconds);
    for (std::uint32_t p = 0; p < PathCount; ++p) {
        const std::uint64_t c = t.cycles[p].exchange(0, std::memory_order_relaxed);
        const std::uint64_t k = t.calls[p].exchange(0, std::memory_order_relaxed);
        const double ms = double(c) / hz * 1000.0;
        n += std::snprintf(line + n, sizeof(line) - n, " %s=%.1fms/s,%.2fus,%llu/s", names[p],
                           ms / seconds, k ? ms * 1000.0 / double(k) : 0.0,
                           (unsigned long long)(double(k) / seconds));
    }
    std::fprintf(stderr, "%s\n", line);
    t.start_time = now;
    t.start_cycles = cycles;
}

struct Frame {
    std::int32_t path = -1;
    std::uint64_t start{};
};
inline thread_local Frame current;

// Exclusive accounting: an inner scope pauses the outer one.
class Scope {
public:
    explicit Scope(Path path) : enabled{Enabled()} {
        if (!enabled) return;
        const std::uint64_t now = __rdtsc();
        if (current.path >= 0)
            State().cycles[current.path].fetch_add(now - current.start, std::memory_order_relaxed);
        saved = current;
        current = {std::int32_t(path), now};
        State().calls[path].fetch_add(1, std::memory_order_relaxed);
    }
    ~Scope() {
        if (!enabled) return;
        const std::uint64_t now = __rdtsc();
        State().cycles[current.path].fetch_add(now - current.start, std::memory_order_relaxed);
        current = {saved.path, now};
        if ((State().events.fetch_add(1, std::memory_order_relaxed) & 0xFFFF) == 0) Report();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    bool enabled;
    Frame saved{};
};

// The current scope continues as `path` (phases of one function).
inline void Switch(Path path) {
    if (!Enabled() || current.path < 0) return;
    const std::uint64_t now = __rdtsc();
    State().cycles[current.path].fetch_add(now - current.start, std::memory_order_relaxed);
    current = {std::int32_t(path), now};
}

} // namespace iss::cp_path
