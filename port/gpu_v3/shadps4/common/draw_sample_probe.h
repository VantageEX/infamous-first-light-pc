// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include "common/logging/log.h"
namespace Common {

// Disabled mode does not read the clock or advance the sampling PRNG.
class DrawSampleProbe {
public:
    enum class Phase : std::uint8_t {
        Pipeline,
        Prepare,
        BindResources,
        BuildRenderingState,
        VertexIndexBuffers,
        PipelineResources,
        DynamicState,
        SchedulerBeginRendering,
        ActualDraw,
        ResetImageBindings,
        Count,
    };
    static DrawSampleProbe Begin() {
        if (!Enabled())
            return DrawSampleProbe(false);
        auto& state = RandomState();
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        if ((state & 127u) != 0)
            return DrawSampleProbe(false);
        return DrawSampleProbe(true);
    }
    void BeginPhase(const Phase phase) {
        if (active)
            marks[2 * Index(phase)] = NowTicks();
    }
    void EndPhase(const Phase phase) {
        if (active)
            marks[2 * Index(phase) + 1] = NowTicks();
    }
    void Finish() {
        if (!active)
            return;
        auto& stats = ThreadStats();
        for (std::size_t i = 0; i < PhaseCount; ++i) {
            const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   Clock::duration{marks[2 * i + 1] - marks[2 * i]})
                                   .count();
            stats.phase_ns[i] += static_cast<std::uint64_t>(delta);
        }
        stats.span_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           Clock::duration{marks[2 * PhaseCount - 1] - marks[0]})
                                           .count());
        ++stats.window_count;
        ++stats.total_count;
        active = false;
        if (stats.window_count != 4096)
            return;
        ++stats.sequence;
        const auto mean_us = [count = stats.window_count](std::uint64_t ns) {
            return static_cast<double>(ns) / (1000.0 * static_cast<double>(count));
        };
        LOG_INFO(Render_Vulkan,
                 "DrawSample seq={} window_count={} total_count={} span={:.3f}us "
                 "pipeline={:.3f} prepare={:.3f} bindres={:.3f} buildstate={:.3f} "
                 "vtxidx={:.3f} descriptors={:.3f} dynstate={:.3f} scheduler_begin={:.3f} "
                 "draw={:.3f} reset={:.3f} steady_clock_mark_overhead_included=true",
                 stats.sequence, stats.window_count, stats.total_count, mean_us(stats.span_ns),
                 mean_us(stats.phase_ns[Index(Phase::Pipeline)]),
                 mean_us(stats.phase_ns[Index(Phase::Prepare)]),
                 mean_us(stats.phase_ns[Index(Phase::BindResources)]),
                 mean_us(stats.phase_ns[Index(Phase::BuildRenderingState)]),
                 mean_us(stats.phase_ns[Index(Phase::VertexIndexBuffers)]),
                 mean_us(stats.phase_ns[Index(Phase::PipelineResources)]),
                 mean_us(stats.phase_ns[Index(Phase::DynamicState)]),
                 mean_us(stats.phase_ns[Index(Phase::SchedulerBeginRendering)]),
                 mean_us(stats.phase_ns[Index(Phase::ActualDraw)]),
                 mean_us(stats.phase_ns[Index(Phase::ResetImageBindings)]));
        stats.window_count = stats.span_ns = 0;
        stats.phase_ns.fill(0);
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr std::size_t PhaseCount = static_cast<std::size_t>(Phase::Count);
    struct Stats {
        std::uint64_t window_count{};
        std::uint64_t total_count{};
        std::uint64_t sequence{};
        std::uint64_t span_ns{};
        std::array<std::uint64_t, PhaseCount> phase_ns{};
    };
    static constexpr std::size_t Index(const Phase phase) {
        return static_cast<std::size_t>(phase);
    }
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_DRAW_SAMPLE");
            return value && value[0] == '1';
        }();
        return enabled;
    }
    static std::int64_t NowTicks() {
        return static_cast<std::int64_t>(Clock::now().time_since_epoch().count());
    }
    static std::uint32_t& RandomState() {
        static thread_local std::uint32_t state = 0xA57A'6123u;
        return state;
    }
    static Stats& ThreadStats() {
        static thread_local Stats stats;
        return stats;
    }
    explicit DrawSampleProbe(const bool active_) : active{active_} {}
    bool active;
    std::int64_t marks[PhaseCount * 2];
};
} // namespace Common
