// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: numbers for the performance overlay (tlg_overlay.cpp). A thread started once samples
// CPU, RAM, GPU use (Windows counters) and VRAM (VK_EXT_memory_budget) twice a second;
// a value that cannot be measured stays negative and shows as "n/a".

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace Vulkan {
class Instance;
}

namespace TlgPerfStats {

/// Overlay items (TlgSettings::Values::overlay_items).
enum Item : unsigned {
    Fps = 1u << 0,
    FrameTime = 1u << 1,
    Upscaler = 1u << 2,
    Gpu = 1u << 3,
    Cpu = 1u << 4,
    Ram = 1u << 5,
    Vram = 1u << 6,
    Graph = 1u << 7, ///< frame time graph (drawn by the overlay, not part of the text)
    AllItems = (1u << 8) - 1,
};
inline constexpr unsigned NumItems = 8;
// tlg: the game renders natively (no upscaler), so the defaults show GPU load and VRAM instead.
inline constexpr unsigned DefaultItems = Fps | FrameTime | Gpu | Vram | Graph;

/// A thread's CPU use over the last sample, in percent of one core.
struct ThreadLoad {
    std::string name;
    float percent = 0.0f;
};

struct Sample {
    float fps = 0.0f, frame_ms = 0.0f;
    std::string upscaler;
    std::string gpu_name;
    float gpu_percent = -1.0f, cpu_percent = -1.0f;
    double ram_used_gb = -1.0, ram_total_gb = -1.0, vram_used_gb = -1.0, vram_total_gb = -1.0;
    /// The busiest threads, the busiest first (only while WantThreads is set).
    std::vector<ThreadLoad> threads;
    /// GPU memory tracking faults and pipelines made per second; -1 before the second sample.
    float faults_per_second = -1.0f, pipelines_per_second = -1.0f;
    uint64_t pipelines = 0; ///< pipelines made since the start
};

/// Counters the renderers bump: a guest memory fault the GPU's page tracking handled
/// (tlggpu_handle_fault), a graphics or compute pipeline made (from the driver's cache or compiled).
inline std::atomic<uint64_t> fault_count{0}, pipeline_count{0};
inline void CountFault() { fault_count.fetch_add(1, std::memory_order_relaxed); }
inline void CountPipeline() { pipeline_count.fetch_add(1, std::memory_order_relaxed); }
/// Set by the overlay while it shows the threads (the telemetry style): enumerating them costs
/// a system-wide thread snapshot on Windows.
inline std::atomic<bool> want_threads{false};

/// Process CPU time over wall time on every core, 0 to 100; -1 without a wall interval.
inline float CpuPercent(uint64_t process_delta, uint64_t wall_delta, unsigned cores) {
    if (!wall_delta || !cores) {
        return -1.0f;
    }
    return std::clamp(float(double(process_delta) * 100.0 / (double(wall_delta) * cores)), 0.0f, 100.0f);
}

/// The overlay text for the items in `mask`: one line each, or one line when `row`.
inline std::string Format(const Sample& s, unsigned mask, bool row) {
    std::string out;
    char part[160];
    const auto add = [&] {
        if (!out.empty()) {
            out += row ? "   " : "\n";
        }
        out += part;
    };
    const auto pair = [](char* to, size_t size, const char* label, double used, double total) {
        if (used < 0.0) {
            std::snprintf(to, size, "%s n/a", label);
        } else if (total < 0.0) {
            std::snprintf(to, size, "%s %.1f GB", label, used);
        } else {
            std::snprintf(to, size, "%s %.1f / %.1f GB", label, used, total);
        }
    };
    if (mask & Fps) {
        std::snprintf(part, sizeof(part), "%.0f FPS", s.fps);
        add();
    }
    if (mask & FrameTime) {
        std::snprintf(part, sizeof(part), "%.1f ms", s.frame_ms);
        add();
    }
    if (mask & Upscaler) {
        std::snprintf(part, sizeof(part), "%s", s.upscaler.empty() ? "Native" : s.upscaler.c_str());
        add();
    }
    if (mask & Gpu) {
        const char* name = s.gpu_name.empty() ? "GPU" : s.gpu_name.c_str();
        if (s.gpu_percent < 0.0f) {
            std::snprintf(part, sizeof(part), "%s n/a", name);
        } else {
            std::snprintf(part, sizeof(part), "%s %.0f%%", name, s.gpu_percent);
        }
        add();
    }
    if (mask & Cpu) {
        if (s.cpu_percent < 0.0f) {
            std::snprintf(part, sizeof(part), "CPU n/a");
        } else {
            std::snprintf(part, sizeof(part), "CPU %.0f%%", s.cpu_percent);
        }
        add();
    }
    if (mask & Ram) {
        pair(part, sizeof(part), "RAM", s.ram_used_gb, s.ram_total_gb);
        add();
    }
    if (mask & Vram) {
        pair(part, sizeof(part), "VRAM", s.vram_used_gb, s.vram_total_gb);
        add();
    }
    return out;
}

/// Starts the sampling thread the first time; later calls do nothing.
void Start(const Vulkan::Instance& instance);
/// The latest CPU, RAM, GPU and VRAM values (the caller fills FPS, frametime and upscaler).
Sample Latest();

} // namespace TlgPerfStats
