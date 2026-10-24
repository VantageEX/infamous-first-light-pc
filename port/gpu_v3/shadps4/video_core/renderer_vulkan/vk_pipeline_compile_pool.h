// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/// SHADPS4_ASYNC_COMPUTE_PIPELINES=1 (default off, read once): the driver compile of a compute
/// pipeline (vkCreateComputePipelines: 20-30 ms for SPIR-V the driver has not seen, measured
/// 2026-10-07) runs on a small worker pool instead of on the thread that asked for it. Second
/// Son loads thousands of generated particle compute shaders as effects appear; each first use
/// blocked the CP front, starving the back (notes/parallel-back-build-2026-10-07.md sec. 2).
/// The pipeline object, its layouts and bindings exist at once; only the VkPipeline handle is
/// pending. Pipeline::Handle() before the compile finished runs the compile itself if no worker
/// has started it, else waits for that worker: never later than the synchronous path.
/// The boot preload (WarmUp) defers the same way, so the game starts while they compile.
bool AsyncComputePipelinesEnabled();

/// SHADPS4_PIPELINE_NO_OPT=1 (diagnostic, 2026-10-07): every game pipeline (graphics and compute)
/// is created with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT, to measure how much faster the
/// driver compiles a pipeline it has not seen (the first-use stutter) and what the unoptimized
/// shaders cost the GPU. Empty flags otherwise.
vk::PipelineCreateFlags DiagPipelineCreateFlags();

/// One pending vkCreateComputePipelines, owned by its Pipeline (never destroyed while a worker
/// can run it: pipelines live until the cache is destroyed, after PipelineCompilePool::Shutdown).
struct DeferredPipeline {
    enum : u32 { Queued = 0, Running = 1, Done = 2 };
    std::atomic<u32> state{Queued};
    std::atomic<bool> promoted{}; ///< A preload job re-queued ahead of the backlog (Promote).

    vk::Device device;
    vk::PipelineCache cache;
    vk::ShaderModule module;
    vk::PipelineLayout layout;
    bool subgroup_size_64{};
    bool preload{};
    std::string debug_name;
    vk::UniquePipeline* out{}; ///< The owning Pipeline's handle; written before state = Done.
    std::chrono::steady_clock::time_point queued_at{};

    /// Queued -> Running -> Done on the calling thread. False if another thread took it.
    bool TryRun(bool by_waiter);
    /// Returns once Done: runs the compile if it is still queued, else waits for its runner.
    void Wait();
    /// SHADPS4_PRECOMPILE (Opus review 2026-10-07, fix B): Queued -> Done without compiling, for
    /// a pipeline that will never be used (RetirePrecompiled). `out` stays empty, so the owner
    /// must never call Handle() afterwards. False if a worker or waiter already took it (that
    /// compile then finishes into the parked pipeline, harmless). Queues keep the raw pointer;
    /// a worker that pops it finds it taken.
    bool TryCancel();
    [[nodiscard]] bool IsDone() const noexcept {
        return state.load(std::memory_order_acquire) == Done;
    }
};

/// Process-wide worker pool for DeferredPipeline jobs (SHADPS4_ASYNC_COMPUTE_THREADS, default 4).
/// Jobs the game is about to use (pushed while playing) run before boot-preload jobs.
namespace PipelineCompilePool {
void Push(DeferredPipeline* job);
/// The game looked up a pipeline whose boot-preload compile is still queued: queue it again
/// ahead of the preload backlog (once), so it is compiled by a worker, not by the back's
/// first Handle() (Opus review 2026-10-07).
void Promote(DeferredPipeline* job);
/// Returns once nothing is queued or running (shader hot reload).
void WaitIdle();
/// WaitIdle for at most `timeout`; the jobs still queued or running (0 = idle). Duplicates from
/// Promote count twice (a progress figure, not an exact count).
size_t WaitIdleFor(std::chrono::milliseconds timeout);
/// Stops and joins the workers; queued jobs are dropped (their pipelines are about to be
/// destroyed). Call before destroying any pipeline that may still be queued.
/// RULE: a Pipeline with a deferred job may be destroyed only after Shutdown or a WaitIdle
/// with no new pushes since: the queues hold raw pointers (also duplicates from Promote, and
/// jobs a waiter already ran). Pipeline eviction would need a cancel path first.
void Shutdown();
} // namespace PipelineCompilePool

} // namespace Vulkan
