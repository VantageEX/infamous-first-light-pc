// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_platform.h"

namespace Vulkan {

namespace {

using Clock = std::chrono::steady_clock;

u64 NsSince(Clock::time_point start) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

/// Counters, logged every 128 finished compiles (relaxed: diagnostics only).
struct Stats {
    std::atomic<u64> done{}, done_preload{}, by_waiter{};
    std::atomic<u64> create_ns{}, max_create_ns{}, queue_ns{};
    std::atomic<u64> waits{}, wait_ns{}, max_wait_ns{}, preload_waits{};
    std::atomic<u64> queue_max{}, promoted{};
};
Stats g_stats;

void AtomicMax(std::atomic<u64>& slot, u64 value) {
    u64 cur = slot.load(std::memory_order_relaxed);
    while (value > cur && !slot.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
}

void MaybeLog(u64 done) {
    if (done % 128 != 0) {
        return;
    }
    const auto ms = [](const std::atomic<u64>& ns) {
        return static_cast<double>(ns.load(std::memory_order_relaxed)) / 1e6;
    };
    const u64 waits = g_stats.waits.load(std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan,
             "AsyncCompute: done={} preload={} run_by_waiter={} create_ms total={:.1f} max={:.2f} "
             "queue_ms total={:.1f} | waits={} (preload {}) wait_ms total={:.1f} max={:.2f} | "
             "queue_max={} promoted={}",
             done, g_stats.done_preload.load(std::memory_order_relaxed),
             g_stats.by_waiter.load(std::memory_order_relaxed), ms(g_stats.create_ns),
             ms(g_stats.max_create_ns), ms(g_stats.queue_ns), waits,
             g_stats.preload_waits.load(std::memory_order_relaxed), ms(g_stats.wait_ns),
             ms(g_stats.max_wait_ns), g_stats.queue_max.load(std::memory_order_relaxed),
             g_stats.promoted.load(std::memory_order_relaxed));
}

class Pool {
public:
    void Push(DeferredPipeline* job) {
        {
            std::scoped_lock lk{mutex};
            if (threads.empty() && !stopped) {
                Start();
            }
            (job->preload ? preload : demand).push_back(job);
            AtomicMax(g_stats.queue_max, demand.size() + preload.size());
        }
        cv.notify_one();
    }

    /// A boot-preload job the game is about to use: queue it again ahead of the backlog. The old
    /// entry stays in `preload`; whichever entry a worker pops second finds the job taken.
    void Promote(DeferredPipeline* job) {
        if (job->promoted.exchange(true, std::memory_order_relaxed)) {
            return;
        }
        {
            std::scoped_lock lk{mutex};
            if (threads.empty()) {
                return; // shut down: the waiter runs it
            }
            demand.push_back(job);
        }
        g_stats.promoted.fetch_add(1, std::memory_order_relaxed);
        cv.notify_one();
    }

    void WaitIdle() {
        std::unique_lock lk{mutex};
        if (threads.empty()) {
            // Never started, or shut down: queued jobs run on their waiters.
            return;
        }
        idle_cv.wait(lk, [&] { return demand.empty() && preload.empty() && running == 0; });
    }

    /// WaitIdle for at most `timeout`; returns the jobs still queued or running (0 = idle).
    size_t WaitIdleFor(std::chrono::milliseconds timeout) {
        std::unique_lock lk{mutex};
        if (threads.empty()) {
            return 0;
        }
        idle_cv.wait_for(lk, timeout,
                         [&] { return demand.empty() && preload.empty() && running == 0; });
        return demand.size() + preload.size() + running;
    }

    void Shutdown() {
        std::vector<std::jthread> joining;
        {
            std::scoped_lock lk{mutex};
            stopped = true;
            for (auto& thread : threads) {
                thread.request_stop();
            }
            joining = std::move(threads);
            threads.clear();
        }
        cv.notify_all();
        joining.clear(); // joins
        std::scoped_lock lk{mutex};
        demand.clear();
        preload.clear();
        idle_cv.notify_all();
    }

private:
    // Caller holds the mutex.
    void Start() {
        const char* env = std::getenv("SHADPS4_ASYNC_COMPUTE_THREADS");
        const u32 count = std::clamp<u32>(env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 4u,
                                          1u, 16u);
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this, i](std::stop_token stoken) { Run(stoken, i); });
        }
        LOG_INFO(Render_Vulkan,
                 "AsyncCompute: {} compile threads; a compute pipeline's driver compile runs off the "
                 "thread that asked for it, its first use waits only if it is still compiling",
                 count);
    }

    void Run(std::stop_token stoken, u32 index) {
        const std::string name = fmt::format("shadPS4:PipelineCompile{}", index);
        Common::SetCurrentThreadName(name.c_str());
        while (true) {
            DeferredPipeline* job{};
            {
                std::unique_lock lk{mutex};
                // wait() returns the predicate after a stop request: check the token too, so a
                // shutdown drops the backlog instead of compiling it (Opus review 2026-10-07).
                if (!cv.wait(lk, stoken, [&] { return !demand.empty() || !preload.empty(); }) ||
                    stoken.stop_requested()) {
                    return;
                }
                const bool from_preload = demand.empty();
                auto& queue = from_preload ? preload : demand;
                job = queue.front();
                queue.pop_front();
                ++running;
                if (from_preload && preload.empty() && !preload_drained_logged) {
                    preload_drained_logged = true;
                    LOG_INFO(Render_Vulkan,
                             "AsyncCompute: boot preload queue drained (last job taken) {:.0f} ms "
                             "after it was queued",
                             static_cast<double>(NsSince(job->queued_at)) / 1e6);
                }
            }
            job->TryRun(false);
            {
                std::scoped_lock lk{mutex};
                --running;
                if (running == 0 && demand.empty() && preload.empty()) {
                    idle_cv.notify_all();
                }
            }
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::condition_variable idle_cv;
    std::deque<DeferredPipeline*> demand;
    std::deque<DeferredPipeline*> preload;
    u32 running{};
    bool stopped{};
    bool preload_drained_logged{};
    std::vector<std::jthread> threads;
};

Pool& GetPool() {
    // Never destroyed: a static Pool would die before the global presenter that owns the
    // pipeline cache, whose destructor still calls Shutdown (Opus review 2026-10-07).
    static Pool* pool = new Pool;
    return *pool;
}

} // Anonymous namespace

bool AsyncComputePipelinesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASYNC_COMPUTE_PIPELINES");
        return value && value[0] == '1';
    }();
    return enabled;
}

vk::PipelineCreateFlags DiagPipelineCreateFlags() {
    static const bool no_opt = [] {
        const char* value = std::getenv("SHADPS4_PIPELINE_NO_OPT");
        const bool on = value && value[0] == '1';
        if (on) {
            LOG_WARNING(Render_Vulkan, "SHADPS4_PIPELINE_NO_OPT: pipelines are created without "
                                       "driver optimization (diagnostic)");
        }
        return on;
    }();
    return no_opt ? vk::PipelineCreateFlagBits::eDisableOptimization : vk::PipelineCreateFlags{};
}

bool DeferredPipeline::TryRun(bool by_waiter) {
    u32 expected = Queued;
    if (!state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) {
        return false;
    }
    const auto t_start = Clock::now();
    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = subgroup_size_64 ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .flags = DiagPipelineCreateFlags(),
        .stage = shader_ci,
        .layout = layout,
    };
    auto [result, pipe] = device.createComputePipelineUnique(cache, compute_pipeline_ci);
    ASSERT_MSG(result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(result));
    SetObjectName(device, *pipe, debug_name);
    *out = std::move(pipe);
    const u64 create_ns = NsSince(t_start);
    g_stats.create_ns.fetch_add(create_ns, std::memory_order_relaxed);
    AtomicMax(g_stats.max_create_ns, create_ns);
    g_stats.queue_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(t_start - queued_at)
                             .count()),
        std::memory_order_relaxed);
    g_stats.by_waiter.fetch_add(by_waiter ? 1 : 0, std::memory_order_relaxed);
    g_stats.done_preload.fetch_add(preload ? 1 : 0, std::memory_order_relaxed);
    const u64 done = g_stats.done.fetch_add(1, std::memory_order_relaxed) + 1;
    // Publishes *out to every thread that observes Done with acquire.
    state.store(Done, std::memory_order_release);
    state.notify_all();
    MaybeLog(done);
    return true;
}

bool DeferredPipeline::TryCancel() {
    u32 expected = Queued;
    if (!state.compare_exchange_strong(expected, Done, std::memory_order_acq_rel)) {
        return false;
    }
    state.notify_all();
    return true;
}

void DeferredPipeline::Wait() {
    if (IsDone()) {
        return;
    }
    const auto t_start = Clock::now();
    if (!TryRun(true)) {
        for (u32 s = state.load(std::memory_order_acquire); s != Done;
             s = state.load(std::memory_order_acquire)) {
            state.wait(s, std::memory_order_acquire);
        }
    }
    const u64 wait_ns = NsSince(t_start);
    g_stats.waits.fetch_add(1, std::memory_order_relaxed);
    g_stats.preload_waits.fetch_add(preload ? 1 : 0, std::memory_order_relaxed);
    g_stats.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
    AtomicMax(g_stats.max_wait_ns, wait_ns);
}

namespace PipelineCompilePool {

void Push(DeferredPipeline* job) {
    job->queued_at = Clock::now();
    GetPool().Push(job);
}

void Promote(DeferredPipeline* job) {
    GetPool().Promote(job);
}

void WaitIdle() {
    GetPool().WaitIdle();
}

size_t WaitIdleFor(std::chrono::milliseconds timeout) {
    return GetPool().WaitIdleFor(timeout);
}

void Shutdown() {
    GetPool().Shutdown();
}

} // namespace PipelineCompilePool

} // namespace Vulkan
