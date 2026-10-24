// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <limits>
#include <immintrin.h>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

#include "common/assert.h"
#include "common/fault_monitor.h"
#include "common/scope_exit.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    // SHADPS4_GAME_MAIN_CENSUS (gmc3): Game:Main's GPU waits, timed with their call path.
    const bool gm = Common::FaultMonitor::t_is_game_main && Common::FaultMonitor::GameMainCensus();
    const u64 gm_t0 = gm ? Common::FaultMonitor::CensusTicks() : 0;
    if (gm) {
        Common::FaultMonitor::g_gm_in_gpu_wait.store(1, std::memory_order_relaxed);
    }
    SCOPE_EXIT {
        if (gm) {
            Common::FaultMonitor::g_gm_in_gpu_wait.store(0, std::memory_order_relaxed);
            Common::FaultMonitor::NoteGameMainGpuWait(Common::FaultMonitor::CensusTicks() - gm_t0);
        }
    };
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    while (true) {
        const vk::Result result = instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT);
        if (result == vk::Result::eSuccess) {
            break;
        }
        // A lost device used to spin here forever with nothing in the log (review 2026-09-27).
        if (result == vk::Result::eErrorDeviceLost) {
            instance.LogDeviceFault();
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost in semaphore wait (tick {})",
                   tick);
    }
    Refresh();
}

void MasterSemaphore::SpinWait(u64 tick, u64 spin_ns) {
    if (IsFree(tick)) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(spin_ns);
    do {
        Refresh();
        if (IsFree(tick)) {
            return;
        }
        for (int i = 0; i < 16; ++i) {
            _mm_pause();
        }
    } while (std::chrono::steady_clock::now() < deadline);
    Wait(tick);
}

u64 MasterSemaphore::QueryCounter() const {
    auto [result, value] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
    return result == vk::Result::eSuccess ? value : 0;
}

} // namespace Vulkan
