// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <utility>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace VideoCore {

/**
 * Second queue of the graphics family for guest readback copies (SHADPS4_READBACK_QUEUE=1).
 *
 * On the main queue a readback copy runs after everything submitted before it, although
 * measured 2026-09-23 69.5% of readbacks read data whose producing batch had already
 * completed. Here the copy waits only for its producer's main-timeline value, then signals
 * this queue's own timeline; the timeline wait carries the cross-queue memory dependency.
 * Mode 1 (same family) needs no ownership transfer. Mode 2 (transfer-only family) reads
 * EXCLUSIVE buffers without a queue-family ownership transfer: undefined by the spec for
 * contents, relied on here only as an experiment on NVIDIA, where buffers carry no
 * family-specific layout or compression. Recording is command processor only; a recorded
 * slot may be submitted from any thread (SHADPS4_FAULT_GUEST_SUBMIT, see RecordSlot /
 * SubmitSlot): every submit on the queue, and the timeline value it signals, is taken under
 * queue_mutex, so values reach the queue in increasing order.
 */
class ReadbackQueue {
public:
    explicit ReadbackQueue(const Vulkan::Instance& instance_)
        : ReadbackQueue(instance_, instance_.GetReadbackQueue(),
                        instance_.GetReadbackQueueFamilyIndex(), nullptr) {}

    /// SHADPS4_READBACK_PREBATCH=1: `queue_` is the main graphics queue and `submit_lock_` the
    /// scheduler's submit mutex. A copy submitted here runs after everything already submitted
    /// and before the batch the command processor is still recording.
    ReadbackQueue(const Vulkan::Instance& instance_, vk::Queue queue_, u32 family,
                  std::mutex* submit_lock_)
        : instance{instance_}, queue{queue_}, submit_lock{submit_lock_}, semaphore{instance_} {
        const auto device = instance.GetDevice();
        auto [pool_result, new_pool] = device.createCommandPoolUnique(vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                     vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = family,
        });
        ASSERT_MSG(pool_result == vk::Result::eSuccess, "ReadbackQueue: command pool failed");
        pool = std::move(new_pool);
        auto [alloc_result, buffers] = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<u32>(NumSlots),
        });
        ASSERT_MSG(alloc_result == vk::Result::eSuccess, "ReadbackQueue: allocation failed");
        for (size_t i = 0; i < NumSlots; ++i) {
            cmdbufs[i] = buffers[i];
        }
    }

    ~ReadbackQueue() {
        // Nothing may still execute from this pool when it is destroyed.
        semaphore.Wait(semaphore.CurrentTick() - 1);
    }

    /// Records with `record(cmdbuf)`, submits after `wait_value` on `wait_semaphore` (none if
    /// 0) and returns the value this queue's timeline reaches when the copy is done.
    template <typename Record>
    u64 Submit(vk::Semaphore wait_semaphore, u64 wait_value, Record&& record) {
        return SubmitSlot(RecordSlot(wait_semaphore, wait_value, std::forward<Record>(record)));
    }

    /// Records with `record(cmdbuf)` into a slot and returns it without submitting. The slot
    /// is held (never reset or handed out again) until SubmitSlot runs on it, from any thread.
    /// Recording thread only, like Submit.
    template <typename Record>
    size_t RecordSlot(vk::Semaphore wait_semaphore, u64 wait_value, Record&& record) {
        const size_t slot = AcquireSlot();
        const vk::CommandBuffer cmdbuf = cmdbufs[slot];
        ASSERT(cmdbuf.reset() == vk::Result::eSuccess);
        ASSERT(cmdbuf.begin(vk::CommandBufferBeginInfo{
                   .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) ==
               vk::Result::eSuccess);
        record(cmdbuf);
        ASSERT(cmdbuf.end() == vk::Result::eSuccess);
        slot_wait_semaphores[slot] = wait_semaphore;
        slot_wait_values[slot] = wait_value;
        return slot;
    }

    /// Submits a slot RecordSlot returned and returns the value this queue's timeline reaches
    /// when the copy is done. Any thread; the caller publishes the slot index to the submitting
    /// thread with its own release/acquire pair. The timeline value is taken and submitted
    /// under one lock, so a slot submitted later signals a larger value.
    u64 SubmitSlot(size_t slot) {
        std::scoped_lock lk{queue_mutex};
        const vk::CommandBuffer cmdbuf = cmdbufs[slot];
        const vk::Semaphore wait_semaphore = slot_wait_semaphores[slot];
        const u64 wait_value = slot_wait_values[slot];
        const u64 signal_value = semaphore.NextTick();
        const vk::Semaphore signal_semaphore = semaphore.Handle();
        const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
        const u32 wait_count = wait_value != 0 ? 1u : 0u;
        const vk::TimelineSemaphoreSubmitInfo timeline_info = {
            .waitSemaphoreValueCount = wait_count,
            .pWaitSemaphoreValues = &wait_value,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_value,
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_info,
            .waitSemaphoreCount = wait_count,
            .pWaitSemaphores = &wait_semaphore,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmdbuf,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphore,
        };
        vk::Result result;
        if (submit_lock) {
            std::scoped_lock submit_lk{*submit_lock};
            result = queue.submit(submit_info);
        } else {
            result = queue.submit(submit_info);
        }
        ASSERT_MSG(result == vk::Result::eSuccess, "ReadbackQueue: submit failed: {}",
                   vk::to_string(result));
        slot_values[slot].store(signal_value, std::memory_order_relaxed);
        // After the value: AcquireSlot reads it behind this acquire.
        slot_held[slot].store(false, std::memory_order_release);
        return signal_value;
    }

    Vulkan::MasterSemaphore& Semaphore() noexcept {
        return semaphore;
    }

    /// SHADPS4_READBACK_PREBATCH with SHADPS4_ASYNC_SUBMIT: the command processor reserves the
    /// value here, in order, and the recording worker submits it with SubmitReserved when it
    /// reaches that point of the command stream - after every batch recorded before it and
    /// before the batch still being recorded.
    u64 ReserveTick() noexcept {
        return semaphore.NextTick();
    }

    template <typename Record>
    void SubmitReserved(u64 signal_value, Record&& record) {
        const size_t slot = AcquireSlot();
        const vk::CommandBuffer cmdbuf = cmdbufs[slot];
        ASSERT(cmdbuf.reset() == vk::Result::eSuccess);
        ASSERT(cmdbuf.begin(vk::CommandBufferBeginInfo{
                   .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) ==
               vk::Result::eSuccess);
        record(cmdbuf);
        ASSERT(cmdbuf.end() == vk::Result::eSuccess);
        const vk::Semaphore signal_semaphore = semaphore.Handle();
        const vk::TimelineSemaphoreSubmitInfo timeline_info = {
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_value,
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_info,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmdbuf,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphore,
        };
        vk::Result result;
        if (submit_lock) {
            std::scoped_lock lk{*submit_lock};
            result = queue.submit(submit_info);
        } else {
            result = queue.submit(submit_info);
        }
        ASSERT_MSG(result == vk::Result::eSuccess, "ReadbackQueue: submit failed: {}",
                   vk::to_string(result));
        slot_values[slot].store(signal_value, std::memory_order_relaxed);
        slot_held[slot].store(false, std::memory_order_release);
    }

    /// True when this is the main graphics queue (copies are ordered by submission).
    bool InQueueOrder() const noexcept {
        return submit_lock != nullptr;
    }

private:
    /// Next free slot in round-robin order, its last copy complete. A slot recorded and not yet
    /// submitted is skipped: its command buffer must not be reset, and its value does not exist
    /// yet. At most one slot per fault job is held, far fewer than NumSlots.
    size_t AcquireSlot() {
        for (size_t tries = 0;; ++tries) {
            const size_t slot = next_slot++ % NumSlots;
            if (!slot_held[slot].load(std::memory_order_acquire)) {
                semaphore.Wait(slot_values[slot].load(std::memory_order_relaxed));
                slot_held[slot].store(true, std::memory_order_relaxed);
                return slot;
            }
            ASSERT_MSG(tries < NumSlots, "ReadbackQueue: every slot holds an unsubmitted copy");
        }
    }

    static constexpr size_t NumSlots = 32;
    const Vulkan::Instance& instance;
    vk::Queue queue;
    std::mutex* submit_lock{};
    /// Serializes SubmitSlot across threads (queue access and timeline value order).
    std::mutex queue_mutex;
    Vulkan::MasterSemaphore semaphore;
    vk::UniqueCommandPool pool;
    std::array<vk::CommandBuffer, NumSlots> cmdbufs{};
    std::array<std::atomic<u64>, NumSlots> slot_values{};
    /// Recorded, not yet submitted (see AcquireSlot).
    std::array<std::atomic<bool>, NumSlots> slot_held{};
    std::array<vk::Semaphore, NumSlots> slot_wait_semaphores{};
    std::array<u64, NumSlots> slot_wait_values{};
    size_t next_slot = 0;
};

} // namespace VideoCore
