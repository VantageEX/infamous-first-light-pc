// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_native_bindings.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

static_assert(sizeof(Shader::PushData) <= NativePushIndexAddressOffset);

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() {
    // SHADPS4_ASYNC_COMPUTE_PIPELINES: a worker may not write into a destroyed pipeline. The
    // cache stops the pool (or waits for it to go idle) before destroying pipelines, so a job is
    // at most still running here; a queued one is no longer referenced by the pool (the RULE in
    // vk_pipeline_compile_pool.h).
    if (deferred) {
        for (u32 s = deferred->state.load(std::memory_order_acquire);
             s == DeferredPipeline::Running; s = deferred->state.load(std::memory_order_acquire)) {
            deferred->state.wait(s, std::memory_order_acquire);
        }
    }
}

void Pipeline::WaitDeferred() const noexcept {
    deferred->Wait();
}

bool Pipeline::CancelDeferred() noexcept {
    return deferred && deferred->TryCancel();
}

void Pipeline::PromoteDeferredSlow() const noexcept {
    if (deferred->preload && !deferred->IsDone()) {
        PipelineCompilePool::Promote(deferred.get());
    }
}

void Pipeline::BeginBufferBarriers(const BufferBarriers& buffer_barriers) const {
    // RecordBufferBarriers without its record (kept separate so the legacy path, including the
    // batch timer's end-of-rendering site, stays byte-identical).
    if (buffer_barriers.empty()) {
        return;
    }
    if (GpuBatchTimer::Enabled()) {
        GpuBreaks().buffer_barrier_calls.fetch_add(1, std::memory_order_relaxed);
        GpuBreaks().buffer_barriers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
    }
    scheduler.EndRendering();
}

void Pipeline::RecordBufferBarriers(const BufferBarriers& buffer_barriers) const {
    if (buffer_barriers.empty()) {
        return;
    }
    if (GpuBatchTimer::Enabled()) {
        GpuBreaks().buffer_barrier_calls.fetch_add(1, std::memory_order_relaxed);
        GpuBreaks().buffer_barriers.fetch_add(buffer_barriers.size(), std::memory_order_relaxed);
    }
    scheduler.EndRendering();
    scheduler.RecordArrays(
        [](vk::CommandBuffer cmdbuf, auto barriers) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = u32(barriers.size()),
                .pBufferMemoryBarriers = barriers.data(),
            });
        },
        std::span<const vk::BufferMemoryBarrier2>{buffer_barriers});
}

void Pipeline::BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const {
    ASSERT_MSG(!uses_native_bindings, "Legacy descriptor emission on a descriptor heap pipeline");
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    RecordBufferBarriers(buffer_barriers);

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const auto layout = *pipeline_layout;
    scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });

    // Bind descriptor set.
    if (sparse_dead_mask != 0) {
        // q3: no writes to the bindings the sparse layout dropped (VUID dstBinding-00316).
        std::erase_if(set_writes, [this](const vk::WriteDescriptorSet& w) {
            return w.dstBinding < 64 && ((sparse_dead_mask >> w.dstBinding) & 1) != 0;
        });
    }
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        scheduler.PushDescriptors(bind_point, layout, set_writes);
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.InvalidateNativeHeaps();
    scheduler.Record([bind_point, layout, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

vk::DescriptorSet Pipeline::CommitSetWrites(std::span<vk::WriteDescriptorSet> writes) const {
    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& w : writes) {
        w.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(writes, {});
    return desc_set;
}

void Pipeline::BindResourcesNative(NativeBindings& native, std::span<const u32> indices,
                                   const BufferBarriers& buffer_barriers,
                                   const Shader::PushData& push_data) const {
    ASSERT(uses_native_bindings && indices.size() == native_binding_count);

    RecordBufferBarriers(buffer_barriers);
    native.BindHeapsIfNeeded();

    if (uses_direct_push_index) {
        native.CommitDirect(IsCompute());
        const u32 data_size = NativePushIndexAddressOffset + u32(indices.size_bytes());
        std::array<u8, NativeDirectPushBlockCapacity> block{};
        std::memcpy(block.data(), &push_data, sizeof(push_data));
        if (!indices.empty()) {
            std::memcpy(block.data() + NativePushIndexAddressOffset, indices.data(),
                        indices.size_bytes());
        }
        scheduler.RecordArrays(
            [push = instance.GetNativeApi().cmd_push_data](vk::CommandBuffer cmdbuf,
                                                            std::span<const u8> data) {
                const NativeApi::PushDataInfo info = {
                    .sType = NativeApi::StructureTypePushDataInfo,
                    .offset = 0,
                    .data = {data.data(), data.size()},
                };
                push(cmdbuf, &info);
            },
            std::span<const u8>{block.data(), data_size});
        return;
    }

    // The index array is owned by the native ring until the GPU completes this submission,
    // not by the recording worker's chunk.
    const u64 index_address = native.CommitIndices(indices, IsCompute());
    std::array<u8, NativePushBlockSize> block{};
    std::memcpy(block.data(), &push_data, sizeof(push_data));
    std::memcpy(block.data() + NativePushIndexAddressOffset, &index_address,
                sizeof(index_address));
    scheduler.Record(
        [push = instance.GetNativeApi().cmd_push_data, block](vk::CommandBuffer cmdbuf) {
            const NativeApi::PushDataInfo info = {
                .sType = NativeApi::StructureTypePushDataInfo,
                .offset = 0,
                .data = {block.data(), block.size()},
            };
            push(cmdbuf, &info);
        });
}

void Pipeline::BuildPacketBindings(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    packet_bindings.clear();
    packet_bindings.reserve(bindings.size());
    packet_buffer_infos = 0;
    packet_image_infos = 0;
    for (const auto& binding : bindings) {
        if (IsPacketBufferType(binding.descriptorType)) {
            packet_buffer_infos += binding.descriptorCount;
        } else {
            packet_image_infos += binding.descriptorCount;
        }
        packet_bindings.push_back(PacketBinding{
            .binding = static_cast<u16>(binding.binding),
            .count = static_cast<u16>(binding.descriptorCount),
            .type = static_cast<u32>(binding.descriptorType),
        });
    }
}

void Pipeline::BuildNativeMappings(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    const auto& heap = instance.GetNativeBindingLayout();
    uses_native_bindings = true;
    native_binding_count = 0;
    native_stage_mappings.clear();

    bool all_scalar = true;
    for (const auto& binding : bindings) {
        native_binding_count =
            std::max(native_binding_count, binding.binding + binding.descriptorCount);
        all_scalar &= binding.descriptorCount == 1;
    }
    uses_direct_push_index =
        all_scalar && native_binding_count <= NativeDirectPushBindingCapacity &&
        NativePushIndexAddressOffset + native_binding_count * sizeof(u32) <=
            heap.max_push_data_size;

    for (const auto& binding : bindings) {
        auto it = std::ranges::find_if(native_stage_mappings, [&](const auto& entry) {
            return entry.stage == binding.stageFlags;
        });
        if (it == native_stage_mappings.end()) {
            it = native_stage_mappings.insert(it, NativeStageMappings{.stage = binding.stageFlags});
        }

        u32 base{};
        u32 stride{};
        switch (binding.descriptorType) {
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBuffer:
            base = heap.buffer_base;
            stride = heap.buffer_stride;
            break;
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
            base = heap.image_base;
            stride = heap.image_stride;
            break;
        case vk::DescriptorType::eSampler:
            base = heap.sampler_base;
            stride = heap.sampler_stride;
            break;
        default:
            UNREACHABLE_MSG("Unexpected guest descriptor type {}",
                            vk::to_string(binding.descriptorType));
        }

        // Element b of the per-draw index array holds the heap slot of unified binding b.
        const u32 address_offset = binding.binding * sizeof(u32);
        NativeApi::DescriptorSetAndBindingMapping mapping = {
            .sType = NativeApi::StructureTypeDescriptorSetAndBindingMapping,
            .descriptorSet = 0,
            .firstBinding = binding.binding,
            .bindingCount = 1,
            .resourceMask = NativeApi::SpirvResourceTypeAll,
        };
        if (uses_direct_push_index) {
            mapping.source = NativeApi::MappingSourceHeapWithPushIndex;
            mapping.sourceData.pushIndex = {
                .heapOffset = base,
                .pushOffset = NativePushIndexAddressOffset + address_offset,
                .heapIndexStride = stride,
                .heapArrayStride = stride,
            };
        } else if (binding.descriptorCount > 1) {
            mapping.source = NativeApi::MappingSourceHeapWithIndirectIndexArray;
            mapping.sourceData.indirectIndexArray = {
                .heapOffset = base,
                .pushOffset = NativePushIndexAddressOffset,
                .addressOffset = address_offset,
                .heapIndexStride = stride,
            };
        } else {
            mapping.source = NativeApi::MappingSourceHeapWithIndirectIndex;
            mapping.sourceData.indirectIndex = {
                .heapOffset = base,
                .pushOffset = NativePushIndexAddressOffset,
                .addressOffset = address_offset,
                .heapIndexStride = stride,
                .heapArrayStride = stride,
            };
            // Pure sampler bindings use these primary fields too. The sampler-prefixed
            // fields are used only for combined image/sampler shader variables.
        }
        it->mappings.push_back(mapping);
    }
    ASSERT_MSG(native_binding_count <= NativeMaxBindingsPerDraw,
               "Pipeline declares {} bindings, native limit is {}", native_binding_count,
               NativeMaxBindingsPerDraw);
}

const NativeApi::ShaderDescriptorSetAndBindingMappingInfo* Pipeline::NativeMappingInfo(
    vk::ShaderStageFlagBits stage, const void* next) const {
    for (const auto& entry : native_stage_mappings) {
        if (entry.stage != vk::ShaderStageFlags{stage}) {
            continue;
        }
        entry.info = {
            .sType = NativeApi::StructureTypeShaderDescriptorSetAndBindingMappingInfo,
            .pNext = next,
            .mappingCount = u32(entry.mappings.size()),
            .pMappings = entry.mappings.data(),
        };
        return &entry.info;
    }
    return nullptr;
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
