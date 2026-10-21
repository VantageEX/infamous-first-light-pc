// SPDX-License-Identifier: GPL-2.0-or-later
#include <bit>
#include <cstring>
#include <malloc.h>
#include <memory>
#include <stdexcept>
#include <span>
#include <vector>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "infamous_renderer/host_feedback_buffer.h"
#include "infamous_renderer/generated_feedback_candidates.h"

namespace iss {
std::span<const std::uint32_t> NativeFeedbackShader();
// Isolated host allocation: never point this test at guest/game data. Use the
// renderer scheduler so submission and queue locking retain their normal owner.
void ProbeNativeFeedback(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler) {
    static bool attempted=false;
    if(attempted)return;
    attempted=true;
    const char* setting=std::getenv("ISS_NATIVE_FEEDBACK_PROBE");
    if(!setting||setting[0]!='1')return;
    if(!instance.HasExternalMemoryHost()){
        LOG_INFO(Render_Vulkan,"NATIVE_FEEDBACK unsupported external host memory");return;
    }
    try {
        static_assert(iss::game::feedback::FindAtomicCandidate(0x3EA96DB0)->known_output_stride_bytes == 64);
        static_assert(iss::game::feedback::FindAtomicCandidate(0x63E538C1)->known_output_stride_bytes == 80);
        static_assert(iss::game::feedback::FindAtomicCandidate(0x9C58AA06)->known_output_stride_bytes == 112);
        const auto device=instance.GetDevice();
        constexpr u64 logical_size=65536;
        auto native=iss::game::feedback::HostFeedbackBuffer::Create(instance,logical_size);
        if(!native){
            LOG_INFO(Render_Vulkan,"NATIVE_FEEDBACK no coherent compatible buffer memory type");return;
        }
        const auto buffer=native->Handle();
        const auto words=NativeFeedbackShader();
        auto shader_result=device.createShaderModuleUnique(vk::ShaderModuleCreateInfo{.codeSize=words.size_bytes(),.pCode=words.data()});
        if(shader_result.result!=vk::Result::eSuccess)throw std::runtime_error("probe shader failed");
        auto shader=std::move(shader_result.value);
        const vk::DescriptorSetLayoutBinding binding{.binding=0,.descriptorType=vk::DescriptorType::eStorageBuffer,
            .descriptorCount=1,.stageFlags=vk::ShaderStageFlagBits::eCompute};
        auto set_layout_result=device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{.bindingCount=1,.pBindings=&binding});
        if(set_layout_result.result!=vk::Result::eSuccess)throw std::runtime_error("probe descriptor layout failed");
        auto set_layout=std::move(set_layout_result.value);
        const auto raw_set_layout=*set_layout;
        auto layout_result=device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{.setLayoutCount=1,.pSetLayouts=&raw_set_layout});
        if(layout_result.result!=vk::Result::eSuccess)throw std::runtime_error("probe pipeline layout failed");
        auto layout=std::move(layout_result.value);
        auto pipeline_result=device.createComputePipelineUnique({},vk::ComputePipelineCreateInfo{
            .stage={.stage=vk::ShaderStageFlagBits::eCompute,.module=*shader,.pName="main"},.layout=*layout});
        if(pipeline_result.result!=vk::Result::eSuccess)throw std::runtime_error("probe compute pipeline failed");
        auto pipeline=std::move(pipeline_result.value);
        const vk::DescriptorPoolSize pool_size{.type=vk::DescriptorType::eStorageBuffer,.descriptorCount=1};
        auto pool_result=device.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{.maxSets=1,.poolSizeCount=1,.pPoolSizes=&pool_size});
        if(pool_result.result!=vk::Result::eSuccess)throw std::runtime_error("probe descriptor pool failed");
        auto pool=std::move(pool_result.value);
        auto sets=device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{.descriptorPool=*pool,.descriptorSetCount=1,.pSetLayouts=&raw_set_layout});
        if(sets.result!=vk::Result::eSuccess)throw std::runtime_error("probe descriptor set failed");
        const auto set=sets.value.front();
        const vk::DescriptorBufferInfo descriptor{.buffer=buffer,.offset=0,.range=logical_size};
        device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet=set,.dstBinding=0,.descriptorCount=1,
            .descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&descriptor},{});
        auto* bytes=native->Data();
        std::vector<unsigned char> expected(logical_size);
        scheduler.EndRendering();
        for(u32 round=0;round<16;++round){
            std::memset(bytes,0xa5,logical_size);
            for(u32 i=0;i<48;++i)bytes[128+i]=static_cast<unsigned char>(round*17+i);
            std::memcpy(expected.data(),bytes,logical_size);
            std::memcpy(expected.data()+256,bytes+128,48);
            const u32 pattern=0x12340000u+round;
            for(u32 i=0;i<48;i+=4)std::memcpy(expected.data()+512+i,&pattern,4);
            u32 cpu_value{};std::memcpy(&cpu_value,bytes+128,4);
            const u32 counter=pattern+cpu_value;std::memcpy(expected.data()+516,&counter,4);
            scheduler.Recorder().Record([handle=buffer,pattern,compute=*pipeline,pipeline_layout=*layout,set](vk::CommandBuffer command){
                const vk::MemoryBarrier before{.srcAccessMask=vk::AccessFlagBits::eHostWrite,
                    .dstAccessMask=vk::AccessFlagBits::eTransferRead|vk::AccessFlagBits::eTransferWrite};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                    vk::PipelineStageFlagBits::eTransfer,{},before,{},{});
                command.copyBuffer(handle,handle,vk::BufferCopy{.srcOffset=128,.dstOffset=256,.size=48});
                command.fillBuffer(handle,512,48,pattern);
                const vk::MemoryBarrier compute_barrier{
                    .srcAccessMask=vk::AccessFlagBits::eTransferWrite|vk::AccessFlagBits::eHostWrite,
                    .dstAccessMask=vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer|vk::PipelineStageFlagBits::eHost,
                    vk::PipelineStageFlagBits::eComputeShader,{},compute_barrier,{},{});
                command.bindPipeline(vk::PipelineBindPoint::eCompute,compute);
                command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,pipeline_layout,0,set,{});
                command.dispatch(1,1,1);
                const vk::MemoryBarrier after{.srcAccessMask=vk::AccessFlagBits::eTransferWrite|vk::AccessFlagBits::eShaderWrite,
                    .dstAccessMask=vk::AccessFlagBits::eHostRead};
                command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer|vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eHost,{},after,{},{});
            });
            scheduler.Finish();
            if(std::memcmp(bytes,expected.data(),logical_size)!=0){
                LOG_ERROR(Render_Vulkan,"NATIVE_FEEDBACK FAIL round {} including guard bytes",round);return;
            }
        }
        LOG_INFO(Render_Vulkan,"NATIVE_FEEDBACK PASS rounds=16 transfer_and_compute=1 record_bytes=48 checked_bytes=65536 coherent_type={} alignment={}",native->MemoryTypeIndex(),native->Alignment());
    } catch(const std::exception& error){
        LOG_ERROR(Render_Vulkan,"NATIVE_FEEDBACK failed: {}",error.what());
    }
}
}
