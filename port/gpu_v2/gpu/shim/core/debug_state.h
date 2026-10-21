// Adapted from bloodborne_pc@4d4d20812c0af23a11c8b8d40f5e06a9054925ef, GPL-2.0-or-later.
// bbport: shadPS4's frame/register dump tooling is not part of this port.
#pragma once
#include <span>
#include <atomic>
#include <string>
#include "common/types.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace DebugStateType {
enum class QueueType { dcb = 0, ccb = 1, acb = 2 };
struct QueueDump {
    QueueType type;
    u32 submit_num;
    u32 num2;
    std::vector<u32> data;
    uintptr_t base_addr;
};
class DebugStateImpl {
public:
    using CsState = AmdGpu::ComputeProgram;
    bool is_using_fsr{};
    std::pair<u32,u32> game_resolution{},output_resolution{};
    std::atomic<u64> draws{},dispatches{};
    std::atomic<u64> flip_count{};
    u64 frame_draws{},frame_dispatches{};
    void IncFlipFrameNum() {flip_count.fetch_add(1,std::memory_order_relaxed);frame_draws=draws.exchange(0,std::memory_order_relaxed);frame_dispatches=dispatches.exchange(0,std::memory_order_relaxed);}
    void IncDrawCall() noexcept {draws.fetch_add(1,std::memory_order_relaxed);}
    void IncDispatch() noexcept {dispatches.fetch_add(1,std::memory_order_relaxed);}
    u64 GetDrawCallsAvg() const {return frame_draws;}
    u64 GetDispatchesAvg() const {return frame_dispatches;}
    void IncGnmFrameNum() {}
    u32 GetFrameNum() const { return 0; }
    bool DumpingCurrentFrame() const { return false; }
    bool DumpingCurrentReg() { return false; }
    bool ShouldPauseInSubmit() const { return false; }
    void PushQueueDump(QueueDump) {}
    void PushRegsDump(uintptr_t, uintptr_t, const AmdGpu::Regs&) {}
    void PushRegsDumpCompute(uintptr_t, uintptr_t, const CsState&) {}
    void CollectShader(const std::string&, Shader::SwStage, vk::ShaderModule, std::span<const u32>,
                       std::span<const u32>, std::span<const u32>, bool) {}
    void ShowDebugMessage(std::string) {}
    bool IsGuestThreadsPaused() const { return false; }
    void PauseGuestThreads() {}
    void ResumeGuestThreads() {}
};
} // namespace DebugStateType
inline DebugStateType::DebugStateImpl DebugStateInstance;
#define DebugState DebugStateInstance
