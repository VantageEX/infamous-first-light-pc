// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <atomic>
#include <cstdlib>
#include "common/perf_stats.h"
namespace iss::diagnostics {
enum Phase {GraphicsPipeline,ComputePipeline,Resources,Buffers,Textures,Vertices,Indices,RenderTargets,DynamicState,ObtainBuffer,SyncMemory,Draw,Dispatch,PhaseCount};
inline constexpr const char *Names[]={"graphics_pipeline","compute_pipeline","resources","buffers","textures","vertices","indices","render_targets","dynamic_state","obtain_buffer","sync_memory","draw","dispatch"};
struct Counters {std::array<std::atomic<u64>,PhaseCount> ns{},calls{};};
inline std::array<Counters,32> counters;
inline std::atomic<unsigned> counter_count{};
// Read once at load: a function-local static would check its guard at every scope.
inline const bool g_enabled=[] {const char *p=std::getenv("ISS_DEEP_PROFILE");return p&&p[0]=='1';}();
inline bool Enabled(){return g_enabled;}
class Scope {
    Counters *slot{};Phase phase;u64 start{};
public:
    explicit Scope(Phase p):phase(p){if(!Enabled())return;thread_local unsigned index=counter_count.fetch_add(1,std::memory_order_relaxed);if(index>=counters.size())return;slot=&counters[index];start=Common::Perf::NowNs();}
    ~Scope(){if(slot){const auto elapsed=Common::Perf::NowNs()-start;slot->ns[phase].store(slot->ns[phase].load(std::memory_order_relaxed)+elapsed,std::memory_order_relaxed);slot->calls[phase].store(slot->calls[phase].load(std::memory_order_relaxed)+1,std::memory_order_relaxed);}}
};
void Poll();
}
