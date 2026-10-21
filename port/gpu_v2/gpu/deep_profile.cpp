// SPDX-License-Identifier: GPL-2.0-or-later
#include <atomic>
#include "deep_profile.h"
#include "core/debug_state.h"
#include <cstdio>
#include <mutex>
namespace iss::diagnostics {
namespace {
struct Site {std::atomic<u64> key{},ns{},calls{},maximum{},address{};};
std::array<Site,128> sites;
struct FeedbackBinding {u64 address{},hash{},tick{},sequence{};};
std::array<FeedbackBinding,8192> feedback_bindings;
std::mutex feedback_mutex;
u64 feedback_sequence{};
bool TraceFeedback(){static const bool value=[] {const char *v=std::getenv("ISS_FEEDBACK_TRACE");return v&&v[0]=='1';}();return value;}
void TraceFeedbackRead(u64 rip,u64 address,u64 elapsed){
    if(!TraceFeedback())return;
    // Bounded diagnostic history. Bindings are candidates, not proof of GPU execution.
    std::scoped_lock guard{feedback_mutex};
    FeedbackBinding match{};
    for(const auto& b:feedback_bindings)if(b.sequence>match.sequence&&address>=b.address&&address-b.address<48)match=b;
    if(!match.sequence)return;
    static std::FILE *file=[] {const char *p=std::getenv("ISS_FEEDBACK_LOG");auto *f=p?std::fopen(p,"w"):nullptr;if(f)std::fputs("time_ns,rip,address,binding_address,shader_hash,scheduler_tick,binding_sequence,elapsed_ms\n",f);return f;}();
    if(file){std::fprintf(file,"%llu,0x%llx,0x%llx,0x%llx,0x%llx,%llu,%llu,%.6f\n",(unsigned long long)Common::Perf::NowNs(),(unsigned long long)rip,(unsigned long long)address,(unsigned long long)match.address,(unsigned long long)match.hash,(unsigned long long)match.tick,(unsigned long long)match.sequence,double(elapsed)/1e6);std::fflush(file);}
}
}
void RecordFeedbackBinding(u64 address,u64 size,u64 hash,u64 tick){
    if(size!=48||!TraceFeedback())return;
    std::scoped_lock guard{feedback_mutex};
    const auto sequence=++feedback_sequence;
    feedback_bindings[sequence%feedback_bindings.size()]={address,hash,tick,sequence};
}
void RecordReadbackTime(u64 rip,u64 address,u64 elapsed,unsigned kind){
    if(kind==0)TraceFeedbackRead(rip,address,elapsed);
    if(!Enabled())return;const u64 key=(rip<<2)|(kind+1);const unsigned first=kind==1?64:0;
    for(unsigned i=first;i<first+64;++i){auto& site=sites[i];auto found=site.key.load(std::memory_order_relaxed);
        if(!found){u64 empty=0;site.key.compare_exchange_strong(empty,key,std::memory_order_relaxed);found=site.key.load(std::memory_order_relaxed);}
        if(found!=key)continue;site.ns.fetch_add(elapsed,std::memory_order_relaxed);site.calls.fetch_add(1,std::memory_order_relaxed);site.address.store(address,std::memory_order_relaxed);
        auto maximum=site.maximum.load(std::memory_order_relaxed);while(maximum<elapsed&&!site.maximum.compare_exchange_weak(maximum,elapsed,std::memory_order_relaxed)){}return;}
}
void Poll(){
    if(!Enabled())return;static std::FILE *file=nullptr;static bool initialized=false;
    static u64 last{},last_flips{};static std::array<u64,PhaseCount> previous_ns{},previous_calls{};
    static constexpr Common::Perf::Counter extra[]={Common::Perf::ReadbackProducerDone,Common::Perf::ReadbackProducerSubmitted,Common::Perf::ReadbackProducerQueued,Common::Perf::ReadbackProducerCurrent,Common::Perf::ReadbackQueueReads,Common::Perf::ReadbackQueueFallbacks,Common::Perf::GpuProfileComputeNs,Common::Perf::GpuProfileInFlight,Common::Perf::AscStarts,Common::Perf::AscStartDelayNs,Common::Perf::AscBusyNs,Common::Perf::GfxYields};
    static constexpr const char *extra_names[]={"producer_done","producer_submitted","producer_queued","producer_current","queue_reads","queue_fallbacks","gpu_compute_ns","gpu_pending_sum","compute_starts","compute_start_delay_ns","compute_cpu_ns","graphics_yields"};
    static std::array<u64,std::size(extra)> previous_extra{};
    static std::FILE *site_file=nullptr;static std::array<u64,128> site_ns{},site_calls{};
    if(!initialized){initialized=true;const char *path=std::getenv("ISS_DEEP_LOG");if(path)file=std::fopen(path,"w");
        if(file){std::fputs("time_ns,interval_ms,flips",file);for(auto name:Names)std::fprintf(file,",%s_ms,%s_calls",name,name);for(auto name:extra_names)std::fprintf(file,",%s",name);std::fputc('\n',file);}
        const char *site_path=std::getenv("ISS_ACCESS_TIMES_LOG");if(site_path)site_file=std::fopen(site_path,"w");if(site_file)std::fputs("time_ns,kind,rip,address,calls,total_ms,max_ms\n",site_file);last=Common::Perf::NowNs();}
    if(!file)return;const auto now=Common::Perf::NowNs();if(now-last<1000000000)return;
    std::array<u64,PhaseCount> ns{},calls{};
    for(unsigned t=0;t<std::min<unsigned>(counter_count.load(),unsigned(counters.size()));++t)for(unsigned p=0;p<PhaseCount;++p){ns[p]+=counters[t].ns[p].load(std::memory_order_relaxed);calls[p]+=counters[t].calls[p].load(std::memory_order_relaxed);}
    const auto flips=DebugState.flip_count.load();std::fprintf(file,"%llu,%.3f,%llu",(unsigned long long)now,double(now-last)/1e6,(unsigned long long)(flips-last_flips));
    for(unsigned p=0;p<PhaseCount;++p){std::fprintf(file,",%.6f,%llu",double(ns[p]-previous_ns[p])/1e6,(unsigned long long)(calls[p]-previous_calls[p]));previous_ns[p]=ns[p];previous_calls[p]=calls[p];}
    for(unsigned i=0;i<std::size(extra);++i){const auto value=Common::Perf::Total(extra[i]);std::fprintf(file,",%llu",(unsigned long long)(value-previous_extra[i]));previous_extra[i]=value;}
    std::fputc('\n',file);std::fflush(file);last=now;last_flips=flips;
    if(site_file){for(unsigned i=0;i<sites.size();++i){auto& s=sites[i];const auto key=s.key.load(),n=s.calls.load(),time=s.ns.load();if(!key||n==site_calls[i])continue;
        std::fprintf(site_file,"%llu,%u,0x%llx,0x%llx,%llu,%.6f,%.6f\n",(unsigned long long)now,unsigned((key&3)-1),(unsigned long long)(key>>2),(unsigned long long)s.address.load(),(unsigned long long)(n-site_calls[i]),double(time-site_ns[i])/1e6,double(s.maximum.load())/1e6);site_calls[i]=n;site_ns[i]=time;}
        std::fflush(site_file);}
}
}
