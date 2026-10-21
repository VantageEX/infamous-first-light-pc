// SPDX-License-Identifier: GPL-2.0-or-later
// Passive, bounded join. A binding is a possible writer; publication sequence is NOT an
// allocation/arena generation. No observation from this module authorizes a memory shortcut.
#include "feedback_join.h"
#include "infamous_renderer/feedback_identity_registry.h"
#include "common/perf_stats.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace iss::feedback_join {
const bool enabled = [] { const char* p = std::getenv("ISS_FEEDBACK_JOIN"); return p && p[0]=='1'; }();
namespace {
struct Candidate {
    u64 sequence{}, address{}, physical{}, hash{}, tick{}, epoch{}, buffer_generation{};
    u64 bind_ns{}, enqueue_ns{};
    u32 frame{}, stage{}, qid{}, gpu_queue{};
    game::feedback::IdentityObservation identity{};
    Vulkan::Scheduler* scheduler{};
};
std::array<Candidate, 16384> history{};
std::mutex mutex;
u64 sequence{}, tickets{}, rows{};
u64 outstanding{};
bool sealed{};
struct Event {
    const char* name{};
    ReadTicket ticket{};
    Candidate candidate{};
    u64 ns{}, submitted{}, completed{}, latest{};
    u64 frame{};
    bool handled{};
};
std::array<Event, 32768> events{};
thread_local std::array<u64, 64> pending{};
thread_local std::size_t pending_count{};
u64 Number(const char* name, u64 fallback) {
    const char* p=std::getenv(name); return p ? std::strtoull(p,nullptr,0) : fallback;
}
u64 FirstFrame() { static const auto n=Number("ISS_FEEDBACK_JOIN_START_FRAME",1600);return n; }
u64 FrameCount() { static const auto n=Number("ISS_FEEDBACK_JOIN_FRAMES",60);return n; }
bool Active(bool preroll=false) {
    const auto first=FirstFrame();
    const auto begin=preroll && first>=4?first-4:first;
    const auto frame=Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    return frame>=begin && (frame<first || frame-first<FrameCount());
}
Candidate Latest(u64 record, u64 physical, u64 epoch) {
    Candidate result{};
    for (const auto& c:history) {
        if(c.sequence>result.sequence && c.epoch==epoch &&
           (c.address==record || (physical && c.physical==physical))) result=c;
    }
    return result;
}
void Row(const char* event, const ReadTicket& t, const Candidate& c, bool handled,
         u64 latest_sequence = 0) {
    if(rows>=32768) return;
    if(sealed) return;
    events[rows++] = {event,t,c,Common::Perf::NowNs(),
        c.scheduler?c.scheduler->SubmittedTick():0,
        c.scheduler?c.scheduler->GetWorkSemaphore()->KnownGpuTick():0,
        latest_sequence,Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed),handled};
}
void FlushAfterWindow() {
    const auto frame=Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    if(frame<FirstFrame() || frame-FirstFrame()<FrameCount())return;
    u64 count{};
    {
        std::scoped_lock lock{mutex};
        if(sealed || outstanding)return;
        sealed=true;
        count=rows;
    }
    // Only the CP calls this after the window and after every read_end. No I/O
    // in the exception handler or under the join lock. Sealed events are immutable.
    auto* file=[] {
        const char* p=std::getenv("ISS_FEEDBACK_JOIN_LOG");
        auto* f=p?std::fopen(p,"w"):nullptr;
        if(f) { std::setvbuf(f,nullptr,_IOFBF,1<<20);
            std::fputs("event,ns,frame,ticket,rip,fault_address,consumer_owner,consumer_record,consumer_physical,read_epoch,read_buffer_generation,before_sequence,writer_sequence,writer_address,writer_physical,shader_hash,stage,engine_qid,gpu_queue,writer_tick,submitted_tick,known_complete_tick,writer_epoch,writer_buffer_generation,bind_ns,enqueue_ns,identity_owner,identity_shader,identity_sequence,identity_reuses,handled,latest_sequence\n",f); }
        return f;
    }();
    if(!file)return;
    for(u64 i=0;i<count;++i) {
    const auto& e=events[i];const auto& t=e.ticket;const auto& c=e.candidate;
    std::fprintf(file,
        "%s,%llu,%llu,%llu,0x%llx,0x%llx,0x%llx,0x%llx,0x%llx,%llu,%llu,%llu,%llu,0x%llx,0x%llx,0x%llx,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,0x%llx,0x%llx,%llu,%llu,%u,%llu\n",
        e.name,e.ns,e.frame,
        t.id,t.rip,t.address,t.owner,t.record,t.physical,t.epoch,t.buffer_generation,t.before_sequence,
        c.sequence,c.address,c.physical,c.hash,c.stage,c.qid,c.gpu_queue,c.tick,e.submitted,e.completed,
        c.epoch,c.buffer_generation,c.bind_ns,c.enqueue_ns,c.identity.owner_object,c.identity.shader_object,
        c.identity.sequence,c.identity.address_reuses,unsigned(e.handled),e.latest);
    }
    std::fclose(file);
}
}
void BeginBindings() { if(enabled) {pending_count=0;FlushAfterWindow();} }
void Binding(u64 address,u64 hash,u32 stage,u32 qid,u32 gpu_queue,
             u64 buffer_generation,Vulkan::Scheduler& scheduler) {
    if(!enabled || !Active(true))return;
    const auto identity=game::feedback::IdentityRegistry::Instance().Latest(address);
    auto* memory=Core::Memory::Instance();
    const auto epoch=memory->MappingEpoch();
    const auto backing=memory->DescribeBacking(address,48);
    Candidate candidate{};
    candidate.address=address;candidate.hash=hash;candidate.tick=scheduler.CurrentTick();
    candidate.physical=backing&&backing->size==48?backing->physical_start:0;
    candidate.epoch=epoch;candidate.buffer_generation=buffer_generation;
    candidate.bind_ns=Common::Perf::NowNs();
    candidate.frame=Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    candidate.stage=stage;candidate.qid=qid;candidate.gpu_queue=gpu_queue;candidate.scheduler=&scheduler;
    if(identity)candidate.identity=*identity;
    std::scoped_lock lock{mutex};
    candidate.sequence=++sequence;
    history[candidate.sequence%history.size()]=candidate;
    if(pending_count<pending.size())pending[pending_count++]=candidate.sequence;
}
void Recorded() {
    if(!enabled || !pending_count)return;
    const auto now=Common::Perf::NowNs();
    const bool active=Active();
    std::scoped_lock lock{mutex};
    for(std::size_t i=0;i<pending_count;++i) {
        auto& c=history[pending[i]%history.size()];
        if(c.sequence!=pending[i])continue;
        // The threaded recorder may not have executed vkCmd* yet. This is only
        // enqueue confirmation, never a Vulkan recording/submission/completion ACK.
        c.enqueue_ns=now;
        if(active)Row("writer",{},c,false);
    }
    pending_count=0;
}
ReadTicket BeginRead(u64 rip,u64 address,u64 owner,u64 record,u64 buffer_generation) {
    if(!enabled || !Active())return {};
    static const auto base=Number("ISS_FEEDBACK_JOIN_EBOOT_BASE",0x10000000);
    if(!((rip==base+0x3f3eff && address==record+4) ||
         (rip==base+0x3f3fcc && address==record+0x28)))return {};
    auto* memory=Core::Memory::Instance();
    const auto epoch=memory->MappingEpoch();
    const auto backing=memory->DescribeBacking(record,48);
    const auto physical=backing&&backing->size==48?backing->physical_start:0;
    std::scoped_lock lock{mutex};
    const auto c=Latest(record,physical,epoch);
    ReadTicket t{++tickets,rip,address,owner,record,physical,epoch,buffer_generation,c.sequence};
    ++outstanding;
    Row("read_begin",t,c,false);
    return t;
}
void EndRead(const ReadTicket& original,bool handled,u64 buffer_generation) {
    if(!enabled || !original.id)return;
    auto t=original;
    t.epoch=Core::Memory::Instance()->MappingEpoch();t.buffer_generation=buffer_generation;
    std::scoped_lock lock{mutex};
    const auto& entry=history[t.before_sequence%history.size()];
    const auto frozen=entry.sequence==t.before_sequence?entry:Candidate{};
    Row("read_end",t,frozen,handled,Latest(t.record,t.physical,t.epoch).sequence);
    --outstanding;
}
}
