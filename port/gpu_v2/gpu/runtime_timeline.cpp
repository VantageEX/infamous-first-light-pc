// SPDX-License-Identifier: GPL-2.0-or-later
#include <atomic>
#include "runtime_timeline.h"
#include "timeline_buffer.h"
#include "common/perf_stats.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace iss::timeline {
const bool enabled=[] {const char* p=std::getenv("ISS_TIMELINE");return p&&p[0]=='1';}();
namespace {
u64 ThreadId(){
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
}
struct Event {
    const char* event{};
    u64 ns{},frame{},thread{},owner{},tick{},a{},b{},c{},d{},duration{};
};
struct Capture {
    Buffer<Event,65536> buffer;
    std::atomic<bool> accepting{false};
    std::FILE* file{};
    u64 rows{},bytes{};
    u64 max_rows=2'000'000,max_bytes=384ull*1024*1024;
    // ISS_TIMELINE_HITCH_MS=N: hitch recorder. Events stay in memory for the last
    // ISS_TIMELINE_HITCH_KEEP_MS (default 2500) and only reach the file when a present follows the
    // previous one by more than N ms: a "hitch" row, the kept window, then 1 s more. Runs for hours.
    u64 hitch_ns{},keep_ns{2'500'000'000ull},last_present{},write_until{},hitches{};
    std::deque<Event> kept;
    Capture(){
        const char* path=std::getenv("ISS_TIMELINE_LOG");
        file=path?std::fopen(path,"wb"):nullptr;
        if(!file){std::fputs("TIMELINE disabled: output unavailable\n",stderr);return;}
        std::setvbuf(file,nullptr,_IOFBF,1<<20);
        if(const char* h=std::getenv("ISS_TIMELINE_HITCH_MS");h&&std::atoi(h)>0){
            hitch_ns=u64(std::atoi(h))*1'000'000ull;
            if(const char* k=std::getenv("ISS_TIMELINE_HITCH_KEEP_MS");k&&std::atoi(k)>0)keep_ns=u64(std::atoi(k))*1'000'000ull;
            max_rows=50'000'000;max_bytes=4ull*1024*1024*1024;
            std::fprintf(stderr,"TIMELINE hitch recorder: > %llu ms, keeps %llu ms\n",
                         (unsigned long long)(hitch_ns/1'000'000),(unsigned long long)(keep_ns/1'000'000));
        }
        std::fputs("event,cpu_ns,observed_guest_frame,thread_id,owner,tick,a,b,c,d,duration_ns\n",file);
        accepting=true;
        // Process-lifetime diagnostic writer. No destructor joins during ExitProcess.
        std::thread([this]{WriteLoop();}).detach();
    }
    void Write(const Event& e){
        const int n=std::fprintf(file,"%s,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
            e.event,e.ns,e.frame,e.thread,e.owner,e.tick,e.a,e.b,e.c,e.d,e.duration);
        if(n>0)bytes+=n;
        ++rows;
    }
    void Hitch(const Event& e){
        if(std::strcmp(e.event,"present_end")==0){
            if(last_present&&e.ns>last_present+hitch_ns){
                Write({"hitch",e.ns,e.frame,e.thread,0,0,(e.ns-last_present)/1000,++hitches,kept.size(),0,0});
                for(const auto& k:kept)Write(k);
                kept.clear();
                write_until=e.ns+1'000'000'000ull;
            }
            last_present=e.ns;
        }
        if(e.ns<=write_until){Write(e);return;}
        kept.push_back(e);
        while(!kept.empty()&&kept.front().ns+keep_ns<e.ns)kept.pop_front();
    }
    void WriteLoop(){
#ifdef _WIN32
        SetThreadDescription(GetCurrentThread(),L"ISS:TimelineWriter");
#endif
        std::array<Event,2048> batch;
        u64 health=0;
        // ISS_TIMELINE_START_FRAME=N: events of earlier guest frames are dropped (a full trace of
        // the gameplay instead of the loading screens).
        const u64 start_frame=[]{const char* p=std::getenv("ISS_TIMELINE_START_FRAME");return p?std::strtoull(p,nullptr,10):0ull;}();
        for(;;){
            auto n=buffer.Take(batch);
            for(std::size_t i=0;i<n;++i){
                if(batch[i].frame<start_frame)continue;
                if(rows>=max_rows||bytes>=max_bytes){
                    accepting=false;
                    Write({"trace_truncated",Common::Perf::NowNs(),0,ThreadId(),0,0,rows,bytes,buffer.Dropped()});
                    std::fclose(file);return;
                }
                if(hitch_ns)Hitch(batch[i]);
                else Write(batch[i]);
            }
            const auto now=Common::Perf::NowNs();
            if(now-health>=1'000'000'000ull){
                Write({"trace_health",now,0,ThreadId(),0,0,rows,bytes,buffer.Dropped()});health=now;
            }
            if(std::ferror(file)){accepting=false;std::fclose(file);return;}
            std::fflush(file);
            if(n<batch.size())std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
Capture& State(){static auto* capture=new Capture;return *capture;}
}
void Emit(const char* event,u64 owner,u64 tick,u64 a,u64 b,u64 c,u64 d,u64 duration_ns){
    if(!enabled)return;
    auto& state=State();if(!state.accepting.load(std::memory_order_relaxed))return;
    state.buffer.Push({event,Common::Perf::NowNs(),
        Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed),ThreadId(),
        owner,tick,a,b,c,d,duration_ns});
}
}
