// SPDX-License-Identifier: GPL-2.0-or-later
// Ordered SPSC recording queue. Design adapted from the read-only Bloodborne
// reference's vk_draw_pipe.h; fences stay on the original CP after a full drain.
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <immintrin.h>
#include "common/assert.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "common/types.h"

namespace iss {
class DrawPipe {
public:
    using Handler=void(*)(void*,const u8*,u32);
    enum Reason:u32 { Packet,Commands,Submission,Yield,Fault,Resource,Fence,Indirect,Shutdown,NumReasons };
    static constexpr u64 Capacity=16ull<<20;
    static constexpr u32 MaxPayload=static_cast<u32>(Capacity/4-64);
    DrawPipe(Handler fn,void* data):handler(fn),context(data),ring(std::make_unique<u8[]>(Capacity)) {
        worker=std::jthread([this](std::stop_token stop){Run(stop);});
    }
    ~DrawPipe(){Drain(Shutdown);worker.request_stop();Wake();worker.join();}
    static bool OnWorker() noexcept {return on_worker;}
    bool Idle() const noexcept {return consumed.load(std::memory_order_acquire)==published.load(std::memory_order_acquire);}
    u8* Begin(u32 bytes){
        ASSERT(bytes<=MaxPayload);const u64 size=(u64(bytes)+sizeof(Header)+63)&~u64(63);
        const u64 offset=head%Capacity;
        if(offset+size>Capacity){
            WaitSpace(head,Capacity-offset);reinterpret_cast<Header*>(ring.get()+offset)->size=0;
            head+=Capacity-offset;published.store(head,std::memory_order_seq_cst);Wake();
        }
        WaitSpace(head,size);pending=static_cast<u32>(size);
        auto* h=reinterpret_cast<Header*>(ring.get()+head%Capacity);h->size=pending;h->bytes=bytes;
        return reinterpret_cast<u8*>(h+1);
    }
    void Commit(){head+=pending;++packets;published.store(head,std::memory_order_seq_cst);if(sleeping.load(std::memory_order_seq_cst))Wake();}
    void Drain(u32 reason=Resource){
        u64 done=consumed.load(std::memory_order_acquire);if(done==head)return;
        ++drains;const auto start=Common::Perf::NowNs();
        for(u32 spins=0;done!=head;++spins){
            if(spins<4096) _mm_pause();
            else WaitProgress(done);
            done=consumed.load(std::memory_order_acquire);
        }
        const auto ns=Common::Perf::NowNs()-start;wait_ns+=ns;
        if(reason<NumReasons){++reason_count[reason];reason_ns[reason]+=ns;}
    }
    u64 packets{},drains{},wait_ns{};
    std::array<u64,NumReasons> reason_count{},reason_ns{};
    std::atomic<u64> busy_ns{},completed_packets{};
private:
    struct Header{u32 size,bytes;};
    void Wake(){wake.fetch_add(1,std::memory_order_seq_cst);wake.notify_one();}
    void WaitProgress(u64 seen){
        producer_waiting.store(true,std::memory_order_seq_cst);
        if(consumed.load(std::memory_order_seq_cst)==seen)consumed.wait(seen,std::memory_order_seq_cst);
        producer_waiting.store(false,std::memory_order_seq_cst);
    }
    void WaitSpace(u64 pos,u64 size){
        for(;;){auto tail=consumed.load(std::memory_order_acquire);if(pos+size-tail<=Capacity)return;WaitProgress(tail);}
    }
    void Run(std::stop_token stop){
        Common::SetCurrentThreadName("iss:DrawStageB");on_worker=true;u64 at=0;
        const auto spin_time=std::chrono::microseconds([] {
            if(const char* p=std::getenv("ISS_DRAW_PIPE_SPIN_US"))return std::clamp(std::atoi(p),0,1000);
            return std::thread::hardware_concurrency()>=12?200:50;
        }());
        for(;;){
            u64 end=published.load(std::memory_order_acquire);
            if(at==end){
                const auto until=std::chrono::steady_clock::now()+spin_time;
                for(u32 n=1;at==end;++n){
                    _mm_pause();end=published.load(std::memory_order_acquire);
                    if(!(n&255) && std::chrono::steady_clock::now()>=until)break;
                }
                if(at==end){
                    const auto seen=wake.load(std::memory_order_seq_cst);
                    // SC publication/flag/recheck closes the sleep-vs-publish lost-wake race.
                    sleeping.store(true,std::memory_order_seq_cst);
                    if(stop.stop_requested())break;
                    if(published.load(std::memory_order_seq_cst)==at)wake.wait(seen,std::memory_order_seq_cst);
                    sleeping.store(false,std::memory_order_seq_cst);
                    continue;
                }
            }
            const auto* h=reinterpret_cast<const Header*>(ring.get()+at%Capacity);
            if(!h->size)at+=Capacity-at%Capacity;
            else{
                const auto start=Common::Perf::NowNs();handler(context,reinterpret_cast<const u8*>(h+1),h->bytes);
                busy_ns.fetch_add(Common::Perf::NowNs()-start,std::memory_order_relaxed);
                at+=h->size;completed_packets.fetch_add(1,std::memory_order_relaxed);
            }
            consumed.store(at,std::memory_order_seq_cst);
            if(producer_waiting.load(std::memory_order_seq_cst))consumed.notify_one();
        }
        on_worker=false;
    }
    Handler handler;void* context;std::unique_ptr<u8[]> ring;
    u64 head{};u32 pending{};
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    alignas(64) std::atomic<u64> wake{0};
    std::atomic<bool> sleeping{},producer_waiting{};
    static inline thread_local bool on_worker{};
    std::jthread worker;
};
}
