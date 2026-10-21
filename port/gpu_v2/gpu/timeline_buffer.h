// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
namespace iss::timeline {
template<class Event,std::size_t Capacity> class Buffer {
public:
    bool Push(const Event& event) {
        std::unique_lock lock{mutex,std::try_to_lock};
        if(!lock || count==Capacity){dropped.fetch_add(1,std::memory_order_relaxed);return false;}
        events[(first+count)%Capacity]=event;++count;return true;
    }
    std::size_t Take(std::span<Event> output) {
        std::scoped_lock lock{mutex};
        const auto n=std::min(count,output.size());
        for(std::size_t i=0;i<n;++i)output[i]=events[(first+i)%Capacity];
        first=(first+n)%Capacity;count-=n;return n;
    }
    std::uint64_t Dropped() const{return dropped.load(std::memory_order_relaxed);}
private:
    std::mutex mutex;
    std::array<Event,Capacity> events{};
    std::size_t first{},count{};
    std::atomic<std::uint64_t> dropped{};
};
}
