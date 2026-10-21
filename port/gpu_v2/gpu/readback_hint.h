// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <atomic>
#include <cstdint>
// A hint changes transfer granularity only. Unhinted pages retain the default;
// all callers still cover their requested bytes and wait for GPU completion.
class ReadbackHints {
    std::array<std::atomic<std::uint64_t>,8> pages{};
    std::atomic<unsigned> next{};
public:
    void Remember(std::uint64_t address){
        const auto key=(address&~std::uint64_t{4095})+1;
        for(auto& page:pages)if(page.load(std::memory_order_relaxed)==key)return;
        pages[next.fetch_add(1,std::memory_order_relaxed)%pages.size()].store(key,std::memory_order_relaxed);
    }
    std::uint64_t WindowFor(std::uint64_t address,std::uint64_t fallback) const {
        const auto key=(address&~std::uint64_t{4095})+1;
        for(const auto& page:pages)if(page.load(std::memory_order_relaxed)==key)return 4096;
        return fallback;
    }
};
