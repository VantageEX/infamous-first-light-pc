// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
namespace iss::readback {
inline bool PageProvenanceEnabled() {
    static const bool enabled=[] {const char* p=std::getenv("ISS_NATIVE_READBACK_PAGES");return p&&p[0]=='1';}();
    return enabled;
}
// Caller owns synchronization. Keep BOTH queue dependencies: disjoint writes in
// one page need not supersede each other. Missing history never means completed.
template<std::size_t Pages> class PageWriteProvenance {
public:
    bool Publish(std::size_t first,std::size_t last,std::uint32_t queue,std::uint64_t tick) {
        if(first>last || last>=Pages || queue>=2 || !tick)return false;
        for(auto page=first;page<=last;) {
            const auto word=page/64;
            const auto end=std::min(last,(word+1)*64-1);
            if(page%64==0 && end-page==63) {
                full_words[queue][word]=std::max(full_words[queue][word],tick);
            } else {
                for(auto p=page;p<=end;++p)ticks[queue][p]=std::max(ticks[queue][p],tick);
            }
            page=end+1;
        }
        return true;
    }
    bool Merge(std::size_t page,std::array<std::uint64_t,2>& result) const {
        if(page>=Pages)return false;
        const auto a=std::max(ticks[0][page],full_words[0][page/64]);
        const auto b=std::max(ticks[1][page],full_words[1][page/64]);
        if(!a&&!b)return false;
        result[0]=std::max(result[0],a);result[1]=std::max(result[1],b);
        return true;
    }
private:
    std::array<std::array<std::uint64_t,Pages>,2> ticks{};
    std::array<std::array<std::uint64_t,(Pages+63)/64>,2> full_words{};
};
}
