// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace iss::native_geometry {
// Versioned metadata only. Never guest bytes, GPU ownership or completion.
// The caller revalidates image-absence epochs before using that observation.
// Both mapping generations must match at every use.
template<std::size_t Capacity> class BackingAliasCache {
    static_assert(Capacity && (Capacity&(Capacity-1))==0);
public:
    struct Entry {
        std::uintptr_t owner{};
        std::uint64_t start{},end{},epoch{},generation{},physical{};
        bool unique{};
        std::uint64_t image_epoch{};
        bool no_images{};
    };
    const Entry* Find(std::uintptr_t owner,std::uint64_t start,std::uint64_t end,
                      std::uint64_t epoch,std::uint64_t generation) const {
        const auto& e=entries[Slot(start,end)];
        return owner && epoch && generation && e.owner==owner && e.start==start &&
            e.end==end && e.epoch==epoch && e.generation==generation?&e:nullptr;
    }
    void Store(const Entry& e){entries[Slot(e.start,e.end)]=e;}
private:
    static std::size_t Slot(std::uint64_t start,std::uint64_t end) {
        const auto key=(start>>12)^((end-start)>>12)*0x9e3779b97f4a7c15ull;
        return std::size_t((key^(key>>23))*0x9e3779b97f4a7c15ull)&(Capacity-1);
    }
    std::array<Entry,Capacity> entries{};
};
}
