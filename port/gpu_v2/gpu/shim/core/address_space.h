// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <boost/icl/interval_set.hpp>
#include "common/enum.h"
#include "common/types.h"
#include "common/multi_level_page_table.h"
namespace Core {
enum class MemoryPermission:u32 {None=0,Read=1,Write=2,ReadWrite=3,Execute=4,ReadWriteExecute=7};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)
class AddressSpace {
public:
    struct Traits {using Entry=u8*;static constexpr size_t ADDRESS_SPACE_BITS=40,L1_BITS=16,PAGE_BITS=14;static constexpr bool NULL_CHECK=true;};
    using BackingPageTable=Common::MultiLevelPageTable<Traits>;
    void Protect(VAddr address,u64 size,MemoryPermission permissions);
    boost::icl::interval_set<VAddr> GetUsableRegions();
    VAddr SystemReservedVirtualBase() const{return 0x700000000ULL;}
    u64 SystemReservedVirtualSize() const{return 0x100000000ULL;}
    const BackingPageTable& BackingPages() const{return backing_pages;}
    void Register(VAddr address,u64 size,u8 *backing);
private:
    BackingPageTable backing_pages;
};
}
