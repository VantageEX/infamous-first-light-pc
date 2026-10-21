// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <functional>
#include <optional>
#include <vector>
#include "common/types.h"
#include "common/singleton.h"
#include "core/address_space.h"
namespace Vulkan {class Rasterizer;}
namespace Core {
class MemoryManager {
public:
    AddressSpace& GetAddressSpace(){return space;}
    void SetRasterizer(Vulkan::Rasterizer *value);
    Vulkan::Rasterizer *GetRasterizer() const{return rasterizer;}
    bool IsValidGpuMapping(VAddr address,u64 size){return address<0x10000000000ULL&&size<0x10000000000ULL-address;}
    u64 ClampRangeSize(VAddr address,u64 size);
    struct BackingSpan { u64 physical_start{}; u64 size{}; };
    struct BackingAlias { VAddr guest_address{}; u64 physical_start{}; u64 size{}; };
    // The host shadow address is shared by guest VA aliases of the same physical bytes.
    // Only the contiguous part of the first mapping is returned.
    std::optional<BackingSpan> DescribeBacking(VAddr address,u64 size);
    // Returns every currently mapped VA of the first contiguous physical span, or nullopt
    // when an overlapping/replaced host mapping makes the alias set ambiguous.
    std::optional<std::vector<BackingAlias>> EnumerateBackingAliases(VAddr address,u64 size);
    [[nodiscard]] u64 MappingEpoch() const noexcept;
    // Native mapping's shared lock spans only a short metadata publication. The
    // callback must not dereference guest memory, wait, or call the GPU driver.
    [[nodiscard]] bool WithMappingReadLock(const std::function<bool(u64)>& publish) const;
    u64 PreferredReadbackWindow(VAddr address,u64 fallback);
    void CopySparseMemory(VAddr source,u8 *dest,u64 size);
    u64 HashSparseMemory(VAddr source,u64 size);
    bool TryWriteBacking(void *address,const void *data,u64 size);
    // Conditional direct shadow write for a completed GPU readback. The native
    // mapping shared lock covers predicate and memcpy, so remap cannot land in
    // between. Predicate must use only already prepared atomic/version metadata;
    // it must not wait, read guest memory, or acquire a cache lock.
    [[nodiscard]] bool TryWriteBackingIf(void* address, const void* data, u64 size,
                                         u64 expected_native_epoch,
                                         const std::function<bool()>& predicate);
    void InvalidateMemory(VAddr address,u64 size);
private:
    AddressSpace space;
    Vulkan::Rasterizer *rasterizer{};
};
using Memory=Common::Singleton<MemoryManager>;
}
