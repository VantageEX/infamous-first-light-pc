// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include <memory>
#include <boost/icl/separate_interval_set.hpp>
#include "common/arch.h"
#include "common/enum.h"
#include "common/types.h"

namespace Core {

enum class MemoryPermission : u32 {
    None = 0,
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Execute = 1 << 2,
    ReadWriteExecute = Read | Write | Execute,
};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)

/**
 * Represents the user virtual address space backed by a dmem memory block
 */
class AddressSpace {
public:
    explicit AddressSpace();
    ~AddressSpace();

    [[nodiscard]] u8* BackingBase() const noexcept {
        return backing_base;
    }
    /// Size of the physical backing view at BackingBase().
    [[nodiscard]] u64 BackingSizeBytes() const noexcept;

    [[nodiscard]] VAddr SystemManagedVirtualBase() noexcept {
        return reinterpret_cast<VAddr>(system_managed_base);
    }
    [[nodiscard]] const u8* SystemManagedVirtualBase() const noexcept {
        return system_managed_base;
    }
    [[nodiscard]] u64 SystemManagedVirtualSize() const noexcept {
        return system_managed_size;
    }

    [[nodiscard]] VAddr SystemReservedVirtualBase() noexcept {
        return reinterpret_cast<VAddr>(system_reserved_base);
    }
    [[nodiscard]] const u8* SystemReservedVirtualBase() const noexcept {
        return system_reserved_base;
    }
    [[nodiscard]] u64 SystemReservedVirtualSize() const noexcept {
        return system_reserved_size;
    }

    [[nodiscard]] VAddr UserVirtualBase() noexcept {
        return reinterpret_cast<VAddr>(user_base);
    }
    [[nodiscard]] const u8* UserVirtualBase() const noexcept {
        return user_base;
    }
    [[nodiscard]] u64 UserVirtualSize() const noexcept {
        return user_size;
    }

    /**
     * @brief Maps memory to the specified virtual address.
     * @param virtual_addr The base address to place the mapping.
     *        If zero is provided an address in system managed area is picked.
     * @param size The size of the area to map.
     * @param phys_addr The offset of the backing file handle to map.
     *                  The same backing region may be aliased into different virtual regions.
     *                  If zero is provided the mapping is considered as private.
     * @return A pointer to the mapped memory.
     */
    void* Map(VAddr virtual_addr, u64 size, PAddr phys_addr = -1, bool exec = false);

    /// Memory maps a specified file descriptor.
    void* MapFile(VAddr virtual_addr, u64 size, u64 offset, u32 prot, uintptr_t fd);

    /// Unmaps specified virtual memory area.
    void Unmap(VAddr virtual_addr, u64 size);

    /// Protects requested region.
    void Protect(VAddr virtual_addr, u64 size, MemoryPermission perms);

    /// SHADPS4_PAGE_OS_SHADOW: bumped before and after every Map / MapFile / Unmap and every
    /// protection change made outside the page manager (BeginRemap / EndRemap), so the page
    /// manager's protection shadow can tell that the OS state may have changed under it.
    /// remaps_active counts changes in progress. Both sequentially consistent.
    [[nodiscard]] u64 RemapEpoch() const noexcept {
        return remap_epoch.load();
    }
    [[nodiscard]] u32 RemapsActive() const noexcept {
        return remaps_active.load();
    }
    void BeginRemap() noexcept {
        remap_epoch.fetch_add(1);
        remaps_active.fetch_add(1);
    }
    void EndRemap() noexcept {
        // Epoch before active (Opus review): a reader that loads active == 0 after this remap
        // must then load the bumped epoch (readers load active first).
        remap_epoch.fetch_add(1);
        remaps_active.fetch_sub(1);
    }

    // Returns an interval set containing all usable regions.
    boost::icl::interval_set<VAddr> GetUsableRegions();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    std::atomic<u64> remap_epoch{0};
    std::atomic<u32> remaps_active{0};
    u8* backing_base{};
    u8* system_managed_base{};
    u64 system_managed_size{};
    u8* system_reserved_base{};
    u64 system_reserved_size{};
    u8* user_base{};
    u64 user_size{};
};

/// SHADPS4_DLSS=1: frees the address space held below 1 TB for NVIDIA's driver; call right before
/// vkCreateDevice with VK_NVX_binary_import (address_space.cpp). No-op otherwise.
void ReleaseDriverVaHole();

} // namespace Core
