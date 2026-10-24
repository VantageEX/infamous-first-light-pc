// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"

// ============================================================================
// Windows static guest red-zone protection
// ============================================================================

enum class WindowsGuestRedZoneProtectionMode : u32 {
    Disabled,
    StaticPatching,
};

namespace Core::WindowsGuestRedZoneProtection {

void SetActiveMode(WindowsGuestRedZoneProtectionMode mode) noexcept;
WindowsGuestRedZoneProtectionMode GetActiveMode() noexcept;
bool IsStaticPatchingEnabled() noexcept;

} // namespace Core::WindowsGuestRedZoneProtection

// ============================================================================
// End Windows static guest red-zone protection
// ============================================================================

namespace Core {

// Windows static guest red-zone protection
struct RedZonePatchResult {
    u64 function_count{};
    u64 instruction_count{};
    u64 red_zone_function_count{};
    u64 memory_instruction_count{};
    u64 short_memory_instruction_count{};
    u64 patched_memory_instruction_count{};
    u64 stack_dependent_memory_instruction_count{};
    u64 control_flow_memory_instruction_count{};
    u64 unrelocatable_memory_instruction_count{};
    u64 indirect_red_zone_function_count{};
    u64 cpu_patch_instruction_count{};
    u64 patched_cpu_patch_instruction_count{};
    u64 unsupported_cpu_patch_instruction_count{};
};

/// Registers a module for patching, providing an area to generate trampoline code.
void RegisterPatchModule(void* module_ptr, u64 module_size, void* trampoline_area_ptr,
                         u64 trampoline_area_size);

/// Reserves `size` bytes (64-byte aligned, read / write / execute) in the trampoline area of the
/// module containing `module_address`, within a near jump of its code. nullptr when full.
u8* AllocateTrampolineSpace(const void* module_address, u64 size);

/// Applies CPU patches that need to be done before beginning executions.
void PrePatchInstructions(u64 segment_addr, u64 segment_size);

/// Windows builds: true under Wine / Proton (ntdll exports wine_get_version), where guest fs:
/// reads do not fault and so are never patched lazily (GitHub #17, #29). SHADPS4_AOT_PATCH=1
/// forces it on, =0 off. Logged once.
bool AheadOfTimePatchingWanted();

/// Patches every matching instruction of an executable segment ahead of time, as Linux builds do
/// in PrePatchInstructions. On Windows it runs after the red-zone static patcher (which decodes
/// the original bytes and calls the same patches for the functions it visits).
void PatchSegmentAheadOfTime(u64 segment_addr, u64 segment_size);

// Windows static guest red-zone protection
/// Keeps Windows exception dispatch outside live guest red zones at faultable memory accesses.
RedZonePatchResult PatchRedZoneMemoryInstructions(u64 segment_addr, u64 segment_size,
                                                  std::span<const uintptr_t> function_starts);

} // namespace Core
