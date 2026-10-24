// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/reserved_memory.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace Common {

void* ReserveAddressSpace(std::size_t bytes) {
#ifdef _WIN32
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
    void* const base = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                            -1, 0);
    return base == MAP_FAILED ? nullptr : base;
#endif
}

bool CommitAddressSpace(void* base, std::size_t offset, std::size_t bytes) {
    char* const at = static_cast<char*>(base) + offset;
#ifdef _WIN32
    return VirtualAlloc(at, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    return mprotect(at, bytes, PROT_READ | PROT_WRITE) == 0;
#endif
}

void ReleaseAddressSpace(void* base, std::size_t bytes) {
#ifdef _WIN32
    (void)bytes;
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, bytes);
#endif
}

std::size_t AddressSpacePageSize() {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
}

} // namespace Common
