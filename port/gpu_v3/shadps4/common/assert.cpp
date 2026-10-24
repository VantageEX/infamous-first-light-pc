// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include "common/arch.h"
#include "common/assert.h"
#include "common/fatal_stack.h"
#include "emulator.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

namespace Common {

void PrintStackToStderr(unsigned skip) {
#ifdef _WIN32
    void* frames[24]{};
    const USHORT n = RtlCaptureStackBackTrace(static_cast<DWORD>(skip + 1), 24, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE module = nullptr;
        wchar_t name[MAX_PATH] = L"?";
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCWSTR>(frames[i]), &module)) {
            GetModuleFileNameW(module, name, MAX_PATH);
        }
        const wchar_t* base = wcsrchr(name, L'\\');
        std::fprintf(stderr, "  #%u %ls+%#llx\n", static_cast<unsigned>(i), base ? base + 1 : name,
                     module ? static_cast<unsigned long long>(
                                  reinterpret_cast<uintptr_t>(frames[i]) -
                                  reinterpret_cast<uintptr_t>(module))
                            : static_cast<unsigned long long>(
                                  reinterpret_cast<uintptr_t>(frames[i])));
    }
    std::fflush(stderr);
#else
    (void)skip;
#endif
}

} // namespace Common

void assert_fail_impl() {
    // v1.0.2: the failed check's text reaches fatal.txt through the log's Critical line (log.cpp,
    // written straight to stderr); the call stack below names the code that asserted.
    std::fprintf(stderr, "FATAL-ASSERT-IMPL; call stack (module+offset):\n");
    std::fflush(stderr);
    Common::PrintStackToStderr(0);
    Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    Crash();
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    // 2026-10-17: Shutdown() ends the process before the asynchronous log writes the line below.
    std::fprintf(stderr, "FATAL-ASSERT %s\n", msg);
    std::fflush(stderr);
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
