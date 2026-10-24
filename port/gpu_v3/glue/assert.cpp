// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3: the fork's assert handlers (common/assert.cpp) without its emulator: a failed check
// prints its text and the call stack to the game log (stderr), then stops the process the same
// way (int3: the runtime's crash handler writes its report).
#include <cstdio>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/assert.h"
#include "common/logging/log.h"

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
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                                                     reinterpret_cast<uintptr_t>(module)));
    }
#else
    (void)skip;
#endif
    std::fflush(stderr);
}

} // namespace Common

void assert_fail_impl() {
    std::fprintf(stderr, "V3: assertion failed; call stack (module+offset):\n");
    Common::PrintStackToStderr(0);
    std::fflush(stdout);
    __asm__ __volatile__("int $3");
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    std::fprintf(stderr, "V3: assertion failed: %s\n", msg);
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
