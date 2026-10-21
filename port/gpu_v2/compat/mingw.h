// SPDX-License-Identifier: GPL-2.0-or-later
// Windows with MSYS2 CLANG64 (MinGW): the MSVC intrinsics the renderer uses that MinGW lacks.
#pragma once
#if defined(_WIN32) && !defined(_MSC_VER)
#include <intrin.h>
#ifndef _ReturnAddress
#define _ReturnAddress() __builtin_return_address(0)
#endif
#ifndef _AddressOfReturnAddress
#define _AddressOfReturnAddress() \
    (static_cast<void*>(static_cast<char*>(__builtin_frame_address(0)) + sizeof(void*)))
#endif
#endif
