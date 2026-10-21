// SPDX-License-Identifier: GPL-2.0-or-later
// std::for_each(std::execution::par, ...) where the standard library has it. libc++ (MSYS2
// CLANG64) has no parallel algorithms; libstdc++ without TBB (our Linux build) runs them in
// sequence anyway, so the sequential fallback changes nothing there.
#pragma once
#include <algorithm>
#include <version>
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201902L
#include <execution>
#define ISS_PAR_FOR_EACH(first, last, ...) std::for_each(std::execution::par, first, last, __VA_ARGS__)
#else
#define ISS_PAR_FOR_EACH(first, last, ...) std::for_each(first, last, __VA_ARGS__)
#endif
