// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <span>

namespace Shader::Gcn {

// Compare in program order and stop at the first changed word. An old shader may have
// been replaced by a shorter one immediately before an unmapped page. A bulk memcmp
// of the old length could cross that page before noticing the earlier changed return.
// Volatile reads prevent vectorization/read-ahead; guest submission synchronization,
// as in the decoder itself, is still responsible for excluding concurrent code writes.
inline bool FetchShaderCodeMatches(const std::uint32_t* code,
                                   std::span<const std::uint32_t> snapshot) {
    const volatile std::uint32_t* current = code;
    for (const auto word : snapshot) {
        if (*current++ != word) {
            return false;
        }
    }
    return true;
}

} // namespace Shader::Gcn
