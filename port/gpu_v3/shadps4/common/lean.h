// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

namespace Common {

/// SHADPS4_LEAN=1: skip work on the command processor's hot paths that only feeds diagnostics:
/// log-only lookups, periodic statistics lines and the counters behind them, the frame-graph
/// overlay's draw counters, and the idle BufferSampleProbe object. Rendering is unchanged.
/// See notes/hot-path-diag-overhead-2026-09-25.md.
///
/// Namespace scope on purpose: a function-local `static const` flag costs a guard check (a TLS
/// epoch load chain under the MSVC ABI) on every call, while this is one plain load. A read made
/// during static initialization, before this initializer ran, sees false (the default).
inline const bool g_lean = [] {
    const char* value = std::getenv("SHADPS4_LEAN");
    return value != nullptr && value[0] == '1';
}();

[[nodiscard]] inline bool Lean() noexcept {
    return g_lean;
}

} // namespace Common
