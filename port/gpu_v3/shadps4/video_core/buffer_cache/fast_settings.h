// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include "core/emulator_settings.h"

namespace VideoCore {

/**
 * `EmulatorSettings.X()` takes a process-wide mutex and copies a shared_ptr on every call. The
 * tracker reads the readbacks mode on every guest write fault and every dirty upload walk, from
 * the CP and all guest fault threads at once. SHADPS4_CACHE_READBACKS_MODE=1 reads it once; the
 * mode is fixed for a running game (title config is applied before any GPU work).
 */
inline auto ReadbacksMode() {
    static const bool cache = [] {
        const char* value = std::getenv("SHADPS4_CACHE_READBACKS_MODE");
        return value && value[0] == '1';
    }();
    if (cache) {
        static const auto mode = EmulatorSettings.GetReadbacksMode();
        return mode;
    }
    return EmulatorSettings.GetReadbacksMode();
}

} // namespace VideoCore
