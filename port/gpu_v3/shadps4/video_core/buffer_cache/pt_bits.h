// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "common/types.h"

namespace VideoCore {

/// The GPU page table's coverage when SHADPS4_PT_BITS is not set (BufferCache::BdaNumPages): set
/// by Vulkan::Instance once the video memory size is known, before any table or shader exists.
/// 37 bits (64 MB table) below 10 GiB of video memory, else 40 (512 MB; v1.1, 2026-10-20).
inline std::atomic<u32> g_pt_auto_bits{40};

} // namespace VideoCore
