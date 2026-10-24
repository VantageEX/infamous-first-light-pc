// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace VideoCore {

/// Who asked for a CPU invalidation. Write faults (either kind) need no readback-job pre-settle;
/// only guest-thread faults may have their upload deferred.
enum class InvalidateOrigin : std::uint8_t {
    Explicit,
    GuestWriteFault,
    CpWriteFault,
};

/**
 * Submission-scoped upload deferral (SHADPS4_EPOCH_DEFER_UPLOAD=1, default off).
 *
 * A command the CP is processing can only legally read guest data written before the guest
 * submitted it (or before a label the CP waited on). The epoch below is bumped on the CP thread
 * whenever that boundary moves: a submission starts, a memory wait is satisfied, or the CP writes
 * guest memory itself. A page first dirtied by a guest write fault in the current epoch was
 * therefore written after everything the CP is working on was submitted, so read-only syncs may
 * leave it dirty and writable until the next epoch instead of uploading and re-protecting it
 * while the guest is still filling it.
 *
 * Measured 2026-09-23: ~90 guest graphics submissions/frame, ~2,900 write faults/frame, ~2,150
 * CP re-protect syscalls/frame, 1.3-3 faults per page per frame. Huge read-only texel buffer
 * binds sync every dirty page within ~549 MB, chasing the guest page by page.
 */
inline const bool g_epoch_defer_upload = [] {
    const char* value = std::getenv("SHADPS4_EPOCH_DEFER_UPLOAD");
    return value && value[0] == '1';
}();

inline std::atomic<std::uint64_t> g_submit_epoch{1};
/// Bumps by PM4 opcode (0 = submission start), for the periodic EpochDefer log line.
inline std::array<std::atomic<std::uint64_t>, 256> g_submit_epoch_reasons{};

/// CP thread only. Closes the current deferral window.
template <typename Reason>
inline void BumpSubmitEpoch(Reason reason) noexcept {
    if (g_epoch_defer_upload) {
        g_submit_epoch.fetch_add(1, std::memory_order_acq_rel);
        g_submit_epoch_reasons[static_cast<std::uint32_t>(reason) & 0xff].fetch_add(
            1, std::memory_order_relaxed);
    }
}

} // namespace VideoCore
