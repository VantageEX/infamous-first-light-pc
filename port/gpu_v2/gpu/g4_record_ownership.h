// SPDX-License-Identifier: GPL-2.0-or-later
// Frontend recording identity, independent of the scheduler's submit/record workers.
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>

#include "g4_cp_coordination.h"

namespace iss::g4 {

struct RecordingIdentity {
    CpLane lane{CpLane::Graphics};
    std::uint32_t guest_qid{};
    std::uint64_t guest_generation{};
};

inline thread_local std::optional<RecordingIdentity> active_recording_identity{};
/// Set once any RecordingOwnerScope exists: before that no thread has an identity.
inline std::atomic<bool> recording_identity_used{false};

class RecordingOwnerScope {
public:
    explicit RecordingOwnerScope(RecordingIdentity identity) noexcept
        : previous{active_recording_identity} {
        recording_identity_used.store(true, std::memory_order_relaxed);
        active_recording_identity = identity;
    }
    ~RecordingOwnerScope() { active_recording_identity = previous; }
    RecordingOwnerScope(const RecordingOwnerScope&) = delete;
    RecordingOwnerScope& operator=(const RecordingOwnerScope&) = delete;

private:
    std::optional<RecordingIdentity> previous;
};

[[nodiscard]] inline bool CurrentRecordingOwnerMatches(CpLane expected) noexcept {
    if (!recording_identity_used.load(std::memory_order_relaxed)) {
        return true;
    }
    return !active_recording_identity || active_recording_identity->lane == expected;
}

} // namespace iss::g4
