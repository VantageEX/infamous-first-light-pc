// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"
namespace iss::timeline {
extern const bool enabled;
// Event names must be string literals. Producer path is bounded and never waits
// for the writer lock or performs file I/O after one-time initialization.
// owner+tick are event-specific IDs; observed_guest_frame is NOT frame ownership.
void Emit(const char* event,u64 owner,u64 tick,u64 a=0,u64 b=0,u64 c=0,u64 d=0,
          u64 duration_ns=0);
}
