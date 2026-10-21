// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "common/types.h"
namespace Vulkan { class Scheduler; }
namespace iss::feedback_join {
extern const bool enabled;
struct ReadTicket { u64 id{}, rip{}, address{}, owner{}, record{}, physical{}, epoch{}, buffer_generation{}, before_sequence{}; };
void BeginBindings();
void Binding(u64 address, u64 hash, u32 stage, u32 qid, u32 gpu_queue,
             u64 buffer_generation, Vulkan::Scheduler& scheduler);
void Recorded();
ReadTicket BeginRead(u64 rip, u64 address, u64 owner, u64 record, u64 buffer_generation);
void EndRead(const ReadTicket&, bool handled, u64 buffer_generation);
}
