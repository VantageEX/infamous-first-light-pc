// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <span>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perf_stats.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/scout.h"

namespace AmdGpu {

// ScoutArena

u32* ScoutArena::Allocate(u32 dwords) {
    if (dwords > BlockDwords) {
        return nullptr;
    }
    if (used + dwords > BlockDwords) {
        // A block of the pool (the submissions of the last frames gave theirs back).
        blocks.emplace_back(owner->TakeBlock());
        used = 0;
    }
    u32* result = blocks.back().get() + used;
    used += dwords;
    return result;
}

void ScoutArena::Reset(std::vector<Block>& pool) {
    for (auto& block : blocks) {
        if (pool.size() < MaxPooledBlocks) {
            pool.emplace_back(std::move(block));
        }
    }
    // Blocks beyond the pool's size are freed here.
    blocks.clear();
    used = BlockDwords;
}

// ScoutSubmission

ScoutSubmission::ScoutSubmission(Scout& scout_, u64 seq_, std::span<const u32> dcb_)
    : scout{scout_}, seq{seq_}, dcb{dcb_}, arena{scout_} {}

ScoutSubmission::~ScoutSubmission() {
    std::scoped_lock lk{scout.PoolMutex()};
    for (auto& chunk : chunk_storage) {
        scout.ReturnChunk(std::move(chunk));
    }
    arena.Reset(scout.BlockPool());
}

ScoutDraw* ScoutSubmission::NewDraw(const void* packet) {
    const u32 index = num_draws;
    const u32 chunk_index = index >> ChunkBits;
    if (chunk_index >= MaxChunks) {
        return nullptr;
    }
    if ((index & (ChunkSize - 1)) == 0) {
        auto chunk = scout.TakeChunk();
        chunks[chunk_index].store(chunk.get(), std::memory_order_release);
        chunk_storage.emplace_back(std::move(chunk));
    }
    ScoutDraw* draw = &chunks[chunk_index].load(std::memory_order_relaxed)[index & (ChunkSize - 1)];
    // Not cleared as a whole (6 KB per draw): the stages are filled from their start, and only
    // the first num_stages are read.
    draw->vertex_sgpr_offset = 0;
    draw->instance_sgpr_offset = 0;
    draw->num_stages = 0;
    draw->packet = packet;
    ++num_draws;
    return draw;
}

void ScoutSubmission::Publish() {
    published.store(num_draws, std::memory_order_release);
}

void ScoutSubmission::RecordCondition(const void* packet, bool skip) {
    if (!conditions) {
        conditions = std::make_unique<Condition[]>(MaxConditions);
    }
    if (num_conditions >= MaxConditions) {
        Abandon();
        return;
    }
    conditions[num_conditions] = Condition{.packet = packet, .skip = skip};
    ++num_conditions;
    published_conditions.store(num_conditions, std::memory_order_release);
}

void ScoutSubmission::Abandon() {
    abandoned.store(true, std::memory_order_release);
}

const ScoutDraw* ScoutSubmission::Next(const void* packet) noexcept {
    const u32 index = cursor++;
    if (desynced || abandoned.load(std::memory_order_acquire)) {
        return nullptr;
    }
    if (index >= published.load(std::memory_order_acquire)) {
        Common::Perf::Add(Common::Perf::ScoutBehind);
        return nullptr;
    }
    const ScoutDraw* draw =
        &chunks[index >> ChunkBits].load(std::memory_order_acquire)[index & (ChunkSize - 1)];
    if (draw->packet != packet) {
        // Not the same path through the commands: the entries are not for these draws.
        Common::Perf::Add(Common::Perf::ScoutDesyncs);
        desynced = true;
        return nullptr;
    }
    return draw;
}

const ScoutDraw* ScoutSubmission::Peek() const noexcept {
    const u32 index = cursor;
    if (desynced || abandoned.load(std::memory_order_acquire) ||
        index >= published.load(std::memory_order_acquire)) {
        return nullptr;
    }
    return &chunks[index >> ChunkBits].load(std::memory_order_acquire)[index & (ChunkSize - 1)];
}

void ScoutSubmission::CheckCondition(const void* packet, bool skip) noexcept {
    if (desynced) {
        return;
    }
    const u32 index = condition_cursor++;
    if (index >= published_conditions.load(std::memory_order_acquire)) {
        // The scout has not decided it yet: it reads the same memory later, which may have
        // changed by then. Entries after it are not trusted.
        desynced = true;
        return;
    }
    const Condition& condition = conditions[index];
    if (condition.packet != packet || condition.skip != skip) {
        Common::Perf::Add(Common::Perf::ScoutDesyncs);
        desynced = true;
    }
}

// Scout

Scout::Scout(ScoutClient& client_) : client{client_} {
    thread = std::jthread([this](std::stop_token stoken) { Run(stoken); });
}

Scout::~Scout() {
    thread.request_stop();
    cv.notify_all();
    thread.join();
}

void Scout::Submit(u64 seq, std::span<const u32> dcb) {
    auto submission = std::make_shared<ScoutSubmission>(*this, seq, dcb);
    {
        std::scoped_lock lk{mutex};
        pending.emplace_back(submission);
        live.emplace_back(std::move(submission));
        // Submissions the command processor finished long ago and never asked for.
        while (live.size() > 64) {
            live.pop_front();
        }
    }
    cv.notify_one();
}

void Scout::Skip(u64 seq) {
    // Nothing queued: the thread notices the gap in the sequence numbers and asks for the
    // registers again (Run).
    (void)seq;
}

std::shared_ptr<ScoutSubmission> Scout::Begin(u64 seq, const Regs& cp_regs) {
    cp_seq.store(seq, std::memory_order_release);
    std::shared_ptr<ScoutSubmission> result;
    bool copied = false;
    {
        std::scoped_lock lk{mutex};
        for (const auto& submission : live) {
            if (submission->Seq() == seq) {
                result = submission;
                break;
            }
        }
        if (result && want_regs.load(std::memory_order_acquire)) {
            // The registers before this submission, for the scout to resume from it.
            regs_copy = std::make_unique<Regs>(cp_regs);
            regs_copy_seq = seq;
            want_regs.store(false, std::memory_order_release);
            copied = true;
        }
    }
    if (copied) {
        cv.notify_one();
    }
    return result;
}

void Scout::Finish(u64 seq) {
    cp_finished_seq.store(seq, std::memory_order_release);
    std::scoped_lock lk{mutex};
    while (!live.empty() && live.front()->Seq() <= seq) {
        live.pop_front();
    }
}

bool Scout::MayBeWrittenLater(VAddr address, u64 size) const noexcept {
    const VAddr end = address + size;
    for (const auto& range : written) {
        if (address < range.end && range.begin < end) {
            return true;
        }
    }
    return false;
}

ScoutArena::Block Scout::TakeBlock() {
    {
        std::scoped_lock lk{pool_mutex};
        if (!block_pool.empty()) {
            auto block = std::move(block_pool.back());
            block_pool.pop_back();
            return block;
        }
    }
    // Written before read (flattened user data): no need to clear it.
    return ScoutArena::Block{new u32[ScoutArena::BlockDwords]};
}

ScoutSubmission::Chunk Scout::TakeChunk() {
    {
        std::scoped_lock lk{pool_mutex};
        if (!chunk_pool.empty()) {
            auto chunk = std::move(chunk_pool.back());
            chunk_pool.pop_back();
            return chunk;
        }
    }
    return std::make_unique<ScoutDraw[]>(ScoutSubmission::ChunkSize);
}

void Scout::ReturnChunk(ScoutSubmission::Chunk chunk) {
    // pool_mutex held by the caller (submission destructor).
    if (chunk_pool.size() < 256) {
        chunk_pool.emplace_back(std::move(chunk));
    }
}

void Scout::ReturnBlocks(std::vector<ScoutArena::Block>& blocks) {
    std::scoped_lock lk{pool_mutex};
    for (auto& block : blocks) {
        if (block_pool.size() < ScoutArena::MaxPooledBlocks) {
            block_pool.emplace_back(std::move(block));
        }
    }
    blocks.clear();
}

void Scout::NoteWrite(VAddr address, u64 size) {
    if (size == 0) {
        return;
    }
    if (written.size() >= 4096) {
        // Too many to check cheaply: the whole address space counts as written until the
        // command processor catches up.
        written.clear();
        written.push_back(WrittenRange{.begin = 0, .end = ~VAddr{0}, .seq = current_seq});
        return;
    }
    written.push_back(WrittenRange{.begin = address, .end = address + size, .seq = current_seq});
}

void Scout::Run(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:Scout");
    u64 last_seq = 0; ///< The last submission walked (the registers are those after it).
    while (!stoken.stop_requested()) {
        std::shared_ptr<ScoutSubmission> submission;
        {
            std::unique_lock lk{mutex};
            Common::CondvarWait(cv, lk, stoken, [this] {
                return !pending.empty() && (regs_valid || regs_copy);
            });
            if (stoken.stop_requested()) {
                break;
            }
            if (regs_valid && pending.front()->Seq() != last_seq + 1) {
                // A submission was skipped (scout off, or walked unsafely): the registers do
                // not describe the state before this one.
                regs_valid = false;
            }
            if (!regs_valid) {
                if (!regs_copy) {
                    want_regs.store(true, std::memory_order_release);
                    continue;
                }
                // Submissions before the copy's cannot be scouted (no registers for them).
                while (!pending.empty() && pending.front()->Seq() < regs_copy_seq) {
                    pending.front()->Abandon();
                    pending.pop_front();
                }
                if (pending.empty() || pending.front()->Seq() != regs_copy_seq) {
                    // The copy is for a submission not queued (not submitted yet, or gone).
                    regs_copy.reset();
                    want_regs.store(true, std::memory_order_release);
                    continue;
                }
                regs = *regs_copy;
                regs_copy.reset();
                regs_valid = true;
                written.clear();
            }
            submission = std::move(pending.front());
            pending.pop_front();
        }
        current_seq = submission->Seq();
        last_seq = current_seq;
        // Memory writes of the submissions the command processor finished are done.
        const u64 finished = cp_finished_seq.load(std::memory_order_acquire);
        std::erase_if(written, [finished](const WrittenRange& range) { return range.seq <= finished; });
        if (finished >= current_seq) {
            // The command processor finished it: the guest may have recycled its command
            // buffers, reading them is unsafe. The registers must be copied again.
            Common::Perf::Add(Common::Perf::ScoutLate);
            submission->Abandon();
            regs_valid = false;
            want_regs.store(true, std::memory_order_release);
            continue;
        }
        if (cp_seq.load(std::memory_order_acquire) > current_seq) {
            // The command processor is past it: nothing to precompute, but its register
            // writes and memory writes still count for the next submissions.
            Common::Perf::Add(Common::Perf::ScoutLate);
            submission->Abandon();
        }
        Common::Perf::Add(Common::Perf::ScoutSubmissions);
        const u64 start = Common::Perf::NowNs();
        Walk(*submission, submission->dcb, 0);
        submission->Publish();
        if (cp_finished_seq.load(std::memory_order_acquire) >= current_seq) {
            // The command processor finished it while it was being read: the guest may have
            // recycled its command buffers meanwhile, so the registers replayed from them are
            // not trusted for the next submissions.
            Common::Perf::Add(Common::Perf::ScoutLate);
            regs_valid = false;
            want_regs.store(true, std::memory_order_release);
        }
        Common::Perf::Add(Common::Perf::ScoutBusyNs, Common::Perf::NowNs() - start);
    }
}

static std::span<const u32> SkipPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        return {};
    }
    return span.subspan(offset);
}

void Scout::Walk(ScoutSubmission& sub, std::span<const u32> dcb, u32 depth) {
    const bool precompute = !sub.abandoned.load(std::memory_order_relaxed);
    while (!dcb.empty()) {
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        if (type == 2) {
            dcb = SkipPacket(dcb, 1);
            continue;
        }
        if (type != 3) {
            // The command processor stops on these: so does the scout.
            sub.Abandon();
            return;
        }
        const u32 count = header->type3.NumWords();
        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* payload = reinterpret_cast<const u32*>(header + 2);
        const auto draw = [&](u16 vertex_sgpr_offset = 0, u16 instance_sgpr_offset = 0) {
            if (!precompute || sub.abandoned.load(std::memory_order_relaxed)) {
                return;
            }
            ScoutDraw* entry = sub.NewDraw(header);
            if (entry == nullptr) {
                sub.Abandon();
                return;
            }
            entry->vertex_sgpr_offset = vertex_sgpr_offset;
            entry->instance_sgpr_offset = instance_sgpr_offset;
            client.ScoutDraw(regs, *entry, sub.Arena());
            Common::Perf::Add(Common::Perf::ScoutDraws);
            // Entries are published as they come, the command processor may be close behind.
            sub.Publish();
        };
        switch (opcode) {
        case PM4ItOpcode::ClearState:
            regs.SetDefaults();
            break;
        case PM4ItOpcode::SetConfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const u32 reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
            if (count >= 1 && reg_addr + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
            }
            break;
        }
        case PM4ItOpcode::SetContextReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const u32 reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
            if (count >= 1 && reg_addr + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                // Compute program of the graphics queue: not scouted.
                break;
            }
            const u32 reg_addr = Regs::ShRegWordOffset + set_data->reg_offset;
            if (count >= 1 && reg_addr + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[reg_addr], header + 2, (count - 1) * sizeof(u32));
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const u32 reg_addr = Regs::UconfigRegWordOffset + set_data->reg_offset;
            if (count >= 1 && reg_addr + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[reg_addr], header + 2, (count - 1) * sizeof(u32));
            }
            break;
        }
        case PM4ItOpcode::IndexType: {
            const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
            regs.index_buffer_type.raw = index_type->raw;
            break;
        }
        case PM4ItOpcode::NumInstances: {
            const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
            regs.num_instances.num_instances = num_instances->num_instances;
            break;
        }
        case PM4ItOpcode::IndexBase: {
            const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
            regs.index_base_address.base_addr_lo = index_base->addr_lo;
            regs.index_base_address.base_addr_hi = index_base->addr_hi;
            break;
        }
        case PM4ItOpcode::IndexBufferSize: {
            const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
            regs.num_indices = index_size->num_indices;
            break;
        }
        case PM4ItOpcode::DrawIndex2: {
            const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            regs.max_index_size = draw_index->max_size;
            regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
            regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
            regs.num_indices = draw_index->index_count;
            regs.draw_initiator = draw_index->draw_initiator;
            draw();
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            const auto* draw_index_off = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            regs.max_index_size = draw_index_off->max_size;
            regs.num_indices = draw_index_off->index_count;
            regs.draw_initiator = draw_index_off->draw_initiator;
            draw();
            break;
        }
        case PM4ItOpcode::DrawIndexAuto: {
            const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            regs.num_indices = draw_index->index_count;
            regs.draw_initiator = draw_index->draw_initiator;
            draw();
            break;
        }
        case PM4ItOpcode::DrawIndirect: {
            const auto* packet = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
            draw(packet->base_vtx_loc, packet->start_inst_loc);
            break;
        }
        case PM4ItOpcode::DrawIndirectMulti: {
            const auto* packet = reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
            draw(packet->base_vtx_loc, packet->start_inst_loc);
            break;
        }
        case PM4ItOpcode::DrawIndexIndirect: {
            const auto* packet = reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
            draw(packet->base_vtx_loc, packet->start_inst_loc);
            break;
        }
        case PM4ItOpcode::DrawIndexIndirectMulti: {
            const auto* packet = reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
            draw(packet->base_vtx_loc, packet->start_inst_loc);
            break;
        }
        case PM4ItOpcode::DrawIndexIndirectCountMulti: {
            const auto* packet =
                reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
            draw(packet->base_vtx_loc, packet->start_inst_loc);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            if (depth < 8) {
                Walk(sub, {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size},
                     depth + 1);
            } else {
                sub.Abandon();
            }
            break;
        }
        case PM4ItOpcode::CondExec: {
            const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
            const bool skip = *cond_exec->Address() == false;
            sub.RecordCondition(header, skip);
            if (skip) {
                dcb = SkipPacket(dcb, header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                continue;
            }
            break;
        }
        // Memory the command processor writes when it gets there: reading it before is stale.
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo != 0x3022C && dma_data->dst_sel != DmaDataDst::Gds) {
                NoteWrite(dma_data->DstAddress<VAddr>(), dma_data->NumBytes());
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            NoteWrite(write_data->Address<VAddr>(), (count >= 2 ? count - 2 : 0) * sizeof(u32));
            break;
        }
        case PM4ItOpcode::EventWrite: {
            const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                regs.cp_strmout_cntl.offset_update_done = 1;
            } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                NoteWrite(event->Address<VAddr>(), 0x100);
            }
            break;
        }
        case PM4ItOpcode::EventWriteEos: {
            const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
            NoteWrite(reinterpret_cast<VAddr>(event_eos->Address()), 8);
            break;
        }
        case PM4ItOpcode::EventWriteEop: {
            const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
            NoteWrite(reinterpret_cast<VAddr>(event_eop->Address<u32>()), 8);
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            NoteWrite(mem_semaphore->Address<VAddr>(), 8);
            break;
        }
        default:
            break;
        }
        dcb = SkipPacket(dcb, count + 1);
    }
}

} // namespace AmdGpu
