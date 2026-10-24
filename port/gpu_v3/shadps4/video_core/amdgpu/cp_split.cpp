// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <immintrin.h>
#include <fmt/format.h>

#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/readback_offload.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/buffer_cache/back_clean.h"

namespace AmdGpu::CpSplit {

namespace {
using Clock = std::chrono::steady_clock;

u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

/// Cheap timestamp for the wait counters (converted to ms at log time against steady_clock).
u64 Ticks() {
    return __rdtsc();
}

/// How long either side spins before parking. At depth 0 a hand-over happens every ~1-2 us
/// (derived: ~1.75 us of CP time per draw/dispatch), so parking on every one (5-50 us per wake)
/// would be unusable. SHADPS4_CP_SPLIT_SPIN_US overrides. SPLIT-TODO: tune; measure P-core cost.
u64 SpinNs() {
    static const u64 ns = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_SPIN_US");
        return (value ? std::strtoull(value, nullptr, 10) : 50ull) * 1000ull;
    }();
    return ns;
}

constexpr u32 AlignUp(u32 value, u32 align) {
    return (value + align - 1) & ~(align - 1);
}

/// posted_seqs slots: a power of two > MaxDepth, so the record MaxDepth + 1 posts back is kept.
constexpr u64 PostSlots = 2048;
static_assert(PostSlots > MaxDepth && (PostSlots & (PostSlots - 1)) == 0);

constexpr std::array<const char*, static_cast<size_t>(DrainReason::Count)> DrainNames{
    "depth0",   "verify_full", "dispatch", "pkt_dma",  "pkt_dma_guard", "pkt_fence",
    "pkt_wdata", "pkt_other",  "call",     "idle_tail", "wait_reg_mem", "cond_exec",
    "mem_sem",  "task_done",   "ib_start", "front_fault", "front_exit",
    "verify_disp", "asc_dma_guard", "asc_wdata_guard", "asc_ib_start",
    "call_asc_disp", "call_asc_disp_ind", "call_asc_disp_patch", "call_asc_dma",
    "call_asc_wdata", "call_asc_release", "call_asc_sem", "call_ce_dump",
    "ib_end", "disp_after_pkt", "task_guard", "task_guard_ib", "task_guard_resume",
    "task_guard_ovf", "wrm_partial",
};
static_assert(DrainNames.back() != nullptr, "DrainNames must name every DrainReason");
} // namespace

int Mode() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT");
        const int v = value ? std::atoi(value) : 0;
        return v <= 0 ? 0 : v == 1 ? 1 : 2;
    }();
    return mode;
}

u32 Depth() {
    static const u32 depth = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_DEPTH");
        const unsigned long long v = value ? std::strtoull(value, nullptr, 10) : 64ull;
        return static_cast<u32>(std::clamp<unsigned long long>(v, 1ull, MaxDepth));
    }();
    return depth;
}

bool PostDma() {
    static const bool post = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_POST_DMA");
        return value && value[0] == '1';
    }();
    return post;
}

bool PostAsc() {
    static const bool post = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_POST_ASC");
        return value && value[0] == '1';
    }();
    return post;
}

bool PostTaskDone() {
    static const bool post = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_POST_TASK_DONE");
        return value && value[0] == '1';
    }();
    return post;
}

u32 TaskGuard() {
    static const u32 level = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_TASK_GUARD");
        const unsigned long long v = value ? std::strtoull(value, nullptr, 10) : 0ull;
        return static_cast<u32>(std::min<unsigned long long>(v, 3ull));
    }();
    return level;
}

bool PostFence() {
    static const bool post = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_POST_FENCE");
        return value && value[0] == '1';
    }();
    return post;
}

bool DispGuard() {
    static const bool guard = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_DISP_GUARD");
        return !(value && value[0] == '0');
    }();
    return guard;
}

u32 PostAscMask() {
    static const u32 mask = [] {
        if (!PostAsc()) {
            return 0u;
        }
        const char* value = std::getenv("SHADPS4_CP_SPLIT_POST_ASC_MASK");
        const unsigned long long v =
            value ? std::strtoull(value, nullptr, 0)
                  : static_cast<unsigned long long>(PostAscDefaultMask);
        return static_cast<u32>(v & 0xfu);
    }();
    return mask;
}

bool Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_VERIFY");
        return value && value[0] == '1';
    }();
    return verify;
}

u32 VerifyEvery() {
    static const u32 every = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_VERIFY_EVERY");
        const unsigned long long v = value ? std::strtoull(value, nullptr, 10) : 256ull;
        return static_cast<u32>(std::clamp<unsigned long long>(v, 1ull, 1ull << 20));
    }();
    return every;
}

u32 DoneBatch() {
    static const u32 batch = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_DONE_BATCH");
        const unsigned long long v = value ? std::strtoull(value, nullptr, 10) : 1ull;
        return static_cast<u32>(std::clamp<unsigned long long>(v, 1ull, 1024ull));
    }();
    return batch;
}

u32 PubBatch() {
    static const u32 batch = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_PUB_BATCH");
        const unsigned long long v = value ? std::strtoull(value, nullptr, 10) : 1ull;
        return static_cast<u32>(std::clamp<unsigned long long>(v, 1ull, 256ull));
    }();
    return batch;
}

bool RegBatch() {
    static const bool batch = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_REG_BATCH");
        return value && value[0] == '1';
    }();
    return batch;
}

int FrontDerive() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_FRONT_DERIVE");
        const int v = value ? value[0] - '0' : 0;
        return v == 1 || v == 2 ? v : 0;
    }();
    return mode;
}

bool FrontDeriveVertexInputSkip() {
    static const bool skip = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_FRONT_DERIVE_VI_SKIP");
        return !(value && value[0] == '0');
    }();
    return skip;
}

Channel::Channel(u32 ring_bytes) {
    ASSERT_MSG(ring_bytes >= (64u << 10) && (ring_bytes & (ring_bytes - 1)) == 0,
               "CP split ring size must be a power of two >= 64 KB");
    capacity = ring_bytes;
    mask = ring_bytes - 1;
    // Over-allocate so the ring starts on a cache line.
    storage_owner = std::make_unique<u8[]>(std::size_t{ring_bytes} + 64);
    const auto base = reinterpret_cast<std::uintptr_t>(storage_owner.get());
    storage = reinterpret_cast<u8*>((base + 63) & ~std::uintptr_t{63});
    depth = RunAhead() ? Depth() : 1;
    done_batch = DoneBatch();
    pub_batch = PubBatch();
    reg_batch = RegBatch();
    posted_seqs = std::make_unique<u64[]>(PostSlots);
    posted_writes = std::make_unique<PostedWrite[]>(PostedWriteSlots);
    log_window_start_ns = NowNs();
    log_window_start_ticks = Ticks();
}

Channel::~Channel() = default;

void* Channel::Reserve(RecordType type, u32 payload_bytes, u16 flags) {
    ASSERT_MSG(!record_open, "CP split: Reserve before the previous record was committed");
    if (hold && type != RecordType::RegWrite && type != RecordType::Extent &&
        type != RecordType::Packet && type != RecordType::Diag && type != RecordType::Retire)
        [[unlikely]] {
        // Brick C v3 item 3: a record the hold may not reorder against - the segment builder
        // closes its segment (posting the Segment record, then the held journal) first.
        if (hold_cb) {
            hold_cb(hold_ctx);
        }
        if (hold) {
            FlushHold();
        }
    }
    const u32 bytes = AlignUp(u32(sizeof(RecordHeader)) + payload_bytes, RecordAlign);
    ASSERT_MSG(bytes <= capacity / 2, "CP split record of {} bytes does not fit the ring", bytes);
    u64 pos = reserve_pos;
    const u32 offset = u32(pos & mask);
    if (offset + bytes > capacity) {
        // Records never wrap: fill the tail with a Wrap record and start again at offset 0.
        const u32 filler = capacity - offset;
        WaitForSpace(pos + filler);
        auto* wrap = reinterpret_cast<RecordHeader*>(storage + offset);
        *wrap = RecordHeader{RecordType::Wrap, 0, filler, 0};
        pos += filler;
        reserve_pos = pos;
    }
    WaitForSpace(pos + bytes);
    last_record_pos = pos;
    auto* header = reinterpret_cast<RecordHeader*>(storage + (pos & mask));
    *header = RecordHeader{type, flags, bytes, ++emitted};
    reserve_pos = pos + bytes;
    record_open = true;
    ++counters.records;
    counters.bytes += bytes;
    return header + 1;
}

void Channel::Commit() {
    committed_pos = reserve_pos;
    committed_seq = emitted;
    record_open = false;
    if (asc_tag != 0 && early_items) [[unlikely]] {
        const auto* rh = reinterpret_cast<const RecordHeader*>(storage + (last_record_pos & mask));
        const u64 tail = early_tail.load(std::memory_order_relaxed);
        while (tail - early_head.load(std::memory_order_acquire) >= EarlySlots) {
            // The back consumes the early queue at every record boundary; publish so it can.
            ++early_full_waits;
            Publish();
            _mm_pause();
        }
        early_items[tail & (EarlySlots - 1)] = {last_record_pos, rh->seq, rh->type};
        early_tail.store(tail + 1, std::memory_order_release);
    }
    if (canon && canon_fn) [[unlikely]] {
        const auto* rh = reinterpret_cast<const RecordHeader*>(storage + (last_record_pos & mask));
        if (rh->type != RecordType::RegWrite) {
            canon_fn(*canon, *rh); // register words are recorded word by word in EmitRegWrite
        }
    }
    if (self_consume) [[unlikely]] {
        cached_read = committed_pos;
        read_pos.store(committed_pos, std::memory_order_relaxed);
        done_seq.store(committed_seq, std::memory_order_release);
    }
    if (draw_hook) [[unlikely]] {
        const auto* rh = reinterpret_cast<const RecordHeader*>(storage + (last_record_pos & mask));
        if (rh->type == RecordType::Draw) {
            draw_hook(hook_ctx, last_record_pos);
        }
    }
}

void Channel::Sync(DrainReason reason) {
    const auto r = static_cast<u32>(reason);
    ++counters.drains[r];
    Publish();
    const u64 ticks = WaitDone(committed_seq);
    if (ticks != 0) {
        ++counters.drain_waits[r];
        counters.drain_ticks[r] += ticks;
    }
}

void Channel::PostRunAhead(PostKind kind) {
    ASSERT(!record_open);
    ++counters.posted[static_cast<u32>(kind)];
    if (kind == PostKind::GfxPacket || kind == PostKind::AscPacket) {
        last_posted_packet_seq = committed_seq;
    }
    posted_seqs[posts & (PostSlots - 1)] = committed_seq;
    ++posts;
    if (++unpublished_posts >= pub_batch) {
        Publish();
    }
    // oldest_post only moves forward with done_seq: while the stale count is within the depth,
    // nothing can need a wait, so done_seq (a line the back writes) is not touched.
    if (posts - oldest_post > depth) {
        u64 done = done_seq.load(std::memory_order_acquire);
        while (oldest_post < posts && posted_seqs[oldest_post & (PostSlots - 1)] <= done) {
            ++oldest_post;
        }
        if (posts - oldest_post > depth) {
            // Wait until at most `depth` posted records are outstanding: the one `depth + 1`
            // posts back (the newest that must be done) is kept because PostSlots > MaxDepth.
            Publish(); // the target may still be unpublished (SHADPS4_CP_SPLIT_PUB_BATCH)
            const u64 target = posted_seqs[(posts - depth - 1) & (PostSlots - 1)];
            ++counters.depth_waits;
            counters.depth_wait_ticks += WaitDone(target);
            oldest_post = posts - depth;
            done = done_seq.load(std::memory_order_acquire);
        }
        counters.max_lead_seq = std::max(counters.max_lead_seq, committed_seq - done);
    }
    counters.max_lead = std::max(counters.max_lead, posts - oldest_post);
}

void Channel::SyncIfPostedPackets(DrainReason reason) {
    if (last_posted_packet_seq > done_seq.load(std::memory_order_acquire)) {
        Sync(reason);
    }
}

void Channel::PrunePostedWrites(u64 done) {
    while (pw_head != pw_tail && posted_writes[pw_head & (PostedWriteSlots - 1)].seq <= done) {
        ++pw_head;
    }
}

void Channel::NotePostedWrite(VAddr dst, u64 bytes, bool has_value, u32 value, bool fence,
                              u8 src) {
    ASSERT(!record_open);
    if (bytes == 0) {
        return;
    }
    PrunePostedWrites(done_seq.load(std::memory_order_acquire));
    if (pw_tail - pw_head == PostedWriteSlots) {
        // Full of outstanding writes: forget the oldest, and let every check drain while it may
        // still be outstanding (GuardSpan). Seqs grow along the log, so the newest dropped seq
        // bounds all dropped ones.
        pw_overflow_seq = posted_writes[pw_head & (PostedWriteSlots - 1)].seq;
        ++pw_head;
        ++counters.write_log_overflows;
    }
    posted_writes[pw_tail & (PostedWriteSlots - 1)] =
        PostedWrite{committed_seq, dst, dst + bytes, value, has_value && bytes >= 4, fence, src,
                    note_queue};
    ++pw_tail;
    counters.write_log_max = std::max(counters.write_log_max, pw_tail - pw_head);
}

bool Channel::BeginGuardCheck(DrainReason reason) {
    ++counters.guard_checks[static_cast<u32>(reason)];
    const u64 done = done_seq.load(std::memory_order_acquire);
    PrunePostedWrites(done);
    return pw_head != pw_tail || pw_overflow_seq > done;
}

bool Channel::GuardSpan(VAddr lo, VAddr hi, DrainReason reason) {
    if (lo >= hi) {
        return false;
    }
    // done_seq only grows, so everything BeginGuardCheck kept may still be outstanding; a write
    // that finished since is compared needlessly, which can only cost a drain.
    if (pw_overflow_seq > done_seq.load(std::memory_order_acquire)) {
        Sync(DrainReason::TaskGuardOverflow);
        return true;
    }
    for (u64 i = pw_head; i != pw_tail; ++i) {
        const PostedWrite& write = posted_writes[i & (PostedWriteSlots - 1)];
        if (write.lo < hi && lo < write.hi) {
            Sync(reason);
            return true;
        }
    }
    return false;
}

u64 Channel::OutstandingWriteSeq(VAddr lo, VAddr hi) {
    const u64 done = done_seq.load(std::memory_order_acquire);
    PrunePostedWrites(done);
    if (pw_overflow_seq > done) {
        return ~0ull;
    }
    u64 seq = 0;
    for (u64 i = pw_head; i != pw_tail; ++i) {
        const PostedWrite& write = posted_writes[i & (PostedWriteSlots - 1)];
        if (write.lo < hi && lo < write.hi) {
            seq = write.seq; // seqs grow along the log: the last match is the newest
        }
    }
    return seq;
}

bool Channel::NewestWriteValue(VAddr lo, u64& seq, u32& value, bool& fence, u32* why) {
    const u64 done = done_seq.load(std::memory_order_acquire);
    PrunePostedWrites(done);
    if (pw_overflow_seq > done) {
        return false;
    }
    const PostedWrite* newest = nullptr;
    for (u64 i = pw_head; i != pw_tail; ++i) {
        const PostedWrite& write = posted_writes[i & (PostedWriteSlots - 1)];
        if (write.lo < lo + 4 && lo < write.hi) {
            newest = &write;
        }
    }
    if (!newest || !newest->known || newest->lo != lo || newest->hi < lo + 4) {
        if (why) {
            // 0 unknown value (+ 16 * source), 1 not at the label start / partial
            *why = newest && !newest->known ? 0u + 16u * newest->src : 1u;
        }
        return false;
    }
    seq = newest->seq;
    value = newest->value;
    fence = newest->fence;
    return true;
}

int Channel::NewestWriterQueue(VAddr lo) {
    PrunePostedWrites(done_seq.load(std::memory_order_acquire));
    int q = -1;
    for (u64 i = pw_head; i != pw_tail; ++i) {
        const PostedWrite& write = posted_writes[i & (PostedWriteSlots - 1)];
        if (write.lo < lo + 4 && lo < write.hi) {
            q = write.queue;
        }
    }
    return q;
}

void Channel::WaitThrough(u64 seq, DrainReason reason) {
    const auto r = static_cast<u32>(reason);
    ++counters.drains[r];
    Publish();
    const u64 ticks = WaitDone(seq);
    if (ticks != 0) {
        ++counters.drain_waits[r];
        counters.drain_ticks[r] += ticks;
    }
}

void Channel::ArmDrainProbe() {
    if (probe_seq.load(std::memory_order_relaxed) != ~0ull ||
        done_seq.load(std::memory_order_acquire) >= committed_seq) {
        return;
    }
    probe_t0.store(Ticks(), std::memory_order_relaxed);
    probe_seq.store(committed_seq, std::memory_order_release);
}

void Channel::EmitRegWrite(u32 word_offset, const u32* data, u32 count) {
    ASSERT(u64{word_offset} + count <= Regs::NumRegs);
    if (count == 0) {
        return; // Nothing to copy; a count-0 segment would also end a batched record early.
    }
    if (hold) [[unlikely]] {
        held.push_back(1u);
        held.push_back(word_offset);
        held.push_back(count);
        held.insert(held.end(), data, data + count);
        return;
    }
    if (canon) [[unlikely]] {
        for (u32 i = 0; i < count; ++i) {
            canon->push_back((u64{1} << 56) | (u64{word_offset + i} << 32) | data[i]);
        }
    }
    const u32 seg = u32(sizeof(RegWritePayload)) + count * 4;
    if (reg_batch && !record_open && reg_seq == emitted && reg_seq != 0) {
        // Append to the newest record if it is our RegWrite and still unpublished: the back cannot
        // see it until write_pos passes it, so growing it in place is invisible. Only when it fits
        // without wrapping and without waiting for space (a wait could publish it).
        const u32 offset = u32(reg_pos & mask);
        const u32 used = reg_used + seg;
        const u32 size = AlignUp(used, RecordAlign);
        if (write_pos.load(std::memory_order_relaxed) <= reg_pos && offset + size <= capacity &&
            size <= MaxRegBatchBytes && reg_pos + size - cached_read <= capacity) {
            auto* header = reinterpret_cast<RecordHeader*>(storage + offset);
            auto* payload = reinterpret_cast<RegWritePayload*>(storage + offset + reg_used);
            payload->word_offset = word_offset;
            payload->count = count;
            std::memcpy(payload + 1, data, std::size_t{count} * 4);
            if (size - used >= sizeof(RegWritePayload)) {
                *reinterpret_cast<RegWritePayload*>(storage + offset + used) = {0, 0};
            }
            counters.bytes += size - header->size;
            header->size = size;
            reg_used = used;
            reserve_pos = committed_pos = reg_pos + size;
            ++counters.reg_appends;
            return;
        }
    }
    auto* payload = Reserve<RegWritePayload>(RecordType::RegWrite, seg);
    payload->word_offset = word_offset;
    payload->count = count;
    std::memcpy(payload + 1, data, std::size_t{count} * 4);
    auto* header = reinterpret_cast<RecordHeader*>(payload) - 1;
    const u32 used = u32(sizeof(RecordHeader)) + seg;
    if (header->size - used >= sizeof(RegWritePayload)) {
        *reinterpret_cast<RegWritePayload*>(reinterpret_cast<u8*>(header) + used) = {0, 0};
    }
    reg_pos = reserve_pos - header->size;
    reg_seq = header->seq;
    reg_used = used;
    Commit();
}

void Channel::EmitClearState() {
    Reserve(RecordType::ClearState, 0);
    Commit();
}

void Channel::EmitExtent(u32 slot, u32 raw) {
    if (hold) [[unlikely]] {
        held.push_back(2u);
        held.push_back(slot);
        held.push_back(raw);
        return;
    }
    auto* payload = Reserve<ExtentPayload>(RecordType::Extent);
    payload->slot = slot;
    payload->raw = raw;
    Commit();
}

void Channel::BeginHold() {
    if (!hold) {
        hold = true;
        held.clear();
    }
}

std::vector<u32> Channel::TakeHold() {
    hold = false;
    std::vector<u32> out = std::move(held);
    held.clear();
    return out;
}

void Channel::ReplayHold(const std::vector<u32>& h) {
    size_t i = 0;
    while (i + 3 <= h.size()) {
        const u32 kind = h[i], a = h[i + 1], b = h[i + 2];
        i += 3;
        if (kind == 1u) {
            ASSERT(i + b <= h.size());
            EmitRegWrite(a, h.data() + i, b);
            i += b;
        } else {
            EmitExtent(a, b);
        }
    }
}

void Channel::FlushHold() {
    if (hold) {
        const std::vector<u32> h = TakeHold();
        ReplayHold(h);
    }
}

void Channel::EmitEpochBump(u32 opcode) {
    auto* payload = Reserve<EpochPayload>(RecordType::EpochBump);
    payload->opcode = opcode;
    // SHADPS4_NATIVE_COHERENT: the dirty-log tail now (low 32 bits); for a submission start it
    // bounds what the back's coherence point must upload.
    payload->pad = static_cast<u32>(VideoCore::DirtyLog::Start());
    Commit();
}

void Channel::PostRetire() {
    Reserve(RecordType::Retire, 0);
    Commit();
    Publish();
}

void Channel::FrontExited() {
    FlushHold();
    Sync(DrainReason::FrontExit);
    front_exited.store(true, std::memory_order_release);
    WakeBack();
}

void Channel::Publish() {
    unpublished_posts = 0;
    if (write_pos.load(std::memory_order_relaxed) == committed_pos) {
        return;
    }
    if (pre_publish) [[unlikely]] {
        pre_publish(hook_ctx); // L3: the open stretch ends before its draws become visible
    }
    // seq_cst store, then seq_cst load of back_parked: with the back's store of back_parked and
    // load of write_pos in WaitForWork, one of the two always sees the other (no lost wake-up).
    write_pos.store(committed_pos, std::memory_order_seq_cst);
    WakeBackIfParked();
    RingReaderBell();
}

void Channel::WaitForSpace(u64 end) {
    if (end - cached_read <= capacity) {
        return;
    }
    ++counters.full_waits;
    front_waiting.fetch_add(1, std::memory_order_acq_rel);
    RingReaderBell();
    const u64 t0 = Ticks();
    for (;;) {
        // The back's read position, held lower by any A2 worker still reading.
        cached_read = ReadFloor();
        if (end - cached_read <= capacity) {
            front_waiting.fetch_sub(1, std::memory_order_acq_rel);
            const u64 waited = Ticks() - t0;
            counters.full_wait_ticks += waited;
            Common::FaultMonitor::front_wait_ticks.fetch_add(waited, std::memory_order_relaxed);
            return;
        }
        // The back frees space only by consuming what has been published.
        Publish();
        _mm_pause();
        std::this_thread::yield();
    }
}

u64 Channel::WaitDone(u64 seq) {
    if (done_seq.load(std::memory_order_acquire) >= seq) {
        return 0;
    }
    Publish(); // SHADPS4_CP_SPLIT_PUB_BATCH: never wait on records the back cannot see
    const u64 t0 = Ticks();
    const u64 deadline = NowNs() + SpinNs();
    front_waiting.fetch_add(1, std::memory_order_acq_rel);
    RingReaderBell();
    struct WaitingScope {
        std::atomic<u32>& w;
        ~WaitingScope() {
            w.fetch_sub(1, std::memory_order_acq_rel);
        }
    } waiting_scope{front_waiting};
    for (u32 spins = 1;; ++spins) {
        if (done_seq.load(std::memory_order_acquire) >= seq) {
            const u64 waited = Ticks() - t0;
            Common::FaultMonitor::front_wait_ticks.fetch_add(waited, std::memory_order_relaxed);
            return waited;
        }
        if ((spins & 63) == 0 && NowNs() > deadline) {
            break;
        }
        _mm_pause();
    }
    ++counters.front_parks;
    for (;;) {
        front_parked.store(true, std::memory_order_seq_cst);
        const u64 current = done_seq.load(std::memory_order_seq_cst);
        if (current >= seq) {
            break;
        }
        done_seq.wait(current, std::memory_order_acquire);
    }
    front_parked.store(false, std::memory_order_relaxed);
    const u64 waited = Ticks() - t0;
    Common::FaultMonitor::front_wait_ticks.fetch_add(waited, std::memory_order_relaxed);
    return waited;
}

const RecordHeader* Channel::Next() {
    for (;;) {
        if (read == cached_write) {
            cached_write = write_pos.load(std::memory_order_acquire);
            if (read == cached_write) {
                // Nothing more to run: publish any batched done before the back can park.
                if (done_count != 0) {
                    PublishDone(pending_done);
                }
                return nullptr;
            }
        }
        const auto* header = reinterpret_cast<const RecordHeader*>(storage + (read & mask));
        if (header->type != RecordType::Wrap) {
            return header;
        }
        read += header->size;
        read_pos.store(read, std::memory_order_release);
    }
}

const RecordHeader* Channel::PeekAt(u64& pos) {
    for (;;) {
        if (pos == cached_write) {
            cached_write = write_pos.load(std::memory_order_acquire);
            if (pos == cached_write) {
                return nullptr;
            }
        }
        const auto* header = reinterpret_cast<const RecordHeader*>(storage + (pos & mask));
        if (header->type != RecordType::Wrap) {
            return header;
        }
        pos += header->size;
    }
}

void Channel::Done(const RecordHeader* record) {
    const u64 seq = record->seq;
    read += record->size;
    read_pos.store(read, std::memory_order_release);
    // Batched publication: the front only waits for seqs it has published, and the back always
    // publishes once it has consumed everything it has seen (read == cached_write), so every wait
    // is released within at most done_batch - 1 further records of back work.
    pending_done = seq;
    if (++done_count < done_batch && read != cached_write) {
        return;
    }
    PublishDone(seq);
}

void Channel::DoneThrough(u64 end_pos, u64 last_seq) {
    // Every released record was published, so write_pos >= end_pos; Next() needs
    // cached_write >= read to see "nothing more" as read == cached_write.
    if (cached_write < end_pos) {
        cached_write = write_pos.load(std::memory_order_acquire);
    }
    ASSERT_MSG(cached_write >= end_pos, "CpSplit: DoneThrough past the published end");
    read = end_pos;
    read_pos.store(read, std::memory_order_release);
    pending_done = last_seq;
    PublishDone(last_seq);
}

void Channel::PublishDone(u64 seq) {
    done_count = 0;
    // seq_cst: pairs with the front's store of front_parked / load of done_seq in WaitDone.
    done_seq.store(seq, std::memory_order_seq_cst);
    if (front_parked.load(std::memory_order_seq_cst)) {
        done_seq.notify_one();
    }
    if (seq >= probe_seq.load(std::memory_order_relaxed)) [[unlikely]] {
        // Only the back clears a probe and the front arms only a cleared one: no lost update.
        if (probe_seq.exchange(~0ull, std::memory_order_acq_rel) != ~0ull) {
            probe_ticks.fetch_add(Ticks() - probe_t0.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            probe_n.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool Channel::BackHasWork(const std::atomic<u32>& num_commands) const {
    return write_pos.load(std::memory_order_seq_cst) != read ||
           num_commands.load(std::memory_order_seq_cst) != 0;
}

void Channel::WaitForWork(const std::atomic<u32>& num_commands, const std::stop_token& stoken) {
    const auto done = [&] {
        return BackHasWork(num_commands) || stoken.stop_requested() || FrontHasExited();
    };
    if (done()) {
        return;
    }
    const u64 t0 = Ticks();
    const u64 deadline = NowNs() + SpinNs();
    for (u32 spins = 1;; ++spins) {
        if (done()) {
            const u64 waited = Ticks() - t0;
            back_idle_ticks.fetch_add(waited, std::memory_order_relaxed);
            Common::FaultMonitor::back_idle_ticks.fetch_add(waited, std::memory_order_relaxed);
            Common::FaultMonitor::back_idle_total_ticks.fetch_add(waited,
                                                                 std::memory_order_relaxed);
            return;
        }
        if ((spins & 63) == 0 && NowNs() > deadline) {
            break;
        }
        _mm_pause();
    }
    const u32 ticket = doorbell.load(std::memory_order_acquire);
    back_parked.store(true, std::memory_order_seq_cst);
    if (!done()) {
        doorbell.wait(ticket, std::memory_order_acquire);
    }
    back_parked.store(false, std::memory_order_relaxed);
    const u64 waited = Ticks() - t0;
    back_idle_ticks.fetch_add(waited, std::memory_order_relaxed);
    Common::FaultMonitor::back_idle_ticks.fetch_add(waited, std::memory_order_relaxed);
    Common::FaultMonitor::back_idle_total_ticks.fetch_add(waited, std::memory_order_relaxed);
}

void Channel::WakeBackIfParked() {
    if (back_parked.load(std::memory_order_seq_cst)) {
        WakeBack();
    }
}

void Channel::WakeBack() {
    doorbell.fetch_add(1, std::memory_order_seq_cst);
    doorbell.notify_one();
}

void Channel::MaybeLog() {
    if ((++log_calls & 1023) != 0 || quiet) {
        return;
    }
    const u64 now = NowNs();
    if (now - log_window_start_ns < 10'000'000'000ull) {
        return;
    }
    const u64 ticks_now = Ticks();
    const double ns_per_tick = double(now - log_window_start_ns) /
                               double(std::max<u64>(1, ticks_now - log_window_start_ticks));
    const auto ms = [ns_per_tick](u64 ticks) { return double(ticks) * ns_per_tick / 1e6; };
    const auto& c = counters;
    u64 drains = 0, drain_waits = 0, drain_ticks = 0;
    std::string per_reason;
    for (u32 r = 0; r < NumDrainReasons; ++r) {
        drains += c.drains[r];
        drain_waits += c.drain_waits[r];
        drain_ticks += c.drain_ticks[r];
        if (c.drains[r] != 0) {
            per_reason += fmt::format(" {}={}/{}/{:.1f}", DrainNames[r], c.drains[r],
                                      c.drain_waits[r], ms(c.drain_ticks[r]));
        }
    }
    const u64 idle = back_idle_ticks.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Render,
             "CpSplit: window_s={:.1f} mode={} depth={} records={} bytes={} posted_draws={} "
             "posted_pkts={} drains={} drain_waits={} front_wait_ms={:.1f} drain_ms={:.1f} "
             "depth_wait_ms={:.1f} depth_waits={} max_lead={} max_lead_records={} "
             "back_idle_ms={:.1f} full_waits={} full_wait_ms={:.1f} a2_slot_wait_ms={:.1f} "
             "front_parks={} reg_appends={} posted_gfx_disp={} "
             "posted_asc_disp={} posted_asc_pkts={} posted_fences={}",
             double(now - log_window_start_ns) / 1e9, Mode(), depth, c.records, c.bytes,
             c.posted[static_cast<u32>(PostKind::Draw)],
             c.posted[static_cast<u32>(PostKind::GfxPacket)], drains, drain_waits,
             ms(drain_ticks + c.depth_wait_ticks), ms(drain_ticks), ms(c.depth_wait_ticks),
             c.depth_waits, c.max_lead, c.max_lead_seq, ms(idle), c.full_waits,
             ms(c.full_wait_ticks), ms(std::exchange(slot_wait_ticks, 0)), c.front_parks,
             c.reg_appends, c.posted[static_cast<u32>(PostKind::GfxDispatch)],
             c.posted[static_cast<u32>(PostKind::AscDispatch)],
             c.posted[static_cast<u32>(PostKind::AscPacket)],
             c.posted[static_cast<u32>(PostKind::Fence)]);
    {
        const auto& r = Common::ReadbackOffload::g_regime;
        LOG_INFO(Render,
                 "FaultRegime: jobs={} seq_vetoes={} sync_fallbacks={} exact_clean={} "
                 "exact_touched={} exact_overrun={} (cumulative)",
                 r.jobs.load(std::memory_order_relaxed),
                 r.seq_vetoes.load(std::memory_order_relaxed),
                 r.sync_fallbacks.load(std::memory_order_relaxed),
                 r.exact_clean.load(std::memory_order_relaxed),
                 r.exact_touched.load(std::memory_order_relaxed),
                 r.exact_overrun.load(std::memory_order_relaxed));
    }
    std::string guard_checks;
    for (u32 r = 0; r < NumDrainReasons; ++r) {
        if (c.guard_checks[r] != 0) {
            guard_checks += fmt::format(" {}={}", DrainNames[r], c.guard_checks[r]);
        }
    }
    if (!guard_checks.empty()) {
        // TASK_GUARD: checks per reason; the hits are the drains of the same name above.
        guard_checks = fmt::format(" | guard_checks:{} write_log_max={} write_log_overflows={}",
                                   guard_checks, c.write_log_max, c.write_log_overflows);
    }
    LOG_INFO(Render, "CpSplit drains (n/waited/ms):{}{}", per_reason, guard_checks);
    counters = {};
    log_window_start_ns = now;
    log_window_start_ticks = ticks_now;
}

} // namespace AmdGpu::CpSplit
