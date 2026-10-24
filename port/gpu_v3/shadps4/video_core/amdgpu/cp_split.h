// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <atomic>
#include <cstring>
#include <memory>
#include <stop_token>
#include <vector>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

/**
 * Command-processor front/back split (notes/cp-split-plan-2026-09-25.md,
 * notes/cp-split-inventory-2026-09-25.md). DRAFT, default off.
 *
 * SHADPS4_CP_SPLIT unset or 0: none of this is constructed and the command processor runs as
 * before, on one thread.
 *
 * SHADPS4_CP_SPLIT=1 (depth 0) or 2 (run-ahead): two threads.
 *  - The BACK is the existing CP thread (Liverpool::process_thread). It keeps every identity the
 *    rest of the emulator relies on: the name "shadPS4:GpuCommandProcessor" (the stack sampler
 *    targets it), Liverpool::gpu_id (IsGpuThread, SendCommand runs inline and the guest command
 *    queue runs here), FaultMonitor::t_is_cp_thread, the CP core placement, the scheduler's single
 *    producer. It owns the buffer/texture caches, the scheduler, IrqC signals and every CP-side
 *    guest-memory write, and a mirror of the register file (BackMirror) that it keeps equal to the
 *    front's by applying the front's register journal.
 *  - The FRONT is a new thread "shadPS4:GpuCommandFront" (t_is_front). It runs the PM4 coroutines
 *    (DCB, CCB, ASC), owns Liverpool::regs, FilterDraw's classifier and the pipeline cache lookup,
 *    and appends records to an ordered SPSC ring. It never touches a cache, the scheduler or guest
 *    memory writes, and never sets gpu_id or t_is_cp_thread.
 *
 * Depth 0 (=1): the front publishes and waits for the back after every record that is not pure
 * register state (Sync). Register records ride along with the next synchronous record. So every
 * guest-visible effect happens in the same order and at the same PM4 position as today; only the
 * thread differs.
 *
 * Run-ahead (=2): graphics-queue draw records are posted (PostRunAhead) instead of synced, and the
 * front keeps parsing while the back executes them, bounded by SHADPS4_CP_SPLIT_DEPTH outstanding
 * posted records (default 64). Everything else still syncs, as at depth 0 (a Sync waits for the
 * back to finish EVERYTHING published, so it is a full drain). With SHADPS4_CP_SPLIT_POST_DMA=1,
 * gfx DMA_DATA packets are posted too (see Liverpool::RunGfxDma for the guards). With
 * SHADPS4_CP_SPLIT_POST_ASC=1, dispatches (gfx and ASC) are posted as owned Dispatch records
 * (ComputeProgram copy, stage copy, pipeline pointer), and ASC DMA_DATA (and optionally
 * WRITE_DATA) as Packet records, with the same rest-of-buffer guard (Liverpool::RunAscPacket).
 * Drain points (Sync with a DrainReason) are where the front reads what the back produces:
 * WAIT_REG_MEM, COND_EXEC, MEM_SEMAPHORE waits, the VO label wait, task done (num_submits is the
 * guest's "GPU idle"), the idle tail, every Call record, and read faults taken on the front
 * (Liverpool::DrainForFrontFault). Draw records own everything the back reads besides the mirror:
 * args, class, pipeline pointer (stable: only ReplaceShader erases), ctx epoch, per-stage user
 * data / flat buffer / pgm_base copies; the Info pointer is used for its immutable part only.
 */
namespace AmdGpu::CpSplit {

/// SHADPS4_CP_SPLIT: 0 = off (default), 1 = split at depth 0, 2 = run-ahead. Read once.
int Mode();
inline bool Enabled() {
    return Mode() > 0;
}
inline bool RunAhead() {
    return Mode() >= 2;
}
/// Largest SHADPS4_CP_SPLIT_DEPTH accepted (the front keeps the seq of this many posted records).
constexpr u32 MaxDepth = 1024;
/// SHADPS4_CP_SPLIT_DEPTH: most posted records the back may owe the front (run-ahead only).
/// Default 64, clamped to [1, MaxDepth].
u32 Depth();
/// SHADPS4_CP_SPLIT_POST_DMA=1: run-ahead also posts gfx DMA_DATA (default: they sync).
bool PostDma();
/// SHADPS4_CP_SPLIT_POST_ASC=1: run-ahead also posts dispatches (ASC and gfx) and ASC packets as
/// owned records instead of synchronous Call records (default off). Which classes is set by
/// SHADPS4_CP_SPLIT_POST_ASC_MASK (PostAscBits, default PostAscDefaultMask), for bisecting.
bool PostAsc();
enum PostAscBits : u32 {
    PostAscDispatch = 1u << 0,  ///< ASC DISPATCH_DIRECT / DISPATCH_INDIRECT (incl. DMA-patched).
    PostGfxDispatch = 1u << 1,  ///< Gfx-queue DISPATCH_DIRECT / DISPATCH_INDIRECT.
    PostAscDma = 1u << 2,       ///< ASC DMA_DATA (guarded like gfx POST_DMA).
    PostAscWriteData = 1u << 3, ///< ASC WRITE_DATA (guarded like gfx POST_DMA). Not default.
};
constexpr u32 PostAscDefaultMask = PostAscDispatch | PostGfxDispatch | PostAscDma;
/// SHADPS4_CP_SPLIT_POST_ASC_MASK (decimal or 0x hex). 0 when POST_ASC is off.
u32 PostAscMask();
/// SHADPS4_CP_SPLIT_POST_TASK_DONE=1 (run-ahead only): a finished submission posts a Retire record
/// instead of draining; the guest-visible idle state waits for the back to retire it.
bool PostTaskDone();
/// SHADPS4_CP_SPLIT_TASK_GUARD (0-3, only with POST_TASK_DONE): replaces POST_TASK_DONE's
/// task-end SyncIfPostedPackets with range checks. The front logs the destination of every posted
/// DMA_DATA / WRITE_DATA (NotePostedWrite) and drains only when an outstanding one overlaps a
/// command buffer it is about to parse or resume.
///  1: task end no longer drains; every Process* level checks its buffer at entry; ProcessLoop
///     checks the unparsed rest of every active buffer before resuming a task. Nested-IB guards
///     unchanged (conservative).
///  2: + the IB-start guards (gfx IbStart, AscIbStart) become the IB's own entry check.
///  3: + the IB-end guards (IbEnd) check only the parent's unparsed rest.
u32 TaskGuard();
/// SHADPS4_CP_SPLIT_POST_FENCE=1 (only with TASK_GUARD): gfx EVENT_WRITE_EOP / EVENT_WRITE_EOS and
/// ASC RELEASE_MEM are posted instead of synchronous. The label write and the interrupt happen on
/// the back in stream order, as before; only the front no longer waits for them. Every front
/// reader of a label (WAIT_REG_MEM, COND_EXEC, semaphores) drains first; the label is noted in
/// the TASK_GUARD write log, and a label inside the unparsed rest of its own buffer syncs.
/// GDS-sourced fences stay synchronous.
bool PostFence();
/// SHADPS4_CP_SPLIT_DISP_GUARD=0 drops POST_ASC's drain before a posted dispatch's front lookup
/// when posted DMA / WRITE_DATA is outstanding (disp_after_pkt), as draws never had it. Default 1.
/// The drain also covered the user data / flat buffer / pgm_base copied into the record. VERIFY's
/// periodic synced dispatch never took this drain (it is not posted), so its back re-lookup
/// already measures the unguarded case at either setting.
bool DispGuard();
/// SHADPS4_CP_SPLIT_VERIFY=1. Depth 0: at every draw/dispatch record (the front is parked) the
/// back compares its whole register mirror, CB/DB extents, context epoch and cs snapshot with the
/// front's, and each copied stage (user data, flat buffer, pgm_base) with the Info the front's
/// lookup just refreshed. Run-ahead: each posted draw carries a copy of the register windows the
/// back reads (gfx SH, context, uconfig up to the VGT draw state) plus the extents, and the back
/// memcmps its mirror against it; every VerifyEvery()-th draw is synced and gets the full depth-0
/// check. The first mismatch is logged with the word offset and asserts.
bool Verify();
/// SHADPS4_CP_SPLIT_VERIFY_EVERY: run-ahead VERIFY syncs one draw in this many (default 256).
u32 VerifyEvery();
/// SHADPS4_CP_SPLIT_DONE_BATCH (default 1, max 1024): the back publishes done_seq every N records,
/// and always when it has consumed everything it has seen published.
u32 DoneBatch();
/// SHADPS4_CP_SPLIT_REG_BATCH=1: consecutive register writes are appended to the newest RegWrite
/// record while it is unpublished (one record, several segments). Default off.
bool RegBatch();
/// SHADPS4_CP_SPLIT_FRONT_DERIVE: 0 = off (default); 1 = the front derives the vertex inputs of
/// each draw (VertexInputsPayload) and the back uses them; 2 = verify: the back derives as
/// today, compares with the record, counts and logs mismatches, and uses its own result.
/// Read once.
int FrontDerive();
/// SHADPS4_CP_SPLIT_FRONT_DERIVE_VI_SKIP=0 (bisect switch, default on): mode 1 no longer skips
/// the setVertexInputEXT of a draw whose description the command buffer already holds.
bool FrontDeriveVertexInputSkip();

/// True on the front thread (set by Liverpool::FrontProcess). The fault handlers use it to drain
/// before a read fault downloads GPU data (inventory 4.6).
inline thread_local bool t_is_front = false;

/// The back's copy of the front state it reads (Liverpool::BackRegs() and friends).
struct BackMirror {
    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};
    CbDbExtent db_extent{};
    u64 ctx_epoch{};
};

enum class RecordType : u16 {
    Wrap,       ///< Filler to the end of the ring; the next record starts at offset 0.
    RegWrite,   ///< Segments of RegWritePayload + count dwords, copied into the back's mirror;
                ///< a count-0 segment (or less than a segment left) ends the record.
    ClearState, ///< mirror.regs.SetDefaults().
    Extent,     ///< ExtentPayload.
    EpochBump,  ///< EpochPayload: VideoCore::BumpSubmitEpoch in stream order.
    Draw,       ///< Liverpool's draw record + per-stage data (defined in liverpool.cpp).
    Dispatch,   ///< Liverpool's dispatch record + one stage (defined in liverpool.cpp).
    Packet,     ///< A verbatim copy of one gfx PM4 type-3 packet, run by ExecuteGfxPacket.
    Call,       ///< CallPayload: a function on the front's stack. Always synchronous.
    Retire,     ///< No payload: a guest submission is complete (POST_TASK_DONE); the back
                ///< retires it for WaitGpuIdle.
    Diag,       ///< DiagPayload: a diagnostic marker in stream order (SHADPS4_ASC_RING_LATENCY).
    Segment,    ///< SHADPS4_NATIVE_SEGX: SplitSegmentRecord, a segment of native colour members.
    DcbBegin,   ///< SHADPS4_A1: DcbBeginPayload, a top-level guest command buffer starts here.
    DcbEnd,     ///< SHADPS4_A1: DcbEndPayload, the command buffer's last record came before.
    DcbRef,     ///< SHADPS4_LEAN=2: DcbRefPayload, the DCB's head span is in a lean job channel.
    LeanEnd,    ///< SHADPS4_LEAN=2: the last record of a lean job channel's span (no payload).
};
/// SHADPS4_LEAN=2: the back follows lean job slot `slot` (DCB `index`) to its LeanEnd.
struct DcbRefPayload {
    u32 slot;
    u32 index;
};

/// Flags for Packet records.
enum RecordFlags : u16 {
    GuestMarkers = 1 << 0,
    AscReleaseMem = 1 << 1, ///< POST_FENCE: an ASC RELEASE_MEM; bits 8-15 hold the pipe id.
};

/// Why the front waited for the back to finish everything published (counters only).
enum class DrainReason : u8 {
    DepthZero,       ///< Depth 0: every draw is synchronous.
    VerifyFull,      ///< Run-ahead VERIFY: the periodic synchronous draw.
    Dispatch,        ///< Gfx-queue dispatch record.
    PacketDma,       ///< Gfx DMA_DATA (run-ahead without POST_DMA).
    PacketDmaGuard,  ///< POST_DMA: this DMA writes into the rest of the command buffer.
    PacketFence,     ///< EVENT_WRITE_EOP / EOS.
    PacketWriteData, ///< Gfx WRITE_DATA.
    PacketOther,     ///< Flip / guest-marker NOP, ZPASS_DONE, PFP_SYNC_ME.
    Call,            ///< Untagged Call records (the tagged ones are the Call* reasons below).
    IdleTail,        ///< The submission loop's idle tail (Call).
    WaitRegMem,      ///< Gfx/ASC WAIT_REG_MEM (and the VO label wait).
    CondExec,
    MemSemaphore,
    TaskDone,        ///< Before --num_submits.
    IbStart,         ///< POST_DMA: an INDIRECT_BUFFER while posted packets are outstanding.
    FrontFault,      ///< A fault taken on the front (DrainForFrontFault).
    FrontExit,
    // POST_ASC (run-ahead):
    VerifyDispatch,    ///< VERIFY: the periodic synchronous dispatch.
    AscDmaGuard,       ///< An ASC DMA_DATA writing into the rest of its command buffer.
    AscWriteDataGuard, ///< An ASC WRITE_DATA writing into the rest of its command buffer.
    AscIbStart,        ///< An ASC INDIRECT_BUFFER while posted packets are outstanding.
    // Call records, by origin (together they are what the plain `call` counter used to be).
    CallAscDispatch,         ///< ASC DISPATCH_DIRECT.
    CallAscDispatchIndirect, ///< ASC DISPATCH_INDIRECT.
    CallAscDispatchPatched,  ///< ASC DISPATCH_DIRECT whose dims a DMA patched (indirect).
    CallAscDma,              ///< ASC DMA_DATA.
    CallAscWriteData,        ///< ASC WRITE_DATA.
    CallAscRelease,          ///< ASC RELEASE_MEM.
    CallAscSemaphore,        ///< ASC MEM_SEMAPHORE signal / decrement.
    CallDumpConstRam,        ///< CE DUMP_CONST_RAM.
    IbEnd,            ///< After an INDIRECT_BUFFER task: posted packets may patch the parent level.
    DispatchAfterPkt, ///< POST_ASC: a posted dispatch's front lookup after posted DMA/WRITE_DATA.
    // TASK_GUARD (a posted write overlaps a command buffer about to be parsed):
    TaskGuard,         ///< At the entry of a top-level DCB / CCB / ACB.
    TaskGuardIb,       ///< At the entry of a nested IB (gfx, CE, ASC).
    TaskGuardResume,   ///< An active buffer's unparsed rest, before ProcessLoop resumes a task.
    TaskGuardOverflow, ///< The write log (or span registry) overflowed: conservative drain.
    WaitRegMemPartial, ///< SHADPS4_WRM_PARTIAL: WAIT_REG_MEM waits through its label's last posted write.
    Count,
};

/// What a posted record is (counters). The packet kinds also arm SyncIfPostedPackets.
enum class PostKind : u8 {
    Draw,
    GfxPacket,   ///< Gfx DMA_DATA (POST_DMA).
    GfxDispatch, ///< POST_ASC.
    AscDispatch, ///< POST_ASC.
    AscPacket,   ///< ASC DMA_DATA / WRITE_DATA (POST_ASC).
    Fence,       ///< POST_FENCE: gfx EOP / EOS, ASC RELEASE_MEM. Does not arm SyncIfPostedPackets.
    Count,
};

struct RecordHeader {
    RecordType type;
    u16 flags;
    u32 size; ///< Header included, multiple of RecordAlign.
    u64 seq;  ///< 1-based record number (Wrap: 0).
};
static_assert(sizeof(RecordHeader) == 16);

constexpr u32 RecordAlign = 16;
/// Largest RegWrite record grown by SHADPS4_CP_SPLIT_REG_BATCH appends.
constexpr u32 MaxRegBatchBytes = 4096;

struct RegWritePayload {
    u32 word_offset;
    u32 count;
    // u32 data[count] follows
};
struct ExtentPayload {
    u32 slot; ///< 0-7 = CB, 8 = DB.
    u32 raw;
};
struct EpochPayload {
    u32 opcode;
    u32 pad;
};
struct CallPayload {
    void (*invoke)(void*);
    void* context;
};
/// SHADPS4_A1 (architecture A, notes/NATIVE-RENDERER.md): the start of one top-level guest
/// command buffer (one SubmitGfx); every record up to the next DcbBegin belongs to it.
struct DcbBeginPayload {
    u64 addr;    ///< guest address of the DCB (identification only, never read)
    u32 dwords;  ///< DCB size
    u32 has_ccb; ///< 1: a constant command buffer came with it
    u32 snap_words; ///< A1=2: the front's register windows follow (DcbSnapWindows), else 0
    u32 index;      ///< 1-based DCB number (front count)
};
/// A1=2 entry snapshot windows (word offset, count), in payload order.
constexpr std::array<std::pair<u32, u32>, 3> DcbSnapWindows{{
    {0x2000, 0x1000}, // config + SH
    {0xA000, 0x400},  // context
    {0xC000, 0x400},  // uconfig
}};
constexpr u32 DcbSnapWords = 0x1000 + 0x400 + 0x400;
struct DcbEndPayload {
    u32 index; ///< the DcbBegin's index
    u32 pad;
};
/// Diagnostic marker (never emitted unless a diagnostic flag is on).
struct DiagPayload {
    u32 kind;
    u32 queue;
    u64 values[6];
};

/// SHADPS4_NATIVE_SEGX (notes/native-path-2026-10-09.md sec. 23): one segment of consecutive
/// native colour members the front's segment builder closed. Payload: this header, then member
/// 0's whole Draw record (SplitDrawRecord + stage blocks + vertex-input block, `draw_bytes`,
/// padded to 16), then n_ops SegOp, then n indirect commands (VkDrawIndexedIndirectCommand:
/// indexCount, instanceCount, firstIndex relative to the index union lo, 0, 0), then n list
/// entries (slot << 24 | ring index: the member's job record, brick A). Unions: the index
/// range, each pulled vertex attribute's stream range (by attribute index, `vmask`), each
/// Guest binding slot's job-run union (`gmask`; a superset of the members' ranges).
enum SegOpKind : u32 {
    SegOpDrawRange = 1, ///< a: first member, b: count
    SegOpStencil = 2,   ///< a: DB_STENCILREFMASK, b: DB_STENCILREFMASK_BF (raw)
    SegOpCull = 3,      ///< a: PA_SU_SC_MODE_CNTL (raw)
};
struct SegOp {
    u32 kind;
    u32 a, b, c;
};
constexpr u32 SegMaxOps = 192;
struct SplitSegmentRecord {
    u32 id;
    u32 n;
    u32 n_ops;
    u32 draw_bytes;
    u32 index_type; ///< member 0's index_buffer_type.raw
    u32 index_size;
    u32 num_indices0;   ///< member 0's draw registers (the mirror holds the next draw's)
    u32 num_instances0;
    u64 index_base0;    ///< IndexBufferBase raw
    u64 ilo, ihi;
    u64 gmask;
    u32 vmask;
    u32 pad;
    std::array<u64, 16> vlo, vhi;
    std::array<u64, 40> glo, ghi;
    /// v3 item 4: the distinct materials (job record image-key hashes) in order of first
    /// appearance and each one's representative member (material 0: member 0).
    u32 n_mat;
    u32 pad2;
    std::array<u64, 8> mat_h1, mat_h2;
    std::array<u32, 8> mat_rep;
    u32 n_rec; ///< embedded member Draw records: 1 (member 0, =2) or n (=1, the replay fallback)
    u32 pad3[3];
    std::array<u32, 64> rec_off; ///< each embedded record's byte offset in the record area
};
static_assert(sizeof(SplitSegmentRecord) % 16 == 0);

/// Per-stage user data of one draw/dispatch, copied on the front right after the lookup
/// (inventory section 4.4). Followed by ud_dw user-data dwords, then flat_dw flat-buffer dwords;
/// the whole block is padded to 8 bytes.
struct StagePayload {
    const void* info; ///< const Shader::Info*: only its immutable part is read by the back.
    VAddr pgm_base;
    u32 logical_stage;
    u32 ud_dw;
    u32 flat_dw;
    u32 pad;
};
static_assert(sizeof(StagePayload) % 8 == 0);

/// Draw arguments the back needs besides the registers (the register-side ones - num_indices,
/// num_instances, index base/type - travel in the journal).
struct DrawArgs {
    bool is_indexed{};
    bool is_indirect{};
    u32 index_offset{};
    VAddr arg_address{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_address{};
};

/// FRONT_DERIVE: the vertex-input derivation of BufferCache::BindVertexBuffers (the
/// GraphicsPipeline::GetVertexInputs loop and the merged, sorted guest ranges), computed on the
/// front in EmitDraw from the fetch shader, the vertex stage's user data (the V# table reads the
/// stage copy's user data points at) and the front's VGT step-rate registers. Follows the stage
/// blocks of a draw record (SplitDrawRecord::vi_bytes), before the VERIFY snapshot. Layout:
/// VertexInputsPayload, VertexInputAttrib[count], VertexInputRange[range_count]; every part is a
/// multiple of 8 bytes. Stays on the back: ClampRangeSize, ObtainBuffer, IsRegionGpuModified and
/// the barriers, the recording.
struct VertexInputsPayload {
    u32 count;       ///< Fetch-shader attributes (= bindings). 0: an empty setVertexInputEXT.
    u32 range_count; ///< Merged guest ranges (buffers with a base and a size), sorted by base.
    u64 desc_id;     ///< 1-based; the front bumps it when the description part changes.
};
/// The description part: what setVertexInputEXT takes (a pure function of the fetch shader, the
/// V# format / stride and the step rates).
struct VertexInputDesc {
    u32 vk_format; ///< vk::Format of the V#'s data / number format.
    u32 stride;    ///< V# stride.
    u32 divisor;   ///< Instance divisor (1 unless over a step rate).
    u8 location;   ///< attrib.semantic (= binding).
    u8 instanced;  ///< 1: vk::VertexInputRate::eInstance.
    u16 pad;
};
constexpr u16 VertexInputNoRange = 0xffff;
struct VertexInputAttrib {
    VertexInputDesc desc;
    VAddr base_address; ///< V# base plus the fetch instruction offset.
    u32 size;           ///< V# size. Base 0 or size 0: no buffer, a null handle is bound.
    u16 range;          ///< Index of the merged range holding base_address, else NoRange.
    u16 pad;
};
struct VertexInputRange {
    VAddr base_address;
    VAddr end_address;
};
static_assert(sizeof(VertexInputsPayload) == 16 && sizeof(VertexInputAttrib) == 32 &&
              sizeof(VertexInputRange) == 16);
inline const VertexInputAttrib* VertexInputAttribs(const VertexInputsPayload& vi) {
    return reinterpret_cast<const VertexInputAttrib*>(&vi + 1);
}
inline const VertexInputRange* VertexInputRanges(const VertexInputsPayload& vi) {
    return reinterpret_cast<const VertexInputRange*>(VertexInputAttribs(vi) + vi.count);
}
/// Vulkan::MaxVertexBufferCount (checked where the derivation is defined).
constexpr u32 MaxVertexInputs = 32;
constexpr u32 VertexInputsBytes(u32 count, u32 range_count) {
    return static_cast<u32>(sizeof(VertexInputsPayload) + count * sizeof(VertexInputAttrib) +
                            range_count * sizeof(VertexInputRange));
}
/// A block at fixed capacity: the derivation's output (front), or the back's own derivation in
/// verify mode.
struct VertexInputsDerived {
    VertexInputsPayload header{};
    std::array<VertexInputAttrib, MaxVertexInputs> attribs{};
    std::array<VertexInputRange, MaxVertexInputs> ranges{};

    u32 Bytes() const {
        return VertexInputsBytes(header.count, header.range_count);
    }
    /// Copies the block to `out` (8-aligned). Returns the end.
    u8* Write(u8* out) const {
        std::memcpy(out, &header, sizeof(header));
        out += sizeof(header);
        std::memcpy(out, attribs.data(), header.count * sizeof(VertexInputAttrib));
        out += header.count * sizeof(VertexInputAttrib);
        std::memcpy(out, ranges.data(), header.range_count * sizeof(VertexInputRange));
        return out + header.range_count * sizeof(VertexInputRange);
    }
};

/**
 * Single-producer (front) / single-consumer (back) byte ring plus the two wait points:
 * the back waiting for work and the front waiting for the back (Sync, the depth bound).
 * Waits spin briefly with _mm_pause, then park on std::atomic::wait (WaitOnAddress).
 *
 * A record is Reserve()d, filled, then Commit()ted; only committed records are ever published,
 * so a Sync from a fault handler that interrupted the front between Reserve and Commit (the
 * verbatim DCB copy in RunGfxPacket is the one guest-memory read in that window) publishes and
 * waits for everything but the open record.
 */
class Channel {
public:
    static constexpr u32 DefaultRingBytes = 4u << 20;

    explicit Channel(u32 ring_bytes = DefaultRingBytes);
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // ---- Front (the only producer) ----

    /// Reserves one record with `payload_bytes` of payload and returns the payload (16-aligned,
    /// uninitialised). Invisible to the back until committed and published. Waits while the ring
    /// is full.
    void* Reserve(RecordType type, u32 payload_bytes, u16 flags = 0);
    template <typename T>
    T* Reserve(RecordType type, u32 payload_bytes = sizeof(T), u16 flags = 0) {
        static_assert(alignof(T) <= RecordAlign);
        return static_cast<T*>(Reserve(type, payload_bytes, flags));
    }
    /// Ends the record reserved last. Does not publish: it rides along with the next Post/Sync.
    void Commit();
    /// Publishes everything committed and waits until the back has executed all of it.
    void Sync(DrainReason reason);
    /// Run-ahead: publishes everything committed without waiting, then waits only while more
    /// than Depth() posted records are outstanding. `kind` selects the counter.
    void PostRunAhead(PostKind kind);
    /// Sync(reason) if a posted packet record (POST_DMA / POST_ASC) may not have executed yet.
    void SyncIfPostedPackets(DrainReason reason);

    /// TASK_GUARD: the packet record committed last (about to be posted) writes [dst, dst+bytes).
    /// Call it for every posted packet with a memory destination (GDS destinations cannot overlap
    /// guest memory). Entries are dropped once done_seq passes them; if PostedWriteSlots are
    /// outstanding the oldest is dropped and checks turn conservative until it is done.
    /// `src` (census): 0 other, 1 fence, 2 gfx WRITE_DATA, 3 gfx DMA, 4 ASC packet, 5 ZPASS_DONE.
    void NotePostedWrite(VAddr dst, u64 bytes, bool has_value = false, u32 value = 0,
                         bool fence = false, u8 src = 0);
    /// SHADPS4_CP_SPLIT_POST_ORDER: true when the newest outstanding noted write overlapping
    /// [lo, lo + 4) starts at lo, covers it and its first dword is known (`seq`, `value`).
    bool NewestWriteValue(VAddr lo, u64& seq, u32& value, bool& fence, u32* why = nullptr);
    /// SHADPS4_XQ_CENSUS: the queue of the newest outstanding noted write overlapping [lo, lo + 4)
    /// (0 gfx, vqid + 1 ASC), or -1 when none may be outstanding.
    int NewestWriterQueue(VAddr lo);
    /// SHADPS4_XQ_CENSUS: the queue the front is parsing (set at the top of its packet loops).
    u8 note_queue{0};
    /// SHADPS4_ASC_EARLY (compute lane step 0): records committed while the front parses an ASC
    /// queue (asc_tag != 0) are pushed here in commit order; the back runs the eligible ones ahead
    /// of the graphics backlog. Front: early_tail; back: early_head.
    struct EarlyItem {
        u64 pos;
        u64 seq;
        RecordType type;
    };
    static constexpr u64 EarlySlots = 16384;
    std::unique_ptr<EarlyItem[]> early_items; ///< allocated by EnableEarly()
    alignas(64) std::atomic<u64> early_tail{0};
    alignas(64) std::atomic<u64> early_head{0};
    u8 asc_tag{0};
    u64 early_full_waits{0};
    void EnableEarly() {
        early_items = std::make_unique<EarlyItem[]>(EarlySlots);
    }
    /// TASK_GUARD: counts a check under `reason`, forgets executed writes, and returns whether any
    /// noted write may still be outstanding (false: nothing to compare, no drain needed).
    bool BeginGuardCheck(DrainReason reason);
    /// After BeginGuardCheck returned true: if an outstanding write overlaps [lo, hi), Sync(reason)
    /// (Sync(TaskGuardOverflow) if the log overflowed) and return true.
    bool GuardSpan(VAddr lo, VAddr hi, DrainReason reason);
    /// SHADPS4_WRM_PARTIAL: the seq of the newest outstanding noted write overlapping [lo, hi);
    /// 0 when none may be outstanding, ~0 when the log overflowed and a dropped one may be.
    u64 OutstandingWriteSeq(VAddr lo, VAddr hi);
    /// Publishes and waits until done_seq >= seq (counted under `reason` like a drain).
    void WaitThrough(u64 seq, DrainReason reason);
    /// WAIT_REG_MEM census: the back measures how long it takes from now until it is done with
    /// everything committed now (what a full drain would wait). One probe at a time.
    void ArmDrainProbe();
    /// {probes completed, their ticks} since the last call.
    std::pair<u64, u64> TakeDrainProbe() {
        return {probe_n.exchange(0, std::memory_order_relaxed),
                probe_ticks.exchange(0, std::memory_order_relaxed)};
    }

    void EmitRegWrite(u32 word_offset, const u32* data, u32 count);
    void EmitClearState();
    void EmitExtent(u32 slot, u32 raw);
    /// SHADPS4_NATIVE_SEGX (brick C v3 item 3): while a hold is on, EmitRegWrite / EmitExtent
    /// append to a side buffer instead of the ring, so a Segment record posted later can still
    /// precede them in stream order. ReplayHold re-emits them in order. Reserve() of any record
    /// other than RegWrite / Extent / Packet / Diag / Retire while holding calls the interrupt
    /// first (the segment builder closes its segment, which takes and replays the hold itself).
    void BeginHold();
    bool Holding() const {
        return hold;
    }
    std::vector<u32> TakeHold();
    void ReplayHold(const std::vector<u32>& held);
    void FlushHold();
    void SetHoldInterrupt(void (*fn)(void*), void* context) {
        hold_cb = fn;
        hold_ctx = context;
    }
    void EmitEpochBump(u32 opcode);

    /// Runs `fn` on the back while the front waits. `fn` lives on the front's stack, which is
    /// sound because the Sync waits for it at any depth: the front is parked until it returns.
    template <typename F>
    void CallSync(F& fn, DrainReason reason) {
        auto* call = Reserve<CallPayload>(RecordType::Call);
        call->invoke = [](void* context) { (*static_cast<F*>(context))(); };
        call->context = static_cast<void*>(&fn);
        Commit();
        Sync(reason);
    }

    /// SHADPS4_A2 (front): unmasked ring position of the record reserved last.
    [[nodiscard]] u64 LastRecordPos() const noexcept {
        return last_record_pos;
    }
    /// SHADPS4_A2 (any thread): the published record at `pos` (Wrap records skipped, `pos`
    /// moved past them). The caller guarantees `pos` lies in published, not yet done records.
    const RecordHeader* RecordAtPos(u64& pos) const {
        for (;;) {
            const auto* header = reinterpret_cast<const RecordHeader*>(storage + (pos & mask));
            if (header->type != RecordType::Wrap) {
                return header;
            }
            pos += header->size;
        }
    }
    /// SHADPS4_A2 (any thread): the published end (unmasked). A record whose header position is
    /// below it is complete and immutable until the back releases it.
    /// Records emitted so far (front) / the newest seq the back has published as done.
    [[nodiscard]] u64 EmittedSeq() const noexcept {
        return emitted;
    }
    /// SHADPS4_LEAN=2: the producer's ring-full and depth wait ticks since the last call.
    std::pair<u64, u64> TakeWaitTicks() {
        return {std::exchange(counters.full_wait_ticks, 0), std::exchange(counters.depth_wait_ticks, 0)};
    }
    /// Back: publish batched done seqs now (SHADPS4_LEAN=2, before waiting on a job channel).
    void FlushDone() {
        if (done_count != 0) {
            PublishDone(pending_done);
        }
    }
    [[nodiscard]] u64 CommittedSeq() const noexcept {
        return committed_seq;
    }
    [[nodiscard]] u64 DoneSeq() const noexcept {
        return done_seq.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 PublishedEnd() const noexcept {
        return write_pos.load(std::memory_order_acquire);
    }
    /// SHADPS4_A1: publishes everything committed without counting a post or waiting.
    void PublishNow() {
        Publish();
    }
    /// L3 (lean job channels, producer thread only): draw_hook runs at the end of every Commit()
    /// of a Draw record with its unmasked position; pre_publish runs before every publish that
    /// makes something new visible (count-based, waits, PublishNow).
    void (*draw_hook)(void*, u64){};
    void (*pre_publish)(void*){};
    void* hook_ctx{};
    /// Publish every n posts (n = ~0u: never by count).
    void SetPubBatch(u32 n) {
        pub_batch = n;
    }
    [[nodiscard]] u64 CommittedPos() const noexcept {
        return committed_pos;
    }
    /// SHADPS4_A2: ring space is free only once the back AND every A2 worker are past it. A
    /// worker holds the position of the record it reads next (~0 when it holds no DCB; it only
    /// grows while held, so a stale value is lower = conservative). WaitForSpace frees space up
    /// to the lowest of read_pos and these.
    struct alignas(64) ReaderHold {
        std::atomic<u64> pos{~0ull};
    };
    static constexpr u32 MaxReaderHolds = 8;
    std::array<ReaderHold, MaxReaderHolds> reader_holds{};
    [[nodiscard]] u64 ReadFloor() const noexcept {
        u64 r = read_pos.load(std::memory_order_acquire);
        for (const auto& hold : reader_holds) {
            r = std::min(r, hold.pos.load(std::memory_order_acquire));
        }
        return r;
    }
    /// SHADPS4_LEAN shadow: nobody consumes this channel - every commit counts as executed.
    bool self_consume{false};
    /// SHADPS4_LEAN: no MaybeLog lines from this channel.
    bool quiet{false};
    /// SHADPS4_LEAN shadow compare: when set, every register word written (EmitRegWrite) and
    /// every other committed record (canon_fn) is appended as canonical u64 events.
    std::vector<u64>* canon{};
    void (*canon_fn)(std::vector<u64>&, const RecordHeader&){};
    /// SHADPS4_A2_PARK: A2 workers parked until something they wait for changes (a publish, the
    /// front starting to wait, the back waiting on a stretch). Rung only while one is parked.
    alignas(64) std::atomic<u32> reader_parked{0};
    alignas(64) std::atomic<u32> reader_bell{0};
    void RingReaderBell() {
        if (reader_parked.load(std::memory_order_seq_cst) != 0) {
            reader_bell.fetch_add(1, std::memory_order_seq_cst);
            reader_bell.notify_all();
        }
    }
    /// SHADPS4_A2: non-zero while the front waits for the back (WaitDone / WaitForSpace).
    [[nodiscard]] bool FrontWaitingOnBack() const noexcept {
        return front_waiting.load(std::memory_order_acquire) != 0;
    }
    alignas(64) std::atomic<u32> front_waiting{0};
    /// SHADPS4_A2: front ticks waiting for a DCB table slot (front-written, logged by MaybeLog).
    u64 slot_wait_ticks{};
    /// SHADPS4_A1: complete DCBs the front has published (front writes after the publish).
    alignas(64) std::atomic<u64> dcb_ends_published{0};

    /// The front thread is about to return; lets the back leave its loop.
    void FrontExited();
    /// Posts a Retire record and publishes it (run-ahead POST_TASK_DONE).
    void PostRetire();

    // ---- Back (the only consumer) ----

    /// The next published record, or null. Stays valid until Done(it).
    const RecordHeader* Next();
    /// SHADPS4_MTB_P3: the ring position of the record Next() returned (unmasked).
    [[nodiscard]] u64 ReadPos() const noexcept {
        return read;
    }
    /// SHADPS4_MTB_P3 read-ahead: the published record at ring position `pos` (>= ReadPos(),
    /// Wrap records skipped and `pos` moved past them), or null when the front has not published
    /// it yet. Releases nothing: every record from ReadPos() on stays valid until Done passes it.
    const RecordHeader* PeekAt(u64& pos);
    /// Releases the record returned by Next() and reports it executed.
    void Done(const RecordHeader* record);
    /// SHADPS4_A2: releases every record from the one Next() returned up to `end_pos`
    /// (unmasked, one past the last) and reports them executed through `last_seq`.
    void DoneThrough(u64 end_pos, u64 last_seq);
    /// Returns when a record is published, `num_commands` is non-zero, `stoken` is stopped or the
    /// front has exited (or spuriously).
    void WaitForWork(const std::atomic<u32>& num_commands, const std::stop_token& stoken);
    bool FrontHasExited() const {
        return front_exited.load(std::memory_order_acquire);
    }

    // ---- Any thread ----

    /// Wakes the back if it is parked (SendCommand, stop requests).
    void WakeBackIfParked();
    void WakeBack();

    static constexpr u32 NumDrainReasons = static_cast<u32>(DrainReason::Count);
    static constexpr u32 NumPostKinds = static_cast<u32>(PostKind::Count);
    /// Front-owned counters, logged and reset every ~10 s by MaybeLog. Times are in TSC ticks.
    struct Counters {
        u64 records{}, bytes{}, full_waits{}, front_parks{}, reg_appends{};
        u64 full_wait_ticks{}; ///< WaitForSpace (ring full), TSC
        std::array<u64, NumPostKinds> posted{};
        u64 depth_waits{}, depth_wait_ticks{};
        u64 max_lead{};       ///< Most posted records outstanding at a post (after the bound).
        u64 max_lead_seq{};   ///< Most records (any type) published but not done, at a post.
        std::array<u64, NumDrainReasons> drains{};
        std::array<u64, NumDrainReasons> drain_waits{}; ///< Drains that found the back busy.
        std::array<u64, NumDrainReasons> drain_ticks{};
        std::array<u64, NumDrainReasons> guard_checks{}; ///< TASK_GUARD checks, by hit reason.
        u64 write_log_max{};       ///< Most outstanding noted writes (TASK_GUARD).
        u64 write_log_overflows{}; ///< Noted writes dropped while outstanding.
    };
    /// Front only. Checks the clock every 1024 calls; logs and resets every ~10 s.
    void MaybeLog();

private:
    void Publish();
    void WaitForSpace(u64 end);
    /// Waits until done_seq >= seq. Returns the ticks spent waiting (0 if it was already done).
    u64 WaitDone(u64 seq);
    bool BackHasWork(const std::atomic<u32>& num_commands) const;
    void PublishDone(u64 seq);

    std::unique_ptr<u8[]> storage_owner;
    u8* storage{};
    u32 capacity{};
    u64 mask{};

    // Producer side (front). Written only by the front.
    alignas(64) std::atomic<u64> write_pos{0}; ///< Published end, seq_cst store.
    u64 reserve_pos{0};
    u64 last_record_pos{0}; ///< SHADPS4_A2
    u64 committed_pos{0};
    u64 committed_seq{0};
    bool reg_batch{false};
    bool hold{false};
    std::vector<u32> held; ///< entries: kind (1 reg / 2 extent), a, b, then count data words
    void (*hold_cb)(void*){nullptr};
    void* hold_ctx{nullptr};
    u64 reg_pos{~0ull};  ///< Ring position of the newest RegWrite record header.
    u64 reg_seq{0};      ///< Its seq; appending is allowed only while it is the newest record.
    u32 reg_used{0};     ///< Bytes of it in use (header + segments), before alignment padding.
    u64 cached_read{0};
    u64 emitted{0};
    bool record_open{false};
    // Run-ahead bound: seqs of the last MaxDepth + 1 posted records, oldest not known done first.
    u32 depth{1};
    std::unique_ptr<u64[]> posted_seqs;
    u64 posts{0};        ///< Records posted so far.
    u32 pub_batch{1};    ///< SHADPS4_CP_SPLIT_PUB_BATCH: publish every N posts
    u32 unpublished_posts{0};
    u64 oldest_post{0};  ///< Index of the oldest post not known to be done.
    u64 last_posted_packet_seq{0};
    // TASK_GUARD write log: destinations of posted packets, oldest first, pruned by done_seq.
    struct PostedWrite {
        u64 seq;
        VAddr lo;
        VAddr hi;
        u32 value;  ///< the dword written at lo, when known
        bool known;
        bool fence; ///< written by an end-of-pipe fence (EOP / EOS / RELEASE_MEM)
        u8 src;
        u8 queue; ///< SHADPS4_XQ_CENSUS: the queue the front was parsing (0 gfx, vqid + 1 ASC)
    };
    /// Outstanding noted writes are posted records, so at most Depth() + 1 (the one being noted):
    /// with MaxDepth 1024 the log cannot overflow; the fallback stays as a guard.
    static constexpr u64 PostedWriteSlots = 2048;
    static_assert(PostedWriteSlots > MaxDepth + 1);
    std::unique_ptr<PostedWrite[]> posted_writes;
    u64 pw_head{0};
    u64 pw_tail{0};
    u64 pw_overflow_seq{0}; ///< Newest seq dropped from the full log (checks are conservative
                            ///< while it is outstanding).
    void PrunePostedWrites(u64 done);
    Counters counters{};
    u64 log_window_start_ns{0};
    u64 log_window_start_ticks{0};
    u32 log_calls{0};

    // Consumer side (back).
    alignas(64) std::atomic<u64> read_pos{0};
    u64 read{0};
    u64 cached_write{0};
    u64 pending_done{0}; ///< Newest executed seq not yet published in done_seq.
    u32 done_batch{1};
    u32 done_count{0};
    std::atomic<u64> back_idle_ticks{0}; ///< Written by the back, read and reset by MaybeLog.

    alignas(64) std::atomic<u64> done_seq{0};
    // ArmDrainProbe: armed by the front (~0 = none), completed and cleared by the back.
    alignas(64) std::atomic<u64> probe_seq{~0ull};
    std::atomic<u64> probe_t0{0}, probe_ticks{0}, probe_n{0};
    alignas(64) std::atomic<bool> front_parked{false};
    alignas(64) std::atomic<bool> back_parked{false};
    std::atomic<u32> doorbell{0};
    std::atomic<bool> front_exited{false};
};

} // namespace AmdGpu::CpSplit
