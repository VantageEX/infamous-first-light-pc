// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_PASSES=1: the engine's draws recorded into Vulkan on the engine's own threads, in
// whole passes, without the command processor touching them (native renderer, stage 1).
//
// Every draw of the engine goes through the eboot's DrawCommandBuffer::drawIndex (eboot
// 0x623C60: shadows 0x43D480, G-buffer 0x1A1770, item path 0x1A1960, 0x448B30, 0x5A8F40). A hook
// there keeps, per engine command buffer, the full register state the command buffer has set
// since it started (a "lane", carried across its chunks), and records each draw at once into a
// Vulkan secondary command buffer of the lane's open "segment": consecutive draws of the same
// attachments with only plain register writes between them. Anything else (a render target
// change, a clear, a dispatch, an event or wait, a DMA, an indirect draw, a draw the recorder
// cannot take) closes the segment.
//
// The command processor finds a segment by the address of its first draw packet. It interprets
// what precedes that draw as usual (so it has the exact registers of the first draw and sets up
// the render targets itself), checks once that the registers the engine thread assumed for the
// first draw are its own, executes the secondary and jumps to the end of the segment's last
// draw, applying the register writes the segment made. Nothing between is parsed: one check and
// one vkCmdExecuteCommands per pass instead of the translation of every draw. A segment it
// cannot use is interpreted as before (its commands are still in the command buffer).
//
// A draw whose material the recorder does not know yet posts a learn request at its packet: the
// command processor draws it through the full path and keeps its template
// (native_material_rasterizer.inl), keyed by the draw's whole render state (PassKey), so the
// engine threads take it from then on.
#pragma once

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace iss::native_passes {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

bool Enabled();
int Install(u64 image_base);

/// Compute: consecutive DISPATCH_DIRECT packets of a graphics command buffer with only compute
/// register writes between them, recorded into a secondary outside rendering (its first packet
/// is the key). A Learn request at a dispatch packet: the command processor learns its template.
enum class Kind : u32 { Segment, Learn, Compute };
enum State : u32 { Free, Open, Ready, Claimed, Done, Abandoned };

struct Segment {
    std::atomic<u32> state{Free};
    Kind kind{Kind::Segment};
    u32 generation{};
    u32 token{}; ///< the lane's tag, with the generation in the first draw's trailing NOP
    const u32* first_draw{}; ///< the key: the DRAW_INDEX_2 packet of its first draw
    const u32* end{};        ///< after its last draw's packets
    std::vector<const u32*> draw_packets;
    /// Every register the segment wrote up to its last draw (absolute word, last value).
    std::vector<std::pair<u32, u32>> final_regs;
    bool has_index_type{}, has_instances{};
    u32 final_index_type{}, final_instances{};
    /// The last draw's packet fields (the command processor's registers after the segment).
    u32 last_max_index{}, last_index_lo{}, last_index_hi{}, last_index_count{}, last_initiator{};
    /// The maximum index and index base above were set by an indexed draw of the segment
    /// (DRAW_INDEX_AUTO draws only set the count and the initiator).
    bool has_last_direct{};
    /// Compute: the last dispatch's dimensions and initiator (the command processor's registers
    /// after the segment).
    u32 last_dims[3]{}, last_dispatch_initiator{};
    /// ISS_NATIVE_INDIRECT: the indirect argument table (SET_BASE) after the segment.
    bool has_indirect_base{};
    u64 final_indirect_base{};
    /// What the engine thread assumed at the first draw (the command processor compares).
    u64 first_context{};
    u32 first_index_type{}, first_instances{};
    std::vector<u32> first_user; ///< vertex then pixel user data words (16 + 16)
    /// The dynamic state registers the lane had not written at the first draw, with the values
    /// it assumed (the latest the engine wrote): the command processor compares them.
    std::vector<std::pair<u32, u32>> dyn_check;
    /// Sampled learn requests (diagnosis): the lane's tracked words at the draw.
    std::vector<u32> snapshot;
    /// Learn requests: the tracked words the lane had written at the draw (MaskWords).
    std::vector<u32> mask;
    void* recording{};           ///< the rasterizer's (native_passes_rasterizer.inl)
    void* compute_recording{};   ///< the rasterizer's, Kind::Compute (native_compute_rasterizer.inl)
    void** lane_user{};          ///< the rasterizer's slot in the lane (its command pool)
    std::atomic<u64> executed_tick{}; ///< scheduler tick it was executed in (Done)
};

// Rasterizer callbacks (engine threads). The lane's registers are an AmdGpu::Regs the
// rasterizer allocated (regs: its object, words: its reg_array).
enum DrawResult : u32 { Recorded, NoRecord, NeedLearn, NewSegment };
struct DrawArgs {
    u32 index_count{}, initiator{};
    u64 index_address{};
    /// DRAW_INDEX_AUTO: a non-indexed draw of index_count vertices (no index buffer).
    bool autod{};
    /// ISS_NATIVE_INDIRECT: DRAW_INDEX_INDIRECT, its arguments (VkDrawIndexedIndirectCommand) at
    /// args_address; the index buffer is the lane's (INDEX_BASE, INDEX_BUFFER_SIZE).
    bool indirect{};
    u64 args_address{};
};
/// ISS_NATIVE_INDIRECT: DRAW_INDEX_INDIRECT packets recorded into segments.
bool IndirectEnabled();
/// The render state key of an indexed indirect draw: its pipeline also depends on the user data
/// registers the command processor writes the base vertex and instance to (packet words 2 and 3).
inline u64 IndirectKey(u64 key, u32 base_vtx_loc, u32 start_inst_loc) {
    u64 x = key ^ (0x25ull << 56) ^ (u64(base_vtx_loc & 0xFFFF) << 16) ^
            (u64(start_inst_loc & 0xFFFF) << 32);
    x *= 0x9E3779B97F4A7C15ull;
    return (x ^ (x >> 29)) | 1;
}
/// Where a draw packet's trailing NOP data (the segment token) starts: after DRAW_INDEX_2 (6
/// dwords) or DRAW_INDEX_AUTO (3 dwords).
inline const u32* TokenWords(const u32* packet) {
    const u32 opcode = (packet[0] >> 8) & 0xFF;
    // DISPATCH_DIRECT and DRAW_INDEX_INDIRECT: 5 dwords, then gnmdriver's trailing NOP.
    return packet + (opcode == 0x2D ? 4 : (opcode == 0x15 || opcode == 0x25) ? 6 : 7);
}
struct Callbacks {
    /// Absolute words of AmdGpu::Regs::index_buffer_type and num_instances (INDEX_TYPE and
    /// NUM_INSTANCES packets write them).
    u32 index_type_word{}, instances_word{};
    /// Absolute words of Regs::index_base_address (its high byte word first), max_index_size and
    /// num_indices (INDEX_BASE, DRAW_INDEX_2 and INDEX_BUFFER_SIZE write them).
    u32 index_base_word{}, max_index_word{}, num_indices_word{};
    void* (*new_regs)(u32*& words);
    void (*clear_state)(void* regs);
    /// Records the draw at the lane's registers into the segment (beginning its secondary when
    /// it has none yet). NewSegment: it would need another segment (other attachments).
    DrawResult (*record)(Segment& segment, void* regs, u64 key, const u32* written,
                         const DrawArgs& draw);
    /// The render state key of the registers (PassKey).
    /// The render state key of the registers the lane has written (`written`, MaskWords bits
    /// over the tracked words), without the ones it inherited.
    u64 (*key)(void* regs, const u32* written);
    /// Ends the segment's secondary (any thread, under the lane's lock).
    bool (*close)(Segment& segment);
    /// Releases what a segment that will not be executed holds.
    void (*abandon)(Segment& segment);
    /// Whether the segment's recording is reusable (executed and retired, or abandoned).
    bool (*retired)(const Segment& segment);
    /// Whether the command processor gave up learning this key.
    bool (*unlearnable)(u64 key);
    /// Periodic report (any thread).
    void (*report)();
    /// Compute (ISS_NATIVE_COMPUTE): the key of the compute registers (words CsBase.. of the
    /// lane's registers: the command processor's ComputeProgram; nullptr: off), and the
    /// recording of one DISPATCH_DIRECT packet into a Kind::Compute segment.
    u64 (*compute_key)(void* regs);
    DrawResult (*record_dispatch)(Segment& segment, void* regs, u64 key, const u32* packet);
};
void SetCallbacks(const Callbacks& callbacks);

// Command processor.
/// The segment or learn request whose first draw packet is `packet`, claimed (state Claimed),
/// nullptr when none is ready there.
Segment* Claim(const u32* packet);
/// After a claim: executed (tick) or not used (the command processor interpreted it).
void Finish(Segment& segment, bool executed, u64 tick);
/// The command processor's registers at the start of a command buffer it processes: an engine
/// lane that starts (over) at that address assumes them for what it has not written.
void PublishSpanStart(const u32* span, const u32* reg_array);
/// Engine submit (gnmdriver), before the command processor gets the command buffers. Draw time
/// mode: every open segment is closed. ISS_NATIVE_PASSES_SUBMIT=1: the segments of these
/// command buffers are recorded now, from the exact registers at each of their draws (a scan of
/// the register writes of every command buffer in submission order, from where the previous
/// submission left them), on a pool of threads; the call returns when they are recorded.
void BeforeSubmit(u32 count, const u32* const* dcbs, const u32* sizes_in_bytes);
/// ISS_NATIVE_PASSES_SUBMIT: whether the register scan must start from the command processor's
/// registers now (once, when recording starts: the caller waits for the GPU to be idle first).
bool NeedsSeed();
/// cs_words: the graphics queue's compute registers (its ComputeProgram, from COMPUTE 0x200).
void Seed(const u32* reg_array, const u32* cs_words, u32 cs_count);
/// ISS_NATIVE_PASSES_SUBMIT: packets the driver submits on its own (the init sequence).
void ScanOnly(const u32* dcb, u32 dwords);
bool SubmitMode();
/// ISS_NATIVE_PASSES_SUBMIT, command processor: waits until the recording of the command buffer
/// at `dcb` is done (the submission returns before it is; the pool records while the command
/// processor works through the earlier command buffers).
void WaitRecorded(const u32* dcb);
/// ISS_NATIVE_PASSES_JOB_WAIT (default on, =0: WaitRecorded waits for the whole command buffer):
/// WaitRecorded waits for the first job only; the command processor calls WaitRecordedAt with each
/// packet it reaches, which waits for the job recording that packet once it crosses into it.
extern thread_local const u32* next_job_boundary;
void WaitRecordedSlow(const u32* packet);
inline void WaitRecordedAt(const u32* packet) {
    if (packet >= next_job_boundary) WaitRecordedSlow(packet);
}
/// ISS_NATIVE_ASC=1 (with ISS_NATIVE_PASSES_SUBMIT and ISS_NATIVE_COMPUTE): the dispatches of
/// the asynchronous compute queues recorded as the graphics queue's are.
bool AscEnabled();
/// A DingDong of compute queue `qid` (the ring span the command processor will process; cp_cs:
/// that queue's ComputeProgram words as the command processor has them). Scans the ring's
/// indirect buffers from the queue's compute registers, has the pool record their dispatches and
/// waits for it (bounded): the command processor claims the segments at their first dispatch.
void BeforeAscSubmit(u32 qid, const u32* ring, u32 dwords, const u32* cp_cs, u32 cp_words);
/// ISS_NATIVE_ASC_ASYNC=1: whether the pool has recorded the compute submission at ring (true
/// when none is pending, or after waiting 50 ms: its dispatches then take the full path).
bool AscRecorded(const u32* ring);
/// Command processor, at a learn request: which registers the lane assumed differently.
void NoteDivergence(const Segment& segment, const u32* reg_array);
void Report();

// Registers the lanes follow (graphics context, vertex/pixel shader, VGT user config).
constexpr u32 TrackedFirst[3] = {0xA000, 0x2C00, 0xC240};
constexpr u32 TrackedCount[3] = {0x400, 0x80, 0x20};
constexpr u32 VsUser0 = 0x2C4C, PsUser0 = 0x2C0C, UserWords = 16;
constexpr u32 TrackedTotal = 0x400 + 0x80 + 0x20;
/// The compute registers (SET_SH_REG 0x200..): kept in the lane's registers at CsBase, outside
/// the tracked graphics words (no graphics key reads them).
constexpr u32 CsBase = 0x2E00, CsWords = 0x80;
constexpr u32 MaskWords = (TrackedTotal + 31) / 32;
/// Registers the recorder turns into Vulkan dynamic state from the draw's own values (not part of
/// the render state key): depth bounds, scissors, window offset, viewport depths and transforms,
/// blend constants, stencil reference and masks, line width, polygon offset.
inline bool IsDynamicReg(u32 reg) {
    if (reg < 0xA000 || reg >= 0xA400) return false;
    const u32 r = reg - 0xA000;
    return r == 0x008 || r == 0x009 || r == 0x00C || r == 0x00D || (r >= 0x080 && r <= 0x082) ||
           r == 0x090 || r == 0x091 || (r >= 0x094 && r <= 0x0D3) || (r >= 0x105 && r <= 0x108) ||
           r == 0x10C || r == 0x10D || (r >= 0x10F && r <= 0x16E) || r == 0x282 ||
           (r >= 0x2DE && r <= 0x2E3);
}
/// Registers no recorded draw reads: occlusion count control (open queries keep segments off), the
/// user clip planes (draws with clip planes enabled are not recorded), and those below.
inline bool IsUnreadReg(u32 reg) {
    return reg == 0xA001 || (reg >= 0xA16F && reg <= 0xA186) ||
           // Clip rectangles and the hardware screen offset, the guard band: no Vulkan state.
           (reg >= 0xA083 && reg <= 0xA08D) || (reg >= 0xA2FA && reg <= 0xA2FD);
}
/// Index of a tracked register in the mask (-1: not tracked).
inline int TrackedIndex(u32 reg) {
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        if (reg >= TrackedFirst[r] && reg < TrackedFirst[r] + TrackedCount[r])
            return int(base + reg - TrackedFirst[r]);
        base += TrackedCount[r];
    }
    return -1;
}

} // namespace iss::native_passes
