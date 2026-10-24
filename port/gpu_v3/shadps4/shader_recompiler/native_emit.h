// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include "common/types.h"
#include "shader_recompiler/resource.h"

namespace Shader {

/// SHADPS4_NATIVE_COLOUR: how the native vertex shader decodes one fetch attribute it pulls
/// itself from the vertex storage buffer (what the Vulkan vertex input format would do).
enum class NativeVtxKind : u8 {
    None,   // not pullable
    Float,  // 32-bit float components
    Uint,   // 32-bit integer components, as uint
    Sint,   // 32-bit integer components, as int
    Half,   // 16-bit float
    Unorm,  // 8 / 16-bit normalized
    Snorm,
    Uint16, // 8 / 16-bit integer, zero-extended
    Sint16, // 8 / 16-bit integer, sign-extended
    Uscaled,
    Sscaled,
};
struct NativeVtxAttr {
    NativeVtxKind kind{};
    u8 bits{};       // component width: 8, 16 or 32; 0 = packed 10:10:10:2 in one dword
    u8 count{};      // components: 1-4
    u8 valid{};
    u32 stride{};    // V# stride (baked: part of the variant key)
};

// SHADPS4_NATIVE_PASS (2026-10-09, notes/native-renderer-2026-10-09.md sec. 3): a native batch
// variant of a graphics pipeline draws a run of guest draws with one indirect multi-draw. Its
// vertex shader reads each draw's user-data registers and full buffer byte offsets from a
// records storage buffer indexed by the draw index, instead of the push constants. Set by the
// pipeline cache around the compile of a native variant (thread-local: the compile runs on the
// calling thread).
// SHADPS4_NATIVE_COLOUR (notes/colour-pass-2026-10-09.md sec. 3): `colour` also makes the
// fragment shader native (its record base comes from a flat draw-index varying the vertex shader
// writes at NativeDrawIndexLocation), and `pull` makes the vertex shader fetch every vertex
// attribute itself from one whole vertex storage buffer at `vertex_binding`, at the record's
// per-location byte offset, so draws whose vertex streams sit at unrelated offsets can share
// one multi-draw.
struct NativeEmit {
    bool active{};
    u32 records_binding{};
    bool colour{};
    bool pull{};
    u32 vertex_binding{};
    std::array<NativeVtxAttr, 32> attrs{}; // by fetch semantic (= vertex input location)
    /// Diagnostic (SHADPS4_NATIVE_COLOUR_DIAG 128): locations (bit mask) that keep a real
    /// vertex input instead of being pulled (valid only for unjoined batches).
    u32 keep_inputs{};
    /// Diagnostic (SHADPS4_NATIVE_COLOUR_DIAG 4096): every attribute keeps its real vertex input
    /// AND is pulled; mismatches go to a debug SSBO at `debug_binding` (layout: dword 0 count,
    /// entries of 16 dwords from dword 16: location, vertex index, draw, stream offset, pulled
    /// x y z w bits, real x y z w bits, vs hash, kind, bits, count; slot 64 = discard).
    bool compare{};
    /// The flat draw-index varying's location: just above every VS output / PS input (the
    /// fixed 31 overflowed maxVertexOutputComponents with the built-ins, s324 validation).
    u32 draw_location{};
    u32 debug_binding{};
    u32 vs_hash{};
    /// SHADPS4_NATIVE_GPU_DIRECT=1 (notes/native-path-2026-10-09.md sec. 10): the record of draw
    /// i is entry (table[first] + i) & table[mask] of the records buffer (a game job thread's
    /// record ring, or the batch's own records with first 0); each Guest binding's record word and
    /// each pulled location's are guest addresses (low 32 bits) minus the table's base.
    bool ring{};
    u32 table_binding{};
    /// SHADPS4_NATIVE_PT (sec. 18, milestone 2a; ring mode only): Guest bindings are read
    /// through the buffer cache's GPU page table (u64 device address per 16 KB guest page, read
    /// by pointer at the table's NativeTablePtFallback address + page * 8: the page table buffer
    /// itself, so it costs no set-0 binding) and a PhysicalStorageBuffer pointer, from the record's full guest
    /// address (low 32 bits in the offset word, bits 32..39 in NativeRecordHiOffset; inherited
    /// slots: the table's NativeTablePtLo / PtHi), not through their set-0 descriptors (still
    /// bound, layout unchanged). =2: every such load also reads the descriptor and mismatches go
    /// to a debug SSBO at `pt_debug_binding` (dword 0 count, entries of 16 dwords from dword 16:
    /// binding, relative index, pt value, descriptor value, draw, guest lo, guest hi, tag).
    bool pt{};
    bool pt_compare{};
    u32 pt_debug_binding{};
    /// SHADPS4_NATIVE_BINDLESS (sec. 12): the PS samples image i / sampler j from the global
    /// arrays (set 1, binding 0 / 1) at the record's slot words NativeRecordImgOffset + i /
    /// NativeRecordSmpOffset + j. Set-0 numbering is unchanged.
    bool bindless{};
    /// SHADPS4_NATIVE_SEG (sec. 20 brick A; ring + pt only): draw i's record is named by the
    /// batch's u32 list at the table's NativeTableSegList address: entry = list[i], slot =
    /// entry >> 24 (a job thread's ring, NativeSegLegacySlot = the batch's own records), ring
    /// index = entry & NativeSegRingMask; record = the slot's device address (table) + index x
    /// NativeRecordDwords x 4, every record word a PhysicalStorageBuffer load. The records
    /// SSBO stays declared and bound (layout unchanged). =2: each word is also read the SSBO
    /// way and mismatches go to the pt compare SSBO (dword 1 count, entries from dword 2048).
    bool seg{};
    bool seg_compare{};
    /// q3 (SHADPS4_NATIVE_BINDLESS_VS): the VS samples bindless too; its image i / sampler j
    /// use slot words vs_img_base + i / vs_smp_base + j (after the FS's).
    bool bindless_vs{};
    u32 vs_img_base{};
    u32 vs_smp_base{};
    /// q53 (SHADPS4_A2_VBUF2): a second whole vertex storage buffer at `vertex2_binding` (after
    /// every other set-0 binding); location l pulls from it when bit 16 + l of the push word
    /// PushData::buf_offsets[0..3] is set (bits 0-15 stay q18's page-table mask).
    bool vbuf2{};
    u32 vertex2_binding{};
};
inline thread_local NativeEmit t_native_emit{};
/// q18: SHADPS4_A2_PT_VERTEX=1 - colour vertex streams outside the bound vertex buffer are pulled through the page
/// table (mask bit per location in PushData::buf_offsets[0..1]).
/// =2 (bit-exact gate only): EVERY pulled location reads through the page table (no mask), so
/// SHADPS4_NATIVE_COLOUR_DIAG=4096 compares the page-table fetch against the real vertex input on the back's path.
inline const int g_pt_vertex = [] {
    const char* e = std::getenv("SHADPS4_A2_PT_VERTEX");
    return e ? std::atoi(e) : 0;
}();
/// q53: SHADPS4_A2_VBUF2=1 - colour variants get a second vertex SSBO; the A2 translator takes draws whose streams
/// sit in two cached buffers (the second one's locations: mask bits 16 + l of buf_offsets[0..3]).
/// =2 / =3 (bit-exact gate only, with SHADPS4_NATIVE_COLOUR_DIAG=4096 on the back's path): the back binds its one
/// vertex buffer to both bindings and sets mask 0xffff (every location through SSBO 2) / 0x5555 (alternating).
inline const int g_vbuf2 = [] {
    const char* e = std::getenv("SHADPS4_A2_VBUF2");
    return e ? std::atoi(e) : 0;
}();
/// q3: SHADPS4_NATIVE_BINDLESS_VS=1.
inline const bool g_native_bindless_vs = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_BINDLESS_VS");
    return e && e[0] == '1';
}();

constexpr u32 NativeRecordUdOffset = 0;
constexpr u32 NativeRecordOffOffset = NUM_USER_DATA_REGS;
/// Each Guest binding's V# size in bytes: native loads past it return 0, as the generic path's
/// descriptor range makes robustBufferAccess do (Opus code review 2026-10-09 #1).
constexpr u32 NativeRecordSizeOffset = NUM_USER_DATA_REGS + NUM_BUFFERS;
/// SHADPS4_NATIVE_COLOUR: per vertex input location (< 16), the byte offset of the draw's
/// stream in the vertex storage buffer and the bytes available from there (the bound size).
constexpr u32 NativeMaxPulledLocations = 16;
constexpr u32 NativeRecordVoffOffset = NUM_USER_DATA_REGS + 2 * NUM_BUFFERS;
constexpr u32 NativeRecordVsizeOffset = NativeRecordVoffOffset + NativeMaxPulledLocations;
/// SHADPS4_NATIVE_BINDLESS: PS image / sampler slots in the global descriptor arrays: NativeBindlessSlots record
/// words from NativeRecordImgOffset, the images first, the samplers from slot word g_bl_img_max.
/// q54 (SHADPS4_BINDLESS_SPLIT=1): 22 image + 10 sampler slots instead of 16 + 16 (8 city / fire materials sample
/// 17-22 images with at most 9 samplers: no bindless variant, > 32 push descriptors -> the back's old path).
constexpr u32 NativeBindlessSlots = 32;
constexpr u32 NativeBindlessImgCap = 22; ///< array bounds: the most image / sampler slots any split gives
constexpr u32 NativeBindlessSmpCap = 16;
inline const u32 g_bl_img_max = [] {
    const char* e = std::getenv("SHADPS4_BINDLESS_SPLIT");
    return e && e[0] == '1' ? 22u : 16u;
}();
inline const u32 g_bl_smp_max = NativeBindlessSlots - g_bl_img_max;
static_assert(NativeBindlessImgCap + NativeBindlessSmpCap >= NativeBindlessSlots);
constexpr u32 NativeRecordImgOffset = NativeRecordVsizeOffset + NativeMaxPulledLocations; // 128
/// SHADPS4_NATIVE_PT: byte b % 4 of dword NativeRecordHiOffset + b / 4 = bits 32..39 of Guest
/// binding b's guest address; the same for pulled vertex locations at NativeRecordVhiOffset
/// (written, not read by the shaders yet). 174-175 spare.
constexpr u32 NativeRecordHiOffset = NativeRecordImgOffset + NativeBindlessSlots;          // 160
constexpr u32 NativeRecordHiDwords = (NUM_BUFFERS + 3) / 4;                                 // 10
constexpr u32 NativeRecordVhiOffset = NativeRecordHiOffset + NativeRecordHiDwords;           // 170
constexpr u32 NativeRecordVhiDwords = NativeMaxPulledLocations / 4;                          // 4
constexpr u32 NativeRecordDwords = 176;
static_assert(NativeRecordHiOffset == 160);
static_assert(NativeRecordVhiOffset + NativeRecordVhiDwords <= NativeRecordDwords);
/// SHADPS4_NATIVE_GPU_DIRECT: the per-batch table (ring mode).
constexpr u32 NativeTableFirst = 0;
constexpr u32 NativeTableMask = 1;
constexpr u32 NativeTableBase = 2;
constexpr u32 NativeTableVbase = 2 + NUM_BUFFERS;
/// Coverage (sec. 11): push-layout user-data slots taken from the table (mask) and their values;
/// Guest binding slots whose size comes from the table (mask over two dwords) and the sizes
/// (their offset is folded into the base: record word - base = the opener's offset).
constexpr u32 NativeTablePushInh = NativeTableVbase + NativeMaxPulledLocations; // 58
constexpr u32 NativeTableInhSlots = NativeTablePushInh + 1;                      // 59, 60
constexpr u32 NativeTableUd = 64;
constexpr u32 NativeTableSize = NativeTableUd + NUM_USER_DATA_REGS; // 80
/// SHADPS4_NATIVE_PT: an inherited slot's guest address (low 32 bits per binding, bits 32..39
/// as bytes) and the 64-bit fallback device address of a page-table miss (two dwords: the
/// page table buffer itself, always mapped memory).
constexpr u32 NativeTablePtLo = NativeTableSize + NUM_BUFFERS;             // 120
constexpr u32 NativeTablePtHi = NativeTablePtLo + NUM_BUFFERS;             // 160
constexpr u32 NativeTablePtFallback = NativeTablePtHi + NativeRecordHiDwords; // 170, 171
/// SHADPS4_NATIVE_SEG: per job slot the u64 device address of its record ring, slot
/// NativeSegLegacySlot the batch's own records; then the u64 address of the batch's record list
/// (a u32 per draw: slot << 24 | ring index). The ring mask is the job ring's (asserted against
/// AmdGpu::NativeRec::RingSize in vk_rasterizer.cpp).
constexpr u32 NativeSegLegacySlot = 16; // = AmdGpu::NativeRec::MaxSlots
constexpr u32 NativeSegSlots = NativeSegLegacySlot + 1;
constexpr u32 NativeSegRingMask = (1u << 16) - 1; // r3: = NativeRec::RingSize - 1
constexpr u32 NativeTableSlotAddr = 176;                                       // 176..209
constexpr u32 NativeTableSegList = NativeTableSlotAddr + 2 * NativeSegSlots; // 210, 211
/// SHADPS4_NATIVE_PT (brick C v2): the fault buffer's u64 device address (a non-contiguous or
/// uncached Guest range flags its pages there; ProcessFaultBuffer creates the buffer).
constexpr u32 NativeTableFaultAddr = NativeTableSegList + 2; // 212, 213
/// Brick C v3 item 4 (multi-material segments): a list entry's slot is bits 24..28, its
/// material index bits 29..31 (0..NativeSegMaxMaterials - 1); the table's NativeTableSegMat =
/// the u64 device address of the segment's slot table (per material NativeSegMatWords words:
/// g_bl_img_max image slots, then g_bl_smp_max sampler slots), 0 = no table: the
/// slots come from the record words (legacy / ring batches).
constexpr u32 NativeSegSlotMask = 0x1f;
constexpr u32 NativeSegMatShift = 29;
constexpr u32 NativeSegMaxMaterials = 8;
constexpr u32 NativeSegMatWords = NativeBindlessSlots;
constexpr u32 NativeTableSegMat = NativeTableFaultAddr + 2; // 214, 215
constexpr u32 NativeTableDwords = 256;
static_assert(NativeTableSegMat + 2 <= NativeTableDwords);
static_assert(NativeSegSlots <= NativeSegSlotMask + 1);
static_assert(NativeTableInhSlots + 2 <= NativeTableUd);
static_assert(NativeTableSize + NUM_BUFFERS <= NativeTablePtLo);
static_assert(NativeTablePtFallback + 2 <= NativeTableSlotAddr);
static_assert(NativeTableFaultAddr + 2 <= NativeTableDwords);
/// SHADPS4_NATIVE_GPU_DIRECT=1: native variants are compiled in ring mode.
inline const bool g_native_ring = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_GPU_DIRECT");
    return v && (v[0] == '1' || v[0] == '3') && v[1] == 0;
}();
/// SHADPS4_NATIVE_PT=1 / 2 (meaningful with SHADPS4_NATIVE_GPU_DIRECT=1 only): native variants
/// read Guest bindings through the GPU page table; 2 = compare against the descriptor loads.
inline const int g_native_pt = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_PT");
    return v && (v[0] == '1' || v[0] == '2') && v[1] == 0 ? v[0] - '0' : 0;
}();
/// SHADPS4_NATIVE_SEG=1 / 2 (meaningful with GPU_DIRECT=1 and NATIVE_PT only): native variants
/// read their records by pointer through the batch's list; 2 = compare against the SSBO path
/// (through NATIVE_PT=2's compare SSBO, so it needs that mode: otherwise it runs as 1).
inline const int g_native_seg = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_SEG");
    return v && (v[0] == '1' || v[0] == '2') && v[1] == 0 ? v[0] - '0' : 0;
}();
inline const bool g_native_seg_compare = g_native_seg == 2 && g_native_pt == 2;
/// The highest location the flat draw-index varying may use (with 4 built-in components the
/// 128-component output budget allows locations up to 30; one spare).
constexpr u32 NativeDrawIndexMaxLocation = 29;

} // namespace Shader
