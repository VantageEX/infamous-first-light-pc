// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <immintrin.h>

#include "common/fault_monitor.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

/// SHADPS4_BACK_CLEAN=1 (=2 verify), default off.
///
/// The command processor asks the memory tracker "is any page of this range CPU-dirty?" for every
/// read binding of every draw (~49,000 times a frame in Second Son) and 99.7% of the answers are
/// "no". Each question reads tracker state that guest write-fault handlers modify on other cores
/// (region lock, dirty bits, summary flags: ~2,850 faults a frame), so the reader keeps paying
/// for cache lines it does not own, and takes the region lock whenever the region has any dirty
/// page. Measured 2026-09-30: MemoryTracker::ForEachUploadRange is 8.6% of the CP back's wall
/// time in the fire scene, 3.3 of it in the region spin lock.
///
/// The tracker stays as it is and stays authoritative. On top of it the back keeps a private
/// bit per page, "verified clean": set after the locked walk of that page found it clean or took
/// its upload, cleared when the page is marked dirty again. Whoever marks pages dirty appends the
/// page range to a log (DirtyLog) while holding the region lock and before setting the tracker
/// bits; the back drains the log before every question. A question whose pages are all verified
/// clean touches no shared tracker state except the log's tail counter; otherwise only the
/// unverified page runs take the locked walk.
///
/// Why this is the same answer the tracker gives (notes/render-state-memo-2026-10-01.md):
/// - A page turns dirty in exactly one place, RegionManager::ChangeRegionState<CPU, true>, always
///   under the region lock; the append sits in front of it. So "tracker bit set" implies "entry
///   appended", and a drain that starts later sees the entry.
/// - A guest write that precedes the submission being processed had its fault handler return
///   before the write, so its entry precedes the back's drain (same happens-before chain the
///   lock-based read relies on).
/// - The walk of an unverified run always takes the region lock (no lock-free "region is clean"
///   shortcut): a handler that has appended but not yet set its bits holds that lock, so the walk
///   waits for the bits. A handler that starts after the walk appends after the walk's drain, and
///   that entry clears the bit again at the next drain.
/// - Overflow (the back did not drain for more than half the ring, e.g. while idle on a loading
///   screen) drops every verified bit. Entries skipped that way belong to handlers that took
///   their region lock before the back's tail read, so later walks see their bits.
inline const int g_back_clean = [] {
    const char* env = std::getenv("SHADPS4_BACK_CLEAN");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

/// SHADPS4_RMEMO_CENSUS=1 (diagnostic, default off; notes/buffer-sync-validity-2026-10-07.md):
/// classifies read-binding memo misses by page-granular validity (the BackClean answer) and by a
/// per-Buffer dirty counter (a second DirtyLog consumer on the back), and times the miss path.
/// Needs SHADPS4_BACK_CLEAN=1 (DirtyLog appends nothing without it). Excludes SHADPS4_PB_CENSUS
/// (both use ReadMemoEntry::pb_written).
inline const bool g_rmemo_census = [] {
    const char* env = std::getenv("SHADPS4_RMEMO_CENSUS");
    return env && env[0] == '1';
}();

/// SHADPS4_READ_SYNC_FAST=1 (=2 verify), default off (notes/buffer-sync-validity-2026-10-07.md
/// sec. 6): a read sync that BackClean answers "no unverified page" returns from ObtainBuffer
/// before SynchronizeBuffer. Needs SHADPS4_BACK_CLEAN=1.
inline const int g_read_sync_fast = [] {
    const char* env = std::getenv("SHADPS4_READ_SYNC_FAST");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

/// SHADPS4_NATIVE_COHERENT=1 (=2 verify), default off (notes/native-path-2026-10-09.md sec. 17,
/// milestone 1 of the native renderer): CPU-written guest memory is made coherent with the cached
/// GPU buffers once per guest submission instead of per binding. At every submission start (and
/// at every other point the CP's legal view of guest memory moves: a satisfied memory wait, a
/// CP-side guest write) the back drains this log and uploads every dirty page that lies in a
/// cached buffer (BufferCache::CoherenceSync); new buffers are synced whole when created, the
/// CP's own DMA writes when they happen. A read binding then needs no validity check at all: it
/// is the buffer lookup and the access barrier. A page the guest writes after its submission
/// started is not legally visible to that submission (submit_epoch.h) and is uploaded at the next
/// point. Needs SHADPS4_BACK_CLEAN=1 (the log) and the CP split (the points arrive as records).
/// SHADPS4_ASC_EARLY (l4a3): set while the back runs a record ahead of the graphics backlog -
/// its read bindings take the per-binding validity sync instead of the coherence skip.
inline thread_local bool t_force_binding_sync = false;
inline thread_local u64 t_forced_binding_checks = 0;
/// SHADPS4_ASC_EARLY_INV=1 (l4a4 census, back thread only): ring position of the record the back runs
/// (in order: t_inorder_pos; early: t_early_pos) and the ranges early records wrote.
inline const bool g_early_inv = [] {
    const char* env = std::getenv("SHADPS4_ASC_EARLY_INV");
    return env && env[0] == '1';
}();
inline thread_local u64 t_inorder_pos = 0;
inline thread_local u64 t_early_pos = 0;
struct EarlyWrite {
    u64 lo, hi, pos;
};
inline std::array<EarlyWrite, 512> g_early_writes{};
inline u64 g_early_w_head = 0, g_early_w_tail = 0; // back only
inline std::atomic<u64> g_early_inv_reads{0}, g_early_inv_checks{0}, g_early_inv_writes{0};
/// SHADPS4_LANE_CENSUS=1 (n1, sec. 21 step 0, back thread only): see n1_lane_census.py.
/// q11a (SHADPS4_BACK_CYCLES=1, back thread): 0/1 early ASC service cycles / services, 2/3 dispatches on
/// the CP thread (cycles / n), 4/5 dispatches inside the early service, 6 lane dependency work per binding
/// (ObtainBuffer lane block), 7 lane notes (LaneNoteRange), 8/9 lane submits (cycles / n).
inline const bool g_lb_on = [] {
    const char* env = std::getenv("SHADPS4_BACK_CYCLES");
    return env && env[0] == '1';
}();
inline std::array<u64, 10> g_lb{};
/// q19 (SHADPS4_BACK_CYCLES=1): Call-record phases, cycles - 0 lazy fences, 1 OnSubmit, 2 fault buffer, 3 image
/// downloads, 4 texture GC, 5 buffer GC, 6 Flush, 7 ProtectFence, 8 GpuIdle IRQ, 9 WaitBarrier; counts 10 idle
/// tails, 11 wait barriers. Back thread only.
inline std::array<u64, 12> g_g1{};
struct G1Scope {
    u32 i;
    u64 t0;
    explicit G1Scope(u32 i_) noexcept : i{i_}, t0{g_lb_on ? __rdtsc() : 0} {}
    ~G1Scope() {
        if (t0 != 0) {
            g_g1[i] += __rdtsc() - t0;
        }
    }
};
inline bool g_lb_in_service = false;
struct LbScope {
    u32 i;
    u64 t0;
    explicit LbScope(u32 i_) noexcept : i{i_}, t0{g_lb_on ? __rdtsc() : 0} {}
    ~LbScope() {
        if (g_lb_on) {
            g_lb[i] += __rdtsc() - t0;
        }
    }
};
inline const bool g_lane_census = [] {
    const char* env = std::getenv("SHADPS4_LANE_CENSUS");
    return env && env[0] == '1';
}();
/// q59: SHADPS4_LANE_WAIT_CENSUS=1 (with SHADPS4_LANE_SEG_CENSUS=1; back thread) - for every lane segment that waits on an
/// unfinished main batch: the waited batch's frame age against the game's frame-end label, whether the bytes really
/// overlap, where the access sits in its batch (record time) and its pipeline (Scheduler::LaneSegCensus).
inline const bool g_lane_wait_census = [] {
    const char* e = std::getenv("SHADPS4_LANE_WAIT_CENSUS");
    return e && e[0] == '1';
}();
/// q59: overlap class of the last LaneProbeWrite (war) / LaneMainProducerFull (raw) answer and the record tsc of the
/// access that set it. 1 war exact byte overlap, 2 war window hit without byte overlap, 3 war older-tick floor,
/// 4 war ReadAll / probe floor (range unknown), 5 war new or re-based probe (conservative), 6 war stamp list overflow,
/// 7 war grown probe rescan, 8 raw note overlap (binding ranges), 9 raw coarse (window table / big list / lost note).
inline u8 g_lane_war_exact = 0, g_lc_hit_exact = 0;
inline u64 g_lane_war_tsc = 0, g_lc_hit_tsc = 0;
/// q59: the game's end-of-frame label (0x1f1e30's EOP; notes/zero-copy-heap-2026-10-07.md) and the main ticks that wrote it.
inline const u64 g_frame_label_addr = [] {
    const char* e = std::getenv("SHADPS4_LANE_WAIT_FRAME_LABEL");
    return e ? std::strtoull(e, nullptr, 16) : 0x248000000ull;
}();
inline std::array<u64, 4> g_frame_end_ticks{};
inline u64 g_frame_end_n = 0, g_frame_label_hits = 0;
/// q59b: s932 showed the label above is written ~6 times per frame in fire: counted only. Frame ends = the back's
/// PatchedFlip packets (LaneWaitFrameEnd).
inline void LaneWaitFrameLabel(u64 addr, u64) noexcept {
    g_frame_label_hits += addr == g_frame_label_addr ? 1 : 0;
}
inline void LaneWaitFrameEnd(u64 tick) noexcept {
    if (g_frame_end_n == 0 || g_frame_end_ticks[(g_frame_end_n - 1) & 3] != tick) {
        g_frame_end_ticks[g_frame_end_n++ & 3] = tick;
    }
}
/// q60: SHADPS4_LANE_FRAME_SCOPE=1 (user option 2, 2026-10-13): a lane WRITE into the game's alternating particle arena
/// (two 30 MiB halves picked by the frame parity [eboot+0x7d3eb38]; each half = record pool + draw list + control blocks
/// / args; REFERENCE game map) waits for the host-GPU end of the newest frame the game's frame fence had seen when the
/// chain was kicked, not for the newest main batch that read the range. Why it is the game's own guarantee: its ASC ring
/// never waits on graphics (q59: 0/s); 0x1f1e30 submits frame F-1 only after the end-of-frame label reached F-2, and
/// 0x3ee7d0 kicks chain F after that; graphics of frame F-2 is the last reader of half F % 2 (graphics of F reads chain
/// F's output, ordered by Game:Main's wait #2). The label value L read at the arena hook is >= F-2, so waiting for
/// frame L's end is safe (one frame conservative at worst). Producer (RAW) waits on main uploads stay.
inline const bool g_lane_frame_scope = [] {
    const char* e = std::getenv("SHADPS4_LANE_FRAME_SCOPE");
    return e && e[0] == '1';
}();
inline std::atomic<u64> g_lane_frame_label{0}; // Game:Main at the arena hook: label 0x248000000's value (low 32 bits) + 1
inline std::array<std::pair<u32, u64>, 8> g_frame_label_ring{}; // back: (frame-end label value, main tick of its EOP)
inline u64 g_frame_label_ring_n = 0;
inline void NoteFrameEndLabel(u32 value, u64 tick) noexcept {
    g_frame_label_ring[g_frame_label_ring_n++ & 7] = {value, tick};
}
/// q60 (back): the main tick holding the EOP of frame g_lane_frame_label; false when the back has not recorded it yet.
inline bool LaneFrameTick(u64& tick) noexcept {
    const u64 want1 = g_lane_frame_label.load(std::memory_order_acquire);
    if (want1 == 0 || g_frame_label_ring_n == 0) {
        return false;
    }
    const u32 want = static_cast<u32>(want1 - 1);
    const u64 n = std::min<u64>(g_frame_label_ring_n, 8);
    for (u64 k = 0; k < n; ++k) {
        const auto& e = g_frame_label_ring[(g_frame_label_ring_n - 1 - k) & 7];
        const s32 d = static_cast<s32>(e.first - want);
        if (d == 0) {
            tick = e.second;
            return true;
        }
        if (d < 0) {
            return false; // a gap: that frame's EOP was not recorded (or not yet)
        }
        if (k + 1 == n) {
            tick = e.second; // older than the ring: wait for the oldest frame end kept (later than needed)
            return true;
        }
    }
    return false;
}
inline std::array<std::atomic<u64>, 4> g_lane_frame_c{}; // q60: scoped, not recorded yet, outside the arena, no hook
struct LaneNote {
    u64 lo, hi, tick, seq;
    u32 kind; // 0 main write, 1 main upload, 2 early write, 3 early upload
    u32 wk{}; // q9 census: writer kind (BufferCache::SetWriterTag; 9 upload)
    u64 wid{}; // q9 census: writer id (pipeline hash)
    u64 tsc{}; // q59 census: record tsc
};
struct LaneTable {
    std::array<std::array<LaneNote, 8>, 4096> b{};
    std::array<u8, 4096> pos{};
};
inline LaneTable g_lc_main{}, g_lc_early{};
inline u64 g_lc_seq = 0;
inline u64 g_lc_main_seq = 0;
inline u64 g_lc_lost_max = 0, g_lc_big_max = 0; // q28: newest tick ever written to g_lc_lost_win / g_lc_big
inline const bool g_lc_fast = [] {
    const char* e = std::getenv("SHADPS4_LANE_PRODUCER_FAST");
    return e && e[0] == '1';
}(); // q25: bumped by every main note (kind < 2) before it changes main state
/// SHADPS4_ASC_LANE (p1): the n1 notes are the lane's producer index; the main batch of the newest
/// gfx EOP label the back wrote (labels are written at record time: WAR rule); big main notes.
inline bool g_lane_on = false;
inline std::array<u64, 16>* g_lane_counters = nullptr; // Scheduler::lane_c (back thread)
inline std::array<u64, 7>* g_lane_open_src = nullptr;  // Scheduler::lane_open_src (p1d / p1e)
inline u64 g_gfx_label_tick = 0;
struct LaneBig {
    u64 lo, hi, tick;
};
inline std::array<LaneBig, 64> g_lc_big{};
inline u64 g_lc_big_pos = 0;
inline u64 g_lc_big_evicted = 0; // newest tick of an unfinished big note overwritten
inline std::atomic<u64> g_lc_big_lost{0};
/// p1b: newest main tick noted per 1 MB (hashed; a big query reads these) and newest unfinished main
/// tick lost to a bucket overwrite per 1 MB.
inline std::array<u64, 65536> g_lc_win{};      // p1e: newest main tick per 64 KB window (hashed)
inline std::array<u64, 65536> g_lc_lost_win{}; // p1e: newest unfinished main tick lost per window
inline u32 LaneWin(u64 w) noexcept {
    return u32((w * 0x9E3779B97F4A7C15ull) >> 48);
}
/// q39: newest main tick per 4 MiB block (64 windows of 64 KB; hashed) - a filter in front of g_lc_win.
inline std::array<u64, 4096> g_lc_super{};
inline u32 LaneSuper(u64 s) noexcept {
    return u32((s * 0x9E3779B97F4A7C15ull) >> 52);
}
inline const bool g_lane_lean_notes = [] {
    const char* e = std::getenv("SHADPS4_LANE_LEAN_NOTES");
    return e && e[0] == '1';
}();
inline const int g_lane_producer_super = [] {
    const char* e = std::getenv("SHADPS4_LANE_PRODUCER_SUPER");
    return e && (e[0] == '1' || e[0] == '2') ? e[0] - '0' : 0;
}();
inline std::array<u64, 3> g_lc_super_c{}; // q39: super blocks skipped, verify compares, verify mismatches
inline u64 g_lc_main_newest = 0;
inline bool g_lc_coarse = false; // p1d: the last LaneMainProducer answer came from a coarse source
inline u32 g_lc_src = 0; // 0 exact, 1 window table (> 1024 windows), 2 window confirmed exact (p1g),
                         // 5 big list, 6 lost (bucket / big list)
inline u64 g_lc_done = 0;     // main ticks complete (known semaphore value)
inline u32 t_lc_rec_flags = 0; // bit 0: open-batch hit, bit 1: in-flight hit (current early record)
/// 0 early reads checked, 1/2 newest producer main write open / in flight, 3/4 main upload open / in
/// flight, 5 early writes checked, 6/7 over main write-or-upload open / in flight, 8 in-order reads
/// checked, 9/10/11 in-order reads over unfinished early writes via coherent skip / full / lean,
/// 12 notes truncated to 64 windows, 13 notes off the back thread, 14 notes, 15 early reads whose
/// newest producer is an early record (lane-internal), 16 early records with an open-batch hit,
/// 17 early records with an in-flight hit only, 18 early records.
inline std::array<std::atomic<u64>, 20> g_lc{};
inline u32 LaneBucket(u64 w) noexcept {
    return u32((w * 0x9E3779B97F4A7C15ull) >> 52) & 4095;
}
inline void LaneNoteRange(u64 lo, u64 hi, u64 tick, u32 kind, u32 wk = 0, u64 wid = 0) noexcept {
    if (hi <= lo) {
        return;
    }
    if (kind >= 2 && g_lane_lean_notes && !g_lane_census) {
        return; // q39: the early table is read only by SHADPS4_LANE_CENSUS
    }
    LbScope lb_scope{7}; // q11a
    LaneTable& t = kind >= 2 ? g_lc_early : g_lc_main;
    const u64 seq = ++g_lc_seq;
    if (kind < 2) {
        ++g_lc_main_seq; // q25
    }
    const u64 w0 = lo >> 16;
    u64 w1 = (hi - 1) >> 16;
    if (w1 - w0 > 63) {
        w1 = w0 + 63;
        g_lc[12].fetch_add(1, std::memory_order_relaxed);
        if (kind < 2) {
            // p1: a big main note is kept whole in a small list (the windows hold its first 4 MB).
            g_lc[19].fetch_add(1, std::memory_order_relaxed);
            auto& slot = g_lc_big[g_lc_big_pos++ % g_lc_big.size()];
            if (slot.tick > g_lc_done) {
                g_lc_big_evicted = std::max(g_lc_big_evicted, slot.tick);
                g_lc_big_lost.fetch_add(1, std::memory_order_relaxed);
            }
            slot = {lo, hi, tick};
            g_lc_big_max = std::max(g_lc_big_max, tick); // q28
        }
    }
    g_lc[14].fetch_add(1, std::memory_order_relaxed);
    if (kind < 2 && g_lane_on) {
        g_lc_main_newest = std::max(g_lc_main_newest, tick);
        const u64 wl = std::min((hi - 1) >> 16, w0 + 1023);
        for (u64 w = w0; w <= wl; ++w) {
            u64& e = g_lc_win[LaneWin(w)];
            e = std::max(e, tick);
        }
        for (u64 s = w0 >> 6; s <= (wl >> 6); ++s) { // q39 (always kept: the filter must see every note)
            u64& e = g_lc_super[LaneSuper(s)];
            e = std::max(e, tick);
        }
    }
    for (u64 w = w0; w <= w1; ++w) {
        const u32 k = LaneBucket(w);
        auto& last = t.b[k][(t.pos[k] - 1) & 7];
        if (last.seq != 0 && last.lo == lo && last.hi == hi && last.kind == kind) {
            last.tick = tick;
            last.seq = seq;
            last.wk = wk;
            last.wid = wid;
            last.tsc = g_lane_wait_census ? __rdtsc() : 0; // q59
            continue;
        }
        auto& slot = t.b[k][t.pos[k]++ & 7];
        if (kind < 2 && g_lane_on && slot.seq != 0 && slot.tick > g_lc_done) {
            u64& e = g_lc_lost_win[LaneWin(w)];
            e = std::max(e, slot.tick);
            g_lc_lost_max = std::max(g_lc_lost_max, slot.tick); // q28
        }
        slot = {lo, hi, tick, seq, kind, wk, wid, g_lane_wait_census ? __rdtsc() : 0}; // q59
    }
}
/// The newest note overlapping [lo, hi) whose main tick is not complete.
inline const LaneNote* LaneFind(const LaneTable& t, u64 lo, u64 hi) noexcept {
    const LaneNote* best = nullptr;
    if (hi <= lo) {
        return best;
    }
    const u64 w0 = lo >> 16;
    const u64 w1 = std::min((hi - 1) >> 16, w0 + 63);
    for (u64 w = w0; w <= w1; ++w) {
        for (const LaneNote& e : t.b[LaneBucket(w)]) {
            if (e.seq != 0 && e.tick > g_lc_done && e.lo < hi && lo < e.hi &&
                (!best || e.seq > best->seq)) {
                best = &e;
            }
        }
    }
    return best;
}
/// q9 census (SHADPS4_LANE_WHO=1, back thread): the writer of LaneMainProducer's answer (kind 250 + src
/// for the coarse sources, which name no note).
inline const bool g_lane_who = [] {
    const char* env = std::getenv("SHADPS4_LANE_WHO");
    return env && env[0] == '1';
}();
inline u32 g_lc_hit_wk = 0;
inline u64 g_lc_hit_wid = 0;
/// q9b: SHADPS4_LANE_WAR=1 - scoped WAR waits from the probes' newest main reads.
inline const int g_lane_war_mode = [] {
    const char* env = std::getenv("SHADPS4_LANE_WAR");
    return env && (env[0] == '1' || env[0] == '2') ? env[0] - '0' : 0;
}();
inline const bool g_lane_war = g_lane_war_mode != 0;
inline std::atomic<u64> g_war_dma_all{0}; // q9c: probes stamped whole by a main DMA bind / draw repeat
inline const bool g_lane_probe = g_lane_who || g_lane_war;
/// q9b: bumped (release) after every probe registration / extension; a stretch begun under an older
/// value stamps every probe at publish.
inline std::atomic<u64> g_lprobe_gen{0};
inline std::atomic<u64> g_war_scoped{0}, g_war_label{0}, g_war_open{0};
/// q9 census: large written lane bindings (>= 8 MiB: the particle pools / arenas) and the newest main
/// batch that READ each. Registered and stamped by the back; workers only test overlap.
struct LaneProbe {
    std::atomic<u64> lo{0}, hi{0};
    u64 rd{0};
    // q26 E1 (back thread only): read tick per 256 KB window from lo, a floor from ReadAll / registration, and the
    // stamping site of each (E0).
    u64 all{0};
    u8 all_site{0};
    std::array<u64, 256> wrd{};
    std::array<u8, 256> wsite{};
    std::array<u64, 256> wid{}; // q34: draw pipeline id of the newest stamp (census)
    std::array<u64, 256> wargs{}; // q35 H5: the stamping draw's indirect args address (0 = not indirect)
    std::array<u16, 256> wflip{}; // q35 H5: the back's flip count at the stamp
    // q38: byte extent (offsets in the window) of the stamps at wrd's tick; wprev = max tick of every older stamp.
    std::array<u64, 256> wprev{};
    std::array<u32, 256> wxlo{}, wxhi{};
    // q59 census: up to 16 exact [lo, hi) offsets (merged when they touch) of the stamps at wrd's tick (count 17 =
    // overflow) and the newest stamp tsc; q59b: the same for the tick before (wtick2).
    std::array<std::array<u32, 32>, 256> wr{}, wr2{};
    std::array<u8, 256> wrn{}, wrn2{};
    std::array<u64, 256> wtsc{}, wtsc2{}, wtick2{};
};
inline constexpr u8 LaneWrCap = 16;
/// q59b: adds [rlo, rhi) to a window's range list (merging touching ranges); n = LaneWrCap + 1 marks overflow.
inline void LaneWrAdd(std::array<u32, 32>& r, u8& n, u32 rlo, u32 rhi) noexcept {
    if (n > LaneWrCap) {
        return;
    }
    for (u32 j = 0; j < n; ++j) {
        if (r[2 * j] <= rhi && rlo <= r[2 * j + 1]) {
            r[2 * j] = std::min(r[2 * j], rlo);
            r[2 * j + 1] = std::max(r[2 * j + 1], rhi);
            return;
        }
    }
    if (n == LaneWrCap) {
        n = LaneWrCap + 1;
        return;
    }
    r[2 * n] = rlo;
    r[2 * n + 1] = rhi;
    ++n;
}
inline std::array<LaneProbe, 64> g_lprobe{};
/// q26 flags (see q26_lane_exact_waits.py).
inline const int g_lane_win = [] {
    const char* e = std::getenv("SHADPS4_LANE_WIN");
    return e && (e[0] == '1' || e[0] == '2') ? e[0] - '0' : 0;
}();
inline const bool g_lane_seg_census = [] {
    const char* e = std::getenv("SHADPS4_LANE_SEG_CENSUS");
    return e && e[0] == '1';
}();
inline const bool g_lane_native_exact = [] {
    const char* e = std::getenv("SHADPS4_LANE_NATIVE_EXACT");
    return e && e[0] == '1';
}();
inline const bool g_lane_small_probes = [] {
    const char* e = std::getenv("SHADPS4_LANE_SMALL_PROBES");
    return e && e[0] == '1';
}();
inline const bool g_lane_probe_merge = [] { // q30 E4
    const char* e = std::getenv("SHADPS4_LANE_PROBE_MERGE");
    return e && e[0] == '1';
}();
inline u32 g_lane_war_site = 0;
inline u64 g_lane_war_wid = 0;                 // q34: pipeline id of the window that set LaneProbeWrite's rd
inline u64 g_lane_stamp_id = 0;                // q34: set by the stamping caller (ObtainBuffer: census_id)
inline u64 g_lane_stamp_args = 0;              // q35 H5: the stamping draw's indirect args address
inline u64 g_lane_war_win = 0, g_lane_war_args = 0; // q36c: the window (and its stamp's args) that set the WAR rd
/// q38: SHADPS4_LANE_WIN_EXTENT=1 - window reads are byte extents of the newest tick + an older-tick floor (see q38 patch).
inline const bool g_lane_win_extent = [] {
    const char* e = std::getenv("SHADPS4_LANE_WIN_EXTENT");
    return e && e[0] == '1';
}();
inline u64 g_lane_ext_saved = 0; // q38 census: window reads that the extent check lowered
/// q36c census: SHADPS4_Q35_FS=hash[,hash...] (hex fs hashes) - log those draws' s15 stamps over probes and the lane
/// writes their stamps hold.
inline const std::array<u64, 4> g_q35_fs = [] {
    std::array<u64, 4> a{};
    if (const char* e = std::getenv("SHADPS4_Q35_FS")) {
        u32 n = 0;
        for (const char* s = e; *s && n < a.size();) {
            char* end = nullptr;
            a[n++] = std::strtoull(s, &end, 16);
            s = (end && *end == ',') ? end + 1 : (end ? end : s + 1);
            if (end && *end == 0) {
                break;
            }
        }
    }
    return a;
}();
inline bool Q35Fs(u64 h) noexcept {
    return h != 0 && (g_q35_fs[0] == h || g_q35_fs[1] == h || g_q35_fs[2] == h || g_q35_fs[3] == h);
}
inline u64 g_census_args = 0;                  // q35: the current draw's indirect args (rasterizer, census only)
inline u64 g_census_vs = 0;                    // q35: the current draw's vertex shader hash (census only)
inline u64 g_back_flip_seq = 0;                // q35 H5: PatchedFlip packets the back executed
/// q40 (DIAGNOSTIC, never for release): SHADPS4_NO_PARTICLES=1 - no ASC dispatches, no particle card draw.
inline const bool g_no_particles = [] {
    const char* e = std::getenv("SHADPS4_NO_PARTICLES");
    return e && e[0] == '1';
}();
inline std::atomic<u64> g_np_disp{0}, g_np_draw{0}; // q40: skipped ASC dispatches / card draws
void NoParticlesLog(); // buffer_cache.cpp
inline void NoParticlesNote(bool draw) noexcept {
    const u64 n = (draw ? g_np_draw : g_np_disp).fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & (n - 1)) == 0 || (n & 0x3ffff) == 0) {
        NoParticlesLog();
    }
}
/// q36: the particle sort replacement's guard counter {hits, first, last, overflow} (host-visible; null = none).
inline volatile u32* g_sort_guard_words = nullptr;
/// q35 H1: SHADPS4_LANE_COPY_NOSTAMP=1 - a fault copy recorded on the lane is not stamped as a main read.
inline const bool g_lane_copy_nostamp = [] {
    const char* e = std::getenv("SHADPS4_LANE_COPY_NOSTAMP");
    return e && e[0] == '1';
}();
/// q35 H5: wid (fs hash, low 32 bits) + meta in bits 56..59: class (0 unknown, 1 same group, 2 other) | flips (0..3) << 2.
inline u64 LaneH5Meta(u64 wid, u64 args, u16 flip, u64 lo) noexcept {
    if (wid == 0) {
        return 0;
    }
    const u64 cls = args == 0 ? 0 : (lo + (u64{16} << 20) >= args && lo < args + (u64{2} << 20)) ? 1 : 2;
    const u64 fd = std::min<u64>(static_cast<u16>(static_cast<u16>(g_back_flip_seq) - flip), 3);
    return (wid & 0xffffffffULL) | ((cls | (fd << 2)) << 56);
}
inline std::array<u64, 8> g_s15_size_hist{};   // q34: s15 stamps over a probe by size <64K/256K/1M/4M/12M/32M/128M/more                      // q26 E0: stamping site of the window that set LaneProbeWrite's rd
inline std::array<u64, 32> g_lprobe_all_c{};        // q26 E0: LaneProbeReadAll calls by site
/// q26 E0: compact wait-site codes for Scheduler::LaneNeedMain.
constexpr u32 LaneSiteRaw(u32 wk) {
    return wk & 255;
}
constexpr u32 LaneSiteWar(u32 stamp_site) {
    return 256 + (stamp_site & 63);
}
inline void LaneProbeWindows(LaneProbe& p, u64 lo, u64 hi, u64 v, u32 site) noexcept {
    const u64 plo = p.lo.load(std::memory_order_relaxed);
    const u64 phi = p.hi.load(std::memory_order_relaxed);
    const u64 a = std::max(lo, plo), b = std::min(hi, phi);
    if (b <= a) {
        return;
    }
    const u64 i0 = std::min<u64>((a - plo) >> 18, 255), i1 = std::min<u64>((b - 1 - plo) >> 18, 255);
    for (u64 i = i0; i <= i1; ++i) {
        if (g_lane_win_extent && !(i == 255 && phi - plo > (u64{64} << 20))) {
            // q38: this stamp's bytes inside window i (q38b: not in a clamped window 255).
            const u64 wb = plo + (i << 18);
            const u32 xlo = static_cast<u32>(std::max(a, wb) - wb);
            const u32 xhi = static_cast<u32>(std::min(b, wb + (u64{1} << 18)) - wb);
            if (v > p.wrd[i]) {
                p.wprev[i] = p.wrd[i]; // the old extent's tick now covers the whole window
                p.wxlo[i] = xlo;
                p.wxhi[i] = xhi;
            } else if (v == p.wrd[i]) {
                p.wxlo[i] = std::min(p.wxlo[i], xlo);
                p.wxhi[i] = std::max(p.wxhi[i], xhi);
            } else {
                p.wprev[i] = std::max(p.wprev[i], v);
            }
        }
        if (g_lane_wait_census) [[unlikely]] {
            // q59: the exact stamp ranges at the newest tick (window 255 of a > 64 MiB probe: overflow).
            const u64 wb = plo + (i << 18);
            const bool clamped = i == 255 && phi - plo > (u64{64} << 20);
            const u32 rlo = static_cast<u32>(std::max(a, wb) - wb);
            const u32 rhi = static_cast<u32>(std::min(b, wb + (u64{1} << 18)) - wb);
            if (v > p.wrd[i]) {
                p.wr2[i] = p.wr[i]; // q59b: the previous tick's list
                p.wrn2[i] = p.wrn[i];
                p.wtsc2[i] = p.wtsc[i];
                p.wtick2[i] = p.wrd[i];
                p.wrn[i] = 0;
            }
            if (v >= p.wrd[i]) {
                if (clamped) {
                    p.wrn[i] = LaneWrCap + 1;
                } else {
                    LaneWrAdd(p.wr[i], p.wrn[i], rlo, rhi);
                }
                p.wtsc[i] = __rdtsc();
            } else if (v == p.wtick2[i]) {
                if (clamped) {
                    p.wrn2[i] = LaneWrCap + 1;
                } else {
                    LaneWrAdd(p.wr2[i], p.wrn2[i], rlo, rhi);
                }
            }
        }
        if (v > p.wrd[i]) {
            p.wrd[i] = v;
            p.wsite[i] = static_cast<u8>(site);
            p.wid[i] = g_lane_stamp_id;
            p.wargs[i] = g_lane_stamp_args;
            p.wflip[i] = static_cast<u16>(g_back_flip_seq);
        }
    }
}
/// q38: the read tick window k of probe p (base plo) holds for a write [lo, hi).
inline u64 LaneWinRead(const LaneProbe& p, u64 k, u64 plo, u64 lo, u64 hi) noexcept {
    if (!g_lane_win_extent) {
        return p.wrd[k];
    }
    if (k == 255 && p.hi.load(std::memory_order_relaxed) - plo > (u64{64} << 20)) {
        return p.wrd[k]; // q38b (review 1): window 255 also collects everything past 64 MiB - no extent there
    }
    const u64 wb = plo + (k << 18);
    const bool over = p.wxlo[k] < p.wxhi[k] && wb + p.wxlo[k] < hi && lo < wb + p.wxhi[k];
    const u64 r = std::max(p.wprev[k], over ? p.wrd[k] : 0);
    g_lane_ext_saved += r < p.wrd[k] ? 1 : 0;
    return r;
}
/// q38: a window whose tick is set without a byte range (growth, re-base): full extent.
inline void LaneWinFull(LaneProbe& p, u64 k) noexcept {
    p.wprev[k] = p.wrd[k];
    p.wxlo[k] = 0;
    p.wxhi[k] = 1u << 18;
}
/// q59: overlap class (see g_lane_war_exact) of window k's read tick r for a lane write [lo, hi), and its stamp tsc.
inline void LaneWarClass(const LaneProbe& p, u64 k, u64 plo, u64 lo, u64 hi, u64 r, u8& cls, u64& tsc) noexcept {
    const bool cur = r == p.wrd[k];
    if (!cur && (r != p.wtick2[k] || r == 0)) {
        cls = 3; // older than the two ticks kept
        tsc = 0;
        return;
    }
    tsc = cur ? p.wtsc[k] : p.wtsc2[k];
    if (cur && p.wsite[k] == 22) {
        cls = 5;
        return;
    }
    const u8 n = cur ? p.wrn[k] : p.wrn2[k];
    const auto& rr = cur ? p.wr[k] : p.wr2[k];
    if (n == 0 || n > LaneWrCap) {
        cls = 6;
        return;
    }
    const u64 wb = plo + (k << 18);
    for (u32 j = 0; j < n; ++j) {
        if (wb + rr[2 * j] < hi && lo < wb + rr[2 * j + 1]) {
            cls = 1;
            return;
        }
    }
    cls = 2;
}
inline std::atomic<u32> g_lprobe_n{0};
/// 0..4 open - newest read (4 = 4+), 5 read done, 6 never read, 7 newest read after the newest gfx label,
/// 8 written lane bindings over a probe, 9 A2 stamps, 10 old-path / copy stamps, 11 registered.
inline std::array<u64, 12> g_lprobe_dist{};
/// open - g_gfx_label_tick at the same bindings (0..4 as above, 5 done).
inline std::array<u64, 6> g_label_dist{};
inline bool LaneProbeAny(u64 lo, u64 hi) noexcept {
    const u32 n = g_lprobe_n.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_lprobe[i].lo.load(std::memory_order_relaxed) < hi &&
            lo < g_lprobe[i].hi.load(std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}
/// q29 E6: probe read ranges stamped inside the current graphics BindResources (back thread).
inline const bool g_lane_repeat_exact = [] {
    const char* e = std::getenv("SHADPS4_LANE_REPEAT_EXACT");
    return e && e[0] == '1';
}();
inline bool g_draw_stamps_on = false;   // inside a graphics BindResources
inline bool g_draw_stamps_valid = false; // the list describes the last full graphics bind
inline bool g_draw_stamps_over = false;
inline u32 g_draw_stamps_n = 0;
inline std::array<std::pair<u64, u64>, 64> g_draw_stamps{};
inline void LaneProbeRead(u64 lo, u64 hi, u64 tick, u32 ctr, u32 site = 10) noexcept {
    const u32 n = g_lprobe_n.load(std::memory_order_relaxed);
    if (g_draw_stamps_on) [[unlikely]] {
        if (g_draw_stamps_n < g_draw_stamps.size()) {
            g_draw_stamps[g_draw_stamps_n++] = {lo, hi};
        } else {
            g_draw_stamps_over = true;
        }
    }
    for (u32 i = 0; i < n; ++i) {
        auto& p = g_lprobe[i];
        if (p.lo.load(std::memory_order_relaxed) < hi && lo < p.hi.load(std::memory_order_relaxed)) {
            p.rd = std::max(p.rd, tick);
            ++g_lprobe_dist[ctr];
            if (site == 15 && g_lane_seg_census) [[unlikely]] {
                const u64 sz = hi - lo; // q34
                ++g_s15_size_hist[sz < (64u << 10) ? 0 : sz < (256u << 10) ? 1 : sz < (1u << 20) ? 2 : sz < (4u << 20) ? 3
                                  : sz < (12u << 20) ? 4 : sz < (32u << 20) ? 5 : sz < (128u << 20) ? 6 : 7];
            }
            if (g_lane_win != 0) {
                // q26 E1 / E3-lite: =2 stores lookup stamps pre-adjusted (+1); A2 publish stamps (9) are exact.
                LaneProbeWindows(p, lo, hi, (g_lane_win == 2 && site != 9) ? tick + 1 : tick, site);
            }
        }
    }
}
/// Stamps every probe (a read whose range is unknown: a stretch begun before a probe change).
inline void LaneProbeReadAll(u64 tick, u32 site = 18) noexcept {
    const u32 n = g_lprobe_n.load(std::memory_order_relaxed);
    ++g_lprobe_all_c[site & 31];
    const u64 v = g_lane_win == 2 ? tick + 1 : tick;
    for (u32 i = 0; i < n; ++i) {
        g_lprobe[i].rd = std::max(g_lprobe[i].rd, tick);
        if (g_lane_win != 0 && v > g_lprobe[i].all) {
            g_lprobe[i].all = v;
            g_lprobe[i].all_site = static_cast<u8>(site);
        }
    }
}
/// A written lane binding (back): the newest main read tick over the probes it overlaps. A big binding
/// over no probe registers one; one that sticks out of the probe it overlaps extends it (<= 64 MiB).
/// A new or extended range has unknown history: its newest read becomes `open` (conservative).
/// `hit`: a probe overlapped before this call (census). Returns true when one probe contains
/// [lo, hi) after the call (the WAR wait may be scoped to `rd`).
inline bool LaneProbeWrite(u64 lo, u64 hi, u64 open, u64& rd, bool& hit) noexcept {
    rd = 0;
    hit = false;
    g_lane_war_exact = 0; // q59
    g_lane_war_tsc = 0;
    bool covered = false;
    const u32 n = g_lprobe_n.load(std::memory_order_relaxed);
    u32 first = n;
    for (u32 i = 0; i < n; ++i) {
        auto& p = g_lprobe[i];
        const u64 plo = p.lo.load(std::memory_order_relaxed);
        const u64 phi = p.hi.load(std::memory_order_relaxed);
        if (plo < hi && lo < phi) {
            hit = true;
            u64 r = p.rd;
            u32 s = 10;
            u64 wid = 0, wargs = 0;
            u16 wflip = 0;
            u8 wc = 4; // q59
            u64 wt = 0;
            if (g_lane_win != 0) {
                // q26 E1: the newest read over this write's windows only (+ the probe's floor).
                r = p.all;
                s = p.all_site;
                wc = (s == 21 || s == 22) ? 5 : 4;
                const u64 a = std::max(lo, plo), b = std::min(hi, phi);
                const u64 i0 = std::min<u64>((a - plo) >> 18, 255), i1 = std::min<u64>((b - 1 - plo) >> 18, 255);
                for (u64 k = i0; k <= i1; ++k) {
                    const u64 wr = LaneWinRead(p, k, plo, lo, hi); // q38
                    if (wr > r) {
                        r = wr;
                        s = p.wsite[k];
                        wid = p.wid[k];
                        wargs = p.wargs[k];
                        wflip = p.wflip[k];
                        g_lane_war_win = plo + (k << 18); // q36c census
                        if (g_lane_wait_census) [[unlikely]] {
                            LaneWarClass(p, k, plo, lo, hi, wr, wc, wt); // q59
                        }
                    }
                }
            }
            if (r > rd) {
                rd = r;
                g_lane_war_site = s;
                g_lane_war_wid = g_lane_seg_census ? LaneH5Meta(wid, wargs, wflip, lo) : wid;
                g_lane_war_args = wargs; // q36c census
                g_lane_war_exact = wc;   // q59
                g_lane_war_tsc = wt;
            }
            first = std::min(first, i);
            covered = covered || (plo <= lo && hi <= phi);
        }
    }
    if (g_lane_probe_merge && g_lane_win != 0 && !covered) {
        // q30 E4: grow the overlapping / nearest probe (window ticks kept), else register an aligned one.
        constexpr u64 W = u64{1} << 18;
        constexpr u64 MaxSpan = u64{64} << 20;
        const u64 alo = lo & ~(W - 1), ahi = (hi + W - 1) & ~(W - 1);
        u32 pick = hit ? first : n;
        if (!hit) {
            u64 best = (u64{4} << 20) + 1;
            for (u32 i = 0; i < n; ++i) {
                const u64 plo = g_lprobe[i].lo.load(std::memory_order_relaxed);
                const u64 phi = g_lprobe[i].hi.load(std::memory_order_relaxed);
                const u64 gap = phi <= alo ? alo - phi : (ahi <= plo ? plo - ahi : 0);
                if (gap < best && std::max(phi, ahi) - std::min(plo, alo) <= MaxSpan) {
                    best = gap;
                    pick = i;
                }
            }
        }
        if (pick < n) {
            auto& p = g_lprobe[pick];
            const u64 plo = p.lo.load(std::memory_order_relaxed);
            const u64 phi = p.hi.load(std::memory_order_relaxed);
            const u64 nlo = std::min(alo, plo), nhi = std::max(ahi, phi);
            if (nhi - nlo <= MaxSpan && (plo & (W - 1)) == 0 && (phi & (W - 1)) == 0) { // q30b: aligned both ends
                const u64 shift = (plo - nlo) >> 18;
                const u64 oldn = std::min<u64>((phi - plo + W - 1) >> 18, 256);
                if (shift != 0) {
                    for (u64 i = 256; i-- > shift;) {
                        p.wrd[i] = p.wrd[i - shift];
                        p.wsite[i] = p.wsite[i - shift];
                        p.wprev[i] = p.wprev[i - shift]; // q38
                        p.wxlo[i] = p.wxlo[i - shift];
                        p.wxhi[i] = p.wxhi[i - shift];
                        p.wr[i] = p.wr[i - shift]; // q59
                        p.wrn[i] = p.wrn[i - shift];
                        p.wtsc[i] = p.wtsc[i - shift];
                        p.wr2[i] = p.wr2[i - shift];
                        p.wrn2[i] = p.wrn2[i - shift];
                        p.wtsc2[i] = p.wtsc2[i - shift];
                        p.wtick2[i] = p.wtick2[i - shift];
                    }
                }
                const u64 newn = std::min<u64>((nhi - nlo) >> 18, 256);
                for (u64 i = 0; i < newn; ++i) {
                    if (i < shift || i >= shift + oldn) {
                        p.wrd[i] = open; // newly covered: earlier main reads were never stamped
                        p.wsite[i] = 22;
                        LaneWinFull(p, i); // q38
                    }
                }
                p.lo.store(nlo, std::memory_order_relaxed);
                p.hi.store(nhi, std::memory_order_relaxed);
                p.rd = std::max(p.rd, open);
                g_lprobe_gen.fetch_add(1, std::memory_order_release);
                u64 r = p.all;
                u32 s = p.all_site;
                const u64 i0 = (lo - nlo) >> 18, i1 = std::min<u64>((hi - 1 - nlo) >> 18, 255);
                for (u64 k = i0; k <= i1; ++k) {
                    const u64 wr = LaneWinRead(p, k, nlo, lo, hi); // q38
                    if (wr > r) {
                        r = wr;
                        s = p.wsite[k];
                    }
                }
                if (r > rd) {
                    rd = r;
                    g_lane_war_site = s;
                    g_lane_war_exact = 7; // q59: grown probe rescan
                    g_lane_war_tsc = 0;
                }
                return true;
            }
        }
        if (!hit && n < g_lprobe.size() && ahi - alo <= MaxSpan) {
            g_lprobe[n].lo.store(alo, std::memory_order_relaxed);
            g_lprobe[n].hi.store(ahi, std::memory_order_relaxed);
            g_lprobe[n].rd = open;
            g_lprobe[n].all = open;
            g_lprobe[n].all_site = 21;
            g_lane_war_site = 21;
            g_lane_war_exact = 5; // q59
            g_lprobe_n.store(n + 1, std::memory_order_release);
            g_lprobe_gen.fetch_add(1, std::memory_order_release);
            ++g_lprobe_dist[11];
            rd = open;
            return true;
        }
        return covered;
    }
    if (!hit && hi - lo >= (g_lane_small_probes ? u64{4096} : (u64{1} << 20)) &&
        n < (g_lane_small_probes ? g_lprobe.size() : 16)) {
        g_lprobe[n].lo.store(lo, std::memory_order_relaxed);
        g_lprobe[n].hi.store(hi, std::memory_order_relaxed);
        g_lprobe[n].rd = open;
        g_lprobe[n].all = open; // q26 E1: unknown history
        g_lprobe[n].all_site = 21;
        g_lane_war_site = 21;
        g_lane_war_exact = 5; // q59
        g_lprobe_n.store(n + 1, std::memory_order_release);
        g_lprobe_gen.fetch_add(1, std::memory_order_release);
        ++g_lprobe_dist[11];
        rd = open;
        return true;
    }
    if (hit && !covered) {
        auto& p = g_lprobe[first];
        const u64 nlo = std::min(lo, p.lo.load(std::memory_order_relaxed));
        const u64 nhi = std::max(hi, p.hi.load(std::memory_order_relaxed));
        if (nhi - nlo <= (u64{64} << 20)) {
            p.lo.store(nlo, std::memory_order_relaxed);
            p.hi.store(nhi, std::memory_order_relaxed);
            p.rd = std::max(p.rd, open);
            if (g_lane_win != 0) {
                // q26 E1: re-based windows; open covers every earlier stamp (conservative, as rd).
                p.all = std::max(p.all, open);
                p.all_site = 22;
                p.wrd.fill(0);
                p.wsite.fill(0);
                p.wprev.fill(0); // q38: the floor `all` covers every earlier stamp
                p.wxlo.fill(0);
                p.wxhi.fill(0);
            }
            g_lane_war_site = 22;
            g_lane_war_exact = 5; // q59
            g_lprobe_gen.fetch_add(1, std::memory_order_release);
            rd = std::max(rd, open);
            covered = true;
        }
    }
    return covered;
}
/// q31 E8: one probe contains [lo, hi) and nothing at or after `open` read it (window ticks / floor < open).
inline bool LaneProbeQuiet(u64 lo, u64 hi, u64 open) noexcept {
    if (g_lane_win != 2 || hi <= lo) { // q31b: stored values are the required wait only under =2
        return false;
    }
    bool contained = false;
    const u32 n = g_lprobe_n.load(std::memory_order_relaxed);
    for (u32 i = 0; i < n; ++i) {
        const auto& p = g_lprobe[i];
        const u64 plo = p.lo.load(std::memory_order_relaxed);
        const u64 phi = p.hi.load(std::memory_order_relaxed);
        if (!(plo < hi && lo < phi)) {
            continue;
        }
        // q31b: every overlapping probe (LaneProbeWrite takes the max over all of them).
        contained = contained || (plo <= lo && hi <= phi);
        if (p.all >= open) {
            return false;
        }
        const u64 a = std::max(lo, plo), b = std::min(hi, phi);
        const u64 i0 = std::min<u64>((a - plo) >> 18, 255), i1 = std::min<u64>((b - 1 - plo) >> 18, 255);
        for (u64 k = i0; k <= i1; ++k) {
            if (LaneWinRead(p, k, plo, lo, hi) >= open) { // q38
                return false;
            }
        }
    }
    return contained;
}
/// q31b: SHADPS4_LANE_BIND_UPLOADS and ObtainBuffer's quiet-before-stamp capture (back thread).
inline const bool g_lane_bind_uploads = [] {
    const char* e = std::getenv("SHADPS4_LANE_BIND_UPLOADS");
    return e && e[0] == '1';
}();
inline bool g_q31_pre = false;
inline u64 g_q31_pre_lo = 0, g_q31_pre_hi = 0, g_q31_pre_tick = 0;
/// q9c: stamps every probe (a main read of unknown range) - back thread, outside lane mode.
inline void LaneProbeReadAllMain(u64 tick, u32 site = 19) noexcept {
    if (g_lane_war_mode == 1 && g_lprobe_n.load(std::memory_order_relaxed) != 0) {
        if (site == 19 && g_lane_repeat_exact && g_draw_stamps_valid && !g_draw_stamps_over) {
            // q29 E6: the repeated draw binds what the last full graphics bind bound.
            for (u32 i = 0; i < g_draw_stamps_n; ++i) {
                LaneProbeRead(g_draw_stamps[i].first, g_draw_stamps[i].second, tick, 10, 24);
            }
            return;
        }
        g_war_dma_all.fetch_add(1, std::memory_order_relaxed);
        LaneProbeReadAll(tick, site);
    }
}
/// Defined in buffer_cache.cpp (back thread): tallies one lane binding's producer answer / probe.
void LaneWhoNote(u64 t, u64 open, bool written, u64 rd, bool hit);
void LaneWhoKept(u32 reason, u64 hash);

/// p1: the newest unfinished main tick that wrote or uploaded [lo, hi) (0: none known).
inline u64 LaneMainProducerFull(u64 lo, u64 hi) noexcept {
    u64 t = 0;
    if (hi <= lo) {
        return 0;
    }
    const u64 w0 = lo >> 16, w1 = (hi - 1) >> 16;
    g_lc_src = 0;
    g_lc_hit_wk = 0;
    g_lc_hit_wid = 0;
    g_lc_hit_exact = 0; // q59
    g_lc_hit_tsc = 0;
    if (w1 - w0 > 63) {
        // p1e: more than the 64 windows LaneFind reads: the per-window newest ticks.
        if (w1 - w0 > 1023) {
            t = g_lc_main_newest;
            g_lc_src = 1;
            g_lc_hit_wk = 251;
            g_lc_hit_exact = 9; // q59
        } else {
            // p1g: the window table pre-filters; a flagged window is confirmed against the notes
            // of its bucket ring (exact range overlap). Bucket rings hold several windows' notes
            // (hashed), so a note found there must overlap [lo, hi) itself.
            for (u64 w = w0; w <= w1; ++w) {
                if (g_lane_producer_super == 1 && ((w & 63) == 0 || w == w0)) {
                    // q39: a whole 4 MiB block with no unfinished main note above the answer so far.
                    if (g_lc_super[LaneSuper(w >> 6)] <= std::max(g_lc_done, t)) {
                        ++g_lc_super_c[0];
                        w |= 63; // the loop's ++w moves to the next block
                        continue;
                    }
                }
                if (g_lc_win[LaneWin(w)] <= std::max(g_lc_done, t)) {
                    continue;
                }
                for (const LaneNote& e : g_lc_main.b[LaneBucket(w)]) {
                    if (e.seq != 0 && e.tick > g_lc_done && e.tick > t && e.lo < hi && lo < e.hi) {
                        t = e.tick;
                        g_lc_src = 2;
                        g_lc_hit_wk = e.wk;
                        g_lc_hit_wid = e.wid;
                        g_lc_hit_exact = 8; // q59
                        g_lc_hit_tsc = e.tsc;
                    }
                }
            }
        }
    } else if (const LaneNote* m = LaneFind(g_lc_main, lo, hi)) {
        t = m->tick;
        g_lc_hit_wk = m->wk;
        g_lc_hit_wid = m->wid;
        g_lc_hit_exact = 8; // q59
        g_lc_hit_tsc = m->tsc;
    }
    const auto raise = [&](u64 v, u32 src) {
        if (v > g_lc_done && v > t) {
            t = v;
            g_lc_src = src;
            g_lc_hit_wk = 250 + src;
            g_lc_hit_wid = 0;
            g_lc_hit_exact = 9; // q59
            g_lc_hit_tsc = 0;
        }
    };
    if (!g_lc_fast || g_lc_lost_max > g_lc_done) { // q28: no lost note is unfinished -> nothing to raise
        for (u64 w = w0; w <= w1 && w <= w0 + 1023; ++w) {
            raise(g_lc_lost_win[LaneWin(w)], 6); // a lost unfinished note in this window
        }
    }
    if (!g_lc_fast || g_lc_big_max > g_lc_done) { // q28
        for (const LaneBig& b : g_lc_big) {
            if (b.lo < hi && lo < b.hi) {
                raise(b.tick, 5);
            }
        }
    }
    raise(g_lc_big_evicted, 6); // a big note was lost: assume it covered this range
    if (t <= g_lc_done) {
        t = 0;
    }
    g_lc_coarse = g_lc_src != 0;
    return t;
}
/// q25: SHADPS4_LANE_PRODUCER_MEMO (1 memo, 2 verify).
inline const int g_lc_memo_mode = [] {
    const char* e = std::getenv("SHADPS4_LANE_PRODUCER_MEMO");
    return e && (e[0] == '1' || e[0] == '2') ? e[0] - '0' : 0;
}();
struct LaneProdMemo {
    u64 lo = 0, hi = 0, seq = 0, t = 0, wid = 0;
    u32 src = 0, wk = 0;
};
inline std::array<LaneProdMemo, 16> g_lc_memo{};
inline std::array<u64, 4> g_lc_memo_c{}; // hits, misses, verified, mismatches
inline u64 LaneMainProducerSuperCheck(u64 lo, u64 hi) noexcept {
    // q39 =2: the p1g answer with the super filter must equal the plain one (same tick).
    const u64 w0 = lo >> 16, w1 = (hi - 1) >> 16;
    if (hi > lo && w1 - w0 > 63 && w1 - w0 <= 1023) {
        u64 a = 0, b = 0;
        for (u64 w = w0; w <= w1; ++w) {
            if (((w & 63) == 0 || w == w0) && g_lc_super[LaneSuper(w >> 6)] <= std::max(g_lc_done, a)) {
                w |= 63;
                continue;
            }
            if (g_lc_win[LaneWin(w)] <= std::max(g_lc_done, a)) {
                continue;
            }
            for (const LaneNote& e : g_lc_main.b[LaneBucket(w)]) {
                if (e.seq != 0 && e.tick > g_lc_done && e.tick > a && e.lo < hi && lo < e.hi) {
                    a = e.tick;
                }
            }
        }
        for (u64 w = w0; w <= w1; ++w) {
            if (g_lc_win[LaneWin(w)] <= std::max(g_lc_done, b)) {
                continue;
            }
            for (const LaneNote& e : g_lc_main.b[LaneBucket(w)]) {
                if (e.seq != 0 && e.tick > g_lc_done && e.tick > b && e.lo < hi && lo < e.hi) {
                    b = e.tick;
                }
            }
        }
        ++g_lc_super_c[1];
        g_lc_super_c[2] += a != b ? 1 : 0;
    }
    return LaneMainProducerFull(lo, hi);
}
inline u64 LaneMainProducer(u64 lo, u64 hi) noexcept {
    if (g_lane_producer_super == 2) [[unlikely]] {
        return LaneMainProducerSuperCheck(lo, hi);
    }
    if (g_lc_memo_mode == 0 || hi <= lo) {
        return LaneMainProducerFull(lo, hi);
    }
    LaneProdMemo& m = g_lc_memo[((lo >> 12) ^ ((hi >> 12) * 0x9E3779B1ULL)) & 15];
    if (m.seq != 0 && m.seq == g_lc_main_seq && m.lo == lo && m.hi == hi) {
        const u64 t = m.t > g_lc_done ? m.t : 0;
        ++g_lc_memo_c[0];
        if (g_lc_memo_mode == 2) {
            const u64 full = LaneMainProducerFull(lo, hi);
            ++g_lc_memo_c[2];
            g_lc_memo_c[3] += full != t ? 1 : 0;
            return full;
        }
        g_lc_src = t != 0 ? m.src : 0;
        g_lc_hit_wk = t != 0 ? m.wk : 0;
        g_lc_hit_wid = t != 0 ? m.wid : 0;
        g_lc_coarse = g_lc_src != 0;
        return t;
    }
    ++g_lc_memo_c[1];
    const u64 t = LaneMainProducerFull(lo, hi);
    m = {lo, hi, g_lc_main_seq, t, g_lc_hit_wid, g_lc_src, g_lc_hit_wk};
    return t;
}
inline void EarlyInvPrune(u64 read_pos) noexcept {
    while (g_early_w_head != g_early_w_tail &&
           g_early_writes[g_early_w_head % g_early_writes.size()].pos < read_pos) {
        ++g_early_w_head;
    }
}

inline const int g_native_coherent = [] {
    const char* env = std::getenv("SHADPS4_NATIVE_COHERENT");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

class DirtyLog {
public:
    static constexpr u64 Capacity = 1ULL << 16;
    static constexpr u64 Mask = Capacity - 1;
    static constexpr u64 TagMask = (1ULL << 24) - 1;

    /// Any thread, under the region lock of the pages, before their tracker bits are set.
    /// `count` is at most NUM_PAGES_PER_REGION (one region per entry).
    static void Append(u64 first_page, u64 count) noexcept {
        if (g_back_clean == 0) {
            return;
        }
        const u64 index = Tail().fetch_add(1, std::memory_order_seq_cst);
        // Bit 10: appended by the CP thread (its own guest writes; NATIVE_COHERENT=2 classifies).
        const u64 cp = Common::FaultMonitor::t_is_cp_thread ? (1ULL << 10) : 0;
        slots[index & Mask].store(((index & TagMask) << 40) | (first_page << 12) | cp | (count - 1),
                                  std::memory_order_release);
    }

    /// As Drain, but stops at `end` (clamped to the tail) and passes f(first, count, by_cp).
    /// A second consumer with its own head (NATIVE_COHERENT). Returns false if entries were lost
    /// (head is then the tail).
    template <typename F>
    static bool DrainTo(u64& head, u64 end, F&& f) {
        const u64 tail = Tail().load(std::memory_order_seq_cst);
        if (end > tail) {
            end = tail;
        }
        if (end <= head) [[likely]] {
            return true;
        }
        if (tail - head > Capacity / 2) {
            head = tail;
            return false;
        }
        while (head != end) {
            const u64 want = head & TagMask;
            u64 value = slots[head & Mask].load(std::memory_order_acquire);
            for (u32 spins = 0; (value >> 40) != want; ++spins) {
                if (spins == 2048 || Tail().load(std::memory_order_relaxed) - head >= Capacity) {
                    head = Tail().load(std::memory_order_seq_cst);
                    return false;
                }
                _mm_pause();
                value = slots[head & Mask].load(std::memory_order_acquire);
            }
            f((value >> 12) & 0xfffffffULL, (value & 0x3ff) + 1, (value & (1ULL << 10)) != 0);
            ++head;
        }
        return true;
    }

    /// The single consumer (CP back). Calls f(first_page, count) for every entry appended before
    /// the call. Returns false if entries were lost: the caller must forget everything it knows.
    template <typename F>
    static bool Drain(u64& head, F&& f) {
        const u64 end = Tail().load(std::memory_order_seq_cst);
        if (end == head) [[likely]] {
            return true;
        }
        if (end - head > Capacity / 2) {
            head = end;
            return false;
        }
        while (head != end) {
            const u64 want = head & TagMask;
            u64 value = slots[head & Mask].load(std::memory_order_acquire);
            for (u32 spins = 0; (value >> 40) != want; ++spins) {
                // The appender is between its fetch_add and its store (it holds a region lock),
                // or the ring lapped us. Do not wait long: the appender may be suspended
                // (debugger pause, stack sampler, shutdown), and unlike the region lock this
                // wait would stall every read sync. Giving up is the overflow path: all
                // verified bits are dropped, and the skipped appender took its region lock
                // before this tail read, so the next locked walk of its pages waits for it.
                if (spins == 2048 || Tail().load(std::memory_order_relaxed) - head >= Capacity) {
                    head = Tail().load(std::memory_order_seq_cst);
                    return false;
                }
                _mm_pause();
                value = slots[head & Mask].load(std::memory_order_acquire);
            }
            f((value >> 12) & 0xfffffffULL, (value & 0x3ff) + 1);
            ++head;
        }
        return true;
    }

    static u64 Start() noexcept {
        return Tail().load(std::memory_order_seq_cst);
    }

private:
    static_assert(NUM_PAGES_PER_REGION <= 1024, "count - 1 must fit 10 bits");
    static_assert(40 - TRACKER_PAGE_BITS <= 28, "page number must fit 28 bits");
    struct alignas(64) PaddedCounter {
        std::atomic<u64> value;
    };
    static_assert(sizeof(PaddedCounter) == 64);
    // Starts one lap in, so no index of the first lap carries tag 0 (the slots' initial value).
    static inline PaddedCounter tail_storage{Capacity};
    static std::atomic<u64>& Tail() noexcept {
        return tail_storage.value;
    }
    alignas(64) static inline std::array<std::atomic<u64>, Capacity> slots{};
};

/// The back's private "verified clean" bits. One thread only.
class BackCleanCache {
public:
    using Run = std::pair<u64, u64>; ///< first page, page count

    struct Stats {
        u64 queries{}, clean{}, runs{}, run_pages{}, drained{}, resets{}, raced{}, mismatches{};
    };
    Stats stats;

    BackCleanCache() {
        if (g_back_clean != 0) {
            top = std::make_unique<Region*[]>(NumRegions);
            full = std::make_unique<u64[]>(NumRegions / 64);
            head = DirtyLog::Start();
        }
    }

    /// Applies every dirty marking made since the last call.
    void Sync() {
        const bool ok = DirtyLog::Drain(head, [this](u64 first_page, u64 count) {
            ++stats.drained;
            Clear(first_page, count);
        });
        if (!ok) [[unlikely]] {
            ++generation;
            ++stats.resets;
            std::fill_n(full.get(), NumRegions / 64, u64{0});
        }
    }

    /// The dirty-log index the last Sync() drained up to (entries before it are applied).
    u64 DrainedTo() const noexcept {
        return head;
    }

    /// Appends the page runs of [addr, addr + size) that are not verified clean, merged.
    void CollectUnverified(VAddr addr, u64 size, std::vector<Run>& runs) const {
        u64 page = addr >> TRACKER_PAGE_BITS;
        const u64 end = (addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
            // Fully verified regions are skipped through the summary, 64 regions per word:
            // the average query spans ~11 regions (4 MiB each) and a per-region block lookup
            // cost as much as the tracker walk it replaced (profile 2026-10-01).
            const u32 skip = static_cast<u32>(std::countr_one(full[region / 64] >> (region % 64)));
            if (skip != 0) {
                page = (region + skip) * NUM_PAGES_PER_REGION;
                continue;
            }
            const u64 base = region * NUM_PAGES_PER_REGION;
            const u64 lo = page - base;
            const u64 hi = std::min<u64>(end - base, NUM_PAGES_PER_REGION);
            const Region* r = top[region];
            if (!r || r->generation != generation) {
                AddRun(runs, page, hi - lo);
            } else if (r->count != NUM_PAGES_PER_REGION) {
                for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                    u64 missing = ~r->bits[word] & WordMask(word, lo, hi);
                    while (missing != 0) {
                        const u64 start = std::countr_zero(missing);
                        const u64 length = std::countr_one(missing >> start);
                        AddRun(runs, base + word * 64 + start, length);
                        missing &= length == 64 ? 0 : ~(((1ULL << length) - 1) << start);
                    }
                }
            }
            page = base + hi;
        }
    }

    /// True when CollectUnverified(addr, size) would append nothing: every page of the range is
    /// verified clean. Same walk, stops at the first unverified page and builds no run list.
    bool AllVerified(VAddr addr, u64 size) const {
        u64 page = addr >> TRACKER_PAGE_BITS;
        const u64 end = (addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
            const u32 skip = static_cast<u32>(std::countr_one(full[region / 64] >> (region % 64)));
            if (skip != 0) {
                page = (region + skip) * NUM_PAGES_PER_REGION;
                continue;
            }
            const u64 base = region * NUM_PAGES_PER_REGION;
            const u64 lo = page - base;
            const u64 hi = std::min<u64>(end - base, NUM_PAGES_PER_REGION);
            const Region* r = top[region];
            if (!r || r->generation != generation) {
                return false;
            }
            if (r->count != NUM_PAGES_PER_REGION) {
                for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                    if ((~r->bits[word] & WordMask(word, lo, hi)) != 0) {
                        return false;
                    }
                }
            }
            page = base + hi;
        }
        return true;
    }

    /// The locked walk took these pages: each is clean in the tracker, or will be logged again.
    void MarkVerified(u64 first_page, u64 count) {
        u64 page = first_page;
        const u64 end = first_page + count;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
            const u64 base = region * NUM_PAGES_PER_REGION;
            const u64 lo = page - base;
            const u64 hi = std::min<u64>(end - base, NUM_PAGES_PER_REGION);
            Region*& r = top[region];
            if (!r) {
                r = &pool.emplace_back();
            }
            if (r->generation != generation) {
                r->generation = generation;
                r->count = 0;
                r->bits.fill(0);
            }
            for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                const u64 mask = WordMask(word, lo, hi);
                r->count += static_cast<u32>(std::popcount(mask & ~r->bits[word]));
                r->bits[word] |= mask;
            }
            if (r->count == NUM_PAGES_PER_REGION) {
                full[region / 64] |= 1ULL << (region % 64);
            }
            page = base + hi;
        }
    }

    static constexpr u64 NumRegions = 1ULL << (40 - TRACKER_HIGHER_PAGE_BITS);

private:
    struct alignas(64) Region {
        u64 generation{};
        u32 count{}; ///< set bits
        std::array<u64, NUM_PAGES_PER_REGION / 64> bits{};
    };

    /// Bits of `word` that lie in the region-relative page range [lo, hi).
    static u64 WordMask(u64 word, u64 lo, u64 hi) {
        const u64 first = std::max<u64>(lo, word * 64) - word * 64;
        const u64 last = std::min<u64>(hi, word * 64 + 64) - word * 64; // exclusive, 1..64
        const u64 upto = last == 64 ? ~0ULL : (1ULL << last) - 1;
        return upto & ~((1ULL << first) - 1);
    }

    static void AddRun(std::vector<Run>& runs, u64 first_page, u64 count) {
        if (!runs.empty() && runs.back().first + runs.back().second == first_page) {
            runs.back().second += count;
        } else {
            runs.emplace_back(first_page, count);
        }
    }

    void Clear(u64 first_page, u64 count) {
        const u64 region = first_page / NUM_PAGES_PER_REGION;
        if (region >= NumRegions) {
            return;
        }
        Region* r = top[region];
        if (!r || r->generation != generation) {
            return;
        }
        full[region / 64] &= ~(1ULL << (region % 64));
        const u64 lo = first_page - region * NUM_PAGES_PER_REGION;
        const u64 hi = std::min<u64>(lo + count, NUM_PAGES_PER_REGION);
        for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
            const u64 mask = WordMask(word, lo, hi);
            r->count -= static_cast<u32>(std::popcount(mask & r->bits[word]));
            r->bits[word] &= ~mask;
        }
    }

    std::unique_ptr<Region*[]> top;
    /// One bit per region: every page of it is verified (current generation). Private.
    std::unique_ptr<u64[]> full;
    std::deque<Region> pool;
    u64 generation{1};
    u64 head{};
};

} // namespace VideoCore
