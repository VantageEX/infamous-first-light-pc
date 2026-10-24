// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstddef>
#include <immintrin.h>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <boost/preprocessor/stringize.hpp>

#include "video_core/texture_cache/tex_table.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/decoder.h"
#include "common/fault_monitor.h"
#include "common/fence_watch.h"
#include "video_core/buffer_cache/protect_queue.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/submit_epoch.h"
#include "common/polyfill_thread.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/native_census.h"
#include <xxhash.h>
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "common/readback_offload.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/renderdoc.h"
#include "video_core/amdgpu/lean_front_api.h"
#include "video_core/amdgpu/lean_parse.h"
#include "video_core/amdgpu/split_records.h"
#include "video_core/renderer_vulkan/vk_a2.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/amdgpu/mtb_census.h"
#include "video_core/amdgpu/mtb_conflict.h"

namespace Common::FenceWatch {

void NoteRelease(VAddr addr, u64 value, u32 size) {
    auto& st = Get();
    const u64 now = NowNs();
    Slot* slot = nullptr;
    Slot* oldest = &st.slots[0];
    if (addr >= State::PinnedBase && addr < State::PinnedBase + State::NumPinned * 8) {
        slot = &st.slots[State::NumSlots + (addr - State::PinnedBase) / 8];
    }
    for (size_t i = 0; slot == nullptr && i < State::NumSlots; ++i) {
        Slot& s = st.slots[i];
        const u64 a = s.addr.load(std::memory_order_relaxed);
        if (a == addr || a == 0) {
            slot = &s;
            break;
        }
        if (s.written_ns.load(std::memory_order_relaxed) <
            oldest->written_ns.load(std::memory_order_relaxed)) {
            oldest = &s;
        }
    }
    if (slot == nullptr) {
        slot = oldest;
        slot->releases.store(0, std::memory_order_relaxed);
    }
    slot->value.store(value, std::memory_order_relaxed);
    slot->size.store(size, std::memory_order_relaxed);
    slot->written_ns.store(now, std::memory_order_relaxed);
    slot->addr.store(addr, std::memory_order_release);
    slot->releases.fetch_add(1, std::memory_order_relaxed);
    st.last_release_ns.store(now, std::memory_order_release);
    static std::atomic<u64> total{0};
    const u64 n = total.fetch_add(1, std::memory_order_relaxed);
    if (n == 0 || (n & 0x3fff) == 0) {
        LOG_INFO(Render, "FenceWatch: armed; compute releases {} (latest {:#x} <- {:#x})", n + 1, addr, value);
    }
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread([] {
            u64 reported_for = 0;
            u64 last_report_ns = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                auto& w = Get();
                // The newest write among busy frame-fence slots; without one, any compute release.
                u64 last = 0;
                for (size_t i = State::NumSlots; i < w.slots.size(); ++i) {
                    if (w.slots[i].releases.load(std::memory_order_relaxed) > 100) {
                        last = std::max(last, w.slots[i].written_ns.load(std::memory_order_acquire));
                    }
                }
                if (last == 0) {
                    last = w.last_release_ns.load(std::memory_order_acquire);
                }
                const u64 now = NowNs();
                if (last == 0 || now - last < 2'000'000'000ULL) {
                    continue;
                }
                if (last == reported_for && now - last_report_ns < 10'000'000'000ULL) {
                    continue;
                }
                reported_for = last;
                last_report_ns = now;
                LOG_WARNING(Render, "FenceWatch: no frame-fence release for {:.1f} s (any compute release {:.1f} s ago); "
                            "covering download writes {}; gnm submission_lock {} waiters {} resets {} races {}",
                            double(now - last) / 1e9,
                            double(now - w.last_release_ns.load()) / 1e9, w.covering_writes.load(),
                            w.gnm_lock.load(), w.gnm_waiters.load(), w.gnm_resets.load(),
                            w.gnm_races.load());
                for (const Slot& s : w.slots) {
                    const VAddr a = s.addr.load(std::memory_order_acquire);
                    if (a == 0) {
                        continue;
                    }
                    const u32 sz = s.size.load(std::memory_order_relaxed);
                    const u64 v = s.value.load(std::memory_order_relaxed);
                    u64 live = 0;
                    const bool ok = Core::Memory::Instance()->TryReadBacking(
                        a, reinterpret_cast<u8*>(&live), sz);
                    const char* verdict = !ok          ? "unreadable"
                                          : live == v  ? "equal"
                                          : live < v   ? "LOWER (clobbered?)"
                                                       : "HIGHER";
                    LOG_WARNING(Render,
                                "FenceWatch:   {:#x} last signalled {:#x} ({} B, {:.1f} s ago, {} releases); "
                                "live {:#x}: {}",
                                a, v, sz, double(now - s.written_ns.load()) / 1e9, s.releases.load(),
                                live, verdict);
                }
            }
        }).detach();
    });
}

void NoteGuestWrite(VAddr addr, const void* data, u64 size, const char* who) {
    auto& st = Get();
    for (const Slot& s : st.slots) {
        const VAddr fence = s.addr.load(std::memory_order_acquire);
        const u32 sz = s.size.load(std::memory_order_relaxed);
        if (fence == 0 || fence < addr || fence + sz > addr + size) {
            continue;
        }
        const u64 n = st.covering_writes.fetch_add(1, std::memory_order_relaxed);
        u64 incoming = 0;
        std::memcpy(&incoming, static_cast<const u8*>(data) + (fence - addr), sz);
        const u64 last = s.value.load(std::memory_order_relaxed);
        if (incoming != last || n < 16) {
            LOG_WARNING(Render,
                        "FenceWatch: {} writes {:#x} over fence {:#x} (last signalled {:#x}, {:.3f} ms ago){}",
                        who, incoming, fence, last,
                        double(NowNs() - s.written_ns.load(std::memory_order_relaxed)) / 1e6,
                        incoming != last ? " CLOBBER" : "");
        }
    }
}

} // namespace Common::FenceWatch

namespace AmdGpu {

/// SHADPS4_DRAW_COST=1: cycles the CP back spent in WaitForWork (read and reset by NoteDrawCost).
thread_local u64 g_back_idle_cyc = 0;
/// SHADPS4_ASC_LANE: the early packet being executed records on the lane (back thread).
thread_local bool t_lane_packet = false;

/// SHADPS4_MTB_P1_BREAK=1 (multi-threaded back P1d, notes/parrec-build-2026-10-08.md sec. 14):
/// prices option (b) - every in-batch sequence point (packet record) ends an open render pass, as
/// a worker segment starting after it would; the next draw re-begins the pass (LOAD).
static const bool g_mtb_break = [] {
    const char* value = std::getenv("SHADPS4_MTB_P1_BREAK");
    return value != nullptr && value[0] == '1';
}();
/// SHADPS4_MTB_P3 (with SHADPS4_MTB_P1=1; P3.0, notes/parrec-build-2026-10-08.md sec. 14d): the
/// parked translation worker. =1: it takes the CP identity (t_is_cp_thread) while the back is
/// parked on its run; =2 (probe): it stays a foreign thread - every texture-cache lock then costs
/// a FlushProcessWriteBuffers (s265: 20 FPS).
static const int g_mtb_p3 = [] {
    const char* value = std::getenv("SHADPS4_MTB_P3");
    return value != nullptr && (value[0] == '1' || value[0] == '2') ? value[0] - '0' : 0;
}();

/// SHADPS4_A1=1 (architecture A step A1, notes/NATIVE-RENDERER.md): DcbBegin records and the
/// per-command-buffer census on the back (observation only; every record runs as before).
static const int g_a1 = [] {
    const char* value = std::getenv("SHADPS4_A1");
    return value != nullptr && (value[0] == '1' || value[0] == '2') ? value[0] - '0' : 0;
}();
static u32 g_a1_front_index = 0; ///< front only: DCBs begun
/// SHADPS4_A2 (architecture A step A2, notes/architecture-a-2026-10-10.md sec. 6): 1 workers
/// translate native stretches, the back publishes them; 2 verify / count (old path for all).
static const int g_a2 = Vulkan::A2::Mode();
/// SHADPS4_A2_PARK=1: caught-up workers park (OS wait) after a short spin.
static const bool g_a2_park = [] {
    const char* v = std::getenv("SHADPS4_A2_PARK");
    return v != nullptr && v[0] == '1';
}();
/// SHADPS4_CLAIM_WAKE=1 (2026-10-15, notes/gamemain-saturation-2026-10-15.md sec. 5): an A2 or
/// lean worker waiting for the DCB index it claimed spins at most SHADPS4_CLAIM_SPIN_US
/// (default 2 us) and then sleeps on a bell only that index's publication rings. Off: 2,048
/// pauses (~80 us) per claim and every parked worker woken per DCB (s1100: six A2 workers used
/// 0.89 cores for 33 ms/s of translation).
static const bool g_claim_wake = [] {
    const char* v = std::getenv("SHADPS4_CLAIM_WAKE");
    return v != nullptr && v[0] == '1';
}();
/// SHADPS4_LEAN_PERDRAW bit 2 or 4 (2026-10-15): lean-worker per-draw costs. Here: a finished stretch's
/// register deltas SWAP with its xl_side slot's previous vectors instead of moving out (the
/// worker's vectors keep their capacity; no allocation per stretch). Also read by lean_parse.cpp
/// (bigger / set-associative worker memos) and vk_a2.cpp (sub-batch growth, fence queues).
/// The value is a decimal bit mask: 1 memos, 2 every allocation part (s2 screen: =3 halved FPS,
/// inferred from the sub-batch reserve), 4 this swap alone, 8 the fence queues alone (vk_a2.cpp).
static const bool g_lean_perdraw = [] {
    const char* v = std::getenv("SHADPS4_LEAN_PERDRAW");
    return v != nullptr && ((std::strtoul(v, nullptr, 10) & (2 | 4)) != 0);
}();
static const u64 g_claim_spin_ticks = [] {
    const char* v = std::getenv("SHADPS4_CLAIM_SPIN_US");
    const u64 us = std::min<u64>(v ? std::strtoull(v, nullptr, 10) : 2, 1000); // review NIT: clamp
    if (!g_claim_wake) {
        return u64{0};
    }
    // TSC ticks per us, measured once against the steady clock (~5 ms at startup).
    using Clock = std::chrono::steady_clock;
    const auto c0 = Clock::now();
    const u64 t0 = __rdtsc();
    auto c1 = c0;
    do {
        c1 = Clock::now();
    } while (c1 - c0 < std::chrono::milliseconds(5));
    const double ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
    const double per_us = double(__rdtsc() - t0) / (ns / 1000.0);
    return static_cast<u64>(per_us * double(us));
}();
namespace {
/// SHADPS4_CLAIM_WAKE: per-index wake slots of a claim counter. The publisher stores the index
/// it published into its slot after the counter, so a waiter that read the slot before that
/// store wakes on the change, and one that read it after sees the counter (release / acquire
/// on the slot). Waiters on indices 16 apart share a slot (a spurious wake, re-checked).
struct ClaimBells {
    struct alignas(64) Bell {
        std::atomic<u32> v{0};
    };
    std::array<Bell, 16> b{};
    void Ring(u32 idx) {
        auto& x = b[idx & 15].v;
        x.store(idx, std::memory_order_release);
        x.notify_all();
    }
    void RingAll() {
        for (auto& x : b) {
            x.v.fetch_add(1, std::memory_order_acq_rel);
            x.v.notify_all();
        }
    }
    /// Waits until `published` >= idx; false when `stopping()` turned true first.
    template <class Stop>
    bool Wait(const std::atomic<u32>& published, u32 idx, Stop&& stopping) {
        const u64 t0 = __rdtsc();
        while (published.load(std::memory_order_acquire) < idx) {
            if (stopping()) {
                return false;
            }
            if (__rdtsc() - t0 < g_claim_spin_ticks) {
                _mm_pause();
                continue;
            }
            auto& x = b[idx & 15].v;
            const u32 seen = x.load(std::memory_order_acquire);
            if (published.load(std::memory_order_acquire) < idx && !stopping()) {
                x.wait(seen, std::memory_order_acquire);
            }
        }
        return true;
    }
};
constexpr u32 A2Slots = 256;
/// Stretches one DCB can hand over (more: the rest of the DCB runs on the old path, counted).
constexpr u32 A2MaxStretches = 128;
/// One DCB of the A2 table. state: 0 free, 1 begun (the front wrote its DcbBegin), 2 a worker
/// follows it, 3 the worker finished it, 4 the back took it (old path, whole DCB).
/// The worker hands stretches over while the front still parses the DCB: it fills stretches[k]
/// and side[k], then ready = k + 1 (release), then progress = the newest draw seq it decided
/// (every draw up to it is either in one of the first `ready` stretches or on the old path).
struct A2Entry {
    std::atomic<u32> state{0};
    std::atomic<u32> index{0};
    std::atomic<u64> begin_pos{0}; ///< atomic: a worker reads it before its claim (ring hold)
    u32 refused{}; ///< 1: no snapshot (every draw on the old path)
    // Written by the worker.
    alignas(64) std::atomic<u32> ready{0};
    std::atomic<u64> progress{0};
    std::atomic<u64> walked{0}; ///< ring position the worker last waited at (caught up there)
    // Written by the back.
    alignas(64) std::atomic<u64> back_wait{0}; ///< draw seq the back waits for (0 none)
    std::atomic<u32> cancel{0}; ///< safety net: the worker stops, the rest runs on the old path
    /// Fixed size A2MaxStretches (sized once at start; never reallocated under the back's reads).
    alignas(64) std::vector<std::unique_ptr<Vulkan::A2::Stretch>> stretches;
    /// Per stretch (same index): one past its last draw record, that draw's ctx epoch, and the
    /// register words written between its first and last draw (final values, in order).
    struct Side {
        u64 end_pos{};
        u64 ctx_epoch{};
        std::vector<std::pair<u32, u32>> delta;
        u64 render_key{}; ///< full A2RenderKey at the stretch's first draw (A2=2 verify)
        /// Lean L3 pre-delta: the RegWrite run right before the first draw (its first record's
        /// position, ~0 none), the first draw's position, the run's words at the first draw.
        u64 pre_pos{~u64{0}};
        u64 first_pos{};
        std::vector<std::pair<u32, u32>> pre_delta;
    };
    std::vector<Side> side;
};
struct A2Counters {
    u64 dcbs{}, back_dcbs{}, worker_dcbs{}, waits{}, wait_cyc{};
    u64 translated{}, walked{}, would{}, stretches{}, barriers{}, occl_rejects{};
    u64 delta_words{}, back_records{}, skipped_records{}, gen_rejects{};
    u64 steal_unclaimed{};
    u64 wait_complete{}, wait_complete_cyc{}, steal_front_waits{}, steal_timeout{};
    u64 rk_cmp{}, rk_bad{}, whole_stale{}, whole_refused{}, whole_st4{};
    u64 snap_cmp{}, snap_bad{};
    u64 dwait_behind{}, dwait_behind_cyc{}, end_waits{}, end_wait_cyc{}, net_cancel{};
    u64 col_stretches{}, materials{}, mat_rejects{}, rt_rejects{}, col_resolve_cyc{};
    u64 slot_cmp{}, slot_bad{}, slot_unwritten{};
    u64 start_ns{}, start_tsc{};
};
struct A2State {
    std::array<A2Entry, A2Slots> e;
    alignas(64) std::atomic<u32> next_claim{1};
    alignas(64) std::atomic<u32> begun{0};     ///< newest index the front began (monotonic)
    ClaimBells bells;                           ///< SHADPS4_CLAIM_WAKE: wakes for `begun`
    alignas(64) std::atomic<u32> published{0}; ///< newest index the front completed (diagnostics)
    alignas(64) std::atomic<bool> stop{false};
    std::atomic<u64> worker_cyc{0}, worker_dcbs{0}, worker_draws{0}; ///< worker-side totals
    std::atomic<u64> take_cyc{0}; ///< inside StretchTranslator::TakeDraw
    std::atomic<u64> publish_wait_cyc{0}; ///< workers waiting for the front's next DCB
    std::atomic<u64> record_wait_cyc{0};  ///< workers caught up, waiting for the next record
    std::atomic<u64> park_cyc{0};         ///< of which parked (SHADPS4_A2_PARK)
    std::atomic<u64> cuts_back{0}, cuts_front{0}, cap_hits{0}, seq_breaks{0}, cancels{0};
    /// Per worker: the DCB it holds (0 none) | phase << 28 (1 waits for a DCB, 2 translates,
    /// 3 caught up with the front).
    std::array<std::atomic<u32>, 8> worker_at{};
    std::vector<std::jthread> workers;
    std::vector<Vulkan::A2::StretchTranslator*> translators; ///< for the refusal log
    bool started{false};
    u64 t_start{}; ///< ns, when the workers started
    // back
    A2Entry* cur{};
    bool cur_back{};
    size_t stretch_i{};
    u64 seen_progress{}; ///< the worker's progress as last loaded
    /// Colour verify (A2=2): the stretch being compared, its next SlotWrite, per material the
    /// resolved image / sampler counts, and whether its materials resolved.
    const Vulkan::A2::Stretch* v_stretch{};
    size_t v_slot_k{};
    std::array<std::pair<u32, u32>, Vulkan::A2::MaxMaterials> v_mat_n{};
    bool v_resolved{};
    u32 cur_n{};         ///< ready stretches as loaded with it
    bool reject_cur{}; ///< the current stretch runs on the old path (occlusion open)
    bool gpu_write{true}; ///< a GPU write was recorded since the last global barrier
    u64 skip_pos{}, skip_seq{}; ///< set by A2Execute: BackLoop releases through here
    A2Counters c{};
    // front
    u64 front_begin_pos{};
};
A2State g_a2s;
u64 A2NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
} // namespace

void Liverpool::MtbBreakAtPacket(const PM4Header* header) {
    // The DMA_DATA to 0x3022C is skipped by ExecuteGfxPacket: not a sequence point.
    const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
    if (header->type3.opcode == PM4ItOpcode::DmaData && dma->dst_addr_lo == 0x3022C) {
        return;
    }
    auto& scheduler = rasterizer->GetScheduler();
    if (scheduler.IsRenderingNow()) {
        scheduler.EndRendering();
        scheduler.NoteMtbBreak();
    }
}


namespace {
bool ViewportMemoEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_VIEWPORT_MEMO");
        return value && value[0] >= '1';
    }();
    return enabled;
}

// SHADPS4_IB_TASK_FREE=1 (port of upstream PR #4996): an INDIRECT_BUFFER packet runs a nested
// coroutine whose final_suspend keeps its frame alive; nothing destroyed it, so every IB packet
// leaked one frame.
bool IbTaskFree() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_IB_TASK_FREE");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_ASC_YIELD=1: the graphics task yields to the round-robin at the next packet boundary
/// when the guest submitted async-compute work since it last looked. On GCN the ACEs parse and
/// run compute queues beside the graphics CP; the emulator's front otherwise parses a whole
/// graphics submission (thousands of draws, at back-thread speed) before an ASC submission gets a
/// turn, unless the graphics stream blocks on a wait. Second Son submits its particle compute and
/// reads the results in the same frame (REFERENCE 4.2 0x29c8d0), so that delay lands on Game:Main.
/// Only the interleaving changes, and the graphics task already yields mid-stream at unsatisfied
/// WAIT_REG_MEM / MEM_SEMAPHORE / REWIND; the round-robin resumes it with the task guards as there.
bool AscYield() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_YIELD");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_ASC_START_STATS=1 (diagnostic): time from SubmitAsc to the first run of that buffer's
/// task, and the number of ASC_YIELD yields; logged every 2,000 ASC submissions.
bool AscStartStats() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_START_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}

u64 AscNowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

struct AscStartCounters {
    u64 starts = 0, total_ns = 0, max_ns = 0, over_1ms = 0, yields = 0;
    std::array<u64, 6> hist{}; ///< <50 us, <200 us, <500 us, <1 ms, <3 ms, >=3 ms
    /// Split by whether the queue was busy (an earlier submission unfinished) at submit.
    std::array<u64, 2> split_n{}, split_ns{}, split_over_1ms{};
    /// Per mapped queue: starts, total ns, submits that found the queue busy.
    std::array<u64, 64> q_n{}, q_ns{}, q_busy{};
};
AscStartCounters g_asc_start; // front thread only

void NoteAscStart(u64 stamp, u32 queue) {
    auto& c = g_asc_start;
    const bool busy = (stamp & 1) != 0;
    const u64 submit_ns = stamp & ~u64{1};
    const u64 now = AscNowNs();
    const u64 ns = now > submit_ns ? now - submit_ns : 0;
    ++c.split_n[busy];
    c.split_ns[busy] += ns;
    c.split_over_1ms[busy] += ns >= 1'000'000;
    if (queue < c.q_n.size()) {
        ++c.q_n[queue];
        c.q_ns[queue] += ns;
        c.q_busy[queue] += busy;
    }
    ++c.starts;
    c.total_ns += ns;
    c.max_ns = std::max(c.max_ns, ns);
    const u64 us = ns / 1000;
    ++c.hist[us < 50 ? 0 : us < 200 ? 1 : us < 500 ? 2 : us < 1000 ? 3 : us < 3000 ? 4 : 5];
    if (c.starts % 2000 == 0) {
        LOG_INFO(Render, "AscStart: starts={} mean_us={:.1f} max_us={:.1f} gfx_yields={} "
                         "hist[<50,<200,<500,<1000,<3000,>=3000 us]={}/{}/{}/{}/{}/{}",
                 c.starts, double(c.total_ns) / double(c.starts) / 1000.0,
                 double(c.max_ns) / 1000.0, c.yields, c.hist[0], c.hist[1], c.hist[2],
                 c.hist[3], c.hist[4], c.hist[5]);
        std::string per_queue;
        for (u32 q = 0; q < c.q_n.size(); ++q) {
            if (c.q_n[q] != 0) {
                per_queue += fmt::format(" q{}={}/{:.0f}us/busy{}", q, c.q_n[q],
                                         double(c.q_ns[q]) / double(c.q_n[q]) / 1000.0,
                                         c.q_busy[q]);
            }
        }
        LOG_INFO(Render,
                 "AscStartSplit: idle_at_submit n={} mean_us={:.1f} over_1ms={} | busy_at_submit "
                 "n={} mean_us={:.1f} over_1ms={} | per queue (starts/mean/busy):{}",
                 c.split_n[0], c.split_n[0] ? double(c.split_ns[0]) / double(c.split_n[0]) / 1e3 : 0.0,
                 c.split_over_1ms[0], c.split_n[1],
                 c.split_n[1] ? double(c.split_ns[1]) / double(c.split_n[1]) / 1e3 : 0.0,
                 c.split_over_1ms[1], per_queue);
        c = AscStartCounters{};
    }
}

/// SHADPS4_ASC_RING_LATENCY=1 (diagnostic, CP split run-ahead only): where the time between a
/// guest ASC submission (sceGnmDingDong -> SubmitAsc) and the back reaching that submission's
/// RELEASE_MEM label goes. Second Son's Game:Main polls the particle ring's label right after
/// submitting the particle compute (0x1695e0), so this window is on the frame's critical path.
/// The back keeps a class clock (graphics stream / each ASC queue / idle) that changes only at
/// stream switches (Diag records the front emits when its round-robin changes queue) and around
/// WaitForWork; SubmitAsc snapshots it on the guest thread, the back snapshots it at the label.
/// Logged every ~10 s per ASC queue: per submission, and per group of submissions whose submit
/// times lie within AscRingGroupGapUs of each other (one particle update, inferred).
bool AscRingLatency() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_RING_LATENCY");
        return value && value[0] == '1';
    }();
    return enabled;
}

constexpr u32 AscRingQueues = 64;
constexpr u64 AscRingGroupGapUs = 3000;
enum AscRingDiagKind : u32 { AscRingSwitch = 1, AscRingBegin = 2, AscRingEnd = 3 };

/// The class clock at one instant (TSC ticks spent per class since start).
struct AscRingSnap {
    u64 tsc = 0, gfx = 0, asc = 0, own = 0, idle = 0;
};

/// Written by the back only. A seqlock lets SubmitAsc (guest threads) read a consistent snapshot.
struct AscRingClock {
    std::atomic<u32> seq{0};
    std::atomic<u64> since{0};
    std::atomic<u32> queue{0}; ///< Stream the back is executing: 0 = graphics, else ASC queue.
    std::atomic<bool> idle{false};
    std::atomic<u64> gfx{0}, asc{0}, idle_ticks{0};
    std::array<std::atomic<u64>, AscRingQueues> asc_q{};

    void Set(u32 new_queue, bool new_idle) { // back only
        const u64 now = __rdtsc();
        seq.fetch_add(1);
        const u64 start = since.load();
        if (start != 0) {
            const u64 d = now - start;
            if (idle.load()) {
                idle_ticks.fetch_add(d);
            } else if (const u32 q = queue.load(); q == 0) {
                gfx.fetch_add(d);
            } else {
                asc.fetch_add(d);
                asc_q[q % AscRingQueues].fetch_add(d);
            }
        }
        since.store(now);
        queue.store(new_queue);
        idle.store(new_idle);
        seq.fetch_add(1);
    }

    AscRingSnap Snap(u32 q) const {
        for (;;) {
            const u32 s = seq.load();
            if (s & 1) {
                _mm_pause();
                continue;
            }
            AscRingSnap snap;
            snap.gfx = gfx.load();
            snap.asc = asc.load();
            snap.own = asc_q[q % AscRingQueues].load();
            snap.idle = idle_ticks.load();
            const u64 start = since.load();
            const u32 cur = queue.load();
            const bool is_idle = idle.load();
            snap.tsc = __rdtsc();
            if (seq.load() != s) {
                continue;
            }
            if (start != 0 && snap.tsc > start) {
                const u64 d = snap.tsc - start;
                if (is_idle) {
                    snap.idle += d;
                } else if (cur == 0) {
                    snap.gfx += d;
                } else {
                    snap.asc += d;
                    if (cur == q) {
                        snap.own += d;
                    }
                }
            }
            return snap;
        }
    }
};
AscRingClock g_asc_ring_clock;

/// Sums of a class-clock difference (TSC ticks).
struct AscRingSpan {
    u64 wall = 0, gfx = 0, own = 0, other = 0, idle = 0;
    void Add(const AscRingSnap& from, const AscRingSnap& to) {
        wall += to.tsc - from.tsc;
        gfx += to.gfx - from.gfx;
        own += to.own - from.own;
        other += (to.asc - from.asc) - (to.own - from.own);
        idle += to.idle - from.idle;
    }
};

struct AscRingQueueState { // back only
    // The submission being executed (a queue's submissions are parsed one at a time, in order).
    bool open = false;
    AscRingSnap submit{};
    u64 front_tsc = 0, back_tsc = 0;
    u32 labels = 0;
    // The group (submissions close in submit time).
    bool group_open = false;
    AscRingSnap group_first{}, group_last_submit{}, group_label{}, group_label_submit{};
    u32 group_subs = 0, group_labels = 0;
    // The ~10 s window.
    u64 subs = 0, subs_nolabel = 0, label_count = 0, deferred_labels = 0, dispatches = 0;
    u64 to_front = 0, to_back = 0, to_label = 0;
    AscRingSpan sub_span{};
    u64 groups = 0, group_sub_sum = 0, group_submit_span = 0;
    AscRingSpan window{}, tail{};
    std::array<u64, 6> tail_hist{}; ///< <250 us, <500, <1000, <2000, <4000, >= 4000
    VAddr last_label_addr = 0;
};

/// What the front is doing (written by the front, sampled by the back when it runs out of work).
enum AscRingFrontState : u32 { FrontUnknown, FrontGfxTask, FrontAscTask, FrontGuestIdle };
std::atomic<u32> g_asc_ring_front_state{FrontUnknown};

struct AscRingState {
    std::array<AscRingQueueState, AscRingQueues> q{};
    u64 window_tsc = 0;
    u64 window_ns = 0;
    /// Back idle (WaitForWork) by the front's state when the back ran out of work.
    std::array<u64, 4> idle_n{}, idle_ticks{};
    std::array<u64, 5> idle_hist{}; ///< <5 us, <20, <100, <500, >= 500 (approx, 3.5 GHz)
    /// Back time in guest commands (SendCommand: read-fault downloads and the like); it is
    /// part of whatever stream class is current.
    u64 cmd_n = 0, cmd_ticks = 0;
    u64 prev_gfx = 0, prev_asc = 0, prev_idle = 0; ///< Class clock at the window start.
    /// Frame-fence discriminator: TSC of the newest gfx EOP label write (back), and at each
    /// SubmitGfx (guest thread) the time since it. Second Son waits for the previous frame's EOP
    /// label before it submits the next frame (0x1f1e30) with a 50 us poll: if Game:Main is
    /// waiting there, each submit follows a label write within about one poll.
    std::atomic<u64> last_eop_tsc{0};
    std::atomic<u64> sg_n{0}, sg_ticks{0};
    std::array<std::atomic<u64>, 5> sg_hist{}; ///< <100 us, <300, <1000, <3000, >= 3000 (approx)
    u32 front_queue = 0; ///< Front only: the stream of the last Switch marker.
    std::mutex submit_mutex; ///< Guards `submits` (SubmitAsc on guest threads, the front pops).
    std::array<std::deque<AscRingSnap>, AscRingQueues> submits{};
};
AscRingState g_asc_ring;

void EmitAscRingDiag(CpSplit::Channel& channel, u32 kind, u32 queue,
                     const AscRingSnap* snap = nullptr, u64 front_tsc = 0) {
    auto* diag = channel.Reserve<CpSplit::DiagPayload>(CpSplit::RecordType::Diag);
    diag->kind = kind;
    diag->queue = queue;
    if (snap) {
        diag->values[0] = snap->tsc;
        diag->values[1] = snap->gfx;
        diag->values[2] = snap->asc;
        diag->values[3] = snap->own;
        diag->values[4] = snap->idle;
    } else {
        diag->values[0] = diag->values[1] = diag->values[2] = diag->values[3] = 0;
        diag->values[4] = 0;
    }
    diag->values[5] = front_tsc;
    channel.Commit();
}

void AscRingCloseGroup(AscRingQueueState& s) {
    if (s.group_open && s.group_labels != 0) {
        ++s.groups;
        s.group_sub_sum += s.group_subs;
        s.group_submit_span += s.group_last_submit.tsc - s.group_first.tsc;
        s.window.Add(s.group_first, s.group_label);
        const u64 before = s.tail.wall;
        s.tail.Add(s.group_label_submit, s.group_label);
        // Bucketed in ticks here, converted by the logger's ratio only for the means; the
        // histogram uses a fixed 3 ticks/ns guess corrected below when logged (labelled approx).
        const u64 ticks = s.tail.wall - before;
        const u64 us = ticks / 3500; // ~3.5 GHz TSC (i5-13600K); the means use the measured ratio
        ++s.tail_hist[us < 250 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : us < 4000 ? 4 : 5];
    }
    s.group_open = false;
    s.group_subs = s.group_labels = 0;
}

/// Returns true when it logged a window (every ~10 s).
bool AscRingLog() {
    auto& g = g_asc_ring;
    const u64 now_tsc = __rdtsc();
    const u64 now_ns = AscNowNs();
    if (g.window_tsc == 0) {
        g.window_tsc = now_tsc;
        g.window_ns = now_ns;
        return false;
    }
    if (now_ns - g.window_ns < 10'000'000'000ull) {
        return false;
    }
    const double ticks_per_us =
        double(now_tsc - g.window_tsc) / (double(now_ns - g.window_ns) / 1000.0);
    const auto us = [&](u64 ticks, u64 n) {
        return n ? double(ticks) / ticks_per_us / double(n) : 0.0;
    };
    for (u32 q = 1; q < AscRingQueues; ++q) {
        auto& s = g.q[q];
        if (s.subs == 0 && s.groups == 0) {
            continue;
        }
        const u64 labelled = s.subs - s.subs_nolabel;
        LOG_INFO(Render,
                 "AscRing q{}: window_s={:.1f} tsc_mhz={:.0f} subs={} no_label={} labels={} "
                 "deferred_labels={} dispatches={} label_addr={:#x} | per labelled sub (us): "
                 "submit->front={:.0f} submit->back={:.0f} submit->label={:.0f} [gfx={:.0f} "
                 "own={:.0f} other_asc={:.0f} idle={:.0f}]",
                 q, double(now_ns - g.window_ns) / 1e9, ticks_per_us, s.subs, s.subs_nolabel,
                 s.label_count, s.deferred_labels, s.dispatches, s.last_label_addr,
                 us(s.to_front, labelled), us(s.to_back, labelled), us(s.to_label, labelled),
                 us(s.sub_span.gfx, labelled), us(s.sub_span.own, labelled),
                 us(s.sub_span.other, labelled), us(s.sub_span.idle, labelled));
        LOG_INFO(Render,
                 "AscRingGroup q{}: groups={} subs_per_group={:.1f} dispatches_per_group={:.0f} | "
                 "mean us: submit_span={:.0f} | first_submit->last_label={:.0f} [gfx={:.0f} "
                 "own={:.0f} other_asc={:.0f} idle={:.0f}] | last_submit->last_label={:.0f} "
                 "[gfx={:.0f} own={:.0f} other_asc={:.0f} idle={:.0f}] "
                 "tail_hist_approx[<250,<500,<1000,<2000,<4000,>=4000 us]={}/{}/{}/{}/{}/{}",
                 q, s.groups, s.groups ? double(s.group_sub_sum) / double(s.groups) : 0.0,
                 s.groups ? double(s.dispatches) / double(s.groups) : 0.0,
                 us(s.group_submit_span, s.groups), us(s.window.wall, s.groups),
                 us(s.window.gfx, s.groups), us(s.window.own, s.groups),
                 us(s.window.other, s.groups), us(s.window.idle, s.groups),
                 us(s.tail.wall, s.groups), us(s.tail.gfx, s.groups), us(s.tail.own, s.groups),
                 us(s.tail.other, s.groups), us(s.tail.idle, s.groups), s.tail_hist[0],
                 s.tail_hist[1], s.tail_hist[2], s.tail_hist[3], s.tail_hist[4], s.tail_hist[5]);
        // Reset the window sums; keep the open submission and group.
        s.subs = s.subs_nolabel = s.label_count = s.deferred_labels = s.dispatches = 0;
        s.to_front = s.to_back = s.to_label = 0;
        s.sub_span = {};
        s.groups = s.group_sub_sum = s.group_submit_span = 0;
        s.window = {};
        s.tail = {};
        s.tail_hist = {};
    }
    LOG_INFO(Render,
             "AscRingBackIdle: window_s={:.1f} back idle ms (waits) by front state at idle start: "
             "front_in_gfx_task={:.1f} ({}) front_in_asc_task={:.1f} ({}) front_waits_for_guest="
             "{:.1f} ({}) unknown={:.1f} ({}) | idle_hist_approx[<5,<20,<100,<500,>=500 us]="
             "{}/{}/{}/{}/{}",
             double(now_ns - g.window_ns) / 1e9, us(g.idle_ticks[FrontGfxTask], 1) / 1e3,
             g.idle_n[FrontGfxTask], us(g.idle_ticks[FrontAscTask], 1) / 1e3,
             g.idle_n[FrontAscTask], us(g.idle_ticks[FrontGuestIdle], 1) / 1e3,
             g.idle_n[FrontGuestIdle], us(g.idle_ticks[FrontUnknown], 1) / 1e3,
             g.idle_n[FrontUnknown], g.idle_hist[0], g.idle_hist[1], g.idle_hist[2],
             g.idle_hist[3], g.idle_hist[4]);
    {
        // The class clock over the window (whole back wall time: every tick is in one class).
        const AscRingSnap snap = g_asc_ring_clock.Snap(0);
        LOG_INFO(Render,
                 "AscRingBackSplit: window_s={:.1f} back ms: gfx_stream={:.1f} asc_streams={:.1f} "
                 "idle={:.1f} | of which guest commands (SendCommand) ms={:.1f} n={}",
                 double(now_ns - g.window_ns) / 1e9, us(snap.gfx - g.prev_gfx, 1) / 1e3,
                 us(snap.asc - g.prev_asc, 1) / 1e3, us(snap.idle - g.prev_idle, 1) / 1e3,
                 us(g.cmd_ticks, 1) / 1e3, g.cmd_n);
        g.prev_gfx = snap.gfx;
        g.prev_asc = snap.asc;
        g.prev_idle = snap.idle;
        g.cmd_n = g.cmd_ticks = 0;
    }
    {
        const u64 n = g.sg_n.exchange(0, std::memory_order_relaxed);
        const u64 ticks = g.sg_ticks.exchange(0, std::memory_order_relaxed);
        std::array<u64, 5> h{};
        for (u32 i = 0; i < h.size(); ++i) {
            h[i] = g.sg_hist[i].exchange(0, std::memory_order_relaxed);
        }
        LOG_INFO(Render,
                 "AscRingFrameFence: gfx_submits={} mean_us_since_last_eop_label={:.0f} "
                 "hist_approx[<100,<300,<1000,<3000,>=3000 us]={}/{}/{}/{}/{}",
                 n, us(ticks, n), h[0], h[1], h[2], h[3], h[4]);
    }
    g.idle_n = {};
    g.idle_ticks = {};
    g.idle_hist = {};
    g.window_tsc = now_tsc;
    g.window_ns = now_ns;
    return true;
}

/// Back: around WaitForWork.
u32 AscRingIdleEnter() {
    g_asc_ring_clock.Set(g_asc_ring_clock.queue.load(), true);
    return g_asc_ring_front_state.load(std::memory_order_relaxed);
}
void AscRingIdleExit(u32 front_state) {
    const u64 start = g_asc_ring_clock.since.load();
    g_asc_ring_clock.Set(g_asc_ring_clock.queue.load(), false);
    const u64 ticks = g_asc_ring_clock.since.load() - start;
    auto& g = g_asc_ring;
    ++g.idle_n[front_state & 3];
    g.idle_ticks[front_state & 3] += ticks;
    const u64 us = ticks / 3500;
    ++g.idle_hist[us < 5 ? 0 : us < 20 ? 1 : us < 100 ? 2 : us < 500 ? 3 : 4];
}

/// Back: a Diag record of the ring-latency instrument, in stream order. True: a window was logged.
bool AscRingBack(const CpSplit::DiagPayload& diag) {
    const u32 q = diag.queue % AscRingQueues;
    auto& s = g_asc_ring.q[q];
    switch (diag.kind) {
    case AscRingSwitch:
        g_asc_ring_clock.Set(diag.queue, false);
        break;
    case AscRingBegin: {
        const AscRingSnap submit{diag.values[0], diag.values[1], diag.values[2], diag.values[3],
                                 diag.values[4]};
        const u64 gap_ticks = AscRingGroupGapUs * 3500; // ~3.5 GHz TSC, see AscRingCloseGroup
        if (s.group_open && submit.tsc - s.group_last_submit.tsc > gap_ticks) {
            AscRingCloseGroup(s);
        }
        if (!s.group_open) {
            s.group_open = true;
            s.group_first = submit;
        }
        s.group_last_submit = submit;
        ++s.group_subs;
        s.open = true;
        s.submit = submit;
        s.front_tsc = diag.values[5];
        s.back_tsc = __rdtsc();
        s.labels = 0;
        break;
    }
    case AscRingEnd:
        if (s.open) {
            ++s.subs;
            s.subs_nolabel += s.labels == 0;
            s.open = false;
        }
        return AscRingLog();
    default:
        break;
    }
    return false;
}

const bool g_asc_early = [] {
    const char* v = std::getenv("SHADPS4_ASC_EARLY");
    return v && v[0] == '1';
}();
// m2: SHADPS4_ASC_SERVICE bits - 1 publish at ASC yields / submission end, 2 service inside follows.
const int g_asc_service = [] {
    const char* v = std::getenv("SHADPS4_ASC_SERVICE");
    return v ? std::atoi(v) : 0;
}();
// q14: SHADPS4_EARLY_CMD_POLL=1 - guest commands run between early ASC records (Game:Main's fault round trips).
const bool g_early_cmd_poll = [] {
    const char* v = std::getenv("SHADPS4_EARLY_CMD_POLL");
    return v && v[0] == '1';
}();
std::atomic<u64> g_early_cmd_polls{0};
struct AscEarlyStats {
    std::atomic<u64> dispatches{}, packets{}, in_order{}, blocked{}, skipped{};
    std::atomic<u64> forced_checks{}, upload_bytes{}, uploading_records{}, cross_waits{};
    // l4f: early record cycles (dispatch / packet), service gaps with pending work, unpublished stops
    std::atomic<u64> c_disp{}, c_pkt{}, gap_ticks{}, gap_max{}, unpublished{};
} g_asc_early_stats;
const bool g_xq = [] {
    const char* v = std::getenv("SHADPS4_XQ_CENSUS");
    return v && v[0] == '1';
}();
constexpr u32 XqQueues = 9; // 0 gfx, 1..8 = vqid + 1
struct XqCensus {
    // waits[waiter][writer]: writer 0 gfx, 1 same ASC queue, 2 other ASC queue, 3 none (CPU /
    // done), 4 FENCE_FORWARD skip (same queue)
    std::array<std::array<std::atomic<u64>, 5>, XqQueues> waits{};
    std::array<std::atomic<u64>, XqQueues> disp{}, disp_img{}, upload_bytes{}, segs{}, segs_img{},
        segs_xwait{};
    std::array<std::atomic<u8>, XqQueues> seg_img{}, seg_xwait{};
    std::array<std::atomic<u64>, XqQueues> label_addr{};
} g_xq_c;
/// Front: a WAIT_REG_MEM of queue `waiter` (0 gfx, vqid + 1) on memory label `lo`.
void XqNoteWait(CpSplit::Channel& ch, u32 waiter, VAddr lo, bool forwarded) {
    if (!g_xq || waiter >= XqQueues) {
        return;
    }
    u32 w = 4;
    if (!forwarded) {
        const int q = ch.NewestWriterQueue(lo);
        w = q < 0 ? 3 : q == 0 ? 0 : u32(q) == waiter ? 1 : 2;
    }
    g_xq_c.waits[waiter][w].fetch_add(1, std::memory_order_relaxed);
    if (waiter != 0 && (w == 0 || w == 3)) {
        g_xq_c.seg_xwait[waiter].store(1, std::memory_order_relaxed);
    }
}

/// Back: an ASC RELEASE_MEM (the queue's completion label) is being executed.
void AscRingLabel(VAddr address, bool deferred_to_gpu) {
    const u32 q = g_asc_ring_clock.queue.load() % AscRingQueues;
    if (g_xq) [[unlikely]] {
        const u32 qi = std::min<u32>(q, XqQueues - 1);
        g_xq_c.segs[qi].fetch_add(1, std::memory_order_relaxed);
        g_xq_c.segs_img[qi].fetch_add(g_xq_c.seg_img[qi].exchange(0), std::memory_order_relaxed);
    }
    auto& s = g_asc_ring.q[q];
    ++s.label_count;
    s.deferred_labels += deferred_to_gpu;
    s.last_label_addr = address;
    if (!s.open) {
        return;
    }
    const AscRingSnap label = g_asc_ring_clock.Snap(q);
    if (s.labels++ == 0) {
        s.to_front += s.front_tsc - s.submit.tsc;
        s.to_back += s.back_tsc - s.submit.tsc;
        s.to_label += label.tsc - s.submit.tsc;
        s.sub_span.Add(s.submit, label);
    }
    if (s.group_open) {
        s.group_label = label;
        s.group_label_submit = s.submit;
        ++s.group_labels;
    }
}

void PrefetchProgram(const ShaderProgram& program) {
    const auto* code = program.Address<const char*>();
    if (!code) {
        return;
    }
    _mm_prefetch(code, _MM_HINT_T0);
    const void* info = g_bininfo_hints[BinaryInfoHintSlot(code)].load(std::memory_order_relaxed);
    if (info) {
        _mm_prefetch(static_cast<const char*>(info), _MM_HINT_T0);
        _mm_prefetch(static_cast<const char*>(info) + 63, _MM_HINT_T0);
    }
}

// Called after SET_SH_REG wrote words [offset, offset + count) of the SH register space.
void PrefetchProgramsOnWrite(const Regs& regs, u32 offset, u32 count) {
    constexpr u32 ps_lo = offsetof(Regs, ps_program) / sizeof(u32) - Regs::ShRegWordOffset;
    constexpr u32 vs_lo = offsetof(Regs, vs_program) / sizeof(u32) - Regs::ShRegWordOffset;
    const u32 end = offset + count;
    // The 40-bit address spans the LO and HI words.
    if (offset <= ps_lo + 1 && ps_lo < end) {
        PrefetchProgram(regs.ps_program);
    }
    if (offset <= vs_lo + 1 && vs_lo < end) {
        PrefetchProgram(regs.vs_program);
    }
}
} // namespace

/// SHADPS4_FENCE_AT_GPU=1: end-of-pipe/end-of-shader fences (EVENT_WRITE_EOP, EVENT_WRITE_EOS,
/// RELEASE_MEM) are written, and their interrupts raised, when the host GPU has actually
/// finished the preceding work, as on the real GPU - not when the CP parses the packet. Needed
/// once the GPU reads guest memory at execution time (SHADPS4_HOST_MIRROR): with CP-time fences
/// the game may reuse a buffer the host GPU has not read yet. GDS-sourced fences stay CP-time.
/// Mode 1 submits the batch at every fence; mode 2 defers without submitting (see
/// Rasterizer::DeferToGpu) and flushes before the CP waits on memory or goes idle; mode 3 is
/// mode 1 for EVENT_WRITE_EOP only (EOS and RELEASE_MEM stay CP-time).
static int FenceAtGpu() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_FENCE_AT_GPU");
        return value && value[0] >= '1' && value[0] <= '4' ? value[0] - '0' : 0;
    }();
    return mode;
}

// q58 (B0 port of 3ef44d28, zero-copy gate; notes/zero-copy-heap-2026-10-07.md sec. 7): SHADPS4_FENCE_AT_GPU=4 writes
// only the graphics EOP / EOS label at SHADPS4_FENCE_AT_GPU_ADDR=<hex> (the game's end-of-frame label 0x248000000) at
// host-GPU completion (mode 1 for that label); every other label stays CP-time and keeps fence forwarding.
static u64 GpuTimedLabel() {
    static const u64 address = [] {
        const char* value = std::getenv("SHADPS4_FENCE_AT_GPU_ADDR");
        return value ? std::strtoull(value, nullptr, 16) : 0ull;
    }();
    return address;
}
static bool GpuTimedLabelHit(u64 address) {
    return FenceAtGpu() == 4 && address != 0 && address == GpuTimedLabel();
}
/// Whether the label at `address` is written at host-GPU completion (modes 1-3: every label of their kinds; 4: one).
static bool FenceDeferredAt(u64 address) {
    const int mode = FenceAtGpu();
    return mode != 0 && (mode != 4 || GpuTimedLabelHit(address));
}

static void WriteFenceValue(void* address, u64 data, u32 num_bytes) {
    auto* memory = Core::Memory::Instance();
    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
        memcpy(address, &data, num_bytes);
    }
}

namespace {
/// SHADPS4_CP_WAIT_STATS=1: where the command processor's wall time goes. WaitRegMem waits are
/// split by whether the polled address is one the CP itself writes (EOS/EOP/RELEASE_MEM/
/// WRITE_DATA: a GPU-side dependency) or not (written by guest CPU code). Wait intervals
/// overlap with other queues' work done between yields; they are latencies, not CPU cost.
struct CpWaitStats {
    bool enabled = [] {
        const char* v = std::getenv("SHADPS4_CP_WAIT_STATS");
        return v && v[0] == '1';
    }();
    std::unordered_set<VAddr> cp_written;
    u64 idle_ns = 0;
    u64 gfx_wrm_cp_ns = 0, gfx_wrm_cpu_ns = 0, gfx_wrm_cp_n = 0, gfx_wrm_cpu_n = 0;
    u64 asc_wrm_cp_ns = 0, asc_wrm_cpu_ns = 0, asc_wrm_cp_n = 0, asc_wrm_cpu_n = 0;
    u64 gfx_wrm_instant = 0;
    u64 labels = 0;
    u64 window_start = 0;

    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    void NoteWrite(const void* addr) {
        if (enabled) {
            cp_written.insert(reinterpret_cast<VAddr>(addr));
            ++labels;
        }
    }
    void CountWait(bool gfx, const void* addr, u64 ns) {
        const bool by_cp = cp_written.contains(reinterpret_cast<VAddr>(addr));
        if (gfx) {
            (by_cp ? gfx_wrm_cp_ns : gfx_wrm_cpu_ns) += ns;
            ++(by_cp ? gfx_wrm_cp_n : gfx_wrm_cpu_n);
        } else {
            (by_cp ? asc_wrm_cp_ns : asc_wrm_cpu_ns) += ns;
            ++(by_cp ? asc_wrm_cp_n : asc_wrm_cpu_n);
        }
        MaybeLog();
    }
    void MaybeLog() {
        const u64 now = Now();
        if (window_start == 0) {
            window_start = now;
            return;
        }
        const u64 span = now - window_start;
        if (span < 10'000'000'000ULL) {
            return;
        }
        const auto pct = [span](u64 ns) { return 100.0 * double(ns) / double(span); };
        LOG_INFO(Render, "CpWait: window_s={:.1f} idle_pct={:.1f} gfx_wrm_cp_pct={:.1f} "
                 "gfx_wrm_cpu_pct={:.1f} gfx_wrm_cp_n={} gfx_wrm_cpu_n={} gfx_wrm_instant={} "
                 "asc_wrm_cp_pct={:.1f} asc_wrm_cpu_pct={:.1f} asc_wrm_cp_n={} asc_wrm_cpu_n={} "
                 "labels={} label_addrs={}",
                 double(span) / 1e9, pct(idle_ns), pct(gfx_wrm_cp_ns), pct(gfx_wrm_cpu_ns),
                 gfx_wrm_cp_n, gfx_wrm_cpu_n, gfx_wrm_instant, pct(asc_wrm_cp_ns),
                 pct(asc_wrm_cpu_ns), asc_wrm_cp_n, asc_wrm_cpu_n, labels, cp_written.size());
        idle_ns = gfx_wrm_cp_ns = gfx_wrm_cpu_ns = gfx_wrm_cp_n = gfx_wrm_cpu_n = 0;
        asc_wrm_cp_ns = asc_wrm_cpu_ns = asc_wrm_cp_n = asc_wrm_cpu_n = 0;
        gfx_wrm_instant = labels = 0;
        window_start = now;
    }
};
CpWaitStats g_cp_wait;
} // namespace

namespace {
/// SHADPS4_FRONT_COST=1 (counter run): front cycles per PM4 opcode, waits excluded.
const bool g_front_cost = [] {
    const char* e = std::getenv("SHADPS4_FRONT_COST");
    return e && e[0] == '1';
}();
struct FrontCost {
    u64 last_tsc{}, last_wait{}, win_tsc{};
    u32 last_op{};
    // 0x00-0xff graphics opcodes, 0x100 | op compute opcodes, FrontCostLeanHead the L2 head pass
    // (LeanFrontBeginL2), FrontCostBetween from a top-level DCB's end to the next tick.
    std::array<u64, 512> cyc{}, n{};
};
constexpr u32 FrontCostLeanHead = 0x1FE;
constexpr u32 FrontCostBetween = 0x1FF;
thread_local FrontCost t_front_cost;
void FrontCostTick(u32 opcode) {
    auto& f = t_front_cost;
    const u64 now = __rdtsc();
    const u64 wait = Common::FaultMonitor::front_wait_ticks.load(std::memory_order_relaxed);
    if (f.last_tsc != 0) {
        const u64 dt = now - f.last_tsc;
        const u64 dw = wait - f.last_wait;
        f.cyc[f.last_op & 511] += dt > dw ? dt - dw : 0;
        ++f.n[f.last_op & 511];
    } else {
        f.win_tsc = now;
    }
    f.last_tsc = now;
    f.last_wait = wait;
    f.last_op = opcode;
    const u64 win = now - f.win_tsc;
    if (win > 35'000'000'000ull) { // ~10 s
        const double s = double(win) / 3.5e9;
        std::array<u32, 512> ord{};
        for (u32 i = 0; i < 512; ++i) {
            ord[i] = i;
        }
        std::sort(ord.begin(), ord.end(), [&](u32 a, u32 b) { return f.cyc[a] > f.cyc[b]; });
        u64 total = 0;
        for (const u64 c : f.cyc) {
            total += c;
        }
        std::string out;
        for (u32 k = 0; k < 18 && f.cyc[ord[k]] != 0; ++k) {
            const u32 op = ord[k];
            const std::string name = op == FrontCostLeanHead ? std::string{"L2head"}
                                     : op == FrontCostBetween ? std::string{"between"}
                                     : op >= 0x100 ? fmt::format("c{:#04x}", op & 255)
                                                   : fmt::format("op{:#04x}", op);
            out += fmt::format(" {}={:.1f}ms/s({:.0f}k/s,{:.0f}cyc)", name,
                               double(f.cyc[op]) / 3.5e6 / s, double(f.n[op]) / 1e3 / s,
                               double(f.cyc[op]) / double(std::max<u64>(f.n[op], 1)));
        }
        LOG_INFO(Render, "FrontCost: window {:.1f}s (TSC ~3.5 GHz) front busy {:.1f} ms/s, by opcode:{}",
                 s, double(total) / 3.5e6 / s, out);
        f.cyc = {};
        f.n = {};
        f.win_tsc = now;
    }
}

/// SHADPS4_REG_SCAN=1 (counter run): the register-only pass of a chained worker, timed and verified.
const bool g_reg_scan = [] {
    const char* e = std::getenv("SHADPS4_REG_SCAN");
    return e && e[0] == '1';
}();
struct RegScanStats {
    u64 cyc{}, dcbs{}, pkts{}, reg_pkts{}, ibs{}, cond_exec{}, cmp{}, bad{}, bad_words{};
    u64 post_cyc{}, posts{}; ///< the front's JournalRegs (posting register records)
    // Constant engine: packets by kind, dwords written to / dumped from CE RAM.
    u64 ce_write{}, ce_write_dw{}, ce_dump{}, ce_dump_dw{}, ce_load{}, ce_other{}, ccbs{};
    // ASC SET_SH_REG to the shared (non-compute) SH registers: all / while a gfx DCB is open.
    u64 asc_sh{}, asc_sh_dw{}, asc_sh_in_gfx{}, asc_sh_in_gfx_dw{};
    u32 asc_sh_min{~0u}, asc_sh_max{};
    // Lean-parser census: draws in total / before the first packet outside S / outside S+DMA_DATA,
    // DCBs fully inside, and the first cutting opcode (S, S+DMA).
    u64 draws{}, draws_s{}, draws_sd{}, dcb_full_s{}, dcb_full_sd{};
    std::array<u64, 256> cut_s{}, cut_sd{};
    // per DCB (reset at each top-level scan)
    bool cur_cut_s{}, cur_cut_sd{};
    u64 win_tsc{};
};
thread_local RegScanStats t_reg_scan;
thread_local bool t_reg_scan_gfx_open = false; ///< a top-level gfx DCB is in progress (front)
thread_local std::unique_ptr<AmdGpu::Regs> t_reg_scan_regs;

/// The register-only pass (see SHADPS4_REG_SCAN). Only what writes the register file.
void RegScanRun(std::span<const u32> dcb, AmdGpu::Regs& r, RegScanStats& st, u32 depth) {
    while (!dcb.empty()) {
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        if (header->type == 2) {
            dcb = dcb.subspan(1);
            continue;
        }
        if (header->type != 3) {
            return; // the front asserts on these
        }
        const u32 count = header->type3.NumWords();
        const auto* w = reinterpret_cast<const u32*>(header);
        ++st.pkts;
        {
            const PM4ItOpcode op = header->type3.opcode;
            const bool is_draw = op == PM4ItOpcode::DrawIndex2 || op == PM4ItOpcode::DrawIndexOffset2 ||
                                 op == PM4ItOpcode::DrawIndexAuto || op == PM4ItOpcode::DrawIndirect ||
                                 op == PM4ItOpcode::DrawIndexIndirect ||
                                 op == PM4ItOpcode::DrawIndirectMulti ||
                                 op == PM4ItOpcode::DrawIndexIndirectMulti ||
                                 op == PM4ItOpcode::DrawIndexIndirectCountMulti;
            const bool in_s =
                op == PM4ItOpcode::SetConfigReg || op == PM4ItOpcode::SetContextReg ||
                op == PM4ItOpcode::SetShReg || op == PM4ItOpcode::SetUconfigReg ||
                op == PM4ItOpcode::ClearState || op == PM4ItOpcode::Nop ||
                op == PM4ItOpcode::IndexType || op == PM4ItOpcode::IndexBase ||
                op == PM4ItOpcode::IndexBufferSize || op == PM4ItOpcode::NumInstances ||
                op == PM4ItOpcode::DrawIndex2 || op == PM4ItOpcode::DrawIndexOffset2 ||
                op == PM4ItOpcode::DrawIndexAuto ||
                // user ~13:55: the no-op DMA (destination 0x3022C, skipped by the back) is supported
                (op == PM4ItOpcode::DmaData &&
                 reinterpret_cast<const PM4DmaData*>(header)->dst_addr_lo == 0x3022C) ||
                op == PM4ItOpcode::AcquireMem; // the front ignores it (user ~14:00)
            // Second column: + SET_BASE (indirect_args_addr) and CONTEXT_CONTROL.
            const bool in_sd =
                in_s || op == PM4ItOpcode::SetBase || op == PM4ItOpcode::ContextControl;
            if (!in_s && !st.cur_cut_s) {
                st.cur_cut_s = true;
                ++st.cut_s[static_cast<u32>(op) & 255];
            }
            if (!in_sd && !st.cur_cut_sd) {
                st.cur_cut_sd = true;
                ++st.cut_sd[static_cast<u32>(op) & 255];
            }
            if (is_draw) {
                ++st.draws;
                st.draws_s += st.cur_cut_s ? 0 : 1;
                st.draws_sd += st.cur_cut_sd ? 0 : 1;
            }
        }
        switch (header->type3.opcode) {
        case PM4ItOpcode::ClearState:
            r.SetDefaults();
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::SetConfigReg:
            std::memcpy(&r.reg_array[AmdGpu::Regs::ConfigRegWordOffset + (w[1] & 0xffff)], w + 2,
                        size_t{count - 1} * 4);
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::SetContextReg:
            std::memcpy(&r.reg_array[AmdGpu::Regs::ContextRegWordOffset + (w[1] & 0xffff)], w + 2,
                        size_t{count - 1} * 4);
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::SetShReg: {
            const u32 off = w[1] & 0xffff;
            if (!(off >= 0x200 && off <= 0x200 + sizeof(ComputeProgram) / 4)) {
                std::memcpy(&r.reg_array[AmdGpu::Regs::ShRegWordOffset + off], w + 2,
                            size_t{count - 1} * 4);
            }
            ++st.reg_pkts;
            break;
        }
        case PM4ItOpcode::SetUconfigReg:
            std::memcpy(&r.reg_array[AmdGpu::Regs::UconfigRegWordOffset + (w[1] & 0xffff)], w + 2,
                        size_t{count - 1} * 4);
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::IndexType:
            r.index_buffer_type.raw = reinterpret_cast<const PM4CmdDrawIndexType*>(header)->raw;
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::DrawIndex2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            r.max_index_size = d->max_size;
            r.index_base_address.base_addr_lo = d->index_base_lo;
            r.index_base_address.base_addr_hi = d->index_base_hi;
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            r.max_index_size = d->max_size;
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            break;
        }
        case PM4ItOpcode::DrawIndexAuto: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            break;
        }
        case PM4ItOpcode::NumInstances:
            r.num_instances.num_instances =
                reinterpret_cast<const PM4CmdDrawNumInstances*>(header)->num_instances;
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::IndexBase: {
            const auto* b = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
            r.index_base_address.base_addr_lo = b->addr_lo;
            r.index_base_address.base_addr_hi = b->addr_hi;
            ++st.reg_pkts;
            break;
        }
        case PM4ItOpcode::IndexBufferSize:
            r.num_indices = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header)->num_indices;
            r.max_index_size = r.num_indices; // r4: INDEX_BUFFER_SIZE = VGT_DMA_MAX_SIZE (upstream #5278)
            ++st.reg_pkts;
            break;
        case PM4ItOpcode::EventWrite: {
            const auto* ev = reinterpret_cast<const PM4CmdEventWrite*>(header);
            if (ev->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                r.cp_strmout_cntl.offset_update_done = 1;
            }
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* ib = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            ++st.ibs;
            if (depth < 8) {
                RegScanRun({ib->Address<const u32>(), ib->ib_size}, r, st, depth + 1);
            }
            break;
        }
        case PM4ItOpcode::CondExec:
            ++st.cond_exec; // reads GPU-written memory: a chained scan cannot evaluate it ahead
            break;
        default:
            break;
        }
        dcb = dcb.subspan(std::min<size_t>(dcb.size(), size_t{count} + 1));
    }
}

void RegScanLog(RegScanStats& st, u64 now) {
    const double s = double(now - st.win_tsc) / 3.5e9;
    {
        const auto top = [](const std::array<u64, 256>& h) {
            std::array<u32, 256> ord{};
            for (u32 i = 0; i < 256; ++i) {
                ord[i] = i;
            }
            std::sort(ord.begin(), ord.end(), [&](u32 a, u32 b) { return h[a] > h[b]; });
            std::string out;
            for (u32 k = 0; k < 6 && h[ord[k]] != 0; ++k) {
                out += fmt::format(" op{:#04x}={}", ord[k], h[ord[k]]);
            }
            return out;
        };
        LOG_INFO(Render,
                 "LeanParse census (S + no-op DMA + ACQUIRE_MEM): draws {} | before the first packet outside S {} ({:.1f}%), DCBs "
                 "fully in S {} of {} | outside S+SET_BASE+CONTEXT_CONTROL {} ({:.1f}%), DCBs fully in it {} | "
                 "first cut, S:{} | S+DMA:{}",
                 st.draws, st.draws_s, st.draws ? 100.0 * double(st.draws_s) / double(st.draws) : 0.0,
                 st.dcb_full_s, st.dcbs, st.draws_sd,
                 st.draws ? 100.0 * double(st.draws_sd) / double(st.draws) : 0.0, st.dcb_full_sd,
                 top(st.cut_s), top(st.cut_sd));
    }
    LOG_INFO(Render,
             "RegScan: window {:.1f}s (TSC ~3.5 GHz) register-only pass {:.2f} ms/s ({:.0f} cycles "
             "per DCB, {} DCBs, {:.0f} packets/s of which register {:.0f}/s, IBs {}, CondExec {}) | "
             "end-state verify {} DCBs, {} bad ({} words) | front posting (JournalRegs) {:.2f} ms/s "
             "({:.0f}/s, {:.0f} cycles each) | CE: CCBs {:.0f}/s, WRITE_CONST_RAM {:.0f}/s ({:.0f} dw/s), "
             "DUMP_CONST_RAM {:.0f}/s ({:.0f} dw/s), LOAD_CONST_RAM {:.0f}/s, other CE {:.0f}/s | ASC "
             "SET_SH_REG (non-compute) {:.0f}/s ({:.0f} dw/s), while a gfx DCB is open {:.0f}/s ({:.0f} "
             "dw/s), SH offsets {:#x}..{:#x}",
             s, double(st.cyc) / 3.5e6 / s, st.dcbs ? double(st.cyc) / double(st.dcbs) : 0.0,
             st.dcbs, double(st.pkts) / s, double(st.reg_pkts) / s, st.ibs, st.cond_exec, st.cmp,
             st.bad, st.bad_words, double(st.post_cyc) / 3.5e6 / s, double(st.posts) / s,
             st.posts ? double(st.post_cyc) / double(st.posts) : 0.0, double(st.ccbs) / s,
             double(st.ce_write) / s, double(st.ce_write_dw) / s, double(st.ce_dump) / s,
             double(st.ce_dump_dw) / s, double(st.ce_load) / s, double(st.ce_other) / s,
             double(st.asc_sh) / s, double(st.asc_sh_dw) / s, double(st.asc_sh_in_gfx) / s,
             double(st.asc_sh_in_gfx_dw) / s, st.asc_sh_min == ~0u ? 0u : st.asc_sh_min,
             st.asc_sh_max);
    st = RegScanStats{};
    st.win_tsc = now;
}

/// SHADPS4_LEAN_PARSE (architecture A piece 2, lean worker parser; notes/architecture-a-2026-10-10.md
/// sec. 16): 1 = L1 shadow (the front still parses everything; workers parse each DCB's head span
/// into a scratch channel and the two record streams are compared as canonical events), 2 = L2.
/// (SHADPS4_LEAN is Common::Lean, the diagnostic skip; every arm sets it to 1.)
const int g_lean = [] {
    const char* v = std::getenv("SHADPS4_LEAN_PARSE");
    return v ? v[0] - '0' : 0;
}();
constexpr u32 LeanSlots = 128;
struct LeanJob {
    std::atomic<u32> state{0};     ///< 0 free, 1 queued (worker may take it)
    std::atomic<u32> done_mask{0}; ///< 1 front finished its events, 2 worker finished
    std::atomic<u32> index{0};
    std::span<const u32> dcb;
    u32 cut_dw{};
    u32 dcb_pkts{}, dcb_draws{}; ///< whole DCB (chain scan)
    std::vector<u32> win; ///< the chain's register windows at the DCB start (DcbSnapWindows order)
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb{};
    CbDbExtent db{};
    u64 epoch{};
    ComputeProgram cs{};
    VAddr iaa{};
    std::vector<u64> front_canon;
    std::vector<std::pair<u32, u32>> front_marks; ///< (packet dword offset, events before it)
    std::vector<u64> worker_canon;
    Lean::SpanResult res{};
    u64 dcb_hash{};           ///< XXH3 of the DCB words at queue time (L1 lag check)
    std::atomic<bool> lag{};  ///< the words changed before the worker's parse ended
    /// L3: the worker's stretches of this job in order (fixed size A2MaxStretches, sized once at
    /// LeanStart, never reallocated under the back's reads). The worker writes xl[k] and
    /// xl_side[k], then xl_ready = k + 1 (release), then publishes the channel.
    std::vector<std::unique_ptr<Vulkan::A2::Stretch>> xl;
    std::vector<A2Entry::Side> xl_side;
    std::atomic<u32> xl_ready{0};
    /// L3: the back found this job's channel empty; the worker publishes at its next draw.
    std::atomic<u32> back_waiting{0};
};
struct LeanCounters {
    std::atomic<u64> dcbs{}, eligible{}, skipped{}, cmp{}, bad{}, events{}, pkts{}, pkts_span{},
        draws{}, draws_span{}, no_mark{}, epoch_bumps{}, run_bumps{};
    std::atomic<u32> epoch_bump_op{}; ///< first offending opcode | 0x100 (0: none)
    /// L2 stage b: parity breaks, L1 lag-excluded jobs, BAD spans by first-difference kind, the
    /// front's draws and its NativeRec member-path draws (job_member = 1).
    std::atomic<u64> parity{}, parity_changed{}, lag{}, front_draws{}, front_member{};
    std::array<std::atomic<u64>, 32> bad_kind{};
    std::array<std::atomic<u64>, 32> stops{};
    std::array<std::atomic<u64>, 256> stop_ops{};
};
struct LeanState {
    std::array<LeanJob, LeanSlots> jobs;
    std::atomic<u32> next_claim{1};
    alignas(64) std::atomic<u32> queued{0}; ///< newest DCB index handed to the workers
    ClaimBells bells;                        ///< SHADPS4_CLAIM_WAKE: wakes for `queued`
    std::atomic<bool> stop{false};
    std::vector<std::jthread> workers;
    bool started{};
    // The chain (front-owned in L1).
    std::unique_ptr<Regs> regs;
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb{};
    CbDbExtent db{};
    u64 epoch{};
    ComputeProgram cs{};
    VAddr iaa{};
    bool seeded{};
    u32 index{};
    u64 chain_cyc{}, chain_cmp{}, chain_bad_win{}, chain_bad_full{}, chain_bad_extra{}, reseeds{};
    u64 chain_bad_db{}, chain_bad_cb{}, chain_bad_cs{}, chain_bad_epoch{}, chain_bad_iaa{};
    LeanCounters c;
    u64 win_tsc{};
    // SHADPS4_LEAN=2
    bool l2{};                                                   ///< mode 2 and the config allows it
    std::array<std::unique_ptr<CpSplit::Channel>, LeanSlots> ch; ///< one per job slot
    alignas(64) std::atomic<u32> back_cur{0};                    ///< DCB index the back follows
    std::atomic<u64> l2_taken{}, l2_skipped{}, l2_early{}, head_draws{}, back_records{},
        back_wait_ticks{};
    u64 split_spans{}; ///< extra jobs made by head splitting (front-owned)
    std::unique_ptr<Lean::Parser> toucher; ///< front: L2 Pretouch before the exclusive lookup
};
LeanState g_lean_s;
/// After `queued` advanced to idx: wake its claimer (SHADPS4_CLAIM_WAKE) or every waiter.
inline void LeanRing(LeanState& s, u32 idx) {
    if (g_claim_wake) {
        s.bells.Ring(idx);
    } else {
        s.queued.notify_all();
    }
}
/// L3 counters (workers add theirs per job; the back its own) and the per-second placement samples.
struct LeanXlStats {
    bool on{}; ///< L2 + SHADPS4_A2 != 0: workers translate
    std::atomic<u64> taken{}, stretches{}, cap{}, cut_pub{}, walked{}, mat_cuts{};
    std::atomic<u64> published{}, pub_draws{}, rej_occl{}, rej_gen{}, rej_mat{}, rej_rt{},
        would{}, back_draws{}, seq_skips{};
    std::atomic<u64> pre_folds{}, pre_records{}, pre_cmp{}, pre_bad{}, wait_pubs{};
    /// SHADPS4_A2=2 field compares (gate 0 bad): render key per draw, delta words per stretch.
    std::atomic<u64> rk_cmp{}, rk_bad{}, delta_cmp{}, delta_bad{}, expect_filed{};
    /// l3e direct checks: inherited-slot words (offset + size vs the back's locked lookup and
    /// clamp), the record's VS / PS user data vs the mirror (inherited-slot draws), clip planes.
    std::atomic<u64> inh_cmp{}, inh_bad{}, ud_cmp{}, ud_bad{}, clip_cmp{}, clip_bad{};
    /// l3g: formatted slots - texel syncs at publish and rejected stretches; A2=2 base compares.
    std::atomic<u64> texel_syncs{}, rej_texel{}, fmt_cmp{}, fmt_bad{};
    std::atomic<u64> verify_skips{}; ///< A2=2: stretches A2=1 would reject (not compared)
    std::atomic<u64> mat_resolves{}; ///< materials resolved at publish (A2ResolveMaterial calls)
    std::atomic<u64> verify_mid{};   ///< A2=2: buffer checks skipped, generation moved mid-stretch
    std::atomic<u64> front_p{}, front_e{}, lean_p{}, lean_e{};
    std::array<std::atomic<Vulkan::A2::StretchTranslator*>, 8> tr{};
};
LeanXlStats g_lean_xl;
/// SHADPS4_BACK_CYCLES=1 on the A2 / lean path (back thread only): cycles by record type on the
/// main ring (A2Execute, DcbRef excluded) and in the job channels, the stretch publishes, the
/// guest commands run inside LeanBackFollow, its wait for workers.
struct BackLeanCensus {
    std::array<u64, 32> main_cyc{}, main_n{}, job_cyc{}, job_n{};
    std::array<u64, 64> why_cyc{}, why_n{}; ///< old-path job-channel draws by job_pad
    u64 pub_cyc{}, pub_n{}, pc_cyc{}, pc_n{}, wait_cyc{}, follow_cyc{}, follows{};
    u64 pub_resolve{}, pub_begin{}, pub_record{}, pub_delta{};
    u64 pro_cyc{}, epi_cyc{}, loop_recs{}; ///< per follow: before / after the loop; records visited
    std::array<u64, 256> pk_cyc{}, pk_n{}; ///< q4: main-ring Packet records by PM4 opcode
    u64 win_tsc{};
    Vulkan::BackCycleStats base{};
};
BackLeanCensus g_blc;
const char* RecordTypeName(u32 t) {
    static constexpr const char* names[] = {"wrap",  "regwrite", "clearstate", "extent",
                                            "epoch", "draw",     "dispatch",   "packet",
                                            "call",  "retire",   "diag",       "segment",
                                            "dcbbegin", "dcbend", "dcbref",    "leanend"};
    return t < std::size(names) ? names[t] : "?";
}
void BackLeanMaybeLog() {
    auto& b = g_blc;
    const u64 now = __rdtsc();
    if (b.win_tsc == 0) {
        b.win_tsc = now;
        b.base = Vulkan::t_back_cycles;
        return;
    }
    if (now - b.win_tsc < 35'000'000'000ull) {
        return;
    }
    const double s = double(now - b.win_tsc) / 3.5e9;
    const auto ms = [&](u64 c) { return double(c) / 3.5e6 / s; };
    const auto ks = [&](u64 n) { return double(n) / 1e3 / s; };
    std::string main_s, job_s;
    u64 main_all = 0, job_all = 0;
    for (u32 t = 0; t < 32; ++t) {
        main_all += b.main_cyc[t];
        job_all += b.job_cyc[t];
        if (b.main_cyc[t] != 0) {
            main_s += fmt::format(" {}={:.1f}({:.0f}k/s,{:.0f}cyc)", RecordTypeName(t),
                                  ms(b.main_cyc[t]), ks(b.main_n[t]),
                                  double(b.main_cyc[t]) / double(std::max<u64>(b.main_n[t], 1)));
        }
        if (b.job_cyc[t] != 0) {
            job_s += fmt::format(" {}={:.1f}({:.0f}k/s,{:.0f}cyc)", RecordTypeName(t),
                                 ms(b.job_cyc[t]), ks(b.job_n[t]),
                                 double(b.job_cyc[t]) / double(std::max<u64>(b.job_n[t], 1)));
        }
    }
    std::string why_s;
    for (u32 i = 0; i < 64; ++i) {
        if (b.why_cyc[i] == 0) {
            continue;
        }
        const std::string name = i == 0    ? std::string{"taken(stretch rejected / verify)"}
                                 : i == 40 ? std::string{"not offered"}
                                 : i == 41 ? std::string{"stretch cap"}
                                 : i <= 32 ? std::string{Vulkan::A2::RefusalName(i - 1)}
                                           : fmt::format("?{}", i);
        why_s += fmt::format(" {}={:.1f}({:.0f}k/s,{:.0f}cyc)", name, ms(b.why_cyc[i]),
                             ks(b.why_n[i]),
                             double(b.why_cyc[i]) / double(std::max<u64>(b.why_n[i], 1)));
    }
    LOG_INFO(Render, "BackLean old-path job draws by why (ms per s):{}", why_s);
    {
        namespace TT = VideoCore::TexTable;
        const auto tk = [](std::atomic<u64>& a) { return a.exchange(0); };
        LOG_INFO(Render,
                 "q8 texture table (window): worker materials looked up {} all hit {} | published "
                 "pre-resolved {} rejected (generation moved) {} | back image inserts {} "
                 "ineligible {} | kills {} image kills {} | verify {} BAD {} (cumulative) stale {} evictions {}",
                 tk(TT::g_c[0]), tk(TT::g_c[1]), tk(TT::g_c[2]), tk(TT::g_c[3]),
                 tk(TT::g_c[4]), tk(TT::g_c[7]), tk(TT::g_kills), tk(TT::g_img_kills),
                 tk(TT::g_c[5]), TT::g_c[6].load(), tk(TT::g_c[8]), tk(TT::g_c[9]));
    }
    {
        std::string pk_s; // q4
        for (u32 i = 0; i < 256; ++i) {
            if (b.pk_cyc[i] != 0) {
                pk_s += fmt::format(" {:#x}={:.1f}({:.1f}k/s,{:.0f}cyc)", i, ms(b.pk_cyc[i]),
                                    ks(b.pk_n[i]),
                                    double(b.pk_cyc[i]) / double(std::max<u64>(b.pk_n[i], 1)));
            }
        }
        LOG_INFO(Render, "BackLean main-ring packets by opcode (ms per s):{}", pk_s);
    }
    {
        auto& lb = VideoCore::g_lb;
        const auto per = [](u64 c, u64 n) { return double(c) / double(std::max<u64>(n, 1)); };
        LOG_INFO(Render,
                 "BackLean lane split (ms per s): early ASC service {:.1f} ({:.1f}k/s) | dispatches on "
                 "the CP {:.1f} ({:.1f}k/s, {:.0f} cyc), of which in the early service {:.1f} ({:.1f}k/s, "
                 "{:.0f} cyc) | lane dependency work per binding {:.1f} | lane notes {:.1f} | lane "
                 "submits {:.1f} ({:.1f}k/s)",
                 ms(lb[0]), ks(lb[1]), ms(lb[2]), ks(lb[3]), per(lb[2], lb[3]), ms(lb[4]), ks(lb[5]),
                 per(lb[4], lb[5]), ms(lb[6]), ms(lb[7]), ms(lb[8]), ks(lb[9]));
        lb.fill(0);
        auto& g = VideoCore::g_g1;
        LOG_INFO(Render,
                 "q19 BackLean Call phases (ms per s): idle tails {:.1f}k/s, wait barriers {:.1f}k/s | lazy fences "
                 "{:.1f} | OnSubmit {:.1f} (fault buffer {:.1f}, image downloads {:.1f}, texture GC {:.1f}, buffer "
                 "GC {:.1f}) | Flush {:.1f} | ProtectFence {:.1f} | GpuIdle IRQ {:.1f} | WaitBarrier {:.1f}",
                 ks(g[10]), ks(g[11]), ms(g[0]), ms(g[1]), ms(g[2]), ms(g[3]), ms(g[4]), ms(g[5]), ms(g[6]),
                 ms(g[7]), ms(g[8]), ms(g[9]));
        g.fill(0);
    }
    const u64 other_all = b.follow_cyc > job_all + b.pub_cyc + b.pc_cyc + b.wait_cyc
                              ? b.follow_cyc - job_all - b.pub_cyc - b.pc_cyc - b.wait_cyc
                              : 0;
    const u64 loop_other =
        other_all > b.pro_cyc + b.epi_cyc ? other_all - b.pro_cyc - b.epi_cyc : 0;
    const auto& n = Vulkan::t_back_cycles;
    LOG_INFO(Render,
             "BackLean: window {:.1f}s | ms per s: main ring {:.1f}:{} | job channels {:.1f}:{} | "
             "stretch publish {:.1f} ({:.0f}k/s, {:.0f}cyc) | guest commands in follow {:.1f} | "
             "wait for workers {:.1f} | follow total {:.1f} ({:.1f}k/s), other (switching / Next / "
             "Done) {:.1f} [per switch: prologue {:.1f} epilogue {:.1f} = {:.0f} cyc per follow | per "
             "record: {:.1f} = {:.0f} cyc over {:.0f}k records/s] | publish split: material resolve {:.1f} pass begin {:.1f} record {:.1f} "
             "delta {:.1f} | inside all: flush {:.1f} "
             "({:.0f}/s) gpu wait {:.1f} stream waits {:.2f}",
             s, ms(main_all), main_s, ms(job_all), job_s, ms(b.pub_cyc), ks(b.pub_n),
             double(b.pub_cyc) / double(std::max<u64>(b.pub_n, 1)), ms(b.pc_cyc), ms(b.wait_cyc),
             ms(b.follow_cyc), ks(b.follows),
             ms(b.follow_cyc > job_all + b.pub_cyc + b.pc_cyc + b.wait_cyc
                    ? b.follow_cyc - job_all - b.pub_cyc - b.pc_cyc - b.wait_cyc
                    : 0),
             ms(b.pro_cyc), ms(b.epi_cyc),
             double(b.pro_cyc + b.epi_cyc) / double(std::max<u64>(b.follows, 1)),
             ms(loop_other), double(loop_other) / double(std::max<u64>(b.loop_recs, 1)),
             ks(b.loop_recs),
             ms(b.pub_resolve), ms(b.pub_begin), ms(b.pub_record), ms(b.pub_delta), ms(n.flush_cyc - b.base.flush_cyc),
             double(n.flush_calls - b.base.flush_calls) / s, ms(n.wait_cyc - b.base.wait_cyc),
             ms(n.stream_wait_cyc - b.base.stream_wait_cyc));
    const u64 keep = now;
    b = {};
    b.win_tsc = keep;
    b.base = Vulkan::t_back_cycles;
}
inline void LeanPlaceSample(std::atomic<u64>& p, std::atomic<u64>& e) {
    (Common::IsPerformanceProcessor(Common::CurrentProcessor()) ? p : e)
        .fetch_add(1, std::memory_order_relaxed);
}
/// The front's DCB being recorded (top level only).
struct LeanFront {
    LeanJob* job{};
    const u32* base{};
    bool recording{};
};
LeanFront g_lean_front;

/// Canonical events of one committed record (not RegWrite: those are recorded word by word).
/// Fields, never struct bytes; the vertex-input description id is excluded (worker numbering).
void LeanCanonRecord(std::vector<u64>& v, const CpSplit::RecordHeader& h) {
    using RT = CpSplit::RecordType;
    const auto ev = [&](u64 kind, u64 x) { v.push_back((kind << 56) | (x & 0x00ffffffffffffffull)); };
    const auto* p = reinterpret_cast<const u8*>(&h + 1);
    switch (h.type) {
    case RT::ClearState:
        ev(2, 0);
        break;
    case RT::Extent: {
        const auto& e = *reinterpret_cast<const CpSplit::ExtentPayload*>(p);
        ev(3, (u64{e.slot} << 32) | e.raw);
        break;
    }
    case RT::EpochBump:
        ev(4, reinterpret_cast<const CpSplit::EpochPayload*>(p)->opcode);
        break;
    case RT::Draw: {
        const auto& r = *reinterpret_cast<const SplitDrawRecord*>(p);
        ev(5, reinterpret_cast<u64>(r.draw.pipeline));
        ev(6, u64{r.draw.args.is_indexed} | u64{r.draw.args.is_indirect} << 1 |
                  u64(static_cast<u32>(r.draw.draw_class)) << 2 | u64{r.draw.skip} << 10 |
                  u64{r.draw.args.index_offset} << 16);
        ev(7, r.ctx_epoch);
        ev(8, (u64{r.job_tag} << 32) | r.job_idx);
        ev(9, reinterpret_cast<u64>(r.cmd_address));
        ev(10, (u64{r.stage_count} << 32) | (r.marker != nullptr ? 1u : 0u) | (u64{r.verify} << 8));
        std::array<Shader::BoundStage, Shader::MaxStageTypes> st{};
        const u8* end = ReadStages(reinterpret_cast<const u8*>(&r + 1), r.stage_count, st);
        for (u32 s = 0; s < st.size(); ++s) {
            if (!st[s].info) {
                continue;
            }
            ev(11, (u64{s} << 48) | reinterpret_cast<u64>(st[s].info));
            ev(12, st[s].pgm_base);
            ev(13, (u64(st[s].user_data.size()) << 32) | st[s].flattened_ud_buf.size());
            for (const u32 w : st[s].user_data) {
                ev(14, w);
            }
            for (const u32 w : st[s].flattened_ud_buf) {
                ev(15, w);
            }
        }
        if (r.vi_bytes != 0) {
            const auto& vi = *reinterpret_cast<const CpSplit::VertexInputsPayload*>(end);
            ev(16, (u64{vi.count} << 32) | vi.range_count);
            const auto* at = CpSplit::VertexInputAttribs(vi);
            for (u32 k = 0; k < vi.count; ++k) {
                ev(17, (u64{at[k].desc.vk_format} << 32) | at[k].desc.stride);
                ev(18, (u64{at[k].desc.divisor} << 16) | (u64{at[k].desc.location} << 8) |
                           at[k].desc.instanced);
                ev(19, at[k].base_address);
                ev(20, (u64{at[k].size} << 16) | at[k].range);
            }
            const auto* rg = CpSplit::VertexInputRanges(vi);
            for (u32 k = 0; k < vi.range_count; ++k) {
                ev(21, rg[k].base_address);
                ev(22, rg[k].end_address);
            }
        }
        break;
    }
    case RT::Packet: {
        const auto* w = reinterpret_cast<const u32*>(p);
        ev(23, (u64{h.size} << 32) | w[0]);
        break;
    }
    default:
        ev(30, static_cast<u32>(h.type));
        break;
    }
}

/// The lean parser's packet set (the worker stops before the first packet outside it).
bool LeanSupported(const PM4Header* header, bool guest_markers) {
    switch (header->type3.opcode) {
    case PM4ItOpcode::Nop:
        return !LeanNopHasBackEffect(header, guest_markers);
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndexIndirect:
        return true;
    case PM4ItOpcode::DmaData:
        return reinterpret_cast<const PM4DmaData*>(header)->dst_addr_lo == 0x3022C;
    default:
        return false;
    }
}

/// The chain's register-only pass over one DCB (+ the non-register items the supported packets
/// change, mirroring ProcessGraphics), finding the cut and counting packets / draws.
/// The state a scan updates: the chain's copy (L1) or the front's live members (L2 head pass).
struct LeanScanRefs {
    Regs& r;
    std::array<CbDbExtent, NUM_COLOR_BUFFERS>& cb;
    CbDbExtent& db;
    u64& epoch;
    ComputeProgram& cs;
    VAddr& iaa;
};
/// DispatchSyncStats::RunNeutral (defined later): packets that do not end a dispatch run.
bool LeanRunNeutral(PM4ItOpcode op) {
    switch (op) {
    case PM4ItOpcode::Nop:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::IndirectBuffer:
    case PM4ItOpcode::IndirectBufferConst:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
        return true;
    default:
        return false;
    }
}
/// r3 (2026-10-14): SHADPS4_LEAN_SPLIT_ANY=N - head splitting every N draws at ANY draw, not only
/// after a render-target change (0 = off: the 256-draw render-target rule). The rooftop's long
/// single-target passes (3,158 distant houses, a 1,791-draw shadow cascade) were never split, so
/// one lean worker parsed each at ~3.4 us per draw while the back waited on it (~2.9 ms per frame).
/// A mid-pass span boundary is an ordinary stretch boundary: the next span's entry state is the
/// front's live state there, as at a render-target split.
u32 LeanSplitAny() {
    static const u32 n = [] {
        const char* v = std::getenv("SHADPS4_LEAN_SPLIT_ANY");
        const long x = v ? std::atol(v) : 0;
        return static_cast<u32>(x < 0 ? 0 : x > 65536 ? 65536 : x);
    }();
    return n;
}
void LeanChainScan(std::span<const u32> dcb, LeanScanRefs s, bool guest_markers, bool post_dma,
                   u32& cut_dw, u32& pkts, u32& draws, bool stop_at_cut = false,
                   u32* run_bumps = nullptr, u32 max_draws = 0, bool* split = nullptr) {
    const u32* base = dcb.data();
    const std::span<const u32> full = dcb;
    const Lean::StaticEnv senv{.guest_markers = guest_markers, .split_post_dma = post_dma};
    auto& r = s.r;
    cut_dw = static_cast<u32>(dcb.size());
    bool cut = false;
    // Head splitting cuts only where a stretch ends anyway: a real change of the render windows
    // (A2RenderWindows: CB / DB / scissor / blend context registers) or a CLEAR_STATE since the
    // last draw.
    bool rt_changed = false;
    const auto touches_rt = [](u32 ctx_off, u32 n) {
        constexpr std::array<std::pair<u32, u32>, 4> win{
            {{0x000, 0x020}, {0x080, 0x0A0}, {0x200, 0x210}, {0x318, 0x390}}};
        for (const auto& [lo, hi] : win) {
            if (ctx_off < hi && ctx_off + n > lo) {
                return true;
            }
        }
        return false;
    };
    while (!dcb.empty()) {
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        if (header->type == 2) {
            dcb = dcb.subspan(1);
            continue;
        }
        if (header->type != 3) {
            break;
        }
        const u32 count = header->type3.NumWords();
        const auto* w = reinterpret_cast<const u32*>(header);
        const auto op = static_cast<PM4ItOpcode>(header->type3.opcode.Value());
        ++pkts;
        if (!cut) {
            u32 sop = 0;
            const u32 at = static_cast<u32>(dcb.data() - base);
            if (Lean::StaticCheck(full, at, r, senv, sop) != Lean::Stop::None) {
                cut = true;
                cut_dw = at;
                if (stop_at_cut) {
                    return; // L2 head pass: the cut packet and the rest are the front's
                }
            }
            // Head splitting: after max_draws draws, end this span right before the next draw
            // packet; the live state here is the next span's entry state.
            if (!cut && stop_at_cut && max_draws != 0 && draws >= max_draws &&
                (rt_changed || LeanSplitAny() != 0) &&
                (op == PM4ItOpcode::DrawIndex2 || op == PM4ItOpcode::DrawIndexOffset2 ||
                 op == PM4ItOpcode::DrawIndexAuto)) {
                cut_dw = static_cast<u32>(dcb.data() - base);
                --pkts;
                if (split) {
                    *split = true;
                }
                return;
            }
        }
        if (op == PM4ItOpcode::DrawIndex2 || op == PM4ItOpcode::DrawIndexOffset2 ||
            op == PM4ItOpcode::DrawIndexAuto) {
            rt_changed = false;
        }
        switch (op) {
        case PM4ItOpcode::ClearState:
            r.SetDefaults();
            ++s.epoch;
            rt_changed = true;
            break;
        case PM4ItOpcode::SetConfigReg:
            std::memcpy(&r.reg_array[Regs::ConfigRegWordOffset + (w[1] & 0xffff)], w + 2,
                        size_t{count - 1} * 4);
            break;
        case PM4ItOpcode::SetContextReg: {
            const u32 reg_addr = Regs::ContextRegWordOffset + (w[1] & 0xffff);
            const u32* payload = w + 2;
            const size_t sz = size_t{count - 1} * 4;
            if (max_draws != 0 && !rt_changed && touches_rt(w[1] & 0xffff, count - 1) &&
                std::memcmp(&r.reg_array[reg_addr], payload, sz) != 0) {
                rt_changed = true;
            }
            if (LeanViewportMemoEnabled() && std::memcmp(&r.reg_array[reg_addr], payload, sz) != 0) {
                ++s.epoch;
            }
            std::memcpy(&r.reg_array[reg_addr], payload, sz);
            // The CB / DB extent NOP hints (as ProcessGraphics' SET_CONTEXT_REG).
            const u32 nop_offset = header->type3.count;
            if (reg_addr >= Liverpool::ContextRegs::CbColor0Base && reg_addr <= Liverpool::ContextRegs::CbColor7Base &&
                (reg_addr - Liverpool::ContextRegs::CbColor0Base) %
                        (Liverpool::ContextRegs::CbColor1Base - Liverpool::ContextRegs::CbColor0Base) == 0) {
                const u32 id = (reg_addr - Liverpool::ContextRegs::CbColor0Base) /
                               (Liverpool::ContextRegs::CbColor1Base - Liverpool::ContextRegs::CbColor0Base);
                if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                    s.cb[id].raw = payload[nop_offset + 1];
                } else {
                    s.cb[id].raw = 0;
                }
            } else if (reg_addr >= Liverpool::ContextRegs::CbColor0Cmask &&
                       reg_addr <= Liverpool::ContextRegs::CbColor7Cmask &&
                       (reg_addr - Liverpool::ContextRegs::CbColor0Cmask) %
                               (Liverpool::ContextRegs::CbColor1Cmask - Liverpool::ContextRegs::CbColor0Cmask) == 0) {
                const u32 id = (reg_addr - Liverpool::ContextRegs::CbColor0Cmask) /
                               (Liverpool::ContextRegs::CbColor1Cmask - Liverpool::ContextRegs::CbColor0Cmask);
                if (nop_offset == 0x04) {
                    s.cb[id].raw = payload[nop_offset + 1];
                }
            } else if (reg_addr == Liverpool::ContextRegs::DbZInfo) {
                s.db.raw = header->type3.count == 8 ? payload[21] : 0;
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const u32 off = w[1] & 0xffff;
            if (off >= 0x200 && off <= 0x200 + sizeof(ComputeProgram) / 4) {
                std::memcpy(reinterpret_cast<u32*>(&s.cs) + (off - 0x200), w + 2,
                            size_t{count - 1} * 4);
            } else {
                std::memcpy(&r.reg_array[Regs::ShRegWordOffset + off], w + 2, size_t{count - 1} * 4);
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg:
            std::memcpy(&r.reg_array[Regs::UconfigRegWordOffset + (w[1] & 0xffff)], w + 2,
                        size_t{count - 1} * 4);
            break;
        case PM4ItOpcode::IndexType:
            r.index_buffer_type.raw = reinterpret_cast<const PM4CmdDrawIndexType*>(header)->raw;
            break;
        case PM4ItOpcode::DrawIndex2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            r.max_index_size = d->max_size;
            r.index_base_address.base_addr_lo = d->index_base_lo;
            r.index_base_address.base_addr_hi = d->index_base_hi;
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            ++draws;
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            r.max_index_size = d->max_size;
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            ++draws;
            break;
        }
        case PM4ItOpcode::DrawIndexAuto: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            r.num_indices = d->index_count;
            r.draw_initiator = d->draw_initiator;
            ++draws;
            break;
        }
        case PM4ItOpcode::DrawIndirect:
        case PM4ItOpcode::DrawIndexIndirect:
        case PM4ItOpcode::DrawIndirectMulti:
        case PM4ItOpcode::DrawIndexIndirectMulti:
        case PM4ItOpcode::DrawIndexIndirectCountMulti:
            ++draws;
            break;
        case PM4ItOpcode::DispatchDirect: {
            // ProcessGraphics writes the dimensions and the initiator into the gfx cs_state.
            const auto* d = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            s.cs.dim_x = d->dim_x;
            s.cs.dim_y = d->dim_y;
            s.cs.dim_z = d->dim_z;
            s.cs.dispatch_initiator = d->dispatch_initiator;
            break;
        }
        case PM4ItOpcode::NumInstances:
            r.num_instances.num_instances =
                reinterpret_cast<const PM4CmdDrawNumInstances*>(header)->num_instances;
            break;
        case PM4ItOpcode::IndexBase: {
            const auto* b = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
            r.index_base_address.base_addr_lo = b->addr_lo;
            r.index_base_address.base_addr_hi = b->addr_hi;
            break;
        }
        case PM4ItOpcode::IndexBufferSize:
            r.num_indices = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header)->num_indices;
            r.max_index_size = r.num_indices; // r4: INDEX_BUFFER_SIZE = VGT_DMA_MAX_SIZE (upstream #5278)
            break;
        case PM4ItOpcode::SetBase:
            s.iaa = reinterpret_cast<const PM4CmdSetBase*>(header)->Address<u64>();
            break;
        case PM4ItOpcode::EventWrite:
            if (reinterpret_cast<const PM4CmdEventWrite*>(header)->event_type.Value() ==
                EventType::SoVgtStreamoutFlush) {
                r.cp_strmout_cntl.offset_update_done = 1;
            }
            break;
        default:
            break;
        }
        if (run_bumps && !LeanRunNeutral(op)) {
            ++*run_bumps;
        }
        dcb = dcb.subspan(std::min<size_t>(dcb.size(), size_t{count} + 1));
    }
}

/// Compares the front's and the worker's events of one DCB (whoever finishes second).
void LeanCompare(LeanJob& j) {
    auto& c = g_lean_s.c;
    const auto& res = j.res;
    size_t n_front = j.front_canon.size();
    if (res.end_dw < j.dcb.size()) {
        n_front = ~size_t{0};
        for (const auto& [off, n] : j.front_marks) {
            if (off == res.end_dw) {
                n_front = n;
                break;
            }
        }
    }
    c.pkts_span += res.packets;
    c.run_bumps += res.run_bumps;
    c.draws_span += res.draws;
    for (const u64 e : j.worker_canon) {
        if ((e >> 56) == 4) { // EpochBump inside the worker's span: a bug (hard check)
            c.epoch_bumps += 1;
            u32 none = 0;
            c.epoch_bump_op.compare_exchange_strong(none, 0x100u | static_cast<u32>(e & 0xff));
        }
    }
    c.stops[std::min<u32>(static_cast<u32>(res.stop), 31)] += 1;
    if (Lean::IsStaticStop(res.stop)) {
        // The chain should have cut at this packet: a changed buffer (lag) or a real bug.
        if (j.lag.load(std::memory_order_acquire)) {
            ++c.parity_changed;
        } else if (++c.parity <= 8) {
            LOG_WARNING(Render, "Lean PARITY (buffer unchanged): DCB {} worker stop {} at dw {} "
                                "(opcode {:#x}), chain cut {} of {} dw",
                        j.index.load(), Lean::StopName(res.stop), res.end_dw, res.stop_opcode,
                        j.cut_dw, j.dcb.size());
        }
    }
    if (res.stop != Lean::Stop::End && res.stop != Lean::Stop::Cut) {
        c.stop_ops[res.stop_opcode & 255] += 1;
    }
    if (j.lag.load(std::memory_order_acquire)) {
        ++c.lag; // the guest may have rewritten the buffer before the worker read it
    } else if (n_front == ~size_t{0}) {
        ++c.no_mark;
    } else {
        ++c.cmp;
        c.events += j.worker_canon.size();
        size_t i = 0;
        const size_t n = std::min(n_front, j.worker_canon.size());
        while (i < n && j.front_canon[i] == j.worker_canon[i]) {
            ++i;
        }
        if (i != n_front || i != j.worker_canon.size()) {
            const u64 fe = i < n_front ? j.front_canon[i] : (i < j.worker_canon.size() ? j.worker_canon[i] : 0);
            c.bad_kind[(fe >> 56) & 31] += 1;
            if (++c.bad <= 24) {
                LOG_WARNING(Render,
                            "Lean verify: DCB {} differs at event {} of front {} / worker {}: front "
                            "{:#018x} worker {:#018x} (span {} packets {} draws, stop {})",
                            j.index.load(), i, n_front, j.worker_canon.size(),
                            i < n_front ? j.front_canon[i] : 0ull,
                            i < j.worker_canon.size() ? j.worker_canon[i] : 0ull, res.packets,
                            res.draws, static_cast<u32>(res.stop));
            }
            static std::atomic<u32> ud_logged{0};
            if (i < n && (j.front_canon[i] >> 56) == 14 && ud_logged.fetch_add(1) < 6) {
                // The truth: entry windows + register events up to i.
                auto truth = std::make_unique<Regs>();
                const u32* w = j.win.data();
                for (const auto& [off, cnt] : CpSplit::DcbSnapWindows) {
                    std::memcpy(&truth->reg_array[off], w, size_t{cnt} * 4);
                    w += cnt;
                }
                s64 stage = -1, first_ud = -1;
                for (size_t k = 0; k < i; ++k) {
                    const u64 e = j.front_canon[k];
                    const u64 kind = e >> 56;
                    if (kind == 1) {
                        truth->reg_array[(e >> 32) & 0xffffff] = static_cast<u32>(e);
                    } else if (kind == 2) {
                        truth->SetDefaults();
                    } else if (kind == 11) {
                        stage = static_cast<s64>((e >> 48) & 0xff);
                        first_ud = -1;
                    } else if (kind == 14 && first_ud < 0) {
                        first_ud = static_cast<s64>(k);
                    }
                }
                if (first_ud < 0) {
                    first_ud = static_cast<s64>(i);
                }
                const u64 slot = i - static_cast<u64>(first_ud);
                const u32 in_stage = stage == static_cast<s64>(Shader::LogicalStage::Fragment)
                                         ? static_cast<u32>(Shader::Stage::Fragment)
                                         : static_cast<u32>(Shader::Stage::Vertex);
                const auto* pgm = truth->ProgramForStage(in_stage);
                const u32 tv = pgm && slot < pgm->user_data.size() ? pgm->user_data[slot] : 0xdeadbeefu;
                std::string fr, wk;
                for (size_t k = i >= 24 ? i - 24 : 0; k < std::min(i + 4, n); ++k) {
                    fr += fmt::format(" {:x}", j.front_canon[k]);
                    wk += fmt::format(" {:x}", j.worker_canon[k]);
                }
                LOG_WARNING(Render, "Lean verify user data: DCB {} event {}: stages_en {:#x} ({}) logical "
                                    "stage {} user-data slot {}: TRUE (registers) {:#x} front {:#x} worker "
                                    "{:#x} -> {}",
                            j.index.load(), i, static_cast<u32>(truth->stage_enable.raw),
                            truth->stage_enable.raw == AmdGpu::ShaderStageEnable::VgtStages::Vs
                                ? "plain Vs"
                                : "NOT plain Vs",
                            stage, slot, tv, static_cast<u32>(j.front_canon[i]),
                            static_cast<u32>(j.worker_canon[i]),
                            tv == static_cast<u32>(j.front_canon[i])
                                ? "worker wrong"
                                : (tv == static_cast<u32>(j.worker_canon[i]) ? "FRONT wrong" : "neither"));
                LOG_WARNING(Render, "Lean verify user data: DCB {} front around:{}", j.index.load(), fr);
                LOG_WARNING(Render, "Lean verify user data: DCB {} worker around:{}", j.index.load(), wk);
            }
            if (c.bad <= 3) {
                const auto dump = [](const std::vector<u64>& v, size_t lim) {
                    std::string s;
                    for (size_t k = 0; k < std::min<size_t>(lim, 96); ++k) {
                        s += fmt::format(" {:x}", v[k]);
                    }
                    return s;
                };
                std::string pk;
                u32 off = 0, pads = 0;
                const u32 end = std::min<u32>(res.end_dw + 1, static_cast<u32>(j.dcb.size()));
                while (off < end) {
                    const auto* h = reinterpret_cast<const PM4Header*>(j.dcb.data() + off);
                    if (h->type == 2) {
                        ++pads;
                        ++off;
                        continue;
                    }
                    if (pads) {
                        pk += fmt::format(" pad{}", pads);
                        pads = 0;
                    }
                    if (h->type != 3) {
                        pk += fmt::format(" {}:type{}", off, u32(h->type));
                        break;
                    }
                    pk += fmt::format(" {}:{:#x}", off, u32(h->type3.opcode.Value()));
                    if (h->type3.opcode == PM4ItOpcode::SetContextReg ||
                        h->type3.opcode == PM4ItOpcode::SetShReg ||
                        h->type3.opcode == PM4ItOpcode::SetUconfigReg) {
                        pk += fmt::format("[{:#x}x{}]", j.dcb[off + 1] & 0xffff, h->type3.NumWords() - 1);
                    }
                    off += h->type3.NumWords() + 1;
                }
                LOG_WARNING(Render, "Lean verify dump DCB {}: packets (end_dw {}):{}", j.index.load(),
                            res.end_dw, pk);
                LOG_WARNING(Render, "Lean verify dump DCB {}: front:{}", j.index.load(),
                            dump(j.front_canon, n_front));
                LOG_WARNING(Render, "Lean verify dump DCB {}: worker:{}", j.index.load(),
                            dump(j.worker_canon, j.worker_canon.size()));
            }
        }
    }
    j.done_mask.store(0, std::memory_order_relaxed);
    j.state.store(0, std::memory_order_release);
}

// SHADPS4_WRM_PARTIAL=1 (l3l): a WAIT_REG_MEM waits only through the last outstanding posted write to
// its label (TASK_GUARD write log), not for the whole backlog; VO labels without any drain. Census
// (front-written, logged by LeanLog): kinds 0 posted write in the log, 1 VO label, 2 none,
// 3 register, 4 log overflow.
const bool g_wrm_partial = [] {
    const char* v = std::getenv("SHADPS4_WRM_PARTIAL");
    return v && v[0] == '1';
}();
struct WrmCensus {
    std::array<std::atomic<u64>, 5> kind{}, passed{}, fallback{};
    std::atomic<u64> waits{}, vo_waits{}, backlog{}, partial_records{}, partial_ticks{},
        partial_waits{};
    std::atomic<CpSplit::Channel*> ch{}; ///< the front's channel, for the drain probe at log time
} g_wrm;
enum class WrmPre { Full, Passed, VoWait, PassedFwdFence };
const bool g_rptr_backing = [] {
    const char* v = std::getenv("SHADPS4_CP_SPLIT_RPTR_BACKING");
    return v && v[0] == '1';
}();
std::atomic<u64> g_rptr_writes{0}, g_rptr_fallbacks{0}, g_rptr_changed{0};
std::array<std::atomic<u64>, 4> g_fwd_why{};
std::array<std::atomic<u64>, 6> g_fwd_unknown_src{}; // unknown value by write-log source // not forwarded: 0 unknown value, 1 not at the label
                                             // start / partial, 2 value fails the test, 3 newer
const int g_wait_barrier = [] {
    const char* v = std::getenv("SHADPS4_CP_SPLIT_WAIT_BARRIER");
    return v && v[0] >= '1' && v[0] <= '2' ? v[0] - '0' : 0;
}();
std::array<std::atomic<u64>, 2> g_wait_barriers{};
bool WaitPassesWith(const PM4CmdWaitRegMem* w, u32 value); // defined with FENCE_FORWARD
const bool g_post_order = [] {
    const char* v = std::getenv("SHADPS4_CP_SPLIT_POST_ORDER");
    return v && v[0] == '1';
}();
struct PostOrderStats {
    std::atomic<u64> idle_posted{}, pfp_posted{}, zpass_posted{}, zpass_guarded{}, forwarded{},
        fwd_cmp{}, fwd_bad{};
    std::array<std::atomic<u64>, 8> fault{}; // read / invalidate x span, label, logged, other
} g_po;
thread_local VAddr t_wrm_label = 0; ///< the label the front's WAIT_REG_MEM tested last
/// A verbatim copy of a gfx packet, committed (not posted yet).
void CommitPacketCopy(CpSplit::Channel& ch, const PM4Header* header, u16 flags) {
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = ch.Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32), flags);
    std::memcpy(copy, header, dwords * sizeof(u32));
    ch.Commit();
}
const bool g_post_wdata = [] {
    const char* v = std::getenv("SHADPS4_CP_SPLIT_POST_WDATA");
    return v && v[0] == '1';
}();
const bool g_post_flip = [] {
    const char* v = std::getenv("SHADPS4_CP_SPLIT_POST_FLIP");
    return v && v[0] == '1';
}();
struct PostWdFlipStats {
    std::atomic<u64> wdata_posted{}, wdata_guarded{}, flips_posted{};
} g_pwf;
WrmPre WrmPartial(CpSplit::Channel& ch, const PM4CmdWaitRegMem* w, bool vo_label,
                  std::span<const u32> regs) {
    const auto add = [](std::atomic<u64>& a, u64 v) { a.fetch_add(v, std::memory_order_relaxed); };
    add(g_wrm.waits, 1);
    g_wrm.ch.store(&ch, std::memory_order_relaxed);
    if (w->mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        // The front's own register file: the back owes it nothing.
        add(g_wrm.kind[3], 1);
        if (g_wrm_partial && w->Test(regs)) {
            add(g_wrm.passed[3], 1);
            return WrmPre::Passed;
        }
        add(g_wrm.fallback[3], 1);
        return WrmPre::Full;
    }
    const VAddr lo = reinterpret_cast<VAddr>(w->Address<u32*>());
    t_wrm_label = lo;
    const u64 done0 = ch.DoneSeq();
    const u64 committed = ch.CommittedSeq();
    add(g_wrm.backlog, committed > done0 ? committed - done0 : 0);
    ch.ArmDrainProbe();
    const u64 seq = ch.OutstandingWriteSeq(lo, lo + sizeof(u32));
    const u32 k = seq == ~0ull ? 4 : seq != 0 ? 0 : vo_label ? 1 : 2;
    add(g_wrm.kind[k], 1);
    // No logged writer (2): the label may be written by something the log does not cover (a
    // shader result read back, another queue): the old path, counted (advisor 2026-10-10).
    if (!g_wrm_partial || k == 4 || k == 2) {
        add(g_wrm.fallback[k], 1);
        return WrmPre::Full;
    }
    bool fwd = false, fwd_fence = false;
    u32 fwd_value = 0;
    if (k == 0 && g_post_order) {
        // SHADPS4_CP_SPLIT_POST_ORDER: the newest logged writer's value passes the test - an
        // ordering point only; the back's stream order runs the write before what follows.
        u64 fseq = 0;
        u32 why = 3;
        const bool known = ch.NewestWriteValue(lo, fseq, fwd_value, fwd_fence, &why);
        fwd = known && fseq == seq && WaitPassesWith(w, fwd_value);
        if (!fwd) {
            g_fwd_why[known ? (fseq == seq ? 2 : 3) : (why & 15)].fetch_add(
                1, std::memory_order_relaxed);
            if (!known && (why & 15) == 0) {
                g_fwd_unknown_src[std::min<u32>(why >> 4, 5)].fetch_add(1,
                                                                       std::memory_order_relaxed);
            }
        }
        if (fwd && g_a2 != 2) {
            add(g_po.forwarded, 1);
            add(g_wrm.passed[0], 1);
            return fwd_fence ? WrmPre::PassedFwdFence : WrmPre::Passed;
        }
    }
    if (k == 0) {
        add(g_wrm.partial_records, seq > done0 ? seq - done0 : 0);
        add(g_wrm.partial_waits, 1);
        const u64 t0 = __rdtsc();
        ch.WaitThrough(seq, CpSplit::DrainReason::WaitRegMemPartial);
        add(g_wrm.partial_ticks, __rdtsc() - t0);
        if (fwd) { // A2=2: the forwarded value against the label after the real wait
            add(g_po.fwd_cmp, 1);
            if (*w->Address<u32*>() != fwd_value &&
                g_po.fwd_bad.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render, "POST_ORDER verify: label {:#x} holds {:#x}, the log predicted {:#x}",
                            lo, *w->Address<u32*>(), fwd_value);
            }
        }
    } else {
        ch.PublishNow(); // VO label: the back must be able to reach the flip it waits for
    }
    if (w->Test(regs)) {
        add(g_wrm.passed[k], 1);
        return fwd && fwd_fence ? WrmPre::PassedFwdFence : WrmPre::Passed; // A2=2 keeps the barrier
    }
    if (vo_label) {
        add(g_wrm.vo_waits, 1);
        return WrmPre::VoWait;
    }
    add(g_wrm.fallback[k], 1);
    return WrmPre::Full;
}

void LeanLog() {
    auto& s = g_lean_s;
    auto& c = s.c;
    const u64 now = __rdtsc();
    const double sec = double(now - s.win_tsc) / 3.5e9;
    std::string stops;
    for (u32 i = 0; i < 32; ++i) {
        if (c.stops[i] != 0) {
            stops += fmt::format(" {}={}", i, c.stops[i].load());
        }
    }
    std::string kinds, misses;
    for (u32 i = 0; i < 32; ++i) {
        if (c.bad_kind[i] != 0) {
            kinds += fmt::format(" {}={}", i, c.bad_kind[i].load());
            c.bad_kind[i] = 0;
        }
    }
    for (u32 i = 0; i < 8; ++i) {
        if (g_lean_fault_stats.miss[i] != 0) {
            misses += fmt::format(" {}={}", i, g_lean_fault_stats.miss[i].load());
        }
    }
    if (g_lean == 2) {
        const double hd = double(s.head_draws.load()), fd = double(c.front_draws.load());
        LOG_INFO(Render,
                 "Lean L2: window {:.1f}s | DCBs {} taken {} not taken {} (extra split spans {}) | head "
                 "draws {} front draws {} "
                 "(workers {:.1f}%) | EARLY STOPS {} (gate 0) | back: worker-channel records {:.0f}/s, "
                 "waiting for workers {:.2f} ms/s | worker: fault waits {} ({:.2f} ms/s), faults in lock {} "
                 "(first {:#x}), touches {}, builds {} (refresh failed {}), null builds {}, native preps {}, "
                 "colour preps {} | head pass {:.2f} ms/s | front faults in lock {}, register-only "
                 "completions {} (draws dropped {}), back lock-service waits {}",
                 sec, c.dcbs.load(), s.l2_taken.load(), s.l2_skipped.load(),
                 std::exchange(s.split_spans, 0), s.head_draws.load(),
                 c.front_draws.load(), hd + fd > 0 ? 100.0 * hd / (hd + fd) : 0.0, s.l2_early.load(),
                 double(s.back_records.load()) / std::max(sec, 1e-9),
                 double(s.back_wait_ticks.load()) / 3.5e6 / std::max(sec, 1e-9),
                 g_lean_l2.fault_waits.load(),
                 double(g_lean_l2.wait_ticks.load()) / 3.5e6 / std::max(sec, 1e-9),
                 g_lean_l2.faults_in_lock.load(), g_lean_l2.first_in_lock.load(),
                 g_lean_l2.touches.load(), g_lean_l2.builds.load(), g_lean_l2.build_refresh_fail.load(),
                 g_lean_l2.build_null.load(), g_lean_l2.native_preps.load(),
                 g_lean_l2.colour_preps.load(), double(s.chain_cyc) / 3.5e6 / std::max(sec, 1e-9),
                 g_lean_l2.front_faults_in_lock.load(), g_lean_l2.regs_only_runs.load(),
                 g_lean_l2.dropped_draws.load(), g_lean_l2.lock_service_waits.load());
        {
            auto& p = g_lean_phase;
            const double d = std::max(1.0, double(p.draws.load()));
            const auto us = [&](const std::atomic<u64>& x) { return double(x.load()) / 3500.0 / d; };
            u64 full = 0, depth = 0;
            for (auto& ch : s.ch) {
                if (ch) {
                    const auto [fw_, dw_] = ch->TakeWaitTicks();
                    full += fw_;
                    depth += dw_;
                }
            }
            const double fw = double(g_lean_l2.wait_ticks.load());
            const double parse = double(p.parse.load());
            const double other = parse - double(p.prewalk + p.lookup + p.derive + p.emit) -
                                 double(full + depth) - fw;
            const double nw = double(std::max<size_t>(1, s.workers.size()));
            const double wall = sec * 3.5e9 * nw;
            std::string per_worker;
            u64 mh = 0, mm = 0;
            for (u32 i = 0; i < p.memo_hit.size(); ++i) {
                const u64 h = p.memo_hit[i].exchange(0), m = p.memo_miss[i].exchange(0);
                mh += h;
                mm += m;
                if (h + m != 0) {
                    per_worker += fmt::format(" w{} {:.1f}%", i, 100.0 * double(h) / double(h + m));
                }
            }
            LOG_INFO(Render,
                     "Lean L2 phases: {} worker draws | us per worker draw: prewalk {:.2f} (header "
                     "{:.2f} info {:.2f} walk {:.2f} fetch {:.2f}; lock takes per draw: map {:.3f} info "
                     "{:.3f}) lookup "
                     "{:.2f} derive {:.2f} emit {:.2f} (record+commit incl. L3 translate {:.2f}, program "
                     "table {:.2f}, table publishes skipped {:.1f}%) other-parse {:.2f} | waits per draw: "
                     "channel "
                     "full {:.2f} depth {:.2f} fault {:.2f} | seed {:.2f} us per DCB | worker time: "
                     "claim wait {:.1f}%, busy (parse+seed) {:.1f}% | memo hits {:.1f}% (per worker:{}) "
                     "ineligible {} verify {} MISMATCH {} | header hits {:.1f}% fetch hits {:.1f}%",
                     u64(d), us(p.prewalk), us(p.pw_header), us(p.pw_info), us(p.pw_walk),
                     us(p.pw_fetch), double(p.map_lock.load()) / d, double(p.info_lock.load()) / d,
                     us(p.lookup), us(p.derive), us(p.emit), us(p.emit_commit), us(p.emit_table),
                     100.0 * double(p.table_skips.load()) / d,
                     other / 3500.0 / d, double(full) / 3500.0 / d, double(depth) / 3500.0 / d,
                     fw / 3500.0 / d,
                     double(p.seed.load()) / 3500.0 / std::max(1.0, double(s.l2_taken.load())),
                     100.0 * double(p.claim_wait.load()) / wall,
                     100.0 * (parse + double(p.seed.load())) / wall,
                     100.0 * double(mh) / std::max(1.0, double(mh + mm)), per_worker,
                     p.memo_ineligible.load(), p.memo_verify.load(), p.memo_verify_bad.load(),
                     100.0 * double(p.hdr_hit) / std::max(1.0, double(p.hdr_hit + p.hdr_miss)),
                     100.0 * double(p.fetch_hit) / std::max(1.0, double(p.fetch_hit + p.fetch_miss)));
            p.claim_wait = p.seed = p.parse = p.prewalk = p.lookup = p.derive = p.emit = p.draws = 0;
            p.hdr_hit = p.hdr_miss = p.fetch_hit = p.fetch_miss = 0;
            p.pw_header = p.pw_info = p.pw_walk = p.pw_fetch = p.map_lock = p.info_lock = 0;
            p.emit_commit = p.emit_table = p.table_skips = 0;
            p.memo_ineligible = 0;
        }
        if (g_lean_xl.on) {
            auto& x = g_lean_xl;
            const double ps = 1.0 / std::max(sec, 1e-9);
            std::array<u64, 32> rf{};
            std::array<u64, 52> sb{};
            for (const auto& t : x.tr) {
                if (const auto* tr = t.load(std::memory_order_acquire)) {
                    for (u32 i = 0; i < 32; ++i) {
                        rf[i] += tr->refusals[i]; // cumulative (racy reads, as A2Log)
                    }
                    for (u32 i = 0; i < 52; ++i) {
                        sb[i] += tr->sub[i];
                    }
                }
            }
            LOG_INFO(Render,
                     "Lean L3 stamp refusals (cumulative): stamp below idx+1 {} above {} gpu_ok 0 {} "
                     "cmd differs {} | record fields: inst_known 0 {} io_ud {} index_count 0 {} "
                     "push_inh {} | slot kind: shadow+images {} clip {} not Guest {} written {} "
                     "formatted {} colour inline/null/empty {} | guest lookup: no buffer {} outside "
                     "{} | inherited refused: size {} no buffer {} offset {} | taken with inherited "
                     "slots {} with clip planes {} | guest-lookup fallback {} | stretches cut by a 9th "
                     "material {} | no variant: pipeline undecided {} eligible (no variant for this "
                     "vertex description yet) {} untextured {} other ineligible {} | eligible by slot: refused (final) {} "
                     "claimed without variant {} not claimed, free slot {} not claimed, slots full {} | indirect: "
                     "with a variant {} without {} (q1 why: no vi {} instanced {} undecided {} eligible {} "
                     "untextured {} other ineligible {}; eligible by slot: refused (final) {} "
                     "built {} building {} not claimed {}) | q4 colour vertex: instanced/location {} "
                     "second VkBuffer {} offset > 4 GB {} all null {} | q53 vbuf2: third VkBuffer {} taken with a second {}",
                     sb[12], sb[13], sb[14], sb[15], sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], sb[6],
                     sb[7], sb[8], sb[9], sb[10], sb[11], sb[16], sb[17], sb[18], sb[19], sb[20],
                     sb[23], x.mat_cuts.exchange(0), sb[24], sb[25], sb[26], sb[27], sb[28], sb[29], sb[30], sb[31], sb[32], sb[33], sb[34], sb[35], sb[36], sb[37], sb[38], sb[39], sb[40], sb[41], sb[42], sb[43], sb[44], sb[45], sb[46], sb[47], sb[48], sb[49]);
            LOG_INFO(Render,
                     "Lean L3 direct checks (A2=2): inherited-slot words {} BAD {} | user-data "
                     "words {} BAD {} | clip planes {} BAD {} | formatted + fallback table bases {} BAD {} | "
                     "formatted: draws taken (cumulative) {}, texel syncs {} rejected stretches {}, "
                     "17th range refused (cumulative) {} | verify skipped (A2=1 would reject) {}",
                     x.inh_cmp.exchange(0), x.inh_bad.load(), x.ud_cmp.exchange(0), x.ud_bad.load(),
                     x.clip_cmp.exchange(0), x.clip_bad.load(), x.fmt_cmp.exchange(0),
                     x.fmt_bad.load(), sb[22], x.texel_syncs.exchange(0), x.rej_texel.exchange(0),
                     sb[21], x.verify_skips.exchange(0));
            LOG_INFO(Render, "Lean L3 verify: buffer checks skipped (generation moved mid-stretch by "
                             "the verify's own old path) {}",
                     x.verify_mid.exchange(0));
            {
                auto& w = g_wrm;
                const auto tk = [](std::atomic<u64>& a) { return a.exchange(0); };
                const u64 n = tk(w.waits);
                std::array<u64, 5> kd{}, ps_{}, fb{};
                for (u32 i = 0; i < 5; ++i) {
                    kd[i] = tk(w.kind[i]);
                    ps_[i] = tk(w.passed[i]);
                    fb[i] = tk(w.fallback[i]);
                }
                const u64 mem = kd[0] + kd[1] + kd[2] + kd[4];
                const u64 pw = tk(w.partial_waits);
                auto* wch = w.ch.load(std::memory_order_relaxed);
                const auto [pn, pt] = wch ? wch->TakeDrainProbe() : std::pair<u64, u64>{};
                LOG_INFO(Render,
                         "WRM census (partial {}): waits {:.0f}/s | writer: posted write in log {} VO "
                         "label {} none {} register {} log overflow {} | passed after the partial "
                         "wait: log {} VO {} none {} register {} | VO waits without drain {} | old "
                         "path (full drain): log {} VO {} none {} register {} overflow {} | backlog at "
                         "a memory wait {:.0f} records avg | partial waits {} for {:.0f} records avg, "
                         "{:.3f} ms avg | a full drain of that backlog takes {:.3f} ms avg (probes {})",
                         g_wrm_partial ? 1 : 0, double(n) * ps, kd[0], kd[1], kd[2], kd[3], kd[4],
                         ps_[0], ps_[1], ps_[2], ps_[3], tk(w.vo_waits), fb[0], fb[1], fb[2], fb[3],
                         fb[4], double(tk(w.backlog)) / double(std::max<u64>(mem, 1)), pw,
                         double(tk(w.partial_records)) / double(std::max<u64>(pw, 1)),
                         double(tk(w.partial_ticks)) / 3.5e6 / double(std::max<u64>(pw, 1)),
                         double(pt) / 3.5e6 / double(std::max<u64>(pn, 1)), pn);
                const auto tk2 = [](std::atomic<u64>& a) { return a.exchange(0); };
                if (g_asc_early) {
                    auto& e = g_asc_early_stats;
                    LOG_INFO(Render,
                             "ASC_EARLY: ran ahead dispatches {:.0f}/s packets {:.0f}/s | in ring "
                             "order (registers / epoch / diag) {:.0f}/s | early service blocked {:.0f}/s "
                             "| skipped in ring order {:.0f}/s | early per-binding sync: checks "
                             "{:.0f}/s, records that uploaded {:.0f}/s, {:.2f} MB/s | inverted reads "
                             "(gfx read, earlier in ring order, of a range an early ASC record wrote) "
                             "{:.0f}/s of {:.0f}/s checked, early writes {:.0f}/s (shader write "
                             "bindings vs old-path reads only) | ASC waits on another queue's label "
                             "(ordering Call) {:.0f}/s | front waits for early-queue space {} | l4f: "
                             "early record time dispatches {:.1f} packets {:.1f} ms/s | service gap "
                             "with work pending {:.1f} ms/s (largest {:.2f} ms) | stops at unpublished "
                             "work {:.0f}/s",
                             double(tk2(e.dispatches)) * ps, double(tk2(e.packets)) * ps,
                             double(tk2(e.in_order)) * ps, double(tk2(e.blocked)) * ps,
                             double(tk2(e.skipped)) * ps, double(tk2(e.forced_checks)) * ps,
                             double(tk2(e.uploading_records)) * ps,
                             double(tk2(e.upload_bytes)) * ps / 1e6,
                             double(VideoCore::g_early_inv_reads.exchange(0)) * ps,
                             double(VideoCore::g_early_inv_checks.exchange(0)) * ps,
                             double(VideoCore::g_early_inv_writes.exchange(0)) * ps,
                             double(tk2(e.cross_waits)) * ps,
                             g_wrm.ch.load() ? g_wrm.ch.load()->early_full_waits : 0,
                             double(tk2(e.c_disp)) / 3.5e6 * ps, double(tk2(e.c_pkt)) / 3.5e6 * ps,
                             double(tk2(e.gap_ticks)) / 3.5e6 * ps, double(tk2(e.gap_max)) / 3.5e6,
                             double(tk2(e.unpublished)) * ps);
                    if (VideoCore::g_lane_on && VideoCore::g_lane_counters) {
                        auto& c = *VideoCore::g_lane_counters;
                        std::array<double, 16> v{};
                        for (u32 i = 0; i < 16; ++i) {
                            v[i] = double(c[i]) * ps;
                            c[i] = 0;
                        }
                        std::array<double, 7> os{};
                        for (u32 i = 0; i < 7; ++i) {
                            os[i] = double((*VideoCore::g_lane_open_src)[i]) * ps;
                            (*VideoCore::g_lane_open_src)[i] = 0;
                        }
                        LOG_INFO(Render,
                                 "ASC_LANE: submits {:.0f}/s (deferred behind the open batch {:.0f}, "
                                 "main flushes for a deferred cut {:.0f}, waiting on an unsubmitted "
                                 "main tick {:.0f}) | label cuts {:.0f}/s | dispatches on the lane "
                                 "{:.0f}/s, early records kept on main {:.0f}/s (unrouted {:.0f}) | "
                                 "packets lane {:.0f} main {:.0f} | rule A waits in flight {:.0f}/s, "
                                 "on the open batch {:.0f}/s | main flushes waiting on the lane "
                                 "{:.0f}/s | fault pre-cuts {:.0f}/s | p1e: pinned segments cut "
                                 "{:.0f}/s, WAR / GDS capped {:.0f}/s, open-batch waits by source "
                                 "exact {:.0f} window table (huge) {:.0f} window confirmed {:.0f} GDS "
                                 "{:.0f} carry {:.0f} big list {:.0f} lost {:.0f} | big main notes "
                                 "{:.0f}/s, lost unfinished {:.0f}/s",
                                 v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[12], v[7], v[8], v[9],
                                 v[10], v[11], v[13], v[14], v[15], os[0], os[1], os[2], os[3],
                                 os[4], os[5], os[6],
                                 double(VideoCore::g_lc[19].exchange(0)) * ps,
                                 double(VideoCore::g_lc_big_lost.exchange(0)) * ps);
                    }
                    if (VideoCore::g_lane_census) {
                        auto& lc = VideoCore::g_lc;
                        std::array<double, 20> v{};
                        for (u32 i = 0; i < 20; ++i) {
                            v[i] = double(lc[i].exchange(0)) * ps;
                        }
                        LOG_INFO(Render,
                                 "LANE census (n1): early records {:.0f}/s, with an open-batch main "
                                 "producer {:.0f}/s, in-flight only {:.0f}/s | early reads {:.0f}/s: "
                                 "newest producer main write open {:.0f} in flight {:.0f}, main upload "
                                 "open {:.0f} in flight {:.0f}, early record {:.0f} | early writes "
                                 "{:.0f}/s over main open {:.0f} in flight {:.0f} | in-order reads "
                                 "{:.0f}/s over unfinished early writes: coherent skip {:.0f} full {:.0f} "
                                 "lean {:.0f} | notes {:.0f}/s, truncated {:.0f}, off the back {:.0f}",
                                 v[18], v[16], v[17], v[0], v[1], v[2], v[3], v[4], v[15], v[5], v[6],
                                 v[7], v[8], v[9], v[10], v[11], v[14], v[12], v[13]);
                    }
                }
                if (g_xq) {
                    std::string ws, as;
                    static constexpr const char* wn[] = {"gfx", "same", "otherASC", "none", "fwd"};
                    for (u32 q = 0; q < XqQueues; ++q) {
                        std::string row;
                        u64 tot = 0;
                        for (u32 w = 0; w < 5; ++w) {
                            const u64 n = tk2(g_xq_c.waits[q][w]);
                            tot += n;
                            if (n) {
                                row += fmt::format(" {} {:.0f}/s", wn[w], double(n) * ps);
                            }
                        }
                        if (tot) {
                            ws += fmt::format(" | {}:{}", q == 0 ? std::string("gfx") : fmt::format("asc{}", q), row);
                        }
                        const u64 d = tk2(g_xq_c.disp[q]);
                        if (q != 0 && d) {
                            const u64 di = tk2(g_xq_c.disp_img[q]), s = tk2(g_xq_c.segs[q]),
                                      si = tk2(g_xq_c.segs_img[q]), sx = tk2(g_xq_c.segs_xwait[q]),
                                      ub = tk2(g_xq_c.upload_bytes[q]);
                            as += fmt::format(
                                " | asc{} (label {:#x}): dispatches {:.0f}/s, binding images {:.1f}% | "
                                "segments {:.0f}/s, with an image dispatch {:.1f}%, with a wait on a "
                                "gfx / CPU label {:.1f}% | uploads for its dispatches {:.1f} MB/s",
                                q, g_xq_c.label_addr[q].load(), double(d) * ps,
                                100.0 * double(di) / double(d), double(s) * ps,
                                s ? 100.0 * double(si) / double(s) : 0.0,
                                s ? 100.0 * double(sx) / double(s) : 0.0, double(ub) * ps / 1e6);
                        }
                    }
                    LOG_INFO(Render, "XQ census waits (waiter: writer){}", ws);
                    LOG_INFO(Render, "XQ census ASC queues{}", as);
                }
                LOG_INFO(Render,
                         "POST_ORDER: idle tails posted {:.0f}/s | PFP_SYNC_ME posted {:.0f}/s | "
                         "ZPASS_DONE posted {:.0f}/s (buffer-guarded {:.0f}/s) | waits forwarded "
                         "{:.0f}/s (not forwarded: unknown value {} [other {} fence {} wdata {} "
                         "dma {} asc {} zpass {}] not at label start {} value fails {} newer {}) | "
                         "wait barriers (mode {}) POST_ORDER {:.0f}/s FENCE_FORWARD "
                         "{:.0f}/s | A2=2 forwarded-value compares {} BAD {} | front faults read: "
                         "span {} label {} logged {} other {} | invalidate: span {} label {} logged "
                         "{} other {}",
                         double(tk2(g_po.idle_posted)) * ps, double(tk2(g_po.pfp_posted)) * ps,
                         double(tk2(g_po.zpass_posted)) * ps, double(tk2(g_po.zpass_guarded)) * ps,
                         double(tk2(g_po.forwarded)) * ps, tk2(g_fwd_why[0]),
                         tk2(g_fwd_unknown_src[0]), tk2(g_fwd_unknown_src[1]),
                         tk2(g_fwd_unknown_src[2]), tk2(g_fwd_unknown_src[3]),
                         tk2(g_fwd_unknown_src[4]), tk2(g_fwd_unknown_src[5]), tk2(g_fwd_why[1]),
                         tk2(g_fwd_why[2]), tk2(g_fwd_why[3]), g_wait_barrier,
                         double(tk2(g_wait_barriers[0])) * ps, double(tk2(g_wait_barriers[1])) * ps,
                         tk2(g_po.fwd_cmp), g_po.fwd_bad.load(),
                         tk2(g_po.fault[0]), tk2(g_po.fault[1]), tk2(g_po.fault[2]),
                         tk2(g_po.fault[3]), tk2(g_po.fault[4]), tk2(g_po.fault[5]),
                         tk2(g_po.fault[6]), tk2(g_po.fault[7]));
                LOG_INFO(Render,
                         "RPTR_BACKING ({}): read-pointer writes {:.0f}/s, backing fallbacks {}, "
                         "changed under the front {} (gate 0), download words skipped {}",
                         g_rptr_backing ? 1 : 0, double(tk2(g_rptr_writes)) * ps,
                         tk2(g_rptr_fallbacks), g_rptr_changed.load(),
                         Core::Memory::Instance()->download_exclusion_hits.load());
                LOG_INFO(Render,
                         "Posted gfx WRITE_DATA {:.0f}/s (kept synchronous by the buffer guard "
                         "{:.0f}/s) | posted flips {:.0f}/s",
                         double(g_pwf.wdata_posted.exchange(0)) * ps,
                         double(g_pwf.wdata_guarded.exchange(0)) * ps,
                         double(g_pwf.flips_posted.exchange(0)) * ps);
            }
            LOG_INFO(Render,
                     "Lean L3 pre-delta: folds {:.0f}/s, register records folded {:.0f}/s | A2=2 "
                     "pre-delta words {} BAD {} | publishes on back wait {:.0f}/s",
                     double(x.pre_folds.exchange(0)) * ps, double(x.pre_records.exchange(0)) * ps,
                     x.pre_cmp.exchange(0), x.pre_bad.load(), double(x.wait_pubs.exchange(0)) * ps);
            std::array<u32, 32> ord{};
            for (u32 i = 0; i < 32; ++i) {
                ord[i] = i;
            }
            std::sort(ord.begin(), ord.end(), [&](u32 a, u32 b) { return rf[a] > rf[b]; });
            u64 rtot = 0;
            for (const u64 v : rf) {
                rtot += v;
            }
            std::string top;
            for (u32 k = 0; k < 8 && rf[ord[k]] != 0; ++k) {
                top += fmt::format(" {}={:.1f}%", Vulkan::A2::RefusalName(ord[k]),
                                   100.0 * double(rf[ord[k]]) / double(std::max<u64>(rtot, 1)));
            }
            const u64 st_ = x.stretches.exchange(0), tk = x.taken.exchange(0),
                      wk = x.walked.exchange(0);
            // The back's counters are monotonic (the per-second line diffs them too).
            static u64 prev_pub = 0, prev_pd = 0, prev_bd = 0;
            const u64 pub_now = x.published.load(), pd_now = x.pub_draws.load(),
                      bd_now = x.back_draws.load();
            const u64 d_pub = pub_now - prev_pub, d_pd = pd_now - prev_pd, d_bd = bd_now - prev_bd;
            prev_pub = pub_now;
            prev_pd = pd_now;
            prev_bd = bd_now;
            LOG_INFO(Render,
                     "Lean L3: worker draws walked {:.0f}/s taken {:.0f}/s ({:.1f}%) in {:.0f} "
                     "stretches/s ({:.1f} draws each; cut by publish {:.0f}/s, cap hits {}; material "
                     "resolves {:.0f}/s) | back: "
                     "published {:.0f}/s ({:.0f} draws/s), rejected occlusion {} buffer-gen {} "
                     "material {} target {} | A2=2 would {:.0f}/s, render key {} BAD {}, delta words {} "
                     "BAD {}, slot words filed {} | back old-path draw records in job "
                     "channels {:.0f}/s, seq skips {} | refusals (cumulative, of {}):{}",
                     double(wk) * ps, double(tk) * ps,
                     100.0 * double(tk) / double(std::max<u64>(wk, 1)), double(st_) * ps, double(tk) / double(std::max<u64>(st_, 1)),
                     double(x.cut_pub.exchange(0)) * ps, x.cap.exchange(0),
                     double(x.mat_resolves.exchange(0)) * ps,
                     double(d_pub) * ps, double(d_pd) * ps,
                     x.rej_occl.exchange(0), x.rej_gen.exchange(0), x.rej_mat.exchange(0),
                     x.rej_rt.exchange(0), double(x.would.exchange(0)) * ps, x.rk_cmp.exchange(0),
                     x.rk_bad.load(), x.delta_cmp.exchange(0), x.delta_bad.load(),
                     x.expect_filed.exchange(0),
                     double(d_bd) * ps, x.seq_skips.exchange(0), rtot, top);
        }
        s.l2_taken = s.l2_skipped = s.head_draws = s.back_records = s.back_wait_ticks = 0;
        g_lean_l2.fault_waits = 0;
        g_lean_l2.wait_ticks = 0;
    }
    LOG_INFO(Render,
             "Lean (mode {}) L2 counters: parity breaks, buffer UNCHANGED {} (gate 0) / buffer changed {} | "
             "lag-excluded jobs (buffer changed) {} | BAD (unchanged buffers) by "
             "first-difference kind:{} | pipeline-miss sub-reasons (cumulative; 1 unsupported stages, 2 "
             "no program, 3 precompiled first use, 4 no permutation, 5 no pipeline, 6 pending):{} | "
             "front draws {} NativeRec member-path {} ({:.1f}%)",
             g_lean, c.parity.load(), c.parity_changed.load(), c.lag.load(), kinds, misses,
             c.front_draws.load(),
             c.front_member.load(),
             c.front_draws ? 100.0 * double(c.front_member) / double(c.front_draws) : 0.0);
    c.parity = c.parity_changed = c.lag = c.front_draws = c.front_member = 0;
    std::string ops;
    for (u32 i = 0; i < 256; ++i) {
        if (c.stop_ops[i] != 0) {
            ops += fmt::format(" op{:#04x}={}", i, c.stop_ops[i].load());
        }
    }
    const double pk = double(c.pkts.load()), dr = double(c.draws.load());
    LOG_INFO(Render,
             "Lean (mode {}): window {:.1f}s | DCBs {} eligible {} skipped (slot busy) {} | chain "
             "{:.2f} ms/s, verified {} DCBs: windows bad {} whole-file bad {} extras bad {} reseeds "
             "{} (extras by kind: db {} cb {} cs {} epoch {} iaa {}) | worker spans compared {} BAD {} "
             "(events {}, no mark {}) | dispatch-run bumps in spans {} | coverage: packets "
             "{:.1f}% draws {:.1f}% | epoch bumps inside worker spans {} (first opcode {:#x}) | stops (by Lean::Stop "
             "value):{} | other-stop opcodes:{} | faults: pre-check stops SRT protected {} "
             "unmapped {} V# protected {} unmapped {} walker not understood {} flat overflow {} code "
             "protected {} code unmapped / no header {}; "
             "FALLBACK faults (must be 0) {} first {:#x} {:#x} | worker shared-lock takes per draw {:.2f}",
             g_lean, sec, c.dcbs.load(), c.eligible.load(), c.skipped.load(),
             double(s.chain_cyc) / 3.5e6 / std::max(sec, 1e-9), s.chain_cmp, s.chain_bad_win,
             s.chain_bad_full, s.chain_bad_extra, s.reseeds, s.chain_bad_db, s.chain_bad_cb,
             s.chain_bad_cs, s.chain_bad_epoch, s.chain_bad_iaa, c.cmp.load(), c.bad.load(),
             c.events.load(), c.no_mark.load(), c.run_bumps.load(), pk > 0 ? 100.0 * double(c.pkts_span) / pk : 0.0,
             dr > 0 ? 100.0 * double(c.draws_span) / dr : 0.0, c.epoch_bumps.load(),
             c.epoch_bump_op.load() & 0xff, stops, ops,
             g_lean_fault_stats.precheck[0].load(), g_lean_fault_stats.precheck[1].load(),
             g_lean_fault_stats.precheck[2].load(), g_lean_fault_stats.precheck[3].load(),
             g_lean_fault_stats.precheck[4].load(), g_lean_fault_stats.precheck[5].load(),
             g_lean_fault_stats.precheck[6].load(), g_lean_fault_stats.precheck[7].load(),
             g_lean_fault_stats.fallback.load(), g_lean_fault_stats.first[0].load(),
             g_lean_fault_stats.first[1].load(),
             g_lean_fault_stats.lock_draws.load()
                 ? double(g_lean_fault_stats.lock_takes.load()) / double(g_lean_fault_stats.lock_draws.load())
                 : 0.0);
    for (auto& x : c.stops) {
        x = 0;
    }
    for (auto& x : c.stop_ops) {
        x = 0;
    }
    c.dcbs = c.eligible = c.skipped = c.cmp = c.bad = c.events = c.pkts = c.pkts_span = c.draws =
        c.draws_span = c.no_mark = c.epoch_bumps = c.run_bumps = 0;
    s.chain_cyc = s.chain_cmp = s.chain_bad_win = s.chain_bad_full = s.chain_bad_extra = s.reseeds = 0;
    s.chain_bad_db = s.chain_bad_cb = s.chain_bad_cs = s.chain_bad_epoch = s.chain_bad_iaa = 0;
    s.win_tsc = now;
}

const bool g_native_timing_rec = [] {
    const char* e = std::getenv("SHADPS4_NATIVE_TIMING");
    return e && e[0] == '1';
}();

// ---- CP front/back split records (cp_split.h). Built on the front, read on the back. ----


/// SplitDrawRecord::verify. Full: the front is parked on this record (depth 0, or the periodic
/// synchronous run-ahead draw), so the back compares against the front's live state. Snapshot:
/// a posted draw; a register-window copy follows the stage blocks.
constexpr u32 DrawVerifyFull = 1u << 0;
constexpr u32 DrawVerifySnapshot = 1u << 1;

/// Run-ahead VERIFY: the register ranges the back's draw path reads (inventory 4.3), copied by
/// the front into each posted draw. Gfx SH (ps_program .. cs_program), all context registers,
/// and uconfig up to the VGT draw state (primitive type, num_indices, num_instances, TF base).
/// cs registers never go through regs (they live in mapped_queues[].cs_state).
struct RegWindow {
    u32 begin;
    u32 end;
};
constexpr u32 WordOf(size_t byte_offset) {
    return static_cast<u32>(byte_offset / sizeof(u32));
}
constexpr std::array<RegWindow, 3> VerifyWindows{{
    {WordOf(offsetof(Regs, ps_program)), WordOf(offsetof(Regs, cs_program))},
    {Regs::ContextRegWordOffset, Regs::ContextRegWordOffset + 0x400},
    {Regs::UconfigRegWordOffset,
     WordOf(offsetof(Regs, vgt_tf_memory_base) + sizeof(Regs::vgt_tf_memory_base))},
}};
constexpr u32 VerifyWindowDwords() {
    u32 dwords = 0;
    for (const auto& window : VerifyWindows) {
        dwords += window.end - window.begin;
    }
    return dwords;
}
/// Extents (8 CB + 1 DB) then the windows, padded to 8 bytes.
constexpr u32 VerifySnapshotBytes =
    ((NUM_COLOR_BUFFERS + 1 + VerifyWindowDwords()) * sizeof(u32) + 7) & ~u32{7};
static_assert(VerifyWindows[0].begin < VerifyWindows[0].end &&
              VerifyWindows[2].end <= Regs::NumRegs);

/// A dispatch (gfx queue, or ASC with POST_ASC). `cs` is the front's cs registers at the packet
/// (GetCsRegs() depends on the front's curr_qid; inventory section 0.3). Followed by at most one
/// stage block, then (VERIFY snapshot only) the register-window copy.
struct SplitDispatchRecord {
    ComputeProgram cs;
    const Vulkan::ComputePipeline* pipeline;
    const char* marker;
    const void* cmd_address;
    VAddr address;
    u32 offset;
    u32 size;
    u32 is_indirect;
    u32 stage_count;
    u32 queue;  ///< mapped_queues index: 0 = gfx, vqid + 1 = ASC.
    u32 verify; ///< DrawVerifyFull / DrawVerifySnapshot (VERIFY only), else 0.
    u64 run;    ///< SHADPS4_GUEST_DISPATCH_ORDER guest run id, else 0.
};
static_assert(sizeof(SplitDispatchRecord) % 8 == 0);


/// Run-ahead VERIFY, front: the extents and register windows the back must match (see
/// VerifyWindows). `out` is 8-aligned (stage blocks end 8-aligned).
void WriteVerifySnapshot(u8* out, const Regs& regs,
                         const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& cb_extent,
                         const CbDbExtent& db_extent) {
    auto* words = reinterpret_cast<u32*>(out);
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        *words++ = cb_extent[cb].raw;
    }
    *words++ = db_extent.raw;
    for (const auto& window : VerifyWindows) {
        const u32 count = window.end - window.begin;
        std::memcpy(words, &regs.reg_array[window.begin], size_t{count} * sizeof(u32));
        words += count;
    }
}

/// VERIFY: the Info still holds what the front's lookup left there. Only valid while the front
/// is parked on this very record (VerifyFull): at run-ahead the front's later lookups rewrite it.
void VerifyStages(const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages, u64 seq) {
    for (const auto& bound : stages) {
        if (!bound.info) {
            continue;
        }
        const auto& info = *bound.info;
        ASSERT_MSG(bound.pgm_base == info.pgm_base &&
                       std::ranges::equal(bound.user_data, info.user_data) &&
                       std::ranges::equal(bound.flattened_ud_buf, info.flattened_ud_buf),
                   "CpSplit VERIFY: stage {} data differs from its Info at record {}",
                   static_cast<u32>(info.l_stage), seq);
    }
}

/// VERIFY with POST_ASC, on a synchronous dispatch record (the front is parked on it, and the back
/// has executed everything published before it): re-runs the front's compute lookup on the back
/// and compares it with what the record carries. A difference means the front's lookup read
/// inputs (user-data / SRT tables or shader code in guest memory) that earlier posted work had not
/// written yet: the run-ahead read hazard. The guest CPU writing those tables in between also
/// shows up here, so the caller logs instead of asserting. Reads the front's pipeline cache and
/// cs_state (GetCsRegs), which is sound only while the front is parked.
bool DispatchRelookupMatches(Vulkan::Rasterizer& rasterizer,
                             const Vulkan::ComputePipeline* carried_pipeline,
                             const Shader::BoundStage& carried) {
    const Vulkan::ComputePipeline* pipeline = rasterizer.PrepareDispatchFront();
    if (pipeline != carried_pipeline) {
        return false;
    }
    if (!pipeline) {
        return true;
    }
    const Shader::Info* info = pipeline->GetStages()[0];
    if (!info) {
        return carried.info == nullptr;
    }
    return carried.info == info && carried.pgm_base == info->pgm_base &&
           std::ranges::equal(carried.user_data, info->user_data) &&
           std::ranges::equal(carried.flattened_ud_buf, info->flattened_ud_buf);
}

/// What DrawRelookup found. `what` is null when the re-lookup agrees with the record.
struct DrawRelookupDiff {
    const char* what{};
    u32 stage{};       ///< LogicalStage, for the per-stage differences.
    u32 index{};       ///< First differing dword (user data / flat buffer).
    u32 carried_dw{};  ///< Dword count the record carries.
    u32 relooked_dw{}; ///< Dword count the re-lookup produced.
};

u32 FirstDifferingDword(std::span<const u32> a, std::span<const u32> b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return static_cast<u32>(i);
}

/// The draw equivalent of DispatchRelookupMatches. VERIFY at run-ahead, on the periodic
/// synchronous draw record: the front is parked on this record and the back has executed every
/// record published before it, so guest memory now holds what that earlier posted work writes (a
/// GPU-side write is marked GPU-modified, and reading it here takes the ordinary back-thread
/// read fault and download, as the unsplit path would at this stream position). The front's
/// lookup for this draw ran BEFORE its Sync (EmitDraw), while that work could still be
/// outstanding; so a stale read of user-data / SRT tables, fetch V# tables or shader code there
/// shows up as a difference here. The guest CPU writing those tables between the two lookups
/// shows up too, so the caller logs instead of asserting.
///
/// `regs` is the FRONT's register file (what GetGraphicsPipeline reads); VerifyMirror has just
/// proven it equal to the back's mirror. Re-runs the pure classifier and the pipeline lookup only
/// (Rasterizer::RelookupDrawFront): not the hazard note or the empty-draw counters (the skip
/// decision is a pure function of the registers, so it cannot differ). Side effects, sound only
/// while the front is parked (as for DispatchRelookupMatches): it rewrites the pipeline cache's
/// scratch selection (key, infos, modules, fetch shader) and the stages' Info user data / flat
/// buffer / pgm_base, all of which the front's next lookup rewrites and which the back reads
/// only from records; it may bump prepared-pipeline LRU/stat counters or seed a prepared entry
/// (guarded memo: performance-only), and on a difference it may compile a pipeline (the cache
/// is keyed by content). The Info comparison must therefore run after VerifyStages.
DrawRelookupDiff DrawRelookup(Vulkan::Rasterizer& rasterizer, const Regs& regs,
                              const Vulkan::Rasterizer::SplitDraw& carried_draw,
                              const std::array<Shader::BoundStage, Shader::MaxStageTypes>& carried) {
    using Vulkan::Rasterizer;
    if (Rasterizer::ClassifyDraw(regs) != carried_draw.draw_class) {
        return {.what = "draw class"};
    }
    if (carried_draw.draw_class != Rasterizer::DrawClass::Draw || carried_draw.skip) {
        // The front ran no lookup (EmitDraw carried no stages).
        return {};
    }
    const Vulkan::GraphicsPipeline* pipeline = rasterizer.RelookupDrawFront();
    if (pipeline != carried_draw.pipeline) {
        return {.what = "pipeline"};
    }
    if (!pipeline) {
        return {};
    }
    const auto infos = pipeline->GetStages();
    for (u32 i = 0; i < infos.size() && i < carried.size(); ++i) {
        const Shader::Info* info = infos[i];
        const Shader::BoundStage& bound = carried[i];
        if (bound.info != info) {
            return {.what = "stage Info", .stage = i};
        }
        if (!info) {
            continue;
        }
        if (bound.pgm_base != info->pgm_base) {
            return {.what = "pgm_base", .stage = i};
        }
        const std::span<const u32> ud = info->user_data;
        if (!std::ranges::equal(bound.user_data, ud)) {
            return {.what = "user data",
                    .stage = i,
                    .index = FirstDifferingDword(bound.user_data, ud),
                    .carried_dw = static_cast<u32>(bound.user_data.size()),
                    .relooked_dw = static_cast<u32>(ud.size())};
        }
        const std::span<const u32> flat{info->flattened_ud_buf.data(),
                                        info->flattened_ud_buf.size()};
        if (!std::ranges::equal(bound.flattened_ud_buf, flat)) {
            return {.what = "flat buffer",
                    .stage = i,
                    .index = FirstDifferingDword(bound.flattened_ud_buf, flat),
                    .carried_dw = static_cast<u32>(bound.flattened_ud_buf.size()),
                    .relooked_dw = static_cast<u32>(flat.size())};
        }
    }
    return {};
}

/// A NOP whose payload has a back-side effect (flip IRQ, guest markers). Other NOPs (CB/DB
/// extent hints and the like) are front-only.
bool NopHasBackEffect(const PM4Header* header, bool guest_markers_enabled) {
    const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
    if (nop->header.count.Value() == 0) {
        return false;
    }
    switch (nop->data_block[0]) {
    case PM4CmdNop::PayloadType::PatchedFlip:
        return true;
    case PM4CmdNop::PayloadType::DebugMarkerPush:
    case PM4CmdNop::PayloadType::DebugColorMarkerPush:
    case PM4CmdNop::PayloadType::DebugMarkerPop:
        return guest_markers_enabled;
    default:
        return false;
    }
}
} // namespace

template <typename F>
void Liverpool::OnBack(F&& fn, CpSplit::DrainReason reason) {
    if (split_enabled) [[unlikely]] {
        // Synchronous at every depth: fn and what it captures live on the front's stack.
        split->CallSync(fn, reason);
    } else {
        fn();
    }
}

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
// SHADPS4_ASC_EARLY: records committed outside ASC parsing (ProcessLoop, other queues) are never
// tagged as ASC records, so the tag is dropped around every ASC yield.
#define YIELD_ASC(id)                                                                              \
    if (split_enabled) {                                                                           \
        split->asc_tag = 0;                                                                        \
        if (g_asc_early && (g_asc_service & 1)) {                                                  \
            split->PublishNow(); /* m2: early service sees this ASC submission's records now */       \
        }                                                                                          \
    }                                                                                              \
    YIELD(acb_task_name[id]);                                                                      \
    if (split_enabled && g_asc_early) {                                                            \
        split->asc_tag = 1;                                                                        \
    }

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

/// Packets that neither write guest memory from the CP nor wait on it: register state, draws,
/// dispatches, cache-control events and nested IBs (which bump at their own start). Any other
/// packet closes the upload-deferral window (submit_epoch.h). Conservative on purpose: an extra
/// bump only uploads a deferred page earlier, a missing one could upload it too late.
static constexpr bool KeepsSubmitEpoch(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::Nop:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetQueueReg:
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetPredication:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
    // GPU copy/fill; its CPU fast paths clear deferral for the bytes they write themselves
    // (BufferCache::FillBuffer/CopyBuffer). Measured ~3,100 per frame in the forest.
    case PM4ItOpcode::DmaData:
    case PM4ItOpcode::EventWrite:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::PfpSyncMe:
    case PM4ItOpcode::IndirectBuffer:
        return true;
    default:
        return false;
    }
}

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    if (CpSplit::Enabled()) {
        split_enabled = true;
        split_verify = CpSplit::Verify();
        split_run_ahead = CpSplit::RunAhead();
        split_post_dma = split_run_ahead && CpSplit::PostDma();
        if (split_post_dma && SplitHazard::Enabled()) {
            // SplitHazard's page map is written by NoteWrite (DMA CPU paths, back) and read by
            // NoteDraw (front): only race-free while DMAs are synchronous.
            LOG_WARNING(Render, "CpSplit: POST_DMA off, SHADPS4_SPLIT_HAZARD_STATS is on");
            split_post_dma = false;
        }
        split_post_asc = split_run_ahead ? CpSplit::PostAscMask() : 0u;
        if (split_post_asc != 0 && SplitHazard::Enabled()) {
            // Same race as POST_DMA: posted ASC DMAs note writes on the back.
            LOG_WARNING(Render, "CpSplit: POST_ASC off, SHADPS4_SPLIT_HAZARD_STATS is on");
            split_post_asc = 0;
        }
        if ((split_post_asc & CpSplit::PostAscWriteData) && g_cp_wait.enabled) {
            // ExecuteGfxPacket's WRITE_DATA notes the address in g_cp_wait on the back, which the
            // front reads: only race-free while WRITE_DATA is synchronous.
            LOG_WARNING(Render, "CpSplit: POST_ASC WRITE_DATA off, SHADPS4_CP_WAIT_STATS is on");
            split_post_asc &= ~u32{CpSplit::PostAscWriteData};
        }
        split_post_task_done = split_run_ahead && CpSplit::PostTaskDone();
        split_task_guard = split_post_task_done ? CpSplit::TaskGuard() : 0u;
        if (CpSplit::TaskGuard() != 0 && !split_post_task_done) {
            LOG_WARNING(Render, "CpSplit: TASK_GUARD ignored without run-ahead POST_TASK_DONE");
        }
        split_post_fence = split_task_guard != 0 && CpSplit::PostFence();
        if (CpSplit::PostFence() && split_task_guard == 0) {
            LOG_WARNING(Render, "CpSplit: POST_FENCE ignored without TASK_GUARD");
        }
        if (split_post_fence && (SplitHazard::Enabled() || g_cp_wait.enabled)) {
            // The fence bodies note the label in SplitHazard / g_cp_wait on the back, which the
            // front reads: only race-free while fences are synchronous.
            LOG_WARNING(Render, "CpSplit: POST_FENCE off, SPLIT_HAZARD_STATS or CP_WAIT_STATS on");
            split_post_fence = false;
        }
        split_disp_guard = CpSplit::DispGuard();
        split_front_derive = CpSplit::FrontDerive();
        if (split_front_derive != 0) {
            front_vi = std::make_unique<CpSplit::VertexInputsDerived>();
        }
        split_verify_every = split_run_ahead ? CpSplit::VerifyEvery() : 1u;
        split = std::make_unique<CpSplit::Channel>();
        if (g_asc_early && split_run_ahead) {
            split->EnableEarly();
            LOG_INFO(Render, "SHADPS4_ASC_EARLY: ASC records run ahead of the graphics backlog");
        }
        mirror = std::make_unique<CpSplit::BackMirror>();
        std::memcpy(&mirror->regs, &regs, sizeof(Regs));
        mirror->cb_extent = last_cb_extent;
        mirror->db_extent = last_db_extent;
        mirror->ctx_epoch = context_change_epoch;
        back_regs = &mirror->regs;
        back_cb_extent = mirror->cb_extent.data();
        back_db_extent = &mirror->db_extent;
        back_ctx_epoch = &mirror->ctx_epoch;
        if (split_run_ahead) {
            LOG_INFO(Render, "CpSplit: on (run-ahead, depth {}{}{}{}){}", CpSplit::Depth(),
                     split_post_dma ? ", POST_DMA" : "",
                     split_post_asc ? fmt::format(", POST_ASC mask {:#x}", split_post_asc)
                                    : std::string{},
                     split_post_task_done
                         ? fmt::format(", POST_TASK_DONE{}",
                                       split_task_guard
                                           ? fmt::format(" TASK_GUARD {}{}", split_task_guard,
                                                         split_post_fence ? ", POST_FENCE" : "")
                                           : std::string{})
                         : std::string{},
                     split_verify ? fmt::format(", VERIFY every {}", split_verify_every)
                                  : std::string{});
            if (!split_disp_guard) {
                LOG_INFO(Render, "CpSplit: DISP_GUARD 0 (no disp_after_pkt drain)");
            }
        } else {
            LOG_INFO(Render, "CpSplit: on (depth 0){}", split_verify ? ", VERIFY" : "");
        }
        if (split_front_derive != 0) {
            const char* detail = split_front_derive == 2 ? ", verify"
                                 : CpSplit::FrontDeriveVertexInputSkip()
                                     ? ", setVertexInputEXT skip"
                                     : "";
            LOG_INFO(Render, "CpSplit: FRONT_DERIVE {} (vertex inputs{})", split_front_derive,
                     detail);
        }
    }
    if (AmdGpu::NativeRec::g_segb != 0) {
        LOG_INFO(Render, "NativeSegB: on (front segment builder, census + verify only; pt {})",
                 Shader::g_native_pt);
    }
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // CP split: commands need the caches and the scheduler, so only the back (gpu_id) runs them,
    // between records and while idle. The front's per-packet calls return here.
    if (split_enabled && !IsGpuThread()) [[unlikely]] {
        return;
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        u32 prev_stream = 0;
        u64 lane_wait = 0;
        if (rasterizer) {
            prev_stream = rasterizer->GetScheduler().ActiveStream();
            lane_wait = rasterizer->GetScheduler().LaneWaitMain();
            rasterizer->GetScheduler().SwitchStream(0);
        }
        callback();
        if (rasterizer && prev_stream != 0 && rasterizer->GetScheduler().HasLane()) {
            rasterizer->GetScheduler().SwitchStream(prev_stream); // SHADPS4_ASC_LANE
            rasterizer->GetScheduler().LaneNeedMain(lane_wait, 4);
        }
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
    // Bill page faults taken on this thread separately: this thread's activity IS the frame.
    Common::FaultMonitor::MarkCpThread();
    Common::ApplyCommandProcessorPlacement();

    if (split_enabled) {
        // CP split: this thread keeps every CP identity (name, gpu_id, t_is_cp_thread, placement,
        // scheduler producer, guest command queue) and becomes the back. The PM4 parser moves to
        // a new front thread, started only now so gpu_id is set before it can send a command.
        front_thread = std::jthread{std::bind_front(&Liverpool::FrontProcess, this)};
        BackLoop(stoken);
        return;
    }
    ProcessLoop(stoken);
}

void Liverpool::FrontProcess(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandFront");
    CpSplit::t_is_front = true;
    Common::ApplyHotThreadPlacement(Common::HotThread::Front);
    // SPLIT-TODO: placement. ApplyCommandProcessorPlacement pins the back; the front wants its
    // own P-core (with the recording worker and Game:Main, 4 hot threads on 6 P-cores).
    ProcessLoop(stoken);
    split->FrontExited();
}

namespace {
bool A2NoopDma(const CpSplit::RecordHeader& record); // l4a6: used by the ASC early service
} // namespace

void Liverpool::BackLoop(std::stop_token stoken) {
    std::stop_callback wake_on_stop{stoken, [this] { split->WakeBack(); }};
    if (g_lean == 2) {
        // SHADPS4_LEAN=2: while this thread waits for the pipeline cache's lock it runs guest
        // commands (a lean worker faulting under that lock waits for one).
        t_lock_service_ctx = this;
        t_lock_service = [](void* p) { static_cast<Liverpool*>(p)->ProcessCommands(); };
    }
    bool stop_forwarded = false;
    // SHADPS4_MTB_P1 (P1d, notes/parrec-build-2026-10-08.md sec. 14): the translating context is
    // installed before the first record that can record a command (a rasterizer is bound).
    bool mtb_install = Vulkan::MtbP1Requested();
    bool mtb_p3 = false;
    // SHADPS4_ASC_EARLY: seqs of records already run ahead, in ring order (back only).
    std::deque<u64> early_done;
    const bool asc_early = g_asc_early && split->early_items != nullptr && g_mtb_p3 == 0;
    u64 early_last_tsc = 0;
    bool early_in_service = false; // q14: a command run inside the service never re-enters it
    auto early_service = [&] {
        if (split->early_head.load(std::memory_order_relaxed) ==
            split->early_tail.load(std::memory_order_acquire)) {
            return; // q11b: nothing queued (the common case between job records)
        }
        if (early_in_service) [[unlikely]] {
            return;
        }
        early_in_service = true;
        SCOPE_EXIT {
            early_in_service = false;
        };
        VideoCore::LbScope lb_scope{0}; // q11a
        VideoCore::g_lb_in_service = true;
        ++VideoCore::g_lb[1];
        auto& ch = *split;
        u64 head = ch.early_head.load(std::memory_order_relaxed);
        const u64 tail = ch.early_tail.load(std::memory_order_acquire);
        {
            // l4f: the time since the previous service, when something is waiting now
            const u64 now = __rdtsc();
            if (head != tail && early_last_tsc != 0) {
                const u64 gap = now - early_last_tsc;
                g_asc_early_stats.gap_ticks.fetch_add(gap, std::memory_order_relaxed);
                if (gap > g_asc_early_stats.gap_max.load(std::memory_order_relaxed)) {
                    g_asc_early_stats.gap_max.store(gap, std::memory_order_relaxed);
                }
            }
        }
        const u64 published = ch.PublishedEnd();
        const u64 read = ch.ReadPos();
        // q24: SHADPS4_CHAIN_TIMING - this run's dispatches timed on the GPU (lane off).
        Vulkan::Scheduler* const q24_sch =
            rasterizer && !VideoCore::g_lane_on && rasterizer->GetScheduler().ChainTiming()
                ? &rasterizer->GetScheduler()
                : nullptr;
        u32 q24_disp = 0;
        bool q24_timed = false;
        SCOPE_EXIT {
            if (q24_timed) {
                q24_sch->ChainTimerEnd(q24_disp);
            }
        };
        while (head != tail) {
            const auto& it = ch.early_items[head & (CpSplit::Channel::EarlySlots - 1)];
            if (it.pos < read) {
                ++head; // already reached in ring order
                continue;
            }
            if (it.pos >= published) {
                g_asc_early_stats.unpublished.fetch_add(1, std::memory_order_relaxed);
                break; // committed, not published yet
            }
            using RT = CpSplit::RecordType;
            if (it.type == RT::Dispatch || it.type == RT::Packet) {
                u64 p = it.pos;
                const auto* r = ch.RecordAtPos(p);
                if (it.type == RT::Dispatch &&
                    reinterpret_cast<const SplitDispatchRecord*>(r + 1)->verify != 0) {
                    // l4a6 (review #4): VerifyMirror assumes the front is parked on it.
                    g_asc_early_stats.blocked.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                // l4a3: per-binding validity for this early record (NATIVE_COHERENT's
                // submission-start upload for it has not run yet).
                const u64 up0 =
                    rasterizer ? rasterizer->GetBufferCache().SyncUploadBytesTotal() : 0;
                const u64 chk0 = VideoCore::t_forced_binding_checks;
                VideoCore::t_force_binding_sync = true;
                VideoCore::t_early_pos = it.pos;
                if (VideoCore::g_lane_census && rasterizer) [[unlikely]] {
                    auto* ms = rasterizer->GetScheduler().GetMasterSemaphore();
                    ms->Refresh();
                    VideoCore::g_lc_done = ms->KnownGpuTick();
                    VideoCore::t_lc_rec_flags = 0;
                }
                // SHADPS4_ASC_LANE: DMA / WRITE_DATA / ACQUIRE_MEM / NOP / ASC labels run on the
                // lane; any other early packet runs on main after the lane's work so far.
                bool lane_label = false;
                if (VideoCore::g_lane_on && it.type == RT::Packet) [[unlikely]] {
                    const auto* h = reinterpret_cast<const PM4Header*>(r + 1);
                    lane_label = (r->flags & CpSplit::AscReleaseMem) != 0;
                    const auto op = h->type3.opcode.Value();
                    // p1f: ASC labels run on the lane too (their GDS store reads lane-written GDS).
                    t_lane_packet = h->type == 3 && (lane_label || op == PM4ItOpcode::DmaData);
                    auto& sch = rasterizer->GetScheduler();
                    if (t_lane_packet) {
                        sch.LaneCount(7);
                    } else {
                        sch.LaneCount(8);
                        const bool quiet = h->type == 3 && (op == PM4ItOpcode::WriteData ||
                                                            op == PM4ItOpcode::AcquireMem ||
                                                            op == PM4ItOpcode::Nop);
                        if (!quiet) {
                            sch.LaneFallbackMain();
                        }
                    }
                }
                if (q24_sch && it.type == RT::Dispatch) [[unlikely]] {
                    if (!q24_timed) {
                        q24_sch->ChainTimerBegin();
                        q24_timed = true;
                    }
                    ++q24_disp;
                }
                const u64 tx0 = __rdtsc();
                ExecuteRecordImpl(*r);
                if (it.type == RT::Packet && (r->flags & CpSplit::AscReleaseMem) && rasterizer &&
                    Vulkan::Rasterizer::EarlyCutMode() == 2) [[unlikely]] {
                    // q7c (2): the ASC submission's end label - the chain that wrote a
                    // read-back window ended; cut here, not at the next main record.
                    VideoCore::t_force_binding_sync = false;
                    rasterizer->MaybeEarlyCut(true);
                    VideoCore::t_force_binding_sync = true;
                }
                if (t_lane_packet) [[unlikely]] {
                    t_lane_packet = false;
                    // f2 (H12): prefetch copies of the windows this label ends, while the lane is active.
                    const u64 pf = lane_label && VideoCore::BufferCache::PrefetchMode() != 0
                                       ? rasterizer->GetBufferCache().PrefetchAfterLabel(VideoCore::g_pf_label_addr)
                                       : 0;
                    rasterizer->GetScheduler().SwitchStream(0);
                    if (lane_label) {
                        rasterizer->GetScheduler().LaneLabelCut(); // marks lane_label_value
                    }
                    if (pf != 0) {
                        rasterizer->GetScheduler().LaneCutNow(); // submit the segment now
                    }
                }
                if (VideoCore::g_lane_census) [[unlikely]] {
                    auto& lc = VideoCore::g_lc;
                    lc[18].fetch_add(1, std::memory_order_relaxed);
                    if (VideoCore::t_lc_rec_flags & 1) {
                        lc[16].fetch_add(1, std::memory_order_relaxed);
                    } else if (VideoCore::t_lc_rec_flags & 2) {
                        lc[17].fetch_add(1, std::memory_order_relaxed);
                    }
                }
                (it.type == RT::Dispatch ? g_asc_early_stats.c_disp : g_asc_early_stats.c_pkt)
                    .fetch_add(__rdtsc() - tx0, std::memory_order_relaxed);
                VideoCore::t_force_binding_sync = false;
                // l4a6 (review #1): as the L3 path - the next stretch replay needs its barrier.
                g_a2s.gpu_write = g_a2s.gpu_write || it.type == RT::Dispatch || !A2NoopDma(*r);
                const u64 up =
                    rasterizer ? rasterizer->GetBufferCache().SyncUploadBytesTotal() - up0 : 0;
                g_asc_early_stats.forced_checks.fetch_add(VideoCore::t_forced_binding_checks - chk0,
                                                         std::memory_order_relaxed);
                g_asc_early_stats.upload_bytes.fetch_add(up, std::memory_order_relaxed);
                g_asc_early_stats.uploading_records.fetch_add(up != 0, std::memory_order_relaxed);
                early_done.push_back(r->seq);
                (it.type == RT::Dispatch ? g_asc_early_stats.dispatches : g_asc_early_stats.packets)
                    .fetch_add(1, std::memory_order_relaxed);
                ++head;
                if (g_early_cmd_poll && num_commands.load(std::memory_order_relaxed) != 0) [[unlikely]] {
                    // q14: Game:Main's fault commands do not wait for the whole service.
                    ch.early_head.store(head, std::memory_order_release);
                    ProcessCommands();
                    const u64 n = g_early_cmd_polls.fetch_add(1, std::memory_order_relaxed) + 1;
                    if ((n & (n - 1)) == 0) {
                        LOG_INFO(Render, "q14: {} command runs inside the early ASC service", n);
                    }
                }
                continue;
            }
            if (it.type == RT::RegWrite || it.type == RT::EpochBump || it.type == RT::Diag) {
                g_asc_early_stats.in_order.fetch_add(1, std::memory_order_relaxed);
                ++head; // runs in ring order, as before
                continue;
            }
            g_asc_early_stats.blocked.fetch_add(1, std::memory_order_relaxed);
            break; // wait until the back passes it in ring order
        }
        ch.early_head.store(head, std::memory_order_release);
        early_last_tsc = __rdtsc(); // m2: gaps are measured from the end of a service
        VideoCore::g_lb_in_service = false;
    };
    if (asc_early && (g_asc_service & 2)) {
        early_service_ctx = &early_service;
        early_service_fn = [](void* p) { (*static_cast<decltype(early_service)*>(p))(); };
    }
    for (;;) {
        if (asc_early) [[unlikely]] {
            early_service();
        }
        if (const auto* record = split->Next()) {
            while (asc_early && !early_done.empty() && early_done.front() < record->seq) {
                early_done.pop_front(); // released without passing here (a DoneThrough range)
            }
            if (asc_early && !early_done.empty() && record->seq == early_done.front()) [[unlikely]] {
                early_done.pop_front();
                g_asc_early_stats.skipped.fetch_add(1, std::memory_order_relaxed);
                split->Done(record); // ran ahead already
                ProcessCommands();
                continue;
            }
            if (mtb_install && rasterizer) [[unlikely]] {
                mtb_install = false;
                // SHADPS4_MTB_P3 (P3.0, sec. 14d): the committer keeps the real tick; the
                // parked translation worker reserves at each run.
                const bool want_p3 = g_mtb_p3 != 0 && rasterizer->HasMtbWorker();
                mtb_p3 = rasterizer->GetScheduler().InstallMtbContext(!want_p3) && want_p3;
                if (mtb_p3) {
                    mtb_thread = std::jthread{std::bind_front(&Liverpool::MtbWorkerLoop, this)};
                    LOG_INFO(Render, "MtbP3: parked translation worker started");
                }
            }
            if (mtb_p3 && MtbRunAhead(record)) {
                ProcessCommands();
                continue;
            }
            Common::SampleThreadPlacement("back", 8);
            if (VideoCore::g_early_inv) [[unlikely]] {
                VideoCore::t_inorder_pos = split->ReadPos();
                VideoCore::EarlyInvPrune(split->ReadPos());
            }
            ExecuteRecord(*record);
            if (g_a2s.skip_pos != 0) [[unlikely]] {
                // SHADPS4_A2: a worker stretch - every record through its last draw at once.
                split->DoneThrough(std::exchange(g_a2s.skip_pos, 0), g_a2s.skip_seq);
            } else {
                split->Done(record);
            }
            // Guest commands run at record boundaries, as they ran at packet boundaries before.
            if (num_commands.load(std::memory_order_relaxed) != 0 && AscRingLatency())
                [[unlikely]] {
                const u64 t0 = __rdtsc();
                ProcessCommands();
                g_asc_ring.cmd_ticks += __rdtsc() - t0;
                ++g_asc_ring.cmd_n;
                continue;
            }
            ProcessCommands();
            continue;
        }
        if (num_commands.load(std::memory_order_relaxed) != 0 && AscRingLatency()) [[unlikely]] {
            const u64 t0 = __rdtsc();
            ProcessCommands();
            g_asc_ring.cmd_ticks += __rdtsc() - t0;
            ++g_asc_ring.cmd_n;
        }
        ProcessCommands();
        if (split->FrontHasExited()) {
            break;
        }
        if (stoken.stop_requested() && !stop_forwarded) {
            // The front owns the submission loop and leaves it on its own stop token. Keep
            // executing its records until it has exited (it drains first).
            // SPLIT-TODO: a front stuck in a guest wait loop hangs shutdown, as the CP does today.
            stop_forwarded = true;
            front_thread.request_stop();
        }
        {
            // SHADPS4_DRAW_COST=1 (diagnostic): back idle time, reported by NoteDrawCost.
            static const bool draw_cost = [] {
                const char* value = std::getenv("SHADPS4_DRAW_COST");
                return value != nullptr && value[0] == '1';
            }();
            const u64 idle_t0 = draw_cost ? __rdtsc() : 0;
            const bool ring_latency = AscRingLatency();
            const u32 ring_front_state = ring_latency ? AscRingIdleEnter() : 0;
            split->WaitForWork(num_commands, stoken);
            if (ring_latency) [[unlikely]] {
                AscRingIdleExit(ring_front_state);
            }
            if (draw_cost) {
                g_back_idle_cyc += __rdtsc() - idle_t0;
            }
        }
    }
    if (mtb_thread.joinable()) {
        mtb_thread.request_stop();
        mtb_thread.join();
    }
    A2Shutdown();
    if (rasterizer) {
        rasterizer->GetScheduler().MtbBackExit();
    }
    front_thread.join();
}

namespace {
bool MtbRunRecord(const CpSplit::RecordHeader* record) {
    switch (record->type) {
    case CpSplit::RecordType::Draw:
    case CpSplit::RecordType::Dispatch:
    case CpSplit::RecordType::RegWrite:
    case CpSplit::RecordType::Extent:
    case CpSplit::RecordType::ClearState:
    // Bookkeeping that translation does not read (s266: runs of ~16 draws without them).
    case CpSplit::RecordType::Retire:
    case CpSplit::RecordType::EpochBump:
    case CpSplit::RecordType::Diag:
        return true;
    case CpSplit::RecordType::Packet: {
        // The DMA_DATA to 0x3022C (~0.4 per draw) is skipped by ExecuteGfxPacket: it would cut
        // every run to ~5 draws (s265).
        if (record->flags & CpSplit::AscReleaseMem) {
            return false;
        }
        const auto* header = reinterpret_cast<const PM4Header*>(record + 1);
        const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
        return header->type3.opcode == PM4ItOpcode::DmaData && dma->dst_addr_lo == 0x3022C;
    }
    default:
        return false;
    }
}
constexpr size_t MtbMaxRun = 256;
} // namespace

bool Liverpool::MtbRunAhead(const CpSplit::RecordHeader* record) {
    if (!MtbRunRecord(record) || num_commands.load(std::memory_order_relaxed) != 0) {
        return false;
    }
    auto& h = mtb_handoff;
    h.run.clear();
    h.run.push_back(record);
    u32 draws = record->type == CpSplit::RecordType::Draw ||
                record->type == CpSplit::RecordType::Dispatch;
    // Read ahead: every record from ReadPos() on stays valid until Done passes it.
    u64 pos = split->ReadPos() + record->size;
    while (h.run.size() < MtbMaxRun && num_commands.load(std::memory_order_relaxed) == 0) {
        const auto* next = split->PeekAt(pos);
        if (!next || !MtbRunRecord(next)) {
            break;
        }
        draws += next->type == CpSplit::RecordType::Draw ||
                 next->type == CpSplit::RecordType::Dispatch;
        h.run.push_back(next);
        pos += next->size;
    }
    if (draws == 0) {
        return false; // register state only: cheaper on the back
    }
    h.state.store(1, std::memory_order_release);
    h.state.notify_one();
    for (u32 spins = 0; h.state.load(std::memory_order_acquire) != 2;) {
        if (++spins < 4096) {
            _mm_pause();
        } else {
            ++h.spin_yields;
            std::this_thread::yield();
        }
    }
    h.state.store(0, std::memory_order_relaxed);
    // The worker may stop early for guest commands; the rest is collected again.
    for (size_t i = 0; i < h.executed; ++i) {
        const auto* executed = split->Next();
        ASSERT_MSG(executed == h.run[i], "MtbP3: ring record order");
        split->Done(executed);
    }
    ++h.runs;
    h.records += h.executed;
    h.draws += draws;
    if (++h.log_runs >= 65536) {
        LOG_INFO(Render, "MtbP3: runs={} records_per_run={:.1f} draws_per_run={:.1f} back_yields={}",
                 h.runs, double(h.records) / double(h.runs), double(h.draws) / double(h.runs),
                 h.spin_yields);
        h.log_runs = 0;
    }
    return true;
}

void Liverpool::MtbWorkerLoop(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:MtbWorker");
    // As urgent as the back it stands in for (SHADPS4_CP_QOS keeps one P-core for the back).
    Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
    auto& h = mtb_handoff;
    std::stop_callback wake{stoken, [&h] {
        h.state.store(3, std::memory_order_release);
        h.state.notify_one();
    }};
    auto& scheduler = rasterizer->GetScheduler();
    scheduler.AttachMtbContext();
    // The worker only runs while the back is parked on its run: it acts for the CP then (=1:
    // also the CP-only fast paths, whose single-writer rule the hand-off keeps).
    Common::FaultMonitor::t_acting_cp = true;
    if (g_mtb_p3 == 1) {
        Common::FaultMonitor::t_is_cp_thread = true;
    }
    LOG_INFO(Render, "MtbP3: worker running, mode {}", g_mtb_p3);
    for (;;) {
        u32 state = h.state.load(std::memory_order_acquire);
        for (u32 spins = 0; state != 1 && state != 3;) {
            if (++spins < 16384) {
                _mm_pause();
            } else if (spins < 16384 + 2048) {
                std::this_thread::yield();
            } else {
                h.state.wait(state, std::memory_order_acquire);
                spins = 0;
            }
            state = h.state.load(std::memory_order_acquire);
        }
        if (state == 3) {
            break;
        }
        scheduler.ReserveTickForRun();
        size_t executed = 0;
        for (const auto* r : h.run) {
            ExecuteRecordImpl(*r);
            ++executed;
            // Guest threads waiting in SendCommand are served at record boundaries, as before.
            if (num_commands.load(std::memory_order_relaxed) != 0) {
                break;
            }
        }
        h.executed = executed;
        h.state.store(2, std::memory_order_release);
    }
}

namespace {
// SHADPS4_MTB_CENSUS (notes/multithreaded-back-design-2026-10-08.md P0): window statistics.
struct MtbWindow {
    u64 seg_draws = 0;
    bool seg_bailed = false;
    bool seg_dependent = false;
    u64 dependents = 0, segments_dependent = 0, tail_draws = 0;
    u64 segments = 0, segments_bailed = 0, seg_len_sum = 0, seqpoints = 0, dma_noop = 0, bookkeeping = 0;
    u64 draws = 0, par_draws = 0;
    u64 par_cyc = 0, ser_draw_cyc = 0, seq_cyc = 0, other_cyc = 0;
    std::array<u64, AmdGpu::MtbCensus::Count> bails{};
    // v4 (P1d, notes/parrec-build-2026-10-08.md sec. 10): sequence points by kind (0 WRITE_DATA,
    // 1 EVENT_WRITE_EOP, 2 EVENT_WRITE_EOS, 3 DMA_DATA, 4 other packet, 5 Call): total, inside a
    // batch (no submission during it), render pass left open, render pass ended; submissions;
    // CurrentTick() reads during draw / dispatch translation.
    std::array<u64, 6> sp_total{}, sp_in_batch{}, sp_pass_kept{}, sp_pass_ended{};
    u64 submits = 0, tick_reads = 0;
};
MtbWindow g_mtb;
const u64 g_mtb_segment = [] {
    const char* env = std::getenv("SHADPS4_MTB_SEGMENT");
    const u64 v = env ? std::strtoull(env, nullptr, 0) : 0;
    return v != 0 ? v : u64{128};
}();

void MtbEndSegment() {
    if (g_mtb.seg_draws != 0) {
        ++g_mtb.segments;
        g_mtb.seg_len_sum += g_mtb.seg_draws;
        g_mtb.segments_bailed += g_mtb.seg_bailed ? 1 : 0;
    }
    g_mtb.seg_draws = 0;
    g_mtb.seg_bailed = false;
    g_mtb.seg_dependent = false;
    AmdGpu::MtbCensus::g_seg_events.clear();
}

void MtbLog() {
    auto& w = g_mtb;
    const double all = double(w.par_cyc + w.ser_draw_cyc + w.seq_cyc + w.other_cyc);
    const double par = all > 0 ? double(w.par_cyc) / all : 0.0;
    const auto amdahl = [par](double k) { return 1.0 / ((1.0 - par) + par / k); };
    const auto& b = w.bails;
    LOG_INFO(Render,
             "MtbCensus: segment={} draws={} parallel_draws={} ({:.1f}%) segments={} mean_len={:.1f} "
             "bailed={:.1f}% seqpoints={} dma_noop={} bookkeeping={} | cycles: parallel {:.1f}% serial_draws {:.1f}% "
             "seqpoints {:.1f}% other {:.1f}% | Amdahl K=2 {:.2f}x K=4 {:.2f}x | first bails: "
             "buf_create={} buf_delete={} buf_upload={} img_register={} img_unregister={} "
             "img_upload={} stream_alloc={} occlusion={} img_invalidate={} | v3: dependent_draws={} "
             "({:.2f}%) segments_with_dependent={} ({:.1f}%) tail_draws_from_first_dependent={} ({:.1f}%) "
             "invalidate_hits={}",
             g_mtb_segment, w.draws, w.par_draws, 100.0 * double(w.par_draws) / double(std::max<u64>(w.draws, 1)),
             w.segments, double(w.seg_len_sum) / double(std::max<u64>(w.segments, 1)),
             100.0 * double(w.segments_bailed) / double(std::max<u64>(w.segments, 1)), w.seqpoints,
             w.dma_noop, w.bookkeeping,
             100.0 * par, all > 0 ? 100.0 * double(w.ser_draw_cyc) / all : 0.0,
             all > 0 ? 100.0 * double(w.seq_cyc) / all : 0.0,
             all > 0 ? 100.0 * double(w.other_cyc) / all : 0.0, amdahl(2.0), amdahl(4.0),
             b[AmdGpu::MtbCensus::BufferCreate], b[AmdGpu::MtbCensus::BufferDelete],
             b[AmdGpu::MtbCensus::BufferUpload], b[AmdGpu::MtbCensus::ImageRegister],
             b[AmdGpu::MtbCensus::ImageUnregister], b[AmdGpu::MtbCensus::ImageUpload],
             b[AmdGpu::MtbCensus::StreamAlloc], b[AmdGpu::MtbCensus::OcclusionSlot],
             b[AmdGpu::MtbCensus::ImageInvalidate], w.dependents,
             100.0 * double(w.dependents) / double(std::max<u64>(w.draws, 1)), w.segments_dependent,
             100.0 * double(w.segments_dependent) / double(std::max<u64>(w.segments, 1)), w.tail_draws,
             100.0 * double(w.tail_draws) / double(std::max<u64>(w.draws, 1)),
             AmdGpu::MtbCensus::g_invalidate_hits);
    const auto kinds = [&w](const std::array<u64, 6>& a) {
        return fmt::format("{}/{}/{}/{}/{}/{}", a[0], a[1], a[2], a[3], a[4], a[5]);
    };
    LOG_INFO(Render,
             "MtbP1d: draws={} submits={} tick_reads_per_draw={:.2f} | seqpoints by kind "
             "[write_data/eop/eos/dma/other_pkt/call] total={} in_batch={} pass_kept={} pass_ended={}",
             w.draws, w.submits, double(w.tick_reads) / double(std::max<u64>(w.draws, 1)),
             kinds(w.sp_total), kinds(w.sp_in_batch), kinds(w.sp_pass_kept), kinds(w.sp_pass_ended));
    AmdGpu::MtbCensus::g_invalidate_hits = 0;
    const u64 open = w.seg_draws;
    const bool open_bailed = w.seg_bailed;
    const bool open_dependent = w.seg_dependent;
    w = MtbWindow{};
    w.seg_draws = open;
    w.seg_bailed = open_bailed;
    w.seg_dependent = open_dependent;
}
} // namespace

namespace {
// SHADPS4_BACK_CYCLES=1 (2026-10-09): the back's cycles per record type per second, and inside
// them the GPU waits / submits / stream-buffer wraps (Vulkan::t_back_cycles). Back thread only.
struct BackCycleWindow {
    std::array<u64, 16> cyc{};
    std::array<u64, 16> n{};
    u64 start_ns{};
    u64 start_tsc{};
    u64 calls{};
    Vulkan::BackCycleStats base{};
    u64 thread_cyc{};
    std::pair<u64, u64> prod{};
};
BackCycleWindow g_back_cyc;

u64 ThreadCycles() {
    return Common::CurrentThreadCycles();
}

void BackCyclesNote(u32 type, u64 cyc, Vulkan::Scheduler* sched) {
    auto& w = g_back_cyc;
    w.cyc[type & 15] += cyc;
    ++w.n[type & 15];
    if ((++w.calls & 4095) != 0) {
        return;
    }
    const u64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    const u64 tsc = __rdtsc();
    if (w.start_ns == 0) {
        w.start_ns = now;
        w.start_tsc = tsc;
        w.cyc = {};
        w.n = {};
        w.base = Vulkan::t_back_cycles;
        w.thread_cyc = ThreadCycles();
        w.prod = sched ? sched->ProducerWaitNs() : std::pair<u64, u64>{};
        return;
    }
    if (now - w.start_ns < 10'000'000'000ull) {
        return;
    }
    const double secs = double(now - w.start_ns) / 1e9;
    const double ms_per_cyc = double(now - w.start_ns) / 1e6 / double(tsc - w.start_tsc);
    const auto ms = [&](u64 c) { return double(c) * ms_per_cyc / secs; }; // ms per second
    const auto per_s = [&](u64 v) { return double(v) / secs; };
    u64 all = 0;
    for (const u64 c : w.cyc) {
        all += c;
    }
    using RT = CpSplit::RecordType;
    const auto t = [](RT r) { return static_cast<u32>(r); };
    const auto& now_s = Vulkan::t_back_cycles;
    const auto& b = w.base;
    const u64 thread_now = ThreadCycles();
    const auto prod = sched ? sched->ProducerWaitNs() : std::pair<u64, u64>{};
    // Thread CPU cycles (QueryThreadCycleTime) as a share of the window's TSC cycles: well below
    // records + spin = the back is descheduled (blocked) inside its records.
    const double cpu_pct = 100.0 * double(thread_now - w.thread_cyc) / double(tsc - w.start_tsc);
    LOG_INFO(Render,
             "BackCycles2: thread_cpu={:.1f}% of wall | producer waits ms per s: chunk_space={:.1f} "
             "drain={:.1f}",
             cpu_pct, double(prod.first - w.prod.first) / 1e6 / secs,
             double(prod.second - w.prod.second) / 1e6 / secs);
    w.thread_cyc = thread_now;
    w.prod = prod;
    LOG_INFO(Render,
             "BackCycles: window_s={:.1f} ghz={:.3f} | ms per s: records={:.1f} draw={:.1f} "
             "dispatch={:.1f} packet={:.1f} call={:.1f} regwrite={:.1f} other={:.1f} | per s: "
             "draws={:.0f} dispatches={:.0f} packets={:.0f} calls={:.0f} regwrites={:.0f} | "
             "inside: gpu_wait_ms={:.1f} waits={:.0f} blocked={:.0f} submit_ms={:.1f} "
             "submits={:.0f} stream_wraps={:.1f} stream_waits={:.1f} stream_wait_ms={:.2f}",
             secs, 1e-6 / ms_per_cyc, ms(all), ms(w.cyc[t(RT::Draw)]), ms(w.cyc[t(RT::Dispatch)]),
             ms(w.cyc[t(RT::Packet)]), ms(w.cyc[t(RT::Call)]), ms(w.cyc[t(RT::RegWrite)]),
             ms(all - w.cyc[t(RT::Draw)] - w.cyc[t(RT::Dispatch)] - w.cyc[t(RT::Packet)] -
                w.cyc[t(RT::Call)] - w.cyc[t(RT::RegWrite)]),
             per_s(w.n[t(RT::Draw)]), per_s(w.n[t(RT::Dispatch)]), per_s(w.n[t(RT::Packet)]),
             per_s(w.n[t(RT::Call)]), per_s(w.n[t(RT::RegWrite)]),
             ms(now_s.wait_cyc - b.wait_cyc), per_s(now_s.wait_calls - b.wait_calls),
             per_s(now_s.wait_blocked - b.wait_blocked), ms(now_s.flush_cyc - b.flush_cyc),
             per_s(now_s.flush_calls - b.flush_calls), per_s(now_s.stream_wraps - b.stream_wraps),
             per_s(now_s.stream_waits - b.stream_waits),
             ms(now_s.stream_wait_cyc - b.stream_wait_cyc));
    w.start_ns = now;
    w.start_tsc = tsc;
    w.cyc = {};
    w.n = {};
    w.base = now_s;
}
} // namespace

namespace {
// SHADPS4_A1 (architecture A step A1, notes/NATIVE-RENDERER.md): per top-level DCB, what a
// worker translating the whole DCB with today's native pieces would meet. Back thread only.
// Bits of what a DCB contains besides native draws (a worker cannot take it whole while any is
// set, except A1Sync, which A handles at the DCB boundary when it sits after the last draw):
enum A1Bits : u32 {
    A1Generic = 1u << 0,  ///< a draw took the generic (old per-draw) path
    A1Indirect = 1u << 1, ///< an indirect draw
    A1Dispatch = 1u << 2, ///< a dispatch
    A1Sync = 1u << 3,     ///< a sync packet (EOP / EOS / WRITE_DATA / DMA other than the no-op /
                          ///< RELEASE_MEM / other) or a Call record
    A1SyncMid = 1u << 4,  ///< ... with a draw after it inside the same DCB
};
struct A1Dcb {
    bool open{};
    u32 bits{};
    u32 draws{}, native{}, shadow{}, colour{}, skipped{};
    u64 cyc{};      ///< back cycles of the DCB's records (the census' own rdtsc included)
    const char* why{}; ///< the first generic draw's reason
    u32 gmask{};       ///< every generic reason seen (A1Reason bits)
    u32 generic{};     ///< generic draws
};
struct A1Census {
    A1Dcb cur{};
    u64 dcbs{}, empty{}, draws{}, native{}, records{}, asc_dispatches{};
    u64 cyc_all{}, cyc_outside{};
    std::array<u64, 32> by_bits{}, draws_by_bits{}, cyc_by_bits{};
    u64 elig_shadow_only{}, elig_colour_only{}, elig_mixed{};
    u64 snap_cyc{}, snap_n{};
    u64 max_draws{};
    std::array<u64, 8> draw_hist{}; ///< 0, 1-15, 16-63, 64-127, 128-255, 256-511, 512+
    std::unordered_map<const char*, u64> why;
    /// DCBs whose only blockers are generic draws (no I / D / mid sync), by reason set.
    std::array<u64, 32> g_dcbs{}, g_draws{}, g_cyc{}, g_generic{};
    u64 start_ns{}, start_tsc{};
    std::unique_ptr<u32[]> snap; ///< entry snapshot: config + SH, context, uconfig windows
    u64 begun{}; ///< DCBs begun on the back (whole run, never reset)
    /// complete DCBs published ahead at each DcbBegin: <0 (this one incomplete), 0, 1, 2, 3-4,
    /// 5-8, 9-16, 17+
    std::array<u64, 8> lead{};
    u64 snap_cmp{}, snap_bad{}, snap_bad_words{};
    u32 snap_logged{};
};
A1Census g_a1c;
u64 A1NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
constexpr u32 A1SnapWords = 0x1000 + 0x400 + 0x400;
/// Generic-path reasons: 0 peek single draw (policy), 1 not indexed, 2 NativeOpen failed, 3 no
/// variant for the vertex description, 4 anything else.
u32 A1Reason(const char* why) {
    if (why == nullptr) {
        return 4;
    }
    const std::string_view w{why};
    return w == "colour: peek says single draw"                    ? 0
           : w == "not indexed"                                    ? 1
           : w == "NativeOpen failed"                              ? 2
           : w == "colour: no variant for this vertex description" ? 3
                                                                   : 4;
}

void A1Log(A1Census& c) {
    const u64 now = A1NowNs();
    const double s = double(now - c.start_ns) * 1e-9;
    const u64 tsc = __rdtsc();
    const double tsc_per_ns = c.start_tsc && now > c.start_ns
                                  ? double(tsc - c.start_tsc) / double(now - c.start_ns)
                                  : 3.5;
    const auto ms_s = [&](u64 cyc) { return s > 0 ? double(cyc) / tsc_per_ns * 1e-6 / s : 0.0; };
    const u64 elig = c.by_bits[0] + c.by_bits[A1Sync];
    const u64 elig_d = c.draws_by_bits[0] + c.draws_by_bits[A1Sync];
    const u64 elig_c = c.cyc_by_bits[0] + c.cyc_by_bits[A1Sync];
    LOG_INFO(Render,
             "A1: window {:.1f}s dcbs/s={:.0f} draw_recs/s={:.0f} draws/dcb={:.1f} max={} empty={} "
             "native_draw_share={:.3f} | eligible(no generic/indirect/dispatch, sync only after "
             "the last draw) dcbs={:.3f} draws={:.3f} back_cyc={:.3f} (shadow-only {} colour-only "
             "{} mixed {}) | back ms/s: dcb records {:.1f} outside dcbs {:.1f} | entry snapshot "
             "{} B, {:.0f} cyc each, {:.2f} ms/s | asc dispatches/s {:.0f}",
             s, s > 0 ? double(c.dcbs) / s : 0.0, s > 0 ? double(c.draws) / s : 0.0,
             c.dcbs - c.empty ? double(c.draws) / double(c.dcbs - c.empty) : 0.0, c.max_draws,
             c.empty, c.draws ? double(c.native) / double(c.draws) : 0.0,
             c.dcbs - c.empty ? double(elig) / double(c.dcbs - c.empty) : 0.0,
             c.draws ? double(elig_d) / double(c.draws) : 0.0,
             c.cyc_all ? double(elig_c) / double(c.cyc_all) : 0.0, c.elig_shadow_only,
             c.elig_colour_only, c.elig_mixed, ms_s(c.cyc_all), ms_s(c.cyc_outside),
             A1SnapWords * 4, c.snap_n ? double(c.snap_cyc) / double(c.snap_n) : 0.0,
             ms_s(c.snap_cyc), s > 0 ? double(c.asc_dispatches) / s : 0.0);
    std::string bits;
    for (u32 b = 0; b < 32; ++b) {
        if (c.by_bits[b] != 0) {
            bits += fmt::format(" [{}{}{}{}{}]={}/{}d/{:.3f}c", (b & A1Generic) ? "G" : "-",
                                (b & A1Indirect) ? "I" : "-", (b & A1Dispatch) ? "D" : "-",
                                (b & A1Sync) ? "S" : "-", (b & A1SyncMid) ? "M" : "-",
                                c.by_bits[b], c.draws_by_bits[b],
                                c.cyc_all ? double(c.cyc_by_bits[b]) / double(c.cyc_all) : 0.0);
        }
    }
    LOG_INFO(Render, "A1 dcbs by content (G generic I indirect D dispatch S sync M sync before a "
                     "draw; dcbs/draws/back-cycle share):{} | draws per dcb hist {}",
             bits, fmt::format("{},{},{},{},{},{},{}", c.draw_hist[0], c.draw_hist[1],
                               c.draw_hist[2], c.draw_hist[3], c.draw_hist[4], c.draw_hist[5],
                               c.draw_hist[6]));
    std::vector<std::pair<u64, const char*>> whys;
    for (const auto& [k, v] : c.why) {
        whys.emplace_back(v, k);
    }
    std::sort(whys.rbegin(), whys.rend());
    std::string w;
    for (size_t i = 0; i < whys.size() && i < 10; ++i) {
        w += fmt::format(" '{}'={}", whys[i].second ? whys[i].second : "(none: not indexed / "
                                                                       "no native branch)",
                         whys[i].first);
    }
    LOG_INFO(Render, "A1 first generic draw per dcb, by reason:{}", w);
    LOG_INFO(Render,
             "A1 lead (complete dcbs published ahead at each dcb start; <0 = this one "
             "incomplete): <0={} 0={} 1={} 2={} 3-4={} 5-8={} 9-16={} 17+={} | front snapshot "
             "verify: compared={} bad={} bad_words={}",
             c.lead[0], c.lead[1], c.lead[2], c.lead[3], c.lead[4], c.lead[5], c.lead[6],
             c.lead[7], c.snap_cmp, c.snap_bad, c.snap_bad_words);
    std::string g;
    for (u32 m = 1; m < 32; ++m) {
        if (c.g_dcbs[m] != 0) {
            g += fmt::format(" [{}{}{}{}{}]={}/{}d/{}g/{:.3f}c", (m & 1) ? "P" : "-",
                             (m & 2) ? "N" : "-", (m & 4) ? "F" : "-", (m & 8) ? "V" : "-",
                             (m & 16) ? "O" : "-", c.g_dcbs[m], c.g_draws[m], c.g_generic[m],
                             c.cyc_all ? double(c.g_cyc[m]) / double(c.cyc_all) : 0.0);
        }
    }
    LOG_INFO(Render, "A1 dcbs blocked only by generic draws, by reason set (P peek single, N not "
                     "indexed, F NativeOpen failed, V no vertex variant, O other; "
                     "dcbs/draws/generic draws/back-cycle share):{}",
             g);
    auto keep = std::move(c.snap);
    const u64 begun = c.begun;
    const u32 logged = c.snap_logged;
    c = A1Census{};
    c.snap = std::move(keep);
    c.begun = begun;
    c.snap_logged = logged;
    c.start_ns = now;
    c.start_tsc = tsc;
}

void A1Close(A1Census& c) {
    auto& d = c.cur;
    if (!d.open) {
        return;
    }
    ++c.dcbs;
    c.draws += d.draws;
    c.native += d.native;
    c.max_draws = std::max<u64>(c.max_draws, d.draws);
    const u32 h = d.draws == 0 ? 0 : d.draws < 16 ? 1 : d.draws < 64 ? 2 : d.draws < 128 ? 3
                : d.draws < 256 ? 4 : d.draws < 512 ? 5 : 6;
    ++c.draw_hist[h];
    if (d.draws == 0) {
        ++c.empty;
    } else {
        const u32 b = d.bits & 31;
        ++c.by_bits[b];
        c.draws_by_bits[b] += d.draws;
        c.cyc_by_bits[b] += d.cyc;
        if (b == 0 || b == A1Sync) {
            if (d.colour == 0) {
                ++c.elig_shadow_only;
            } else if (d.shadow == 0) {
                ++c.elig_colour_only;
            } else {
                ++c.elig_mixed;
            }
        }
        if (d.why != nullptr || (d.bits & A1Generic)) {
            ++c.why[d.why];
        }
        if (b == A1Generic || b == (A1Generic | A1Sync)) {
            ++c.g_dcbs[d.gmask & 31];
            c.g_draws[d.gmask & 31] += d.draws;
            c.g_cyc[d.gmask & 31] += d.cyc;
            c.g_generic[d.gmask & 31] += d.generic;
        }
    }
    d = A1Dcb{};
    if (c.dcbs >= 65536) {
        A1Log(c);
    }
}
} // namespace

void Liverpool::A1Execute(const CpSplit::RecordHeader& record) {
    using RT = CpSplit::RecordType;
    auto& c = g_a1c;
    if (c.start_ns == 0) [[unlikely]] {
        c.start_ns = A1NowNs();
        c.start_tsc = __rdtsc();
    }
    if (!c.snap) [[unlikely]] {
        c.snap = std::make_unique<u32[]>(A1SnapWords);
    }
    if (record.type == RT::DcbBegin) {
        A1Close(c);
        // Entry snapshot (what a worker would start from): config + SH, context, uconfig windows.
        const u64 t0 = __rdtsc();
        const auto& r = mirror->regs.reg_array;
        std::memcpy(c.snap.get(), &r[0x2000], 0x1000 * 4);
        std::memcpy(c.snap.get() + 0x1000, &r[0xA000], 0x400 * 4);
        std::memcpy(c.snap.get() + 0x1400, &r[0xC000], 0x400 * 4);
        c.snap_cyc += __rdtsc() - t0;
        ++c.snap_n;
        c.cur.open = true;
        ++c.begun;
        if (split) {
            const s64 ahead =
                s64(split->dcb_ends_published.load(std::memory_order_acquire)) - s64(c.begun);
            const u32 li = ahead < 0 ? 0 : ahead == 0 ? 1 : ahead == 1 ? 2 : ahead == 2 ? 3
                         : ahead <= 4 ? 4 : ahead <= 8 ? 5 : ahead <= 16 ? 6 : 7;
            ++c.lead[li];
        }
        const auto& bp = *reinterpret_cast<const CpSplit::DcbBeginPayload*>(&record + 1);
        if (bp.snap_words == CpSplit::DcbSnapWords) {
            const u32* fs = reinterpret_cast<const u32*>(&bp + 1);
            ++c.snap_cmp;
            u32 bad = 0;
            for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
                for (u32 i = 0; i < n; ++i) {
                    if (fs[i] != r[off + i]) [[unlikely]] {
                        ++bad;
                        if (c.snap_logged < 40) {
                            ++c.snap_logged;
                            LOG_WARNING(Render,
                                        "A1 snapshot mismatch dcb {} reg {:#x}: front {:#x} "
                                        "back {:#x}",
                                        bp.index, off + i, fs[i], r[off + i]);
                        }
                    }
                }
                fs += n;
            }
            if (bad != 0) {
                ++c.snap_bad;
                c.snap_bad_words += bad;
            }
        }
        return;
    }
    auto& d = c.cur;
    u32 bits = 0;
    const u64 t0 = __rdtsc();
    switch (record.type) {
    case RT::Draw: {
        const auto& r = *reinterpret_cast<const SplitDrawRecord*>(&record + 1);
        Vulkan::Rasterizer* tr = rasterizer ? &rasterizer->Translating() : nullptr;
        const u64 opens0 = tr ? tr->A1NativeOpens() : 0;
        if (tr) {
            tr->rec_outcome = 0;
            (void)tr->A1TakeWhy();
        }
        ExecuteRecordImpl(record);
        ++d.draws;
        if (r.draw.args.is_indirect) {
            bits |= A1Indirect;
            break;
        }
        if (!tr) {
            break;
        }
        const u32 o = tr->rec_outcome;
        const bool opened = tr->A1NativeOpens() != opens0;
        if (o == 1 || o == 4) {
            ++d.native;
            ++d.shadow;
        } else if (o == 2 || o == 5) {
            ++d.native;
            ++d.colour;
        } else if (o == 3 && opened) {
            ++d.native;
            ++(tr->A1LastOpenColour() ? d.colour : d.shadow);
        } else if (o == 0) {
            ++d.skipped; // filtered / no draw (begin() refused)
            ++d.native;
        } else {
            bits |= A1Generic;
            const char* why = tr->A1TakeWhy();
            d.gmask |= 1u << A1Reason(why);
            ++d.generic;
            if (!(d.bits & A1Generic)) {
                d.why = why;
            }
        }
        if (d.bits & A1Sync) {
            bits |= A1SyncMid;
        }
        break;
    }
    case RT::Segment:
        ExecuteRecordImpl(record);
        ++d.draws;
        ++d.native;
        ++d.colour;
        break;
    case RT::Dispatch: {
        ExecuteRecordImpl(record);
        // ASC dispatches interleave with the gfx DCBs in the stream; only gfx-queue ones count.
        const auto& r = *reinterpret_cast<const SplitDispatchRecord*>(&record + 1);
        if (r.queue == 0) {
            bits |= A1Dispatch;
        } else {
            ++c.asc_dispatches;
        }
        break;
    }
    case RT::Packet: {
        ExecuteRecordImpl(record);
        const auto* header = reinterpret_cast<const PM4Header*>(&record + 1);
        const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
        const bool noop = !(record.flags & CpSplit::AscReleaseMem) &&
                          header->type3.opcode == PM4ItOpcode::DmaData &&
                          dma->dst_addr_lo == 0x3022C;
        if (!noop) {
            bits |= A1Sync;
        }
        break;
    }
    case RT::Call:
        ExecuteRecordImpl(record);
        bits |= A1Sync;
        break;
    default:
        ExecuteRecordImpl(record);
        break;
    }
    const u64 dt = __rdtsc() - t0;
    ++c.records;
    c.cyc_all += dt;
    if (d.open) {
        d.cyc += dt;
        d.bits |= bits;
    } else {
        c.cyc_outside += dt;
    }
}

namespace {
/// Context-register windows a render pass depends on (DB, window / generic scissors, colour
/// control, CB): a change ends a worker stretch (the back begins rendering once per stretch).
constexpr std::array<std::pair<u32, u32>, 4> A2RenderWindows{{
    {0x000, 0x020}, {0x080, 0x0A0}, {0x200, 0x210}, {0x318, 0x390}}};

/// The render-relevant pipeline key fields (compared when the pipeline changes).
struct A2PipeRender {
    u32 num_color;
    std::array<Shader::PsColorBuffer, AmdGpu::NUM_COLOR_BUFFERS> color;
    u8 samples, depth_samples;
    std::array<u8, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    u32 mrt_mask, z;
    explicit A2PipeRender(const Vulkan::GraphicsPipeline* p) {
        std::memset(this, 0, sizeof(*this));
        const auto& k = p->GetGraphicsKey();
        num_color = k.num_color_attachments;
        color = k.color_buffers;
        samples = k.num_samples;
        depth_samples = k.depth_samples;
        color_samples = k.color_samples;
        mrt_mask = k.mrt_mask;
        z = static_cast<u32>(k.z_format) | static_cast<u32>(k.stencil_format) << 4;
    }
    bool operator==(const A2PipeRender& o) const {
        return std::memcmp(this, &o, sizeof(*this)) == 0;
    }
};
/// Copies the render windows (compared with memcmp on a later change).
void A2CopyRenderWindows(const Regs& regs, std::array<u32, 0x20 + 0x20 + 0x10 + 0x78>& out) {
    u32 o = 0;
    for (const auto& [lo, hi] : A2RenderWindows) {
        std::memcpy(&out[o], &regs.reg_array[Regs::ContextRegWordOffset + lo], (hi - lo) * 4);
        o += hi - lo;
    }
}
bool A2TouchesRenderWindows(u32 off, u32 count) {
    for (const auto& [lo, hi] : A2RenderWindows) {
        const u32 a = Regs::ContextRegWordOffset + lo, b = Regs::ContextRegWordOffset + hi;
        if (off < b && off + count > a) {
            return true;
        }
    }
    return false;
}

u64 A2RenderKey(const Regs& regs, const Vulkan::GraphicsPipeline* pipeline) {
    u64 h = 0xcbf29ce484222325ull;
    const auto mix = [&](const void* p, size_t n) {
        const auto* b = static_cast<const u8*>(p);
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ b[i]) * 0x100000001B3ull;
        }
    };
    for (const auto& [lo, hi] : A2RenderWindows) {
        mix(&regs.reg_array[Regs::ContextRegWordOffset + lo], (hi - lo) * sizeof(u32));
    }
    const auto& k = pipeline->GetGraphicsKey();
    mix(&k.num_color_attachments, sizeof(k.num_color_attachments));
    mix(k.color_buffers.data(), sizeof(k.color_buffers));
    mix(&k.num_samples, sizeof(k.num_samples));
    mix(&k.depth_samples, sizeof(k.depth_samples));
    mix(k.color_samples.data(), sizeof(k.color_samples));
    mix(&k.mrt_mask, sizeof(k.mrt_mask));
    const u32 z = static_cast<u32>(k.z_format) | static_cast<u32>(k.stencil_format) << 4;
    mix(&z, sizeof(z));
    return h;
}

bool A2NoopDma(const CpSplit::RecordHeader& record) {
    if (record.flags & CpSplit::AscReleaseMem) {
        return false;
    }
    const auto* header = reinterpret_cast<const PM4Header*>(&record + 1);
    const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
    return header->type3.opcode == PM4ItOpcode::DmaData && dma->dst_addr_lo == 0x3022C;
}

void A2Log(u64 slot_cmp, u64 slot_bad, u64 expect_n) {
    auto& g = g_a2s;
    auto& c = g.c;
    const u64 now = A2NowNs();
    const u64 tsc = __rdtsc();
    const double s = double(now - c.start_ns) * 1e-9;
    const double tpn = now > c.start_ns ? double(tsc - c.start_tsc) / double(now - c.start_ns) : 3.5;
    const auto ps = [&](u64 n) { return s > 0 ? double(n) / s : 0.0; };
    const auto ms = [&](u64 cyc) { return s > 0 ? double(cyc) / tpn * 1e-6 / s : 0.0; };
    const double ws = std::max(1e-9, double(now - g.t_start) * 1e-9);
    const auto wms = [&](u64 cyc) { return double(cyc) / tpn * 1e-6 / ws; };
    const u64 all = c.translated + c.walked;
    std::array<u64, 32> refusals{};
    for (const auto* t : g.translators) {
        for (u32 i = 0; i < 32; ++i) {
            refusals[i] += t->refusals[i];
        }
    }
    std::string rf;
    for (u32 i = 0; i < 32; ++i) {
        if (refusals[i] != 0) {
            rf += fmt::format(" '{}'={}", Vulkan::A2::RefusalName(i), refusals[i]);
        }
    }
    std::array<u64, 16> subs{};
    for (const auto* t : g.translators) {
        for (u32 i = 0; i < 16; ++i) {
            subs[i] += t->sub[i];
        }
    }
    const std::string sub = fmt::format(
        " | sub-reasons: record fields inst_known0 {} io_ud {} index_count0 {} push_inh {}; slot "
        "kind shadow-images {} clip {} not-guest {} written {} formatted {} colour-inline/null {}; "
        "guest lookup no-buffer {} outside {}",
        subs[0], subs[1], subs[2], subs[3], subs[4], subs[5], subs[6], subs[7], subs[8], subs[9],
        subs[10], subs[11]);
    LOG_INFO(Render,
             "A2 (mode {}): window {:.1f}s | back-TRANSLATED draw records/s={:.0f} back-walked "
             "(worker) draw records/s={:.0f} fallback share={:.3f} | stretches/s={:.0f} "
             "draws/stretch={:.1f} | dcbs/s={:.0f} taken whole by the back={} (front waiting {} "
             "time-out {} unclaimed {}) followed by workers={} | back waits on workers at draws {} "
             "({:.2f} ms/s), of which worker behind the publish {} ({:.2f} ms/s); at DcbEnd {} "
             "({:.2f} ms/s); for a claim {} ({:.2f} ms/s); safety cancels {} | global "
             "barriers/s={:.0f} occlusion rejects={} buffer-structure rejects {} | back records/s "
             "{:.0f} (released in bulk/s {:.0f}, delta words per stretch {:.1f}) | verify: front "
             "snapshot {} bad {}, render key {} bad {}, would-walk {} | whole: stale {} refused {} "
             "state4 {} | workers {} (since start): translating {:.0f} ms/s, waiting for the next "
             "DCB {:.0f} ms/s, caught up waiting for records {:.0f} ms/s (parked {:.0f}, park {}); cuts: back waited {} "
             "front waited {}; stretch cap {} seq breaks {} cancels {}; translator cyc per draw "
             "{:.0f} | colour: stretches {} materials {} (resolve {:.2f} ms/s) rejected: material {} render target {} | slot verify {} bad {} not written by the old path {} | worker refusals (cumulative):{}",
             g_a2, s, ps(c.translated), ps(c.walked),
             all ? double(c.translated) / double(all) : 0.0, ps(c.stretches),
             c.stretches ? double(c.walked) / double(c.stretches) : 0.0, ps(c.dcbs), c.back_dcbs,
             c.steal_front_waits, c.steal_timeout, c.steal_unclaimed, c.worker_dcbs, c.waits,
             ms(c.wait_cyc), c.dwait_behind, ms(c.dwait_behind_cyc), c.end_waits,
             ms(c.end_wait_cyc), c.wait_complete, ms(c.wait_complete_cyc), c.net_cancel,
             ps(c.barriers), c.occl_rejects, c.gen_rejects, ps(c.back_records),
             ps(c.skipped_records),
             c.stretches ? double(c.delta_words) / double(c.stretches) : 0.0, c.snap_cmp,
             c.snap_bad, c.rk_cmp, c.rk_bad, c.would, c.whole_stale, c.whole_refused,
             c.whole_st4, Vulkan::A2::NumWorkers(), wms(g.worker_cyc), wms(g.publish_wait_cyc),
             wms(g.record_wait_cyc), wms(g.park_cyc), g_a2_park ? "on" : "off",
             g.cuts_back.load(), g.cuts_front.load(), g.cap_hits.load(),
             g.seq_breaks.load(), g.cancels.load(),
             g.worker_draws ? double(g.take_cyc) / double(g.worker_draws) : 0.0,
             c.col_stretches, c.materials, ms(c.col_resolve_cyc), c.mat_rejects, c.rt_rejects,
             slot_cmp, slot_bad,
             c.slot_unwritten + expect_n, rf + sub);
    c = A2Counters{};
    c.start_ns = now;
    c.start_tsc = tsc;
}
} // namespace

void Liverpool::A2WorkerLoop(std::stop_token stoken, u32 index) {
    Common::SetCurrentThreadName(fmt::format("shadPS4:A2Worker{}", index).c_str());
    Common::ApplyHelperThreadPlacement("A2Worker");
    auto& g = g_a2s;
    auto arena = std::make_unique<Vulkan::A2::Arena>();
    Vulkan::A2::WorkerContext ctx{};
    ctx.instance = &rasterizer->A2Instance();
    ctx.scheduler = &rasterizer->GetScheduler();
    ctx.buffer_cache = &rasterizer->A2BufferCache();
    ctx.arena = arena.get();
    ctx.worker_index = index;
    Vulkan::A2::StretchTranslator tr{ctx};
    {
        static std::mutex reg_mutex;
        std::scoped_lock lock{reg_mutex};
        g.translators.push_back(&tr);
    }
    auto regs = std::make_unique<Regs>();
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    std::vector<u32> pending;               // words written since the last taken draw
    std::vector<std::pair<u32, u32>> delta; // committed through the last taken draw
    if (index >= CpSplit::Channel::MaxReaderHolds) {
        LOG_ERROR(Render, "A2: worker {} not started (at most {} workers hold ring space)", index,
                  CpSplit::Channel::MaxReaderHolds);
        return;
    }
    auto& at = g.worker_at[index & 7];
    auto& hold = split->reader_holds[index].pos;
    const auto stopping = [&] {
        return stoken.stop_requested() || g.stop.load(std::memory_order_acquire);
    };
    while (!stopping()) {
        const u32 idx = g.next_claim.fetch_add(1, std::memory_order_acq_rel);
        at.store(idx | (1u << 28), std::memory_order_relaxed);
        // Wait until the front began `idx` (the entry is written before its DcbBegin record can
        // be published).
        {
            const u64 pw0 = __rdtsc();
            if (g_claim_wake) {
                if (!g.bells.Wait(g.begun, idx, stopping)) {
                    return;
                }
            }
            for (u32 spins = 0; g.begun.load(std::memory_order_acquire) < idx;) {
                if (stopping()) {
                    return;
                }
                if (++spins < 2048) {
                    _mm_pause();
                } else {
                    const u32 seen = g.begun.load(std::memory_order_acquire);
                    if (seen < idx) {
                        g.begun.wait(seen, std::memory_order_acquire);
                    }
                }
            }
            g.publish_wait_cyc.fetch_add(__rdtsc() - pw0, std::memory_order_relaxed);
            Common::FaultMonitor::SampleEmuPlacement(4);
        }
        auto& e = g.e[idx % A2Slots];
        if (e.index.load(std::memory_order_acquire) != idx) {
            continue; // the back took it and released it
        }
        // Hold ring space BEFORE the claim: the back proceeds into the DCB (and releases its
        // records) only after it sees state 2. A begin_pos read here is this DCB's or, if the
        // slot was reused, a later one - both safe (the floor only has to be low enough).
        hold.store(e.begin_pos.load(std::memory_order_acquire), std::memory_order_seq_cst);
        u32 expect = 1;
        if (!e.state.compare_exchange_strong(expect, 2, std::memory_order_seq_cst)) {
            hold.store(~0ull, std::memory_order_release);
            continue; // the back took it
        }
        // Stable while we hold it (state 2). If the slot was released and begun again for
        // idx + A2Slots between the two loads, we simply follow that DCB (its own claimer will
        // find it taken).
        const u32 dcb = e.index.load(std::memory_order_acquire);
        at.store(dcb | (2u << 28), std::memory_order_relaxed);
        const u64 w_t0 = __rdtsc();
        u64 w_draws = 0, rec_wait = 0;
        u32 n = 0;        // stretches handed over
        u64 decided = 0;  // progress published
        u64 last_seq = 0; // seq of the newest record walked
        bool open = false;
        bool occlusion = false;
        bool rt_dirty = false;                       // render windows written since saved
        const Vulkan::GraphicsPipeline* rt_pipe = nullptr; // pipeline the stretch's key holds
        std::optional<A2PipeRender> rt_pkey;
        std::array<u32, 0x20 + 0x20 + 0x10 + 0x78> rt_win{};
        u64 last_end = 0, last_epoch = 0, open_key = 0, open_first = 0;
        pending.clear();
        delta.clear();
        const auto publish = [&](u64 seq) {
            if (seq > decided) {
                decided = seq;
                e.progress.store(seq, std::memory_order_release);
            }
        };
        const auto end_stretch = [&] {
            if (open) {
                if (auto s = tr.EndStretch()) {
                    e.stretches[n] = std::move(s);
                    e.side[n] = {last_end, last_epoch, std::move(delta), open_key};
                    e.ready.store(++n, std::memory_order_release);
                }
                open = false;
            }
            pending.clear();
            delta.clear();
        };
        // 0: the record at p is published; 1: stop; 2: the back cancelled (safety net).
        const auto wait_pub = [&](u64 p) -> int {
            if (split->PublishedEnd() > p) {
                return 0;
            }
            // Caught up with the front: everything before an open stretch is decided.
            publish(open ? open_first - 1 : last_seq);
            e.walked.store(p, std::memory_order_release);
            at.store(dcb | (3u << 28), std::memory_order_relaxed);
            const u64 t0 = __rdtsc();
            int r = 0;
            for (u32 spins = 0; split->PublishedEnd() <= p; ++spins) {
                if (stopping()) {
                    r = 1;
                    break;
                }
                if (e.cancel.load(std::memory_order_acquire) != 0) {
                    r = 2;
                    break;
                }
                if (open) {
                    // Somebody needs the open stretch now: the back waits for its first draw,
                    // or the front waits for the back (a drain publishes without a record that
                    // would close the stretch).
                    const bool back = e.back_wait.load(std::memory_order_acquire) != 0;
                    if (back || split->FrontWaitingOnBack()) {
                        (back ? g.cuts_back : g.cuts_front).fetch_add(1, std::memory_order_relaxed);
                        end_stretch();
                        publish(last_seq);
                    }
                }
                if (g_a2_park && spins >= 256) {
                    // Park: every waker rings the bell after changing what we re-check here
                    // (seq_cst both sides, so either we see the change or it sees us parked).
                    const u64 k0 = __rdtsc();
                    split->reader_parked.fetch_add(1, std::memory_order_seq_cst);
                    const u32 bell = split->reader_bell.load(std::memory_order_seq_cst);
                    const bool need =
                        open && (e.back_wait.load(std::memory_order_seq_cst) != 0 ||
                                 split->FrontWaitingOnBack());
                    if (split->PublishedEnd() <= p && !need && !stopping() &&
                        e.cancel.load(std::memory_order_seq_cst) == 0) {
                        split->reader_bell.wait(bell, std::memory_order_acquire);
                    }
                    split->reader_parked.fetch_sub(1, std::memory_order_seq_cst);
                    g.park_cyc.fetch_add(__rdtsc() - k0, std::memory_order_relaxed);
                    spins = 0;
                    continue;
                }
                spins < 1024 ? _mm_pause() : std::this_thread::yield();
            }
            rec_wait += __rdtsc() - t0;
            at.store(dcb | (2u << 28), std::memory_order_relaxed);
            return r;
        };
        u64 pos = e.begin_pos.load(std::memory_order_acquire);
        hold.store(pos, std::memory_order_release); // >= the value held before the claim
        bool ok = wait_pub(pos) == 0;
        if (ok) {
            const auto* h = split->RecordAtPos(pos);
            const auto& bp = *reinterpret_cast<const CpSplit::DcbBeginPayload*>(h + 1);
            if (h->type != CpSplit::RecordType::DcbBegin || bp.index != dcb ||
                bp.snap_words != CpSplit::DcbSnapWords) {
                e.refused = 1;
                ok = false;
            } else {
                const u32* fs = reinterpret_cast<const u32*>(&bp + 1);
                for (const auto& [off, cnt] : CpSplit::DcbSnapWindows) {
                    std::memcpy(&regs->reg_array[off], fs, size_t{cnt} * sizeof(u32));
                    fs += cnt;
                }
                last_seq = h->seq;
                pos += h->size;
            }
        }
        while (ok) {
            if (wait_pub(pos) != 0) {
                break; // stop or cancel: the rest runs on the old path
            }
            hold.store(pos, std::memory_order_release); // grows only: stale = lower = safe
            e.walked.store(pos, std::memory_order_relaxed);
            u64 rpos = pos;
            const auto* h = split->RecordAtPos(rpos);
            if (rpos != pos) {
                pos = rpos; // skipped a Wrap: wait for the record after it
                continue;
            }
            if (h->seq != last_seq + 1) {
                // An alarm only (the ring hold makes an overwrite impossible): stop here, the
                // rest runs on the old path.
                g.seq_breaks.fetch_add(1, std::memory_order_relaxed);
                break;
            }
            if (h->type == CpSplit::RecordType::DcbEnd ||
                h->type == CpSplit::RecordType::DcbBegin ||
                h->type == CpSplit::RecordType::DcbRef) {
                // DcbRef (SHADPS4_LEAN=2): the head's register writes are in a job channel this
                // walk does not see - the rest of the DCB runs on the old path.
                last_seq = h->seq;
                break;
            }
            const auto* payload = reinterpret_cast<const u8*>(h + 1);
            switch (h->type) {
            case CpSplit::RecordType::RegWrite: {
                const u8* end = reinterpret_cast<const u8*>(h) + h->size;
                const u8* p = payload;
                while (p + sizeof(CpSplit::RegWritePayload) <= end) {
                    const auto* w = reinterpret_cast<const CpSplit::RegWritePayload*>(p);
                    if (w->count == 0) {
                        break;
                    }
                    std::memcpy(&regs->reg_array[w->word_offset], w + 1,
                                size_t{w->count} * sizeof(u32));
                    if (open) {
                        for (u32 i = 0; i < w->count; ++i) {
                            pending.push_back(w->word_offset + i);
                        }
                        rt_dirty = rt_dirty || A2TouchesRenderWindows(w->word_offset, w->count);
                    }
                    p += sizeof(CpSplit::RegWritePayload) + size_t{w->count} * sizeof(u32);
                }
                break;
            }
            case CpSplit::RecordType::ClearState:
                end_stretch(); // the back must see it (a whole-mirror reset)
                regs->SetDefaults();
                break;
            case CpSplit::RecordType::Draw: {
                const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(payload);
                stages = {};
                const u8* stages_end =
                    ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, stages);
                Vulkan::A2::DrawIn in{};
                in.pipeline = rec.draw.pipeline;
                in.is_indexed = rec.draw.args.is_indexed;
                in.is_indirect = rec.draw.args.is_indirect;
                in.index_offset = rec.draw.args.index_offset;
                in.stages = &stages;
                in.vi = rec.vi_bytes != 0
                            ? reinterpret_cast<const CpSplit::VertexInputsPayload*>(stages_end)
                            : nullptr;
                in.job_tag = rec.job_tag;
                in.job_idx = rec.job_idx;
                in.job_cmd = rec.cmd_address;
                in.seq = h->seq;
                in.rec_pos = pos;
                in.arg_address = rec.draw.args.arg_address + rec.draw.args.offset;
                ++w_draws;
                if (occlusion || !in.pipeline || rec.marker || rec.verify != 0) {
                    end_stretch();
                    break;
                }
                if (open && (rt_dirty || in.pipeline != rt_pipe)) {
                    // Same pass only if the render windows and the pipeline's attachment
                    // setup are unchanged since the stretch's first draw.
                    bool same = true;
                    if (rt_dirty) {
                        std::array<u32, 0x20 + 0x20 + 0x10 + 0x78> now{};
                        A2CopyRenderWindows(*regs, now);
                        same = now == rt_win;
                        rt_dirty = false;
                    }
                    if (same && in.pipeline != rt_pipe) {
                        same = A2PipeRender{in.pipeline} == *rt_pkey;
                        rt_pipe = in.pipeline;
                    }
                    if (!same) {
                        end_stretch();
                    }
                }
                if (!open && n >= A2MaxStretches) {
                    g.cap_hits.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                const u64 tk0 = __rdtsc();
                const bool took = tr.TakeDraw(*regs, in);
                g.take_cyc.fetch_add(__rdtsc() - tk0, std::memory_order_relaxed);
                if (took) {
                    if (!open) {
                        open = true;
                        open_first = h->seq;
                        rt_dirty = false;
                        rt_pipe = in.pipeline;
                        rt_pkey.emplace(in.pipeline);
                        A2CopyRenderWindows(*regs, rt_win);
                        if (g_a2 == 2) {
                            open_key = A2RenderKey(*regs, in.pipeline);
                        }
                        pending.clear();
                        delta.clear();
                    }
                    // The words written before this draw inside the stretch are the back's
                    // to apply (it skips their records); later ones stay pending.
                    for (const u32 off : pending) {
                        delta.emplace_back(off, regs->reg_array[off]);
                    }
                    pending.clear();
                    last_end = pos + h->size;
                    last_epoch = rec.ctx_epoch;
                } else {
                    end_stretch();
                }
                break;
            }
            case CpSplit::RecordType::Packet:
                if (!A2NoopDma(*h)) {
                    // A ZPASS_DONE counter dump opens / closes occlusion wrapping: the rest of
                    // this DCB stays on the old path (the back decides per draw).
                    const auto* ph = reinterpret_cast<const PM4Header*>(payload);
                    if (ph->type3.opcode == PM4ItOpcode::EventWrite) {
                        occlusion = true;
                    }
                    end_stretch();
                }
                break;
            default: // Extent, EpochBump, Call, Dispatch, Segment, Retire, Diag: a boundary
                end_stretch();
                break;
            }
            last_seq = h->seq;
            if (!open && h->type == CpSplit::RecordType::Draw) {
                publish(last_seq);
            }
            pos += h->size;
        }
        end_stretch();
        if (e.cancel.load(std::memory_order_acquire) != 0) {
            g.cancels.fetch_add(1, std::memory_order_relaxed);
        }
        g.worker_cyc.fetch_add(__rdtsc() - w_t0 - rec_wait, std::memory_order_relaxed);
        g.record_wait_cyc.fetch_add(rec_wait, std::memory_order_relaxed);
        g.worker_dcbs.fetch_add(1, std::memory_order_relaxed);
        g.worker_draws.fetch_add(w_draws, std::memory_order_relaxed);
        at.store(0, std::memory_order_relaxed);
        hold.store(~0ull, std::memory_order_release); // done reading this DCB's records
        e.progress.store(~0ull, std::memory_order_release);
        e.state.store(3, std::memory_order_release);
        e.state.notify_all();
    }
}

/// L3 (SHADPS4_LEAN_PARSE=2 + SHADPS4_A2; notes/architecture-a-2026-10-10.md sec. 16): a lean worker
/// translates its own draws into A2 stretches while it commits them - A2WorkerLoop's record walk
/// over its own job channel, with its chain registers as the mirror (at a Draw record's commit they
/// hold the state at that draw; every record before it is final, a RegWrite record only grows while
/// it is the newest). Safety argument: every publish of the job channel first ends the open stretch
/// (pre_publish), so a stretch is complete before its first draw is visible - the back never waits
/// on a stretch, and its DoneThrough to end_pos never passes the published end.
struct LeanXl {
    Vulkan::A2::StretchTranslator* tr{};
    LeanJob* job{};
    CpSplit::Channel* ch{};
    const Regs* regs{};
    u32 idx{};
    u64 walk_pos{}; ///< next record to walk (unmasked)
    u32 n{};        ///< stretches handed over in this job
    bool open{}, occlusion{}, rt_dirty{};
    const Vulkan::GraphicsPipeline* rt_pipe{}; ///< pipeline the stretch's key holds
    std::optional<A2PipeRender> rt_pkey;
    std::array<u32, 0x20 + 0x20 + 0x10 + 0x78> rt_win{};
    u64 last_end{}, last_epoch{};
    std::vector<u32> pending;               // words written since the last taken draw
    std::vector<std::pair<u32, u32>> delta; // committed through the last taken draw
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    u32 since_pub{};
    u64 open_key{}; ///< SHADPS4_A2=2: A2RenderKey at the open stretch's first draw
    // The run of RegWrite records right before the current record (pre-delta) and the open
    // stretch's own pre-run.
    u64 pre_start{~u64{0}};
    std::vector<u32> pre_offs;
    u64 s_pre_pos{~u64{0}}, s_first_pos{};
    std::vector<std::pair<u32, u32>> s_pre_delta;
    u64 taken{}, stretches{}, cap{}, cut_pub{}, walked{}, mat_cuts{};
};

void LeanXlEnd(LeanXl& x) {
    if (x.open) {
        if (auto s = x.tr->EndStretch()) {
            x.job->xl[x.n] = std::move(s);
            auto& sd = x.job->xl_side[x.n];
            if (g_lean_perdraw) {
                // Every field, the two vectors by swap: the slot's previous buffers (its last
                // job's; the back read them before it freed the job slot with state = 0, release,
                // and this worker's claim acquired state = 1 after the front re-queued it) become
                // this worker's next ones, capacity kept. Both are cleared below.
                sd.end_pos = x.last_end;
                sd.ctx_epoch = x.last_epoch;
                sd.delta.swap(x.delta);
                sd.render_key = x.open_key;
                sd.pre_pos = x.s_pre_pos;
                sd.first_pos = x.s_first_pos;
                sd.pre_delta.swap(x.s_pre_delta);
                // Review RISK 1: a recycled buffer keeps its largest capacity; drop outliers.
                if (x.delta.capacity() > 4096) {
                    decltype(x.delta){}.swap(x.delta);
                }
                if (x.s_pre_delta.capacity() > 4096) {
                    decltype(x.s_pre_delta){}.swap(x.s_pre_delta);
                }
            } else {
                sd = {x.last_end, x.last_epoch, std::move(x.delta), x.open_key};
                sd.pre_pos = x.s_pre_pos;
                sd.first_pos = x.s_first_pos;
                sd.pre_delta = std::move(x.s_pre_delta);
            }
            x.s_pre_delta.clear();
            x.s_pre_pos = ~u64{0};
            x.job->xl_ready.store(++x.n, std::memory_order_release);
            ++x.stretches;
        }
        x.open = false;
    }
    x.pending.clear();
    x.delta.clear();
}

void LeanXlDraw(LeanXl& x, const CpSplit::RecordHeader& h, u64 pos) {
    const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(&h + 1);
    x.stages = {};
    const u8* stages_end = ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, x.stages);
    Vulkan::A2::DrawIn in{};
    in.pipeline = rec.draw.pipeline;
    in.is_indexed = rec.draw.args.is_indexed;
    in.is_indirect = rec.draw.args.is_indirect;
    in.index_offset = rec.draw.args.index_offset;
    in.stages = &x.stages;
    in.vi = rec.vi_bytes != 0 ? reinterpret_cast<const CpSplit::VertexInputsPayload*>(stages_end)
                              : nullptr;
    in.job_tag = rec.job_tag;
    in.job_idx = rec.job_idx;
    in.job_cmd = rec.cmd_address;
    in.seq = h.seq;
    in.rec_pos = pos;
    in.arg_address = rec.draw.args.arg_address + rec.draw.args.offset;
    ++x.walked;
    // The back census's bucket for this draw if it runs on the old path (job_pad is unused; the
    // record is committed but not yet published).
    auto& why = const_cast<SplitDrawRecord&>(rec).job_pad;
    if (x.occlusion || !in.pipeline || rec.marker || rec.verify != 0) {
        why = 40;
        LeanXlEnd(x);
        return;
    }
    const Regs& regs = *x.regs;
    if (x.open && (x.rt_dirty || in.pipeline != x.rt_pipe)) {
        // Same pass only if the render windows and the pipeline's attachment setup are unchanged
        // since the stretch's first draw.
        bool same = true;
        if (x.rt_dirty) {
            std::array<u32, 0x20 + 0x20 + 0x10 + 0x78> now{};
            A2CopyRenderWindows(regs, now);
            same = now == x.rt_win;
            x.rt_dirty = false;
        }
        if (same && in.pipeline != x.rt_pipe) {
            same = A2PipeRender{in.pipeline} == *x.rt_pkey;
            x.rt_pipe = in.pipeline;
        }
        if (!same) {
            LeanXlEnd(x);
        }
    }
    if (!x.open && x.n >= A2MaxStretches) {
        ++x.cap;
        why = 41;
        return;
    }
    why = 0;
    bool took = x.tr->TakeDraw(regs, in);
    if (!took && x.open && x.tr->last_refusal == Vulkan::A2::RefuseMaterials) {
        // A 9th distinct material: end the stretch here, the draw opens the next one.
        LeanXlEnd(x);
        ++x.mat_cuts;
        took = x.tr->TakeDraw(regs, in);
    }
    if (took) {
        if (!x.open) {
            x.open = true;
            x.rt_dirty = false;
            x.rt_pipe = in.pipeline;
            x.rt_pkey.emplace(in.pipeline);
            A2CopyRenderWindows(regs, x.rt_win);
            x.open_key = g_a2 == 2 ? A2RenderKey(regs, in.pipeline) : 0;
            // The RegWrite run right before this first draw rides with the stretch.
            x.s_first_pos = pos;
            x.s_pre_delta.clear();
            x.s_pre_pos = x.pre_start;
            if (x.pre_start != ~u64{0}) {
                for (const u32 off : x.pre_offs) {
                    x.s_pre_delta.emplace_back(off, regs.reg_array[off]);
                }
            }
            x.pending.clear();
            x.delta.clear();
        }
        // The words written before this draw inside the stretch are the back's to apply (it
        // skips their records); later ones stay pending.
        for (const u32 off : x.pending) {
            x.delta.emplace_back(off, regs.reg_array[off]);
        }
        x.pending.clear();
        x.last_end = pos + h.size;
        x.last_epoch = rec.ctx_epoch;
        ++x.taken;
    } else {
        why = 1 + x.tr->last_refusal;
        LeanXlEnd(x);
    }
}

/// Walks the worker's committed records that start before `end` (unmasked).
void LeanXlWalk(LeanXl& x, u64 end) {
    while (x.walk_pos < end) {
        u64 pos = x.walk_pos;
        const auto* h = x.ch->RecordAtPos(pos); // skips a Wrap
        if (pos >= end) {
            break; // only a Wrap before `end`
        }
        const auto* payload = reinterpret_cast<const u8*>(h + 1);
        if (h->type == CpSplit::RecordType::RegWrite) {
            if (x.pre_start == ~u64{0}) {
                x.pre_start = pos;
            }
            const u8* pend = reinterpret_cast<const u8*>(h) + h->size;
            for (const u8* q = payload; q + sizeof(CpSplit::RegWritePayload) <= pend;) {
                const auto* w = reinterpret_cast<const CpSplit::RegWritePayload*>(q);
                if (w->count == 0) {
                    break;
                }
                for (u32 i = 0; i < w->count; ++i) {
                    x.pre_offs.push_back(w->word_offset + i);
                }
                q += sizeof(CpSplit::RegWritePayload) + size_t{w->count} * sizeof(u32);
            }
        }
        switch (h->type) {
        case CpSplit::RecordType::RegWrite:
            if (x.open) {
                const u8* end = reinterpret_cast<const u8*>(h) + h->size;
                const u8* q = payload;
                while (q + sizeof(CpSplit::RegWritePayload) <= end) {
                    const auto* w = reinterpret_cast<const CpSplit::RegWritePayload*>(q);
                    if (w->count == 0) {
                        break;
                    }
                    for (u32 i = 0; i < w->count; ++i) {
                        x.pending.push_back(w->word_offset + i);
                    }
                    x.rt_dirty = x.rt_dirty || A2TouchesRenderWindows(w->word_offset, w->count);
                    q += sizeof(CpSplit::RegWritePayload) + size_t{w->count} * sizeof(u32);
                }
            }
            break;
        case CpSplit::RecordType::Draw:
            LeanXlDraw(x, *h, pos);
            break;
        case CpSplit::RecordType::Packet:
            if (!A2NoopDma(*h)) {
                const auto* ph = reinterpret_cast<const PM4Header*>(payload);
                if (ph->type3.opcode == PM4ItOpcode::EventWrite) {
                    x.occlusion = true; // the rest of this job stays on the old path
                }
                LeanXlEnd(x);
            }
            break;
        default: // ClearState, Extent, EpochBump, ...: a boundary
            LeanXlEnd(x);
            break;
        }
        if (h->type != CpSplit::RecordType::RegWrite) {
            x.pre_start = ~u64{0}; // the run must end right at the first draw
            x.pre_offs.clear();
        }
        x.walk_pos = pos + h->size;
    }
}

/// Channel::draw_hook: walk the worker's committed records through the Draw record at draw_pos.
void LeanXlDrawHook(void* p, u64 draw_pos) {
    auto& x = *static_cast<LeanXl*>(p);
    LeanXlWalk(x, draw_pos + 1);
    // Lean channels publish by count never: right away when the back waits on this channel,
    // and every 8 draws while it follows this job (each publish ends the open stretch).
    ++x.since_pub;
    const bool waiting = x.job->back_waiting.load(std::memory_order_relaxed) != 0;
    if (waiting ||
        (x.since_pub >= 8 && g_lean_s.back_cur.load(std::memory_order_relaxed) == x.idx)) {
        if (waiting) {
            x.job->back_waiting.store(0, std::memory_order_relaxed);
            g_lean_xl.wait_pubs.fetch_add(1, std::memory_order_relaxed);
        }
        x.ch->PublishNow();
    }
}

/// Channel::pre_publish: end the open stretch before anything new becomes visible.
void LeanXlPrePublish(void* p) {
    auto& x = *static_cast<LeanXl*>(p);
    // Everything about to be visible is walked first: the back may run and release it (and the
    // ring space be reused) before the next Draw commit would have walked it (Opus review).
    LeanXlWalk(x, x.ch->CommittedPos());
    x.cut_pub += x.open ? 1 : 0;
    LeanXlEnd(x);
    x.since_pub = 0;
    x.pre_start = ~u64{0}; // published records are never folded into a later stretch
    x.pre_offs.clear();
}

void Liverpool::LeanStart() {
    auto& s = g_lean_s;
    s.started = true;
    if (g_lean == 2) {
        // The parser's own config gates (Lean::Parser's constructor) + a split with run-ahead.
        const auto flag = [](const char* name) {
            const char* v = std::getenv(name);
            return v && v[0] == '1';
        };
        s.l2 = split_enabled && split_run_ahead && !split_verify && NativeRec::g_segb == 0 &&
               !flag("SHADPS4_DRAW_TAG") && !flag("SHADPS4_DISPATCH_SYNC_STATS") &&
               !SplitHazard::Enabled();
        if (s.l2) {
            for (auto& ch : s.ch) {
                ch = std::make_unique<CpSplit::Channel>(1u << 20);
                ch->quiet = true;
            }
            g_lean_xl.on = g_a2 != 0;
            if (g_lean_xl.on) {
                for (u32 i = 0; i < LeanSlots; ++i) {
                    s.ch[i]->SetPubBatch(~0u);
                    s.jobs[i].xl.resize(A2MaxStretches);
                    s.jobs[i].xl_side.resize(A2MaxStretches);
                }
            }
        }
        LOG_INFO(Render, "Lean L2: {} ({} job channels of 1 MB)", s.l2 ? "on" : "OFF (config)",
                 s.l2 ? LeanSlots : 0u);
        if (s.l2) {
            Lean::Env fe{};
            fe.rasterizer = rasterizer;
            fe.pipeline_cache = &rasterizer->GetPipelineCache();
            fe.split_enabled = split_enabled;
            fe.split_run_ahead = split_run_ahead;
            fe.split_verify = split_verify;
            fe.split_post_dma = split_post_dma;
            fe.split_task_guard = split_task_guard;
            fe.split_front_derive = split_front_derive;
            fe.l2 = true;
            s.toucher = std::make_unique<Lean::Parser>(fe);
        }
    }
    split->canon_fn = &LeanCanonRecord; // the front's records too (canon is set per packet)
    s.regs = std::make_unique<Regs>();
    s.win_tsc = __rdtsc();
    static const u32 n = [] {
        if (const u32 a = Common::AutoPoolWorkers(true)) {
            return a; // SHADPS4_POOL_AUTO
        }
        const char* v = std::getenv("SHADPS4_LEAN_WORKERS");
        const u32 x = v ? static_cast<u32>(std::strtoul(v, nullptr, 10)) : 4u;
        return std::clamp<u32>(x, 1, 8);
    }();
    for (u32 i = 0; i < n; ++i) {
        s.workers.emplace_back(std::bind_front(&Liverpool::LeanWorkerLoop, this), i);
    }
    LOG_INFO(Render, "Lean: {} shadow parse workers started (mode {})", n, g_lean);
}

void Liverpool::LeanWorkerLoop(std::stop_token stoken, u32 index) {
    Common::SetCurrentThreadName(fmt::format("shadPS4:LeanWorker{}", index).c_str());
    Common::ApplyHelperThreadPlacement("LeanWorker");
    t_lean_worker = true; // its faults are counted, never treated as the front's
    auto& s = g_lean_s;
    Lean::Env env{};
    env.rasterizer = rasterizer;
    env.pipeline_cache = &rasterizer->GetPipelineCache();
    env.split_enabled = split_enabled;
    env.split_run_ahead = split_run_ahead;
    env.split_verify = split_verify;
    env.split_post_dma = split_post_dma;
    env.split_task_guard = split_task_guard;
    env.split_front_derive = split_front_derive;
    env.host_markers = EmulatorSettings.IsVkHostMarkersEnabled();
    env.guest_markers = EmulatorSettings.IsVkGuestMarkersEnabled();
    {
        const char* v = std::getenv("SHADPS4_DRAW_TAG"); // = g_draw_tag_front (defined later)
        env.draw_tag_front = v && v[0] == '1';
    }
    env.l2 = g_lean == 2 && s.l2;
    Lean::Parser parser{env};
    // L3: this worker's stretch translator and arena.
    std::unique_ptr<Vulkan::A2::Arena> xl_arena;
    std::unique_ptr<Vulkan::A2::StretchTranslator> xl_tr;
    LeanXl xl;
    if (env.l2 && g_lean_xl.on) {
        xl_arena = std::make_unique<Vulkan::A2::Arena>();
        Vulkan::A2::WorkerContext actx{};
        actx.instance = &rasterizer->A2Instance();
        actx.scheduler = &rasterizer->GetScheduler();
        actx.buffer_cache = &rasterizer->A2BufferCache();
        actx.arena = xl_arena.get();
        actx.worker_index = 16 + index;
        xl_tr = std::make_unique<Vulkan::A2::StretchTranslator>(actx);
        xl.tr = xl_tr.get();
        g_lean_xl.tr[index & 7].store(xl.tr, std::memory_order_release);
    }
    SCOPE_EXIT {
        if (xl.tr) {
            g_lean_xl.tr[index & 7].store(nullptr, std::memory_order_release); // LeanLog reads it
        }
    };
    // L2 fault hook context: publish this job's channel, wait for the back up to its position.
    struct L2Wait {
        CpSplit::Channel* ch{};
        u32 idx{};
    } l2w;
    if (env.l2) {
        t_lean_l2_ctx = &l2w;
        t_lean_l2_wait = [](void* p) {
            auto& w = *static_cast<L2Wait*>(p);
            if (!w.ch) {
                return;
            }
            w.ch->PublishNow();
            const u64 want = w.ch->CommittedSeq();
            const u64 t0 = __rdtsc();
            for (u32 spins = 0; g_lean_s.back_cur.load(std::memory_order_acquire) != w.idx ||
                                w.ch->DoneSeq() < want;
                 ++spins) {
                spins < 4096 ? _mm_pause() : std::this_thread::yield();
                if (g_lean_s.stop.load(std::memory_order_relaxed)) {
                    break;
                }
            }
            g_lean_l2.wait_ticks.fetch_add(__rdtsc() - t0, std::memory_order_relaxed);
        };
    }
    CpSplit::Channel ch; // scratch: nobody reads it
    ch.self_consume = true;
    ch.quiet = true;
    ch.canon_fn = &LeanCanonRecord;
    Lean::ChainState st;
    st.regs = std::make_unique<Regs>();
    const auto stopping = [&] {
        return stoken.stop_requested() || s.stop.load(std::memory_order_acquire);
    };
    while (!stopping()) {
        const u32 idx = s.next_claim.fetch_add(1, std::memory_order_acq_rel);
        const u64 tw0 = __rdtsc(); // claim wait: until the job is queued (or found not taken)
        if (g_claim_wake && !s.bells.Wait(s.queued, idx, stopping)) {
            return;
        }
        for (u32 spins = 0; s.queued.load(std::memory_order_acquire) < idx;) {
            if (stopping()) {
                return;
            }
            if (++spins < 2048) {
                _mm_pause();
            } else {
                const u32 seen = s.queued.load(std::memory_order_acquire);
                if (seen < idx) {
                    s.queued.wait(seen, std::memory_order_acquire);
                }
            }
        }
        const u64 tw1 = __rdtsc();
        g_lean_phase.claim_wait += tw1 - tw0;
        Common::FaultMonitor::SampleEmuPlacement(3);
        auto& j = s.jobs[idx % LeanSlots];
        if (j.index.load(std::memory_order_acquire) != idx ||
            j.state.load(std::memory_order_acquire) != 1) {
            continue; // not queued (slot busy / not eligible)
        }
        const u32* w = j.win.data();
        for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
            std::memcpy(&st.regs->reg_array[off], w, size_t{n} * 4);
            w += n;
        }
        st.cb_extent = j.cb;
        st.db_extent = j.db;
        st.ctx_epoch = j.epoch;
        st.cs_state = j.cs;
        st.indirect_args_addr = j.iaa;
        st.vi_desc_id = (u64{1} << 63) | (u64{idx} << 20);
        const u64 ts1 = __rdtsc();
        g_lean_phase.seed += ts1 - tw1;
        if (env.l2) {
            // L2: the real channel of this slot; the back follows it from the DcbRef.
            auto& out = *s.ch[idx % LeanSlots];
            l2w = {&out, idx};
            LeanPlaceSample(g_lean_xl.lean_p, g_lean_xl.lean_e);
            if (xl.tr) {
                xl.job = &j;
                xl.ch = &out;
                xl.idx = idx;
                xl.regs = st.regs.get();
                xl.walk_pos = out.CommittedPos();
                xl.n = 0;
                xl.open = xl.occlusion = xl.rt_dirty = false;
                xl.rt_pipe = nullptr;
                xl.rt_pkey.reset();
                xl.pending.clear();
                xl.delta.clear();
                xl.since_pub = 0;
                xl.pre_start = ~u64{0};
                xl.pre_offs.clear();
                xl.s_pre_pos = ~u64{0};
                xl.s_pre_delta.clear();
                j.back_waiting.store(0, std::memory_order_relaxed);
                out.hook_ctx = &xl;
                out.draw_hook = &LeanXlDrawHook;
                out.pre_publish = &LeanXlPrePublish;
            }
            j.res = parser.Parse(j.dcb, j.cut_dw, st, out);
            g_lean_phase.parse += __rdtsc() - ts1;
            parser.FlushPhases(index);
            if (j.res.end_dw != j.cut_dw) {
                // An early stop (gate 0): finish to the cut in register-only mode so the back's
                // mirror reaches the front's state at the cut; the draws in between are dropped.
                if (s.l2_early.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_CRITICAL(Render, "Lean L2 EARLY STOP: DCB {} stop {} at dw {} (opcode {:#x}), "
                                         "cut {} of {} dw",
                                 idx, Lean::StopName(j.res.stop), j.res.end_dw,
                                 j.res.stop_opcode, j.cut_dw, j.dcb.size());
                }
                parser.SetRegsOnly(true, j.res.end_dw);
                const auto rest = parser.Parse(j.dcb, j.cut_dw, st, out);
                parser.SetRegsOnly(false, 0);
                g_lean_l2.regs_only_runs.fetch_add(1, std::memory_order_relaxed);
                g_lean_l2.dropped_draws.fetch_add(rest.dropped, std::memory_order_relaxed);
                if (rest.end_dw != j.cut_dw) {
                    LOG_CRITICAL(Render, "Lean L2: register-only completion of DCB {} stopped at {} "
                                         "({}), cut {}",
                                 idx, rest.end_dw, Lean::StopName(rest.stop), j.cut_dw);
                }
            }
            if (xl.tr) {
                // End the last stretch and leave the channel BEFORE the LeanEnd publish: the back
                // may free the slot right after it (Opus review).
                LeanXlWalk(xl, out.CommittedPos());
                LeanXlEnd(xl);
                out.draw_hook = nullptr;
                out.pre_publish = nullptr;
                out.hook_ctx = nullptr;
                xl.job = nullptr;
                auto& gx = g_lean_xl;
                gx.taken.fetch_add(std::exchange(xl.taken, 0), std::memory_order_relaxed);
                gx.stretches.fetch_add(std::exchange(xl.stretches, 0), std::memory_order_relaxed);
                gx.cap.fetch_add(std::exchange(xl.cap, 0), std::memory_order_relaxed);
                gx.cut_pub.fetch_add(std::exchange(xl.cut_pub, 0), std::memory_order_relaxed);
                gx.walked.fetch_add(std::exchange(xl.walked, 0), std::memory_order_relaxed);
                gx.mat_cuts.fetch_add(std::exchange(xl.mat_cuts, 0), std::memory_order_relaxed);
            }
            out.Reserve(CpSplit::RecordType::LeanEnd, 0);
            out.Commit();
            out.PublishNow();
            l2w = {};
            continue; // the back frees the slot at the LeanEnd
        }
        ch.canon = &j.worker_canon;
        j.res = parser.Parse(j.dcb, j.cut_dw, st, ch);
        ch.canon = nullptr;
        // L1 lag: the back already ran (part of) this DCB, so the game may have rewritten it.
        j.lag.store(XXH3_64bits(j.dcb.data(), j.dcb.size_bytes()) != j.dcb_hash,
                    std::memory_order_release);
        if (j.done_mask.fetch_or(2, std::memory_order_acq_rel) == 1) {
            LeanCompare(j);
        }
    }
}

void Liverpool::A2Shutdown() {
    if (g_lean_s.started) {
        auto& s = g_lean_s;
        s.stop.store(true, std::memory_order_release);
        s.queued.fetch_add(0x40000000u, std::memory_order_acq_rel);
        s.queued.notify_all();
        if (g_claim_wake) {
            s.bells.RingAll();
        }
        for (auto& w : s.workers) {
            w.request_stop();
        }
        s.workers.clear(); // joins
        s.started = false;
    }
    auto& g = g_a2s;
    if (!g.started) {
        return;
    }
    g.stop.store(true, std::memory_order_release);
    g.begun.fetch_add(0x40000000u, std::memory_order_acq_rel); // wake every waiter
    g.begun.notify_all();
    if (g_claim_wake) {
        g.bells.RingAll();
    }
    split->reader_bell.fetch_add(1, std::memory_order_seq_cst);
    split->reader_bell.notify_all();
    for (auto& t : g.workers) {
        t.request_stop();
    }
    g.workers.clear(); // joins
}

void Liverpool::A2Execute(const CpSplit::RecordHeader& record) {
    using RT = CpSplit::RecordType;
    auto& g = g_a2s;
    auto& c = g.c;
    if (record.type != RT::Dispatch && rasterizer &&
        Vulkan::Rasterizer::EarlyCutMode() == 2) [[unlikely]] {
        rasterizer->MaybeEarlyCut(true); // q7b: the dispatch run that wrote a hot window ended
    }
    if (!g.started && rasterizer) [[unlikely]] {
        g.started = true;
        c.start_ns = A2NowNs();
        c.start_tsc = __rdtsc();
        g.t_start = c.start_ns;
        for (auto& entry : g.e) {
            entry.stretches.resize(A2MaxStretches);
            entry.side.resize(A2MaxStretches);
        }
        const u32 n = Vulkan::A2::NumWorkers();
        for (u32 i = 0; i < n; ++i) {
            g.workers.emplace_back(std::bind_front(&Liverpool::A2WorkerLoop, this), i);
        }
        LOG_INFO(Render, "A2: {} stretch workers started (mode {})", n, g_a2);
    }
    const auto release_cur = [&] {
        if (g.cur) {
            auto& e = *g.cur;
            if (!g.cur_back) {
                // The worker may still walk the tail records (non-draw records never wait on
                // it): the slot stays ours until it is done. Bounded: every record through this
                // DcbEnd is published.
                const u64 t0 = __rdtsc();
                bool waited = false;
                for (u32 spins = 0;; ++spins) {
                    u32 st = e.state.load(std::memory_order_acquire);
                    if (st == 1) {
                        if (e.state.compare_exchange_strong(st, 4, std::memory_order_acq_rel)) {
                            break;
                        }
                        continue;
                    }
                    if (st != 2) {
                        break;
                    }
                    waited = true;
                    if (spins < 4096) {
                        _mm_pause();
                    } else {
                        e.state.wait(2, std::memory_order_acquire);
                    }
                }
                if (waited) {
                    ++c.end_waits;
                    c.end_wait_cyc += __rdtsc() - t0;
                }
            }
            const u32 n = e.ready.load(std::memory_order_acquire);
            for (u32 i = 0; i < n; ++i) {
                auto& s = e.stretches[i];
                if (s && s->tick && s->tick->load(std::memory_order_relaxed) == 0) {
                    s->tick->store(1, std::memory_order_release); // never published: free
                }
                s.reset();
            }
            e.ready.store(0, std::memory_order_relaxed);
            e.state.store(0, std::memory_order_release);
            g.cur = nullptr;
        }
    };
    switch (record.type) {
    case RT::DcbBegin: {
        release_cur();
        const auto& bp = *reinterpret_cast<const CpSplit::DcbBeginPayload*>(&record + 1);
        const u32 idx = bp.index;
        if (g_a2 == 2 && bp.snap_words == CpSplit::DcbSnapWords) {
            // The entry-state verify: the front's windows = the back's mirror here.
            const u32* fs = reinterpret_cast<const u32*>(&bp + 1);
            bool bad = false;
            for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
                bad = bad || std::memcmp(fs, &mirror->regs.reg_array[off], size_t{n} * 4) != 0;
                fs += n;
            }
            ++c.snap_cmp;
            c.snap_bad += bad;
        }
        auto& e = g.e[idx % A2Slots];
        g.cur_back = false;
        g.stretch_i = 0;
        g.reject_cur = false;
        g.seen_progress = 0;
        g.cur_n = 0;
        ++c.dcbs;
        if (e.index.load(std::memory_order_acquire) != idx ||
            e.state.load(std::memory_order_acquire) == 0) {
            // Cannot happen: the front writes the entry before the DcbBegin record is
            // published. Old path, and the slot is not ours to release.
            ++c.whole_stale;
            g.cur_back = true;
            g.cur = nullptr;
            ++c.back_dcbs;
            return;
        }
        {
            // Begun but not yet claimed by a worker: wait for one unless the front is waiting
            // on the back (it may never publish more) or the safety net passes; then the back
            // takes it below.
            const u64 t0 = __rdtsc();
            constexpr u64 UnclaimedTicks = 175'000'000; // ~50 ms
            bool waited = false;
            for (u32 spins = 0; e.state.load(std::memory_order_acquire) == 1; ++spins) {
                if (split->FrontWaitingOnBack()) {
                    ++c.steal_front_waits;
                    break;
                }
                if (__rdtsc() - t0 > UnclaimedTicks) {
                    ++c.steal_timeout;
                    std::string ws;
                    for (u32 w = 0; w < Vulkan::A2::NumWorkers() && w < 8; ++w) {
                        const u32 at = g.worker_at[w].load(std::memory_order_relaxed);
                        ws += fmt::format(" w{}={}/{}", w, at & 0x0fffffffu, at >> 28);
                    }
                    LOG_WARNING(Render,
                                "A2 safety net (50 ms, unclaimed): dcb {} - begun {} next_claim "
                                "{} front waiting {}; workers (dcb/phase 1 wait-dcb 2 translate "
                                "3 caught-up):{}",
                                idx, g.begun.load(std::memory_order_relaxed),
                                g.next_claim.load(std::memory_order_relaxed),
                                split->FrontWaitingOnBack(), ws);
                    break;
                }
                waited = true;
                spins < 4096 ? _mm_pause() : std::this_thread::yield();
            }
            if (waited) {
                ++c.wait_complete;
                c.wait_complete_cyc += __rdtsc() - t0;
            }
        }
        for (;;) {
            u32 st = e.state.load(std::memory_order_acquire);
            if (st == 1) {
                if (e.state.compare_exchange_strong(st, 4, std::memory_order_acq_rel)) {
                    g.cur_back = true;
                    ++c.steal_unclaimed;
                    break;
                }
                continue;
            }
            // 2: a worker follows the front through it; 3: the worker finished it.
            g.cur_back = st == 4;
            c.whole_refused += st == 3 && e.refused != 0;
            c.whole_st4 += st == 4;
            break;
        }
        g.cur = &e;
        ++(g.cur_back ? c.back_dcbs : c.worker_dcbs);
        if (c.dcbs >= 65536) {
            A2Log(rasterizer->a2_slot_cmp, rasterizer->a2_slot_bad, rasterizer->a2_expect.size());
        }
        return;
    }
    case RT::DcbEnd:
        release_cur();
        return;
    case RT::Draw: {
        if (g.cur && !g.cur_back && record.seq > g.seen_progress) {
            auto& e = *g.cur;
            u64 p = e.progress.load(std::memory_order_acquire);
            if (p < record.seq) [[unlikely]] {
                // The worker following this DCB has not decided this draw yet (it is behind,
                // or the draw opens its stretch: it hands it over when it sees back_wait).
                const u64 t0 = __rdtsc();
                const bool behind =
                    e.walked.load(std::memory_order_acquire) < split->PublishedEnd();
                e.back_wait.store(record.seq, std::memory_order_seq_cst);
                split->RingReaderBell(); // a parked worker must hand the stretch over
                bool netted = false;
                for (u32 spins = 0;
                     (p = e.progress.load(std::memory_order_acquire)) < record.seq; ++spins) {
                    if (!netted && __rdtsc() - t0 > 175'000'000) {
                        // Safety net (~50 ms; a scheduling bug if it fires in normal play): the
                        // worker stops and the rest of the DCB runs on the old path.
                        netted = true;
                        ++c.net_cancel;
                        std::string ws;
                        for (u32 w = 0; w < Vulkan::A2::NumWorkers() && w < 8; ++w) {
                            const u32 at = g.worker_at[w].load(std::memory_order_relaxed);
                            ws += fmt::format(" w{}={}/{}", w, at & 0x0fffffffu, at >> 28);
                        }
                        LOG_WARNING(Render,
                                    "A2 safety net (50 ms, draw): dcb {} draw seq {} progress {} "
                                    "ready {} state {} walked {} published end {} front waiting "
                                    "{}; workers (dcb/phase):{}",
                                    e.index.load(std::memory_order_relaxed), record.seq, p,
                                    e.ready.load(std::memory_order_relaxed),
                                    e.state.load(std::memory_order_relaxed),
                                    e.walked.load(std::memory_order_relaxed),
                                    split->PublishedEnd(), split->FrontWaitingOnBack(), ws);
                        e.cancel.store(1, std::memory_order_release);
                    }
                    spins < 4096 ? _mm_pause() : std::this_thread::yield();
                }
                e.back_wait.store(0, std::memory_order_release);
                const u64 dt = __rdtsc() - t0;
                ++c.waits;
                c.wait_cyc += dt;
                if (behind) {
                    ++c.dwait_behind;
                    c.dwait_behind_cyc += dt;
                }
            }
            g.seen_progress = p;
            g.cur_n = e.ready.load(std::memory_order_acquire);
        }
        if (g.cur && !g.cur_back && g.stretch_i < g.cur_n) {
            auto& s = *g.cur->stretches[g.stretch_i];
            if (record.seq >= s.first_seq && record.seq <= s.last_seq) {
                const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(&record + 1);
                if (record.seq == s.first_seq) {
                    const bool occl = rasterizer->A2OcclusionOpen();
                    // A buffer created / deleted / joined since the worker's lookups: its VkBuffer
                    // handles may be stale (A2LookupBuffer reads slots without a lock).
                    const bool gen_moved =
                        rasterizer->A2BufferCache().StructGeneration() != s.buf_gen;
                    g.reject_cur = g_a2 == 2 || occl || gen_moved;
                    c.occl_rejects += g_a2 == 1 && occl;
                    c.gen_rejects += g_a2 == 1 && gen_moved && !occl;
                    // Colour: every distinct material once, before the pass (sec. 10).
                    static thread_local std::vector<VideoCore::ImageId> mat_ids;
                    mat_ids.clear();
                    g.v_stretch = nullptr;
                    const bool colour = !s.materials.empty();
                    if (colour && (!g.reject_cur || (g_a2 == 2 && !occl && !gen_moved))) {
                        if (g_a2 == 1) {
                            rasterizer->GetScheduler().SwitchStream(0);
                        }
                        const u64 r0 = __rdtsc();
                        const u32 set0 = rasterizer->A2BindlessCur();
                        bool ok = s.materials.size() <= Vulkan::A2::MaxMaterials &&
                                  rasterizer->A2ColourGates();
                        {
                            // Samplers resolve with the mirror's border-colour base at this
                            // draw: the stretch must not change it.
                            const u32 bc0 = static_cast<u32>(
                                reinterpret_cast<const u32*>(&mirror->regs.ta_bc_base) -
                                mirror->regs.reg_array.data());
                            const u32 bc1 = bc0 + static_cast<u32>(
                                (sizeof(mirror->regs.ta_bc_base) + 3) / 4);
                            for (const auto& [off, val] : g.cur->side[g.stretch_i].delta) {
                                ok = ok && (off < bc0 || off >= bc1);
                            }
                        }
                        for (size_t m = 0; ok && m < s.materials.size(); ++m) {
                            const auto& mr = s.materials[m];
                            u64 mp = mr.rec_pos;
                            const auto* mh = split->RecordAtPos(mp);
                            ok = mh->type == CpSplit::RecordType::Draw;
                            if (!ok) {
                                break;
                            }
                            const auto& mrec = *reinterpret_cast<const SplitDrawRecord*>(mh + 1);
                            std::array<Shader::BoundStage, Shader::MaxStageTypes> mst{};
                            ReadStages(reinterpret_cast<const u8*>(&mrec + 1), mrec.stage_count, mst);
                            auto& [ni, ns] = g.v_mat_n[m];
                            ni = ns = 0;
                            ok = rasterizer->A2ResolveMaterial(
                                mr.base, mst[u32(Shader::LogicalStage::Fragment)],
                                s.cmds->mat_slots[m].data(), ni, ns, mat_ids, 0, 0, 0,
                                &mst[u32(Shader::LogicalStage::Vertex)]);
                        }
                        // A rotation mid-loop invalidates the earlier materials' slots.
                        ok = ok && rasterizer->A2BindlessCur() == set0;
                        c.col_resolve_cyc += __rdtsc() - r0;
                        ++c.col_stretches;
                        c.materials += s.materials.size();
                        if (ok && g_a2 == 1) {
                            s.cmds->bindless_set = rasterizer->A2BindlessCommit();
                        } else if (!ok) {
                            c.mat_rejects += g_a2 == 1;
                            g.reject_cur = true;
                        }
                        if (ok && g_a2 == 2) {
                            auto& ex = rasterizer->a2_expect;
                            if (ex.size() > 200000) {
                                c.slot_unwritten += ex.size(); // never written: stale
                                ex.clear();
                            }
                            for (const auto& sw : s.cmds->slot_writes) {
                                if (sw.mat < s.materials.size()) {
                                    auto& e2 = ex[u64{sw.tag} << 32 | sw.idx];
                                    e2.slots = s.cmds->mat_slots[sw.mat];
                                    e2.n_img = g.v_mat_n[sw.mat].first;
                                    e2.n_smp = g.v_mat_n[sw.mat].second;
                                }
                            }
                        }
                    }
                    if (!g.reject_cur) {
                        rasterizer->GetScheduler().SwitchStream(0);
                        if (g.gpu_write) {
                            rasterizer->A2GlobalBarrier();
                            g.gpu_write = false;
                            ++c.barriers;
                        }
                        mirror->ctx_epoch = rec.ctx_epoch;
                        rasterizer->A2BeginStretch(rec.draw.pipeline);
                    }
                    if (!g.reject_cur && colour && !rasterizer->A2ImagesSafe(mat_ids)) {
                        // A sampled image is a target of this pass: the old path (per-draw
                        // barriers) runs the stretch; the pass it began is reused.
                        ++c.rt_rejects;
                        g.reject_cur = true;
                    }
                    if (!g.reject_cur) {
                        auto& sched = rasterizer->GetScheduler();
                        sched.Record([cmds = s.cmds, inst = &rasterizer->A2Instance()](
                                         vk::CommandBuffer cmdbuf) {
                            Vulkan::A2::Replay(*cmds, cmdbuf, *inst);
                        });
                        s.tick->store(sched.CurrentTick(), std::memory_order_release);
                        if (VideoCore::g_lane_probe) [[unlikely]] {
                            if (s.probe_over || s.probe_gen != VideoCore::g_lprobe_gen.load(
                                                                  std::memory_order_relaxed)) {
                                VideoCore::LaneProbeReadAll(sched.CurrentTick(), s.probe_over ? 26u : 18u); // q9b; q32 site
                            } else {
                                for (const auto& [plo, phi] : s.probe_reads) {
                                    VideoCore::LaneProbeRead(plo, phi, sched.CurrentTick(), 9, 9);
                                }
                            }
                        }
                        rasterizer->A2EndStretch();
                        // The stretch's register writes (its records are never visited), after the
                        // pass was resolved from the mirror at its first draw.
                        const auto& sd = g.cur->side[g.stretch_i];
                        for (const auto& [off, val] : sd.delta) {
                            mirror->regs.reg_array[off] = val;
                        }
                        mirror->ctx_epoch = sd.ctx_epoch;
                        c.delta_words += sd.delta.size();
                        ++c.stretches;
                        c.walked += s.draws;
                        ++c.back_records;
                        g.skip_pos = sd.end_pos;
                        g.skip_seq = s.last_seq;
                        c.skipped_records += s.last_seq - record.seq;
                        ++g.stretch_i;
                        return;
                    }
                }
                if (g_a2 == 2) {
                    // The shortcut check: the full key from the BACK's mirror at this draw.
                    const u64 bk = A2RenderKey(mirror->regs, rec.draw.pipeline);
                    ++c.rk_cmp;
                    if (bk != g.cur->side[g.stretch_i].render_key) {
                        if (++c.rk_bad <= 16) {
                            LOG_WARNING(Render,
                                        "A2 verify: render key mismatch at draw seq {} "
                                        "(stretch {}..{})",
                                        record.seq, s.first_seq, s.last_seq);
                        }
                    }
                }
                if (record.seq == s.last_seq) {
                    ++g.stretch_i;
                }
                // Rejected / verify: the old path runs the draw.
                c.would += g_a2 == 2 ? 1 : 0;
                if (false && g.v_stretch == &s && g.v_slot_k < s.cmds->slot_writes.size()) {
                    const auto& sw = s.cmds->slot_writes[g.v_slot_k];
                    if (sw.tag == rec.job_tag && sw.idx == rec.job_idx) {
                        ++g.v_slot_k;
                        const u64 n0 = rasterizer->a2_probe.n;
                        ++c.translated;
                        ++c.back_records;
                        ExecuteRecordImpl(record);
                        const auto& pr = rasterizer->a2_probe;
                        if (pr.n == n0 || pr.tag != sw.tag || pr.idx != sw.idx) {
                            ++c.slot_unwritten;
                        } else {
                            const auto& want = s.cmds->mat_slots[sw.mat];
                            const auto [ni, ns] = g.v_mat_n[sw.mat];
                            bool bad = sw.mat >= s.materials.size();
                            for (u32 i = 0; !bad && i < ni; ++i) {
                                bad = pr.slots[i] != want[i];
                            }
                            for (u32 j = 0; !bad && j < ns; ++j) {
                                bad = pr.slots[Shader::g_bl_img_max + j] !=
                                      want[Shader::g_bl_img_max + j];
                            }
                            ++c.slot_cmp;
                            if (bad && ++c.slot_bad <= 16) {
                                LOG_WARNING(Render,
                                            "A2 verify: slot words differ at draw seq {} (material "
                                            "{} of {}, {} images {} samplers)",
                                            record.seq, sw.mat, s.materials.size(), ni, ns);
                            }
                        }
                        return;
                    }
                }
            }
        }
        ++c.translated;
        ++c.back_records;
        ExecuteRecordImpl(record);
        return;
    }
    case RT::Dispatch:
    case RT::Call:
        g.gpu_write = true;
        ++c.back_records;
        ExecuteRecordImpl(record);
        return;
    case RT::Packet:
        if (!A2NoopDma(record)) {
            g.gpu_write = true;
        }
        ++c.back_records;
        ExecuteRecordImpl(record);
        return;
    default:
        ++c.back_records;
        ExecuteRecordImpl(record);
        return;
    }
}

void Liverpool::ExecuteRecord(const CpSplit::RecordHeader& record) {
    if (g_a2 != 0) [[unlikely]] {
        static const bool blc = Vulkan::BackCyclesEnabled();
        if (blc && record.type != CpSplit::RecordType::DcbRef) [[unlikely]] {
            const u64 t0 = __rdtsc();
            A2Execute(record);
            const u32 t = static_cast<u32>(record.type) & 31;
            const u64 dt = __rdtsc() - t0;
            g_blc.main_cyc[t] += dt;
            ++g_blc.main_n[t];
            if (record.type == CpSplit::RecordType::Packet) {
                // q4: by opcode (the record's payload starts with the PM4 header)
                const auto* ph = reinterpret_cast<const PM4Header*>(&record + 1);
                const u32 op = ph->type == 3 ? static_cast<u32>(ph->type3.opcode.Value()) & 0xff : 0;
                g_blc.pk_cyc[op] += dt;
                ++g_blc.pk_n[op];
            }
            BackLeanMaybeLog();
            return;
        }
        A2Execute(record);
        return;
    }
    if (g_a1 != 0) [[unlikely]] {
        A1Execute(record);
        return;
    }
    static const bool back_cycles = Vulkan::BackCyclesEnabled();
    if (back_cycles) [[unlikely]] {
        const u64 t0 = __rdtsc();
        ExecuteRecordImpl(record);
        BackCyclesNote(static_cast<u32>(record.type), __rdtsc() - t0,
                       rasterizer ? &rasterizer->GetScheduler() : nullptr);
        return;
    }
    if (!AmdGpu::MtbCensus::g_enabled) [[likely]] {
        ExecuteRecordImpl(record);
        return;
    }
    namespace MC = AmdGpu::MtbCensus;
    const bool is_work = record.type == CpSplit::RecordType::Draw ||
                         record.type == CpSplit::RecordType::Dispatch;
    if (is_work && g_mtb.seg_draws >= g_mtb_segment) {
        MtbEndSegment();
    }
    MC::g_draw_event = MC::None;
    MC::g_draw_dependent = false;
    MC::g_draw_events.clear();
    Vulkan::Scheduler* const sched = rasterizer ? &rasterizer->GetScheduler() : nullptr;
    const bool rend_in = sched && sched->IsRenderingNow();
    const u64 tick_in = sched ? sched->GetMasterSemaphore()->CurrentTickRaw() : 0;
    const u64 reads_in = Vulkan::MasterSemaphore::t_tick_reads;
    const u64 t0 = __rdtsc();
    ExecuteRecordImpl(record);
    const u64 cyc = __rdtsc() - t0;
    const bool rend_out = sched && sched->IsRenderingNow();
    const u64 tick_out = sched ? sched->GetMasterSemaphore()->CurrentTickRaw() : 0;
    g_mtb.submits += tick_out - tick_in;
    const auto note_seqpoint = [&](u32 kind) {
        ++g_mtb.sp_total[kind];
        if (tick_out == tick_in) {
            ++g_mtb.sp_in_batch[kind];
            g_mtb.sp_pass_kept[kind] += rend_in && rend_out ? 1 : 0;
        }
        g_mtb.sp_pass_ended[kind] += rend_in && !rend_out ? 1 : 0;
    };
    switch (record.type) {
    case CpSplit::RecordType::Draw:
    case CpSplit::RecordType::Dispatch: {
        ++g_mtb.draws;
        ++g_mtb.seg_draws;
        g_mtb.tick_reads += Vulkan::MasterSemaphore::t_tick_reads - reads_in;
        if (MC::g_draw_dependent) {
            ++g_mtb.dependents;
            if (!g_mtb.seg_dependent) {
                g_mtb.seg_dependent = true;
                ++g_mtb.segments_dependent;
            }
        }
        g_mtb.tail_draws += g_mtb.seg_dependent ? 1 : 0;
        MC::g_seg_events.insert(MC::g_seg_events.end(), MC::g_draw_events.begin(),
                                MC::g_draw_events.end());
        const u32 ev = MC::g_draw_event;
        if (!g_mtb.seg_bailed && ev == MC::None) {
            ++g_mtb.par_draws;
            g_mtb.par_cyc += cyc;
        } else {
            if (!g_mtb.seg_bailed) {
                g_mtb.seg_bailed = true;
                ++g_mtb.bails[ev];
            }
            g_mtb.ser_draw_cyc += cyc;
        }
        if ((g_mtb.draws & 0x7fffff) == 0) {
            MtbLog();
        }
        break;
    }
    case CpSplit::RecordType::Packet: {
        // The DMA_DATA to 0x3022C is skipped by ExecuteGfxPacket: not a dependency.
        const auto* header = reinterpret_cast<const PM4Header*>(&record + 1);
        const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
        if (header->type3.opcode == PM4ItOpcode::DmaData && dma->dst_addr_lo == 0x3022C) {
            ++g_mtb.dma_noop;
            g_mtb.other_cyc += cyc;
            break;
        }
        g_mtb.seq_cyc += cyc;
        ++g_mtb.seqpoints;
        MtbEndSegment();
        if (record.flags & CpSplit::AscReleaseMem) {
            note_seqpoint(4);
        } else {
            switch (header->type3.opcode) {
            case PM4ItOpcode::WriteData:
                note_seqpoint(0);
                break;
            case PM4ItOpcode::EventWriteEop:
                note_seqpoint(1);
                break;
            case PM4ItOpcode::EventWriteEos:
                note_seqpoint(2);
                break;
            case PM4ItOpcode::DmaData:
                note_seqpoint(3);
                break;
            default:
                note_seqpoint(4);
                break;
            }
        }
        break;
    }
    case CpSplit::RecordType::Call:
        g_mtb.seq_cyc += cyc;
        ++g_mtb.seqpoints;
        MtbEndSegment();
        note_seqpoint(5);
        break;
    case CpSplit::RecordType::Retire:
    case CpSplit::RecordType::ClearState:
    case CpSplit::RecordType::EpochBump:
        // In-order bookkeeping the committer does; a worker's translation does not read it
        // (ClearState resets the register mirror, which each worker applies itself).
        ++g_mtb.bookkeeping;
        g_mtb.other_cyc += cyc;
        break;
    default:
        g_mtb.other_cyc += cyc;
        break;
    }
    MC::g_draw_event = MC::None;
}

void Liverpool::ExecuteRecordImpl(const CpSplit::RecordHeader& record) {
    const auto* payload = reinterpret_cast<const u8*>(&record + 1);
    switch (record.type) {
    case CpSplit::RecordType::RegWrite: {
        const u8* end = reinterpret_cast<const u8*>(&record) + record.size;
        const u8* p = payload;
        while (p + sizeof(CpSplit::RegWritePayload) <= end) {
            const auto* write = reinterpret_cast<const CpSplit::RegWritePayload*>(p);
            if (write->count == 0) {
                break;
            }
            std::memcpy(&mirror->regs.reg_array[write->word_offset], write + 1,
                        size_t{write->count} * sizeof(u32));
            p += sizeof(CpSplit::RegWritePayload) + size_t{write->count} * sizeof(u32);
        }
        break;
    }
    case CpSplit::RecordType::ClearState:
        mirror->regs.SetDefaults();
        break;
    case CpSplit::RecordType::Extent: {
        const auto* extent = reinterpret_cast<const CpSplit::ExtentPayload*>(payload);
        if (extent->slot < NUM_COLOR_BUFFERS) {
            mirror->cb_extent[extent->slot].raw = extent->raw;
        } else {
            mirror->db_extent.raw = extent->raw;
        }
        break;
    }
    case CpSplit::RecordType::EpochBump: {
        const auto* epoch = reinterpret_cast<const CpSplit::EpochPayload*>(payload);
        VideoCore::BumpSubmitEpoch(static_cast<PM4ItOpcode>(epoch->opcode));
        static const bool mtb = Vulkan::MtbP1Requested() || g_mtb_p3 != 0;
        if (VideoCore::g_native_coherent != 0 && rasterizer && !mtb) {
            // (Never under MTB P1 / P3: a worker could skip-bind past a point the committer has not
            // run yet; Opus review F. Without points the skip never goes live.)
            // Milestone 1 (native-path sec. 17): memory coherence once per submission. A
            // submission start (Nop) is bounded by the tail the front read; other points (memory
            // wait satisfied, CP-side guest write) take everything logged so far.
            const auto op = static_cast<PM4ItOpcode>(epoch->opcode);
            rasterizer->GetBufferCache().CoherenceSync(op == PM4ItOpcode::Nop, epoch->pad);
        }
        break;
    }
    case CpSplit::RecordType::Draw:
        rasterizer->GetScheduler().SwitchStream(0);
        if (g_native_timing_rec) [[unlikely]] {
            // SHADPS4_NATIVE_TIMING: whole draw records by what the back did with them (sec. 9
            // gate 0: the per-record path around a join / confirm, measured, not subtracted).
            auto& tr = rasterizer->Translating();
            tr.rec_outcome = 0;
            const u64 t0 = __rdtsc();
            ExecuteDrawRecord(record);
            tr.NoteRecordCycles(__rdtsc() - t0);
        } else {
            ExecuteDrawRecord(record);
        }
        break;
    case CpSplit::RecordType::Segment:
        rasterizer->GetScheduler().SwitchStream(0);
        ExecuteSegmentRecord(record);
        break;
    case CpSplit::RecordType::Dispatch:
        ExecuteDispatchRecord(record);
        break;
    case CpSplit::RecordType::Packet:
        // SHADPS4_ASC_STREAM: packets (DMA, WRITE_DATA, fences, flips) record on the main stream.
        // SHADPS4_ASC_LANE: an early lane packet records on the lane (t_lane_packet).
        if (rasterizer) {
            auto& sch = rasterizer->GetScheduler();
            sch.SwitchStream(t_lane_packet ? 1 : 0);
            if (t_lane_packet) {
                // DMA_DATA: rule A and the WAR wait are taken in CopyBuffer / FillBuffer.
            } else if (VideoCore::g_lane_on && !VideoCore::t_force_binding_sync &&
                       !(record.flags & CpSplit::AscReleaseMem)) {
                // A gfx EOP label: the game may act on it before the GPU ran this batch.
                const auto* h = reinterpret_cast<const PM4Header*>(payload);
                const auto op = h->type3.opcode.Value();
                if (h->type == 3 && (op == PM4ItOpcode::ReleaseMem ||
                                     op == PM4ItOpcode::EventWriteEop ||
                                     op == PM4ItOpcode::EventWriteEos)) {
                    VideoCore::g_gfx_label_tick = sch.CurrentTick();
                    if (VideoCore::g_lane_frame_scope && op == PM4ItOpcode::EventWriteEop) {
                        // q60: the end-of-frame label's value -> its main tick.
                        const auto* eop = reinterpret_cast<const PM4CmdEventWriteEop*>(h);
                        if (reinterpret_cast<u64>(eop->Address<u32>()) == 0x248000000ULL) {
                            VideoCore::NoteFrameEndLabel(eop->DataDWord(), sch.CurrentTick());
                        }
                    }
                    if (VideoCore::g_lane_wait_census) [[unlikely]] {
                        // q59: the game's frame-end label batches (frame age of lane waits).
                        const u64 addr =
                            op == PM4ItOpcode::EventWriteEop
                                ? reinterpret_cast<u64>(reinterpret_cast<const PM4CmdEventWriteEop*>(h)->Address<u32>())
                            : op == PM4ItOpcode::EventWriteEos
                                ? reinterpret_cast<u64>(reinterpret_cast<const PM4CmdEventWriteEos*>(h)->Address())
                                : u64{reinterpret_cast<const PM4CmdReleaseMem*>(h)->Address<VAddr>()};
                        VideoCore::LaneWaitFrameLabel(addr, sch.CurrentTick());
                    }
                }
            }
        }
        if (g_mtb_break && rasterizer) [[unlikely]] {
            MtbBreakAtPacket(reinterpret_cast<const PM4Header*>(payload));
        }
        if (record.flags & CpSplit::AscReleaseMem) [[unlikely]] {
            ExecuteAscReleaseMem(reinterpret_cast<const PM4CmdReleaseMem*>(payload),
                                 record.flags >> 8);
            break;
        }
        ExecuteGfxPacket(reinterpret_cast<const PM4Header*>(payload),
                         (record.flags & CpSplit::GuestMarkers) != 0);
        break;
    case CpSplit::RecordType::Call: {
        const auto* call = reinterpret_cast<const CpSplit::CallPayload*>(payload);
        if (rasterizer) {
            rasterizer->GetScheduler().SwitchStream(0);
        }
        back_in_call = true;
        call->invoke(call->context);
        back_in_call = false;
        break;
    }
    case CpSplit::RecordType::Retire: {
        {
            std::scoped_lock lock{submit_mutex};
            pending_retire.fetch_sub(1, std::memory_order_seq_cst);
        }
        submit_cv.notify_all();
        break;
    }
    case CpSplit::RecordType::DcbBegin:
    case CpSplit::RecordType::DcbEnd:
        break; // SHADPS4_A1: bookkeeping in A1Execute
    case CpSplit::RecordType::DcbRef:
        LeanBackFollow(*reinterpret_cast<const CpSplit::DcbRefPayload*>(payload));
        break;
    case CpSplit::RecordType::LeanEnd:
        break; // only in a job channel (LeanBackFollow consumes it)
    case CpSplit::RecordType::Diag:
        if (AscRingBack(*reinterpret_cast<const CpSplit::DiagPayload*>(payload)) && rasterizer) {
            rasterizer->GetScheduler().LogRecordingWindow();
        }
        break;
    case CpSplit::RecordType::Wrap:
    default:
        UNREACHABLE_MSG("CpSplit: bad record type {}", static_cast<u32>(record.type));
    }
}

int Liverpool::NativePeekNextDraw(const void* pipeline) {
    if (!split || !back_draw_record || lean_following) {
        return -1;
    }
    u64 pos = split->ReadPos() + back_draw_record->size;
    for (u32 i = 0; i < 16; ++i) {
        const auto* next = split->PeekAt(pos);
        if (!next) {
            return -1;
        }
        switch (next->type) {
        case CpSplit::RecordType::RegWrite:
        case CpSplit::RecordType::Diag:
        case CpSplit::RecordType::DcbBegin:
        case CpSplit::RecordType::DcbEnd:
            pos += next->size;
            continue;
        case CpSplit::RecordType::Draw: {
            const auto& r = *reinterpret_cast<const SplitDrawRecord*>(next + 1);
            return r.draw.pipeline == pipeline ? 1 : 0;
        }
        default:
            return 0;
        }
    }
    return -1;
}

namespace {
/// SHADPS4_NATIVE_SEGB, back thread: defined after the SegB namespace below.
void NativeSegBBack(const SplitDrawRecord& rec, const Regs& regs, Vulkan::Rasterizer& tr);
} // namespace

void Liverpool::ExecuteDrawRecord(const CpSplit::RecordHeader& record, const void* draw_payload) {
    back_draw_record = &record;
    struct ResetPeek {
        const CpSplit::RecordHeader*& ref;
        ~ResetPeek() {
            ref = nullptr;
        }
    } reset_peek{back_draw_record};
    const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(
        draw_payload ? draw_payload : static_cast<const void*>(&record + 1));
    mirror->ctx_epoch = rec.ctx_epoch;
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    const u8* stages_end =
        ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, stages);
    // FRONT_DERIVE block (read by BufferCache::BindVertexBuffers), between stages and snapshot.
    const CpSplit::VertexInputsPayload* vi_block = nullptr;
    if (rec.vi_bytes != 0) {
        vi_block = reinterpret_cast<const CpSplit::VertexInputsPayload*>(stages_end);
        stages_end += rec.vi_bytes;
    }
    if (rec.verify & DrawVerifyFull) {
        VerifyMirror("draw", record.seq, nullptr);
        VerifyStages(stages, record.seq);
        if (split_run_ahead) {
            // Posted work before this record has now executed: does the lookup still agree?
            // (Depth 0 posts nothing, so there it could only see guest CPU writes.)
            ++split_draw_relookup_checks;
            const DrawRelookupDiff diff = DrawRelookup(*rasterizer, regs, rec.draw, stages);
            if (diff.what && ++split_draw_relookup_diffs <= 16) {
                LOG_WARNING(Render,
                            "CpSplit VERIFY: draw record {}: re-lookup on the back differs from "
                            "the front's: {} (stage {}, first dword {}, carried {} dw, re-lookup "
                            "{} dw)",
                            record.seq, diff.what, diff.stage, diff.index, diff.carried_dw,
                            diff.relooked_dw);
            }
            if ((split_draw_relookup_checks & 0xfff) == 0) {
                LOG_INFO(Render,
                         "CpSplit VERIFY: {} synced draws re-looked-up on the back, {} differed",
                         split_draw_relookup_checks, split_draw_relookup_diffs);
            }
        }
    } else if (rec.verify & DrawVerifySnapshot) {
        VerifySnapshot(stages_end, record.seq);
    }
    // SHADPS4_MTB_P1: split draws translate on the worker instance (P1d).
    Vulkan::Rasterizer& translating = rasterizer->Translating();
    translating.draw_tag = rec.pad;
    translating.nr_head_tag = rec.nr_tag;
    translating.nr_head_idx = rec.nr_idx;
    translating.draw_job_tag = rec.job_tag;
    translating.draw_job_idx = rec.job_idx;
    translating.draw_job_cmd = rec.cmd_address;
    translating.draw_job_member = rec.job_member != 0;
    // SHADPS4_NATIVE_SEGB: what this record becomes (rec_outcome 4/5 confirm, 1/2 join, 3 tail
    // = opener if segb_opened else generic) and the front's prediction against the mirror.
    const bool segb = AmdGpu::NativeRec::g_segb != 0;
    if (segb) [[unlikely]] {
        translating.rec_outcome = 0;
        translating.segb_opened = false;
    }
    // SHADPS4_NATIVE_REC_PREP (notes/native-path-2026-10-09.md sec. 9): a run member the batch
    // open already prepared from the job records is confirmed here (count + 1) instead of being
    // joined; any refusal falls through to the normal path below.
    if (translating.native_prep.active) [[unlikely]] {
        if (translating.NativePrepConfirm(rec.draw, rec.verify == 0 && !rec.marker, vi_block)) {
            if (segb) [[unlikely]] {
                NativeSegBBack(rec, BackRegs(), translating);
            }
            return;
        }
    }
    MtbConflict::g_in_draw = MtbConflict::g_enabled;
    SCOPE_EXIT {
        MtbConflict::g_in_draw = false;
    };
    if (rec.marker) {
        translating.ScopeMarkerBegin(fmt::format("gfx:{}:{}", rec.cmd_address, rec.marker));
        translating.ExecuteDraw(rec.draw, stages, vi_block);
        translating.ScopeMarkerEnd();
    } else {
        translating.ExecuteDraw(rec.draw, stages, vi_block);
    }
    if (segb) [[unlikely]] {
        NativeSegBBack(rec, BackRegs(), translating);
    }
}

void Liverpool::ExecuteDispatchRecord(const CpSplit::RecordHeader& record) {
    const auto& rec = *reinterpret_cast<const SplitDispatchRecord*>(&record + 1);
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    const u8* stages_end =
        ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, stages);
    if (rec.verify & DrawVerifyFull) {
        // The front is parked on this record: its cs_state for this queue, registers and the
        // Info its lookup just refreshed must equal the record (this is what the Call path read).
        VerifyMirror("dispatch", record.seq, &rec.cs, rec.queue);
        VerifyStages(stages, record.seq);
        if (split_post_asc != 0) {
            // Posted work before this record has now executed: does the lookup still agree?
            ++split_relookup_checks;
            const auto& carried = stages[static_cast<u32>(Shader::LogicalStage::Compute)];
            if (!DispatchRelookupMatches(*rasterizer, rec.pipeline, carried) &&
                ++split_relookup_diffs <= 16) {
                LOG_WARNING(Render,
                            "CpSplit VERIFY: dispatch record {} (queue {}): re-lookup on the back "
                            "differs from the front's (pipeline or user data / flat buffer)",
                            record.seq, rec.queue);
            }
            if ((split_relookup_checks & 0xfff) == 0) {
                LOG_INFO(Render,
                         "CpSplit VERIFY: {} synced dispatches re-looked-up on the back, {} "
                         "differed",
                         split_relookup_checks, split_relookup_diffs);
            }
        }
    } else if (rec.verify & DrawVerifySnapshot) {
        VerifySnapshot(stages_end, record.seq);
    }
    if (rec.queue != GfxQueueId && AscRingLatency()) [[unlikely]] {
        ++g_asc_ring.q[rec.queue % AscRingQueues].dispatches;
    }
    const u32 xq_qi = std::min<u32>(rec.queue, XqQueues - 1);
    const u64 xq_up0 =
        g_xq && rec.queue != GfxQueueId ? rasterizer->GetBufferCache().SyncUploadBytesTotal() : 0;
    SCOPE_EXIT {
        if (g_xq && rec.queue != GfxQueueId) [[unlikely]] {
            g_xq_c.upload_bytes[xq_qi].fetch_add(
                rasterizer->GetBufferCache().SyncUploadBytesTotal() - xq_up0,
                std::memory_order_relaxed);
            g_xq_c.disp[xq_qi].fetch_add(1, std::memory_order_relaxed);
            bool img = false;
            if (rec.pipeline) {
                for (const auto* info : rec.pipeline->GetStages()) {
                    img |= info && !info->images.empty();
                }
            }
            if (img) {
                g_xq_c.disp_img[xq_qi].fetch_add(1, std::memory_order_relaxed);
                g_xq_c.seg_img[xq_qi].store(1, std::memory_order_relaxed);
            }
        }
    };
    // BackCsRegs() returns the snapshot while this dispatch runs (HLE, IsCompute*, BindBuffers).
    back_cs = &rec.cs;
    // SHADPS4_MTB_P1: split dispatches (and their run / queue) go to the worker instance (P1d).
    Vulkan::Rasterizer& translating = rasterizer->Translating();
    MtbConflict::g_in_draw = MtbConflict::g_enabled;
    SCOPE_EXIT {
        MtbConflict::g_in_draw = false;
    };
    translating.SetDispatchRun(rec.run);
    translating.SetDispatchQueue(rec.queue);
    Common::ReadbackOffload::AscSpan().in_asc_dispatch = rec.queue != GfxQueueId;
    if (VideoCore::g_no_particles && rec.queue != GfxQueueId) [[unlikely]] {
        VideoCore::NoParticlesNote(false); // q40: an ASC dispatch record, not executed
    } else if (rec.marker && rec.queue != GfxQueueId) {
        translating.ScopeMarkerBegin(
            fmt::format("asc[{}]:{}:{}", rec.queue - 1, rec.cmd_address, rec.marker));
        translating.ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
        translating.ScopeMarkerEnd();
    } else if (rec.marker) {
        translating.ScopeMarkerBegin(fmt::format("gfx:{}:{}", rec.cmd_address, rec.marker));
        translating.ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
        translating.ScopeMarkerEnd();
    } else {
        translating.ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
    }
    translating.SetDispatchRun(0);
    translating.SetDispatchQueue(0);
    Common::ReadbackOffload::AscSpan().in_asc_dispatch = false;
    back_cs = nullptr;
}

void Liverpool::VerifyMirror(const char* what, u64 seq, const ComputeProgram* cs,
                             u32 cs_queue) const {
    // Only for records the front is parked on (Sync): depth 0, dispatches, and the periodic
    // synchronous run-ahead draw. The front's state is then stable and published to this thread
    // (release store of write_pos, acquire load in Next).
    // A full memcmp rather than a hash: same pass over 213 KB, exact, and it names the word.
    const auto& back = mirror->regs.reg_array;
    const auto& front = regs.reg_array;
    if (std::memcmp(back.data(), front.data(), sizeof(back)) != 0) {
        u32 i = 0;
        while (back[i] == front[i]) {
            ++i;
        }
        UNREACHABLE_MSG("CpSplit VERIFY: register word {:#x} differs at {} record {}: back {:#x} "
                        "front {:#x}",
                        i, what, seq, back[i], front[i]);
    }
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        ASSERT_MSG(mirror->cb_extent[cb].raw == last_cb_extent[cb].raw,
                   "CpSplit VERIFY: CB{} extent differs at {} record {}", cb, what, seq);
    }
    ASSERT_MSG(mirror->db_extent.raw == last_db_extent.raw,
               "CpSplit VERIFY: DB extent differs at {} record {}", what, seq);
    // The context epoch only feeds the draw path's viewport memo; dispatch records do not carry
    // it, so the mirror's copy is only current at draws.
    ASSERT_MSG(cs || mirror->ctx_epoch == context_change_epoch,
               "CpSplit VERIFY: context epoch differs at {} record {}", what, seq);
    if (cs) {
        // The Call path read mapped_queues[curr_qid] on the back; the record copied GetCsRegs()
        // on the front. Both must be this queue's state.
        ASSERT_MSG(cs_queue < NumTotalQueues && s32(cs_queue) == curr_qid &&
                       std::memcmp(cs, &mapped_queues[cs_queue].cs_state, sizeof(*cs)) == 0,
                   "CpSplit VERIFY: cs registers differ at {} record {} (queue {}, curr_qid {})",
                   what, seq, cs_queue, curr_qid);
    }
    static u64 checks = 0;
    if ((++checks & 0xffff) == 0) {
        LOG_INFO(Render, "CpSplit VERIFY: {} draw/dispatch records checked, 0 mismatches",
                 checks);
    }
}

void Liverpool::VerifySnapshot(const u8* snapshot, u64 seq) const {
    // Run-ahead: the front has moved on, so compare against the copy it made at this draw.
    // Registers outside the windows are covered by the periodic full check (VerifyMirror).
    const auto* words = reinterpret_cast<const u32*>(snapshot);
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        ASSERT_MSG(mirror->cb_extent[cb].raw == words[cb],
                   "CpSplit VERIFY: CB{} extent differs at posted record {}", cb, seq);
    }
    ASSERT_MSG(mirror->db_extent.raw == words[NUM_COLOR_BUFFERS],
               "CpSplit VERIFY: DB extent differs at posted record {}", seq);
    words += NUM_COLOR_BUFFERS + 1;
    for (const auto& window : VerifyWindows) {
        const u32 count = window.end - window.begin;
        const u32* back = &mirror->regs.reg_array[window.begin];
        if (std::memcmp(back, words, size_t{count} * sizeof(u32)) != 0) {
            u32 i = 0;
            while (back[i] == words[i]) {
                ++i;
            }
            UNREACHABLE_MSG("CpSplit VERIFY: register word {:#x} differs at posted record "
                            "{}: back {:#x} front {:#x}",
                            window.begin + i, seq, back[i], words[i]);
        }
        words += count;
    }
    static u64 checks = 0;
    if ((++checks & 0xffff) == 0) {
        LOG_INFO(Render, "CpSplit VERIFY: {} posted draws/dispatches checked (register windows), 0 mismatches",
                 checks);
    }
}

void Liverpool::JournalRegs(u32 word_offset, u32 count) {
    const u64 jr0 = g_reg_scan ? __rdtsc() : 0;
    SCOPE_EXIT {
        if (g_reg_scan) [[unlikely]] {
            t_reg_scan.post_cyc += __rdtsc() - jr0;
            ++t_reg_scan.posts;
        }
    };
    if (split_enabled) [[unlikely]] {
        split->EmitRegWrite(word_offset, &regs.reg_array[word_offset], count);
    }
}

void Liverpool::JournalExtent(u32 reg_addr) {
    if (!split_enabled) [[likely]] {
        return;
    }
    switch (reg_addr) {
    case ContextRegs::CbColor0Base:
    case ContextRegs::CbColor1Base:
    case ContextRegs::CbColor2Base:
    case ContextRegs::CbColor3Base:
    case ContextRegs::CbColor4Base:
    case ContextRegs::CbColor5Base:
    case ContextRegs::CbColor6Base:
    case ContextRegs::CbColor7Base: {
        const u32 cb = (reg_addr - ContextRegs::CbColor0Base) /
                       (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
        split->EmitExtent(cb, last_cb_extent[cb].raw);
        break;
    }
    case ContextRegs::CbColor0Cmask:
    case ContextRegs::CbColor1Cmask:
    case ContextRegs::CbColor2Cmask:
    case ContextRegs::CbColor3Cmask:
    case ContextRegs::CbColor4Cmask:
    case ContextRegs::CbColor5Cmask:
    case ContextRegs::CbColor6Cmask:
    case ContextRegs::CbColor7Cmask: {
        const u32 cb = (reg_addr - ContextRegs::CbColor0Cmask) /
                       (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
        split->EmitExtent(cb, last_cb_extent[cb].raw);
        break;
    }
    case ContextRegs::DbZInfo:
        split->EmitExtent(NUM_COLOR_BUFFERS, last_db_extent.raw);
        break;
    default:
        break;
    }
}

void Liverpool::BumpEpoch(PM4ItOpcode opcode) {
    if (split_enabled) [[unlikely]] {
        // In stream order on the back. Delaying a bump only makes an upload happen earlier
        // (submit_epoch.h: an extra bump is safe, a missing one is not).
        if (VideoCore::g_epoch_defer_upload || VideoCore::g_native_coherent != 0) {
            split->EmitEpochBump(static_cast<u32>(opcode));
        }
        return;
    }
    VideoCore::BumpSubmitEpoch(opcode);
}

void Liverpool::RunGfxPacket(const PM4Header* header, bool guest_markers_enabled,
                             CpSplit::DrainReason reason) {
    if (!split_enabled) [[likely]] {
        ExecuteGfxPacket(header, guest_markers_enabled);
        return;
    }
    // Verbatim copy: the guest may reuse the DCB once the front has moved on (run-ahead).
    // The copy reads guest memory between Reserve and Commit; a fault there is safe because a
    // fault drain never publishes the open record (Channel).
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                guest_markers_enabled ? CpSplit::GuestMarkers : 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    split->Sync(reason);
}

void Liverpool::RunGfxDma(const PM4Header* header, bool guest_markers_enabled,
                          std::span<const u32> rest) {
    if (!split_post_dma) [[likely]] {
        RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketDma);
        return;
    }
    // POST_DMA (run-ahead): nothing the front reads after this packet is produced by it, except
    //  - guest memory read by later pipeline lookups (shader code, SRT tables, fetch V# tables):
    //    measured 0 of ~4 M top-level checks against DMA CPU-path writes (SPLIT_HAZARD_STATS,
    //    2026-09-25); nested tables unmeasured (SPLIT-TODO);
    //  - the rest of this command buffer (a GPU-patched packet, as ASC does for dispatch dims):
    //    guarded here, synchronous if the destination overlaps it;
    //  - nested IBs: IndirectBuffer syncs while posted packets are outstanding;
    //  - WAIT_REG_MEM / COND_EXEC / semaphores / faults: those drain anyway.
    // SPLIT-TODO: only the innermost DCB level is guarded; a parent level's remainder is not.
    const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
    VAddr write_dst = 0;
    u64 write_bytes = 0; ///< 0: a GDS destination, which no command buffer can be in.
    if (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) {
        const VAddr dst = dma->DstAddress<VAddr>();
        const VAddr dst_end = dst + dma->NumBytes();
        const auto lo = reinterpret_cast<VAddr>(rest.data());
        const VAddr hi = lo + rest.size_bytes();
        if (dst < hi && lo < dst_end) {
            RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketDmaGuard);
            return;
        }
        write_dst = dst;
        write_bytes = dst_end - dst;
    }
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                guest_markers_enabled ? CpSplit::GuestMarkers : 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    // The no-op DMA (destination 0x3022C) never writes: the back's ExecuteGfxPacket skips it.
    if (split_task_guard && dma->dst_addr_lo != 0x3022C) [[unlikely]] {
        // DMA_DATA from immediate data fills its destination with `data` (known value).
        split->NotePostedWrite(write_dst, write_bytes, dma->src_sel == DmaDataSrc::Data, dma->data,
                               false, 3);
    }
    split->PostRunAhead(CpSplit::PostKind::GfxPacket);
}

void Liverpool::AdvanceAscReadPtr(AscQueueInfo& queue, u32 dwords) {
    if (!g_rptr_backing || !split_run_ahead) {
        *queue.read_addr += dwords;
        *queue.read_addr %= queue.ring_size_dw;
        return;
    }
    const u32 cur = *queue.read_addr; // reads never fault here (s548/s549: write faults only)
    if (queue.rptr_last != ~0u && cur != queue.rptr_last &&
        g_rptr_changed.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render, "RPTR_BACKING: read pointer {:#x} holds {:#x}, the front wrote {:#x}",
                    reinterpret_cast<VAddr>(queue.read_addr), cur, queue.rptr_last);
    }
    const u32 next = (cur + dwords) % queue.ring_size_dw;
    auto* memory = Core::Memory::Instance();
    if (queue.rptr_last == ~0u) {
        memory->ExcludeFromDownloads(reinterpret_cast<VAddr>(queue.read_addr));
    }
    if (memory->TryWriteBacking(queue.read_addr, &next, sizeof(next))) {
        g_rptr_writes.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_rptr_fallbacks.fetch_add(1, std::memory_order_relaxed);
        *queue.read_addr = next;
    }
    queue.rptr_last = next;
}

void Liverpool::PostWaitBarrier(u32 kind) {
    auto* call = split->Reserve<CpSplit::CallPayload>(CpSplit::RecordType::Call);
    call->invoke = [](void* context) {
        auto* lp = static_cast<Liverpool*>(context);
        ++VideoCore::g_g1[11]; // q19
        if (lp->rasterizer) {
            VideoCore::G1Scope g1{9};
            lp->rasterizer->WaitBarrier();
        }
    };
    call->context = this;
    split->Commit();
    split->PostRunAhead(CpSplit::PostKind::Fence);
    g_wait_barriers[kind].fetch_add(1, std::memory_order_relaxed);
}

void Liverpool::NoteFrontFault(VAddr addr, u64 size, bool read) {
    if (!split_task_guard) {
        return;
    }
    const u64 n = std::max<u64>(size, 1);
    u32 cls = 3;
    if (SplitSpansOverlap(addr, addr + n)) {
        cls = 0;
    } else if (t_wrm_label != 0 && (addr >> 12) <= (t_wrm_label >> 12) &&
               (t_wrm_label >> 12) <= ((addr + n - 1) >> 12)) {
        cls = 1;
    } else if (split->OutstandingWriteSeq(addr, addr + n) != 0) {
        cls = 2;
    }
    const u32 i = (read ? 0 : 4) + cls;
    static std::atomic<u32> samples{0};
    if (samples.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_INFO(Render, "POST_ORDER front fault sample: {} {:#x} size {:#x} class {} (0 span, 1 "
                         "label, 2 logged, 3 other), last tested label {:#x}",
                 read ? "read" : "invalidate", addr, size, cls, t_wrm_label);
    }
    g_po.fault[i].fetch_add(1, std::memory_order_relaxed);
}

void Liverpool::RunAscPacket(const PM4Header* header, std::span<const u32> rest, VAddr dst,
                             u64 bytes, CpSplit::DrainReason guard_reason) {
    // POST_ASC: the same argument as RunGfxDma. What the front reads after this packet:
    //  - the rest of this ASC command buffer (the title patches dispatch dims this way; that
    //    case is the front's indirect_patches and never gets here): guarded, synchronous if
    //    [dst, dst + bytes) overlaps it;
    //  - nested ASC IBs: IndirectBuffer syncs while posted packets are outstanding;
    //  - guest memory read by later pipeline lookups: unmeasured for ASC (SPLIT-TODO); the
    //    VERIFY re-lookup on synced dispatches is the check;
    //  - WAIT_REG_MEM / semaphores / RELEASE_MEM / task done / faults: those drain anyway.
    // The packet is copied: the front advances the ring's read_addr right after parsing it, so
    // the guest may reuse the slot before the back runs it.
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32), 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    const auto lo = reinterpret_cast<VAddr>(rest.data());
    const VAddr hi = lo + rest.size_bytes();
    if (bytes != 0 && dst < hi && lo < dst + bytes) {
        split->Sync(guard_reason);
        return;
    }
    if (split_task_guard) [[unlikely]] {
        // bytes == 0 (GDS destination): nothing to note. DMA_DATA from immediate data: known value.
        const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
        const bool imm = header->type3.opcode == PM4ItOpcode::DmaData &&
                         dma->src_sel == DmaDataSrc::Data;
        split->NotePostedWrite(dst, bytes, imm, imm ? dma->data : 0, false, 4);
    }
    split->PostRunAhead(CpSplit::PostKind::AscPacket);
    split->MaybeLog();
}

namespace {
// SHADPS4_FENCE_CENSUS=1 (2026-10-09, notes/dcb-parallel-2026-10-09.md sec. 4): when a posted fence
// syncs because its label lies in command memory the front has not parsed yet, where does the
// label sit? Inside a packet the front will read (header / body of which opcode), or in data it
// skips (a NOP body), and does a later WAIT_REG_MEM of the same span poll it? Front thread only.
const bool g_fence_census = [] {
    const char* v = std::getenv("SHADPS4_FENCE_CENSUS");
    return v && v[0] == '1';
}();
struct FenceCensusState {
    std::unordered_map<u32, u64> where; // key: asc << 24 | in_rest << 23 | body << 22 | type << 8 | opcode
    std::array<u64, 5> dist{};          // dwords from span start to the label: <16, <256, <4K, <64K, more
    u64 syncs = 0, polled_later = 0, not_found = 0;
    std::array<u64, 8> rel{}; // label - fence packet, dwords: <-4096 <-16 <0 0..1 <16 <256 <4096 more
    std::array<u64, 4> at_type{}; // PM4 type bits of the dword at the label
    u64 span_dw = 0;
} g_fence_census_state;

void FenceCensusNote(std::span<const u32> span, VAddr label, bool asc, bool in_rest,
                     const PM4Header* fence) {
    auto& st = g_fence_census_state;
    ++st.syncs;
    st.span_dw += span.size();
    {
        const s64 r = (s64(label) - s64(reinterpret_cast<VAddr>(fence))) / 4;
        st.rel[r < -4096 ? 0 : r < -16 ? 1 : r < 0 ? 2 : r < 2 ? 3 : r < 16 ? 4 : r < 256 ? 5 : r < 4096 ? 6 : 7] += 1;
        st.at_type[(*reinterpret_cast<const u32*>(label) >> 30) & 3] += 1;
    }
    const u32* p = span.data();
    const u32* end = p + span.size();
    const VAddr base = reinterpret_cast<VAddr>(p);
    const u64 dw = (label - base) / 4;
    st.dist[dw < 16 ? 0 : dw < 256 ? 1 : dw < 4096 ? 2 : dw < 65536 ? 3 : 4] += 1;
    bool found = false;
    bool polled = false;
    while (p < end) {
        const auto* h = reinterpret_cast<const PM4Header*>(p);
        u32 n = 1;
        u32 opcode = 0;
        const u32 type = h->type;
        if (type == 3) {
            n = h->type3.NumWords() + 1;
            opcode = static_cast<u32>(h->type3.opcode.Value());
            if (h->type3.opcode == PM4ItOpcode::WaitRegMem && !polled) {
                const auto* w = reinterpret_cast<const PM4CmdWaitRegMem*>(h);
                const VAddr a = reinterpret_cast<VAddr>(w->Address<u64*>());
                polled = a >= label && a < label + 8;
            }
        } else if (type == 0) {
            n = h->type0.NumWords() + 1;
        }
        const VAddr lo = reinterpret_cast<VAddr>(p);
        if (!found && label >= lo && label < lo + u64{n} * 4) {
            found = true;
            const u32 body = label != lo ? 1u : 0u;
            ++st.where[(asc ? 1u << 24 : 0u) | (in_rest ? 1u << 23 : 0u) | (body << 22) |
                       (type << 8) | (opcode & 0xff)];
        }
        p += n;
    }
    st.not_found += !found;
    st.polled_later += polled;
    if ((st.syncs & 0x3fff) == 0) {
        std::vector<std::pair<u32, u64>> top(st.where.begin(), st.where.end());
        std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 12); ++i) {
            const u32 k = top[i].first;
            line += fmt::format(" [{}{}{} type{} op{:#04x}]={}", (k >> 24) & 1 ? "asc " : "gfx ",
                                (k >> 23) & 1 ? "rest " : "other ", (k >> 22) & 1 ? "body" : "header",
                                (k >> 8) & 3, k & 0xff, top[i].second);
        }
        LOG_INFO(Render,
                 "FenceCensus: syncs={} not_found={} polled_by_later_wait={} dist(dw <16/<256/<4K/<64K/more)="
                 "{}/{}/{}/{}/{} | label-fence dw <-4096/<-16/<0/0-1/<16/<256/<4096/more={}/{}/{}/{}/{}/{}/{}/{} "
                 "| dword at label type0/1/2/3={}/{}/{}/{} span_dw_mean={:.0f} |{}",
                 st.syncs, st.not_found, st.polled_later, st.dist[0], st.dist[1], st.dist[2],
                 st.dist[3], st.dist[4], st.rel[0], st.rel[1], st.rel[2], st.rel[3], st.rel[4],
                 st.rel[5], st.rel[6], st.rel[7], st.at_type[0], st.at_type[1], st.at_type[2],
                 st.at_type[3], double(st.span_dw) / double(st.syncs), line);
    }
}
/// Census v3: from the fence forward to the end of the span, the first WAIT_REG_MEM on memory
/// polling the label: distance (packets, draws) and whether the fence's own value passes it.
struct FenceWaitState {
    u64 n = 0, eos = 0, no_wait = 0, pass = 0, fail = 0, other_writer = 0;
    std::array<u64, 5> draws_between{}; // 0, 1-3, 4-15, 16-63, 64+
    std::array<u64, 5> packets_between{};
} g_fence_wait;

void FenceWaitNote(const PM4Header* fence, const u32* end, VAddr label) {
    auto& st = g_fence_wait;
    ++st.n;
    u32 value = 0;
    switch (fence->type3.opcode.Value()) {
    case PM4ItOpcode::EventWriteEop:
        value = reinterpret_cast<const PM4CmdEventWriteEop*>(fence)->DataDWord();
        break;
    case PM4ItOpcode::ReleaseMem:
        value = reinterpret_cast<const PM4CmdReleaseMem*>(fence)->DataDWord();
        break;
    default:
        ++st.eos;
        return;
    }
    const auto bucket = [](u64 v) { return v == 0 ? 0 : v < 4 ? 1 : v < 16 ? 2 : v < 64 ? 3 : 4; };
    const u32* p = reinterpret_cast<const u32*>(fence) + fence->type3.NumWords() + 1;
    u64 draws = 0, packets = 0;
    while (p < end) {
        const auto* h = reinterpret_cast<const PM4Header*>(p);
        u32 n = 1;
        if (h->type == 3) {
            n = h->type3.NumWords() + 1;
            const auto op = h->type3.opcode.Value();
            using O = PM4ItOpcode;
            draws += op == O::DrawIndex2 || op == O::DrawIndexOffset2 || op == O::DrawIndexAuto ||
                     op == O::DispatchDirect || op == O::DrawIndexIndirect ||
                     op == O::DispatchIndirect;
            if (op == O::WaitRegMem) {
                const auto* w = reinterpret_cast<const PM4CmdWaitRegMem*>(h);
                const VAddr a = reinterpret_cast<VAddr>(w->Address<u32*>());
                if (w->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory && a == label) {
                    std::array<u32, 1> fake{};
                    // Evaluate the predicate on the value the fence will write.
                    const u32 v = value & w->mask;
                    bool ok = false;
                    using F = PM4CmdWaitRegMem::Function;
                    switch (w->function.Value()) {
                    case F::Always: ok = true; break;
                    case F::LessThan: ok = v < w->ref; break;
                    case F::LessThanEqual: ok = v <= w->ref; break;
                    case F::Equal: ok = v == w->ref; break;
                    case F::NotEqual: ok = v != w->ref; break;
                    case F::GreaterThanEqual: ok = v >= w->ref; break;
                    case F::GreaterThan: ok = v > w->ref; break;
                    default: break;
                    }
                    (void)fake;
                    ++(ok ? st.pass : st.fail);
                    ++st.draws_between[bucket(draws)];
                    ++st.packets_between[bucket(packets)];
                    break;
                }
            }
            if ((op == O::EventWriteEop || op == O::ReleaseMem || op == O::WriteData) && p != reinterpret_cast<const u32*>(fence)) {
                // another writer that might target the label before the wait
                if (op == O::EventWriteEop &&
                    reinterpret_cast<VAddr>(reinterpret_cast<const PM4CmdEventWriteEop*>(h)->Address<u32>()) == label) {
                    ++st.other_writer;
                }
            }
        } else if (h->type == 0) {
            n = h->type0.NumWords() + 1;
        }
        ++packets;
        p += n;
    }
    if (p >= end) {
        ++st.no_wait;
    }
    if ((st.n & 0x3fff) == 0) {
        LOG_INFO(Render,
                 "FenceWait: fences={} eos={} no_wait_in_span={} wait_pass_with_fence_value={} "
                 "wait_fail={} other_eop_writer={} draws_between 0/1-3/4-15/16-63/64+={}/{}/{}/{}/{} "
                 "packets_between={}/{}/{}/{}/{}",
                 st.n, st.eos, st.no_wait, st.pass, st.fail, st.other_writer, st.draws_between[0],
                 st.draws_between[1], st.draws_between[2], st.draws_between[3],
                 st.draws_between[4], st.packets_between[0], st.packets_between[1],
                 st.packets_between[2], st.packets_between[3], st.packets_between[4]);
    }
}
// SHADPS4_FENCE_FORWARD=1 (2026-10-09, notes/dcb-parallel-2026-10-09.md sec. 4; census s340 / s341:
// every syncing fence label sits just BEHIND its own packet, and the very next packet is a
// WAIT_REG_MEM on that label which the fence's own value satisfies). The front posts such a fence
// without syncing and skips the adjacent wait: in stream order the back writes the label before
// anything after the wait runs, and the wait's test passes on that value. Front thread only.
const bool g_fence_forward = [] {
    const char* v = std::getenv("SHADPS4_FENCE_FORWARD");
    return v && v[0] == '1';
}();
struct ForwardedFence {
    const u32* next_packet = nullptr; ///< the packet right after the fence (the only one we skip)
    VAddr label = 0;
    u32 value = 0;
};
std::array<ForwardedFence, 257> g_fwd{}; // 0 = gfx, 1 + pipe_id = ASC
struct ForwardStats {
    u64 posted = 0, skipped_waits = 0, not_adjacent = 0, failed_test = 0;
} g_fwd_stats;

/// The low dword a fence packet writes to its label and the write's width in bytes, if it is a
/// plain data write (Opus review 2026-10-09 #2: the exact width, not 8).
struct FenceWrite {
    u32 value;
    u32 width;
};
std::optional<FenceWrite> FenceValue(const PM4Header* fence) {
    const auto data = [](DataSelect sel, u32 value) -> std::optional<FenceWrite> {
        if (sel == DataSelect::Data32Low) {
            return FenceWrite{value, 4};
        }
        if (sel == DataSelect::Data64) {
            return FenceWrite{value, 8};
        }
        return std::nullopt;
    };
    switch (fence->type3.opcode.Value()) {
    case PM4ItOpcode::EventWriteEop: {
        const auto* eop = reinterpret_cast<const PM4CmdEventWriteEop*>(fence);
        return data(eop->data_sel.Value(), eop->DataDWord());
    }
    case PM4ItOpcode::EventWriteEos: {
        const auto* eos = reinterpret_cast<const PM4CmdEventWriteEos*>(fence);
        if (eos->command.Value() == PM4CmdEventWriteEos::Command::SignalFence) {
            return FenceWrite{eos->DataDWord(), 4};
        }
        return std::nullopt;
    }
    case PM4ItOpcode::ReleaseMem: {
        const auto* rm = reinterpret_cast<const PM4CmdReleaseMem*>(fence);
        return data(rm->data_sel.Value(), rm->DataDWord());
    }
    default:
        return std::nullopt;
    }
}

/// Opus review #6: ASC fences forward only from indirect buffers (a top-level ASC ring's read
/// pointer moves past the skipped wait before the back writes a label that may sit in the ring).
thread_local bool t_fwd_asc_allowed = false;

bool WaitPassesWith(const PM4CmdWaitRegMem* w, u32 value) {
    const u32 v = value & w->mask;
    using F = PM4CmdWaitRegMem::Function;
    switch (w->function.Value()) {
    case F::Always:
        return true;
    case F::LessThan:
        return v < w->ref;
    case F::LessThanEqual:
        return v <= w->ref;
    case F::Equal:
        return v == w->ref;
    case F::NotEqual:
        return v != w->ref;
    case F::GreaterThanEqual:
        return v >= w->ref;
    case F::GreaterThan:
        return v > w->ref;
    default:
        return false;
    }
}

/// True: this WAIT_REG_MEM is the packet right after a forwarded fence on the same queue, polls
/// its label and passes on its value - the caller skips the drain and the wait.
bool ForwardedWait(u32 queue_key, const PM4Header* header) {
    auto& f = g_fwd[queue_key];
    if (f.next_packet == nullptr) {
        return false;
    }
    const bool adjacent = reinterpret_cast<const u32*>(header) == f.next_packet;
    const auto* w = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
    const bool ok = adjacent && w->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                    reinterpret_cast<VAddr>(w->Address<u32*>()) == f.label &&
                    WaitPassesWith(w, f.value);
    f.next_packet = nullptr;
    if (!adjacent) {
        ++g_fwd_stats.not_adjacent;
    } else if (!ok) {
        ++g_fwd_stats.failed_test;
    } else if ((++g_fwd_stats.skipped_waits & 0xffff) == 0) {
        LOG_INFO(Render, "FenceForward: posted={} skipped_waits={} not_adjacent={} failed_test={}",
                 g_fwd_stats.posted, g_fwd_stats.skipped_waits, g_fwd_stats.not_adjacent,
                 g_fwd_stats.failed_test);
    }
    return ok;
}
} // namespace

void Liverpool::RunFence(const PM4Header* header, std::span<const u32> rest, VAddr label,
                         u16 flags, CpSplit::DrainReason sync_reason) {
    // POST_FENCE: the label is written (and the interrupt raised) when the back reaches this
    // record, exactly when the synchronous path wrote it; the front just does not wait.
    // Front readers of the label drain first (WAIT_REG_MEM, COND_EXEC, semaphores); a label in
    // another buffer is caught by the TASK_GUARD checks; one in this buffer's rest syncs here.
    // The copy: the guest may reuse the command memory once the front has moved on.
    // Fences do not arm SyncIfPostedPackets, so the IB-end guards do not cover a label in a
    // parent level's rest: check every active level here (`rest` is the innermost one).
    constexpr u64 LabelBytes = 8; // 32- or 64-bit value; 8 covers both.
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32), flags);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    const auto lo = reinterpret_cast<VAddr>(rest.data());
    const VAddr hi = lo + rest.size_bytes();
    if (g_fence_forward) {
        // A label wholly behind this packet's header (memory the front has already parsed) and
        // in no other active span cannot change anything the front reads next; when the very next
        // packet is a WAIT_REG_MEM on that label which the fence's value passes, post without
        // syncing and let ForwardedWait skip that wait. Opus review 2026-10-09: GPU-delayed fences
        // (#1) keep the old path; exact width, strictly before the header (#2); the other spans
        // still checked (#3); the adjacent wait checked here, so no stale entry survives (#4);
        // ASC only from indirect buffers (#6).
        const bool asc = (flags & CpSplit::AscReleaseMem) != 0;
        const u32 key = asc ? 1 + ((flags >> 8) & 0xff) : 0;
        g_fwd[key].next_packet = nullptr;
        const auto write = FenceValue(header);
        const auto* next = reinterpret_cast<const u32*>(header) + dwords;
        const VAddr head = reinterpret_cast<VAddr>(header);
        if (write && !FenceDeferredAt(label) && (!asc || t_fwd_asc_allowed) &&
            label + write->width <= head && label + 256 >= head &&
            reinterpret_cast<VAddr>(next) < hi &&
            !SplitSpansOverlap(label, label + write->width)) {
            const auto* nh = reinterpret_cast<const PM4Header*>(next);
            const auto* w = reinterpret_cast<const PM4CmdWaitRegMem*>(next);
            if (nh->type == 3 && nh->type3.opcode == PM4ItOpcode::WaitRegMem &&
                w->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                reinterpret_cast<VAddr>(w->Address<u32*>()) == label &&
                WaitPassesWith(w, write->value)) {
                g_fwd[key] = {next, label, write->value};
                ++g_fwd_stats.posted;
                split->NotePostedWrite(label, write->width, true, write->value, true, 1);
                split->PostRunAhead(CpSplit::PostKind::Fence);
                split->MaybeLog();
                return;
            }
        }
    }
    if ((label < hi && lo < label + LabelBytes) ||
        SplitSpansOverlap(label, label + LabelBytes)) {
        if (g_fence_census) [[unlikely]] {
            FenceWaitNote(header, reinterpret_cast<const u32*>(hi), label);
            const bool asc = (flags & CpSplit::AscReleaseMem) != 0;
            if (label < hi && lo < label + LabelBytes) {
                FenceCensusNote(rest, label, asc, true, header);
            } else {
                for (u32 i = 0; i < split_span_hwm; ++i) {
                    const auto* span = split_spans[i];
                    if (!span) {
                        continue;
                    }
                    const auto s_lo = reinterpret_cast<VAddr>(span->data());
                    if (label < s_lo + span->size_bytes() && s_lo < label + LabelBytes) {
                        FenceCensusNote(*span, label, asc, false, header);
                        break;
                    }
                }
            }
        }
        split->Sync(sync_reason);
        return;
    }
    if (const auto fw = FenceValue(header); fw && !FenceDeferredAt(label)) {
        split->NotePostedWrite(label, LabelBytes, true, fw->value, true, 1);
    } else {
        split->NotePostedWrite(label, LabelBytes, false, 0, true, 1);
    }
    split->PostRunAhead(CpSplit::PostKind::Fence);
    split->MaybeLog();
}



static void NoteAscFence(const PM4CmdReleaseMem& packet) {
    if (!Common::FenceWatch::Enabled()) {
        return;
    }
    switch (packet.data_sel.Value()) {
    case DataSelect::Data32Low:
        Common::FenceWatch::NoteRelease(packet.Address<VAddr>(), packet.DataDWord(), 4);
        break;
    case DataSelect::Data64:
        Common::FenceWatch::NoteRelease(packet.Address<VAddr>(), packet.DataQWord(), 8);
        break;
    default:
        break;
    }
}

/// SHADPS4_ASC_RELEASE_FLUSH=1: cut and submit the batch being recorded when an ASC RELEASE_MEM
/// runs on the back. The label is written at CP time, so the guest's poll (Second Son's 0x1695e0
/// particle wait) passes at once; without a cut the producing dispatches stay in the unsubmitted
/// batch until the guest's first readback fault cuts it (46% of control-block jobs, run
/// diag-fire-fastlane3-20260928-i, Fable review 2026-09-28).
static bool AscReleaseFlush() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_RELEASE_FLUSH");
        return value && value[0] == '1';
    }();
    return enabled;
}

void Liverpool::ExecuteAscReleaseMem(const PM4CmdReleaseMem* release_mem, u32 pipe_id) {
    VideoCore::g_pf_label_addr = release_mem->Address<VAddr>(); // f2: the label the guest polls
    if (rasterizer) {
        // p1f: an ASC label on the lane - image downloads are main-only work (review B2).
        auto& sch = rasterizer->GetScheduler();
        const u32 prev = sch.ActiveStream();
        const u64 lane_wait = sch.LaneWaitMain();
        if (prev != 0 && sch.HasLane()) {
            sch.SwitchStream(0);
        }
        rasterizer->ProcessDownloadImages();
        if (prev != 0 && sch.HasLane()) {
            sch.SwitchStream(prev);
            sch.LaneNeedMain(lane_wait, 4);
        }
    }
    if (AscReleaseFlush() && rasterizer) {
        auto& sched = rasterizer->GetScheduler();
        sched.EndRendering();
        sched.Flush();
    }
    if (rasterizer && rasterizer->GetScheduler().HasAscStream()) {
        // SHADPS4_ASC_STREAM: the guest's compute label. Hand the ASC work recorded so far to its
        // queue now (its own queue: this does not deepen the main one).
        rasterizer->GetScheduler().SubmitAscSegment();
    }
    if (Common::ReadbackOffload::AscSpanStatsEnabled() && rasterizer) [[unlikely]] {
        auto& sp = Common::ReadbackOffload::AscSpan();
        auto& sched = rasterizer->GetScheduler();
        auto* ms = sched.GetMasterSemaphore();
        ms->Refresh();
        const u64 cur = sched.CurrentTick();
        const u64 known = ms->KnownGpuTick();
        ++sp.spans;
        if (sp.binds == 0) {
            ++sp.empty;
        } else {
            const u64 floor_a = std::max(sp.w_access, known);
            const u64 floor_w = std::max(sp.w_write, known);
            // Unfinished batches newer than the newest one the span depends on (the open batch
            // included: an ASC submission could go ahead of it).
            const u64 over_a = cur > floor_a ? cur - floor_a : 0;
            const u64 over_w = cur > floor_w ? cur - floor_w : 0;
            sp.unfinished_sum += cur > known ? cur - known : 0;
            sp.over_access_sum += over_a;
            sp.over_write_sum += over_w;
            sp.access_is_current += sp.w_access == cur;
            sp.write_is_current += sp.w_write == cur;
            sp.big_buffer_spans += sp.max_buffer_mb >= 16;
            ++sp.over_access_hist[over_a == 0 ? 0 : over_a == 1 ? 1 : over_a < 4 ? 2 : over_a < 8 ? 3 : 4];
        }
        sp.w_access = sp.w_write = sp.binds = sp.max_buffer_mb = 0;
        if ((sp.spans & 0x7ff) == 0) {
            const double n = double(std::max<u64>(1, sp.spans - sp.empty));
            LOG_INFO(Render,
                     "AscSpan: spans={} empty={} unfinished_batches={:.2f} overtakeable_by_access="
                     "{:.2f} overtakeable_by_write={:.2f} access_newest_is_current={} "
                     "write_newest_is_current={} spans_binding_>=16MB={} "
                     "hist_overtake_access[0,1,2-3,4-7,8+]={}/{}/{}/{}/{}",
                     sp.spans, sp.empty, double(sp.unfinished_sum) / n,
                     double(sp.over_access_sum) / n, double(sp.over_write_sum) / n,
                     sp.access_is_current, sp.write_is_current, sp.big_buffer_spans,
                     sp.over_access_hist[0], sp.over_access_hist[1], sp.over_access_hist[2],
                     sp.over_access_hist[3], sp.over_access_hist[4]);
            const auto keep = sp.in_asc_dispatch;
            sp = {};
            sp.in_asc_dispatch = keep;
        }
    }
    g_cp_wait.NoteWrite(release_mem->Address<void*>());
    VideoCore::ProtectFence();
    if (AscRingLatency()) [[unlikely]] {
        // Just before the label write (or its deferral to the GPU, counted separately).
        AscRingLabel(release_mem->Address<VAddr>(),
                     (FenceAtGpu() == 1 || FenceAtGpu() == 2) && rasterizer &&
                         release_mem->data_sel != DataSelect::GdsMemStore);
    }
    if ((FenceAtGpu() == 1 || FenceAtGpu() == 2) && rasterizer &&
        release_mem->data_sel != DataSelect::GdsMemStore) {
        rasterizer->DeferToGpu(
            [packet = *release_mem, pipe_id] {
                packet.SignalFence(
                    [pipe_id] {
                        Platform::IrqC::Instance()->Signal(
                            static_cast<Platform::InterruptId>(pipe_id));
                    },
                    [](VAddr, u16, u16) {});
                NoteAscFence(packet);
            },
            FenceAtGpu() == 1);
        return;
    }
    release_mem->SignalFence(
        [pipe_id] {
            Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
        },
        [this](VAddr dst, u16 gds_index, u16 num_dwords) {
            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
        });
    NoteAscFence(*release_mem);
}

Liverpool::SplitSpanScope::SplitSpanScope(Liverpool& liverpool, const std::span<const u32>* span,
                                          CpSplit::DrainReason entry_reason) {
    if (liverpool.split_task_guard == 0) [[likely]] {
        return;
    }
    lp = &liverpool;
    for (u32 i = 0; i < SplitSpanSlots; ++i) {
        if (!lp->split_spans[i]) {
            lp->split_spans[i] = span;
            slot = static_cast<s32>(i);
            lp->split_span_hwm = std::max(lp->split_span_hwm, i + 1);
            break;
        }
    }
    if (slot < 0) {
        ++lp->split_span_untracked;
    }
    // Nothing of this buffer has been parsed yet.
    lp->SplitGuardSpan(*span, entry_reason);
}

Liverpool::SplitSpanScope::~SplitSpanScope() {
    if (!lp) {
        return;
    }
    if (slot < 0) {
        --lp->split_span_untracked;
        return;
    }
    lp->split_spans[slot] = nullptr;
    while (lp->split_span_hwm > 0 && !lp->split_spans[lp->split_span_hwm - 1]) {
        --lp->split_span_hwm;
    }
}

void Liverpool::SplitGuardSpan(std::span<const u32> span, CpSplit::DrainReason reason) {
    if (split->BeginGuardCheck(reason)) {
        const auto lo = reinterpret_cast<VAddr>(span.data());
        split->GuardSpan(lo, lo + span.size_bytes(), reason);
    }
}

void Liverpool::SplitGuardActive() {
    if (!split->BeginGuardCheck(CpSplit::DrainReason::TaskGuardResume)) {
        return;
    }
    if (split_span_untracked != 0) {
        split->Sync(CpSplit::DrainReason::TaskGuardOverflow);
        return;
    }
    for (u32 i = 0; i < split_span_hwm; ++i) {
        if (const auto* span = split_spans[i]) {
            const auto lo = reinterpret_cast<VAddr>(span->data());
            if (split->GuardSpan(lo, lo + span->size_bytes(),
                                 CpSplit::DrainReason::TaskGuardResume)) {
                return;
            }
        }
    }
}

bool Liverpool::SplitSpansOverlap(VAddr lo, VAddr hi) const {
    if (split_span_untracked != 0) {
        return true;
    }
    for (u32 i = 0; i < split_span_hwm; ++i) {
        if (const auto* span = split_spans[i]) {
            const auto s_lo = reinterpret_cast<VAddr>(span->data());
            if (lo < s_lo + span->size_bytes() && s_lo < hi) {
                return true;
            }
        }
    }
    return false;
}

namespace {
// SHADPS4_DRAW_TAG=1: see gnmdriver.cpp. The front copies the job thread's tag from the
// DrawIndex trailing NOP into the draw record (SplitDrawRecord::pad) and times EmitDraw.
const bool g_draw_tag_front = [] {
    const char* v = std::getenv("SHADPS4_DRAW_TAG");
    return v && v[0] == '1';
}();
struct FrontTagStats {
    u64 c_tagged{}, n_tagged{}, c_untagged{}, n_untagged{};
} g_front_tag;
} // namespace

namespace {
// SHADPS4_NATIVE_REC=1 (native_rec.h): the front checks every job-thread record against its own
// state for the same draw - user data (each register the bound VS / PS reads), index base /
// count, instance count - and that a run keeps one pipeline and one context epoch. Front thread.
struct NativeRecFrontStats {
    u64 draws{}, tagged{}, no_ring{}, stale{}, heads{}, members{}, broken{}, pipe_moved{},
        epoch_moved{}, ud_ok{}, ud_bad{}, ud_unknown{}, idx_bad{}, inst_bad{}, inst_unknown{},
        other_stage{}, run_draws_2plus{}, gap_dw{};
    // Stage a2: the job thread's own derivation vs the front's.
    u64 derived{}, not_derived{}, d_pipe_diff{}, d_flat_ok{}, d_flat_bad{}, d_vi_ok{}, d_vi_bad{},
        d_stale{}, members_derived{}, pipe_moved_derived{};
    std::array<u64, 8> hist{}; // run length of heads: 1, 2-3, 4-7, ..., 128+
    u32 run_tag{};
    u32 run_head{};
    u32 run_next_k{};
    const void* run_pipeline{};
    u64 run_epoch{};
    u32 logged{};
};
NativeRecFrontStats g_nrf;

u32 NativeNeedCached(const Shader::Info* info, const Shader::Gcn::FetchShaderData* fetch);

/// Stage a2's publish: (VS hash, PS hash, fetch shader) -> pipeline for the job threads, from the
/// front (NativeRecFront) or a lean worker (any thread). `vud`: the draw's VS user data (empty:
/// the shared Info's). Returns the VS Info if the draw qualifies (Vertex + optional Fragment only).
const Shader::Info* NativeRecPublishProgramImpl(const Vulkan::GraphicsPipeline* pipeline,
                                                std::span<const u32> vud,
                                                const Shader::Info** psi_out) {
    namespace NR = AmdGpu::NativeRec;
    const auto stages = pipeline->GetStages();
    const Shader::Info* vsi = stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
    const Shader::Info* psi = stages[static_cast<u32>(Shader::LogicalStage::Fragment)];
    if (psi_out) {
        *psi_out = psi;
    }
    bool other = false;
    for (u32 i = 0; i < stages.size(); ++i) {
        other = other || (stages[i] && i != static_cast<u32>(Shader::LogicalStage::Vertex) &&
                          i != static_cast<u32>(Shader::LogicalStage::Fragment));
    }
    if (!vsi || vsi->stage != Shader::Stage::Vertex || other) {
        return nullptr;
    }
    u64 fetch = 0;
    u32 fb = 0xffffffffu;
    if (vsi->has_fetch_shader) {
        const u32 b = vsi->fetch_shader_sgpr_base;
        fb = b + 1;
        if (vud.empty()) {
            vud = vsi->user_data;
        }
        if (b + 1 < vud.size()) {
            fetch = (u64(vud[b + 1]) << 32 | vud[b]) & 0xFFFFFFFFFFFFull;
        }
    }
    NR::VsPublish(vsi->pgm_hash, fb);
    NR::TablePublish(vsi->pgm_hash, psi ? psi->pgm_hash : 0, fetch, pipeline, [&] {
        if (!NR::g_needs_on) {
            return ~u64{0};
        }
        const u64 nv = NativeNeedCached(vsi, pipeline->GetFetchShader());
        const u64 np = psi ? NativeNeedCached(psi, nullptr) : 0;
        return nv | (np << 32);
    });
    return vsi;
}

void NativeRecFront(const PM4Header* header, const Vulkan::Rasterizer::SplitDraw& draw,
                    const Regs& regs, u64 ctx_epoch,
                    const AmdGpu::CpSplit::VertexInputsDerived* front_vi,
                    const Vulkan::GraphicsLookupScratch* cap) {
    namespace NR = AmdGpu::NativeRec;
    auto& st = g_nrf;
    ++st.draws;
    const auto log = [&] {
        if ((st.draws & 0x3ffff) != 0) {
            return;
        }
        std::string h;
        for (u32 b = 0; b < 8; ++b) {
            h += fmt::format(" {}", st.hist[b]);
        }
        LOG_INFO(Render,
                 "NativeRec front: draws={} tagged={} no_ring={} stale={} | runs={} members={} "
                 "(draws in runs of 2+ {:.1f}%) broken={} pipeline_moved={} epoch_moved={} | ud "
                 "regs ok={} MISMATCH={} unknown={} | index MISMATCH={} instances MISMATCH={} "
                 "unknown={} | other-stage draws={} | gap dwords per member {:.1f} | heads by run "
                 "length 1/2-3/4-7/8-15/16-31/32-63/64-127/128+:{}",
                 st.draws, st.tagged, st.no_ring, st.stale, st.heads, st.members,
                 st.tagged ? 100.0 * double(st.run_draws_2plus) / double(st.tagged) : 0.0,
                 st.broken, st.pipe_moved, st.epoch_moved, st.ud_ok, st.ud_bad, st.ud_unknown,
                 st.idx_bad, st.inst_bad, st.inst_unknown, st.other_stage,
                 st.members ? double(st.gap_dw) / double(st.members) : 0.0, h);
        LOG_INFO(Render,
                 "NativeRec front a2: derived={} not_derived={} pipeline differs={} | flat ok={} "
                 "MISMATCH={} | vertex inputs ok={} MISMATCH={} | payload stale={} | derived "
                 "members={} of which the pipeline moved={}",
                 st.derived, st.not_derived, st.d_pipe_diff, st.d_flat_ok, st.d_flat_bad,
                 st.d_vi_ok, st.d_vi_bad, st.d_stale, st.members_derived, st.pipe_moved_derived);
    };
    const auto* w = reinterpret_cast<const u32*>(header);
    if (header->type3.opcode != PM4ItOpcode::DrawIndex2 || w[6] != 0xc0021000u ||
        w[7] != NR::Magic) {
        return log();
    }
    ++st.tagged;
    const u32 tag = w[8];
    const u32 idx = w[9];
    const NR::Record* ring = NR::g_rings[tag >> 28].load(std::memory_order_acquire);
    if (!ring) {
        ++st.no_ring;
        return log();
    }
    const NR::Record& src = ring[idx & (NR::RingSize - 1)];
    if (src.stamp.load(std::memory_order_acquire) != idx + 1 || src.cmd != w) {
        ++st.stale;
        return log();
    }
    const u32 run_k = src.run_k;
    const u32 head_idx = src.head_idx;
    const auto ud_vs = src.ud_vs;
    const auto ud_ps = src.ud_ps;
    const u32 vs_known = src.vs_known, ps_known = src.ps_known;
    const u64 index_addr = src.index_addr;
    const u32 index_count = src.index_count, instances = src.instances;
    const bool inst_known = src.inst_known != 0;
    const u32 gap = src.gap_dwords;
    const u32 count = run_k == 0 ? src.count.load(std::memory_order_acquire) : 0;
    const bool derived = src.derived != 0;
    const void* d_pipeline = src.pipeline;
    const u32 d_nvs = src.flat_vs_dw, d_nps = src.flat_ps_dw, d_vi_bytes = src.vi_bytes;
    const u64 d_pos = src.payload_pos;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (src.stamp.load(std::memory_order_relaxed) != idx + 1) {
        ++st.stale;
        return log();
    }
    if (run_k == 0) {
        ++st.heads;
        st.hist[std::min<u32>(std::bit_width(std::max<u32>(count, 1)) - 1, 7u)] += 1;
        st.run_draws_2plus += count >= 2 ? count : 0;
        st.run_tag = tag;
        st.run_head = idx;
        st.run_next_k = 1;
        st.run_pipeline = draw.pipeline;
        st.run_epoch = ctx_epoch;
    } else {
        ++st.members;
        st.gap_dw += gap;
        if (tag != st.run_tag || head_idx != st.run_head || run_k != st.run_next_k) {
            ++st.broken;
        } else {
            st.pipe_moved += draw.pipeline != st.run_pipeline;
            if (src.derived) {
                ++st.members_derived;
                st.pipe_moved_derived += draw.pipeline != st.run_pipeline;
            }
            st.epoch_moved += ctx_epoch != st.run_epoch;
        }
        st.run_next_k = run_k + 1;
    }
    if (regs.index_base_address.Address() != index_addr || regs.num_indices != index_count) {
        if (++st.idx_bad <= 8) {
            LOG_WARNING(Render, "NativeRec MISMATCH index: front {:#x} x{} record {:#x} x{}",
                        regs.index_base_address.Address(), regs.num_indices, index_addr,
                        index_count);
        }
    }
    if (!inst_known) {
        ++st.inst_unknown;
    } else if (regs.num_instances.NumInstances() != std::max<u32>(instances, 1)) {
        ++st.inst_bad;
    }
    if (draw.pipeline && draw.draw_class == Vulkan::Rasterizer::DrawClass::Draw && !draw.skip) {
        for (const Shader::Info* info : draw.pipeline->GetStages()) {
            if (!info) {
                continue;
            }
            const bool vs = info->l_stage == Shader::LogicalStage::Vertex &&
                            info->stage == Shader::Stage::Vertex;
            const bool ps = info->l_stage == Shader::LogicalStage::Fragment;
            if (!vs && !ps) {
                ++st.other_stage;
                continue;
            }
            const auto& rec_ud = vs ? ud_vs : ud_ps;
            const u32 known = vs ? vs_known : ps_known;
            std::span<const u32> info_ud = info->user_data;
            if (cap) {
                info_ud = cap->stages[static_cast<u32>(info->l_stage)].user_data;
            }
            // Every user-data register the stage is given (RSRC2 USER_SGPR): direct reads, the
            // SRT walker's pointers and the fetch shader address all come from these.
            const u32 n_ud = std::min<u32>(
                vs ? regs.vs_program.settings.num_user_regs : regs.ps_program.settings.num_user_regs,
                16u);
            for (u32 i = 0; i < n_ud; ++i) {
                if (i >= info_ud.size()) {
                    continue;
                }
                if ((known >> i & 1) == 0) {
                    ++st.ud_unknown;
                } else if (rec_ud[i] != info_ud[i]) {
                    if (++st.ud_bad <= 16) {
                        LOG_WARNING(Render,
                                    "NativeRec MISMATCH ud: stage {} reg {} front {:#x} record "
                                    "{:#x} (run k {}, gap {} dw)",
                                    vs ? "VS" : "PS", i, info_ud[i], rec_ud[i], run_k,
                                    gap);
                    }
                } else {
                    ++st.ud_ok;
                }
            }
        }
    }
    // Stage a2: publish (VS hash, PS hash, fetch shader) -> pipeline for the job threads, then
    // compare what the job thread derived for this draw.
    const Shader::Info* vsi = nullptr;
    const Shader::Info* psi = nullptr;
    if (draw.pipeline && draw.draw_class == Vulkan::Rasterizer::DrawClass::Draw && !draw.skip) {
        std::span<const u32> vud{};
        if (cap) {
            vud = cap->stages[static_cast<u32>(Shader::LogicalStage::Vertex)].user_data;
        }
        vsi = NativeRecPublishProgramImpl(draw.pipeline, vud, &psi);
    }
    if (!derived) {
        ++st.not_derived;
    } else if (vsi) {
        ++st.derived;
        st.d_pipe_diff += d_pipeline != static_cast<const void*>(draw.pipeline);
        const u8* payload = NR::g_payloads[tag >> 28].load(std::memory_order_acquire);
        if (payload) {
            const u8* base = payload + (d_pos & (NR::PayloadSize - 1));
            const u32* fvs = reinterpret_cast<const u32*>(base);
            const u32* fps = fvs + d_nvs;
            // The flat buffer starts with the 16 user-data registers; only the stage's own
            // (RSRC2 count) are read, the SRT walker's part after them always is.
            const auto same_flat = [](const u32* job, std::span<const u32> front, u32 n,
                                      u32 n_user, u32 need) {
                if (n != front.size()) {
                    return false;
                }
                for (u32 i = 0; i < n; ++i) {
                    if ((i >= 16 || (i < n_user && ((need >> i) & 1))) && job[i] != front[i]) {
                        return false;
                    }
                }
                return true;
            };
            const u32 need_vs = NR::g_needs_on
                                    ? NativeNeedCached(vsi, draw.pipeline->GetFetchShader())
                                    : 0xffffffffu;
            const u32 need_ps = NR::g_needs_on && psi ? NativeNeedCached(psi, nullptr) : 0xffffffffu;
            // Inherited registers (needed, unknown to the job) are stale on purpose (sec. 11).
            bool flat_ok = same_flat(fvs, vsi->flattened_ud_buf, d_nvs,
                                     regs.vs_program.settings.num_user_regs, need_vs & vs_known);
            if (psi) {
                flat_ok = flat_ok && same_flat(fps, psi->flattened_ud_buf, d_nps,
                                               regs.ps_program.settings.num_user_regs,
                                               need_ps & ps_known);
            }
            bool vi_ok = true;
            if (front_vi) {
                const auto* hdr = reinterpret_cast<const AmdGpu::CpSplit::VertexInputsPayload*>(
                    base + (u64(d_nvs + d_nps) * 4 + 7) / 8 * 8);
                const auto* att =
                    reinterpret_cast<const AmdGpu::CpSplit::VertexInputAttrib*>(hdr + 1);
                const auto* rng =
                    reinterpret_cast<const AmdGpu::CpSplit::VertexInputRange*>(att + hdr->count);
                vi_ok = d_vi_bytes == front_vi->Bytes() && hdr->count == front_vi->header.count &&
                        hdr->range_count == front_vi->header.range_count;
                for (u32 a = 0; vi_ok && a < hdr->count; ++a) {
                    const auto& x = att[a];
                    const auto& y = front_vi->attribs[a];
                    vi_ok = x.desc.vk_format == y.desc.vk_format &&
                            x.desc.stride == y.desc.stride &&
                            x.desc.location == y.desc.location &&
                            x.desc.instanced == y.desc.instanced &&
                            x.base_address == y.base_address && x.size == y.size &&
                            x.range == y.range;
                }
                for (u32 g = 0; vi_ok && g < hdr->range_count; ++g) {
                    vi_ok = rng[g].base_address == front_vi->ranges[g].base_address &&
                            rng[g].end_address == front_vi->ranges[g].end_address;
                }
            }
            if (src.stamp.load(std::memory_order_acquire) != idx + 1) {
                ++st.d_stale;
            } else {
                (flat_ok ? st.d_flat_ok : st.d_flat_bad) += 1;
                if (front_vi) {
                    (vi_ok ? st.d_vi_ok : st.d_vi_bad) += 1;
                }
                if ((!flat_ok || (front_vi && !vi_ok)) && ++st.logged <= 16) {
                    LOG_WARNING(Render,
                                "NativeRec a2 MISMATCH: flat {} vi {} (VS flat {} vs {} dw, "
                                "pipeline {})",
                                flat_ok ? "ok" : "BAD", vi_ok ? "ok" : "BAD", d_nvs,
                                vsi->flattened_ud_buf.size(),
                                d_pipeline == static_cast<const void*>(draw.pipeline)
                                    ? "same"
                                    : "differs");
                }
            }
        }
    }
    log();
}

} // namespace

namespace {
// SHADPS4_NATIVE_REC=2 (stage b, notes/native-path-2026-10-09.md sec. 6): the front hands the
// members of a job-thread run to the back without its own pipeline lookup and derivation. The
// head went through EmitDraw normally and opened the run (NativeRecRun); each member's draw
// record is built from the job thread's record: the head's pipeline, the member's own user data,
// flattened user data and program bases, the job-derived vertex inputs with the head's
// descriptions. =3: the member record is built into a scratch buffer and compared with the
// normal EmitDraw record of the same draw (which still renders).
struct NativeRecRun {
    bool active{};
    u32 tag{}, head_idx{}, next_k{};
    u64 ctx_epoch{};
    const Vulkan::GraphicsPipeline* pipeline{};
    const Shader::Info* vsi{};
    const Shader::Info* psi{};
    u64 desc_id{};
    u32 desc_count{};
    u32 prim{}, index_type{}, vs_nuser{}, ps_nuser{}; ///< front registers at the head (review 4)
    std::array<CpSplit::VertexInputDesc, CpSplit::MaxVertexInputs> descs{};
};
NativeRecRun g_nr_run;
struct NativeRecMemberStats {
    u64 members{}, refused{}, c_member{}, c_normal{}, n_normal{};
    u64 cmp_ok{}, cmp_bad{};
    std::array<u64, 8> why{};
    u32 logged{};
} g_nr_mst;
std::vector<u8> g_nr_scratch;
bool g_nr_scratch_valid{};

// SHADPS4_NATIVE_SEGB (notes/native-path-2026-10-09.md sec. 20, milestone 3 brick B): the front
// groups consecutive draws into the segments a native Segment record could execute - every
// member has a derived job record with a GPU record, the segment's pipeline with a published
// ring (+ pt when SHADPS4_NATIVE_PT is on) bindless colour variant for its vertex description,
// the same index type, instance offset 0, only whitelisted packets since the previous member
// (user-data SET_SH_REG, plain NOP, INDEX_TYPE, NUM_INSTANCES, INDEX_BASE, INDEX_BUFFER_SIZE),
// fewer than 64 members, and per-class unions (index range, each pulled vertex stream, each
// Guest binding slot) within 64 MB - and predicts each member's indirect command. Census and
// verify only: every draw is still posted and the back compares the prediction with what it
// draws (NativeSegBBack). The Guest slot ranges come from the job run's union (RunUnion, CPU
// memory): a superset of the draw's own, since the GPU record ring is write-combined memory
// (a read there costs ~10 k cycles, s395). Front thread.
namespace SegB {
constexpr u32 MaxMembers = 64;
constexpr u64 MaxUnion = 64ull << 20;
/// SEGX=1: the embedded member records of one Segment record (the channel takes 2 MB records).
constexpr u32 SegMaxRecordBytes = 768u << 10;
/// Why a draw stayed outside (1-9, also closing the open segment) or a segment closed (10-20).
enum Reason : u32 {
    RNone = 0,
    RNoTag,          ///< not a tagged DrawIndex2 (every other draw packet included)
    RStale,          ///< no job ring, or the record was overwritten
    RNoGpuRec,       ///< not derived by the job thread, or no GPU record
    RNoVariant,      ///< no usable colour variant (split by NvReason)
    RVariantNoRing,  ///< the variant is not ring (+ pt) + bindless
    RInstanceOffset, ///< the fetch shader's instance offset SGPR is not 0
    RNotPlain,       ///< filtered / skipped / no pipeline / no vertex inputs
    RNoUnion,        ///< the job run's union entry is gone (NativeRecGpu killed it)
    RPipeline,       ///< another pipeline
    RIndexType,      ///< the index type changed
    RPacket,         ///< an off-whitelist packet since the previous member (kind in pk[])
    RCount,          ///< 64 members
    RIndexUnion,     ///< the index union would pass 64 MB
    RVertexUnion,
    RGuestUnion,     ///< (SEGB only: with SEGX the back checks the job-run unions)
    RMisaligned,     ///< the index address is not a whole number of indices from the opener's
    RInheritedDiff,  ///< v2: the member's inherited set / values differ from the opener's
    RMaterial,       ///< v3 (SEGX): a second material (image-key hashes)
    ROps,            ///< v3 (SEGX): the op list is full
    RRecordFull,     ///< v2 (SEGX=1): the members' records would pass the Segment record cap
    NumReasons,
};
enum PacketKind : u32 {
    PkNone = 0,
    PkCtx,      ///< v2: a SET_CONTEXT_REG that changes a value (or CONFIG / CONTEXT_CONTROL / CLEAR_STATE)
    PkSh,       ///< SET_SH_REG outside the VS / PS user-data registers
    PkUconfig,  ///< v2: a SET_UCONFIG_REG other than index type / counts, or a changed primitive type
    PkNop,      ///< a NOP with a back effect (flip, guest marker)
    PkDispatch,
    PkEvent,    ///< EVENT_WRITE / EOP / EOS / RELEASE_MEM
    PkDma,      ///< DMA_DATA / WRITE_DATA / COPY_DATA
    PkWait,     ///< WAIT_REG_MEM, semaphores, COND_EXEC, PFP_SYNC_ME, ACQUIRE_MEM, REWIND
    PkIb,
    PkSubmit,   ///< a new top-level command buffer (ProcessGraphics entry)
    PkOther,
    NumPacketKinds,
};
/// v2: the no-variant split (why the pipeline has no usable colour variant for this draw).
enum NvReason : u32 {
    NvShadow = 0,    ///< the pipeline has a shadow / legacy native_variant (out of v1)
    NvNotEligible,   ///< PrepareNativeColour refused the pipeline (native_colour_why)
    NvUndecided,     ///< PrepareNativeColour has not run for the pipeline yet
    NvNotClaimed,    ///< eligible, but no slot holds this vertex description (slots full / rate)
    NvRefused,       ///< the slot is claimed and its build was refused (no push descriptors,
                     ///< per-description rule; native_colour_why)
    NvBuilding,      ///< the slot is claimed, the variant not published yet
    NumNv,
};
/// VGT uconfig registers a member's command / record carries (SET_UCONFIG_REG whitelist).
constexpr u32 UcPrim = WordOf(offsetof(Regs, primitive_type)) - Regs::UconfigRegWordOffset;
constexpr u32 UcIndexType = 0x243; // VGT_INDEX_TYPE (the front keeps it in index_buffer_type)
constexpr u32 UcNumIndices = WordOf(offsetof(Regs, num_indices)) - Regs::UconfigRegWordOffset;
constexpr u32 UcNumInstances = WordOf(offsetof(Regs, num_instances)) - Regs::UconfigRegWordOffset;
/// v3 (SEGX): context registers that become ops inside a segment (dynamic state on the back).
constexpr u32 CtxStencilF = WordOf(offsetof(Regs, stencil_ref_front)) - Regs::ContextRegWordOffset;
constexpr u32 CtxStencilB = WordOf(offsetof(Regs, stencil_ref_back)) - Regs::ContextRegWordOffset;
constexpr u32 CtxPolyCtl = WordOf(offsetof(Regs, polygon_control)) - Regs::ContextRegWordOffset;
static_assert(CtxStencilF == 0x10C && CtxStencilB == 0x10D && CtxPolyCtl == 0x205);
struct Pred {
    u32 id{}, member{};
    u64 index_addr{};
    u32 index_count{}, instances{};
};
/// A small "by key" histogram (register offsets, reason strings), first come first kept.
template <typename K>
struct KeyHist {
    std::array<std::pair<K, u64>, 24> e{};
    u32 n{};
    u64 overflow{};
    void Note(K k) {
        for (u32 i = 0; i < n; ++i) {
            if (e[i].first == k) {
                ++e[i].second;
                return;
            }
        }
        if (n < e.size()) {
            e[n++] = {k, 1};
        } else {
            ++overflow;
        }
    }
    void Reset() {
        n = 0;
        overflow = 0;
    }
};
struct State {
    bool open{};
    u32 next_id{}, id{}, n{};
    const void* pipeline{};
    u32 index_type{}, index_size{};
    u32 prim{}; ///< v2: the primitive type register at the open
    u64 addr0{}; ///< the opener's index address (the alignment reference: lo may still drop)
    u64 ilo{}, ihi{};
    std::array<u64, 16> vlo{}, vhi{};
    u64 gmask{};
    std::array<u64, AmdGpu::NativeRec::UnionSlots> glo{}, ghi{};
    u32 nmat{};
    std::array<u64, Shader::NativeSegMaxMaterials> mat_h1{}, mat_h2{};
    std::array<u32, Shader::NativeSegMaxMaterials> mat_rep{}; ///< v3 item 4: first member per material
    // v2: the opener's inherited set and the front's values of those registers.
    u32 inh_vs{}, inh_ps{}, push_inh{};
    u64 inh_slots{};
    std::array<u32, 16> inh_val_vs{}, inh_val_ps{};
    u32 packet{}; ///< first off-whitelist packet kind since the last draw (PkNone: clean)
    /// v3 item 3 (SEGX): the context block at the segment open (stencil / cull ops update it)
    /// and whether a value-changing context write is pending the next member's block compare.
    std::array<u32, NativeCtxBlockWords> ctx0{};
    bool ctx_dirty{};
    // SHADPS4_NATIVE_SEGX (sec. 23): the segment as the Segment record carries it. `segx`: a
    // Segment record is posted at every close; `exec` (=1): member Draw records are withheld.
    bool init{}, segx{}, exec{};
    CpSplit::Channel* split{};
    bool run_ahead{};
    std::vector<u8> recs; ///< the members' Draw records (member 0 only in =2), kept at their post
    u32 rec_bytes{}, n_rec{};
    std::array<u32, MaxMembers> rec_off{};
    std::array<u64, MaxMembers> m_addr{};
    std::array<u32, MaxMembers> m_count{}, m_inst{}, m_entry{}, m_vo{};
    std::array<CpSplit::SegOp, CpSplit::SegMaxOps> ops{};
    u32 n_ops{};
    u32 range_first{}, range_count{}; ///< the open DRAW_RANGE
    u32 init_sf{}, init_sb{}, init_pc{}, cur_sf{}, cur_sb{}, cur_pc{};
    bool st_dirty{}, cull_dirty{}, any_st{}, any_cull{};
    u64 ibase0{}; ///< member 0's index base / count / NUM_INSTANCES registers (raw)
    u32 nidx0{}, ninst0{};
    // Counters, reset at every log.
    u64 draws{}, in_seg{}, segments{}, mat_sum{}, inh_members{};
    u64 posted{}, posted_members{}, op_range{}, op_st{}, op_cull{}, singles{};
    std::array<u64, 7> hist{}; ///< closed segments by member count 1/2-3/4-7/8-15/16-31/32-63/64
    std::array<u64, 4> mat_hist{};
    std::array<u64, NumReasons> close{}, outside{};
    std::array<u64, NumPacketKinds> pk{};
    std::array<u64, NumNv> nv{};
    KeyHist<u32> uc_off{}, ctx_off{};   ///< registers that closed segments (0xffff: CLEAR / CONTROL / CONFIG)
    KeyHist<u32> sh_off{};              ///< SET_SH_REG offsets (first register) that closed segments
    u64 drop_st{}, drop_cull{};         ///< SEGX: state changes pending at a close (no member followed)
    u64 ctx_cont{}, dma_cont{};         ///< v3 item 3: members continued by the block compare / no-op DMAs
    KeyHist<const char*> why{};         ///< native_colour_why of not-eligible / refused pipelines
    u32 calls{};
    std::chrono::steady_clock::time_point window{};
};
State g_state;
thread_local std::vector<u8> g_scratch; ///< SEGX=1: a withheld member's record is built here

void Close(State& s, u32 why);
void FlushHold(State& s);

/// v3 item 3: a record the journal hold may not reorder against is about to be reserved - the
/// open segment closes here (its Segment record first, then the held journal).
void HoldInterrupt(void*) {
    auto& s = g_state;
    if (s.open) {
        ++s.pk[PkOther];
        Close(s, RPacket);
    } else {
        FlushHold(s);
    }
}

/// Front thread, once: the channel and the SEGX gates (never under CP-split VERIFY).
void Init(State& s, CpSplit::Channel* split, bool run_ahead, bool verify) {
    s.init = true;
    s.split = split;
    s.run_ahead = run_ahead;
    s.segx = AmdGpu::NativeRec::g_segx != 0 && !verify;
    s.exec = AmdGpu::NativeRec::g_segx == 1 && !verify;
    if (s.segx && split) {
        split->SetHoldInterrupt(&HoldInterrupt, nullptr);
    }
}

void FlushHold(State& s) {
    s.ctx_dirty = false;
    if (s.split) {
        s.split->FlushHold();
    }
}

SplitDrawRecord* ScratchRecord(u32 bytes) {
    g_scratch.resize(bytes);
    return reinterpret_cast<SplitDrawRecord*>(g_scratch.data());
}

/// A member's record bytes, kept until the segment closes (member 0 in =2; every member in =1,
/// for the replay fallback).
void KeepMember(const void* rec, u32 bytes) {
    auto& s = g_state;
    if (s.n_rec >= MaxMembers) {
        return;
    }
    const u32 at = (s.rec_bytes + 15u) & ~15u;
    s.recs.resize(at + bytes);
    std::memcpy(s.recs.data() + at, rec, bytes);
    s.rec_off[s.n_rec++] = at;
    s.rec_bytes = at + bytes;
}

/// SEGX: one Segment record for the open segment, in stream position (before whatever closes
/// it). Member commands against the final index union lo; state ops first restore the values
/// member 0 ran with (the back's mirror holds the LAST member's), then the recorded ops.
void PostSegment(State& s) {
    if (!s.split || s.n == 0 || s.rec_bytes == 0) {
        return;
    }
    // v3 item 3: the Segment record precedes any journal the builder held back.
    const std::vector<u32> held = s.split->TakeHold();
    s.ctx_dirty = false;
    s.drop_st += s.st_dirty;
    s.drop_cull += s.cull_dirty;
    std::array<CpSplit::SegOp, CpSplit::SegMaxOps + 3> ops{};
    u32 n_ops = 0;
    if (s.any_st) {
        ops[n_ops++] = {CpSplit::SegOpStencil, s.init_sf, s.init_sb, 0};
    }
    if (s.any_cull) {
        ops[n_ops++] = {CpSplit::SegOpCull, s.init_pc, 0, 0};
    }
    for (u32 i = 0; i < s.n_ops; ++i) {
        ops[n_ops++] = s.ops[i];
    }
    if (s.range_count != 0) {
        ops[n_ops++] = {CpSplit::SegOpDrawRange, s.range_first, s.range_count, 0};
    }
    const u32 recs_al = (s.rec_bytes + 15u) & ~15u;
    const u32 bytes = static_cast<u32>(sizeof(CpSplit::SplitSegmentRecord)) + recs_al +
                      n_ops * u32(sizeof(CpSplit::SegOp)) + s.n * 20u + s.n * 4u;
    auto* seg = s.split->Reserve<CpSplit::SplitSegmentRecord>(CpSplit::RecordType::Segment,
                                                              (bytes + 15u) & ~15u);
    *seg = CpSplit::SplitSegmentRecord{};
    seg->id = s.id;
    seg->n = s.n;
    seg->n_ops = n_ops;
    seg->draw_bytes = s.rec_bytes;
    seg->n_rec = s.n_rec;
    seg->rec_off = s.rec_off;
    seg->index_type = s.index_type;
    seg->index_size = s.index_size;
    seg->num_indices0 = s.nidx0;
    seg->num_instances0 = s.ninst0;
    seg->index_base0 = s.ibase0;
    seg->ilo = s.ilo;
    seg->ihi = s.ihi;
    seg->gmask = s.gmask;
    for (u32 a = 0; a < 16; ++a) {
        if (s.vhi[a] > s.vlo[a]) {
            seg->vmask |= 1u << a;
            seg->vlo[a] = s.vlo[a];
            seg->vhi[a] = s.vhi[a];
        }
    }
    seg->glo = s.glo;
    seg->ghi = s.ghi;
    seg->n_mat = s.nmat;
    for (u32 m = 0; m < s.nmat; ++m) {
        seg->mat_h1[m] = s.mat_h1[m];
        seg->mat_h2[m] = s.mat_h2[m];
        seg->mat_rep[m] = s.mat_rep[m];
    }
    u8* p = reinterpret_cast<u8*>(seg + 1);
    std::memcpy(p, s.recs.data(), s.rec_bytes);
    p += recs_al;
    std::memcpy(p, ops.data(), n_ops * sizeof(CpSplit::SegOp));
    p += n_ops * sizeof(CpSplit::SegOp);
    for (u32 j = 0; j < s.n; ++j) {
        // vertexOffset as today's colour join writes it (the verified convention).
        const u32 cmd[5] = {s.m_count[j], s.m_inst[j],
                            static_cast<u32>((s.m_addr[j] - s.ilo) / s.index_size), s.m_vo[j], 0u};
        std::memcpy(p, cmd, sizeof(cmd));
        p += sizeof(cmd);
    }
    std::memcpy(p, s.m_entry.data(), s.n * 4);
    s.split->Commit();
    if (s.run_ahead) {
        s.split->PostRunAhead(CpSplit::PostKind::Draw);
    } else {
        s.split->Sync(CpSplit::DrainReason::DepthZero);
    }
    ++s.posted;
    s.posted_members += s.n;
    s.singles += s.n == 1;
    for (u32 i = 0; i < n_ops; ++i) {
        (ops[i].kind == CpSplit::SegOpDrawRange ? s.op_range
         : ops[i].kind == CpSplit::SegOpStencil  ? s.op_st
                                                  : s.op_cull) += 1;
    }
    s.rec_bytes = 0;
    s.n_rec = 0;
    s.split->ReplayHold(held);
}

void NoteSegment(State& s) {
    const u32 n = s.n;
    ++s.segments;
    ++s.hist[n <= 1 ? 0 : n <= 3 ? 1 : n <= 7 ? 2 : n <= 15 ? 3 : n <= 31 ? 4 : n <= 63 ? 5 : 6];
    s.mat_sum += s.nmat;
    ++s.mat_hist[s.nmat <= 1 ? 0 : s.nmat == 2 ? 1 : s.nmat <= 4 ? 2 : 3];
    s.open = false;
}

void Close(State& s, u32 why) {
    if (s.open) {
        ++s.close[why];
        if (s.segx) {
            PostSegment(s);
        }
        NoteSegment(s);
    }
    FlushHold(s);
}

template <typename K, typename F>
std::string TopKeys(KeyHist<K>& h, F&& fmt_key) {
    std::sort(h.e.begin(), h.e.begin() + h.n,
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string out;
    for (u32 i = 0; i < h.n && i < 10; ++i) {
        out += fmt::format(" {}:{}", fmt_key(h.e[i].first), h.e[i].second);
    }
    if (h.overflow != 0) {
        out += fmt::format(" (+{} in other keys)", h.overflow);
    }
    return out.empty() ? std::string{" -"} : out;
}

void MaybeLog(State& s) {
    if ((++s.calls & 1023) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (s.window == std::chrono::steady_clock::time_point{}) {
        s.window = now;
        return;
    }
    if (now - s.window < std::chrono::seconds(10)) {
        return;
    }
    s.window = now;
    const auto& c = s.close;
    const auto& o = s.outside;
    const auto& p = s.pk;
    LOG_INFO(Render,
             "NativeSegB front: draws={} in_segments={} ({:.1f}%) segments={} | members per "
             "segment 1/2-3/4-7/8-15/16-31/32-63/64: {} {} {} {} {} {} {} | close: pipeline={} "
             "index_type={} packet={} (ctx {} sh {} uconfig {} nop {} dispatch {} event {} dma {} "
             "wait {} ib {} submit {} other {}) count={} index_union={} vertex_union={} "
             "guest_union={} index_misaligned={} inherited_differs={} material={} ops_full={} "
             "record_full={} outside_draw={} | outside: no_tag={} stale={} no_gpu_rec={} no_variant={} "
             "variant_not_ring_pt={} instance_offset={} not_plain={} no_union={} | materials per "
             "segment mean {:.2f} 1/2/3-4/5-8: {} {} {} {}",
             s.draws, s.in_seg, s.draws ? 100.0 * double(s.in_seg) / double(s.draws) : 0.0,
             s.segments, s.hist[0], s.hist[1], s.hist[2], s.hist[3], s.hist[4], s.hist[5],
             s.hist[6], c[RPipeline], c[RIndexType], c[RPacket], p[PkCtx], p[PkSh], p[PkUconfig],
             p[PkNop], p[PkDispatch], p[PkEvent], p[PkDma], p[PkWait], p[PkIb], p[PkSubmit],
             p[PkOther], c[RCount], c[RIndexUnion], c[RVertexUnion], c[RGuestUnion],
             c[RMisaligned], c[RInheritedDiff], c[RMaterial], c[ROps], c[RRecordFull],
             c[RNoTag] + c[RStale] + c[RNoGpuRec] + c[RNoVariant] + c[RVariantNoRing] +
                 c[RInstanceOffset] + c[RNotPlain] + c[RNoUnion],
             o[RNoTag], o[RStale], o[RNoGpuRec], o[RNoVariant], o[RVariantNoRing],
             o[RInstanceOffset], o[RNotPlain], o[RNoUnion],
             s.segments ? double(s.mat_sum) / double(s.segments) : 0.0, s.mat_hist[0],
             s.mat_hist[1], s.mat_hist[2], s.mat_hist[3]);
    const auto hex = [](u32 k) { return fmt::format("{:#x}", k); };
    const auto str = [](const char* k) { return std::string{k ? k : "?"}; };
    LOG_INFO(Render,
             "NativeSegB front detail: no_variant: shadow={} not_eligible={} undecided={} "
             "not_claimed={} refused={} building={} | why:{} | inherited members={} | uconfig "
             "close offsets:{} | ctx close offsets:{} | sh close offsets:{} | ctx continued by "
             "block compare={} dma no-op continued={}",
             s.nv[NvShadow], s.nv[NvNotEligible], s.nv[NvUndecided], s.nv[NvNotClaimed],
             s.nv[NvRefused], s.nv[NvBuilding], TopKeys(s.why, str), s.inh_members,
             TopKeys(s.uc_off, hex), TopKeys(s.ctx_off, hex), TopKeys(s.sh_off, hex),
             s.ctx_cont, s.dma_cont);
    if (s.segx) {
        LOG_INFO(Render,
                 "NativeSegX front: mode {} segment records posted={} members={} ops: draw "
                 "ranges={} stencil={} cull={} | state changes pending at a close (dropped, no "
                 "member followed): stencil={} cull={} | singles returned={}",
                 s.exec ? 1 : 2, s.posted, s.posted_members, s.op_range, s.op_st, s.op_cull,
                 s.drop_st, s.drop_cull, s.singles);
    }
    s.draws = s.in_seg = s.segments = s.mat_sum = s.inh_members = 0;
    s.posted = s.posted_members = s.op_range = s.op_st = s.op_cull = s.singles = 0;
    s.hist.fill(0);
    s.mat_hist.fill(0);
    s.close.fill(0);
    s.outside.fill(0);
    s.pk.fill(0);
    s.nv.fill(0);
    s.uc_off.Reset();
    s.ctx_off.Reset();
    s.sh_off.Reset();
    s.why.Reset();
    s.drop_st = s.drop_cull = 0;
    s.ctx_cont = s.dma_cont = 0;
}

/// Every gfx type-3 packet the front parses (ProcessGraphics, before the packet is applied, so
/// `regs` is the shadow BEFORE the write); draws are classified in NativeSegBDraw. Keeps the
/// first off-whitelist kind since the last draw. v2: a SET_CONTEXT_REG whose every value equals
/// the shadow's (= the segment's start value, since equal writes leave it unchanged) and a
/// SET_UCONFIG_REG of the index type / counts (per-member parameters) or an unchanged primitive
/// type continue the segment. v3 (SEGX): a changed stencil ref / mask or cull register becomes
/// an op; any other closing kind closes the segment HERE (its Segment record must precede the
/// packet's own record: fences, waits and DMAs execute in stream order).
void NotePacket(PM4ItOpcode opcode, const PM4Header* header, u32 count, bool guest_markers,
                const Regs& regs) {
    auto& s = g_state;
    if (s.packet != PkNone) {
        return;
    }
    u32 kind = PkNone;
    switch (opcode) {
    case PM4ItOpcode::Nop:
        kind = NopHasBackEffect(header, guest_markers) ? PkNop : PkNone;
        break;
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
        break;
    case PM4ItOpcode::SetShReg: {
        // The job walker's ranges: VS user data 0x4c-0x5b, PS user data 0x0c-0x1b. v2 (SEGX):
        // the program registers (PGM_LO / HI / RSRC1 / RSRC2, VS 0x48-0x4b, PS 0x08-0x0b: the
        // game binds a shader copy per object) continue too - the next draw's pipeline
        // identity check closes when the pipeline really differs.
        const u32 off = reinterpret_cast<const PM4CmdSetData*>(header)->reg_offset;
        const u32 n = count - 1;
        const bool vs = off >= 0x4c && off + n <= 0x5c;
        const bool ps = off >= 0x0c && off + n <= 0x1c;
        const bool vs_pgm = s.segx && off >= 0x48 && off + n <= 0x4c;
        const bool ps_pgm = s.segx && off >= 0x08 && off + n <= 0x0c;
        kind = (vs || ps || vs_pgm || ps_pgm) ? PkNone : PkSh;
        if (kind == PkSh && s.open) {
            s.sh_off.Note(off);
        }
        break;
    }
    case PM4ItOpcode::SetContextReg: {
        const u32 off = reinterpret_cast<const PM4CmdSetData*>(header)->reg_offset;
        const u32* data = reinterpret_cast<const u32*>(header) + 2;
        for (u32 j = 0; j + 1 < count; ++j) {
            const u32 reg = off + j;
            if (reg < 0x400 && data[j] == regs.reg_array[Regs::ContextRegWordOffset + reg]) {
                continue;
            }
            if (s.segx && s.open &&
                (reg == CtxStencilF || reg == CtxStencilB || reg == CtxPolyCtl)) {
                if (reg == CtxStencilF) {
                    s.cur_sf = data[j];
                    s.st_dirty = true;
                } else if (reg == CtxStencilB) {
                    s.cur_sb = data[j];
                    s.st_dirty = true;
                } else {
                    s.cur_pc = data[j];
                    s.cull_dirty = true;
                }
                s.ctx0[reg] = data[j]; // an op: the block compare must not see it as a change
                continue;
            }
            if (s.segx && s.open && reg < NativeCtxBlockWords) {
                // v3 item 3: the back's rule - the next member's block compare decides; the
                // journal is held so a close there still posts the Segment record first.
                s.ctx_dirty = true;
                s.split->BeginHold();
                continue;
            }
            kind = PkCtx;
            if (s.open) {
                s.ctx_off.Note(reg);
            }
        }
        break;
    }
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
        kind = PkCtx;
        if (s.open) {
            s.ctx_off.Note(0xffffu);
        }
        break;
    case PM4ItOpcode::SetUconfigReg: {
        const u32 off = reinterpret_cast<const PM4CmdSetData*>(header)->reg_offset;
        const u32* data = reinterpret_cast<const u32*>(header) + 2;
        for (u32 j = 0; j + 1 < count; ++j) {
            const u32 reg = off + j;
            if (reg == UcIndexType || reg == UcNumIndices || reg == UcNumInstances ||
                (reg == UcPrim && (!s.open || data[j] == s.prim))) {
                continue;
            }
            kind = PkUconfig;
            if (s.open) {
                s.uc_off.Note(reg);
            }
        }
        break;
    }
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
        kind = PkDispatch;
        break;
    case PM4ItOpcode::EventWrite:
    case PM4ItOpcode::EventWriteEop:
    case PM4ItOpcode::EventWriteEos:
    case PM4ItOpcode::ReleaseMem:
        kind = PkEvent;
        break;
    case PM4ItOpcode::DmaData:
        // v3 item 3: the DMA_DATA to 0x3022C is skipped by the back's ExecuteGfxPacket (not a
        // sequence point, MtbBreakAtPacket): it does not close.
        if (reinterpret_cast<const PM4DmaData*>(header)->dst_addr_lo == 0x3022C) {
            s.dma_cont += s.open;
            break;
        }
        kind = PkDma;
        break;
    case PM4ItOpcode::WriteData:
    case PM4ItOpcode::CopyData:
        kind = PkDma;
        break;
    case PM4ItOpcode::WaitRegMem:
    case PM4ItOpcode::MemSemaphore:
    case PM4ItOpcode::WaitOnCeCounter:
    case PM4ItOpcode::CondExec:
    case PM4ItOpcode::PfpSyncMe:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::Rewind:
        kind = PkWait;
        break;
    case PM4ItOpcode::IndirectBuffer:
        kind = PkIb;
        break;
    default:
        kind = PkOther;
        break;
    }
    if (s.segx && s.open && kind != PkNone) {
        ++s.pk[kind];
        Close(s, RPacket); // eager: the Segment record goes out before this packet's record
        return;
    }
    s.packet = kind;
}

/// A new top-level command buffer: a sequence point (epoch bump, coherence sync) with no packet.
void NoteSubmit() {
    auto& s = g_state;
    if (s.segx && s.open) {
        ++s.pk[PkSubmit];
        Close(s, RPacket);
        return;
    }
    if (s.packet == PkNone) {
        s.packet = PkSubmit;
    }
}

/// One call per graphics draw the front posts (the member path or EmitDraw). `plain`: a draw
/// of class Draw with a pipeline that was not skipped. True = the draw joined / opened a segment
/// and `pred` is its prediction.
bool NativeSegBDraw(const PM4Header* header, const Vulkan::GraphicsPipeline* pipeline, bool plain,
                    const CpSplit::VertexInputAttrib* attribs, u32 vi_count, const Regs& regs,
                    u32 rec_bytes, Pred& pred) {
    namespace NR = AmdGpu::NativeRec;
    auto& s = g_state;
    ++s.draws;
    const u32 packet = std::exchange(s.packet, u32{PkNone});
    MaybeLog(s);
    const auto outside = [&](u32 why) {
        ++s.outside[why];
        Close(s, why);
        return false;
    };
    const auto no_variant = [&](u32 nv) {
        ++s.nv[nv];
        return outside(RNoVariant);
    };
    const auto* w = reinterpret_cast<const u32*>(header);
    if (header->type3.opcode != PM4ItOpcode::DrawIndex2 || w[6] != 0xc0021000u ||
        w[7] != NR::Magic) {
        return outside(RNoTag);
    }
    const u32 tag = w[8];
    const u32 idx = w[9];
    const u32 slot = tag >> 28;
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    if (!ring) {
        return outside(RStale);
    }
    const NR::Record& src = ring[idx & (NR::RingSize - 1)];
    if (src.stamp.load(std::memory_order_acquire) != idx + 1) {
        return outside(RStale);
    }
    // Seqlock read (Opus review 1): copy, fence, re-check the stamp, then use.
    const bool cmd_ok = src.cmd == w;
    const bool derived = src.derived != 0, gpu_ok = src.gpu_ok == 1; // q18 B: 2 = A2 translator only
    const u32 inh_vs = src.inh_vs, inh_ps = src.inh_ps, push_inh = src.push_inh;
    const u64 inh_slots = src.inh_slots;
    const u32 io_ud = src.io_ud, vo_ud = src.vo_ud;
    const u64 index_addr = src.index_addr;
    const u32 index_count = src.index_count;
    const u32 union_idx = src.union_idx, head_idx = src.head_idx;
    const u64 h1 = src.img_h1, h2 = src.img_h2;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (src.stamp.load(std::memory_order_relaxed) != idx + 1 || !cmd_ok) {
        return outside(RStale);
    }
    if (!derived || !gpu_ok) {
        return outside(RNoGpuRec);
    }
    if (!plain || !pipeline || !attribs || vi_count == 0) {
        return outside(RNotPlain);
    }
    if (io_ud != 0) {
        return outside(RInstanceOffset);
    }
    // The variant the back would bind (NativeDirectCandidate's rule): a colour variant of this
    // vertex description, bindless, with the batch table (ring) and, in pt mode, the page table.
    // v2: the no-variant cases apart (NvReason).
    if (pipeline->native_variant.load(std::memory_order_relaxed) != nullptr) {
        return no_variant(NvShadow);
    }
    if (pipeline->native_colour_static == 2) {
        s.why.Note(pipeline->native_colour_why.load(std::memory_order_relaxed));
        return no_variant(NvNotEligible);
    }
    if (pipeline->native_colour_static == 0) {
        return no_variant(NvUndecided);
    }
    const u64 vkey = VideoCore::NativeVertexKey(attribs, vi_count);
    const Vulkan::GraphicsPipeline::NativeColourSlot* cslot = nullptr;
    for (const auto& cs : pipeline->native_colour) {
        if (cs.key.load(std::memory_order_acquire) == vkey) {
            cslot = &cs;
            break;
        }
    }
    if (!cslot) {
        return no_variant(NvNotClaimed);
    }
    const auto* v = cslot->variant.load(std::memory_order_acquire);
    if (!v) {
        if (cslot->state.load(std::memory_order_relaxed) == 2) {
            s.why.Note(pipeline->native_colour_why.load(std::memory_order_relaxed));
            return no_variant(NvRefused);
        }
        return no_variant(NvBuilding);
    }
    if (!v->native_bindless ||
        v->native_table_binding == Vulkan::GraphicsPipeline::NoNativeRecords) {
        return outside(RVariantNoRing); // (pt mode needs no binding: the page table is read by pointer)
    }
    // The Guest binding ranges: the job run's union so far (CPU memory; a superset).
    NR::RunUnion* unions = NR::g_unions[slot].load(std::memory_order_acquire);
    if (!unions || union_idx == ~0u) {
        return outside(RNoUnion);
    }
    const NR::RunUnion& ru = unions[union_idx & NR::UnionMask];
    if (ru.stamp.load(std::memory_order_acquire) != head_idx + 1) {
        return outside(RNoUnion);
    }
    const u64 gmask = ru.guest_mask;
    std::array<u64, NR::UnionSlots> glo{}, ghi{};
    for (u64 m = gmask; m != 0; m &= m - 1) {
        const u32 bi = static_cast<u32>(std::countr_zero(m));
        glo[bi] = ru.lo[bi];
        ghi[bi] = ru.hi[bi];
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (ru.stamp.load(std::memory_order_relaxed) != head_idx + 1) {
        return outside(RNoUnion);
    }
    const u32 isize = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
    const u64 ilo = index_addr, ihi = index_addr + u64(index_count) * isize;
    const u32 vcount = std::min<u32>(vi_count, 16);
    // v2: the inherited registers' values as the front has them (the table carries the opener's).
    const auto inh_same = [&] {
        if (inh_vs != s.inh_vs || inh_ps != s.inh_ps || push_inh != s.push_inh ||
            inh_slots != s.inh_slots) {
            return false;
        }
        for (u32 i = 0; i < 16; ++i) {
            if (((inh_vs >> i) & 1) && regs.vs_program.user_data[i] != s.inh_val_vs[i]) {
                return false;
            }
            if (((inh_ps >> i) & 1) && regs.ps_program.user_data[i] != s.inh_val_ps[i]) {
                return false;
            }
        }
        return true;
    };
    bool seen = false;
    u32 mi = 0; // v3 item 4: the member's material index
    for (u32 i = 0; i < s.nmat && !seen; ++i) {
        seen = s.mat_h1[i] == h1 && s.mat_h2[i] == h2;
        mi = i;
    }
    if (s.open) {
        u32 why = RNone;
        bool ctx_changed = false;
        if (s.ctx_dirty) {
            // v3 item 3: context writes since the previous member - the back's block compare.
            const u32* now = &regs.reg_array[Regs::ContextRegWordOffset];
            if (NativeCtxBlockSame(now, s.ctx0.data())) {
                ++s.ctx_cont;
                FlushHold(s);
            } else {
                s.ctx_off.Note(NativeCtxBlockFirstDiff(now, s.ctx0.data()));
                ctx_changed = true;
            }
        }
        if (s.pipeline != pipeline) {
            why = RPipeline;
        } else if (s.index_type != regs.index_buffer_type.raw) {
            why = RIndexType;
        } else if (packet != PkNone) {
            why = RPacket;
            ++s.pk[packet];
        } else if (ctx_changed) {
            why = RPacket;
            ++s.pk[PkCtx];
        } else if (s.n >= MaxMembers) {
            why = RCount;
        } else if (!inh_same()) {
            why = RInheritedDiff;
        } else if ((index_addr - s.addr0) % isize != 0) {
            why = RMisaligned;
        } else if (std::max(ihi, s.ihi) - std::min(ilo, s.ilo) > MaxUnion) {
            why = RIndexUnion;
        } else if (!seen && s.nmat >= Shader::NativeSegMaxMaterials) {
            why = RMaterial; // v3 item 4: the material table is full (a 9th material)
        } else if (s.segx && s.n_ops + 3 > CpSplit::SegMaxOps) {
            why = ROps;
        } else if (s.exec && s.rec_bytes + 16u + rec_bytes > SegMaxRecordBytes) {
            why = RRecordFull;
        }
        for (u32 a = 0; why == RNone && a < vcount; ++a) {
            if (attribs[a].base_address == 0 || attribs[a].size == 0) {
                continue;
            }
            const u64 lo = std::min<u64>(s.vlo[a], attribs[a].base_address);
            const u64 hi = std::max<u64>(s.vhi[a], attribs[a].base_address + attribs[a].size);
            if (hi - lo > MaxUnion) {
                why = RVertexUnion;
            }
        }
        for (u64 m = gmask; why == RNone && !s.segx && m != 0; m &= m - 1) {
            const u32 bi = static_cast<u32>(std::countr_zero(m));
            const bool had = ((s.gmask >> bi) & 1) != 0;
            const u64 lo = had ? std::min(s.glo[bi], glo[bi]) : glo[bi];
            const u64 hi = had ? std::max(s.ghi[bi], ghi[bi]) : ghi[bi];
            if (hi - lo > MaxUnion) {
                why = RGuestUnion;
            }
        }
        if (why != RNone) {
            Close(s, why);
        }
    }
    if (!s.open) {
        s.open = true;
        s.id = ++s.next_id;
        s.n = 0;
        s.pipeline = pipeline;
        s.index_type = regs.index_buffer_type.raw;
        s.index_size = isize;
        s.prim = regs.reg_array[Regs::UconfigRegWordOffset + UcPrim];
        s.addr0 = index_addr;
        s.ilo = ~u64{0};
        s.ihi = 0;
        s.vlo.fill(~u64{0});
        s.vhi.fill(0);
        s.gmask = 0;
        s.nmat = 0;
        s.inh_vs = inh_vs;
        s.inh_ps = inh_ps;
        s.push_inh = push_inh;
        s.inh_slots = inh_slots;
        for (u32 i = 0; i < 16; ++i) {
            s.inh_val_vs[i] = regs.vs_program.user_data[i];
            s.inh_val_ps[i] = regs.ps_program.user_data[i];
        }
        // SEGX: the state member 0 runs with, the op list, member 0's draw registers.
        s.init_sf = s.cur_sf = regs.reg_array[Regs::ContextRegWordOffset + CtxStencilF];
        s.init_sb = s.cur_sb = regs.reg_array[Regs::ContextRegWordOffset + CtxStencilB];
        s.init_pc = s.cur_pc = regs.reg_array[Regs::ContextRegWordOffset + CtxPolyCtl];
        std::memcpy(s.ctx0.data(), &regs.reg_array[Regs::ContextRegWordOffset], sizeof(s.ctx0));
        s.ctx_dirty = false;
        s.st_dirty = s.cull_dirty = s.any_st = s.any_cull = false;
        s.n_ops = 0;
        s.range_first = 0;
        s.range_count = 0;
        std::memcpy(&s.ibase0, &regs.index_base_address, sizeof(u64));
        s.nidx0 = regs.num_indices;
        s.ninst0 = regs.num_instances.num_instances;
        s.rec_bytes = 0;
        s.n_rec = 0;
        seen = false;
        mi = 0;
    } else if (s.segx && (s.st_dirty || s.cull_dirty)) {
        // State changed since the previous member: close the open draw range, then the ops.
        if (s.range_count != 0) {
            s.ops[s.n_ops++] = {CpSplit::SegOpDrawRange, s.range_first, s.range_count, 0};
            s.range_count = 0;
        }
        if (s.st_dirty) {
            s.ops[s.n_ops++] = {CpSplit::SegOpStencil, s.cur_sf, s.cur_sb, 0};
            s.st_dirty = false;
            s.any_st = true;
        }
        if (s.cull_dirty) {
            s.ops[s.n_ops++] = {CpSplit::SegOpCull, s.cur_pc, 0, 0};
            s.cull_dirty = false;
            s.any_cull = true;
        }
    }
    // The member's command against the union lo as it stands (exact division: every member is
    // a whole number of indices from the opener, so from any lo).
    const u64 lo_now = std::min(s.ilo, ilo);
    const u64 first_index = (index_addr - lo_now) / isize;
    pred.id = s.id;
    pred.member = s.n;
    pred.index_addr = lo_now + first_index * isize;
    pred.index_count = index_count;
    pred.instances = regs.num_instances.NumInstances();
    s.m_addr[s.n] = index_addr;
    s.m_count[s.n] = index_count;
    s.m_inst[s.n] = pred.instances;
    // The low 24 bits of the job's ring counter (unmasked: the back ages it against g_job_next;
    // the shaders mask it to the ring), the slot, and (v3 item 4) the material index.
    if (!seen) {
        mi = s.nmat; // appended below
    }
    s.m_entry[s.n] = (slot << 24) | (idx & 0xffffffu) | (mi << Shader::NativeSegMatShift);
    s.m_vo[s.n] = regs.index_offset != 0 ? regs.index_offset : vo_ud;
    if (s.range_count == 0) {
        s.range_first = s.n;
    }
    ++s.range_count;
    s.ilo = lo_now;
    s.ihi = std::max(s.ihi, ihi);
    for (u32 a = 0; a < vcount; ++a) {
        if (attribs[a].base_address != 0 && attribs[a].size != 0) {
            s.vlo[a] = std::min<u64>(s.vlo[a], attribs[a].base_address);
            s.vhi[a] = std::max<u64>(s.vhi[a], attribs[a].base_address + attribs[a].size);
        }
    }
    for (u64 m = gmask; m != 0; m &= m - 1) {
        const u32 bi = static_cast<u32>(std::countr_zero(m));
        const bool had = ((s.gmask >> bi) & 1) != 0;
        s.glo[bi] = had ? std::min(s.glo[bi], glo[bi]) : glo[bi];
        s.ghi[bi] = had ? std::max(s.ghi[bi], ghi[bi]) : ghi[bi];
    }
    s.gmask |= gmask;
    if (!seen && s.nmat < Shader::NativeSegMaxMaterials) {
        s.mat_h1[s.nmat] = h1;
        s.mat_h2[s.nmat] = h2;
        s.mat_rep[s.nmat] = s.n;
        ++s.nmat;
    }
    if ((inh_vs | inh_ps) != 0 || inh_slots != 0) {
        ++s.inh_members;
    }
    ++s.n;
    ++s.in_seg;
    return true;
}

/// Back thread, after each draw record ran: the prediction against the mirror's draw registers
/// and what today's path did with the member (rec_outcome / segb_opened).
struct BackStats {
    u64 predicted{}, ok{}, bad{}, opener{}, join{}, confirm{}, generic{}, other{};
    u32 logged{};
    u32 calls{};
    std::chrono::steady_clock::time_point window{};
} g_back;
} // namespace SegB

void NativeSegBBack(const SplitDrawRecord& rec, const Regs& regs, Vulkan::Rasterizer& tr) {
    auto& s = SegB::g_back;
    if (rec.segb_id != 0) {
        ++s.predicted;
        const u32 isize = regs.index_buffer_type.index_type == IndexType::Index16 ? 2u : 4u;
        const u64 addr = regs.index_base_address.Address<u64>() +
                         u64(rec.draw.args.index_offset) * isize;
        const u32 count = regs.num_indices;
        const u32 inst = regs.num_instances.NumInstances();
        if (addr == rec.segb_index_addr && count == rec.segb_index_count &&
            inst == rec.segb_instances) {
            ++s.ok;
        } else {
            ++s.bad;
            if (++s.logged <= 20) {
                LOG_WARNING(Render,
                            "NativeSegB back MISMATCH: segment {} member {} (slot {} ring {}): "
                            "index address predicted {:#x} real {:#x}, count {} / {}, instances "
                            "{} / {}",
                            rec.segb_id, rec.segb_member, rec.job_tag >> 28, rec.job_idx,
                            rec.segb_index_addr, addr, rec.segb_index_count, count,
                            rec.segb_instances, inst);
            }
        }
        const u32 o = tr.rec_outcome;
        if (o == 4 || o == 5) {
            ++s.confirm;
        } else if (o == 1 || o == 2) {
            ++s.join;
        } else if (o == 3) {
            ++(tr.segb_opened ? s.opener : s.generic);
        } else {
            ++s.other;
        }
    }
    if ((++s.calls & 1023) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (s.window == std::chrono::steady_clock::time_point{}) {
        s.window = now;
        return;
    }
    if (now - s.window < std::chrono::seconds(10)) {
        return;
    }
    s.window = now;
    LOG_INFO(Render,
             "NativeSegB back: predicted members={} compare ok={} MISMATCH={} | handled today as: "
             "opener={} legacy_join={} ring_confirm={} generic={} other={}",
             s.predicted, s.ok, s.bad, s.opener, s.join, s.confirm, s.generic, s.other);
    s.predicted = s.ok = s.bad = s.opener = s.join = s.confirm = s.generic = s.other = 0;
}

u32 StageBlockBytes(u32 ud_dw, u32 flat_dw) {
    return static_cast<u32>((sizeof(CpSplit::StagePayload) + size_t(ud_dw + flat_dw) * 4 + 7) &
                            ~size_t{7});
}

/// Compares the member path's record bytes (after the SplitDrawRecord header) with the normal
/// path's: stage blocks (ud only below the stage's user-SGPR count, flat as NativeRecFront) and
/// the vertex-input block byte for byte.

// SHADPS4_NATIVE_REC coverage (notes/native-path-2026-10-09.md sec. 11): the user-data registers
// a stage really reads: its pushed user data (ud_mask), descriptors read straight from user-data
// registers (sharp_idx < 16), the SRT walker's root pointers and, for the VS, the fetch shader's
// SGPRs. The walker roots are read back from the walker's x86 code (its emitter writes, per
// pointer, `push rdi; mov rdi, [rdi + 4 * sgpr]`, roots at depth 1; the fault patch only rewrites
// nested loads); anything unexpected -> 0xffffffff (unknown: the RSRC2-count rule stays).
u32 NativeWalkerRoots(const Shader::PersistentSrtInfo& srt, bool& ok) {
    ok = true;
    u32 m = 0;
    if (!srt.walker_func || srt.walker_func_size == 0) {
        return 0;
    }
    auto* code = reinterpret_cast<u8*>(srt.walker_func);
    const size_t n = srt.walker_func_size;
    size_t pos = 0;
    int depth = 0;
    bool after_push = false;
    while (pos < n) {
        ZydisDecodedInstruction ins;
        ZydisDecodedOperand ops[ZYDIS_MAX_OPERAND_COUNT];
        if (!ZYAN_SUCCESS(
                Common::Decoder::Instance()->decodeInstruction(ins, ops, code + pos, n - pos))) {
            ok = false;
            return 0;
        }
        const bool reg0_rdi = ops[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
                              ops[0].reg.value == ZYDIS_REGISTER_RDI;
        if (ins.mnemonic == ZYDIS_MNEMONIC_PUSH && reg0_rdi) {
            ++depth;
            after_push = true;
            pos += ins.length;
            continue;
        }
        if (ins.mnemonic == ZYDIS_MNEMONIC_RET) {
            break;
        }
        if (ins.mnemonic == ZYDIS_MNEMONIC_POP && reg0_rdi) {
            --depth;
        } else if (after_push && depth == 1) {
            // The root load: mov rdi, qword ptr [rdi + disp].
            if (ins.mnemonic != ZYDIS_MNEMONIC_MOV || !reg0_rdi ||
                ops[1].type != ZYDIS_OPERAND_TYPE_MEMORY || ops[1].mem.base != ZYDIS_REGISTER_RDI ||
                ops[1].mem.index != ZYDIS_REGISTER_NONE) {
                ok = false;
                return 0;
            }
            const s64 disp = ops[1].mem.disp.value;
            if (disp < 0 || disp % 4 != 0 || disp / 4 >= 15) {
                ok = false;
                return 0;
            }
            m |= 3u << (disp / 4);
        }
        after_push = false;
        pos += ins.length;
    }
    return ok ? m : 0;
}

u32 NativeNeedMask(const Shader::Info* info, const Shader::Gcn::FetchShaderData* fetch) {
    if (!info) {
        return 0;
    }
    u32 m = info->ud_mask.mask;
    const auto add = [&m](u32 idx, u32 count) {
        for (u32 i = 0; i < count; ++i) {
            if (idx + i < 16) {
                m |= 1u << (idx + i);
            }
        }
    };
    for (const auto& d : info->buffers) {
        if (!d.inline_cbuf && d.sharp_idx < 16) {
            add(d.sharp_idx, 4);
        }
    }
    for (const auto& d : info->images) {
        if (d.sharp_idx < 16) {
            add(d.sharp_idx, 8);
        }
    }
    for (const auto& d : info->samplers) {
        if (!d.is_inline_sampler && d.sharp_idx < 16) {
            add(d.sharp_idx, 4);
        }
    }
    for (const auto& d : info->fmasks) {
        if (d.sharp_idx < 16) {
            add(d.sharp_idx, 8);
        }
    }
    // Strict (never inherited): the walker's roots (its flat output would be stale) and the
    // fetch shader's SGPRs (vertex inputs, offsets).
    u32 strict = 0;
    const auto add_strict = [&](u32 idx, u32 count) {
        for (u32 i = 0; i < count; ++i) {
            if (idx + i < 16) {
                strict |= 1u << (idx + i);
            }
        }
    };
    bool ok = true;
    strict |= NativeWalkerRoots(info->srt_info, ok);
    if (!ok) {
        return 0xffffffffu;
    }
    if (info->stage == Shader::Stage::Vertex) {
        if (info->has_fetch_shader) {
            add_strict(info->fetch_shader_sgpr_base, 2);
        }
        if (fetch) {
            if (fetch->vertex_offset_sgpr >= 0) {
                add_strict(static_cast<u32>(fetch->vertex_offset_sgpr), 1);
            }
            if (fetch->instance_offset_sgpr >= 0) {
                add_strict(static_cast<u32>(fetch->instance_offset_sgpr), 1);
            }
            for (const auto& attrib : fetch->attributes) {
                if (attrib.sgpr_base != Shader::IR::NumScalarRegs) {
                    add_strict(attrib.sgpr_base, 2); // a pointer to the V# list
                } else {
                    add_strict(attrib.dword_offset, 4);
                }
            }
        }
    }
    m |= strict;
    return (m & 0xffffu) | ((strict & 0xffffu) << 16);
}

/// Front thread: NativeNeedMask per Info, cached (Infos live as long as the pipeline cache).
u32 NativeNeedCached(const Shader::Info* info, const Shader::Gcn::FetchShaderData* fetch) {
    // Per thread: the front and the lean workers (NativeRecPublishProgramImpl) call it.
    static thread_local std::unordered_map<const void*, u32> cache;
    if (!info) {
        return 0;
    }
    const void* key = static_cast<const u8*>(static_cast<const void*>(info)) +
                      (fetch ? 1 : 0); // a VS with / without its fetch shader
    if (const auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }
    const u32 m = NativeNeedMask(info, fetch);
    cache.emplace(key, m);
    return m;
}

bool SameMemberRecord(const u8* normal, const u8* member, u32 stage_count, u32 vi_bytes,
                      const Regs& regs, const Vulkan::GraphicsPipeline* pipeline) {
    for (u32 s = 0; s < stage_count; ++s) {
        const auto* a = reinterpret_cast<const CpSplit::StagePayload*>(normal);
        const auto* b = reinterpret_cast<const CpSplit::StagePayload*>(member);
        if (a->info != b->info || a->pgm_base != b->pgm_base ||
            a->logical_stage != b->logical_stage || a->ud_dw != b->ud_dw ||
            a->flat_dw != b->flat_dw) {
            return false;
        }
        const bool vs = a->logical_stage == static_cast<u32>(Shader::LogicalStage::Vertex);
        const u32 n_user = std::min<u32>(
            vs ? regs.vs_program.settings.num_user_regs : regs.ps_program.settings.num_user_regs,
            16u);
        const auto* da = reinterpret_cast<const u32*>(a + 1);
        const auto* db = reinterpret_cast<const u32*>(b + 1);
        // Coverage (sec. 11): only the registers the stage reads (the job thread may not know
        // the others).
        u32 need = 0xffffffffu;
        if (AmdGpu::NativeRec::g_needs_on && pipeline) {
            need = NativeNeedCached(static_cast<const Shader::Info*>(a->info),
                                    vs ? pipeline->GetFetchShader() : nullptr);
        }
        for (u32 i = 0; i < a->ud_dw; ++i) {
            if (i < n_user && ((need >> i) & 1) && da[i] != db[i]) {
                return false;
            }
        }
        for (u32 i = 0; i < a->flat_dw; ++i) {
            if ((i >= 16 || (i < n_user && ((need >> i) & 1))) &&
                da[a->ud_dw + i] != db[a->ud_dw + i]) {
                return false;
            }
        }
        const u32 step = StageBlockBytes(a->ud_dw, a->flat_dw);
        normal += step;
        member += step;
    }
    return std::memcmp(normal, member, vi_bytes) == 0;
}
} // namespace

namespace {
/// SHADPS4_NATIVE_SEGX, back thread: Segment records by outcome. Reasons: 1 no usable variant,
/// 2 job ring lifetime, 3 index union, 4 vertex union, 5 vertex streams in several buffers,
/// 6 Guest union, 7 ops, 8 segment prepare failed, 9 the open was not reached, 10 other.
struct SegXBack {
    u64 segments{}, members{}, ok{}, refused{}, op_range{}, op_st{}, op_cull{}, cycles{};
    u64 replayed{}, replayed_members{}, singles{}, mats_extra{};
    std::array<u64, 4> mat_hist{}; ///< executed segments by materials 1 / 2 / 3-4 / 5-8
    std::array<u64, 12> why{};
    u32 logged{};
    u32 calls{};
    std::chrono::steady_clock::time_point window{};
} g_segx_back;
} // namespace

void Liverpool::ExecuteSegmentRecord(const CpSplit::RecordHeader& record) {
    namespace NR = AmdGpu::NativeRec;
    auto& s = g_segx_back;
    const auto* seg = reinterpret_cast<const CpSplit::SplitSegmentRecord*>(&record + 1);
    const u8* p = reinterpret_cast<const u8*>(seg + 1);
    const u8* recs = p; // the embedded member records, at seg->rec_off[j]
    const auto* drec = reinterpret_cast<const SplitDrawRecord*>(recs + seg->rec_off[0]);
    p += (seg->draw_bytes + 15u) & ~15u;
    const auto* ops = reinterpret_cast<const CpSplit::SegOp*>(p);
    p += size_t(seg->n_ops) * sizeof(CpSplit::SegOp);
    const u32* cmds = reinterpret_cast<const u32*>(p);
    p += size_t(seg->n) * 20;
    const u32* list = reinterpret_cast<const u32*>(p);
    // The embedded record's vertex-input block follows its stage blocks.
    const u8* body = reinterpret_cast<const u8*>(drec + 1);
    for (u32 i = 0; i < drec->stage_count; ++i) {
        const auto* sp = reinterpret_cast<const CpSplit::StagePayload*>(body);
        body += StageBlockBytes(sp->ud_dw, sp->flat_dw);
    }
    const auto* vi =
        drec->vi_bytes != 0 ? reinterpret_cast<const CpSplit::VertexInputsPayload*>(body) : nullptr;
    Vulkan::Rasterizer& tr = rasterizer->Translating();
    if (seg->n == 1) {
        // Brick C v3 item 1: a single-member segment is an ordinary draw of today's path (no
        // evaluation, no segment open); in =1 its withheld record runs here with ITS draw
        // registers and stencil / cull state (the mirror holds the next draw's).
        ++s.singles;
        if (NR::g_segx == 1 && seg->n_rec >= 1) {
            auto& mregs = mirror->regs;
            constexpr u32 ctx0 = Regs::ContextRegWordOffset;
            const auto save_ib = mregs.index_base_address;
            const u32 save_ni = mregs.num_indices;
            const auto save_inst = mregs.num_instances;
            const auto save_it = mregs.index_buffer_type;
            const u32 save_sf = mregs.reg_array[ctx0 + 0x10C];
            const u32 save_sb = mregs.reg_array[ctx0 + 0x10D];
            const u32 save_pc = mregs.reg_array[ctx0 + 0x205];
            for (u32 i = 0; i < seg->n_ops && ops[i].kind != CpSplit::SegOpDrawRange; ++i) {
                if (ops[i].kind == CpSplit::SegOpStencil) {
                    mregs.reg_array[ctx0 + 0x10C] = ops[i].a;
                    mregs.reg_array[ctx0 + 0x10D] = ops[i].b;
                } else if (ops[i].kind == CpSplit::SegOpCull) {
                    mregs.reg_array[ctx0 + 0x205] = ops[i].a;
                }
            }
            std::memcpy(&mregs.index_base_address, &seg->index_base0, sizeof(u64));
            mregs.num_indices = seg->num_indices0;
            mregs.num_instances.num_instances = seg->num_instances0;
            mregs.index_buffer_type.raw = seg->index_type;
            tr.segx_pending = nullptr;
            tr.segx_left = 0;
            tr.segx_n_mat = 0;
            ExecuteDrawRecord(record, drec);
            mregs.index_base_address = save_ib;
            mregs.num_indices = save_ni;
            mregs.num_instances = save_inst;
            mregs.index_buffer_type = save_it;
            mregs.reg_array[ctx0 + 0x10C] = save_sf;
            mregs.reg_array[ctx0 + 0x10D] = save_sb;
            mregs.reg_array[ctx0 + 0x205] = save_pc;
        }
        return;
    }
    const u64 t0 = __rdtsc();
    u32 why = 0;
    bool ok = tr.NativeSegEvaluate(*seg, drec->draw.pipeline, vi, list, why);
    // v3 item 4: each further distinct material's representative member - its stage blocks
    // (raw T# / S# words) from its embedded record, for the open's one resolve per material.
    tr.segx_n_mat = 0;
    if (ok && NR::g_segx == 1) {
        if (seg->n_mat == 0 || seg->n_mat > Shader::NativeSegMaxMaterials ||
            (seg->n_mat > 1 && seg->n_rec != seg->n)) {
            ok = false;
            why = 10;
        } else {
            tr.segx_n_mat = seg->n_mat;
            for (u32 m = 1; m < seg->n_mat && ok; ++m) {
                const u32 rep = seg->mat_rep[m];
                if (rep >= seg->n) {
                    ok = false;
                    why = 10;
                    break;
                }
                const auto* mrec =
                    reinterpret_cast<const SplitDrawRecord*>(recs + seg->rec_off[rep]);
                auto& st = tr.segx_mat_stages[m];
                st = {};
                ReadStages(reinterpret_cast<const u8*>(mrec + 1), mrec->stage_count, st);
                if (!st[static_cast<u32>(Shader::LogicalStage::Fragment)].info) {
                    ok = false;
                    why = 10;
                }
            }
        }
    }
    if (NR::g_segx == 1) {
        // The members' Draw records were withheld: member 0 runs through its embedded record
        // with ITS draw registers in the mirror (the journal already holds the next draw's) and
        // the stencil / cull state it ran with (the leading state ops; the mirror holds the last
        // member's). A refusal replays every other member through today's path, in op order,
        // each with its own draw registers (brick C v2: no holes).
        auto& mregs = mirror->regs;
        constexpr u32 ctx0 = Regs::ContextRegWordOffset;
        const auto save_ib = mregs.index_base_address;
        const u32 save_ni = mregs.num_indices;
        const auto save_inst = mregs.num_instances;
        const auto save_it = mregs.index_buffer_type;
        const u32 save_sf = mregs.reg_array[ctx0 + 0x10C];
        const u32 save_sb = mregs.reg_array[ctx0 + 0x10D];
        const u32 save_pc = mregs.reg_array[ctx0 + 0x205];
        const auto apply_state = [&](const CpSplit::SegOp& op) {
            if (op.kind == CpSplit::SegOpStencil) {
                mregs.reg_array[ctx0 + 0x10C] = op.a;
                mregs.reg_array[ctx0 + 0x10D] = op.b;
            } else if (op.kind == CpSplit::SegOpCull) {
                mregs.reg_array[ctx0 + 0x205] = op.a;
            }
        };
        for (u32 i = 0; i < seg->n_ops && ops[i].kind != CpSplit::SegOpDrawRange; ++i) {
            apply_state(ops[i]);
        }
        std::memcpy(&mregs.index_base_address, &seg->index_base0, sizeof(u64));
        mregs.num_indices = seg->num_indices0;
        mregs.num_instances.num_instances = seg->num_instances0;
        mregs.index_buffer_type.raw = seg->index_type;
        tr.segx_pending = ok ? seg : nullptr;
        tr.segx_ops = ops;
        tr.segx_cmds = cmds;
        tr.segx_list = list;
        tr.segx_result = 0;
        tr.segx_left = seg->n - 1;
        ExecuteDrawRecord(record, drec);
        tr.segx_pending = nullptr;
        tr.segx_left = 0;
        tr.segx_n_mat = 0;
        if (ok) {
            ok = tr.segx_result == 1;
            why = ok ? 0 : tr.segx_result == 0 ? 9 : tr.segx_result;
        }
        if (!ok && seg->n_rec == seg->n && seg->n > 1) {
            ++s.replayed;
            for (u32 i = 0; i < seg->n_ops; ++i) {
                const auto& op = ops[i];
                if (op.kind != CpSplit::SegOpDrawRange) {
                    apply_state(op);
                    continue;
                }
                for (u32 j = op.a; j < op.a + op.b && j < seg->n; ++j) {
                    if (j == 0) {
                        continue; // drawn above
                    }
                    const u32* c = cmds + size_t(j) * 5;
                    const u64 addr = seg->ilo + u64(c[2]) * seg->index_size;
                    mregs.index_base_address.base_addr_lo = static_cast<u32>(addr);
                    mregs.index_base_address.base_addr_hi = static_cast<u32>(addr >> 32);
                    mregs.num_indices = c[0];
                    mregs.num_instances.num_instances = c[1];
                    tr.segx_left = seg->n - 1 - j;
                    ExecuteDrawRecord(record, recs + seg->rec_off[j]);
                    tr.segx_left = 0;
                    ++s.replayed_members;
                }
            }
        }
        mregs.index_base_address = save_ib;
        mregs.num_indices = save_ni;
        mregs.num_instances = save_inst;
        mregs.index_buffer_type = save_it;
        mregs.reg_array[ctx0 + 0x10C] = save_sf;
        mregs.reg_array[ctx0 + 0x10D] = save_sb;
        mregs.reg_array[ctx0 + 0x205] = save_pc;
    }
    s.cycles += __rdtsc() - t0;
    ++s.segments;
    s.members += seg->n;
    for (u32 i = 0; i < seg->n_ops; ++i) {
        (ops[i].kind == CpSplit::SegOpDrawRange ? s.op_range
         : ops[i].kind == CpSplit::SegOpStencil  ? s.op_st
                                                  : s.op_cull) += 1;
    }
    if (ok) {
        ++s.ok;
        s.mats_extra += seg->n_mat > 0 ? seg->n_mat - 1 : 0;
        ++s.mat_hist[seg->n_mat <= 1 ? 0 : seg->n_mat == 2 ? 1 : seg->n_mat <= 4 ? 2 : 3];
    } else {
        ++s.refused;
        ++s.why[std::min<u32>(why, 11)];
        if (NR::g_segx == 1 && ++s.logged <= 20) {
            LOG_WARNING(Render, "NativeSegX refused segment {} ({} members, {} ops): reason {}",
                        seg->id, seg->n, seg->n_ops, why);
        }
    }
    if ((++s.calls & 255) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (s.window == std::chrono::steady_clock::time_point{}) {
        s.window = now;
        return;
    }
    if (now - s.window < std::chrono::seconds(10)) {
        return;
    }
    s.window = now;
    LOG_INFO(Render,
             "NativeSegX: mode {} segments={} members={} evaluated ok={} refused={} (variant {} "
             "lifetime {} index_union {} vertex_union {} vertex_bufs {} guest_union {} ops {} "
             "prepare {} open {} materials {} other {}) ops: draw ranges={} stencil={} cull={} | "
             "cycles per segment open {:.0f} | replayed segments={} members={} | singles "
             "returned={} | materials per segment 1/2/3-4/5-8: {} {} {} {} (extra materials "
             "resolved={})",
             NR::g_segx, s.segments, s.members, s.ok, s.refused, s.why[1], s.why[2], s.why[3],
             s.why[4], s.why[5], s.why[6], s.why[7], s.why[8], s.why[9], s.why[10], s.why[11],
             s.op_range, s.op_st, s.op_cull,
             s.segments ? double(s.cycles) / double(s.segments) : 0.0, s.replayed,
             s.replayed_members, s.singles, s.mat_hist[0], s.mat_hist[1], s.mat_hist[2],
             s.mat_hist[3], s.mats_extra);
    s.segments = s.members = s.ok = s.refused = s.op_range = s.op_st = s.op_cull = s.cycles = 0;
    s.replayed = s.replayed_members = s.singles = s.mats_extra = 0;
    s.mat_hist.fill(0);
    s.why.fill(0);
}

bool Liverpool::NativeRecMember(const PM4Header* header, const char* name, bool host_marker,
                                const CpSplit::DrawArgs& args) {
    namespace NR = AmdGpu::NativeRec;
    auto& run = g_nr_run;
    auto& st = g_nr_mst;
    g_nr_scratch_valid = false;
    const auto refuse = [&](u32 why) {
        ++st.why[std::min<u32>(why, 7)];
        ++st.refused;
        run.active = false;
        return false;
    };
    if (!run.active) {
        return false;
    }
    const auto* w = reinterpret_cast<const u32*>(header);
    if (header->type3.opcode != PM4ItOpcode::DrawIndex2 || w[6] != 0xc0021000u ||
        w[7] != NR::Magic || split_verify || split_front_derive == 0) {
        return refuse(0);
    }
    const u32 tag = w[8];
    const u32 idx = w[9];
    const NR::Record* ring = NR::g_rings[tag >> 28].load(std::memory_order_acquire);
    const u8* payload = NR::g_payloads[tag >> 28].load(std::memory_order_acquire);
    if (!ring || !payload) {
        return refuse(1);
    }
    const NR::Record& src = ring[idx & (NR::RingSize - 1)];
    if (src.stamp.load(std::memory_order_acquire) != idx + 1) {
        return refuse(1);
    }
    // Opus review 1: copy every field first, then fence + re-check the stamp, then bounds, and
    // only then dereference or copy through them.
    const bool cmd_ok = src.cmd == w;
    const u32 run_k = src.run_k, head_idx = src.head_idx, src_tag = src.tag;
    const bool derived = src.derived != 0;
    const void* src_pipeline = src.pipeline;
    const u32 nvs = src.flat_vs_dw, nps = src.flat_ps_dw, vi_bytes = src.vi_bytes;
    const u64 pos = src.payload_pos;
    const auto ud_vs = src.ud_vs;
    const auto ud_ps = src.ud_ps;
    const VAddr vs_pgm = VAddr{src.vs_pgm} << 8, ps_pgm = VAddr{src.ps_pgm} << 8;
    const u32 inh = src.inh_vs | src.inh_ps;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (src.stamp.load(std::memory_order_relaxed) != idx + 1 || !cmd_ok) {
        return refuse(1);
    }
    if (inh != 0) {
        // Coverage (sec. 11): the job's user data holds stale inherited registers; the normal
        // path (the front's own registers) builds this record.
        return refuse(1);
    }
    if (run_k != run.next_k || head_idx != run.head_idx || src_tag != run.tag || !derived) {
        return refuse(2);
    }
    // Opus review 4: the context epoch does not cover the uconfig primitive type, the index
    // type or the stages' user-SGPR counts.
    if (context_change_epoch != run.ctx_epoch ||
        static_cast<u32>(regs.primitive_type) != run.prim ||
        regs.index_buffer_type.raw != run.index_type ||
        regs.vs_program.settings.num_user_regs != run.vs_nuser ||
        regs.ps_program.settings.num_user_regs != run.ps_nuser) {
        return refuse(3);
    }
    const auto* gp = static_cast<const Vulkan::GraphicsPipeline*>(src_pipeline);
    if (!gp || nvs > 1024 || nps > 1024 || vi_bytes < sizeof(CpSplit::VertexInputsPayload) ||
        vi_bytes > sizeof(CpSplit::VertexInputsDerived)) {
        return refuse(1);
    }
    const auto gstages = gp->GetStages();
    if (gstages[static_cast<u32>(Shader::LogicalStage::Vertex)] != run.vsi ||
        gstages[static_cast<u32>(Shader::LogicalStage::Fragment)] != run.psi) {
        return refuse(4);
    }
    const u64 t0 = __rdtsc();
    const u8* pl = payload + (pos & (NR::PayloadSize - 1));
    const u32 vs_block = StageBlockBytes(16, nvs);
    const u32 ps_block = run.psi ? StageBlockBytes(16, nps) : 0u;
    const u32 body = vs_block + ps_block + vi_bytes;
    thread_local std::vector<u8> local;
    local.resize(body);
    u8* out = local.data();
    for (u32 i = 0; i < gstages.size(); ++i) {
        const bool is_vs = i == static_cast<u32>(Shader::LogicalStage::Vertex);
        const bool is_ps = i == static_cast<u32>(Shader::LogicalStage::Fragment);
        if (!gstages[i] || (!is_vs && !is_ps)) {
            continue;
        }
        auto* sp = reinterpret_cast<CpSplit::StagePayload*>(out);
        sp->info = gstages[i];
        sp->pgm_base = is_vs ? vs_pgm : ps_pgm;
        sp->logical_stage = i;
        sp->ud_dw = 16;
        sp->flat_dw = is_vs ? nvs : nps;
        sp->pad = 0;
        auto* dw = reinterpret_cast<u32*>(sp + 1);
        std::memcpy(dw, is_vs ? ud_vs.data() : ud_ps.data(), 16 * 4);
        std::memcpy(dw + 16, is_vs ? pl : pl + size_t(nvs) * 4, size_t(sp->flat_dw) * 4);
        out += StageBlockBytes(16, sp->flat_dw);
    }
    const u8* vi_src = pl + (u64(nvs + nps) * 4 + 7) / 8 * 8;
    std::memcpy(out, vi_src, vi_bytes);
    // Opus review 2: the payload is intact only if the job thread has not claimed past it.
    std::atomic_thread_fence(std::memory_order_acquire);
    if (NR::g_payload_claim[tag >> 28].load(std::memory_order_relaxed) - pos > NR::PayloadSize ||
        src.stamp.load(std::memory_order_relaxed) != idx + 1) {
        return refuse(7);
    }
    {
        auto* hdr = reinterpret_cast<CpSplit::VertexInputsPayload*>(out);
        auto* att = reinterpret_cast<CpSplit::VertexInputAttrib*>(hdr + 1);
        if (hdr->count != run.desc_count ||
            CpSplit::VertexInputsBytes(hdr->count, hdr->range_count) != vi_bytes) {
            return refuse(6);
        }
        hdr->desc_id = run.desc_id;
        for (u32 a = 0; a < hdr->count; ++a) {
            att[a].desc = run.descs[a]; // the head's (divisors come from context registers)
        }
    }
    // Last (Opus review 9): the hazard note and empty-draw counter run once per taken member.
    Vulkan::Rasterizer::SplitDraw draw{.args = args};
    if (!rasterizer->PrepareMemberFront(draw, run.pipeline) || draw.skip) {
        return refuse(5);
    }
    const u32 stage_count = run.psi ? 2u : 1u;
    if (NR::g_mode == 3) {
        // Verify: keep the bytes for EmitDraw's compare; the normal path renders.
        g_nr_scratch.assign(local.begin(), local.end());
        g_nr_scratch_valid = true;
        ++run.next_k;
        return false;
    }
    SegB::Pred segb{};
    bool segx_hold = false;
    const u32 rec_bytes = static_cast<u32>(sizeof(SplitDrawRecord) + body);
    if (NR::g_segb != 0) [[unlikely]] {
        // SHADPS4_NATIVE_SEGB: the vertex block in `local` (the head's descriptions patched in).
        if (!SegB::g_state.init) {
            SegB::Init(SegB::g_state, split.get(), split_run_ahead, split_verify);
        }
        const auto* vih = reinterpret_cast<const CpSplit::VertexInputsPayload*>(out);
        const bool member = SegB::NativeSegBDraw(header, run.pipeline, true,
                                                 CpSplit::VertexInputAttribs(*vih), vih->count,
                                                 regs, rec_bytes, segb);
        segx_hold = member && SegB::g_state.exec;
    }
    // SHADPS4_NATIVE_SEGX=1: a segment member's record is built but not posted (it rides in the
    // Segment record at the close).
    auto* rec = segx_hold ? SegB::ScratchRecord(rec_bytes)
                          : split->Reserve<SplitDrawRecord>(CpSplit::RecordType::Draw, rec_bytes);
    rec->draw = draw;
    rec->ctx_epoch = context_change_epoch;
    rec->marker = host_marker ? name : nullptr;
    rec->cmd_address = header;
    rec->stage_count = stage_count;
    rec->verify = 0;
    rec->vi_bytes = vi_bytes;
    rec->pad = 0;
    rec->nr_tag = 0;
    rec->nr_idx = 0;
    rec->job_tag = tag;
    rec->job_idx = idx;
    rec->job_member = 1;
    rec->job_pad = 0;
    rec->segb_id = segb.id;
    rec->segb_member = segb.member;
    rec->segb_index_addr = segb.index_addr;
    rec->segb_index_count = segb.index_count;
    rec->segb_instances = segb.instances;
    std::memcpy(reinterpret_cast<u8*>(rec + 1), local.data(), body);
    if (segb.id != 0 && (segb.member == 0 || SegB::g_state.exec) && SegB::g_state.segx) [[unlikely]] {
        SegB::KeepMember(rec, rec_bytes);
    }
    if (!segx_hold) {
        split->Commit();
        if (split_run_ahead) {
            split->PostRunAhead(CpSplit::PostKind::Draw);
        } else {
            split->Sync(CpSplit::DrainReason::DepthZero);
        }
    }
    split->MaybeLog();
    ++run.next_k;
    ++st.members;
    st.c_member += __rdtsc() - t0;
    return true;
}

void Liverpool::NativeRecHead(const PM4Header* header, const void* pipeline_ptr, bool plain_draw,
                              u32 vi_bytes, const u8* record_body, u32 stage_count) {
    namespace NR = AmdGpu::NativeRec;
    const auto* draw_pipeline = static_cast<const Vulkan::GraphicsPipeline*>(pipeline_ptr);
    auto& run = g_nr_run;
    auto& st = g_nr_mst;
    // Mode 3: compare the member path's bytes with this (normal) record's.
    if (g_nr_scratch_valid) {
        g_nr_scratch_valid = false;
        const bool same = g_nr_scratch.size() >= vi_bytes &&
                          SameMemberRecord(record_body, g_nr_scratch.data(), stage_count,
                                           vi_bytes, regs, draw_pipeline) &&
                          draw_pipeline == run.pipeline;
        (same ? st.cmp_ok : st.cmp_bad) += 1;
        if (!same && ++st.logged <= 16) {
            LOG_WARNING(Render, "NativeRec member MISMATCH (k {}, pipeline {})", run.next_k - 1,
                        draw_pipeline == run.pipeline ? "same" : "differs");
        }
        return; // a verified member is not a head
    }
    run.active = false;
    const auto* w = reinterpret_cast<const u32*>(header);
    if (header->type3.opcode != PM4ItOpcode::DrawIndex2 || w[6] != 0xc0021000u ||
        w[7] != NR::Magic || !draw_pipeline || !plain_draw || vi_bytes == 0) {
        return;
    }
    const u32 tag = w[8];
    const u32 idx = w[9];
    const NR::Record* ring = NR::g_rings[tag >> 28].load(std::memory_order_acquire);
    if (!ring) {
        return;
    }
    const NR::Record& src = ring[idx & (NR::RingSize - 1)];
    if (src.stamp.load(std::memory_order_acquire) != idx + 1 || src.cmd != w ||
        src.run_k != 0 || src.count.load(std::memory_order_acquire) < 2 || !src.derived) {
        return;
    }
    const auto stages = draw_pipeline->GetStages();
    for (u32 i = 0; i < stages.size(); ++i) {
        if (stages[i] && i != static_cast<u32>(Shader::LogicalStage::Vertex) &&
            i != static_cast<u32>(Shader::LogicalStage::Fragment)) {
            return;
        }
    }
    const Shader::Info* vsi = stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (!vsi || vsi->stage != Shader::Stage::Vertex) {
        return;
    }
    run.active = true;
    run.tag = tag;
    run.head_idx = idx;
    run.next_k = 1;
    run.ctx_epoch = context_change_epoch;
    run.pipeline = draw_pipeline;
    run.vsi = vsi;
    run.psi = stages[static_cast<u32>(Shader::LogicalStage::Fragment)];
    run.prim = static_cast<u32>(regs.primitive_type);
    run.index_type = regs.index_buffer_type.raw;
    run.vs_nuser = regs.vs_program.settings.num_user_regs;
    run.ps_nuser = regs.ps_program.settings.num_user_regs;
    run.desc_id = front_vi->header.desc_id;
    run.desc_count = front_vi->header.count;
    for (u32 a = 0; a < run.desc_count; ++a) {
        run.descs[a] = front_vi->attribs[a].desc;
    }
}

void Liverpool::LeanFrontBegin(std::span<const u32> dcb, std::span<const u32> ccb,
                               bool guest_markers) {
    auto& s = g_lean_s;
    auto& c = s.c;
    if (!s.started) {
        LeanStart();
    }
    ++c.dcbs;
    if (s.win_tsc == 0) {
        s.win_tsc = __rdtsc();
    }
    const u64 t0 = __rdtsc();
    auto& gfx_cs = mapped_queues[GfxQueueId].cs_state;
    // The chain must equal the front's live state here (the front parsed everything before).
    const auto seed = [&] {
        *s.regs = regs;
        s.cb = last_cb_extent;
        s.db = last_db_extent;
        s.epoch = context_change_epoch;
        s.cs = gfx_cs;
        s.iaa = indirect_args_addr;
        s.seeded = true;
    };
    if (!s.seeded) {
        seed();
    } else {
        ++s.chain_cmp;
        bool bad_win = false;
        for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
            bad_win = bad_win ||
                      std::memcmp(&s.regs->reg_array[off], &regs.reg_array[off], size_t{n} * 4) != 0;
        }
        bool bad_full = false;
        if ((s.chain_cmp & 15) == 0) {
            bad_full = std::memcmp(s.regs->reg_array.data(), regs.reg_array.data(),
                                   sizeof(regs.reg_array)) != 0;
        }
        const bool bad_db = s.db.raw != last_db_extent.raw;
        const bool bad_epoch = s.epoch != context_change_epoch;
        const bool bad_iaa = s.iaa != indirect_args_addr;
        s32 cs_dw = -1; // first differing cs_state dword
        for (u32 i = 0; i < sizeof(ComputeProgram) / 4 && cs_dw < 0; ++i) {
            if (reinterpret_cast<const u32*>(&s.cs)[i] != reinterpret_cast<const u32*>(&gfx_cs)[i]) {
                cs_dw = static_cast<s32>(i);
            }
        }
        u32 cb_mask = 0;
        for (u32 i = 0; i < NUM_COLOR_BUFFERS; ++i) {
            cb_mask |= s.cb[i].raw != last_cb_extent[i].raw ? 1u << i : 0u;
        }
        const bool bad_extra = bad_db || bad_epoch || bad_iaa || cs_dw >= 0 || cb_mask != 0;
        s.chain_bad_db += bad_db;
        s.chain_bad_cb += cb_mask != 0;
        s.chain_bad_cs += cs_dw >= 0;
        s.chain_bad_epoch += bad_epoch;
        s.chain_bad_iaa += bad_iaa;
        s.chain_bad_win += bad_win;
        s.chain_bad_full += bad_full;
        s.chain_bad_extra += bad_extra;
        if (bad_win || bad_full || bad_extra) {
            if (s.chain_bad_win + s.chain_bad_full + s.chain_bad_extra <= 8) {
                LOG_WARNING(Render, "Lean chain: state differs from the front at DCB {} (windows {} "
                                    "whole file {} extras {}: db {} cb mask {:#x} cs first dword {} "
                                    "(chain {:#x} front {:#x}), epoch chain {} front {}, iaa {:#x} / {:#x})",
                            s.index + 1, bad_win, bad_full, bad_extra, bad_db, cb_mask, cs_dw,
                            cs_dw >= 0 ? reinterpret_cast<const u32*>(&s.cs)[cs_dw] : 0u,
                            cs_dw >= 0 ? reinterpret_cast<const u32*>(&gfx_cs)[cs_dw] : 0u, s.epoch,
                            context_change_epoch, s.iaa, indirect_args_addr);
            }
            ++s.reseeds;
            seed();
        }
    }
    const u32 idx = ++s.index;
    auto& j = s.jobs[idx % LeanSlots];
    const bool eligible = ccb.empty();
    bool queued = false;
    if (eligible && j.state.load(std::memory_order_acquire) == 0 &&
        j.done_mask.load(std::memory_order_acquire) == 0) {
        // Entry state E(n) for the worker, then the scan carries the chain to E(n+1).
        j.index.store(idx, std::memory_order_relaxed);
        j.dcb = dcb;
        j.win.resize(CpSplit::DcbSnapWords);
        u32* out = j.win.data();
        for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
            std::memcpy(out, &s.regs->reg_array[off], size_t{n} * 4);
            out += n;
        }
        j.cb = s.cb;
        j.db = s.db;
        j.epoch = s.epoch;
        j.cs = s.cs;
        j.iaa = s.iaa;
        j.front_canon.clear();
        j.front_marks.clear();
        j.worker_canon.clear();
        j.dcb_hash = XXH3_64bits(dcb.data(), dcb.size_bytes());
        j.lag.store(false, std::memory_order_relaxed);
        queued = true;
    }
    u32 cut = 0, pkts = 0, draws = 0;
    LeanChainScan(dcb, LeanScanRefs{*s.regs, s.cb, s.db, s.epoch, s.cs, s.iaa}, guest_markers,
                  split_post_dma, cut, pkts, draws);
    s.chain_cyc += __rdtsc() - t0;
    c.pkts += pkts;
    c.draws += draws;
    if (queued) {
        ++c.eligible;
        j.cut_dw = cut;
        j.dcb_pkts = pkts;
        j.dcb_draws = draws;
        g_lean_front = {&j, dcb.data(), true};
        j.state.store(1, std::memory_order_release);
        s.queued.store(idx, std::memory_order_release);
        LeanRing(s, idx);
    } else {
        c.skipped += eligible ? 1 : 0;
        // Not queued: the next index still advances so the workers' claims stay in step.
        s.queued.store(idx, std::memory_order_release);
        LeanRing(s, idx);
    }
    if (c.dcbs >= 65536) {
        LeanLog();
    }
}

Liverpool::LeanL2Begin Liverpool::LeanFrontBeginL2(std::span<const u32> dcb,
                                                   std::span<const u32> ccb, bool guest_markers) {
    auto& s = g_lean_s;
    auto& c = s.c;
    if (!s.started) {
        LeanStart();
    }
    ++c.dcbs;
    if (s.win_tsc == 0) {
        s.win_tsc = __rdtsc();
    }
    LeanL2Begin res{};
    const u32 idx = ++s.index;
    auto& j = s.jobs[idx % LeanSlots];
    const auto not_taken = [&] {
        ++s.l2_skipped;
        s.queued.store(idx, std::memory_order_release); // the workers' claims stay in step
        LeanRing(s, idx);
    };
    if (!s.l2 || !ccb.empty() || j.state.load(std::memory_order_acquire) != 0) {
        not_taken();
        if (c.dcbs >= 65536) {
            LeanLog();
        }
        return res;
    }
    const u64 t0 = __rdtsc();
    auto& gfx_cs = mapped_queues[GfxQueueId].cs_state;
    // Heads longer than LeanSplitDraws draws are cut (head splitting) at the first render-target
    // change after each LeanSplitDraws draws - where the worker's stretch ends anyway: each span
    // is one job (own slot, channel, DcbRef), its entry state the front's live state at its start.
    // heads above it split, at a render-target change (r3: at any draw, SHADPS4_LEAN_SPLIT_ANY=N)
    const u32 LeanSplitDraws = LeanSplitAny() != 0 ? LeanSplitAny() : 256u;
    u32 start = 0, cur = idx, total_bumps = 0, spans = 0;
    for (;;) {
        auto& jj = s.jobs[cur % LeanSlots];
        const auto span = dcb.subspan(start);
        jj.index.store(cur, std::memory_order_relaxed);
        jj.dcb = span;
        jj.win.resize(CpSplit::DcbSnapWords);
        u32* out = jj.win.data();
        for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
            std::memcpy(out, &regs.reg_array[off], size_t{n} * 4);
            out += n;
        }
        jj.cb = last_cb_extent;
        jj.db = last_db_extent;
        jj.epoch = context_change_epoch;
        jj.cs = gfx_cs;
        jj.iaa = indirect_args_addr;
        // The head pass ON the live state, to the StaticCheck cut or the split point.
        u32 cut = 0, pkts = 0, draws = 0, bumps = 0;
        bool split_here = false;
        LeanChainScan(span,
                      LeanScanRefs{regs, last_cb_extent, last_db_extent, context_change_epoch,
                                   gfx_cs, indirect_args_addr},
                      guest_markers, split_post_dma, cut, pkts, draws, true, &bumps,
                      LeanSplitDraws, &split_here);
        if (cut == 0) {
            // Nothing for a worker from here (the first packet is the front's).
            ++s.l2_skipped;
            s.queued.store(cur, std::memory_order_release); // the workers' claims stay in step
            LeanRing(s, cur);
            break;
        }
        c.pkts_span += pkts;
        s.head_draws += draws;
        ++s.l2_taken;
        ++spans;
        jj.cut_dw = cut;
        jj.state.store(1, std::memory_order_release);
        s.queued.store(cur, std::memory_order_release);
        LeanRing(s, cur);
        // The back runs the span from the job's channel when it reaches this record.
        auto* ref = split->Reserve<CpSplit::DcbRefPayload>(CpSplit::RecordType::DcbRef);
        ref->slot = cur % LeanSlots;
        ref->index = cur;
        split->Commit();
        split->PublishNow();
        total_bumps += bumps;
        start += cut;
        if (!split_here || start >= dcb.size()) {
            break;
        }
        const u32 next = ++s.index;
        if (s.jobs[next % LeanSlots].state.load(std::memory_order_acquire) != 0) {
            // Slot busy: the rest of the head is the front's.
            ++s.l2_skipped;
            s.queued.store(next, std::memory_order_release);
            LeanRing(s, next);
            break;
        }
        cur = next;
    }
    s.chain_cyc += __rdtsc() - t0;
    s.split_spans += spans > 1 ? spans - 1 : 0;
    res.cut = start;
    res.run_bumps = total_bumps;
    if (c.dcbs >= 65536) {
        LeanLog();
    }
    return res;
}

void Liverpool::LeanBackFollow(const CpSplit::DcbRefPayload& ref) {
    using RT = CpSplit::RecordType;
    static const bool blc = Vulkan::BackCyclesEnabled();
    const u64 follow_t0 = blc ? __rdtsc() : 0;
    auto& s = g_lean_s;
    auto& ch = *s.ch[ref.slot];
    auto& job = s.jobs[ref.slot];
    auto& x = g_lean_xl;
    s.back_cur.store(ref.index, std::memory_order_release);
    split->FlushDone(); // the front's depth waits must see everything before this record
    lean_following = true;
    // L3: replays the worker's stretch at its first draw (A2Execute's publish, mode 1) and returns
    // true; false = the old path runs the stretch's records (rejected, or A2=2 verify).
    bool v_pass = false; // A2=2: the last stretch passed every check A2=1 makes before publishing
    const auto publish = [&](Vulkan::A2::Stretch& st, const A2Entry::Side& sd,
                             const CpSplit::RecordHeader& first) {
        v_pass = false;
        const bool verify = g_a2 == 2;
        if (verify) {
            x.would.fetch_add(st.draws, std::memory_order_relaxed);
        }
        auto& g = g_a2s;
        const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(&first + 1);
        if (rasterizer->A2OcclusionOpen()) {
            x.rej_occl.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!verify) {
            // Formatted slots: the back's own typed lookup (image -> buffer sync) + barrier per
            // range, before the generation check (a buffer it creates rejects the stretch).
            for (const auto& [ta, ts] : st.texel) {
                x.texel_syncs.fetch_add(1, std::memory_order_relaxed);
                if (!rasterizer->A2SyncTexel(ta, ts)) {
                    x.rej_texel.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
            }
        }
        // A buffer created / deleted / joined since the worker's lookups: its VkBuffer handles may
        // be stale (A2LookupBuffer reads slots without a lock).
        if (rasterizer->A2BufferCache().StructGeneration() != st.buf_gen) {
            x.rej_gen.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!verify) {
            rasterizer->GetScheduler().SwitchStream(0);
        }
        std::array<std::pair<u32, u32>, Vulkan::A2::MaxMaterials> mat_n{};
        const bool bl = Vulkan::BackCyclesEnabled();
        u64 tq = bl ? __rdtsc() : 0;
        const auto lap = [&](u64& acc) {
            if (bl) [[unlikely]] {
                const u64 n = __rdtsc();
                acc += n - tq;
                tq = n;
            }
        };
        static thread_local std::vector<VideoCore::ImageId> mat_ids;
        mat_ids.clear();
        const bool colour = !st.materials.empty();
        bool tab_used = false; // q8b (B): the table supplied materials of this stretch
        if (colour) {
            // Every distinct material once, before the pass (sec. 10); the representative Draw
            // records are in THIS job channel (not yet released: inside the stretch).
            const u32 set0 = rasterizer->A2BindlessCur();
            bool ok = st.materials.size() <= Vulkan::A2::MaxMaterials && rasterizer->A2ColourGates();
            {
                // Samplers resolve with the mirror's border-colour base at this draw: the stretch
                // must not change it.
                const u32 bc0 = static_cast<u32>(
                    reinterpret_cast<const u32*>(&mirror->regs.ta_bc_base) -
                    mirror->regs.reg_array.data());
                const u32 bc1 = bc0 + static_cast<u32>((sizeof(mirror->regs.ta_bc_base) + 3) / 4);
                for (const auto& [off, val] : sd.delta) {
                    ok = ok && (off < bc0 || off >= bc1);
                }
            }
            // q8: materials the worker resolved from the texture table hold while its generation
            // and the bindless set are unchanged - no per-material work here (=2: verified).
            namespace TT = VideoCore::TexTable;
            const int tab_mode = TT::Mode();
            const bool tab_ok = st.mat_pre != 0 && st.tab_gen == TT::Gen() &&
                                rasterizer->A2BindlessCur() == st.tab_set &&
                                rasterizer->A2TexTableStateOk() && // q8b (A)
                                TT::ImgsCurrent(st.tab_kseq, st.tab_imgs); // q10
            tab_used = tab_ok;
            if (st.mat_pre != 0 && !tab_ok) {
                TT::g_c[3].fetch_add(1, std::memory_order_relaxed);
            }
            for (size_t m = 0; ok && m < st.materials.size(); ++m) {
                const auto& mr = st.materials[m];
                std::array<u32, Shader::NativeBindlessSlots> pre_slots{};
                const bool pre = tab_ok && ((st.mat_pre >> m) & 1) != 0;
                if (pre) {
                    TT::g_c[2].fetch_add(1, std::memory_order_relaxed);
                    mat_n[m] = st.pre_n[m];
                    if (tab_mode != 2) {
                        continue;
                    }
                    pre_slots = st.cmds->mat_slots[m];
                }
                const auto tab_verify = [&] {
                    if (!pre) {
                        return;
                    }
                    if (TT::Gen() != st.tab_gen || !TT::ImgsCurrent(st.tab_kseq, st.tab_imgs)) {
                        TT::g_c[8].fetch_add(1, std::memory_order_relaxed); // q8b (E): stale
                        return;
                    }
                    TT::g_c[5].fetch_add(1, std::memory_order_relaxed);
                    bool bad = mat_n[m] != st.pre_n[m];
                    for (u32 i = 0; !bad && i < mat_n[m].first; ++i) {
                        bad = pre_slots[i] != st.cmds->mat_slots[m][i];
                    }
                    for (u32 j = 0; !bad && j < mat_n[m].second; ++j) {
                        bad = pre_slots[Shader::g_bl_img_max + j] !=
                              st.cmds->mat_slots[m][Shader::g_bl_img_max + j];
                    }
                    if (bad && TT::g_c[6].fetch_add(1, std::memory_order_relaxed) < 8) {
                        LOG_WARNING(Render, "q8 texture table VERIFY BAD: material {} ({} / {} "
                                            "images, {} / {} samplers)",
                                    m, st.pre_n[m].first, mat_n[m].first, st.pre_n[m].second,
                                    mat_n[m].second);
                    }
                };
                if (rasterizer->A2MaterialMemo(mr.base, mr.h1, mr.h2, mr.kn,
                                               st.cmds->mat_slots[m].data(), mat_n[m].first,
                                               mat_n[m].second, mat_ids)) {
                    x.mat_resolves.fetch_add(1, std::memory_order_relaxed);
                    continue; // l4d: no ReadStages, no per-image hashing
                }
                u64 mp = mr.rec_pos;
                const auto* mh = ch.RecordAtPos(mp);
                ok = mh->type == RT::Draw;
                if (!ok) {
                    break;
                }
                const auto& mrec = *reinterpret_cast<const SplitDrawRecord*>(mh + 1);
                std::array<Shader::BoundStage, Shader::MaxStageTypes> mst{};
                ReadStages(reinterpret_cast<const u8*>(&mrec + 1), mrec.stage_count, mst);
                auto& [ni, ns] = mat_n[m];
                x.mat_resolves.fetch_add(1, std::memory_order_relaxed);
                ok = rasterizer->A2ResolveMaterial(mr.base, mst[u32(Shader::LogicalStage::Fragment)],
                                                   st.cmds->mat_slots[m].data(), ni, ns, mat_ids,
                                                   mr.h1, mr.h2, mr.kn,
                                                   &mst[u32(Shader::LogicalStage::Vertex)]);
                if (ok) {
                    tab_verify();
                }
            }
            if (tab_mode != 0) {
                rasterizer->A2TexTableTouch(); // q8
            }
            // A rotation mid-loop invalidates the earlier materials' slots.
            ok = ok && rasterizer->A2BindlessCur() == set0;
            if (!ok) {
                x.rej_mat.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (verify) {
                // The old path's ring write of each colour draw compares its slot words with these.
                auto& ex = rasterizer->a2_expect;
                if (ex.size() > 200000) {
                    ex.clear(); // never written: stale
                }
                v_pass = true;
                for (const auto& sw : st.cmds->slot_writes) {
                    if (sw.mat < st.materials.size()) {
                        auto& e2 = ex[u64{sw.tag} << 32 | sw.idx];
                        e2.slots = st.cmds->mat_slots[sw.mat];
                        e2.n_img = mat_n[sw.mat].first;
                        e2.n_smp = mat_n[sw.mat].second;
                        x.expect_filed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                return false;
            }
            st.cmds->bindless_set = rasterizer->A2BindlessCommit();
        }
        lap(g_blc.pub_resolve);
        if (verify) {
            v_pass = true;
            return false;
        }
        // l4g: non-push sub-batches get their set now (before anything of the stretch is recorded)
        if (!Vulkan::A2::CommitSets(*st.cmds)) [[unlikely]] {
            x.rej_mat.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (g.gpu_write) {
            rasterizer->A2GlobalBarrier();
            g.gpu_write = false;
        }
        mirror->ctx_epoch = rec.ctx_epoch;
        rasterizer->A2BeginStretch(rec.draw.pipeline);
        if (tab_used && VideoCore::TexTable::Mode() != 2 &&
            (st.tab_gen != VideoCore::TexTable::Gen() ||
             !VideoCore::TexTable::ImgsCurrent(st.tab_kseq, st.tab_imgs))) { // q10
            // q8: an image of a table material changed (e.g. became a target of this pass).
            VideoCore::TexTable::g_c[3].fetch_add(1, std::memory_order_relaxed);
            x.rej_rt.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (tab_used && VideoCore::TexTable::V2()) {
            for (const auto& [tid, tep] : st.tab_imgs) { // q10b: the table's images too
                mat_ids.push_back(VideoCore::ImageId{tid});
            }
        }
        if (colour && !rasterizer->A2ImagesSafe(mat_ids)) {
            // A sampled image is a target of this pass: the old path (per-draw barriers) runs the
            // stretch; the pass it began is reused.
            x.rej_rt.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        lap(g_blc.pub_begin);
        auto& sched = rasterizer->GetScheduler();
        // p2 (SHADPS4_A2_SEC): execute the worker's secondary when it fits this pass and set 1.
        vk::CommandBuffer sec = st.secondary;
        auto& sc = Vulkan::A2::g_sec_c;
        if (!sec) {
            if (Vulkan::A2::SecondaryMode()) {
                sc[4].fetch_add(1, std::memory_order_relaxed);
            }
        } else if (colour && rasterizer->A2BindlessCur() != st.bl_idx) {
            sc[5].fetch_add(1, std::memory_order_relaxed);
            sec = vk::CommandBuffer{};
        } else {
            const auto& rs = sched.GetRenderState();
            bool fits = rs.num_color_attachments == st.inh.n_color;
            for (u32 i = 0; fits && i < st.inh.n_color; ++i) {
                fits = bool(rs.color_attachments[i].image_view) ==
                       (st.inh.color[i] != vk::Format::eUndefined);
            }
            const auto& db = rs.depth_stencil_attachment;
            fits = fits && bool(db.has_depth) == (st.inh.depth != vk::Format::eUndefined) &&
                   bool(db.has_stencil) == (st.inh.stencil != vk::Format::eUndefined);
            if (!fits) {
                sc[6].fetch_add(1, std::memory_order_relaxed);
                sec = vk::CommandBuffer{};
            }
        }
        if (sec) {
            // p2c: the lane writes the slot words and drops the last StretchCmds reference (as
            // Replay did) - on the back they cost ~0.3 ms per frame (s603).
            sched.Record([cmds = st.cmds, sec](vk::CommandBuffer cmdbuf) {
                Vulkan::A2::WriteSlots(*cmds);
                cmdbuf.executeCommands(sec);
            });
            const u64 n = sc[3].fetch_add(1, std::memory_order_relaxed) + 1;
            if ((n & 0xffff) == 0) {
                LOG_INFO(Render, "A2 secondaries: recorded {} skipped (no pool {} l4g set {}) "
                                 "executed {} closure (none {} set moved {} pass differs {}) "
                                 "pool resets {}",
                         sc[0].load(), sc[1].load(), sc[2].load(), sc[3].load(), sc[4].load(),
                         sc[5].load(), sc[6].load(), sc[7].load());
            }
        } else {
            sched.Record([cmds = st.cmds, inst = &rasterizer->A2Instance()](
                             vk::CommandBuffer cmdbuf) { Vulkan::A2::Replay(*cmds, cmdbuf, *inst); });
        }
        st.tick->store(sched.CurrentTick(), std::memory_order_release);
        if (VideoCore::g_lane_probe) [[unlikely]] {
            if (st.probe_over ||
                st.probe_gen != VideoCore::g_lprobe_gen.load(std::memory_order_relaxed)) {
                VideoCore::LaneProbeReadAll(sched.CurrentTick(), st.probe_over ? 26u : 18u); // q9b; q32 site
            } else {
                for (const auto& [plo, phi] : st.probe_reads) {
                    VideoCore::LaneProbeRead(plo, phi, sched.CurrentTick(), 9, 9);
                }
            }
        }
        rasterizer->A2EndStretch();
        lap(g_blc.pub_record);
        // The stretch's register writes (its records are never visited), after the pass was
        // resolved from the mirror at its first draw.
        for (const auto& [off, val] : sd.delta) {
            mirror->regs.reg_array[off] = val;
        }
        mirror->ctx_epoch = sd.ctx_epoch;
        lap(g_blc.pub_delta);
        x.published.fetch_add(1, std::memory_order_relaxed);
        x.pub_draws.fetch_add(st.draws, std::memory_order_relaxed);
        return true;
    };
    // Per-second line (the s525 swings): this thread's logical processor and clock probe, its
    // old-path draws and published stretches, and the P/E samples of the front and lean workers.
    static thread_local u64 sec_tsc = __rdtsc(), sec_draws = 0, sec_pub = 0;
    if (x.on) [[likely]] {
        const u64 now = __rdtsc();
        if (now - sec_tsc > 3'500'000'000ull) {
            const u32 cpu = Common::CurrentProcessor();
            const u32 probe = Common::ClockProbe();
            const u64 bd = x.back_draws.load(std::memory_order_relaxed);
            const u64 pb = x.published.load(std::memory_order_relaxed);
            LOG_INFO(Render,
                     "Lean second: back cpu {} ({}) clock probe {} ticks (~{:.2f} GHz) | back old-path "
                     "draws {} stretches {} | front P/E {}/{} lean P/E {}/{}",
                     cpu, Common::IsPerformanceProcessor(cpu) ? "P" : "E", probe,
                     probe ? 8192.0 / double(probe) * 3.5 : 0.0, bd - sec_draws, pb - sec_pub,
                     x.front_p.exchange(0), x.front_e.exchange(0), x.lean_p.exchange(0),
                     x.lean_e.exchange(0));
            sec_tsc = now;
            sec_draws = bd;
            sec_pub = pb;
        }
    }
    u32 si = 0; // next stretch of this job
    // SHADPS4_A2=2: the stretch whose draws run on the old path and are compared.
    const Vulkan::A2::Stretch* vs = nullptr;
    const A2Entry::Side* vsd = nullptr;
    size_t vs_inh = 0, vs_clip = 0, vs_fmt = 0; // next direct-check entries of `vs`
    u64 wait = 0;
    if (blc) [[unlikely]] {
        g_blc.pro_cyc += __rdtsc() - follow_t0;
    }
    for (u32 spins = 0;;) {
        if (split->FrontHasExited() || s.stop.load(std::memory_order_relaxed)) [[unlikely]] {
            break;
        }
        if (const auto* r = ch.Next()) {
            spins = 0;
            if (blc) [[unlikely]] {
                ++g_blc.loop_recs;
            }
            if (r->type == RT::LeanEnd) {
                ch.Done(r);
                break;
            }
            if (r->type == RT::RegWrite && x.on && g_a2 == 1) {
                // A stretch's pre-run starts here: its words into the mirror (a RegWrite record
                // is a plain copy into it), the run released up to the stretch's first draw.
                const u32 ready = job.xl_ready.load(std::memory_order_acquire);
                if (si < ready && job.xl_side[si].pre_pos == ch.ReadPos()) {
                    const auto& sd = job.xl_side[si];
                    for (const auto& [off, val] : sd.pre_delta) {
                        mirror->regs.reg_array[off] = val;
                    }
                    const u64 last = job.xl[si]->first_seq - 1;
                    x.pre_folds.fetch_add(1, std::memory_order_relaxed);
                    x.pre_records.fetch_add(last + 1 - r->seq, std::memory_order_relaxed);
                    ch.DoneThrough(sd.first_pos, last);
                    continue;
                }
            }
            if (r->type == RT::Draw && x.on) {
                const u32 ready = job.xl_ready.load(std::memory_order_acquire);
                while (si < ready && job.xl[si]->first_seq < r->seq) {
                    ++si; // cannot happen (stretches start at a visited draw): counted
                    x.seq_skips.fetch_add(1, std::memory_order_relaxed);
                }
                if (si < ready && job.xl[si]->first_seq == r->seq) {
                    auto& st = *job.xl[si];
                    const auto& sd = job.xl_side[si];
                    ++si;
                    const u64 p0 = blc ? __rdtsc() : 0;
                    const bool pubd = publish(st, sd, *r);
                    if (blc) [[unlikely]] {
                        g_blc.pub_cyc += __rdtsc() - p0;
                        ++g_blc.pub_n;
                    }
                    if (pubd) {
                        ch.DoneThrough(sd.end_pos, st.last_seq);
                        ++s.back_records;
                        if (num_commands.load(std::memory_order_relaxed) != 0) {
                            const u64 c0 = blc ? __rdtsc() : 0;
                            ProcessCommands();
                            if (blc) [[unlikely]] {
                                g_blc.pc_cyc += __rdtsc() - c0;
                            }
                        }
                        continue;
                    }
                    if (g_a2 == 2 && !v_pass) {
                        x.verify_skips.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (g_a2 == 2 && v_pass) {
                        vs = &st;
                        vsd = &sd;
                        vs_inh = 0;
                        vs_clip = 0;
                        vs_fmt = 0;
                    }
                }
                if (vs && r->seq == vs->first_seq && vsd->pre_pos != ~u64{0}) {
                    // The pre-delta = the back's mirror at the first draw (last value per offset).
                    static thread_local std::unordered_set<u32> pseen;
                    pseen.clear();
                    u32 bad = 0;
                    for (auto it = vsd->pre_delta.rbegin(); it != vsd->pre_delta.rend(); ++it) {
                        if (pseen.insert(it->first).second) {
                            bad += mirror->regs.reg_array[it->first] != it->second ? 1 : 0;
                        }
                    }
                    x.pre_cmp.fetch_add(pseen.size(), std::memory_order_relaxed);
                    if (bad != 0 && x.pre_bad.fetch_add(bad, std::memory_order_relaxed) < 16) {
                        LOG_WARNING(Render, "Lean L3 verify: {} pre-delta words differ at job draw "
                                            "seq {}",
                                    bad, r->seq);
                    }
                }
                if (vs && r->seq >= vs->first_seq && r->seq <= vs->last_seq) {
                    // The full key from the BACK's mirror at this draw (its registers before it).
                    const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(r + 1);
                    if (rec.draw.pipeline) {
                        const u64 bk = A2RenderKey(mirror->regs, rec.draw.pipeline);
                        x.rk_cmp.fetch_add(1, std::memory_order_relaxed);
                        if (bk != vsd->render_key &&
                            x.rk_bad.fetch_add(1, std::memory_order_relaxed) < 16) {
                            LOG_WARNING(Render, "Lean L3 verify: render key mismatch at job draw seq "
                                                "{} (stretch {}..{})",
                                        r->seq, vs->first_seq, vs->last_seq);
                        }
                    }
                    // l3e direct checks (the mirror = the registers at this draw).
                    const bool gen_same =
                        rasterizer->A2BufferCache().StructGeneration() == vs->buf_gen;
                    const auto& ic = vs->cmds->inh_checks;
                    bool ud_done = false;
                    for (; vs_inh < ic.size() && ic[vs_inh].seq <= r->seq; ++vs_inh) {
                        const auto& c = ic[vs_inh];
                        if (c.seq != r->seq) {
                            continue;
                        }
                        if (!gen_same) {
                            x.verify_mid.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        const auto* buf = rasterizer->A2BufferCache().NativeBufferAt(c.addr, 0, 32);
                        const u64 clamp =
                            Core::Memory::Instance()->ClampRangeSize(c.addr, c.vsize);
                        const bool ok = buf && c.addr - buf->CpuAddr() == c.off &&
                                        std::min<u64>(clamp, 0xffffffffull) == c.size;
                        x.inh_cmp.fetch_add(1, std::memory_order_relaxed);
                        if (!ok && x.inh_bad.fetch_add(1, std::memory_order_relaxed) < 16) {
                            LOG_WARNING(Render,
                                        "Lean L3 verify: inherited slot {:#x} worker off {:#x} size "
                                        "{:#x}, back off {:#x} clamp {:#x} (job draw seq {})",
                                        c.addr, c.off, c.size,
                                        buf ? c.addr - buf->CpuAddr() : ~u64{0}, clamp, r->seq);
                        }
                        if (!ud_done) {
                            ud_done = true;
                            std::array<Shader::BoundStage, Shader::MaxStageTypes> rst{};
                            ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, rst);
                            const auto cmp_ud = [&](const Shader::BoundStage& bs,
                                                    const std::array<u32, 16>& m) {
                                const size_t n = std::min<size_t>(bs.user_data.size(), 16);
                                u32 bad = 0;
                                for (size_t i = 0; i < n; ++i) {
                                    bad += bs.user_data[i] != m[i] ? 1 : 0;
                                }
                                x.ud_cmp.fetch_add(n, std::memory_order_relaxed);
                                if (bad != 0 &&
                                    x.ud_bad.fetch_add(bad, std::memory_order_relaxed) < 16) {
                                    LOG_WARNING(Render, "Lean L3 verify: {} user-data words differ "
                                                        "from the mirror (job draw seq {})",
                                                bad, r->seq);
                                }
                            };
                            const auto& vsb = rst[u32(Shader::LogicalStage::Vertex)];
                            const auto& psb = rst[u32(Shader::LogicalStage::Fragment)];
                            if (vsb.info) {
                                cmp_ud(vsb, mirror->regs.vs_program.user_data);
                            }
                            if (psb.info) {
                                cmp_ud(psb, mirror->regs.ps_program.user_data);
                            }
                        }
                    }
                    const auto& fc = vs->cmds->fmt_checks;
                    for (; vs_fmt < fc.size() && fc[vs_fmt].seq <= r->seq; ++vs_fmt) {
                        const auto& c = fc[vs_fmt];
                        if (c.seq != r->seq) {
                            continue;
                        }
                        if (!gen_same) {
                            x.verify_mid.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        const auto* buf = rasterizer->A2BufferCache().NativeBufferAt(c.addr, 0, 33);
                        const bool ok = buf && static_cast<u32>(buf->CpuAddr()) == c.base;
                        x.fmt_cmp.fetch_add(1, std::memory_order_relaxed);
                        if (!ok && x.fmt_bad.fetch_add(1, std::memory_order_relaxed) < 16) {
                            LOG_WARNING(Render, "Lean L3 verify: formatted slot {:#x} worker base "
                                                "{:#x}, back {:#x} (job draw seq {})",
                                        c.addr, c.base, buf ? buf->CpuAddr() : ~u64{0}, r->seq);
                        }
                    }
                    const auto& cc = vs->cmds->clip_checks;
                    for (; vs_clip < cc.size() && cc[vs_clip].seq <= r->seq; ++vs_clip) {
                        const auto& c = cc[vs_clip];
                        if (c.seq != r->seq) {
                            continue;
                        }
                        bool ok = mirror->regs.clipper_control.user_clip_plane_enable == c.enable;
                        for (u32 i = 0; ok && i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                            const auto& p = mirror->regs.clip_user_data[i];
                            ok = p.data_x == c.planes[i * 4] && p.data_y == c.planes[i * 4 + 1] &&
                                 p.data_z == c.planes[i * 4 + 2] && p.data_w == c.planes[i * 4 + 3];
                        }
                        x.clip_cmp.fetch_add(1, std::memory_order_relaxed);
                        if (!ok && x.clip_bad.fetch_add(1, std::memory_order_relaxed) < 16) {
                            LOG_WARNING(Render, "Lean L3 verify: clip planes differ from the mirror "
                                                "(job draw seq {})",
                                        r->seq);
                        }
                    }
                }
            }
            switch (r->type) {
            case RT::Dispatch:
            case RT::Call:
                g_a2s.gpu_write = true;
                break;
            case RT::Packet:
                g_a2s.gpu_write = g_a2s.gpu_write || !A2NoopDma(*r);
                break;
            case RT::Draw:
                // Monotonic, one writer (this thread): no locked RMW per draw.
                x.back_draws.store(x.back_draws.load(std::memory_order_relaxed) + 1,
                                   std::memory_order_relaxed);
                break;
            default:
                break;
            }
            const u64 e0 = blc ? __rdtsc() : 0;
            ExecuteRecordImpl(*r); // the plain path for everything not in a published stretch
            if (blc) [[unlikely]] {
                const u32 t = static_cast<u32>(r->type) & 31;
                const u64 dt = __rdtsc() - e0;
                g_blc.job_cyc[t] += dt;
                ++g_blc.job_n[t];
                if (r->type == RT::Draw) {
                    const u32 w =
                        reinterpret_cast<const SplitDrawRecord*>(r + 1)->job_pad & 63;
                    g_blc.why_cyc[w] += dt;
                    ++g_blc.why_n[w];
                }
            }
            if (vs && r->type == RT::Draw && r->seq == vs->last_seq) {
                // The delta the back would have applied = its mirror after the stretch's last draw
                // (the last value per offset).
                static thread_local std::unordered_set<u32> seen;
                seen.clear();
                u32 bad = 0;
                for (auto it = vsd->delta.rbegin(); it != vsd->delta.rend(); ++it) {
                    if (seen.insert(it->first).second) {
                        bad += mirror->regs.reg_array[it->first] != it->second ? 1 : 0;
                    }
                }
                x.delta_cmp.fetch_add(seen.size(), std::memory_order_relaxed);
                if (bad != 0 && x.delta_bad.fetch_add(bad, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render, "Lean L3 verify: {} delta words differ after job draw seq {} "
                                        "(stretch {}..{})",
                                bad, r->seq, vs->first_seq, vs->last_seq);
                }
                vs = nullptr;
                vsd = nullptr;
            }
            ch.Done(r);
            ++s.back_records;
            if (num_commands.load(std::memory_order_relaxed) != 0) {
                const u64 c0 = blc ? __rdtsc() : 0;
                ProcessCommands();
                if (blc) [[unlikely]] {
                    g_blc.pc_cyc += __rdtsc() - c0;
                }
            }
            if (early_service_fn) [[unlikely]] {
                early_service_fn(early_service_ctx); // m2: particle work between job records
            }
            continue;
        }
        if (early_service_fn) [[unlikely]] {
            early_service_fn(early_service_ctx); // m2: and while the workers are behind
        }
        const u64 c0 = blc ? __rdtsc() : 0;
        ProcessCommands(); // guest commands run while the back waits for the worker
        if (blc) [[unlikely]] {
            g_blc.pc_cyc += __rdtsc() - c0;
        }
        const u64 w0 = __rdtsc();
        ++spins < 4096 ? _mm_pause() : std::this_thread::yield();
        wait += __rdtsc() - w0;
    }
    lean_following = false;
    s.back_wait_ticks += wait;
    const u64 epi_t0 = blc ? __rdtsc() : 0;
    if (x.on) {
        const u32 n = job.xl_ready.load(std::memory_order_acquire);
        for (u32 i = 0; i < n; ++i) {
            auto& st = job.xl[i];
            if (st && st->tick && st->tick->load(std::memory_order_relaxed) == 0) {
                st->tick->store(1, std::memory_order_release); // never published: free
            }
            st.reset();
        }
        job.xl_ready.store(0, std::memory_order_relaxed);
        job.back_waiting.store(0, std::memory_order_relaxed);
    }
    job.state.store(0, std::memory_order_release); // the slot is free again
    if (blc) [[unlikely]] {
        const u64 t1 = __rdtsc();
        g_blc.epi_cyc += t1 - epi_t0;
        g_blc.wait_cyc += wait;
        g_blc.follow_cyc += t1 - follow_t0;
        ++g_blc.follows;
    }
}

void Liverpool::EmitDraw(const PM4Header* header, const char* name, bool host_marker,
                         const CpSplit::DrawArgs& args) {
    if (g_lean != 0) [[unlikely]] {
        ++g_lean_s.c.front_draws;
    }
    if (AmdGpu::NativeRec::g_mode >= 2) [[unlikely]] {
        if (NativeRecMember(header, name, host_marker, args)) {
            if (g_lean != 0) [[unlikely]] {
                ++g_lean_s.c.front_member;
            }
            return;
        }
    }
    const u64 t_nr = AmdGpu::NativeRec::g_mode >= 2 ? __rdtsc() : 0;
    u32 tag = 0;
    u64 t_front = 0;
    if (g_draw_tag_front) [[unlikely]] {
        t_front = __rdtsc();
        if (header->type3.opcode == PM4ItOpcode::DrawIndex2) {
            const auto* w = reinterpret_cast<const u32*>(header);
            if (w[6] == 0xc0021000u && w[7] == PM4CmdNop::PayloadType::NativeDrawTag) {
                tag = w[8];
            }
        }
    }
    SCOPE_EXIT {
        if (g_draw_tag_front) [[unlikely]] {
            const u64 dt = __rdtsc() - t_front;
            auto& f = g_front_tag;
            (tag ? f.c_tagged : f.c_untagged) += dt;
            ++(tag ? f.n_tagged : f.n_untagged);
            if (((f.n_tagged + f.n_untagged) & 0xfffff) == 0) {
                LOG_INFO(Render, "DrawTagFront: EmitDraw cycles per draw: tagged {:.0f} (n={}) "
                                 "untagged {:.0f} (n={})",
                         f.n_tagged ? double(f.c_tagged) / double(f.n_tagged) : 0.0, f.n_tagged,
                         f.n_untagged ? double(f.c_untagged) / double(f.n_untagged) : 0.0,
                         f.n_untagged);
                f = {};
            }
        }
    };
    Vulkan::Rasterizer::SplitDraw draw{.args = args};
    // Keep this lookup before the Sync below, also for the periodic VERIFY draw: the back's
    // re-lookup of that draw (DrawRelookup) is only meaningful because this one ran while
    // earlier posted records could still be outstanding. A drain before this line would blind it.
    if (g_lean_s.toucher) [[unlikely]] {
        g_lean_s.toucher->Pretouch(regs); // faults before PrepareDrawFront's exclusive lock
    }
    rasterizer->PrepareDrawFront(draw);
    const bool has_stages = draw.pipeline && draw.draw_class ==
                                                 Vulkan::Rasterizer::DrawClass::Draw &&
                            !draw.skip;
    // Run-ahead: posted, except the periodic VERIFY draw, which syncs for the full check.
    // Depth 0: every draw syncs.
    bool sync = !split_run_ahead;
    u32 verify = 0;
    if (split_verify) {
        if (!split_run_ahead || (++split_verify_draws % split_verify_every) == 0) {
            sync = true;
            verify = DrawVerifyFull;
        } else {
            verify = DrawVerifySnapshot;
        }
    }
    u32 bytes = sizeof(SplitDrawRecord);
    // SHADPS4_LEAN=2: the stage fields captured under the cache's lock (FrontCapture).
    const Vulkan::GraphicsLookupScratch* cap = has_stages ? rasterizer->FrontCapture() : nullptr;
    if (has_stages && cap) {
        for (const auto& slot : cap->stages) {
            if (slot.info) {
                bytes += StageBytes(static_cast<u32>(slot.user_data.size()),
                                    static_cast<u32>(slot.flattened_ud_buf.size()));
            }
        }
    } else if (has_stages) {
        for (const Shader::Info* info : draw.pipeline->GetStages()) {
            if (info) {
                bytes += StageBytes(*info);
            }
        }
    }
    // FRONT_DERIVE: derived here, before the Sync below and from the same Info the stage copy
    // reads, so the V# table reads happen at parse time (whether that differs from the back's
    // read at execute time is what mode 2 measures). Not for indirect draws: DrawIndirectTail
    // keeps the back-side derivation.
    u32 vi_bytes = 0;
    // q1: SHADPS4_DERIVE_INDIRECT=1 (as the lean parser).
    static const bool derive_indirect = [] {
        const char* e = std::getenv("SHADPS4_DERIVE_INDIRECT");
        return e && e[0] == '1';
    }();
    if (split_front_derive != 0 && has_stages && (!args.is_indirect || derive_indirect)) {
        const auto stages = draw.pipeline->GetStages();
        Shader::BoundStage vs{stages[static_cast<u32>(Shader::LogicalStage::Vertex)]};
        if (cap) {
            const auto& slot = cap->stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
            vs.info = slot.info;
            vs.user_data = slot.user_data;
            vs.flattened_ud_buf = slot.flattened_ud_buf;
            vs.pgm_base = slot.pgm_base;
        }
        vi_bytes = draw.pipeline->DeriveVertexInputs(*front_vi, regs.vgt_instance_step_rate_0,
                                                     regs.vgt_instance_step_rate_1, vs);
        bytes += vi_bytes;
        // Number the description part: the back skips the setVertexInputEXT of a repeated id.
        auto& vi = *front_vi;
        bool same = vi.header.count == front_vi_prev_count;
        for (u32 i = 0; same && i < vi.header.count; ++i) {
            same = std::memcmp(&vi.attribs[i].desc, &front_vi_prev[i],
                               sizeof(CpSplit::VertexInputDesc)) == 0;
        }
        if (!same) {
            ++front_vi_desc_id;
            front_vi_prev_count = vi.header.count;
            for (u32 i = 0; i < vi.header.count; ++i) {
                front_vi_prev[i] = vi.attribs[i].desc;
            }
        }
        vi.header.desc_id = front_vi_desc_id;
        if (VideoCore::g_native_colour) [[unlikely]] {
            // SHADPS4_NATIVE_COLOUR: a variant per (pipeline, vertex description), front-side.
            rasterizer->PrepareNativeColourFront(draw.pipeline, vi);
        }
    }
    if (verify == DrawVerifySnapshot) {
        bytes += VerifySnapshotBytes;
    }
    if (AmdGpu::NativeRec::g_mode != 0) [[unlikely]] {
        NativeRecFront(header, draw, regs, context_change_epoch, vi_bytes ? &*front_vi : nullptr,
                       rasterizer->FrontCapture());
    }
    SegB::Pred segb{};
    bool segx_hold = false;
    if (AmdGpu::NativeRec::g_segb != 0) [[unlikely]] {
        if (!SegB::g_state.init) {
            SegB::Init(SegB::g_state, split.get(), split_run_ahead, split_verify);
        }
        const bool member = SegB::NativeSegBDraw(header, draw.pipeline, has_stages,
                                                 vi_bytes ? front_vi->attribs.data() : nullptr,
                                                 vi_bytes ? front_vi->header.count : 0u, regs,
                                                 bytes, segb);
        segx_hold = member && SegB::g_state.exec;
    }
    // SHADPS4_NATIVE_SEGX=1: a segment member's record is built but not posted (member 0's
    // rides in the Segment record at the close).
    auto* rec = segx_hold ? SegB::ScratchRecord(bytes)
                          : split->Reserve<SplitDrawRecord>(CpSplit::RecordType::Draw, bytes);
    rec->draw = draw;
    rec->ctx_epoch = context_change_epoch;
    rec->marker = host_marker ? name : nullptr;
    rec->cmd_address = header;
    rec->stage_count = 0;
    rec->verify = verify;
    rec->vi_bytes = vi_bytes;
    rec->pad = tag;
    rec->nr_tag = 0;
    rec->nr_idx = 0;
    rec->job_tag = 0;
    rec->job_idx = 0;
    rec->job_member = 0;
    rec->job_pad = 0;
    rec->segb_id = segb.id;
    rec->segb_member = segb.member;
    rec->segb_index_addr = segb.index_addr;
    rec->segb_index_count = segb.index_count;
    rec->segb_instances = segb.instances;
    if (AmdGpu::NativeRec::g_mode >= 2 && header->type3.opcode == PM4ItOpcode::DrawIndex2) {
        const auto* w = reinterpret_cast<const u32*>(header);
        if (w[6] == 0xc0021000u && w[7] == AmdGpu::NativeRec::Magic) {
            rec->job_tag = w[8];
            rec->job_idx = w[9];
        }
    } else if (AmdGpu::NativeRec::g_mode >= 2 &&
               header->type3.opcode == PM4ItOpcode::DrawIndexIndirect) {
        // q2: the HLE's tagged NOP after DRAW_INDEX_INDIRECT (the front parsed it: in bounds).
        const auto* w = reinterpret_cast<const u32*>(header);
        if (w[5] == 0xc0021000u && w[6] == AmdGpu::NativeRec::Magic) {
            rec->job_tag = w[7];
            rec->job_idx = w[8];
        }
    }
    // Copied now: the front's next lookup rewrites these Info fields (inventory 4.4). With
    // run-ahead this is what makes the record self-contained: the back reads user data, flat
    // buffer and pgm_base only from here, and the Info only for its immutable part.
    auto* out = reinterpret_cast<u8*>(rec + 1);
    if (has_stages && cap) {
        for (u32 i = 0; i < cap->stages.size(); ++i) {
            const auto& slot = cap->stages[i];
            if (slot.info) {
                out = WriteStage(out, slot.info, slot.pgm_base, slot.user_data,
                                 slot.flattened_ud_buf, i);
                ++rec->stage_count;
            }
        }
    } else if (has_stages) {
        const auto stages = draw.pipeline->GetStages();
        for (u32 i = 0; i < stages.size(); ++i) {
            if (stages[i]) {
                out = WriteStage(out, *stages[i], i);
                ++rec->stage_count;
            }
        }
    }
    if (vi_bytes != 0) {
        out = front_vi->Write(out);
    }
    if (verify == DrawVerifySnapshot) {
        WriteVerifySnapshot(out, regs, last_cb_extent, last_db_extent);
    }
    if (AmdGpu::NativeRec::g_mode >= 2) [[unlikely]] {
        NativeRecHead(header, draw.pipeline,
                      draw.draw_class == Vulkan::Rasterizer::DrawClass::Draw && !draw.skip,
                      vi_bytes, reinterpret_cast<const u8*>(rec + 1),
                      rec->stage_count);
        if (g_nr_run.active && g_nr_run.next_k == 1) {
            rec->nr_tag = g_nr_run.tag;
            rec->nr_idx = g_nr_run.head_idx;
        }
        auto& st = g_nr_mst;
        st.c_normal += __rdtsc() - t_nr;
        if ((++st.n_normal & 0x3ffff) == 0) {
            LOG_INFO(Render,
                     "NativeRec members: taken={} refused={} (why 0 tag/mode 1 ring/stale 2 "
                     "run order 3 epoch/regs 4 infos 5 class/skip 6 vi count 7 payload: {} {} {} {} {} {} {} {}) | "
                     "mode-3 compare ok={} MISMATCH={} | front cycles per draw: member {:.0f} "
                     "normal EmitDraw {:.0f} (n={})",
                     st.members, st.refused, st.why[0], st.why[1], st.why[2], st.why[3],
                     st.why[4], st.why[5], st.why[6], st.why[7], st.cmp_ok, st.cmp_bad,
                     st.members ? double(st.c_member) / double(st.members) : 0.0,
                     double(st.c_normal) / double(st.n_normal), st.n_normal);
        }
    }
    if (segb.id != 0 && (segb.member == 0 || SegB::g_state.exec) && SegB::g_state.segx) [[unlikely]] {
        SegB::KeepMember(rec, bytes);
    }
    if (segx_hold) [[unlikely]] {
        split->MaybeLog();
        return;
    }
    split->Commit();
    if (sync) {
        split->Sync(split_run_ahead ? CpSplit::DrainReason::VerifyFull
                                    : CpSplit::DrainReason::DepthZero);
    } else {
        split->PostRunAhead(CpSplit::PostKind::Draw);
    }
    split->MaybeLog();
}

namespace {
/// SHADPS4_DISPATCH_SYNC_STATS=1 (diagnostic, default off, front thread only): for every dispatch,
/// which guest sync-type packets the same queue issued since its previous dispatch. Decides whether
/// the game orders its back-to-back compute dispatches itself (CS_PARTIAL_FLUSH / ACQUIRE_MEM /
/// RELEASE_MEM / WAIT_REG_MEM) or relies on GCN overlapping them (2026-09-27, fire particles).
struct DispatchSyncStats {
    enum Bit : u32 {
        EvCsFlush = 1u << 0,
        EvCacheFlush = 1u << 1,
        EvOther = 1u << 2,
        Acquire = 1u << 3,
        Release = 1u << 4,
        WaitReg = 1u << 5,
        Dma = 1u << 6,
        WriteData = 1u << 7,
        EopEos = 1u << 8,
        Draw = 1u << 9,
        Count = 10
    };
    bool enabled = [] {
        const char* v = std::getenv("SHADPS4_DISPATCH_SYNC_STATS");
        return v && v[0] == '1';
    }();
    /// SHADPS4_GUEST_DISPATCH_ORDER=1: also number guest dispatch runs per queue (a run ends at
    /// every sync-type packet or draw of that queue); the back skips buffer barriers inside a run.
    bool order = [] {
        const char* v = std::getenv("SHADPS4_GUEST_DISPATCH_ORDER");
        return v && v[0] == '1';
    }();
    std::array<u64, 65> run{};
    u64 run_counter{};
    u64 Run(u32 key) {
        if (run[key] == 0) {
            run[key] = ++run_counter;
        }
        return run[key];
    }
    std::array<u32, 65> since{};          // per queue key: bits seen since its last dispatch
    std::array<bool, 65> had_dispatch{};  // queue key has dispatched before
    std::array<std::array<u64, Count>, 2> bit_hits{}; // [gfx=0/asc=1][bit]
    std::array<u64, 2> dispatches{}, back_to_back{}, first{};
    std::array<u64, 64> ev_types{};
    u64 total{};

    /// Packets that do not end a guest dispatch run: register writes, padding/markers and IB jumps.
    /// Everything else (events, fences, waits, DMA, draws, semaphores, ...) does.
    static bool RunNeutral(PM4ItOpcode op) {
        switch (op) {
        case PM4ItOpcode::Nop:
        case PM4ItOpcode::SetShReg:
        case PM4ItOpcode::SetContextReg:
        case PM4ItOpcode::SetUconfigReg:
        case PM4ItOpcode::SetConfigReg:
        case PM4ItOpcode::IndirectBuffer:
        case PM4ItOpcode::IndirectBufferConst:
        case PM4ItOpcode::DispatchDirect:
        case PM4ItOpcode::DispatchIndirect:
            return true;
        default:
            return false;
        }
    }

    void Note(u32 key, PM4ItOpcode op, const PM4Header* header) {
        if (!enabled && !order) {
            return;
        }
        if (order && !RunNeutral(op)) {
            run[key] = ++run_counter; // a new run starts after this packet
        }
        if (!enabled) {
            return;
        }
        u32 bit = 0;
        switch (op) {
        case PM4ItOpcode::EventWrite: {
            const auto type =
                static_cast<u32>(reinterpret_cast<const PM4CmdEventWrite*>(header)->event_type.Value());
            ++ev_types[type & 63];
            bit = type == static_cast<u32>(EventType::CsPartialFlush) ? EvCsFlush
                  : (type == static_cast<u32>(EventType::CacheFlush) ||
                     type == static_cast<u32>(EventType::CacheFlushTs))
                      ? EvCacheFlush
                      : EvOther;
            break;
        }
        case PM4ItOpcode::AcquireMem:
            bit = Acquire;
            break;
        case PM4ItOpcode::ReleaseMem:
            bit = Release;
            break;
        case PM4ItOpcode::WaitRegMem:
            bit = WaitReg;
            break;
        case PM4ItOpcode::DmaData:
            bit = Dma;
            break;
        case PM4ItOpcode::WriteData:
            bit = WriteData;
            break;
        case PM4ItOpcode::EventWriteEop:
        case PM4ItOpcode::EventWriteEos:
            bit = EopEos;
            break;
        case PM4ItOpcode::DrawIndex2:
        case PM4ItOpcode::DrawIndexAuto:
        case PM4ItOpcode::DrawIndexOffset2:
        case PM4ItOpcode::DrawIndexIndirect:
        case PM4ItOpcode::DrawIndirect:
        case PM4ItOpcode::DrawIndexIndirectMulti:
            bit = Draw;
            break;
        case PM4ItOpcode::DispatchDirect:
        case PM4ItOpcode::DispatchIndirect: {
            const u32 q = key == 64 ? 0 : 1;
            ++dispatches[q];
            if (!had_dispatch[key]) {
                had_dispatch[key] = true;
                ++first[q];
            } else if (since[key] == 0) {
                ++back_to_back[q];
            } else {
                for (u32 b = 0; b < Count; ++b) {
                    bit_hits[q][b] += (since[key] >> b) & 1;
                }
            }
            since[key] = 0;
            if ((++total & 0x7fff) == 0) {
                Log();
            }
            return;
        }
        default:
            return;
        }
        since[key] |= bit;
    }

    void Log() const {
        std::string ev;
        for (u32 t = 0; t < 64; ++t) {
            if (ev_types[t]) {
                ev += fmt::format(" {}:{}", t, ev_types[t]);
            }
        }
        for (u32 q = 0; q < 2; ++q) {
            const auto& h = bit_hits[q];
            LOG_INFO(Render,
                     "DispatchSync: queue={} dispatches={} back_to_back={} first={} with_sync: "
                     "cs_flush={} cache_flush={} ev_other={} acquire={} release={} wait_reg={} "
                     "dma={} write_data={} eop_eos={} draw={}",
                     q ? "asc" : "gfx", dispatches[q], back_to_back[q], first[q], h[0], h[1], h[2],
                     h[3], h[4], h[5], h[6], h[7], h[8], h[9]);
        }
        LOG_INFO(Render, "DispatchSync: event types (type:count){}", ev);
    }
};
DispatchSyncStats g_dispatch_sync;
} // namespace

void Liverpool::EmitDispatch(const PM4Header* header, const char* name, bool host_marker,
                             bool is_indirect, VAddr address, u32 offset, u32 size, u32 queue) {
    // POST_ASC (run-ahead): posted like draws. The record owns everything the back reads besides
    // the register mirror: the cs registers (copied here; the back's BackCsRegs() returns the
    // copy), the stage's user data / flat buffer / pgm_base, the pipeline pointer (stable) and
    // the indirect address. The ASC ring slot may be reused once the front moves on.
    const u32 post_bit =
        queue == GfxQueueId ? u32{CpSplit::PostGfxDispatch} : u32{CpSplit::PostAscDispatch};
    bool post = (split_post_asc & post_bit) != 0;
    u32 verify = 0;
    if (split_verify) {
        if (!post || (++split_verify_dispatches % split_verify_every) == 0) {
            verify = DrawVerifyFull;
        } else {
            verify = DrawVerifySnapshot;
        }
    }
    const bool verify_sync = post && verify == DrawVerifyFull;
    post = post && !verify_sync;
    if (post && split_disp_guard) {
        // The front lookup below reads guest memory (user data tables, flat buffer) that a posted
        // DMA_DATA / WRITE_DATA may not have written yet. DISP_GUARD=0 drops this, as for draws.
        split->SyncIfPostedPackets(CpSplit::DrainReason::DispatchAfterPkt);
    }
    const Vulkan::ComputePipeline* pipeline = rasterizer->PrepareDispatchFront();
    const Shader::Info* info = pipeline ? pipeline->GetStages()[0] : nullptr;
    const u32 stage_bytes = info ? StageBytes(*info) : 0u;
    u32 bytes = sizeof(SplitDispatchRecord) + stage_bytes;
    if (verify == DrawVerifySnapshot) {
        bytes += VerifySnapshotBytes;
    }
    auto* rec = split->Reserve<SplitDispatchRecord>(CpSplit::RecordType::Dispatch, bytes);
    std::memcpy(&rec->cs, &GetCsRegs(), sizeof(ComputeProgram));
    rec->queue = queue;
    rec->verify = verify;
    rec->pipeline = pipeline;
    rec->marker = host_marker ? name : nullptr;
    rec->cmd_address = header;
    rec->address = address;
    rec->offset = offset;
    rec->size = size;
    rec->is_indirect = is_indirect ? 1u : 0u;
    rec->stage_count = 0;
    rec->run = g_dispatch_sync.order
                   ? g_dispatch_sync.Run(queue == GfxQueueId ? 64u : ((queue - 1) & 63u))
                   : 0;
    auto* out = reinterpret_cast<u8*>(rec + 1);
    if (info) {
        out = WriteStage(out, *info, u32(Shader::LogicalStage::Compute));
        rec->stage_count = 1;
    }
    if (verify == DrawVerifySnapshot) {
        WriteVerifySnapshot(out, regs, last_cb_extent, last_db_extent);
    }
    split->Commit();
    if (post) {
        split->PostRunAhead(queue == GfxQueueId ? CpSplit::PostKind::GfxDispatch
                                                : CpSplit::PostKind::AscDispatch);
    } else {
        // Without POST_ASC (or for the periodic VERIFY dispatch) synchronous, as before.
        split->Sync(verify_sync ? CpSplit::DrainReason::VerifyDispatch
                                : CpSplit::DrainReason::Dispatch);
    }
    split->MaybeLog();
}

void Liverpool::ExecuteGfxPacket(const PM4Header* header, bool guest_markers_enabled) {
    // Bodies moved verbatim from ProcessGraphics (the inline path calls this directly).
    const PM4ItOpcode opcode = header->type3.opcode;
    switch (opcode) {
    case PM4ItOpcode::Nop: {
        const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
        if (nop->header.count.Value() == 0) {
            break;
        }

        switch (nop->data_block[0]) {
        case PM4CmdNop::PayloadType::PatchedFlip: {
            // There is no evidence that GPU CP drives flip events by parsing
            // special NOP packets. For convenience lets assume that it does.
            ++VideoCore::g_back_flip_seq; // q35 H5
            if (VideoCore::g_lane_wait_census && rasterizer) [[unlikely]] {
                VideoCore::LaneWaitFrameEnd(rasterizer->GetScheduler().CurrentTick()); // q59b
            }
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
            break;
        }
        case PM4CmdNop::PayloadType::DebugMarkerPush: {
            if (guest_markers_enabled) {
                const auto marker_sz = nop->header.count.Value() * 2;
                const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                             marker_sz};
                rasterizer->ScopeMarkerBegin(label, true);
            }
            break;
        }
        case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
            if (guest_markers_enabled) {
                const auto marker_sz = nop->header.count.Value() * 2;
                const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                             marker_sz};
                const u32 color = *reinterpret_cast<const u32*>(
                    reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                rasterizer->ScopedMarkerInsertColor(label, color, true);
            }
            break;
        }
        case PM4CmdNop::PayloadType::DebugMarkerPop: {
            if (guest_markers_enabled) {
                rasterizer->ScopeMarkerEnd(true);
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    case PM4ItOpcode::EventWrite: {
        // The streamout-flush part is a register write and stays in ProcessGraphics.
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
        // SHADPS4_OCCLUSION_FAKE / SHADPS4_OCCLUSION_LOG (2026-10-02, probe; see
        // scripts/patch/occlusion-probe-2026-10-02.py): what the game's occlusion queries are answered
        // with, and a log of their pattern. Defaults keep the upstream fake (always "visible").
        static const int occlusion_fake = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_FAKE");
            return value ? std::atoi(value) : 1;
        }();
        static const bool occlusion_log = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_LOG");
            return value && value[0] == '1';
        }();
        if (occlusion_log && event->event_type.Value() != EventType::PixelPipeStatDump &&
            (event->event_index.Value() == EventIndex::ZpassDone ||
             event->event_type.Value() == EventType::PixelPipeStatControl ||
             event->event_type.Value() == EventType::PixelPipeStatReset)) {
            static u32 other_logged = 0;
            if (other_logged < 16) {
                ++other_logged;
                LOG_INFO(Render, "OcclusionQ: other event type={} index={} frame={}",
                         static_cast<u32>(event->event_type.Value()),
                         static_cast<u32>(event->event_index.Value()),
                         Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed));
            }
        }
        if (event->event_index.Value() == EventIndex::ZpassDone) {
            if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                u64* results = event->Address<u64*>();
                // SHADPS4_OCCLUSION_QUERIES=1: real queries (vk_occlusion.h) publish their counts
                // here; the probe and the fake below are then unused.
                if (rasterizer && rasterizer->OcclusionDump(results, num_counter_pairs)) {
                    // EVENT_WRITE keeps the submit epoch: no coherence point follows (review B).
                    rasterizer->GetBufferCache().CoherentCpWrite(reinterpret_cast<VAddr>(results),
                                                                 u64{num_counter_pairs} * 16);
                    break;
                }
                if (occlusion_log) [[unlikely]] {
                    static u64 dumps = 0, window_dumps = 0, window_slot8 = 0, window_frame0 = 0;
                    static std::unordered_set<u64> window_addrs;
                    const u64 frame =
                        Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
                    const u64 addr = reinterpret_cast<u64>(results);
                    if (dumps < 48) {
                        LOG_INFO(Render,
                                 "OcclusionQ: dump #{} addr={:#x} slot={} had={:#x} next={:#x} "
                                 "counter={:#x} frame={}",
                                 dumps, addr, (addr & 8) ? 1 : 0, results[0], results[1],
                                 pixel_counter, frame);
                    }
                    if (window_dumps == 0) {
                        window_frame0 = frame;
                    }
                    ++dumps;
                    ++window_dumps;
                    window_slot8 += (addr & 8) ? 1 : 0;
                    window_addrs.insert(addr);
                    if (window_dumps == 2048) {
                        const u64 frames = frame - window_frame0;
                        LOG_INFO(Render,
                                 "OcclusionQ: dumps={} frames={} dumps/frame={:.1f} distinct_addrs={} "
                                 "slot1_share={:.1f}% fake_mode={} total={}",
                                 window_dumps, frames,
                                 frames ? static_cast<double>(window_dumps) / frames : 0.0,
                                 window_addrs.size(), 100.0 * window_slot8 / window_dumps,
                                 occlusion_fake, dumps);
                        window_dumps = 0;
                        window_slot8 = 0;
                        window_addrs.clear();
                    }
                }
                if (occlusion_fake != 2) {
                    u64* const first = results;
                    for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                        *results = pixel_counter | OcclusionCounterValidMask;
                    }
                    if (occlusion_fake != 0) {
                        pixel_counter += OcclusionCounterStep;
                    }
                    if (rasterizer) {
                        rasterizer->GetBufferCache().CoherentCpWrite(
                            reinterpret_cast<VAddr>(first), u64{num_counter_pairs} * 16);
                    }
                }
            }
        }
        break;
    }
    case PM4ItOpcode::EventWriteEos: {
        const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
        if (rasterizer) {
            rasterizer->ProcessDownloadImages();
        }
        // g_cp_wait and SplitHazard are front-side diagnostics noted here on the back. Race-free
        // because fence/WRITE_DATA packets stay synchronous at every depth (front parked), and
        // POST_DMA is refused while SplitHazard is on. POST_ASC is refused with SplitHazard, and
        // its WRITE_DATA bit (posted ASC WRITE_DATA runs the WRITE_DATA case below) with
        // CP_WAIT_STATS. SPLIT-TODO(run-ahead): note these writes at parse time if these packets
        // are ever posted (inventory section 3).
        g_cp_wait.NoteWrite(event_eos->Address());
        SplitHazard::NoteWrite(reinterpret_cast<u64>(event_eos->Address()), 8);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if ((FenceAtGpu() == 1 || FenceAtGpu() == 2 ||
             GpuTimedLabelHit(reinterpret_cast<u64>(event_eos->Address()))) &&
            rasterizer && event_eos->command != PM4CmdEventWriteEos::Command::GdsStore) {
            rasterizer->DeferToGpu([packet = *event_eos] { packet.SignalFence(WriteFenceValue); },
                                   FenceAtGpu() != 2);
        } else {
            event_eos->SignalFence(WriteFenceValue);
        }
        if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
            ASSERT(event_eos->size == 1);
            if (rasterizer) {
                rasterizer->Finish();
                const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                *event_eos->Address() = value;
            }
        }
        break;
    }
    case PM4ItOpcode::EventWriteEop: {
        const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
        if (rasterizer) {
            rasterizer->ProcessDownloadImages();
        }
        g_cp_wait.NoteWrite(event_eop->Address<u32>());
        SplitHazard::NoteWrite(reinterpret_cast<u64>(event_eop->Address<u32>()), 8);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if (((FenceAtGpu() != 0 && FenceAtGpu() != 4) ||
             GpuTimedLabelHit(reinterpret_cast<u64>(event_eop->Address<u32>()))) &&
            rasterizer) {
            rasterizer->DeferToGpu(
                [packet = *event_eop] {
                    packet.SignalFence(WriteFenceValue, [] {
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                    });
                },
                FenceAtGpu() != 2);
            break;
        }
        event_eop->SignalFence(
            [](void* address, u64 data, u32 num_bytes) {
                auto* memory = Core::Memory::Instance();
                if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                    memcpy(address, &data, num_bytes);
                }
            },
            [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
        if (AscRingLatency()) [[unlikely]] {
            g_asc_ring.last_eop_tsc.store(__rdtsc(), std::memory_order_relaxed);
        }
        break;
    }
    case PM4ItOpcode::DmaData: {
        const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
        if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
            break;
        }
        if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
            rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                   true);
        } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                    dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                   dma_data->dst_sel == DmaDataDst::Gds) {
            rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                   dma_data->NumBytes(), true, false);
        } else if (dma_data->src_sel == DmaDataSrc::Data &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                   dma_data->data, false);
        } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                   dma_data->NumBytes(), false, true);
        } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                    dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->SrcAddress<VAddr>(),
                                   dma_data->NumBytes(), false, false);
        } else {
            UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel.Value()),
                            u32(dma_data->dst_sel.Value()));
        }
        break;
    }
    case PM4ItOpcode::WriteData: {
        const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
        ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
        const u32 data_size = (header->type3.count.Value() - 2) * 4;
        u64* address = write_data->Address<u64*>();
        g_cp_wait.NoteWrite(address);
        SplitHazard::NoteWrite(reinterpret_cast<u64>(address), data_size);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if (!write_data->wr_one_addr.Value()) {
            std::memcpy(address, write_data->data, data_size);
        } else {
            UNREACHABLE();
        }
        break;
    }
    case PM4ItOpcode::MemSemaphore: {
        // The wait half of a waiting semaphore stays in ProcessGraphics (it yields).
        const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
        if (mem_semaphore->IsSignaling()) {
            mem_semaphore->Signal();
        } else {
            mem_semaphore->Decrement();
        }
        break;
    }
    case PM4ItOpcode::PfpSyncMe: {
        if (rasterizer) {
            rasterizer->CpSync();
        }
        break;
    }
    default:
        UNREACHABLE_MSG("CpSplit: no back action for PM4 opcode {:#x}", static_cast<u32>(opcode));
    }
}

void Liverpool::ProcessLoop(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        Common::SampleThreadPlacement(CpSplit::t_is_front ? "front" : "cp", 2);
        {
            // Idle here is the CP thread waiting on the GUEST, not emulator cost. Splitting it out
            // is what says whether parallel command recording can pay off - see
            // notes\parallel-command-recording-plan.md.
            const auto t_wait = Common::FaultMonitor::Now();
            const u64 idle_start = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
            g_asc_ring_front_state.store(FrontGuestIdle, std::memory_order_relaxed);
            std::unique_lock lk{submit_mutex};
            if (!(num_commands || num_submits || submit_done)) {
                // SHADPS4_CP_SPLIT_PUB_BATCH: nothing may stay unpublished while the front idles.
                if (split_enabled && CpSplit::t_is_front) {
                    split->PublishNow();
                }
                // Thread-wait column front_idle_ms (always on): only a wait that really blocks.
                const auto idle_from = std::chrono::steady_clock::now();
                Common::CondvarWait(submit_cv, lk, stoken, [this] {
                    return num_commands || num_submits || submit_done;
                });
                Common::FaultMonitor::front_idle_ns.fetch_add(
                    static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - idle_from)
                                         .count()),
                    std::memory_order_relaxed);
            }
            Common::FaultMonitor::CountCpWait(Common::FaultMonitor::SinceNs(t_wait));
            if (g_cp_wait.enabled) {
                g_cp_wait.idle_ns += CpWaitStats::Now() - idle_start;
                g_cp_wait.MaybeLog();
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        // NOTE: StartCapture() used to be called here, when this thread woke with work. Combined
        // with ending at the first completed submit, that recorded an arbitrary slice of a frame.
        // Both capture boundaries now sit at present, in Presenter::Present.

        curr_qid = -1;

        const auto t_active = Common::FaultMonitor::Now();
        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            if (split_task_guard) [[unlikely]] {
                // TASK_GUARD: a finished task no longer drains, so a DMA / WRITE_DATA it posted
                // may still be outstanding when the front resumes a task that is part-way through
                // its buffers. A task not started yet checks its own buffer at entry instead.
                SplitGuardActive();
            }
            const bool ring_latency = split_run_ahead && AscRingLatency();
            if (ring_latency) [[unlikely]] {
                g_asc_ring_front_state.store(curr_qid == s32(GfxQueueId) ? FrontGfxTask
                                                                         : FrontAscTask,
                                             std::memory_order_relaxed);
                if (g_asc_ring.front_queue != u32(curr_qid)) {
                    // Stream switch, in stream order: rides along with the task's next record.
                    g_asc_ring.front_queue = u32(curr_qid);
                    EmitAscRingDiag(*split, AscRingSwitch, u32(curr_qid));
                }
            }
            task.resume();

            if (task.done()) {
                if (ring_latency && curr_qid != s32(GfxQueueId)) [[unlikely]] {
                    EmitAscRingDiag(*split, AscRingEnd, u32(curr_qid));
                }
                // CP split: the guest reads num_submits == 0 as "GPU idle", so the back must
                // have finished the submission first. POST_TASK_DONE: instead of draining, count
                // it pending and post a Retire record; WaitGpuIdle/IsGpuIdle also wait for
                // pending_retire, which the back decrements when it reaches the record.
                if (split_post_task_done) {
                    if (!split_task_guard) {
                        // The old drain also made this task's posted DMA / WRITE_DATA land before
                        // the next top-level buffer is parsed (the nested-IB guards cover only
                        // IBs). TASK_GUARD checks each buffer against the posted writes instead,
                        // when the front starts it (SplitSpanScope) or resumes a task
                        // (SplitGuardActive above).
                        split->SyncIfPostedPackets(CpSplit::DrainReason::TaskDone);
                    }
                    pending_retire.fetch_add(1, std::memory_order_seq_cst);
                    split->PostRetire();
                } else {
                    FrontDrain(CpSplit::DrainReason::TaskDone);
                }
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();
                if (curr_qid != static_cast<s32>(GfxQueueId)) {
                    asc_outstanding.fetch_sub(1, std::memory_order_release);
                    ++asc_done_seq;
                }

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }
        Common::FaultMonitor::CountCpActive(Common::FaultMonitor::SinceNs(t_active));
        if (split_enabled) {
            // CP split: the idle tail uses the scheduler, the caches and IrqC (back work). Same
            // order as below; submit_done is consumed here on the front.
            const bool flush_submit = submit_done;
            if (flush_submit) {
                submit_done = false;
            }
            if (g_post_order && split_task_guard) {
                // SHADPS4_CP_SPLIT_POST_ORDER: the same work, posted; it runs (and the guest sees
                // GPU idle) when the back reaches it. No stack captures: `this` | flush flag.
                auto* call = split->Reserve<CpSplit::CallPayload>(CpSplit::RecordType::Call);
                call->invoke = [](void* context) {
                    const auto bits = reinterpret_cast<uintptr_t>(context);
                    auto* lp = reinterpret_cast<Liverpool*>(bits & ~uintptr_t{1});
                    ++VideoCore::g_g1[10]; // q19
                    if (FenceAtGpu() == 2 && lp->rasterizer) {
                        VideoCore::G1Scope g1{0};
                        lp->rasterizer->FlushLazyFences();
                    }
                    if ((bits & 1) != 0 && lp->rasterizer) {
                        {
                            VideoCore::G1Scope g1{1};
                            lp->rasterizer->OnSubmit();
                        }
                        VideoCore::G1Scope g1{6};
                        lp->rasterizer->Flush();
                    }
                    {
                        VideoCore::G1Scope g1{7};
                        VideoCore::ProtectFence();
                    }
                    VideoCore::G1Scope g1{8};
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
                };
                static_assert(alignof(Liverpool) >= 2);
                call->context = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(this) |
                                                        (flush_submit ? 1 : 0));
                split->Commit();
                split->PostRunAhead(CpSplit::PostKind::Fence);
                split->PublishNow();
                split->MaybeLog();
                g_po.idle_posted.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            OnBack(
                [&] {
                    if (FenceAtGpu() == 2 && rasterizer) {
                        rasterizer->FlushLazyFences();
                    }
                    if (flush_submit && rasterizer) {
                        rasterizer->OnSubmit();
                        rasterizer->Flush();
                    }
                    VideoCore::ProtectFence();
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
                },
                CpSplit::DrainReason::IdleTail);
            continue;
        }
        if (FenceAtGpu() == 2 && rasterizer) {
            rasterizer->FlushLazyFences();
        }

        if (submit_done) {
            // NOTE: EndCapture() used to live here, which ended the RenderDoc capture at the first
            // completed submit. A title splits one displayed frame across several submits, so that
            // recorded an arbitrary batch rather than a frame: captures had wildly varying content
            // and any batch that missed the final composite presented a black image. The capture
            // now ends after present, in Presenter::Present.
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        VideoCore::ProtectFence();
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb, bool is_indirect) {
    FIBER_ENTER(ccb_task_name);
    // TASK_GUARD: covers the top-level CCB and CE IBs (which had no posted-write guard before).
    SplitSpanScope split_span{*this, &ccb,
                              is_indirect ? CpSplit::DrainReason::TaskGuardIb
                                          : CpSplit::DrainReason::TaskGuard};

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        if (g_reg_scan) [[unlikely]] {
            auto& st = t_reg_scan;
            const u32 nw = header->type3.NumWords();
            if (opcode == PM4ItOpcode::WriteConstRam) {
                ++st.ce_write;
                st.ce_write_dw += nw > 1 ? nw - 1 : 0;
            } else if (opcode == PM4ItOpcode::DumpConstRam) {
                ++st.ce_dump;
                st.ce_dump_dw += reinterpret_cast<const PM4DumpConstRam*>(header)->Size() / 4;
            } else if (opcode == PM4ItOpcode::LoadConstRam) {
                ++st.ce_load;
            } else if (opcode != PM4ItOpcode::Nop) {
                ++st.ce_other;
            }
        }
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            // CP split: a CP-side guest-memory write, so it runs on the back (fault identity).
            // A Call record: synchronous at every depth, so reading the front's heap is sound.
            // SPLIT-TODO(run-ahead): copy the heap slice into an owned record instead of reading
            // the front's constants_heap from the back.
            OnBack([&] {
                memcpy(dump_const->Address<void*>(),
                       cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            }, CpSplit::DrainReason::CallDumpConstRam);
            // A CP-side guest memory write: later draws may read it (submit_epoch.h).
            BumpEpoch(PM4ItOpcode::DumpConstRam);
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCeUpdate(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, true);
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}


Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);
    // Everything the guest wrote before this submission is now due (submit_epoch.h). An indirect
    // buffer is part of its parent submission, so it does not reopen the window.
    const bool is_indirect_gfx = std::exchange(starting_indirect_gfx, false);
    if (!is_indirect_gfx) {
        BumpEpoch(PM4ItOpcode::Nop);
        if (AmdGpu::NativeRec::g_segb != 0) [[unlikely]] {
            SegB::NoteSubmit();
        }
    }
    if (!is_indirect_gfx && split_enabled && (g_a1 != 0 || g_a2 != 0)) [[unlikely]] {
        // Architecture A step A1: the game's top-level command buffer is the unit of work.
        const u32 snap_words = (g_a1 == 2 || g_a2 != 0) ? CpSplit::DcbSnapWords : 0u;
        auto* begin = split->Reserve<CpSplit::DcbBeginPayload>(
            CpSplit::RecordType::DcbBegin,
            static_cast<u32>(sizeof(CpSplit::DcbBeginPayload) + snap_words * sizeof(u32)));
        begin->addr = reinterpret_cast<u64>(dcb.data());
        begin->dwords = static_cast<u32>(dcb.size());
        begin->has_ccb = ccb.empty() ? 0u : 1u;
        begin->snap_words = snap_words;
        begin->index = ++g_a1_front_index;
        g_a2s.front_begin_pos = split->LastRecordPos();
        if (snap_words != 0) {
            // The front's live register file at the DCB start = the back's mirror once it has
            // applied every record before this one (verified on the back).
            u32* out = reinterpret_cast<u32*>(begin + 1);
            for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
                std::memcpy(out, &regs.reg_array[off], size_t{n} * sizeof(u32));
                out += n;
            }
        }
        split->Commit();
        if (g_a2 != 0) {
            // Hand the DCB to the workers now - they follow the front record by record. The
            // entry is written before the DcbBegin record can be published (nothing publishes
            // between the Commit above and here).
            const u32 idx = g_a1_front_index;
            auto& e = g_a2s.e[idx % A2Slots];
            if (e.state.load(std::memory_order_acquire) != 0) {
                // The back still holds idx - A2Slots (it releases at that DcbEnd, published).
                split->front_waiting.fetch_add(1, std::memory_order_acq_rel);
                split->RingReaderBell();
                const u64 sw0 = __rdtsc();
                for (u32 spins = 0; e.state.load(std::memory_order_acquire) != 0; ++spins) {
                    spins < 4096 ? _mm_pause() : std::this_thread::yield();
                }
                const u64 swt = __rdtsc() - sw0;
                split->slot_wait_ticks += swt;
                Common::FaultMonitor::front_wait_ticks.fetch_add(swt, std::memory_order_relaxed);
                split->front_waiting.fetch_sub(1, std::memory_order_acq_rel);
            }
            e.begin_pos.store(g_a2s.front_begin_pos, std::memory_order_relaxed);
            e.refused = 0;
            e.ready.store(0, std::memory_order_relaxed);
            e.progress.store(0, std::memory_order_relaxed);
            e.walked.store(0, std::memory_order_relaxed);
            e.back_wait.store(0, std::memory_order_relaxed);
            e.cancel.store(0, std::memory_order_relaxed);
            e.index.store(idx, std::memory_order_release);
            e.state.store(1, std::memory_order_release);
            g_a2s.begun.store(idx, std::memory_order_release);
            if (g_claim_wake) {
                g_a2s.bells.Ring(idx);
            } else {
                g_a2s.begun.notify_all();
            }
            Common::FaultMonitor::SampleEmuPlacement(5);
        }
    }
    if (g_reg_scan && !is_indirect_gfx) [[unlikely]] {
        auto& st = t_reg_scan;
        auto& sr = t_reg_scan_regs;
        if (!sr) {
            sr = std::make_unique<AmdGpu::Regs>();
        }
        *sr = regs; // untimed: a chained worker gets this from the previous one
        const u64 t0 = __rdtsc();
        if (st.win_tsc == 0) {
            st.win_tsc = t0;
        }
        st.cur_cut_s = st.cur_cut_sd = false;
        RegScanRun(dcb, *sr, st, 0);
        st.cyc += __rdtsc() - t0;
        st.dcb_full_s += st.cur_cut_s ? 0 : 1;
        st.dcb_full_sd += st.cur_cut_sd ? 0 : 1;
        ++st.dcbs;
        st.ccbs += ccb.empty() ? 0 : 1;
        t_reg_scan_gfx_open = true;
    }
    if (g_lean != 0 && !is_indirect_gfx && split_enabled && rasterizer) [[unlikely]] {
        if (g_lean == 2) {
            if (split_task_guard) {
                // The head pass (and the worker) read [0, cut) before SplitSpanScope guards the
                // tail: guard the whole buffer first.
                const auto lo = reinterpret_cast<VAddr>(dcb.data());
                split->GuardSpan(lo, lo + dcb.size_bytes(), CpSplit::DrainReason::TaskGuard);
            }
            if (g_front_cost) [[unlikely]] {
                FrontCostTick(FrontCostLeanHead);
            }
            if (g_lean_xl.on) {
                LeanPlaceSample(g_lean_xl.front_p, g_lean_xl.front_e);
            }
            const auto l2 =
                LeanFrontBeginL2(dcb, ccb, EmulatorSettings.IsVkGuestMarkersEnabled());
            if (l2.cut != 0) {
                // The head span is a worker's: one fresh gfx dispatch run if it had run-ending
                // packets (runs compare only for equality), and no NativeRec run continues
                // across it. The front parses the tail from the state the head pass left.
                if (l2.run_bumps != 0 && g_dispatch_sync.order) {
                    g_dispatch_sync.run[64] = ++g_dispatch_sync.run_counter;
                }
                g_nr_run.active = false;
                dcb = dcb.subspan(l2.cut);
            }
        } else {
            LeanFrontBegin(dcb, ccb, rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled());
        }
    }
    SplitSpanScope split_span{*this, &dcb,
                              is_indirect_gfx ? CpSplit::DrainReason::TaskGuardIb
                                              : CpSplit::DrainReason::TaskGuard};

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    const bool asc_yield = AscYield();
    while (!dcb.empty()) {
        if (g_lean_front.job != nullptr) [[unlikely]] {
            split->canon = nullptr; // nothing between packets (host commands, ASC) is recorded
        }
        if ((g_xq || g_asc_early) && split_enabled) [[unlikely]] {
            split->note_queue = 0;
        }
        if (g_asc_early && split_enabled) [[unlikely]] {
            split->asc_tag = 0;
        }
        ProcessCommands();

        if (asc_yield) {
            // A new ASC submission, or one finished while more are queued (the round-robin
            // returns here after each finished submission). A blocked ASC task finishes nothing,
            // so this yields at most once per ASC submit and once per ASC completion.
            const u64 seq = asc_submit_seq.load(std::memory_order_acquire);
            const bool next_queued = asc_done_seq != gfx_seen_asc_done &&
                                     asc_outstanding.load(std::memory_order_acquire) != 0;
            if (seq != gfx_seen_asc_seq || next_queued) {
                gfx_seen_asc_seq = seq;
                gfx_seen_asc_done = asc_done_seq;
                ++g_asc_start.yields;
                YIELD_GFX();
            }
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            if (g_front_cost) [[unlikely]] {
                FrontCostTick(static_cast<u32>(opcode));
            }
            if (g_lean_front.recording &&
                g_lean_front.base == reinterpret_cast<const u32*>(base_addr)) [[unlikely]] {
                auto& lj = *g_lean_front.job;
                const u32 off = static_cast<u32>(dcb.data() - g_lean_front.base);
                lj.front_marks.emplace_back(off, static_cast<u32>(lj.front_canon.size()));
                if (off >= lj.cut_dw) {
                    g_lean_front.recording = false; // the front parses this and the rest
                } else {
                    split->canon = &lj.front_canon;
                }
            }
            g_dispatch_sync.Note(64, opcode, header);
            if (AmdGpu::NativeRec::g_segb != 0) [[unlikely]] {
                SegB::NotePacket(opcode, header, count, guest_markers_enabled, regs);
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                // Flip IRQ and guest markers are back work; the body is in ExecuteGfxPacket.
                if (split_enabled) {
                    if (NopHasBackEffect(header, guest_markers_enabled)) {
                        const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                        if (g_post_flip && split_task_guard &&
                            nop->data_block[0] == PM4CmdNop::PayloadType::PatchedFlip) {
                            // SHADPS4_CP_SPLIT_POST_FLIP: no guest-memory write; the IRQ is
                            // raised on the back in stream order, as before.
                            const u32 dwords = header->type3.NumWords() + 1;
                            void* copy = split->Reserve(
                                CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                guest_markers_enabled ? CpSplit::GuestMarkers : 0);
                            std::memcpy(copy, header, dwords * sizeof(u32));
                            split->Commit();
                            split->PostRunAhead(CpSplit::PostKind::Fence);
                            split->MaybeLog();
                            g_pwf.flips_posted.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                        RunGfxPacket(header, guest_markers_enabled,
                                     CpSplit::DrainReason::PacketOther);
                    }
                    break;
                }
                ExecuteGfxPacket(header, guest_markers_enabled);
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                ++context_change_epoch;
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) {
                    ++regs_generation;
                    ++regs_generation_ctx;
                }
                if (split_enabled) [[unlikely]] {
                    split->EmitClearState();
                }
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                const auto sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, sz);
                JournalRegs(reg_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                const auto ctx_sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, ctx_sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                if (ViewportMemoEnabled() &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, ctx_sz) != 0) {
                    ++context_change_epoch;
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, ctx_sz);
                JournalRegs(reg_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                JournalExtent(reg_addr);
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    JournalRegs(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                    if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                    if (PgmPrefetchEnabled()) {
                        PrefetchProgramsOnWrite(regs, set_data->reg_offset, count - 1);
                    }
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto uc_addr = Regs::UconfigRegWordOffset + set_data->reg_offset;
                const auto uc_sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[uc_addr], header + 2, uc_sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                std::memcpy(&regs.reg_array[uc_addr], header + 2, uc_sz);
                JournalRegs(uc_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                JournalField(regs.index_buffer_type);
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                JournalField(regs.max_index_size);
                JournalField(regs.index_base_address);
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndex2", host_markers_enabled, {.is_indexed = true});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                JournalField(regs.max_index_size);
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexOffset2", host_markers_enabled,
                             {.is_indexed = true, .index_offset = draw_index_off->index_offset});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexAuto", host_markers_enabled, {});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndirect", host_markers_enabled,
                             {.is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = u32(stride),
                              .max_count = 1});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndirectMulti", host_markers_enabled,
                             {.is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_indirect->stride,
                              .max_count = draw_indirect->count});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirect", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = u32(stride),
                              .max_count = 1});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirectMulti", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_index_indirect->stride,
                              .max_count = draw_index_indirect->count});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirectCountMulti", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_index_indirect->stride,
                              .max_count = draw_index_indirect->count,
                              .count_address = draw_index_indirect->count_indirect_enable.Value()
                                                   ? VAddr(draw_index_indirect->count_addr)
                                                   : VAddr{0}});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (split_enabled) {
                    if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                        EmitDispatch(header, "DispatchDirect", host_markers_enabled, false, 0, 0,
                                     0, GfxQueueId);
                    }
                    break;
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (split_enabled) {
                    if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                        EmitDispatch(header, "DispatchIndirect", host_markers_enabled, true,
                                     indirect_args_addr, offset, size, GfxQueueId);
                    }
                    break;
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                JournalField(regs.num_instances);
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                JournalField(regs.index_base_address);
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                JournalField(regs.num_indices);
                // r4 (upstream #5278): INDEX_BUFFER_SIZE also sets VGT_DMA_MAX_SIZE, the capacity an
                // indexed INDIRECT draw reads from INDEX_BASE; direct draws overwrite num_indices.
                regs.max_index_size = index_size->num_indices;
                JournalField(regs.max_index_size);
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    JournalField(regs.cp_strmout_cntl);
                } else if (!split_enabled) {
                    ExecuteGfxPacket(header, guest_markers_enabled);
                } else if (event->event_index.Value() == EventIndex::ZpassDone &&
                           event->event_type.Value() == EventType::PixelPipeStatDump) {
                    // Occlusion counter write into guest memory: back work.
                    if (g_post_order && split_task_guard) {
                        // SHADPS4_CP_SPLIT_POST_ORDER: posted, its results noted in the write log.
                        const VAddr dst = event->Address<VAddr>();
                        const u64 bytes = u64{num_counter_pairs} * 16;
                        const auto lo = reinterpret_cast<VAddr>(dcb.data());
                        const VAddr hi = lo + dcb.size_bytes();
                        if (!(dst < hi && lo < dst + bytes) && !SplitSpansOverlap(dst, dst + bytes)) {
                            CommitPacketCopy(*split, header,
                                             guest_markers_enabled ? u16{CpSplit::GuestMarkers}
                                                                   : u16{0});
                            split->NotePostedWrite(dst, bytes, false, 0, false, 5);
                            split->PostRunAhead(CpSplit::PostKind::GfxPacket);
                            split->MaybeLog();
                            g_po.zpass_posted.fetch_add(1, std::memory_order_relaxed);
                            break;
                        }
                        g_po.zpass_guarded.fetch_add(1, std::memory_order_relaxed);
                    }
                    RunGfxPacket(header, guest_markers_enabled,
                                 CpSplit::DrainReason::PacketOther);
                }
                break;
            }
            // Fences, DMA, CP memory writes and the CP sync barrier: back work, bodies in
            // ExecuteGfxPacket. Synchronous at every depth, except DMA_DATA under POST_DMA.
            case PM4ItOpcode::EventWriteEos:
            case PM4ItOpcode::EventWriteEop: {
                if (split_post_fence) [[unlikely]] {
                    VAddr label = 0;
                    if (opcode == PM4ItOpcode::EventWriteEop) {
                        label = reinterpret_cast<VAddr>(
                            reinterpret_cast<const PM4CmdEventWriteEop*>(header)->Address<u32>());
                    } else if (const auto* eos =
                                   reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                               eos->command != PM4CmdEventWriteEos::Command::GdsStore) {
                        label = eos->Address<VAddr>();
                    }
                    if (label != 0) {
                        RunFence(header, dcb, label,
                                 guest_markers_enabled ? u16{CpSplit::GuestMarkers} : u16{0},
                                 CpSplit::DrainReason::PacketFence);
                        break;
                    }
                }
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketFence);
                break;
            }
            case PM4ItOpcode::DmaData: {
                RunGfxDma(header, guest_markers_enabled, dcb);
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* wd = reinterpret_cast<const PM4CmdWriteData*>(header);
                if (g_post_wdata && split_task_guard &&
                    (wd->dst_sel.Value() == 2 || wd->dst_sel.Value() == 5) &&
                    !wd->wr_one_addr.Value() && header->type3.count.Value() > 2) {
                    // SHADPS4_CP_SPLIT_POST_WDATA: posted like a POST_DMA write; synchronous
                    // only when it writes into the rest of this buffer or an active span.
                    const VAddr dst = wd->Address<VAddr>();
                    const u64 bytes = (header->type3.count.Value() - 2) * 4ull;
                    const auto lo = reinterpret_cast<VAddr>(dcb.data());
                    const VAddr hi = lo + dcb.size_bytes();
                    if ((dst < hi && lo < dst + bytes) || SplitSpansOverlap(dst, dst + bytes)) {
                        g_pwf.wdata_guarded.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        const u32 dwords = header->type3.NumWords() + 1;
                        void* copy = split->Reserve(
                            CpSplit::RecordType::Packet, dwords * sizeof(u32),
                            guest_markers_enabled ? CpSplit::GuestMarkers : 0);
                        std::memcpy(copy, header, dwords * sizeof(u32));
                        split->Commit();
                        split->NotePostedWrite(dst, bytes, true, wd->data[0], false, 2);
                        split->PostRunAhead(CpSplit::PostKind::GfxPacket);
                        split->MaybeLog();
                        g_pwf.wdata_posted.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
                RunGfxPacket(header, guest_markers_enabled,
                             CpSplit::DrainReason::PacketWriteData);
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (g_post_order && split_task_guard) {
                    // SHADPS4_CP_SPLIT_POST_ORDER: a barrier on the back; nothing for the front.
                    CommitPacketCopy(*split, header,
                                     guest_markers_enabled ? u16{CpSplit::GuestMarkers} : u16{0});
                    split->PostRunAhead(CpSplit::PostKind::Fence);
                    split->MaybeLog();
                    g_po.pfp_posted.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketOther);
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (!mem_semaphore->IsSignaling()) {
                    // The front reads guest memory the back may still write: drain first.
                    FrontDrain(CpSplit::DrainReason::MemSemaphore);
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                }
                // Signal() or Decrement(), in ExecuteGfxPacket.
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::MemSemaphore);
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                // The flag is written by the guest CPU, not the back: no drain needed.
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (g_xq && split_enabled && wait_reg_mem->mem_space.Value() ==
                                                 PM4CmdWaitRegMem::MemSpace::Memory) [[unlikely]] {
                    XqNoteWait(*split, 0, reinterpret_cast<VAddr>(wait_addr),
                               g_fence_forward && g_fwd[0].next_packet ==
                                                      reinterpret_cast<const u32*>(header));
                }
                if (g_fence_forward && split_enabled && ForwardedWait(0, header)) {
                    if (g_wait_barrier >= 2) {
                        PostWaitBarrier(1);
                    }
                    break; // SHADPS4_FENCE_FORWARD: passes in stream order on the fence's value
                }
                const bool vo_wait = vo_port->IsVoLabel(wait_addr) &&
                                     num_submits == mapped_queues[GfxQueueId].submits.size();
                const WrmPre pre = split_enabled && split_task_guard
                                       ? WrmPartial(*split, wait_reg_mem, vo_wait, regs.reg_array)
                                       : WrmPre::Full;
                if (pre == WrmPre::PassedFwdFence) {
                    if (g_wait_barrier >= 1) {
                        PostWaitBarrier(0);
                    }
                    break; // SHADPS4_CP_SPLIT_POST_ORDER: forwarded on an end-of-pipe label
                }
                if (pre == WrmPre::Passed) {
                    break; // SHADPS4_WRM_PARTIAL: passed after the label's own last write
                }
                if (pre == WrmPre::VoWait) {
                    // SHADPS4_WRM_PARTIAL: the presenter writes it; the backlog is not drained.
                    const auto vo_start = Common::FaultMonitor::Now();
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    Common::FaultMonitor::CountVoLabelWait(
                        Common::FaultMonitor::SinceNs(vo_start));
                    break;
                }
                // CP split: the back may still owe writes this wait reads (inventory 0.6), so
                // drain before testing. At depth 0 this only flushes register records.
                FrontDrain(CpSplit::DrainReason::WaitRegMem);
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    const auto vo_start = Common::FaultMonitor::Now();
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    Common::FaultMonitor::CountVoLabelWait(
                        Common::FaultMonitor::SinceNs(vo_start));
                    break;
                }
                const auto wrm_start = Common::FaultMonitor::Now();
                const u64 wrm_ns0 = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
                bool wrm_waited = false;
                if (FenceAtGpu() == 2 && rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                    OnBack([&] { rasterizer->FlushLazyFences(); },
                           CpSplit::DrainReason::WaitRegMem);
                }
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    wrm_waited = true;
                    YIELD_GFX();
                }
                if (g_cp_wait.enabled) {
                    if (wrm_waited) {
                        g_cp_wait.CountWait(true, wait_reg_mem->Address<u64*>(),
                                            CpWaitStats::Now() - wrm_ns0);
                    } else {
                        ++g_cp_wait.gfx_wrm_instant;
                    }
                }
                Common::FaultMonitor::CountWaitRegMem(Common::FaultMonitor::SinceNs(wrm_start));
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                if (split_task_guard < 2 &&
                    (split_post_dma || (split_post_asc & CpSplit::PostAscDma) ||
                     (split_post_asc & CpSplit::PostAscWriteData))) [[unlikely]] {
                    // A posted DMA may still be writing the IB about to be parsed. TASK_GUARD >= 2:
                    // the IB's own entry check (SplitSpanScope) does this for its span only.
                    split->SyncIfPostedPackets(CpSplit::DrainReason::IbStart);
                }
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                starting_indirect_gfx = true;
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                if (split_post_dma || (split_post_asc & CpSplit::PostAscWriteData)) [[unlikely]] {
                    // A posted DMA inside the IB may patch the rest of the parent level.
                    // TASK_GUARD 3: only a write into that rest drains (`dcb` still starts at
                    // this packet); higher levels check theirs at their own IB end.
                    if (split_task_guard >= 3) {
                        SplitGuardSpan(dcb, CpSplit::DrainReason::IbEnd);
                    } else {
                        split->SyncIfPostedPackets(CpSplit::DrainReason::IbEnd);
                    }
                }
                if (IbTaskFree()) {
                    task.handle.destroy();
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    if (split_task_guard) [[unlikely]] {
                        // The DE resumes the CE directly, not through ProcessLoop: check the
                        // CE's unparsed rest against this task's posted writes.
                        SplitGuardActive();
                    }
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                // Reads guest memory the back may write.
                FrontDrain(CpSplit::DrainReason::CondExec);
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            if (!KeepsSubmitEpoch(opcode)) {
                BumpEpoch(opcode);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            if (split_task_guard) [[unlikely]] {
                SplitGuardActive(); // As at WAIT_ON_CE_COUNTER.
            }
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    if (g_lean != 0 && !is_indirect_gfx && g_lean_front.job != nullptr) [[unlikely]] {
        split->canon = nullptr;
        auto& lj = *g_lean_front.job;
        g_lean_front = {};
        if (lj.done_mask.fetch_or(1, std::memory_order_acq_rel) == 2) {
            LeanCompare(lj);
        }
    }
    if (g_reg_scan && !is_indirect_gfx && t_reg_scan_regs) [[unlikely]] {
        auto& st = t_reg_scan;
        t_reg_scan_gfx_open = false;
        u32 bad = 0;
        for (const auto& [off, n] : CpSplit::DcbSnapWindows) {
            for (u32 i = 0; i < n; ++i) {
                if (t_reg_scan_regs->reg_array[off + i] != regs.reg_array[off + i]) {
                    if (++bad <= 4 && st.bad < 8) {
                        LOG_WARNING(Render, "RegScan verify: word {:#x} scan {:#x} front {:#x}",
                                    off + i, t_reg_scan_regs->reg_array[off + i],
                                    regs.reg_array[off + i]);
                    }
                }
            }
        }
        ++st.cmp;
        st.bad += bad != 0;
        st.bad_words += bad;
        const u64 now = __rdtsc();
        if (now - st.win_tsc > 35'000'000'000ull) {
            RegScanLog(st, now);
        }
    }
    if (!is_indirect_gfx && split_enabled && (g_a1 != 0 || g_a2 != 0)) [[unlikely]] {
        auto* end = split->Reserve<CpSplit::DcbEndPayload>(CpSplit::RecordType::DcbEnd);
        end->index = g_a1_front_index;
        end->pad = 0;
        split->Commit();
        split->PublishNow();
        split->dcb_ends_published.fetch_add(1, std::memory_order_release);
        if (g_a2 != 0) {
            g_a2s.published.store(g_a1_front_index, std::memory_order_release);
        }
    }
    if (g_front_cost && !is_indirect_gfx) [[unlikely]] {
        FrontCostTick(FrontCostBetween);
    }
    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    // SHADPS4_ASC_EARLY: this level's caller state, restored when it ends (a nested IB level
    // returns into its parent's tagged parse; the top level returns to ProcessLoop untagged).
    const u8 asc_tag_saved = split_enabled ? split->asc_tag : u8{0};
    // CP split: ASC packets are parsed on the front like the DCB. Their back work runs through
    // Call records (OnBack), including the pipeline lookup inside DispatchDirect/Indirect. A
    // Call is a full drain at every depth (the front stays parked until it returns), so this
    // stays sound with run-ahead; each one costs a round trip. With POST_ASC (run-ahead),
    // dispatches are owned Dispatch records with the lookup on the front (EmitDispatch), and
    // DMA_DATA (and optionally WRITE_DATA) owned Packet records (RunAscPacket), all posted.
    // RELEASE_MEM, MEM_SEMAPHORE and WAIT_REG_MEM stay synchronous.
    if constexpr (!is_indirect) {
        BumpEpoch(PM4ItOpcode::Nop);
        if (AscStartStats()) [[unlikely]] {
            u64 submit_ns = 0;
            {
                auto& mq = mapped_queues[vqid + 1];
                std::scoped_lock lock{mq.m_access};
                auto& times = asc_submit_ns[vqid + 1];
                if (!times.empty()) {
                    submit_ns = times.front();
                    times.pop_front();
                }
            }
            if (submit_ns != 0) {
                NoteAscStart(submit_ns, vqid + 1);
            }
        }
        if (split_run_ahead && AscRingLatency()) [[unlikely]] {
            AscRingSnap submit{};
            bool have = false;
            {
                std::scoped_lock lock{g_asc_ring.submit_mutex};
                auto& stamps = g_asc_ring.submits[(vqid + 1) % AscRingQueues];
                if (!stamps.empty()) {
                    submit = stamps.front();
                    stamps.pop_front();
                    have = true;
                }
            }
            if (have) {
                EmitAscRingDiag(*split, AscRingBegin, vqid + 1, &submit, __rdtsc());
            }
        }
    }
    SplitSpanScope split_span{*this, &acb,
                              is_indirect ? CpSplit::DrainReason::TaskGuardIb
                                          : CpSplit::DrainReason::TaskGuard};
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();
        if (g_asc_early && split_enabled) [[unlikely]] {
            split->asc_tag = 1;
        }
        if ((g_xq || g_asc_early) && split_enabled) [[unlikely]] {
            split->note_queue = static_cast<u8>(std::min<u32>(vqid + 1, XqQueues - 1));
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                AdvanceAscReadPtr(queue, acb.size());
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                AdvanceAscReadPtr(queue, next_dw_off);
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        if (g_front_cost) [[unlikely]] {
            FrontCostTick(0x100u | static_cast<u32>(opcode));
        }
        g_dispatch_sync.Note(vqid & 63, opcode, header);

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            if (split_task_guard < 2 &&
                (split_post_dma || (split_post_asc & CpSplit::PostAscDma) ||
                 (split_post_asc & CpSplit::PostAscWriteData))) [[unlikely]] {
                // A posted DMA / WRITE_DATA may still be writing the IB about to be parsed.
                // TASK_GUARD >= 2: the IB's own entry check (SplitSpanScope) does this.
                split->SyncIfPostedPackets(CpSplit::DrainReason::AscIbStart);
            }
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            if ((split_post_asc & CpSplit::PostAscDma) ||
                (split_post_asc & CpSplit::PostAscWriteData)) [[unlikely]] {
                // A posted DMA / WRITE_DATA inside the IB may patch the rest of the parent level.
                // TASK_GUARD 3: only a write into that rest drains.
                if (split_task_guard >= 3) {
                    SplitGuardSpan(acb, CpSplit::DrainReason::IbEnd);
                } else {
                    split->SyncIfPostedPackets(CpSplit::DrainReason::IbEnd);
                }
            }
            if (IbTaskFree()) {
                task.handle.destroy();
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (split_post_asc & CpSplit::PostAscDma) [[unlikely]] {
                // POST_ASC: the back runs ExecuteGfxPacket's DMA_DATA body, which is this one
                // minus the indirect-patch detection below (front-only bookkeeping, kept here).
                const bool src_mem = dma_data->src_sel == DmaDataSrc::Memory ||
                                     dma_data->src_sel == DmaDataSrc::MemoryUsingL2;
                const bool dst_mem = dma_data->dst_sel == DmaDataDst::Memory ||
                                     dma_data->dst_sel == DmaDataDst::MemoryUsingL2;
                if (src_mem && dst_mem) {
                    const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                    const auto* patched =
                        reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                    if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                        dma_data->NumBytes() == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                        patched->type == 3 &&
                        patched->type3.opcode == PM4ItOpcode::DispatchDirect) {
                        indirect_patches.emplace_back(patched, dma_data->SrcAddress<VAddr>());
                        break;
                    }
                }
                RunAscPacket(header, acb, dst_mem ? dma_data->DstAddress<VAddr>() : VAddr{0},
                             dst_mem ? dma_data->NumBytes() : 0u,
                             CpSplit::DrainReason::AscDmaGuard);
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                OnBack(
                    [&] {
                        rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                               dma_data->data, true);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                OnBack(
                    [&] {
                        rasterizer->CopyBuffer(dma_data->dst_addr_lo,
                                               dma_data->SrcAddress<VAddr>(),
                                               dma_data->NumBytes(), true, false);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                OnBack(
                    [&] {
                        rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->NumBytes(), dma_data->data, false);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                OnBack(
                    [&] {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->src_addr_lo, dma_data->NumBytes(),
                                               false, true);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    OnBack(
                        [&] {
                            rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                        },
                        CpSplit::DrainReason::CallAscDma);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                if (g_reg_scan) [[unlikely]] {
                    auto& st = t_reg_scan;
                    const u32 dw = u32(set_size / sizeof(u32));
                    ++st.asc_sh;
                    st.asc_sh_dw += dw;
                    if (t_reg_scan_gfx_open) {
                        ++st.asc_sh_in_gfx;
                        st.asc_sh_in_gfx_dw += dw;
                    }
                    st.asc_sh_min = std::min<u32>(st.asc_sh_min, set_data->reg_offset);
                    st.asc_sh_max = std::max<u32>(st.asc_sh_max, set_data->reg_offset + dw);
                }
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                JournalRegs(Regs::ShRegWordOffset + set_data->reg_offset,
                            u32(set_size / sizeof(u32)));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if ((split_post_asc & CpSplit::PostAscDispatch) && rasterizer) [[unlikely]] {
                    // As the Call below: the queue's cs_state as it is (dims not updated), no
                    // host marker. The back reads the patched dims in stream order.
                    EmitDispatch(header, "DispatchIndirect", false, true, it->indirect_addr, 0,
                                 u32(size), vqid + 1);
                    break;
                }
                OnBack([&] {
                    if (VideoCore::g_no_particles) {
                        VideoCore::NoParticlesNote(false); // q40
                    } else {
                        rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                    }
                }, CpSplit::DrainReason::CallAscDispatchPatched);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (split_post_asc & CpSplit::PostAscDispatch) [[unlikely]] {
                    EmitDispatch(header, "DispatchDirect", host_markers_enabled, false, 0, 0, 0,
                                 vqid + 1);
                    break;
                }
                OnBack(
                    [&] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        if (host_markers_enabled) {
                            rasterizer->ScopeMarkerBegin(
                                fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                            rasterizer->DispatchDirect();
                            rasterizer->ScopeMarkerEnd();
                        } else if (VideoCore::g_no_particles) {
                            VideoCore::NoParticlesNote(false); // q40
                        } else {
                            rasterizer->DispatchDirect();
                        }
                    },
                    CpSplit::DrainReason::CallAscDispatch);
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (split_post_asc & CpSplit::PostAscDispatch) [[unlikely]] {
                    EmitDispatch(header, "DispatchIndirect", host_markers_enabled, true,
                                 ib_address, 0, u32(size), vqid + 1);
                    break;
                }
                OnBack(
                    [&] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        if (host_markers_enabled) {
                            rasterizer->ScopeMarkerBegin(
                                fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                            rasterizer->DispatchIndirect(ib_address, 0, size);
                            rasterizer->ScopeMarkerEnd();
                        } else if (VideoCore::g_no_particles) {
                            VideoCore::NoParticlesNote(false); // q40
                        } else {
                            rasterizer->DispatchIndirect(ib_address, 0, size);
                        }
                    },
                    CpSplit::DrainReason::CallAscDispatchIndirect);
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (split_post_asc & CpSplit::PostAscWriteData) [[unlikely]] {
                // POST_ASC: ExecuteGfxPacket's WRITE_DATA body, which adds SplitHazard::
                // NoteWrite (off: POST_ASC is refused with it) and g_cp_wait.NoteWrite (this bit
                // is refused with CP_WAIT_STATS). Readers of the label (WAIT_REG_MEM, semaphore
                // waits, task done) drain first; the guest CPU sees it later, never earlier.
                RunAscPacket(header, acb,
                             reinterpret_cast<VAddr>(write_data->Address<const void*>()),
                             data_size, CpSplit::DrainReason::AscWriteDataGuard);
                break;
            }
            OnBack(
                [&] {
                    g_cp_wait.NoteWrite(write_data->Address<void*>());
                    VideoCore::ProtectFence();
                    if (!write_data->wr_one_addr.Value()) {
                        std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
                    } else {
                        UNREACHABLE();
                    }
                },
                CpSplit::DrainReason::CallAscWriteData);
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            // Stays synchronous under POST_ASC (a Call waits for everything posted). Posting the
            // signal would keep stream order, and its readers drain first, but it is not
            // measured to matter yet (call_asc_sem says). SPLIT-TODO(run-ahead).
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                OnBack([&] { mem_semaphore->Signal(); }, CpSplit::DrainReason::CallAscSemaphore);
            } else {
                FrontDrain(CpSplit::DrainReason::MemSemaphore);
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                OnBack([&] { mem_semaphore->Decrement(); },
                       CpSplit::DrainReason::CallAscSemaphore);
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (g_xq && split_enabled && wait_reg_mem->mem_space.Value() ==
                                             PM4CmdWaitRegMem::MemSpace::Memory) [[unlikely]] {
                XqNoteWait(*split, vqid + 1, reinterpret_cast<VAddr>(wait_reg_mem->Address<u32*>()),
                           g_fence_forward && g_fwd[1 + (queue.pipe_id & 0xff)].next_packet ==
                                                  reinterpret_cast<const u32*>(header));
            }
            if (g_fence_forward && split_enabled && ForwardedWait(1 + (queue.pipe_id & 0xff), header)) {
                if (g_wait_barrier >= 2) {
                    PostWaitBarrier(1);
                }
                break; // SHADPS4_FENCE_FORWARD (ASC)
            }
            if (split_enabled && split_task_guard) {
                // l4a6 (review #2): the label's newest logged writer, before WrmPartial prunes.
                const int writer_q =
                    g_asc_early && split->early_items
                        ? split->NewestWriterQueue(
                              reinterpret_cast<VAddr>(wait_reg_mem->Address<u32*>()))
                        : -1;
                const WrmPre pre = WrmPartial(*split, wait_reg_mem, false, regs.reg_array);
                const bool fwd_barrier = pre == WrmPre::PassedFwdFence && g_wait_barrier >= 1;
                if (fwd_barrier) {
                    PostWaitBarrier(0);
                }
                if (!fwd_barrier && writer_q >= 0 && u32(writer_q) != std::min<u32>(vqid + 1, XqQueues - 1) &&
                    (pre == WrmPre::Passed || pre == WrmPre::PassedFwdFence)) [[unlikely]] {
                    // Another queue wrote the label: a tagged Call keeps the ASC records after
                    // this wait behind that writer in ring order (early service blocks on it).
                    g_asc_early_stats.cross_waits.fetch_add(1, std::memory_order_relaxed);
                    PostWaitBarrier(0);
                }
                if (pre == WrmPre::Passed || pre == WrmPre::PassedFwdFence) {
                    break; // SHADPS4_WRM_PARTIAL / POST_ORDER (ASC)
                }
            }
            const auto wrm_start = Common::FaultMonitor::Now();
            const u64 wrm_ns0 = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
            bool wrm_waited = false;
            FrontDrain(CpSplit::DrainReason::WaitRegMem);
            if (FenceAtGpu() == 2 && rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                OnBack([&] { rasterizer->FlushLazyFences(); }, CpSplit::DrainReason::WaitRegMem);
            }
            while (!wait_reg_mem->Test(regs.reg_array)) {
                wrm_waited = true;
                YIELD_ASC(vqid);
            }
            if (g_cp_wait.enabled && wrm_waited) {
                g_cp_wait.CountWait(false, wait_reg_mem->Address<u64*>(),
                                    CpWaitStats::Now() - wrm_ns0);
            }
            Common::FaultMonitor::CountWaitRegMem(Common::FaultMonitor::SinceNs(wrm_start));
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (g_xq) [[unlikely]] {
                const u32 qi = std::min<u32>(vqid + 1, XqQueues - 1);
                g_xq_c.segs_xwait[qi].fetch_add(g_xq_c.seg_xwait[qi].exchange(0),
                                                std::memory_order_relaxed);
                g_xq_c.label_addr[qi].store(release_mem->Address<VAddr>(),
                                            std::memory_order_relaxed);
            }
            if (split_post_fence && release_mem->data_sel != DataSelect::GdsMemStore)
                [[unlikely]] {
                ASSERT(queue.pipe_id <= 0xff);
                t_fwd_asc_allowed = is_indirect;
                RunFence(header, acb, release_mem->Address<VAddr>(),
                         static_cast<u16>(CpSplit::AscReleaseMem | (queue.pipe_id << 8)),
                         CpSplit::DrainReason::CallAscRelease);
                break;
            }
            // Synchronous under POST_ASC alone: a fence + IRQ (and g_cp_wait's label set).
            OnBack([&] { ExecuteAscReleaseMem(release_mem, queue.pipe_id); },
                   CpSplit::DrainReason::CallAscRelease);
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        if (!KeepsSubmitEpoch(opcode)) {
            BumpEpoch(opcode);
        }
        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            static std::atomic<u32> rp_logged{0};
            if (rp_logged.load(std::memory_order_relaxed) < 8 &&
                rp_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_INFO(Render, "POST_ORDER: ASC vqid {} read pointer at {:#x}, ring {:#x}", vqid,
                         reinterpret_cast<VAddr>(queue.read_addr), queue.map_addr);
            }
            AdvanceAscReadPtr(queue, next_dw_off);
        }
    }

    if (split_enabled) {
        split->asc_tag = asc_tag_saved;
        if (g_asc_early && (g_asc_service & 1)) {
            split->PublishNow(); // m2: the end of an ASC submission
        }
    }
    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    Common::FaultMonitor::CountGuestSubmit(true);
    Common::SampleThreadPlacement("guest-submit-gfx", 4);
    if (AscRingLatency()) [[unlikely]] {
        auto& g = g_asc_ring;
        const u64 eop = g.last_eop_tsc.load(std::memory_order_relaxed);
        const u64 now = __rdtsc();
        if (eop != 0 && now > eop) {
            const u64 ticks = now - eop;
            const u64 us = ticks / 3500; // ~3.5 GHz TSC; the mean uses the measured ratio
            g.sg_n.fetch_add(1, std::memory_order_relaxed);
            g.sg_ticks.fetch_add(ticks, std::memory_order_relaxed);
            g.sg_hist[us < 100 ? 0 : us < 300 ? 1 : us < 1000 ? 2 : us < 3000 ? 3 : 4].fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    Common::FaultMonitor::CountGuestSubmit(false);
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        if (split_run_ahead && AscRingLatency()) [[unlikely]] {
            // Under m_access and before the push below, so stamps keep the queue's task order.
            std::scoped_lock ring_lock{g_asc_ring.submit_mutex};
            g_asc_ring.submits[gnm_vqid % AscRingQueues].push_back(
                g_asc_ring_clock.Snap(gnm_vqid));
        }
        // Before the push: the front may run and pop this task before this function returns.
        asc_outstanding.fetch_add(1, std::memory_order_relaxed);
        queue.submits.emplace(task.handle);
        if (AscStartStats()) [[unlikely]] {
            // Low bit: an earlier submission of this queue is still queued or running.
            asc_submit_ns[gnm_vqid].push_back((AscNowNs() & ~u64{1}) |
                                              (queue.submits.size() > 1 ? 1 : 0));
        }
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    asc_submit_seq.fetch_add(1, std::memory_order_release);
    submit_cv.notify_one();
}

bool LeanViewportMemoEnabled() {
    return ViewportMemoEnabled();
}
const Shader::Info* NativeRecPublishProgram(const Vulkan::GraphicsPipeline* pipeline,
                                            std::span<const u32> vs_user_data) {
    return NativeRecPublishProgramImpl(pipeline, vs_user_data, nullptr);
}
bool LeanNopHasBackEffect(const PM4Header* header, bool guest_markers_enabled) {
    return NopHasBackEffect(header, guest_markers_enabled);
}
bool LeanKeepsSubmitEpoch(PM4ItOpcode opcode) {
    return KeepsSubmitEpoch(opcode);
}

} // namespace AmdGpu
