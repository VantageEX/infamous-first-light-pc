// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <immintrin.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include "common/types.h"

// SHADPS4_NATIVE_REC (2026-10-09 evening, notes/native-path-2026-10-09.md sec. 4): the game's job
// threads write one record per DrawIndex at the HLE call, and group consecutive draws into runs
// (same shader hashes, only whitelisted packets between: user-data SET_SH_REG, plain NOPs,
// NUM_INSTANCES, the same INDEX_TYPE / primitive type, the no-op L2-prefetch DMA). The record
// index rides in the DrawIndex trailing NOP. =1: the front verifies every record against its own
// state (no behaviour change).
namespace AmdGpu::NativeRec {

constexpr u32 Magic = 0x4e415431u; // 'NAT1' in the trailing NOP's data_block[0]
// r3 (2026-10-14): 2^16 records per job slot (was 2^14). The rooftop (~11 k draws per frame, a slot
// writing several thousand per frame) refused ~1,900 draws per frame at the translator's lifetime
// check: the job thread was more than RingSize / 4 records ahead when a worker reached the draw.
constexpr u32 RingBits = 16;
constexpr u32 RingSize = 1u << RingBits;
/// r3: SHADPS4_NR_WINDOW = how far (records) the job thread may be ahead of a draw a translator
/// takes (the rest of the ring is the GPU's safe window). Default 4096 = the 2^14 ring's quarter
/// (the old behaviour); at most RingSize / 4.
inline u32 LifetimeWindow() {
    static const u32 w = [] {
        const char* v = std::getenv("SHADPS4_NR_WINDOW");
        const long n = v ? std::atol(v) : 4096;
        return static_cast<u32>(n < 1 ? 1 : n > long(RingSize / 4) ? long(RingSize / 4) : n);
    }();
    return w;
}
constexpr u32 MaxSlots = 16;
constexpr u32 MaxRun = 4096;

struct Record {
    std::atomic<u32> stamp; ///< idx + 1 once written (seqlock: 0 while being written)
    u32 run_k;              ///< position in the run, 0 = the run's head
    u32 tag;                ///< (slot << 28) | shader-hash run id
    u32 head_idx;           ///< ring index of the run's head
    const u32* cmd;         ///< this draw's DRAW_INDEX_2 packet
    u64 index_addr;
    u32 index_count;
    u32 instances;          ///< NUM_INSTANCES (valid when inst_known)
    u32 inst_known;
    u32 vs_known;           ///< bits of ud_vs the job thread's mirror knows
    u32 ps_known;
    u32 gap_dwords;         ///< dwords walked before this draw (0 for a head after a break)
    std::array<u32, 16> ud_vs;
    std::array<u32, 16> ud_ps;
    // Head only, updated by every draw the run takes (final once the command buffer is
    // submitted: the game submits after its jobs finished).
    // Opus review 7: written by later draws after the head's stamp was published.
    std::atomic<u32> count;
    std::atomic<const u32*> end; ///< dword after the run's last trailing NOP
    // Stage a2: what the job thread derived itself through the program table (0: not derived).
    u32 derived;
    u32 flat_vs_dw;
    u32 flat_ps_dw;
    u32 vi_bytes;
    u64 payload_pos;      ///< byte position in the thread's payload ring: flat VS, flat PS, vi
    const void* pipeline; ///< the Vulkan::GraphicsPipeline the derivation used
    u64 fetch_addr;
    u32 vs_pgm; ///< the bound VS / PS program address >> 8 (the game binds per-object copies)
    u32 ps_pgm;
    // SHADPS4_NATIVE_GPU_DIRECT (sec. 10): 1 = the GPU record of this draw is written (g_gpu_recs
    // at the same ring index); the run's ranges live in g_unions[union_idx & UnionMask]; the
    // values of the fetch shader's vertex / instance offset SGPRs (0 when none); per fetch
    // attribute, the stream's guest address (0: null stream).
    u32 gpu_ok;
    u32 union_idx;
    u32 vo_ud;
    u32 io_ud;
    u32 vcount;
    std::array<u64, 16> vaddr;
    /// The raw T# / S# words of every image and sampler of every stage (NativeImageKey's order):
    /// count and two independent 64-bit hashes (ImageKeyHash).
    u32 img_n;
    u64 img_h1, img_h2;
    /// Coverage (sec. 11): needed user-data registers the job thread never saw written - state a
    /// command buffer inherited from an earlier one (another job thread's). Constant over a run
    /// (a write in any gap of the run makes them known and ends the run's inherited set). The
    /// GPU record holds stale values for them: the push-layout user-data slots (push_inh) and
    /// the Guest binding slots whose V# sits in them (inh_slots) come from the batch opener.
    u32 inh_vs, inh_ps;
    u32 push_inh;
    u64 inh_slots;
};

/// SHADPS4_NATIVE_GPU_DIRECT: two independent 64-bit hashes of an image key (raw T# / S# words).
/// Record::img_n bit 31: a T# / S# of the draw sits in inherited registers (zeros in the key).
constexpr u32 ImgInhBit = 0x80000000u;

inline void ImageKeyHash(const u32* w, u32 n, u64& h1, u64& h2) {
    u64 a = 0x9E3779B97F4A7C15ull ^ n, b = 0xC2B2AE3D27D4EB4Full + n;
    for (u32 i = 0; i < n; ++i) {
        a = (a ^ w[i]) * 0x100000001B3ull;
        a ^= a >> 29;
        b = (b + w[i]) * 0xFF51AFD7ED558CCDull;
        b ^= b >> 33;
    }
    h1 = a;
    h2 = b;
}

// SHADPS4_NATIVE_GPU_DIRECT (notes/native-path-2026-10-09.md sec. 10): the job thread writes,
// per derived draw, the record the native shaders read (Shader::NativeRecordDwords dwords: user
// data in push layout, each Guest binding's guest address (low 32 bits) and size, each pulled
// vertex location's guest address (low 32 bits)); the GPU subtracts a per-batch base. And per
// run, the union of every binding's range over the run's draws so far (count published last).
/// 1: ring mode; 2: G1 verify (records vs every open / join); 3: ring verify (ring prepares run,
/// batches stay legacy, every prepared member's join is compared with the ring's prediction).
inline const int g_gpu_mode = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_GPU_DIRECT");
    return v ? std::atoi(v) : 0;
}();
constexpr u32 GpuRecDwords = 176; // = Shader::NativeRecordDwords (bindless slot words 128-159)
/// The job thread writes dwords 0-127 and the guest-address hi bytes 160-175
/// (SHADPS4_NATIVE_PT); the back the bindless slot words 128-159.
constexpr u32 GpuRecJobDwords = 128;
constexpr u32 GpuRecHiOffset = 160;
inline std::array<std::atomic<u32*>, MaxSlots> g_gpu_recs{};
constexpr u32 UnionBits = 12;
constexpr u32 UnionMask = (1u << UnionBits) - 1;
constexpr u32 UnionSlots = 40; // Shader::NUM_BUFFERS
struct RunUnion {
    std::atomic<u32> stamp; ///< head ring index + 1 (0 while being started)
    std::atomic<u32> n;     ///< draws of the run the ranges cover (published after them)
    u64 guest_mask;         ///< Guest binding slots
    u64 eq_mask;            ///< slots whose (address, size) was the same for every draw
    std::array<u64, UnionSlots> lo, hi, first_addr;
    std::array<u32, UnionSlots> first_size;
    u32 vcount;
    std::array<u64, 16> vlo, vhi;
    u64 ilo, ihi;
};
inline std::array<std::atomic<RunUnion*>, MaxSlots> g_unions{};
/// SHADPS4_NATIVE_GPU_DIRECT=1: the GPU record ring of a job slot lives in a mapped Vulkan buffer
/// (allocated through this hook, set by the rasterizer; null: CPU memory, ring mode off); its
/// VkBuffer handle per slot.
inline std::atomic<u32* (*)(u32 slot)> g_gpu_alloc{nullptr};
inline std::array<std::atomic<u64>, MaxSlots> g_gpu_handles{};
/// SHADPS4_NATIVE_SEG: the ring's device address (published before its handle; 0: none).
inline std::array<std::atomic<u64>, MaxSlots> g_gpu_bda{};
/// q47: one cache line per job slot for the per-draw stores below (16 slots shared two lines:
/// every job thread's per-draw store took the line from the others).
template <typename T>
struct alignas(64) PaddedAtomic : std::atomic<T> {
    PaddedAtomic() noexcept : std::atomic<T>(T{}) {}
};
/// The job thread's next ring index (relaxed; the back's ring-lifetime check).
inline std::array<PaddedAtomic<u32>, MaxSlots> g_job_next{};

constexpr u32 PayloadBits = 23; // 8 MB per thread
constexpr u64 PayloadSize = u64{1} << PayloadBits;
inline std::array<std::atomic<u8*>, MaxSlots> g_payloads{};
/// Opus review 2: per slot, the payload position the job thread has claimed up to (published
/// before it writes there). A reader's bytes at [pos, ...) are intact iff claim - pos <= size.
inline std::array<PaddedAtomic<u64>, MaxSlots> g_payload_claim{};
/// Slot 15 is shared by every thread past the 14th: no records there (review 6).
constexpr u32 MaxRecordSlot = 14;

/// Program table: (VS hash, PS hash, fetch shader address) -> the pipeline the front (or a lean
/// worker, SHADPS4_LEAN_PARSE=2) looked up for such a draw. Any number of writers (an empty slot is
/// claimed by CAS to TableClaiming, filled, then keyed); read lock-free by the job threads.
struct ProgramEntry {
    std::atomic<u64> key{};
    u64 vs_hash{}, ps_hash{}, fetch_addr{};
    std::atomic<const void*> pipeline{};
    /// Coverage (sec. 11): per stage (VS low 32 bits, PS high) the user-data registers it really
    /// reads (low 16 bits) and the strict subset that may not be inherited (SRT walker roots,
    /// fetch-shader SGPRs; high 16 bits); a half of 0xffffffff = unknown -> the RSRC2-count rule.
    /// Published before the pipeline / key.
    std::atomic<u64> needs{~u64{0}};
};
constexpr u32 TableSize = 8192;
inline std::array<ProgramEntry, TableSize> g_table{};
/// VS hash -> fetch_shader_sgpr_base + 1 (0: unknown), 0xffffffff: no fetch shader.
struct VsEntry {
    std::atomic<u64> vs_hash{};
    std::atomic<u32> fetch_base{};
};
inline std::array<VsEntry, TableSize> g_vs_table{};

/// A slot being filled by a writer (never a real key; readers skip it).
constexpr u64 TableClaiming = ~u64{0};

inline u64 TableKey(u64 vs, u64 ps, u64 fetch) {
    const u64 k = vs * 0x9E3779B97F4A7C15ull ^ (ps + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full ^
                  (fetch + 1) * 0x165667B19E3779F9ull;
    return k && k != TableClaiming ? k : 1;
}

/// Waits while another writer fills the slot (a few stores).
inline u64 TableSettled(const std::atomic<u64>& word) {
    u64 v = word.load(std::memory_order_acquire);
    while (v == TableClaiming) {
        _mm_pause();
        v = word.load(std::memory_order_acquire);
    }
    return v;
}

inline const void* TableFind(u64 vs, u64 ps, u64 fetch, u64* needs = nullptr) {
    const u64 k = TableKey(vs, ps, fetch);
    for (u32 i = 0; i < 16; ++i) {
        const auto& e = g_table[(k + i) & (TableSize - 1)];
        const u64 ek = e.key.load(std::memory_order_acquire);
        if (ek == 0) {
            return nullptr;
        }
        if (ek == k && e.vs_hash == vs && e.ps_hash == ps && e.fetch_addr == fetch) {
            const void* p = e.pipeline.load(std::memory_order_acquire);
            if (needs) {
                *needs = e.needs.load(std::memory_order_acquire);
            }
            return p;
        }
    }
    return nullptr;
}

/// Any thread. `needs_fn()` (the stages' read masks) runs only when an entry is created or
/// re-pointed. Concurrent re-points of one entry may pair one writer's needs with another's
/// pipeline: harmless, the needs depend only on the key's shaders (VS, PS, fetch shader).
template <typename NeedsFn>
inline void TablePublish(u64 vs, u64 ps, u64 fetch, const void* pipeline, NeedsFn&& needs_fn) {
    const u64 k = TableKey(vs, ps, fetch);
    for (u32 i = 0; i < 16; ++i) {
        auto& e = g_table[(k + i) & (TableSize - 1)];
        u64 ek = TableSettled(e.key);
        if (ek == 0) {
            if (e.key.compare_exchange_strong(ek, TableClaiming, std::memory_order_acq_rel)) {
                e.vs_hash = vs;
                e.ps_hash = ps;
                e.fetch_addr = fetch;
                e.needs.store(needs_fn(), std::memory_order_relaxed);
                e.pipeline.store(pipeline, std::memory_order_relaxed);
                e.key.store(k, std::memory_order_release);
                return;
            }
            ek = TableSettled(e.key); // another writer took it: compare with what it wrote
        }
        if (ek == k && e.vs_hash == vs && e.ps_hash == ps && e.fetch_addr == fetch) {
            if (e.pipeline.load(std::memory_order_relaxed) != pipeline) {
                e.needs.store(needs_fn(), std::memory_order_relaxed);
                e.pipeline.store(pipeline, std::memory_order_release);
            }
            return;
        }
    }
}

/// SHADPS4_NATIVE_REC_INHERIT=0: no inherited registers (every needed register must be known).
inline const bool g_inherit_on = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_REC_INHERIT");
    return !v || v[0] != '0';
}();
/// SHADPS4_NATIVE_REC_NEEDS=0 turns the read-mask rule off (the RSRC2-count rule everywhere).
inline const bool g_needs_on = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_REC_NEEDS");
    return !v || v[0] != '0';
}();

inline u32 VsFetchBase(u64 vs) {
    const u64 k = vs ? vs : 1;
    for (u32 i = 0; i < 16; ++i) {
        const auto& e = g_vs_table[(k * 0x9E3779B97F4A7C15ull + i) & (TableSize - 1)];
        const u64 h = e.vs_hash.load(std::memory_order_acquire);
        if (h == 0) {
            return 0;
        }
        if (h == k) { // (TableClaiming: being filled, never equal to k)
            return e.fetch_base.load(std::memory_order_relaxed);
        }
    }
    return 0;
}

/// Any thread (empty slots claimed by CAS). fetch_base: fetch_shader_sgpr_base + 1, or
/// 0xffffffff for none.
inline void VsPublish(u64 vs, u32 fetch_base) {
    const u64 k = vs && vs != TableClaiming ? vs : 1;
    for (u32 i = 0; i < 16; ++i) {
        auto& e = g_vs_table[(k * 0x9E3779B97F4A7C15ull + i) & (TableSize - 1)];
        u64 h = TableSettled(e.vs_hash);
        if (h == 0) {
            if (e.vs_hash.compare_exchange_strong(h, TableClaiming, std::memory_order_acq_rel)) {
                e.fetch_base.store(fetch_base, std::memory_order_relaxed);
                e.vs_hash.store(k, std::memory_order_release);
                return;
            }
            h = TableSettled(e.vs_hash);
        }
        if (h == k) {
            return;
        }
    }
}

inline std::array<std::atomic<Record*>, MaxSlots> g_rings{};

inline const int g_mode = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_REC");
    return v ? std::atoi(v) : 0;
}();

/// SHADPS4_NATIVE_SEGB (notes/native-path-2026-10-09.md sec. 20, milestone 3 brick B): the front
/// builds segments of consecutive native members and predicts each member's indirect command;
/// census and verify only (every draw still runs as today; the back compares). 0 = off.
inline const int g_segb = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_SEGB");
    return v ? std::atoi(v) : 0;
}();
/// SHADPS4_NATIVE_SEGX (sec. 23, milestone 3 brick C): the front posts closed segments as one
/// Segment record each (needs SEGB, GPU_DIRECT=1, PT, SEG). 2 = evaluate only (members still
/// drawn by today's path), 1 = execute (member Draw records withheld). 0 = off.
inline const int g_segx = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_SEGX");
    const int m = v ? std::atoi(v) : 0;
    return (m == 1 || m == 2) && g_segb != 0 ? m : 0;
}();

} // namespace AmdGpu::NativeRec
