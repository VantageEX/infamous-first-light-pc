// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/native_census.h"

#include <shared_mutex>

#include <map>
#include <tuple>

#include <array>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stop_token>
#include <thread>
#include <span>
#include <unordered_map>
#include <vector>
#include <boost/container/small_vector.hpp>
#include "common/lru_cache.h"
#include "common/readback_unmap_gate.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/pt_bits.h"
#include "video_core/buffer_cache/submit_epoch.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/gpu_range_set.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/page_manager.h"

namespace AmdGpu {
struct Liverpool;
namespace CpSplit {
struct VertexInputsPayload;
}
} // namespace AmdGpu

namespace Core {
class MemoryManager;
}

namespace Shader {
struct BoundStage;
}

namespace Vulkan {
class GraphicsPipeline;
class MasterSemaphore;
}

namespace Common {
class BufferSampleProbe;
enum class BufferSampleCaller : u8;
} // namespace Common

namespace Common::ReadbackOffload {
enum class Origin : u32;
struct SyncTiming;
} // namespace Common::ReadbackOffload

namespace AmdGpu {
union Regs;
}

namespace VideoCore {

/// SHADPS4_DRAW_COST=1 (diagnostic): cycles of the last SynchronizeBuffer on this thread, split into
/// the dirty-page walk (ForEachUploadRange) and the rest; callers reset them before a call.
u64& DiagSyncWalkCycles();
u64& DiagSyncRestCycles();

using BufferId = Common::SlotId;

/// SHADPS4_LEAN_BIND=1 (=2 verify), default off (notes/lean-draw-path-design-2026-10-08.md item
/// A): FindBufferHinted in the binding pass 1 and ObtainReadLean for plain read shader bindings.
inline const int g_lean_bind = [] {
    const char* env = std::getenv("SHADPS4_LEAN_BIND");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

/// r4 (2026-10-14, upstream #5278): SHADPS4_IB_INDIRECT_CAP. An indexed INDIRECT draw reads its
/// indices up to the capacity INDEX_BUFFER_SIZE set (VGT_DMA_MAX_SIZE = regs.max_index_size), but
/// the bound / checked range came from regs.num_indices, which every direct draw overwrites with
/// its own index count: the index buffer could end before the indices the GPU reads (the rooftop
/// load's device losses, s947 ff.: ReadInvalid just past one resource). 1 = bind / check
/// max(num_indices, max_index_size) indices (capped at 2^24) + census; 2 = census only.
inline const int g_ib_indirect_cap = [] {
    const char* env = std::getenv("SHADPS4_IB_INDIRECT_CAP");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();
/// The index count an indexed indirect draw binds (0: regs.num_indices, the old rule). `path`:
/// 0 = the generic DrawIndirectTail, 1 = the A2 worker translator (census only).
u32 IndirectIndexCapacity(const AmdGpu::Regs& regs, u32 path);

class TextureCache;
class MemoryTracker;
class PageManager;
class ReadbackQueue;

enum class NativeRunAccess : u32 { Storage, Vertex, Index };

/// q7c: set by the rasterizer while a dispatch executes (CP thread).
extern thread_local bool t_in_dispatch;
/// r20 SHADPS4_EXPOSURE_STALE_READ: set by the rasterizer while it binds a dispatch of the game's
/// exposure metering shader (cs_a8b68717); NoteBatchWrite then records the written ranges.
extern thread_local bool t_tolerant_writer;

class BufferCache {
public:
    /// q7 (SHADPS4_READBACK_EARLY_CUT): true once after a GPU write into a recently read-back
    /// window was noted on the CP thread (the dispatch then ends its batch).
    bool TakeEarlyCut() noexcept {
        const bool c = early_cut_pending;
        early_cut_pending = false;
        if (c) {
            DisarmEarlyCut(early_cut_mask); // q7c: one cut per readback
            early_cut_mask = 0;
        }
        return c;
    }
    bool EarlyCutPending() const noexcept {
        return early_cut_pending;
    }
    void DisarmEarlyCut(u32 mask);
    /// q56 (SHADPS4_HOT_PREFETCH_AT_WRITE): after a dispatch (CP thread), prefetch hot pages it wrote.
    void HotPrefetchAfterWrite();
    u32 early_cut_mask{};

    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    /// SHADPS4_PT_BITS (36-40; default g_pt_auto_bits: 37 below 10 GiB of video memory, else 40;
    /// PERF-PLAN 11, 2026-10-20; s1461 4 GB sim: +22% mean / +21% p1): the guest address bits the
    /// GPU page table (a device address per 16 KB page) covers. 40 = the PS4 GPU's whole range,
    /// 512 MB of video memory; the game maps GPU memory no higher than 63.5 GB (s1457), so 37 bits
    /// (128 GB) = 64 MB. Above the table: CPU entry writes are clipped, the descriptor readers get
    /// 0 (robustness2: a miss), the native pointer readers clamp to a miss (NativePtTranslate).
    static u64 BdaNumPages() {
        static const u64 pages = [] {
            u32 bits = g_pt_auto_bits.load(std::memory_order_relaxed);
            if (const char* v = std::getenv("SHADPS4_PT_BITS"); v && std::atoi(v) >= 36 &&
                                                                std::atoi(v) <= 40) {
                bits = static_cast<u32>(std::atoi(v));
            }
            return u64{1} << (bits - CACHING_PAGEBITS);
        }();
        return pages;
    }
    static u64 BdaPageTableSize() {
        return BdaNumPages() * sizeof(vk::DeviceAddress);
    }

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
        bool mesh = false; ///< a BIG_GROW mesh-sized join: video memory if it fits (GitHub #30)
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }
    [[nodiscard]] u64 FaultPagesProcessed() const noexcept {
        return fault_manager.PagesProcessed();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size,
                          InvalidateOrigin origin = InvalidateOrigin::Explicit);

    /// SHADPS4_PARTICLE_ARENA_DISCARD: drop the GPU-modified state of a dead particle arena half
    /// (no download). Posted from Game:Main; DiscardDeadGpuData runs on the CP thread.
    void PostArenaDiscard(VAddr lo, u64 size, VAddr arena_lo, VAddr arena_hi);
    /// SHADPS4_ARENA_READ_PREFETCH (q48): Game:Main at the frame's submit; posts PrefetchArenaReads.
    void PostArenaReadPrefetch(VAddr half);
    /// SHADPS4_ARENA_PREFETCH_RETRY: a submit thread after vkQueueSubmit of `submitted`; posts the
    /// armed PrefetchArenaReads retry once `submitted` covers it (never from the CP thread).
    void PostArenaReadRetry(u64 submitted);
    void DiscardDeadGpuData(VAddr lo, u64 size,
                            const std::vector<std::pair<VAddr, std::vector<u64>>>& chunks);

    /// Block new asynchronous readbacks before GPU unmap and settle earlier ones.
    void BeginGpuUnmap(VAddr device_addr, u64 size);

    /// A completed GPU mapping can admit asynchronous readbacks again; never waits on CP.
    void EndGpuMap(VAddr device_addr, u64 size);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    /// `offload_write`: a guest thread's write fault may become an asynchronous job
    /// (SHADPS4_WRITE_FAULT_OFFLOAD); every other write stays synchronous.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false,
                    bool offload_write = false);

    /// Binds host vertex buffers for the current draw. `front_vi`: with CP split FRONT_DERIVE,
    /// the vertex inputs the front derived for this draw (cp_split.h), else null.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline, const Shader::BoundStage& vs,
                           boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
                           const AmdGpu::CpSplit::VertexInputsPayload* front_vi = nullptr);

    /// Bind host index buffer for the current draw.
    /// SHADPS4_NATIVE_PASS join (notes/native-renderer-2026-10-09.md sec. 3): the same buffer
    /// lookups / syncs BindVertexBuffersDerived and BindIndexBuffer make, without recording
    /// anything; host handles and offsets go to `out`. With `barriers` non-null the hazard state
    /// is advanced exactly as those functions do and any barrier is appended there.
    /// SHADPS4_NATIVE_REC_RUN (stage c1): the one existing buffer that contains [lo, hi) - no
    /// creation, no merge - synced once through ObtainBuffer, with the barrier the access needs
    /// appended to `barriers`. nullptr when the span is not inside one buffer or over 64 MB.
    /// SHADPS4_NATIVE_REC_PREP: the existing buffer holding `addr` (page table only: no
    /// creation, merge or sync), or nullptr.
    const Buffer* NativeBufferAt(VAddr addr, u64 size = 0, u32 site = 23); // q26 E2: size != 0 = the caller's range; q37 site
    /// A2: lock-free, read-only lookup for the A2 worker threads (vk_a2.cpp): the cached buffer
    /// holding `addr`, or null. Reads the page table's second-level page with an acquire load
    /// (the CP publishes it with release), the entry's id as a plain word (the CP's page-table
    /// edits are plain stores: a torn or stale id is caught by the bounds check below and by
    /// the back's StructGeneration check at publish, which refuses a stretch whose lookups
    /// predate a buffer delete / join), and the Buffer in the stable SlotVector (objects keep
    /// their address until erased, commit d1ba1774). Mutates nothing; never creates, merges or
    /// syncs. `is_deleted` is read racily too (same publish-time guard).
    [[nodiscard]] const Buffer* A2LookupBuffer(VAddr addr) const noexcept {
        if (addr == 0) {
            return nullptr;
        }
        const PageData* entry = page_table.find_acquire(addr >> CACHING_PAGEBITS);
        if (!entry) {
            return nullptr;
        }
        const BufferId id = entry->buffer_id;
        if (!id || id.index >= Common::StableSlotCapacity) {
            return nullptr;
        }
        const Buffer& found = slot_buffers[id];
        if (found.is_deleted || !found.IsInBounds(addr, 1)) {
            return nullptr;
        }
        return &found;
    }
    const Buffer* NativeRunSpan(VAddr lo, VAddr hi, NativeRunAccess access,
                                boost::container::small_vector<vk::BufferMemoryBarrier2, 16>* barriers,
                                u64 dispatch_run, bool sync = true, int gpu_mod_hint = -1,
                                bool* gpu_mod_out = nullptr);
    void NativeJoinVertexIndex(const AmdGpu::CpSplit::VertexInputsPayload& vi, u32 index_offset,
                               NativeJoinVi& out,
                               boost::container::small_vector<vk::BufferMemoryBarrier2, 16>* barriers);

    /// r4: `num_indices` != 0 overrides regs.num_indices (indexed indirect draws: the capacity).
    void BindIndexBuffer(u32 index_offset,
                         boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
                         u32 num_indices = 0);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region. `sample_caller` only labels the opt-in
    /// BufferSampleProbe rows; it never affects the returned buffer.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(
        VAddr gpu_addr, u32 size, bool is_written, bool is_texel_buffer = false,
        BufferId buffer_id = {}, Common::BufferSampleCaller sample_caller = {});

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// r13: hands the buffer destroys parked since the last draw / dispatch entry to the
    /// scheduler at the tick current now (call at every draw / dispatch entry, back thread).
    void CommitDeletes();

    /// SHADPS4_BIND_MEMO_STATS: lock-free exact per-page CPU-dirty check; a range without a
    /// tracker region counts as dirty. CP thread.
    [[nodiscard]] bool IsRegionCpuDirtyRelaxed(VAddr addr, u64 size) const;

    /// SHADPS4_BIND_MEMO_STATS: bumped on every buffer register/unregister (create, join,
    /// delete) and around GPU map/unmap, i.e. whenever FindBuffer or ClampRangeSize could
    /// answer differently for an unchanged V#.
    [[nodiscard]] u64 StructGeneration() const noexcept {
        return struct_generation.load(std::memory_order_acquire);
    }
    void BumpStructGeneration() noexcept {
        struct_generation.fetch_add(1, std::memory_order_acq_rel);
    }

    /// SHADPS4_BIND_MEMO_STATS: bytes uploaded by every SynchronizeBuffer so far. CP thread.
    [[nodiscard]] u64 SyncUploadBytesTotal() const noexcept {
        return sync_upload_bytes_total;
    }

    /// SHADPS4_PB_CENSUS: read-bind-memo misses so far (both caller classes). CP thread.
    [[nodiscard]] u64 ReadMemoMissesTotal() const noexcept {
        return (read_memo_stats[0][0] - read_memo_stats[0][1]) +
               (read_memo_stats[1][0] - read_memo_stats[1][1]);
    }
    [[nodiscard]] u64 FaultJobsAdmitted() const noexcept {
        return fault_jobs_admitted.load(std::memory_order_acquire);
    }

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// SHADPS4_LEAN_BIND=1 (=2 verify): FindBuffer through a 4-entry bounds hint of the buffers
    /// found last, valid while StructGeneration is unchanged (every page-table edit passes
    /// ChangeRegister, which bumps it, on the CP thread). Identical to FindBuffer as long as
    /// cached buffers never overlap; =2 runs FindBuffer on every hint hit and compares. CP thread.
    BufferId FindBufferHinted(VAddr device_addr, u32 size);

    /// SHADPS4_LEAN_BIND: ObtainBuffer for a plain read shader binding (not written, not
    /// formatted) when every per-call option it would test is off (see LeanReadOk): the
    /// READ_BIND_MEMO=1/3 probe, then READ_SYNC_FAST's clean check, without ObtainBuffer's
    /// statistics and option tests. Returns {nullptr, 0} when the full path must run (invalid id,
    /// not clean); the full path then repeats the probe and the check. =2 runs the full path
    /// after every lean answer and compares the pair. CP thread.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainReadLean(VAddr device_addr, u32 size,
                                                         BufferId buffer_id);
    [[nodiscard]] bool LeanReadOk();

    /// SHADPS4_NATIVE_COHERENT (back_clean.h, native-path sec. 17): a coherence point in stream
    /// order (EpochBump record). `bounded`: a submission start, whose record carries the low 32
    /// bits of the dirty-log tail the front read when it began parsing the submission; entries
    /// after it are guest writes made after the submit and wait for the next point. CP thread.
    void CoherenceSync(bool bounded, u32 bound_lo);
    /// True when a binding of this kind needs no validity work (memory is coherent for the
    /// submission): reads that are not texel buffers, on the CP thread, after the first point.
    [[nodiscard]] bool CoherentSkip(bool is_written, bool is_texel) const noexcept {
        if (t_force_binding_sync) [[unlikely]] {
            ++t_forced_binding_checks;
            return false;
        }
        return g_native_coherent != 0 && coherent_live && !is_written && !is_texel &&
               Common::FaultMonitor::t_is_cp_thread;
    }
    /// NATIVE_COHERENT=2: runs the per-binding sync the skip replaced and classifies any upload.
    void CoherentVerify(Buffer& buffer, VAddr addr, u32 size);
    /// SHADPS4_NATIVE_COHERENT: the CP wrote these guest bytes inside a submission with no
    /// coherence point after it (DMA CPU paths, occlusion dumps): sync them now. CP thread.
    void CoherentCpWrite(VAddr addr, u64 size) {
        if (g_native_coherent != 0 && coherent_live && size != 0) {
            ++coherent_stats.dma_syncs;
            SynchronizeBuffersInRange(addr, size);
        }
    }
    void NoteCoherentSpanSkip() noexcept {
        ++coherent_stats.span_skips;
    }

    /// SHADPS4_DRAW_REPEAT_HEAP: the read-sync fast path's clean answer for a range (no upload due).
    [[nodiscard]] bool ReadRangeClean(VAddr addr, u64 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector();

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        CheckCpThread(1);
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    /// SHADPS4_THREAD_ASSERT=1 (diagnostic): logs one Critical line per site the first time
    /// buffer_ranges is walked (site 1, ForEachBufferInRange) or edited (site 0, ChangeRegister)
    /// off the command processor thread. Those users take no lock; only IsRegionRegistered may
    /// run on guest threads (see buffer_ranges_mutex). No-op when the flag is off.
    void CheckCpThread(int site) const;

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    /// Returns false when nothing was GPU-dirty, in which case no tracker state was changed.
    /// timing, when set, receives the split ns of the synchronous submit and wait.
    template <bool async>
    bool DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write,
                              Common::ReadbackOffload::SyncTiming* timing = nullptr);
    /// SHADPS4_WRITE_FAULT_WIDEN=2: downloads every GPU-modified range of the window, across all
    /// buffers in it, with one submit and wait. Returns false when nothing was GPU-modified.
    bool DownloadWindowMultiBuffer(VAddr window_addr, u64 window_size);

    // Readback wait offload (SHADPS4_READBACK_OFFLOAD, default off). A guest fault is split
    // into two command processor steps around a GPU wait taken on the faulting thread.
    // Everything below is owned by the command processor thread, except fault_guest_waiters
    // and the two members guests read before an invalidation: pending_fault_jobs and
    // fault_windows.

    /// Guest bytes covered by one staging copy, in absolute addresses so the list still
    /// compares after the source buffer has been merged into another.
    struct FaultCopy {
        VAddr addr;
        u64 staging_offset;
        u64 size;

        bool operator==(const FaultCopy&) const = default;
    };

    enum class FaultJobState : u8 {
        Free,
        Pending,  ///< Copy submitted; tracker state still exactly as the sync path left it.
        Settling, ///< Being finalized; skipped by nested settles.
        Success,  ///< Published. Kept until the owning thread consumes the result.
        Vetoed,   ///< Snapshot dropped, tracking untouched. The owner runs one sync fallback.
    };

    struct FaultJob {
        u64 id{};
        FaultJobState state{FaultJobState::Free};
        bool abandoned{};
        bool is_write{};
        VAddr addr{};
        u64 size{};
        u64 tick{};
        /// `tick` is a value of the readback queue's timeline, not the main one.
        bool on_readback_queue{};
        /// With on_readback_queue: the timeline is the fast lane's (SHADPS4_READBACK_FAST_LANE).
        bool on_fast_lane{};
        /// With on_readback_queue: the ASC stream's timeline (SHADPS4_ASC_STREAM).
        bool on_asc{};
        /// With on_readback_queue: the arena retry queue's timeline (SHADPS4_ARENA_RETRY_QUEUE).
        bool on_retry_lane{};
        u64 total_bytes{};
        std::vector<FaultCopy> copies;
        /// Buffer spans the copies were collected in, in order: [addr, addr + size) for a read
        /// job, one span per buffer of the window for a write job (SHADPS4_WRITE_FAULT_OFFLOAD).
        std::vector<std::pair<VAddr, VAddr>> spans;
        /// A write job's exact faulting range, CPU-marked at publication.
        VAddr mark_addr{};
        u64 mark_size{};
        std::vector<u64> seqs;
        /// SHADPS4_FAULT_VETO_EXACT: GpuWriteLog position at the snapshot of `seqs`.
        u64 write_log_pos{};
        std::unique_ptr<Buffer> staging;
        /// SHADPS4_FAULT_GUEST_COPY: who copies the staging bytes into guest memory.
        /// 0 open, 1 the owning guest thread is copying, 2 it has copied, 3 closed by the
        /// command processor (it copies itself, or the job is vetoed). See GuestCopyFaultJob.
        std::atomic<u32> guest_copy{0};
        /// SHADPS4_FAULT_GUEST_SUBMIT: who submits the recorded lane copy (ReadbackQueue slot
        /// `submit_slot`). 0 recorded and open to the owning guest thread, 1 the owner is
        /// submitting, 2 it has submitted, 3 the command processor is submitting, 4 submitted
        /// by the command processor or nothing deferred. `tick` is valid in 2 and 4 only; the
        /// command processor reads it through CloseGuestSubmit. See GuestSubmitFaultJob.
        std::atomic<u32> guest_submit{4};
        u32 submit_slot{};
        /// f2 (SHADPS4_LANE_PREFETCH): a lane prefetch job (no owner thread; abandoned from the start).
        bool prefetch{};
        /// f2: the slot is held for the prefetch waiter until the back's publish has run (never reused before).
        /// A hot-page prefetch (not on_asc) holds it only until FinalizeFaultJob; an arena read
        /// prefetch (q48, rena) like f2, until PublishPrefetch.
        std::atomic<bool> pf_hold{false};
        /// q48 (SHADPS4_ARENA_READ_PREFETCH): an arena read prefetch. Holds its slot until
        /// PublishPrefetch like an f2 job, so the waiter copies the bytes, not the back.
        bool arena{};
        /// SHADPS4_ARENA_PREFETCH_RETRY: an arena job of a retry run (set with `arena`, read
        /// only with it).
        bool arena_retry{};
        /// r20 SHADPS4_EXPOSURE_STALE_READ: a pending hot-page job whose window a guest read was
        /// let through with the stale bytes (tracker already CPU-valid); its publish only writes
        /// the fresh bytes into guest memory. Reset when the job is finalized.
        bool released{};
    };
    /// The vote of a fault job: true if no GPU write marking touched it since its snapshot
    /// (region sequences; with SHADPS4_FAULT_VETO_EXACT=1 a moved sequence is refined by the
    /// logged ranges of those markings against the job's window). Command processor only.
    bool FaultJobUnwritten(const FaultJob& job);
    /// Guest thread that owns `job`, after its GPU wait and before FinishFaultDownload.
    void GuestCopyFaultJob(FaultJob& job);
    /// Command processor, before a job is finalized: true if the guest thread has copied.
    bool CloseGuestCopy(FaultJob& job);
    /// Guest thread that owns `job`, before its GPU wait: submits the lane copy, or takes the
    /// value the command processor submitted. Returns the lane tick to wait for.
    u64 GuestSubmitFaultJob(FaultJob& job);
    /// Command processor, before it reads a job's tick: waits for an in-flight guest submit
    /// or submits the recorded slot itself. No-op for a job that is not deferred.
    void CloseGuestSubmit(FaultJob& job);
    /// Closes the submit of every unfinalized job whose window overlaps the range (a buffer
    /// about to be deleted may be a source of its recorded copy).
    void CloseGuestSubmitsOver(VAddr device_addr, u64 size);

    /// Returned to the faulting thread by value. id 0 means the fault was resolved already.
    struct FaultTicket;
    /// m2: Game:Main's main-semaphore download waits by (window, writer), logged every ~10 s.
    static void M2NoteMainWait(const FaultTicket& t, u64 wait_tsc, u64 tick_at_fault);
    struct FaultTicket {
        u64 id{};
        u64 tick{};
        /// SHADPS4_FAULT_OVERLAP_RETRY: `id` names an overlapping pending job to wait for on the
        /// faulting thread, settle, then retry, instead of a job created for this fault.
        bool retry{};
        /// `tick` belongs to the readback queue's timeline.
        bool on_readback_queue{};
        /// With on_readback_queue: the fast lane's timeline.
        bool on_fast_lane{};
        /// With on_readback_queue: the ASC stream's timeline.
        bool on_asc{};
        /// With on_readback_queue: the arena retry queue's timeline (SHADPS4_ARENA_RETRY_QUEUE).
        bool on_retry_lane{};
        /// SHADPS4_READBACK_LATENCY diagnostics, set by PrepareFaultDownload.
        u64 prep_end_ns{};
        u64 flush_end_ns{};
        u32 batch_draws{};
        u32 inflight{};
        /// SHADPS4_GPU_BATCH_TIMING batch mix: main tick known complete before the flush, the
        /// last main batch ahead of the copy, the copy's own main batch (0: separate submission),
        /// the newest batch that wrote the copied bytes, and the flush time (0: not recorded).
        u64 done_tick{};
        u64 last_ahead{};
        u64 own_tick{};
        u64 producer{};
        u64 mix_flush_ns{};
        u64 label_tick{};
        /// m2: a download waiting on the main semaphore for its batch - the copied window and
        /// the newest writer of its bytes in the write history (tick, kind, id; kind 0xff none).
        bool m2_main{};
        u64 m2_addr{}, m2_bytes{}, m2_writer_tick{}, m2_id{};
        u8 m2_kind{0xff};
        /// SHADPS4_FAULT_GUEST_COPY: the job created for this fault (null for a retry ticket).
        FaultJob* job{};
        /// SHADPS4_FAULT_GUEST_SUBMIT: `job` has a recorded lane copy the owner submits
        /// (GuestSubmitFaultJob) before it waits; `tick` is then unknown.
        bool deferred_submit{};
    };
    void RecordFaultLatency(const FaultTicket& ticket, u32 passes, u64 t_fault, u64 t_cp,
                            u64 t_wait, u64 t_woke, bool free_at_wait, VAddr device_addr);

    /// Waits until a job's copy has completed on whichever queue it ran. Closes a deferred
    /// guest submit first, so the tick exists.
    void WaitFaultJob(FaultJob& job);
    /// Newest main-timeline tick that GPU-wrote any byte of the range (0 if never).
    u64 ProducerTick(VAddr addr, u64 size) const;
    void RecordFaultCopiesOnReadbackQueue(vk::CommandBuffer cmdbuf, vk::Buffer source,
                                          vk::Buffer staging,
                                          std::span<const vk::BufferCopy> copies,
                                          bool in_queue_order);
    std::unique_ptr<ReadbackQueue> readback_queue;
    /// SHADPS4_READBACK_FAST_LANE: second queue for copies whose producer already completed.
    std::unique_ptr<ReadbackQueue> fast_lane;
    /// SHADPS4_ARENA_RETRY_QUEUE (r18): the arena prefetch retry copies, which wait for their
    /// producer, on their own queue (never ahead of a fault copy on the fast lane).
    std::unique_ptr<ReadbackQueue> retry_lane;
    ReadbackQueue& LaneOf(bool on_fast_lane) {
        return on_fast_lane ? *fast_lane : *readback_queue;
    }
public:
    /// SHADPS4_ASC_STREAM: an indirect dispatch's argument range, checked with its bindings.
    /// SHADPS4_ASC_V2_CENSUS=1 (diagnostic, notes/asc-stream-v2-design-2026-09-28.md step 0): who
    /// writes. The rasterizer sets the draw / dispatch tag, the DMA / fill / WRITE_DATA / join /
    /// image paths set their own; NoteBatchWrite stores it next to each main write-history entry.
    /// Kinds: 1 draw, 2 dispatch from the graphics queue, 3 dispatch from an async queue recorded on
    /// main, 4 DMA_DATA copy, 5 WRITE_DATA, 6 JoinOverlap, 7 image->buffer, 8 fill, 0 other.
    void SetWriterTag(u8 kind, u64 id) noexcept {
        census_kind = kind;
        census_id = id;
    }
    [[nodiscard]] static bool CensusEnabled();
    /// r20 SHADPS4_EXPOSURE_STALE_READ (game-specific, inFAMOUS Second Son): the CPU reads the
    /// exposure metering output a frame late with no wait (REFERENCE 0x171610); like the PS4, a
    /// read before the GPU finished sees the bytes already in memory instead of waiting.
    [[nodiscard]] static bool ExposureStaleRead();
    /// Census (b): one ASC candidate binding (before the stream decision).
    void CensusAscBinding(BufferId buffer_id, VAddr addr, u64 size, bool written);
    [[nodiscard]] static bool RangeShadowEnabled();
    /// SHADPS4_ASC_RANGE=1 (needs ASC_STREAM): the ASC decision uses the open batch's exact
    /// ranges (RangeConflict) instead of the per-Buffer tick. Range notes are then always on.
    [[nodiscard]] static bool AscRangeEnabled();
    /// SHADPS4_ASC_I1=0/2: ASC accesses of the dispatch being recorded, stamped with the
    /// segment value after its tail (a flush inside pass 2 reopens the segment).
    struct PendingAscAccess {
        BufferId id;          ///< cache Buffer (ids survive slot_buffers growth)
        const Buffer* fixed;  ///< non-slot Buffers (GDS), else null
        VAddr lo;
        u64 size;
        bool write;
    };
    std::vector<PendingAscAccess> pending_asc;
    void StampPendingAsc(u64 value) {
        for (const auto& a : pending_asc) {
            const Buffer* buffer = a.fixed;
            if (!buffer) {
                if (IsBufferInvalid(a.id)) {
                    continue; // deleted: its memory is retired by R0 before reuse
                }
                buffer = &slot_buffers[a.id];
            }
            buffer->NoteAscAccess(a.lo, a.size, a.write, value);
        }
        pending_asc.clear();
    }
    [[nodiscard]] static bool AscDepsEnabled();
    /// SHADPS4_UPLOAD_AHEAD_STATS (diagnostic).
    void NoteTransferBegin();
    void NoteTransferEnd();
    void NoteUploadAhead(const Buffer& buffer, VAddr device_addr, u32 size, u64 bytes);
    /// SHADPS4_UPLOAD_AHEAD: upload the dirty pages after a small upload (see buffer_cache.cpp).
    void UploadAhead(Buffer& buffer, VAddr device_addr, u32 size);
    /// Marks a Buffer as bound by an ASC candidate (its main accesses are noted by range).
    /// A Buffer bound for the first time in the middle of a batch has no range log for the
    /// accesses already made in it: those count as whole-Buffer (conservative).
    void MarkAscBound(BufferId buffer_id) const;
    /// Census mode 2: does the open main batch really touch this ASC binding's range? Returns 0
    /// (no), else the writer kind of the first overlapping access (255 = whole-Buffer note).
    u8 RangeConflict(BufferId buffer_id, VAddr addr, u64 size, bool written, u64 now,
                     u64& id) const;

    void SetAscIndirectArgs(VAddr addr, u64 size) noexcept {
        asc_indirect_addr = addr;
        asc_indirect_size = size;
    }
    VAddr asc_indirect_addr{};
    u64 asc_indirect_size{};
private:
    /// SHADPS4_ASC_STREAM: newest ASC value whose history slot was overwritten.
    u64 asc_hist_evicted{};
    Vulkan::MasterSemaphore& JobSemaphore(bool on_fast_lane, bool on_asc,
                                          bool on_retry_lane = false);
    /// SHADPS4_ASC_STREAM: GPU write ranges recorded in the last WriteHistory ASC segments.
    std::array<std::vector<std::pair<VAddr, VAddr>>, 512> asc_write_hist;
    std::array<u64, 512> asc_write_hist_value{};
    /// q11e: per-slot [min begin, max end) of its ranges.
    std::array<std::pair<VAddr, VAddr>, 512> asc_write_bbox{};
    /// Newest ASC segment value that wrote any copied byte (0 = none in the history). q11e: values
    /// <= `done_below` (finished segments) are not searched.
    u64 NewestAscWriter(std::span<const FaultCopy> copies, u64 done_below = 0) const;
    /// q11e: the done_below for a caller that treats writer w as finished when IsFree(w + margin).
    u64 AscDoneBelow(u64 margin);

public:
    /// f2 (SHADPS4_LANE_PREFETCH=1 run, =2 verify; needs the lane): 0 off.
    static int PrefetchMode();
    /// f2: back, lane stream active, right after the ASC label at `label` was recorded on the lane.
    /// Records the prefetch copies of the armed windows whose learned label it is; returns the number recorded
    /// (the caller then submits the segment: LaneCutNow).
    u64 PrefetchAfterLabel(VAddr label);

private:
    /// f2: back; the waiter posts it after the job's lane value completed (and its guest copy).
    /// `lane` is the PrefetchItem's (0 f2, 1 / 2 hot-page jobs). Clears the slot's hold only
    /// while the slot still holds job `id`.
    void PublishPrefetch(u64 id, u32 slot, u8 lane);
    void PrefetchNoteWindow(VAddr addr, u64 size);
    /// `retry_lane`: the r18 waiter of the arena retry items (pf_retry_queue).
    void PrefetchWaiterLoop(std::stop_token stop, bool own = false);
    void PrefetchMaybeLog();
    /// SHADPS4_HOT_PAGE_PREFETCH: back, once per frame (DiscardDeadGpuData); owner-less read jobs
    /// over the pages of guest readback-class write faults, published by the f2 waiter.
    void PrefetchHotPages(VAddr only_page = 0);
    /// SHADPS4_ARENA_READ_PREFETCH (q48): back, once per frame end; owner-less read jobs over the
    /// pages of `half` at the half-relative offsets where guest reads faulted lately.
    /// SHADPS4_ARENA_PREFETCH_RETRY: `gen` the discard hook generation of `half`, `retry` a run
    /// posted by the scheduler's submitted hook.
    void PrefetchArenaReads(VAddr half, u64 gen = 0, bool retry = false);
    struct PrefetchItem {
        u64 id;
        u32 slot;
        u64 value;
        u8 lane{}; ///< timeline of `value`: 0 the ASC lane (f2), 1 the fast lane, 2 main,
                   ///< 3 the arena retry queue (r18)
        bool arena{}; ///< q48: the waiter copies the bytes (the slot is held until the publish)
    };
    std::mutex pf_mutex;
    std::condition_variable_any pf_cv;
    std::vector<PrefetchItem> pf_queue;
    /// r18 SHADPS4_ARENA_RETRY_QUEUE: retry items (lane 3) wait in their own queue and thread,
    /// so an item waiting for its producer never holds up the publish of the others.
    std::mutex pf_retry_mutex;
    std::condition_variable_any pf_retry_cv;
    std::vector<PrefetchItem> pf_retry_queue;
    /// f2: back-thread state. Hot windows: first page << 24 | pages, the learned label, idle triggers.
    struct PrefetchWindow {
        u64 packed{};
        VAddr label{};
        u32 idle{};
    };
    std::array<PrefetchWindow, 16> pf_win{};
    u32 pf_win_next = 0;
    u32 pf_pending = 0; ///< prefetch jobs admitted and not yet published (back)
    bool pf_last_ok = false;
    std::array<u64, 20> pf_c{};
    std::chrono::steady_clock::time_point pf_log_t{};
    u64 producer_floor = 0; ///< Tick of the newest write too large to record per slot.
    /// SHADPS4_READBACK_EARLY_FLUSH: main tick of the last readback job per 1 MB slot, and a
    /// request raised when a GPU write lands in a recently read-back slot.
    std::unique_ptr<u64[]> readback_hot_ticks;
    bool readback_flush_requested = false;

public:
    /// True once per GPU write into a recently read-back 1 MB slot since the last call.
    /// SHADPS4_READ_PRODUCER_STATS=1 (diagnostic, CP back thread): the shader (pgm hash) whose
    /// buffer binding last wrote each 64 KiB window, and per read fault a histogram of those
    /// producers ("0" = written by no tracked shader binding, e.g. CP DMA or images).
    void SetDiagProducer(u64 pgm_hash) {
        diag_producer = pgm_hash;
    }
    static bool ReadProducerStatsEnabled();
    void NoteDiagRead(VAddr addr);

    bool TakeReadbackFlushRequest() {
        return std::exchange(readback_flush_requested, false);
    }

private:

    /// Settles job `id` if it is still pending (its tick is normally already reached).
    void SettlePendingJob(u64 id);

    /// Tracker-page span of an unfinalized job, indexed like fault_jobs. Empty when end is 0.
    struct FaultWindow {
        VAddr begin{};
        VAddr end{};
    };

    enum class VerifyResult : u8 {
        Matched,    ///< An independent second readback held the same bytes.
        Unverified, ///< No comparison was made. The job is vetoed, never published.
        Mismatch,
    };

    /// q51: back to 8 (s867, q50 with 16: page_manager "Not enough watchers" assert in arm A).
    static constexpr size_t MaxFaultJobs = 8;
    /// SHADPS4_FAULT_STAGING_MB=N (default 2): per-job readback staging size.
    static u64 FaultStagingSize() {
        static const u64 size = [] {
            const char* value = std::getenv("SHADPS4_FAULT_STAGING_MB");
            const u64 mb = value ? std::strtoull(value, nullptr, 10) : 0;
            return (mb >= 2 && mb <= 64 ? mb : 2) * 1024 * 1024;
        }();
        return size;
    }

    /// The window a fault downloads: exact for writes, widened and preflighted for reads.
    std::pair<VAddr, u64> ResolveFaultWindow(const Buffer& buffer, VAddr device_addr, u64 size,
                                             bool is_write);

    /// The existing synchronous fault download, after settling overlapping jobs.
    void NoteWriteFaultOverlap(VAddr device_addr, u64 size);
    void ResolveFaultSync(VAddr device_addr, u64 size, bool is_write,
                          Common::ReadbackOffload::Origin origin);

    FaultTicket PrepareFaultDownload(VAddr device_addr, u64 size, bool is_write);

    /// Returns true when the job was vetoed and `allow_retry` asked for another asynchronous
    /// attempt instead of the synchronous fallback (SHADPS4_FAULT_VETO_RETRY): nothing was
    /// published or downloaded, the caller prepares a new job.
    bool FinishFaultDownload(u64 id, VAddr device_addr, u64 size, bool is_write,
                             bool allow_retry = false);

    /// Finalizes every pending job sharing a tracker page with the range, waiting on its tick
    /// here. Must precede any download, publication or tracker clear over that range.
    void SettleOverlappingJobs(VAddr device_addr, u64 size);

    /// Callable from any thread. True if an unfinalized job shares a tracker page with the
    /// range, in which case the caller settles it on the command processor before it lets
    /// the tracker decide anything about that range.
    bool HasOverlappingFaultJob(VAddr device_addr, u64 size);

    /// An invalidation that found nothing left to download still owes its CPU marking.
    void CompleteEmptyWriteFault(VAddr device_addr, u64 size);

    /// Transfer read of a cache buffer into staging, entered into the buffer's barrier chain.
    void RecordFaultCopies(vk::CommandBuffer cmdbuf, Buffer& source, vk::Buffer staging,
                           std::span<const vk::BufferCopy> copies);

    void FinalizeFaultJob(FaultJob& job);

    VerifyResult VerifyFaultJob(FaultJob& job);

    bool PublishFaultJob(FaultJob& job, bool guest_copied = false);

    /// Same walk and staging layout as DownloadBufferMemory, without subtracting anything.
    /// Staging offsets continue from `total_size_bytes`; returns the new total.
    u64 CollectFaultCopies(VAddr device_addr, u64 size, std::vector<FaultCopy>& copies,
                           u64 total_size_bytes = 0);

    std::unique_ptr<Buffer> AcquireFaultStaging();

    void ReleaseFaultStaging(std::unique_ptr<Buffer> staging);

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer, Common::BufferSampleProbe* sample_probe);

    vk::Buffer ReserveDeferredCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    void UploadCopiesInPieces(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                              size_t total_size_bytes, bool test);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);
    /// r13 SHADPS4_DELETE_AT_NEXT_DRAW (default on; =0 off): destroys parked until CommitDeletes.
    static bool DeleteAtNextDraw();
    u64 big_grow_count{}; ///< r15 SHADPS4_BIG_GROW: big joins sized ahead
    std::mutex pending_deletes_mutex;
    std::vector<Common::UniqueFunction<void>> pending_deletes;

    /// FRONT_DERIVE=1: BindVertexBuffers from the front's block (no V# reads, no sort here).
    void BindVertexBuffersDerived(
        const AmdGpu::CpSplit::VertexInputsPayload& vi,
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    PageManager& page_manager;
    /// SHADPS4_DEFERRED_PROTECT scratch, command processor only.
    std::vector<PageManager::ProtectRange> deferred_protects;
    struct DeferredCopy {
        VAddr addr;
        u8* dest;
        u64 size;
        /// SHADPS4_DEFERRED_COPY_CENSUS only: a sampled hash of the part of this copy that lies in
        /// the bound range (hash_off / hash_len inside the copy), and the guest label count when
        /// the copy was recorded. census_kind: 0 = not counted, 1 = bound range, 2 = upload-ahead
        /// page (speculative, nothing consumes it yet), 3 = unreadable at record time.
        u64 hash{};
        u64 label_seq{};
        u32 hash_off{};
        u32 hash_len{};
        u32 census_kind{};
    };
    /// SHADPS4_DEFERRED_COPY_CENSUS: the binding SynchronizeBuffer is serving (CP thread only).
    VAddr census_lo{};
    VAddr census_hi{};
    std::vector<DeferredCopy> deferred_copies;
    /// SHADPS4_READBACK_PRODUCER_STATS=1: last command tick that GPU-wrote each 1 MB slot, to
    /// measure whether readback data is already complete when a fault asks for it.
    std::unique_ptr<u64[]> producer_ticks;
    void NoteGpuWrite(VAddr addr, u64 size);
    u64 diag_producer{};
    std::unordered_map<u64, u64> diag_window_writer;
    std::unordered_map<u64, u64> diag_read_hist;
    /// SHADPS4_READ_FAULT_RIP=1: tick of the last writable binding per 64 KiB window, the windows
    /// read-faulted so far, and per (pgm, addr, size) how often a writable binding covered one.
    std::unordered_map<u64, u64> diag_window_tick;
    std::unordered_map<u64, u64> diag_faulted_windows;
    std::map<std::tuple<u64, u64, u64>, u64> diag_fault_window_writers;
    u64 diag_reads{};
    struct WrittenMemoEntry {
        VAddr addr{};
        u64 size{};
        u64 epoch{};
    };
    std::array<WrittenMemoEntry, 4096> written_memo_table{};
    /// SHADPS4_READ_BIND_MEMO (default off): a read-only, non-texel ObtainBuffer of exactly this
    /// (buffer, range) ran in full at dirty epoch `epoch` with the cache structure at `gen`.
    struct ReadMemoEntry {
        VAddr addr{};
        u64 size{};
        u64 epoch{};
        u64 gen{};
        u32 buffer_index{};
        u64 pb_written{}; ///< SHADPS4_PB_CENSUS: census draw index of the last write
    };
    std::array<ReadMemoEntry, 8192> read_memo_table{};
    /// SHADPS4_LEAN_BIND: FindBufferHinted's bounds hints and LeanReadOk's answer (-1 unknown).
    struct BufferHint {
        VAddr lo{};
        VAddr hi{};
        BufferId id{};
    };
    std::array<BufferHint, 4> buffer_hints{};
    u64 buffer_hint_gen{~0ULL};
    u32 buffer_hint_next{};
    int lean_read_state{-1};
    /// hint hits, hint misses, hint verify mismatches, lean read hits, lean read fallbacks,
    /// lean read verify mismatches
    std::array<u64, 6> lean_stats{};
    void LogLeanStats();

public:
    /// SHADPS4_READ_MEMO_PREFETCH: pulls the READ_BIND_MEMO entry that ObtainBuffer will probe for
    /// this read binding into the cache. The hash must stay the one in ObtainBuffer (a mismatch
    /// only wastes the prefetch).
    void PrefetchReadMemo(VAddr device_addr, u32 size, BufferId buffer_id) const noexcept {
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL) ^
                      u64(buffer_id.index);
        const char* entry = reinterpret_cast<const char*>(
            &read_memo_table[(h ^ (h >> 31)) & (read_memo_table.size() - 1)]);
        _mm_prefetch(entry, _MM_HINT_T0);
        // The 40-byte entry is unaligned: its tail can sit on the next cache line.
        _mm_prefetch(entry + sizeof(ReadMemoEntry) - 1, _MM_HINT_T0);
    }

private:
    /// SHADPS4_WRITTEN_RUN_MEMO (default off): a written ObtainBuffer of exactly this range ran in
    /// full in batch `tick` with these epochs and this many admitted fault jobs.
    struct WrittenRunEntry {
        VAddr addr{};
        u64 size{};
        u64 serial{};
        u64 tick{};
        u64 dirty_epoch{};
        u64 clear_epoch{};
        u64 jobs{};
    };
    std::array<WrittenRunEntry, 4096> written_run_table{};
    /// calls, hits, miss key, miss tick, miss dirty epoch, miss clear epoch, miss jobs, verify bad.
    std::array<u64, 8> written_run_stats{};
    /// Fault jobs admitted so far (monotonic; PrepareFaultDownload, CP thread).
    std::atomic<u64> fault_jobs_admitted{0};
    u64 read_memo_calls{};
    u64 read_memo_verify_bad{};
    u64 buffer_memo_serial{};
    /// SHADPS4_NATIVE_COHERENT state (CP thread).
    bool coherent_live{};
    bool coherent_warned{};
    u64 coherent_head{DirtyLog::Start()};
    std::vector<std::pair<u64, u64>> coherent_runs;
    struct CoherentStats {
        u64 points{}, bounded{}, sweeps{}, runs{}, bytes{}, cyc{}, max_cyc{}, dma_syncs{},
            cp_peeked{};
        u64 skips{}, span_skips{}, created{}, verified{}, v_late{}, v_cp{}, v_bad{}, v_raced{},
            v_bytes{};
        u64 lane_bufs{}, lane_bytes{}, lane_points{}, lane_kept{}; // q22
    } coherent_stats;
    std::vector<std::tuple<BufferId, VAddr, VAddr>> coherent_lane; // q22: the lane pass of one point
    /// [0] BindBuffers (uniform/storage) [1] other callers: calls, hits, key miss, epoch miss,
    /// guard (generation or serial) miss.
    std::array<std::array<u64, 5>, 2> read_memo_stats{};
    /// SHADPS4_READ_SYNC_FAST: checks, clean answers, verify (=2): the full path uploaded after a
    /// clean answer with new dirty-log entries since the check (a guest write raced it) / without
    /// any (a real mismatch).
    std::array<u64, 4> read_sync_fast_stats{};
    /// SHADPS4_RMEMO_CENSUS (notes/buffer-sync-validity-2026-10-07.md). Classes: 0 memo hit,
    /// 1 key miss, 2 epoch miss (shader read bindings), 3 Other callers (vertex / index, no memo).
    /// Paths of a miss: 0 BackClean found no unverified page, 1 walked and uploaded nothing,
    /// 2 uploaded, 3 not the BackClean path and uploaded nothing.
    struct RmemoCensus {
        bool started{};
        u64 seq{};
        u64 dirty_head{};
        u64 gen{};
        u64 drained{}, resets{}, bumps{};
        std::array<u64, 4> calls{};
        std::array<std::array<u64, 4>, 4> path{};
        std::array<std::array<u64, 7>, 4> size_class{};
        std::array<u64, 4> epoch_same_path{}; ///< epoch misses, own Buffer not dirtied, by path
        std::array<u64, 4> timed{}, timed_cyc{};             ///< all of a class (hit: whole call)
        std::array<u64, 4> clean_timed{}, clean_cyc{}, clean_check_cyc{}; ///< path 0 only
        u64 tsc0{};
        std::chrono::steady_clock::time_point t0{};
        u64 next_log{1ULL << 22};
    } rmemo_census;
    u64 RmemoCensusStamp(const Buffer& buffer);
    void RmemoCensusCount(int cls, const Buffer& buffer, bool epoch_same, u64 cyc, bool timed);
    bool AsyncFaultCopies();
    void RecordFaultCopiesAsync(Buffer& source, vk::Buffer staging,
                                std::span<const vk::BufferCopy> copies);
    /// SHADPS4_READBACK_PREBATCH: guest ranges the batch being recorded writes on the GPU.
    std::vector<std::pair<VAddr, VAddr>> batch_writes;
    u64 batch_writes_tick = 0;
    /// SHADPS4_PRECISE_PRODUCER: the GPU write ranges of the last WriteHistory batches.
    static constexpr u64 WriteHistory = 64;
    std::array<std::vector<std::pair<VAddr, VAddr>>, WriteHistory> write_hist;
    /// SHADPS4_ASC_V2_CENSUS: writer kind and pipeline hash per write_hist entry (same index).
    std::array<std::vector<std::pair<u8, u64>>, WriteHistory> write_hist_tag;
    u8 census_kind{};
    u64 census_id{};
    struct WriterTagScope {
        BufferCache& cache;
        u8 kind;
        u64 id;
        bool tolerant; ///< r20 review: the cache's own writes (DMA, joins, fills, image copies) inside
                       ///< a metering dispatch tail are never exposure metering output
        WriterTagScope(BufferCache& cache_, u8 new_kind)
            : cache{cache_}, kind{cache_.census_kind}, id{cache_.census_id},
              tolerant{t_tolerant_writer} {
            cache.census_kind = new_kind;
            cache.census_id = 0;
            t_tolerant_writer = false;
        }
        ~WriterTagScope() {
            cache.census_kind = kind;
            cache.census_id = id;
            t_tolerant_writer = tolerant;
        }
    };
    /// Newest main write-history entry in [first, last] overlapping the copies: its tag, or
    /// kind 255 when the answer is "older than the history" (conservative writer).
    std::pair<u8, u64> NewestMainWriterTag(std::span<const FaultCopy> copies, u64 first,
                                           u64 last) const;
    void CensusCtrlJob(std::span<const FaultCopy> copies, u64 tick, u64 asc_producer,
                       bool asc_done);
    std::array<u64, WriteHistory> write_hist_tick{};
    /// Newest batch tick that wrote any of the copies; 0 when none of the last WriteHistory did.
    u64 PreciseProducer(std::span<const FaultCopy> copies) const;
    /// SHADPS4_READBACK_FAST_LANE: true if a batch in [first, last] GPU-wrote any copied byte.
    bool BatchesWriteAny(std::span<const FaultCopy> copies, u64 first, u64 last) const;
    void NoteBatchWrite(VAddr addr, u64 size);
    bool early_cut_pending{};
    bool BatchWritesAny(std::span<const FaultCopy> copies);
    void NoteReadbackProducer(VAddr addr, u64 size);
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    u64 last_sync_upload_bytes{}; // SHADPS4_BUFFER_TRAFFIC: bytes the last SynchronizeBuffer uploaded
    // SHADPS4_BIND_MEMO_STATS: see SyncUploadBytesTotal and StructGeneration.
    u64 sync_upload_bytes_total{};
    u64 sync_upload_calls{};
    u64 sync_upload_log_tsc{};
    std::atomic<u64> struct_generation{0};

    /**
     * SHADPS4_HOST_MIRROR=1 (plan b prototype, default off): read-only uniform/storage bindings
     * (rasterizer BindBuffers, texel only with SHADPS4_HOST_MIRROR_TEXEL=1, <= SHADPS4_HOST_MIRROR_MAX
     * bytes, default 256 MB; SHADPS4_HOST_MIRROR_RANGE=lo-hi instead mirrors bindings overlapping it) whose
     * range has CPU-modified pages and no GPU-modified pages are bound straight to guest memory:
     * the always-writable physical backing view is imported with VK_EXT_external_memory_host in
     * 1 GB chunks and the binding becomes chunk + offset. No upload, no dirty-bit clear, no
     * re-protect - the pages stay writable, so the game's next write does not fault, and the
     * tracked path (which still sees them CPU-dirty) uploads current data if it is used later.
     * GPU reads then happen at execution time, not at CP time: see SHADPS4_FENCE_AT_GPU.
     */
    std::pair<Buffer*, u32> TryHostMirror(VAddr device_addr, u32 size, bool is_texel);
    Buffer* HostMirrorChunk(u64 index);
    bool MirrorBackingOffset(VAddr device_addr, u32 size, u64& backing);
    struct BackingSpanEntry {
        VAddr start{}, end{};
        u64 backing{};
        u64 generation{~0ULL};
    };
    std::array<BackingSpanEntry, 4> backing_spans{};
    u32 backing_span_next{};
    std::atomic<u64> backing_span_generation{0};
    struct HostMirrorSlot {
        std::unique_ptr<Buffer> buffer;
        bool failed{};
    };
    static constexpr u32 HostMirrorChunkBits = 30;
    std::vector<HostMirrorSlot> host_mirror_chunks;
    bool host_mirror{};
    u32 host_mirror_max{};
    u64 hm_calls{}, hm_hits{}, hm_clean{}, hm_gpu{}, hm_nobacking{}, hm_straddle{}, hm_bytes{};
    u64 hm_toobig{}, hm_max_seen{}, hm_outside{}, hm_samples{}, hm_image{};
    bool host_mirror_texel{}; // SHADPS4_HOST_MIRROR_TEXEL=1: formatted buffers too
    VAddr host_mirror_lo{}, host_mirror_hi{}; // SHADPS4_HOST_MIRROR_RANGE
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer, Common::StableSlotCapacity> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 vram_peak_vma = 0;    ///< v1.0.2 VideoMemory log line: VMA device-local peak
    u64 vram_peak_driver = 0; ///< and the driver's process-usage peak
    std::chrono::steady_clock::time_point vram_last_log{};
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    GpuRangeSet gpu_modified_ranges;
    SplitRangeMap<BufferId> buffer_ranges;
    /// Guards buffer_ranges between the command processor (Register/Unregister edit the tree) and
    /// guest threads (write faults read it in IsRegionRegistered). Other readers run on the command
    /// processor. SHADPS4_BUFFER_RANGES_LOCK=0 disables it (A/B only).
    mutable std::shared_mutex buffer_ranges_mutex;
    PageTable page_table;
    std::array<FaultJob, MaxFaultJobs> fault_jobs;
    std::vector<std::unique_ptr<Buffer>> fault_staging_pool;
    u32 fault_staging_created = 0;
    u64 next_fault_job_id = 1;
    /// Jobs admitted and not yet finalized. Written by the command processor only, under
    /// fault_window_mutex together with fault_windows; guests read it without the mutex as
    /// the no-pending fast path of InvalidateMemory.
    std::atomic<u32> pending_fault_jobs{0};
    std::mutex fault_window_mutex;
    std::array<FaultWindow, MaxFaultJobs> fault_windows{};
    Common::ReadbackUnmapGate fault_mapping_gate;
    /// Guest threads between entering the offload path and consuming their result.
    std::atomic<u32> fault_guest_waiters{0};
    /// f2: the prefetch waiter (last members: stopped and joined first).
    std::jthread pf_thread;
    std::jthread pf_retry_thread; ///< r18: the retry items' waiter
};

/// f2: the guest address of the last ASC RELEASE_MEM the back executed (back thread only).
inline VAddr g_pf_label_addr = 0;

} // namespace VideoCore
