// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <vector>
#include "g4_packet_hazards.h"
#include "g4_graphics_control.h"
#include "g4_readback_writers.h"
#include "g4_sparse_residency.h"
#include "g4_texture_negative_index.h"
#include "buffer_read_memo.h"
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/feedback_writer_probe.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/buffer_cache/upload_worker.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

/// A GPU queue the command processor records work for: the graphics queue, or the compute queue
/// running the work of the guest's asynchronous compute queues alongside it (asc_queue). Each
/// has its own command buffers and timeline (scheduler), barrier tracking and staging memory
/// (runtime) and stream buffer.
struct QueueContext {
    Vulkan::Scheduler* scheduler;
    Vulkan::Runtime* runtime;
    StreamBuffer* stream_buffer;
    u32 index; ///< 0 graphics, 1 compute (RegionManager::CrossQueueAccess).
};

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer of the active queue.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return *ActiveQueuePtr()->stream_buffer;
    }

    /// The queue the bindings, uploads and copies are recorded for (asc_queue): nullptr for the
    /// graphics queue. Command processor thread only.
    void SetQueue(QueueContext* context) noexcept {
        QueueContext* selected = context != nullptr ? context : &graphics_queue;
        if (cp_split_enabled) {
            thread_queue_selection = {this, selected};
        } else {
            active = selected;
        }
    }
    // Startup-only gate. The split worker must select its queue before touching BufferCache.
    void SetCpSplitEnabled(bool enabled) noexcept { cp_split_enabled = enabled; }
    /// Conservative version of renderer-visible guest mappings. Any map, registration or
    /// unmap invalidates plans prepared against an older backing, including unrelated ranges.
    [[nodiscard]] u64 MappingGeneration() const noexcept {
        return mapping_generation.load(std::memory_order_acquire);
    }
    void NoteMappingChange(VAddr address, u64 size) noexcept;
    /// Stage binding memo inputs: arena handle changes and GPU writes published so far, and
    /// whether a separate compute queue (cross queue tracking) exists.
    [[nodiscard]] u64 ArenaBindingGeneration() const noexcept {
        return arena_binding_generation.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 GpuWriteSeq() const noexcept;
    [[nodiscard]] bool HasComputeQueue() const noexcept {
        return compute_queue != nullptr;
    }
    // Prepare only: resolve a guest range to host-shadow physical spans so VA aliases conflict.
    // Returns false for gaps, remaps during the snapshot, or an unbounded span list.
    [[nodiscard]] bool PlanGuestAccess(VAddr address, u64 size, bool write,
                                       u64 prepared_mapping_generation,
                                       std::vector<iss::g4::PacketResourceAccess>& out) const;
    // Call before a direct guest write (and on CPU fault) so an unpublished packet
    // cannot retain a snapshot of bytes that are about to change.
    void InvalidatePreparedPacketAccess(VAddr address, u64 size);
    // Coarse reservations for shared non-guest allocations. They avoid false negatives
    // until a finer range proof is available for GDS and the BDA page table.
    static void PlanGdsAccess(bool write,
                              std::vector<iss::g4::PacketResourceAccess>& out) {
        out.push_back({iss::g4::GdsBackingKey, 0, 1, write});
    }
    static void PlanBdaAccess(std::vector<iss::g4::PacketResourceAccess>& out) {
        out.push_back({iss::g4::BdaBackingKey, 0, 1, false});
    }
    struct PreparedGuestBufferBinding {
        const Buffer* buffer{};
        u64 offset{};
        u64 size{};
        VAddr guest_address{};
        bool write{};
        u64 residency_tick{};
        u64 graphics_bda_tick{};
        u64 mapping_generation{};
        std::array<u64, 2> cpu_versions{};
        std::array<u64, 2> writer_ticks{};
        u64 writer_write_seq{};
        bool gpu_owned{};
    };
    [[nodiscard]] iss::g4::PacketReserveResult TryReservePacket(
        iss::g4::PacketLeaseKey key,
        std::span<const iss::g4::PacketResourceAccess> accesses,
        u64 prepared_mapping_generation) {
        std::scoped_lock lock{coherence_mutex};
        return packet_hazards.TryReserve(key, accesses, prepared_mapping_generation,
                                         MappingGeneration());
    }
    [[nodiscard]] bool RevalidatePacket(u64 lease_id, bool guest_inputs_stable) const {
        return packet_hazards.Revalidate(lease_id, MappingGeneration(), guest_inputs_stable);
    }
    [[nodiscard]] std::vector<iss::g4::QidTick> PacketDependencies(u64 lease_id) const {
        return packet_hazards.Dependencies(lease_id);
    }
    [[nodiscard]] bool PublishPacket(u64 lease_id, iss::g4::QidTick token,
                                     bool guest_inputs_stable) {
        std::scoped_lock lock{coherence_mutex};
        return packet_hazards.Publish(lease_id, token, MappingGeneration(),
                                      guest_inputs_stable);
    }
    [[nodiscard]] bool PublishPacketWithBufferWrites(
        u64 lease_id, iss::g4::QidTick token, bool guest_inputs_stable,
        std::span<const PreparedGuestBufferBinding> bindings,
        std::span<const iss::g4::NegativeImageSnapshot> negative_images = {});
    [[nodiscard]] bool CancelPacket(u64 lease_id) {
        return packet_hazards.CancelUnpublished(lease_id);
    }
    [[nodiscard]] bool FinishPacketRecord(u64 lease_id) {
        return packet_hazards.FinishRecord(lease_id);
    }

    [[nodiscard]] const QueueContext& ActiveQueue() const noexcept {
        return *ActiveQueuePtr();
    }
    // Stage 2 diagnostic identity. The caller restores the previous value after a serial scope.
    struct SerialProducerIdentity {
        iss::g4::CpOwner owner{iss::g4::CpOwner::Graphics};
        u32 qid{};
        u64 generation{};
    };
    SerialProducerIdentity SetSerialProducerIdentity(iss::g4::CpOwner owner, u32 qid,
                                                      u64 generation) noexcept {
        return std::exchange(serial_producer_identity, {owner, qid, generation});
    }
    void RestoreSerialProducerIdentity(SerialProducerIdentity previous) noexcept {
        serial_producer_identity = previous;
    }

    [[nodiscard]] bool OnComputeQueue() const noexcept {
        return ActiveQueuePtr()->index != 0;
    }

    /// Cross queue dependencies are tracked while the compute queue is in use (asc_queue). The
    /// compute scheduler waits for the upload worker like the graphics one.
    void SetCrossQueueTracking(QueueContext* compute);

    /// asc_queue: a submission of the compute queue waits for the sparse binds of the arena
    /// done so far (issued on the graphics queue, which waits for them in order).
    void WaitArenaBinds(Vulkan::SubmitInfo& info);
    /// The highest sparse bind ticket handed to the graphics queue. A ticket alone
    /// is insufficient for a BDA page-table copy; that has a scheduler dependency.
    [[nodiscard]] u64 PublishedResidencyTick() const noexcept {
        return residency_publication.SubmittedTick();
    }
    [[nodiscard]] u64 PublishedBdaGraphicsTick() const noexcept {
        return published_bda_g_tick.load(std::memory_order_acquire);
    }
    void AddResidencyWait(Vulkan::SubmitInfo& info, u64 required_tick);

    /// Cross queue dependencies (asc_queue): the active queue accesses the GDS buffer (its own
    /// tracking: the GDS is not guest memory).
    void NoteGdsAccess(bool is_written);

    /// Cross queue dependencies (asc_queue): the active queue reads the BDA page table.
    void NoteBdaAccess();

    /// Cross queue dependencies (asc_queue): the active queue accesses a range of guest memory
    /// through the GPU copy, in its current command buffer: a wait on the other queue is added
    /// to the current session when needed (and the graphics session is flushed first when the
    /// compute queue depends on it).
    void NoteQueueAccess(VAddr device_addr, u64 size, bool is_written);

    /// Cross queue dependencies (asc_queue): the active queue uploads the range to the GPU copy
    /// in its current command buffer.
    void NoteQueueUpload(VAddr device_addr, u64 size);

    /// Waits for the GPU to finish the work of every queue submitted so far (readbacks on the
    /// command processor thread).
    void FinishAllQueues();

    /// Submits the session of the compute queue if it recorded commands, after the graphics
    /// work it waits for has been handed to the driver (asc_queue).
    void FlushComputeQueue();

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    /// ISS_UNIFIED_MEMORY (unified_memory.h): whether the range lies in the imported guest heap
    /// the GPU uses in place (false until it is imported).
    [[nodiscard]] bool IsUnified(VAddr address, u64 size) const noexcept {
        return address >= unified_base && size != 0 && address + size <= unified_end;
    }
    /// ISS_NATIVE_RECORDER: the imported heap's buffer and guest base (null before import).
    [[nodiscard]] const Buffer* UnifiedBuffer() const noexcept {
        return unified_buffer.get();
    }
    [[nodiscard]] VAddr UnifiedBase() const noexcept {
        return unified_base;
    }
    /// Whether the GPU copy of the range is current (uploaded, not written by the CPU since,
    /// no unprotected hot page).
    [[nodiscard]] bool IsRegionCpuClean(VAddr address, u64 size) const noexcept;
    /// Whether the GPU was given a write into the unified range (since the heap was imported).
    [[nodiscard]] bool UnifiedTouched(VAddr address, u64 size) const noexcept;
    /// Command processor reads of guest memory in the unified heap: the GPU writes recorded
    /// there before are submitted (behind a host barrier) and waited for.
    void SyncUnifiedRead(VAddr address, u64 size);
    /// Makes the GPU writes recorded so far visible to the host (before a deferred fence).
    void UnifiedHostBarrier();

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    /// guest_rip: the faulting instruction of a guest read (readback_relaxed_sites), 0 if none.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false,
                    u64 guest_rip = 0);

    /// The command processor wrote guest memory directly (constant RAM dumps, WRITE_DATA, DMA
    /// done on the CPU): pages written without a fault are uploaded again.
    void OnGpuThreadWrite(VAddr device_addr, u64 size);

    /// Called at guest fence points: downloads what the GPU wrote in memory the guest read back
    /// recently, so its next reads find the data ready instead of waiting for the GPU.
    void PrefetchReadbacks();

    /// Writes the pending readbacks the GPU has finished. Cheap: only checks the GPU tick known
    /// from the last submission or wait.
    void RetireReadbacks();

    /// Tuning::bind_prefetch: prefetches what ObtainBuffer reads first for a read only binding
    /// of the range (its stream copy or sync memo entry, its tracking region), command processor.
    void PrefetchLookup(VAddr device_addr, u64 size) const noexcept;

    /// Tuning::bind_prefetch: prefetches the guest memory a small read only binding of the range
    /// will copy into the stream buffer, unless its last copy is reused or its pages are bound
    /// in place (stream_static). Called a little before the binding, once its PrefetchLookup
    /// entries are cached (the copies read memory guest threads just wrote, in their caches).
    void PrefetchStreamSource(VAddr device_addr, u64 size) noexcept;

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false,
                                                             bool* known_unwritten = nullptr);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// ISS_NATIVE_PASSES_SYNC_READS: a read of a native segment, recorded where its arena has it,
    /// whose pages the CPU wrote since: the guest data uploaded in place (as ObtainBuffer does for
    /// ranges it does not stream), returns where the range is now.
    [[nodiscard]] std::pair<const Buffer*, u64> SyncReadInPlace(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Whether the GPU copy of the region has data guest memory lacks: GPU modified, or bytes
    /// the GPU wrote in pages whose GPU state a guest write discarded (write merge). Such memory
    /// must be read and written on the GPU.
    [[nodiscard]] bool HasGpuData(VAddr addr, size_t size);

    // Native material proof, serial graphics only. Tokens never retain stream copies.
    // Caller supplies an ordinary read-only, unformatted binding and its clamped size.
    struct PreparedReadToken {
        VAddr address{};
        u32 size{};
        const Buffer* buffer{};
        u64 offset{}, writer_sequence{};
        std::array<u64, 2> cpu_versions{}, mapping_versions{};
    };
    [[nodiscard]] std::optional<PreparedReadToken> CapturePreparedRead(
        VAddr address, u32 size, const Buffer* buffer, u64 offset);
    [[nodiscard]] bool ValidatePreparedRead(const PreparedReadToken& token);
    // Call only after every material guard passed; preserves ordinary read tracking.
    [[nodiscard]] bool CommitPreparedRead(const PreparedReadToken& token);

    /// Frame x-ray: the arena holding the range and the range's offset in it, null when its
    /// pages are not resident. Nothing is synchronized or changed.
    [[nodiscard]] std::pair<const Buffer*, u64> PeekArena(VAddr device_addr, u32 size);

    struct PublishedArena {
        const Buffer* arena{};
        u64 offset{};
        u64 residency_tick{};
        // Highest G scheduler tick that recorded a BDA page-table copy.
        u64 graphics_bda_tick{};
        explicit operator bool() const noexcept { return arena != nullptr; }
    };
    /// Read-only C preflight. Returns empty while a required bind is still pending.
    [[nodiscard]] PublishedArena PeekPublishedArena(VAddr device_addr, u32 size) const;
    struct AscArenaPreflight {
        u64 graphics_tick{};
        u64 mapping_generation{};
    };
    // G safe point only. Materializes a CPU-owned guest range in an arena before C
    // packet publication; rejects GPU-owned bytes and cross-page arena migration.
    [[nodiscard]] std::optional<AscArenaPreflight> PrepareArenaForAsc(
        VAddr device_addr, u32 size);
    // Fast prepared path: only an existing, fully resident GPU copy with clean CPU
    // pages. Anything requiring upload, stream copy or sparse allocation fails closed.
    [[nodiscard]] std::optional<PreparedGuestBufferBinding> PrepareGuestBufferBinding(
        VAddr device_addr, u32 size, bool is_written) const;
    [[nodiscard]] bool RevalidateGuestBufferBinding(
        const PreparedGuestBufferBinding& plan) const;
    // Exact queue/tick/qid/generation provenance for a C safe-point readback.
    // Unknown or evicted history fails closed; this is not authorization to copy bytes.
    [[nodiscard]] std::optional<iss::g4::ReadbackWriterSnapshot>
    CaptureG4ReadbackWriters(VAddr address, u64 size) const;
    [[nodiscard]] bool RevalidateG4ReadbackWriters(
        const iss::g4::ReadbackWriterSnapshot& snapshot) const;
    enum class G4ReadbackPreflightState : u8 {
        Pending, Stale, NeedsSeal, Submitted, Unsupported,
    };
    struct G4ReadbackPreflight {
        iss::g4::GraphicsControlMailbox::Request control{};
        std::shared_ptr<std::atomic<G4ReadbackPreflightState>> result{};
    };
    // Nonblocking C->G metadata control. Submitted means producer timelines are
    // submitted; it does not mean bytes were downloaded. C must yield until the
    // control is Done, then revalidate before a separate G-owned copy.
    [[nodiscard]] G4ReadbackPreflight EnqueueG4ReadbackPreflight(
        iss::g4::ReadbackWriterSnapshot snapshot);
    enum class G4FeedbackCopyState : u8 {
        Pending, Stale, NeedsSeal, Unsupported, Copied,
    };
    struct G4FeedbackCopyData {
        std::atomic<G4FeedbackCopyState> state{G4FeedbackCopyState::Pending};
        std::array<u8, 8> bytes{};
    };
    struct G4FeedbackCopy {
        iss::g4::GraphicsControlMailbox::Request control{};
        std::shared_ptr<G4FeedbackCopyData> result{};
    };
    // Exact 8-byte feedback copy into an owning result. Does not write guest
    // memory or clear dirty state. Only usable after C revalidates provenance.
    [[nodiscard]] G4FeedbackCopy EnqueueG4FeedbackCopy(
        iss::g4::ReadbackWriterSnapshot snapshot);

    /// Frame x-ray: the tracking state of the range's pages, as text.
    [[nodiscard]] std::string DescribePages(VAddr device_addr, u32 size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info, bool graphics_submit = true);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }
    void NoteArenaBindLocked(const Buffer* arena);

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    /// Whether every block of the range is backed by memory (blocks never stop being resident).
    bool IsResident(u64 first_block, u64 last_block) const;
    void MarkResident(u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    /// GPU modified ranges come as many small scattered islands, so a readback downloads a
    /// window around the requested address.
    std::pair<VAddr, VAddr> GetReadbackWindow(const Buffer* arena, VAddr device_addr, u64 size);

    /// A download from the GPU that was recorded and submitted without waiting for it. The guest
    /// thread that requested it waits for the GPU, then the data is written to guest memory.
    struct PendingReadback {
        u64 tick{};
        u64 record_seq{}; ///< GPU writes numbered above it came after the download was recorded.
        Vulkan::StagingBufferRef download{};
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        VAddr arena_base{};
        VAddr window_start{};
        VAddr window_end{};
        vk::Buffer arena_handle{};
        bool prefetch{};
        bool on_readback_queue{}; ///< tick is on the readback queue, else the graphics queue.
        /// readback_relaxed_sites: pages a guest read was let through while the download was in
        /// flight (no longer GPU modified): they get the new data too if unchanged since.
        boost::container::small_vector<VAddr, 4> relaxed_pages;
    };

    /// Copies GPU data to host memory on the second queue of the graphics family. A copy waits
    /// only for the graphics submission that last wrote its data, not for the work recorded
    /// after it, which is what a readback recorded in the graphics command buffer waits for.
    class ReadbackQueue {
    public:
        struct Job {
            vk::Buffer src;
            vk::Buffer dst;
            std::span<const vk::BufferCopy> copies;
        };

        explicit ReadbackQueue(const Vulkan::Instance& instance);
        ~ReadbackQueue();

        /// A timeline semaphore tick the copies wait for (the queue that wrote the data).
        struct Wait {
            vk::Semaphore semaphore;
            u64 tick;
        };

        /// Submits the copies once the timelines reach their ticks. Returns the readback
        /// semaphore tick signaled when the data is visible to the host.
        u64 Submit(std::span<const Job> jobs, std::span<const Wait> waits);

        Vulkan::Semaphore& Semaphore() noexcept {
            return semaphore;
        }

        bool IsFree(u64 tick) {
            if (semaphore.IsFree(tick)) {
                return true;
            }
            semaphore.Refresh();
            return semaphore.IsFree(tick);
        }

    private:
        const Vulkan::Instance& instance;
        Vulkan::Semaphore semaphore;
        Vulkan::CommandPool command_pool;
    };

    /// True when readbacks go through the readback queue (available and enabled).
    bool UseReadbackQueue() const;
    void AssertGraphicsReadbackOwner() const;

    /// The GPU work that last wrote the GPU modified pages of a window: the tick of each queue
    /// (0 graphics, 1 compute; 0 if none) and where that work is.
    struct WindowWriters {
        std::array<u64, 2> ticks{};
        bool submitted{true}; ///< Every writer is handed to the driver (readback queue usable).
        bool done{true};      ///< Every writer is done on the GPU.
        bool current{};       ///< A writer is in a session being recorded.
    };
    WindowWriters WindowWritersOf(VAddr start, u64 size, bool* precise = nullptr);
    /// The waits of a readback queue copy of a window.
    boost::container::static_vector<ReadbackQueue::Wait, 3> ReadbackWaits(
        const WindowWriters&, u64 residency_tick = 0);
    /// A copy of the window recorded in the graphics session: it waits for the compute writer.
    void WaitComputeWriter(const WindowWriters& writers);

    bool IsReadbackDone(bool on_readback_queue, u64 tick);

    /// Registers the window as pending and collects the copies of its GPU modified ranges into a
    /// download buffer, without recording them. Returns false when there is nothing to download
    /// (the window stays registered).
    bool CollectReadback(const Buffer* arena, PendingReadback& readback);

    /// Collects the window and records its copies in the graphics command buffer.
    bool RecordReadback(const Buffer* arena, PendingReadback& readback);

    /// Remembers a window the guest read back, candidate for prefetching.
    void NoteHotWindow(VAddr start, VAddr end);

    struct HotWindow {
        VAddr start;
        VAddr end;
        u64 last_read_ns;
        u64 prefetch_tick; ///< Tick of the last prefetch of the window, at most one in flight.
        u64 prefetch_seq;  ///< Write sequence the last prefetch covered (readback_keep_ranges).
        bool prefetch_on_queue; ///< prefetch_tick is on the readback queue.
    };

    /// Downloads the GPU modified memory around a request and waits for it (command processor).
    void ReadbackSync(VAddr device_addr, u64 size, bool is_write);
    /// Paused G-owner path: submit a full-window copy after its exact C producer tick and
    /// wait only for readbacks overlapping this request. Never submits C's scheduler.
    void ReadbackG4Window(VAddr device_addr, u64 size, bool is_write);
    /// Tuning::readback_keep_ranges, see CollectReadback.
    [[nodiscard]] bool KeepRanges() const;

    /// Records and submits the download of the GPU modified memory around a request. Returns the
    /// tick to wait for on wait_semaphore before calling CompleteReadbacks, or 0 when there is
    /// nothing to wait for.
    u64 BeginReadback(VAddr device_addr, u64 size, bool is_write,
                      Vulkan::Semaphore*& wait_semaphore);

    /// Writes the pending readbacks the GPU has finished to guest memory: in any order, except
    /// after an unfinished earlier readback of the same window. With wait, waits for all of them
    /// and writes them in submission order.
    void CompleteReadbacks(bool wait);

    /// Writes the pending readback at this position of pending_readbacks to guest memory.
    void FinishReadback(size_t index);

    /// Guest thread: waits for a readback tick (reported if it takes seconds).
    void WaitReadback(VAddr device_addr, Vulkan::Semaphore* semaphore, u64 tick);

    /// Command processor: the in flight readback to wait for before the range can be written,
    /// nullptr if none (see CompleteReadbacks).
    const PendingReadback* FirstUnfinishedReadback(VAddr device_addr, u64 size);

    /// Whether a guest read by this instruction is let through without waiting for the GPU
    /// (readback_relaxed_sites).
    [[nodiscard]] bool IsRelaxedSite(u64 guest_rip) const;

    /// Command processor: the download of the window of a relaxed read is recorded (or the one in
    /// flight is used), then its page becomes readable at once with the data guest memory has.
    void BeginRelaxedReadback(VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    /// Tests whether the GPU could use the guest memory around this address in place (import of
    /// host memory), through the guest mapping and through its backing, and logs the results.
    void ProbeHostImport(VAddr address);
    u32 probed_host_import{};

    /// The scheduler, runtime, staging pool and stream buffer of the active queue (asc_queue).
    [[nodiscard]] Vulkan::Scheduler& Sched() const noexcept {
        return *ActiveQueuePtr()->scheduler;
    }
    [[nodiscard]] Vulkan::Runtime& Rt() const noexcept {
        return *ActiveQueuePtr()->runtime;
    }
    [[nodiscard]] Vulkan::StagingBufferPool& Staging() const noexcept;
    [[nodiscard]] StreamBuffer& Stream() const noexcept {
        return *ActiveQueuePtr()->stream_buffer;
    }

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler; ///< Graphics queue.
    Vulkan::Runtime& runtime;     ///< Graphics queue.
    Vulkan::StagingBufferPool& staging_pool; ///< Graphics queue.
    QueueContext graphics_queue;
    QueueContext* active;
    struct ThreadQueueSelection {
        const BufferCache* cache{};
        QueueContext* queue{};
    };
    static thread_local ThreadQueueSelection thread_queue_selection;
    bool cp_split_enabled{};
    std::atomic<u64> mapping_generation{1};
    iss::g4::PacketHazards packet_hazards{};
    [[nodiscard]] QueueContext* ActiveQueuePtr() const noexcept {
        return cp_split_enabled && thread_queue_selection.cache == this
                   ? thread_queue_selection.queue : active;
    }
    QueueContext* compute_queue{}; ///< Set while asc_queue is in use (cross queue tracking).
    /// Cross queue dependencies of the GDS buffer: last tick of each queue reading and writing;
    /// of the BDA page table: last tick of each queue writing it.
    std::array<u64, 2> gds_read_tick{};
    std::array<u64, 2> gds_write_tick{};
    std::array<u64, 2> bda_write_tick{};
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;
    SerialProducerIdentity serial_producer_identity{};
    FeedbackWriterProbe feedback_writer_probe{}; // passive, opt-in only
    iss::g4::ReadbackWriterLedger<> readback_writers{}; // under coherence_mutex

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    // Coherent guest GPU byte ownership. Never hold across guest reads, driver calls or waits.
    mutable std::mutex coherence_mutex;
    // Only populated by the opt-in native API; mapping changes invalidate intersecting
    // 16 MB regions, never unrelated prepared resources. Protected by coherence_mutex.
    std::vector<std::pair<u64, u64>> prepared_read_mapping_versions;
    u64 coherence_write_generation{1};
    [[nodiscard]] std::vector<std::pair<VAddr, VAddr>> SnapshotGpuRanges(VAddr address,
                                                                          u64 size,
                                                                          u64* write_generation = nullptr) const;
    [[nodiscard]] bool GpuRangesIntersect(VAddr address, u64 size) const;
    void SubtractGpuRange(VAddr address, u64 size);
    void SubtractGpuRangeIfUnchanged(VAddr address, u64 size, u64 write_generation);
    [[nodiscard]] u64 PublishGpuWrite(VAddr address, u64 size, u64 tick, u32 queue);
    /// ISS_PUBLISH_REPEAT_SKIP: the last publishes (under coherence_mutex), and how many times
    /// ranges were subtracted from gpu_modified_ranges (a memo entry is only valid without one).
    struct PublishMemo {
        VAddr address;
        u64 size, tick;
        u32 queue;
        u64 subtracts;
    };
    std::array<PublishMemo, 256> publish_memo{};
    u64 gpu_ranges_subtracts{};
    /// ISS_PREFETCH_PER_SUBMIT: the tick of the last readback prefetch scan.
    u64 prefetch_scan_tick{~0ULL};
    RangeSet pending_download_ranges; ///< readback_keep_ranges: downloads in flight.
    std::deque<PendingReadback> pending_readbacks;
    std::unique_ptr<ReadbackQueue> readback_queue;
    boost::container::static_vector<HotWindow, 64> hot_windows;

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    /// Recent copies of small read only buffers in the stream buffer, reused while nothing can
    /// have changed them: same command buffer (the stream buffer keeps its data until that
    /// command buffer completes), same upload epoch (no guest CPU write released since) and no
    /// memory written by the emulator since. Command processor thread only.
    struct StreamCopy {
        VAddr address{};
        u64 tick{};
        u64 epoch{};
        u64 host_write_gen{};
        u64 offset{};
        u32 size{};
        u32 queue{}; ///< The stream buffer (queue) holding the copy; ticks are per queue.
    };
    static constexpr size_t NumStreamCopies = 4096;
    std::array<StreamCopy, NumStreamCopies> stream_copies{};
    u32 stream_clean_sample{};
    // Serial small-buffer arena reuse only. Stream copies and cross-queue/G4
    // resource lifetime are deliberately outside this contract.
    iss::buffers::SmallReadMemo<const Buffer*> small_read_memo;

    /// Sync memo (Tuning::sync_memo): read only ranges found up to date in the GPU copy of guest
    /// memory, with the CPU modification versions of their regions at that time. Command
    /// processor thread only.
    struct SyncMemo {
        VAddr address{};
        const Buffer* arena{};
        std::array<u64, 2> versions{};
        u32 size{};
    };
    static constexpr size_t NumSyncMemos = 4096;
    std::array<SyncMemo, NumSyncMemos> sync_memos{};
    struct LargeReadMemo {
        const Buffer* arena{};
        VAddr address{};
        u64 size{}, epoch{}, host_write_gen{};
        u32 queue{};
        bool valid{};
        std::vector<u64> versions;
    };
    std::array<LargeReadMemo, 16> large_read_memos;
    struct ComputeScratch {
        std::array<StreamCopy, NumStreamCopies> stream_copies{};
        u32 stream_clean_sample{};
        std::array<SyncMemo, NumSyncMemos> sync_memos{};
        std::array<LargeReadMemo, 16> large_read_memos;
        std::vector<std::pair<u64, u64>> deferred_pages;
    } compute_scratch{};
    [[nodiscard]] bool ComputeScratchActive() const noexcept {
        return cp_split_enabled && ActiveQueuePtr()->index == 1;
    }

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    // Address-space arena migration creates a new Vulkan handle for the same
    // physical storage. Cached bindings must switch handles with the live table.
    std::atomic<u64> arena_binding_generation{1};
    std::deque<Buffer> arenas;
    mutable std::mutex residency_mutex;
    std::vector<ArenaBinds> pending_binds;
    iss::g4::SparseResidencyPublication<const Buffer*> residency_publication;
    std::atomic<u64> published_bda_g_tick{};
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;
    /// One bit per block of the address space, set once the block is resident: answers the
    /// residency check of every binding without searching resident_ranges.
    std::vector<u64> resident_bits;
    std::vector<u64> resident_full; ///< A bit per word of resident_bits that is all ones.

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};

    /// Upload worker (async_uploads) and the deferred protections of the upload being prepared.
    std::unique_ptr<UploadWorker> upload_worker;
    std::vector<std::pair<u64, u64>> deferred_pages;

    /// ISS_UNIFIED_MEMORY: the imported heap [unified_base, unified_end) (both 0 until imported),
    /// its buffer, and per 64 KB chunk the tick of the last GPU write recorded into it (0: never).
    static constexpr u32 UNIFIED_CHUNK_SHIFT = 16;
    bool EnableUnified();
    std::pair<const Buffer*, u64> ObtainUnified(VAddr device_addr, u32 size, bool is_written,
                                                bool* known_unwritten);
    VAddr unified_base{};
    VAddr unified_end{};
    bool unified_failed{};
    std::unique_ptr<Buffer> unified_buffer;
    vk::DeviceMemory unified_memory{};
    std::vector<u64> unified_chunk_tick;
};

} // namespace VideoCore
