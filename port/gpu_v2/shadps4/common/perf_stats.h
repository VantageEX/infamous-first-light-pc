// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "common/logging/log.h"
#include "common/types.h"

#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentProcessorNumber(void);
#else
#include <sched.h>
#endif

/**
 * Lightweight performance counters used to find what causes slow frames.
 *
 * Counters are accumulated by the emulator threads and drained on every guest flip. The
 * presenter logs a summary once per second, plus a breakdown of every frame slower than
 * SlowFrameMs, so a slowdown can be attributed to shader compilation, GPU synchronization,
 * readbacks, page faults or uploads.
 */
namespace Common::Perf {

enum Counter : u32 {
    ShaderCompiles,
    ShaderCompileNs,
    PipelineCreates,
    PipelineCreateNs,
    PipelineStallNs,
    PipelineSkips,
    GpuWaits,
    GpuWaitNs,
    GpuBusyNs,
    GpuSpinNs,
    ReadbackReads,
    ReadbackWrites,
    ReadbackDiscards,
    ReadbackStallNs,
    ReadbackQueueNs,
    ReadbackSyncNs,
    ReadbackBytes,
    ReadbackAsync,
    ReadbackAsyncWaitNs,
    ReadbackAsyncRetries,
    ReadbackPrefetches,
    ReadbackPrefetchFlushes,
    ReadbackPrefetchHits,
    ReadbackDiscardsBlocked,
    FaultReads,
    FaultWrites,
    ProtectCalls,
    ProtectNs,
    BufferUploads,
    BufferUploadBytes,
    BufferUploadNs,
    ImageRefreshes,
    ImageRefreshNs,
    ImageUploadBytes,
    ImageCreates,
    BarrierCalls,
    BarrierSubresources,
    BarriersEmitted,
    BarrierFastPath,
    BarrierMemoHit,
    BarrierDirtyPath,
    HotPagePromotions,
    HotPageDemotions,
    UploadEpochs,
    ReadbackQueueSubmits,
    ReadbackQueueFlushes,
    // Sources of upload epochs, see VideoCore::NextUploadEpoch.
    EpochGfxSubmits,
    EpochAscSubmits,
    EpochCpuWaits,
    EpochSemaphoreWaits,
    EpochRewinds,
    // Waits that did not start an epoch: label written by the command processor, video out.
    EpochCpWaitsSkipped,
    EpochVoWaitsSkipped,
    // Small read only buffers copied in the stream buffer at each binding.
    StreamCopies,
    StreamCopyBytes,
    StreamReuses,
    StreamReuseBytes,
    StreamCleanCopies,
    StreamArenaBinds,
    // Clean ranges copied anyway because reading them in place needed a barrier.
    StreamBarrierAvoided,
    // Hot pages compared with their last upload at a new epoch.
    HotPageHashSkips,
    HotPageHashUploads,
    HotPageHashMismatches,
    HostWrites,
    // Written buffer bindings the GPU already owned (no locking needed).
    WrittenRangeSkips,
    // Texture cache invalidations (guest write faults, GPU writes) of pages without images,
    // answered without its lock, and those that took it.
    TexInvalidateSkips,
    TexInvalidateLocked,
    // Pipeline barriers of buffer memory (each ends the render pass), and those caused by an
    // upload or copy overlapping earlier accesses.
    BufferBarrierFlushes,
    CopyBarrierFlushes,
    // Guest frames not yet processed by the command processor, summed at each frame end, and
    // time guest frame ends waited for it (max_gpu_lag_frames).
    GpuLagFrames,
    GpuLagSamples,
    GpuLagWaitNs,
    // Asynchronous compute submissions started by the command processor, the time they waited
    // to start since the guest submitted them, the command processor time spent in them, and the
    // times the graphics queue gave way to them (asc_yield_packets).
    AscStarts,
    AscStartDelayNs,
    AscBusyNs,
    GfxYields,
    // Guest occlusion queries counted by the GPU (real_occlusion), the Vulkan queries they used
    // (one per rendering scope and per dump of an overlapping query), the counting segments that
    // could not start, the queries reported with emulated counters (all samples visible), and the
    // command processor waits for counted results satisfied without waiting for the GPU (and
    // those that had to read the results back).
    OcclusionQueries,
    OcclusionSegments,
    OcclusionDropped,
    OcclusionEmulated,
    OcclusionWaitsSkipped,
    OcclusionWaitsRead,
    // Command processor waits on memory satisfied whatever its value (sync points), skipped
    // without reading it.
    WaitsTrivial,
    // Sync memo: read only buffer bindings found up to date without walking the page tracking
    // tables, and those that had to (then remembered).
    SyncMemoHits,
    SyncMemoMisses,
    // Shader permutation memo: stages whose permutation was found from the hash of their
    // resource descriptors instead of specializing them again, and those specialized.
    SpecMemoHits,
    SpecMemoMisses,
    StageLastMemoChecks,
    StageLastMemoHits,
    LargeReadMemoChecks,
    LargeReadMemoHits,
    BufferChecksReused,
    // Region summary: read only buffer bindings skipped without visiting any region (no region
    // of their range has CPU modified pages).
    RegionSummarySkips,
    // Shader stage lookups (PipelineCache::GetProgram) and the user data dwords flattened.
    // Scout: submissions read ahead, draws precomputed, time spent, submissions the command
    // processor was already past; command processor side: stage lookups taken from the scout,
    // entries whose inputs did not match (stale), draws without entry (scout behind), and
    // submissions whose entries stopped being used (another path through the commands).
    ScoutSubmissions,
    ScoutDraws,
    ScoutBusyNs,
    ScoutLate,
    ScoutHits,
    ScoutStale,
    ScoutMisses,
    ScoutBehind,
    ScoutDesyncs,
    SpecCalls,
    SpecFlatDwords,
    // Texture binding memo: textures bound again with the same descriptor while the texture
    // cache did not change, found without searching the cache, and the others (searched, then
    // remembered).
    TexMemoHits,
    TexMemoMisses,
    // Write merge (readback_write_merge): pages whose content was kept when a guest write
    // discarded their GPU data (and those not kept, too many), pages uploaded with a merge, bytes
    // the GPU wrote that these uploads left on the GPU, and bytes the GPU wrote that the guest
    // changed since (uploaded).
    MergeSnapshots,
    MergeSnapshotsSkipped,
    MergePages,
    MergeKeptBytes,
    MergeGuestBytes,
    // Threaded recording: command batches given to the recording thread, the commands in them,
    // the times the command processor waited for it to record the session itself, how long, and
    // the time the recording thread spent recording and submitting.
    RecorderJobs,
    RecorderCommands,
    RecorderSyncs,
    RecorderSyncNs,
    RecorderBusyNs,
    // Times the command processor woke the sleeping recording thread (a system call each).
    RecorderWakes,
    // Upload worker (async_uploads): buffer uploads whose page protection and copy to staging
    // memory another thread did, their bytes, the pages it protected, the time it spent, and the
    // time submissions waited for it.
    AsyncUploadJobs,
    AsyncUploadBytes,
    AsyncUploadPages,
    AsyncUploadBusyNs,
    AsyncUploadWaitNs,
    // Binds recorded in the command buffers through CommandRecorder (pipelines, push constants,
    // vertex inputs), and those left out because the command buffer already had them
    // (skip_redundant_binds).
    PipelineBinds,
    PipelineBindsSkipped,
    ConstantPushes,
    ConstantPushesSkipped,
    VertexInputSets,
    VertexInputSetsSkipped,
    DescriptorPushes,
    DescriptorPushesSkipped,
    // Shader params memo (shader_params_memo): stage programs found without reading their code
    // again, and the others.
    ParamsMemoHits,
    ParamsMemoMisses,
    // Draws without vertices and dispatches without workgroups (left out with
    // skip_empty_draws).
    EmptyDraws,
    EmptyDispatches,
    // Stage lookups the scout resolved, taken as is (scout_fast), and draws whose pipeline was
    // the last one (pipeline_key_memo).
    ScoutFast,
    PipelineKeyMemoHits,
    // Draws whose resource lookups were prefetched from the scout's entries a draw ahead, and
    // texture memo keys taken from the scout instead of being hashed (bind_prefetch).
    PrefetchedDraws,
    ScoutTexKeys,
    // Draws without context register change since the draw before, and with its pipeline
    // (what a fast path for such draws could skip).
    DrawsSameContext,
    DrawsSamePipeline,
    // Render and depth targets found again from the registers of the draw before (rt_memo),
    // and the others.
    RtMemoHits,
    RtMemoMisses,
    // Protections applied after the region lock was released (late_protect).
    LateProtects,
    // Guest readbacks by the state of the graphics submission that last wrote their window when
    // they are requested: executed by the GPU, submitted to the driver, handed to the recording
    // thread, or in the command buffer being recorded.
    ReadbackProducerDone,
    ReadbackProducerSubmitted,
    ReadbackProducerQueued,
    ReadbackProducerCurrent,
    // readback_queue: guest readbacks copied by the readback queue, those copied by the graphics
    // command buffer because their data was not submitted yet, and readbacks written to guest
    // memory before an earlier one (different window) still in flight.
    ReadbackQueueReads,
    ReadbackQueueFallbacks,
    ReadbackOutOfOrder,
    // gpu_profile: GPU time of the submissions (busy time without overlaps), their number, those
    // carrying readbacks and their GPU time, submissions already in flight when one is submitted
    // (summed), and the GPU time of the compute dispatches.
    GpuProfileBusyNs,
    GpuProfileSessions,
    GpuProfileSessionNs,
    GpuProfileReadbackSessions,
    GpuProfileReadbackNs,
    GpuProfileInFlight,
    GpuProfileComputeNs,
    GpuProfileDispatches,
    // readback_relaxed_sites: guest reads let through without waiting for the GPU, pages of them
    // given the new data when its download landed, and pages left as they were (changed since).
    ReadbackRelaxed,
    ReadbackRelaxedRefreshes,
    ReadbackRelaxedSkips,
    // asc_queue: dispatches of the guest's compute queues run on the compute queue, those kept
    // on the graphics queue (images, HLE...), submissions of the compute queue, waits of each
    // queue on the other, sessions flushed early for a wait, words of guest memory the compute
    // queue touched for the first time.
    AscQueueDispatches,
    AscQueueFallbacks,
    ComputeQueueSubmits,
    CrossWaitsGraphics,
    CrossWaitsCompute,
    CrossQueueFlushes,
    CrossQueueFirstTouches,
    CounterCount,
};

/**
 * Counters are kept per thread and summed when read: counting is then a plain addition (no
 * locked instruction, which would also wait for the pending writes of the thread to GPU memory
 * to complete). A thread's counters are only written by that thread; the reporter reads them
 * from another thread (each read is atomic, the sums are approximate by a few events).
 */
struct ThreadCounters {
    std::array<std::atomic<u64>, CounterCount> values{};
};

constexpr size_t MaxCounterThreads = 256;

struct CounterRegistry {
    std::array<ThreadCounters, MaxCounterThreads> threads{};
    std::atomic<size_t> num_threads{0};
    /// Threads beyond the slots share the last slot, counted with locked additions.
    std::atomic<bool> overflow{false};
};

inline CounterRegistry& Registry() {
    static CounterRegistry registry{};
    return registry;
}

/// Counters of the calling thread (a slot taken at its first count; no allocation, this may run
/// in a fault handler).
inline ThreadCounters* LocalCounters(bool& shared) {
    thread_local ThreadCounters* local = nullptr;
    thread_local bool local_shared = false;
    if (local == nullptr) [[unlikely]] {
        auto& registry = Registry();
        const size_t index = registry.num_threads.fetch_add(1, std::memory_order_acq_rel);
        if (index < MaxCounterThreads - 1) {
            local = &registry.threads[index];
        } else {
            local = &registry.threads[MaxCounterThreads - 1];
            local_shared = true;
            registry.overflow.store(true, std::memory_order_relaxed);
        }
    }
    shared = local_shared;
    return local;
}

/// Sum of the counter over all threads.
inline u64 Total(Counter counter) {
    auto& registry = Registry();
    const size_t num_threads =
        std::min(registry.num_threads.load(std::memory_order_acquire), MaxCounterThreads);
    u64 total = 0;
    for (size_t i = 0; i < num_threads; ++i) {
        total += registry.threads[i].values[counter].load(std::memory_order_relaxed);
    }
    return total;
}

/// Parameters that can be changed while the game runs (perf_tuning.ini in the user folder), to
/// compare strategies without restarting the game.
struct Tuning {
    /// Size of the window downloaded around a precise readback, a power of two.
    std::atomic<u64> readback_window_bytes{512 * 1024};
    /// See GPUSettings::readbacks_write_discard.
    std::atomic<bool> readback_write_discard{false};
    /// perf_tuning.ini sets readback_write_discard: the emulator settings do not override it.
    std::atomic<bool> readback_write_discard_from_ini{false};
    /// With readback_write_discard: the bytes the GPU wrote in a page whose GPU data a guest
    /// write discarded stay on the GPU. The next upload of the page only sends the bytes outside
    /// them and those the guest changed, instead of overwriting the GPU results with the stale
    /// guest copy (pages shared by guest data and GPU results).
    std::atomic<bool> readback_write_merge{false};
    // GPU modified ranges stay known while their download is in flight (BufferCache).
    std::atomic<bool> readback_keep_ranges{false};
    /// Readbacks requested by guest threads wait for the GPU on that thread, so the command
    /// processor keeps working meanwhile.
    std::atomic<bool> async_readbacks{true};
    /// Memory the guest read back recently is downloaded again at fence points, so later reads
    /// find it ready (requires async_readbacks).
    std::atomic<bool> readback_prefetch{true};
    /// 0: pipelines are compiled on the command processor thread, which stalls meanwhile.
    /// 1: graphics pipelines are compiled by worker threads and draws using a pipeline that is
    /// not ready yet are skipped. 2: same for compute pipelines and dispatches.
    std::atomic<u32> async_pipelines{2};
    /// How long the command processor waits for a new pipeline before skipping its work.
    std::atomic<u32> async_pipeline_wait_us{3000};
    /// Complete compute work using a fast first pipeline, optimize a replacement in the background.
    std::atomic<bool> tiered_compute_pipelines{false};
    /// Records a GPU checkpoint before each draw and dispatch, reported with the device fault
    /// information when the GPU crashes (device lost).
    std::atomic<bool> gpu_checkpoints{false};
    /// Logs the buffers and images bound to the shaders with these hashes (a few times), and the
    /// shaders that later write the buffers they read.
    std::array<std::atomic<u64>, 4> trace_shaders{};
    /// Particle trace: guest reads of GPU written memory (readbacks) overlapping the guest range
    /// [trace_reads_lo, trace_reads_hi) are logged with the bytes the guest gets and the frame.
    std::atomic<u64> trace_reads_lo{};
    std::atomic<u64> trace_reads_hi{};
    /// Particle trace: the indirect draws of this pixel shader log their arguments (as guest
    /// memory has them) each frame, 0: off.
    std::atomic<u64> trace_draw{};
    /// Frame x-ray: guest addresses of images written to disk (raw) at the end of each dumped
    /// frame (up to 8, 0: none).
    std::array<std::atomic<u64>, 8> dump_images{};
    std::array<std::atomic<u64>, 8> dump_images_size{}; ///< (width << 32) | height, 0: any
    // Frame x-ray: images dumped after the n-th draw or dispatch of a dumped frame using the
    // shader (pixel or compute hash), optionally only the image of the given size.
    struct DumpAfter {
        std::atomic<u64> shader{};
        std::atomic<u64> address{};
        std::atomic<u32> width{};
        std::atomic<u32> height{};
        std::atomic<u32> count{};
    };
    std::array<DumpAfter, 8> dump_after{};
    /// A page the guest writes in this many consecutive frames (one write fault each) becomes
    /// hot: it is no longer write protected and is uploaded once per upload epoch instead.
    /// 0 disables hot pages.
    std::atomic<u32> hot_page_faults{0};
    /// Every this many frames, hot pages are write protected again to notice those the guest
    /// no longer writes (0: never).
    std::atomic<u32> hot_page_cooldown_frames{240};
    /// At a new upload epoch, a hot page whose content did not change since its last upload
    /// (64 bit hash) is not uploaded again.
    std::atomic<bool> hot_page_hash{true};
    /// GPU readback copies run on a second queue and wait only for the submission that wrote the
    /// data (when the GPU has a second queue). Slower on NVIDIA, where the queues share the GPU.
    std::atomic<bool> readback_queue{false};
    /// A small read only buffer bound again in the same command buffer and upload epoch, with no
    /// memory written by the emulator meanwhile, reuses its stream buffer copy.
    std::atomic<bool> stream_dedup{false};
    /// A small read only buffer whose pages are already up to date in the GPU copy of guest
    /// memory (uploaded and not written by the CPU since) is bound there instead of copied.
    std::atomic<bool> stream_static{false};
    /// stream_static only binds in place pages without write fault in this many frames (0: any
    /// clean page). Pages the guest rewrites often would make their next upload wait for the
    /// reads in place with a full pipeline barrier.
    std::atomic<u32> stream_static_frames{8};
    /// The guest frame end (SubmitDone) waits until the command processor is at most this many
    /// guest frames behind (0: no limit). Bounded wait.
    std::atomic<u32> max_gpu_lag_frames{0};
    /// GPU timestamps written by end of pipe commands advance this many times slower (1: real
    /// time), so the game measures shorter GPU frame times (dynamic quality scaling).
    std::atomic<u32> gpu_clock_div{1};
    /// While asynchronous compute work is queued, the graphics queue gives way to it every this
    /// many packets (0: only when it waits), as the PS4 runs both queues at the same time.
    std::atomic<u32> asc_yield_packets{0};
    /// Guest occlusion queries count the samples that really pass the depth test (Vulkan
    /// occlusion queries) instead of reporting every tested object as visible.
    std::atomic<bool> real_occlusion{false};
    /// Compute shaders of workgroups larger than a GCN wave get LDS barriers in their uniform
    /// parts (hosts with 32 wide subgroups split the waves the code expects in lockstep). Read
    /// when the emulator starts: changing it recompiles the compute shaders once.
    std::atomic<bool> lds_barriers_multi_wave{false};
    /// Compute workgroups spanning several host subgroups get a workgroup barrier after the
    /// buffer stores of their uniform loops that also load buffer memory: GCN waves exchange
    /// data through memory in lockstep (sorting passes within a wave), split host subgroups
    /// race. Read when the emulator starts: changing it recompiles the compute shaders once.
    std::atomic<bool> wave_memory_barriers{false};
    /// Any new non zero value logs every draw, dispatch and DMA of the next 2 guest frames, with
    /// the shaders, render targets and resources they use (frame "x-ray").
    std::atomic<u32> dump_frames{0};
    /// The Vulkan commands of the command processor are recorded in the command buffer by a
    /// thread of their own (from the next submission): the command processor no longer waits in
    /// the driver for each command.
    std::atomic<bool> threaded_recording{false};
    /// At each guest frame end, the guest waits until the command processor has processed the
    /// work submitted so far (bounded wait), as the PS4 GPU keeps up with the frame: GPU results
    /// the guest reads back during the next frame are produced by then. 1: compute queues only,
    /// 2: all queues. 0: off.
    std::atomic<u32> gpu_sync_frame_end{0};
    /// Buffer uploads protect their pages again and copy guest memory to staging memory on a
    /// worker thread instead of the command processor (the submission waits for it).
    std::atomic<bool> async_uploads{false};
    /// The recording thread waits for new commands this long (microseconds) before sleeping,
    /// so the command processor rarely has to wake it up with a system call (0: sleeps at once).
    std::atomic<u32> recorder_spin_us{0};
    /// The profiler also walks the call stacks of the emulator threads: profiles then give the
    /// time of each function including what it calls (Perf inclusive / Perf path lines).
    std::atomic<bool> profile_stacks{false};
    /// Any new non zero value tests once, at the next buffer binding, whether the GPU can use
    /// guest memory in place (VK_EXT_external_memory_host import of its pages), and logs it.
    std::atomic<u32> probe_host_import{0};
    /// Read only buffer bindings found up to date are remembered with the modification versions
    /// of their memory regions: bound again while no guest write touched those regions, they
    /// skip the page tracking checks.
    std::atomic<bool> sync_memo{false};
    std::atomic<bool> large_read_memo{false};
    std::atomic<bool> reuse_buffer_checks{false};
    /// Shader stages bound again with the same resource descriptors (hash) reuse their last
    /// permutation instead of specializing again.
    std::atomic<bool> spec_memo{false};
    /// Render targets created without size hint take the size of the texture descriptors that
    /// sample them (shrunk once), so that sampling covers the rendered area, not the padding.
    std::atomic<bool> rt_size_learn{false};
    /// Non zero: the GPU clocks the guest reads through commands advance by exactly this many
    /// microseconds per guest frame (plus a microsecond per event, to stay monotonic), as on a
    /// console that holds its frame rate, instead of following the host time (gpu_clock_div).
    /// For games that adapt their detail to the measured GPU time.
    std::atomic<u32> gpu_clock_frame_us{0};
    /// Region summary: read only buffer bindings skip the page tracking words of regions
    /// (and whole ranges) without CPU modified pages, from lock free summaries.
    std::atomic<bool> region_summary{false};
    /// Texture binding memo: a texture descriptor bound again while the texture cache did not
    /// change reuses the image found for it, without searching the cache or locking it.
    std::atomic<bool> tex_memo{false};
    /// Upload worker also handles the ranges the GPU writes: their guest pages are copied and
    /// protected (against writes and reads) by the worker, before the submission.
    std::atomic<bool> async_written_uploads{false};
    /// Scout: a thread reads the graphics submissions ahead of the command processor and
    /// precomputes the shader stage lookups of the draws (AmdGpu::Scout).
    std::atomic<bool> scout{false};
    /// Pipelines, push constants and vertex inputs the command buffer already has (same as the
    /// last ones recorded in it) are not recorded again.
    std::atomic<bool> skip_redundant_binds{false};
    /// The code size and hash of a shader address are reused until the next submission or guest
    /// wait, instead of being read in guest memory for every stage of every draw.
    std::atomic<bool> shader_params_memo{false};
    /// Draws without vertices and dispatches without workgroups are left out before their
    /// pipeline and resources are looked up.
    std::atomic<bool> skip_empty_draws{false};
    /// Logical processors the emulator runs on (hexadecimal mask, 0: all), applied while the
    /// game runs: to keep its threads on one die of a two die CPU, the one with the larger L3.
    std::atomic<u64> cpu_affinity{0};
    /// With scout: the stage lookups the scout resolved completely are taken as is by the
    /// command processor (no shader header, program, flattening, memo or attribute check).
    std::atomic<bool> scout_fast{false};
    /// A draw with the key of the draw before reuses its pipeline without hashing the key.
    std::atomic<bool> pipeline_key_memo{false};
    /// The memo entries of the resource lookups of a draw (texture, stream copy and sync memos)
    /// are prefetched before they are read: a draw ahead from the scout's entries (which also
    /// give the texture memo keys), else just before the bindings of each shader stage.
    std::atomic<bool> bind_prefetch{false};
    /// Logical processors of the emulator threads translating the graphics commands (command
    /// processor, scout, recorder, upload worker), hexadecimal mask, 0: those of the process.
    std::atomic<u64> gpu_threads_affinity{0};
    /// Logical processors of the other threads of the emulator (game threads...), hexadecimal
    /// mask, 0: those of the process.
    std::atomic<u64> other_threads_affinity{0};
    /// A render or depth target slot with the registers (and size hint) of the draw before
    /// keeps its image while the texture cache does not change: no description built from the
    /// registers, no search, no lock.
    std::atomic<bool> rt_memo{false};
    /// The page protections decided under a region lock (guest write faults, uploads of the
    /// command processor) are applied once the region lock is released, their pages staying
    /// locked: the system calls no longer hold up the other threads using the region.
    std::atomic<bool> late_protect{false};
    /// Page protections of different threads run at the same time (the address space lock is
    /// shared by protections, exclusive only for mappings), instead of one at a time.
    std::atomic<bool> parallel_protect{false};
    /// Priority of the threads translating the graphics commands (command processor, scout,
    /// recorder, upload worker): 0 unchanged, 1 above normal, 2 highest.
    std::atomic<u32> gpu_threads_priority{0};
    /// GPU timestamps (diagnostics): 1 around each submission of the command processor (GPU busy
    /// time, submissions in flight, those carrying readbacks), 2 also around each compute
    /// dispatch (GPU time per compute shader, top shaders logged every 5 s).
    std::atomic<u32> gpu_profile{0};
    /// readback_relaxed_sites: code of the game (offsets of 256 byte blocks of its executable, as
    /// the readback site report shows them) whose reads of GPU written memory do not wait for
    /// the GPU. The read gets guest memory as it is (data of the previous download, a frame or
    /// two old), the download of the new data goes on and lands in guest memory once the GPU is
    /// done, unless the guest or the GPU changed the page meanwhile. relaxed_all: every read.
    static constexpr u32 MaxRelaxedSites = 8;
    std::array<std::atomic<u64>, MaxRelaxedSites> relaxed_sites{};
    std::atomic<u32> num_relaxed_sites{0};
    std::atomic<bool> relaxed_all{false};
    /// The compute work of the guest's asynchronous compute queues runs on a GPU queue of its
    /// own, alongside the graphics work, as on the PS4 (dispatches reading or writing images
    /// stay on the graphics queue). Dependencies between the queues are tracked per 256 KB of
    /// guest memory and waited for with semaphores.
    std::atomic<bool> asc_queue{false};
    /// Performance reports in the log: the per second counters line, slow frames, the CPU usage
    /// of the emulator threads and their profiles (perf_tuning.ini perf_log = 1). Off by default.
    std::atomic<bool> perf_log{false};
};

/// gpu_threads_affinity value chosen by the emulator: on CPUs whose L3 caches differ in size
/// (a Ryzen with 3D V-Cache on one of its dies), the threads translating the graphics commands
/// run on the cores of the largest L3 (one logical processor per core, the first core left to
/// the other threads); elsewhere they are not pinned.
constexpr u64 AffinityAuto = ~0ULL;

inline Tuning& GetTuning() {
    static Tuning tuning{};
    return tuning;
}

/// Frame x-ray in progress (Tuning::dump_frames): caches log what they do.
inline std::atomic<bool>& FrameDumpActive() {
    static std::atomic<bool> active{false};
    return active;
}

/// Guest frames submitted (SubmitDone), for clocks locked to the frame (Tuning::gpu_clock_frame_us).
inline std::atomic<u64>& GuestFrameCounter() {
    static std::atomic<u64> counter{0};
    return counter;
}

inline std::atomic<u32>& GpuThreadCpu() {
    static std::atomic<u32> cpu{~0U};
    return cpu;
}

inline void Add(Counter counter, u64 value = 1) {
    bool shared;
    auto& slot = LocalCounters(shared)->values[counter];
    if (shared) [[unlikely]] {
        slot.fetch_add(value, std::memory_order_relaxed);
        return;
    }
    // Single writer: a plain addition (the reporter reads the whole value atomically).
    slot.store(slot.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline u32 CurrentCpu() {
#ifdef _WIN32
    return static_cast<u32>(GetCurrentProcessorNumber());
#else
    const int cpu = sched_getcpu();
    return cpu < 0 ? ~0U : static_cast<u32>(cpu);
#endif
}

/// Counts one event and the time spent in the enclosing scope.
class ScopedTimer {
public:
    ScopedTimer(Counter count_counter_, Counter ns_counter_)
        : count_counter{count_counter_}, ns_counter{ns_counter_}, start{NowNs()} {}
    ~ScopedTimer() {
        Add(count_counter);
        Add(ns_counter, NowNs() - start);
    }
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Counter count_counter;
    Counter ns_counter;
    u64 start;
};

/// Frames slower than this (below 20 fps) are reported individually.
constexpr double SlowFrameMs = 50.0;
constexpr u32 MaxSlowFramesPerSecond = 10;

/// Called by the presenter for every presented guest frame. Not thread safe, single caller.
struct FrameCsvField {
    const char* name;
    Counter counter;
    bool nanoseconds;
};
inline constexpr FrameCsvField FrameCsvDetails[] = {
    {"readback_queue_ms", ReadbackQueueNs, true},
    {"readback_sync_ms", ReadbackSyncNs, true},
    {"readback_async_wait_ms", ReadbackAsyncWaitNs, true},
    {"readback_bytes", ReadbackBytes, false},
    {"readback_reads", ReadbackReads, false},
    {"readback_retries", ReadbackAsyncRetries, false},
    {"readback_prefetch_hits", ReadbackPrefetchHits, false},
    {"readback_queue_fallbacks", ReadbackQueueFallbacks, false},
    {"recorder_work_ms", RecorderBusyNs, true},
    {"recorder_sync_ms", RecorderSyncNs, true},
    {"recorder_jobs", RecorderJobs, false},
    {"recorder_commands", RecorderCommands, false},
    {"recorder_wakes", RecorderWakes, false},
    {"upload_worker_work_ms", AsyncUploadBusyNs, true},
    {"upload_worker_wait_ms", AsyncUploadWaitNs, true},
    {"upload_worker_bytes", AsyncUploadBytes, false},
    {"protect_calls", ProtectCalls, false},
    {"image_barrier_calls", BarrierCalls, false},
    {"image_barriers_emitted", BarriersEmitted, false},
    {"buffer_barrier_flushes", BufferBarrierFlushes, false},
    {"stage_lookups", SpecCalls, false},
    {"specialization_memo_hits", SpecMemoHits, false},
    {"specialization_memo_misses", SpecMemoMisses, false},
    {"stage_last_memo_checks", StageLastMemoChecks, false},
    {"stage_last_memo_hits", StageLastMemoHits, false},
    {"large_read_memo_checks", LargeReadMemoChecks, false},
    {"large_read_memo_hits", LargeReadMemoHits, false},
    {"buffer_checks_reused", BufferChecksReused, false},
    {"texture_memo_hits", TexMemoHits, false},
    {"texture_memo_misses", TexMemoMisses, false},
    {"gpu_lag_wait_ms", GpuLagWaitNs, true},
    // GPU results are counted when polled, which may be later than execution.
    {"gpu_completed_work_ms", GpuProfileBusyNs, true},
    {"gpu_completed_submissions", GpuProfileSessions, false},
    {"gpu_completed_readback_ms", GpuProfileReadbackNs, true},
};

inline void CaptureFrameCsv(float draws_per_frame, float dispatches_per_frame) {
    struct Capture {
        std::FILE* file{};
        u64 last{};
        u64 index{};
        std::array<u64, 18> previous{};
        std::array<u64, std::size(FrameCsvDetails)> detail_previous{};
        u64 last_flush{};
        Capture() {
            if (const char* path = std::getenv("SHADPS4_FRAME_CSV"); path && *path) {
                file = std::fopen(path, "wb");
                if (file) {
                    std::setvbuf(file, nullptr, _IOFBF, 256 * 1024);
                    std::fputs("frame,present_ms,draws,dispatches,shader_compile_ms,pipeline_stall_"
                               "ms,gpu_wait_ms,readback_stall_ms,new_images,pipeline_skips,buffer_upload_ms,image_refresh_ms,page_protect_ms,pipeline_compile_ms,shader_compiles,pipelines_created,buffer_upload_bytes,image_upload_bytes,fault_reads,fault_writes,command_processor_ms,command_spin_ms",
                               file);
                    for (const auto& field : FrameCsvDetails) {
                        std::fprintf(file, ",%s", field.name);
                    }
                    std::fputs(",gpu_profile_enabled\n", file);
                }
            }
        }
        ~Capture() {
            if (file)
                std::fclose(file);
        }
    };
    static Capture capture;
    if (!capture.file)
        return;
    const u64 now = NowNs();
    const std::array<u64, 18> totals{Total(ShaderCompileNs), Total(PipelineStallNs),
                                    Total(GpuWaitNs), Total(ReadbackStallNs), Total(ImageCreates),
                                    Total(PipelineSkips), Total(BufferUploadNs), Total(ImageRefreshNs),
                                    Total(ProtectNs), Total(PipelineCreateNs), Total(ShaderCompiles),
                                    Total(PipelineCreates), Total(BufferUploadBytes), Total(ImageUploadBytes),
                                    Total(FaultReads), Total(FaultWrites), Total(GpuBusyNs), Total(GpuSpinNs)};
    if (capture.last != 0) {
        std::fprintf(capture.file, "%llu,%.6f,%.1f,%.1f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%.6f,%.6f",
                     static_cast<unsigned long long>(++capture.index),
                     double(now - capture.last) / 1e6, draws_per_frame, dispatches_per_frame,
                     double(totals[0] - capture.previous[0]) / 1e6,
                     double(totals[1] - capture.previous[1]) / 1e6,
                     double(totals[2] - capture.previous[2]) / 1e6,
                     double(totals[3] - capture.previous[3]) / 1e6,
                     static_cast<unsigned long long>(totals[4] - capture.previous[4]),
                     static_cast<unsigned long long>(totals[5] - capture.previous[5]),
                     double(totals[6] - capture.previous[6]) / 1e6,
                     double(totals[7] - capture.previous[7]) / 1e6,
                     double(totals[8] - capture.previous[8]) / 1e6,
                     double(totals[9] - capture.previous[9]) / 1e6,
                     static_cast<unsigned long long>(totals[10] - capture.previous[10]),
                     static_cast<unsigned long long>(totals[11] - capture.previous[11]),
                     static_cast<unsigned long long>(totals[12] - capture.previous[12]),
                     static_cast<unsigned long long>(totals[13] - capture.previous[13]),
                     static_cast<unsigned long long>(totals[14] - capture.previous[14]),
                     static_cast<unsigned long long>(totals[15] - capture.previous[15]),
                     double(totals[16] - capture.previous[16]) / 1e6,
                     double(totals[17] - capture.previous[17]) / 1e6);
    }
    for (size_t i = 0; i < std::size(FrameCsvDetails); ++i) {
        const auto& field = FrameCsvDetails[i];
        const u64 total = Total(field.counter);
        if (capture.last != 0) {
            const u64 delta = total - capture.detail_previous[i];
            if (field.nanoseconds) {
                std::fprintf(capture.file, ",%.6f", double(delta) / 1e6);
            } else {
                std::fprintf(capture.file, ",%llu", static_cast<unsigned long long>(delta));
            }
        }
        capture.detail_previous[i] = total;
    }
    if (capture.last != 0) {
        std::fprintf(capture.file, ",%u\n",
                     GetTuning().gpu_profile.load(std::memory_order_relaxed) != 0 ? 1u : 0u);
    }
    // Bounded live visibility without flushing for each frame.
    if (now - capture.last_flush >= 1'000'000'000ULL) {
        std::fflush(capture.file);
        capture.last_flush = now;
    }
    capture.last = now;
    capture.previous = totals;
}

inline void OnFlip(float draws_per_frame, float dispatches_per_frame) {
    CaptureFrameCsv(draws_per_frame, dispatches_per_frame);
    if (!GetTuning().perf_log.load(std::memory_order_relaxed)) {
        return;
    }
    struct Reporter {
        u64 last_flip{};
        u64 window_start{};
        u64 frames{};
        double frame_ms_sum{};
        double frame_ms_max{};
        u32 frames_over_33{};
        u32 frames_over_50{};
        u32 slow_logged{};
        u64 flip_index{};
        std::array<u64, CounterCount> totals{};
        std::array<u64, CounterCount> last_seen{};
    };
    static Reporter r;

    const u64 now = NowNs();
    // Events since the previous flip: the counters only grow, the difference is this frame's.
    std::array<u64, CounterCount> frame{};
    for (u32 i = 0; i < CounterCount; ++i) {
        const u64 total = Total(static_cast<Counter>(i));
        frame[i] = total - r.last_seen[i];
        r.last_seen[i] = total;
    }
    ++r.flip_index;
    if (r.last_flip == 0) {
        r.last_flip = now;
        r.window_start = now;
        return;
    }

    const double frame_ms = static_cast<double>(now - r.last_flip) / 1e6;
    r.last_flip = now;
    ++r.frames;
    r.frame_ms_sum += frame_ms;
    r.frame_ms_max = std::max(r.frame_ms_max, frame_ms);
    r.frames_over_33 += frame_ms > 33.4 ? 1 : 0;
    r.frames_over_50 += frame_ms > SlowFrameMs ? 1 : 0;
    for (u32 i = 0; i < CounterCount; ++i) {
        r.totals[i] += frame[i];
    }

    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    const auto mb = [](u64 bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

    if (frame_ms > SlowFrameMs && r.slow_logged < MaxSlowFramesPerSecond) {
        ++r.slow_logged;
        LOG_WARNING(Debug,
                    "Perf slow frame #{}: {:.1f} ms | shaders {} ({:.1f} ms) pipelines {} ({:.1f} "
                    "ms compile, {:.1f} ms stall, {} skipped) | gpu waits {} ({:.1f} ms) | "
                    "readbacks R{} W{} D{} ({:.1f} ms stalled: queue {:.1f} ms, gpu sync {:.1f} "
                    "ms, {:.2f} MB, async {} waiting {:.1f} ms) | faults R{} W{} | protect "
                    "{} ({:.1f} ms) | buffer uploads {} ({:.2f} MB, {:.1f} ms) | image refresh {} "
                    "({:.2f} MB, {:.1f} ms) new images {} | gpu thread work {:.1f} ms spin {:.1f} "
                    "ms",
                    r.flip_index, frame_ms, frame[ShaderCompiles], ms(frame[ShaderCompileNs]),
                    frame[PipelineCreates], ms(frame[PipelineCreateNs]), ms(frame[PipelineStallNs]),
                    frame[PipelineSkips], frame[GpuWaits], ms(frame[GpuWaitNs]),
                    frame[ReadbackReads], frame[ReadbackWrites], frame[ReadbackDiscards],
                    ms(frame[ReadbackStallNs]), ms(frame[ReadbackQueueNs]),
                    ms(frame[ReadbackSyncNs]), mb(frame[ReadbackBytes]), frame[ReadbackAsync],
                    ms(frame[ReadbackAsyncWaitNs]), frame[FaultReads], frame[FaultWrites],
                    frame[ProtectCalls], ms(frame[ProtectNs]), frame[BufferUploads],
                    mb(frame[BufferUploadBytes]), ms(frame[BufferUploadNs]), frame[ImageRefreshes],
                    mb(frame[ImageUploadBytes]), ms(frame[ImageRefreshNs]), frame[ImageCreates],
                    ms(frame[GpuBusyNs]), ms(frame[GpuSpinNs]));
    }

    const u64 window_ns = now - r.window_start;
    if (window_ns < 1'000'000'000ULL) {
        return;
    }
    const auto& t = r.totals;
    const double seconds = static_cast<double>(window_ns) / 1e9;
    const u32 gpu_cpu = GpuThreadCpu().load(std::memory_order_relaxed);
    const auto pct = [window_ns](u64 ns) {
        return 100.0 * static_cast<double>(ns) / static_cast<double>(window_ns);
    };
    LOG_INFO(Debug,
             "Perf: {:.1f} fps | frame avg {:.1f} ms max {:.1f} ms | >33ms {} >50ms {} | gpu "
             "thread work {:.0f}% spin {:.0f}% (cpu {}) | shaders {} ({:.0f} ms) pipelines {} "
             "({:.0f} ms compile, {:.0f} ms stall, {} skipped) | gpu waits {} ({:.0f} ms) | "
             "readbacks R{} W{} D{} ({:.0f} ms stalled: queue {:.0f} ms, gpu sync {:.0f} ms, "
             "{:.1f} MB, async {} waiting {:.0f} ms retried {}, prefetch {} flushes {} hits {}, "
             "discards blocked {}, queue {} flushes {}) | "
             "faults R{} W{} | protect {} "
             "({:.0f} ms) | hot pages +{} -{} hash skip {} upload {} changed {} | epochs {} (gfx {} "
             "asc {} "
             "wait {} sem {} rewind {}, skipped cp {} vo {}) host writes {} | stream {} ({:.1f} "
             "MB) reused {} ({:.1f} MB) clean {} arena {} (barrier avoided {}) | written skips {} | "
             "texture invalidate "
             "skips {} locked {} | buffer barriers {} (copies {}) | gpu lag {:.2f} frames (waited "
             "{:.0f} ms) | asc {} (start delay {:.2f} ms, busy {:.0f}%) gfx yields {} | occlusion "
             "{} (segments {} dropped {} emulated {} waits skipped {} read {} trivial {}) | memo sync {}/{} spec {}/{} tex {}/{} region skips {} | stage lookups {} ({:.0f} dwords avg) | scout submissions {} draws {} ({:.0f} ms) late {} hits {} stale {} misses {} behind {} desyncs {} | write merge snapshots {} (skipped {}) pages {} "
             "kept {} B guest {} B | recorder jobs {} commands {} syncs {} ({:.1f} ms) busy {:.0f}% "
             "wakes {} | upload worker jobs {} ({:.1f} MB, {} pages protected) busy {:.0f}% waited "
             "{:.1f} ms | "
             "buffer uploads {} ({:.1f} MB, "
             "{:.0f} "
             "ms) | image refresh {} ({:.1f} "
             "MB, {:.0f} ms) new images {} | image barriers {} calls {} fast {} memo {} dirty {} "
             "subresources {} "
             "emitted | "
             "binds left out: pipelines {}/{} constants {}/{} vertex inputs {}/{} "
             "descriptors {}/{} | shader params memo {}/{} | empty draws {} dispatches {} | "
             "scout fast {} | pipeline key memo {} | bind prefetch draws {} scout texture "
             "keys {} | draws with the context of the draw before {} with its pipeline {} | "
             "rt memo {}/{} | late protects {} | readback producers done {} submitted {} "
             "queued {} current {}, queue reads {} fallbacks {} out of order {} | gpu busy "
             "{:.0f}% submissions {} ({:.2f} ms avg, in flight {:.1f} avg) with readbacks {} "
             "({:.2f} ms avg) compute {} ({:.1f} ms) | relaxed readbacks {} (pages refreshed {} "
             "left {}) | asc queue dispatches {} fallbacks {} submits {} waits gfx {} compute {} "
             "flushes {} first touches {} | draws/frame {:.0f} dispatches/frame {:.0f}",
             static_cast<double>(r.frames) / seconds,
             r.frames ? r.frame_ms_sum / static_cast<double>(r.frames) : 0.0, r.frame_ms_max,
             r.frames_over_33, r.frames_over_50, pct(t[GpuBusyNs]), pct(t[GpuSpinNs]),
             static_cast<s32>(gpu_cpu), t[ShaderCompiles], ms(t[ShaderCompileNs]),
             t[PipelineCreates], ms(t[PipelineCreateNs]), ms(t[PipelineStallNs]), t[PipelineSkips],
             t[GpuWaits], ms(t[GpuWaitNs]), t[ReadbackReads], t[ReadbackWrites],
             t[ReadbackDiscards], ms(t[ReadbackStallNs]), ms(t[ReadbackQueueNs]),
             ms(t[ReadbackSyncNs]), mb(t[ReadbackBytes]), t[ReadbackAsync],
             ms(t[ReadbackAsyncWaitNs]), t[ReadbackAsyncRetries], t[ReadbackPrefetches],
             t[ReadbackPrefetchFlushes], t[ReadbackPrefetchHits], t[ReadbackDiscardsBlocked],
             t[ReadbackQueueSubmits], t[ReadbackQueueFlushes],
             t[FaultReads], t[FaultWrites], t[ProtectCalls], ms(t[ProtectNs]),
             t[HotPagePromotions], t[HotPageDemotions], t[HotPageHashSkips],
             t[HotPageHashUploads], t[HotPageHashMismatches], t[UploadEpochs], t[EpochGfxSubmits],
             t[EpochAscSubmits],
             t[EpochCpuWaits], t[EpochSemaphoreWaits], t[EpochRewinds], t[EpochCpWaitsSkipped],
             t[EpochVoWaitsSkipped], t[HostWrites], t[StreamCopies], mb(t[StreamCopyBytes]),
             t[StreamReuses], mb(t[StreamReuseBytes]), t[StreamCleanCopies], t[StreamArenaBinds],
             t[StreamBarrierAvoided], t[WrittenRangeSkips], t[TexInvalidateSkips], t[TexInvalidateLocked],
             t[BufferBarrierFlushes], t[CopyBarrierFlushes],
             t[GpuLagSamples] ? static_cast<double>(t[GpuLagFrames]) /
                                    static_cast<double>(t[GpuLagSamples])
                              : 0.0,
             ms(t[GpuLagWaitNs]), t[AscStarts],
             t[AscStarts] ? ms(t[AscStartDelayNs]) / static_cast<double>(t[AscStarts]) : 0.0,
             pct(t[AscBusyNs]), t[GfxYields], t[OcclusionQueries], t[OcclusionSegments],
             t[OcclusionDropped], t[OcclusionEmulated], t[OcclusionWaitsSkipped],
             t[OcclusionWaitsRead], t[WaitsTrivial], t[SyncMemoHits], t[SyncMemoMisses],
             t[SpecMemoHits], t[SpecMemoMisses], t[TexMemoHits], t[TexMemoMisses],
             t[RegionSummarySkips], t[SpecCalls],
             t[SpecCalls] ? static_cast<double>(t[SpecFlatDwords]) / static_cast<double>(t[SpecCalls])
                          : 0.0,
             t[ScoutSubmissions], t[ScoutDraws], ms(t[ScoutBusyNs]), t[ScoutLate], t[ScoutHits],
             t[ScoutStale], t[ScoutMisses], t[ScoutBehind], t[ScoutDesyncs], t[MergeSnapshots],
             t[MergeSnapshotsSkipped], t[MergePages], t[MergeKeptBytes], t[MergeGuestBytes],
             t[RecorderJobs], t[RecorderCommands], t[RecorderSyncs], ms(t[RecorderSyncNs]),
             pct(t[RecorderBusyNs]), t[RecorderWakes], t[AsyncUploadJobs],
             mb(t[AsyncUploadBytes]), t[AsyncUploadPages], pct(t[AsyncUploadBusyNs]),
             ms(t[AsyncUploadWaitNs]), t[BufferUploads],
             mb(t[BufferUploadBytes]), ms(t[BufferUploadNs]), t[ImageRefreshes],
             mb(t[ImageUploadBytes]), ms(t[ImageRefreshNs]), t[ImageCreates], t[BarrierCalls],
             t[BarrierFastPath], t[BarrierMemoHit], t[BarrierDirtyPath], t[BarrierSubresources],
             t[BarriersEmitted], t[PipelineBindsSkipped], t[PipelineBinds],
             t[ConstantPushesSkipped], t[ConstantPushes], t[VertexInputSetsSkipped],
             t[VertexInputSets], t[DescriptorPushesSkipped], t[DescriptorPushes],
             t[ParamsMemoHits], t[ParamsMemoMisses], t[EmptyDraws], t[EmptyDispatches],
             t[ScoutFast], t[PipelineKeyMemoHits], t[PrefetchedDraws], t[ScoutTexKeys],
             t[DrawsSameContext], t[DrawsSamePipeline], t[RtMemoHits], t[RtMemoMisses],
             t[LateProtects], t[ReadbackProducerDone], t[ReadbackProducerSubmitted],
             t[ReadbackProducerQueued], t[ReadbackProducerCurrent], t[ReadbackQueueReads],
             t[ReadbackQueueFallbacks], t[ReadbackOutOfOrder], pct(t[GpuProfileBusyNs]),
             t[GpuProfileSessions],
             t[GpuProfileSessions] ? ms(t[GpuProfileSessionNs]) /
                                         static_cast<double>(t[GpuProfileSessions])
                                   : 0.0,
             t[GpuProfileSessions] ? static_cast<double>(t[GpuProfileInFlight]) /
                                         static_cast<double>(t[GpuProfileSessions])
                                   : 0.0,
             t[GpuProfileReadbackSessions],
             t[GpuProfileReadbackSessions] ? ms(t[GpuProfileReadbackNs]) /
                                                 static_cast<double>(t[GpuProfileReadbackSessions])
                                           : 0.0,
             t[GpuProfileDispatches], ms(t[GpuProfileComputeNs]), t[ReadbackRelaxed],
             t[ReadbackRelaxedRefreshes], t[ReadbackRelaxedSkips], t[AscQueueDispatches],
             t[AscQueueFallbacks], t[ComputeQueueSubmits], t[CrossWaitsGraphics],
             t[CrossWaitsCompute], t[CrossQueueFlushes], t[CrossQueueFirstTouches],
             draws_per_frame, dispatches_per_frame);

    const u64 last_flip = r.last_flip;
    const u64 flip_index = r.flip_index;
    const auto last_seen = r.last_seen;
    r = Reporter{};
    r.last_flip = last_flip;
    r.window_start = now;
    r.flip_index = flip_index;
    r.last_seen = last_seen;
}

} // namespace Common::Perf
