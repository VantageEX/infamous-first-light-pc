// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>

namespace Core::PerfSampler {

/**
 * Starts a background thread that reports, once per second, the CPU usage of the busiest
 * emulator threads, and periodically a statistical profile of where those threads spend their
 * time (sampled instruction addresses, grouped by function). Windows only, no-op elsewhere.
 * Set the environment variable SHADPS4_PERF_SAMPLER=0 to disable it.
 */
void Start();

/// Stops the sampler thread, before shutting down the logger.
void Stop();

/**
 * Settings measured on some games, applied at startup before perf_tuning.ini (which can change
 * any of them): inFAMOUS Second Son (CUSA00004, CUSA00223). SHADPS4_GAME_PROFILE=0 disables it.
 */
void ApplyGameProfile(std::string_view serial);

/**
 * Applies perf_tuning.ini from the user folder when it changes (checked at most once per second),
 * so strategies can be compared while the game runs. Lines are "key = value":
 *   readback_window_kb = 512      window downloaded around a precise readback (power of two)
 *   readback_write_discard = 0    see GPUSettings::readbacks_write_discard
 *   readback_write_merge = 0      with write discard, the bytes the GPU wrote in a page whose
 *                                 GPU data a guest write discarded stay on the GPU (only the
 *                                 bytes the guest changed are uploaded)
 *   readback_keep_ranges = 0      1: what the GPU wrote stays known as GPU data until its download
 *                                 has landed in guest memory (not from the moment the download
 *                                 is recorded): DMA copies and small reads of it meanwhile use
 *                                 the GPU copy, and write merge keeps it
 *   async_readbacks = 1           the faulting thread waits for the GPU, not the command processor
 *   readback_prefetch = 1         download memory read back recently again at fence points
 *   async_pipelines = 2           0 off, 1 graphics pipelines compiled by worker threads (draws
 *                                 skipped until ready), 2 graphics and compute
 *   async_pipeline_wait_ms = 3    wait for a new pipeline at most this long before skipping
 *   gpu_checkpoints = 0           1: mark each draw and dispatch so a GPU crash reports the
 *                                 work the GPU was executing (small cost, for diagnostics)
 *   trace_shader = 761c6f93,...   log the buffers and images bound to these shaders (hex
 *                                 hashes, up to 4) and the shaders writing what they read
 *   hot_page_faults = 0           a page the guest writes in this many consecutive frames is
 *                                 no longer write protected, it is uploaded once per upload
 *                                 epoch instead (0 disables)
 *   hot_page_cooldown_frames = 240  hot pages are protected again every N frames
 *   hot_page_hash = 1             a hot page whose content did not change since its last upload
 *                                 is not uploaded again at a new epoch
 *   readback_queue = 0            readback copies run on a second GPU queue and wait only for
 *                                 the submission that wrote the data, once it is submitted to
 *                                 the driver (data of commands not submitted yet is copied by
 *                                 the graphics queue)
 *   stream_dedup = 0              a small read only buffer bound again in the same command
 *                                 buffer and epoch reuses its stream buffer copy
 *   stream_static = 0             a small read only buffer already up to date in the GPU copy
 *                                 of guest memory is bound there instead of copied
 *   stream_static_frames = 8      ... only if its pages had no write fault in this many frames
 *   max_gpu_lag_frames = 0        the guest frame end waits until the command processor is at
 *                                 most this many frames behind (0: no limit)
 *   gpu_clock_div = 1             GPU timestamps read by the game advance this many times slower
 *   asc_yield_packets = 0         while compute work is queued, the graphics queue gives way to
 *                                 it every this many packets (0: only when it waits)
 *   real_occlusion = 0            1: occlusion queries count the samples that pass the depth
 *                                 test on the GPU instead of reporting everything visible
 *   lds_barriers_multi_wave = 0   1: compute workgroups larger than a GCN wave get LDS barriers
 *                                 (read at startup, changing it recompiles compute shaders)
 *   wave_memory_barriers = 0      1: compute workgroups split across host subgroups get a
 *                                 barrier after the buffer stores of their uniform loops (GPU
 *                                 particle sorting passes within a wave; read at startup,
 *                                 changing it recompiles compute shaders)
 *   dump_frames = 0               any change to a non zero value while the game runs logs every
 *                                 draw, dispatch and DMA of the next guest frames (x-ray); the
 *                                 value is the number of frames (2 to 30)
 *   threaded_recording = 0        1: the Vulkan commands of the command processor are recorded
 *                                 by a thread of their own (from the next submission)
 *   gpu_sync_frame_end = 0        at each guest frame end, the guest waits until the command
 *                                 processor has processed the compute (1) or all (2) work
 *                                 submitted so far, so GPU results read back next frame exist
 *   async_uploads = 0             1: buffer uploads protect their pages again and copy guest
 *                                 memory on a worker thread (another core), not the command
 *                                 processor
 *   recorder_spin_us = 0          the recording thread waits this long for new commands before
 *                                 sleeping (fewer wake up system calls for the command processor)
 *   profile_stacks = 0            1: the profiles of the emulator threads also give the time of
 *                                 each function including its callees, and the busiest call paths
 *   sync_memo = 0                 1: a read only buffer bound again while no guest write touched
 *                                 its memory regions skips the page tracking checks
 *   spec_memo = 0                 1: a shader stage bound again with resource descriptors of the
 *                                 same layout (formats, strides, swizzles: addresses do not
 *                                 count) and the same fetch shader reuses the permutation found
 *                                 then instead of specializing again (16 remembered per shader)
 *   rt_size_learn = 0             1: render targets created without size hint are shrunk to the
 *                                 size of the textures sampling them (padding not sampled)
 *   region_summary = 0            1: read only buffer bindings skip the page tracking tables of
 *                                 the regions (16 MB) without CPU modified pages, told by lock
 *                                 free summaries (bindings covering the whole guest memory)
 *   tex_memo = 0                  1: a texture descriptor bound again while the texture cache
 *                                 did not change reuses the image found for it, without the
 *                                 cache search and its locks
 *   async_written_uploads = 0     1: the upload worker (async_uploads) also handles the buffers
 *                                 the GPU writes: guest pages copied and protected against
 *                                 reads and writes by the worker before the submission
 *   trace_reads = lo-hi          particle trace: every guest read of GPU written memory in this
 *                                 guest range (hex) is logged (PTRACE read) with the bytes read
 *   trace_draw = 24dad4be         particle trace: the indirect draws of this pixel shader log
 *                                 their arguments each frame (PTRACE draw)
 *   scout = 0                     1: a thread reads the graphics submissions ahead of the command
 *                                 processor and precomputes the shader stage lookups of the
 *                                 draws (checked against the processor's state before use)
 *   gpu_clock_frame_us = 0        non zero: the GPU clocks the game reads advance by exactly this
 *                                 many microseconds per guest frame (console pace), instead of
 *                                 following the host time, for games adapting their detail to
 *                                 the measured GPU time (16667 = 60 fps, 33333 = 30 fps)
 *   dump_images = 316ad0000,...   frame x-ray: the images at these guest addresses (hex, up to
 *                                 8) are written raw to the log folder at the end of each dumped
 *                                 frame (name: frame, address, size, Vulkan format); the image
 *                                 used most recently among those at the address, or the one of
 *                                 the given size: 316ad0000/3200x1800
 *   dump_after = 29c851e0:30bda0000/3200x1800:1,...
 *                                 frame x-ray: the image is written after the n-th draw or
 *                                 dispatch of each dumped frame whose pixel or compute shader
 *                                 has the hash (n = 1 if omitted), up to 8 entries; "hash:vb"
 *                                 writes the vertex buffers of the draw instead (GPU copy and
 *                                 guest memory, to compare)
 *   skip_redundant_binds = 0      1: pipelines, push constants, vertex inputs and descriptor
 *                                 pushes equal to the last ones of the command buffer are not
 *                                 recorded again
 *   shader_params_memo = 0        1: the code size and hash of a shader address are reused until
 *                                 the next submission or guest wait
 *   skip_empty_draws = 0          1: draws without vertices and dispatches without workgroups
 *                                 are left out
 *   cpu_affinity = 0              logical processors of the emulator (hexadecimal mask, 0: all)
 *   scout_fast = 0                1 (with scout): the stage lookups the scout resolved are taken
 *                                 as they are by the command processor
 *   pipeline_key_memo = 0         1: a draw with the pipeline key of the draw before reuses its
 *                                 pipeline
 *   bind_prefetch = 0             1: the memo entries of the resource lookups of a draw are
 *                                 prefetched, a draw ahead with the scout (whose texture keys
 *                                 are then taken as they are)
 *   gpu_threads_affinity = 0      logical processors of the command processor, scout, recorder
 *                                 and upload worker threads (hexadecimal mask, 0: the process's,
 *                                 auto: the cores of the largest L3 cache when the L3 caches
 *                                 differ in size, such as a Ryzen with 3D V-Cache)
 *   other_threads_affinity = 0    logical processors of the other threads (game threads...),
 *                                 hexadecimal mask, 0: the process's
 *   rt_memo = 0                   1: a render or depth target with the registers of the draw
 *                                 before keeps its image while the texture cache does not change
 *   late_protect = 0              1: page protections decided under a region lock are applied
 *                                 after it is released (the pages stay locked until then)
 *   parallel_protect = 0          1: page protections of different threads run at the same time
 *                                 (address space lock shared, exclusive only for mappings)
 *   gpu_threads_priority = 0      priority of the command processor, scout, recorder and upload
 *                                 worker threads: 0 unchanged, 1 above normal, 2 highest
 *   gpu_profile = 0               GPU timestamps (diagnostics): 1 per submission (GPU busy time,
 *                                 submissions in flight, those carrying readbacks), 2 also per
 *                                 compute dispatch (top compute shaders logged every 5 s)
 *   perf_log = 0                  1: performance reports in the log (counters every second, slow
 *                                 frames, CPU usage and profiles of the emulator threads)
 *   asc_queue = 0                 1: the compute work of the guest's asynchronous compute queues
 *                                 runs on a GPU queue of its own alongside the graphics work
 *                                 (as on the PS4), except the dispatches using images
 */
void PollTuning();

/// Records a guest read of GPU written memory that needed a precise readback (address and guest
/// instruction), reported with the profiles to find which data the game reads back.
void RecordReadback(unsigned long long address, unsigned long long rip);

} // namespace Core::PerfSampler
