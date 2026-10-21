// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: frame recorder, for effects that flicker in a static scene. TLG_FRAME_RECORD_TRIGGER=<file>:
// once the file exists, the next two complete frames (flip to flip) are recorded command by
// command, graphics and compute queues in the order the GPU command thread runs them, and
// written to TLG_DUMP_DIR (default out/dump) as frame_record_0.txt and frame_record_1.txt; the
// trigger file is removed when they are written. Each line is one command with the inputs of
// draws and dispatches: shaders and permutations, counts (indirect arguments as the GPU reads
// them), bound buffers with a content hash, images, render targets, viewport and blend/depth/
// stencil state. scripts/frame_diff.py compares the two files.
//
// While a recording is armed or running the draw pipeline is off (Rasterizer::UseDrawPipe), so
// every command runs on the GPU command thread, in stream order. Otherwise the only cost is the
// Active() check.

#pragma once

#include <atomic>
#include <string>
#include <string_view>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace AmdGpu {
union Regs;
}

namespace Vulkan {

class Instance;
class Scheduler;

class FrameRecord {
public:
    /// Armed or recording (the draw pipeline stays off).
    static bool Active() {
        return state.load(std::memory_order_relaxed) != Idle;
    }
    /// Commands are being recorded (GPU command thread).
    static bool IsRecording() {
        return state.load(std::memory_order_relaxed) == Recording;
    }

    static void Attach(const Instance& instance, Scheduler& scheduler);

    /// Every flip (VideoOutDriver::SubmitFlipInternal): trigger file and frame boundaries.
    /// `gpu_thread`: called on the GPU command thread (stream order); other flips only arm.
    static void OnFlip(VAddr buffer, bool gpu_thread);

    /// Liverpool: a type-3 packet of the graphics queue (queue < 0, `regs` its registers) or
    /// of compute queue `queue` (vqid). Register packets are left out.
    static void Packet(int queue, const void* header, const AmdGpu::Regs* regs);

    /// Rasterizer: the draw or dispatch the last packet started takes details (the packet
    /// may also end in no Vulkan work: filtered draws, HLE shaders).
    static bool Open();
    static void Append(std::string_view text);

    /// Hash of up to 64 KiB of guest memory; a fault (unmapped, or read-protected because
    /// the GPU wrote the pages) is caught and gives "fault".
    static std::string GuestHash(VAddr address, u64 size);
    /// The first `size` bytes of guest memory as 32-bit words ("fault" when unreadable).
    static std::string GuestWords(VAddr address, u64 size);

    /// Appends `key` and the hash (GpuHash, up to 64 KiB) or the words (GpuWords) of `buffer`
    /// at `offset` as the GPU sees it at this point of the stream, resolved after the frames.
    static void GpuHash(std::string_view key, vk::Buffer buffer, u64 offset, u64 size);
    static void GpuWords(std::string_view key, vk::Buffer buffer, u64 offset, u64 size);
    /// Whether copies queued by GpuHash/GpuWords would be recorded (budget left).
    static bool GpuCopiesAvailable();

    /// TLG_FRAME_RECORD_DUMP=<hash>[,<hash>...] (program hashes, as the lines print them, leading
    /// zeros optional): the inputs of draws and dispatches with one of these shaders are also
    /// written whole (up to 4 MiB each) as frame_record[_<n>]_f<frame>_c<command>_<key>.bin.
    static bool DumpShader(u64 pgm_hash);
    /// The buffer as the GPU sees it at this point of the stream (memory the GPU wrote).
    static void GpuDump(std::string_view key, vk::Buffer buffer, u64 offset, u64 size);
    /// Guest memory, read now.
    static void GuestDump(std::string_view key, VAddr address, u64 size);
    /// Records the copies queued since the last call. Ends the render pass: call only where
    /// the next draw begins its own (it must not repeat a clear) or outside of one.
    static void FlushGpuCopies();

private:
    enum : u32 { Idle, Armed, Warmup, Recording };
    static inline std::atomic<u32> state{Idle};
};

} // namespace Vulkan
