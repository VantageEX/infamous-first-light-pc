// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_frame_record.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <vector>
#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>
#include <xxhash.h>

#include "tlg_toggles.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

using AmdGpu::PM4ItOpcode;

constexpr u64 MaxHashBytes = 64_KB;
constexpr u64 MaxWordBytes = 1_KB;
constexpr u64 MaxDumpBytes = 4_MB;
constexpr u32 NumFrames = 2;

const Instance* instance{};
Scheduler* scheduler{};

// GPU command thread only (OnFlip with gpu_thread, Packet and the rasterizer details).
std::array<std::vector<std::string>, NumFrames> frames;
u32 frame = 0;
bool open = false; ///< the last command is a draw/dispatch taking details

// GPU copies: what the GPU read at a draw, resolved once both frames are done.
struct Snapshot {
    u64 offset;
    u64 size;
    bool words;
    std::string dump; ///< TLG_FRAME_RECORD_DUMP file name; the snapshot is only written out
};
/// TLG_FRAME_RECORD_DUMP inputs read from guest memory at the draw.
struct GuestDumpData {
    std::string name;
    std::vector<u8> bytes;
};
std::vector<GuestDumpData> guest_dumps;
struct Copy {
    vk::Buffer source;
    u64 source_offset;
    u64 offset;
    u64 size;
};
std::vector<Snapshot> snapshots;
std::vector<Copy> copies;
VkBuffer staging{};
VmaAllocation staging_allocation{};
VmaAllocationInfo staging_info{};
u64 staging_size = 0;
u64 staging_used = 0;
u64 snapshots_dropped = 0;

const char* Trigger() {
    static const char* trigger = std::getenv("TLG_FRAME_RECORD_TRIGGER");
    return trigger && trigger[0] ? trigger : nullptr;
}

std::filesystem::path DumpDir() {
    const char* dir = std::getenv("TLG_DUMP_DIR");
    return dir && dir[0] ? dir : "out/dump";
}

// A fault while reading jumps back to the sigsetjmp (runtime_fault_recover: the probe's signal
// handler takes it before page tracking, so a read-protected page is not read back here).
[[gnu::noinline]] bool SafeHash(VAddr address, u64 size, u64* out) {
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        return false;
    }
    runtime_fault_recover = &recover;
    *out = XXH3_64bits(reinterpret_cast<const void*>(address), size);
    runtime_fault_recover = nullptr;
    return true;
}

[[gnu::noinline]] bool SafeCopy(VAddr address, u64 size, void* out) {
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        return false;
    }
    runtime_fault_recover = &recover;
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    runtime_fault_recover = nullptr;
    return true;
}

std::string Words(const u32* words, u64 count) {
    std::string out;
    for (u64 i = 0; i < count; ++i) {
        out += std::format("{}{:x}", i ? "," : "", words[i]);
    }
    return out.empty() ? "-" : out;
}

const char* Kind(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
        return "draw_indexed";
    case PM4ItOpcode::DrawIndexAuto:
        return "draw";
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
        return "draw_indirect";
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
        return "draw_indexed_indirect";
    case PM4ItOpcode::DispatchDirect:
        return "dispatch";
    case PM4ItOpcode::DispatchIndirect:
        return "dispatch_indirect";
    case PM4ItOpcode::DmaData:
        return "dma";
    case PM4ItOpcode::WriteData:
        return "write_data";
    case PM4ItOpcode::EventWrite:
        return "event";
    case PM4ItOpcode::EventWriteEop:
        return "event_eop";
    case PM4ItOpcode::EventWriteEos:
        return "event_eos";
    case PM4ItOpcode::WaitRegMem:
        return "wait_reg_mem";
    case PM4ItOpcode::IndirectBuffer:
        return "indirect_buffer";
    default:
        return "other";
    }
}

std::string Prefix(int queue, std::string_view op, std::string_view kind) {
    const std::string q = queue < 0 ? std::string{"gfx"} : std::format("c{}", queue);
    return std::format("#{} q={} op={} kind={}", frames[frame].size(), q, op, kind);
}

void Begin() {
    for (auto& f : frames) {
        f.clear();
    }
    frame = 0;
    open = false;
    snapshots.clear();
    copies.clear();
    guest_dumps.clear();
    staging_used = 0;
    snapshots_dropped = 0;
    // TLG_FRAME_RECORD_GPU_MB (default 256): room for the buffers copied from the GPU.
    const char* env = std::getenv("TLG_FRAME_RECORD_GPU_MB");
    const u64 mb = env && env[0] ? std::strtoull(env, nullptr, 10) : 256;
    staging_size = mb * 1_MB;
    if (!staging_size || !instance) {
        return;
    }
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = staging_size,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    if (vmaCreateBuffer(instance->GetAllocator(), &buffer_ci, &alloc_ci, &staging,
                        &staging_allocation, &staging_info) != VK_SUCCESS) {
        std::printf("Frame record: no %llu MB staging buffer, GPU-side contents left out\n",
                    static_cast<unsigned long long>(mb));
        staging = VK_NULL_HANDLE;
        staging_size = 0;
    }
}

std::string Resolve(const std::string& line) {
    if (line.find('\x01') == std::string::npos) {
        return line;
    }
    std::string out;
    size_t at = 0;
    while (true) {
        const size_t start = line.find('\x01', at);
        if (start == std::string::npos) {
            out.append(line, at);
            break;
        }
        const size_t end = line.find('\x02', start);
        out.append(line, at, start - at);
        const u64 index = std::strtoull(line.c_str() + start + 1, nullptr, 10);
        const auto& snap = snapshots[index];
        const auto* data = static_cast<const u8*>(staging_info.pMappedData) + snap.offset;
        if (snap.words) {
            std::vector<u32> words(snap.size / 4);
            std::memcpy(words.data(), data, words.size() * 4);
            out += Words(words.data(), words.size());
        } else {
            out += std::format("g{:016x}", XXH3_64bits(data, snap.size));
        }
        at = end + 1;
    }
    return out;
}

void Write() {
    // The copies were recorded with the frames: wait for them.
    scheduler->Finish();
    if (staging) {
        vmaInvalidateAllocation(instance->GetAllocator(), staging_allocation, 0, VK_WHOLE_SIZE);
    }
    const auto dir = DumpDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    // Every recording of a session keeps its files (frame_record_<n>_<frame>.txt from the second
    // on), so two moments can be compared: the same street with and without an effect.
    static u32 recording = 0;
    const u32 number = recording++;
    for (u32 f = 0; f < NumFrames; ++f) {
        const auto path = (dir / (number ? std::format("frame_record_{}_{}.txt", number, f)
                                         : std::format("frame_record_{}.txt", f)))
                              .string();
        FILE* out = std::fopen(path.c_str(), "w");
        if (!out) {
            std::printf("Frame record: cannot write %s\n", path.c_str());
            continue;
        }
        std::fprintf(out,
                     "# frame_record frame %u: %zu commands, %llu MB copied from the GPU, %llu "
                     "copies over the budget (TLG_FRAME_RECORD_GPU_MB)\n",
                     f, frames[f].size(), static_cast<unsigned long long>(staging_used >> 20),
                     static_cast<unsigned long long>(snapshots_dropped));
        for (const auto& line : frames[f]) {
            const auto text = Resolve(line);
            std::fwrite(text.data(), 1, text.size(), out);
            std::fputc('\n', out);
        }
        std::fclose(out);
        std::printf("Frame record: %zu commands written to %s\n", frames[f].size(), path.c_str());
    }
    // TLG_FRAME_RECORD_DUMP: whole inputs of the selected shaders.
    const std::string base = number ? std::format("frame_record_{}_", number) : "frame_record_";
    u32 dumped = 0;
    const auto dump = [&](const std::string& name, const void* data, u64 size) {
        const auto path = (dir / (base + name + ".bin")).string();
        if (FILE* out = std::fopen(path.c_str(), "wb")) {
            std::fwrite(data, 1, size, out);
            std::fclose(out);
            ++dumped;
        }
    };
    for (const auto& snap : snapshots) {
        if (!snap.dump.empty()) {
            dump(snap.dump, static_cast<const u8*>(staging_info.pMappedData) + snap.offset, snap.size);
        }
    }
    for (const auto& d : guest_dumps) {
        dump(d.name, d.bytes.data(), d.bytes.size());
    }
    if (dumped) {
        std::printf("Frame record: %u input buffers dumped (TLG_FRAME_RECORD_DUMP)\n", dumped);
    }
    guest_dumps.clear();
    guest_dumps.shrink_to_fit();
    if (staging) {
        vmaDestroyBuffer(instance->GetAllocator(), staging, staging_allocation);
        staging = VK_NULL_HANDLE;
    }
    for (auto& f : frames) {
        f.clear();
        f.shrink_to_fit();
    }
    snapshots.clear();
    snapshots.shrink_to_fit();
}

} // namespace

void FrameRecord::Attach(const Instance& instance_, Scheduler& scheduler_) {
    instance = &instance_;
    scheduler = &scheduler_;
}

void FrameRecord::OnFlip(VAddr buffer, bool gpu_thread) {
    const char* trigger = Trigger();
    if (!trigger) {
        return;
    }
    switch (state.load(std::memory_order_relaxed)) {
    case Idle: {
        std::error_code ec;
        if (std::filesystem::exists(trigger, ec)) {
            // The draw pipeline goes off now; one frame lets it run dry before recording.
            state.store(Armed, std::memory_order_relaxed);
            std::printf("Frame record: armed\n");
        }
        break;
    }
    case Armed:
        if (gpu_thread) {
            state.store(Warmup, std::memory_order_relaxed);
        }
        break;
    case Warmup:
        if (gpu_thread && scheduler) {
            Begin();
            state.store(Recording, std::memory_order_relaxed);
            std::printf("Frame record: recording %u frames\n", NumFrames);
        }
        break;
    case Recording:
        if (!gpu_thread) {
            break;
        }
        frames[frame].push_back(Prefix(-1, "Flip", "flip") + std::format(" buffer={:#x}", buffer));
        open = false;
        if (++frame < NumFrames) {
            break;
        }
        frame = NumFrames - 1;
        Write();
        {
            std::error_code ec;
            std::filesystem::remove(trigger, ec);
        }
        state.store(Idle, std::memory_order_relaxed);
        break;
    }
}

void FrameRecord::Packet(int queue, const void* header_ptr, const AmdGpu::Regs* regs) {
    if (state.load(std::memory_order_relaxed) != Recording) {
        return;
    }
    open = false;
    const auto* header = static_cast<const AmdGpu::PM4Header*>(header_ptr);
    const auto opcode = header->type3.opcode.Value();
    const u32 count = header->type3.NumWords();
    const auto* body = reinterpret_cast<const u32*>(header) + 1;
    const auto op = magic_enum::enum_name(opcode);
    switch (opcode) {
    case PM4ItOpcode::Nop: {
        if (header->type3.count.Value() == 0) {
            return;
        }
        const u32 payload = body[0];
        if (payload == AmdGpu::PM4CmdNop::PayloadType::PatchedFlip) {
            frames[frame].push_back(Prefix(queue, op, "flip_packet"));
        } else if (payload == AmdGpu::PM4CmdNop::PayloadType::DebugMarkerPush ||
                   payload == AmdGpu::PM4CmdNop::PayloadType::DebugColorMarkerPush) {
            // The label follows the payload word, NUL-terminated within the packet.
            const auto* text = reinterpret_cast<const char*>(body + 1);
            const size_t max = (count - 1) * sizeof(u32);
            std::string label{text, strnlen(text, max)};
            for (char& c : label) {
                if (!std::isgraph(static_cast<unsigned char>(c))) {
                    c = '_';
                }
            }
            frames[frame].push_back(Prefix(queue, op, "marker") + " label=" + label);
        } else if (payload == AmdGpu::PM4CmdNop::PayloadType::DebugMarkerPop) {
            frames[frame].push_back(Prefix(queue, op, "marker_pop"));
        }
        return;
    }
    // State packets: their effect shows in the inputs of the draws.
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetQueueReg:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::SetPredication:
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::WaitOnCeCounter:
    case PM4ItOpcode::PfpSyncMe:
        return;
    default:
        break;
    }
    std::string line = Prefix(queue, op, Kind(opcode));
    switch (opcode) {
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
        if (regs) {
            line += std::format(" n={} inst={}", regs->num_indices,
                                regs->num_instances.NumInstances());
        }
        [[fallthrough]];
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
        frames[frame].push_back(std::move(line));
        open = true;
        return;
    case PM4ItOpcode::DmaData: {
        const auto* dma = reinterpret_cast<const AmdGpu::PM4DmaData*>(header);
        const u64 bytes = dma->NumBytes();
        line += std::format(" src_sel={} dst_sel={} bytes={}", u32(dma->src_sel), u32(dma->dst_sel),
                            bytes);
        if (dma->dst_sel != AmdGpu::DmaDataDst::Gds) {
            line += std::format(" dst={:#x}", dma->DstAddress<VAddr>());
        }
        const auto src_sel = dma->src_sel;
        if (src_sel == AmdGpu::DmaDataSrc::Data) {
            line += std::format(" data={:x}", dma->data);
        } else if (src_sel == AmdGpu::DmaDataSrc::Memory ||
                   src_sel == AmdGpu::DmaDataSrc::MemoryUsingL2) {
            const VAddr src = dma->SrcAddress<VAddr>();
            line += std::format(" src={:#x} srch={}", src, GuestHash(src, bytes));
        }
        break;
    }
    default:
        break;
    }
    // Everything else: the packet body (addresses, values, event types).
    line += " raw=" + Words(body, std::min<u32>(count, 16));
    frames[frame].push_back(std::move(line));
}

bool FrameRecord::Open() {
    return open && state.load(std::memory_order_relaxed) == Recording;
}

void FrameRecord::Append(std::string_view text) {
    if (!frames[frame].empty()) {
        frames[frame].back() += text;
    }
}

std::string FrameRecord::GuestHash(VAddr address, u64 size) {
    if (!address || !size) {
        return "null";
    }
    u64 hash;
    if (!SafeHash(address, std::min(size, MaxHashBytes), &hash)) {
        return "fault";
    }
    return std::format("c{:016x}", hash);
}

std::string FrameRecord::GuestWords(VAddr address, u64 size) {
    std::array<u32, MaxWordBytes / 4> words;
    const u64 bytes = std::min(size, MaxWordBytes) & ~u64(3);
    if (!address || !SafeCopy(address, bytes, words.data())) {
        return "fault";
    }
    return Words(words.data(), bytes / 4);
}

bool FrameRecord::GpuCopiesAvailable() {
    return staging != VK_NULL_HANDLE && staging_used < staging_size;
}

static void QueueCopy(std::string_view key, vk::Buffer buffer, u64 offset, u64 size, bool words) {
    size = std::min(size, words ? MaxWordBytes : MaxHashBytes) & ~u64(3);
    if (!buffer || !size) {
        FrameRecord::Append(std::string{key} + "null");
        return;
    }
    if (!staging || staging_used + size > staging_size) {
        ++snapshots_dropped;
        FrameRecord::Append(std::string{key} + "nogpu");
        return;
    }
    copies.push_back({buffer, offset, staging_used, size});
    snapshots.push_back({staging_used, size, words});
    staging_used += (size + 15) & ~u64(15);
    FrameRecord::Append(std::format("{}\x01{}\x02", key, snapshots.size() - 1));
}

void FrameRecord::GpuHash(std::string_view key, vk::Buffer buffer, u64 offset, u64 size) {
    QueueCopy(key, buffer, offset, size, false);
}

void FrameRecord::GpuWords(std::string_view key, vk::Buffer buffer, u64 offset, u64 size) {
    QueueCopy(key, buffer, offset, size, true);
}

bool FrameRecord::DumpShader(u64 pgm_hash) {
    static const std::vector<u64> hashes = [] {
        std::vector<u64> out;
        const char* env = std::getenv("TLG_FRAME_RECORD_DUMP");
        for (const char* p = env; p && *p;) {
            char* end = nullptr;
            const u64 value = std::strtoull(p, &end, 16);
            if (end == p) {
                break;
            }
            out.push_back(value);
            p = *end ? end + 1 : end;
        }
        return out;
    }();
    return std::find(hashes.begin(), hashes.end(), pgm_hash) != hashes.end();
}

static std::string DumpName(std::string_view key) {
    const size_t command = frames[frame].empty() ? 0 : frames[frame].size() - 1;
    std::string name = std::format("f{}_c{}_{}", frame, command, key);
    for (char& c : name) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
            c = '_';
        }
    }
    return name;
}

void FrameRecord::GpuDump(std::string_view key, vk::Buffer buffer, u64 offset, u64 size) {
    size = std::min(size, MaxDumpBytes) & ~u64(3);
    if (!buffer || !size || !staging || staging_used + size > staging_size) {
        ++snapshots_dropped;
        return;
    }
    copies.push_back({buffer, offset, staging_used, size});
    snapshots.push_back({staging_used, size, false, DumpName(key)});
    staging_used += (size + 15) & ~u64(15);
}

void FrameRecord::GuestDump(std::string_view key, VAddr address, u64 size) {
    size = std::min(size, MaxDumpBytes);
    GuestDumpData d{DumpName(key), std::vector<u8>(size)};
    if (address && size && SafeCopy(address, size, d.bytes.data())) {
        guest_dumps.push_back(std::move(d));
    }
}

void FrameRecord::FlushGpuCopies() {
    if (copies.empty()) {
        return;
    }
    scheduler->EndRendering();
    scheduler->Record([pending = std::move(copies), target = vk::Buffer{staging}](
                          vk::CommandBuffer cmdbuf) {
        // Everything before (shader and transfer writes) is visible to the copies, and later
        // writes wait for them.
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                   .pMemoryBarriers = &before});
        for (const auto& c : pending) {
            cmdbuf.copyBuffer(c.source, target,
                              vk::BufferCopy{.srcOffset = c.source_offset,
                                             .dstOffset = c.offset,
                                             .size = c.size});
        }
        const vk::MemoryBarrier2 after{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1,
                                                   .pMemoryBarriers = &after});
    });
    copies = {};
}

} // namespace Vulkan
