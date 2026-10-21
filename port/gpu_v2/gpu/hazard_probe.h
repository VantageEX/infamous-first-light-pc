// Session 12 diagnostic: record the buffer checks behind compute barriers.
// This does not change hazard decisions or GPU commands.
#pragma once

#include <array>
#include <cstdint>

namespace iss::hazard_probe {

struct BufferCheck {
    std::uint32_t descriptor_index{};
    std::uint32_t buffer_type{}; // static_cast<u32>(Shader::BufferType)
    std::array<std::uint32_t, 4> raw_vsharp{}; // Before post_op; zero for special buffers.
    std::uint64_t effective_address{}; // Guest address, zero for special buffers.
    std::uint64_t effective_size{}; // Clamped guest size; GDS uses its 64 KiB size.
    std::uint64_t backing_id{}; // reinterpret_cast<uintptr_t>(VideoCore::Buffer*).
    std::uint64_t buffer_offset{}; // Offset passed to IsBufferAccessed.
    std::uint64_t tracked_size{}; // Size passed to IsBufferAccessed.
    bool is_written{};
    bool check_performed{}; // False for null/stream/special cases with no check.
    bool hit{}; // Actual IsBufferAccessed result, never an inferred hazard.
    bool needs_before{};
    bool needs_after{};
    bool is_special{};
};

// Begin/Buffer/End or Begin/Abort must all run on the same serialized CP/Stage-B thread.
// Capture starts 30 seconds after the first BeginDispatch, stops after 3000 dispatches or
// 32 MiB, and is disabled unless both ISS_GPU_HAZARD_DIAG=1 and ISS_GPU_HAZARD_LOG are set.
bool Enabled() noexcept;
void BeginDispatch(std::uint64_t shader_hash, std::uint64_t permutation_hash,
                   std::uint64_t tick, std::uint32_t queue, std::uint32_t stage,
                   bool inherited_needs_barrier) noexcept;
void Buffer(const BufferCheck& check) noexcept;
// flushed means Runtime::FlushBarriers was called for this dispatch, not GPU completion.
void EndDispatch(bool needs_barrier, bool flushed) noexcept;
void AbortDispatch(std::uint32_t reason) noexcept;

} // namespace iss::hazard_probe
