// SPDX-License-Identifier: GPL-2.0-or-later
// ASC -> graphics fallback handoff. This protocol owns frontend inputs; it does not
// record commands or make the SRT walker safe by itself.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "g4_cp_coordination.h"
#include "shader_recompiler/resource.h"

namespace iss::g4 {

class SrtReadWatchLease;
struct SrtCaptureResult;

inline constexpr std::size_t FallbackUserDataWords = 16;

struct GuestPacketIdentity {
    std::uint32_t qid{};
    std::uint64_t generation{};
    std::uint64_t submission_sequence{};
    std::uint64_t packet_sequence{};
};

struct SrtReadSnapshot {
    std::uint64_t guest_address{};
    std::uint64_t memory_generation{};
    std::vector<std::byte> bytes{};
};

// GPU-written indirect arguments must remain a GPU resource reference. A CPU copy
// would silently replace a producer's result with stale dimensions.
struct IndirectArgsRef {
    std::uint64_t gpu_address{};
    std::uint64_t backing_generation{};
    std::uint32_t size{};
};

struct ShaderCodeSnapshot {
    std::uint64_t guest_base{};
    std::uint64_t hash{};
    std::vector<std::uint32_t> code{};
};

enum class SrtCoverage : std::uint8_t {
    Unknown,
    Complete, // every transitive guest read and alias is captured and versioned
};

// The resource preparer must supply this proof after the final generation check.
// In particular, the current walker_func cannot provide Complete by itself: it
// dereferences guest addresses without returning its transitive read set.
struct SrtSnapshotProof {
    SrtCoverage coverage{SrtCoverage::Unknown};
    bool flattened_from_owned_reads{};
    bool no_late_guest_reads{};
    bool generations_stable{};
    bool no_guest_reads{}; // e.g. no walker_func, only copied user-data words
};

enum class FallbackGate : std::uint8_t {
    Ready,
    InvalidIdentity,
    MissingComputeState,
    MissingIndirectReference,
    MissingShaderCode,
    IncompleteSrtCoverage,
};

struct FallbackPacket {
    GuestPacketIdentity guest{};
    std::vector<std::byte> compute_state{}; // full ComputeProgram bytes, including CS UD
    std::size_t required_compute_state_bytes{}; // caller passes sizeof(AmdGpu::ComputeProgram)
    std::array<std::uint32_t, FallbackUserDataWords> user_data{};
    std::vector<std::uint32_t> flattened_srt{};
    std::vector<SrtReadSnapshot> srt_reads{};
    SrtSnapshotProof srt_proof{};
    std::shared_ptr<SrtReadWatchLease> srt_lease{}; // keeps guest writes watched through publish
    std::shared_ptr<const SrtCaptureResult> srt_capture{};
    Shader::PushData push_data{};
    std::optional<IndirectArgsRef> indirect_args{};
    ShaderCodeSnapshot shader{};
    bool needs_indirect_args{};
    // Captured with the ASC packet. The first serial image route rejects samplers,
    // but a later sampler route must not borrow G's live border-color register.
    std::uint64_t border_color_base{};
    bool serial_image{};

    [[nodiscard]] FallbackGate Gate() const noexcept;
    [[nodiscard]] bool PublishSrtIfValid(const std::function<bool()>& publish) const;
};

// Packet publication copies all frontend inputs. The source spans may be mutated
// immediately after this call. No guest or other owner's scratch pointer is saved.
[[nodiscard]] FallbackPacket CaptureFallbackPacket(
    GuestPacketIdentity guest, std::span<const std::byte> compute_state,
    std::size_t required_compute_state_bytes,
    std::span<const std::uint32_t, FallbackUserDataWords> user_data,
    std::span<const std::uint32_t> flattened_srt,
    std::span<const SrtReadSnapshot> srt_reads, SrtSnapshotProof proof,
    std::optional<IndirectArgsRef> indirect_args, bool needs_indirect_args,
    std::shared_ptr<SrtReadWatchLease> srt_lease = {},
    ShaderCodeSnapshot shader = {},
    std::shared_ptr<const SrtCaptureResult> srt_capture = {},
    Shader::PushData push_data = {},
    std::uint64_t border_color_base = 0, bool serial_image = false);

// Liverpool passes its AmdGpu::ComputeProgram here so the entire CS register
// block is copied with the compiler's actual layout and size.
template <class ComputeProgram>
[[nodiscard]] FallbackPacket CaptureTypedFallbackPacket(
    GuestPacketIdentity guest, const ComputeProgram& cs,
    std::span<const std::uint32_t, FallbackUserDataWords> user_data,
    std::span<const std::uint32_t> flattened_srt,
    std::span<const SrtReadSnapshot> srt_reads, SrtSnapshotProof proof,
    std::optional<IndirectArgsRef> indirect_args, bool needs_indirect_args,
    std::shared_ptr<SrtReadWatchLease> srt_lease = {},
    ShaderCodeSnapshot shader = {},
    std::shared_ptr<const SrtCaptureResult> srt_capture = {},
    Shader::PushData push_data = {},
    std::uint64_t border_color_base = 0, bool serial_image = false) {
    static_assert(std::is_trivially_copyable_v<ComputeProgram>);
    return CaptureFallbackPacket(guest, std::as_bytes(std::span{&cs, 1}), sizeof(cs),
                                 user_data, flattened_srt, srt_reads, proof,
                                 indirect_args, needs_indirect_args, std::move(srt_lease),
                                 std::move(shader), std::move(srt_capture), push_data,
                                 border_color_base, serial_image);
}

// Supplied by G only after recording the fallback. Never carry a C Program::info
// pointer in the packet or use it to satisfy this proof.
struct GraphicsRecordingProof {
    std::uint64_t graphics_tick{};
    std::size_t expected_flattened_words{}; // from the G partition's Program::info
    bool graphics_pipeline_partition{};
    bool no_guest_dereference{};
    bool all_resources_prepared{};
};

enum class FallbackState : std::uint8_t {
    Queued, RecordingGraphics, GraphicsRecorded, GraphicsSealed,
    Failed, Cancelled,
};

class FallbackRequest {
public:
    [[nodiscard]] const FallbackPacket& Packet() const noexcept { return packet; }
    [[nodiscard]] const QidTick& AscPrefix() const noexcept { return asc_prefix; }
    [[nodiscard]] FallbackState State() const noexcept;
    // Only GraphicsSealed yields a token. C must add it as a dependency before
    // its next packet can consume the fallback's writes.
    [[nodiscard]] std::optional<QidTick> GraphicsResult() const noexcept;

private:
    friend class FallbackMailbox;
    FallbackRequest(FallbackPacket packet_, QidTick asc_prefix_)
        : packet{std::move(packet_)}, asc_prefix{asc_prefix_} {}

    FallbackPacket packet;
    QidTick asc_prefix;
    mutable std::mutex mutex;
    FallbackState state{FallbackState::Queued};
    std::uint64_t recorded_graphics_tick{};
    std::optional<QidTick> graphics_result{};
};

// C queues only after its preceding prefix has actually sealed. G takes a
// request only at a frontend safe point. This class deliberately has no wait or
// reentrant callback API; owner control pumping belongs to the CP coordinator.
class FallbackMailbox {
public:
    using Request = std::shared_ptr<FallbackRequest>;

    [[nodiscard]] Request Enqueue(FallbackPacket packet, const SealRequest& sealed_asc_prefix);
    [[nodiscard]] Request TakeAtGraphicsSafePoint(bool at_safe_point);
    [[nodiscard]] bool MarkGraphicsRecorded(const Request& request,
                                             std::uint64_t current_guest_generation,
                                             GraphicsRecordingProof proof);
    [[nodiscard]] bool PublishGraphicsSeal(const Request& request, const SealRequest& gfx_seal,
                                            std::uint64_t current_guest_generation);
    void Fail(const Request& request) noexcept;
    void Stop() noexcept;

private:
    std::mutex mutex;
    std::deque<Request> pending{};
    Request active{};
    std::array<GuestPacketIdentity, 57> latest_guest{};
    std::optional<QidTick> last_graphics_result{};
    bool stopped{};
};

} // namespace iss::g4
