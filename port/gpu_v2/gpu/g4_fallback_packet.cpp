// SPDX-License-Identifier: GPL-2.0-or-later
#include "g4_fallback_packet.h"
#include "srt_walker_capture.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace iss::g4 {

FallbackGate FallbackPacket::Gate() const noexcept {
    if (guest.qid == 0 || guest.qid >= 57 || guest.generation == 0 ||
        guest.submission_sequence == 0 || guest.packet_sequence == 0) {
        return FallbackGate::InvalidIdentity;
    }
    if (required_compute_state_bytes == 0 ||
        compute_state.size() != required_compute_state_bytes)
        return FallbackGate::MissingComputeState;
    if (needs_indirect_args &&
        (!indirect_args || indirect_args->gpu_address == 0 ||
         indirect_args->backing_generation == 0 || indirect_args->size == 0 ||
         indirect_args->gpu_address >
             std::numeric_limits<std::uint64_t>::max() - indirect_args->size)) {
        return FallbackGate::MissingIndirectReference;
    }
    const auto code_bytes = shader.code.size() * sizeof(std::uint32_t);
    if (shader.guest_base == 0 || shader.hash == 0 || shader.code.empty() ||
        shader.code.size() > 0x4000 ||
        shader.guest_base > std::numeric_limits<std::uint64_t>::max() - code_bytes ||
        std::none_of(srt_reads.begin(), srt_reads.end(), [&](const auto& read) {
            return read.guest_address == shader.guest_base && read.bytes.size() >= code_bytes &&
                   std::memcmp(read.bytes.data(), shader.code.data(), code_bytes) == 0;
        })) return FallbackGate::MissingShaderCode;
    if (srt_proof.coverage != SrtCoverage::Complete ||
        !srt_proof.flattened_from_owned_reads || !srt_proof.no_late_guest_reads ||
        !srt_proof.generations_stable ||
        (srt_proof.no_guest_reads ? !srt_reads.empty()
                                  : (!srt_lease || !srt_lease->Valid())) ||
        std::any_of(srt_reads.begin(), srt_reads.end(), [](const auto& read) {
            return read.guest_address == 0 || read.memory_generation == 0 || read.bytes.empty() ||
                   read.guest_address >
                       std::numeric_limits<std::uint64_t>::max() - read.bytes.size();
        })) {
        return FallbackGate::IncompleteSrtCoverage;
    }
    return FallbackGate::Ready;
}

bool FallbackPacket::PublishSrtIfValid(const std::function<bool()>& publish) const {
    if (Gate() != FallbackGate::Ready) return false;
    if (srt_proof.no_guest_reads) return publish();
    return srt_lease->PublishIfValid(publish);
}

FallbackPacket CaptureFallbackPacket(
    GuestPacketIdentity guest, std::span<const std::byte> compute_state,
    std::size_t required_compute_state_bytes,
    std::span<const std::uint32_t, FallbackUserDataWords> user_data,
    std::span<const std::uint32_t> flattened_srt,
    std::span<const SrtReadSnapshot> srt_reads, SrtSnapshotProof proof,
    std::optional<IndirectArgsRef> indirect_args, bool needs_indirect_args,
    std::shared_ptr<SrtReadWatchLease> srt_lease, ShaderCodeSnapshot shader,
    std::shared_ptr<const SrtCaptureResult> srt_capture, Shader::PushData push_data,
    std::uint64_t border_color_base, bool serial_image) {
    FallbackPacket packet{};
    packet.guest = guest;
    packet.compute_state.assign(compute_state.begin(), compute_state.end());
    packet.required_compute_state_bytes = required_compute_state_bytes;
    std::copy(user_data.begin(), user_data.end(), packet.user_data.begin());
    packet.flattened_srt.assign(flattened_srt.begin(), flattened_srt.end());
    packet.srt_reads.assign(srt_reads.begin(), srt_reads.end());
    packet.srt_proof = proof;
    packet.srt_lease = std::move(srt_lease);
    packet.srt_capture = std::move(srt_capture);
    packet.push_data = push_data;
    packet.indirect_args = indirect_args;
    packet.shader = std::move(shader);
    packet.needs_indirect_args = needs_indirect_args;
    packet.border_color_base = border_color_base;
    packet.serial_image = serial_image;
    return packet;
}

FallbackState FallbackRequest::State() const noexcept {
    std::scoped_lock lock{mutex};
    return state;
}

std::optional<QidTick> FallbackRequest::GraphicsResult() const noexcept {
    std::scoped_lock lock{mutex};
    return state == FallbackState::GraphicsSealed ? graphics_result : std::nullopt;
}

FallbackMailbox::Request FallbackMailbox::Enqueue(FallbackPacket packet,
                                                  const SealRequest& sealed_asc_prefix) {
    const auto prefix = sealed_asc_prefix.target;
    if (packet.Gate() != FallbackGate::Ready ||
        sealed_asc_prefix.State() != SealState::Done ||
        sealed_asc_prefix.SealedTick() < prefix.tick ||
        prefix.lane != CpLane::Asc || prefix.tick == 0 ||
        prefix.qid != packet.guest.qid ||
        prefix.generation != packet.guest.generation) return {};
    std::scoped_lock lock{mutex};
    if (stopped) return {};
    auto& latest = latest_guest[packet.guest.qid];
    if (latest.generation > packet.guest.generation ||
        (latest.generation == packet.guest.generation &&
         (latest.submission_sequence > packet.guest.submission_sequence ||
          (latest.submission_sequence == packet.guest.submission_sequence &&
           latest.packet_sequence >= packet.guest.packet_sequence)))) return {};
    latest = packet.guest;
    auto request = Request{new FallbackRequest{std::move(packet), prefix}};
    pending.push_back(request);
    return request;
}

FallbackMailbox::Request FallbackMailbox::TakeAtGraphicsSafePoint(bool at_safe_point) {
    if (!at_safe_point) return {};
    std::scoped_lock lock{mutex};
    if (stopped || active || pending.empty()) return {};
    auto request = pending.front();
    pending.pop_front();
    {
        std::scoped_lock request_lock{request->mutex};
        request->state = FallbackState::RecordingGraphics;
    }
    active = request;
    return request;
}

bool FallbackMailbox::MarkGraphicsRecorded(const Request& request,
                                            std::uint64_t current_guest_generation,
                                            GraphicsRecordingProof proof) {
    if (!request) return false;
    std::scoped_lock lock{request->mutex};
    if (request->state != FallbackState::RecordingGraphics ||
        request->packet.Gate() != FallbackGate::Ready ||
        request->packet.guest.generation != current_guest_generation ||
        proof.graphics_tick == 0 || !proof.graphics_pipeline_partition ||
        proof.expected_flattened_words != request->packet.flattened_srt.size() ||
        !proof.no_guest_dereference || !proof.all_resources_prepared) return false;
    request->recorded_graphics_tick = proof.graphics_tick;
    request->state = FallbackState::GraphicsRecorded;
    return true;
}

bool FallbackMailbox::PublishGraphicsSeal(const Request& request, const SealRequest& gfx_seal,
                                           std::uint64_t current_guest_generation) {
    const auto gfx_token = gfx_seal.target;
    if (!request || gfx_seal.State() != SealState::Done ||
        gfx_seal.SealedTick() < gfx_token.tick ||
        gfx_token.lane != CpLane::Graphics || gfx_token.qid != 0 ||
        gfx_token.generation == 0 || gfx_token.tick == 0) return false;
    std::scoped_lock lock{mutex, request->mutex};
    if (active != request || stopped ||
        request->recorded_graphics_tick != gfx_token.tick ||
        (last_graphics_result &&
         (last_graphics_result->generation != gfx_token.generation ||
          last_graphics_result->tick >= gfx_token.tick))) return false;
    if (request->state != FallbackState::GraphicsRecorded ||
        request->packet.guest.generation != current_guest_generation) return false;
    request->graphics_result = gfx_token;
    request->state = FallbackState::GraphicsSealed;
    last_graphics_result = gfx_token;
    active.reset();
    return true;
}

void FallbackMailbox::Fail(const Request& request) noexcept {
    if (!request) return;
    std::scoped_lock lock{mutex, request->mutex};
    if (request->state != FallbackState::GraphicsSealed &&
        request->state != FallbackState::Cancelled) request->state = FallbackState::Failed;
    if (active == request) active.reset();
    std::erase(pending, request);
}

void FallbackMailbox::Stop() noexcept {
    std::scoped_lock lock{mutex};
    stopped = true;
    for (const auto& request : pending) {
        std::scoped_lock request_lock{request->mutex};
        request->state = FallbackState::Cancelled;
    }
    if (active) {
        std::scoped_lock request_lock{active->mutex};
        if (active->state != FallbackState::GraphicsSealed &&
            active->state != FallbackState::Failed) active->state = FallbackState::Cancelled;
    }
    pending.clear();
    active.reset();
}

} // namespace iss::g4
