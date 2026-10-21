// SPDX-License-Identifier: GPL-2.0-or-later
// C-owner pause while a graphics safe-point request owns a prepared upload.
// Liverpool serializes calls with submit_mutex; no PM4 may run from any ASC qid
// until the originating qid has observed the terminal result.
#pragma once

#include <cstdint>
#include <utility>

#include "g4_graphics_control.h"

namespace iss::g4 {

class AscOwnerPause {
public:
    using Request = GraphicsControlMailbox::Request;
    enum class Gate : std::uint8_t { Open, Waiting, ResumeOrigin, ReleaseGuest };
    struct Decision {
        Gate gate{Gate::Open};
        std::uint32_t qid{};
        std::uint64_t prefix_tick{};
    };

    bool Begin(std::uint32_t qid, Request request, std::uint64_t sealed_tick) noexcept {
        if (active || !request || qid == 0 || qid >= 57 || sealed_tick == 0) return false;
        origin_qid = qid;
        prefix_tick = sealed_tick;
        paused_ack = false;
        active = std::move(request);
        return true;
    }

    bool BeginGuest(Request request, std::uint64_t sealed_tick) noexcept {
        if (active || !request || sealed_tick == 0) return false;
        origin_qid = 0;
        prefix_tick = sealed_tick;
        paused_ack = false;
        active = std::move(request);
        return true;
    }

    [[nodiscard]] bool Active() const noexcept { return static_cast<bool>(active); }
    [[nodiscard]] bool Acknowledged() const noexcept {
        return active && paused_ack && Next().gate == Gate::Waiting;
    }
    [[nodiscard]] std::uint64_t PrefixTick() const noexcept { return prefix_tick; }
    bool AcknowledgePaused(std::uint64_t expected_prefix_tick,
                           std::uint64_t submitted_tick, bool scheduler_idle) noexcept {
        if (Next().gate != Gate::Waiting || prefix_tick != expected_prefix_tick ||
            !scheduler_idle ||
            submitted_tick < prefix_tick) return false;
        paused_ack = true;
        return true;
    }

    [[nodiscard]] Decision Next() const noexcept {
        if (!active) return {};
        const auto state = active->Status();
        if (state == GraphicsControlMailbox::Control::State::Queued ||
            state == GraphicsControlMailbox::Control::State::Running)
            return {Gate::Waiting, origin_qid, prefix_tick};
        return origin_qid ? Decision{Gate::ResumeOrigin, origin_qid, prefix_tick}
                          : Decision{Gate::ReleaseGuest, 0, prefix_tick};
    }

    // Called immediately before ProcessAsc resumes the terminal request's origin.
    // The choice of qid is already fixed, and the coroutine may start another
    // owning control during that resume.
    bool AcknowledgeOriginResume(std::uint32_t qid) noexcept {
        if (Next().gate != Gate::ResumeOrigin || qid != origin_qid) return false;
        active.reset();
        origin_qid = 0;
        prefix_tick = 0;
        paused_ack = false;
        return true;
    }

    bool AcknowledgeGuestRelease() noexcept {
        if (Next().gate != Gate::ReleaseGuest) return false;
        active.reset();
        origin_qid = 0;
        prefix_tick = 0;
        paused_ack = false;
        return true;
    }

    void Stop() noexcept {
        active.reset();
        origin_qid = 0;
        prefix_tick = 0;
        paused_ack = false;
    }

private:
    Request active{};
    std::uint32_t origin_qid{};
    std::uint64_t prefix_tick{};
    bool paused_ack{};
};

} // namespace iss::g4
