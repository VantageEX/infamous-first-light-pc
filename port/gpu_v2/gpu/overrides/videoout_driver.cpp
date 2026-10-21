// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstdlib>
#include "common/assert.h"
#include "common/debug.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "runtime_timeline.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

#include <atomic>

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Libraries::VideoOut {

namespace {
std::atomic<u64> next_trace_id{1};
}

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req, u64 vblank_count) {
    const u64 present_start = iss::timeline::enabled ? Common::Perf::NowNs() : 0;
    const u64 frame_address = iss::timeline::enabled
                                  ? reinterpret_cast<u64>(req.frame) : 0;
    const u64 frame_id = iss::timeline::enabled ? req.frame->id : 0;
    const u64 frame_ready_tick = iss::timeline::enabled ? req.frame->ready_tick : 0;
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_flip_begin", req.trace_id, vblank_count,
                            static_cast<u64>(static_cast<s64>(req.index)),
                            static_cast<u64>(req.eop), static_cast<u64>(req.flip_arg),
                            frame_address);
    }
    // Update HDR status before presenting.
    presenter->SetHDR(req.port->is_hdr);

    // Present the frame.
    presenter->Present(req.frame);
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_present_return", req.trace_id, vblank_count,
                            frame_address, frame_ready_tick,
                            frame_id, 0, Common::Perf::NowNs() - present_start);
    }

    // Update flip status.
    auto* port = req.port;
    u64 flip_count{};
    s32 flip_pending{};
    s32 gc_pending{};
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
        flip_count = flip_status.count;
        flip_pending = flip_status.flip_pending_num;
        gc_pending = flip_status.gc_queue_num;
    }
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_flip_status", req.trace_id, vblank_count,
                            flip_count, static_cast<u64>(flip_pending),
                            static_cast<u64>(gc_pending),
                            static_cast<u64>(static_cast<s64>(req.index)));
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset prev flip label
    const s32 previous_index = port->prev_index;
    if (previous_index != -1) {
        port->buffer_labels[previous_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_flip_events_labels_done", req.trace_id, vblank_count,
                            static_cast<u64>(static_cast<s64>(previous_index)),
                            static_cast<u64>(static_cast<s64>(req.index)),
                            static_cast<u64>(port->flip_events.size()),
                            static_cast<u64>(req.eop));
    }
}

void VideoOutDriver::DrawBlankFrame() {
    const auto empty_frame = presenter->PrepareBlankFrame(true);
    presenter->Present(empty_frame, false, false);
}

void VideoOutDriver::DrawLastFrame() {
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
    }
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    u64 trace_id{};
    s32 pending{};
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
        trace_id = next_trace_id.fetch_add(1, std::memory_order_relaxed);
        pending = port->flip_status.flip_pending_num;
    }
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_flip_accept", trace_id, 0,
                            static_cast<u64>(static_cast<s64>(index)),
                            static_cast<u64>(is_eop), static_cast<u64>(flip_arg),
                            static_cast<u64>(pending));
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() {
            SubmitFlipInternal(port, index, flip_arg, is_eop, trace_id);
        });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop, trace_id);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop,
                                        u64 trace_id) {
    const u64 prepare_start = iss::timeline::enabled ? Common::Perf::NowNs() : 0;
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_prepare_begin", trace_id, 0,
                            static_cast<u64>(static_cast<s64>(index)),
                            static_cast<u64>(is_eop));
    }
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }
    if (iss::timeline::enabled) {
        iss::timeline::Emit("vo_prepare_end", trace_id, 0,
                            reinterpret_cast<u64>(frame), frame->ready_tick,
                            static_cast<u64>(frame->id),
                            static_cast<u64>(static_cast<s64>(index)),
                            Common::Perf::NowNs() - prepare_start);
    }

    {
        std::scoped_lock lock{mutex};
        requests.push({
            .frame = frame,
            .port = port,
            .flip_arg = flip_arg,
            .index = index,
            .eop = is_eop,
            .trace_id = trace_id,
        });
        if (iss::timeline::enabled) {
            iss::timeline::Emit("vo_queue_push", trace_id, 0,
                                requests.size(), reinterpret_cast<u64>(frame),
                                frame->ready_tick, static_cast<u64>(frame->id));
        }
    }
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this](u64 vblank_count, s32 flip_rate,
                                        u64& queue_depth) -> Request {
        std::scoped_lock lk{mutex};
        queue_depth = requests.size();
        if (iss::timeline::enabled) {
            iss::timeline::Emit("vo_poll", requests.empty() ? 0 : requests.front().trace_id,
                                vblank_count, queue_depth,
                                static_cast<u64>(flip_rate),
                                static_cast<u64>(!requests.empty()));
        }
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            if (iss::timeline::enabled) {
                iss::timeline::Emit("vo_queue_pop", request.trace_id, vblank_count,
                                    requests.size(), reinterpret_cast<u64>(request.frame),
                                    request.frame->ready_tick,
                                    static_cast<u64>(request.frame->id));
            }
            return request;
        }
        return {};
    };

    // ISS_FLIP_RATE=n: the flips are taken every n+1 vblanks whatever rate the guest set (its
    // 30 FPS option sets 1: with BB_VBLANK_HZ=240 the flips snap to a 120 Hz grid).
    const s32 forced_flip_rate = [] {
        const char* value = std::getenv("ISS_FLIP_RATE");
        return value && *value ? std::atoi(value) : -1;
    }();
    // BB_FPS_LIMIT=n (launcher "FPS limit", 0 = off): a flip is taken only once 1/n s passed since
    // the last one (half a vblank of tolerance, so jitter does not cost a whole tick); the guest
    // waits for its queued flip, which paces the game itself.
    const u32 frame_limit = EmulatorSettings.GetFrameLimit();
    const auto min_flip_interval =
        frame_limit ? std::chrono::nanoseconds(1000000000 / frame_limit) - vblank_period / 2
                    : std::chrono::nanoseconds(0);
    if (frame_limit) std::fprintf(stderr, "VIDEOOUT fps_limit=%u\n", frame_limit);
    auto last_flip = std::chrono::steady_clock::time_point{};
    s32 seen_flip_rate = -1;
    bool was_paused = false;
    while (!token.stop_requested()) {
        timer.Start();
        if (main_port.flip_rate != seen_flip_rate) {
            seen_flip_rate = main_port.flip_rate;
            std::fprintf(stderr, "VIDEOOUT flip_rate=%d (guest) forced=%d\n", seen_flip_rate,
                         forced_flip_rate);
        }
        const s32 flip_rate = forced_flip_rate >= 0 ? forced_flip_rate : main_port.flip_rate;

        const bool paused = DebugState.IsGuestThreadsPaused();
        if (iss::timeline::enabled && paused != was_paused) {
            iss::timeline::Emit("vo_pause_change", 0, 0,
                                static_cast<u64>(paused));
        }
        was_paused = paused;
        if (paused) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        const u64 vblank_count = vblank_status.count;
        u64 presented_trace_id{};
        const bool limit_ok =
            frame_limit == 0 || std::chrono::steady_clock::now() - last_flip >= min_flip_interval;
        if (vblank_status.count % (flip_rate + 1) == 0 && limit_ok) {
            u64 queue_depth{};
            const auto request = receive_request(vblank_count, flip_rate,
                                                 queue_depth);
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                presented_trace_id = request.trace_id;
                last_flip = std::chrono::steady_clock::now();
                Flip(request, vblank_count);
                FRAME_END;
            }
        }

        u64 new_vblank_count{};
        u64 vblank_event_count{};
        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
            new_vblank_count = vblank_status.count;
            vblank_event_count = main_port.vblank_events.size();
        }
        if (iss::timeline::enabled) {
            iss::timeline::Emit("vo_vblank", presented_trace_id, vblank_count,
                                new_vblank_count,
                                static_cast<u64>(main_port.flip_rate),
                                vblank_event_count);
        }

        timer.End();
    }
}

} // namespace Libraries::VideoOut
