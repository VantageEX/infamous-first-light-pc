// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

// GPU black box (ISS_GPU_BLACKBOX, on unless =0). Every marked GPU command (command processor
// draws and dispatches, native segments and each of their draws) writes its number with
// VK_AMD_buffer_marker to host memory when the GPU starts it and when it completes, and its
// description (path, shaders, targets, blend, pause) stays in a ring on the CPU. A diary keeps the
// recent events (pipeline compiles, image creation and deletion, pause, native learning). A
// watchdog writes gpu-stall-N.txt when the GPU stops advancing for 1 s, before the driver reset
// (2 s), and gpu-crash.txt/.json at device lost, in the run folder. scripts/gpu_crash_report.py
// explains them and compares all captured crashes.
// Implementation: gpu_blackbox.inl (in overrides/vk_scheduler.cpp).

#include <cstdint>

namespace iss::blackbox {

/// ISS_GPU_BLACKBOX (default on). Valid once the Vulkan device exists.
bool On();

/// Diary categories: each keeps its own recent events in the reports.
enum Category : int { General = 0, Compile = 1, Image = 2 };

/// A diary event (any thread), printf style, kept with its time.
void Event(int category, const char* format, ...);

/// An image the texture cache deleted (its uid): the reports flag commands that still use it.
void ImageDeleted(std::uint64_t uid, std::uint64_t address);

} // namespace iss::blackbox
