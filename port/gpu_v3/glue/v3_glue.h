// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3's glue: what gpu_host.cpp (this port's GPU interface) hands to bridge.cpp (the
// fork's emulator entry points).
#pragma once

#include <cstdint>
#include <string>

namespace V3 {

struct HostConfig {
    std::string title, serial, user_dir;
    uint32_t sdk_version{};
    uint32_t psf_attributes{};
    int width{1920}, height{1080};
    uint64_t image_base{};
    uint64_t (*tsc_frequency)(){};
    uint64_t (*process_time_us)(){};
    int32_t* (*errno_location)(){};
};
extern HostConfig g_host;
/// While set (this thread), a failed assert throws std::runtime_error instead of stopping.
extern thread_local bool g_assert_throws;
/// The job file's records compiled before the game starts (precompile_jobs.cpp).
void PrecompileJobs();

} // namespace V3
