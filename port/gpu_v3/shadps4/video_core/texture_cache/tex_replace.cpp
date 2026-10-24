// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <xxhash.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "common/logging/log.h"
#include "common/stb.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/tex_replace.h"

namespace VideoCore::TexReplace {

namespace {

const bool g_flag = [] {
    const char* v = std::getenv("SHADPS4_TEX_REPLACE");
    return v && v[0] == '1';
}();

std::vector<Target> g_targets;
std::once_flag g_load_once;
std::atomic<u32> g_logged{0};

const bool g_auto = [] {
    const char* v = std::getenv("SHADPS4_TEX_REPLACE_AUTO");
    return !(v && v[0] == '0');
}();
std::atomic<bool> g_gamepad{false};
std::mutex g_seen_mutex;
std::vector<std::pair<VAddr, u32>> g_seen; ///< guest ranges of matched textures

std::filesystem::path ExeDir() {
#ifdef _WIN32
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return std::filesystem::current_path();
        }
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    return std::filesystem::path(buf).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

bool LoadPng(const std::filesystem::path& path, std::vector<u8>& rgba, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    const std::vector<u8> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int channels = 0;
    u8* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h,
                                       &channels, 4);
    if (!pixels) {
        return false;
    }
    rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

void Load() {
    const char* dir_env = std::getenv("SHADPS4_TEX_REPLACE_DIR");
    const std::filesystem::path dir =
        dir_env && dir_env[0] ? std::filesystem::path(dir_env) : ExeDir() / "tex_replace";
    const auto manifest = dir / "replacements.txt";
    std::ifstream in(manifest);
    if (!in) {
        LOG_WARNING(Render_Vulkan, "TexReplace: no manifest at {} - nothing replaced",
                    manifest.string());
        return;
    }
    std::string line;
    u32 line_no = 0;
    u32 loaded = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (const auto hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        std::istringstream ls(line);
        std::string key_str, file, opt;
        u32 tex_w = 0, tex_h = 0, x = 0, y = 0;
        if (!(ls >> key_str)) {
            continue; // blank or comment
        }
        if (!(ls >> tex_w >> tex_h >> x >> y >> file)) {
            LOG_WARNING(Render_Vulkan, "TexReplace: line {} not understood", line_no);
            continue;
        }
        ls >> opt;
        const bool flip = opt != "noflip";
        const u64 key = std::strtoull(key_str.c_str(), nullptr, 16);

        Patch patch{};
        int w = 0, h = 0;
        std::vector<u8> rgba;
        if (!LoadPng(dir / file, rgba, w, h)) {
            LOG_WARNING(Render_Vulkan, "TexReplace: line {}: cannot read {} as PNG", line_no, file);
            continue;
        }
        if (w <= 0 || h <= 0 || x + u32(w) > tex_w || y + u32(h) > tex_h) {
            LOG_WARNING(Render_Vulkan,
                        "TexReplace: line {}: {} is {}x{} at {},{} - outside the {}x{} texture",
                        line_no, file, w, h, x, y, tex_w, tex_h);
            continue;
        }
        patch.x = x;
        patch.width = u32(w);
        patch.height = u32(h);
        patch.y = flip ? tex_h - y - u32(h) : y;
        patch.rgba.resize(rgba.size());
        const size_t row = size_t(w) * 4;
        for (u32 r = 0; r < u32(h); ++r) {
            const u32 src_row = flip ? u32(h) - 1 - r : r;
            std::memcpy(patch.rgba.data() + r * row, rgba.data() + src_row * row, row);
        }

        Target* target = nullptr;
        for (auto& t : g_targets) {
            if (t.key == key && t.width == tex_w && t.height == tex_h) {
                target = &t;
            }
        }
        if (!target) {
            target = &g_targets.emplace_back(Target{key, tex_w, tex_h, {}});
        }
        target->patches.push_back(std::move(patch));
        ++loaded;
    }
    LOG_INFO(Render_Vulkan, "TexReplace: {} pictures for {} textures from {}", loaded,
             g_targets.size(), manifest.string());
}

} // Anonymous namespace

bool Enabled() {
    if (!g_flag) {
        return false;
    }
    std::call_once(g_load_once, Load);
    return !g_targets.empty();
}

const Target* Find(const ImageInfo& info) {
    if (info.num_samples != 1 || info.resources.levels != 1 || info.resources.layers != 1 ||
        info.num_bits != 32 || info.props.is_block || info.type != AmdGpu::ImageType::Color2D) {
        return nullptr;
    }
    bool size_hit = false;
    for (const auto& t : g_targets) {
        size_hit |= t.width == info.size.width && t.height == info.size.height;
    }
    if (!size_hit || info.guest_address == 0 || info.guest_size == 0) {
        return nullptr;
    }
    const u64 key =
        XXH3_64bits(reinterpret_cast<const void*>(info.guest_address), info.guest_size);
    const Target* found = nullptr;
    for (const auto& t : g_targets) {
        if (t.key == key && t.width == info.size.width && t.height == info.size.height) {
            found = &t;
        }
    }
    if (found) {
        std::scoped_lock lk{g_seen_mutex};
        const std::pair<VAddr, u32> range{info.guest_address, info.guest_size};
        if (std::find(g_seen.begin(), g_seen.end(), range) == g_seen.end()) {
            g_seen.push_back(range);
        }
    }
    if (g_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
        LOG_INFO(Render_Vulkan, "TexReplace: {}x{} texture at {:#x} ({} bytes, vk format {}) key {:016x}: {}",
                 info.size.width, info.size.height, info.guest_address, info.guest_size,
                 static_cast<u32>(info.pixel_format), key,
                 found ? fmt::format("{} pictures applied", found->patches.size()) : "no match");
    }
    return found;
}

bool AutoSwitch() {
    return g_flag && g_auto;
}

bool PicturesActive() {
    return !g_gamepad.load(std::memory_order_relaxed);
}

void NoteInputDevice(bool gamepad) {
    if (g_gamepad.load(std::memory_order_relaxed) == gamepad ||
        g_gamepad.exchange(gamepad, std::memory_order_relaxed) == gamepad) {
        return;
    }
    std::vector<std::pair<VAddr, u32>> ranges;
    {
        std::scoped_lock lk{g_seen_mutex};
        ranges = g_seen;
    }
    LOG_INFO(Render_Vulkan, "TexReplace: {} input - {} glyphs ({} textures re-uploaded)",
             gamepad ? "pad" : "keyboard / mouse", gamepad ? "game's own" : "replaced",
             ranges.size());
    if (ranges.empty()) {
        return;
    }
    // On a thread of its own, like a game thread's CPU write (Opus review 10-18): the invalidation
    // may wait on the command processor (a GPU-written range), and the caller is the SDL thread,
    // which the present thread can wait on (ImGui IME callback) - a cycle. Nobody waits on this one.
    std::thread([ranges = std::move(ranges)] {
        for (const auto& [addr, size] : ranges) {
            Core::Memory::Instance()->InvalidateMemory(addr, size);
        }
    }).detach();
}

bool SwapRedBlue(const ImageInfo& info) {
    switch (info.pixel_format) {
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eB8G8R8A8Snorm:
    case vk::Format::eB8G8R8A8Uint:
        return true;
    default:
        return false;
    }
}

} // namespace VideoCore::TexReplace
