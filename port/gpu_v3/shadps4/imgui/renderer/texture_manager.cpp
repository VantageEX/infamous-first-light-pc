// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <deque>
#include <utility>

#include <imgui.h>
#include "common/assert.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/stb.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui_impl_vulkan.h"
#include "texture_manager.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace ImGui {

namespace Core::TextureManager {
struct Inner {
    std::atomic_int count = 0;
    ImTextureID texture_id = nullptr;
    u32 width = 0;
    u32 height = 0;

    Vulkan::UploadTextureData upload_data;

    ~Inner();
};
} // namespace Core::TextureManager

using namespace Core::TextureManager;

RefCountedTexture::RefCountedTexture(Inner* inner) : inner(inner) {
    ++inner->count;
}

RefCountedTexture RefCountedTexture::DecodePngTexture(std::vector<u8> data) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngTexture(std::move(data), core);
    return RefCountedTexture(core);
}

RefCountedTexture RefCountedTexture::DecodePngFile(std::filesystem::path path) {
    const auto core = new Inner;
    Core::TextureManager::DecodePngFile(std::move(path), core);
    return RefCountedTexture(core);
}

RefCountedTexture::RefCountedTexture() : inner(nullptr) {}

RefCountedTexture::RefCountedTexture(const RefCountedTexture& other) : inner(other.inner) {
    if (inner != nullptr) {
        ++inner->count;
    }
}

RefCountedTexture::RefCountedTexture(RefCountedTexture&& other) noexcept : inner(other.inner) {
    other.inner = nullptr;
}

RefCountedTexture& RefCountedTexture::operator=(const RefCountedTexture& other) {
    if (this == &other)
        return *this;
    inner = other.inner;
    if (inner != nullptr) {
        ++inner->count;
    }
    return *this;
}

RefCountedTexture& RefCountedTexture::operator=(RefCountedTexture&& other) noexcept {
    if (this == &other)
        return *this;
    std::swap(inner, other.inner);
    return *this;
}

RefCountedTexture::~RefCountedTexture() {
    if (inner != nullptr) {
        if (inner->count.fetch_sub(1) == 1) {
            delete inner;
        }
    }
}

RefCountedTexture::Image RefCountedTexture::GetTexture() const {
    if (inner == nullptr) {
        return {};
    }
    return Image{
        .im_id = inner->texture_id,
        .width = inner->width,
        .height = inner->height,
    };
}

RefCountedTexture::operator bool() const {
    return inner != nullptr && inner->texture_id != nullptr;
}

struct Job {
    Inner* core;
    std::vector<u8> data;
    std::filesystem::path path;
};

struct UploadJob {
    Inner* core = nullptr;
    Vulkan::UploadTextureData data;
    int tick = 0; // Used to skip the first frame when destroying to await the current frame to draw
    // SHADPS4_SWAPCHAIN_SAFE_WAIT=1 (review 10-15): `tick` counts Submit() calls, which run on
    // every scheduler submit (hundreds per frame), so a destroy could free a texture that a present
    // command buffer recorded but not yet submitted still samples. Destroy only once the presented
    // frame count reaches this (0 = use `tick`).
    u64 present_at = 0;
};

// Frames the presenter has submitted (NotePresented), for UploadJob::present_at.
static std::atomic<u64> g_presented{0};
// When only not-yet-due destroys are queued, Submit clears g_upload_pending (so submitters skip
// the lock again, SUBMIT_LOCK_SKIP) and NotePresented re-arms it once g_presented reaches this.
static std::atomic<u64> g_destroy_due{~0ULL};

static bool g_is_worker_running = false;
static std::jthread g_worker_thread;
static std::condition_variable g_worker_cv;

static std::mutex g_job_list_mtx;
static std::deque<Job> g_job_list;

static std::mutex g_upload_mtx;
static std::deque<UploadJob> g_upload_list;
// Mirrors !g_upload_list.empty(), written under g_upload_mtx after every change; read without it
// by HasPendingUploads (a stale false only delays an upload to the next submit).
static std::atomic<bool> g_upload_pending{false};

namespace Core::TextureManager {

Inner::~Inner() {
    if (upload_data.im_texture != nullptr) {
        // The frame being built now may still draw it: it is presented as g_presented + 1, so
        // wait for + 2.
        const bool by_present = ::Vulkan::SwapchainSafeWait();
        std::unique_lock lk{g_upload_mtx};
        g_upload_list.emplace_back(UploadJob{
            .data = this->upload_data,
            .tick = by_present ? 0 : 2,
            .present_at = by_present ? g_presented.load(std::memory_order_acquire) + 2 : 0,
        });
        g_upload_pending.store(true, std::memory_order_relaxed);
    }
}

void WorkerLoop() {
    Common::SetCurrentThreadName("shadPS4:ImGuiTextureManager");
    std::mutex mtx;
    while (g_is_worker_running) {
        std::unique_lock lk{mtx};
        g_worker_cv.wait(lk);
        if (!g_is_worker_running) {
            break;
        }
        while (true) {
            g_job_list_mtx.lock();
            if (g_job_list.empty()) {
                g_job_list_mtx.unlock();
                break;
            }
            auto [core, png_raw, path] = std::move(g_job_list.front());
            g_job_list.pop_front();
            g_job_list_mtx.unlock();

            if (EmulatorSettings.IsVkCrashDiagnosticEnabled()) {
                // FIXME: Crash diagnostic hangs when building the command buffer here
                continue;
            }

            if (!path.empty()) { // Decode PNG from file
                Common::FS::IOFile file(path, Common::FS::FileAccessMode::Read);
                if (!file.IsOpen()) {
                    LOG_ERROR(ImGui, "Failed to open PNG file: {}", path.string());
                    continue;
                }
                png_raw.resize(file.GetSize());
                file.Seek(0);
                file.ReadRaw<u8>(png_raw.data(), png_raw.size());
                file.Close();
            }

            int width, height;
            const stbi_uc* pixels =
                stbi_load_from_memory(png_raw.data(), png_raw.size(), &width, &height, nullptr, 4);

            auto texture = Vulkan::UploadTexture(pixels, vk::Format::eR8G8B8A8Unorm, width, height,
                                                 width * height * 4 * sizeof(stbi_uc));
            stbi_image_free((void*)pixels);

            core->upload_data = texture;
            core->width = width;
            core->height = height;

            std::unique_lock upload_lk{g_upload_mtx};
            g_upload_list.emplace_back(UploadJob{
                .core = core,
            });
            g_upload_pending.store(true, std::memory_order_relaxed);
        }
    }
}

void StartWorker() {
    ASSERT(!g_is_worker_running);
    g_worker_thread = std::jthread(WorkerLoop);
    g_is_worker_running = true;
}

void StopWorker() {
    ASSERT(g_is_worker_running);
    g_is_worker_running = false;
    g_worker_cv.notify_one();
}

void DecodePngTexture(std::vector<u8> data, Inner* core) {
    ++core->count;
    Job job{
        .core = core,
        .data = std::move(data),
    };
    std::unique_lock lk{g_job_list_mtx};
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

void DecodePngFile(std::filesystem::path path, Inner* core) {
    ++core->count;
    Job job{
        .core = core,
        .path = std::move(path),
    };
    std::unique_lock lk{g_job_list_mtx};
    g_job_list.push_back(std::move(job));
    g_worker_cv.notify_one();
}

bool HasPendingUploads() {
    return g_upload_pending.load(std::memory_order_relaxed);
}

void NotePresented() {
    const u64 presented = g_presented.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (presented >= g_destroy_due.load(std::memory_order_acquire)) {
        // Under the list lock (once per frame at most), so Submit's clear cannot overwrite it.
        std::unique_lock lk{g_upload_mtx};
        g_destroy_due.store(~0ULL, std::memory_order_relaxed);
        g_upload_pending.store(!g_upload_list.empty(), std::memory_order_relaxed);
    }
}

void Submit() {
    UploadJob upload;
    {
        std::unique_lock lk{g_upload_mtx};
        if (g_upload_list.empty()) {
            g_upload_pending.store(false, std::memory_order_relaxed);
            return;
        }
        // Upload one texture at a time to avoid slow down
        upload = g_upload_list.front();
        g_upload_list.pop_front();
        if (upload.tick > 0) {
            --upload.tick;
            g_upload_list.emplace_back(upload);
            return;
        }
        if (upload.present_at != 0) {
            const u64 presented = g_presented.load(std::memory_order_acquire);
            if (presented < upload.present_at) {
                g_upload_list.emplace_back(std::move(upload));
                u64 due = ~0ULL;
                for (const UploadJob& job : g_upload_list) {
                    if (job.core != nullptr || job.tick > 0 || job.present_at <= presented) {
                        return; // other work is due: keep the hint set
                    }
                    due = std::min(due, job.present_at);
                }
                // A present landing between the load above and this store re-arms one frame
                // late at worst (the next NotePresented sees due <= presented).
                g_destroy_due.store(due, std::memory_order_release);
                g_upload_pending.store(false, std::memory_order_relaxed);
                return;
            }
        }
        g_upload_pending.store(!g_upload_list.empty(), std::memory_order_relaxed);
    }
    if (upload.core != nullptr) {
        upload.core->upload_data.Upload();
        upload.core->texture_id = upload.core->upload_data.im_texture;
        if (upload.core->count.fetch_sub(1) == 1) {
            delete upload.core;
        }
    } else {
        upload.data.Destroy();
    }
}
} // namespace Core::TextureManager

} // namespace ImGui