// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"

#include "video_core/cache_storage.h"
#include "video_core/cache_write_queue.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

#include <miniz.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <utility>

namespace {

Storage::CacheWriteQueue write_queue;

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

// ISS_CACHE_PREFETCH (default on, =0 off): the cache folder's files read by many threads at
// Open, the preload then reads them from memory. Each file open of the game process takes ~6 ms
// (antivirus/sync filters on every new file of an unsigned process; Python opens the same ones in
// 0.2 ms), one after the other on the preload thread: 4200 pipelines took 20-28 s to preload,
// 95% of it in IOFile::Open while the compiler threads waited (preload-prof2/3 2026-10-18).
// Opens in parallel overlap that latency. The copies are dropped at FinishPreload.
std::mutex prefetch_mutex;
std::unordered_map<std::string, std::vector<u8>> prefetched;

bool PrefetchEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_CACHE_PREFETCH");
        return !value || value[0] != '0';
    }();
    return enabled;
}

void Prefetch(const std::filesystem::path& folder) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::filesystem::path> paths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator{folder, ec}) {
        if (entry.is_regular_file(ec)) paths.push_back(entry.path());
    }
    if (paths.empty()) return;
    const u32 threads = std::clamp<u32>(std::thread::hardware_concurrency(), 4U, 24U);
    std::atomic<size_t> next{0};
    std::atomic<u64> bytes{0};
    std::vector<std::pair<std::string, std::vector<u8>>> results(paths.size());
    {
        std::vector<std::jthread> workers;
        for (u32 t = 0; t < threads; ++t) {
            workers.emplace_back([&] {
                for (size_t i = next.fetch_add(1); i < paths.size(); i = next.fetch_add(1)) {
                    using namespace Common::FS;
                    const auto file = IOFile{paths[i], FileAccessMode::Read};
                    if (!file.IsOpen()) continue;
                    std::vector<u8> data(file.GetSize());
                    if (file.Read(data) != data.size()) continue;
                    bytes.fetch_add(data.size(), std::memory_order_relaxed);
                    results[i] = {paths[i].string(), std::move(data)};
                }
            });
        }
    }
    std::scoped_lock lk{prefetch_mutex};
    for (auto& [name, data] : results) {
        if (!name.empty()) prefetched.emplace(std::move(name), std::move(data));
    }
    std::fprintf(stderr, "CACHE_PREFETCH files=%zu mb=%.1f threads=%u ms=%.0f\n", prefetched.size(),
                 double(bytes.load()) / 1048576.0, threads,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count());
}

/// The prefetched bytes of a file, moved out (each is read once by the preload).
bool TakePrefetched(const std::filesystem::path& path, std::vector<u8>& out) {
    std::scoped_lock lk{prefetch_mutex};
    if (prefetched.empty()) return false;
    const auto it = prefetched.find(path.string());
    if (it == prefetched.end()) return false;
    out = std::move(it->second);
    prefetched.erase(it);
    return true;
}

} // namespace

namespace Storage {

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadPS4:PipelineCacheIO");

    while (auto request = write_queue.Next(stoken)) {
        if (request->valid()) (*request)();
    }
}

constexpr std::string GetBlobFileExtension(BlobType type) {
    switch (type) {
    case BlobType::ShaderMeta: {
        return "meta";
    }
    case BlobType::ShaderBinary: {
        return "spv";
    }
    case BlobType::PipelineKey: {
        return "key";
    }
    case BlobType::ShaderProfile: {
        return "bin";
    }
    default:
        UNREACHABLE();
    }
}

void DataBase::Open() {
    if (opened) {
        return;
    }

    const auto& game_info = Common::ElfInfo::Instance();

    using namespace Common::FS;
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_zero_struct(&zip_ar);

        cache_path = GetUserPath(PathType::CacheDir) /
                     std::filesystem::path{game_info.GameSerial()}.replace_extension(".zip");

        if (!mz_zip_reader_init_file(&zip_ar, cache_path.string().c_str(),
                                     MZ_ZIP_FLAG_READ_ALLOW_WRITING) ||
            !mz_zip_validate_archive(&zip_ar, 0)) {
            LOG_INFO(Render, "Cache archive {} is not found or archive is corrupted",
                     cache_path.string().c_str());
            mz_zip_reader_end(&zip_ar);
            mz_zip_writer_init_file(&zip_ar, cache_path.string().c_str(), 0);
        }
    } else {
        cache_path = GetUserPath(PathType::CacheDir) / game_info.GameSerial();
        if (!std::filesystem::exists(cache_path)) {
            std::filesystem::create_directories(cache_path);
        }
        if (PrefetchEnabled()) {
            Prefetch(cache_path);
        }
    }

    ASSERT_MSG(write_queue.Begin(), "Previous pipeline cache writes were not drained");
    io_worker = std::jthread{ProcessIO};
    opened = true;
}

void DataBase::Close() {
    if (!IsOpened()) {
        return;
    }

    // Reject new work and complete every already accepted write before joining.
    opened = false;
    if (auto callback = std::exchange(before_close, {})) callback();
    write_queue.Seal();
    io_worker.request_stop();
    io_worker.join();

    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_finalize_archive(&zip_ar);
        mz_zip_writer_end(&zip_ar);
    }

    opened = false;
    LOG_INFO(Render, "Cache dumped");
}

void DataBase::Reset() {
    if (!IsOpened()) {
        return;
    }
    if (EmulatorSettings.IsPipelineCacheArchived() && zip_ar.m_zip_mode == MZ_ZIP_MODE_READING) {
        // Release the file handle of the old archive so it can be moved aside.
        mz_zip_reader_end(&zip_ar);
    }
    Close();

    // Keep a single copy of the previous cache next to the new one instead of deleting it.
    std::error_code ec;
    auto old_path = cache_path;
    old_path += ".incompatible";
    std::filesystem::remove_all(old_path, ec);
    std::filesystem::rename(cache_path, old_path, ec);
    if (ec) {
        LOG_WARNING(Render, "Could not move the old cache aside ({}), deleting it", ec.message());
        std::filesystem::remove_all(cache_path, ec);
    }
    LOG_INFO(Render, "Pipeline cache reset, the previous one was moved to {}", old_path.string());
    Open();
}

template <typename T>
bool WriteVector(const BlobType type, std::filesystem::path&& path_, std::vector<T>&& v) {
    {
        auto request = std::packaged_task<void()>{[type, path = std::move(path_),
                                                 data = std::move(v)]() mutable {
            path.replace_extension(GetBlobFileExtension(type));
            if (EmulatorSettings.IsPipelineCacheArchived()) {
                ASSERT_MSG(!ar_is_read_only,
                           "The archive is read-only. Did you forget to call `FinishPreload`?");
                if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), data.data(),
                                           data.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                    LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
                }
            } else {
                using namespace Common::FS;
                const auto file = IOFile{path, FileAccessMode::Create};
                file.Write(data);
            }
        }};
        return write_queue.Submit(std::move(request));
    }

}

template <typename T>
void LoadVector(BlobType type, std::filesystem::path& path, std::vector<T>& v) {
    using namespace Common::FS;
    path.replace_extension(GetBlobFileExtension(type));
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        int index{-1};
        index = mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0);
        if (index < 0) {
            LOG_WARNING(Render, "File {} is not found in the archive", path.string().c_str());
            return;
        }
        mz_zip_archive_file_stat stat{};
        mz_zip_reader_file_stat(&zip_ar, index, &stat);
        v.resize(stat.m_uncomp_size / sizeof(T));
        mz_zip_reader_extract_to_mem(&zip_ar, index, v.data(), stat.m_uncomp_size, 0);
    } else {
        if (std::vector<u8> bytes; TakePrefetched(path, bytes)) {
            v.resize(bytes.size() / sizeof(T));
            std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
            return;
        }
        const auto file = IOFile{path, FileAccessMode::Read};
        v.resize(file.GetSize() / sizeof(T));
        file.Read(v);
    }
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u8>&& data) {
    if (!opened) {
        return false;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u32>&& data) {
    if (!opened) {
        return false;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u8>& data) {
    if (!opened) {
        return;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    if (!opened) {
        return;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func) {
    const auto& ext = GetBlobFileExtension(type);
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            file_name.fill(0);
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            if (std::string{file_name.data()}.ends_with(ext)) {
                mz_zip_archive_file_stat stat{};
                mz_zip_reader_file_stat(&zip_ar, index, &stat);
                std::vector<u8> data(stat.m_uncomp_size);
                mz_zip_reader_extract_to_mem(&zip_ar, index, data.data(), data.size(), 0);
                func(std::move(data));
            }
        }
    } else {
        for (const auto& file_name : std::filesystem::directory_iterator{cache_path}) {
            if (file_name.path().extension().string().ends_with(ext)) {
                if (std::vector<u8> data; TakePrefetched(file_name.path(), data)) {
                    func(std::move(data));
                    continue;
                }
                using namespace Common::FS;
                const auto& file = IOFile{file_name, FileAccessMode::Read};
                if (file.IsOpen()) {
                    std::vector<u8> data(file.GetSize());
                    file.Read(data);
                    func(std::move(data));
                }
            }
        }
    }
}

void DataBase::FinishPreload() {
    {
        std::scoped_lock lk{prefetch_mutex};
        prefetched.clear();
        prefetched.rehash(0);
    }
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

} // namespace Storage
