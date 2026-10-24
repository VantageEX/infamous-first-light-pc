// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"

#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"

#include <miniz.h>

#include <atomic>
#include <string>
#include <unordered_map>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <system_error>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

namespace {

std::mutex submit_mutex{};
u32 num_requests{};
std::condition_variable_any request_cv{};
std::queue<std::packaged_task<void()>> req_queue{};
std::mutex m_request{};

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

// SHADPS4_CACHE_PREFETCH=2 (2026-10-16): file name -> bytes of every .key / .meta read by the
// prefetch, until FinishPreload. Opening a cache file costs ~3-8 ms while the antivirus has no
// verdict cached for it (measured 10-16 on files made that day); WarmUp then opens none of them.
std::unordered_map<std::string, std::vector<u8>> g_prefetched{};

std::string BlobFileName(const std::string& name, std::string_view ext) {
    std::string file_name;
    file_name.reserve(name.size() + 1 + ext.size());
    file_name.append(name).append(1, '.').append(ext);
    return file_name;
}

} // namespace

namespace Storage {

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadPS4:PipelineCacheIO");

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(request_cv, lk, stoken, [&] { return num_requests; });
        }

        if (stoken.stop_requested()) {
            break;
        }

        while (num_requests) {
            std::packaged_task<void()> request{};
            {
                std::scoped_lock lock{m_request};
                if (req_queue.empty()) {
                    continue;
                }
                request = std::move(req_queue.front());
                req_queue.pop();
            }

            if (request.valid()) {
                request();
                request.get_future().wait();
            }

            --num_requests;
        }
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

void DataBase::Open(bool native_bindings) {
    if (opened) {
        return;
    }

    const auto& game_info = Common::ElfInfo::Instance();
    // Keep experimental binding-model caches separate from the playable legacy cache.
    // Shader metadata and permutation-indexed binaries must come from the same namespace.
    // Builds on either side of the ShaderBinaryVersion 3 -> 4 bump share this folder: version 4
    // accepts every version-3 entry except GDS shaders (LoadShaderMeta), so only those few
    // recompile when the user alternates builds. (A per-version folder seeded by a copy was
    // tried on 2026-10-07: the copy of ~38,000 files took 120 s at first launch.)
    auto cache_name = std::string{game_info.GameSerial()} +
                      (native_bindings ? "_native_bindings_v1" : "");
    // SHADPS4_CACHE_NAME (2026-10-07): another folder / archive name, e.g. a copy for test runs
    // that write entries the user's own cache should not get.
    if (const char* name = std::getenv("SHADPS4_CACHE_NAME"); name && *name) {
        cache_name = name;
        LOG_INFO(Render, "Pipeline cache name from SHADPS4_CACHE_NAME: {}", cache_name);
    }

    using namespace Common::FS;
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_zero_struct(&zip_ar);

        cache_path = GetUserPath(PathType::CacheDir) /
                     std::filesystem::path{cache_name}.replace_extension(".zip");

        if (!mz_zip_reader_init_file(&zip_ar, cache_path.string().c_str(),
                                     MZ_ZIP_FLAG_READ_ALLOW_WRITING) ||
            !mz_zip_validate_archive(&zip_ar, 0)) {
            LOG_INFO(Render, "Cache archive {} is not found or archive is corrupted",
                     cache_path.string().c_str());
            mz_zip_reader_end(&zip_ar);
            mz_zip_writer_init_file(&zip_ar, cache_path.string().c_str(), 0);
        }
    } else {
        cache_path = GetUserPath(PathType::CacheDir) / cache_name;
        if (!std::filesystem::exists(cache_path)) {
            std::filesystem::create_directories(cache_path);
        }
    }

    io_worker = std::jthread{ProcessIO};
    opened = true;
}

void DataBase::Close() {
    if (!IsOpened()) {
        return;
    }

    io_worker.request_stop();
    io_worker.join();

    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_finalize_archive(&zip_ar);
        mz_zip_writer_end(&zip_ar);
    }

    opened = false;
    LOG_INFO(Render, "Cache dumped");
}

template <typename T>
bool WriteVector(const BlobType type, std::filesystem::path&& path_, std::vector<T>&& v) {
    {
        auto request = std::packaged_task<void()>{[=]() {
            auto path{path_};
            path.replace_extension(GetBlobFileExtension(type));
            if (EmulatorSettings.IsPipelineCacheArchived()) {
                ASSERT_MSG(!ar_is_read_only,
                           "The archive is read-only. Did you forget to call `FinishPreload`?");
                if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), v.data(),
                                           v.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                    LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
                }
            } else {
                using namespace Common::FS;
                const auto file = IOFile{path, FileAccessMode::Create};
                file.Write(v);
            }
        }};
        std::scoped_lock lock{m_request};
        req_queue.emplace(std::move(request));
    }

    std::scoped_lock lk{submit_mutex};
    ++num_requests;
    request_cv.notify_one();
    return true;
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
    if (!g_prefetched.empty()) {
        if (const auto it = g_prefetched.find(BlobFileName(name, GetBlobFileExtension(type)));
            it != g_prefetched.end()) {
            data = it->second;
            return;
        }
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    if (!opened) {
        return;
    }
    if (!g_prefetched.empty()) {
        if (const auto it = g_prefetched.find(BlobFileName(name, GetBlobFileExtension(type)));
            it != g_prefetched.end()) {
            data.resize(it->second.size() / sizeof(u32));
            std::memcpy(data.data(), it->second.data(), data.size() * sizeof(u32));
            return;
        }
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
                if (!g_prefetched.empty()) {
                    if (const auto it = g_prefetched.find(file_name.path().filename().string());
                        it != g_prefetched.end()) {
                        func(std::vector<u8>(it->second));
                        continue;
                    }
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

void DataBase::Prefetch(const std::function<void(u64 done, u64 total)>& progress) {
    // The first open of a cache file is slow (~3.7 ms each on the user's PC, 2026-10-07: 38,368
    // files took 141 s on one thread and 13.2 s on 16; a second pass 3.0 s). WarmUp opens every blob
    // one by one, so a build's first launch, or any launch after the files' scan results expired,
    // showed a black screen for minutes. Reading them once on many threads first makes that serial
    // loop hit warm files.
    const char* env = std::getenv("SHADPS4_CACHE_PREFETCH");
    if (!opened || !env || (env[0] != '1' && env[0] != '2' && env[0] != '3') ||
        EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    const bool keep = env[0] == '2' || env[0] == '3';
    const bool keep_spv = env[0] == '3';
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(cache_path, ec)) {
        if (entry.is_regular_file(ec)) {
            files.push_back(entry.path());
        }
    }
    constexpr u32 NumThreads = 16;
    std::atomic<size_t> next{0};
    std::atomic<u64> bytes{0};
    // =2: one slot per file, written only by the thread that took that index.
    std::vector<std::vector<u8>> kept(keep ? files.size() : 0);
    std::vector<char> keep_file(files.size(), 0);
    if (keep) {
        for (size_t i = 0; i < files.size(); ++i) {
            const auto ext = files[i].extension();
            keep_file[i] = ext == ".key" || ext == ".meta" || (keep_spv && ext == ".spv");
        }
    }
    {
        std::vector<std::jthread> threads;
        for (u32 t = 0; t < NumThreads; ++t) {
            threads.emplace_back([&] {
                std::vector<char> buffer(64 * 1024);
                for (size_t i = next.fetch_add(1); i < files.size(); i = next.fetch_add(1)) {
                    if (keep_file[i]) {
                        std::ifstream file(files[i], std::ios::binary | std::ios::ate);
                        if (!file) {
                            continue;
                        }
                        auto& out = kept[i];
                        out.resize(static_cast<size_t>(file.tellg()));
                        file.seekg(0);
                        file.read(reinterpret_cast<char*>(out.data()),
                                  static_cast<std::streamsize>(out.size()));
                        if (!file) {
                            out.clear();
                            keep_file[i] = 0; // left to an ordinary load
                            continue;
                        }
                        bytes.fetch_add(out.size(), std::memory_order_relaxed);
                        continue;
                    }
                    std::ifstream file(files[i], std::ios::binary);
                    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
                        bytes.fetch_add(static_cast<u64>(file.gcount()), std::memory_order_relaxed);
                    }
                }
            });
        }
        // Boot screen (2026-10-17): the caller's thread reports progress while the readers work.
        if (progress) {
            for (size_t taken = next.load(); taken < files.size(); taken = next.load()) {
                progress(taken, files.size());
                std::this_thread::sleep_for(std::chrono::milliseconds{50});
            }
        }
    } // joins
    if (keep) {
        g_prefetched.reserve(files.size());
        for (size_t i = 0; i < files.size(); ++i) {
            if (keep_file[i]) {
                g_prefetched.emplace(files[i].filename().string(), std::move(kept[i]));
            }
        }
        LOG_INFO(Render, "Pipeline cache prefetch: {} files kept in memory ({})", g_prefetched.size(),
                 keep_spv ? "keys, metas, SPIR-V" : "keys, metas");
    }
    LOG_INFO(Render, "Pipeline cache prefetch: {} files, {} MB read on {} threads in {} ms",
             files.size(), bytes.load() >> 20, NumThreads,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t_start)
                 .count());
}

size_t DataBase::CountBlobs(BlobType type) const {
    if (!opened || EmulatorSettings.IsPipelineCacheArchived()) {
        return 0;
    }
    const auto ext = GetBlobFileExtension(type);
    size_t count = 0;
    if (!g_prefetched.empty() && (ext == "key" || ext == "meta")) {
        for (const auto& [name, bytes] : g_prefetched) {
            count += name.ends_with(ext) ? 1 : 0;
        }
        return count;
    }
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(cache_path, ec)) {
        count += entry.path().extension().string().ends_with(ext) ? 1 : 0;
    }
    return count;
}

std::filesystem::path DataBase::SidecarPath(std::string_view file_name) const {
    if (!opened || EmulatorSettings.IsPipelineCacheArchived()) {
        return {};
    }
    return cache_path / file_name;
}

const std::vector<u8>* DataBase::Prefetched(const std::string& file_name) const {
    const auto it = g_prefetched.find(file_name);
    return it != g_prefetched.end() ? &it->second : nullptr;
}

void DataBase::FinishPreload() {
    std::unordered_map<std::string, std::vector<u8>>{}.swap(g_prefetched);
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

} // namespace Storage
