// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"

#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <miniz.h>

#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

namespace {

std::mutex submit_mutex{};
// tlg: atomic: the build threads count requests in under submit_mutex while the IO thread counts
// them out without it. A lost update left the count above zero with an empty queue: the IO
// thread spun on it and WaitIdle never returned (first-start precompile hang).
std::atomic<u32> num_requests{};
std::condition_variable_any request_cv{};
std::queue<std::packaged_task<void()>> req_queue{};
std::mutex m_request{};

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

#ifdef TLG_SHADER_CACHE_KEY
// tlg: the GPU user folder used to be chosen at load time, before the probe set
// TLG_GPU_USER_DIR: ~/.local/share/shadPS4 on Linux (shadPS4's own folder) and <saves>/user on
// Windows. On the first start with the port's folder, this game's cache for the current key is
// moved here once, so nothing is recompiled. Only that <serial>-<key> folder is taken; nothing
// else in the shadPS4 folder is touched (an installed shadPS4 emulator may own it).
void MigrateLegacyCache(const std::filesystem::path& cache_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(cache_path, ec)) {
        return;
    }
    const auto name = cache_path.filename();
    std::vector<fs::path> sources{Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "user" /
                                  "cache" / name};
#ifndef _WIN32
    if (const char* data = std::getenv("XDG_DATA_HOME"); data && data[0]) {
        sources.push_back(fs::path{data} / "shadPS4" / "cache" / name);
    } else if (const char* home = std::getenv("HOME"); home && home[0]) {
        sources.push_back(fs::path{home} / ".local" / "share" / "shadPS4" / "cache" / name);
    }
#endif
    for (const auto& source : sources) {
        if (!fs::is_directory(source, ec)) {
            continue;
        }
        fs::create_directories(cache_path.parent_path(), ec);
        fs::rename(source, cache_path, ec);
        if (ec) {
            // Another file system: copy, then delete the old folder only if the copy is whole.
            ec.clear();
            fs::copy(source, cache_path, fs::copy_options::recursive, ec);
            if (ec) {
                LOG_WARNING(Render, "Shader cache: could not move {} here ({}); rebuilding it",
                            source.string(), ec.message());
                fs::remove_all(cache_path, ec);
                return;
            }
            fs::remove_all(source, ec);
        }
        LOG_INFO(Render, "Shader cache moved from {} to {}", source.string(), cache_path.string());
        return;
    }
}

// tlg: shaders translated by an older recompiler are never reused: this game's folders of other
// keys (and the unkeyed <serial>) are removed, in the port's own cache folder only.
void PruneOtherRevisions(const std::filesystem::path& cache_path) {
    const std::string serial{Common::ElfInfo::Instance().GameSerial()};
    const auto name = cache_path.filename().string();
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(cache_path.parent_path(), ec)) {
        const auto other = entry.path().filename().string();
        if (entry.is_directory(ec) && other != name &&
            (other == serial || other.rfind(serial + "-", 0) == 0)) {
            LOG_INFO(Render, "Removing shader cache of another recompiler revision: {}", other);
            std::filesystem::remove_all(entry.path(), ec);
        }
    }
}
#endif

} // namespace

namespace Storage {

std::filesystem::path GameCacheDir() {
    const std::string serial{Common::ElfInfo::Instance().GameSerial()};
#ifdef TLG_SHADER_CACHE_KEY
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
           (serial + "-" + TLG_SHADER_CACHE_KEY);
#else
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / serial;
#endif
}

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadPS4:PipelineCacheIO");

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(request_cv, lk, stoken, [&] { return num_requests.load() != 0; });
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
        cache_path = GameCacheDir();
#ifdef TLG_SHADER_CACHE_KEY
        MigrateLegacyCache(cache_path);
        PruneOtherRevisions(cache_path);
#endif
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
        // tlg: the blob and its path move into the job: capturing them by value copied the whole
        // blob on the caller, once per pipeline the game builds (tens per second while it plays).
        auto request = std::packaged_task<void()>{
            [type, v = std::move(v), path_ = std::move(path_)]() mutable {
                auto path{std::move(path_)};
                path.replace_extension(GetBlobFileExtension(type));
                if (EmulatorSettings.IsPipelineCacheArchived()) {
                    ASSERT_MSG(!ar_is_read_only,
                               "The archive is read-only. Did you forget to call `FinishPreload`?");
                    if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), v.data(),
                                               v.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                        LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
                    }
                } else {
                    // tlg: write beside and rename into place, so a game that ends during the
                    // write (the queue is not drained at exit) never leaves a truncated blob that
                    // the next start would load.
                    using namespace Common::FS;
                    auto temp{path};
                    temp += ".tmp";
                    {
                        const auto file = IOFile{temp, FileAccessMode::Create};
                        if (!file.IsOpen() || file.Write(v) != v.size()) {
                            return;
                        }
                    }
                    std::error_code error;
                    std::filesystem::rename(temp, path, error);
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

size_t DataBase::CountBlobs(BlobType type) {
    const auto& ext = GetBlobFileExtension(type);
    size_t count = 0;
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            count += std::string{file_name.data()}.ends_with(ext);
        }
        return count;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{cache_path, error}) {
        count += entry.path().extension().string().ends_with(ext);
    }
    return count;
}

bool DataBase::ConcurrentReads() const {
    return !EmulatorSettings.IsPipelineCacheArchived();
}

void DataBase::ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func,
                           const std::function<bool()>& stop) {
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
            if (stop && stop()) {
                return;
            }
            if (file_name.path().extension().string().ends_with(ext)) {
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

void DataBase::ForEachBlobName(BlobType type,
                               const std::function<void(std::string_view name)>& func) {
    const auto ext = "." + GetBlobFileExtension(type);
    const auto visit = [&](std::string_view name) {
        if (name.ends_with(ext)) {
            func(name.substr(0, name.size() - ext.size()));
        }
    };
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            visit(file_name.data());
        }
        return;
    }
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator{cache_path, error}) {
        visit(entry.path().filename().string());
    }
}

void DataBase::WaitIdle() {
    while (opened && num_requests.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void DataBase::Clear() {
    if (!opened || EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    std::error_code ec;
    u64 removed = 0;
    for (const auto& entry : std::filesystem::directory_iterator(cache_path, ec)) {
        if (entry.is_regular_file(ec) && std::filesystem::remove(entry.path(), ec)) {
            ++removed;
        }
    }
    LOG_WARNING(Render, "Pipeline cache cleared ({} files): it is rebuilt for this build", removed);
}

void DataBase::FinishPreload() {
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

} // namespace Storage
