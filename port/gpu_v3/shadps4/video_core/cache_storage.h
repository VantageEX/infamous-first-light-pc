// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <string_view>
#include <thread>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open(bool native_bindings = false);
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();
    // SHADPS4_CACHE_PREFETCH=1: read every cache file once on several threads (folder mode).
    // =2 also keeps every .key and .meta file's bytes until FinishPreload, so WarmUp's loads open
    // no file (2026-10-16: s1322 spent 104.5 s opening the seed's metas one by one); =3 keeps the
    // .spv files too (s1330: 95 s opening 22 k SPIR-V files; ~0.8 GB held until FinishPreload).
    // progress (2026-10-17, boot screen): called on the calling thread about every 50 ms with the
    // files taken so far and the total, while the reader threads work.
    void Prefetch(const std::function<void(u64 done, u64 total)>& progress = {});
    // Number of blobs of one type in the folder (the prefetched names when kept, else a scan).
    [[nodiscard]] size_t CountBlobs(BlobType type) const;
    // =2: the kept bytes of a cache file ("<name>.<ext>"), or null. Read-only between Prefetch
    // and FinishPreload, so boot threads may call it concurrently.
    [[nodiscard]] const std::vector<u8>* Prefetched(const std::string& file_name) const;
    // SHADPS4_PRECOMPILE's sidecar files (plain text, written synchronously so a crash keeps
    // them) live in the cache folder; empty in archive mode or before Open.
    [[nodiscard]] std::filesystem::path SidecarPath(std::string_view file_name) const;

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

private:
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
