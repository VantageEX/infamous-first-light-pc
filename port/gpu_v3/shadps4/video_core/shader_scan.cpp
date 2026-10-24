// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Shader index from the game's own files (2026-10-16, notes/study-shader-stutter-2026-10-16.md
// sec. 4 step 1). A port of the remake project's tools/shaders.py detector: every PS4 shader binary
// ends with a 32-byte "OrbShdr" BinaryInfo record, and its code starts at the nearest
// 0xBEEB03FF token (s_mov_b32 vcc_hi, literal) before the record whose literal puts the record at
// code + (literal + 1) * 8 - the rule SearchBinaryInfo (regs_shader.h) applies forward. On the
// Second Son dump the tool found 314,869 records = 10,374 distinct shaders, all copies of a hash
// byte-identical, and the emulator's own dumps byte-equal for every dumped hash (measured
// 2026-10-07). The index keeps file offsets only; the pre-compile reads the code at its offset.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

#include "common/logging/log.h"
#include "video_core/shader_scan.h"

#include <xxhash.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace VideoCore {

namespace {

constexpr std::array<u8, 7> Signature = {0x4f, 0x72, 0x62, 0x53, 0x68, 0x64, 0x72}; // OrbShdr
constexpr u32 CodeToken = 0xBEEB03FFu;
constexpr size_t RecordSize = 32;
constexpr size_t SearchBack = 0x10000 * 4; // as tools/shaders.py (the emulator scans 0x4000 dwords)
constexpr size_t ChunkSize = 32ull << 20;
constexpr u32 IndexMagic = 0x58494853; // "SHIX"
constexpr u32 IndexVersion = 3; // 2: ShaderLocation::crc32; 3: per-file sizes, no write times

struct Fingerprint {
    u64 files{};
    u64 bytes{};
    u64 newest{};
    bool operator==(const Fingerprint&) const = default;
};

struct FileEntry {
    std::filesystem::path path;
    u64 size{};
};

std::vector<FileEntry> ListFiles(const std::filesystem::path& folder, Fingerprint& fp) {
    std::vector<FileEntry> files;
    std::error_code ec;
    // follow_directory_symlink (2026-10-19): a game folder whose subfolders are junctions / links
    // (for example data kept on another drive) was indexed without them, so the seed found "no code
    // in the game index" and rebuilt nothing. The depth limit stops a link that loops.
    for (std::filesystem::recursive_directory_iterator it{
             folder,
             std::filesystem::directory_options::skip_permission_denied |
                 std::filesystem::directory_options::follow_directory_symlink,
             ec},
         end;
         it != end; it.increment(ec)) {
        if (ec) {
            break;
        }
        if (it.depth() > 16) {
            it.disable_recursion_pending();
        }
        if (!it->is_regular_file(ec)) {
            continue;
        }
        const u64 size = it->file_size(ec);
        if (ec || size < RecordSize * 2) {
            continue;
        }
        const auto time = it->last_write_time(ec).time_since_epoch().count();
        ++fp.files;
        fp.bytes += size;
        fp.newest = std::max<u64>(fp.newest, static_cast<u64>(time));
        files.push_back({it->path(), size});
    }
    std::ranges::sort(files, {}, &FileEntry::path);
    return files;
}

u32 Load32(const u8* p) {
    u32 v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// One file: every record whose code start is found. `out` keeps the first location per hash.
void ScanFile(const FileEntry& file, u32 file_idx, std::unordered_map<u64, ShaderLocation>& out,
              u64& records, u64& no_code) {
    std::ifstream in{file.path, std::ios::binary};
    if (!in) {
        return;
    }
    std::vector<u8> buf;
    buf.reserve(ChunkSize + SearchBack + 2 * RecordSize);
    u64 buf_base = 0;    // file offset of buf[0]
    u64 search_from = 0; // file offset where the next record search starts
    while (true) {
        const size_t old = buf.size();
        buf.resize(old + ChunkSize);
        in.read(reinterpret_cast<char*>(buf.data() + old), ChunkSize);
        const size_t got = static_cast<size_t>(in.gcount());
        buf.resize(old + got);
        const bool eof = got < ChunkSize;
        if (buf.size() >= RecordSize) {
            const size_t last = buf.size() - RecordSize; // the last index a whole record fits
            size_t pos = static_cast<size_t>(search_from - buf_base);
            while (pos <= last) {
                const void* hit = std::memchr(buf.data() + pos, Signature[0], last + 1 - pos);
                if (!hit) {
                    break;
                }
                pos = static_cast<size_t>(static_cast<const u8*>(hit) - buf.data());
                if (std::memcmp(buf.data() + pos, Signature.data(), Signature.size()) != 0) {
                    ++pos;
                    continue;
                }
                ++records;
                const u8* record = buf.data() + pos;
                const u32 bits = Load32(record + 8);
                const u32 type = (bits >> 2) & 0xF;
                const u32 length = bits >> 8;
                u64 hash;
                std::memcpy(&hash, record + 16, sizeof(hash));
                const u32 crc32 = Load32(record + 24);
                // The nearest token before the record whose literal points exactly at it.
                const size_t low = pos > SearchBack ? pos - SearchBack : 0;
                size_t code = SIZE_MAX;
                for (size_t q = pos >= 8 ? pos - 8 : 0; q + 1 > low && q != SIZE_MAX; --q) {
                    if (Load32(buf.data() + q) == CodeToken &&
                        (u64{Load32(buf.data() + q + 4)} + 1) * 8 == pos - q) {
                        code = q;
                        break;
                    }
                    if (q == 0) {
                        break;
                    }
                }
                if (code == SIZE_MAX || length == 0 || length > pos - code || (length & 3) != 0) {
                    ++no_code;
                } else if (!out.contains(hash)) {
                    out.emplace(hash,
                                ShaderLocation{file_idx, buf_base + code, length, type, crc32});
                }
                ++pos;
            }
            search_from = buf_base + last + 1;
        }
        if (eof) {
            break;
        }
        // Keep the tail the next chunk's records may search back into.
        const size_t keep = SearchBack + 2 * RecordSize;
        if (buf.size() > keep) {
            const size_t drop = buf.size() - keep;
            std::memmove(buf.data(), buf.data() + drop, keep);
            buf.resize(keep);
            buf_base += drop;
        }
    }
}

template <typename T>
void Put(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
bool Get(std::ifstream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return in.gcount() == static_cast<std::streamsize>(sizeof(T));
}

bool SaveIndex(const std::filesystem::path& path, const std::filesystem::path& folder,
               const Fingerprint& fp, const std::vector<FileEntry>& entries,
               const ShaderIndex& index) {
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    if (!out) {
        return false;
    }
    Put(out, IndexMagic);
    Put(out, IndexVersion);
    Put(out, fp);
    Put(out, static_cast<u32>(index.files.size()));
    for (size_t i = 0; i < index.files.size(); ++i) {
        // Lexical (review 10-19): relative() resolves junctions, so files behind a junction to
        // another drive all became "" and the index never matched again.
        const auto rel = index.files[i].lexically_relative(folder).generic_u8string();
        Put(out, static_cast<u32>(rel.size()));
        out.write(reinterpret_cast<const char*>(rel.data()), rel.size());
        Put(out, i < entries.size() ? entries[i].size : u64{0});
    }
    Put(out, static_cast<u32>(index.shaders.size()));
    for (const auto& [hash, loc] : index.shaders) {
        Put(out, hash);
        Put(out, loc);
    }
    return static_cast<bool>(out);
}

/// What a saved index gave for this folder's listing (v1.0.2).
struct LoadedIndex {
    bool exact{};            ///< every file the same: nothing to read
    std::vector<bool> known; ///< per listed file: same relative path and size as saved
    u64 reused{};            ///< shaders kept (their file is unchanged)
    u64 lost{};              ///< shaders whose saved file is gone or changed
};

/// Reads the saved index. Shaders whose file has the same relative path and size here are kept in
/// `index.shaders` (file numbers mapped to this listing); `out.known` marks those files.
/// v1.0.1 needed the whole listing equal, so a player's dump with a few extra or different files
/// (a player's EU copy: 1,874 files against the seed's 1,866) read all 29.7 GB again on the first
/// start - 13.7 s on an NVMe drive, minutes on a hard drive, with a frozen screen at 15%.
bool LoadIndex(const std::filesystem::path& path, const std::filesystem::path& folder,
               const std::vector<FileEntry>& entries, ShaderIndex& index, LoadedIndex& out) {
    std::ifstream in{path, std::ios::binary};
    u32 magic{}, version{}, num_files{}, num_shaders{};
    Fingerprint saved{};
    if (!Get(in, magic) || !Get(in, version) || magic != IndexMagic || version != IndexVersion ||
        !Get(in, saved) || !Get(in, num_files) || num_files > (1u << 24)) {
        return false;
    }
    std::unordered_map<std::u8string, u32> listed;
    listed.reserve(entries.size());
    for (u32 i = 0; i < entries.size(); ++i) {
        listed.emplace(entries[i].path.lexically_relative(folder).generic_u8string(), i);
    }
    out.known.assign(entries.size(), false);
    std::vector<u32> saved_to_listed(num_files, UINT32_MAX);
    u32 matched = 0;
    for (u32 j = 0; j < num_files; ++j) {
        u32 len{};
        if (!Get(in, len) || len > 4096) {
            return false;
        }
        std::u8string rel(len, u8'\0');
        in.read(reinterpret_cast<char*>(rel.data()), len);
        u64 size{};
        if (in.gcount() != static_cast<std::streamsize>(len) || !Get(in, size)) {
            return false;
        }
        const auto it = listed.find(rel);
        if (it != listed.end() && entries[it->second].size == size && !out.known[it->second]) {
            saved_to_listed[j] = it->second;
            out.known[it->second] = true;
            ++matched;
        }
    }
    if (!Get(in, num_shaders) || num_shaders > (1u << 24)) {
        return false;
    }
    for (u32 i = 0; i < num_shaders; ++i) {
        u64 hash{};
        ShaderLocation loc{};
        if (!Get(in, hash) || !Get(in, loc) || loc.file >= num_files) {
            return false;
        }
        if (saved_to_listed[loc.file] == UINT32_MAX) {
            ++out.lost;
            continue;
        }
        loc.file = saved_to_listed[loc.file];
        index.shaders.emplace(hash, loc);
        ++out.reused;
    }
    out.exact = matched == num_files && matched == entries.size();
    return true;
}

} // Anonymous namespace

ShaderIndex BuildShaderIndex(const std::filesystem::path& game_folder,
                             const std::filesystem::path& index_file,
                             const std::function<void(u64, u64)>& progress) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    auto ms = [&] {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    Fingerprint fp{};
    const auto files = ListFiles(game_folder, fp);
    ShaderIndex index;
    for (const auto& file : files) {
        index.files.push_back(file.path);
    }
    // The files to read: all, or (v1.0.2) only the new / changed ones when the saved index keeps
    // every shader of the rest.
    std::vector<u32> to_scan;
    LoadedIndex loaded;
    const bool have_saved =
        !index_file.empty() && LoadIndex(index_file, game_folder, files, index, loaded);
    if (have_saved && loaded.exact) {
        LOG_INFO(Render_Vulkan, "ShaderScan: index {} read: {} shaders in {} files ({:.0f} ms)",
                 index_file.string(), index.shaders.size(), index.files.size(), ms());
        return index;
    }
    if (have_saved && loaded.lost == 0) {
        for (u32 i = 0; i < files.size(); ++i) {
            if (!loaded.known[i]) {
                to_scan.push_back(i);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "ShaderScan: index {} kept {} shaders of the unchanged files; reading the {} new "
                 "or changed files of {}",
                 index_file.string(), loaded.reused, to_scan.size(), files.size());
    } else {
        if (have_saved) {
            LOG_INFO(Render_Vulkan,
                     "ShaderScan: index {} does not fit this game folder ({} shaders sat in files "
                     "that are gone or changed); reading every file",
                     index_file.string(), loaded.lost);
        }
        index.shaders.clear();
        to_scan.resize(files.size());
        for (u32 i = 0; i < files.size(); ++i) {
            to_scan[i] = i;
        }
    }
    u64 scan_bytes = 0;
    for (const u32 i : to_scan) {
        scan_bytes += files[i].size;
    }
    const u32 hw = std::max(2u, std::thread::hardware_concurrency());
    const u32 num_threads = std::clamp(hw / 2, 2u, 8u);
    std::atomic<size_t> next{0};
    std::atomic<size_t> files_done{0};
    std::atomic<u64> bytes_done{0};
    std::mutex merge_mutex;
    std::vector<std::unordered_map<u64, ShaderLocation>> found(to_scan.size() > 0 ? num_threads
                                                                                  : 0);
    std::atomic<u64> records{0}, no_code{0};
    std::vector<std::thread> threads;
    for (u32 t = 0; t < num_threads && !to_scan.empty(); ++t) {
        threads.emplace_back([&, t] {
            std::unordered_map<u64, ShaderLocation> mine;
            u64 my_records = 0, my_no_code = 0;
            for (size_t k = next++; k < to_scan.size(); k = next++) {
                const u32 i = to_scan[k];
                // A per-file map keeps "first file wins" independent of the thread order.
                std::unordered_map<u64, ShaderLocation> local;
                ScanFile(files[i], i, local, my_records, my_no_code);
                for (auto& [hash, loc] : local) {
                    const auto [it, inserted] = mine.try_emplace(hash, loc);
                    if (!inserted && loc.file < it->second.file) {
                        it->second = loc;
                    }
                }
                bytes_done += files[i].size;
                ++files_done;
            }
            records += my_records;
            no_code += my_no_code;
            std::scoped_lock lk{merge_mutex};
            found[t] = std::move(mine);
        });
    }
    // v1.0.2: the caller's thread (the boot's main thread) waits in short steps and reports
    // progress, so the boot screen moves and Windows never marks the window "Not responding".
    while (files_done.load() < to_scan.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        if (progress) {
            progress(bytes_done.load(), scan_bytes);
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (auto& part : found) {
        for (auto& [hash, loc] : part) {
            const auto [it, inserted] = index.shaders.try_emplace(hash, loc);
            if (!inserted && loc.file < it->second.file) {
                it->second = loc;
            }
        }
    }
    u32 by_type[16] = {};
    for (const auto& [hash, loc] : index.shaders) {
        ++by_type[loc.type & 0xF];
    }
    const bool saved =
        !index_file.empty() && SaveIndex(index_file, game_folder, fp, files, index);
    LOG_INFO(Render_Vulkan,
             "ShaderScan: {} of {} files read ({} of {} MiB), {} records ({} without a code start) "
             "-> {} shaders (cs {}, ps {}, vs {}, other {}) on {} threads in {:.0f} ms; index {}",
             to_scan.size(), files.size(), scan_bytes >> 20, fp.bytes >> 20, records.load(),
             no_code.load(), index.shaders.size(),
             by_type[4], by_type[0], by_type[1],
             index.shaders.size() - by_type[4] - by_type[0] - by_type[1], num_threads, ms(),
             saved ? index_file.string() : std::string{"not saved"});
    return index;
}

size_t CompactProcessHeap() {
#ifdef _WIN32
    return HeapCompact(GetProcessHeap(), 0);
#else
    return 0;
#endif
}

u64 ExecutableHash() {
    std::filesystem::path exe;
#ifdef _WIN32
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) {
        return 0;
    }
    buf.resize(n);
    exe = buf;
#else
    std::error_code ec;
    exe = std::filesystem::read_symlink("/proc/self/exe", ec);
#endif
    std::ifstream in{exe, std::ios::binary | std::ios::ate};
    if (!in) {
        return 0;
    }
    std::vector<char> bytes(static_cast<size_t>(in.tellg()));
    in.seekg(0);
    in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        return 0;
    }
    return XXH3_64bits(bytes.data(), bytes.size());
}

std::vector<u32> ReadShaderCode(const ShaderIndex& index, const ShaderLocation& location,
                                u64 expected_hash) {
    std::vector<u32> code;
    if (location.file >= index.files.size() || location.code_length == 0 ||
        (location.code_length & 3) != 0) {
        return code;
    }
    std::ifstream in{index.files[location.file], std::ios::binary};
    in.seekg(static_cast<std::streamoff>(location.offset));
    code.resize(location.code_length / sizeof(u32));
    in.read(reinterpret_cast<char*>(code.data()), location.code_length);
    if (in.gcount() != static_cast<std::streamsize>(location.code_length)) {
        code.clear();
        return code;
    }
    if (expected_hash != 0) {
        // The record the scan found sits (code[1] + 1) * 8 bytes after the code start.
        std::array<u8, RecordSize> record{};
        if (code.size() < 2 || code[0] != CodeToken) {
            code.clear();
            return code;
        }
        in.seekg(static_cast<std::streamoff>(location.offset + (u64{code[1]} + 1) * 8));
        in.read(reinterpret_cast<char*>(record.data()), record.size());
        u64 hash{};
        std::memcpy(&hash, record.data() + 16, sizeof(hash));
        const u32 bits = Load32(record.data() + 8);
        if (in.gcount() != static_cast<std::streamsize>(record.size()) ||
            std::memcmp(record.data(), Signature.data(), Signature.size()) != 0 ||
            hash != expected_hash || Load32(record.data() + 24) != location.crc32 ||
            (bits >> 8) != location.code_length) {
            code.clear();
        }
    }
    return code;
}

} // namespace VideoCore
