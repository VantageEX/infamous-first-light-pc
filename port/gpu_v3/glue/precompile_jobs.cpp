// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3: this port's first-start shader compilation (the job file, <user>/cache/<serial>-
// jobs.bin, made by run.py from the game's own .xpps packages with tools/xpps), as renderer v2
// (gpu_v2/gpu/precompile_jobs.inl) and this port's renderer (vk_precompile.cpp) run it. Each
// record is a draw (kind 0: its registers, shaders, flattened user data, fetch shader and vertex
// buffer descriptors) or a dispatch (kind 1: the compute registers and its shader) as the
// pipeline cache saw it. Before the game starts, every record is looked up exactly as if the game
// drew or dispatched it: the fork's pipeline cache translates its shaders, builds the pipeline
// and saves both to its on-disk cache, so the game finds them there instead of compiling while
// it plays. A record the translator refuses is skipped (V3::g_assert_throws). Done once per job
// file: a marker in the cache folder skips the pass on later starts.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/path_util.h"
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "tlg_overlay.h"
#include "v3_glue.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace V3 {
namespace {

constexpr char Magic[8] = {'I', 'S', 'S', 'P', 'J', 'O', 'B', '1'};

struct VertexSharp {
    u32 sgpr_base, dword_offset;
    std::array<u32, 4> sharp;
};
struct Stage {
    u32 sw_stage, hw_stage;
    u64 hash;
    std::vector<u8> code;
    std::array<u32, 16> user_data;
    std::vector<u32> flat;
    std::vector<u8> fetch_code;
    u32 fetch_sgpr = 0;
    std::vector<VertexSharp> vertex_sharps;
};
struct Record {
    u32 kind;
    std::vector<std::pair<u32, u32>> regs;
    std::vector<Stage> stages;
};

class Reader {
public:
    explicit Reader(std::vector<u8> bytes_) : bytes{std::move(bytes_)} {}
    template <typename T>
    bool Get(T& value) {
        if (at + sizeof(T) > bytes.size()) return false;
        std::memcpy(&value, bytes.data() + at, sizeof(T));
        at += sizeof(T);
        return true;
    }
    bool Bytes(std::vector<u8>& out, u32 size) {
        if (at + size > bytes.size()) return false;
        out.assign(bytes.begin() + at, bytes.begin() + at + size);
        at += size;
        return true;
    }
    bool Done() const {
        return at == bytes.size();
    }

private:
    std::vector<u8> bytes;
    size_t at = 0;
};

bool Read(const std::filesystem::path& path, std::vector<Record>& records) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    Reader r{std::vector<u8>{std::istreambuf_iterator<char>(in), {}}};
    char magic[8];
    u32 version;
    if (!r.Get(magic) || std::memcmp(magic, Magic, 8) != 0 || !r.Get(version) || version != 1) {
        return false;
    }
    while (!r.Done()) {
        Record rec{};
        u32 n;
        if (!r.Get(rec.kind) || !r.Get(n)) return false;
        rec.regs.resize(n);
        for (auto& [reg, value] : rec.regs) {
            if (!r.Get(reg) || !r.Get(value)) return false;
        }
        u32 stages;
        if (!r.Get(stages)) return false;
        rec.stages.resize(stages);
        for (auto& st : rec.stages) {
            u32 size;
            if (!r.Get(st.sw_stage) || !r.Get(st.hw_stage) || !r.Get(st.hash) || !r.Get(size) ||
                !r.Bytes(st.code, size) || !r.Get(st.user_data) || !r.Get(n))
                return false;
            st.flat.resize(n);
            for (auto& w : st.flat) {
                if (!r.Get(w)) return false;
            }
            if (!r.Get(size) || !r.Bytes(st.fetch_code, size) || !r.Get(st.fetch_sgpr) || !r.Get(n))
                return false;
            st.vertex_sharps.resize(n);
            for (auto& vs : st.vertex_sharps) {
                if (!r.Get(vs.sgpr_base) || !r.Get(vs.dword_offset) || !r.Get(vs.sharp)) return false;
            }
        }
        records.push_back(std::move(rec));
    }
    return true;
}

/// Host memory standing in for the guest memory the registers point at (shader code, fetch
/// shaders, vertex buffer descriptor tables): page-aligned, below 2^48 (40-bit, 256-byte units).
class Arena {
public:
    void* Place(const void* data, size_t size) {
        constexpr size_t Page = 4096;
        // Whole pages and a spare one: the translator reads code up to the end of its page.
        const size_t bytes = (size + Page - 1) / Page * Page + Page;
        auto& block = blocks.emplace_back(std::make_unique<u8[]>(bytes + Page));
        auto* aligned = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(block.get()) + Page - 1) &
                                              ~uintptr_t{Page - 1});
        // The slack holds s_endpgm, not zeros: a decoder that runs past a block stops there.
        auto* words = reinterpret_cast<u32*>(aligned);
        std::fill(words, words + bytes / sizeof(u32), data ? 0xBF810000u : 0u);
        if (data) std::memcpy(aligned, data, size);
        return aligned;
    }

private:
    std::vector<std::unique_ptr<u8[]>> blocks;
};

void SetPointer(AmdGpu::UserData& user_data, u32 sgpr, const void* pointer) {
    const auto value = reinterpret_cast<uintptr_t>(pointer);
    user_data[sgpr] = u32(value);
    user_data[sgpr + 1] = u32(value >> 32);
}

/// One copy of each shader's code for the whole run: the program cache keeps a pointer to the code
/// of the record that first used a shader and translates its later permutations from it.
struct CodeStore {
    Arena arena;
    std::unordered_map<u64, void*> by_hash;
    void* Get(const Stage& stage) {
        const u64 key = stage.hash ^ (u64(stage.code.size()) << 40);
        auto [it, is_new] = by_hash.try_emplace(key, nullptr);
        if (is_new) it->second = arena.Place(stage.code.data(), stage.code.size());
        return it->second;
    }
};

void PlaceStage(CodeStore& code, Arena& arena, const Stage& stage, u64& address,
                AmdGpu::UserData& user_data) {
    address = reinterpret_cast<uintptr_t>(code.Get(stage)) >> 8;
    std::copy(stage.user_data.begin(), stage.user_data.end(), user_data.begin());
    if (!stage.fetch_code.empty() && stage.fetch_sgpr + 1 < user_data.size()) {
        SetPointer(user_data, stage.fetch_sgpr,
                   arena.Place(stage.fetch_code.data(), stage.fetch_code.size()));
    }
    std::unordered_map<u32, u32> table_dwords;
    for (const auto& vs : stage.vertex_sharps) {
        auto& dwords = table_dwords[vs.sgpr_base];
        dwords = std::max(dwords, vs.dword_offset + 4);
    }
    for (const auto& [sgpr, dwords] : table_dwords) {
        auto* table = static_cast<u32*>(arena.Place(nullptr, dwords * sizeof(u32)));
        for (const auto& vs : stage.vertex_sharps) {
            if (vs.sgpr_base == sgpr) std::memcpy(table + vs.dword_offset, vs.sharp.data(), 16);
        }
        if (sgpr + 1 < user_data.size()) SetPointer(user_data, sgpr, table);
    }
}

/// The marker of a finished pass over this job file (its size and time).
std::filesystem::path DoneMarker(const std::filesystem::path& jobs) {
    std::error_code error;
    const auto size = std::filesystem::file_size(jobs, error);
    const auto time = std::filesystem::last_write_time(jobs, error).time_since_epoch().count();
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
           fmt::format("jobs-done-{:x}-{:x}.txt", size, static_cast<u64>(time));
}

} // namespace

void PrecompileJobs() {
    const char* user = std::getenv("TLG_GPU_USER_DIR");
    const auto jobs = std::filesystem::u8path(user && *user ? user : g_host.user_dir) / "cache" /
                      (g_host.serial + "-jobs.bin");
    std::error_code error;
    if (std::getenv("TLG_PRECOMPILE_JOBS") && std::string_view{std::getenv("TLG_PRECOMPILE_JOBS")} == "0") {
        return;
    }
    if (!std::filesystem::is_regular_file(jobs, error)) {
        std::fprintf(stderr, "V3 precompile: no job file (%s)\n", jobs.string().c_str());
        return;
    }
    const auto marker = DoneMarker(jobs);
    if (std::filesystem::exists(marker, error)) {
        std::fprintf(stderr, "V3 precompile: done on an earlier start (%s)\n", marker.filename().string().c_str());
        return;
    }
    std::vector<Record> records;
    if (!Read(jobs, records)) {
        std::fprintf(stderr, "V3 precompile: %s is not a job file\n", jobs.string().c_str());
        return;
    }
    if (!presenter || !liverpool) return;
    auto& cache = presenter->GetRasterizer().GetPipelineCache();
    const u32 total = static_cast<u32>(records.size());
    std::fprintf(stderr, "V3 precompile: %u records from %s\n", total, jobs.string().c_str());

    // This port's "Compiling shaders" screen: the VideoOut present thread draws a blank frame
    // every vblank until the game opens its port, and the presenter draws the overlay over each.
    // Never presented from here: two threads presenting raced on the swapchain (a crash in
    // NVIDIA's driver, a read at 0x130, on the present thread).
    TlgOverlay::SetLoadingCompile(true);
    TlgOverlay::SetLoading(true, 0, total);
    // TLG_V3_PRECOMPILE_TRACE=N: the first N records named before they are looked up.
    const char* trace_value = std::getenv("TLG_V3_PRECOMPILE_TRACE");
    const u32 trace = trace_value ? static_cast<u32>(std::strtoul(trace_value, nullptr, 10)) : 0;

    static auto* const code = new CodeStore; // kept: later permutations translate from it
    auto regs = std::make_unique<AmdGpu::Regs>();
    auto& cs = liverpool->GetCsRegs();
    const AmdGpu::ComputeProgram saved_cs = cs;
    u32 done = 0, graphics = 0, compute = 0, failed = 0, skipped = 0;
    std::unordered_set<u64> seen_compute;
    const auto started = std::chrono::steady_clock::now();
    for (const auto& record : records) {
        ++done;
        if (done <= trace) {
            std::fprintf(stderr, "V3 precompile: record %u kind %u, %zu stages\n", done, record.kind,
                         record.stages.size());
            std::fflush(stderr);
        }
        if ((done & 63) == 0) TlgOverlay::SetLoading(true, done, total);
        if ((done % 2000) == 0) {
            std::fprintf(stderr, "V3 precompile: %u / %u, %u graphics, %u compute, %u skipped, %u failed, %.0f s\n",
                         done, total, graphics, compute, skipped, failed,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        }
        Arena arena;
        std::unordered_map<u64, std::vector<u32>> flats;
        for (const auto& stage : record.stages) flats[stage.hash] = stage.flat;
        Shader::g_synthetic_flats = &flats;
        g_assert_throws = true;
        try {
            if (record.kind == 0) {
                regs->reg_array.fill(0);
                for (const auto& [reg, value] : record.regs) {
                    if (reg < regs->reg_array.size()) regs->reg_array[reg] = value;
                }
                for (const auto& stage : record.stages) {
                    auto* program = const_cast<AmdGpu::ShaderProgram*>(regs->ProgramForStage(stage.hw_stage));
                    if (!program) continue;
                    u64 address{};
                    PlaceStage(*code, arena, stage, address, program->user_data);
                    program->address = address;
                }
                Vulkan::GraphicsLookupScratch scratch{};
                if (cache.LookupOrBuildGraphics(*regs, scratch)) {
                    ++graphics;
                } else {
                    ++failed;
                }
            } else {
                // The compute record's registers are the ComputeProgram's words (same layout here).
                AmdGpu::ComputeProgram program{};
                auto* words = reinterpret_cast<u32*>(&program);
                for (const auto& [reg, value] : record.regs) {
                    if (reg < sizeof(program) / sizeof(u32)) words[reg] = value;
                }
                if (record.stages.size() == 1) {
                    u64 address{};
                    PlaceStage(*code, arena, record.stages[0], address, program.user_data);
                    program.address = address;
                }
                cs = program;
                if (cache.GetComputePipeline()) {
                    ++compute;
                } else {
                    ++failed;
                }
            }
        } catch (const std::exception& e) {
            if (failed < 12) {
                std::fprintf(stderr, "V3 precompile: record %u (kind %u) skipped: %s\n", done, record.kind, e.what());
            }
            ++failed;
        }
        g_assert_throws = false;
        Shader::g_synthetic_flats = nullptr;
    }
    cs = saved_cs;
    TlgOverlay::SetLoadingCompile(false);
    TlgOverlay::SetLoading(false, 0, 0);
    std::fprintf(stderr, "V3 precompile: %u records, %u graphics, %u compute, %u failed in %.0f s\n", total,
                 graphics, compute, failed,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    std::ofstream{marker} << graphics << " graphics, " << compute << " compute, " << failed << " failed\n";
}

} // namespace V3
