// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include "shader_recompiler/spirv_variants.h"
#include "common/thread.h"
#include "common/elf_info.h"
#include "common/hash.h"
#include "common/scm_rev.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/shader_scan.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 4u; // GDS append index computed at run time (d6857fbc)
static constexpr u32 ShaderMetaVersion = 4u; // ImageSpecialization gained is_block_1d
static constexpr u32 PipelineKeyVersion = 4u; // SerializationSupport gained image_binding_counts
// SHADPS4_PRECOMPILE: metas of permutations built from guessed contexts carry this bit, so builds
// whose compute key lacks num_user_data / LDS size (before 2026-10-07) reject them as stale.
static constexpr u32 PrecompiledMetaBit = 0x80000000u;
} // namespace Serialization

namespace Vulkan {

namespace {
// SHADPS4_CACHE_REBUILD WarmUp time split (warmup_times).
using WarmClock = std::chrono::steady_clock;
u64 NsSince(WarmClock::time_point t0) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(WarmClock::now() - t0).count());
}
} // Anonymous namespace

u32 GetShaderBinaryVersion() {
    return Serialization::ShaderBinaryVersion;
}

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx, bool precompiled) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }
    if (info.nodma_pinned || Shader::NoDmaPinnedHash(info.pgm_hash)) {
        return; // q9b: a NODMA module never enters the shader cache
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion |
               (precompiled ? Serialization::PrecompiledMetaBit : 0u));
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }
    if (Shader::NoDmaPinnedHash(pgm_hash)) {
        return; // q9b
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool SameBindingLayout(const Shader::Info& a, const Shader::Info& b) {
    // ud_mask: dispatch pushes user data by the program's shared mask (Fable finding 1).
    if (a.uses_dma != b.uses_dma || a.buffers.size() != b.buffers.size() ||
        a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size() ||
        a.fmasks.size() != b.fmasks.size() || a.ud_mask.mask != b.ud_mask.mask) {
        return false;
    }
    for (size_t i = 0; i < a.buffers.size(); ++i) {
        if (a.buffers[i].buffer_type != b.buffers[i].buffer_type) {
            return false;
        }
    }
    for (size_t i = 0; i < a.images.size(); ++i) {
        if (a.images[i].is_written != b.images[i].is_written) {
            return false;
        }
    }
    return true;
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx, bool* precompiled) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if ((meta_version & ~Serialization::PrecompiledMetaBit) != Serialization::ShaderMetaVersion) {
        return false;
    }
    if (precompiled) {
        *precompiled = (meta_version & Serialization::PrecompiledMetaBit) != 0;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    // Version 3 -> 4 changed only shaders that append / consume through GDS (the counter index
    // was baked at compile time, d6857fbc). A version-3 entry of any other shader is still exact,
    // so the shared cache folder (cache_storage.cpp Open) stays warm across the bump.
    const bool upgradable = binary_version == 3u && Serialization::ShaderBinaryVersion == 4u;
    if (binary_version != Serialization::ShaderBinaryVersion && !upgradable) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);
    if (Shader::NoDmaMode() && info.stage == Shader::Stage::Compute && info.uses_dma) {
        return false; // q9b: translate it (the NODMA rule may apply)
    }
    if (binary_version != Serialization::ShaderBinaryVersion) {
        for (const auto& buffer : info.buffers) {
            if (buffer.buffer_type == Shader::BufferType::GdsBuffer) {
                return false; // recompiled with the fix, stored as version 4
            }
        }
    }

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(image_binding_counts);
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(image_binding_counts);
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    // A prior preload may have returned midway through loading its stages.
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    const auto t_meta = WarmClock::now();
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    warmup_times.meta_load += NsSince(t_meta);
    if (meta_blob.empty()) {
        return false;
    }
    // SHADPS4_PRECOMPILE fix A: a guessed (marked) meta loads after every unmarked one. The
    // version word written first by RegisterShaderMeta carries the marker.
    if (defer_marked_compute && meta_blob.size() >= sizeof(u32)) {
        u32 meta_version{};
        std::memcpy(&meta_version, meta_blob.data(), sizeof(meta_version));
        if ((meta_version & Serialization::PrecompiledMetaBit) != 0) {
            defer_marked_compute->push_back({compute_key, std::move(sdata), std::move(meta_blob)});
            return true; // counted as loaded; WarmUp corrects the count if it fails later
        }
    }
    return BuildComputePipeline(compute_key, sdata, std::move(meta_blob));
}

bool PipelineCache::BuildComputePipeline(const ComputePipelineKey& key,
                                         ComputePipeline::SerializationSupport& sdata,
                                         std::vector<u8>&& meta_blob) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    compute_key = key;

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    const auto t_ctor = WarmClock::now();
    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);
    warmup_times.compute_ctor += NsSince(t_ctor);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
    sdata.Write(image_binding_counts);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    sdata.Read(image_binding_counts);
    return true;
}

namespace {
/// SHADPS4_PARALLEL_WARMUP=1 (default off, 2026-10-07): the boot preload constructs the cached
/// graphics pipelines on several threads after loading them in order. The constructor holds the
/// driver compile, ~30 ms per pipeline the driver's own cache does not hold (measured s176): with
/// ~5,700 cached graphics pipelines that was the minutes-long black screen of a build's first
/// launch. Everything the constructor reads (Infos, modules, fetch data) is complete before the
/// threads start and nothing else runs (the game has not started).
struct PreloadGraphicsJob {
    GraphicsPipelineKey key{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipeline::SerializationSupport sdata{};
    std::unique_ptr<GraphicsPipeline> pipeline;
};
/// Set only while WarmUp runs with SHADPS4_PARALLEL_WARMUP (single-threaded loading phase).
std::vector<PreloadGraphicsJob>* g_preload_jobs = nullptr;

bool ParallelWarmUpEnabled() {
    const char* value = std::getenv("SHADPS4_PARALLEL_WARMUP");
    return value && value[0] == '1';
}

/// SHADPS4_CACHE_REBUILD (2026-10-16): 1 rebuild missing SPIR-V from the game's files, 2 VERIFY
/// cached SPIR-V against a rebuild, 3 both (LoadPipelineStage).
int CacheRebuildMode() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_CACHE_REBUILD");
        return value ? std::atoi(value) : 0;
    }();
    return mode;
}

/// vs-formats.bin key: shader hashes have a zero upper half (measured), perm < 256.
u64 VsFormatKey(u64 pgm_hash, size_t perm_idx) {
    return (pgm_hash << 8) | (perm_idx & 0xff);
}

constexpr u32 VsFormatsMagic = 0x4d465356; // "VSFM"
// 3: per attribute semantic 6 bytes - conversion code, number class, swizzle x y z w (0xFF = use the spec)
constexpr u32 VsFormatsVersion = 3;
constexpr u32 VsFormatsRecord = 6;

/// seed-stamp.bin (2026-10-16, study sec. 5 findings 1-4): written by a =2 VERIFY run, shipped with
/// a seed: game serial + APP_VER (logged only), emulator build (scm description), the eboot code hash
/// (v3, 2026-10-19: the game identity), the Shader::Profile and dynamic
/// vertex input of the machine that verified it, and per shader hash the game file's OrbShdr crc32
/// and code length.
constexpr u32 SeedStampMagic = 0x50545353; // "SSTP"
constexpr u32 SeedStampVersion = 3; // 2: permutations per hash, EXE hash in the build id; 3: eboot code hash

/// The emulator build a seed was verified with: the scm description plus the running EXE file's
/// hash (Opus review D: s1300 and s1309-s1312, five different builds, shared one description).
const std::string& BuildId() {
    static const std::string id =
        fmt::format("{} exe {:016x}", Common::g_scm_desc, VideoCore::ExecutableHash());
    return id;
}
constexpr std::string_view SeedStampFile = "seed-stamp.bin";
/// The seed's g_ key names (scratch/make-release script; PipelineCache::seed_graphics_keys).
constexpr std::string_view SeedGraphicsKeysFile = "seed-graphics-keys.txt";

/// A cached .spv is used only when whole: SPIR-V magic, every instruction inside the file, the
/// last one OpFunctionEnd. A first boot killed while the IO thread wrote a file can leave it
/// truncated (study sec. 5 finding 5).
bool SpirvLooksWhole(const std::vector<u32>& spv) {
    if (spv.size() < 6 || spv[0] != 0x07230203u) {
        return false;
    }
    size_t i = 5;
    u32 last_op = 0;
    while (i < spv.size()) {
        const u32 n = spv[i] >> 16;
        if (n == 0 || n > spv.size() - i) {
            return false;
        }
        last_op = spv[i] & 0xffff;
        i += n;
    }
    return last_op == 56; // OpFunctionEnd
}

/// SHADPS4_CACHE_REBUILD_THREADS=N (default 0 = off): with =1, the stages without SPIR-V are
/// translated on N threads before the serial preload (which then takes them from prebuilt).
u32 RebuildThreads() {
    static const u32 n = [] {
        const char* value = std::getenv("SHADPS4_CACHE_REBUILD_THREADS");
        return value ? std::min<u32>(static_cast<u32>(std::strtoul(value, nullptr, 10)), 64u) : 0u;
    }();
    return n;
}

/// SHADPS4_CACHE_REBUILD_SIM_PROFILE=1 (=2 only): RebuildStageSpv translates with an AMD-like
/// profile; the VERIFY then counts layout-equal stages instead of word-equal ones.
bool SimProfileActive() {
    static const bool on = [] {
        const char* value = std::getenv("SHADPS4_CACHE_REBUILD_SIM_PROFILE");
        return value && value[0] == '1';
    }();
    return on;
}

void PutString(std::ofstream& out, std::string_view text) {
    const u32 n = static_cast<u32>(text.size());
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(text.data(), n);
}

bool GetString(std::ifstream& in, std::string& text) {
    u32 n{};
    in.read(reinterpret_cast<char*>(&n), 4);
    if (!in || n > 4096) {
        return false;
    }
    text.resize(n);
    in.read(text.data(), n);
    return static_cast<bool>(in);
}
} // Anonymous namespace

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    graphics_key.Deserialize(ar);
    // A seed key preloads only where the stamp allows graphics (seed_graphics_keys).
    if (!seed_stamp.graphics &&
        seed_graphics_keys.contains(std::hash<GraphicsPipelineKey>{}(graphics_key))) {
        ++seed_graphics_keys_skipped;
        return false;
    }

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        const auto t_meta = WarmClock::now();
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        warmup_times.meta_load += NsSince(t_meta);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    if (g_preload_jobs) {
        // SHADPS4_PARALLEL_WARMUP: constructed by WarmUp's threads, inserted after them.
        auto& job = g_preload_jobs->emplace_back();
        job.key = graphics_key;
        job.infos = infos;
        job.runtime_infos = runtime_infos;
        job.modules = modules;
        job.fetch_shader = fetch_shader;
        job.sdata = std::move(sdata);
        infos.fill(nullptr);
        modules.fill(nullptr);
        fetch_shader.reset();
        return true;
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();

    if (!it.value()->IsCreated()) {
        graphics_pipelines.erase(it);
        ++preload_refused;
        return false;
    }
    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    std::optional<Shader::Gcn::FetchShaderData> loaded_fetch;
    bool marked = false;
    if (!LoadShaderMeta(ar, program->info, loaded_fetch, spec, perm_idx, &marked)) {
        return false;
    }
    if (marked && precompile_retired.contains(program->info.pgm_hash)) {
        // Fix A: a guess left over from a program retired after a live miss in an earlier launch.
        ++retired_metas_skipped;
        return false;
    }
    // SHADPS4_PRECOMPILE (Fable review 2026-10-07 finding 5): keep the marker across launches, so
    // the first live dispatch still reports PrecompiledUse and a MISS still retires the program.
    // A program counts as precompiled only while every loaded permutation is (fix A, below).
    program->precompiled = marked;
    // Preserve loader scratch state on later failure; on success use the module's
    // immutable owner below. Serialized values and cache format stay unchanged.
    fetch_shader = loaded_fetch
                       ? std::make_shared<const Shader::Gcn::FetchShaderData>(*loaded_fetch)
                       : nullptr;

    std::vector<u32> spv{};
    const std::string spv_name = fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx);
    {
        const auto t_spv = WarmClock::now();
        // With a rebuild a missing .spv is the normal case (a seed): look before opening, which
        // logged an Error line per missing file (21,896 in s1314).
        std::error_code ec;
        const auto spv_path = CacheRebuildMode() != 0
                                  ? Storage::DataBase::Instance().SidecarPath(spv_name + ".spv")
                                  : std::filesystem::path{};
        // A stage the pre-pass made is taken from `prebuilt` below, not read back from the file
        // the pre-pass saved meanwhile (s1325: 7,630 modules made twice that way).
        if (prebuilt.contains(spv_name)) {
        } else if (spv_path.empty() ||
                   Storage::DataBase::Instance().Prefetched(spv_name + ".spv") != nullptr ||
                   std::filesystem::exists(spv_path, ec)) {
            Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary, spv_name, spv);
        }
        warmup_times.spv_load += NsSince(t_spv);
    }
    if (CacheRebuildMode() != 0 && !spv.empty() && !SpirvLooksWhole(spv)) {
        if (++rebuild_stats.bad_spv <= 8) {
            LOG_WARNING(Render_Vulkan,
                        "CacheRebuild: {}.spv is not a whole SPIR-V module ({} words): rebuilt or "
                        "compiled at first use",
                        spv_name, spv.size());
        }
        spv.clear();
    }
    // SHADPS4_CACHE_REBUILD (2026-10-16): =1 a stage whose SPIR-V is missing is made again from the
    // game's own code + this meta (RebuildStageSpv) and saved; =2 VERIFY: every stage that has
    // SPIR-V is made again and compared word for word (nothing changes). =3 both.
    bool rebuilt_known = false;
    // A pre-pass module that is not inserted below (an existing permutation, another index, a
    // layout refusal) is destroyed when this returns.
    vk::ShaderModule prebuilt_module{};
    struct ModuleGuard {
        vk::Device device;
        vk::ShaderModule* module;
        ~ModuleGuard() {
            if (*module) {
                device.destroyShaderModule(*module);
            }
        }
    } prebuilt_guard{instance.GetDevice(), &prebuilt_module};
    if (const int rebuild = CacheRebuildMode(); rebuild != 0) {
        const size_t si = std::min<size_t>(static_cast<size_t>(program->info.stage), 7);
        auto& rs = rebuild_stats;
        // Bit 4 TRACE (s1297 stopped inside WarmUp without a log line): the stage about to be
        // rebuilt, written and flushed to the cache folder's rebuild-inprogress.txt first.
        if (rebuild & 4) {
            static std::ofstream trace{
                Storage::DataBase::Instance().SidecarPath("rebuild-inprogress.txt"),
                std::ios::binary | std::ios::trunc};
            static u64 traced = 0;
            trace.seekp(0);
            trace << fmt::format("{} {:#018x} perm {} marked {} spv {} n {}            \n",
                                 program->info.stage, program->info.pgm_hash, perm_idx,
                                 marked ? 1 : 0, spv.size(), ++traced);
            trace.flush();
        }
        // A permutation another pipeline already rebuilt (or loaded) needs no SPIR-V: the module
        // is taken from program_cache below.
        const auto it_known = program_cache.find(program->info.pgm_hash);
        const bool in_cache = it_known != program_cache.end() &&
                              std::ranges::find(it_known.value()->modules, spec,
                                                &Program::Module::spec) !=
                                  it_known.value()->modules.end();
        const bool known = spv.empty() && in_cache;
        if (known && (rebuild & 1)) {
            ++rs.reused[si];
            rebuilt_known = true;
        } else if (spv.empty() && (rebuild & 1)) {
            // Seed stamp gate (study sec. 5): graphics only on the seed's GPU profile + dynamic
            // vertex input, compute on any; a hash the stamp does not list is never rebuilt.
            const bool is_compute = program->info.stage == Shader::Stage::Compute;
            if (!(is_compute ? seed_stamp.compute : seed_stamp.graphics)) {
                ++rs.kind_refused[si];
                return false;
            }
            const auto it_stamp = seed_stamp.shaders.find(program->info.pgm_hash);
            if (it_stamp == seed_stamp.shaders.end() ||
                !seed_stamp.perms.contains(VsFormatKey(program->info.pgm_hash, perm_idx))) {
                ++rs.not_stamped[si];
                return false;
            }
            if (const auto it_pre = prebuilt.find(spv_name); it_pre != prebuilt.end()) {
                // Translated, saved and made a module by the pre-pass.
                const PrebuiltStage pre = it_pre->second;
                prebuilt.erase(it_pre);
                if (!pre.same_layout) {
                    ++rs.layout_mismatch[si];
                    return false;
                }
                prebuilt_module = pre.module;
                ++rs.prebuilt[si];
                ++rs.rebuilt[si];
            } else {
                bool same_layout = false;
                bool identity_mismatch = false;
                const auto t_rebuild = WarmClock::now();
                const auto it_fmt = vs_formats.find(VsFormatKey(program->info.pgm_hash, perm_idx));
                spv = RebuildStageSpv(program->info, loaded_fetch, spec, &same_layout,
                                      it_fmt != vs_formats.end() ? &it_fmt->second : nullptr,
                                      &it_stamp->second, &identity_mismatch);
                warmup_times.rebuild += NsSince(t_rebuild);
                if (identity_mismatch) {
                    ++rs.identity_mismatch[si];
                    return false;
                }
                if (spv.empty()) {
                    ++rs.no_code[si];
                    return false;
                }
                if (!same_layout) {
                    ++rs.layout_mismatch[si];
                    return false;
                }
                ++rs.rebuilt[si];
                RegisterShaderBinary(std::vector<u32>(spv), program->info.pgm_hash, perm_idx);
            }
        } else if (!spv.empty() && (rebuild & 2) && !in_cache) { // each permutation once
            bool same_layout = false;
            const u64 fmt_key = VsFormatKey(program->info.pgm_hash, perm_idx);
            const auto it_fmt = vs_formats.find(fmt_key);
            const auto again = RebuildStageSpv(program->info, loaded_fetch, spec, &same_layout,
                                               it_fmt != vs_formats.end() ? &it_fmt->second
                                                                          : nullptr);
            if (SimProfileActive()) {
                // Simulated other GPU: words differ by design; count the binding layout only.
                if (again.empty()) {
                    ++rs.no_code[si];
                } else if (same_layout) {
                    ++rs.equal[si];
                } else {
                    ++rs.layout_mismatch[si];
                }
            }
            std::vector<u8> found;
            if (!SimProfileActive() && !again.empty() && again != spv && same_layout &&
                (rebuild & 8) &&
                program->info.stage == Shader::Stage::Vertex) {
                found = AnnotateVsConversions(program->info, loaded_fetch, spec, spv);
                if (found.empty()) {
                    ++rs.annotate_failed[si];
                } else {
                    ++rs.annotated[si];
                }
            }
            if (SimProfileActive()) {
                // counted above
            } else if (!found.empty()) {
                vs_formats[fmt_key] = std::move(found);
                vs_formats_dirty = true;
            } else if (again.empty()) {
                ++rs.no_code[si];
            } else if (again == spv && same_layout) {
                ++rs.equal[si];
                if (perm_idx < 256) {
                    stamp_perms[program->info.pgm_hash].push_back(static_cast<u8>(perm_idx));
                }
            } else {
                ++rs.differ[si];
                stamp_differ.insert(program->info.pgm_hash);
                // The shader meta of every differing permutation (file name = HashCombine(hash,
                // perm), as GetProgram saves it): a seed for players drops these, so they compile
                // at play time from the real V#s instead of a rebuild that may decode differently.
                {
                    std::ofstream list{Storage::DataBase::Instance().SidecarPath("rebuild-differ.txt"),
                                       std::ios::app};
                    list << fmt::format("{:#018x} {} {:#x} {}\n",
                                        HashCombine(program->info.pgm_hash, perm_idx),
                                        program->info.stage, program->info.pgm_hash, perm_idx);
                }
                u32 differs = 0;
                for (const u32 n : rs.differ) {
                    differs += n;
                }
                if (differs <= 3) {
                    // Both SPIR-V blobs next to the cache, for spirv-dis + diff.
                    for (const auto& [suffix, words] :
                         {std::pair<const char*, const std::vector<u32>*>{"cached", &spv},
                          std::pair<const char*, const std::vector<u32>*>{"rebuilt", &again}}) {
                        std::ofstream out{Storage::DataBase::Instance().SidecarPath(fmt::format(
                                              "rebuild-diff-{:016x}-{}-{}.spv",
                                              program->info.pgm_hash, perm_idx, suffix)),
                                          std::ios::binary | std::ios::trunc};
                        out.write(reinterpret_cast<const char*>(words->data()),
                                  words->size() * sizeof(u32));
                    }
                }
                if (differs <= 16) {
                    size_t first = 0;
                    while (first < spv.size() && first < again.size() && spv[first] == again[first]) {
                        ++first;
                    }
                    LOG_WARNING(Render_Vulkan,
                                "CacheRebuild verify: {} {:#x} perm {} differs: {} vs cached {} "
                                "words, first difference at word {}, layout {}",
                                program->info.stage, program->info.pgm_hash, perm_idx, again.size(),
                                spv.size(), first, same_layout ? "same" : "DIFFERENT");
                }
            }
        }
    }
    if (spv.empty() && !rebuilt_known && !prebuilt_module) {
        return false;
    }
    DumpOneShader(spv, program->info.pgm_hash, program->info.stage, perm_idx, "cached");

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        const auto t_module = WarmClock::now();
        module = prebuilt_module ? std::exchange(prebuilt_module, vk::ShaderModule{})
                                 : CompileSPV(spv, instance.GetDevice());
        warmup_times.compile_spv += NsSince(t_module);
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // A matching permutation is valid only at its original index. A different index means
            // the store holds entries from more than one cache generation, so this pipeline is
            // left to compile at runtime.
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, program->info.stage, program->info.pgm_hash, idx);
                return false;
            }
            module = it->module;
        } else {
            // Every permutation is bound through the program's first Info. A cached permutation
            // from another cache generation can have other bindings (DMA buffers shift every
            // image), so it is left to compile at runtime, where it is pinned to that Info.
            if (!SameBindingLayout(program->info, it_pgm.value()->info)) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} has a different binding layout "
                            "than its program, skipping preload",
                            perm_idx, program->info.stage, program->info.pgm_hash);
                return false;
            }
            const auto t_module = WarmClock::now();
            module = prebuilt_module ? std::exchange(prebuilt_module, vk::ShaderModule{})
                                     : CompileSPV(spv, instance.GetDevice());
            warmup_times.compile_spv += NsSince(t_module);
        }
        // Fix A: one live (unmarked) permutation makes the program a live one: no PrecompiledUse
        // line and no retire on its first dispatch. Set only for a permutation actually inserted;
        // a live meta skipped above for its layout leaves a guessed program to retire at its
        // first miss.
        it_pgm.value()->precompiled = it_pgm.value()->precompiled && marked;
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    fetch_shader = it_pgm.value()->modules[perm_idx].fetch_shader;

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;

    return true;
}

void PipelineCache::ReadSeedGraphicsKeys() {
    seed_graphics_keys.clear();
    std::ifstream in{Storage::DataBase::Instance().SidecarPath(SeedGraphicsKeysFile)};
    std::string line;
    while (std::getline(in, line)) {
        // One key file name per line, without ".key": g_0x<16 hex digits>.
        if (line.size() >= 20 && line.starts_with("g_0x")) {
            seed_graphics_keys.insert(std::strtoull(line.c_str() + 4, nullptr, 16));
        }
    }
    if (!seed_graphics_keys.empty()) {
        LOG_INFO(Render, "CacheRebuild: {} lists {} seed graphics keys", SeedGraphicsKeysFile,
                 seed_graphics_keys.size());
    }
}

void PipelineCache::ReadSeedStamp() {
    seed_stamp = {};
    std::ifstream in{Storage::DataBase::Instance().SidecarPath(SeedStampFile),
                     std::ios::binary | std::ios::ate};
    const std::streamoff file_size = in ? static_cast<std::streamoff>(in.tellg()) : 0;
    in.seekg(0);
    u32 magic{}, version{}, profile_size{}, count{};
    u64 code_hash{};
    std::string serial, app_ver, build;
    std::vector<u8> profile_bytes;
    u8 dynamic_vertex_input{};
    std::unordered_map<u64, std::pair<u32, u32>> shaders;
    std::unordered_set<u64> perms;
    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    bool ok = in && magic == SeedStampMagic && version == SeedStampVersion &&
              GetString(in, serial) && GetString(in, app_ver) && GetString(in, build);
    if (ok) {
        in.read(reinterpret_cast<char*>(&code_hash), 8);
        in.read(reinterpret_cast<char*>(&profile_size), 4);
        ok = in && profile_size <= 4096;
    }
    if (ok) {
        profile_bytes.resize(profile_size);
        in.read(reinterpret_cast<char*>(profile_bytes.data()), profile_size);
        in.read(reinterpret_cast<char*>(&dynamic_vertex_input), 1);
        in.read(reinterpret_cast<char*>(&count), 4);
        // Each entry is at least 17 bytes (review D: a corrupt count must not reserve hundreds of
        // MB before the read fails).
        const std::streamoff left = file_size - static_cast<std::streamoff>(in.tellg());
        ok = in && left >= 0 && static_cast<std::streamoff>(count) * 17 <= left;
    }
    if (ok) {
        shaders.reserve(count);
        for (u32 i = 0; i < count && in; ++i) {
            u64 hash{};
            u32 crc32{}, length{};
            u8 num_perms{};
            in.read(reinterpret_cast<char*>(&hash), 8);
            in.read(reinterpret_cast<char*>(&crc32), 4);
            in.read(reinterpret_cast<char*>(&length), 4);
            in.read(reinterpret_cast<char*>(&num_perms), 1);
            std::array<u8, 256> perm_list{};
            in.read(reinterpret_cast<char*>(perm_list.data()), num_perms);
            if (in) {
                shaders.emplace(hash, std::make_pair(crc32, length));
                for (u32 k = 0; k < num_perms; ++k) {
                    perms.insert(VsFormatKey(hash, perm_list[k]));
                }
            }
        }
        ok = static_cast<bool>(in) && shaders.size() == count;
    }
    if (!ok) {
        LOG_WARNING(Render, "CacheRebuild: no valid {} in the cache folder: no stage is rebuilt",
                    SeedStampFile);
        return;
    }
    // The game is identified by its eboot code, not by param.sfo (2026-10-19: a DLC param.sfo in a
    // player's game folder read as "Unknown version"); every rebuilt stage is also checked against
    // the game file's shader crc32 + length.
    const auto& elf = Common::ElfInfo::Instance();
    if (code_hash == 0 || code_hash != elf.EbootCodeHash()) {
        LOG_WARNING(Render,
                    "CacheRebuild: the seed was made for game code {:#018x} ({} {}), this game's "
                    "code is {:#018x} ({} {}): no stage is rebuilt",
                    code_hash, serial, app_ver, elf.EbootCodeHash(), elf.GameSerial(),
                    elf.AppVer());
        return;
    }
    if (serial != elf.GameSerial() || app_ver != elf.AppVer()) {
        LOG_INFO(Render,
                 "CacheRebuild: param.sfo says {} {}, the seed's says {} {}; the game code matches "
                 "the seed's ({:#018x}), so the seed applies",
                 elf.GameSerial(), elf.AppVer(), serial, app_ver, code_hash);
    }
    if (build != BuildId()) {
        LOG_WARNING(Render,
                    "CacheRebuild: the seed was verified with emulator build {}, this is {}: no "
                    "stage is rebuilt (a =2 VERIFY run with this build writes a new stamp)",
                    build, BuildId());
        return;
    }
    Shader::Profile seed_profile{};
    const bool same_profile = profile_bytes.size() == sizeof(seed_profile) &&
                              (std::memcpy(&seed_profile, profile_bytes.data(),
                                           sizeof(seed_profile)),
                               seed_profile == profile);
    const bool same_vertex_input =
        (dynamic_vertex_input != 0) == instance.IsVertexInputDynamicState();
    seed_stamp.compute = true;
    seed_stamp.graphics = same_profile && same_vertex_input;
    seed_stamp.shaders = std::move(shaders);
    seed_stamp.perms = std::move(perms);
    if (seed_stamp.graphics) {
        LOG_INFO(Render,
                 "CacheRebuild: seed stamp {} {} build {}: {} shaders, {} permutations, all stages",
                 serial, app_ver, build, seed_stamp.shaders.size(), seed_stamp.perms.size());
    } else {
        LOG_WARNING(Render,
                    "CacheRebuild: seed stamp {} {} build {}: {} shaders, {} permutations; this "
                    "GPU's profile {} the seed's, dynamic vertex input {}: compute stages only",
                    serial, app_ver, build, seed_stamp.shaders.size(), seed_stamp.perms.size(),
                    same_profile ? "equals" : "differs from",
                    same_vertex_input ? "same" : "differs");
    }
}

void PipelineCache::WriteSeedStamp() const {
    struct Entry {
        u64 hash;
        u32 crc32;
        u32 length;
        std::vector<u8> perms;
    };
    std::vector<Entry> entries;
    entries.reserve(stamp_perms.size());
    u64 total_perms = 0;
    u32 left_out = 0;
    for (const auto& [hash, perm_list] : stamp_perms) {
        u32 crc32{}, length{};
        if (stamp_differ.contains(hash)) {
            ++left_out; // a program with a differing permutation compiles live as a whole
            continue;
        }
        if (!GameShaderIdentity(hash, &crc32, &length)) {
            continue;
        }
        auto perms = perm_list;
        std::ranges::sort(perms);
        perms.erase(std::unique(perms.begin(), perms.end()), perms.end());
        total_perms += perms.size();
        entries.push_back({hash, crc32, length, std::move(perms)});
    }
    std::ranges::sort(entries, {}, &Entry::hash);
    std::ofstream out{Storage::DataBase::Instance().SidecarPath(SeedStampFile),
                      std::ios::binary | std::ios::trunc};
    const auto& elf = Common::ElfInfo::Instance();
    out.write(reinterpret_cast<const char*>(&SeedStampMagic), 4);
    out.write(reinterpret_cast<const char*>(&SeedStampVersion), 4);
    PutString(out, elf.GameSerial());
    PutString(out, elf.AppVer());
    PutString(out, BuildId());
    const u64 code_hash = elf.EbootCodeHash();
    out.write(reinterpret_cast<const char*>(&code_hash), 8);
    const u32 profile_size = sizeof(profile);
    out.write(reinterpret_cast<const char*>(&profile_size), 4);
    out.write(reinterpret_cast<const char*>(&profile), profile_size);
    const u8 dynamic_vertex_input = instance.IsVertexInputDynamicState() ? 1 : 0;
    out.write(reinterpret_cast<const char*>(&dynamic_vertex_input), 1);
    const u32 count = static_cast<u32>(entries.size());
    out.write(reinterpret_cast<const char*>(&count), 4);
    for (const auto& entry : entries) {
        const u8 num_perms = static_cast<u8>(std::min<size_t>(entry.perms.size(), 255));
        out.write(reinterpret_cast<const char*>(&entry.hash), 8);
        out.write(reinterpret_cast<const char*>(&entry.crc32), 4);
        out.write(reinterpret_cast<const char*>(&entry.length), 4);
        out.write(reinterpret_cast<const char*>(&num_perms), 1);
        out.write(reinterpret_cast<const char*>(entry.perms.data()), num_perms);
    }
    LOG_INFO(Render,
             "CacheRebuild: {} written: {} shaders, {} permutations verified word-equal ({} programs "
             "with a differing permutation left out), game {} {} (code {:#018x}), build {}, dynamic "
             "vertex input {}",
             SeedStampFile, count, total_perms, left_out, elf.GameSerial(), elf.AppVer(), code_hash,
             BuildId(), dynamic_vertex_input);
}

void PipelineCache::PrebuildMissingStages(u32 num_threads) {
    const auto folder = Storage::DataBase::Instance().SidecarPath("");
    if (folder.empty()) {
        return;
    }
    u32 crc_unused{}, len_unused{};
    GameShaderIdentity(0, &crc_unused, &len_unused); // builds the index here, before the threads
    std::vector<std::filesystem::path> metas;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
        if (entry.path().extension() == ".meta") {
            metas.push_back(entry.path());
        }
    }
    num_threads =
        std::clamp<u32>(num_threads, 1u, std::max(1u, std::thread::hardware_concurrency()));
    // Workers translate; the main thread saves each SPIR-V and makes its module as it arrives, so
    // only modules are kept (review E).
    struct Ready {
        std::string name;
        u64 hash{};
        size_t perm{};
        std::vector<u32> spv;
        bool same_layout{};
    };
    std::mutex ready_mutex;
    std::condition_variable ready_cv;
    std::deque<Ready> ready;
    std::atomic<size_t> next{0};
    std::atomic<u32> finished{0}, failed{0};
    u32 made = 0, refused_layout = 0;
    auto drain = [&] {
        std::deque<Ready> batch;
        {
            std::scoped_lock lock{ready_mutex};
            batch.swap(ready);
        }
        for (auto& item : batch) {
            if (!item.same_layout) {
                prebuilt.emplace(std::move(item.name), PrebuiltStage{{}, false});
                ++refused_layout;
                continue;
            }
            const auto t_module = WarmClock::now();
            const vk::ShaderModule module = CompileSPV(item.spv, instance.GetDevice());
            warmup_times.compile_spv += NsSince(t_module);
            const auto [it, inserted] =
                prebuilt.emplace(std::move(item.name), PrebuiltStage{module, true});
            if (!inserted) {
                instance.GetDevice().destroyShaderModule(module);
                continue;
            }
            RegisterShaderBinary(std::move(item.spv), item.hash, item.perm);
            ++made;
        }
    };
    {
        std::vector<std::jthread> threads;
        for (u32 t = 0; t < num_threads; ++t) {
            threads.emplace_back([&] {
                Shader::Pools thread_pools;
                for (size_t i = next.fetch_add(1); i < metas.size();
                     i = next.fetch_add(1), finished.fetch_add(1)) {
                    std::vector<u8> bytes;
                    if (const auto* kept = Storage::DataBase::Instance().Prefetched(
                            metas[i].filename().string())) {
                        bytes = *kept; // SHADPS4_CACHE_PREFETCH=2: no second open
                    } else {
                        std::ifstream file{metas[i], std::ios::binary | std::ios::ate};
                        if (!file) {
                            continue;
                        }
                        bytes.resize(static_cast<size_t>(file.tellg()));
                        file.seekg(0);
                        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
                        if (!file) {
                            continue;
                        }
                    }
                    Serialization::Archive ar{std::move(bytes)};
                    auto program = std::make_unique<Program>();
                    Shader::StageSpecialization spec{};
                    spec.info = &program->info;
                    size_t perm_idx{};
                    std::optional<Shader::Gcn::FetchShaderData> fetch;
                    bool marked = false;
                    if (!LoadShaderMeta(ar, program->info, fetch, spec, perm_idx, &marked)) {
                        continue;
                    }
                    const u64 hash = program->info.pgm_hash;
                    if (marked && precompile_retired.contains(hash)) {
                        continue;
                    }
                    const bool is_compute = program->info.stage == Shader::Stage::Compute;
                    if (!(is_compute ? seed_stamp.compute : seed_stamp.graphics)) {
                        continue;
                    }
                    const auto it_stamp = seed_stamp.shaders.find(hash);
                    if (it_stamp == seed_stamp.shaders.end() ||
                        !seed_stamp.perms.contains(VsFormatKey(hash, perm_idx))) {
                        continue;
                    }
                    std::string name = fmt::format("{:#018x}_{}", hash, perm_idx);
                    std::error_code exists_ec;
                    if (std::filesystem::exists(folder / (name + ".spv"), exists_ec)) {
                        continue;
                    }
                    const auto it_fmt = vs_formats.find(VsFormatKey(hash, perm_idx));
                    bool same_layout = false;
                    bool identity_mismatch = false;
                    auto spv = RebuildStageSpv(
                        program->info, fetch, spec, &same_layout,
                        it_fmt != vs_formats.end() ? &it_fmt->second : nullptr, &it_stamp->second,
                        &identity_mismatch, &thread_pools);
                    if (spv.empty() || identity_mismatch) {
                        failed.fetch_add(1);
                        continue; // the serial preload tries again and counts the reason
                    }
                    {
                        std::scoped_lock lock{ready_mutex};
                        ready.push_back({std::move(name), hash, perm_idx, std::move(spv),
                                         same_layout});
                    }
                    ready_cv.notify_one();
                }
            });
        }
        // The main thread saves the results, keeps the window alive and shows progress
        // (ShowBootProgress pumps events) until every meta has been taken.
        while (finished.load() < metas.size()) {
            {
                std::unique_lock lock{ready_mutex};
                ready_cv.wait_for(lock, std::chrono::milliseconds{100},
                                  [&] { return !ready.empty(); });
            }
            drain();
            ShowBootProgress("shaders prepared", finished.load(), metas.size());
        }
    } // joins
    drain(); // results pushed after the last check
    const auto t_compact = WarmClock::now();
    const size_t largest_free = VideoCore::CompactProcessHeap();
    LOG_INFO(Render,
             "CacheRebuild: pre-pass on {} threads: {} metas, {} stages translated and saved, {} "
             "refused for their layout, {} failed (left to the serial preload); heap compacted in "
             "{} ms (largest free block {} KiB)",
             num_threads, metas.size(), made, refused_layout, failed.load(),
             NsSince(t_compact) / 1000000, largest_free >> 10);
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }
    if (Shader::SpirvVariantActive()) {
        LOG_WARNING(Render, "A SPIR-V variant flag is set: the pipeline cache is not used this run");
        return;
    }
    // SHADPS4_PIPELINE_CACHE_SKIP=1: neither preload nor save the on-disk cache this run, so
    // every pipeline compiles at first use (a test bed for SHADPS4_ASYNC_PIPELINES that needs no
    // new area). Unlike the SPIR-V variant flags it changes nothing in the shaders; the store
    // stays unopened, so Register* and Sync are no-ops.
    static const bool skip_cache = [] {
        const char* value = std::getenv("SHADPS4_PIPELINE_CACHE_SKIP");
        return value && value[0] == '1';
    }();
    if (skip_cache) {
        LOG_WARNING(Render, "SHADPS4_PIPELINE_CACHE_SKIP: the pipeline cache is neither read nor "
                            "written this run");
        return;
    }

    Storage::DataBase::Instance().Open(instance.IsNativeBindingsActive());
    // Check if cache is compatible (read before the prefetch: the boot title / screen says from
    // the start whether this launch prepares shaders)
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    // Boot title (2026-10-17, user): "Preparing shaders (one time)" only on a seeded folder's first
    // launch or while stages are really rebuilt; any other launch says "Loading shaders".
    const bool fresh_folder = profile_data.empty();
    const auto preparing = [&] {
        if (fresh_folder) {
            return true;
        }
        for (size_t s = 0; s < 8; ++s) {
            if (rebuild_stats.rebuilt[s] + rebuild_stats.prebuilt[s] != 0) {
                return true;
            }
        }
        return false;
    };
    // Boot progress for SHADPS4_BOOT_SCREEN: the phases weighted by rc22's cold first launch
    // (s1348: prefetch 16.6 s, read + rebuild 61.0 s, compile drain 33.8 s) as 0-15 / 15-70 /
    // 70-100 %; inside a phase the real counts.
    const bool show_boot = CacheRebuildMode() != 0;
    Storage::DataBase::Instance().Prefetch([&](u64 done, u64 total) {
        if (show_boot) {
            ShowBootProgress("cache files read", done, total, fresh_folder,
                             total ? 0.15f * static_cast<float>(done) / static_cast<float>(total)
                                   : 0.0f);
        }
    });

    if (profile_data.empty()) {
        std::vector<u8> fresh(sizeof(profile));
        std::memcpy(fresh.data(), &profile, sizeof(profile));
        // SHADPS4_CACHE_REBUILD=1 (2026-10-16): a folder seeded with pipeline keys + shader metas
        // but no profile and no SPIR-V (a shipped set) goes on to the preload below, which rebuilds
        // every stage from the game's own files for this machine's profile (LoadPipelineStage).
        if ((CacheRebuildMode() & 1) == 0) {
            Storage::DataBase::Instance().FinishPreload();
            Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                               std::move(fresh));
            // SHADPS4_PRECOMPILE (2026-10-16, notes/study-shader-stutter-2026-10-16.md sec. 4
            // step 3): a fresh cache folder (a new player's first launch) has nothing to preload
            // but still gets the pre-compile pass; this branch used to return, so the pass (and a
            // context table) took effect only from the second launch on. Returns at once when the
            // flag is off.
            precompile_retired = ReadPrecompileHashes(PrecompileRetiredFile);
            PrecompileFromExtract();
            return;
        }
        profile_data = fresh;
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(fresh));
        LOG_INFO(Render, "CacheRebuild: no profile in the cache folder; this machine's saved, "
                         "stages without SPIR-V are rebuilt from the game's files");
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    const auto t_warmup = std::chrono::steady_clock::now();
    std::vector<PreloadGraphicsJob> preload_jobs;
    if (ParallelWarmUpEnabled()) {
        g_preload_jobs = &preload_jobs;
    }
    // SHADPS4_CACHE_REBUILD: the vertex-attribute conversions of vs permutations (ANNOTATE output,
    // shipped with a seed folder).
    if (CacheRebuildMode() != 0) {
        std::ifstream in{Storage::DataBase::Instance().SidecarPath("vs-formats.bin"),
                         std::ios::binary};
        u32 magic{}, version{}, count{};
        in.read(reinterpret_cast<char*>(&magic), 4);
        in.read(reinterpret_cast<char*>(&version), 4);
        in.read(reinterpret_cast<char*>(&count), 4);
        if (in && magic == VsFormatsMagic && version == VsFormatsVersion && count < (1u << 22)) {
            for (u32 e = 0; e < count && in; ++e) {
                u64 key{};
                u8 n{};
                in.read(reinterpret_cast<char*>(&key), 8);
                in.read(reinterpret_cast<char*>(&n), 1);
                std::vector<u8> codes(size_t{n} * VsFormatsRecord);
                in.read(reinterpret_cast<char*>(codes.data()), codes.size());
                if (in) {
                    vs_formats[key] = std::move(codes);
                }
            }
        }
        LOG_INFO(Render, "CacheRebuild: vs-formats.bin entries {}", vs_formats.size());
    }
    // SHADPS4_PRECOMPILE fix A (any build of this branch, whatever the flag: the cache may hold
    // marked metas from an earlier PRECOMPILE launch).
    precompile_retired = ReadPrecompileHashes(PrecompileRetiredFile);
    ReadSeedGraphicsKeys(); // any rebuild mode: seed keys stay out unless the stamp allows them
    if (CacheRebuildMode() & 1) {
        ReadSeedStamp();
        if (RebuildThreads() > 0 && (seed_stamp.compute || seed_stamp.graphics)) {
            const auto t_pre = WarmClock::now();
            PrebuildMissingStages(RebuildThreads());
            warmup_times.prepass += NsSince(t_pre);
        }
    }
    std::vector<DeferredComputeLoad> deferred_marked;
    defer_marked_compute = &deferred_marked;
    const size_t total_keys =
        show_boot ? Storage::DataBase::Instance().CountBlobs(Storage::BlobType::PipelineKey) : 0;
    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;
            if (show_boot) {
                const float read =
                    total_keys ? std::min(1.0f, static_cast<float>(num_total_pipelines) /
                                                    static_cast<float>(total_keys))
                               : 0.0f;
                ShowBootProgress("pipelines read", num_total_pipelines, total_keys, preparing(),
                                 0.15f + 0.55f * read);
            }

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });
    defer_marked_compute = nullptr;
    // Fix A: guessed permutations after every live one (counted as loaded above).
    u32 deferred_failed = 0;
    // 1.1 (PERF-PLAN 11): below 24 GB of RAM the guessed permutations are not preloaded. Each
    // preloaded pipeline costs ~160 KB of driver memory (s1434 / s1435: 20,683 pipelines = ~3.3 GB
    // outside the process heap); the ~10,000 guesses are the largest group and only cover first
    // uses (a live miss compiles as before). SHADPS4_PRELOAD_GUESSES=1 / =0 forces either way.
    const bool skip_guesses = [] {
        if (const char* v = std::getenv("SHADPS4_PRELOAD_GUESSES")) {
            return v[0] == '0';
        }
        const u64 ram = Common::PhysicalRamBytes();
        return ram != 0 && ram < (24ull << 30);
    }();
    if (skip_guesses && !deferred_marked.empty()) {
        LOG_INFO(Render, "WarmUp: {} guessed compute permutations not preloaded (RAM {} MB < 24 GB; "
                         "SHADPS4_PRELOAD_GUESSES=1 loads them)",
                 deferred_marked.size(), Common::PhysicalRamBytes() >> 20);
        num_pipelines -= static_cast<u32>(deferred_marked.size());
        deferred_marked.clear();
    }
    for (auto& load : deferred_marked) {
        if (!BuildComputePipeline(load.key, load.sdata, std::move(load.meta))) {
            --num_pipelines;
            ++deferred_failed;
        }
    }
    if (!deferred_marked.empty() || !precompile_retired.empty()) {
        LOG_INFO(Render, "WarmUp: precompiled compute pipelines loaded last {} (not loaded {}), "
                         "retired programs {} (their guessed metas skipped {})",
                 deferred_marked.size(), deferred_failed, precompile_retired.size(),
                 retired_metas_skipped);
    }
    g_preload_jobs = nullptr;

    if (!preload_jobs.empty()) {
        const auto t_build = std::chrono::steady_clock::now();
        const char* env = std::getenv("SHADPS4_WARMUP_THREADS");
        const u32 hw = std::max(2u, std::thread::hardware_concurrency());
        const u32 num_threads = std::clamp<u32>(
            env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : hw / 2, 1u, 16u);
        std::atomic<size_t> next{0};
        {
            std::vector<std::jthread> threads;
            for (u32 t = 0; t < num_threads; ++t) {
                threads.emplace_back([&] {
                    for (size_t i = next.fetch_add(1); i < preload_jobs.size();
                         i = next.fetch_add(1)) {
                        auto& job = preload_jobs[i];
                        job.pipeline = std::make_unique<GraphicsPipeline>(
                            instance, scheduler, desc_heap, profile, job.key, *pipeline_cache,
                            job.infos, job.runtime_infos, job.fetch_shader, job.modules, job.sdata,
                            true);
                    }
                });
            }
        } // joins
        u32 refused = 0;
        for (auto& job : preload_jobs) {
            if (!job.pipeline->IsCreated()) {
                ++refused;
                continue;
            }
            const auto [it, is_new] = graphics_pipelines.try_emplace(job.key);
            ASSERT(is_new);
            it.value() = std::move(job.pipeline);
        }
        preload_refused += refused;
        num_pipelines -= refused;
        LOG_INFO(Render, "WarmUp: {} graphics pipelines built on {} threads in {} ms",
                 preload_jobs.size() - refused, num_threads,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_build)
                     .count());
    }
    LOG_INFO(Render, "WarmUp: total {} ms",
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t_warmup)
                 .count());

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (seed_graphics_keys_skipped != 0 || preload_refused != 0) {
        LOG_INFO(Render,
                 "WarmUp: {} seed graphics keys not preloaded (stamp allows graphics: {}), {} "
                 "cached graphics pipelines refused by the driver",
                 seed_graphics_keys_skipped, seed_stamp.graphics, preload_refused);
    }
    if (CacheRebuildMode() != 0) {
        // SHADPS4_PRECOMPILE_WAIT=1 with a rebuild: the boot also waits for the compute pipelines
        // the preload deferred to the compile pool (cold driver: ~17 k x ~50 ms), so the first
        // effects never meet them still queued.
        const char* wait_env = std::getenv("SHADPS4_PRECOMPILE_WAIT");
        if ((CacheRebuildMode() & 1) && wait_env && wait_env[0] == '1') {
            const auto t_wait = std::chrono::steady_clock::now();
            size_t left_max = 0;
            while (const size_t left =
                       PipelineCompilePool::WaitIdleFor(std::chrono::milliseconds{250})) {
                left_max = std::max(left_max, left);
                ShowBootProgress("pipelines left", left, 0, preparing(),
                                 0.70f + 0.30f * (1.0f - static_cast<float>(left) /
                                                             static_cast<float>(left_max)));
            }
            LOG_INFO(Render, "CacheRebuild: waited {} ms for the compile pool to drain",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t_wait)
                         .count());
        }
        // The last boot frame stays up until the game's first one.
        const std::string_view head =
            preparing() ? "Preparing shaders (one time)" : "Loading shaders";
        PresentBootScreen(head, "Starting the game", 1.0f, false);
        // SHADPS4_BOOT_SCREEN_MIN_MS (test): another minimum than the user's 3 s.
        const char* min_env = std::getenv("SHADPS4_BOOT_SCREEN_MIN_MS");
        HoldBootScreen(head, "Starting the game", false,
                       std::chrono::milliseconds{min_env ? std::atoi(min_env) : 3000});
        ShowBootProgress({}, 0, ~0ull);
        const auto& rs = rebuild_stats;
        for (size_t s = 0; s < 8; ++s) {
            if (rs.equal[s] + rs.differ[s] + rs.no_code[s] + rs.rebuilt[s] + rs.reused[s] +
                rs.layout_mismatch[s] + rs.not_stamped[s] + rs.identity_mismatch[s] +
                rs.kind_refused[s]) {
                LOG_INFO(Render,
                         "CacheRebuild: stage {}: verify equal={} differ={} annotated={} "
                         "annotate_failed={} | rebuilt={} (prebuilt {}) reused={} "
                         "layout_mismatch={} | no code in the game index={} | refused: kind={} "
                         "not in stamp={} game file differs={}",
                         static_cast<Shader::Stage>(s), rs.equal[s], rs.differ[s],
                         rs.annotated[s], rs.annotate_failed[s], rs.rebuilt[s], rs.prebuilt[s],
                         rs.reused[s], rs.layout_mismatch[s], rs.no_code[s], rs.kind_refused[s],
                         rs.not_stamped[s], rs.identity_mismatch[s]);
            }
        }
        const auto& wt = warmup_times;
        LOG_INFO(Render,
                 "CacheRebuild: WarmUp main thread ms: meta load {} | spv load {} | rebuild {} | "
                 "shader modules {} | compute pipeline objects {} | pre-pass {} (threads {}, left "
                 "unused {}) | spv not whole {} | SRT walker arena {} KiB",
                 wt.meta_load / 1000000, wt.spv_load / 1000000, wt.rebuild / 1000000,
                 wt.compile_spv / 1000000, wt.compute_ctor / 1000000, wt.prepass / 1000000,
                 RebuildThreads(), prebuilt.size(), rs.bad_spv, Shader::SrtWalkerArenaUsed() >> 10);
        for (auto& [name, pre] : prebuilt) {
            if (pre.module) {
                instance.GetDevice().destroyShaderModule(pre.module);
            }
        }
        prebuilt.clear();
        if ((CacheRebuildMode() & 2) && !SimProfileActive() && !stamp_perms.empty()) {
            WriteSeedStamp();
        }
        if (vs_formats_dirty) {
            std::ofstream out{Storage::DataBase::Instance().SidecarPath("vs-formats.bin"),
                              std::ios::binary | std::ios::trunc};
            const u32 count = static_cast<u32>(vs_formats.size());
            out.write(reinterpret_cast<const char*>(&VsFormatsMagic), 4);
            out.write(reinterpret_cast<const char*>(&VsFormatsVersion), 4);
            out.write(reinterpret_cast<const char*>(&count), 4);
            for (const auto& [key, codes] : vs_formats) {
                const u8 n = static_cast<u8>(std::min<size_t>(codes.size() / VsFormatsRecord, 255));
                out.write(reinterpret_cast<const char*>(&key), 8);
                out.write(reinterpret_cast<const char*>(&n), 1);
                out.write(reinterpret_cast<const char*>(codes.data()), size_t{n} * VsFormatsRecord);
            }
            LOG_INFO(Render, "CacheRebuild: vs-formats.bin written, {} entries", count);
        }
    }
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    PrecompileFromExtract();

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
