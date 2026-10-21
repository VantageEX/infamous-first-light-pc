// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <memory>
#include <shared_mutex>
#include <variant>
#include <tsl/robin_map.h>
#include <unordered_set>
#include <xxhash.h>
#include "g4_qid1_owner.h"
#include "graphics_pipeline_reuse.h"
#include "video_core/amdgpu/scout.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/preload_identity.h"
#include "video_core/renderer_vulkan/last_stage_lookup.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
struct ComputeProgram;
}

namespace iss::g4 { struct FallbackPacket; }

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;
class PipelineCompiler;

/// Key of the texture binding memo (Rasterizer::BindTextures, Tuning::tex_memo) for binding
/// `index` of an image resource with this T#: what the image description depends on, without
/// padding bytes. Also computed by the scout thread for the stages it precomputes.
inline u64 TexMemoKey(const AmdGpu::Image& tsharp, u32 index,
                      const Shader::ImageResource& image_desc) noexcept {
    struct KeyData {
        AmdGpu::Image tsharp;
        u32 index;
        u32 flags;
    } key_data{
        .tsharp = tsharp,
        .index = index,
        .flags = (image_desc.is_depth ? 1u : 0u) | (image_desc.is_array ? 2u : 0u) |
                 (image_desc.is_written ? 4u : 0u) |
                 (static_cast<u32>(image_desc.mip_fallback_mode) << 8) |
                 (static_cast<u32>(image_desc.constant_mip_index) << 16),
    };
    static_assert(sizeof(KeyData) == sizeof(AmdGpu::Image) + 8);
    return XXH3_64bits(&key_data, sizeof(key_data)) | 1;
}

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    std::array<u32, Shader::NUM_USER_DATA_REGS> replay_user_data{};
    ModuleList modules{};
    LastStageLookup last_stage_lookup;

    /// Permutation memo (Tuning::spec_memo): hash of the resource descriptors, runtime info and
    /// start bindings of recent bindings of this program, and the permutation they selected.
    /// Written by the command processor only, also read by the scout thread: each entry is a
    /// sequence lock (odd while being written).
    struct MemoEntry {
        std::atomic<u32> seq{};
        std::atomic<u32> perm_idx{};
        std::atomic<u64> key{};
        std::atomic<bool> valid{};
    };
    static constexpr size_t NumMemoEntries = 16;
    std::array<MemoEntry, NumMemoEntries> memo{};
    std::atomic<u32> memo_next{}; ///< Published to the graphics scout thread.

    /// Command processor: remembers the permutation of a key (oldest entry replaced).
    void WriteMemo(u64 key, u32 perm_idx) noexcept {
        const u32 next = memo_next.load(std::memory_order_relaxed);
        auto& entry = memo[next];
        const u32 seq = entry.seq.load(std::memory_order_relaxed);
        entry.seq.store(seq + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        entry.key.store(key, std::memory_order_relaxed);
        entry.perm_idx.store(perm_idx, std::memory_order_relaxed);
        entry.valid.store(true, std::memory_order_relaxed);
        entry.seq.store(seq + 2, std::memory_order_release);
        memo_next.store((next + 1) % NumMemoEntries, std::memory_order_release);
    }

    /// Scout thread: the permutation remembered for the key, read consistently, most recent
    /// entries first.
    bool ReadMemo(u64 key, u32& perm_idx) const noexcept {
        const u32 next = memo_next.load(std::memory_order_acquire);
        for (size_t n = 0; n < NumMemoEntries; ++n) {
            // A stale next value only changes the order of the atomic memo probes.
            const auto& entry =
                memo[(NumMemoEntries + next - 1 - n) % NumMemoEntries];
            const u32 seq = entry.seq.load(std::memory_order_acquire);
            if ((seq & 1) != 0) {
                continue;
            }
            const bool valid = entry.valid.load(std::memory_order_relaxed);
            const u64 entry_key = entry.key.load(std::memory_order_relaxed);
            const u32 entry_perm = entry.perm_idx.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (entry.seq.load(std::memory_order_relaxed) != seq) {
                continue;
            }
            if (valid && entry_key == key) {
                perm_idx = entry_perm;
                return true;
            }
        }
        return false;
    }

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
};

class PipelineCache : public AmdGpu::ScoutClient {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache() override;

    /// Scout thread: the shader stage lookups of a draw, from the registers at that draw.
    void ScoutDraw(const AmdGpu::Regs& regs, AmdGpu::ScoutDraw& draw,
                   AmdGpu::ScoutArena& arena) override;
    /// ISS_SPEC_COMPUTE (default on with ISS_SPEC_TRANSLATE, =0 off): a native passes/compute
    /// recording thread met a dispatch with this compute program before the command processor: a new
    /// compute program is translated ahead by the speculative translator (any thread).
    void RequestComputeSpecTranslation(const AmdGpu::ComputeProgram& cs);

    void WarmUp();
    void Sync();
    // PC port: thread-safe driver-cache checkpoint; no draws are dropped.
    bool CheckpointDriverCache();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage, u64 expected_hash);
    PreloadIdentity preload_identity;
    std::unordered_map<u64, std::pair<Program*, size_t>> preload_stages;
    u64 preload_stage_hits{};
    bool LoadCachedPipelineStage(u64 hash, size_t stage);
    bool parallel_preload{};
    void QueuePreload(Pipeline* pipeline);
    void WaitPreload();
    /// A thread that needs a pipeline still waiting in the compiler queue takes it out and
    /// compiles it itself (true); false when a worker has it (or it is done).
    bool CompileNowIfQueued(const Pipeline* pipeline);
    /// ISS_ASYNC_SKIP_DRAWS: the pipeline of a skipped draw to the front of the compile queue.
    void PrioritizeCompile(const Pipeline* pipeline);

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});
    /// Linux host: compiles the graphics pipelines of this port's first-start job file
    /// (<user>/cache/<serial>-jobs.bin, port/tools/xpps) before the game starts; once per job
    /// file, driver and shader version (gpu/precompile_jobs.inl).
    void PrecompileJobs(const std::function<void(u32 done, u32 total)>& progress);
    // Conservative engine-owned snapshot. Does not retain submission-owned scout pointers.
    struct PreparedGraphicsState;
    std::shared_ptr<PreparedGraphicsState> CapturePreparedGraphics(const GraphicsPipeline* pipeline);
    const GraphicsPipeline* TryPreparedGraphics(const PreparedGraphicsState& state);

    const ComputePipeline* GetComputePipeline();
    const ComputePipeline* GetComputePipeline(iss::g4::CpOwner owner);
    // G-only replay of an ASC fallback. All CS, code, UD and flattened SRT inputs are
    // packet-owned; mismatch returns nullptr before recording any dispatch.
    const ComputePipeline* GetComputePipelineFromSnapshot(
        const iss::g4::FallbackPacket& packet);

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);
    Result GetProgram(iss::g4::CpOwner owner, Shader::HwStage hw_stage,
                      Shader::SwStage sw_stage, const Shader::ShaderParams& params,
                      Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

    /// The scout stage taken as is (scout_fast) for this stage info in the current draw, null
    /// when the stage was looked up by the command processor: the scout's precomputed resource
    /// lookups (Tuning::bind_prefetch) apply to the info's bindings.
    /// Fast draws: the scout entries taken belong to the draw whose lookup took them.
    [[nodiscard]] u32 NewPipelineCount() const {
        return num_new_pipelines.load(std::memory_order_relaxed);
    }
    void SetTakenScout(u32 slot, const Shader::Info* info,
                       const AmdGpu::ScoutStage* stage) noexcept {
        taken_scout_infos.fill(nullptr);
        taken_scout_stages.fill(nullptr);
        if (info && stage && slot < taken_scout_infos.size()) {
            taken_scout_infos[slot] = info;
            taken_scout_stages[slot] = stage;
        }
    }
    void AddTakenScout(u32 slot, const Shader::Info* info,
                       const AmdGpu::ScoutStage* stage) noexcept {
        if (info && stage && slot < taken_scout_infos.size()) {
            taken_scout_infos[slot] = info;
            taken_scout_stages[slot] = stage;
        }
    }
    const AmdGpu::ScoutStage* TakenScoutStage(const Shader::Info& info) const noexcept {
        for (size_t i = 0; i < taken_scout_infos.size(); ++i) {
            if (taken_scout_infos[i] == &info) {
                return taken_scout_stages[i];
            }
        }
        return nullptr;
    }

    /// The shader hash of a graphics stage program (what the stage lookup is keyed by), through
    /// the params memo. Command processor thread only.
    u64 GraphicsProgramHash(const AmdGpu::ShaderProgram& pgm);

private:
    inline static thread_local const iss::g4::FallbackPacket* active_compute_snapshot{};
    inline static thread_local const AmdGpu::ComputeProgram* active_snapshot_cs{};
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey(iss::g4::CpOwner owner);
    iss::pipeline_reuse::Stamp GraphicsReuseStamp() const;
    const GraphicsPipeline* TryReuseGraphicsState();
    void RememberGraphicsState(const GraphicsPipeline* pipeline);

    /// AmdGpu::GetParams of a stage program, through the params memo (shader_params_memo,
    /// command processor thread only).
    template <typename ProgramRegs>
    Shader::ShaderParams GetStageParams(const ProgramRegs& pgm, iss::g4::CpOwner owner);

    /// Tuning::scout_fast: the stage lookup the scout thread resolved for this draw, taken as
    /// is when its inputs are the command processor's (fills infos, modules, the key's stage
    /// hash and the bindings). False: GetProgram as usual.
    template <typename ProgramRegs>
    bool TakeScoutedStage(Shader::HwStage stage_in, Shader::SwStage stage_out,
                          const ProgramRegs& pgm, Shader::Backend::Bindings& binding);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding,
                                   iss::g4::CpOwner owner);
    /// CompileModule after the translation (done by another thread): emit, module, register.
    vk::ShaderModule CompileModuleIR(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                     const Shader::IR::Program& ir_program,
                                     const std::span<const u32>& code, size_t perm_idx,
                                     Shader::Backend::Bindings& binding, iss::g4::CpOwner owner);
    /// ISS_PARALLEL_TRANSLATE: the vertex stage of a draw whose two stages are new programs,
    /// translated by a helper thread while the command processor translates the pixel stage.
    struct PreTranslation;
    PreTranslation* pre_translation{};
    /// ISS_SPEC_TRANSLATE: new programs the scout meets are translated ahead by worker threads
    /// (the translation does not depend on the bindings, only the emission does).
    std::unique_ptr<class SpecTranslator> spec_translator;
    /// ISS_ASYNC_SKIP_DRAWS: a stage of the current draw is still being translated by the
    /// speculative translator (the draw is skipped; RefreshGraphicsStages).
    bool async_translation_pending{};
    void RequestSpecTranslation(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                                const Shader::ShaderParams& params,
                                const Shader::RuntimeInfo& runtime_info);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage,
                                                iss::g4::CpOwner owner);
    /// The part of the runtime info that only depends on the registers (any thread). Returns
    /// false for the stages that read more (tessellation).
    bool BuildRuntimeInfoFrom(Shader::RuntimeInfo& info, const AmdGpu::Regs& regs,
                              Shader::HwStage stage, Shader::SwStage l_stage,
                              const std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers>&
                                  color_buffers,
                              const DrawIndirectParams& indirect_params) const;
    /// The color targets of the graphics key, from the registers.
    static void FillColorBuffers(
        const AmdGpu::Regs& regs,
        std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers>& color_buffers);
    /// The program of a shader, null when not compiled yet (any thread).
    Program* FindProgram(u64 hash);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    /// Hands a pipeline created with a deferred compilation to the worker threads, then waits
    /// for it at most async_pipeline_wait_ms. Callers skip their work while it is not ready.
    void CompileAsync(Pipeline* pipeline);

private:
    void SaveDriverCache();
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    std::filesystem::path driver_cache_path;
    bool driver_cache_saved{};
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    /// Held exclusively while the command processor adds a program, shared by the scout
    /// thread while it looks one up (the command processor reads without it: it is the only
    /// writer).
    std::shared_mutex program_cache_mutex;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    /// Shader params memo (shader_params_memo): the code size and hash found for a shader
    /// address, reused while the upload epoch and the emulator write generation stay the same.
    /// Until then there was no new submission or guest wait, so the guest cannot have written
    /// other code at that address for the commands being processed. Reading the code start and
    /// its binary info at the end of the code missed the caches twice per stage of every draw.
    struct ParamsMemo {
        VAddr address{};
        u64 epoch{};
        u64 host_write_gen{};
        u64 hash{};
        u32 num_dwords{};
    };
    static constexpr u32 NumParamsMemoBits = 6;
    std::array<ParamsMemo, 1U << NumParamsMemoBits> params_memos{};
    /// Pipeline key memo (pipeline_key_memo): the key and pipeline of the last draw.
    GraphicsPipelineKey last_graphics_key{};
    GraphicsPipeline* last_graphics_pipeline{};
    struct GraphicsReuseState {
        struct Stage {
            iss::pipeline_reuse::StageIdentity identity;
            std::array<std::byte, offsetof(AmdGpu::ShaderProgram, user_data)> registers{};
        };
        bool valid{};
        const GraphicsPipeline* pipeline{};
        iss::pipeline_reuse::Stamp stamp{};
        std::array<Stage, 2> stages{};
    } graphics_reuse;
    u64 graphics_reuse_checks{}, graphics_reuse_hits{}, graphics_reuse_verified{};
    std::unordered_set<u64> xray_dumped; ///< Shaders whose code the frame x-ray wrote.
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    /// The stages of the current draw taken from the scout (TakeScoutedStage), by stage.
    std::array<const Shader::Info*, MaxShaderStages> taken_scout_infos{};
    std::array<const AmdGpu::ScoutStage*, MaxShaderStages> taken_scout_stages{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    std::atomic<u32> num_new_pipelines{}; // new pipelines added by both owners

    // The preload/scout-facing fields above remain the graphics partition. ASC never reads or
    // mutates a graphics Program::info, permutation vector, or lookup scratch. In serial mode
    // every legacy call continues to select the graphics partition.
    struct ComputeLookupContext {
        Shader::Pools pools;
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
        std::array<ParamsMemo, 1U << NumParamsMemoBits> params_memos{};
        std::array<const Shader::Info*, MaxShaderStages> infos{};
        std::array<vk::ShaderModule, MaxShaderStages> modules{};
        Shader::Gcn::FetchShaderData* fetch_shader{};
        ComputePipelineKey compute_key{};
        std::array<Shader::PsColorBuffer, Shader::MaxColorBuffers> color_buffers{};
        DrawIndirectParams indirect_params{};
    } asc_lookup;
    tsl::robin_map<size_t, std::unique_ptr<Program>> asc_program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> asc_compute_pipelines;
    std::mutex module_related_pipelines_mutex;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    std::unique_ptr<PipelineCompiler> compiler;

    /// ISS_AOT_COMPUTE=<index> (scripts/aot_index.py): every compute shader of the game files
    /// translated and compiled ahead of time by low priority threads; the command processor
    /// only adopts the finished programs and pipelines (DrainAot).
    std::unique_ptr<struct AotCompute> aot;
    void StartAotCompute(const char* index_path);
    /// ISS_AOT_RECIPES: the SPIR-V of the recipe pack's stages, made from the game files.
    void PrepareRecipeStages();
    void DrainAot();
};

} // namespace Vulkan
