// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/prepared_specialization.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/prepared_graphics_state.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
union Regs;
}

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

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
        // Immutable and independently owned: module-vector growth/replacement
        // cannot invalidate metadata retained by a pipeline or a current draw.
        std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    // MRU guards only; guest user data and descriptors still refresh before each probe.
    // The secondary is allocated only for the opt-in two-entry experiment.
    std::unique_ptr<Shader::PreparedSpecialization> prepared;
    std::unique_ptr<Shader::PreparedSpecialization> prepared_secondary;
    // Bumped by every module-list or module-handle mutation. Prepared graphics pipeline
    // entries record it, since a new/inserted permutation can change first-match selection.
    u64 generation{};
    // SHADPS4_PRECOMPILE: built before the game dispatched it; first live use counted once.
    bool precompiled{};
    bool precompiled_seen{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void InvalidatePrepared() {
        ++generation;
        if (prepared) {
            prepared->valid = false;
        }
        if (prepared_secondary) {
            prepared_secondary->valid = false;
        }
    }

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        InvalidatePrepared();
        auto fetch = spec.fetch_shader_data
                         ? std::make_shared<const Shader::Gcn::FetchShaderData>(*spec.fetch_shader_data)
                         : nullptr;
        modules.emplace_back(module, std::move(spec), std::move(fetch));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        InvalidatePrepared();
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        auto fetch = spec.fetch_shader_data
                         ? std::make_shared<const Shader::Gcn::FetchShaderData>(*spec.fetch_shader_data)
                         : nullptr;
        modules[perm_idx] = {module, std::move(spec), std::move(fetch)};
    }
};

/// Architecture A (notes/NATIVE-RENDERER.md sec. 9): caller-owned scratch of one graphics pipeline
/// lookup over a private register file (PipelineCache::LookupGraphics / LookupOrBuildGraphics /
/// CaptureFront). One per worker thread, reused across draws (its vectors keep their capacity).
/// After a successful lookup it holds everything the front copies into a draw record today
/// (liverpool.cpp WriteStage): per logical stage the SHARED Shader::Info pointer (the back reads
/// only its immutable part), pgm_base, the user data words and the flattened user-data buffer,
/// exactly as the front's lookup would have left them in that Info for the same registers; plus
/// the key, the per-stage runtime infos and modules of the selection and the pipeline's fetch
/// shader. Valid until the next lookup on the same scratch.
struct GraphicsLookupScratch {
    struct Stage {
        const Shader::Info* info{}; ///< Program::info (shared, immutable part); null = unbound
        VAddr pgm_base{};
        std::vector<u32> user_data; ///< Shader::ShaderParams::NumShaderUserData words
        std::vector<u32> flattened_ud_buf;
        vk::ShaderModule module{};
    };
    /// Why LookupGraphics returned null (the read-only path never compiles).
    enum class Miss : u8 {
        None,
        UnsupportedStages,   ///< not the plain VS(+PS) configuration: exclusive path only
        NoProgram,           ///< a bound stage's program hash is not in program_cache (would compile)
        PrecompiledFirstUse, ///< SHADPS4_PRECOMPILE first live use (the front path writes state)
        NoPermutation,       ///< no module matches the specialization (would compile)
        NoPipeline,          ///< the key has no pipeline (would compile)
        PipelinePending,     ///< SHADPS4_ASYNC_PIPELINES: compiling; the front returns null too
    };

    const GraphicsPipeline* pipeline{};
    GraphicsPipelineKey key{};
    std::array<Stage, MaxShaderStages> stages{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    const Shader::Gcn::FetchShaderData* fetch_shader{}; ///< = pipeline->GetFetchShader()
    Miss miss{};

    /// Private copies of the shared Infos this scratch has resolved, keyed by the shared Info's
    /// address (Programs are never destroyed, so an address names one program for the run). The
    /// immutable half of a shared Info never changes after its program is published (see the
    /// thread-safety note in vk_pipeline_cache.cpp), so a copy is refreshed only when evicted;
    /// the per-draw half (pgm_base, user_data, flattened_ud_buf) is rewritten in the copy on every
    /// lookup, which is what lets the read-only path run the unmodified StageSpecialization /
    /// fetch-shader / SRT-walker code (all of which take a Shader::Info) without touching the
    /// shared object.
    struct InfoView {
        const Shader::Info* source{};
        u64 last_use{};
        Shader::Info info{};
    };
    static constexpr size_t NumViews = 16;
    std::array<InfoView, NumViews> views{};
    u64 view_clock{};

    Shader::Info& View(const Shader::Info& shared);
    /// SHADPS4_LEAN worker memo: when set, ResolveStageReadOnly also captures per bound stage the
    /// PreparedSpecialization guard of the selected permutation and the private view it used.
    bool capture_guards{};
    std::array<Shader::PreparedSpecialization, MaxShaderStages> guards{};
    std::array<const Shader::Info*, MaxShaderStages> guard_views{};
    void Reset();

    /// SHADPS4_LEAN worker (architecture-a sec. 16): the guest reads done OUTSIDE the cache's lock,
    /// per logical stage - the shader params (GetParams reads the shader header) and the flattened
    /// user data (the worker's checked SRT walk). When `prewalked`, ResolveStageReadOnly uses them
    /// and reads no guest memory under the lock. Not touched by Reset (set before each lookup).
    bool prewalked{};
    std::array<std::optional<Shader::ShaderParams>, MaxShaderStages> pre_params{};
    std::array<std::vector<u32>, MaxShaderStages> pre_flat{};
    /// The program was not in program_cache at the prewalk (no flat buffer was built): a miss.
    std::array<bool, MaxShaderStages> pre_missing{};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool);
    ~PipelineCache();

    void WarmUp();
    // SHADPS4_PRECOMPILE (vk_pipeline_precompile.cpp): run by WarmUp on the main thread.
    void PrecompileFromExtract();
    // SHADPS4_CACHE_REBUILD (2026-10-16, vk_pipeline_precompile.cpp): a cached stage's SPIR-V made
    // again from the game's own code (VideoCore::BuildShaderIndex) and its cached meta (user data,
    // flattened sharps, fetch data, runtime info, start bindings). Empty if the code is not found.
    // WarmUp only; the synthetic flat / fetch inputs are thread_local, so boot threads may run it
    // with their own `thread_pools`. `same_layout`: the new translation binds exactly as the
    // cached Info (SameResourceFields). `expect_identity`: the seed stamp's (crc32, code length)
    // for this hash; a different game file sets `identity_mismatch` and returns empty.
    // `vs_conv`: per fetch attribute a number-conversion code (0 = the spec's number class, 1
    // Uscaled, 2 Sscaled, 3 SnormNz 8-bit, 4 SnormNz 16-bit, 5 Unorm 32-bit): the cached spec
    // keeps only the class, the conversion comes from the vs-formats table (ANNOTATE).
    std::vector<u32> RebuildStageSpv(const Shader::Info& cached,
                                     const std::optional<Shader::Gcn::FetchShaderData>& fetch,
                                     const Shader::StageSpecialization& spec,
                                     bool* same_layout = nullptr,
                                     const std::vector<u8>* vs_conv = nullptr,
                                     const std::pair<u32, u32>* expect_identity = nullptr,
                                     bool* identity_mismatch = nullptr,
                                     Shader::Pools* thread_pools = nullptr);
    // SHADPS4_CACHE_REBUILD ANNOTATE (bit 8, with VERIFY): the conversion codes that make a cached
    // vertex shader's rebuild word-equal, or empty. vs_formats: (hash, perm) -> codes, read from /
    // written to the cache folder's vs-formats.bin.
    std::vector<u8> AnnotateVsConversions(const Shader::Info& cached,
                                          const std::optional<Shader::Gcn::FetchShaderData>& fetch,
                                          const Shader::StageSpecialization& spec,
                                          const std::vector<u32>& cached_spv);
    std::unordered_map<u64, std::vector<u8>> vs_formats;
    bool vs_formats_dirty{};
    struct RebuildStats {
        u32 equal[8]{}, differ[8]{}, no_code[8]{}, rebuilt[8]{}, reused[8]{}, layout_mismatch[8]{};
        u32 annotated[8]{}, annotate_failed[8]{};
        u32 not_stamped[8]{}, identity_mismatch[8]{}, kind_refused[8]{}, prebuilt[8]{};
        u32 bad_spv{};
    };
    RebuildStats rebuild_stats{};
    // SHADPS4_CACHE_REBUILD seed stamp (seed-stamp.bin, written by a =2 VERIFY run; study sec. 5
    // findings 1-4): =1 rebuilds only for the game serial / version and emulator build the seed was
    // verified with; graphics only on the same GPU profile + dynamic vertex input, else compute
    // only; per shader only when the game file's crc32 + length match the stamp.
    struct SeedStamp {
        bool compute{};
        bool graphics{};
        std::unordered_map<u64, std::pair<u32, u32>> shaders; // hash -> (crc32, code length)
        std::unordered_set<u64> perms; // (hash << 8) | perm of every permutation verified equal
    };
    SeedStamp seed_stamp{};
    // seed-graphics-keys.txt (2026-10-19): the g_ key names a seed shipped with. Unless the stamp
    // allows graphics on this GPU, these keys are not preloaded: once this GPU has compiled its own
    // permutations, which reuse the seed's index-based meta and .spv names, a seed key would pair
    // the seed machine's pipeline state with this GPU's modules.
    std::unordered_set<u64> seed_graphics_keys;
    u32 seed_graphics_keys_skipped{};
    u32 preload_refused{};
    void ReadSeedGraphicsKeys();
    void ReadSeedStamp();
    void WriteSeedStamp() const;
    // SHADPS4_CACHE_REBUILD_THREADS: every meta in the folder without a .spv whose rebuild the
    // stamp allows, translated on `num_threads` (own Pools each) into prebuilt_spv.
    void PrebuildMissingStages(u32 num_threads);
    // =2: per hash the permutations rebuilt word-equal (the stamp's content; Opus review B1: the
    // stamp lists permutations, so a differing one is refused by the code, not only by packaging).
    std::unordered_map<u64, std::vector<u8>> stamp_perms;
    // =2: programs with at least one permutation that differs: left out of the stamp entirely, so a
    // seeded boot compiles the whole program live (s1339: a VS program with rebuilt and live
    // permutations mixed gave VUID-VkGraphicsPipelineCreateInfo-layout-07990 at the next boot).
    std::unordered_set<u64> stamp_differ;
    // SHADPS4_CACHE_REBUILD_THREADS=N: stages translated on N boot threads before the serial
    // preload, by spv file name ("{hash:#018x}_{perm}"): the SPIR-V is saved and its module made
    // as it arrives (main thread), only the module is kept (review E: no 0.7 GB of SPIR-V held);
    // LoadPipelineStage takes it from here. same_layout false: refused, not translated again.
    struct PrebuiltStage {
        vk::ShaderModule module{};
        bool same_layout{};
    };
    std::unordered_map<std::string, PrebuiltStage> prebuilt;
    // WarmUp time split (SHADPS4_CACHE_REBUILD set): ns spent in each part on the main thread.
    struct WarmUpTimes {
        u64 meta_load{}, spv_load{}, rebuild{}, compile_spv{}, compute_ctor{}, prepass{};
    };
    WarmUpTimes warmup_times{};
    // SHADPS4_PRECOMPILE, Fable finding 1: a precompiled program whose first live dispatch matched
    // none of its guessed permutations leaves program_cache with its compute pipelines; both are
    // parked (a pool compile of such a pipeline may still be running), never destroyed early.
    void RetirePrecompiled(u64 hash);
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    // The second half of LoadComputePipeline (WarmUp runs it later for precompiled metas).
    bool BuildComputePipeline(const ComputePipelineKey& key,
                              ComputePipeline::SerializationSupport& sdata,
                              std::vector<u8>&& meta_blob);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    /// Front (command-processor) thread: the lookup for liverpool->regs, writing the member scratch
    /// and each bound stage's per-draw fields in the shared Infos (what WriteStage reads). Takes the
    /// cache's exclusive lock for its duration (= GetGraphicsPipelineLocked under AcquireExclusive).
    const GraphicsPipeline* GetGraphicsPipeline();

    // ---- Architecture A: lookups from any thread over an explicit register file --------------
    // Locking (one std::shared_mutex, `graphics_mutex`, non-recursive): LookupGraphics holds it
    // SHARED for the whole lookup; GetGraphicsPipeline, GetComputePipeline, LookupOrBuildGraphics'
    // miss path, PrepareNativeVariant / PrepareNativeColour and ReplaceShader's mutation hold it
    // EXCLUSIVE. Every *Locked member assumes the caller holds the exclusive lock and takes none.

    /// Any thread, concurrently with other LookupGraphics calls: computes the key from `regs`
    /// (exactly what the front computes from liverpool->regs for a plain VS(+PS) draw, every
    /// early-out included), looks the pipeline up and fills `out` with each bound stage's per-draw
    /// data as WriteStage would read it after the front's lookup. Returns null and sets out.miss
    /// on a miss (never compiles, never writes member state or any shared Shader::Info field) and
    /// whenever the front's lookup would return null. Configurations other than
    /// stage_enable == Vs are reported as Miss::UnsupportedStages (LookupOrBuildGraphics handles
    /// them on the exclusive path).
    /// SHADPS4_LEAN: the program's shared Info for a shader hash (short shared lock); null when
    /// absent or a precompiled program's first use (the lookup would miss).
    const Shader::Info* ProgramInfoFor(u64 hash) const;
    const GraphicsPipeline* LookupGraphics(const AmdGpu::Regs& regs,
                                           GraphicsLookupScratch& out) const;

    /// Any thread: LookupGraphics, and on a miss the exclusive path: re-runs the front's complete
    /// lookup over `regs` (prepared memo, async queueing, program / permutation / pipeline
    /// compiles, all as GetGraphicsPipeline would) and captures its result into `out`. NOTE: the
    /// exclusive path leaves the shared Infos holding THIS draw's per-draw fields, as the front's
    /// own lookup would; a front that reads them without holding AcquireExclusive() across its
    /// lookup + WriteStage can observe them.
    const GraphicsPipeline* LookupOrBuildGraphics(const AmdGpu::Regs& regs,
                                                  GraphicsLookupScratch& out);

    /// SHADPS4_LEAN=2 workers: refresh the shared Infos with THIS draw (GraphicsLookupLocked over
    /// `regs`) and decide the native / colour variant, under one exclusive hold.
    void PrepareNativeVariantRefreshed(const GraphicsPipeline* base, const AmdGpu::Regs& regs);
    void PrepareNativeColourRefreshed(const GraphicsPipeline* base,
                                      const AmdGpu::CpSplit::VertexInputsDerived& vi,
                                      const AmdGpu::Regs& regs);
    /// The exclusive lock; on the back thread (t_lock_service set) a try-lock loop that runs
    /// guest commands while it waits (a worker faulting under the lock needs them).
    [[nodiscard]] std::unique_lock<std::shared_mutex> LockExclusiveServicing() const;
    /// Bumped whenever prepared pipelines may name erased pipelines (read racily by worker memos).
    [[nodiscard]] u64 PreparedEpoch() const noexcept {
        return prepared_pipeline_epoch;
    }
    /// The exclusive guard; hold it across GetGraphicsPipelineLocked + WriteStage on the front
    /// once workers may take LookupOrBuildGraphics' build path concurrently.
    [[nodiscard]] std::unique_lock<std::shared_mutex> AcquireExclusive() const;
    /// GetGraphicsPipeline without taking the lock: the caller holds AcquireExclusive().
    const GraphicsPipeline* GetGraphicsPipelineLocked();

    /// Front thread, right after its lookup returned `pipeline` (same exclusive guard, or before
    /// any other lookup): copies what the lookup left in the member scratch and the shared Infos
    /// (per stage: pgm_base, user data words, flat buffer words) into `out`.
    void CaptureFront(GraphicsLookupScratch& out, const GraphicsPipeline* pipeline) const;

    /// Verify mode, front thread, right after GetGraphicsPipeline returned `front_result` for the
    /// same registers a worker's LookupGraphics used (nothing else may look up in between):
    /// compares `s` FIELD BY FIELD with what the front's lookup left behind: pipeline pointer,
    /// key, and per logical stage the Info pointer, pgm_base, every user data word, every
    /// flattened word, the module and the runtime info fields. `why` (optional) names the first
    /// difference.
    bool LookupMatchesFront(const GraphicsLookupScratch& s, const GraphicsPipeline* front_result,
                            const char** why = nullptr) const;

    /// SHADPS4_NATIVE_PASS (front / lookup thread, right after GetGraphicsPipeline returned
    /// `base` for the current registers): decides once whether `base` can draw native batches
    /// and, if so, builds its native variant (vertex shader re-emitted to read per-draw records,
    /// layout + one records SSBO) and publishes it in base->native_variant.
    void PrepareNativeVariant(const GraphicsPipeline* base);
    /// Any thread: the same over `regs` (the registers `base` was looked up with). Exclusive lock.
    void PrepareNativeVariant(const GraphicsPipeline* base, const AmdGpu::Regs& regs);
    /// The same without taking the lock: the caller holds AcquireExclusive() (a front that holds
    /// the guard across its lookup + WriteStage must call this, not the locking variant).
    void PrepareNativeVariantLocked(const GraphicsPipeline* base, const AmdGpu::Regs& regs);

    /// SHADPS4_NATIVE_COLOUR (front / lookup thread, right after the front derived the draw's
    /// vertex inputs): decides once whether `base` (a textured pipeline) can draw native batches
    /// and, for each new vertex description, translates its native VS + PS here and queues the
    /// driver compile on the native compiler thread, which publishes it in base->native_colour.
    void PrepareNativeColour(const GraphicsPipeline* base,
                             const AmdGpu::CpSplit::VertexInputsDerived& vi);
    /// Any thread: the same over `regs` (the registers `base` was looked up with). Exclusive lock.
    void PrepareNativeColour(const GraphicsPipeline* base,
                             const AmdGpu::CpSplit::VertexInputsDerived& vi,
                             const AmdGpu::Regs& regs);
    /// The same without taking the lock: the caller holds AcquireExclusive().
    void PrepareNativeColourLocked(const GraphicsPipeline* base,
                                   const AmdGpu::CpSplit::VertexInputsDerived& vi,
                                   const AmdGpu::Regs& regs);

    /// Front thread (the current queue's cs registers). Exclusive lock.
    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::shared_ptr<const Shader::Gcn::FetchShaderData>, u64>;
    /// Caller holds the exclusive lock (internal callers do). Registers = liverpool->regs.
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});
    /// SHADPS4_DUMP_SHADER_HASH=<hex> (2026-10-14): writes that program's SPIR-V - as compiled
    /// or as loaded from the pipeline cache - to <user>/shader/dumps/one_<name>_<origin>.spv,
    /// once per permutation and origin, whatever the dump setting (the W4 water patch is built
    /// from it).
    static void DumpOneShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                              size_t perm_idx, const char* origin);
    /// <user>/shader/patch/<stage>_<hash>_src<fnv1a64 of `source`>.spv, if present (2026-10-14).
    static std::optional<std::vector<u32>> GetShaderPatchBySource(u64 hash, Shader::Stage stage,
                                                                  std::span<const u32> source);
    /// Any patch file for this program in <user>/shader/patch (by index or by source).
    static bool HasAnyShaderPatch(u64 hash, Shader::Stage stage);

    auto& GetProfile() const {
        return profile;
    }

private:
    // The front path over an explicit register file (member scratch + shared Infos; exclusive
    // lock held by the caller). GetGraphicsPipeline / PrepareNative* pass liverpool->regs.
    const GraphicsPipeline* GraphicsLookupLocked(const AmdGpu::Regs& regs);
    bool RefreshGraphicsKey(const AmdGpu::Regs& regs);
    bool RefreshGraphicsStages(const AmdGpu::Regs& regs);
    bool RefreshComputeKey();
    Result GetProgram(const AmdGpu::Regs& regs, Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);
    void CaptureFrontLocked(GraphicsLookupScratch& out, const GraphicsPipeline* pipeline) const;

    // Pure pieces of the key / runtime-info computation, shared by the front path and the
    // read-only path (no member state written; `key` is the caller's).
    static void GraphicsKeyFirstPass(const AmdGpu::Regs& regs, GraphicsPipelineKey& key,
                                     bool& db_enabled, bool& skip_cb_binding);
    void GraphicsKeySecondPass(const AmdGpu::Regs& regs, GraphicsPipelineKey& key,
                               bool db_enabled, bool skip_cb_binding, bool allow_log) const;
    void FillStageRuntimeInfo(Shader::RuntimeInfo& info, const AmdGpu::Regs& regs,
                              Shader::Stage stage, Shader::LogicalStage l_stage,
                              const Shader::Info* hull_info, const GraphicsPipelineKey& key,
                              bool dump) const;

    // The read-only path (shared lock held by the caller).
    enum class Resolve { Unbound, Bound, Miss };
    const GraphicsPipeline* LookupGraphicsShared(const AmdGpu::Regs& regs,
                                                 GraphicsLookupScratch& out) const;
    Resolve ResolveStageReadOnly(const AmdGpu::Regs& regs, Shader::Stage stage_in,
                                 Shader::LogicalStage stage_out,
                                 Shader::Backend::Bindings& binding, GraphicsLookupScratch& out,
                                 const Shader::Info*& view_out,
                                 const Program::Module*& module_out) const;

    // SHADPS4_PREPARED_PIPELINES: one successful original VS+PS selection, reused while its
    // exact register snapshot, both per-stage specialization guards, both program
    // generations and the table epoch still hold. Command-processor thread only.
    struct PreparedPipelineEntry {
        bool valid{};
        u64 epoch{};
        u64 last_use{}; // SHADPS4_PREPARED_LRU: prepared_clock at the last hit or seed
        PreparedGraphicsState state;
        Program* fs_program{};
        Program* vs_program{};
        u64 fs_generation{};
        u64 vs_generation{};
        // Their captured RuntimeInfo bytes/permutation also restore the cache outputs.
        Shader::PreparedSpecialization fs_guard;
        Shader::PreparedSpecialization vs_guard;
        const GraphicsPipeline* pipeline{};
        vk::ShaderModule fs_module;
        vk::ShaderModule vs_module;
        std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    };
    struct StageSelection {
        Program* program{};
        size_t permutation{};
        bool compiled{};
    };
    static constexpr size_t PreparedPipelineSets = 1024;
    static constexpr size_t PreparedPipelineMaxWays = 32;
    /// SHADPS4_PREPARED_WAYS=<4..32> (default 4): associativity of the prepared-pipeline table.
    static size_t PreparedPipelineWays();

    PreparedPipelineEntry* TryPreparedPipeline(const AmdGpu::Regs& regs,
                                               PreparedGraphicsState& state, bool& eligible);
    void SeedPreparedPipeline(const PreparedGraphicsState& state, const GraphicsPipeline* pipeline);
    void VerifyPreparedPipeline(PreparedPipelineEntry& entry,
                                const GraphicsPipeline* pipeline, bool is_new);
    void NoteStageSelection(Shader::LogicalStage l_stage, Program* program, size_t permutation,
                            bool compiled);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext) const;
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(const AmdGpu::Regs& regs, Shader::Stage stage,
                                                Shader::LogicalStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    // SHADPS4_ASYNC_PIPELINES (default off): a new graphics key is handed to one compiler thread
    // and the lookup returns null (the draw is skipped) until a later lookup publishes the
    // finished pipeline. graphics_pipelines keeps its single mutator: only the lookup thread
    // inserts (a null entry marks a pending key) and assigns. Compute pipelines stay synchronous.
    struct AsyncJob;
    struct AsyncCompiler;
    void QueueAsyncPipeline(u64 pipeline_hash);
    void PublishAsyncPipelines();
    void NoteCompiledPipeline(u64 hash, double translate_ms, double create_ms, u32 queued_ahead,
                              bool async);

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    // Architecture A: guards program_cache, Program::modules / prepared, graphics_pipelines,
    // compute_pipelines, the member lookup scratch (graphics_key, infos, modules, runtime_infos,
    // fetch_shader, stage_selections, prepared tables, stats) and the per-draw fields of every
    // shared Shader::Info. See the thread-safety note in vk_pipeline_cache.cpp.
    mutable std::shared_mutex graphics_mutex;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    std::vector<std::unique_ptr<Program>> retired_programs;          ///< RetirePrecompiled
    std::vector<std::unique_ptr<ComputePipeline>> retired_pipelines; ///< RetirePrecompiled
    // SHADPS4_PRECOMPILE, Opus review 2026-10-07 fix A. A retire leaves the guessed (marked)
    // metas of the program's higher permutation indices in the cache next to the live rebuild's.
    // Programs retired in any earlier launch (sidecar precompile-retired.txt, read by WarmUp):
    // their marked metas are not loaded. All other marked compute metas load after the unmarked
    // ones (deferred), so a live meta, never a guessed one, creates a program's shared Info.
    std::unordered_set<u64> precompile_retired;
    struct DeferredComputeLoad {
        ComputePipelineKey key{};
        ComputePipeline::SerializationSupport sdata{};
        std::vector<u8> meta;
    };
    std::vector<DeferredComputeLoad>* defer_marked_compute{}; ///< set only during WarmUp's pass
    u32 retired_metas_skipped{};
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::vector<std::unique_ptr<GraphicsPipeline>> native_variants; // SHADPS4_NATIVE_PASS
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    std::unique_ptr<AsyncCompiler> async_compiler; // started by the first async job
    struct NativeColourJob;
    struct NativeColourCompiler;
    std::chrono::steady_clock::time_point native_colour_last{};
    u32 native_colour_logged{};
public:
    /// l4e census: colour variant claims - 0 claimed, 1 refused with a reason after the claim,
    /// 2 vertex format not pulled, 3 records binding, 4 pushed to the compiler, 5 built, 6 built
    /// without push descriptors, 7 of those kept as set variants (l4g).
    static inline std::array<std::atomic<u64>, 8> g_nc_claims{};
private:
    u32 async_pending{};         // queued, not yet published; lookup thread only
    double lookup_translate_ms{}; // GetProgram's module creation time within this lookup
    // PipelineCompile / PipelineCompileSummary log lines (both modes). skipped_draws counts
    // lookups that returned null because the key's compile was pending.
    struct CompileStats {
        u64 n{};
        u64 skipped_draws{};
        double translate_ms_total{};
        double create_ms_total{};
        double max_create_ms{};
    } compile_stats{};

    // Allocated on first use when SHADPS4_PREPARED_PIPELINES=1. The epoch invalidates every
    // entry at once whenever a pipeline is erased or a module handle is replaced.
    std::vector<PreparedPipelineEntry> prepared_pipelines;
    // SHADPS4_PREPARED_TAGS=1: state hash per way, written at seeding, so a lookup only
    // touches the (large, ~1 KB apart) entries whose hash matches. A stale tag only costs the
    // full compare the lookup did before.
    std::vector<u64> prepared_tags;
    std::array<StageSelection, MaxShaderStages> stage_selections{};
    u64 prepared_pipeline_epoch{1};
    u32 prepared_pipeline_victim{};
    u64 prepared_clock{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // SHADPS4_NATIVE_COLOUR: last member, so it stops (and destroys the variants it owns)
    // before anything else of the cache goes.
    std::unique_ptr<NativeColourCompiler> native_colour_compiler;
};

} // namespace Vulkan
