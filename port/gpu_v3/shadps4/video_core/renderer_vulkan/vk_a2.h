// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

/**
 * Architecture A, step A2 (notes/NATIVE-RENDERER.md, notes/architecture-a-2026-10-10.md sec. 6):
 * worker threads translate the NATIVE STRETCHES of the game's top-level command buffers (DCBs)
 * into plain command lists (StretchCmds); the back runs everything else on the old path and
 * replays the lists in stream order (one Scheduler::Record closure per stretch, inside the
 * render pass it began), so the GPU sees the original order.
 *
 * Split of work:
 *  - Liverpool (liverpool.cpp): the DCB table, the worker threads, the record walk with a private
 *    register mirror per worker, the stretch boundaries, the back-side execution of a DCB.
 *  - StretchTranslator (vk_a2.cpp): per draw, decides whether the worker can take it and appends
 *    it to the open stretch's command list. Runs on a WORKER thread: it must not touch the buffer
 *    cache's mutable state, the texture cache, the scheduler (Record / Rc / ticks / sinks), the
 *    stream buffer or any Rasterizer member. Allowed: immutable pipeline data and atomics
 *    (native_variant, native_colour slots), the job-record rings (NativeRec, read-only,
 *    stamp-checked), the buffer cache's lock-free lookup (BufferCache::A2LookupBuffer), the
 *    worker's own arena, its own DynamicState, and the master semaphore's known GPU tick
 *    (read-only, for arena reuse).
 *  - Colour draws (v2, "deferred material resolve"): the worker takes a native COLOUR draw
 *    (bindless textures + pulled vertices) with every check that depends only on registers, the
 *    draw record's stage blocks, the job record and immutable pipeline data; it records the
 *    draw's raw T# / S# key as a MATERIAL of the stretch (at most MaxMaterials distinct ones) and
 *    a SlotWrite per draw. The bindless slots need the texture cache, so the BACK resolves each
 *    material once at publish (ReadStages of the representative Draw record at
 *    MaterialRef::rec_pos, BindlessResolve with that fragment stage, the feedback-loop / bound
 *    target / force_general checks), refuses the whole stretch if any fails, fills
 *    StretchCmds::mat_slots and bindless_set, and only then records the Replay closure, which
 *    writes every draw's slot words into the GPU ring record before the draws.
 */
namespace Vulkan {
class GraphicsPipeline;
class Instance;
} // namespace Vulkan
namespace VideoCore {
class BufferCache;
class Buffer;
} // namespace VideoCore
namespace AmdGpu::CpSplit {
struct VertexInputsPayload;
}

namespace Vulkan::A2 {

/// SHADPS4_A2: 0 off (default); 1 workers record native stretches and the back publishes them;
/// 2 verify: workers translate, the back discards every stretch and runs the old path (counts).
int Mode();
/// SHADPS4_A2_WORKERS (default 3, 1-8).
u32 NumWorkers();

/// The draw as the worker loop decoded it from the Draw record (pointers into the CpSplit ring:
/// valid while the worker runs on this DCB).
struct DrawIn {
    const GraphicsPipeline* pipeline{}; ///< the front's lookup (base pipeline)
    bool is_indexed{};
    bool is_indirect{};
    u32 index_offset{};
    const std::array<Shader::BoundStage, Shader::MaxStageTypes>* stages{}; ///< ReadStages output
    const AmdGpu::CpSplit::VertexInputsPayload* vi{}; ///< FRONT_DERIVE block, may be null
    u32 job_tag{}, job_idx{}; ///< the draw's own job record (tag 0: none)
    /// The draw's DRAW_INDEX_2 packet address (the Draw record's cmd_address): the job record's
    /// `cmd` must equal it (NativeRingPrepare's `cmd0 != draw_job_cmd` check). Null: the check
    /// is skipped (stamp + fields only) and counted under RefuseNoCmd once per session.
    const void* job_cmd{};
    u64 seq{};                ///< the Draw record's CpSplit seq
    u64 rec_pos{}; ///< unmasked ring position of the Draw record (a material's representative)
    u64 arg_address{}; ///< q2: an indirect draw's args (SET_BASE + data offset)
};

/// Colour draws: distinct materials (raw T# / S# keys) a stretch may carry; the back resolves
/// each once at publish.
constexpr u32 MaxMaterials = 8;

/// One material of a stretch: the draw's base pipeline (DrawIn::pipeline) and the Draw record the
/// back re-reads (ReadStages -> BindlessResolve with its fragment stage) to get the slots.
struct StretchCmds;
/// l4g: the back, at publish: a descriptor set per non-push sub-batch. False: none (old path).
bool CommitSets(StretchCmds& cmds);

struct MaterialRef {
    const GraphicsPipeline* base{};
    u64 rec_pos{};
    u64 h1{}, h2{}; ///< l4d: the worker's material key hash (NativeRec::ImageKeyHash)
    u32 kn{};       ///< l4d: its length in words
};

/// Per-worker memory the GPU reads (tables, indirect commands, counts): a host-visible coherent
/// ring. Regions are handed out in order; a region is reusable once the GPU passed the tick of the
/// submission batch its stretch was PUBLISHED into (the back stamps Stretch::tick at publish).
struct Arena {
    vk::Buffer buffer{};
    u8* mapped{};
    u64 device_address{};
    u64 size{};
    u64 head{}; ///< next free byte (unmasked)
    /// Reuse fence per ended stretch: [unmasked start, unmasked end, tick] in order, oldest
    /// first. tick 0 = not published yet = not free. The tick object is shared with the Stretch
    /// (the back destroys the Stretch when the DCB is done; the fence keeps the tick alive).
    struct Fence {
        u64 start{};
        u64 end{};
        std::shared_ptr<std::atomic<u64>> tick;
    };
    std::vector<Fence> fences;
    /// SHADPS4_LEAN_PERDRAW bit 2 or 8: index of the first live entry of `fences` (popped in O(1);
    /// the dead prefix is erased in bulk). Stays 0 with those bits off.
    size_t fence_head{};
};

/// One sub-batch of a stretch: everything one drawIndexedIndirectCount needs, as values. The
/// first sub-batch of a stretch carries the FULL state (pipeline, push constants, descriptors,
/// vertex input, index / vertex buffers, every dynamic state); later ones only what changed
/// (the emit_* flags and the dynamic state's dirty bits say what Replay records).
struct SubBatch {
    bool emit_pipeline{}, emit_push{}, emit_descriptors{}, emit_vertex_input{}, emit_index{},
        emit_vertex_buffers{};
    /// A colour sub-batch (bindless variant): Replay binds StretchCmds::bindless_set at set 1
    /// with `layout` after the set-0 push, as the back's native colour batch does.
    bool bindless{};
    vk::Pipeline pipeline{};
    vk::PipelineLayout layout{};
    /// l4g: a non-push variant - the back commits `set` (set 0) from these writes at publish.
    const GraphicsPipeline* set_owner{};
    vk::DescriptorSet set{};
    Shader::PushData push{};
    /// Set-0 push-descriptor writes (storage buffers only), in binding order.
    struct Write {
        u32 binding{};
        vk::Buffer buffer{}; ///< VK_NULL_HANDLE: a null descriptor (unused clip planes)
        u64 offset{};
        u64 range{};
    };
    u32 num_writes{};
    std::array<Write, Shader::NUM_BUFFERS + 4> writes{};
    /// setVertexInputEXT arrays (the FRONT_DERIVE block's description part).
    u32 vi_count{};
    std::array<vk::VertexInputAttributeDescription2EXT, 16> vi_attrs{};
    std::array<vk::VertexInputBindingDescription2EXT, 16> vi_bindings{};
    vk::Buffer ib{};
    vk::IndexType ib_type{};
    u32 vb_n{};
    std::array<vk::Buffer, 16> vb{};
    std::array<vk::DeviceSize, 16> vb_off{};
    /// Dynamic state values; its dirty bits are what this sub-batch records (Commit).
    DynamicState dyn{};
    /// The indirect draw: commands and count in the arena buffer.
    vk::Buffer arena{};
    u64 cmd_off{};
    u64 count_off{};
    u32 n{}; ///< draws (maxDrawCount of the indirect count draw)
    /// q2: a guest indirect draw - drawIndexedIndirect(ind_buf, ind_off, 1, 20) instead of the
    /// count draw; the index buffer is bound at ib_off (the args' firstIndex is relative to it).
    vk::Buffer ind_buf{};
    u64 ind_off{};
    u64 ib_off{};
};

struct StretchCmds {
    std::vector<SubBatch> batches;
    /// Colour (bindless) sub-batches: the set-1 copy and the per-material slot words
    /// (Shader::g_bl_img_max image slots, then the sampler slots: the record words
    /// Shader::NativeRecordImgOffset .. +NativeBindlessSlots). Both set by the back at publish
    /// (BindlessResolve per material); unused when slot_writes is empty.
    vk::DescriptorSet bindless_set{}; ///< set by the back at publish
    std::array<std::array<u32, Shader::NativeBindlessSlots>, MaxMaterials>
        mat_slots{}; ///< set by the back
    /// One per colour draw: Replay writes mat_slots[mat] into job record (tag, idx) of the GPU
    /// ring (as Rasterizer::BindlessRingWrite) before it records the draws.
    struct SlotWrite {
        u32 tag{};
        u32 idx{};
        u32 mat{};
        u32* ring{}; ///< r9 (SHADPS4_NR_WORKER=1): the translator's own record ring (null: the job slot's)
    };
    std::vector<SlotWrite> slot_writes;
    /// SHADPS4_A2=2 direct checks, one entry per taken draw (the back compares at that draw):
    /// an inherited Guest slot's table words, and the clip planes the sub-batch binds.
    struct InhCheck {
        u64 seq{};
        VAddr addr{};
        u64 vsize{};
        u32 off{};
        u32 size{};
    };
    std::vector<InhCheck> inh_checks;
    struct ClipCheck {
        u64 seq{};
        u32 enable{};
        std::array<u32, 24> planes{};
    };
    std::vector<ClipCheck> clip_checks;
    struct FmtCheck {
        u64 seq{};
        VAddr addr{};
        u32 base{};
    };
    std::vector<FmtCheck> fmt_checks;
};

/// Records `cmds` into `cmdbuf` exactly as the translator's closures would have, in order. Called
/// by the recording lane inside the stretch's render pass; after it the recorder's tracked
/// pipeline / dynamic state / vertex input / descriptor state is unknown (the back invalidates).
void Replay(const StretchCmds& cmds, vk::CommandBuffer cmdbuf, const Instance& instance);
/// p2: Replay's two halves - the slot words into the job ring (host memory), and the commands
/// with `set1` as the bindless set.
void WriteSlots(const StretchCmds& cmds);
void RecordBatches(const StretchCmds& cmds, vk::DescriptorSet set1, vk::CommandBuffer cmdbuf,
                   const Instance& instance);
/// SHADPS4_A2_SEC=1 (sec. 20 v1): workers record each stretch into a secondary command buffer.
bool SecondaryMode();
/// The back's current bindless set index (mirrored by Rasterizer for the workers).
inline std::atomic<u32> g_bindless_cur{0};
/// Counters: 0 secondaries recorded, 1 skipped (no free pool), 2 skipped (l4g committed set),
/// 3 executed at publish, 4 closure (no secondary), 5 closure (set 1 moved), 6 closure
/// (inheritance differs from the pass), 7 pool resets.
inline std::array<std::atomic<u64>, 8> g_sec_c{};

/// One stretch's output: what the back publishes at the stretch's first draw.
struct Stretch {
    u64 first_seq{}, last_seq{}; ///< Draw record seqs of its first and last draw
    u32 draws{};
    std::shared_ptr<StretchCmds> cmds;
    u64 arena_end{};            ///< unmasked arena end of its regions (for the reuse fence)
    /// Set by the back at publish (store release): the batch tick the stretch was recorded into;
    /// verify mode stores 1 right away. 0 = unpublished (its arena regions are not free).
    std::shared_ptr<std::atomic<u64>> tick;
    /// BufferCache::StructGeneration() when the first lookup of the stretch ran: the back refuses
    /// the stretch (old path) if the generation moved before publish (a buffer was deleted or
    /// joined since the worker's lookups).
    u64 buf_gen{};
    /// Colour draws' materials in first-seen order (index = SlotWrite::mat / mat_slots row).
    std::vector<MaterialRef> materials; ///< empty: no colour draws
    /// Formatted Guest slots' (V# address, V# size), distinct, at most 16: the back syncs each
    /// with its own typed lookup (A2SyncTexel) before the stretch's pass.
    std::vector<std::pair<VAddr, u64>> texel;
    /// q9 census (SHADPS4_LANE_WHO=1): reads over a lane pool probe ([lo, hi)), stamped at publish.
    std::vector<std::pair<VAddr, VAddr>> probe_reads;
    /// q9b: g_lprobe_gen when the stretch began (a newer one at publish: every probe is stamped).
    u64 probe_gen{};
    bool probe_over{}; ///< more than 256 probed ranges: every probe is stamped
    /// q8 (SHADPS4_TEX_TABLE): materials the worker resolved from the texture table (bit m: its
    /// slot words are in cmds->mat_slots[m]), the table generation / bindless set they hold for,
    /// and their image / sampler counts.
    u32 mat_pre{};
    u64 tab_gen{};
    u32 tab_set{};
    /// q10: the kill sequence read before the first table material's lookups, and the (image id,
    /// epoch) of every image the stretch took from the table (ImgsCurrent at publish).
    u64 tab_kseq{};
    std::vector<std::pair<u32, u32>> tab_imgs;
    std::array<std::pair<u8, u8>, 8> pre_n{};
    /// p2 (SHADPS4_A2_SEC): the worker's secondary (null: the closure path), the bindless set
    /// index it binds, and its rendering inheritance.
    const GraphicsPipeline* rt_pipe{};
    vk::CommandBuffer secondary{};
    u32 bl_idx{};
    struct Inh {
        u32 n_color{};
        std::array<vk::Format, 8> color{};
        vk::Format depth{vk::Format::eUndefined};
        vk::Format stencil{vk::Format::eUndefined};
    } inh{};
};

/// Context a translator gets (set up once per worker thread).
struct WorkerContext {
    const Instance* instance{};
    Scheduler* scheduler{};               ///< only GetMasterSemaphore()->KnownGpuTick() is read
    VideoCore::BufferCache* buffer_cache{}; ///< only A2-safe lookups may be called
    Arena* arena{};
    u32 worker_index{};
};

/// Refusal reasons (StretchTranslator::refusals index). RefusalName() gives the log label.
enum Refusal : u32 {
    RefuseOff = 0,       ///< the translator is off (config gates, arena not coherent)
    RefuseNotIndexed,
    RefuseIndirect,
    RefuseNoVariant,     ///< no native shadow variant (yet)
    RefuseVariantKind,   ///< variant without ring table / with the PT=2 debug binding
    RefuseNoVi,          ///< no FRONT_DERIVE block, or more than 16 attributes
    RefuseNoJob,         ///< job_tag 0 (no job record)
    RefuseNoRing,        ///< the job slot has no ring / unions / GPU ring handle
    RefuseStamp,         ///< record stamp / cmd / gpu_ok mismatch
    RefuseLifetime,      ///< the job thread is more than a quarter ring ahead
    RefuseRecordFields,  ///< inst_known 0, io_ud != 0, index_count 0
    RefuseInheritedSlot, ///< a Guest V# sits in inherited registers (v1: not handled)
    RefuseUnion,         ///< the run union is not published / does not cover the draw
    RefuseSlotKind,      ///< a non-Guest binding, images / samplers, or enabled user clip planes
    RefuseGuestLookup,   ///< a Guest slot's union range is not in one cached buffer
    RefuseIndexLookup,   ///< the index range is not in one cached buffer
    RefuseVertexLookup,  ///< a vertex stream's base is not in a cached buffer
    RefuseRecordRegs,    ///< the record disagrees with the mirror (counts, index / vertex addr)
    RefuseInstanceOffset,///< firstInstance != 0
    RefuseVertexOffset,  ///< the rebased vertexOffset leaves [0, INT32_MAX] or no common quotient
    RefuseRestartIndex,  ///< primitive restart with an index other than -1
    RefuseArena,         ///< no free arena region (unpublished or GPU behind) - never waits
    RefuseStages,        ///< no vertex stage data
    RefuseLayout,        ///< the variant's binding numbering does not match the stages
    RefuseNoCmd,         ///< DrawIn::job_cmd null (counted once; the draw is not refused)
    // Colour draws (v2).
    RefuseColourGates,   ///< colour off / diag bits 1 or 4096 / variant not bindless-ring-pt
    RefuseColourStages,  ///< images or samplers outside the fragment stage, or no fragment stage
    RefuseColourImages,  ///< > g_bl_img_max / g_bl_smp_max, storage / atomic image, mip fallback DynamicIndex,
                         ///< null or invalid-format T#, key over 512 words
    RefuseColourKeyHash, ///< the job record's image count / key hash disagree with the stages
    RefuseColourVertex,  ///< instanced attribute, location >= 16, streams in more than one buffer,
                         ///< offset > 32 bits, or no stream at all
    RefuseColourStream,  ///< a stream's [base, base + size) is not inside its cached buffer
    RefuseMaterials,     ///< a 9th distinct material in the stretch
    NumRefusals          ///< 32 = the refusals array (a colour Guest V# with an inline cbuf,
                         ///< null base or zero size counts under RefuseSlotKind)
};
static_assert(NumRefusals <= 32);
const char* RefusalName(u32 reason);

/// Per-draw translation on a worker. One instance per worker thread.
class StretchTranslator {
public:
    explicit StretchTranslator(const WorkerContext& ctx);
    ~StretchTranslator();

    /// True if the worker can take this draw into the open stretch (opening one if none is open);
    /// it is appended to the stretch's command list then. False: the draw is the back's (old
    /// path); the caller ends the open stretch first (EndStretch) - a refused draw never changes
    /// the list. `regs` = the worker's private mirror at this draw.
    bool TakeDraw(const AmdGpu::Regs& regs, const DrawIn& draw);
    /// Ends the open stretch (flushes the pending sub-batch) and returns it (null if none open).
    std::unique_ptr<Stretch> EndStretch();
    /// Refusal counters by reason (logged by the caller).
    std::array<u64, 32> refusals{};
    /// Sub-reasons (census): 0-3 record fields (inst_known 0, io_ud, index_count 0, push_inh),
    /// 4-9 slot kind (shadow with images, clip planes, not Guest, written, formatted, colour
    /// inline / null / empty V#), 10-11 guest lookup (no buffer, address outside it), 12-15 stamp
    /// (stamp below idx + 1, above it, gpu_ok 0, cmd differs), 16-18 inherited slot refused (size
    /// 0 or >= 1 GB, no cached buffer holding it, offset over 32 bits), 19 draws taken with an
    /// inherited slot, 20 draws taken with user clip planes, 21 a 17th texel range refused,
    /// 22 draws taken with a formatted slot, 23 guest slots resolved by the opener's own V#.
    std::array<u64, 52> sub{}; ///< 48-49 (q53): colour vertex third VkBuffer refused / draws taken with a second one
    ///< 34-43 (q1): indirect without a variant by cause; 32-33 (m3): indirect draws with / without a variant ///< 24-27 (l4b): "no variant" by pipeline state; 28-31 (l4c) slots
    /// Draws taken and sub-batches opened so far (logged by the caller).
    u64 taken{}, sub_batches{};
    /// The reason of the last refusal (TakeDraw returned false).
    u32 last_refusal{};

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::A2
