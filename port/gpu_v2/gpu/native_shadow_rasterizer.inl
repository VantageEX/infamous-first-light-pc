// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_depth_run.inl.
//
// ISS_NATIVE_SHADOW=1 (native_shadow.h): the draws of an engine shadow batch, recorded from the
// batch instead of interpreting their packets.
//
// A draw's render state is everything the range's registers make of it: the registers before
// the range (context, primitive and program registers, hashed at the marker, render target
// addresses left out) and the values the range had set at that draw (Batch::states). The
// pipeline the translated path chose for a render state is learned the first time it is seen:
// the range is then interpreted, each draw of an unknown state taking the full path. Later
// ranges whose states are all known and of the audited shape are recorded here, each draw as
// the translated path records it (DrawRecord) with the bindings it would get:
// - the vertex and index ranges go through BufferCache::ObtainBuffer with the same sizes and
//   hazard checks as BindVertexBuffers/BindIndexBuffer;
// - the per instance records (SSBO0, small, without GPU data) are read from the current guest
//   bytes into one stream region, each draw bound to its own copy, as the translated path reads
//   them (stream copy or clean arena);
// - the fetch and SSBO descriptors must keep the format words seen when the state was learned.
// A pipeline change sets everything up as DrawRecord does; a state change of the same pipeline
// updates the dynamic state. When a setup fails after some draws, the command processor
// interprets the rest of the range.

namespace NativeShadow {
namespace ns = iss::native_shadow;
static_assert(offsetof(AmdGpu::Regs, index_base_address) / sizeof(u32) == 0xA1F9);
static_assert(offsetof(AmdGpu::Regs, draw_initiator) / sizeof(u32) == 0xA1FC);
static_assert(offsetof(AmdGpu::Regs, max_index_size) / sizeof(u32) == 0xA29E);
static_assert(offsetof(AmdGpu::Regs, num_indices) / sizeof(u32) == 0xC24C);
static_assert(offsetof(AmdGpu::Regs, num_instances) / sizeof(u32) == 0xC24D);
static_assert(offsetof(AmdGpu::Regs, index_buffer_type) / sizeof(u32) == 0xA29F);
static_assert(offsetof(AmdGpu::Regs, depth_render_override) / sizeof(u32) == 0xA003);
static_assert(offsetof(AmdGpu::Regs, depth_bounds_min) / sizeof(u32) == 0xA008);
struct Learned {
    const GraphicsPipeline* pipeline{};
    std::array<u32, 2> fetch_words{}, ssbo_words{}; // descriptor words 1 and 3
    bool ok{};
};
static std::unordered_map<u64, Learned>& Memo() {
    static std::unordered_map<u64, Learned> memo; // command processor thread only
    return memo;
}
struct DrawTotals { u64 native{}, interpreted{}, reference{}, ranges{}; };
static DrawTotals& ActualDraws() { static DrawTotals totals; return totals; }
static u32& LastRejectedShape() { static u32 state = ~0u; return state; }
// Only a synchronous CP suffix call installs this request. No borrowed batch
// storage reaches the recorder: RecordGroup copies its resolved Vulkan payload.
struct SuffixViewRequest { const ns::Batch* batch{}; u32 first{}; };
static thread_local SuffixViewRequest suffix_view_request;
static bool SuffixViewEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_SUFFIX_VIEW");
    return enabled;
}
static std::span<const ns::Draw> ExecutionDraws(const ns::Batch& batch) {
    const u32 first = suffix_view_request.batch == &batch ? suffix_view_request.first : 0;
    const std::span<const ns::Draw> draws(batch.draws);
    return first <= draws.size() ? draws.subspan(first) : draws.subspan(draws.size());
}
// Opt-in diagnosis of memo churn, sampled to keep normal runs unaffected.
struct MemoStats {
    u64 attempts{}, inserts{}, clears{}, no_pipeline{}, bad_shape{};
    std::array<u32, 0x3000> previous{}, changes{};
    bool have_previous{};
};
static bool MemoStatsEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_STATS");
    return enabled;
}
struct CostTrace {
    enum PhaseId { Lookup, Resolve, Copy, Setup, Record, Count };
    struct Totals { u64 calls{}, draws{}, ticks[Count]{}; };
    static bool Enabled() {
        static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_COST");
        return enabled;
    }
    static Totals& Stats() { static Totals stats; return stats; }
    bool enabled = Enabled();
    PhaseId phase = Lookup;
    u64 last = enabled ? __rdtsc() : 0;
    void Phase(PhaseId next) {
        if (!enabled) return;
        const u64 now = __rdtsc();
        Stats().ticks[phase] += now - last;
        last = now;
        phase = next;
    }
    ~CostTrace() {
        if (!enabled) return;
        Phase(phase);
        auto& s = Stats();
        if ((++s.calls & 16383) != 0) return;
        u64 total = 0;
        for (u64 ticks : s.ticks) total += ticks;
        std::fprintf(stderr, "NATIVE_SHADOW_COST calls=%llu draws=%llu ticks=%llu lookup=%.2f resolve=%.2f copy=%.2f setup=%.2f record=%.2f\n",
                     s.calls, s.draws, total, 100.0*s.ticks[Lookup]/total,
                     100.0*s.ticks[Resolve]/total, 100.0*s.ticks[Copy]/total,
                     100.0*s.ticks[Setup]/total, 100.0*s.ticks[Record]/total);
    }
};
static MemoStats& MemoCounters() { static MemoStats stats; return stats; }
static void DiagnoseMemo(const AmdGpu::Regs& regs, const ns::Batch& batch) {
    if (!MemoStatsEnabled()) return;
    auto& s = MemoCounters();
    if ((++s.attempts & 1023) == 1) {
        const auto sample = [&](u32 first, u32 count) {
            for (u32 reg = first; reg < first + count; ++reg) {
                if (reg >= ns::VsUserData2 && reg < ns::VsUserData2 + ns::DrawUserWords) continue;
                const u32 slot = reg >= 0xA000 ? reg - 0xA000 : 0x2800 + reg - 0x2C00;
                const u32 value = regs.reg_array[reg];
                if (s.have_previous && s.previous[slot] != value) ++s.changes[slot];
                s.previous[slot] = value;
            }
        };
        sample(0xA000, 0x400); sample(0xC240, 0x20);
        sample(0x2C08, 0x14); sample(0x2C48, 0x14);
        s.have_previous = true;
    }
    if ((s.attempts & 65535) != 1) return;
    std::fprintf(stderr, "NATIVE_SHADOW_MEMO attempts=%llu inserts=%llu clears=%llu no_pipeline=%llu bad_shape=%llu size=%zu regs=",
                 s.attempts, s.inserts, s.clears, s.no_pipeline, s.bad_shape, Memo().size());
    for (u32 reg : batch.state_regs) std::fprintf(stderr, "%x,", reg);
    auto changes = s.changes;
    std::fprintf(stderr, " changing=");
    for (u32 i = 0; i < 12; ++i) {
        auto it = std::max_element(changes.begin(), changes.end());
        if (*it == 0) break;
        const u32 slot = u32(it - changes.begin());
        std::fprintf(stderr, "%x:%u,", slot < 0x2800 ? 0xA000 + slot : 0x2C00 + slot - 0x2800, *it);
        *it = 0;
    }
    std::fprintf(stderr, "\n");
}
struct Resolved {
    const VideoCore::Buffer* vb{};
    u64 vb_offset{};
    const VideoCore::Buffer* ib{};
    u64 ib_offset{};
    VAddr ssbo{};
    u32 ssbo_size{}, region_offset{}, index_count{}, instances{}, instances_raw{};
    std::array<u32, ns::DrawUserWords> user{};
};
struct GroupDraw {
    vk::Buffer vb{}, ib{};
    vk::DeviceSize vb_offset{}, ib_offset{};
    u32 index_count{}, instances{}, first_instance{};
    vk::DeviceSize indirect_offset{};
};
static bool GroupEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_GROUP");
    return enabled;
}
static bool SplitEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_SPLIT");
    return enabled;
}
static bool RecordGroupEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_RECORD_GROUP");
    return enabled;
}
static bool IndirectGroupEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_INDIRECT");
    return enabled;
}
static void NoteGroup(u32 count, bool queued) {
    static u64 groups{}, draws{}, queued_groups{};
    ++groups; draws += count; queued_groups += queued;
    if ((groups & 16383) == 1)
        std::fprintf(stderr, "NATIVE_SHADOW_GROUP groups=%llu draws=%llu queued_groups=%llu\n",
                     groups, draws, queued_groups);
}
static void RecordGroup(Scheduler& scheduler, const std::vector<GroupDraw>& draws,
                        vk::Buffer indirect_buffer) {
    const auto record = [count = u32(draws.size()), indirect_buffer](vk::CommandBuffer cmd, const GroupDraw* data) {
        for (u32 i = 0; i < count; ++i) {
            const auto& d = data[i];
            cmd.bindVertexBuffers(0, 1, &d.vb, &d.vb_offset);
            cmd.bindIndexBuffer(d.ib, d.ib_offset, vk::IndexType::eUint16);
            u32 end = i + 1;
            if (indirect_buffer) {
                while (end < count && data[end].vb == d.vb && data[end].ib == d.ib &&
                       data[end].vb_offset == d.vb_offset && data[end].ib_offset == d.ib_offset) ++end;
            }
            if (end - i > 1) {
                // Exact geometry bindings: no arena offset rebasing (the earlier failed mode).
                cmd.drawIndexedIndirect(indirect_buffer, d.indirect_offset, end - i,
                                        sizeof(vk::DrawIndexedIndirectCommand));
                i = end - 1;
            } else cmd.drawIndexed(d.index_count, d.instances, 0, 0, d.first_instance);
        }
    };
    if (const auto direct = scheduler.DirectCommandBuffer()) record(direct, draws.data());
    else {
        auto& commands = scheduler.Stream();
        const auto* owned = commands.Copy(draws.data(), draws.size());
        commands.Record([=](vk::CommandBuffer cmd) { record(cmd, owned); });
    }
}
static std::vector<Resolved>& Scratch() {
    static std::vector<Resolved> scratch;
    return scratch;
}
static std::vector<const Learned*>& StateScratch() {
    static std::vector<const Learned*> scratch;
    return scratch;
}

static bool InactiveShadowUser(u32 reg) {
    // ShapeOk admits only PS with zero user SGPRs and VS with at most eight.
    return (reg >= 0x2C0C && reg < 0x2C1C) || (reg >= 0x2C54 && reg < 0x2C5C);
}
// ISS_NATIVE_SHADOW_LOOSE_KEY=1: words that differ per object but never select the pipeline
// leave the memo key. VS user data 0/1: the audited shape reads its user data only as the fetch
// list pointer (2/3) and the SSBO sharp (4..7), with no flattened user data buffer, so 0/1 reach
// no descriptor and no specialization. The context registers the fast draw path classifies as
// Free (depth bounds, scissors, viewport depths, blend constants, stencil ops, polygon offset,
// line width): dynamic state only, applied from the registers by UpdateDynamicState at every
// setup and state change, read by no pipeline key. Without this, moving through the city gives
// every range new keys (the memo fills and is cleared over and over, the ranges are interpreted).
static bool LooseKeyEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_LOOSE_KEY");
    return enabled;
}
static bool LooseKeyReg(u32 reg) {
    if (!LooseKeyEnabled()) return false;
    if (reg == 0x2C4C || reg == 0x2C4D) return true;
    return reg >= 0xA000 && reg < 0xA400 && ClassifyFastReg(reg) == FastRegClass::Free;
}

static bool EquivalentStateEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_EQUIVALENT_STATE");
    return enabled;
}
static bool EquivalentState(const ns::Batch& b, u32 left, u32 right, const u32* entry) {
    if (left == right) return true;
    if (left >= b.states.size() || right >= b.states.size()) return false;
    const auto& a = b.states[left]; const auto& c = b.states[right];
    for (u32 k = 0; k < b.state_regs.size(); ++k) {
        if (InactiveShadowUser(b.state_regs[k])) continue;
        const u32 av = ((a.inherit >> k) & 1) ? entry[k] : b.state_values[a.first_value + k];
        const u32 cv = ((c.inherit >> k) & 1) ? entry[k] : b.state_values[c.first_value + k];
        if (av != cv) return false;
    }
    return true;
}
static bool ResourceReuseEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_RESOURCE_REUSE");
    return enabled;
}
static bool EquivalentResources(const ns::Batch& b, u32 left, u32 right, const u32* entry) {
    if (left == right) return true;
    if (left >= b.states.size() || right >= b.states.size()) return false;
    const auto& a = b.states[left]; const auto& c = b.states[right];
    for (u32 k = 0; k < b.state_regs.size(); ++k) {
        const u32 reg = b.state_regs[k];
        // Reuse the existing fast-draw classification: Free fields are applied
        // by UpdateDynamicState, never target/resource preparation.
        if (InactiveShadowUser(reg) || ClassifyFastReg(reg) == FastRegClass::Free) continue;
        const u32 av = ((a.inherit >> k) & 1) ? entry[k] : b.state_values[a.first_value + k];
        const u32 cv = ((c.inherit >> k) & 1) ? entry[k] : b.state_values[c.first_value + k];
        if (av != cv) return false;
    }
    return true;
}

static u64 BeginHash(const AmdGpu::Regs& regs, const ns::Batch& batch) {
    // Context registers without the render target addresses (formats stay), the primitive
    // registers and the VS/PS program registers without the per draw VS user data.
    std::array<u32, 0x400> context;
    std::memcpy(context.data(), &regs.reg_array[0xA000], sizeof(context));
    // These words are represented by StateKey, including their inherited values.
    // Hashing their previous values too makes equivalent draws miss the memo.
    for (const u32 reg : batch.state_regs)
        if (reg >= 0xA000 && reg < 0xA400) context[reg - 0xA000] = 0;
    for (const u32 reg : {0x005u, 0x012u, 0x013u, 0x014u, 0x015u}) context[reg] = 0;
    // DRAW_INDEX_2 replaces these before each draw; its payload is stored in
    // Batch::draws. The previous draw's index address/count is not pipeline state.
    for (const u32 reg : {0x1F9u, 0x1FAu, 0x1FCu, 0x29Eu, 0x29Fu}) context[reg] = 0;
    if (LooseKeyEnabled()) {
        static const auto loose = [] {
            std::array<bool, 0x400> mask{};
            for (u32 r = 0; r < 0x400; ++r) mask[r] = LooseKeyReg(0xA000 + r);
            return mask;
        }();
        for (u32 r = 0; r < 0x400; ++r)
            if (loose[r]) context[r] = 0;
    }
    for (u32 n = 0; n < 8; ++n)
        for (const u32 field : {0u, 7u, 9u}) context[0x318 + n * 15 + field] = 0;
    u64 h = XXH3_64bits(context.data(), sizeof(context));
    const auto hash_range = [&](u32 first, u32 count, u64 seed) {
        std::array<u32, 0x20> words;
        std::memcpy(words.data(), &regs.reg_array[first], count * sizeof(u32));
        for (u32 k = 0; k < count; ++k)
            if (InactiveShadowUser(first + k)) words[k] = 0;
        for (u32 reg : batch.state_regs)
            if (reg >= first && reg < first + count) words[reg - first] = 0;
        if (first == 0xC240) words[0xC24C - 0xC240] = 0; // previous num_indices
        if (first == 0xC240) words[0xC24D - 0xC240] = 0; // per-draw instances
        return XXH3_64bits_withSeed(words.data(), count * sizeof(u32), seed);
    };
    h = hash_range(0xC240, 0x20, h);
    h = hash_range(0x2C08, 0x14, h);
    std::array<u32, 0x14> vs;
    std::memcpy(vs.data(), &regs.reg_array[0x2C48], sizeof(vs));
    for (u32 reg : batch.state_regs)
        if (reg >= 0x2C48 && reg < 0x2C5C) vs[reg - 0x2C48] = 0;
    for (u32 k = 0; k < vs.size(); ++k)
        if (InactiveShadowUser(0x2C48 + k) || LooseKeyReg(0x2C48 + k)) vs[k] = 0;
    for (u32 k = 0; k < ns::DrawUserWords; ++k) vs[ns::VsUserData2 - 0x2C48 + k] = 0;
    return XXH3_64bits_withSeed(vs.data(), sizeof(vs), h);
}

// The key of a render state: `value(k)` gives the value of state register k at the draw.
template <typename Value>
static u64 StateKey(const ns::Batch& b, u64 begin_hash, Value&& value) {
    std::array<u32, 2 * ns::MaxStateRegs> words;
    const u32 n = u32(b.state_regs.size());
    for (u32 k = 0; k < n; ++k) {
        words[2 * k] = b.state_regs[k];
        words[2 * k + 1] =
            InactiveShadowUser(b.state_regs[k]) || LooseKeyReg(b.state_regs[k]) ? 0 : value(k);
    }
    return XXH3_64bits_withSeed(words.data(), 2 * n * sizeof(u32), begin_hash) | 1;
}

// The audited shape (as ISS_NATIVE_DEPTH_RUN): VS and PS only, one fetch attribute at user
// data 2, SSBO0 read only at user data 4 and the clip planes, nothing in the PS.
static bool ShapeOk(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs) {
    if (regs.reg_array[0xA003] & 0x78000000u) return false;
    if (regs.ps_program.settings.num_user_regs != 0 ||
        regs.vs_program.settings.num_user_regs > 8) return false;
    if (u32(regs.primitive_type) != 4 || regs.index_buffer_type.raw != 0 ||
        regs.enable_primitive_restart || regs.clipper_control.user_clip_plane_enable ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable) return false;
    const auto stages = pipeline.GetStages();
    const auto* vs_stage = stages[u32(Shader::SwStage::Vertex)];
    const auto* fs_stage = stages[u32(Shader::SwStage::Fragment)];
    if (!vs_stage || !fs_stage) return false;
    for (u32 i = 0; i < stages.size(); ++i)
        if (stages[i] && stages[i] != vs_stage && stages[i] != fs_stage) return false;
    const auto& vs = *vs_stage;
    const auto& fs = *fs_stage;
    const auto& fetch = pipeline.GetFetchShader();
    if (fetch.attributes.size() != 1 || fetch.vertex_offset_sgpr != -1 ||
        fetch.instance_offset_sgpr != -1 || fetch.attributes[0].sgpr_base != 2 ||
        fetch.attributes[0].dword_offset != 0 || fetch.attributes[0].instance_data != 0 ||
        fetch.attributes[0].inst_offset != 0 || vs.buffers.size() != 2 || !vs.images.empty() ||
        !vs.samplers.empty() || !fs.buffers.empty() || !fs.images.empty() ||
        !fs.samplers.empty() || vs.uses_dma || fs.uses_dma || !vs.fmasks.empty() ||
        !fs.fmasks.empty() || vs.buffers[0].IsSpecial() || vs.buffers[0].is_written ||
        vs.buffers[0].is_formatted || !vs.buffers[1].IsSpecial() ||
        vs.buffers[1].buffer_type != Shader::BufferType::ClipPlanes) return false;
    const std::array<u32, 4> ssbo{regs.vs_program.user_data[4], regs.vs_program.user_data[5],
                                  regs.vs_program.user_data[6], regs.vs_program.user_data[7]};
    if (std::bit_cast<std::array<u32, 4>>(vs.buffers[0].GetSharp(vs)) != ssbo) return false;
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    return vertex_offset == 0 && instance_offset == 0;
}

// ISS_NATIVE_SHADOW_VERIFY=1 with ISS_NATIVE_SHADOW_VERIFY_DIR (benchmarks; the visible frame
// is the translated one): a sampled range is recorded from the batch into private copies of its
// targets, then interpreted as usual into the real targets; both results, from the same
// starting bytes, must be identical (NativeVisual::Complete: before, reference, native).
static bool VerifyEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_SHADOW_VERIFY");
    return enabled;
}
static std::shared_ptr<NativeVisual::State> VerifyState() {
    static auto state = std::make_shared<NativeVisual::State>();
    return state;
}
struct Verify {
    std::shared_ptr<NativeVisual::Work> work;
    const ns::Batch* batch{};
    u32 last_draw{~0u};
    u32 expected_last{~0u};
};
static Verify& PendingVerify() {
    static Verify verify; // command processor thread only
    return verify;
}
} // namespace NativeShadow

// The capture of a verified range: the real targets of the first draw's render state are copied
// (before), and into private images the native draws go to. Returns the private render state.
std::optional<RenderState> Rasterizer::NativeShadowVerifyStart(const RenderState& render) {
    using namespace NativeVisual;
    const auto state = NativeShadow::VerifyState();
    const auto* output = std::getenv("ISS_NATIVE_SHADOW_VERIFY_DIR");
    if (!output || !*output || g_iss_frame.load(std::memory_order_relaxed) < 1200 ||
        render.num_layers != 1 || render.num_color_attachments > 8 || attachment_feedback_loop ||
        render.width > 4096 || render.height > 4096 ||
        state->pending.load(std::memory_order_acquire) || state->failed.load() ||
        state->qualified.load() >= 12 || state->attempts.load() >= 64) return std::nullopt;
    static u64 rejected{};
    const auto reject = [&](const char* why) -> std::optional<RenderState> {
        if (++rejected <= 16 || (rejected & 4095) == 0)
            std::fprintf(stderr, "NATIVE_SHADOW_VERIFY_REJECT reason=%s count=%llu\n", why, rejected);
        return std::nullopt;
    };
    auto work = std::make_shared<Work>();
    work->frame = g_iss_frame.load();
    work->width = render.width;
    work->height = render.height;
    work->directory = std::filesystem::path(output);
    u64 bytes = 0;
    // Whole single layer/level targets with plain views, as the depth run pair.
    const auto admit = [&](VideoCore::ImageId id, vk::ImageView view, u32 slot, bool depth) {
        if (!id || !view) return false;
        auto& image = texture_cache.GetImage(id);
        if (!image.backing || image.binding.is_bound) return false;
        const auto& backing = *image.backing;
        const auto& ci = backing.image.image_ci;
        if (ci.imageType != vk::ImageType::e2D || ci.samples != vk::SampleCountFlagBits::e1 ||
            ci.mipLevels != 1 || ci.arrayLayers != 1 || ci.extent.depth != 1 ||
            ci.extent.width != render.width || ci.extent.height != render.height ||
            ci.sharingMode != vk::SharingMode::eExclusive || !backing.subresource_states.empty() ||
            !(ci.usage & vk::ImageUsageFlagBits::eTransferSrc) ||
            (ci.flags & (vk::ImageCreateFlagBits::eSparseBinding |
                         vk::ImageCreateFlagBits::eSparseResidency))) return false;
        const VideoCore::ImageViewInfo* info = nullptr;
        for (auto view_id : backing.image_view_ids) {
            const auto& live = texture_cache.GetImageView(view_id);
            if (*live.image_view == view) { info = &live.info; break; }
        }
        if (!info || info->format != ci.format || info->range.base.level ||
            info->range.base.layer || info->range.extent.levels != 1 ||
            info->range.extent.layers != 1 || info->min_lod) return false;
        Target target;
        target.source = &image;
        target.saved = backing.state;
        target.view_info = *info;
        target.aspects = image.aspect_mask;
        target.slot = slot;
        target.depth = depth;
        target.name = depth ? "depth" : "color" + std::to_string(slot);
        if (depth) {
            if (ci.format == vk::Format::eD32Sfloat || ci.format == vk::Format::eD32SfloatS8Uint)
                target.planes.push_back({vk::ImageAspectFlagBits::eDepth, 4});
            else if (ci.format == vk::Format::eD16Unorm || ci.format == vk::Format::eD16UnormS8Uint)
                target.planes.push_back({vk::ImageAspectFlagBits::eDepth, 2});
            else return false;
            if (ci.format == vk::Format::eD32SfloatS8Uint || ci.format == vk::Format::eD16UnormS8Uint)
                target.planes.push_back({vk::ImageAspectFlagBits::eStencil, 1});
        } else {
            const auto bpp = ColorBytes(ci.format);
            if (!bpp) return false;
            target.planes.push_back({vk::ImageAspectFlagBits::eColor, bpp});
        }
        for (auto& plane : target.planes) {
            plane.offset = target.stride;
            plane.size = u64{render.width} * render.height * plane.bytes;
            target.stride = Common::AlignUp(target.stride + plane.size, u64{4});
        }
        bytes += target.stride;
        work->targets.push_back(std::move(target));
        return true;
    };
    for (u32 slot = 0; slot < render.num_color_attachments; ++slot) {
        if (!render.color_attachments[slot].image_view) continue;
        if (!admit(cb_descs[slot].first, render.color_attachments[slot].image_view, slot, false))
            return reject("color_target_shape_or_view");
    }
    if (render.depth_stencil_attachment.image_view &&
        !admit(db_desc.first, render.depth_stencil_attachment.image_view, 0, true))
        return reject("depth_target_shape_or_view");
    if (work->targets.empty()) return reject("no_targets");
    if (bytes > 256ull * 1024 * 1024) return reject("capture_budget");
    try {
        std::filesystem::create_directories(work->directory);
        for (auto& target : work->targets) {
            auto ci = target.source->backing->image.image_ci;
            ci.pNext = nullptr;
            ci.initialLayout = vk::ImageLayout::eUndefined;
            ci.usage |= vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
            target.images[1] = std::make_unique<VideoCore::UniqueImage>(instance.GetDevice(),
                                                                        instance.GetAllocator());
            target.images[1]->Create(ci);
            vk::ImageViewUsageCreateInfo usage{.usage = target.source->usage_flags & ci.usage};
            if (!target.view_info.is_storage) usage.usage &= ~vk::ImageUsageFlagBits::eStorage;
            const vk::ImageViewCreateInfo view_ci{
                .pNext = &usage, .image = target.images[1]->image,
                .viewType = vk::ImageViewType::e2D, .format = target.view_info.format,
                .components = target.view_info.mapping,
                .subresourceRange = {target.aspects, 0, 1, 0, 1}};
            auto [result, view] = instance.GetDevice().createImageViewUnique(view_ci);
            if (result != vk::Result::eSuccess)
                throw std::runtime_error("shadow verify image view allocation failed");
            target.views[1] = std::move(view);
            target.readback = std::make_unique<VideoCore::Buffer>(
                instance, 0, target.stride * 3, VideoCore::MemoryType::HostCached, "shadow verify");
        }
    } catch (const std::exception& error) {
        state->failed.store(true);
        std::fprintf(stderr, "NATIVE_SHADOW_VERIFY_ERROR allocation=%s\n", error.what());
        return std::nullopt;
    }
    work->sample = state->attempts.fetch_add(1) + 1;
    state->pending.store(true, std::memory_order_release);
    // The real pass begins (its clear, if any, is recorded) and ends; its bytes are the start.
    scheduler.BeginRendering(render);
    scheduler.EndRendering();
    const auto cmd = scheduler.Recorder();
    for (auto& target : work->targets)
        runtime.Transit(target.source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    for (auto& target : work->targets) {
        CopyOut(cmd, target, target.source->GetImage(), 0, render.width, render.height);
        Barrier(cmd, target.images[1]->image, target.aspects, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe, {},
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
        for (const auto& plane : target.planes) {
            const vk::ImageCopy copy{.srcSubresource = {plane.aspect, 0, 0, 1},
                                     .dstSubresource = {plane.aspect, 0, 0, 1},
                                     .extent = {render.width, render.height, 1}};
            cmd.copyImage(target.source->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                          target.images[1]->image, vk::ImageLayout::eTransferDstOptimal,
                          std::span{&copy, 1});
        }
        Barrier(cmd, target.images[1]->image, target.aspects, vk::ImageLayout::eTransferDstOptimal,
                target.saved.layout, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllGraphics,
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
        runtime.Transit(target.source, target.saved.layout, target.saved.pl_stage,
                        target.saved.access_mask);
    }
    runtime.FlushBarriers();
    NativeShadow::PendingVerify().work = std::move(work);
    return NativeShadowVerifyTargets(render);
}

// A render state of the verified range with its targets replaced by the private copies.
RenderState Rasterizer::NativeShadowVerifyTargets(const RenderState& render) {
    auto clone = render;
    for (auto& attachment : clone.color_attachments) attachment.is_clear = 0;
    clone.depth_stencil_attachment.depth_clear = false;
    clone.depth_stencil_attachment.stencil_clear = false;
    for (const auto& target : NativeShadow::PendingVerify().work->targets) {
        if (target.depth) clone.depth_stencil_attachment.image_view = *target.views[1];
        else clone.color_attachments[target.slot].image_view = *target.views[1];
    }
    return clone;
}

// After the native draws: the private targets are copied out (native).
void Rasterizer::NativeShadowVerifyNative(const iss::native_shadow::Batch& batch) {
    using namespace NativeVisual;
    auto& pending = NativeShadow::PendingVerify();
    scheduler.EndRendering();
    const auto cmd = scheduler.Recorder();
    for (auto& target : pending.work->targets) {
        Barrier(cmd, target.images[1]->image, target.aspects, target.saved.layout,
                vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllGraphics,
                vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferRead);
        CopyOut(cmd, target, target.images[1]->image, 2, pending.work->width, pending.work->height);
    }
    pending.batch = &batch;
    pending.last_draw = ~0u;
    pending.expected_last = u32(batch.draws.size() - 1);
}

// The end of the interpreted range: the real targets are copied out (reference) and compared.
void Rasterizer::NativeShadowVerifyFinish(bool complete) {
    using namespace NativeVisual;
    auto& pending = NativeShadow::PendingVerify();
    if (!pending.work) return;
    const auto state = NativeShadow::VerifyState();
    auto work = std::move(pending.work);
    pending = {};
    if (!complete) {
        std::fprintf(stderr, "NATIVE_SHADOW_VERIFY_ABORT sample=%u\n", work->sample);
        state->pending.store(false, std::memory_order_release);
        return;
    }
    scheduler.EndRendering();
    const auto cmd = scheduler.Recorder();
    for (auto& target : work->targets)
        runtime.Transit(target.source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    for (auto& target : work->targets)
        CopyOut(cmd, target, target.source->GetImage(), 1, work->width, work->height);
    for (auto& target : work->targets)
        runtime.Transit(target.source, target.saved.layout, target.saved.pl_stage,
                        target.saved.access_mask);
    runtime.FlushBarriers();
    const vk::MemoryBarrier2 host_read{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                       .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                       .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                                       .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host_read});
    scheduler.DeferPriorityOperation([work = std::move(work), state] {
        Complete(work, state, "NATIVE_SHADOW_PAIR", "shadow_pixels");
    });
}

void Rasterizer::NativeShadowLearnBegin(const iss::native_shadow::Batch& batch) {
    shadow_learn = {&batch, NativeShadow::BeginHash(DrawRegs(), batch), 0, false};
}

void Rasterizer::NativeShadowLearnEnd() {
    auto& pending = NativeShadow::PendingVerify();
    if (pending.work && pending.batch) {
        NativeShadowVerifyFinish(pending.batch == shadow_learn.batch &&
                                 pending.last_draw == pending.expected_last);
    }
    shadow_learn = {};
    auto& totals = NativeShadow::ActualDraws();
    if ((++totals.ranges & 8191) == 1)
        std::fprintf(stderr, "NATIVE_SHADOW_DRAW_TOTALS native=%llu interpreted=%llu reference=%llu\n",
                     totals.native, totals.interpreted, totals.reference);
}

void Rasterizer::NativeShadowBeforeDraw(u32) {
    using namespace NativeShadow;
    auto& l = shadow_learn;
    if (!l.batch) return;
    const auto& regs = DrawRegs();
    l.key = StateKey(*l.batch, l.begin_hash,
                     [&](u32 k) { return regs.reg_array[l.batch->state_regs[k]]; });
    l.learn = !Memo().contains(l.key);
    if (l.learn) {
        DiagnoseMemo(regs, *l.batch);
        // The full path: DrawRecord names the pipeline.
        fast_draw.valid = false;
        last_draw_pipeline = nullptr;
    }
}

void Rasterizer::NativeShadowAfterDraw(u32 draw) {
    using namespace NativeShadow;
    auto& l = shadow_learn;
    if (l.batch) ++ActualDraws().interpreted;
    if (auto& pending = PendingVerify(); pending.batch && pending.batch == l.batch) {
        ++ActualDraws().reference;
        pending.last_draw = draw;
        if (draw == pending.expected_last) NativeShadowVerifyFinish(true);
    }
    if (!l.batch || !l.learn) return;
    if (!last_draw_pipeline) {
        if (MemoStatsEnabled()) ++MemoCounters().no_pipeline;
        return;
    }
    const auto& regs = DrawRegs();
    Learned e{last_draw_pipeline};
    e.ok = ShapeOk(*last_draw_pipeline, regs);
    if (e.ok) {
        std::array<u32, 4> fetch{};
        const VAddr list = VAddr(regs.vs_program.user_data[2]) |
                           (VAddr(regs.vs_program.user_data[3]) << 32);
        memory->CopySparseMemory(list, reinterpret_cast<u8*>(fetch.data()), sizeof(fetch));
        e.fetch_words = {fetch[1], fetch[3]};
        e.ssbo_words = {regs.vs_program.user_data[5], regs.vs_program.user_data[7]};
    }
    auto& memo = Memo();
    if (MemoStatsEnabled()) {
        ++MemoCounters().inserts;
        if (!e.ok) ++MemoCounters().bad_shape;
    }
    if (memo.size() > 65536) {
        memo.clear();
        if (MemoStatsEnabled()) ++MemoCounters().clears;
    }
    memo[l.key] = e;
    l.learn = false;
}

u32 Rasterizer::NativeShadowExecute(const iss::native_shadow::Batch& b) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::ShadowNative};
    using namespace NativeShadow;
    LastRejectedShape() = ~0u;
    CostTrace cost;
    const auto draws = ExecutionDraws(b);
    const u32 first_draw = u32(b.draws.size() - draws.size());
    u32 n = u32(draws.size());
    if (!n) return 0;
    if (cost.enabled) CostTrace::Stats().draws += n;
    if (cp_split_enabled || draw_pipe_state || OnDrawWorker() || DrawQueue() != 0 ||
        iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
        guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        occlusion.HasOpen() || buffer_cache.HasComputeQueue() ||
        !instance.IsVertexInputDynamicState()) {
        ns::NoteRange(ns::Context, n);
        return 0;
    }
    auto& regs = DrawRegs();
    static const bool suffix_hash = iss::native_item::Flag("ISS_NATIVE_SHADOW_SUFFIX_HASH");
    static const bool suffix_hash_check = iss::native_item::Flag("ISS_NATIVE_SHADOW_SUFFIX_HASH_CHECK");
    static bool suffix_hash_rejected{};
    static u64 suffix_hash_hits{}, suffix_hash_checks{}, suffix_hash_mismatches{};
    // The learner is reset at every marker/end. Reuse only within that active
    // invocation, never by batch pointer/generation across DCB submissions.
    const bool reuse_hash = suffix_hash && !suffix_hash_rejected &&
        suffix_view_request.batch == &b && shadow_learn.batch == &b;
    u64 begin_hash = reuse_hash ? shadow_learn.begin_hash : BeginHash(regs, b);
    if (reuse_hash) {
        ++suffix_hash_hits;
        if (suffix_hash_check) {
            ++suffix_hash_checks;
            const u64 reference = BeginHash(regs, b);
            if (reference != begin_hash) {
                ++suffix_hash_mismatches;
                suffix_hash_rejected = true;
                begin_hash = reference;
            }
        }
        if ((suffix_hash_hits & 16383) == 1 || suffix_hash_rejected)
            std::fprintf(stderr, "NATIVE_SHADOW_SUFFIX_HASH hits=%llu checks=%llu mismatches=%llu disabled=%u\n",
                         suffix_hash_hits, suffix_hash_checks, suffix_hash_mismatches,
                         unsigned(suffix_hash_rejected));
    }
    auto& memo = Memo();
    auto& learned = StateScratch();
    learned.assign(b.states.size(), nullptr);
    for (u32 i = 0; i < n; ++i) {
        const auto& d = draws[i];
        if (learned[d.state]) continue;
        const auto& state = b.states[d.state];
        const u64 key = StateKey(b, begin_hash, [&](u32 k) {
            return (state.inherit >> k) & 1 ? regs.reg_array[b.state_regs[k]]
                                            : b.state_values[state.first_value + k];
        });
        const auto it = memo.find(key);
        if (it == memo.end()) {
            if (SplitEnabled() && i) { n = i; break; }
            ns::NoteRange(ns::Learn, n);
            return 0;
        }
        if (!it->second.ok || !it->second.pipeline->IsReady()) {
            if (!it->second.ok) LastRejectedShape() = d.state;
            if (SplitEnabled() && i) { n = i; break; }
            ns::NoteRange(ns::Missing, n);
            return 0;
        }
        learned[d.state] = &it->second;
    }

    cost.Phase(CostTrace::Resolve);
    // The CP replays from the marker on fallback/verification and commits final
    // registers on success. Do not let native setup change that starting state.
    std::array<u32, ns::MaxStateRegs> saved_state;
    for (u32 k = 0; k < b.state_regs.size(); ++k)
        saved_state[k] = regs.reg_array[b.state_regs[k]];
    std::array<u32, ns::DrawUserWords> saved_user;
    for (u32 k = 0; k < ns::DrawUserWords; ++k)
        saved_user[k] = regs.reg_array[ns::VsUserData2 + k];
    const auto saved_instances = regs.num_instances;
    const auto saved_index_type = regs.index_buffer_type;
    const auto saved_index_base = regs.index_base_address;
    const auto saved_num_indices = regs.num_indices;
    const auto saved_max_index = regs.max_index_size;
    const auto saved_initiator = regs.draw_initiator;
    SCOPE_EXIT {
        for (u32 k = 0; k < b.state_regs.size(); ++k)
            regs.reg_array[b.state_regs[k]] = saved_state[k];
        for (u32 k = 0; k < ns::DrawUserWords; ++k)
            regs.reg_array[ns::VsUserData2 + k] = saved_user[k];
        regs.num_instances = saved_instances;
        regs.index_buffer_type = saved_index_type;
        regs.index_base_address = saved_index_base;
        regs.num_indices = saved_num_indices;
        regs.max_index_size = saved_max_index;
        regs.draw_initiator = saved_initiator;
    };

    // As Draw does before each draw.
    DrainDrawPipe(5);
    SubmitFeedbackBatch(scheduler);
    PeriodicSubmit(scheduler);
    scheduler.PopPendingOperations();

    // Resolution of every draw, as the translated path resolves it, before any recording.
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 tick = scheduler.CurrentTick();
    const bool reuse_check =
        Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    static const bool geometry_reuse_flag = iss::native_item::Flag("ISS_NATIVE_SHADOW_GEOMETRY_REUSE");
    static const bool geometry_verify = iss::native_item::Flag("ISS_NATIVE_SHADOW_GEOMETRY_VERIFY");
    const bool geometry_reuse = geometry_reuse_flag && StageMemoContextEligible();
    struct GeometryEntry {
        const Rasterizer* owner{};
        VAddr address{};
        u32 size{};
        const VideoCore::Buffer* buffer{};
        u64 offset{}, tick{};
        StageMemoStamp stamp{};
        bool unwritten{};
    };
    static std::array<GeometryEntry, 2048> geometry_cache;
    static u64 geometry_hits{}, geometry_misses{}, geometry_checks{}, geometry_mismatches{};
    const auto geometry_stamp = [&] {
        auto stamp = CaptureStageMemoStamp();
        // Binding bookkeeping does not modify an arena's contents or lifetime.
        // Keep every memory/upload/dirty/hazard/arena generation, as StageMemo.
        stamp.bind_epoch = 0;
        return stamp;
    };
    const auto obtain_geometry = [&](VAddr address, u32 size, bool& unwritten)
        -> std::pair<const VideoCore::Buffer*, u64> {
        const auto obtain = [&] {
            return buffer_cache.ObtainBuffer(address, size, false, false,
                                              reuse_check ? &unwritten : nullptr);
        };
        if (!geometry_reuse) return obtain();
        const u64 now_tick = scheduler.CurrentTick();
        const auto before = geometry_stamp();
        const u64 key = address ^ (u64(size) << 32);
        auto& entry = geometry_cache[(key * 0x9E3779B97F4A7C15ull) >> (64 - 11)];
        std::pair<const VideoCore::Buffer*, u64> result;
        if (entry.owner == this && entry.address == address && entry.size == size &&
            entry.buffer && entry.tick == now_tick && entry.stamp == before) {
            ++geometry_hits;
            unwritten = entry.unwritten;
            result = {entry.buffer, entry.offset};
            if (geometry_verify) {
                ++geometry_checks;
                const bool expected_unwritten = unwritten;
                unwritten = false;
                const auto reference = obtain();
                if (reference != result || unwritten != expected_unwritten) ++geometry_mismatches;
                result = reference;
                if (geometry_stamp() != before) entry.buffer = nullptr;
            }
        } else {
            ++geometry_misses;
            result = obtain();
            const auto after = geometry_stamp();
            entry.buffer = nullptr;
            // Stream allocations have different reuse rules; leave them to the
            // existing cache. An arena result must have caused no state mutation.
            if (result.first != &stream && before == after && scheduler.CurrentTick() == now_tick)
                entry = {this, address, size, result.first, result.second, now_tick, after, unwritten};
        }
        if (((geometry_hits + geometry_misses) & 16383) == 1)
            std::fprintf(stderr, "NATIVE_SHADOW_GEOMETRY hits=%llu misses=%llu checks=%llu mismatches=%llu\n",
                         geometry_hits, geometry_misses, geometry_checks, geometry_mismatches);
        return result;
    };
    const u64 align = std::max<u64>(instance.StorageMinAlignment(), 4);
    auto& out = Scratch();
    out.assign(n, {});
    bool barrier = false;
    bool shared_records = GroupEnabled();
    u64 packed_size = 0;
    u64 region_size = 0;
    const auto fail = [&] {
        ns::NoteRange(ns::Missing, n);
        return 0u;
    };
    for (u32 i = 0; i < n; ++i) {
        const auto& d = draws[i];
        auto& r = out[i];
        if (d.index_count == 0) continue; // nothing drawn, as the translated path
        const auto& l = *learned[d.state];
        for (u32 k = 0; k < ns::DrawUserWords; ++k)
            r.user[k] = (d.user_inherit >> k) & 1 ? regs.reg_array[ns::VsUserData2 + k] : d.user[k];
        const u32 index_type = d.index_type_inherit ? regs.index_buffer_type.raw : d.index_type;
        r.instances_raw = d.instances_inherit ? regs.num_instances.num_instances : d.instances;
        r.instances = r.instances_raw == 0 ? 1 : r.instances_raw;
        if (index_type != 0) return fail();
        std::array<u32, 4> raw{};
        const VAddr list = VAddr(r.user[0]) | (VAddr(r.user[1]) << 32);
        memory->CopySparseMemory(list, reinterpret_cast<u8*>(raw.data()), sizeof(raw));
        const auto sharp = std::bit_cast<AmdGpu::Buffer>(raw);
        if (raw[1] != l.fetch_words[0] || raw[3] != l.fetch_words[1] || !sharp.base_address ||
            !sharp.GetSize()) return fail();
        // As BindVertexBuffers (one guest buffer, so one merged range).
        const u64 vb_size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
        if (!vb_size || vb_size > std::numeric_limits<u32>::max()) return fail();
        bool vb_unwritten = false;
        const auto [vb, vb_offset] = obtain_geometry(sharp.base_address, u32(vb_size), vb_unwritten);
        if (vb != &stream && !vb_unwritten) barrier |= runtime.IsBufferAccessed(vb, vb_offset, vb_size);
        // As BindIndexBuffer (16 bit indices, index offset 0).
        decltype(regs.index_base_address) index_base{};
        index_base.base_addr_lo = d.index_lo;
        index_base.base_addr_hi = d.index_hi;
        const u32 ib_size = d.index_count * 2;
        bool ib_unwritten = false;
        const auto [ib, ib_offset] = obtain_geometry(index_base.Address<VAddr>(), ib_size, ib_unwritten);
        if (ib != &stream && !ib_unwritten) barrier |= runtime.IsBufferAccessed(ib, ib_offset, ib_size);
        // SSBO0: a small read only range without GPU data is read from the current guest bytes.
        if (r.user[3] != l.ssbo_words[0] || r.user[5] != l.ssbo_words[1]) return fail();
        const auto ssbo = std::bit_cast<AmdGpu::Buffer>(
            std::array<u32, 4>{r.user[2], r.user[3], r.user[4], r.user[5]});
        const u64 ssbo_size = ssbo.GetSize();
        const auto& vs = l.pipeline->GetStage(Shader::SwStage::Vertex);
        const auto& ps = l.pipeline->GetStage(Shader::SwStage::Fragment);
        shared_records &= (vs.pgm_hash == 0xC01162F7 || vs.pgm_hash == 0x47B0694B) &&
                          ps.pgm_hash == 0x1318DDD4 && ssbo.GetStride() == 72 &&
                          r.user[3] == 4718592 && r.user[5] == 537031172 &&
                          ssbo_size == u64(r.instances) * 72;
        if (!ssbo.base_address || !ssbo_size || ssbo_size > NativeDepthRun::StreamThreshold ||
            memory->ClampRangeSize(ssbo.base_address, ssbo_size) != ssbo_size ||
            buffer_cache.HasGpuData(ssbo.base_address, ssbo_size)) return fail();
        if (scheduler.CurrentTick() != tick) return fail();
        r.vb = vb;
        r.vb_offset = vb_offset;
        r.ib = ib;
        r.ib_offset = ib_offset;
        r.index_count = d.index_count;
        r.ssbo = ssbo.base_address;
        r.ssbo_size = u32(ssbo_size);
        packed_size += ssbo_size;
        r.region_offset = u32(region_size);
        region_size += (ssbo_size + align - 1) / align * align;
    }
    if (region_size == 0) return fail();
    if (shared_records) {
        static thread_local const Instance* caps_owner{};
        static thread_local std::optional<iss::native_geometry::InstancedDeviceCaps> caps;
        if (caps_owner != &instance) {
            caps = iss::native_geometry::InstancedDeviceCaps::Capture(instance);
            caps_owner = &instance;
        }
        shared_records = packed_size <= std::numeric_limits<u32>::max() &&
                         caps->Supports(instance, n, packed_size);
        if (shared_records) {
            region_size = 0;
            for (auto& r : out) {
                r.region_offset = u32(region_size);
                region_size += r.ssbo_size;
            }
        }
    }
    cost.Phase(CostTrace::Copy);
    const auto [region, region_base] = stream.Map(region_size, align);
    if (!region) return fail();
    for (const auto& r : out)
        if (r.vb) memory->CopySparseMemory(r.ssbo, region + r.region_offset, r.ssbo_size);
    stream.Commit();
    if (scheduler.CurrentTick() != tick) return fail();

    u64 indirect_base = 0;
    const bool indirect = shared_records && IndirectGroupEnabled();
    if (indirect) {
        const auto [data, offset] = stream.Map(u64(n) * sizeof(vk::DrawIndexedIndirectCommand), 4);
        if (!data) return fail();
        indirect_base = offset;
        for (u32 i = 0; i < n; ++i) {
            const auto& r = out[i];
            const vk::DrawIndexedIndirectCommand draw{r.index_count, r.instances, 0, 0, r.region_offset / 72};
            std::memcpy(data + u64(i) * sizeof(draw), &draw, sizeof(draw));
        }
        stream.Commit();
        if (scheduler.CurrentTick() != tick) return fail();
    }

    cost.Phase(CostTrace::Record);
    // Recording, draw by draw, with the registers each draw had.
    static u64 verify_counter{};
    bool verifying = VerifyEnabled() && (++verify_counter % 61) == 0;
    const auto cmd = scheduler.Recorder();
    const GraphicsPipeline* current = nullptr;
    u32 current_state = ~0u;
    int ssbo_write = -1;
    u32 ssbo_index = 0;
    bool descriptors_bound = false;
    RenderState render_state{};
    const auto set_registers = [&](u32 i) {
        const auto& d = draws[i];
        const auto& r = out[i];
        if (d.state != current_state) {
            const auto& state = b.states[d.state];
            for (u32 k = 0; k < b.state_regs.size(); ++k)
                if (!((state.inherit >> k) & 1))
                    regs.reg_array[b.state_regs[k]] = b.state_values[state.first_value + k];
        }
        for (u32 k = 0; k < ns::DrawUserWords; ++k) regs.reg_array[ns::VsUserData2 + k] = r.user[k];
        regs.num_instances.num_instances = r.instances_raw;
        regs.index_base_address.base_addr_lo = d.index_lo;
        regs.index_base_address.base_addr_hi = d.index_hi;
        regs.num_indices = d.index_count;
        regs.max_index_size = d.max_index;
        regs.draw_initiator = d.initiator;
    };
    // As DrawRecord, up to the draw: render state, descriptors, targets, pipeline, vertex input.
    const auto setup = [&](const GraphicsPipeline* pipeline, bool first) {
        descriptors_bound = false;
        cost.Phase(CostTrace::Setup);
        SCOPE_EXIT { cost.Phase(CostTrace::Record); };
        PrepareRenderState(pipeline);
        if (first) {
            for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i)
                if (cb_descs[i].first &&
                    texture_cache.IsMetaCleared(regs.color_buffers[i].CmaskAddress(),
                                                cb_descs[i].second.view_info.range.base.layer))
                    return false;
            if (db_desc.first &&
                texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                            db_desc.second.view_info.range.base.layer))
                return false;
        }
        const auto target_count = bound_images.size();
        static const bool direct_resources = iss::native_item::Flag("ISS_NATIVE_SHADOW_DIRECT_RESOURCES");
        static const bool check_resources = iss::native_item::Flag("ISS_NATIVE_SHADOW_DIRECT_RESOURCES_CHECK");
        static bool direct_resources_rejected{};
        static u64 direct_setups{}, checked_setups{}, resource_mismatches{}, resource_calls{};
        bool direct = direct_resources && !direct_resources_rejected && !iss::hazard_probe::Enabled();
        bool resource_mismatch = false;
        const bool generic = !direct || check_resources;
        if (generic && (!BindResources(pipeline) || bound_images.size() != target_count)) return false;
        if (direct && check_resources) {
            ++checked_setups;
            // Compare the generic layout and push data before discarding its
            // unused original SSBO binding. The instance stream replaces slot0
            // below in both paths; the disabled clip planes must remain null.
            auto expected_push = MakeUserData(regs);
            auto actual_push = push_data;
            actual_push.buf_offsets[0] = 0;
            bool equal = set_write_index == 2 && set_writes.size() == 2 &&
                std::memcmp(&expected_push.xoffset, &actual_push.xoffset, 4 * sizeof(float)) == 0 &&
                expected_push.buf_offsets == actual_push.buf_offsets;
            for (u32 k = 0; equal && k < 2; ++k) {
                const auto& w = set_writes[k];
                equal = !w.dstSet && !w.pNext && w.dstBinding == k && w.dstArrayElement == 0 &&
                    w.descriptorCount == 1 && w.descriptorType == vk::DescriptorType::eStorageBuffer &&
                    w.pBufferInfo && !w.pImageInfo && !w.pTexelBufferView;
            }
            if (equal) {
                const auto& clip = *set_writes[1].pBufferInfo;
                equal = bool(set_writes[0].pBufferInfo->buffer) &&
                    !clip.buffer && clip.offset == 0 && clip.range == VK_WHOLE_SIZE;
            }
            if (!equal) {
                ++resource_mismatches;
                resource_mismatch = true;
                direct_resources_rejected = true;
                direct = false; // Keep the already prepared generic resources.
            }
        }
        if (direct) {
            if (!generic) {
                iss::feedback_join::BeginBindings();
                ++bind_resources_seq;
            }
            // ShapeOk admits precisely one read-only SSBO and disabled clip
            // planes, no other stage resources/DMA. The SSBO bytes have already
            // been validated and copied into this command buffer's stream.
            PrepareDescriptorWrites(set_writes, 2, false);
            buffer_infos.clear();
            image_infos.clear();
            image_bindings.clear();
            image_sources.clear();
            buffer_infos.emplace_back(stream.Handle(), region_base, region_size);
            buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            for (u32 k = 0; k < 2; ++k) WriteBufferDescriptor(set_writes[k], k, &buffer_infos[k]);
            set_write_index = 2;
            push_data = MakeUserData(regs);
            // Preserve bound_images (targets) and existing bound_buffers: their
            // ResetBindings bookkeeping still belongs to the regular renderer.
            ++direct_setups;
        }
        if (direct_resources && ((++resource_calls & 16383) == 1 || resource_mismatch))
            std::fprintf(stderr, "NATIVE_SHADOW_DIRECT_RESOURCES setups=%llu checks=%llu mismatches=%llu disabled=%u\n",
                         direct_setups, checked_setups, resource_mismatches,
                         unsigned(direct_resources_rejected));
        ssbo_write = -1;
        for (u32 i = 0, buffer_index = 0; i < set_write_index; ++i) {
            const auto& write = set_writes[i];
            if (!write.pBufferInfo) continue;
            if (write.pBufferInfo->buffer) {
                if (ssbo_write >= 0) return false;
                ssbo_write = int(i);
                ssbo_index = buffer_index;
            }
            buffer_index += write.descriptorCount;
        }
        if (ssbo_write < 0 || ssbo_index >= push_data.buf_offsets.size()) return false;
        render_state = BeginRendering(pipeline);
        if (first) needs_barrier |= barrier;
        if (needs_barrier) runtime.FlushBarriers();
        UpdateDynamicState(pipeline, true);
        if (attachment_feedback_loop) return false;
        if (verifying && first) {
            const auto clone = NativeShadowVerifyStart(render_state);
            verifying = clone.has_value();
            scheduler.BeginRendering(verifying ? *clone : render_state);
        } else {
            scheduler.BeginRendering(verifying ? NativeShadowVerifyTargets(render_state)
                                               : render_state);
        }
        VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
        VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
        VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                                  regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
        if (attributes.size() != 1 || bindings.size() != 1 || !divisors.empty()) return false;
        BindGraphicsPipeline(pipeline);
        if (!keep_vertex_input) cmd.setVertexInputEXT(bindings, attributes);
        return true;
    };
    // A state change of the same pipeline only updates the dynamic state, unless the range sets
    // shader registers (user data in the push constants) or clip planes (a bound buffer).
    const bool full_on_state = std::any_of(b.state_regs.begin(), b.state_regs.end(), [](u32 reg) {
        return (reg >= 0x2C00 && reg < 0x3000) || (reg >= 0xA16F && reg < 0xA187);
    });
    u32 recorded = 0;
    for (u32 i = 0; i < n; ++i) {
        const auto& d = draws[i];
        const auto& r = out[i];
        if (!r.vb) {
            ++recorded;
            continue;
        }
        set_registers(i);
        const auto* pipeline = learned[d.state]->pipeline;
        const bool same_state = d.state == current_state ||
            (EquivalentStateEnabled() && EquivalentState(b, d.state, current_state, saved_state.data()));
        const bool same_resources = same_state || (ResourceReuseEnabled() &&
            EquivalentResources(b, d.state, current_state, saved_state.data()));
        if (pipeline != current || (full_on_state && !same_resources)) {
            if (current) ResetBindings(false);
            const u64 setup_tick = scheduler.CurrentTick();
            if (!setup(pipeline, current == nullptr) ||
                (current && scheduler.CurrentTick() != setup_tick)) {
                ResetBindings(false);
                fast_draw.valid = false;
                if (verifying && PendingVerify().work) {
                    // Only the private copies were drawn: the range is interpreted whole.
                    NativeShadowVerifyFinish(false);
                    return fail();
                }
                if (!current) return fail();
                ns::NoteRange(ns::Partial, recorded);
                ActualDraws().native += recorded;
                return recorded;
            }
            current = pipeline;
        } else if (!same_state) {
            UpdateDynamicState(pipeline, true);
        }
        current_state = d.state;
        // Each draw with its own copy of its records, bound as the translated path binds them.
        static u64 resource_binds{}, reused_binds{};
        if (!ResourceReuseEnabled() || !shared_records || !descriptors_bound) {
            Pipeline::DescriptorWrites writes(set_writes.begin(), set_writes.begin() + set_write_index);
            const vk::DescriptorBufferInfo info{stream.Handle(),
                shared_records ? region_base : region_base + r.region_offset,
                shared_records ? region_size : r.ssbo_size};
            writes[ssbo_write].pBufferInfo = &info;
            auto push = push_data;
            push.buf_offsets[ssbo_index] = 0;
            pipeline->BindResources(writes, push);
            descriptors_bound = true;
            if (ResourceReuseEnabled()) ++resource_binds;
        } else ++reused_binds;
        if (ResourceReuseEnabled() && ((resource_binds + reused_binds) & 16383) == 1)
            std::fprintf(stderr, "NATIVE_SHADOW_RESOURCES binds=%llu reused=%llu\n", resource_binds, reused_binds);
        u32 end = i + 1;
        if (shared_records) {
            while (end < n && out[end].vb && learned[draws[end].state]->pipeline == pipeline &&
                   (draws[end].state == d.state || (EquivalentStateEnabled() &&
                    EquivalentState(b, draws[end].state, d.state, saved_state.data())))) ++end;
        }
        static std::vector<GroupDraw> commands;
        commands.clear();
        for (u32 j = i; j < end; ++j) {
            const auto& draw = out[j];
            const u32 first_instance = shared_records ? draw.region_offset / 72 : 0;
            if (shared_records && (RecordGroupEnabled() || indirect)) {
                commands.push_back({draw.vb->Handle(), draw.ib->Handle(), draw.vb_offset,
                                    draw.ib_offset, draw.index_count, draw.instances, first_instance,
                                    indirect_base + u64(j) * sizeof(vk::DrawIndexedIndirectCommand)});
            } else {
                const vk::Buffer vb_handle = draw.vb->Handle();
                const vk::DeviceSize vb_offset = draw.vb_offset;
                cmd.bindVertexBuffers(0, 1, &vb_handle, &vb_offset);
                cmd.bindIndexBuffer(draw.ib->Handle(), draw.ib_offset, vk::IndexType::eUint16);
                cmd.drawIndexed(draw.index_count, draw.instances, 0, 0, first_instance);
            }
            DebugState.IncDrawCall();
            iss::feedback_join::Recorded();
            ++recorded;
        }
        if (!commands.empty()) {
            if (indirect) {
                static u64 groups{}, draws{};
                for (u32 k = 0; k < commands.size();) {
                    u32 end = k + 1;
                    while (end < commands.size() && commands[end].vb == commands[k].vb &&
                           commands[end].ib == commands[k].ib &&
                           commands[end].vb_offset == commands[k].vb_offset &&
                           commands[end].ib_offset == commands[k].ib_offset) ++end;
                    if (end - k > 1) { ++groups; draws += end - k; }
                    k = end;
                }
                static u64 calls{};
                if ((++calls & 16383) == 1)
                    std::fprintf(stderr, "NATIVE_SHADOW_INDIRECT groups=%llu draws=%llu\n", groups, draws);
            }
            RecordGroup(scheduler, commands, indirect ? stream.Handle() : vk::Buffer{});
        }
        if (shared_records) NoteGroup(end - i, !commands.empty());
        i = end - 1;
    }
    if (verifying) {
        // The native result is in the private copies; the range is now interpreted into the
        // real targets, and compared at its end (NativeShadowLearnEnd).
        NativeShadowVerifyNative(b);
        PendingVerify().expected_last = first_draw + n - 1;
        ResetBindings(false);
        fast_draw.valid = false;
        ns::NoteRange(ns::Verify, n);
        return 0;
    }
    ResetBindings(false);
    fast_draw.valid = false;
    ns::NoteRange(ns::Native, n);
    ActualDraws().native += n;
    return n;
}

// Retry a later run after an unsupported draw was interpreted normally. The CP
// still walks setters and skips only draws already recorded, preserving order.
u32 ExecuteNativeShadowSuffix(Rasterizer& rasterizer, const iss::native_shadow::Batch& b, u32 first) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::ShadowSuffix};
    using namespace NativeShadow;
    if (!SplitEnabled() || PendingVerify().work || first >= b.draws.size()) return 0;
    static ns::Batch slice;
    static const ns::Batch* owner{};
    static u32 generation{};
    static std::vector<bool> rejected_shapes;
    static u64 attempts{}, recorded{}, shape_skips{};
    const u32 current_generation = b.ready.load(std::memory_order_acquire);
    if (owner != &b || generation != current_generation) {
        if (!SuffixViewEnabled()) {
            slice.state_regs = b.state_regs;
            slice.state_values = b.state_values;
            slice.states = b.states;
        }
        rejected_shapes.assign(b.states.size(), false);
        owner = &b; generation = current_generation;
    }
    if (rejected_shapes[b.draws[first].state]) { ++shape_skips; return 0; }
    u32 done;
    if (SuffixViewEnabled()) {
        const auto saved_request = suffix_view_request;
        suffix_view_request = {&b, first};
        SCOPE_EXIT { suffix_view_request = saved_request; };
        done = rasterizer.NativeShadowExecute(b);
    } else {
        slice.draws.assign(b.draws.begin() + first, b.draws.end());
        done = rasterizer.NativeShadowExecute(slice);
    }
    if (LastRejectedShape() != ~0u) rejected_shapes[LastRejectedShape()] = true;
    auto& pending = PendingVerify();
    if (pending.batch == &slice) {
        pending.batch = &b;
        pending.expected_last += first;
    }
    recorded += done;
    if ((++attempts & 16383) == 1)
        std::fprintf(stderr, "NATIVE_SHADOW_SPLIT attempts=%llu recorded=%llu shape_skips=%llu\n", attempts, recorded, shape_skips);
    return done;
}
