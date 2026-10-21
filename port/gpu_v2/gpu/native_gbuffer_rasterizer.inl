// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_shadow_rasterizer.inl.
//
// ISS_NATIVE_GBUFFER=1 (native_gbuffer.h): the draws of an engine G-buffer material range
// recorded from the range instead of interpreting their packets.
//
// Each draw takes the registers the range gives it (its render state version and its per draw
// resource table pointers). A draw whose render state or material differs from the previous one
// takes the full path as DrawRecord does: the pipeline lookup (the permutation depends on the
// material's sharps), the render state, every descriptor. A draw of the same pipeline, state and
// material (same images, samplers and buffer kinds) only resolves the buffers that moved (its
// instance records and per object constants), pushes the descriptor set again and binds its
// geometry. Any draw the path does not take ends the native prefix: the command processor
// interprets the rest of the range from that draw.
//
// ISS_NATIVE_GBUFFER_VERIFY=1 (with ISS_NATIVE_SHADOW_VERIFY_DIR): a sampled range is recorded
// into private copies of its targets and then interpreted into the real targets; both results
// must be identical (NATIVE_GBUFFER_PAIR).

namespace NativeGbuffer {
namespace ng = iss::native_gbuffer;
constexpr u32 MaxSharps = 32;

struct Sharps {
    u32 num_ps_buffers{}, num_ps_images{}, num_ps_samplers{}, num_vs_buffers{}, num_vs_images{},
        num_vs_samplers{}, num_fetch{};
    std::array<AmdGpu::Buffer, MaxSharps> ps_buffers{}, vs_buffers{};
    std::array<AmdGpu::Image, MaxSharps> ps_images{}, vs_images{};
    std::array<AmdGpu::Sampler, MaxSharps> ps_samplers{}, vs_samplers{};
    std::array<std::array<u32, 2>, MaxSharps> fetch{};
};

// The first ShapeOk test a pipeline failed (diagnosis).
static std::array<u64, 8>& ShapeFails() { static std::array<u64, 8> c{}; return c; }
struct ShapeEntry { const GraphicsPipeline* pipeline{}; bool ok{}; };
static std::array<ShapeEntry, 1024>& Shapes() {
    static std::array<ShapeEntry, 1024> shapes; // command processor thread only
    return shapes;
}

// What the path supports: vertex and pixel stages only, read only buffers and sampled images,
// no DMA, push descriptors, at most MaxSharps resources of each kind.
static bool ShapeOk(const GraphicsPipeline& pipeline) {
    auto& entry = Shapes()[(reinterpret_cast<std::uintptr_t>(&pipeline) >> 4) & 1023];
    if (entry.pipeline == &pipeline) return entry.ok;
    u32 fail = ~0u;
    const auto check = [&](bool pass, u32 reason) {
        if (!pass && fail == ~0u) fail = reason;
    };
    check(pipeline.UsesPushDescriptors() || NativeMaterial::SetsEnabled(), 0);
    const auto stages = pipeline.GetStages();
    const auto* vs = stages[u32(Shader::SwStage::Vertex)];
    const auto* ps = stages[u32(Shader::SwStage::Fragment)];
    check(vs && ps, 1);
    for (u32 i = 0; i < stages.size(); ++i)
        check(!stages[i] || stages[i] == vs || stages[i] == ps, 1);
    for (const auto* stage : {vs, ps}) {
        if (!stage) continue;
        check(!stage->uses_dma && stage->fmasks.empty() && stage->buffers.size() <= MaxSharps &&
              stage->images.size() <= MaxSharps && stage->samplers.size() <= MaxSharps, 2);
        for (const auto& b : stage->buffers) {
            check(!b.is_written, 3);
            if (b.IsSpecial())
                check(b.buffer_type == Shader::BufferType::Flatbuf ||
                      b.buffer_type == Shader::BufferType::ClipPlanes, 4);
        }
        for (const auto& image : stage->images) check(!image.is_written, 3);
    }
    // Vertex samplers: compared by the lean path like the others (ISS_NATIVE_GBUFFER_SETS).
    check(!vs || vs->samplers.empty() || NativeMaterial::SetsEnabled(), 5);
    check(pipeline.GetFetchShader().attributes.size() <= MaxSharps, 6);
    const bool ok = fail == ~0u;
    if (!ok) ++ShapeFails()[fail];
    entry = {&pipeline, ok};
    return ok;
}

static void Capture(const GraphicsPipeline& pipeline, Sharps& s) {
    const auto& vs = pipeline.GetStage(Shader::SwStage::Vertex);
    const auto& ps = pipeline.GetStage(Shader::SwStage::Fragment);
    s.num_ps_buffers = u32(ps.buffers.size());
    s.num_ps_images = u32(ps.images.size());
    s.num_ps_samplers = u32(ps.samplers.size());
    s.num_vs_buffers = u32(vs.buffers.size());
    s.num_vs_images = u32(vs.images.size());
    s.num_vs_samplers = u32(vs.samplers.size());
    for (u32 i = 0; i < s.num_ps_buffers; ++i)
        s.ps_buffers[i] = ps.buffers[i].IsSpecial() ? AmdGpu::Buffer{} : ps.buffers[i].GetSharp(ps);
    for (u32 i = 0; i < s.num_ps_images; ++i) s.ps_images[i] = ps.images[i].GetSharp(ps);
    for (u32 i = 0; i < s.num_ps_samplers; ++i) s.ps_samplers[i] = ps.samplers[i].GetSharp(ps);
    for (u32 i = 0; i < s.num_vs_buffers; ++i)
        s.vs_buffers[i] = vs.buffers[i].IsSpecial() ? AmdGpu::Buffer{} : vs.buffers[i].GetSharp(vs);
    for (u32 i = 0; i < s.num_vs_images; ++i) s.vs_images[i] = vs.images[i].GetSharp(vs);
    for (u32 i = 0; i < s.num_vs_samplers; ++i) s.vs_samplers[i] = vs.samplers[i].GetSharp(vs);
    const auto& fetch = pipeline.GetFetchShader();
    s.num_fetch = u32(fetch.attributes.size());
    for (u32 i = 0; i < s.num_fetch; ++i) {
        std::array<u32, 4> words;
        const auto sharp = fetch.attributes[i].GetSharp(vs);
        std::memcpy(words.data(), &sharp, sizeof(words));
        s.fetch[i] = {words[1] >> 12, words[3]};
    }
}

// The same buffer kind: only its address and size may differ (the permutation reads neither).
static bool SameKind(const AmdGpu::Buffer& a, const AmdGpu::Buffer& b) {
    std::array<u32, 4> x, y;
    std::memcpy(x.data(), &a, sizeof(x));
    std::memcpy(y.data(), &b, sizeof(y));
    return (x[1] >> 12) == (y[1] >> 12) && x[3] == y[3];
}

struct Totals { u64 native{}, interpreted{}, full{}, lean{}, templated{}, ranges{}; };
// Why a range's native prefix ended before its first draw (diagnosis, in DRAW_TOTALS).
enum Stop : u32 { StopOther, StopState, StopNoPipeline, StopNotReady, StopShape, StopCount };
static std::array<u64, StopCount>& StopCounts() { static std::array<u64, StopCount> c{}; return c; }
static Totals& Counters() { static Totals totals; return totals; }

static bool VerifyEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_GBUFFER_VERIFY");
    return enabled;
}
// Diagnosis: ISS_NATIVE_GBUFFER_LEAN=0 sends every draw through the full path;
// ISS_NATIVE_GBUFFER_VERIFY_CONTINUE=1 keeps sampling after a differing pair.
static bool LeanEnabled() {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_NATIVE_GBUFFER_LEAN");
        return !p || p[0] != '0';
    }();
    return enabled;
}
static bool VerifyContinue() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_GBUFFER_VERIFY_CONTINUE");
    return enabled;
}
struct Verify {
    std::shared_ptr<NativeVisual::Work> work;
    const ng::Batch* batch{};
    u32 remaining{}; // interpreted draws of the range until the reference is complete
};
static Verify& PendingVerify() { static Verify verify; return verify; }
} // namespace NativeGbuffer

// The end of a verified range's interpretation: the real targets (reference) are copied out and
// compared with the private copies the native draws wrote.
void Rasterizer::NativeGbufferVerifyFinish(bool complete) {
    using namespace NativeVisual;
    auto& pending = NativeGbuffer::PendingVerify();
    auto& shadow_pending = NativeShadow::PendingVerify();
    if (!pending.work) return;
    const auto state = NativeShadow::VerifyState();
    auto work = std::move(pending.work);
    pending = {};
    shadow_pending = {};
    if (!complete) {
        std::fprintf(stderr, "NATIVE_GBUFFER_VERIFY_ABORT sample=%u\n", work->sample);
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
        Complete(work, state, "NATIVE_GBUFFER_PAIR", "gbuffer_pixels");
    });
}

// An interpreted draw of a G-buffer range (command processor): counters and verification.
void Rasterizer::NativeGbufferInterpretedDraw(const iss::native_gbuffer::Batch& batch, bool last) {
    using namespace NativeGbuffer;
    ++Counters().interpreted;
    auto& pending = PendingVerify();
    if (pending.work && pending.batch == &batch && last) NativeGbufferVerifyFinish(true);
}

void Rasterizer::NativeGbufferRangeEnd(const iss::native_gbuffer::Batch* batch) {
    using namespace NativeGbuffer;
    auto& pending = PendingVerify();
    if (pending.work && (!batch || pending.batch == batch)) NativeGbufferVerifyFinish(false);
    auto& c = Counters();
    if ((++c.ranges & 8191) == 1)
    {
        const auto& st = StopCounts();
        const auto& sf = ShapeFails();
        std::fprintf(stderr, "NATIVE_GBUFFER_DRAW_TOTALS native=%llu interpreted=%llu full=%llu lean=%llu templated=%llu "
                     "first_stop[other=%llu state=%llu no_pipeline=%llu not_ready=%llu shape=%llu] "
                     "shape_fail[push=%llu stages=%llu limits=%llu written=%llu special=%llu "
                     "vs_samplers=%llu fetch=%llu]\n",
                     c.native, c.interpreted, c.full, c.lean, c.templated, st[0], st[1], st[2],
                     st[3], st[4], sf[0], sf[1], sf[2], sf[3], sf[4], sf[5], sf[6]);
        if (NativeMaterial::Enabled()) NativeMaterialReport();
    }
}

u32 Rasterizer::NativeGbufferExecute(const iss::native_gbuffer::Batch& b,
                                     const std::function<void(const u32*)>& begin_draw,
                                     const std::function<void()>& end_draw) {
    const iss::cp_path::Scope cp_path_scope{iss::cp_path::GbufferNative};
    using namespace NativeGbuffer;
    const u32 n = u32(b.draws.size());
    if (!n) return 0;
    if (cp_split_enabled || draw_pipe_state || OnDrawWorker() || DrawQueue() != 0 ||
        iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
        guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        occlusion.HasOpen() || buffer_cache.HasComputeQueue() ||
        !instance.IsVertexInputDynamicState() || iss::hazard_probe::Enabled() ||
        g_num_traced_ranges != 0 || NativeShadow::PendingVerify().work ||
        PendingVerify().work) {
        ng::NoteRange(ng::Context, n);
        NativeRecorder::Discard(b);
        return 0;
    }
    auto& regs = DrawRegs();
    // The command processor commits the range's registers itself (all of them on success, those
    // of the last recorded draw on a prefix): the native setup must not change its start state.
    std::array<u32, ng::MaxStateRegs> saved_state;
    for (u32 k = 0; k < b.state_regs.size(); ++k) saved_state[k] = regs.reg_array[b.state_regs[k]];
    std::array<u32, ng::UserWords> saved_vs, saved_ps;
    for (u32 k = 0; k < ng::UserWords; ++k) {
        saved_vs[k] = regs.reg_array[ng::VsUser0 + k];
        saved_ps[k] = regs.reg_array[ng::PsUser0 + k];
    }
    // ISS_NATIVE_MATERIALS: the range context, before any register of the range is set.
    const bool materials = NativeMaterial::Enabled();
    const u64 context_hash = materials ? NativeMaterial::ContextHash(regs, b.state_regs) : 0;
    std::vector<u64> state_keys(materials ? b.states.size() : 0, 0);
    const auto state_key = [&](u32 state) -> u64 {
        if (!materials) return 0;
        u64& key = state_keys[state];
        if (!key) key = NativeMaterial::StateKey(context_hash, b, state) | 1;
        return key;
    };
    std::vector<NativeImageCapture> image_capture;
    const auto saved_instances = regs.num_instances;
    const auto saved_index_type = regs.index_buffer_type;
    const auto saved_index_base = regs.index_base_address;
    const auto saved_num_indices = regs.num_indices;
    const auto saved_max_index = regs.max_index_size;
    const auto saved_initiator = regs.draw_initiator;
    SCOPE_EXIT {
        for (u32 k = 0; k < b.state_regs.size(); ++k) regs.reg_array[b.state_regs[k]] = saved_state[k];
        for (u32 k = 0; k < ng::UserWords; ++k) {
            regs.reg_array[ng::VsUser0 + k] = saved_vs[k];
            regs.reg_array[ng::PsUser0 + k] = saved_ps[k];
        }
        regs.num_instances = saved_instances;
        regs.index_buffer_type = saved_index_type;
        regs.index_base_address = saved_index_base;
        regs.num_indices = saved_num_indices;
        regs.max_index_size = saved_max_index;
        regs.draw_initiator = saved_initiator;
    };
    const auto set_registers = [&](u32 i) {
        const auto& d = b.draws[i];
        const auto& state = b.states[d.state];
        for (u32 k = 0; k < b.state_regs.size(); ++k)
            regs.reg_array[b.state_regs[k]] =
                (state.inherit >> k) & 1 ? saved_state[k] : b.state_values[state.first_value + k];
        for (u32 k = 0; k < ng::UserWords; ++k) {
            regs.reg_array[ng::VsUser0 + k] = (d.vs_inherit >> k) & 1 ? saved_vs[k] : d.vs_user[k];
            regs.reg_array[ng::PsUser0 + k] = (d.ps_inherit >> k) & 1 ? saved_ps[k] : d.ps_user[k];
        }
        regs.num_instances.num_instances =
            d.instances_inherit ? saved_instances.num_instances : d.instances;
        regs.index_buffer_type.raw = d.index_type_inherit ? saved_index_type.raw : d.index_type;
        regs.index_base_address.base_addr_lo = d.index_lo;
        regs.index_base_address.base_addr_hi = d.index_hi;
        regs.num_indices = d.index_count;
        regs.max_index_size = d.max_index;
        regs.draw_initiator = d.initiator;
    };
    // Two render state versions draw alike when their effective values match.
    const auto same_state = [&](u32 left, u32 right) {
        if (left == right) return true;
        const auto& a = b.states[left];
        const auto& c = b.states[right];
        for (u32 k = 0; k < b.state_regs.size(); ++k) {
            const u32 av = (a.inherit >> k) & 1 ? saved_state[k] : b.state_values[a.first_value + k];
            const u32 cv = (c.inherit >> k) & 1 ? saved_state[k] : b.state_values[c.first_value + k];
            if (av != cv) return false;
        }
        return true;
    };

    // As Draw does before each draw.
    DrainDrawPipe(5);
    SubmitFeedbackBatch(scheduler);
    PeriodicSubmit(scheduler);
    scheduler.PopPendingOperations();

    static u64 verify_counter{};
    bool verifying = VerifyEnabled() && (++verify_counter % 61) == 0;

    // ISS_NATIVE_RECORDER: the range as its engine thread recorded it (sampled verification
    // ranges take the paths below).
    if (materials) {
        NativeRecorderSetup();
        if (!verifying) {
            if (const u32 done = NativeRecorderExecute(b, context_hash, set_registers, begin_draw, end_draw)) {
                ng::NoteRange(ng::Native, done);
                Counters().native += done;
                return done;
            }
        } else {
            NativeRecorder::Discard(b);
        }
    }
    if (verifying && VerifyContinue()) NativeShadow::VerifyState()->failed.store(false);
    u32 verify_full = 0, verify_lean = 0;
    std::string verify_programs;
    const auto cmd = scheduler.Recorder();
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 alignment = instance.StorageMinAlignment();
    const GraphicsPipeline* current = nullptr;
    u32 current_state = ~0u;
    u64 state_tick = 0;
    // As the plain fast draw: no upload, CPU or GPU write, transition or texture cache change
    // since the previous draw, or the descriptors kept from it are resolved again.
    StageMemoStamp kept_stamp{};
    RenderState render_state{};
    boost::container::static_vector<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS + 1> targets;
    Sharps kept{};
    u32 recorded = 0;
    // The pipeline and state key the render state was last prepared for (full or template path),
    // and whether a template draw left bindings to release.
    const GraphicsPipeline* rs_pipeline = nullptr;
    u64 rs_key = 0;
    bool first_setup = true, bindings_live = false;
    bool in_scout = false;
    fast_draw.valid = false;
    const auto finish_scout = [&] {
        if (in_scout) end_draw();
        in_scout = false;
    };
    // The prefix recorded so far stays; the command processor interprets from draw `recorded`.
    const auto stop = [&](u32 why = StopOther) -> u32 {
        if (!recorded) ++StopCounts()[why];
        finish_scout();
        ResetBindings(false);
        fast_draw.valid = false;
        if (verifying && PendingVerify().work) {
            // Only the private copies were drawn: the range is interpreted whole, unverified.
            NativeGbufferVerifyFinish(false);
            ng::NoteRange(ng::Missing, n);
            return 0;
        }
        ng::NoteRange(recorded ? ng::Partial : ng::Missing, recorded);
        Counters().native += recorded;
        return recorded;
    };

    for (u32 i = 0; i < n; ++i) {
        const auto& d = b.draws[i];
        set_registers(i);
        if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Normal ||
            regs.primitive_type == AmdGpu::PrimitiveType::None ||
            (regs.reg_array[0xA003] & 0x78000000u) || regs.enable_primitive_restart ||
            regs.depth_render_control.depth_clear_enable ||
            regs.depth_render_control.stencil_clear_enable) return stop(StopState);
        if (regs.num_indices == 0) {
            Common::Perf::Add(Common::Perf::EmptyDraws);
            if (!Common::Perf::GetTuning().skip_empty_draws.load(std::memory_order_relaxed))
                return stop();
            ++recorded;
            continue;
        }
        begin_draw(d.packet);
        in_scout = true;
        const GraphicsPipeline* drawn = nullptr;

        // Same pipeline, state and material: refresh the stage user data and compare.
        // A submission since the setup lost the bound pipeline, vertex input and dynamic state.
        bool lean = LeanEnabled() && current && scheduler.CurrentTick() == state_tick &&
                    StageMemoContextEligible() && CaptureStageMemoStamp() == kept_stamp &&
                    (d.state == current_state || same_state(d.state, current_state));
        u32 changed_ps = 0, changed_vs = 0;
        if (lean) {
            auto& vs = const_cast<Shader::Info&>(current->GetStage(Shader::SwStage::Vertex));
            auto& ps = const_cast<Shader::Info&>(current->GetStage(Shader::SwStage::Fragment));
            vs.RefreshFlatBuf();
            ps.RefreshFlatBuf();
            for (u32 k = 0; lean && k < kept.num_ps_images; ++k) {
                const auto sharp = ps.images[k].GetSharp(ps);
                lean = std::memcmp(&sharp, &kept.ps_images[k], sizeof(sharp)) == 0;
            }
            for (u32 k = 0; lean && k < kept.num_ps_samplers; ++k) {
                const auto sharp = ps.samplers[k].GetSharp(ps);
                lean = std::memcmp(&sharp, &kept.ps_samplers[k], sizeof(sharp)) == 0;
            }
            for (u32 k = 0; lean && k < kept.num_vs_images; ++k) {
                const auto sharp = vs.images[k].GetSharp(vs);
                lean = std::memcmp(&sharp, &kept.vs_images[k], sizeof(sharp)) == 0;
            }
            for (u32 k = 0; lean && k < kept.num_vs_samplers; ++k) {
                const auto sharp = vs.samplers[k].GetSharp(vs);
                lean = std::memcmp(&sharp, &kept.vs_samplers[k], sizeof(sharp)) == 0;
            }
            for (u32 k = 0; lean && k < kept.num_ps_buffers; ++k) {
                if (ps.buffers[k].IsSpecial()) continue;
                const auto sharp = ps.buffers[k].GetSharp(ps);
                if (std::memcmp(&sharp, &kept.ps_buffers[k], sizeof(sharp)) == 0) continue;
                lean = SameKind(sharp, kept.ps_buffers[k]) && !ps.buffers[k].is_formatted;
                changed_ps |= 1u << k;
            }
            for (u32 k = 0; lean && k < kept.num_vs_buffers; ++k) {
                if (vs.buffers[k].IsSpecial()) continue;
                const auto sharp = vs.buffers[k].GetSharp(vs);
                if (std::memcmp(&sharp, &kept.vs_buffers[k], sizeof(sharp)) == 0) continue;
                lean = SameKind(sharp, kept.vs_buffers[k]) && !vs.buffers[k].is_formatted;
                changed_vs |= 1u << k;
            }
            const auto& fetch = current->GetFetchShader();
            for (u32 k = 0; lean && k < kept.num_fetch; ++k) {
                std::array<u32, 4> words;
                const auto sharp = fetch.attributes[k].GetSharp(vs);
                std::memcpy(words.data(), &sharp, sizeof(words));
                lean = (words[1] >> 12) == kept.fetch[k][0] && words[3] == kept.fetch[k][1];
            }
        }

        // ISS_NATIVE_MATERIALS: a material seen before in this render state draws from its
        // template (no pipeline lookup, no texture search).
        const NativeMaterialTemplate* tmpl = nullptr;
        if (!lean && materials && (!verifying || PendingVerify().work)) {
            tmpl = NativeMaterialFind(state_key(d.state));
            if (tmpl && !ShapeOk(*tmpl->pipeline)) tmpl = nullptr;
        }
        if (tmpl) {
            const u64 setup_tick = scheduler.CurrentTick();
            const auto* pipeline = tmpl->pipeline;
            const u64 key = state_key(d.state);
            ResetBindings(false);
            const bool new_state = first_setup || pipeline != rs_pipeline || key != rs_key;
            if (new_state) {
                PrepareRenderState(pipeline);
                if (first_setup) {
                    for (u32 c = 0; c < AmdGpu::NUM_COLOR_BUFFERS; ++c)
                        if (cb_descs[c].first &&
                            texture_cache.IsMetaCleared(regs.color_buffers[c].CmaskAddress(),
                                                        cb_descs[c].second.view_info.range.base.layer))
                            return stop();
                    if (db_desc.first &&
                        texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                                    db_desc.second.view_info.range.base.layer))
                        return stop();
                }
                targets.assign(bound_images.begin(), bound_images.end());
            } else {
                for (const auto image_id : targets) {
                    bound_images.emplace_back(image_id);
                    texture_cache.GetImage(image_id).binding.is_target = 1u;
                }
            }
            bindings_live = true;
            if (!NativeMaterialBind(*tmpl)) {
                // Stale or a target read as a texture: the full path, from clean bindings.
                ResetBindings(false);
                tmpl = nullptr;
                rs_pipeline = nullptr;
            } else {
                last_draw_pipeline = pipeline;
                for (const auto image_id : targets)
                    if (texture_cache.GetImage(image_id).binding.is_bound) return stop(); // feedback
                if (new_state) render_state = BeginRendering(pipeline);
                BindVertexBuffers(pipeline);
                BindIndexBuffer(0);
                if (needs_barrier) runtime.FlushBarriers();
                pipeline->BindResources(set_writes, push_data);
                UpdateDynamicState(pipeline, true);
                if (attachment_feedback_loop) return stop();
                if (recorded && scheduler.CurrentTick() != setup_tick) return stop();
                scheduler.BeginRendering(verifying ? NativeGbufferVerifyTargets(render_state)
                                                   : render_state);
                BindGraphicsPipeline(pipeline);
                rs_pipeline = pipeline;
                rs_key = key;
                first_setup = false;
                // The lean path compares with the full path's captured descriptors only.
                current = nullptr;
                current_state = ~0u;
                drawn = pipeline;
                ++Counters().templated;
                ++verify_full;
            }
        }

        if (drawn) {
        } else if (lean) {
            // Patch the moved buffers and the flattened user data copies in place, as
            // TryLeanRebind does, then push the whole set again.
            auto& vs = const_cast<Shader::Info&>(current->GetStage(Shader::SwStage::Vertex));
            auto& ps = const_cast<Shader::Info&>(current->GetStage(Shader::SwStage::Fragment));
            const u32 vs_write_base = kept.num_ps_buffers + kept.num_ps_images + kept.num_ps_samplers;
            bool ok = true;
            const auto patch = [&](Shader::Info& stage, u32 k, u32 write, u32 binding, bool moved,
                                   AmdGpu::Buffer& kept_sharp) {
                const auto& desc = stage.buffers[k];
                auto* info = const_cast<vk::DescriptorBufferInfo*>(set_writes[write].pBufferInfo);
                if (!info || set_writes[write].descriptorCount != 1) { ok = false; return; }
                if (desc.IsSpecial()) {
                    if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                        ok &= regs.clipper_control.user_clip_plane_enable == 0;
                        return;
                    }
                    const u32 size = u32(stage.FlatUserData().size() * sizeof(u32));
                    const u64 offset = stream.Copy(stage.FlatUserData().data(), size, alignment);
                    *info = vk::DescriptorBufferInfo{stream.Handle(), offset, size};
                    return;
                }
                if (!moved) return;
                const auto sharp = desc.GetSharp(stage);
                const auto& old = kept_sharp;
                const u64 old_size = old.base_address && old.GetSize()
                    ? memory->ClampRangeSize(old.base_address, old.GetSize()) : 0;
                if (old.base_address) {
                    for (auto it = bound_buffers.begin(); it != bound_buffers.end(); ++it)
                        if (it->address == old.base_address && it->size == old_size) {
                            bound_buffers.erase(it);
                            break;
                        }
                }
                if (!sharp.base_address || !sharp.GetSize()) {
                    *info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
                } else {
                    const u64 size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
                    const auto [buffer, offset] =
                        buffer_cache.ObtainBuffer(sharp.base_address, size, false, false, nullptr);
                    const u64 offset_aligned = Common::AlignDown(offset, alignment);
                    const u32 adjust = u32(offset - offset_aligned);
                    *info = vk::DescriptorBufferInfo{buffer->Handle(), offset_aligned, size + adjust};
                    if (binding >= push_data.buf_offsets.size()) { ok = false; return; }
                    push_data.buf_offsets[binding] = u8(adjust);
                    if (buffer != &stream) {
                        bound_buffers.push_back(BoundBuffer{buffer, offset, u32(size), false,
                                                            sharp.base_address});
                        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, false);
                    }
                }
                kept_sharp = sharp;
            };
            if (set_writes.size() < vs_write_base + kept.num_vs_buffers) ok = false;
            for (u32 k = 0; ok && k < kept.num_ps_buffers; ++k)
                patch(ps, k, k, k, (changed_ps >> k) & 1, kept.ps_buffers[k]);
            for (u32 k = 0; ok && k < kept.num_vs_buffers; ++k)
                patch(vs, k, vs_write_base + k, kept.num_ps_buffers + k, (changed_vs >> k) & 1,
                      kept.vs_buffers[k]);
            if (!ok) return stop();
            keep_vertex_input = true;
            BindVertexBuffers(current);
            keep_vertex_input = false;
            BindIndexBuffer(0);
            if (needs_barrier) runtime.FlushBarriers();
            current->BindResources(set_writes, push_data);
            if (scheduler.CurrentTick() != state_tick) return stop();
            scheduler.BeginRendering(verifying ? NativeGbufferVerifyTargets(render_state)
                                               : render_state);
            BindGraphicsPipeline(current);
            drawn = current;
            ++Counters().lean;
            ++verify_lean;
        } else {
            // The full path, as DrawRecord with the pipeline of this draw's registers and material.
            const u64 setup_tick = scheduler.CurrentTick();
            if (current || bindings_live) ResetBindings(false);
            bindings_live = false;
            const auto* pipeline = LookupNativeGraphicsPipeline();
            if (!pipeline) return stop(StopNoPipeline);
            if (!pipeline->IsReady() && SkipUnreadyDraws()) return stop(StopNotReady);
            if (!ShapeOk(*pipeline)) return stop(StopShape);
            const bool new_state = pipeline != current || current_state == ~0u ||
                                   !same_state(d.state, current_state) || pipeline != rs_pipeline;
            last_draw_pipeline = pipeline;
            if (new_state) {
                PrepareRenderState(pipeline);
                if (first_setup) {
                    for (u32 c = 0; c < AmdGpu::NUM_COLOR_BUFFERS; ++c)
                        if (cb_descs[c].first &&
                            texture_cache.IsMetaCleared(regs.color_buffers[c].CmaskAddress(),
                                                        cb_descs[c].second.view_info.range.base.layer))
                            return stop();
                    if (db_desc.first &&
                        texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                                    db_desc.second.view_info.range.base.layer))
                        return stop();
                }
                targets.assign(bound_images.begin(), bound_images.end());
            } else {
                for (const auto image_id : targets) {
                    bound_images.emplace_back(image_id);
                    texture_cache.GetImage(image_id).binding.is_target = 1u;
                }
            }
            image_capture.clear();
            if (materials) native_image_capture = &image_capture;
            const bool bound = BindResources(pipeline);
            native_image_capture = nullptr;
            if (!bound) return stop();
            for (const auto image_id : targets)
                if (texture_cache.GetImage(image_id).binding.is_bound) return stop(); // feedback
            if (new_state) render_state = BeginRendering(pipeline);
            BindVertexBuffers(pipeline);
            BindIndexBuffer(0);
            if (needs_barrier) runtime.FlushBarriers();
            pipeline->BindResources(set_writes, push_data);
            UpdateDynamicState(pipeline, true);
            if (attachment_feedback_loop) return stop();
            if (current && scheduler.CurrentTick() != setup_tick) return stop();
            if (verifying && !current) {
                const auto clone = NativeShadowVerifyStart(render_state);
                verifying = clone.has_value();
                if (verifying) {
                    PendingVerify().work = std::move(NativeShadow::PendingVerify().work);
                    NativeShadow::PendingVerify() = {};
                    PendingVerify().batch = &b;
                }
                scheduler.BeginRendering(verifying ? *clone : render_state);
            } else {
                scheduler.BeginRendering(verifying ? NativeGbufferVerifyTargets(render_state)
                                                   : render_state);
            }
            BindGraphicsPipeline(pipeline);
            Capture(*pipeline, kept);
            if (materials) {
                native_image_capture = &image_capture;
                NativeMaterialLearn(state_key(d.state), pipeline, render_state);
                native_image_capture = nullptr;
            }
            rs_pipeline = pipeline;
            rs_key = state_key(d.state);
            first_setup = false;
            drawn = pipeline;
            ++verify_full;
            if (verifying) {
                char item[48];
                std::snprintf(item, sizeof(item), " %u:%08llx/%08llx", i,
                              (unsigned long long)pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                              (unsigned long long)pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash);
                verify_programs += item;
            }
            state_tick = scheduler.CurrentTick();
            current = pipeline;
            current_state = d.state;
            ++Counters().full;
        }

        const auto& vs = drawn->GetStage(Shader::SwStage::Vertex);
        const auto [vertex_offset, instance_offset] =
            GetDrawOffsets(regs, vs, drawn->GetFetchShader());
        cmd.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0, s32(vertex_offset),
                        instance_offset);
        iss::feedback_join::Recorded();
        DebugState.IncDrawCall();
        kept_stamp = CaptureStageMemoStamp();
        finish_scout();
        ++recorded;
    }
    ResetBindings(false);
    fast_draw.valid = false;
    if (verifying) {
        // The native result is in the private copies; the range is now interpreted into the real
        // targets and compared after its last draw (NativeGbufferInterpretedDraw).
        NativeGbufferVerifyNative();
        PendingVerify().remaining = n;
        std::fprintf(stderr, "NATIVE_GBUFFER_VERIFY_RANGE sample=%u draws=%u full=%u lean=%u programs=%s\n",
                     PendingVerify().work->sample, n, verify_full, verify_lean,
                     verify_programs.c_str());
        ng::NoteRange(ng::Verify, n);
        return 0;
    }
    ng::NoteRange(ng::Native, n);
    Counters().native += n;
    return n;
}

RenderState Rasterizer::NativeGbufferVerifyTargets(const RenderState& render) {
    auto clone = render;
    for (auto& attachment : clone.color_attachments) attachment.is_clear = 0;
    clone.depth_stencil_attachment.depth_clear = false;
    clone.depth_stencil_attachment.stencil_clear = false;
    for (const auto& target : NativeGbuffer::PendingVerify().work->targets) {
        if (target.depth) clone.depth_stencil_attachment.image_view = *target.views[1];
        else clone.color_attachments[target.slot].image_view = *target.views[1];
    }
    return clone;
}

void Rasterizer::NativeGbufferVerifyNative() {
    using namespace NativeVisual;
    auto& pending = NativeGbuffer::PendingVerify();
    scheduler.EndRendering();
    const auto cmd = scheduler.Recorder();
    for (auto& target : pending.work->targets) {
        Barrier(cmd, target.images[1]->image, target.aspects, target.saved.layout,
                vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllGraphics,
                vk::AccessFlagBits2::eMemoryWrite, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferRead);
        CopyOut(cmd, target, target.images[1]->image, 2, pending.work->width, pending.work->height);
    }
}
