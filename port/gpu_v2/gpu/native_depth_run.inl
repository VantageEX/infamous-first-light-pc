// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan, after native_visual_probe.inl.
//
// ISS_NATIVE_DEPTH_RUN=1 (with ISS_NATIVE_INSTANCED_FEATURE=1 for drawIndirectFirstInstance): a
// run of the city's instanced depth family (VS C01162F7/47B0694B, PS 1318DDD4 without resources)
// is recorded as one indexed indirect draw instead of one translated draw per packet.
//
// Every draw of the run keeps the data the translated path would give it:
// - vertex and index ranges go through BufferCache::ObtainBuffer with the same sizes and the
//   same hazard check as BindVertexBuffers/BindIndexBuffer. The run only continues while they
//   land in one Vulkan buffer each: bound at offset 0, vertexOffset/firstIndex select exactly
//   the bytes the separate binds would have selected.
// - the per instance records (SSBO0, 72 bytes each, <= 16 KB, no GPU data): the translated path
//   reads the current guest bytes there (a stream copy or the clean arena). They are copied once
//   into one region; the shader indexes it by gl_InstanceIndex * 18 dwords, rebased by
//   firstInstance.
// The PM4 collector proves every other setter between the draws writes the value already set.
// A draw whose resolution would change anything else (an upload, a recorded write, another
// command buffer) ends the run before it; its packets are interpreted as usual.
//
// ISS_NATIVE_DEPTH_RUN_VERIFY=1 (benchmarks; draws only into private copies): sampled runs are also drawn into
// private copies of their targets twice, draw by draw with separately copied records and the
// separate bindings, then as the indirect draw. The target bytes must be identical.

namespace NativeDepthRun {
// BufferCache::STREAM_THRESHOLD (private): the small read only ranges ObtainBuffer serves from
// the current guest bytes (stream copy or clean arena) when they hold no GPU data.
constexpr u64 StreamThreshold = 16_KB;
enum Cut : u32 { CutVsharp, CutVertexBuffer, CutIndexBuffer, CutSsbo, CutSideEffect, CutTick, CutCount };
constexpr std::array<const char*, CutCount> cut_names{"vsharp", "vertex_buffer", "index_buffer",
                                                      "ssbo", "side_effect", "tick"};
struct Stats {
    u64 attempts{}, runs{}, draws{}, short_runs{}, setup_fails{}, barriers{};
    std::array<u64, CutCount> cut{};
};
static Stats& Get() {
    static Stats stats;
    return stats;
}
enum Early : u32 { EarlyCalls, EarlyNoFast, EarlyFamily, EarlyPrograms, EarlyContext, EarlyRegs,
                   EarlyStages, EarlyShape, EarlyOffsets, EarlySsboSharp, EarlyCollect, EarlyCount };
constexpr std::array<const char*, EarlyCount> early_names{
    "calls", "no_fast", "family", "programs", "context", "regs", "stages", "shape", "offsets",
    "ssbo_sharp", "collect"};
static std::array<u64, EarlyCount>& Early() {
    static std::array<u64, EarlyCount> early{};
    return early;
}
static void NoteEarly(u32 reason) {
    auto& e = Early();
    ++e[reason];
    if ((++e[EarlyCalls] & 0x3FFFF) == 1) {
        std::string line;
        for (u32 i = 0; i < EarlyCount; ++i) line += fmt::format(" {}={}", early_names[i], e[i]);
        std::fprintf(stderr, "NATIVE_DEPTH_RUN_EARLY%s\n", line.c_str());
    }
}
static void Report() {
    const auto& s = Get();
    std::string cuts;
    for (u32 i = 0; i < CutCount; ++i) cuts += fmt::format(" {}={}", cut_names[i], s.cut[i]);
    std::fprintf(stderr, "NATIVE_DEPTH_RUN attempts=%llu runs=%llu draws=%llu short=%llu setup_fails=%llu barriers=%llu cut[%s ]\n",
                 (unsigned long long)s.attempts, (unsigned long long)s.runs,
                 (unsigned long long)s.draws, (unsigned long long)s.short_runs,
                 (unsigned long long)s.setup_fails, (unsigned long long)s.barriers, cuts.c_str());
}
struct Draw {
    const VideoCore::Buffer* vb{};
    u64 vb_offset{};
    const VideoCore::Buffer* ib{};
    u64 ib_offset{};
    u32 index_count{}, instances{}, first_instance{}, ssbo_size{};
    VAddr ssbo{};
};
static std::vector<Draw>& Draws() {
    static std::vector<Draw> draws; // command processor thread only
    return draws;
}
// How a run is recorded: the reference is each draw with its own bindings (what the translated
// path records); the others share the records region and/or the geometry buffers.
enum Mode : u32 { ModeReference, ModeIndirect, ModeDirectMerged, ModeDirectSeparateGeometry, ModeCount };
constexpr std::array<const char*, ModeCount> mode_tags{
    "NATIVE_DEPTH_RUN_PAIR_REF", "NATIVE_DEPTH_RUN_PAIR_INDIRECT", "NATIVE_DEPTH_RUN_PAIR_DIRECT_MERGED",
    "NATIVE_DEPTH_RUN_PAIR_DIRECT_SEPARATE_GEOMETRY"};
static bool DiagEnabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_DEPTH_RUN_DIAG");
    return enabled;
}
static u64& VerifyRuns() {
    static u64 runs{};
    return runs;
}
static bool VerifyEnabled() {
    // Draws only into private copies: the visible frame is unchanged (allowed visible).
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_DEPTH_RUN_VERIFY");
    return enabled;
}
static std::shared_ptr<NativeVisual::State> VerifyState(u32 mode = ModeIndirect) {
    static std::array<std::shared_ptr<NativeVisual::State>, ModeCount> states = [] {
        std::array<std::shared_ptr<NativeVisual::State>, ModeCount> out;
        for (auto& state : out) state = std::make_shared<NativeVisual::State>();
        return out;
    }();
    return states[mode];
}
} // namespace NativeDepthRun

bool Rasterizer::NativeDepthRunEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_DEPTH_RUN");
        const bool on = value && value[0] == '1';
        std::fprintf(stderr, "PORT_TUNING native_depth_run=%d\n", on ? 1 : 0);
        return on;
    }();
    return enabled;
}

const GraphicsPipeline* Rasterizer::DepthRunCandidate() {
    // Cheap filter: the previous draw was of the depth family and this one has the same programs
    // (addresses and descriptors, not user data). Only then the pipeline is looked up, once per
    // run; the run's own checks follow in TryNativeDepthRun.
    using namespace NativeDepthRun;
    const auto& f = fast_draw;
    if (!f.valid || !f.pipeline || OnDrawWorker()) { NoteEarly(EarlyNoFast); return nullptr; }
    const auto stages = f.pipeline->GetStages();
    const auto* vs = stages[u32(Shader::SwStage::Vertex)];
    const auto* fs = stages[u32(Shader::SwStage::Fragment)];
    if (!vs || !fs || fs->pgm_hash != 0x1318DDD4 ||
        (vs->pgm_hash != 0xC01162F7 && vs->pgm_hash != 0x47B0694B)) {
        NoteEarly(EarlyFamily);
        return nullptr;
    }
    constexpr size_t program_regs = offsetof(AmdGpu::ShaderProgram, user_data);
    const auto& regs = DrawRegs();
    if (std::memcmp(&regs.vs_program, f.vs_program.data(), program_regs) != 0 ||
        std::memcmp(&regs.ps_program, f.ps_program.data(), program_regs) != 0) {
        NoteEarly(EarlyPrograms);
        return nullptr;
    }
    fast_lookup_set = false;
    // Always the full lookup: reusing the previous pipeline when no context register changed
    // was verified wrong for some draws (the VS has several permutations), see evidence.
    return pipeline_cache.GetGraphicsPipeline();
}

std::size_t Rasterizer::TryNativeDepthRun(const GraphicsPipeline* pipeline,
                                          std::span<const u32> following,
                                          std::size_t packet_budget) {
    namespace ng = iss::native_geometry;
    using namespace NativeDepthRun;
    if (!NativeDepthRunEnabled() || !pipeline) return 0;
    static thread_local const Instance* caps_owner{};
    static thread_local std::optional<ng::InstancedDeviceCaps> caps;
    if (caps_owner != &instance) {
        caps = ng::InstancedDeviceCaps::Capture(instance);
        caps_owner = &instance;
    }
    if (cp_split_enabled || draw_pipe_state || OnDrawWorker() || DrawQueue() != 0 ||
        iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
        guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        occlusion.HasOpen() || buffer_cache.HasComputeQueue() ||
        !instance.IsVertexInputDynamicState() || scheduler.HasPriorPendingOperations() ||
        (VerifyEnabled() && !DiagEnabled() &&
         VerifyState(ModeDirectSeparateGeometry)->failed.load(std::memory_order_acquire))) {
        NoteEarly(EarlyContext);
        return 0;
    }
    auto& regs = DrawRegs();
    if (u32(regs.primitive_type) != 4 || regs.index_buffer_type.raw != 0 ||
        regs.num_instances.NumInstances() == 0 || regs.num_instances.NumInstances() > 256 ||
        regs.enable_primitive_restart || regs.clipper_control.user_clip_plane_enable ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable) { NoteEarly(EarlyRegs); return 0; }
    const auto stages = pipeline->GetStages();
    const auto* vs_stage = stages[u32(Shader::SwStage::Vertex)];
    const auto* fs_stage = stages[u32(Shader::SwStage::Fragment)];
    if (!vs_stage || !fs_stage) { NoteEarly(EarlyStages); return 0; }
    for (u32 i = 0; i < stages.size(); ++i)
        if (stages[i] && stages[i] != vs_stage && stages[i] != fs_stage) return 0;
    const auto& vs = *vs_stage;
    const auto& fs = *fs_stage;
    if ((vs.pgm_hash != 0xC01162F7 && vs.pgm_hash != 0x47B0694B) || fs.pgm_hash != 0x1318DDD4)
        return 0;
    // The audited shape (evidence/secondary-native-batch-contract-20261013.md): one fetch
    // attribute at user data 2, SSBO0 of 72 byte records, clip planes unused, no PS resources.
    const auto& fetch = pipeline->GetFetchShader();
    if (fetch.size != 24 || fetch.attributes.size() != 1 ||
        fetch.vertex_offset_sgpr != -1 || fetch.instance_offset_sgpr != -1 ||
        fetch.attributes[0].dest_vgpr != 4 || fetch.attributes[0].num_elements != 3 ||
        fetch.attributes[0].sgpr_base != 2 || fetch.attributes[0].dword_offset != 0 ||
        fetch.attributes[0].instance_data != 0 || fetch.attributes[0].inst_offset != 0 ||
        fetch.attributes[0].data_format != 0 || fetch.attributes[0].num_format != 0 ||
        vs.buffers.size() != 2 || !vs.images.empty() || !vs.samplers.empty() ||
        !fs.buffers.empty() || !fs.images.empty() || !fs.samplers.empty() ||
        vs.uses_dma || fs.uses_dma || !vs.fmasks.empty() || !fs.fmasks.empty() ||
        vs.buffers[0].IsSpecial() || vs.buffers[0].is_written || vs.buffers[0].is_formatted ||
        !vs.buffers[1].IsSpecial() ||
        vs.buffers[1].buffer_type != Shader::BufferType::ClipPlanes) { NoteEarly(EarlyShape); return 0; }
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    if (vertex_offset != 0 || instance_offset != 0) { NoteEarly(EarlyOffsets); return 0; }

    const auto pointer_word = reinterpret_cast<const u32*>(&regs.vs_program.user_data[2]) -
                              regs.reg_array.data();
    const auto ssbo_word = reinterpret_cast<const u32*>(&regs.vs_program.user_data[4]) -
                           regs.reg_array.data();
    const u64 first_fetch = u64(regs.vs_program.user_data[2]) |
                            (u64(regs.vs_program.user_data[3]) << 32);
    const std::array<u32, 4> first_ssbo{regs.vs_program.user_data[4], regs.vs_program.user_data[5],
                                        regs.vs_program.user_data[6], regs.vs_program.user_data[7]};
    if (std::bit_cast<std::array<u32, 4>>(vs.buffers[0].GetSharp(vs)) != first_ssbo) {
        NoteEarly(EarlySsboSharp);
        return 0;
    }
    const ng::InstancedRunInputs input{
        regs.reg_array, AmdGpu::Regs::ShRegWordOffset, AmdGpu::Regs::UconfigRegWordOffset,
        std::size_t(pointer_word), std::size_t(ssbo_word), regs.index_buffer_type.raw,
        {first_fetch, first_ssbo, regs.index_base_address.Address<VAddr>(), regs.num_indices,
         regs.max_index_size, regs.draw_initiator, regs.num_instances.NumInstances(), 0}};
    const auto run = ng::CollectInstancedRun(following, input, 128, packet_budget);
    if (run.draws.size() < 2) {
        NoteEarly(EarlyCollect);
        // Diagnostic: the packets that ended the first rejected runs.
        static u32 dumped{};
        if (dumped < 12) {
            ++dumped;
            std::string line = fmt::format("NATIVE_DEPTH_RUN_COLLECT budget={} fetch_word={} ssbo_word={} packets:",
                                           packet_budget, pointer_word, ssbo_word);
            std::size_t cursor = 0;
            for (u32 p = 0; p < 14 && cursor < following.size(); ++p) {
                const u32 header = following[cursor];
                if ((header >> 30) != 3) { line += fmt::format(" t{}", header >> 30); ++cursor; continue; }
                const u32 count = ((header >> 16) & 0x3fff) + 1;
                const u32 opcode = (header >> 8) & 0xff;
                line += fmt::format(" {:02x}/{}", opcode, count);
                if ((opcode == 0x76 || opcode == 0x79 || opcode == 0x69) && cursor + 1 < following.size())
                    line += fmt::format("@{:x}", following[cursor + 1]);
                if (opcode == 0x10 && cursor + 1 < following.size())
                    line += fmt::format("[{:x}]", following[cursor + 1]);
                if (opcode == 0x76 && cursor + 2 < following.size()) {
                    const u32 first = AmdGpu::Regs::ShRegWordOffset + following[cursor + 1];
                    for (u32 i = 0; i < std::min<u32>(count - 1, 4) && cursor + 2 + i < following.size(); ++i)
                        line += fmt::format("{}{}", i ? "," : "=",
                                            following[cursor + 2 + i] == regs.reg_array[first + i] ? "same" : "diff");
                }
                cursor += count + 1;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        return 0;
    }
    auto& stats = Get();
    ++stats.attempts;
    if ((stats.attempts & 4095) == 1) Report();

    // Resolution of every draw, as the translated path resolves it, before any recording.
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 tick = scheduler.CurrentTick();
    const bool reuse_check =
        Common::Perf::GetTuning().reuse_buffer_checks.load(std::memory_order_relaxed);
    auto& draws = Draws();
    draws.clear();
    const VideoCore::Buffer* vb_buffer{};
    const VideoCore::Buffer* ib_buffer{};
    bool shared_geometry = true;
    bool barrier = false;
    u32 records = 0;
    Cut cut = CutCount;
    for (std::size_t n = 0; n < run.draws.size(); ++n) {
        const auto& src = run.draws[n];
        // Side effects of the first draw's lookups precede the whole run; a later draw whose
        // lookups upload or record a write is left to the translated path.
        const u64 epoch_before = VideoCore::CurrentUploadEpoch();
        const u64 hazard_before = runtime.HazardSeq();
        std::array<u32, 4> raw{};
        memory->CopySparseMemory(src.fetch_descriptors, reinterpret_cast<u8*>(raw.data()), sizeof(raw));
        const auto sharp = std::bit_cast<AmdGpu::Buffer>(raw);
        if (sharp.GetStride() != 8 || !sharp.base_address || !sharp.GetSize() ||
            sharp.GetSize() > 64 * 1024 || raw[1] != 524290 || raw[3] != 418732) {
            cut = CutVsharp;
            break;
        }
        bool draw_barrier = false;
        // As BindVertexBuffers (one guest buffer, so one merged range).
        const u64 vb_size = memory->ClampRangeSize(sharp.base_address, sharp.GetSize());
        if (!vb_size) { cut = CutVsharp; break; }
        bool vb_unwritten = false;
        const auto [vb, vb_offset] = buffer_cache.ObtainBuffer(
            sharp.base_address, u32(vb_size), false, false, reuse_check ? &vb_unwritten : nullptr);
        if (vb != &stream && !vb_unwritten)
            draw_barrier |= runtime.IsBufferAccessed(vb, vb_offset, vb_size);
        // Shared geometry bindings (diagnostic modes only) need one buffer and small offsets.
        shared_geometry &= (!vb_buffer || vb == vb_buffer) && vb_offset % 8 == 0 &&
                           vb_offset / 8 <= u64(std::numeric_limits<s32>::max());
        // As BindIndexBuffer (16 bit indices, index offset 0).
        const u32 ib_size = src.index_count * 2;
        bool ib_unwritten = false;
        const auto [ib, ib_offset] = buffer_cache.ObtainBuffer(
            src.index_address, ib_size, false, false, reuse_check ? &ib_unwritten : nullptr);
        if (ib != &stream && !ib_unwritten)
            draw_barrier |= runtime.IsBufferAccessed(ib, ib_offset, ib_size);
        shared_geometry &= (!ib_buffer || ib == ib_buffer) && ib_offset % 2 == 0 &&
                           ib_offset / 2 <= u64(std::numeric_limits<u32>::max());
        // SSBO0: a small read only range without GPU data is read from the current guest bytes.
        const auto ssbo = std::bit_cast<AmdGpu::Buffer>(src.ssbo_raw);
        const u64 ssbo_size = ssbo.GetSize();
        if (!ssbo.base_address || ssbo.GetStride() != 72 || src.ssbo_raw[1] != 4718592 ||
            src.ssbo_raw[3] != 537031172 || ssbo_size != u64(src.instances) * 72 ||
            ssbo_size > StreamThreshold ||
            memory->ClampRangeSize(ssbo.base_address, ssbo_size) != ssbo_size ||
            buffer_cache.HasGpuData(ssbo.base_address, ssbo_size)) {
            cut = CutSsbo;
            break;
        }
        if (n > 0 && (VideoCore::CurrentUploadEpoch() != epoch_before ||
                      runtime.HazardSeq() != hazard_before)) {
            cut = CutSideEffect;
            break;
        }
        if (scheduler.CurrentTick() != tick) {
            cut = CutTick;
            break;
        }
        if (!vb_buffer) vb_buffer = vb;
        if (!ib_buffer) ib_buffer = ib;
        barrier |= draw_barrier;
        draws.push_back({vb, vb_offset, ib, ib_offset, src.index_count, src.instances, records,
                         u32(ssbo_size), ssbo.base_address});
        records += src.instances;
    }
    if (cut != CutCount) ++stats.cut[cut];
    if (draws.size() < 2 || !caps->Supports(instance, u32(draws.size()), u64(records) * 72)) {
        ++stats.short_runs;
        return 0;
    }

    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    if (attributes.size() != 1 || bindings.size() != 1 || bindings[0].stride != 8) {
        ++stats.setup_fails;
        return 0;
    }

    // One region of instance records and the indirect commands, in this command buffer.
    const u64 ssbo_total = u64(records) * 72;
    const auto [ssbo_data, ssbo_offset] = stream.Map(ssbo_total, caps->StorageAlignment());
    if (!ssbo_data) { ++stats.setup_fails; return 0; }
    for (const auto& d : draws)
        memory->CopySparseMemory(d.ssbo, ssbo_data + u64(d.first_instance) * 72, d.ssbo_size);
    stream.Commit();
    // Indirect commands: only the diagnostic shared geometry modes read them.
    u64 command_offset = 0;
    if (VerifyEnabled() && DiagEnabled() && shared_geometry) {
    const u64 commands_size = draws.size() * sizeof(vk::DrawIndexedIndirectCommand);
    const auto [command_data, offset] = stream.Map(commands_size, 4);
    if (!command_data) { ++stats.setup_fails; return 0; }
    command_offset = offset;
    for (std::size_t i = 0; i < draws.size(); ++i) {
        const auto& d = draws[i];
        const vk::DrawIndexedIndirectCommand command{
            .indexCount = d.index_count,
            .instanceCount = d.instances,
            .firstIndex = u32(d.ib_offset / 2),
            .vertexOffset = s32(d.vb_offset / 8),
            .firstInstance = d.first_instance,
        };
        std::memcpy(command_data + i * sizeof(command), &command, sizeof(command));
    }
    stream.Commit();
    }
    if (scheduler.CurrentTick() != tick) { ++stats.setup_fails; return 0; }

    // The first draw's render state and descriptors, as DrawRecord prepares them.
    PrepareRenderState(pipeline);
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        if (cb_descs[i].first && texture_cache.IsMetaCleared(regs.color_buffers[i].CmaskAddress(),
                cb_descs[i].second.view_info.range.base.layer)) {
            ResetBindings(false); ++stats.setup_fails; return 0;
        }
    }
    if (db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
                                                     db_desc.second.view_info.range.base.layer)) {
        ResetBindings(false); ++stats.setup_fails; return 0;
    }
    const auto target_count = bound_images.size();
    if (!BindResources(pipeline) || bound_images.size() != target_count) {
        ResetBindings(false); ++stats.setup_fails; return 0;
    }
    int ssbo_write = -1;
    u32 ssbo_index = 0;
    for (u32 i = 0, buffer_index = 0; i < set_write_index; ++i) {
        const auto& write = set_writes[i];
        if (!write.pBufferInfo) continue;
        if (write.pBufferInfo->buffer) {
            if (ssbo_write >= 0) { ssbo_write = -2; break; }
            ssbo_write = int(i);
            ssbo_index = buffer_index;
        }
        buffer_index += write.descriptorCount;
    }
    if (ssbo_write < 0 || ssbo_index >= push_data.buf_offsets.size()) {
        ResetBindings(false); ++stats.setup_fails; return 0;
    }
    const auto ssbo_info = static_cast<std::size_t>(set_writes[ssbo_write].pBufferInfo - buffer_infos.data());
    buffer_infos[ssbo_info] = vk::DescriptorBufferInfo{stream.Handle(), ssbo_offset, ssbo_total};
    push_data.buf_offsets[ssbo_index] = 0;

    auto render_state = BeginRendering(pipeline);
    needs_barrier |= barrier;
    stats.barriers += needs_barrier ? 1 : 0;
    if (needs_barrier) runtime.FlushBarriers();
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, true);
    scheduler.BeginRendering(render_state);
    if (attachment_feedback_loop) { ResetBindings(false); ++stats.setup_fails; return 0; }
    const auto cmdbuf = scheduler.Recorder();
    BindGraphicsPipeline(pipeline);
    if (!keep_vertex_input) cmdbuf.setVertexInputEXT(bindings, attributes);
    // Recorded: each draw's own geometry bindings and a direct draw, the records shared through
    // one descriptor and firstInstance (verified identical to the draw by draw reference; shared
    // geometry offsets in the arena drew nothing, see evidence). ISS_NATIVE_DEPTH_RUN_DIAG=1
    // (with VERIFY): the real targets get the reference; every mode is compared with it.
    const bool diag = VerifyEnabled() && DiagEnabled();
    if (VerifyEnabled() && (++VerifyRuns() & 15) == 0) {
        for (u32 mode = ModeIndirect; mode < ModeCount; ++mode) {
            if (mode != ModeDirectSeparateGeometry && (!diag || !shared_geometry)) continue;
            VerifyDepthRunPair(pipeline, render_state, u32(ssbo_write), ssbo_index,
                               command_offset, vb_buffer, ib_buffer, mode);
        }
    }
    RecordDepthRunMode(diag ? ModeReference : ModeDirectSeparateGeometry, pipeline,
                       u32(ssbo_write), ssbo_index, command_offset, vb_buffer, ib_buffer);
    // Leave the bound state the translated path leaves after the last draw (its own records,
    // offset 0, as the reference binds them): the next draw can then take the fast path,
    // which compares with the state captured below. The geometry bound last is already its own.
    bool keep_fast = false;
    if (FastDrawEnabled()) {
        const auto& own = draws.back();
        const auto [own_data, own_offset] = stream.Map(own.ssbo_size, instance.StorageMinAlignment());
        if (own_data) {
            memory->CopySparseMemory(own.ssbo, own_data, own.ssbo_size);
            stream.Commit();
            buffer_infos[ssbo_info] = vk::DescriptorBufferInfo{stream.Handle(), own_offset, own.ssbo_size};
            pipeline->BindResources(set_writes, push_data);
            keep_fast = scheduler.CurrentTick() == tick;
        }
    }

    // The registers the skipped packets set: those of the last draw included.
    const auto& last = run.draws[draws.size() - 1];
    regs.vs_program.user_data[2] = u32(last.fetch_descriptors);
    regs.vs_program.user_data[3] = u32(last.fetch_descriptors >> 32);
    for (u32 i = 0; i < 4; ++i) regs.vs_program.user_data[4 + i] = last.ssbo_raw[i];
    regs.num_instances.num_instances = last.instances;
    regs.index_base_address.base_addr_lo = u32(last.index_address);
    regs.index_base_address.base_addr_hi = u32(last.index_address >> 32);
    regs.num_indices = last.index_count;
    regs.max_index_size = last.max_index_size;
    regs.draw_initiator = last.initiator;
    // As at the end of DrawRecord, with the last draw's registers and bindings.
    if (keep_fast) CaptureFastDraw(pipeline, true, render_state);
    ResetBindings(false);
    if (keep_fast) FinishFastDraw();
    else fast_draw.valid = false;
    for (std::size_t i = 0; i < draws.size(); ++i) {
        DebugState.IncDrawCall();
        iss::feedback_join::Recorded();
    }
    ++stats.runs;
    stats.draws += draws.size();
    if ((stats.runs & 4095) == 1) Report();
    return last.packet_word + 6;
}

#include "native_depth_run_verify.inl"
