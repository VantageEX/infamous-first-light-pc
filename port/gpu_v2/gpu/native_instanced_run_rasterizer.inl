// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan. One bounded DCB run is either fully replaced
// or left for ordinary DrawRecord/PM4 interpretation, never deferred.
std::size_t Rasterizer::TryNativeInstancedRun(const GraphicsPipeline* pipeline,
                                             std::span<const u32> following_commands,
                                             std::size_t packet_budget) {
    namespace ng = iss::native_geometry;
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_INSTANCED_BATCH");
        return value && value[0] == '1';
    }();
    static const bool verified_spirv = [] {
        const char* value = std::getenv("ISS_SECONDARY_SPV_VERIFIED");
        return value && value[0] == '1';
    }();
    if (!enabled || !verified_spirv || cp_split_enabled || draw_pipe_state ||
        OnDrawWorker() || DrawQueue() != 0 || iss::command_trace::enabled ||
        IsDumping() || host_markers_enabled || guest_markers_enabled ||
        GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        scheduler.HasPriorPendingOperations()) return 0;
    if (const char* probe = std::getenv("ISS_SECONDARY_BATCH_PROBE");
        probe && probe[0] == '1') return 0;
    auto& regs = DrawRegs();
    if (u32(regs.primitive_type) != 4 || regs.index_buffer_type.raw != 0 ||
        regs.num_instances.NumInstances() == 0 ||
        regs.num_instances.NumInstances() > 256 ||
        regs.enable_primitive_restart || regs.clipper_control.user_clip_plane_enable ||
        regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable) return 0;
    const auto key_hash = std::hash<GraphicsPipelineKey>{}(pipeline->GetGraphicsKey());
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fs = pipeline->GetStage(Shader::SwStage::Fragment);
    const bool c011 = key_hash == 0x3B6D3A6D0DEC8019ull &&
        vs.pgm_hash == 0xC01162F7 &&
        pipeline->GetGraphicsKey().stage_hashes[u32(Shader::SwStage::Vertex)] ==
            0x00000C0100769D1Full;
    const bool v47 = key_hash == 0x85C78D166865FBD7ull &&
        vs.pgm_hash == 0x47B0694B &&
        pipeline->GetGraphicsKey().stage_hashes[u32(Shader::SwStage::Vertex)] ==
            0x0000047BEEF75906ull;
    if ((!c011 && !v47) || fs.pgm_hash != 0x1318DDD4 ||
        pipeline->GetGraphicsKey().stage_hashes[u32(Shader::SwStage::Fragment)] !=
            0x000001323E5E9A42ull) return 0;
    const auto& fetch = pipeline->GetFetchShader();
    if (fetch.size != 24 || fetch.attributes.size() != 1 ||
        fetch.vertex_offset_sgpr != -1 || fetch.instance_offset_sgpr != -1 ||
        fetch.attributes[0].dest_vgpr != 4 || fetch.attributes[0].num_elements != 3 ||
        fetch.attributes[0].sgpr_base != 2 || fetch.attributes[0].dword_offset != 0 ||
        fetch.attributes[0].instance_data != 0 || fetch.attributes[0].inst_offset != 0 ||
        fetch.attributes[0].data_format != 0 || fetch.attributes[0].num_format != 0 ||
        vs.buffers.size() != 2 || !vs.images.empty() || !vs.samplers.empty() ||
        !fs.buffers.empty() || !fs.images.empty() || !fs.samplers.empty()) return 0;
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    if (vertex_offset != 0 || instance_offset != 0) return 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) continue;
        if ((stage != &vs && stage != &fs) || stage->uses_dma || !stage->fmasks.empty()) return 0;
        for (const auto& desc : stage->buffers) {
            if (desc.is_written || (desc.IsSpecial() &&
                desc.buffer_type != Shader::BufferType::ClipPlanes)) return 0;
        }
    }
    if (vs.buffers[0].IsSpecial() || !vs.buffers[1].IsSpecial() ||
        vs.buffers[1].buffer_type != Shader::BufferType::ClipPlanes) return 0;

    const auto pointer_word = reinterpret_cast<const u32*>(&regs.vs_program.user_data[2]) -
                              regs.reg_array.data();
    const auto ssbo_word = reinterpret_cast<const u32*>(&regs.vs_program.user_data[4]) -
                           regs.reg_array.data();
    const u64 first_fetch = u64(regs.vs_program.user_data[2]) |
                            (u64(regs.vs_program.user_data[3]) << 32);
    const std::array<u32, 4> first_ssbo{
        regs.vs_program.user_data[4], regs.vs_program.user_data[5],
        regs.vs_program.user_data[6], regs.vs_program.user_data[7]};
    if (std::bit_cast<std::array<u32, 4>>(vs.buffers[0].GetSharp(vs)) != first_ssbo)
        return 0;
    const ng::InstancedRunInputs input{
        regs.reg_array, AmdGpu::Regs::ShRegWordOffset,
        AmdGpu::Regs::UconfigRegWordOffset, std::size_t(pointer_word),
        std::size_t(ssbo_word), regs.index_buffer_type.raw,
        {first_fetch, first_ssbo, regs.index_base_address.Address<VAddr>(),
         regs.num_indices, regs.max_index_size, regs.draw_initiator,
         regs.num_instances.NumInstances(), 0}};
    const auto run = ng::CollectInstancedRun(following_commands, input, 64, packet_budget);
    if (run.draws.size() < 2) return 0;
    const auto reject = [](const char* reason) -> std::size_t {
        static thread_local unsigned reports{};
        if (reports++ < 8) std::fprintf(stderr, "NATIVE_INSTANCED_BATCH fallback=%s\n", reason);
        return 0;
    };

    const auto mapping_epoch = memory->MappingEpoch();
    const auto buffer_generation = buffer_cache.MappingGeneration();
    const auto upload_epoch = VideoCore::CurrentUploadEpoch();
    const auto host_generation = VideoCore::CurrentHostWriteGen();
    const auto unchanged = [&] {
        return mapping_epoch == memory->MappingEpoch() &&
               buffer_generation == buffer_cache.MappingGeneration() &&
               upload_epoch == VideoCore::CurrentUploadEpoch() &&
               host_generation == VideoCore::CurrentHostWriteGen();
    };
    std::vector<ng::InstancedDraw> draws(run.draws.size());
    std::vector<ng::MemoryRead> requests;
    std::vector<ng::MemoryRead> source_reads;
    std::vector<ng::Bytes*> destinations;
    requests.reserve(run.draws.size() * 3);
    source_reads.reserve(run.draws.size() * 4);
    destinations.reserve(run.draws.size() * 3);
    const auto queue_read = [&](u64 address, u64 size, ng::Bytes& destination) {
        if (!address || !size || !memory->IsValidGpuMapping(address, size)) return false;
        destination = {nullptr, std::size_t(size)};
        requests.push_back({address, size});
        source_reads.push_back({address, size});
        destinations.push_back(&destination);
        return true;
    };
    std::vector<Core::MemoryManager::BackingSpan> read_backings;
    read_backings.reserve(run.draws.size() * 4 + 16);
    // The host shadow is shared by physical aliases. An alias VA can own newer
    // GPU data even when HasGpuData(original VA) is false. Registered images are
    // also rejected: BufferCache alone does not track their GPU-written bytes.
    const auto cpu_current_backing = [&](u64 address, u64 size)
        -> std::optional<Core::MemoryManager::BackingSpan> {
        if (!address || !size || address > UINT64_MAX - size) return std::nullopt;
        const auto generation = buffer_cache.MappingGeneration();
        const auto epoch = memory->MappingEpoch();
        if (!generation || !epoch) return std::nullopt;
        const auto backing = memory->DescribeBacking(address, size);
        const auto aliases = memory->EnumerateBackingAliases(address, size);
        if (!backing || backing->size != size || !aliases) return std::nullopt;
        bool found_original = false;
        for (const auto& alias : *aliases) {
            if (!alias.size || alias.physical_start < backing->physical_start ||
                alias.physical_start - backing->physical_start > backing->size ||
                alias.size > backing->size -
                    (alias.physical_start - backing->physical_start) ||
                alias.guest_address > UINT64_MAX - alias.size ||
                buffer_cache.HasGpuData(alias.guest_address, alias.size) ||
                texture_cache.ProbeNoRegisteredImage(alias.guest_address, alias.size).result !=
                    iss::g4::NegativeImageResult::NoRegisteredPage)
                return std::nullopt;
            found_original |= alias.guest_address == address &&
                              alias.physical_start == backing->physical_start &&
                              alias.size == size;
        }
        if (!found_original || buffer_cache.MappingGeneration() != generation ||
            memory->MappingEpoch() != epoch) return std::nullopt;
        return backing;
    };
    const auto copy_cpu = [&](u64 address, u64 size, void* destination) {
        if (!size || size > UINT32_MAX || memory->ClampRangeSize(address, size) != size)
            return false;
        const auto before = cpu_current_backing(address, size);
        if (!before) return false;
        memory->CopySparseMemory(address, static_cast<u8*>(destination), size);
        const auto after = cpu_current_backing(address, size);
        if (!after || after->physical_start != before->physical_start ||
            after->size != before->size || !unchanged()) return false;
        read_backings.push_back(*after);
        return true;
    };
    const auto sources_current = [&] {
        const auto image_generation = texture_cache.Generation();
        for (const auto& read : source_reads)
            if (!cpu_current_backing(read.address, read.size)) return false;
        return image_generation == texture_cache.Generation() && unchanged();
    };
    // The PM4 parser proves all setters equal to the first draw except VS fetch
    // pointer, SSBO0 descriptor, NumInstances and DrawIndex2 arguments.
    std::vector<std::byte> canonical;
    canonical.reserve(sizeof(regs) + sizeof(GraphicsPipelineKey) + 128);
    const auto append_state = [&](const void* data, std::size_t size) {
        if (!size) return;
        const auto* bytes = static_cast<const std::byte*>(data);
        canonical.insert(canonical.end(), bytes, bytes + size);
    };
    append_state(&regs, sizeof(regs));
    append_state(&pipeline->GetGraphicsKey(), sizeof(GraphicsPipelineKey));
    append_state(&DrawCbExtents(), sizeof(DrawCbExtents()));
    append_state(&DrawDbExtent(), sizeof(DrawDbExtent()));
    append_state(fs.FlatUserData().data(), fs.FlatUserData().size_bytes());
    const ng::StateSnapshot state{{canonical.data(), canonical.size()}, true, true};

    for (std::size_t n = 0; n < run.draws.size(); ++n) {
        const auto& source = run.draws[n];
        auto& draw = draws[n];
        draw.vertex_hash = u32(vs.pgm_hash);
        draw.fragment_hash = u32(fs.pgm_hash);
        draw.graphics_pipeline_key_hash = key_hash;
        draw.vertex_permutation = 0;
        draw.exact_spirv_verified = true;
        draw.primitive_type = 4;
        draw.index_type = 0;
        draw.instance_count = source.instances;
        draw.binding_count = draw.attribute_count = 1;
        draw.index_count = source.index_count;
        draw.state = state;
        std::array<u32, 4> descriptor{};
        if (!copy_cpu(source.fetch_descriptors, sizeof(descriptor), descriptor.data()))
            return reject("fetch_descriptor");
        source_reads.push_back({source.fetch_descriptors, sizeof(descriptor)});
        AmdGpu::Buffer sharp{};
        std::memcpy(&sharp, descriptor.data(), sizeof(sharp));
        const auto& attribute = fetch.attributes[0];
        sharp.base_address += attribute.inst_offset;
        if (attribute.data_format) {
            sharp.data_format = attribute.data_format;
            sharp.num_format = attribute.num_format;
        }
        const auto vertex_raw = std::bit_cast<std::array<u32, 4>>(sharp);
        const u64 vertex_size = sharp.GetSize();
        if (sharp.GetStride() != 8 || vertex_size == 0 || vertex_size > 64 * 1024 ||
            vertex_raw[1] != 524290 || vertex_raw[3] != 418732) return reject("vsharp_layout");
        draw.vertex = {{nullptr, std::size_t(vertex_size)}, sharp.base_address,
            u32(vertex_size), sharp.GetStride(), vertex_raw[1], vertex_raw[3],
            attribute.sgpr_base, attribute.dword_offset, u32(attribute.instance_data),
            true, true, true};
        if (!queue_read(sharp.base_address, vertex_size, draw.vertex.bytes))
            return reject("vertex_range");
        const AmdGpu::Buffer ssbo = std::bit_cast<AmdGpu::Buffer>(source.ssbo_raw);
        const u64 ssbo_size = ssbo.GetSize();
        if (!ssbo.base_address || ssbo.GetStride() != 72 ||
            source.ssbo_raw[1] != 4718592 ||
            source.ssbo_raw[3] != 537031172 ||
            ssbo_size != u64(source.instances) * 72 ||
            ssbo_size > 64 * 1024) return reject("ssbo_layout");
        draw.instance_guest_base = ssbo.base_address;
        draw.instance_cpu_owned = draw.instance_mapped = draw.instance_snapshot_stable = true;
        if (!queue_read(ssbo.base_address, ssbo_size, draw.instance_bytes))
            return reject("ssbo_range");
        if (source.index_count > 32768 ||
            !queue_read(source.index_address, u64(source.index_count) * 2,
                        draw.index_bytes)) return reject("index_range");
        draw.index_cpu_owned = draw.index_mapped = draw.index_snapshot_stable = true;
    }
    const auto plan = ng::PlanReadSnapshot(requests);
    if (!plan) return reject("snapshot_capacity_or_bounds");
    std::vector<std::byte> snapshot(plan->size);
    for (const auto& range : plan->ranges) {
        if (!copy_cpu(range.address, range.size, snapshot.data() + range.offset))
            return reject("geometry_snapshot");
    }
    for (std::size_t i = 0; i < destinations.size(); ++i)
        destinations[i]->data = snapshot.data() + plan->offsets[i];
    auto batch = ng::BuildInstanced(draws);
    if (!batch || !sources_current()) return reject("batch_validation_or_generation");
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    if (attributes.size() != 1 || bindings.size() != 1) return reject("vertex_layout");
    static thread_local const Instance* caps_owner{};
    static thread_local std::optional<ng::InstancedDeviceCaps> caps;
    if (caps_owner != &instance) {
        caps = ng::InstancedDeviceCaps::Capture(instance);
        caps_owner = &instance;
    }
    if (!caps->Supports(instance, batch.batch.draw_count, batch.batch.instances.size()))
        return reject("device_caps");

    PrepareRenderState(pipeline);
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        if (cb_descs[i].first && texture_cache.IsMetaCleared(
            regs.color_buffers[i].CmaskAddress(), cb_descs[i].second.view_info.range.base.layer)) {
            ResetBindings(false); return 0;
        }
    }
    if (db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
        db_desc.second.view_info.range.base.layer)) { ResetBindings(false); return 0; }
    const auto target_count = bound_images.size();
    if (!BindResources(pipeline)) { ResetBindings(false); return 0; }
    // PS0 declares no images. No sampled or storage image may appear here.
    if (bound_images.size() != target_count) { ResetBindings(false); return 0; }
    for (std::size_t i = 0; i < target_count; ++i) {
        const auto& image = texture_cache.GetImage(bound_images[i]);
        const auto target = memory->DescribeBacking(image.info.guest_address, image.info.guest_size);
        if (!target || target->size != image.info.guest_size) { ResetBindings(false); return 0; }
        for (const auto& source : read_backings) {
            if (source.physical_start < target->physical_start + target->size &&
                target->physical_start < source.physical_start + source.size) {
                ResetBindings(false); return 0;
            }
        }
    }
    if (!sources_current() || scheduler.HasPriorPendingOperations()) {
        ResetBindings(false); return 0;
    }
    const auto render_state = BeginRendering(pipeline);
    if (needs_barrier) runtime.FlushBarriers();
    UpdateDynamicState(pipeline, true);
    scheduler.BeginRendering(render_state);
    if (attachment_feedback_loop || !sources_current() ||
        scheduler.HasPriorPendingOperations()) {
        ResetBindings(false); return 0;
    }
    ng::InstancedVulkanRecordContext context{
        instance, *caps, scheduler, buffer_cache.GetStreamBuffer(), *pipeline,
        set_writes, push_data};
    context.vertex_binding = bindings[0];
    context.vertex_attribute = attributes[0];
    context.proof.complete_state_equal_at_record = true;
    context.proof.no_intervening_feedback_or_barrier = true;
    context.opt_in_enabled = context.rendering_scope_active =
        context.dynamic_state_bound = context.exact_spirv_verified = true;
    const auto status = ng::TryUploadAndRecordInstanced(batch, context);
    ResetBindings(false);
    if (status != ng::InstancedRecordStatus::Recorded) return reject("record_preflight");

    const auto& last = run.draws.back();
    regs.vs_program.user_data[2] = u32(last.fetch_descriptors);
    regs.vs_program.user_data[3] = u32(last.fetch_descriptors >> 32);
    for (u32 i = 0; i < 4; ++i) regs.vs_program.user_data[4 + i] = last.ssbo_raw[i];
    regs.num_instances.num_instances = last.instances;
    regs.index_base_address.base_addr_lo = u32(last.index_address);
    regs.index_base_address.base_addr_hi = u32(last.index_address >> 32);
    regs.num_indices = last.index_count;
    regs.max_index_size = last.max_index_size;
    regs.draw_initiator = last.initiator;
    for (std::size_t i = 0; i < run.draws.size(); ++i) DebugState.IncDrawCall();
    static thread_local u64 batches{}, saved_draws{};
    ++batches;
    saved_draws += run.draws.size() - 1;
    if (batches == 1 || (batches & 1023) == 0)
        std::fprintf(stderr, "NATIVE_INSTANCED_BATCH groups=%llu draws_combined=%llu\n",
                     static_cast<unsigned long long>(batches),
                     static_cast<unsigned long long>(saved_draws));
    return run.consumed_words;
}
