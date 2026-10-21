// SPDX-License-Identifier: GPL-2.0-or-later
// Included in namespace Vulkan. The CPU collector substitutes only one bounded,
// synchronous run: no pending draw survives this call or crosses a PM4 boundary.
std::size_t Rasterizer::TryNativeIndexedRun(const GraphicsPipeline* pipeline,
                                           std::span<const u32> following_commands, std::size_t packet_budget) {
    namespace ng = iss::native_geometry;
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_INDEXED_BATCH");
        const bool on = value && value[0] == '1';
        if (on) std::fputs("NATIVE_INDEXED_BATCH enabled: exact pipeline, bounded DCB runs\n", stderr);
        return on;
    }();
    if (!enabled || cp_split_enabled || draw_pipe_state || OnDrawWorker() || DrawQueue() != 0 ||
        iss::command_trace::enabled || IsDumping() || host_markers_enabled ||
        guest_markers_enabled || GpuCrashDiag::Enabled() || EmulatorSettings.IsPatchShaders() ||
        scheduler.HasPriorPendingOperations()) return 0;
    auto& regs = DrawRegs();
    if (regs.num_instances.NumInstances() != 1 || u32(regs.primitive_type) != 4 ||
        regs.index_buffer_type.raw != 0 || regs.enable_primitive_restart ||
        regs.clipper_control.user_clip_plane_enable || regs.depth_render_control.depth_clear_enable ||
        regs.depth_render_control.stencil_clear_enable) return 0;
    const auto key_hash = std::hash<GraphicsPipelineKey>{}(pipeline->GetGraphicsKey());
    if (key_hash != 0x6794975e42e32374ULL ||
        pipeline->GetGraphicsKey().stage_hashes[u32(Shader::SwStage::Vertex)] != 0xbf5de0e198ULL)
        return 0;
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fs = pipeline->GetStage(Shader::SwStage::Fragment);
    const auto& fetch = pipeline->GetFetchShader();
    if (vs.pgm_hash != 0x0beb7156 || fs.pgm_hash != 0x6353e48b ||
        fetch.attributes.size() != 3 || fetch.size != 48 || vs.buffers.size() != 3 ||
        fs.buffers.size() != 5 || !vs.images.empty() || fs.images.size() != 5) return 0;
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, fetch);
    if (vertex_offset != 0 || instance_offset != 0) return 0;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) continue;
        if ((stage != &vs && stage != &fs) || stage->uses_dma || !stage->fmasks.empty()) return 0;
        for (const auto& desc : stage->buffers) {
            if (desc.is_written || (desc.IsSpecial() &&
                desc.buffer_type != Shader::BufferType::ClipPlanes)) return 0;
        }
        for (const auto& desc : stage->images) if (desc.is_written || desc.is_atomic) return 0;
    }
    const auto pointer_word = reinterpret_cast<const u32*>(&regs.vs_program.user_data[2]) -
                              regs.reg_array.data();
    const u64 first_fetch = u64{regs.vs_program.user_data[2]} |
                             (u64{regs.vs_program.user_data[3]} << 32);
    const ng::RunInputs input{regs.reg_array, AmdGpu::Regs::ShRegWordOffset,
        AmdGpu::Regs::UconfigRegWordOffset, static_cast<std::size_t>(pointer_word),
        regs.index_buffer_type.raw, regs.num_instances.num_instances,
        {first_fetch, regs.index_base_address.Address<VAddr>(), regs.num_indices,
         regs.max_index_size, regs.draw_initiator, 0}};
    const auto run = ng::CollectIndexedRun(following_commands, input, 64, packet_budget);
    if (run.draws.size() < 2) return 0;
    static thread_local bool observed_run{};
    if (!observed_run) {
        observed_run = true;
        std::fprintf(stderr, "NATIVE_INDEXED_BATCH first_run_draws=%zu\n", run.draws.size());
    }
    const auto reject = [](const char* reason) -> std::size_t {
        static thread_local unsigned reports{};
        if (reports++ < 8) std::fprintf(stderr, "NATIVE_INDEXED_BATCH fallback=%s\n", reason);
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
    std::vector<ng::Draw> draws(run.draws.size());
    std::vector<ng::MemoryRead> requests;
    std::vector<ng::Bytes*> destinations;
    requests.reserve(run.draws.size() * 4);
    destinations.reserve(run.draws.size() * 4);
    const auto queue_read = [&](u64 address, u64 size, ng::Bytes& destination) {
        if (!address || !size || !memory->IsValidGpuMapping(address, size)) return false;
        destination = {nullptr, static_cast<std::size_t>(size)};
        requests.push_back({address, size});
        destinations.push_back(&destination);
        return true;
    };
    std::vector<Core::MemoryManager::BackingSpan> read_backings;
    read_backings.reserve(run.draws.size() * 5 + 16);
    const auto remember_read = [&](u64 address, u64 size) {
        const auto backing = memory->DescribeBacking(address, size);
        if (!backing || backing->size != size) return false;
        read_backings.push_back(*backing);
        return true;
    };
    for (const auto* stage : pipeline->GetStages()) if (stage) {
        for (const auto& desc : stage->buffers) if (!desc.IsSpecial()) {
            const auto sharp = desc.GetSharp(*stage);
            const auto size = sharp.GetSize();
            if (!sharp.base_address || !size || size > 16384 ||
                buffer_cache.HasGpuData(sharp.base_address, size) ||
                !remember_read(sharp.base_address, size)) return reject("material_buffer");
        }
    }
    // This interned state is immutable until record. CollectIndexedRun proves every accepted
    // setter is equal to these registers except fetch-list words 2/3. No material/attachment,
    // resource pointer or synchronization command can occur inside the accepted run.
    std::vector<std::byte> canonical;
    canonical.reserve(sizeof(regs) + 4096);
    const auto append_state = [&](const void* data, std::size_t size) {
        if (!size) return;
        const auto* bytes = static_cast<const std::byte*>(data);
        canonical.insert(canonical.end(), bytes, bytes + size);
    };
    append_state(&regs, sizeof(regs));
    append_state(&pipeline->GetGraphicsKey(), sizeof(GraphicsPipelineKey));
    append_state(&DrawCbExtents(), sizeof(DrawCbExtents()));
    append_state(&DrawDbExtent(), sizeof(DrawDbExtent()));
    for (const auto* stage : pipeline->GetStages()) if (stage) {
        append_state(stage->FlatUserData().data(), stage->FlatUserData().size_bytes());
        for (const auto& desc : stage->buffers) if (!desc.IsSpecial()) {
            const auto sharp = desc.GetSharp(*stage); append_state(&sharp, sizeof(sharp));
        }
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(*stage); append_state(&sharp, sizeof(sharp));
        }
        for (const auto& desc : stage->samplers) {
            const auto sharp = desc.GetSharp(*stage); append_state(&sharp, sizeof(sharp));
        }
    }
    const ng::StateSnapshot state{{canonical.data(), canonical.size()}, true, true};
    const auto copy_cpu = [&](u64 address, u64 size, void* destination) {
        if (!address || !size || size > UINT32_MAX ||
            memory->ClampRangeSize(address, size) != size || buffer_cache.HasGpuData(address, size) ||
            !remember_read(address, size))
            return false;
        memory->CopySparseMemory(address, static_cast<u8*>(destination), size);
        return unchanged();
    };
    for (std::size_t n = 0; n < run.draws.size(); ++n) {
        const auto& source = run.draws[n];
        auto& d = draws[n];
        d.vertex_hash = u32(vs.pgm_hash); d.fragment_hash = u32(fs.pgm_hash);
        d.graphics_pipeline_key_hash = key_hash; d.vertex_permutation = 0;
        d.primitive_type = 4; d.instance_count = 1;
        d.binding_count = d.attribute_count = 3;
        d.fetch_inputs_only = true;
        // Exact VS permutation 0 was audited: the SPIR-V declares VertexIndex/InstanceIndex
        // but never loads either. Other keys/permutations are rejected above and by Build.
        d.vertex_index_unobserved = true;
        d.index_count = source.index_count; d.state = state;
        std::array<u32, 12> descriptors;
        if (!copy_cpu(source.fetch_descriptors, sizeof(descriptors), descriptors.data())) return reject("fetch_descriptors");
        for (u32 i = 0; i < 3; ++i) {
            const auto& attribute = fetch.attributes[i];
            if (attribute.sgpr_base != 2 || attribute.dword_offset > 8) return 0;
            AmdGpu::Buffer sharp{};
            std::memcpy(&sharp, descriptors.data() + attribute.dword_offset, sizeof(sharp));
            sharp.base_address += attribute.inst_offset;
            if (attribute.data_format) {
                sharp.data_format = attribute.data_format;
                sharp.num_format = attribute.num_format;
            }
            const auto bytes = sharp.GetSize();
            if (!bytes || bytes > 32768) return 0;
            const auto raw = std::bit_cast<std::array<u32, 4>>(sharp);
            d.streams[i] = {{nullptr, bytes}, sharp.base_address,
                static_cast<u32>(bytes), sharp.GetStride(), raw[1], raw[3], attribute.sgpr_base,
                attribute.dword_offset, static_cast<u32>(attribute.instance_data), true, true, true};
            if (!queue_read(sharp.base_address, bytes, d.streams[i].bytes)) return reject("vertex_range");
        }
        if (source.index_count > 32768) return 0;
        if (!queue_read(source.index_address, std::size_t(source.index_count) * 2, d.index_bytes))
            return reject("index_range");
        d.index_cpu_owned = d.index_mapped = d.index_snapshot_stable = true;
    }
    // Fetch streams often overlap (two attributes share an interleaved stream). Snapshot
    // each union once, without copying gaps, then point every draw's views into that immutable
    // storage. Sorting memory reads does not reorder the draws passed to Build.
    const auto plan = ng::PlanReadSnapshot(requests);
    if (!plan) return reject("snapshot_capacity_or_bounds");
    std::vector<std::byte> snapshot(plan->size);
    for (const auto& range : plan->ranges)
        if (!copy_cpu(range.address, range.size, snapshot.data() + range.offset))
            return reject("geometry_snapshot");
    for (std::size_t i = 0; i < destinations.size(); ++i)
        destinations[i]->data = snapshot.data() + plan->offsets[i];
    auto batch = ng::Build(draws);
    if (!batch || !unchanged()) return reject("batch_validation_or_generation");
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    if (attributes.size() != 3 || bindings.size() != 3) return 0;
    static thread_local const Instance* caps_owner{};
    static thread_local std::optional<ng::DeviceCaps> caps;
    if (caps_owner != &instance) { caps = ng::DeviceCaps::Capture(instance); caps_owner = &instance; }
    if (!caps->Supports(instance, batch.batch.draw_count)) return 0;

    PrepareRenderState(pipeline);
    // Keep clear-producing draws on the original path; a failed batch must not consume
    // a one-shot meta clear before the normal DrawRecord gets to see it.
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        if (cb_descs[i].first && texture_cache.IsMetaCleared(regs.color_buffers[i].CmaskAddress(),
            cb_descs[i].second.view_info.range.base.layer)) { ResetBindings(false); return 0; }
    }
    if (db_desc.first && texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(),
        db_desc.second.view_info.range.base.layer)) { ResetBindings(false); return 0; }
    const auto target_count = bound_images.size();
    if (!BindResources(pipeline)) { ResetBindings(false); return 0; }
    for (std::size_t i = target_count; i < bound_images.size(); ++i) {
        const auto& image = texture_cache.GetImage(bound_images[i]);
        if (image.binding.is_target || image.binding.force_general || image.binding.needs_rebind) {
            ResetBindings(false); return 0;
        }
        if (!remember_read(image.info.guest_address, image.info.guest_size)) {
            ResetBindings(false); return 0;
        }
        for (std::size_t j = 0; j < target_count; ++j) {
            if (image.backing == texture_cache.GetImage(bound_images[j]).backing) {
                ResetBindings(false); return 0;
            }
        }
    }
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
    if (!unchanged() || scheduler.HasPriorPendingOperations()) { ResetBindings(false); return 0; }
    const auto render_state = BeginRendering(pipeline);
    if (needs_barrier) runtime.FlushBarriers();
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, true);
    scheduler.BeginRendering(render_state);
    // BeginRendering can consume a late clear. Once called, always commit the render scope
    // before falling back, so the ordinary draw resumes with Load and the clear is not lost.
    if (attachment_feedback_loop || !unchanged() || scheduler.HasPriorPendingOperations()) {
        ResetBindings(false); return 0;
    }
    ng::VulkanRecordContext record{instance, *caps, scheduler, buffer_cache.GetStreamBuffer()};
    record.pipeline = pipeline->Handle(); record.graphics_pipeline_key_hash = key_hash;
    std::copy(attributes.begin(), attributes.end(), record.attributes.begin());
    std::copy(bindings.begin(), bindings.end(), record.bindings.begin());
    record.proof.complete_state_equal_at_record = true;
    record.proof.no_intervening_feedback_or_barrier = true;
    record.opt_in_enabled = record.rendering_scope_active = record.descriptors_and_dynamic_state_bound = true;
    const auto status = ng::TryUploadAndRecord(batch, record);
    ResetBindings(false);
    if (status != ng::RecordStatus::Recorded) return reject("record_preflight");

    const auto& last = run.draws.back();
    regs.vs_program.user_data[2] = u32(last.fetch_descriptors);
    regs.vs_program.user_data[3] = u32(last.fetch_descriptors >> 32);
    regs.index_base_address.base_addr_lo = u32(last.index_address);
    regs.index_base_address.base_addr_hi = u32(last.index_address >> 32);
    regs.num_indices = last.index_count; regs.max_index_size = last.max_index_size;
    regs.draw_initiator = last.initiator;
    for (std::size_t i = 0; i < run.draws.size(); ++i) DebugState.IncDrawCall();
    static thread_local u64 batches{}, saved_draws{};
    ++batches; saved_draws += run.draws.size() - 1;
    if (batches == 1 || (batches & 1023) == 0)
        std::fprintf(stderr, "NATIVE_INDEXED_BATCH groups=%llu draws_combined=%llu\n",
                     static_cast<unsigned long long>(batches), static_cast<unsigned long long>(saved_draws));
    return run.consumed_words;
}
