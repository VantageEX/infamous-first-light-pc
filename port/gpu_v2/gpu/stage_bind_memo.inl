// SPDX-License-Identifier: GPL-2.0-or-later
// Stage binding memo, included by vk_rasterizer.cpp (Rasterizer members).
//
// BindBuffers/BindTextures resolve every V#/T#/S# of a stage through the buffer and texture
// caches on each draw. Most draws of the city repeat the descriptors of an earlier draw for at
// least one stage (same material, a different mesh). When the descriptors are byte-identical and
// no event that could change a lookup result happened since they were resolved, the resolved
// descriptor writes, push offsets and tracked bindings are replayed instead.
//
// A result is only remembered when resolving it changed nothing observable: the stamp read
// before the stage equals the one read after it (no upload, image transition, recorded write,
// new image or sampler...) and the stage needed no barrier. Replaying it is then what the
// original lookups would return again while the stamp still holds. The descriptor writes of a
// stage do not depend on the pipeline: only on the shader, its first binding indices and the
// descriptors read.

static bool StageMemoFlag(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}

static bool StageMemoEnabled() {
    static const bool enabled = StageMemoFlag("ISS_STAGE_MEMO");
    return enabled;
}

static bool StageMemoVerify() {
    static const bool verify = StageMemoFlag("ISS_STAGE_MEMO_VERIFY");
    return verify;
}

// ISS_STAGE_MEMO_LOOSE=1: the stage memo ignores Rasterizer::bind_epoch. That counter exists
// for the fast draw (the recorded command buffer state: pipeline, push descriptors, render
// pass); a resolved descriptor is invalidated by what the other fields count (uploads, CPU and
// GPU writes, barriers and transitions, mappings, arenas, images, samplers).
static bool StageMemoLoose() {
    static const bool loose = StageMemoFlag("ISS_STAGE_MEMO_LOOSE");
    return loose;
}

// ISS_STAGE_MEMO_TEXEL=1: stages reading formatted (texel) buffers are remembered too.
static bool StageMemoTexel() {
    static const bool texel = StageMemoFlag("ISS_STAGE_MEMO_TEXEL");
    return texel;
}

static bool StageMemoCycles() {
    static const bool cycles = StageMemoFlag("ISS_STAGE_MEMO_CYCLES");
    return cycles;
}

Rasterizer::StageMemo& Rasterizer::StageMemoEntry(u32 slot, const Shader::Info& stage,
                                                  const Shader::Backend::Bindings& binding) {
    if (!stage_memo_table) {
        stage_memo_table = static_cast<StageMemo*>(
            std::calloc(size_t{Shader::MaxStageTypes} * StageMemoEntries, sizeof(StageMemo)));
        ASSERT_MSG(stage_memo_table, "stage memo table allocation failed");
    }
    if (memo_key_entry && memo_key_stage == &stage && memo_key_binding == binding) {
        // The store after the lookup of this stage: the same descriptors.
        return *memo_key_entry;
    }
    // The same shader is drawn with many materials. The user data point to per draw memory, the
    // material is what the descriptors read from it are: the entry is chosen by their content.
    u64 key = reinterpret_cast<u64>(&stage) ^ (u64{binding.unified} << 48) ^
              (u64{binding.buffer} << 56);
    const auto mix = [&key](const void* data, size_t size) {
        // Sharps are 16 or 32 bytes: 64 bit words halve the dependent multiplications.
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i < size / 8; ++i) {
            u64 word;
            std::memcpy(&word, bytes + i * 8, sizeof(word));
            key = (key ^ word) * 0x100000001B3ULL;
            key ^= key >> 32;
        }
    };
    // Buffers are left out: the per object constants move every draw (they are resolved again
    // on a hit, see ReplayStageMemo).
    const u32 num_images = std::min<u32>(stage.images.size(), StageMemo::MaxImages);
    for (u32 i = 0; i < num_images; ++i) {
        const auto sharp = stage.images[i].GetSharp(stage);
        mix(&sharp, sizeof(sharp));
    }
    const u32 num_samplers = std::min<u32>(stage.samplers.size(), StageMemo::MaxSamplers);
    for (u32 i = 0; i < num_samplers; ++i) {
        const auto sharp = stage.samplers[i].GetSharp(stage);
        mix(&sharp, sizeof(sharp));
    }
    key *= 0x9E3779B97F4A7C15ULL;
    auto& entry =
        stage_memo_table[size_t{slot} * StageMemoEntries + (key >> (64 - StageMemoBits))];
    memo_key_stage = &stage;
    memo_key_binding = binding;
    memo_key_entry = &entry;
    return entry;
}

Rasterizer::StageMemoStamp Rasterizer::CaptureStageMemoStamp() const noexcept {
    return {
        .upload_epoch = VideoCore::CurrentUploadEpoch(),
        .host_write_gen = VideoCore::CurrentHostWriteGen(),
        .mapping_gen = buffer_cache.MappingGeneration(),
        .arena_gen = buffer_cache.ArenaBindingGeneration(),
        .gpu_write_seq = buffer_cache.GpuWriteSeq(),
        .cpu_dirty_seq = VideoCore::g_cpu_dirty_seq.load(std::memory_order_acquire),
        .hazard_seq = runtime.HazardSeq(),
        .bind_epoch = bind_epoch.load(std::memory_order_acquire),
        .tex_gen = texture_cache.Generation(),
        .sampler_gen = texture_cache.SamplersGeneration(),
    };
}

bool Rasterizer::StageMemoContextEligible() const noexcept {
    static const bool observers = StageMemoFlag("ISS_FEEDBACK_IDENTITY") ||
                                  StageMemoFlag("ISS_CP_OWNER_QID1_STAGE1");
    // The draw pipe worker owns the caches while it records (the command processor drains it
    // before any other access): ISS_DRAW_PIPE_FAST lets it use the memo.
    static const bool pipe_fast = StageMemoFlag("ISS_DRAW_PIPE_FAST");
    if (observers || cp_split_enabled || (OnDrawWorker() && !pipe_fast) || DrawQueue() != 0 ||
        active_record_context != nullptr || active_runtime != &runtime ||
        active_scheduler != &scheduler || buffer_cache.HasComputeQueue() ||
        iss::hazard_probe::Enabled() || g_num_traced_ranges != 0 || g_dump.frames_left != 0 ||
        iss::feedback_join::enabled || iss::command_trace::enabled) {
        return false;
    }
    return Common::Perf::GetTuning().trace_shaders[0].load(std::memory_order_relaxed) == 0;
}

bool Rasterizer::StageMemoStageEligible(const Shader::Info& stage) noexcept {
    if (stage.buffers.size() > StageMemo::MaxBuffers || stage.images.size() > StageMemo::MaxImages ||
        stage.samplers.size() > StageMemo::MaxSamplers) {
        return false;
    }
    // Formatted (texel) buffers read only (ISS_STAGE_MEMO_TEXEL): their lookup only differs by
    // SynchronizeMemoryFromImage, which copies an image or clears a meta range starting at the
    // buffer's base. That changes the stamp during the lookup (the result is not stored), and an
    // image created there later changes the texture generation (stamp).
    for (const auto& desc : stage.buffers) {
        if (desc.is_written || (desc.is_formatted && !StageMemoTexel())) {
            return false;
        }
        if (desc.IsSpecial() && desc.buffer_type != Shader::BufferType::Flatbuf &&
            desc.buffer_type != Shader::BufferType::ClipPlanes) {
            return false;
        }
    }
    for (const auto& desc : stage.images) {
        if (desc.is_written || desc.is_atomic) {
            return false;
        }
    }
    return true;
}

bool Rasterizer::ReplayStageMemo(u32 slot, const Shader::Info& stage, const Pipeline* pipeline,
                                 Shader::Backend::Bindings& binding, Shader::PushData& push_data,
                                 bool verify_only) {
    auto& stats = stage_memo_stats;
    ++stats.checks;
    ++stats.slot_checks[slot & 7];
    memo_key_entry = nullptr;
    const auto& memo = StageMemoEntry(slot, stage, binding);
    if (!memo.valid) {
        ++stats.no_memo;
        return false;
    }
    // The program base only matters through descriptors offset by it, compared below.
    if (memo.info != &stage) {
        ++stats.other_info;
        ++stats.slot_info[slot & 7];
        return false;
    }
    if (memo.binding_in != binding) {
        ++stats.other_binding;
        return false;
    }
    auto now = CaptureStageMemoStamp();
    const bool only_bind_epoch = [&] {
        auto same_epoch = now;
        same_epoch.bind_epoch = memo.stamp.bind_epoch;
        return now != memo.stamp && same_epoch == memo.stamp;
    }();
    if (only_bind_epoch && StageMemoLoose()) {
        now.bind_epoch = memo.stamp.bind_epoch;
    }
    if (now != memo.stamp) {
        ++stats.stamp;
        ++stats.slot_stamp[slot & 7];
        stats.stamp_only_bind_epoch += only_bind_epoch ? 1 : 0;
        const auto& a = memo.stamp;
        stats.stamp_fields[0] += a.upload_epoch != now.upload_epoch;
        stats.stamp_fields[1] += a.host_write_gen != now.host_write_gen;
        stats.stamp_fields[2] += a.mapping_gen != now.mapping_gen;
        stats.stamp_fields[3] += a.arena_gen != now.arena_gen;
        stats.stamp_fields[4] += a.gpu_write_seq != now.gpu_write_seq;
        stats.stamp_fields[5] += a.cpu_dirty_seq != now.cpu_dirty_seq;
        stats.stamp_fields[6] += a.hazard_seq != now.hazard_seq;
        stats.stamp_fields[7] += a.bind_epoch != now.bind_epoch;
        stats.stamp_fields[8] += a.tex_gen != now.tex_gen;
        stats.stamp_fields[9] += a.sampler_gen != now.sampler_gen;
        return false;
    }
    const auto& regs = DrawRegs();
    if (memo.ta_bc_base != regs.ta_bc_base.Address() ||
        memo.clip_enable != static_cast<u32>(regs.clipper_control.user_clip_plane_enable)) {
        ++stats.sharps;
        return false;
    }
    // Buffers whose V# moved (per object constants) are resolved again below; the others, the
    // images and the samplers must be those remembered.
    u32 changed = 0;
    for (u32 i = 0; i < memo.num_buffers; ++i) {
        const auto& desc = stage.buffers[i];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        if (std::memcmp(&vsharp, &memo.vsharps[i], sizeof(vsharp)) != 0) {
            if (!memo.partial_ok || memo.flatbuf[i] != 0 || desc.is_formatted ||
                desc.is_written) {
                ++stats.sharps;
                return false;
            }
            changed |= 1u << i;
        }
    }
    if ((memo.stream_mask & ~changed) != 0 && memo.tick != scheduler.CurrentTick()) {
        // A stream copy kept is only valid in the command buffer it was made in.
        ++stats.tick;
        return false;
    }
    for (u32 i = 0; i < memo.num_images; ++i) {
        const auto tsharp = stage.images[i].GetSharp(stage);
        if (std::memcmp(&tsharp, &memo.tsharps[i], sizeof(tsharp)) != 0) {
            ++stats.sharps;
            return false;
        }
    }
    for (u32 i = 0; i < memo.num_samplers; ++i) {
        const auto ssharp = stage.samplers[i].GetSharp(stage);
        if (std::memcmp(&ssharp, &memo.ssharps[i], sizeof(ssharp)) != 0) {
            ++stats.sharps;
            return false;
        }
    }
    // Render targets of this draw are known (PrepareRenderState ran): a sampled image that is
    // now a target needs the original feedback loop handling.
    for (u32 i = 0; i < memo.num_bound_images; ++i) {
        const auto& image = texture_cache.GetImage(memo.bound_images[i]);
        if (image.binding.is_target || image.binding.force_general) {
            ++stats.target;
            return false;
        }
    }
    if (buffer_infos.size() + memo.num_buffers > buffer_infos.capacity() ||
        image_infos.size() + memo.num_image_infos > image_infos.capacity() ||
        bound_buffers.size() + memo.num_bound_buffers > bound_buffers.capacity() ||
        bound_images.size() + memo.num_bound_images > bound_images.capacity() ||
        set_write_index + memo.num_writes > set_writes.size()) {
        ++stats.capacity;
        return false;
    }
    if (verify_only) {
        return true;
    }

    const u32 buffer_base = static_cast<u32>(buffer_infos.size());
    const u32 image_base = static_cast<u32>(image_infos.size());
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 i = 0; i < memo.num_buffers; ++i) {
        if (memo.flatbuf[i] == 1) {
            // The flattened user data of this draw, copied as BindBuffers does.
            auto& vk_buffer = buffer_cache.GetStreamBuffer();
            const u32 ubo_size = stage.FlatUserData().size() * sizeof(u32);
            const u64 offset = vk_buffer.Copy(stage.FlatUserData().data(), ubo_size, alignment);
            buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
        } else if (memo.flatbuf[i] == 2) {
            // The clip planes of this draw, copied as BindBuffers does.
            auto& vk_buffer = buffer_cache.GetStreamBuffer();
            std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
            for (u32 p = 0; p < AmdGpu::NUM_CLIP_PLANES; ++p) {
                const auto& plane = regs.clip_user_data[p];
                planes[p * 4 + 0] = std::bit_cast<float>(plane.data_x);
                planes[p * 4 + 1] = std::bit_cast<float>(plane.data_y);
                planes[p * 4 + 2] = std::bit_cast<float>(plane.data_z);
                planes[p * 4 + 3] = std::bit_cast<float>(plane.data_w);
            }
            const u32 ubo_size = static_cast<u32>(sizeof(planes));
            const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
            buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
        } else if (((changed >> i) & 1) != 0) {
            // As BindBuffers resolves a read only buffer.
            const auto& desc = stage.buffers[i];
            const auto vsharp = desc.GetSharp(stage);
            ++stats.resolved_buffers;
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                push_data.buf_offsets[binding.buffer + i] = 0;
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                bool known_unwritten = false;
                const bool reuse_check = Common::Perf::GetTuning().reuse_buffer_checks.load(
                    std::memory_order_relaxed);
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, false, false,
                    reuse_check ? &known_unwritten : nullptr);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                push_data.AddOffset(binding.buffer + i, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                if (buffer != &buffer_cache.GetStreamBuffer()) {
                    bound_buffers.emplace_back(buffer, offset, static_cast<u32>(size), false,
                                               vsharp.base_address);
                    if (!known_unwritten) {
                        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, false);
                    }
                }
            }
            continue;
        } else {
            buffer_infos.push_back(memo.buffer_infos[i]);
        }
        push_data.buf_offsets[binding.buffer + i] = memo.push_offsets[i];
        if (memo.partial_ok && memo.bound_index[i] != 0xFF) {
            const auto& bound = memo.bound_buffers[memo.bound_index[i]];
            bound_buffers.emplace_back(bound.buffer, bound.offset, bound.size, false,
                                       bound.address);
        }
    }
    for (u32 i = 0; i < memo.num_image_infos; ++i) {
        image_infos.push_back(memo.image_infos[i]);
    }
    for (u32 i = 0; i < memo.num_writes; ++i) {
        const auto& write = memo.writes[i];
        auto& set_write = set_writes[set_write_index++];
        if (write.is_buffer) {
            WriteBufferDescriptor(set_write, write.binding,
                                  &buffer_infos[buffer_base + write.info_index]);
        } else {
            WriteImageDescriptor(set_write, write.binding, write.count, write.type,
                                 &image_infos[image_base + write.info_index]);
        }
    }
    if (!memo.partial_ok) {
        // Not mapped per buffer (then nothing changed): all the bound entries remembered.
        for (u32 i = 0; i < memo.num_bound_buffers; ++i) {
            const auto& bound = memo.bound_buffers[i];
            bound_buffers.emplace_back(bound.buffer, bound.offset, bound.size, false,
                                       bound.address);
        }
    }
    const u64 tick = scheduler.CurrentTick();
    for (u32 i = 0; i < memo.num_bound_images; ++i) {
        const auto image_id = memo.bound_images[i];
        bound_images.emplace_back(image_id);
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_bound = 1u;
        image.tick_accessed_last = tick;
    }
    binding = memo.binding_out;
    ++stats.hits;
    stats.partial_hits += changed != 0 ? 1 : 0;
    ++stats.slot_hits[slot & 7];
    return true;
}

void Rasterizer::RememberStageMemo(u32 slot, const Shader::Info& stage, const Pipeline* pipeline,
                                   const StageMemoCapture& capture,
                                   const Shader::Backend::Bindings& binding,
                                   const Shader::PushData& push_data, bool verify) {
    auto& stats = stage_memo_stats;
    auto& entry = StageMemoEntry(slot, stage, capture.binding_in);
    memo_key_entry = nullptr;
    // Filled in place (only the used elements): stores are as frequent as misses.
    auto& out = verify ? stage_memo_scratch : entry;
    out.valid = false;
    const auto stamp_now = CaptureStageMemoStamp();
    const bool stamp_changed = stamp_now != capture.stamp;
    if (verify && (stamp_changed || needs_barrier)) {
        // The original lookups had an effect (upload, transition, barrier...) a replay skips.
        const auto& a = capture.stamp;
        const auto& b = stamp_now;
        const u32 fields = (a.upload_epoch != b.upload_epoch ? 1u : 0u) |
                           (a.host_write_gen != b.host_write_gen ? 2u : 0u) |
                           (a.mapping_gen != b.mapping_gen ? 4u : 0u) |
                           (a.arena_gen != b.arena_gen ? 8u : 0u) |
                           (a.gpu_write_seq != b.gpu_write_seq ? 16u : 0u) |
                           (a.cpu_dirty_seq != b.cpu_dirty_seq ? 32u : 0u) |
                           (a.hazard_seq != b.hazard_seq ? 64u : 0u) |
                           (a.bind_epoch != b.bind_epoch ? 128u : 0u) |
                           (a.tex_gen != b.tex_gen ? 256u : 0u) |
                           (a.sampler_gen != b.sampler_gen ? 512u : 0u);
        // Guest threads change the upload epoch, CPU modified pages and invalidations at any
        // time: a change of only those during the lookups is concurrent, not caused by them.
        constexpr u32 concurrent = 1u | 2u | 32u | 128u;
        ++stats.verified;
        if (needs_barrier || (fields & ~concurrent) != 0) {
            ++stats.mismatches;
            if (stats.mismatches <= 64) {
                std::fprintf(stderr,
                             "STAGE_MEMO_MISMATCH what=%s fields=0x%x stage=0x%llx slot=%u\n",
                             needs_barrier ? "barrier" : "side_effect", fields,
                             static_cast<unsigned long long>(stage.pgm_hash), slot);
            }
        } else {
            ++stats.concurrent;
        }
        return;
    }
    if (stamp_changed) {
        ++stats.reject_stamp;
        return;
    }
    if (needs_barrier) {
        ++stats.reject_barrier;
        return;
    }
    const u32 num_buffer_infos = static_cast<u32>(buffer_infos.size()) - capture.buffer_infos;
    const u32 num_image_infos = static_cast<u32>(image_infos.size()) - capture.image_infos;
    const u32 num_writes = set_write_index - capture.writes;
    const u32 num_bound_buffers = static_cast<u32>(bound_buffers.size()) - capture.bound_buffers;
    const u32 num_bound_images = static_cast<u32>(bound_images.size()) - capture.bound_images;
    if (num_buffer_infos != stage.buffers.size() || num_image_infos > StageMemo::MaxImageInfos ||
        num_writes > StageMemo::MaxWrites || num_bound_buffers > StageMemo::MaxBuffers ||
        num_bound_images > StageMemo::MaxImageInfos) {
        ++stats.reject_capacity;
        return;
    }
    const auto& regs = DrawRegs();
    out.ta_bc_base = regs.ta_bc_base.Address();
    out.clip_enable = static_cast<u32>(regs.clipper_control.user_clip_plane_enable);
    out.num_buffers = static_cast<u8>(stage.buffers.size());
    out.num_images = static_cast<u8>(stage.images.size());
    out.num_samplers = static_cast<u8>(stage.samplers.size());
    out.uses_stream = false;
    out.stream_mask = 0;
    const auto stream_handle = buffer_cache.GetStreamBuffer().Handle();
    for (u32 i = 0; i < out.num_buffers; ++i) {
        const auto& desc = stage.buffers[i];
        const bool clip_copy =
            desc.buffer_type == Shader::BufferType::ClipPlanes && out.clip_enable != 0;
        out.flatbuf[i] = desc.buffer_type == Shader::BufferType::Flatbuf ? 1 : clip_copy ? 2 : 0;
        if (!desc.IsSpecial()) {
            out.vsharps[i] = desc.GetSharp(stage);
        }
        const auto& info = buffer_infos[capture.buffer_infos + i];
        out.buffer_infos[i] = info;
        out.uses_stream |= out.flatbuf[i] == 0 && info.buffer == stream_handle;
        if (out.flatbuf[i] == 0 && info.buffer == stream_handle) {
            out.stream_mask |= 1u << i;
        }
        out.push_offsets[i] = push_data.buf_offsets[capture.binding_in.buffer + i];
    }
    for (u32 i = 0; i < out.num_images; ++i) {
        out.tsharps[i] = stage.images[i].GetSharp(stage);
    }
    for (u32 i = 0; i < out.num_samplers; ++i) {
        out.ssharps[i] = stage.samplers[i].GetSharp(stage);
    }
    for (u32 i = 0; i < num_image_infos; ++i) {
        out.image_infos[i] = image_infos[capture.image_infos + i];
    }
    for (u32 i = 0; i < num_writes; ++i) {
        const auto& write = set_writes[capture.writes + i];
        auto& rec = out.writes[i];
        rec.binding = write.dstBinding;
        rec.count = write.descriptorCount;
        rec.type = write.descriptorType;
        if (write.pBufferInfo != nullptr) {
            const auto index = write.pBufferInfo - buffer_infos.data();
            if (write.dstArrayElement != 0 || write.descriptorCount != 1 ||
                write.descriptorType != vk::DescriptorType::eStorageBuffer ||
                index < capture.buffer_infos || index >= capture.buffer_infos + num_buffer_infos) {
                ++stats.reject_capacity;
                return;
            }
            rec.is_buffer = true;
            rec.info_index = static_cast<u16>(index - capture.buffer_infos);
        } else {
            const auto index = write.pImageInfo - image_infos.data();
            if (write.dstArrayElement != 0 || index < capture.image_infos ||
                index + write.descriptorCount > capture.image_infos + num_image_infos) {
                ++stats.reject_capacity;
                return;
            }
            rec.is_buffer = false;
            rec.info_index = static_cast<u16>(index - capture.image_infos);
        }
    }
    for (u32 i = 0; i < num_bound_buffers; ++i) {
        const auto& bound = bound_buffers[capture.bound_buffers + i];
        if (bound.is_written) {
            ++stats.reject_capacity;
            return;
        }
        out.bound_buffers[i] = {bound.buffer, bound.offset, bound.size, bound.address};
    }
    // BindBuffers adds a bound entry, in buffer order, for each non special buffer that is not
    // null and not a stream copy.
    {
        u32 next = 0;
        bool ok = true;
        for (u32 i = 0; i < out.num_buffers; ++i) {
            out.bound_index[i] = 0xFF;
            const auto& desc = stage.buffers[i];
            const auto& info = out.buffer_infos[i];
            if (desc.IsSpecial() || !info.buffer || info.buffer == stream_handle) {
                continue;
            }
            if (next >= num_bound_buffers ||
                out.bound_buffers[next].address != out.vsharps[i].base_address) {
                ok = false;
                break;
            }
            out.bound_index[i] = static_cast<u8>(next++);
        }
        out.partial_ok = ok && next == num_bound_buffers;
    }
    for (u32 i = 0; i < num_bound_images; ++i) {
        const auto image_id = bound_images[capture.bound_images + i];
        const auto& image = texture_cache.GetImage(image_id);
        if (image.binding.is_target || image.binding.force_general) {
            ++stats.reject_target;
            return;
        }
        out.bound_images[i] = image_id;
    }
    out.num_image_infos = static_cast<u8>(num_image_infos);
    out.num_writes = static_cast<u8>(num_writes);
    out.num_bound_buffers = static_cast<u8>(num_bound_buffers);
    out.num_bound_images = static_cast<u8>(num_bound_images);
    out.info = &stage;
    out.pipeline = pipeline;
    out.program_base = stage.ProgramBase();
    out.tick = capture.tick;
    out.binding_in = capture.binding_in;
    out.binding_out = binding;
    out.stamp = capture.stamp;
    out.valid = true;
    if (!verify) {
        ++stats.stores;
        return;
    }

    // The original lookups ran where the memo would have been replayed: compare.
    ++stats.verified;
    const auto& memo = entry;
    bool same = memo.valid && memo.num_buffers == out.num_buffers &&
                memo.num_image_infos == out.num_image_infos && memo.num_writes == out.num_writes &&
                memo.num_bound_buffers == out.num_bound_buffers &&
                memo.num_bound_images == out.num_bound_images &&
                memo.binding_out == out.binding_out;
    const char* what = same ? "" : "shape";
    for (u32 i = 0; same && i < out.num_buffers; ++i) {
        const auto& a = memo.buffer_infos[i];
        const auto& b = out.buffer_infos[i];
        const bool streamed = a.buffer == stream_handle && b.buffer == stream_handle;
        if (memo.flatbuf[i] != out.flatbuf[i] || a.range != b.range ||
            (!streamed && (a.buffer != b.buffer || a.offset != b.offset)) ||
            memo.push_offsets[i] != out.push_offsets[i]) {
            same = false;
            what = "buffer";
        }
    }
    for (u32 i = 0; same && i < out.num_image_infos; ++i) {
        if (memo.image_infos[i] != out.image_infos[i]) {
            same = false;
            what = "image";
        }
    }
    for (u32 i = 0; same && i < out.num_writes; ++i) {
        const auto& a = memo.writes[i];
        const auto& b = out.writes[i];
        if (a.binding != b.binding || a.count != b.count || a.type != b.type ||
            a.info_index != b.info_index || a.is_buffer != b.is_buffer) {
            same = false;
            what = "write";
        }
    }
    for (u32 i = 0; same && i < out.num_bound_buffers; ++i) {
        const auto& a = memo.bound_buffers[i];
        const auto& b = out.bound_buffers[i];
        if (a.buffer != b.buffer || a.offset != b.offset || a.size != b.size ||
            a.address != b.address) {
            same = false;
            what = "bound_buffer";
        }
    }
    for (u32 i = 0; same && i < out.num_bound_images; ++i) {
        if (memo.bound_images[i] != out.bound_images[i]) {
            same = false;
            what = "bound_image";
        }
    }
    if (!same) {
        ++stats.mismatches;
        if (stats.mismatches <= 64) {
            std::fprintf(stderr,
                         "STAGE_MEMO_MISMATCH what=%s stage=0x%llx slot=%u memo[b=%u ii=%u w=%u "
                         "bb=%u bi=%u stream=%u out=%u/%u] now[b=%u ii=%u w=%u bb=%u bi=%u "
                         "stream=%u out=%u/%u]\n",
                         what, static_cast<unsigned long long>(stage.pgm_hash), slot,
                         memo.num_buffers, memo.num_image_infos, memo.num_writes,
                         memo.num_bound_buffers, memo.num_bound_images, memo.uses_stream,
                         memo.binding_out.unified, memo.binding_out.buffer, out.num_buffers,
                         out.num_image_infos, out.num_writes, out.num_bound_buffers,
                         out.num_bound_images, out.uses_stream, out.binding_out.unified,
                         out.binding_out.buffer);
        }
    }
}

void Rasterizer::ReportStageMemo() {
    const auto& s = stage_memo_stats;
    std::fprintf(stderr, "STAGE_MEMO_PARTIAL partial_hits=%llu resolved_buffers=%llu\n",
                 (unsigned long long)s.partial_hits, (unsigned long long)s.resolved_buffers);
    std::fprintf(stderr,
                 "STAGE_MEMO_STAMP upload=%llu host_write=%llu mapping=%llu arena=%llu "
                 "gpu_write=%llu cpu_dirty=%llu hazard=%llu bind_epoch=%llu texture=%llu "
                 "sampler=%llu only_bind_epoch=%llu\n",
                 (unsigned long long)s.stamp_fields[0], (unsigned long long)s.stamp_fields[1],
                 (unsigned long long)s.stamp_fields[2], (unsigned long long)s.stamp_fields[3],
                 (unsigned long long)s.stamp_fields[4], (unsigned long long)s.stamp_fields[5],
                 (unsigned long long)s.stamp_fields[6], (unsigned long long)s.stamp_fields[7],
                 (unsigned long long)s.stamp_fields[8], (unsigned long long)s.stamp_fields[9],
                 (unsigned long long)s.stamp_only_bind_epoch);
    for (u32 i = 0; i < 8; ++i) {
        if (s.slot_checks[i] == 0) continue;
        std::fprintf(stderr,
                     "STAGE_MEMO_SLOT slot=%u checks=%llu hits=%llu info=%llu stamp=%llu "
                     "original_cycles=%.0f originals=%llu\n",
                     i, (unsigned long long)s.slot_checks[i], (unsigned long long)s.slot_hits[i],
                     (unsigned long long)s.slot_info[i], (unsigned long long)s.slot_stamp[i],
                     s.slot_originals[i] ? double(s.slot_cyc_original[i]) / double(s.slot_originals[i]) : 0.0,
                     (unsigned long long)s.slot_originals[i]);
    }
    std::fprintf(stderr,
                 "STAGE_MEMO checks=%llu hits=%llu stores=%llu verified=%llu mismatches=%llu "
                 "concurrent=%llu miss[no_memo=%llu info=%llu binding=%llu stamp=%llu tick=%llu "
                 "sharps=%llu target=%llu capacity=%llu] reject[stamp=%llu barrier=%llu "
                 "target=%llu capacity=%llu]\n",
                 (unsigned long long)s.checks, (unsigned long long)s.hits,
                 (unsigned long long)s.stores, (unsigned long long)s.verified,
                 (unsigned long long)s.mismatches, (unsigned long long)s.concurrent,
                 (unsigned long long)s.no_memo, (unsigned long long)s.other_info,
                 (unsigned long long)s.other_binding, (unsigned long long)s.stamp,
                 (unsigned long long)s.tick, (unsigned long long)s.sharps,
                 (unsigned long long)s.target, (unsigned long long)s.capacity,
                 (unsigned long long)s.reject_stamp, (unsigned long long)s.reject_barrier,
                 (unsigned long long)s.reject_target, (unsigned long long)s.reject_capacity);
    if (s.originals != 0 && s.hits != 0) {
        std::fprintf(stderr,
                     "STAGE_MEMO_CYCLES hit=%llu check_miss=%llu original=%llu remember=%llu "
                     "per_hit=%.0f per_original=%.0f\n",
                     (unsigned long long)s.cyc_hit, (unsigned long long)s.cyc_check,
                     (unsigned long long)s.cyc_original, (unsigned long long)s.cyc_remember,
                     double(s.cyc_hit) / double(s.hits),
                     double(s.cyc_original) / double(s.originals));
    }
}
