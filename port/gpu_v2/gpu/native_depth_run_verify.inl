// SPDX-License-Identifier: GPL-2.0-or-later
// Included by native_depth_run.inl (namespace Vulkan). ISS_NATIVE_DEPTH_RUN_VERIFY=1,
// benchmarks (visible allowed: only private copies are drawn): the run is drawn into two private copies of its targets, draw by draw with the
// separate bindings and separately copied records (reference), and as the indirect draw
// (candidate). No real draw is skipped or duplicated; the caller records the real one once.


void Rasterizer::RecordDepthRunMode(u32 mode, const GraphicsPipeline* pipeline, u32 ssbo_write,
                                    u32 ssbo_index, u64 command_offset,
                                    const VideoCore::Buffer* vertex_buffer,
                                    const VideoCore::Buffer* index_buffer) {
    using namespace NativeDepthRun;
    const auto& draws = Draws();
    auto& stream = buffer_cache.GetStreamBuffer();
    const auto cmd = scheduler.Recorder();
    const vk::DeviceSize zero = 0;
    const auto bind_geometry = [&](const VideoCore::Buffer* vb, vk::DeviceSize vb_offset,
                                   const VideoCore::Buffer* ib, vk::DeviceSize ib_offset) {
        const vk::Buffer handle = vb->Handle();
        cmd.bindVertexBuffers(0, 1, &handle, &vb_offset);
        cmd.bindIndexBuffer(ib->Handle(), ib_offset, vk::IndexType::eUint16);
    };
    if (mode == ModeReference) {
        // Each draw with its own bindings and its own copy of its records, as translated.
        for (const auto& d : draws) {
            const auto [data, offset] = stream.Map(d.ssbo_size, instance.StorageMinAlignment());
            ASSERT_MSG(data, "depth run reference records");
            memory->CopySparseMemory(d.ssbo, data, d.ssbo_size);
            stream.Commit();
            Pipeline::DescriptorWrites writes(set_writes.begin(), set_writes.begin() + set_write_index);
            const vk::DescriptorBufferInfo info{stream.Handle(), offset, d.ssbo_size};
            writes[ssbo_write].pBufferInfo = &info;
            auto push = push_data;
            push.buf_offsets[ssbo_index] = 0;
            pipeline->BindResources(writes, push);
            bind_geometry(d.vb, d.vb_offset, d.ib, d.ib_offset);
            cmd.drawIndexed(d.index_count, d.instances, 0, 0, 0);
        }
        return;
    }
    // The shared records region (set_writes), indexed by firstInstance.
    pipeline->BindResources(set_writes, push_data);
    if (mode == ModeIndirect) {
        bind_geometry(vertex_buffer, zero, index_buffer, zero);
        cmd.drawIndexedIndirect(stream.Handle(), command_offset, u32(draws.size()),
                                sizeof(vk::DrawIndexedIndirectCommand));
    } else if (mode == ModeDirectMerged) {
        bind_geometry(vertex_buffer, zero, index_buffer, zero);
        for (const auto& d : draws)
            cmd.drawIndexed(d.index_count, d.instances, u32(d.ib_offset / 2),
                            s32(d.vb_offset / 8), d.first_instance);
    } else {
        for (const auto& d : draws) {
            bind_geometry(d.vb, d.vb_offset, d.ib, d.ib_offset);
            cmd.drawIndexed(d.index_count, d.instances, 0, 0, d.first_instance);
        }
    }
}

void Rasterizer::VerifyDepthRunPair(const GraphicsPipeline* pipeline, RenderState& render,
                                    u32 ssbo_write, u32 ssbo_index, u64 command_offset,
                                    const VideoCore::Buffer* vertex_buffer,
                                    const VideoCore::Buffer* index_buffer, u32 mode) {
    using namespace NativeVisual;
    const auto state = NativeDepthRun::VerifyState(mode);
    const u32 draw_count = u32(NativeDepthRun::Draws().size());
    if (g_iss_frame.load(std::memory_order_relaxed) < 1200 ||
        render.num_layers != 1 || render.num_color_attachments > 8 || attachment_feedback_loop ||
        render.width > 4096 || render.height > 4096 ||
        state->pending.load(std::memory_order_acquire) || state->failed.load() ||
        state->qualified.load() >= 6 || state->attempts.load() >= 32) return;
    const auto* output = std::getenv("ISS_NATIVE_DEPTH_RUN_VERIFY_DIR");
    if (!output || !*output) return;
    static u64 rejected{};
    const auto reject = [&](const char* why) {
        if (++rejected <= 16 || (rejected & 4095) == 0)
            std::fprintf(stderr, "NATIVE_DEPTH_RUN_REJECT reason=%s count=%llu\n", why, rejected);
    };
    const auto& draws = NativeDepthRun::Draws();
    auto work = std::make_shared<Work>();
    work->frame = g_iss_frame.load();
    work->width = render.width;
    work->height = render.height;
    work->prepared_resources = draw_count;
    work->identity = {pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash, draw_count,
                      pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash, 0};
    work->directory = std::filesystem::path(output);
    u64 bytes = 0;
    // Same admission as the material probe: whole single layer/level targets, plain views.
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
        if (backing.state.layout != (depth ? render.depth_stencil_attachment.image_layout
                                           : render.color_attachments[slot].image_layout))
            return false;
        Target target;
        target.source = &image;
        target.saved = backing.state;
        target.view_info = *info;
        target.aspects = image.aspect_mask;
        target.slot = slot;
        target.depth = depth;
        target.name = depth ? "depth" : "color" + std::to_string(slot);
        if (depth) {
            if (ci.format != vk::Format::eD32Sfloat && ci.format != vk::Format::eD32SfloatS8Uint)
                return false;
            target.planes.push_back({vk::ImageAspectFlagBits::eDepth, 4});
            if (ci.format == vk::Format::eD32SfloatS8Uint)
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
            for (u32 i = 0; i < 2; ++i) {
                target.images[i] = std::make_unique<VideoCore::UniqueImage>(instance.GetDevice(),
                                                                            instance.GetAllocator());
                target.images[i]->Create(ci);
                vk::ImageViewUsageCreateInfo usage{.usage = target.source->usage_flags & ci.usage};
                if (!target.view_info.is_storage) usage.usage &= ~vk::ImageUsageFlagBits::eStorage;
                const vk::ImageViewCreateInfo view_ci{
                    .pNext = &usage, .image = target.images[i]->image,
                    .viewType = vk::ImageViewType::e2D, .format = target.view_info.format,
                    .components = target.view_info.mapping,
                    .subresourceRange = {target.aspects, 0, 1, 0, 1}};
                auto [result, view] = instance.GetDevice().createImageViewUnique(view_ci);
                if (result != vk::Result::eSuccess)
                    throw std::runtime_error("paired image view allocation failed");
                target.views[i] = std::move(view);
            }
            target.readback = std::make_unique<VideoCore::Buffer>(
                instance, 0, target.stride * 3, VideoCore::MemoryType::HostCached, "depth run pair");
        }
    } catch (const std::exception& error) {
        state->failed.store(true);
        std::fprintf(stderr, "NATIVE_DEPTH_RUN_ERROR allocation=%s\n", error.what());
        return;
    }
    work->sample = state->attempts.fetch_add(1) + 1;
    state->pending.store(true, std::memory_order_release);
    scheduler.EndRendering(); // the real clear, if any, has already been recorded
    const auto cmd = scheduler.Recorder();
    for (auto& target : work->targets)
        runtime.Transit(target.source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto attachment_access = [](const Target& target) {
        return target.depth ? vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                  vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                            : vk::AccessFlagBits2::eColorAttachmentRead |
                                  vk::AccessFlagBits2::eColorAttachmentWrite;
    };
    for (auto& target : work->targets) {
        CopyOut(cmd, target, target.source->GetImage(), 0, render.width, render.height);
        for (u32 side = 0; side < 2; ++side) {
            Barrier(cmd, target.images[side]->image, target.aspects, vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe, {},
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
            for (const auto& plane : target.planes) {
                const vk::ImageCopy copy{.srcSubresource = {plane.aspect, 0, 0, 1},
                                         .dstSubresource = {plane.aspect, 0, 0, 1},
                                         .extent = {render.width, render.height, 1}};
                cmd.copyImage(target.source->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                              target.images[side]->image, vk::ImageLayout::eTransferDstOptimal,
                              std::span{&copy, 1});
            }
            Barrier(cmd, target.images[side]->image, target.aspects,
                    vk::ImageLayout::eTransferDstOptimal, target.saved.layout,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eAllGraphics, attachment_access(target));
        }
        runtime.Transit(target.source, target.saved.layout, target.saved.pl_stage,
                        target.saved.access_mask);
    }
    runtime.FlushBarriers();
    RenderState resume = render;
    for (auto& attachment : resume.color_attachments) attachment.is_clear = 0;
    resume.depth_stencil_attachment.depth_clear = false;
    resume.depth_stencil_attachment.stencil_clear = false;
    for (u32 side = 0; side < 2; ++side) {
        auto clone_state = resume;
        for (const auto& target : work->targets) {
            if (target.depth) clone_state.depth_stencil_attachment.image_view = *target.views[side];
            else clone_state.color_attachments[target.slot].image_view = *target.views[side];
        }
        scheduler.BeginRendering(clone_state);
        RecordDepthRunMode(side == 0 ? u32(NativeDepthRun::ModeReference) : mode, pipeline,
                           ssbo_write, ssbo_index, command_offset, vertex_buffer, index_buffer);
        scheduler.EndRendering();
        for (const auto& target : work->targets) {
            Barrier(cmd, target.images[side]->image, target.aspects, target.saved.layout,
                    vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllGraphics,
                    attachment_access(target), vk::PipelineStageFlagBits2::eTransfer,
                    vk::AccessFlagBits2::eTransferRead);
            CopyOut(cmd, target, target.images[side]->image, side + 1, render.width, render.height);
        }
    }
    const vk::MemoryBarrier2 host_read{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                       .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                       .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                                       .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host_read});
    // Back to the real targets; the caller binds and records the real draw.
    scheduler.BeginRendering(resume);
    render = resume;
    const char* tag = NativeDepthRun::mode_tags[mode];
    scheduler.DeferPriorityOperation([work = std::move(work), state, tag] {
        Complete(work, state, tag, "depth_run_pixels");
    });
}
