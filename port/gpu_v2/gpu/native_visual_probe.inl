// SPDX-License-Identifier: GPL-2.0-or-later
// Diagnostic-only paired GPU rendering from independently captured plans in ONE guest frame.
// No actual game draw is skipped or duplicated; auxiliary draws write only private images.
namespace NativeVisual {
static bool Enabled() {
    static const bool enabled = iss::native_item::Flag("ISS_NATIVE_VISUAL_PROBE") &&
        iss::native_item::Flag("ISS_NATIVE_VERIFY") && iss::native_item::Flag("ISS_HYBRID_BENCH_HIDDEN");
    return enabled;
}
struct FrameWatch { u64 frame{}, draws{}, previous{}; };
static FrameWatch& Watch() { static thread_local FrameWatch watch; return watch; }
struct State {
    std::atomic<bool> pending{}, failed{};
    std::atomic<u32> attempts{}, completed{}, qualified{};
};
static std::shared_ptr<State> Shared() { static auto state = std::make_shared<State>(); return state; }
static u32 ColorBytes(vk::Format f) {
    switch (f) {
    case vk::Format::eR8Unorm: case vk::Format::eR8Uint: return 1;
    case vk::Format::eR8G8Unorm: case vk::Format::eR16Unorm: case vk::Format::eR16Sfloat: return 2;
    case vk::Format::eR8G8B8A8Unorm: case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm: case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32: case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eB10G11R11UfloatPack32: case vk::Format::eR16G16Sfloat:
    case vk::Format::eR16G16Unorm: case vk::Format::eR16G16Snorm:
    case vk::Format::eR32Sfloat: case vk::Format::eR32Uint: return 4;
    case vk::Format::eR16G16B16A16Sfloat: case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR16G16B16A16Snorm: case vk::Format::eR32G32Sfloat: return 8;
    case vk::Format::eR32G32B32A32Sfloat: return 16;
    default: return 0;
    }
}
struct Plane { vk::ImageAspectFlagBits aspect; u32 bytes; u64 offset{}, size{}; };
struct Target {
    VideoCore::Image* source{};
    VideoCore::Image::State saved{};
    VideoCore::ImageViewInfo view_info{};
    vk::ImageAspectFlags aspects{};
    std::string name;
    u32 slot{};
    bool depth{};
    std::vector<Plane> planes;
    u64 stride{};
    std::array<std::unique_ptr<VideoCore::UniqueImage>, 2> images;
    std::array<vk::UniqueImageView, 2> views;
    std::unique_ptr<VideoCore::Buffer> readback;
};
struct Work {
    std::vector<Target> targets;
    std::filesystem::path directory;
    u64 frame{}, city_draws{};
    u32 sample{}, width{}, height{}, prepared_resources{};
    std::array<u64, 4> identity{};
};
static void Barrier(const CommandRecorder& cmd, vk::Image image, vk::ImageAspectFlags aspects,
                    vk::ImageLayout before, vk::ImageLayout after,
                    vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                    vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage, .srcAccessMask = src_access,
        .dstStageMask = dst_stage, .dstAccessMask = dst_access,
        .oldLayout = before, .newLayout = after,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image, .subresourceRange = {aspects, 0, 1, 0, 1}};
    cmd.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}
static void CopyOut(const CommandRecorder& cmd, const Target& target, vk::Image image,
                    u32 part, u32 width, u32 height) {
    for (const auto& plane : target.planes) {
        const vk::BufferImageCopy copy{
            .bufferOffset = target.stride * part + plane.offset,
            .imageSubresource = {plane.aspect, 0, 0, 1}, .imageExtent = {width, height, 1}};
        cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal,
            target.readback->Handle(), std::span{&copy, 1});
    }
}
static void Complete(const std::shared_ptr<Work>& work, const std::shared_ptr<State>& state,
                     const char* tag = "NATIVE_VISUAL_PAIR", const char* field = "gpu_visual_pixels") {
    u64 differences = 0, changed_reference = 0, changed_candidate = 0, compared = 0;
    bool saved = true, mapped = true;
    std::vector<std::pair<std::string, u64>> dumps;
    const auto prefix = "sample-" + std::to_string(work->sample) + "-frame-" + std::to_string(work->frame);
    for (const auto& target : work->targets)
        mapped &= target.readback->mapped_data.size() >= target.stride * 3;
    // An unmapped readback can't be compared; it never counts as equal or qualified.
    if (!mapped) { saved = false; differences = 1; }
    for (auto& target : work->targets) {
        if (!mapped) break;
        target.readback->Invalidate(0, target.stride * 3);
        const auto data = target.readback->mapped_data.data();
        for (const auto& plane : target.planes) {
            const auto before = data + plane.offset;
            const auto reference = before + target.stride;
            const auto candidate = reference + target.stride;
            u64 plane_changed = 0;
            for (u64 i = 0; i < plane.size; ++i) differences += reference[i] != candidate[i];
            for (u64 i = 0; i < plane.size; i += plane.bytes) {
                const bool changed = std::memcmp(before + i, reference + i, plane.bytes) != 0;
                changed_reference += changed; plane_changed += changed;
                changed_candidate += std::memcmp(before + i, candidate + i, plane.bytes) != 0;
            }
            compared += plane.size;
            // Keep actual image planes, not normalized/color-adjusted derivatives.
            if (plane_changed || differences) {
                const auto label = plane.aspect == vk::ImageAspectFlagBits::eColor ? "color" :
                                   plane.aspect == vk::ImageAspectFlagBits::eDepth ? "depth" : "stencil";
                for (u32 part = 0; part < 3; ++part) {
                    const char* side = part == 0 ? "before" : part == 1 ? "reference" : "native";
                    const auto name = prefix + "-" + target.name + "-" + label + "-" + side + ".raw";
                    std::ofstream stream(work->directory / name, std::ios::binary);
                    stream.write(reinterpret_cast<const char*>(before + target.stride * part), plane.size);
                    stream.close();
                    saved &= !stream.fail();
                    dumps.emplace_back(name, plane.size);
                }
            }
        }
    }
    const bool nonempty = changed_reference != 0 && changed_candidate != 0;
    const bool equal = differences == 0;
    std::ofstream json(work->directory / (prefix + ".json"));
    json << "{\n  \"sample\": " << work->sample << ",\n  \"frame\": " << work->frame
         << ",\n  \"width\": " << work->width << ",\n  \"height\": " << work->height
         << ",\n  \"previous_city_draws\": " << work->city_draws
         << ",\n  \"prepared_resources\": " << work->prepared_resources
         << ",\n  \"compared_bytes\": " << compared << ",\n  \"different_bytes\": " << differences
         << ",\n  \"changed_reference_pixels\": " << changed_reference
         << ",\n  \"changed_native_pixels\": " << changed_candidate
         << ",\n  \"nonempty\": " << (nonempty ? "true" : "false")
         << ",\n  \"equal\": " << (equal ? "true" : "false")
         << ",\n  \"mapped\": " << (mapped ? "true" : "false")
         << ",\n  \"saved\": " << (saved ? "true" : "false")
         << ",\n  \"dumps\": [";
    for (size_t i = 0; i < dumps.size(); ++i)
        json << (i ? "," : "") << "{\"file\":\"" << dumps[i].first << "\",\"bytes\":" << dumps[i].second << '}';
    json << "],\n  \"planes\": [";
    bool first = true;
    for (const auto& target : work->targets) for (const auto& plane : target.planes) {
        if (!first) json << ','; first = false;
        json << "{\"target\":\"" << target.name << "\",\"format\":" << u32(target.view_info.format)
             << ",\"aspect\":" << u32(plane.aspect) << ",\"bytes_per_pixel\":" << plane.bytes << '}';
    }
    // "saved" above covers the dumps; a failed manifest write shows as saved=0 in the log line.
    json << "]\n}\n"; json.close(); saved &= !json.fail();
    if (!equal || !saved) state->failed.store(true, std::memory_order_release);
    if (!equal)
        std::fprintf(stderr, "NATIVE_VERIFY_MISMATCH field=%s technique=%llx material=%llx sample=%u\n",
            field, work->identity[0], work->identity[2], work->sample);
    if (equal && nonempty && saved) state->qualified.fetch_add(1, std::memory_order_release);
    state->completed.fetch_add(1, std::memory_order_release);
    std::fprintf(stderr, "%s sample=%u frame=%llu city_draws=%llu compared_bytes=%llu different_bytes=%llu changed_reference=%llu changed_native=%llu saved=%u qualified=%u failed=%u\n",
        tag, work->sample, work->frame, work->city_draws, compared, differences, changed_reference,
        changed_candidate, unsigned(saved), state->qualified.load(), unsigned(state->failed.load()));
    state->pending.store(false, std::memory_order_release);
}
} // namespace NativeVisual

void Rasterizer::ObserveNativeVisualDraw() {
    if (!NativeVisual::Enabled()) return;
    native_visual_latch = {};
    auto& watch = NativeVisual::Watch();
    const auto frame = g_iss_frame.load(std::memory_order_relaxed);
    if (frame != watch.frame) { watch.previous = watch.draws; watch.draws = 0; watch.frame = frame; }
    ++watch.draws;
}
bool Rasterizer::NativeVisualFailed() const {
    return NativeVisual::Enabled() && NativeVisual::Shared()->failed.load(std::memory_order_acquire);
}

void Rasterizer::ProbeNativeVisual(const GraphicsPipeline* pipeline, RenderState& render,
                                   s32 vertex_offset, u32 instance_offset) {
    using namespace NativeVisual;
    if (!Enabled() || !native_visual_latch.eligible || !native_visual_latch.prepared_resources || !NativeContext() ||
        attachment_feedback_loop || render.num_layers != 1 || !render.num_color_attachments ||
        render.num_color_attachments > 8 ||
        render.width < 960 || render.height < 540 || Watch().previous < 7500) return;
    const auto state = Shared();
    if (state->pending.load(std::memory_order_acquire) || state->failed.load() ||
        state->qualified.load() >= 3 || state->attempts.load() >= 32) return;
    const auto* output = std::getenv("ISS_NATIVE_VISUAL_PROBE_DIR");
    if (!output || !*output) return;
    static u64 rejected{};
    const auto reject = [&](const char* why) {
        if (++rejected <= 16 || (rejected & 4095) == 0)
            std::fprintf(stderr, "NATIVE_VISUAL_REJECT reason=%s count=%llu\n", why, rejected);
    };
    auto work = std::make_shared<Work>();
    work->frame = g_iss_frame.load(); work->city_draws = Watch().previous;
    work->width = render.width; work->height = render.height;
    work->prepared_resources = native_visual_latch.prepared_resources;
    work->identity = native_visual_latch.identity;
    work->directory = std::filesystem::path(output);
    u64 bytes = 0;
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
            (ci.flags & (vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency))) return false;
        const VideoCore::ImageViewInfo* info = nullptr;
        for (auto view_id : backing.image_view_ids) {
            const auto& live = texture_cache.GetImageView(view_id);
            if (*live.image_view == view) { info = &live.info; break; }
        }
        if (!info || info->format != ci.format || info->range.base.level || info->range.base.layer ||
            info->range.extent.levels != 1 || info->range.extent.layers != 1 || info->min_lod ||
            info->type != AmdGpu::ImageType::Color2D) return false;
        const auto& m = info->mapping;
        if ((m.r != vk::ComponentSwizzle::eIdentity && m.r != vk::ComponentSwizzle::eR) ||
            (m.g != vk::ComponentSwizzle::eIdentity && m.g != vk::ComponentSwizzle::eG) ||
            (m.b != vk::ComponentSwizzle::eIdentity && m.b != vk::ComponentSwizzle::eB) ||
            (m.a != vk::ComponentSwizzle::eIdentity && m.a != vk::ComponentSwizzle::eA)) return false;
        if (backing.state.layout != (depth ? render.depth_stencil_attachment.image_layout
                                          : render.color_attachments[slot].image_layout)) return false;
        Target target;
        target.source = &image; target.saved = backing.state; target.view_info = *info;
        target.aspects = image.aspect_mask; target.slot = slot; target.depth = depth;
        target.name = depth ? "depth" : "color" + std::to_string(slot);
        if (depth) {
            if (ci.format != vk::Format::eD32Sfloat && ci.format != vk::Format::eD32SfloatS8Uint) return false;
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
    // Memory limit of the diagnostic only (2 private copies + 3x readback per target byte).
    if (bytes > 256ull * 1024 * 1024) {
        if (rejected < 16) std::fprintf(stderr, "NATIVE_VISUAL_BUDGET targets=%zu bytes=%llu size=%ux%u\n",
                                        work->targets.size(), bytes, u32(render.width), u32(render.height));
        return reject("capture_budget");
    }
    // All allocations precede recording. Queued commands retain Work until GPU completion.
    try {
        std::filesystem::create_directories(work->directory);
        for (auto& target : work->targets) {
            auto ci = target.source->backing->image.image_ci;
            ci.pNext = nullptr; ci.initialLayout = vk::ImageLayout::eUndefined;
            // Private copies are filled from and read back to transfer; the source needs only Src.
            ci.usage |= vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
            for (u32 i = 0; i < 2; ++i) {
                target.images[i] = std::make_unique<VideoCore::UniqueImage>(instance.GetDevice(), instance.GetAllocator());
                target.images[i]->Create(ci);
                vk::ImageViewUsageCreateInfo usage{.usage = target.source->usage_flags & ci.usage};
                if (!target.view_info.is_storage) usage.usage &= ~vk::ImageUsageFlagBits::eStorage;
                const vk::ImageViewCreateInfo view_ci{
                    .pNext = &usage,
                    .image = target.images[i]->image, .viewType = vk::ImageViewType::e2D,
                    .format = target.view_info.format, .components = target.view_info.mapping,
                    .subresourceRange = {target.aspects, 0, 1, 0, 1}};
                auto [result, view] = instance.GetDevice().createImageViewUnique(view_ci);
                if (result != vk::Result::eSuccess) throw std::runtime_error("paired image view allocation failed");
                target.views[i] = std::move(view);
            }
            target.readback = std::make_unique<VideoCore::Buffer>(instance, 0, target.stride * 3,
                VideoCore::MemoryType::HostCached, "native visual pair");
        }
    } catch (const std::exception& error) {
        state->failed.store(true);
        std::fprintf(stderr, "NATIVE_VISUAL_ERROR allocation=%s\n", error.what());
        return;
    }
    work->sample = state->attempts.fetch_add(1) + 1;
    state->pending.store(true, std::memory_order_release);
    scheduler.EndRendering(); // the game's clear has already occurred; preserve its content
    const auto cmd = scheduler.Recorder();
    for (auto& target : work->targets)
        runtime.Transit(target.source, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    for (auto& target : work->targets) {
        CopyOut(cmd, target, target.source->GetImage(), 0, render.width, render.height);
        for (u32 side = 0; side < 2; ++side) {
            Barrier(cmd, target.images[side]->image, target.aspects, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe, {},
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
            for (const auto& plane : target.planes) {
                const vk::ImageCopy copy{
                    .srcSubresource = {plane.aspect, 0, 0, 1}, .dstSubresource = {plane.aspect, 0, 0, 1},
                    .extent = {render.width, render.height, 1}};
                cmd.copyImage(target.source->GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                    target.images[side]->image, vk::ImageLayout::eTransferDstOptimal, std::span{&copy, 1});
            }
            Barrier(cmd, target.images[side]->image, target.aspects, vk::ImageLayout::eTransferDstOptimal,
                target.saved.layout, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eAllGraphics,
                target.depth ? vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite);
        }
        runtime.Transit(target.source, target.saved.layout, target.saved.pl_stage, target.saved.access_mask);
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
        auto& plan = side == 0 ? native_visual_latch.reference : native_visual_latch.candidate;
        pipeline->BindResources(plan.writes, plan.push);
        scheduler.BeginRendering(clone_state);
        const auto& regs = DrawRegs();
        cmd.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0, vertex_offset, instance_offset);
        scheduler.EndRendering();
        for (const auto& target : work->targets) {
            Barrier(cmd, target.images[side]->image, target.aspects, target.saved.layout,
                vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eAllGraphics,
                target.depth ? vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                             : vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            CopyOut(cmd, target, target.images[side]->image, side + 1, render.width, render.height);
        }
    }
    const vk::MemoryBarrier2 host_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &host_read});
    // Restore the real draw's bindings and scope. The caller records that original draw once.
    pipeline->BindResources(set_writes, push_data);
    scheduler.BeginRendering(resume);
    render = resume;
    if (native_recording) native_recording->render_state = render;
    native_visual_latch.eligible = false;
    scheduler.DeferPriorityOperation([work = std::move(work), state] { Complete(work, state); });
}
