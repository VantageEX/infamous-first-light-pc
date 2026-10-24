// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"
#include <algorithm>
#include <ranges>
#include "common/assert.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_memory_budget.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/water_scale.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};

// SHADPS4_SMART_IMAGE_BARRIERS, read once at startup: GetBarriers runs per image transition, and
// a function-local static would add a guard check to each call.
static const bool g_smart_images = [] {
    const char* value = std::getenv("SHADPS4_SMART_IMAGE_BARRIERS");
    return value != nullptr && value[0] == '1';
}();

static vk::ImageUsageFlags ImageUsageFlags(const Vulkan::Instance* instance,
                                           const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (instance->IsAttachmentFeedbackLoopLayoutSupported()) {
                usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
            }
            // Always create images with storage flag to avoid needing re-creation in case of e.g
            // compute clears This sacrifices a bit of performance but is less work. ExtendedUsage
            // flag is also used.
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

static vk::FormatFeatureFlags2 FormatFeatureFlags(const vk::ImageUsageFlags usage_flags) {
    vk::FormatFeatureFlags2 feature_flags{};
    if (usage_flags & vk::ImageUsageFlagBits::eTransferSrc) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eTransferDst) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eSampled) {
        feature_flags |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eColorAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    // Note: StorageImage is intentionally ignored for now since it is always set, and can mess up
    // compatibility checks.
    return feature_flags;
}

UniqueImage::~UniqueImage() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
    }
}

void UniqueImage::Destroy() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
        image = vk::Image{};
        allocation = {};
    }
}

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci) {
    this->image_ci = image_ci;
    ASSERT(!image);
    VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkImageCreateInfo image_ci_unsafe = static_cast<VkImageCreateInfo>(image_ci);
    VkImage unsafe_image{};
    // v1.0.2: over VMA's budget, retry without it (vk_memory_budget.h) instead of asserting.
    const u64 approx_bytes = u64{image_ci.extent.width} * image_ci.extent.height *
                             image_ci.extent.depth * image_ci.arrayLayers * 4;
    VkResult result = Vulkan::AllocateWithBudgetRetry(
        allocator, alloc_info, "image", approx_bytes,
        [&] {
            return fmt::format("{}x{}x{} layers {} mips {} {} (size ~ at 4 bytes per texel)",
                               image_ci.extent.width, image_ci.extent.height,
                               image_ci.extent.depth, image_ci.arrayLayers, image_ci.mipLevels,
                               vk::to_string(image_ci.format));
        },
        [&](const VmaAllocationCreateInfo& ci) {
            return vmaCreateImage(allocator, &image_ci_unsafe, &ci, &unsafe_image, &allocation,
                                  nullptr);
        });
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating image {}x{} {} with error {}",
               image_ci.extent.width, image_ci.extent.height, vk::to_string(image_ci.format),
               vk::to_string(vk::Result{result}));
    image = vk::Image{unsafe_image};
}

Image::Image(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
             BlitHelper& blit_helper_, Common::SlotVector<ImageView, Common::StableSlotCapacity>& slot_image_views_,
             const ImageInfo& info_)
    : instance{&instance_}, scheduler{&scheduler_}, blit_helper{&blit_helper_},
      slot_image_views{&slot_image_views_}, info{info_} {
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }
    image_uid = global_image_uid.Next();
    mip_hashes.resize(info.resources.levels);
    // Here we force `eExtendedUsage` as don't know all image usage cases beforehand. In normal case
    // the texture cache should re-create the resource with the usage requested
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance->Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance->IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }

    usage_flags = ImageUsageFlags(instance, info);
    format_features = FormatFeatureFlags(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance->GetSupportedFormat(info.pixel_format, format_features);
    const vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = usage_flags,
        .flags = flags,
    };
    const auto image_format_properties =
        instance->GetPhysicalDevice().getImageFormatProperties2(format_info);
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} is not supported (flags {}, usage {})",
                  vk::to_string(supported_format), vk::to_string(format_info.type),
                  vk::to_string(format_info.flags), vk::to_string(format_info.usage));
    }
    supported_samples = image_format_properties.result == vk::Result::eSuccess
                            ? image_format_properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

    const vk::ImageCreateInfo image_ci = {
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = supported_format,
        .extent{
            .width = info.size.width,
            .height = info.size.height,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = LiverpoolToVK::NumSamples(info.num_samples, supported_samples),
        .tiling = tiling,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
    backing->image.Create(image_ci);

    Vulkan::SetObjectName(instance->GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing->num_samples > 1 != info.num_samples > 1) {
        SetBackingSamples(info.num_samples);
    }
    const auto& view_infos = backing->image_view_infos;
    const auto it = std::ranges::find(view_infos, view_info);
    if (it != view_infos.end()) {
        const auto view_id = backing->image_view_ids[std::distance(view_infos.begin(), it)];
        return (*slot_image_views)[view_id];
    }
    const auto view_id = slot_image_views->insert(*instance, view_info, *this);
    backing->image_view_infos.emplace_back(view_info);
    backing->image_view_ids.emplace_back(view_id);
    return (*slot_image_views)[view_id];
}

Image::Barriers Image::GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                                   vk::PipelineStageFlags2 dst_stage,
                                   std::optional<SubresourceRange> subres_range, bool raw_layout) {
    if (!raw_layout) {
        dst_layout = Vulkan::GuestLayout(dst_layout);
    }
    // SHADPS4_MTB_CONFLICT (diagnostic): a draw's first touch in the batch, and any state change.
    const bool cf = AmdGpu::MtbConflict::g_enabled;
    const u64 cf_serial = cf ? backing->state_serial : 0;
    if (cf) [[unlikely]] {
        AmdGpu::MtbConflict::Touch(backing->cf, AmdGpu::MtbConflict::Image);
    }
    BackingImage* const cf_backing = backing;
    SCOPE_EXIT {
        if (cf && cf_backing->state_serial != cf_serial) [[unlikely]] {
            AmdGpu::MtbConflict::Change(cf_backing->cf, AmdGpu::MtbConflict::Image);
        }
    };
    auto& last_state = backing->state;
    auto& runs = backing->subresource_runs;

    const u32 res_levels = info.resources.levels;
    const u32 res_layers = info.resources.layers;

    constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                 vk::AccessFlagBits2::eShaderWrite |
                                 vk::AccessFlagBits2::eMemoryWrite;
    const State dst_state{dst_stage, dst_mask, dst_layout};

    // SHADPS4_SMART_IMAGE_BARRIERS=1: a read after reads in the same layout needs a barrier only to
    // make an earlier write visible to a stage/access that has not been given it yet. `covered`
    // skips reads whose stage and access are already in the state; otherwise the barrier is kept
    // and the state becomes the union of the reads, so alternating read kinds settle after one
    // barrier each (a later write waits for every stage in the union).
    const bool smart_images = g_smart_images;
    constexpr auto any_writes = write_flags | vk::AccessFlagBits2::eShaderStorageWrite |
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    const bool dst_reads_only = !(dst_mask & any_writes);
    const auto read_after_read = [&](const State& st) {
        return smart_images && dst_reads_only && st.layout == dst_layout &&
               !(st.access_mask & any_writes);
    };
    const auto covered = [&](const State& st) {
        return read_after_read(st) && (st.access_mask & dst_mask) == dst_mask &&
               (st.pl_stage & dst_stage) == dst_stage;
    };
    const auto next_state = [&](const State& st) {
        return read_after_read(st)
                   ? State{st.pl_stage | dst_stage, st.access_mask | dst_mask, dst_layout}
                   : dst_state;
    };

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != info.resources);
    const bool promoted = !runs.empty();

    Barriers barriers;

    // Fast path: whole image shares one state (never promoted to per-subresource tracking).
    if (!needs_partial_transition && !promoted) {
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if ((last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) ||
            covered(last_state)) {
            Common::FaultMonitor::CountBarriers(false, 0, 0);
            return {};
        }
        // SHADPS4_IMAGE_WAW_STATS=1 (diagnostic): same-state write-after-write barriers (same
        // layout and access, previous access writes), which upstream #5112 stopped emitting.
        static const bool waw_stats = [] {
            const char* value = std::getenv("SHADPS4_IMAGE_WAW_STATS");
            return value && value[0] == '1';
        }();
        if (waw_stats) [[unlikely]] {
            static u64 total = 0, same_waw = 0, same_waw_color = 0, same_waw_general = 0,
                       same_waw_depth = 0;
            ++total;
            if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && is_write) {
                ++same_waw;
                same_waw_color += dst_layout == vk::ImageLayout::eColorAttachmentOptimal;
                same_waw_general += dst_layout == vk::ImageLayout::eGeneral;
                same_waw_depth += dst_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal ||
                                  dst_layout == vk::ImageLayout::eDepthAttachmentOptimal;
            }
            if ((total & 0x3ffff) == 0) {
                LOG_INFO(Render_Vulkan,
                         "ImageWawStats: whole-image barriers={} same_state_waw={} ({:.1f}%) "
                         "color={} general={} depth={}",
                         total, same_waw, 100.0 * double(same_waw) / double(total),
                         same_waw_color, same_waw_general, same_waw_depth);
            }
        }
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = GetImage(),
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
        last_state = next_state(last_state);
        ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
        Common::FaultMonitor::CountBarriers(false, 0, barriers.size());
        return barriers;
    }

    // Range path. Subresources are tracked as a coalesced list of [begin, end) runs over the
    // linear index (mip * layers + layer). A partial transition of a range that is already uniform
    // in the destination state costs O(runs it overlaps) - a scan with no allocation and no
    // barriers - instead of one comparison per mip x layer slot (up to thousands on array
    // textures, which was the 11.5 ms/frame leaf).
    if (!promoted) {
        runs.push_back(StateRun{0u, res_levels * res_layers, last_state});
        ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
    }

    const u32 base_level = needs_partial_transition ? subres_range->base.level : 0u;
    const u32 base_layer = needs_partial_transition ? subres_range->base.layer : 0u;
    const u32 ext_levels = needs_partial_transition ? subres_range->extent.levels : res_levels;
    const u32 ext_layers = needs_partial_transition ? subres_range->extent.layers : res_layers;

    u64 seg_iters = 0;
    boost::container::small_vector<StateRun, 8> rebuilt;

    for (u32 mip = base_level; mip < base_level + ext_levels; ++mip) {
        // [a, b) is this mip's slice of the requested layer range; it never crosses a mip
        // boundary, so every barrier emitted below is a valid single-mip layer range.
        const u32 a = mip * res_layers + base_layer;
        const u32 b = a + ext_layers;
        if (a >= b) {
            continue;
        }

        // Pass 1: locate the first overlapping run and check whether any overlapped segment
        // actually needs a transition. The converged hot path exits here - no barrier, no realloc.
        size_t first = 0;
        while (first < runs.size() && runs[first].end <= a) {
            ++first;
        }
        bool any = false;
        for (size_t i = first; i < runs.size() && runs[i].begin < b; ++i) {
            ++seg_iters;
            const auto& st = runs[i].state;
            const bool is_write = static_cast<bool>(st.access_mask & write_flags);
            if ((st.layout != dst_layout || st.access_mask != dst_mask || is_write) &&
                !covered(st)) {
                any = true;
                break;
            }
        }
        if (!any) {
            continue;
        }

        // Pass 2: rebuild the run list, setting the needed sub-segments of [a, b) to dst_state and
        // emitting one barrier per changed old-state segment (adjacent equal runs coalesced).
        rebuilt.clear();
        const auto push = [&](u32 bg, u32 en, const State& st) {
            if (bg >= en) {
                return;
            }
            if (!rebuilt.empty() && rebuilt.back().end == bg && rebuilt.back().state == st) {
                rebuilt.back().end = en;
            } else {
                rebuilt.push_back(StateRun{bg, en, st});
            }
        };
        for (const auto& r : runs) {
            // Part of this run left of the query stays as-is.
            const u32 left_end = std::min(r.end, a);
            if (left_end > r.begin) {
                push(r.begin, left_end, r.state);
            }
            // Overlap with [a, b): transition it if the old state differs (or is a write).
            const u32 ov_begin = std::max(r.begin, a);
            const u32 ov_end = std::min(r.end, b);
            if (ov_begin < ov_end) {
                const bool is_write = static_cast<bool>(r.state.access_mask & write_flags);
                const bool needs =
                    (r.state.layout != dst_layout || r.state.access_mask != dst_mask || is_write) &&
                    !covered(r.state);
                if (needs) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = r.state.pl_stage,
                        .srcAccessMask = r.state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = r.state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = GetImage(),
                        .subresourceRange{
                            .aspectMask = aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = ov_begin - mip * res_layers,
                            .layerCount = ov_end - ov_begin,
                        },
                    });
                    push(ov_begin, ov_end, next_state(r.state));
                } else {
                    push(ov_begin, ov_end, r.state);
                }
            }
            // Part right of the query stays as-is.
            const u32 right_begin = std::max(r.begin, b);
            if (right_begin < r.end) {
                push(right_begin, r.end, r.state);
            }
        }
        runs.assign(rebuilt.begin(), rebuilt.end());
        ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
    }

    // A full-resource transition leaves every subresource in one state, so drop back to the cheap
    // uniform path for next time.
    // With smart image barriers a full transition can leave differing (merged or skipped) states;
    // keep per-subresource tracking unless every run agrees.
    if (!needs_partial_transition && !smart_images) {
        if (!runs.empty()) {
            runs.clear();
            ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
        }
    } else if (!needs_partial_transition) {
        const bool uniform = std::ranges::all_of(
            runs, [&](const StateRun& r) { return r.state == runs.front().state; });
        if (uniform) {
            last_state = runs.empty() ? dst_state : runs.front().state;
            runs.clear();
            ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
            Common::FaultMonitor::CountBarriers(true, seg_iters, barriers.size());
            return barriers;
        }
    }

    if (!(last_state == dst_state)) {
        last_state = dst_state;
        ++backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8
    }
    Common::FaultMonitor::CountBarriers(true, seg_iters, barriers.size());
    return barriers;
}

void Image::Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                    std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf /*= {}*/,
                    std::source_location loc) {
    // Adjust pipeline stage
    const vk::PipelineStageFlags2 dst_pl_stage =
        (dst_mask == vk::AccessFlagBits2::eTransferRead ||
         dst_mask == vk::AccessFlagBits2::eTransferWrite)
            ? vk::PipelineStageFlagBits2::eTransfer
            : vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;

    // MEASUREMENT (perf-forest): bt_transit is 11.5 ms/frame, the biggest single leaf, and has
    // never been split into its three parts: GetBarriers (the subresource loop), EndRendering (the
    // render-pass break, only when a barrier is actually emitted), and pipelineBarrier2 (submit).
    // Which dominates decides whether there is any lever left here.
    static u64 tr_calls = 0, tr_empty = 0, tr_getbar_ns = 0, tr_endr_ns = 0, tr_submit_ns = 0;
    const auto tr_t0 = Common::FaultMonitor::Now();
    const auto barriers = GetBarriers(dst_layout, dst_mask, dst_pl_stage, range, bool(cmdbuf));
    const auto tr_t1 = Common::FaultMonitor::Now();
    // SHADPS4_LEAN: no TransitSplit accounting (its timings are 0 unless SHADPS4_FAULT_TIMERS).
    const bool tr_stats = !::Common::Lean();
    if (tr_stats) {
        ++tr_calls;
        tr_getbar_ns += Common::FaultMonitor::DiffNs(tr_t0, tr_t1);
    }
    const auto tr_report = [&] {
        if (!tr_stats || tr_calls < 500000) {
            return;
        }
        const auto nonempty = tr_calls - tr_empty;
        LOG_INFO(Render_Vulkan,
                 "TransitSplit: calls={} empty={:.1f}% getbarriers={:.3f}us/call "
                 "endrender={:.3f}us/emit submit={:.3f}us/emit",
                 tr_calls, 100.0 * static_cast<double>(tr_empty) / tr_calls,
                 static_cast<double>(tr_getbar_ns) / 1e3 / tr_calls,
                 nonempty ? static_cast<double>(tr_endr_ns) / 1e3 / nonempty : 0.0,
                 nonempty ? static_cast<double>(tr_submit_ns) / 1e3 / nonempty : 0.0);
        tr_calls = tr_empty = tr_getbar_ns = tr_endr_ns = tr_submit_ns = 0;
    };
    if (barriers.empty()) {
        tr_empty += tr_stats;
        tr_report();
        return;
    }

    const auto emit_barriers = [](vk::CommandBuffer command, auto owned_barriers) {
        command.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(owned_barriers.size()),
            .pImageMemoryBarriers = owned_barriers.data(),
        });
    };
    if (!cmdbuf) {
        // External command buffers must be protected by their caller's access scope.
        scheduler->EndRendering(loc);
    }
    const auto tr_t2 = Common::FaultMonitor::Now();
    if (cmdbuf) {
        emit_barriers(cmdbuf, std::span<const vk::ImageMemoryBarrier2>{barriers});
    } else {
        scheduler->RecordArrays(emit_barriers, std::span<const vk::ImageMemoryBarrier2>{barriers});
    }
    if (tr_stats) {
        tr_endr_ns += Common::FaultMonitor::DiffNs(tr_t1, tr_t2);
        tr_submit_ns += Common::FaultMonitor::DiffNs(tr_t2, Common::FaultMonitor::Now());
    }
    if (Vulkan::GpuBatchTimer::Enabled()) {
        Vulkan::GpuBreaks().image_barrier_calls.fetch_add(1, std::memory_order_relaxed);
        Vulkan::GpuBreaks().image_barriers.fetch_add(barriers.size(), std::memory_order_relaxed);
        for (const auto& b : barriers) {
            Vulkan::GpuBreakSites().Image(loc, b.oldLayout, b.newLayout, b.srcAccessMask,
                                          b.dstAccessMask);
        }
    }
    tr_report();
}

void Image::Upload(std::span<const vk::BufferImageCopy> upload_copies, vk::Buffer buffer,
                   u64 offset, u64 barrier_size) {
    SetBackingSamples(info.num_samples, false);
    scheduler->EndRendering();
    // barrier_size: the source range when it is not the guest-sized upload (TexReplace pictures).
    const u64 range_size = barrier_size ? barrier_size : info.guest_size;

    const vk::BufferMemoryBarrier2 pre_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer,
        .offset = offset,
        .size = range_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer,
        .offset = offset,
        .size = range_size,
    };
    const auto image_barriers =
        GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eCopy, {});
    scheduler->RecordArrays(
        [buffer, image = GetImage(), pre_barrier,
         post_barrier](vk::CommandBuffer cmdbuf, auto owned_barriers, auto owned_copies) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &pre_barrier,
                .imageMemoryBarrierCount = static_cast<u32>(owned_barriers.size()),
                .pImageMemoryBarriers = owned_barriers.data(),
            });
            cmdbuf.copyBufferToImage(buffer, image,
                                     Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal),
                                     owned_copies);
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &post_barrier,
            });
        },
        std::span<const vk::ImageMemoryBarrier2>{image_barriers}, upload_copies);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
    flags &= ~ImageFlagBits::Dirty;
}

void Image::Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                     u64 offset, u64 download_size) {
    SetBackingSamples(info.num_samples);
    if (backing->scale > 1) [[unlikely]] {
        // SHADPS4_WATER_REFLECTION_SCALE: the guest-sized copies read the top-left part only.
        WaterScale::g_counters.downloads.fetch_add(1, std::memory_order_relaxed);
    }
    scheduler->EndRendering();

    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = download_size,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .buffer = buffer,
        .offset = offset,
        .size = download_size,
    };
    const auto image_barriers =
        GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                    vk::PipelineStageFlagBits2::eCopy, {});
    auto command_access = scheduler->AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
        .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    });
    cmdbuf.copyImageToBuffer(GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
                             buffer,
                             download_copies);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

static std::pair<u32, u32> SanitizeCopyLayers(const ImageInfo& src_info, const ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D source layers {} to 1.", src_layers);
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        LOG_WARNING(Render_Vulkan, "Coercing copy 3D destination layers {} to 1.", dst_layers);
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy source layers {} and destination layers {} to minimum.",
                        src_layers, dst_layers);
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D source layers {} to 3D destination depth {}", src_layers,
                        depth);
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            LOG_WARNING(Render_Vulkan,
                        "Coercing copy 2D destination layers {} to 3D source depth {}", dst_layers,
                        depth);
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

void Image::CopyImage(Image& src_image) {
    const auto& src_info = src_image.info;
    if (backing->scale > 1 || src_image.backing->scale > 1) [[unlikely]] {
        WaterScale::g_counters.copies.fetch_add(1, std::memory_order_relaxed);
    }

    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src_info.pixel_format != info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src_info.pixel_format), vk::to_string(info.pixel_format));
    }

    const u32 base_width = src_info.size.width;
    const u32 base_height = src_info.size.height;
    const u32 base_depth =
        info.type == AmdGpu::ImageType::Color3D ? info.size.depth : src_info.size.depth;

    // Match sample count before copying
    SetBackingSamples(info.num_samples, false);
    src_image.SetBackingSamples(src_info.num_samples);

    boost::container::small_vector<vk::ImageCopy, 8> regions;

    const vk::ImageAspectFlags src_aspect =
        src_image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const vk::ImageAspectFlags dst_aspect = aspect_mask & ~vk::ImageAspectFlagBits::eStencil;

    const bool src_is_2d = ConvertImageType(src_info.type) == vk::ImageType::e2D;
    const bool src_is_3d = ConvertImageType(src_info.type) == vk::ImageType::e3D;

    const bool dst_is_2d = ConvertImageType(info.type) == vk::ImageType::e2D;
    const bool dst_is_3d = ConvertImageType(info.type) == vk::ImageType::e3D;

    const bool is_2d_to_3d = src_is_2d && dst_is_3d;
    const bool is_3d_to_2d = src_is_3d && dst_is_2d;
    const bool is_same_type = !is_2d_to_3d && !is_3d_to_2d;

    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        auto [src_layers, dst_layers] = SanitizeCopyLayers(src_info, info, mip_d);

        vk::ImageCopy region{};

        region.srcSubresource.aspectMask = src_aspect;
        region.srcSubresource.mipLevel = mip;
        region.srcSubresource.baseArrayLayer = 0;

        region.dstSubresource.aspectMask = dst_aspect;
        region.dstSubresource.mipLevel = mip;
        region.dstSubresource.baseArrayLayer = 0;

        if (is_same_type) {
            // 2D->2D OR 3D->3D
            if (src_is_3d) {
                // 3D images must use layerCount=1
                region.srcSubresource.layerCount = 1;
                region.dstSubresource.layerCount = 1;
                region.extent = vk::Extent3D(mip_w, mip_h, mip_d);
            } else {
                // Array images
                const u32 copy_layers = std::min(src_layers, dst_layers);
                region.srcSubresource.layerCount = copy_layers;
                region.dstSubresource.layerCount = copy_layers;
                region.extent = vk::Extent3D(mip_w, mip_h, 1);
            }
        } else if (is_2d_to_3d) {
            // 2D array -> 3D volume
            region.srcSubresource.layerCount = src_layers;
            region.dstSubresource.layerCount = 1;
            region.extent = vk::Extent3D(mip_w, mip_h, src_layers);
        } else if (is_3d_to_2d) {
            // 3D volume -> 2D array
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.layerCount = dst_layers;
            region.extent = vk::Extent3D(mip_w, mip_h, dst_layers);
        }

        regions.push_back(region);
    }

    scheduler->EndRendering();

    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});

    auto command_access = scheduler->AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();

    if (!regions.empty()) {
        cmdbuf.copyImage(src_image.GetImage(), src_image.backing->state.layout, GetImage(),
                         backing->state.layout, regions);
    }

    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}
void Image::CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset) {
    const auto& src_info = src_image.info;
    if (backing->scale > 1 || src_image.backing->scale > 1) [[unlikely]] {
        WaterScale::g_counters.copies.fetch_add(1, std::memory_order_relaxed);
    }
    const u32 num_mips = std::min(src_info.resources.levels, info.resources.levels);
    const u32 num_layers = std::min(src_info.resources.layers, info.resources.layers);
    ASSERT(src_info.resources.layers == info.resources.layers || num_mips == 1);

    SetBackingSamples(info.num_samples, false);
    src_image.SetBackingSamples(src_info.num_samples);

    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const auto mip_w = std::max(src_info.size.width >> mip, 1u);
        const auto mip_h = std::max(src_info.size.height >> mip, 1u);
        const auto mip_d = std::max(src_info.size.depth >> mip, 1u);

        buffer_copies.emplace_back(vk::BufferImageCopy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = src_image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {mip_w, mip_h, mip_d},
        });
    }

    const vk::BufferMemoryBarrier2 pre_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer,
        .offset = offset,
        .size = VK_WHOLE_SIZE,
    };

    const vk::BufferMemoryBarrier2 post_copy_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer,
        .offset = offset,
        .size = VK_WHOLE_SIZE,
    };

    scheduler->EndRendering();
    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});

    auto command_access = scheduler->AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_copy_barrier,
    });

    cmdbuf.copyImageToBuffer(src_image.GetImage(),
                             Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal), buffer,
                             buffer_copies);

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_copy_barrier,
    });

    for (auto& copy : buffer_copies) {
        copy.imageSubresource.aspectMask = aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    }

    cmdbuf.copyBufferToImage(buffer, GetImage(),
                             Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal),
                             buffer_copies);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}

void Image::CopyMip(Image& src_image, u32 mip, u32 slice) {
    const auto& src_info = src_image.info;
    if (backing->scale > 1 || src_image.backing->scale > 1) [[unlikely]] {
        WaterScale::g_counters.copies.fetch_add(1, std::memory_order_relaxed);
    }

    const auto dst_dim = info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max(info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src_info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src_info.size.width >> src_dim));
    ASSERT(mip_block_h == (src_info.size.height >> src_dim));
    ASSERT(mip_block_p == (src_info.pitch >> src_dim));

    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src_info, info, src_info.size.depth);

    const vk::ImageCopy image_copy{
        .srcSubresource{
            .aspectMask = src_image.aspect_mask,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = src_layers,
        },
        .dstSubresource{
            .aspectMask = src_image.aspect_mask,
            .mipLevel = mip,
            .baseArrayLayer = slice,
            .layerCount = dst_layers,
        },
        .extent = {src_info.size.width, src_info.size.height, src_info.size.depth},
    };

    SetBackingSamples(info.num_samples);
    src_image.SetBackingSamples(src_info.num_samples);

    scheduler->EndRendering();
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});

    auto command_access = scheduler->AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.copyImage(src_image.GetImage(), src_image.backing->state.layout, GetImage(),
                     backing->state.layout, image_copy);
    Transit(vk::ImageLayout::eGeneral,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {});
}

void Image::Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                    const VideoCore::SubresourceRange& mrt1_range) {
    SetBackingSamples(1, false);
    scheduler->EndRendering();

    src_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                      mrt0_range);
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, mrt1_range);

    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src_image.info, info, 1);
    if (src_image.backing->num_samples == 1) {
        const vk::ImageCopy region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt0_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt1_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {info.size.width, info.size.height, 1},
        };
        scheduler->AcquireCommandBuffer().Handle().copyImage(
            src_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
            GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal), region);
    } else {
        const vk::ImageResolve region = {
            .srcSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt0_range.base.layer,
                .layerCount = src_layers,
            },
            .srcOffset = {0, 0, 0},
            .dstSubresource{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = mrt1_range.base.layer,
                .layerCount = dst_layers,
            },
            .dstOffset = {0, 0, 0},
            .extent = {info.size.width, info.size.height, 1},
        };
        scheduler->AcquireCommandBuffer().Handle().resolveImage(
            src_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
            GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal), region);
    }

    flags |= VideoCore::ImageFlagBits::GpuModified;
    flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Image::Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range) {
    const vk::ImageSubresourceRange vk_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = range.base.level,
        .levelCount = range.extent.levels,
        .baseArrayLayer = range.base.layer,
        .layerCount = range.extent.layers,
    };
    scheduler->EndRendering();
    Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    auto command_access = scheduler->AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.clearColorImage(GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal),
                           clear_value.color,
                           vk_range);
}

void Image::SetBackingSamples(u32 num_samples, bool copy_backing) {
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!info.props.is_depth, "Swapping samples is only valid for color images");
    BackingImage* new_backing;
    // SHADPS4_WATER_REFLECTION_SCALE: a sample backing of the current scale (always 1 when off).
    const u32 cur_scale = backing->scale;
    auto it = std::ranges::find_if(backing_images, [&](const BackingImage& b) {
        return b.num_samples == num_samples && b.scale == cur_scale;
    });
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.samples = LiverpoolToVK::NumSamples(num_samples, supported_samples);

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->scale = cur_scale;
        new_backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
        new_backing->image.Create(new_image_ci);

        Vulkan::SetObjectName(instance->GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (copy_backing) {
        scheduler->EndRendering();
        ASSERT(info.resources.levels == 1 && info.resources.layers == 1);

        // Transition current backing to shader read layout
        auto barriers =
            GetBarriers(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                        vk::PipelineStageFlagBits2::eFragmentShader, std::nullopt);

        // Transition dest backing to color attachment layout, not caring of previous contents
        constexpr auto dst_stage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        constexpr auto dst_access = vk::AccessFlagBits2::eColorAttachmentWrite;
        const auto dst_layout = Vulkan::GuestLayout(vk::ImageLayout::eColorAttachmentOptimal);
        barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = new_backing->image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = info.resources.layers,
            },
        });
        auto command_access = scheduler->AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(
            info.size.width, info.size.height, new_backing->num_samples, info.pixel_format,
            backing->num_samples > 1, backing->image, new_backing->image);

        // Update current layout in tracker to new backings layout
        new_backing->state.layout = dst_layout;
        new_backing->state.access_mask = dst_access;
        new_backing->state.pl_stage = dst_stage;
        ++new_backing->state_serial;
        VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8 // SHADPS4_TEX_TRANSIT_SKIP: a state write outside GetBarriers
        if (AmdGpu::MtbConflict::g_enabled) [[unlikely]] {
            AmdGpu::MtbConflict::Change(new_backing->cf, AmdGpu::MtbConflict::Image);
        }
    }

    VideoCore::TexTable::NoteImage(tab_gen, tab_id); // q8b: the backing changes
    backing = new_backing;
}

void Image::SetBackingScale(u32 scale) {
    if (!backing || backing->scale == scale) {
        return;
    }
    ASSERT_MSG(info.resources.levels == 1 && info.resources.layers == 1 &&
                   info.type != AmdGpu::ImageType::Color3D,
               "Scaled backings are single-level 2D images");
    const u32 num_samples = backing->num_samples;
    BackingImage* new_backing;
    auto it = std::ranges::find_if(backing_images, [&](const BackingImage& b) {
        return b.num_samples == num_samples && b.scale == scale;
    });
    if (it == backing_images.end()) {
        auto new_image_ci = backing->image.image_ci;
        new_image_ci.extent.width = info.size.width * scale;
        new_image_ci.extent.height = info.size.height * scale;

        new_backing = &backing_images.emplace_back();
        new_backing->num_samples = num_samples;
        new_backing->scale = scale;
        new_backing->image = UniqueImage{instance->GetDevice(), instance->GetAllocator()};
        new_backing->image.Create(new_image_ci);

        Vulkan::SetObjectName(instance->GetDevice(), new_backing->image.image,
                              "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (x{} backing)",
                              info.size.width, info.size.height, info.size.depth,
                              AmdGpu::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                              info.guest_address, info.guest_size, info.resources.layers,
                              info.resources.levels, num_samples, scale);
    } else {
        new_backing = std::addressof(*it);
    }
    // The views, memos and texture-table slots key on the backing: all of them miss now.
    VideoCore::TexTable::NoteImage(tab_gen, tab_id);
    g_image_epoch.fetch_add(1, std::memory_order_release);
    backing = new_backing;
}

} // namespace VideoCore
