"""Renderer v3's presenter: the fork's vk_presenter.cpp with this port's overlay instead of the
fork's ImGui interface. Reads the fork's file, writes glue/overrides/vk_presenter.cpp.

Run from port/gpu_v3 after updating shadps4/:
    python tools/make_presenter.py shadps4/video_core/renderer_vulkan/vk_presenter.cpp glue/overrides/vk_presenter.cpp
"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
s = src.read_text(encoding='utf-8')
nl = '\r\n' if '\r\n' in s else '\n'
s = s.replace('\r\n', '\n')


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:200])
    s = s.replace(old, new)


HEADER = '''// Renderer v3 (port/gpu_v3/README.md): axylpup's vk_presenter.cpp (second-son-v1.1.1) with this
// port's overlay in place of the fork's ImGui interface: the game's frame is blitted into the
// window (fitted, black bars), then gpu/shim/tlg_overlay.cpp draws over it with its own ImGui
// context; no docking window, layers, splash or ImGui boot screen. Everything else is the fork's.
'''
s = HEADER + s

for line in ('#include "core/devtools/layer.h"\n', '#include "imgui/friends_layer.h"\n',
             '#include "imgui/invitation_prompt_layer.h"\n', '#include "imgui/notifications_layer.h"\n',
             '#include "imgui/renderer/imgui_core.h"\n', '#include "imgui/renderer/texture_manager.h"\n',
             '#include "imgui/renderer/imgui_impl_vulkan.h"\n', '#include "imgui/shadnet_notifications_layer.h"\n',
             '#include <imgui.h>\n'):
    rep(line, '')
rep('#include "video_core/renderer_vulkan/vk_presenter.h"\n',
    '#include "video_core/renderer_vulkan/vk_presenter.h"\n#include "tlg_overlay.h"\n')

# A black window behind the frame (letterbox) and behind the boot screen, without a layout change.
rep('static vk::Rect2D FitImage(', '''static void ClearToBlack(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent) {
    const vk::RenderingAttachmentInfo attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}},
    };
    const vk::RenderingInfo info{
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    };
    cmdbuf.beginRendering(info);
    cmdbuf.endRendering();
}

static vk::Rect2D FitImage(''')

# Boot screen: black and this port's overlay (its own progress and notices), no ImGui window.
start = s.index('    ImGui::Core::NewFrame(false);\n    {\n        const ImGuiViewport* vp')
end = s.index('        ImGui::PopStyleColor(2);\n    }\n', start) + len('        ImGui::PopStyleColor(2);\n    }\n')
s = s[:start] + s[end:]
rep('''    ImGui::Core::Render(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    const vk::ImageMemoryBarrier to_present{''', '''    ClearToBlack(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    TlgOverlay::Render(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    const vk::ImageMemoryBarrier to_present{''')
rep('    ImGui::Core::TextureManager::NotePresented();\n', '', 2)

# The boot screen shows the fork's warm-up (its pipeline cache, read while the rasterizer is
# built) through this port's loading screen: the overlay starts before the rasterizer, and each
# boot frame hands it the step's progress (one_time: the first-start preparation).
rep('''      rasterizer{(g_booting_presenter = this,
                  std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool))},''',
    '''      rasterizer{(g_booting_presenter = this,
                  TlgOverlay::Init(instance, swapchain.GetSurfaceFormat().format, 8),
                  std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool))},''')
rep('''    ClearToBlack(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    TlgOverlay::Render(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());''',
    '''    if (!headline.empty()) {
        TlgOverlay::SetLoadingCompile(one_time);
        TlgOverlay::SetLoading(true, static_cast<u32>(std::clamp(fraction, 0.0f, 1.0f) * 100.0f), 100);
    }
    ClearToBlack(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());
    TlgOverlay::Render(cmdbuf, swapchain.ImageView(), swapchain.GetExtent());''')

# The stutter report (glue/v3_stutter.h): once per presented game frame.
rep('''        DebugState.IncFlipFrameNum();
''', '''        DebugState.IncFlipFrameNum();
        V3::Stutter::OnPresent();
''')
rep('#include "tlg_overlay.h"\n', '#include "tlg_overlay.h"\n#include "v3_stutter.h"\n')

# Constructor and destructor: this port's overlay instead of the fork's layers.
rep('''    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    ImGui::Friends::Register();
    ImGui::ShadNetNotify::Register();
    ImGui::InvitationPrompt::Register();
''', '''    // This port's overlay (FPS, menu, save dialogs), with its own ImGui context.
    TlgOverlay::Init(instance, swapchain.GetSurfaceFormat().format, num_images);
''')
rep('''    ImGui::InvitationPrompt::Unregister();
    ImGui::ShadNetNotify::Unregister();
    ImGui::Friends::Unregister();
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
''', '')
rep('''        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);''', '''        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        TlgOverlay::ForgetBackdrop(frame.image_view);
        device.destroyImageView(frame.image_view);''')

# Frames: the overlay's backdrop descriptor instead of an ImGui texture.
rep('''    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);''', '''    if (frame->image_view) {
        TlgOverlay::ForgetBackdrop(frame->image_view);
        device.destroyImageView(frame->image_view);''')
rep('    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);\n', '')

# Present: the frame is a blit source, then the overlay's backdrop.
rep('    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);\n\n', '')
rep('''                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,''', '''                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,''')
rep('''        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);
''', '''        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                   vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);
''')
start = s.index('        { // Draw the game\n')
end_marker = '        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());\n'
end = s.index(end_marker, start) + len(end_marker)
s = s[:start] + '''        { // Draw the game: the frame fitted into the window, then this port's overlay over it.
            const vk::Rect2D rect = FitImage(static_cast<s32>(frame->width), static_cast<s32>(frame->height),
                                             static_cast<s32>(extent.width), static_cast<s32>(extent.height));
            SetExpectedGameSize(static_cast<s32>(extent.width), static_cast<s32>(extent.height));
            ClearToBlack(cmdbuf, swapchain_image_view, extent);
            const vk::ImageSubresourceRange color_range{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            };
            const vk::ImageMemoryBarrier to_blit{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange = color_range,
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
                                   {}, {}, to_blit);
            const vk::ImageSubresourceLayers layers{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            const vk::ImageBlit blit{
                .srcSubresource = layers,
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{static_cast<s32>(frame->width),
                                                      static_cast<s32>(frame->height), 1}},
                .dstSubresource = layers,
                .dstOffsets = std::array{vk::Offset3D{rect.offset.x, rect.offset.y, 0},
                                         vk::Offset3D{rect.offset.x + static_cast<s32>(rect.extent.width),
                                                      rect.offset.y + static_cast<s32>(rect.extent.height), 1}},
            };
            cmdbuf.blitImage(frame->image, vk::ImageLayout::eTransferSrcOptimal, swapchain_image,
                             vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
            // The frame stays shader-readable between presents (PrepareLastFrame expects it), which
            // the overlay's frosted panels sample.
            const std::array after_blit{
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                                     vk::AccessFlagBits::eColorAttachmentRead,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = swapchain_image,
                    .subresourceRange = color_range,
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = frame->image,
                    .subresourceRange = color_range,
                },
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                       vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, after_blit);
            TlgOverlay::SetBackdrop(frame->image_view, vk::ImageLayout::eShaderReadOnlyOptimal, rect);
            TlgOverlay::Render(cmdbuf, swapchain_image_view, extent);
        }
''' + s[end:]

# Screenshot notices: this port's overlay notice instead of the fork's notification layer.
rep('''        shadNotifications::QueueNotification("Saved screenshot:\\n" + primary_path.string(), 3.0f,
                                             shadNotifications::position::BottomRight, imgdata);''',
    '''        (void)imgdata;
        TlgOverlay::Notice("Saved screenshot: " + primary_path.filename().string(), 3.0f);''')
rep('''            shadNotifications::QueueNotification("Saved screenshot:\\n" + path.string(), 3.0f,
                                                 shadNotifications::position::BottomRight, imgdata);''',
    '''            (void)imgdata;
            TlgOverlay::Notice("Saved screenshot: " + path.filename().string(), 3.0f);''')

assert 'ImGui::' not in s, [l for l in s.splitlines() if 'ImGui::' in l][:5]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(s.replace('\n', nl).encode('utf-8'))
print('ok', dst)
