// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3 without DLSS: the fork's own non-Windows stubs (vk_dlss.cpp, !_WIN32), used on
// every system here, since this port ships no Streamline SDK. DLSS stays off (DeviceSupported).
#include "video_core/renderer_vulkan/vk_dlss.h"

namespace Vulkan::Dlss {

void AddInstanceExtensions(std::vector<const char*>&) {}
bool DeviceSupported(vk::PhysicalDevice) {
    return false;
}
const std::vector<const char*>& DeviceExtensions() {
    static const std::vector<const char*> none;
    return none;
}
void DeviceCreated(vk::Instance, vk::PhysicalDevice, vk::Device, u32, u32) {}
PFN_vkQueuePresentKHR PresentFunction() {
    return nullptr;
}
void Shutdown() {}
bool Evaluate(vk::CommandBuffer, u64, const EvalImage&, const EvalImage&, const EvalImage&,
              const EvalImage&, float, float, bool) {
    return false;
}

} // namespace Vulkan::Dlss
