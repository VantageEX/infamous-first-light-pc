#pragma once

namespace Vulkan {
class Instance;
}

namespace iss::native_geometry {

// Tracks the feature bit actually requested for each logical device. Querying
// physical support alone is insufficient for nonzero indirect firstInstance.
void RegisterIndirectFirstInstanceFeature(const Vulkan::Instance& instance, bool enabled);
void UnregisterIndirectFirstInstanceFeature(const Vulkan::Instance& instance);
bool IndirectFirstInstanceEnabledFor(const Vulkan::Instance& instance);

} // namespace iss::native_geometry
