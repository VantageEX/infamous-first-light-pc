#include "infamous_renderer/native_instanced_feature.h"

#include <mutex>
#include <unordered_map>

namespace iss::native_geometry {
namespace {
std::mutex mutex;
std::unordered_map<const Vulkan::Instance*, bool> enabled_instances;
}

void RegisterIndirectFirstInstanceFeature(const Vulkan::Instance& instance, bool enabled) {
    std::scoped_lock lock{mutex};
    enabled_instances[&instance] = enabled;
}

void UnregisterIndirectFirstInstanceFeature(const Vulkan::Instance& instance) {
    std::scoped_lock lock{mutex};
    enabled_instances.erase(&instance);
}

bool IndirectFirstInstanceEnabledFor(const Vulkan::Instance& instance) {
    std::scoped_lock lock{mutex};
    const auto found = enabled_instances.find(&instance);
    return found != enabled_instances.end() && found->second;
}

} // namespace iss::native_geometry
