// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <atomic>

namespace Vulkan {
// Publication only: the owner retains both pipelines until queued GPU work is finished.
template<typename Handle>
class PipelineHandleChoice {
public:
    Handle Select(Handle initial) const noexcept {
        const auto value = replacement.load(std::memory_order_acquire);
        return value == Handle{} ? initial : value;
    }
    void Publish(Handle handle) noexcept { replacement.store(handle, std::memory_order_release); }
private:
    std::atomic<Handle> replacement{};
};
} // namespace Vulkan
