// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace iss::buffers {
// A proof that the ordinary small-buffer path returned an already resident,
// CPU-clean arena. It never represents guest bytes or GPU completion. A hit
// still requires the caller's CURRENT buffer hazard check before it may bind.
struct SmallReadStamp {
    std::uint64_t address{}, size{}, mapping_generation{}, arena_generation{}, gpu_write_sequence{};
    std::uint64_t upload_epoch{}, host_write_generation{}, scheduler_tick{};
    std::array<std::uint64_t, 2> cpu_versions{};
    std::uint32_t queue{}, stable_frames{};
    bool operator==(const SmallReadStamp&) const = default;
};

template<class Handle, std::size_t Capacity = 4096>
class SmallReadMemo {
    static_assert(Capacity && (Capacity & (Capacity - 1)) == 0);
public:
    struct Entry { SmallReadStamp stamp{}; Handle handle{}; std::uint64_t offset{}; };
    struct Counts { std::uint64_t checks{}, matches{}, hits{}, hazards{}, stores{}, unstable{}; };

    const Entry* Find(const SmallReadStamp& stamp) {
        ++counts.checks;
        const auto& entry = entries[Slot(stamp.address, stamp.size)];
        if (!entry.handle || entry.stamp != stamp) return nullptr;
        ++counts.matches;
        return &entry;
    }
    // Sample before the original clean-page check, and again after its resident
    // binding is prepared. Never bless an observation spanning a version change.
    bool Remember(const SmallReadStamp& before, const SmallReadStamp& after,
                  Handle handle, std::uint64_t offset) {
        if (!handle || !before.address || !before.size || before != after) {
            ++counts.unstable;
            return false;
        }
        entries[Slot(before.address, before.size)] = {before, handle, offset};
        ++counts.stores;
        return true;
    }
    void Hit() { ++counts.hits; }
    void Hazard() { ++counts.hazards; }
    const Counts& Statistics() const { return counts; }

private:
    static std::size_t Slot(std::uint64_t address, std::uint64_t size) {
        auto key = (address >> 4) ^ (size * UINT64_C(0x9e3779b97f4a7c15));
        key ^= key >> 29;
        return std::size_t(key * UINT64_C(0xbf58476d1ce4e5b9)) & (Capacity - 1);
    }
    std::array<Entry, Capacity> entries{};
    Counts counts{};
};
} // namespace iss::buffers
