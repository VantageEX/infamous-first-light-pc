// SPDX-License-Identifier: GPL-2.0-or-later
// Conservative, lock-free absence hint for the G-owned texture registry.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace iss::g4 {

enum class NegativeImageResult : std::uint8_t { Unknown, HasRegisteredPage, NoRegisteredPage };

struct NegativeImageSnapshot {
    NegativeImageResult result{NegativeImageResult::Unknown};
    std::uint64_t epoch{};
    std::uint64_t address{};
    std::size_t size{};
};

// Writers are serialized externally. Every registry insertion/removal brackets
// the page-count changes. A reader accepts absence only if it saw one unchanged,
// even epoch on both sides of all atomic page-count loads.
class TextureNegativeIndex {
public:
    void BeginRegistryMutation() noexcept {
        epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    void EndRegistryMutation() noexcept {
        epoch.fetch_add(1, std::memory_order_release);
    }

    template <class PageCount>
    [[nodiscard]] NegativeImageSnapshot Probe(std::uint64_t address, std::size_t size,
                                              unsigned page_shift, std::uint64_t num_pages,
                                              PageCount&& count) const noexcept {
        NegativeImageSnapshot out{NegativeImageResult::Unknown, 0, address, size};
        if (size == 0 || address > std::numeric_limits<std::uint64_t>::max() - (size - 1) ||
            page_shift >= 64) return out;
        const auto first = address >> page_shift;
        const auto last = (address + size - 1) >> page_shift;
        if (last >= num_pages) return out;
        const auto before = epoch.load(std::memory_order_acquire);
        if (before & 1) return out;
        bool has_image = false;
        for (auto page = first; page <= last; ++page) {
            if (count(page) != 0) {
                has_image = true;
                break;
            }
        }
        const auto after = epoch.load(std::memory_order_acquire);
        if (before != after || (after & 1)) return out;
        out.epoch = after;
        out.result = has_image ? NegativeImageResult::HasRegisteredPage
                               : NegativeImageResult::NoRegisteredPage;
        return out;
    }

    [[nodiscard]] bool StillCurrent(const NegativeImageSnapshot& snapshot) const noexcept {
        return snapshot.result == NegativeImageResult::NoRegisteredPage &&
               !(snapshot.epoch & 1) &&
               epoch.load(std::memory_order_acquire) == snapshot.epoch;
    }

private:
    std::atomic<std::uint64_t> epoch{2};
};

} // namespace iss::g4
