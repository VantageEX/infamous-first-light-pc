// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/object_pool.h"
#include "common/types.h"

namespace VideoCore {

template <class Traits>
class MultiLevelPageTable final {
    using Entry = typename Traits::Entry;

    static constexpr size_t AddressSpaceBits = Traits::AddressSpaceBits;
    static constexpr size_t FirstLevelBits = Traits::FirstLevelBits;
    static constexpr size_t PageBits = Traits::PageBits;
    static constexpr size_t FirstLevelShift = AddressSpaceBits - FirstLevelBits;
    static constexpr size_t SecondLevelBits = FirstLevelShift - PageBits;
    static constexpr size_t NumEntriesPerL1Page = 1ULL << SecondLevelBits;

    using L1Page = std::array<Entry, NumEntriesPerL1Page>;

public:
    explicit MultiLevelPageTable() : first_level_map{1ULL << FirstLevelBits, nullptr} {}

    ~MultiLevelPageTable() noexcept = default;

    [[nodiscard]] Entry* find(size_t page) {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            return nullptr;
        }
        return &(*first_level_map[l1_page])[l2_page];
    }

    [[nodiscard]] const Entry* find(size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            return nullptr;
        }
        return &(*first_level_map[l1_page])[l2_page];
    }

    /// For readers on another thread than the one that allocates (see operator[]): one acquire load
    /// of the second-level pointer, which operator[] publishes with release.
    [[nodiscard]] const Entry* find_acquire(size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        L1Page* const l2 = std::atomic_ref<L1Page*>(const_cast<L1Page*&>(first_level_map[l1_page]))
                               .load(std::memory_order_acquire);
        return l2 ? &(*l2)[l2_page] : nullptr;
    }
    [[nodiscard]] const Entry& operator[](size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            std::atomic_ref<L1Page*>(first_level_map[l1_page])
                .store(page_alloc.Create(), std::memory_order_release);
        }
        return (*first_level_map[l1_page])[l2_page];
    }

    [[nodiscard]] Entry& operator[](size_t page) {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            std::atomic_ref<L1Page*>(first_level_map[l1_page])
                .store(page_alloc.Create(), std::memory_order_release);
        }
        return (*first_level_map[l1_page])[l2_page];
    }

private:
    std::vector<L1Page*> first_level_map{};
    Common::ObjectPool<L1Page> page_alloc;
};

} // namespace VideoCore
