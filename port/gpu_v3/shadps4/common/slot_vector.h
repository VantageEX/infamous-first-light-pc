// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <numeric>
#include <utility>
#include <vector>
#include "common/assert.h"
#include "common/reserved_memory.h"
#include "common/types.h"

namespace Common {

struct SlotId {
    static constexpr u32 INVALID_INDEX = std::numeric_limits<u32>::max();

    SlotId() noexcept = default;
    constexpr SlotId(u32 index) noexcept : index(index) {}

    constexpr auto operator<=>(const SlotId&) const noexcept = default;

    constexpr explicit operator bool() const noexcept {
        return index != INVALID_INDEX;
    }

    u32 index = INVALID_INDEX;
};

/// Capacity of the cache SlotVectors (images, image views, buffers) whose objects must never move:
/// a resolver thread may hold a reference while the command processor inserts
/// (notes/parallel-back-design-2026-10-02.md sec. 5 and 7a). Address space only; pages are
/// committed as the vector grows, exactly where the old doubling would have reallocated.
inline constexpr std::size_t StableSlotCapacity = std::size_t{1} << 20;

/// StableCapacity == 0: the classic vector (grows by moving every object into a doubled array).
/// StableCapacity  > 0: storage is one reserved range, committed in place - an object keeps its
/// address until it is erased, and the occupancy bitset never reallocates either. Exceeding the
/// capacity is fatal.
template <class T, std::size_t StableCapacity = 0>
class SlotVector {
    constexpr static std::size_t InitialCapacity = 2048;
    constexpr static bool Stable = StableCapacity != 0;

public:
    template <typename ValueType, typename Pointer, typename Reference>
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = ValueType;
        using difference_type = std::ptrdiff_t;
        using pointer = Pointer;
        using reference = Reference;

        Iterator(SlotVector& vector_, SlotId index_) : vector(vector_), slot(index_) {
            AdvanceToValid();
        }

        reference operator*() const {
            return vector[slot];
        }

        pointer operator->() const {
            return &vector[slot];
        }

        Iterator& operator++() {
            ++slot.index;
            AdvanceToValid();
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const Iterator& other) const {
            return slot == other.slot;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void AdvanceToValid() {
            while (slot < vector.values_capacity && !vector.ReadStorageBit(slot.index)) {
                ++slot.index;
            }
        }

        SlotVector& vector;
        SlotId slot;
    };

    using iterator = Iterator<T, T*, T&>;
    using const_iterator = Iterator<const T, const T*, const T&>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    SlotVector() {
        if constexpr (Stable) {
            const std::size_t page = AddressSpacePageSize();
            reserved_bytes = (StableCapacity * sizeof(Entry) + page - 1) / page * page;
            values = static_cast<Entry*>(ReserveAddressSpace(reserved_bytes));
            ASSERT_MSG(values != nullptr, "SlotVector: cannot reserve {} bytes", reserved_bytes);
            stored_bitset.reserve((StableCapacity + 63) / 64);
        }
        Reserve(InitialCapacity);
    }

    SlotVector(const SlotVector&) = delete;
    SlotVector& operator=(const SlotVector&) = delete;

    ~SlotVector() noexcept {
        std::size_t index = 0;
        for (u64 bits : stored_bitset) {
            for (std::size_t bit = 0; bits; ++bit, bits >>= 1) {
                if ((bits & 1) != 0) {
                    values[index + bit].object.~T();
                }
            }
            index += 64;
        }
        if constexpr (Stable) {
            ReleaseAddressSpace(values, reserved_bytes);
        } else {
            delete[] values;
        }
    }

    [[nodiscard]] T& operator[](SlotId id) noexcept {
        ValidateIndex(id);
        return values[id.index].object;
    }

    [[nodiscard]] const T& operator[](SlotId id) const noexcept {
        ValidateIndex(id);
        return values[id.index].object;
    }

    bool is_allocated(SlotId id) const {
        return ReadStorageBit(id.index);
    }

    template <typename... Args>
    SlotId insert(Args&&... args) noexcept {
        const u32 index = FreeValueIndex();
        new (&values[index].object) T(std::forward<Args>(args)...);
        SetStorageBit(index);

        return SlotId{index};
    }

    void erase(SlotId id) noexcept {
        values[id.index].object.~T();
        free_list.push_back(id.index);
        ResetStorageBit(id.index);
    }

    std::size_t size() const noexcept {
        return values_capacity - free_list.size();
    }

    iterator begin() noexcept {
        return iterator(*this, 0);
    }

    const_iterator begin() const noexcept {
        return const_iterator(*this, 0);
    }

    const_iterator cbegin() const noexcept {
        return begin();
    }

    iterator end() noexcept {
        return iterator(*this, values_capacity);
    }

    const_iterator end() const noexcept {
        return const_iterator(*this, values_capacity);
    }

    const_iterator cend() const noexcept {
        return end();
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const noexcept {
        return rbegin();
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const noexcept {
        return rend();
    }

private:
    struct NonTrivialDummy {
        NonTrivialDummy() noexcept {}
    };

    union Entry {
        Entry() noexcept : dummy{} {}
        ~Entry() noexcept {}

        NonTrivialDummy dummy;
        T object;
    };

    void SetStorageBit(u32 index) noexcept {
        stored_bitset[index / 64] |= u64(1) << (index % 64);
    }

    void ResetStorageBit(u32 index) noexcept {
        stored_bitset[index / 64] &= ~(u64(1) << (index % 64));
    }

    bool ReadStorageBit(u32 index) const noexcept {
        return ((stored_bitset[index / 64] >> (index % 64)) & 1) != 0;
    }

    void ValidateIndex([[maybe_unused]] SlotId id) const noexcept {
        DEBUG_ASSERT(id);
        DEBUG_ASSERT(id.index / 64 < stored_bitset.size());
        DEBUG_ASSERT(((stored_bitset[id.index / 64] >> (id.index % 64)) & 1) != 0);
    }

    [[nodiscard]] u32 FreeValueIndex() noexcept {
        if (free_list.empty()) {
            std::size_t grown = values_capacity ? (values_capacity << 1) : 1;
            if constexpr (Stable) {
                ASSERT_MSG(values_capacity < StableCapacity, "SlotVector: stable capacity {} full",
                           StableCapacity);
                grown = std::min(grown, StableCapacity);
            }
            Reserve(grown);
        }

        const u32 free_index = free_list.back();
        free_list.pop_back();
        return free_index;
    }

    void Reserve(std::size_t new_capacity) noexcept {
        if constexpr (Stable) {
            // Commit the pages the new entries need and construct them in place: nothing moves.
            ASSERT(new_capacity <= StableCapacity);
            const std::size_t page = AddressSpacePageSize();
            const std::size_t need =
                (new_capacity * sizeof(Entry) + page - 1) / page * page;
            if (need > committed_bytes) {
                const bool committed =
                    CommitAddressSpace(values, committed_bytes, need - committed_bytes);
                ASSERT_MSG(committed, "SlotVector: cannot commit {} bytes", need - committed_bytes);
                committed_bytes = need;
            }
            for (std::size_t i = values_capacity; i < new_capacity; ++i) {
                new (&values[i]) Entry();
            }
            stored_bitset.resize((new_capacity + 63) / 64); // within the reserved capacity
            const std::size_t old_free_size = free_list.size();
            free_list.resize(old_free_size + (new_capacity - values_capacity));
            const std::size_t new_free_size = free_list.size();
            std::iota(free_list.rbegin(), free_list.rbegin() + new_free_size - old_free_size,
                      static_cast<u32>(values_capacity));
            values_capacity = new_capacity;
            return;
        }
        Entry* const new_values = new Entry[new_capacity];
        std::size_t index = 0;
        for (u64 bits : stored_bitset) {
            for (std::size_t bit = 0; bits; ++bit, bits >>= 1) {
                const std::size_t i = index + bit;
                if ((bits & 1) == 0) {
                    continue;
                }
                T& old_value = values[i].object;
                new (&new_values[i].object) T(std::move(old_value));
                old_value.~T();
            }
            index += 64;
        }

        stored_bitset.resize((new_capacity + 63) / 64);

        const std::size_t old_free_size = free_list.size();
        free_list.resize(old_free_size + (new_capacity - values_capacity));
        const std::size_t new_free_size = free_list.size();
        std::iota(free_list.rbegin(), free_list.rbegin() + new_free_size - old_free_size,
                  static_cast<u32>(values_capacity));

        delete[] values;
        values = new_values;
        values_capacity = new_capacity;
    }

    Entry* values = nullptr;
    std::size_t values_capacity = 0;
    std::size_t reserved_bytes = 0;  ///< Stable only
    std::size_t committed_bytes = 0; ///< Stable only

    std::vector<u64> stored_bitset;
    std::vector<u32> free_list;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
    std::size_t operator()(const Common::SlotId& id) const noexcept {
        return std::hash<u32>{}(id.index);
    }
};
