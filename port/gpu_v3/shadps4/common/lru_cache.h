// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <deque>
#include <type_traits>

#include "common/types.h"

namespace Common {

template <typename ObjectType, typename TickType>
class LeastRecentlyUsedCache {
    struct Item {
        ObjectType obj;
        TickType tick;
        Item* next{};
        Item* prev{};
        /// In the list. A freed item is not: a late Touch of a freed image's id (a per-frame
        /// table touching it before it noticed the unregister, 10-20 live-resize hang s1513)
        /// re-linked it, the pool handed it out again and the list closed into a loop.
        bool linked{};
    };

public:
    LeastRecentlyUsedCache() : first_item{}, last_item{} {}
    ~LeastRecentlyUsedCache() = default;

    size_t Insert(ObjectType obj, TickType tick) {
        const auto new_id = Build();
        auto& item = item_pool[new_id];
        item.obj = obj;
        item.tick = tick;
        Attach(item);
        return new_id;
    }

    void Touch(size_t id, TickType tick) {
        auto& item = item_pool[id];
        if (!item.linked || item.tick >= tick) {
            return;
        }
        item.tick = tick;
        if (&item == last_item) {
            return;
        }
        Detach(item);
        Attach(item);
    }

    void Free(size_t id) {
        auto& item = item_pool[id];
        if (!item.linked) {
            return; // freed twice
        }
        Detach(item);
        item.linked = false;
        item.prev = nullptr;
        item.next = nullptr;
        free_items.push_back(id);
    }

    /// Every item, in list order (no tick test): for owners that keep the use tick themselves.
    template <typename Func>
    void ForEachItem(Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result_t<Func, ObjectType>, bool>;
        Item* iterator = first_item;
        size_t budget = item_pool.size();
        while (iterator && budget-- > 0) {
            Item* next = iterator->next;
            if constexpr (RETURNS_BOOL) {
                if (func(iterator->obj)) {
                    return;
                }
            } else {
                func(iterator->obj);
            }
            iterator = next;
        }
    }

    template <typename Func>
    void ForEachItemBelow(TickType tick, Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result_t<Func, ObjectType>, bool>;
        Item* iterator = first_item;
        // A loop in the list (it cannot form any more, see Item::linked) would hang the walk:
        // at most one visit per item.
        size_t budget = item_pool.size();
        while (iterator && budget-- > 0) {
            if (static_cast<s64>(tick) - static_cast<s64>(iterator->tick) < 0) {
                return;
            }
            Item* next = iterator->next;
            if constexpr (RETURNS_BOOL) {
                if (func(iterator->obj)) {
                    return;
                }
            } else {
                func(iterator->obj);
            }
            iterator = next;
        }
    }

private:
    size_t Build() {
        if (free_items.empty()) {
            const size_t item_id = item_pool.size();
            auto& item = item_pool.emplace_back();
            item.next = nullptr;
            item.prev = nullptr;
            return item_id;
        }
        const size_t item_id = free_items.front();
        free_items.pop_front();
        auto& item = item_pool[item_id];
        item.next = nullptr;
        item.prev = nullptr;
        return item_id;
    }

    void Attach(Item& item) {
        item.linked = true;
        if (!first_item) {
            first_item = &item;
        }
        if (!last_item) {
            last_item = &item;
        } else {
            item.prev = last_item;
            last_item->next = &item;
            item.next = nullptr;
            last_item = &item;
        }
    }

    void Detach(Item& item) {
        if (item.prev) {
            item.prev->next = item.next;
        }
        if (item.next) {
            item.next->prev = item.prev;
        }
        if (&item == first_item) {
            first_item = item.next;
            if (first_item) {
                first_item->prev = nullptr;
            }
        }
        if (&item == last_item) {
            last_item = item.prev;
            if (last_item) {
                last_item->next = nullptr;
            }
        }
    }

    std::deque<Item> item_pool;
    std::deque<size_t> free_items;
    Item* first_item{};
    Item* last_item{};
};

} // namespace Common
