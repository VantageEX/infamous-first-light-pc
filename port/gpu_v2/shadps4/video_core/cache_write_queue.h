// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <stop_token>

namespace Storage {
class CacheWriteQueue {
public:
    bool Begin() {
        std::lock_guard lock{mutex};
        if (!tasks.empty()) return false;
        accepting = true;
        return true;
    }
    bool Submit(std::packaged_task<void()> task) {
        {
            std::lock_guard lock{mutex};
            if (!accepting) return false;
            tasks.push(std::move(task));
        }
        ready.notify_one();
        return true;
    }
    void Seal() {
        std::lock_guard lock{mutex};
        accepting = false;
    }
    std::optional<std::packaged_task<void()>> Next(std::stop_token stop) {
        std::unique_lock lock{mutex};
        ready.wait(lock, stop, [&] { return !tasks.empty(); });
        // A stop request wakes an idle worker, but accepted writes drain first.
        if (tasks.empty()) return {};
        auto task = std::move(tasks.front());
        tasks.pop();
        return task;
    }
private:
    std::mutex mutex;
    std::condition_variable_any ready;
    std::queue<std::packaged_task<void()>> tasks;
    bool accepting{};
};
} // namespace Storage
