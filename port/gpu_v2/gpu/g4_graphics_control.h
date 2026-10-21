// SPDX-License-Identifier: GPL-2.0-or-later
// Owner-only frontend controls. The caller decides where a G safe point is;
// callbacks must own their inputs and must not wait for PM4 or another CP owner.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace iss::g4 {

class GraphicsControlMailbox {
public:
    struct Control {
        enum class State : std::uint8_t { Queued, Running, Done, Failed, Cancelled };
        [[nodiscard]] State Status() const noexcept {
            return state.load(std::memory_order_acquire);
        }
        [[nodiscard]] std::exception_ptr Failure() const noexcept {
            return Status() == State::Failed ? failure : std::exception_ptr{};
        }

    private:
        friend class GraphicsControlMailbox;
        explicit Control(std::function<void()> operation_) : operation{std::move(operation_)} {}
        std::function<void()> operation;
        std::atomic<State> state{State::Queued};
        std::exception_ptr failure{}; // stored before Failed release
    };
    using Request = std::shared_ptr<Control>;

    [[nodiscard]] Request Enqueue(std::function<void()> operation) {
        if (!operation) return {};
        auto request = Request{new Control{std::move(operation)}};
        std::scoped_lock lock{mutex};
        if (stopped) return {};
        pending.push_back(request);
        pending_count.fetch_add(1, std::memory_order_release);
        return request;
    }

    [[nodiscard]] bool HasPending() const noexcept {
        return pending_count.load(std::memory_order_acquire) != 0;
    }

    // Used when a prerequisite C seal fails before this G control is allowed
    // to run. An active callback cannot be cancelled through this path.
    bool CancelPending(const Request& request) noexcept {
        if (!request) return false;
        std::scoped_lock lock{mutex};
        const auto it = std::find(pending.begin(), pending.end(), request);
        if (it == pending.end()) return false;
        (*it)->operation = {};
        (*it)->state.store(Control::State::Cancelled, std::memory_order_release);
        pending.erase(it);
        pending_count.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }

    // Only G's outer Process loop passes true. No control is executed reentrantly
    // from a guest packet, another control, or ProcessCommands.
    bool ServeOne(bool graphics_safe_point) {
        if (!graphics_safe_point || !HasPending()) return false;
        Request request;
        {
            std::scoped_lock lock{mutex};
            if (stopped || active || pending.empty()) return false;
            request = std::move(pending.front());
            pending.pop_front();
            pending_count.fetch_sub(1, std::memory_order_acq_rel);
            active = request;
            request->state.store(Control::State::Running, std::memory_order_release);
        }
        std::exception_ptr failure;
        try {
            request->operation();
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::scoped_lock lock{mutex};
            if (!stopped) {
                if (failure) request->failure = failure;
                request->state.store(failure ? Control::State::Failed : Control::State::Done,
                                     std::memory_order_release);
            }
            request->operation = {};
            active.reset();
        }
        return true;
    }

    void Stop() noexcept {
        std::scoped_lock lock{mutex};
        stopped = true;
        for (const auto& request : pending)
            request->state.store(Control::State::Cancelled, std::memory_order_release);
        if (active)
            active->state.store(Control::State::Cancelled, std::memory_order_release);
        pending.clear();
        pending_count.store(0, std::memory_order_release);
    }

private:
    mutable std::mutex mutex;
    std::deque<Request> pending{};
    Request active{};
    std::atomic<std::uint32_t> pending_count{};
    bool stopped{};
};

} // namespace iss::g4
