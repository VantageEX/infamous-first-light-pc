// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>

#include <cstdio>

#include <string>

#include <algorithm>

#include <vector>

#include <typeinfo>

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace Common {

/// SHADPS4_RECORD_BYTES_STATS=1, read once at startup (see RecordBytesStatsEnabled).
inline const bool g_record_bytes_stats = [] {
    const char* value = std::getenv("SHADPS4_RECORD_BYTES_STATS");
    return value && value[0] == '1';
}();

/**
 * A bounded, single-producer command queue with one recording worker.
 *
 * The queue owns a fixed pool of chunks. Small callable objects are placement-stored in those
 * chunks, so recording them does not allocate. A chunk is published when it is full or when Drain
 * is called. The worker executes published chunks in FIFO order and recycles a chunk only after all
 * of its callables have been invoked (or discarded) and destroyed. The sole producer owns the
 * unpublished current chunk; when that chunk has room, recording takes the producer-only fast path
 * without acquiring the queue mutex. Publishing, acquiring a replacement chunk, and draining
 * still use the mutex.
 *
 * Only one producer may call Record, RecordWithPayload, or Drain. Factories and callbacks must not
 * call back into this object. A factory runs after reservation but before publication, without the
 * queue mutex held; the producer-owned current chunk keeps its reserved payload private until the
 * returned callable has been constructed.
 *
 * RecordWithPayload reserves the callable and the requested extra storage in one unpublished
 * chunk. The factory receives an aligned byte span in that storage and must return an invocable
 * callable. The callable is responsible for any non-trivial objects it constructs in the byte
 * span; the queue treats that span as raw storage and keeps it alive until the callable is
 * destroyed.
 */
// ChunkCount is the capacity; SHADPS4_REC_CHUNKS (default 10, the old fixed count) sets how many chunks are used.
// ChunkCount 128 (2026-10-08, P1c): the producer uses at most the first half (the old 64),
// record sinks the rest.
template <std::size_t ChunkCount = 128, std::size_t ChunkBytes = 64 * 1024>
class CommandRecordingWorker {
    static_assert(ChunkCount >= 2, "The command worker needs at least two chunks");
    static_assert(ChunkBytes > 0, "The command worker needs non-empty chunks");

    using InvokeFunction = void (*)(void*);
    using DestroyFunction = void (*)(void*) noexcept;

    struct CommandHeader {
        InvokeFunction invoke{};
        DestroyFunction destroy{};
        void* object{};
        std::size_t next_offset{};
    };

    static_assert(alignof(CommandHeader) <= alignof(std::max_align_t));
    static_assert(ChunkBytes >= sizeof(CommandHeader));

    enum class ChunkState : std::uint8_t {
        Free,
        Building,
        Queued,
        Executing,
        Discarding,
    };

    struct Chunk {
        alignas(std::max_align_t) std::array<std::byte, ChunkBytes> storage{};
        std::size_t used{};
        std::size_t first_offset{};
        std::size_t command_count{};
        ChunkState state{ChunkState::Free};
        std::uint64_t batch{}; ///< Lane mode: the submission batch the chunk belongs to.
    };

    struct Layout {
        std::size_t header_offset{};
        std::size_t object_offset{};
        std::size_t payload_offset{};
        std::size_t end_offset{};
    };

    struct Reservation {
        Chunk* chunk{};
        std::size_t old_used{};
        std::size_t old_first_offset{};
        std::size_t old_command_count{};
        Layout layout{};
        CommandHeader* header{};
        void* object{};
        std::span<std::byte> payload{};
    };

public:
    static constexpr std::size_t DefaultChunkCount = ChunkCount;
    [[nodiscard]] std::size_t ActiveChunks() const noexcept {
        return active_chunks_;
    }
    /// Commands accepted so far. Producer thread only (it is the only writer).
    [[nodiscard]] std::uint64_t AcceptedCommandsProducer() const noexcept {
        return accepted_commands_;
    }
    static constexpr std::size_t DefaultChunkBytes = ChunkBytes;

    struct Counters {
        std::uint64_t commands{};
        std::uint64_t chunks{};
        std::uint64_t bytes{};
        std::uint64_t worker_ns{};
        std::uint64_t drain_ns{};
        std::uint64_t space_wait_ns{};
    };

    /// Diagnostic: called on the worker thread once per chunk (SHADPS4_THREAD_STATS).
    static void SetChunkHook(void (*hook)()) {
        chunk_hook_.store(hook, std::memory_order_relaxed);
    }

    /// Lane mode (SHADPS4_PAR_REC, `lanes` >= 1): the producer marks the end of every submission
    /// batch with Publish(true); chunks are tagged with their batch and batch k runs on lane
    /// k % lanes, each lane in FIFO order on its own thread. Lanes record different batches at
    /// once, so their commands must only touch per-lane state (the caller's job). `lane_start`
    /// runs first on each lane thread; `on_failure` runs once if a command throws (wake anything
    /// a lane may be blocked on). lanes == 0: the single ordered worker.
    static constexpr std::size_t MaxLanes = 4;
    /// Chunks [0, SinkBase) serve the producer (SHADPS4_REC_CHUNKS clamps to it), the rest Sinks.
    static constexpr std::size_t SinkBase = ChunkCount / 2;

    /**
     * Multi-threaded back P1c (notes/parrec-build-2026-10-08.md sec. 6): commands recorded by a
     * translating thread into chunks of this worker's sink region, nothing published until the
     * producer hands the sink over with PublishSink - in order with its own commands. One thread
     * fills a sink at a time; the producer publishes it. Callables keep pointers into their
     * chunk (payloads), so chunks are handed over, never copied.
     */
    class Sink {
    public:
        explicit Sink(CommandRecordingWorker& owner) : owner_{owner} {
            chunks_.reserve(8);
        }
        ~Sink() {
            Discard();
        }
        Sink(const Sink&) = delete;
        Sink& operator=(const Sink&) = delete;

        template <typename F>
            requires std::is_object_v<std::remove_cvref_t<F>>
        void Record(F&& function) {
            using Callable = std::remove_cvref_t<F>;
            static_assert(alignof(Callable) <= alignof(std::max_align_t));
            static_assert(std::is_invocable_v<Callable&>, "Command callable must be invocable");
            Append<Callable>(
                [&](void* object, std::span<std::byte>) {
                    std::construct_at(reinterpret_cast<Callable*>(object),
                                      std::forward<F>(function));
                },
                sizeof(Callable), alignof(Callable), 0);
        }

        template <typename Factory>
            requires std::is_object_v<std::remove_reference_t<Factory>>
        void RecordWithPayload(const std::size_t payload_bytes, Factory&& factory) {
            using FactoryType = std::remove_reference_t<Factory>;
            using Result = std::invoke_result_t<FactoryType&, std::span<std::byte>>;
            using Callable = std::remove_cvref_t<Result>;
            static_assert(alignof(Callable) <= alignof(std::max_align_t));
            static_assert(std::is_invocable_v<Callable&>, "Factory result must be invocable");
            Append<Callable>(
                [&](void* object, std::span<std::byte> payload) {
                    decltype(auto) result = std::invoke(factory, payload);
                    if constexpr (std::is_lvalue_reference_v<decltype(result)>) {
                        std::construct_at(reinterpret_cast<Callable*>(object), result);
                    } else {
                        std::construct_at(reinterpret_cast<Callable*>(object), std::move(result));
                    }
                },
                sizeof(Callable), alignof(Callable), payload_bytes);
        }

        /// Monotonic: moves with every command and every MarkBoundary (the RecordSeq of a
        /// translating context: equal values = nothing recorded in between).
        [[nodiscard]] std::uint64_t Seq() const noexcept {
            return seq_;
        }
        /// Something was recorded outside this sink between its commands (a submission).
        void MarkBoundary() noexcept {
            ++seq_;
        }
        /// Continues a sequence another counter started (the RecordSeq before the sink).
        void StartSeq(const std::uint64_t seq) noexcept {
            seq_ = seq;
        }
        /// Serial use only (P1d: the producer fills the sink itself): when a new chunk is
        /// needed, the full ones are published first, like the producer's own chunks.
        void SetEagerPublish(const bool eager) noexcept {
            eager_publish_ = eager;
        }
        [[nodiscard]] std::uint64_t EagerPublishes() const noexcept {
            return eager_publishes_;
        }
        [[nodiscard]] bool Empty() const noexcept {
            return chunks_.empty();
        }
        /// Drops everything recorded and not yet published.
        void Discard() {
            for (Chunk* chunk : chunks_) {
                owner_.FreeSinkChunk(*chunk);
            }
            chunks_.clear();
        }

    private:
        friend class CommandRecordingWorker;

        template <typename Callable, typename Construct>
        void Append(Construct&& construct, const std::size_t object_bytes,
                    const std::size_t object_alignment, const std::size_t payload_bytes) {
            if (!TryLayout(0, object_bytes, object_alignment, payload_bytes)) {
                throw std::length_error("command recording sink: command exceeds chunk size");
            }
            Chunk* chunk = chunks_.empty() ? nullptr : chunks_.back();
            std::optional<Layout> layout;
            if (chunk != nullptr) {
                layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
            }
            if (!layout) {
                if (eager_publish_ && !chunks_.empty()) {
                    owner_.PublishSink(*this);
                    ++eager_publishes_;
                }
                chunk = owner_.AcquireSinkChunk();
                chunks_.push_back(chunk);
                layout = TryLayout(0, object_bytes, object_alignment, payload_bytes);
            }
            const std::size_t old_used = chunk->used;
            chunk->used = layout->end_offset;
            auto* const header =
                reinterpret_cast<CommandHeader*>(chunk->storage.data() + layout->header_offset);
            void* const object = chunk->storage.data() + layout->object_offset;
            try {
                construct(object, std::span<std::byte>{
                                      chunk->storage.data() + layout->payload_offset,
                                      payload_bytes});
            } catch (...) {
                chunk->used = old_used;
                throw;
            }
            std::construct_at(header, CommandHeader{&Invoke<Callable>, &Destroy<Callable>,
                                                    object, layout->end_offset});
            if (chunk->command_count == 0) {
                chunk->first_offset = layout->header_offset;
            }
            ++chunk->command_count;
            ++seq_;
        }

        CommandRecordingWorker& owner_;
        std::vector<Chunk*> chunks_;
        std::uint64_t seq_{};
        bool eager_publish_{};
        std::uint64_t eager_publishes_{};
    };

    /// Producer: queues everything it recorded so far, then the sink's commands, in that order,
    /// and empties the sink (lane mode: tagged with the open batch).
    void PublishSink(Sink& sink) {
        std::unique_lock lock{mutex_};
        if (failure_) {
            lock.unlock();
            sink.Discard();
            return;
        }
        if (current_ != nullptr && current_->command_count != 0) {
            PublishCurrentLocked();
        }
        for (Chunk* chunk : sink.chunks_) {
            if (chunk->command_count == 0) {
                chunk->used = 0;
                chunk->first_offset = 0;
                chunk->state = ChunkState::Free;
                continue;
            }
            accepted_commands_ += chunk->command_count;
            reserved_bytes_ += chunk->used;
            PublishChunkLocked(chunk);
        }
        sink.chunks_.clear();
    }
    [[nodiscard]] std::uint64_t SinkSpaceWaits() const noexcept {
        std::lock_guard lock{mutex_};
        return sink_space_waits_;
    }
    explicit CommandRecordingWorker(const bool measure_timing = false, const std::size_t lanes = 0,
                                    std::function<void(std::size_t)> lane_start = {},
                                    std::function<void()> on_failure = {})
        : measure_timing_{measure_timing}, lanes_{(std::min)(lanes, MaxLanes)},
          lane_start_{std::move(lane_start)}, on_failure_{std::move(on_failure)},
          worker_{lanes_ != 0
                      ? std::thread{&CommandRecordingWorker::LaneLoop, this, std::size_t{0}}
                      : std::thread{&CommandRecordingWorker::WorkerLoop, this}} {
        for (std::size_t lane = 1; lane < lanes_; ++lane) {
            extra_lanes_.emplace_back(&CommandRecordingWorker::LaneLoop, this, lane);
        }
    }

    ~CommandRecordingWorker() noexcept {
        try {
            Drain();
        } catch (...) {
            // Destruction must not throw. Drain has already discarded all queued captures before
            // reporting a callback exception.
        }
        {
            std::lock_guard lock{mutex_};
            stop_requested_ = true;
        }
        work_cv_.notify_all();
        for (auto& lane : lane_queues_) {
            lane.cv.notify_all();
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        for (auto& thread : extra_lanes_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    [[nodiscard]] std::size_t Lanes() const noexcept {
        return lanes_;
    }
    /// Lane mode: the batch the producer is recording (batch ends published so far). Producer
    /// thread only.
    [[nodiscard]] std::uint64_t ProducerBatch() const noexcept {
        return producer_batch_;
    }
    /// Lane mode: nanoseconds lane `lane` spent running chunks (SHADPS4_ASYNC_RECORDING_TIMING).
    [[nodiscard]] std::uint64_t LaneWorkerNs(const std::size_t lane) const noexcept {
        std::lock_guard lock{mutex_};
        return lane < MaxLanes ? lane_queues_[lane].worker_ns : 0;
    }
    [[nodiscard]] std::uint64_t LaneChunks(const std::size_t lane) const noexcept {
        std::lock_guard lock{mutex_};
        return lane < MaxLanes ? lane_queues_[lane].chunks : 0;
    }

    CommandRecordingWorker(const CommandRecordingWorker&) = delete;
    CommandRecordingWorker& operator=(const CommandRecordingWorker&) = delete;
    CommandRecordingWorker(CommandRecordingWorker&&) = delete;
    CommandRecordingWorker& operator=(CommandRecordingWorker&&) = delete;

    template <typename F>
        requires std::is_object_v<std::remove_cvref_t<F>>
    void Record(F&& function) {
        using Callable = std::remove_cvref_t<F>;
        static_assert(alignof(Callable) <= alignof(std::max_align_t),
                      "Command callable alignment exceeds max_align_t");
        static_assert(std::is_invocable_v<Callable&>, "Command callable must be invocable");
        RecordImpl<Callable>(
            [&](void* object) {
                std::construct_at(reinterpret_cast<Callable*>(object), std::forward<F>(function));
            },
            sizeof(Callable), alignof(Callable), 0);
    }

    template <typename Factory>
        requires std::is_object_v<std::remove_reference_t<Factory>>
    void RecordWithPayload(const std::size_t payload_bytes, Factory&& factory) {
        using FactoryType = std::remove_reference_t<Factory>;
        using Result = std::invoke_result_t<FactoryType&, std::span<std::byte>>;
        using Callable = std::remove_cvref_t<Result>;
        static_assert(!std::is_void_v<Result>, "Payload factory must return a callable");
        static_assert(std::is_object_v<Callable>, "Payload factory must return an object");
        static_assert(alignof(Callable) <= alignof(std::max_align_t),
                      "Command callable alignment exceeds max_align_t");
        static_assert(std::is_invocable_v<Callable&>, "Factory result must be invocable");

        RecordImpl<Callable>(
            [&](void* object, std::span<std::byte> payload) {
                // Keep the result's lifetime through placement construction. This also supports
                // factories returning an lvalue callable, although returning by value is preferred.
                decltype(auto) result = std::invoke(factory, payload);
                if constexpr (std::is_lvalue_reference_v<decltype(result)>) {
                    std::construct_at(reinterpret_cast<Callable*>(object), result);
                } else {
                    std::construct_at(reinterpret_cast<Callable*>(object), std::move(result));
                }
            },
            sizeof(Callable), alignof(Callable), payload_bytes);
    }

    /// Publishes the producer's partial chunk without waiting, so everything recorded so far runs
    /// even if nothing else is recorded for a while (an asynchronous submission, for instance).
    /// Lane mode: `batch_end` = the command just recorded ended the batch (its submission);
    /// everything recorded from now on belongs to the next batch.
    void Publish(const bool batch_end = false) {
        std::unique_lock lock{mutex_};
        if (current_ != nullptr && !failure_ && current_->command_count != 0) {
            PublishCurrentLocked();
        }
        if (batch_end) {
            ++producer_batch_;
        }
    }

    /// Publishes the producer's partial chunk and waits for every prior command to finish.
    /// Rethrows the first callback exception after all commands through this drain point have been
    /// invoked or discarded and destroyed.
    void Drain() {
        const auto drain_start = measure_timing_ ? Clock::now() : Clock::time_point{};
        const auto finish_drain_timing = [this, drain_start] {
            if (measure_timing_) {
                drain_ns_ += ToNanoseconds(Clock::now() - drain_start);
            }
        };
        Chunk* discard_current = nullptr;
        std::uint64_t target{};
        {
            std::unique_lock lock{mutex_};
            if (current_ != nullptr) {
                if (failure_) {
                    discard_current = current_;
                    current_ = nullptr;
                    discard_current->state = ChunkState::Discarding;
                } else if (current_->command_count != 0) {
                    PublishCurrentLocked();
                } else {
                    ReleaseEmptyCurrentLocked();
                }
            }
            target = accepted_commands_;
        }

        if (discard_current != nullptr) {
            DestroyChunkContents(*discard_current);
            std::lock_guard lock{mutex_};
            ReleaseDiscardedChunkLocked(*discard_current);
        }

        std::exception_ptr error;
        {
            std::unique_lock lock{mutex_};
            done_cv_.wait(lock, [this, target] { return completed_commands_ >= target; });
            error = failure_;
        }
        if (error) {
            finish_drain_timing();
            std::rethrow_exception(error);
        }
        finish_drain_timing();
    }

    /// Returns cumulative work counters. Only the sole producer may call this, after Drain.
    /// Timing counters written by the worker are synchronized by the queue mutex.
    [[nodiscard]] Counters GetCounters() const noexcept {
        std::lock_guard lock{mutex_};
        return Counters{accepted_commands_, published_chunks_, reserved_bytes_,
                        worker_ns_,         drain_ns_,         space_wait_ns_};
    }

private:
    // SHADPS4_RECORD_BYTES_STATS=1 (diagnostic): bytes recorded per command type (producer only).
    // Namespace-scope flag: a function-local static would add a guard check to every Record.
    static bool RecordBytesStatsEnabled() {
        return g_record_bytes_stats;
    }
    static void NoteRecordBytes(const char* type, std::size_t bytes) {
        struct Entry {
            const char* type;
            std::uint64_t count;
            std::uint64_t bytes;
        };
        static std::vector<Entry> entries;
        static std::uint64_t total = 0;
        auto it = std::find_if(entries.begin(), entries.end(),
                               [type](const Entry& e) { return e.type == type; });
        if (it == entries.end()) {
            entries.push_back({type, 0, 0});
            it = entries.end() - 1;
        }
        ++it->count;
        it->bytes += bytes;
        if ((++total & ((1u << 21) - 1)) != 0) {
            return;
        }
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.bytes > b.bytes; });
        std::uint64_t all_bytes = 0;
        for (const auto& e : entries) {
            all_bytes += e.bytes;
        }
        std::string line = "RECORD_BYTES: records " + std::to_string(total) + " bytes " +
                           std::to_string(all_bytes);
        for (std::size_t i = 0; i < (std::min<std::size_t>)(entries.size(), 8); ++i) {
            std::string name = entries[i].type;
            if (name.size() > 90) {
                name = name.substr(name.size() - 90);
            }
            line += " | " + std::to_string(entries[i].count) + "x " +
                    std::to_string(entries[i].bytes / (std::max<std::uint64_t>)(entries[i].count, 1)) +
                    "B " + name;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
        for (auto& e : entries) {
            e.count = e.bytes = 0;
        }
    }

    using Clock = std::chrono::steady_clock;

    static std::uint64_t ToNanoseconds(const Clock::duration duration) noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    }

    template <typename Callable, typename Construct>
    void RecordImpl(Construct&& construct, const std::size_t object_bytes,
                    const std::size_t object_alignment, const std::size_t payload_bytes) {
        if (RecordBytesStatsEnabled()) {
            NoteRecordBytes(typeid(Callable).name(), object_bytes + payload_bytes);
        }
        if (TryRecordFast<Callable>(construct, object_bytes, object_alignment, payload_bytes)) {
            return;
        }

        std::unique_lock lock{mutex_};
        CheckAcceptingLocked();
        Reservation reservation =
            ReserveLocked(lock, object_bytes, object_alignment, payload_bytes);
        lock.unlock();
        try {
            if constexpr (std::is_invocable_v<Construct&, void*, std::span<std::byte>>) {
                std::invoke(construct, reservation.object, reservation.payload);
            } else {
                std::invoke(construct, reservation.object);
            }
        } catch (...) {
            RollbackReservation(reservation);
            throw;
        }

        std::construct_at(reservation.header,
                          CommandHeader{&Invoke<Callable>, &Destroy<Callable>, reservation.object,
                                        reservation.layout.end_offset});
        CommitReservation(reservation);
        if (reservation.chunk->used == ChunkBytes) {
            std::lock_guard publish_lock{mutex_};
            // A callback may fail while the producer is constructing this command. If that
            // happened before this lock was acquired, leave the raced command in the producer-owned
            // chunk so Drain can destroy it after the worker's failure is observed.
            if (!failure_) {
                assert(current_ == reservation.chunk);
                PublishCurrentLocked();
            }
        }
    }

    template <typename Callable, typename Construct>
    [[nodiscard]] bool TryRecordFast(Construct& construct, const std::size_t object_bytes,
                                     const std::size_t object_alignment,
                                     const std::size_t payload_bytes) {
        // The failure flag is published by the worker while holding mutex_. A false result does
        // not reserve any state that the worker can observe; a raced command remains in current_
        // for Drain to discard if failure is published before its full-chunk handoff.
        if (failure_flag_.load(std::memory_order_acquire)) {
            return false;
        }

        // Check the empty layout first so an oversized command never publishes older work before
        // throwing. This also means the fast path has the same error behavior as ReserveLocked.
        if (!TryLayout(0, object_bytes, object_alignment, payload_bytes)) {
            throw std::length_error("command recording worker command exceeds chunk size");
        }

        Chunk* const chunk = current_;
        if (chunk == nullptr) {
            return false;
        }
        const auto layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
        if (!layout) {
            return false;
        }

        const std::size_t old_used = chunk->used;
        const std::size_t old_first_offset = chunk->first_offset;
        const std::size_t old_command_count = chunk->command_count;
        chunk->used = layout->end_offset;
        Reservation reservation{
            .chunk = chunk,
            .old_used = old_used,
            .old_first_offset = old_first_offset,
            .old_command_count = old_command_count,
            .layout = *layout,
            .header =
                reinterpret_cast<CommandHeader*>(chunk->storage.data() + layout->header_offset),
            .object = chunk->storage.data() + layout->object_offset,
            .payload =
                std::span<std::byte>{chunk->storage.data() + layout->payload_offset, payload_bytes},
        };
        try {
            if constexpr (std::is_invocable_v<Construct&, void*, std::span<std::byte>>) {
                std::invoke(construct, reservation.object, reservation.payload);
            } else {
                std::invoke(construct, reservation.object);
            }
        } catch (...) {
            RollbackReservation(reservation);
            throw;
        }

        std::construct_at(reservation.header,
                          CommandHeader{&Invoke<Callable>, &Destroy<Callable>, reservation.object,
                                        reservation.layout.end_offset});
        CommitReservation(reservation);
        if (reservation.chunk->used == ChunkBytes) {
            std::lock_guard publish_lock{mutex_};
            if (!failure_) {
                assert(current_ == reservation.chunk);
                PublishCurrentLocked();
            }
        }
        return true;
    }

    void CommitReservation(const Reservation& reservation) {
        reservation.chunk->command_count = reservation.old_command_count + 1;
        if (reservation.old_command_count == 0) {
            reservation.chunk->first_offset = reservation.layout.header_offset;
        }
        ++accepted_commands_;
        reserved_bytes_ += reservation.layout.end_offset - reservation.old_used;
    }

    template <typename Callable>
    static void Invoke(void* object) {
        std::invoke(*std::launder(reinterpret_cast<Callable*>(object)));
    }

    template <typename Callable>
    static void Destroy(void* object) noexcept {
        std::destroy_at(std::launder(reinterpret_cast<Callable*>(object)));
    }

    static constexpr std::size_t AlignUp(const std::size_t value, const std::size_t alignment) {
        const std::size_t mask = alignment - 1;
        if (value > (static_cast<std::size_t>(-1) - mask)) {
            throw std::length_error("command recording worker layout overflow");
        }
        return (value + mask) & ~mask;
    }

    [[nodiscard]] static std::optional<Layout> TryLayout(const std::size_t begin,
                                                         const std::size_t object_bytes,
                                                         const std::size_t object_alignment,
                                                         const std::size_t payload_bytes) {
        const std::size_t header_offset = AlignUp(begin, alignof(CommandHeader));
        if (header_offset > (static_cast<std::size_t>(-1) - sizeof(CommandHeader))) {
            throw std::length_error("command recording worker header overflow");
        }
        const std::size_t object_offset =
            AlignUp(header_offset + sizeof(CommandHeader), object_alignment);
        if (object_offset > (static_cast<std::size_t>(-1) - object_bytes)) {
            throw std::length_error("command recording worker callable overflow");
        }
        const std::size_t payload_offset =
            AlignUp(object_offset + object_bytes, alignof(std::max_align_t));
        if (payload_offset > (static_cast<std::size_t>(-1) - payload_bytes)) {
            throw std::length_error("command recording worker payload overflow");
        }
        const std::size_t raw_end_offset = payload_offset + payload_bytes;
        if (raw_end_offset > (static_cast<std::size_t>(-1) - alignof(CommandHeader) + 1)) {
            throw std::length_error("command recording worker end overflow");
        }
        // Traversal starts at the next command header, so include the alignment padding in the
        // reservation. Without this, a non-aligned payload leaves the next header in the gap.
        const std::size_t end_offset = AlignUp(raw_end_offset, alignof(CommandHeader));
        if (end_offset > ChunkBytes) {
            return std::nullopt;
        }
        return Layout{header_offset, object_offset, payload_offset, end_offset};
    }

    void CheckAcceptingLocked() {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        if (stop_requested_) {
            throw std::runtime_error("command recording worker is stopped");
        }
    }

    [[nodiscard]] bool HasFreeChunkLocked() const {
        for (std::size_t i = 0; i < active_chunks_; ++i) {
            if (chunks_[i].state == ChunkState::Free) {
                return true;
            }
        }
        return false;
    }

    /// SHADPS4_REC_CHUNKS=N (2..ChunkCount, default 10): chunks the producer may fill before it
    /// waits for the worker. More chunks let the command processor run further ahead of the
    /// recording thread (fire scene: the CP back waited 6-10% on a full ring, 2026-09-27).
    static std::size_t ReadActiveChunks() {
        const char* value = std::getenv("SHADPS4_REC_CHUNKS");
        unsigned long long n = value ? std::strtoull(value, nullptr, 10) : 10ull;
        if (n == 0) {
            n = 10; // unparseable or 0: the default, not the 2-chunk floor
        }
        return static_cast<std::size_t>(
            std::clamp<unsigned long long>(n, 2ull, static_cast<unsigned long long>(SinkBase)));
    }

    Chunk* AcquireCurrentLocked(std::unique_lock<std::mutex>& lock) {
        while (current_ == nullptr) {
            const auto can_acquire = [this] {
                return stop_requested_ || failure_ || HasFreeChunkLocked();
            };
            if (measure_timing_ && !stop_requested_ && !failure_ && !HasFreeChunkLocked()) {
                const auto wait_start = Clock::now();
                space_cv_.wait(lock, can_acquire);
                space_wait_ns_ += ToNanoseconds(Clock::now() - wait_start);
            } else {
                space_cv_.wait(lock, can_acquire);
            }
            CheckAcceptingLocked();
            for (std::size_t i = 0; i < active_chunks_; ++i) {
                auto& chunk = chunks_[i];
                if (chunk.state == ChunkState::Free) {
                    chunk.state = ChunkState::Building;
                    current_ = &chunk;
                    break;
                }
            }
        }
        return current_;
    }

    [[nodiscard]] Reservation ReserveLocked(std::unique_lock<std::mutex>& lock,
                                            const std::size_t object_bytes,
                                            const std::size_t object_alignment,
                                            const std::size_t payload_bytes) {
        // Validate against an empty chunk before publishing the current one. An oversized command
        // therefore cannot unexpectedly publish older work before throwing.
        const auto empty_layout = TryLayout(0, object_bytes, object_alignment, payload_bytes);
        if (!empty_layout) {
            throw std::length_error("command recording worker command exceeds chunk size");
        }

        Chunk* chunk = AcquireCurrentLocked(lock);
        auto layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
        if (!layout) {
            assert(chunk->command_count != 0);
            PublishCurrentLocked();
            chunk = AcquireCurrentLocked(lock);
            layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
            assert(layout.has_value());
        }

        const std::size_t old_used = chunk->used;
        const std::size_t old_first_offset = chunk->first_offset;
        const std::size_t old_command_count = chunk->command_count;
        chunk->used = layout->end_offset;
        return Reservation{
            .chunk = chunk,
            .old_used = old_used,
            .old_first_offset = old_first_offset,
            .old_command_count = old_command_count,
            .layout = *layout,
            .header =
                reinterpret_cast<CommandHeader*>(chunk->storage.data() + layout->header_offset),
            .object = chunk->storage.data() + layout->object_offset,
            .payload =
                std::span<std::byte>{chunk->storage.data() + layout->payload_offset, payload_bytes},
        };
    }

    void RollbackReservation(const Reservation& reservation) {
        assert(current_ == reservation.chunk);
        assert(reservation.chunk->state == ChunkState::Building);
        reservation.chunk->used = reservation.old_used;
        reservation.chunk->first_offset = reservation.old_first_offset;
        reservation.chunk->command_count = reservation.old_command_count;
    }

    void PublishCurrentLocked() {
        assert(current_ != nullptr);
        Chunk* const chunk = current_;
        current_ = nullptr;
        PublishChunkLocked(chunk);
    }

    void PublishChunkLocked(Chunk* chunk) {
        assert(chunk->command_count != 0);
        chunk->state = ChunkState::Queued;
        if (lanes_ != 0) {
            // A chunk is tagged when published: batch ends publish the chunk holding the
            // submission first, so a chunk never holds two batches.
            chunk->batch = producer_batch_;
            LaneQueue& lane = lane_queues_[producer_batch_ % lanes_];
            assert(lane.size < ChunkCount);
            lane.queue[(lane.head + lane.size) % ChunkCount] = chunk;
            ++lane.size;
            ++published_chunks_;
            lane.cv.notify_one();
            return;
        }
        assert(published_size_ < ChunkCount);
        published_queue_[(published_head_ + published_size_) % ChunkCount] = chunk;
        ++published_size_;
        ++published_chunks_;
        work_cv_.notify_one();
    }

    /// A freed chunk wakes whoever allocates from its region.
    void NotifyFreedLocked(const Chunk& chunk) {
        if (static_cast<std::size_t>(&chunk - chunks_.data()) >= SinkBase) {
            sink_space_cv_.notify_all();
        } else {
            space_cv_.notify_one();
        }
    }

    /// P1c: a free chunk of the sink region for a Sink (blocks until lanes free one).
    Chunk* AcquireSinkChunk() {
        std::unique_lock lock{mutex_};
        for (;;) {
            CheckAcceptingLocked();
            for (std::size_t i = SinkBase; i < ChunkCount; ++i) {
                if (chunks_[i].state == ChunkState::Free) {
                    chunks_[i].state = ChunkState::Building;
                    return &chunks_[i];
                }
            }
            ++sink_space_waits_;
            sink_space_cv_.wait(lock);
        }
    }

    void FreeSinkChunk(Chunk& chunk) {
        DestroyChunkContents(chunk);
        std::lock_guard lock{mutex_};
        chunk.used = 0;
        chunk.first_offset = 0;
        chunk.command_count = 0;
        chunk.state = ChunkState::Free;
        sink_space_cv_.notify_all();
    }

    void ReleaseEmptyCurrentLocked() {
        assert(current_ != nullptr);
        assert(current_->command_count == 0);
        current_->used = 0;
        current_->first_offset = 0;
        current_->state = ChunkState::Free;
        current_ = nullptr;
        space_cv_.notify_one();
    }

    [[nodiscard]] Chunk* PopPublishedLocked() {
        assert(published_size_ != 0);
        Chunk* chunk = published_queue_[published_head_];
        published_head_ = (published_head_ + 1) % ChunkCount;
        --published_size_;
        chunk->state = ChunkState::Executing;
        return chunk;
    }

    static void DestroyChunkContents(Chunk& chunk) noexcept {
        std::size_t offset = chunk.first_offset;
        for (std::size_t index = 0; index < chunk.command_count; ++index) {
            auto* const header =
                std::launder(reinterpret_cast<CommandHeader*>(chunk.storage.data() + offset));
            const std::size_t next_offset = header->next_offset;
            header->destroy(header->object);
            offset = next_offset;
        }
        assert(offset == chunk.used);
    }

    void ReleaseCompletedChunkLocked(Chunk& chunk) {
        assert(chunk.state == ChunkState::Executing);
        completed_commands_ += chunk.command_count;
        chunk.used = 0;
        chunk.first_offset = 0;
        chunk.command_count = 0;
        chunk.state = ChunkState::Free;
        done_cv_.notify_all();
        NotifyFreedLocked(chunk);
    }

    void ReleaseDiscardedChunkLocked(Chunk& chunk) {
        assert(chunk.state == ChunkState::Discarding);
        completed_commands_ += chunk.command_count;
        chunk.used = 0;
        chunk.first_offset = 0;
        chunk.command_count = 0;
        chunk.state = ChunkState::Free;
        done_cv_.notify_all();
        NotifyFreedLocked(chunk);
    }

    void WorkerLoop() {
        for (;;) {
            Chunk* chunk{};
            {
                std::unique_lock lock{mutex_};
                work_cv_.wait(lock, [this] { return stop_requested_ || published_size_ != 0; });
                if (published_size_ == 0) {
                    if (stop_requested_) {
                        return;
                    }
                    continue;
                }
                chunk = PopPublishedLocked();
            }
            if (auto* const hook = chunk_hook_.load(std::memory_order_relaxed)) {
                hook();
            }

            const auto worker_start = measure_timing_ ? Clock::now() : Clock::time_point{};
            std::exception_ptr callback_error;
            std::size_t offset = chunk->first_offset;
            for (std::size_t index = 0; index < chunk->command_count; ++index) {
                auto* const header =
                    std::launder(reinterpret_cast<CommandHeader*>(chunk->storage.data() + offset));
                const std::size_t next_offset = header->next_offset;
                if (!callback_error) {
                    try {
                        header->invoke(header->object);
                    } catch (...) {
                        callback_error = std::current_exception();
                    }
                }
                header->destroy(header->object);
                offset = next_offset;
            }
            assert(offset == chunk->used);
            const std::uint64_t chunk_ns =
                measure_timing_ ? ToNanoseconds(Clock::now() - worker_start) : 0;

            if (!callback_error) {
                std::lock_guard lock{mutex_};
                worker_ns_ += chunk_ns;
                ReleaseCompletedChunkLocked(*chunk);
                continue;
            }

            std::array<Chunk*, ChunkCount> discarded{};
            std::size_t discarded_count = 0;
            {
                std::lock_guard lock{mutex_};
                failure_ = callback_error;
                failure_flag_.store(true, std::memory_order_release);
                worker_ns_ += chunk_ns;
                chunk->state = ChunkState::Discarding;
                completed_commands_ += chunk->command_count;
                chunk->used = 0;
                chunk->first_offset = 0;
                chunk->command_count = 0;
                chunk->state = ChunkState::Free;
                while (published_size_ != 0) {
                    auto* const queued = PopPublishedLocked();
                    queued->state = ChunkState::Discarding;
                    discarded[discarded_count++] = queued;
                }
                done_cv_.notify_all();
                space_cv_.notify_all();
            }

            // Destructors can execute arbitrary user code; keep them outside the queue mutex just
            // like callbacks.
            for (std::size_t index = 0; index < discarded_count; ++index) {
                Chunk& queued = *discarded[index];
                const auto discard_start = measure_timing_ ? Clock::now() : Clock::time_point{};
                DestroyChunkContents(queued);
                const std::uint64_t discard_ns =
                    measure_timing_ ? ToNanoseconds(Clock::now() - discard_start) : 0;
                std::lock_guard lock{mutex_};
                worker_ns_ += discard_ns;
                ReleaseDiscardedChunkLocked(queued);
            }
            done_cv_.notify_all();
            return;
        }
    }

    /// Lane mode: lane `lane` runs the chunks of batches k with k % lanes_ == lane, in order.
    void LaneLoop(const std::size_t lane_index) {
        if (lane_start_) {
            lane_start_(lane_index);
        }
        LaneQueue& lane = lane_queues_[lane_index];
        for (;;) {
            Chunk* chunk{};
            {
                std::unique_lock lock{mutex_};
                lane.cv.wait(lock, [&] { return stop_requested_ || failure_ || lane.size != 0; });
                if (failure_) {
                    return; // the failing lane discarded every queued chunk
                }
                if (lane.size == 0) {
                    if (stop_requested_) {
                        return;
                    }
                    continue;
                }
                chunk = lane.queue[lane.head];
                lane.head = (lane.head + 1) % ChunkCount;
                --lane.size;
                chunk->state = ChunkState::Executing;
            }
            if (auto* const hook = chunk_hook_.load(std::memory_order_relaxed)) {
                hook();
            }

            const auto worker_start = measure_timing_ ? Clock::now() : Clock::time_point{};
            std::exception_ptr callback_error;
            std::size_t offset = chunk->first_offset;
            for (std::size_t index = 0; index < chunk->command_count; ++index) {
                auto* const header =
                    std::launder(reinterpret_cast<CommandHeader*>(chunk->storage.data() + offset));
                const std::size_t next_offset = header->next_offset;
                if (!callback_error) {
                    try {
                        header->invoke(header->object);
                    } catch (...) {
                        callback_error = std::current_exception();
                    }
                }
                header->destroy(header->object);
                offset = next_offset;
            }
            assert(offset == chunk->used);
            const std::uint64_t chunk_ns =
                measure_timing_ ? ToNanoseconds(Clock::now() - worker_start) : 0;

            if (!callback_error) {
                std::lock_guard lock{mutex_};
                worker_ns_ += chunk_ns;
                lane.worker_ns += chunk_ns;
                ++lane.chunks;
                ReleaseCompletedChunkLocked(*chunk);
                continue;
            }

            std::array<Chunk*, ChunkCount> discarded{};
            std::size_t discarded_count = 0;
            bool first_failure = false;
            {
                std::lock_guard lock{mutex_};
                first_failure = !failure_;
                if (first_failure) {
                    failure_ = callback_error;
                    failure_flag_.store(true, std::memory_order_release);
                }
                worker_ns_ += chunk_ns;
                lane.worker_ns += chunk_ns;
                completed_commands_ += chunk->command_count;
                chunk->used = 0;
                chunk->first_offset = 0;
                chunk->command_count = 0;
                chunk->state = ChunkState::Free;
                for (auto& other : lane_queues_) {
                    while (other.size != 0) {
                        Chunk* const queued = other.queue[other.head];
                        other.head = (other.head + 1) % ChunkCount;
                        --other.size;
                        queued->state = ChunkState::Discarding;
                        discarded[discarded_count++] = queued;
                    }
                    other.cv.notify_all();
                }
                done_cv_.notify_all();
                space_cv_.notify_all();
            }
            if (first_failure && on_failure_) {
                on_failure_();
            }
            for (std::size_t index = 0; index < discarded_count; ++index) {
                Chunk& queued = *discarded[index];
                DestroyChunkContents(queued);
                std::lock_guard lock{mutex_};
                ReleaseDiscardedChunkLocked(queued);
            }
            done_cv_.notify_all();
            return;
        }
    }

    struct LaneQueue {
        std::array<Chunk*, ChunkCount> queue{};
        std::size_t head{};
        std::size_t size{};
        std::condition_variable cv;
        std::uint64_t worker_ns{};
        std::uint64_t chunks{};
    };

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::condition_variable space_cv_;
    std::condition_variable sink_space_cv_; ///< P1c: sink-region chunks freed.
    std::uint64_t sink_space_waits_{};
    std::array<Chunk, ChunkCount> chunks_{};
    std::array<Chunk*, ChunkCount> published_queue_{};
    const std::size_t active_chunks_{ReadActiveChunks()};
    Chunk* current_{};
    std::size_t published_head_{};
    std::size_t published_size_{};
    std::uint64_t accepted_commands_{};
    std::uint64_t completed_commands_{};
    std::uint64_t published_chunks_{};
    std::uint64_t reserved_bytes_{};
    std::exception_ptr failure_{};
    std::atomic<bool> failure_flag_{};
    bool stop_requested_{};
    const bool measure_timing_;
    static inline std::atomic<void (*)()> chunk_hook_{nullptr};
    std::uint64_t worker_ns_{};
    std::uint64_t drain_ns_{};
    std::uint64_t space_wait_ns_{};
    // Lane mode (declared before the threads that read them).
    const std::size_t lanes_;
    const std::function<void(std::size_t)> lane_start_;
    const std::function<void()> on_failure_;
    std::array<LaneQueue, MaxLanes> lane_queues_{};
    std::uint64_t producer_batch_{}; ///< Producer + mutex_ (read by the producer alone).
    std::vector<std::thread> extra_lanes_;
    std::thread worker_;
};

} // namespace Common
