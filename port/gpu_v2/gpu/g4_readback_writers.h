// SPDX-License-Identifier: GPL-2.0-or-later
// Bounded exact producer history for a G-owned readback preflight. All calls are
// under BufferCache::coherence_mutex. Unknown history fails closed.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "g4_cp_coordination.h"

namespace iss::g4 {

struct ReadbackWriterRecord {
    QidTick producer{};
    std::uint64_t address{};
    std::uint64_t size{};
    std::uint64_t mapping_generation{};
    std::uint64_t write_sequence{};
};

struct ReadbackWriterSnapshot {
    std::uint64_t address{};
    std::uint64_t size{};
    std::uint64_t mapping_generation{};
    std::uint64_t write_sequence{};
    std::array<std::uint64_t, 2> writer_ticks{};
    std::array<QidTick, 2> producers{};
};

template <std::size_t Capacity = 4096>
class ReadbackWriterLedger {
    static_assert(Capacity > 0);

public:
    void Publish(ReadbackWriterRecord record) noexcept {
        if (!Valid(record)) return;
        records[next] = record;
        next = (next + 1) % Capacity;
        if (count < Capacity) ++count;
    }

    void Invalidate(std::uint64_t address, std::uint64_t size) noexcept {
        if (size == 0) return;
        if (address > UINT64_MAX - size) {
            for (std::size_t i = 0; i < count; ++i) records[i].size = 0;
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            auto& record = records[i];
            if (record.size && Overlaps(address, size, record.address, record.size))
                record.size = 0;
        }
    }

    [[nodiscard]] std::optional<ReadbackWriterSnapshot> Capture(
        std::uint64_t address, std::uint64_t size, std::uint64_t mapping_generation,
        std::uint64_t write_sequence, std::array<std::uint64_t, 2> writer_ticks) const noexcept {
        if (size == 0 || address > UINT64_MAX - size || mapping_generation == 0)
            return std::nullopt;
        ReadbackWriterSnapshot snapshot{address, size, mapping_generation,
                                        write_sequence, writer_ticks, {}};
        for (std::size_t lane = 0; lane < 2; ++lane) {
            if (writer_ticks[lane] == 0) continue;
            bool found = false;
            for (std::size_t i = 0; i < count; ++i) {
                const auto index = (next + Capacity - 1 - i) % Capacity;
                const auto& record = records[index];
                if (record.producer.lane != (lane == 0 ? CpLane::Graphics : CpLane::Asc) ||
                    record.producer.tick != writer_ticks[lane] ||
                    record.mapping_generation != mapping_generation ||
                    !Overlaps(address, size, record.address, record.size)) continue;
                snapshot.producers[lane] = record.producer;
                found = true;
                break;
            }
            if (!found) return std::nullopt;
        }
        return snapshot;
    }

    [[nodiscard]] bool StillCurrent(const ReadbackWriterSnapshot& snapshot,
                                    std::uint64_t mapping_generation,
                                    std::uint64_t write_sequence,
                                    std::array<std::uint64_t, 2> writer_ticks) const noexcept {
        if (snapshot.mapping_generation != mapping_generation ||
            snapshot.write_sequence != write_sequence ||
            snapshot.writer_ticks != writer_ticks) return false;
        const auto fresh = Capture(snapshot.address, snapshot.size, mapping_generation,
                                   write_sequence, writer_ticks);
        return fresh && fresh->producers == snapshot.producers;
    }

private:
    static bool Overlaps(std::uint64_t a, std::uint64_t a_size,
                         std::uint64_t b, std::uint64_t b_size) noexcept {
        return a_size != 0 && b_size != 0 && a < b + b_size && b < a + a_size;
    }
    static bool Valid(const ReadbackWriterRecord& record) noexcept {
        return record.size != 0 && record.address <= UINT64_MAX - record.size &&
               record.mapping_generation != 0 && record.write_sequence != 0 &&
               record.producer.tick != 0 && record.producer.generation != 0 &&
               record.producer.qid < 57 &&
               (record.producer.lane == CpLane::Graphics ? record.producer.qid == 0
                                                        : record.producer.qid != 0);
    }

    std::array<ReadbackWriterRecord, Capacity> records{};
    std::size_t next{};
    std::size_t count{};
};

} // namespace iss::g4
