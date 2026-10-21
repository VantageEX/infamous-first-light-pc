// SPDX-License-Identifier: GPL-2.0-or-later
// Logical all-or-nothing leases for G4 packet preparation. No driver, guest read or wait occurs
// under this table's mutex. The caller still owns backing pins and CPU-version revalidation.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "g4_cp_coordination.h"

namespace iss::g4 {

struct PacketResourceAccess {
    // Stable physical backing identity, shared by every virtual alias. Zero means unresolved.
    std::uint64_t backing_key{};
    std::uint64_t offset{};
    std::uint64_t size{};
    bool write{};

    [[nodiscard]] bool Valid() const noexcept {
        return backing_key != 0 && size != 0 &&
               offset <= std::numeric_limits<std::uint64_t>::max() - size;
    }
};

// Physical guest backing aliases all use the same key. These identities are disjoint
// from host pointers, so GDS and the BDA page table can join the same reservation order.
inline constexpr std::uint64_t GuestBackingKey = 1;
inline constexpr std::uint64_t GdsBackingKey = 2;
inline constexpr std::uint64_t BdaBackingKey = 3;
// Conservative graphics packet admission until exact G resource planning exists.
// A write lease on this key conflicts with every packet resource kind.
inline constexpr std::uint64_t GlobalResourceKey = 4;

struct PacketLeaseKey {
    CpLane lane{CpLane::Graphics};
    std::uint32_t qid{};
    std::uint64_t generation{};
    std::uint64_t packet_sequence{};
    friend bool operator==(const PacketLeaseKey&, const PacketLeaseKey&) = default;
};

enum class PacketReserveStatus : std::uint8_t { Reserved, Busy, Stale, Invalid, Stopped };

struct PacketReserveResult {
    PacketReserveStatus status{PacketReserveStatus::Invalid};
    std::uint64_t lease_id{};
};

class PacketHazards {
public:
    // Reserve the complete access set at once; Busy/Invalid/Stale leave no partial lease.
    [[nodiscard]] PacketReserveResult TryReserve(PacketLeaseKey key,
                                                 std::span<const PacketResourceAccess> accesses,
                                                 std::uint64_t prepared_mapping_generation,
                                                 std::uint64_t current_mapping_generation) {
        std::scoped_lock lock{mutex};
        if (stopped) return {PacketReserveStatus::Stopped, 0};
        if (key.generation == 0 || key.qid >= 57 || key.packet_sequence == 0 ||
            (key.lane == CpLane::Graphics && key.qid != 0) ||
            (key.lane == CpLane::Asc && key.qid == 0) ||
            std::any_of(accesses.begin(), accesses.end(),
                        [](const auto& access) { return !access.Valid(); }))
            return {PacketReserveStatus::Invalid, 0};
        if (prepared_mapping_generation == 0 ||
            prepared_mapping_generation != current_mapping_generation)
            return {PacketReserveStatus::Stale, 0};
        for (const auto& live : leases) {
            if (live.key == key) return {PacketReserveStatus::Invalid, 0};
            for (const auto& proposed : accesses) {
                if (std::any_of(live.accesses.begin(), live.accesses.end(),
                                [&](const auto& held) { return Conflict(proposed, held); }))
                    return {PacketReserveStatus::Busy, 0};
            }
        }
        const std::uint64_t id = next_lease_id++;
        leases.push_back(Lease{id, key, prepared_mapping_generation,
                               {accesses.begin(), accesses.end()}, false, false});
        return {PacketReserveStatus::Reserved, id};
    }

    // A guest write invalidates a prepared lease without waiting for its owner. Hot/unprotected
    // guest writes must additionally be caught by the caller's input versions before Publish.
    void InvalidateCpuWrite(std::uint64_t backing_key, std::uint64_t offset,
                            std::uint64_t size) {
        if (backing_key == 0 || size == 0 ||
            offset > std::numeric_limits<std::uint64_t>::max() - size) return;
        const PacketResourceAccess changed{backing_key, offset, size, true};
        std::scoped_lock lock{mutex};
        for (auto& lease : leases) {
            lease.stale |= std::any_of(lease.accesses.begin(), lease.accesses.end(),
                                       [&](const auto& access) {
                                           // Global graphics admission serializes G/C packets;
                                           // it is not a snapshot of every guest byte. The
                                           // legacy G path tracks CPU writes separately.
                                           return access.backing_key != GlobalResourceKey &&
                                                  Overlaps(changed, access);
                                       });
        }
    }

    // A fault with unknown backing must invalidate every plan awaiting publication.
    // The caller may retry after resolving the map; guessing an alias would be unsafe.
    void InvalidateUnpublished() {
        std::scoped_lock lock{mutex};
        for (auto& lease : leases) {
            if (!lease.published) lease.stale = true;
        }
    }

    [[nodiscard]] bool Revalidate(std::uint64_t lease_id,
                                  std::uint64_t current_mapping_generation,
                                  bool guest_inputs_stable) const {
        std::scoped_lock lock{mutex};
        const auto* lease = Find(lease_id);
        return !stopped && lease && !lease->published && !lease->stale &&
               guest_inputs_stable &&
               lease->mapping_generation == current_mapping_generation;
    }

    // Read-only snapshot of open producer tokens. The owner seals these outside all cache locks.
    [[nodiscard]] std::vector<QidTick> Dependencies(std::uint64_t lease_id) const {
        std::scoped_lock lock{mutex};
        const auto* lease = Find(lease_id);
        std::vector<QidTick> result;
        if (!lease || lease->stale || stopped) return result;
        // A global admission packet conflicts with every access on the other
        // lane. That lane is FIFO, so its newest tick covers all older ticks.
        const bool global = std::any_of(lease->accesses.begin(), lease->accesses.end(),
                                        [](const auto& access) {
                                            return access.backing_key == GlobalResourceKey;
                                        });
        if (global) {
            const auto& newest = latest_published[lease->key.lane == CpLane::Graphics ? 1 : 0];
            if (newest) result.push_back(*newest);
            return result;
        }
        for (const auto& earlier : history) {
            if (earlier.token.lane == lease->key.lane) continue;
            if (!std::any_of(lease->accesses.begin(), lease->accesses.end(),
                             [&](const auto& access) {
                                 return Conflict(access, earlier.access);
                             })) continue;
            if (std::find(result.begin(), result.end(), earlier.token) == result.end())
                result.push_back(earlier.token);
        }
        return result;
    }

    // Called only after all producer seals and input checks. Publication precedes Vulkan record.
    [[nodiscard]] bool Publish(std::uint64_t lease_id, QidTick token,
                               std::uint64_t current_mapping_generation,
                               bool guest_inputs_stable) {
        std::scoped_lock lock{mutex};
        auto* lease = Find(lease_id);
        if (stopped || !lease || lease->published || lease->stale || !guest_inputs_stable ||
            lease->mapping_generation != current_mapping_generation ||
            token.tick == 0 || token.qid != lease->key.qid ||
            token.generation != lease->key.generation || token.lane != lease->key.lane)
            return false;
        lease->published = true;
        for (const auto& access : lease->accesses) {
            if (access.backing_key == GlobalResourceKey) {
                // The later G/C FIFO tick subsumes an older global token of its lane.
                std::erase_if(history, [&](const auto& entry) {
                    return entry.access.backing_key == GlobalResourceKey &&
                           entry.token.lane == token.lane && entry.token.tick <= token.tick;
                });
            }
            history.push_back({access, token});
        }
        auto& latest = latest_published[token.lane == CpLane::Graphics ? 0 : 1];
        if (!latest || token.tick >= latest->tick) latest = token;
        return true;
    }

    // Before Publish this cancels a plan; after Publish the caller must finish or abort the run,
    // never replay a partially recorded packet.
    [[nodiscard]] bool CancelUnpublished(std::uint64_t lease_id) {
        std::scoped_lock lock{mutex};
        const auto it = std::find_if(leases.begin(), leases.end(),
                                     [&](const auto& lease) { return lease.id == lease_id; });
        if (it == leases.end() || it->published) return false;
        leases.erase(it);
        return true;
    }
    [[nodiscard]] bool FinishRecord(std::uint64_t lease_id) {
        std::scoped_lock lock{mutex};
        const auto it = std::find_if(leases.begin(), leases.end(),
                                     [&](const auto& lease) { return lease.id == lease_id; });
        if (it == leases.end() || !it->published) return false;
        leases.erase(it);
        return true;
    }
    void RetireCompleted(CpLane lane, std::uint64_t completed_tick) {
        std::scoped_lock lock{mutex};
        std::erase_if(history, [&](const auto& entry) {
            return entry.token.lane == lane && entry.token.tick <= completed_tick;
        });
        auto& latest = latest_published[lane == CpLane::Graphics ? 0 : 1];
        if (latest && latest->tick <= completed_tick) latest.reset();
    }
    void Stop() {
        std::scoped_lock lock{mutex};
        stopped = true;
        leases.clear();
        history.clear();
        latest_published = {};
    }

private:
    struct Lease {
        std::uint64_t id{};
        PacketLeaseKey key{};
        std::uint64_t mapping_generation{};
        std::vector<PacketResourceAccess> accesses{};
        bool stale{};
        bool published{};
    };
    struct HistoryEntry {
        PacketResourceAccess access{};
        QidTick token{};
    };
    [[nodiscard]] static bool Overlaps(const PacketResourceAccess& a,
                                       const PacketResourceAccess& b) noexcept {
        if (a.backing_key == GlobalResourceKey || b.backing_key == GlobalResourceKey)
            return true;
        return a.backing_key == b.backing_key && a.offset < b.offset + b.size &&
               b.offset < a.offset + a.size;
    }
    [[nodiscard]] static bool Conflict(const PacketResourceAccess& a,
                                       const PacketResourceAccess& b) noexcept {
        return (a.write || b.write) && Overlaps(a, b);
    }
    [[nodiscard]] Lease* Find(std::uint64_t id) noexcept {
        const auto it = std::find_if(leases.begin(), leases.end(),
                                     [&](const auto& lease) { return lease.id == id; });
        return it == leases.end() ? nullptr : &*it;
    }
    [[nodiscard]] const Lease* Find(std::uint64_t id) const noexcept {
        const auto it = std::find_if(leases.begin(), leases.end(),
                                     [&](const auto& lease) { return lease.id == id; });
        return it == leases.end() ? nullptr : &*it;
    }

    mutable std::mutex mutex;
    std::uint64_t next_lease_id{1};
    std::vector<Lease> leases;
    std::vector<HistoryEntry> history;
    std::array<std::optional<QidTick>, 2> latest_published{};
    bool stopped{};
};

} // namespace iss::g4
