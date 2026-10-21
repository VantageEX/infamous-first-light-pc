// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

namespace iss::game::feedback {

// Published before the EBOOT call that binds a feedback record. The object is
// the dispatch owner; the record can belong to a related object. This is an
// observation only: neither the owner nor the last published address proves
// that a later GPU binding is the same arena allocation after address reuse.
struct IdentityObservation {
    std::uint64_t owner_object{};
    std::uint64_t shader_object{};
    std::uint64_t record_address{};
    std::uint64_t sequence{};
    std::uint64_t address_reuses{};
};

class IdentityRegistry {
public:
    static IdentityRegistry& Instance() noexcept {
        static IdentityRegistry registry;
        return registry;
    }

    void Publish(std::uint64_t owner_object, std::uint64_t shader_object,
                 std::uint64_t record_address) noexcept {
        if (owner_object == 0 || shader_object == 0 || record_address == 0) return;
        std::lock_guard lock{mutex};
        auto* entry = &observations[Slot(record_address)];
        const auto reuses = entry->record_address == record_address
                                ? entry->address_reuses + 1 : 0;
        *entry = {.owner_object = owner_object,
                  .shader_object = shader_object,
                  .record_address = record_address,
                  .sequence = ++sequence,
                  .address_reuses = reuses};
    }

    [[nodiscard]] std::optional<IdentityObservation> Latest(
        std::uint64_t record_address) const noexcept {
        if (record_address == 0) return std::nullopt;
        std::lock_guard lock{mutex};
        const auto& entry = observations[Slot(record_address)];
        return entry.record_address == record_address
                   ? std::optional<IdentityObservation>{entry} : std::nullopt;
    }

    // Any remap invalidates even passive address associations. This registry
    // never authorizes changing a descriptor or retiring guest dirty state.
    void Invalidate(std::uint64_t start, std::uint64_t size) noexcept {
        if (size == 0) return;
        std::lock_guard lock{mutex};
        for (auto& entry : observations) {
            const auto address = entry.record_address;
            if (address >= start && address - start < size) entry = {};
        }
    }

private:
    static constexpr std::size_t MaxAddresses = 4096;
    static constexpr std::size_t Slot(std::uint64_t address) noexcept {
        // Guest records are 256-byte aligned. Collision only loses an
        // observation; it can never create a matching record address.
        const auto slot = address >> 8;
        return static_cast<std::size_t>((slot ^ (slot >> 12) ^ (slot >> 24)) &
                                        (MaxAddresses - 1));
    }
    IdentityRegistry() = default;
    mutable std::mutex mutex;
    std::array<IdentityObservation, MaxAddresses> observations{};
    std::uint64_t sequence{};
};

} // namespace iss::game::feedback
