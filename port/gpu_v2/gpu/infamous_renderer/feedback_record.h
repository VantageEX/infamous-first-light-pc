// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace iss::game::feedback {

// Raw 48-byte result bound as storage buffer 0 by the game's feedback compute pass.
// The eboot consumer reads word 1 at offset +4. Other words retain their raw GPU
// representation; the bounds operations in shader 0x3EA96DB0 compare unsigned bits.
struct Record {
    std::array<std::uint32_t, 12> words{};

    static constexpr std::size_t ByteSize = 48;
    static constexpr std::size_t CountWord = 1;
    static constexpr std::size_t FirstOutputWord = 2;
    static constexpr std::size_t SecondaryCountWord = 3;
    static constexpr std::size_t MinWord = 4;
    static constexpr std::size_t MaxWord = 7;
    static constexpr std::size_t OutputStrideWords = 16;

    // A record can be copied only while no GPU/CPU writer is active. Preserve
    // the entire allocation, including word 11, which the observed eboot path
    // does not initialize. The game and this port both use little-endian words.
    [[nodiscard]] static Record FromRawBytes(
        std::span<const std::byte, ByteSize> source) noexcept {
        static_assert(std::endian::native == std::endian::little);
        Record result;
        std::memcpy(result.words.data(), source.data(), ByteSize);
        return result;
    }

    void CopyRawBytes(std::span<std::byte, ByteSize> destination) const noexcept {
        static_assert(std::endian::native == std::endian::little);
        std::memcpy(destination.data(), words.data(), ByteSize);
    }

    // Eligible output branch of translated shader 0x3EA96DB0_0. The inputs
    // correspond to ssbo_6[2], ssbo_4[128], ssbo_3[3], ssbo_3[7] for a new
    // lane. The
    // subtraction/addition wrap as u32 before the signed conversion; the
    // ordered float comparison rejects NaN.
    [[nodiscard]] bool MainLaneEligible(std::uint32_t lane,
                                        std::uint32_t extra_lanes,
                                        std::uint32_t current_tick,
                                        std::uint32_t item_tick,
                                        std::uint32_t threshold_bits) const noexcept {
        const auto existing_count = words[0];
        if (lane < existing_count) return true;
        if (lane >= existing_count + extra_lanes) return false;
        const auto elapsed_bits = current_tick - item_tick;
        const auto elapsed = std::bit_cast<std::int32_t>(elapsed_bits);
        const auto threshold = std::bit_cast<float>(threshold_bits);
        return 0.000694444461f * static_cast<float>(elapsed) < threshold;
    }

    // Mirrors only the stores established in eboot 0x3EFDE0. Word 11 is left
    // untouched because its initialization/use is not yet proven.
    void InitializeKnownWords(std::uint32_t first_output,
                              const std::array<std::uint32_t, 3>& initial_min_bits) noexcept {
        words[0] = 0;
        words[CountWord] = 0;
        words[FirstOutputWord] = first_output;
        words[3] = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            words[MinWord + i] = initial_min_bits[i];
        }
        words[7] = 0;
        words[8] = 0;
        words[9] = 0;
        words[10] = 0;
    }

    // CPU reference for the main output branch of shader 0x3EA96DB0.
    // An item already inside the existing output range keeps its input index;
    // a new item reserves one through word 1's atomic add. Both paths update
    // bounds. The shader also writes a 64-byte output element at index *
    // OutputStrideWords; its payload is outside this record helper. The caller
    // must supply the shader-computed condition and operands.
    [[nodiscard]] std::uint32_t ApplyMainBranch(
        bool in_existing_output_range, std::uint32_t existing_index,
        const std::array<std::uint32_t, 3>& min_bits,
        const std::array<std::uint32_t, 3>& max_bits) noexcept {
        const auto index = in_existing_output_range ? existing_index
            : std::atomic_ref(words[CountWord]).fetch_add(1, std::memory_order_relaxed);
        for (std::size_t i = 0; i < 3; ++i) {
            AtomicMin(words[MinWord + i], min_bits[i]);
            AtomicMax(words[MaxWord + i], max_bits[i]);
        }
        return index;
    }

    // Other branches in the translated feedback programs reserve from word 3.
    // The caller still owns their branch conditions and output payload.
    [[nodiscard]] std::uint32_t ReserveSecondaryIndex() noexcept {
        return std::atomic_ref(words[SecondaryCountWord]).fetch_add(
            1, std::memory_order_relaxed);
    }

private:
    static void AtomicMin(std::uint32_t& word, std::uint32_t value) noexcept {
        std::atomic_ref atomic{word};
        auto old = atomic.load(std::memory_order_relaxed);
        while (value < old && !atomic.compare_exchange_weak(
                   old, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    static void AtomicMax(std::uint32_t& word, std::uint32_t value) noexcept {
        std::atomic_ref atomic{word};
        auto old = atomic.load(std::memory_order_relaxed);
        while (value > old && !atomic.compare_exchange_weak(
                   old, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }
};

static_assert(sizeof(Record) == Record::ByteSize);
static_assert(offsetof(Record, words) == 0);

} // namespace iss::game::feedback
