// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace Vulkan {
inline std::uint64_t CacheChecksum(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (auto value : bytes) hash = (hash ^ value) * 0x100000001b3ULL;
    return hash;
}
inline std::uint64_t ReadCacheWord(std::span<const std::uint8_t> data, std::size_t offset, int size) {
    std::uint64_t value{};
    for (int i = 0; i < size; ++i) value |= std::uint64_t(data[offset + i]) << (i * 8);
    return value;
}
inline std::vector<std::uint8_t> PackDriverCache(std::span<const std::uint8_t> payload) {
    std::vector<std::uint8_t> out;
    const auto append = [&](std::uint64_t value, int size) {
        for (int i = 0; i < size; ++i) out.push_back(std::uint8_t(value >> (i * 8)));
    };
    append(0x434b5653, 4); append(1, 4); append(payload.size(), 8); append(CacheChecksum(payload), 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}
inline std::span<const std::uint8_t> UnpackDriverCache(std::span<const std::uint8_t> bytes,
    std::uint32_t vendor, std::uint32_t device, std::span<const std::uint8_t> uuid) {
    if (bytes.size() < 56 || uuid.size() != 16 || ReadCacheWord(bytes, 0, 4) != 0x434b5653 ||
        ReadCacheWord(bytes, 4, 4) != 1 || ReadCacheWord(bytes, 8, 8) != bytes.size() - 24) return {};
    const auto data = bytes.subspan(24);
    if (ReadCacheWord(bytes, 16, 8) != CacheChecksum(data) || ReadCacheWord(data, 0, 4) < 32 ||
        ReadCacheWord(data, 0, 4) > data.size() || ReadCacheWord(data, 4, 4) != 1 ||
        ReadCacheWord(data, 8, 4) != vendor || ReadCacheWord(data, 12, 4) != device ||
        !std::equal(uuid.begin(), uuid.end(), data.begin() + 16)) return {};
    return data;
}
} // namespace Vulkan
