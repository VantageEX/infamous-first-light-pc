// SPDX-License-Identifier: GPL-2.0-or-later
// Owning, checked control-list plan for the game's generic buffer-copy HLE shader.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace iss::g4 {

struct CopyHleControl {
    std::uint32_t dst_index{};
    std::uint32_t src_index{};
    std::uint32_t end{};
};
static_assert(sizeof(CopyHleControl) == 12);

struct CopyHleOp {
    std::uint64_t src_offset{};
    std::uint64_t dst_offset{};
    std::uint64_t size{};
};

struct CopyHleBatch {
    std::size_t first{}, count{};
    std::uint64_t src_begin{}, src_end{};
    std::uint64_t dst_begin{}, dst_end{};
};

struct CopyHlePlan {
    std::vector<CopyHleOp> copies{};
    std::vector<CopyHleBatch> batches{};
};

[[nodiscard]] inline std::optional<CopyHlePlan> BuildCopyHlePlan(
    std::span<const CopyHleControl> controls, std::uint32_t stride,
    std::uint64_t source_bytes, std::uint64_t destination_bytes) {
    constexpr std::uint64_t MaxBatchDistance = 64ull << 20;
    constexpr std::size_t MaxCopies = 1u << 20;
    if (stride == 0 || controls.size() > MaxCopies) return std::nullopt;
    CopyHlePlan plan;
    plan.copies.reserve(controls.size());
    for (const auto& control : controls) {
        const auto src = std::uint64_t{control.src_index} * stride;
        const auto dst = std::uint64_t{control.dst_index} * stride;
        const auto count = std::uint64_t{control.end} + 1;
        if (count > std::numeric_limits<std::uint64_t>::max() / stride)
            return std::nullopt;
        const auto bytes = count * stride;
        if (src > source_bytes || bytes > source_bytes - src ||
            dst > destination_bytes || bytes > destination_bytes - dst)
            return std::nullopt;
        plan.copies.push_back({src, dst, bytes});
    }
    for (std::size_t i = 0; i < plan.copies.size();) {
        const auto& first = plan.copies[i];
        CopyHleBatch batch{i, 1, first.src_offset, first.src_offset + first.size,
                           first.dst_offset, first.dst_offset + first.size};
        while (batch.first + batch.count < plan.copies.size()) {
            const auto& next = plan.copies[batch.first + batch.count];
            const auto src_begin = std::min(batch.src_begin, next.src_offset);
            const auto src_end = std::max(batch.src_end, next.src_offset + next.size);
            const auto dst_begin = std::min(batch.dst_begin, next.dst_offset);
            const auto dst_end = std::max(batch.dst_end, next.dst_offset + next.size);
            if (src_end - src_begin > MaxBatchDistance ||
                dst_end - dst_begin > MaxBatchDistance) break;
            batch.src_begin = src_begin;
            batch.src_end = src_end;
            batch.dst_begin = dst_begin;
            batch.dst_end = dst_end;
            ++batch.count;
        }
        plan.batches.push_back(batch);
        i += batch.count;
    }
    return plan;
}

} // namespace iss::g4
