#include "infamous_renderer/native_instanced_batch.h"

#include <array>
#include <cstring>
#include <limits>

namespace iss::native_geometry {
namespace {

struct ApprovedKey {
    std::uint64_t hash;
    std::uint32_t vertex;
    std::uint32_t permutation;
};

constexpr std::array<ApprovedKey, 9> kApprovedKeys{{
    {0x0941F2A56E818413ull, 0xC01162F7, 4},
    {0x3B6D3A6D0DEC8019ull, 0xC01162F7, 0},
    {0x5DC3810A508DE8B2ull, 0xC01162F7, 0},
    {0x8542BC9469896934ull, 0xC01162F7, 0},
    {0xF6FFC28F5B41B58Full, 0xC01162F7, 1},
    {0x762B231483A5B016ull, 0x47B0694B, 0},
    {0x7F766609DDBCEBF5ull, 0x47B0694B, 3},
    {0x8225AAAFF659C468ull, 0x47B0694B, 0},
    {0x85C78D166865FBD7ull, 0x47B0694B, 0},
}};
constexpr std::size_t kMaximumDraws = 256;
constexpr std::size_t kMaximumSourceVertexBytes = 64 * 1024;
constexpr std::size_t kMaximumPackedVertexBytes = 16 * 1024 * 1024;
constexpr std::size_t kMaximumPackedInstanceBytes = 16 * 1024 * 1024;
constexpr std::size_t kMaximumPackedIndexBytes = 4 * 1024 * 1024;
constexpr std::uint32_t kInstanceRecordBytes = 72;

bool Equal(Bytes a, Bytes b) {
    return a.size == b.size && a.data && b.data &&
           (a.data == b.data || std::memcmp(a.data, b.data, a.size) == 0);
}

std::uint16_t ReadIndex(const std::byte* p) {
    return std::uint16_t(std::to_integer<unsigned char>(p[0])) |
           (std::uint16_t(std::to_integer<unsigned char>(p[1])) << 8);
}

Reject Validate(const InstancedDraw& d) {
    if (!IsApprovedInstancedKey(d.graphics_pipeline_key_hash, d.vertex_hash,
                                 d.vertex_permutation) ||
        d.fragment_hash != 0x1318DDD4 || !d.exact_spirv_verified || d.queue != 0 ||
        d.primitive_type != 4 || d.index_type != 0 || d.index_offset != 0 ||
        d.first_instance != 0 || !d.instance_count || d.binding_count != 1 ||
        d.attribute_count != 1 || d.vertex.sgpr_base != 2 ||
        d.vertex.dword_offset != 0 || d.vertex.instance_data != 0) {
        return Reject::UnsupportedShaderOrLayout;
    }
    if (!d.state.complete || !d.state.immutable_through_record ||
        !d.state.canonical.data || !d.state.canonical.size) {
        return Reject::UnverifiedState;
    }
    if (!d.vertex.cpu_owned || !d.vertex.mapped || !d.vertex.snapshot_stable ||
        !d.instance_cpu_owned || !d.instance_mapped || !d.instance_snapshot_stable ||
        !d.index_cpu_owned || !d.index_mapped || !d.index_snapshot_stable) {
        return Reject::UnstableSource;
    }
    const std::size_t vertex_size = d.vertex.descriptor_size;
    if (!d.vertex.guest_base || !d.vertex.bytes.data || !vertex_size ||
        vertex_size > kMaximumSourceVertexBytes || d.vertex.bytes.size != vertex_size ||
        !d.vertex.stride || d.vertex.stride > 256 ||
        vertex_size % d.vertex.stride ||
        d.vertex.guest_base > UINT64_MAX - vertex_size ||
        !d.instance_guest_base || !d.instance_bytes.data ||
        d.instance_count > kMaximumPackedInstanceBytes / kInstanceRecordBytes ||
        d.instance_bytes.size != std::size_t(d.instance_count) * kInstanceRecordBytes ||
        d.instance_guest_base > UINT64_MAX - d.instance_bytes.size ||
        !d.index_count || d.index_count > kMaximumPackedIndexBytes / 2 ||
        !d.index_bytes.data ||
        d.index_bytes.size != std::size_t(d.index_count) * 2) {
        return Reject::InvalidRange;
    }
    const std::int64_t vertices = std::int64_t(vertex_size / d.vertex.stride);
    for (std::uint32_t i = 0; i < d.index_count; ++i) {
        const std::int64_t index = ReadIndex(d.index_bytes.data + std::size_t(i) * 2);
        if (index + d.vertex_offset < 0 || index + d.vertex_offset >= vertices) {
            return Reject::OutOfBoundsIndex;
        }
    }
    return Reject::None;
}

} // namespace

bool IsApprovedInstancedKey(std::uint64_t key, std::uint32_t vertex,
                            std::uint32_t permutation) {
    for (const auto& allowed : kApprovedKeys) {
        if (allowed.hash == key && allowed.vertex == vertex &&
            allowed.permutation == permutation) return true;
    }
    return false;
}

InstancedBuildResult BuildInstanced(std::span<const InstancedDraw> draws) {
    InstancedBuildResult result{};
    const auto reject = [&](Reject reason) {
        result.reject = reason;
        result.batch = {};
        return result;
    };
    if (draws.empty() || draws.size() > kMaximumDraws) {
        return reject(Reject::EmptyOrTooManyDraws);
    }
    std::size_t vertex_bytes = 0, instance_bytes = 0, index_bytes = 0;
    const auto& first = draws.front();
    for (const InstancedDraw& d : draws) {
        if (const Reject reason = Validate(d); reason != Reject::None) return reject(reason);
        if (d.graphics_pipeline_key_hash != first.graphics_pipeline_key_hash ||
            d.vertex_hash != first.vertex_hash ||
            d.vertex_permutation != first.vertex_permutation ||
            d.vertex.stride != first.vertex.stride ||
            d.vertex.descriptor_control_word != first.vertex.descriptor_control_word ||
            d.vertex.format_word != first.vertex.format_word ||
            !Equal(d.state.canonical, first.state.canonical)) {
            return reject(Reject::DifferentState);
        }
        if (d.vertex.bytes.size > kMaximumPackedVertexBytes - vertex_bytes ||
            d.instance_bytes.size > kMaximumPackedInstanceBytes - instance_bytes ||
            d.index_bytes.size > kMaximumPackedIndexBytes - index_bytes) {
            return reject(Reject::OutputOverflow);
        }
        const std::int64_t rebased_vertex =
            std::int64_t(d.vertex_offset) + std::int64_t(vertex_bytes / d.vertex.stride);
        const std::size_t first_instance = instance_bytes / kInstanceRecordBytes;
        if (rebased_vertex < INT32_MIN || rebased_vertex > INT32_MAX ||
            first_instance > UINT32_MAX - d.instance_count ||
            index_bytes / 2 > UINT32_MAX) {
            return reject(Reject::OutputOverflow);
        }
        vertex_bytes += d.vertex.bytes.size;
        instance_bytes += d.instance_bytes.size;
        index_bytes += d.index_bytes.size;
    }
    auto& batch = result.batch;
    batch.vertices.reserve(vertex_bytes);
    batch.instances.reserve(instance_bytes);
    batch.indices.reserve(index_bytes);
    batch.commands.reserve(draws.size());
    batch.vertex_stride = first.vertex.stride;
    batch.vertex_control_word = first.vertex.descriptor_control_word;
    batch.vertex_format_word = first.vertex.format_word;
    batch.graphics_pipeline_key_hash = first.graphics_pipeline_key_hash;
    batch.vertex_hash = first.vertex_hash;
    batch.vertex_permutation = first.vertex_permutation;
    for (const InstancedDraw& d : draws) {
        batch.commands.push_back({
            d.index_count, d.instance_count, std::uint32_t(batch.indices.size() / 2),
            std::int32_t(std::int64_t(d.vertex_offset) +
                         std::int64_t(batch.vertices.size() / d.vertex.stride)),
            std::uint32_t(batch.instances.size() / kInstanceRecordBytes),
        });
        batch.vertices.insert(batch.vertices.end(), d.vertex.bytes.data,
                              d.vertex.bytes.data + d.vertex.bytes.size);
        batch.instances.insert(batch.instances.end(), d.instance_bytes.data,
                               d.instance_bytes.data + d.instance_bytes.size);
        batch.indices.insert(batch.indices.end(), d.index_bytes.data,
                             d.index_bytes.data + d.index_bytes.size);
    }
    batch.draw_count = std::uint32_t(draws.size());
    return result;
}

} // namespace iss::native_geometry
