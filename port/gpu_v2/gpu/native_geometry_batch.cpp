#include "infamous_renderer/native_geometry_batch.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace iss::native_geometry {
namespace {

constexpr std::uint32_t kVertexHash = 0x0BEB7156;
constexpr std::uint32_t kFragmentHash = 0x6353E48B;
constexpr std::uint64_t kGraphicsKeyHash = 0x6794975E42E32374;
constexpr std::array<std::uint32_t, 3> kFormats{418732, 299948, 193068};
constexpr std::array<std::uint32_t, 3> kDwordOffsets{0, 8, 4};
constexpr std::size_t kMaximumDraws = 256;
constexpr std::size_t kMaximumSourceStreamBytes = 32 * 1024;
constexpr std::size_t kMaximumPackedStreamBytes = 16 * 1024 * 1024;
constexpr std::size_t kMaximumPackedIndexBytes = 4 * 1024 * 1024;

bool Equal(Bytes a, Bytes b) {
    return a.size == b.size && a.data && b.data &&
           (a.data == b.data || std::memcmp(a.data, b.data, a.size) == 0);
}

std::uint16_t ReadIndex(const std::byte* p) {
    return std::uint16_t(std::to_integer<unsigned char>(p[0])) |
           (std::uint16_t(std::to_integer<unsigned char>(p[1])) << 8);
}

Reject ValidateDraw(const Draw& draw) {
    if (draw.vertex_hash != kVertexHash || draw.fragment_hash != kFragmentHash ||
        draw.graphics_pipeline_key_hash != kGraphicsKeyHash ||
        draw.vertex_permutation != 0 || !draw.vertex_index_unobserved ||
        draw.queue != 0 || draw.primitive_type != 4 || draw.index_type != 0 ||
        draw.instance_count != 1 || draw.first_instance != 0 ||
        draw.binding_count != 3 || draw.attribute_count != 3 ||
        !draw.fetch_inputs_only || draw.index_offset != 0) {
        return Reject::UnsupportedShaderOrLayout;
    }
    if (!draw.state.complete || !draw.state.immutable_through_record ||
        !draw.state.canonical.data || !draw.state.canonical.size) {
        return Reject::UnverifiedState;
    }
    if (!draw.index_cpu_owned || !draw.index_mapped || !draw.index_snapshot_stable) {
        return Reject::UnstableSource;
    }
    if (!draw.index_count || draw.index_count > kMaximumPackedIndexBytes / 2 ||
        !draw.index_bytes.data ||
        draw.index_bytes.size != std::size_t(draw.index_count) * 2) {
        return Reject::InvalidRange;
    }
    const std::size_t size = draw.streams[0].descriptor_size;
    if (!size || size > kMaximumSourceStreamBytes || size % 8) {
        return Reject::InvalidRange;
    }
    for (std::size_t i = 0; i < 3; ++i) {
        const Stream& s = draw.streams[i];
        if (s.stride != 8 || s.descriptor_control_word != 524290 ||
            s.format_word != kFormats[i] || s.sgpr_base != 2 ||
            s.dword_offset != kDwordOffsets[i] || s.instance_data != 0) {
            return Reject::UnsupportedShaderOrLayout;
        }
        if (!s.cpu_owned || !s.mapped || !s.snapshot_stable) {
            return Reject::UnstableSource;
        }
        if (!s.guest_base || !s.bytes.data || s.descriptor_size != size ||
            s.bytes.size != size || s.guest_base > UINT64_MAX - size) {
            return Reject::InvalidRange;
        }
    }
    const std::int64_t vertex_count = std::int64_t(size / 8);
    for (std::uint32_t i = 0; i < draw.index_count; ++i) {
        const std::int64_t vertex =
            std::int64_t(ReadIndex(draw.index_bytes.data + std::size_t(i) * 2)) +
            draw.vertex_offset;
        if (vertex < 0 || vertex >= vertex_count) {
            return Reject::OutOfBoundsIndex;
        }
    }
    return Reject::None;
}

} // namespace

BuildResult Build(std::span<const Draw> draws) {
    BuildResult result{};
    const auto reject = [&](Reject reason) {
        result.reject = reason;
        result.batch = {};
        return result;
    };
    if (draws.empty() || draws.size() > kMaximumDraws) {
        return reject(Reject::EmptyOrTooManyDraws);
    }
    // Validate the complete run and calculate exact capacities before copying any
    // bytes. Besides avoiding growth reallocations, this keeps rejection strictly
    // all-or-nothing even when the last draw exceeds an output limit.
    std::size_t total_stream_bytes = 0;
    std::size_t total_index_bytes = 0;
    for (const Draw& draw : draws) {
        if (const Reject reason = ValidateDraw(draw); reason != Reject::None) {
            return reject(reason);
        }
        if (!Equal(draw.state.canonical, draws.front().state.canonical)) {
            return reject(Reject::DifferentState);
        }
        const std::size_t segment_vertices = total_stream_bytes / 8;
        const std::int64_t rebased_offset =
            std::int64_t(draw.vertex_offset) + std::int64_t(segment_vertices);
        if (draw.streams[0].descriptor_size > kMaximumPackedStreamBytes - total_stream_bytes ||
            draw.index_bytes.size > kMaximumPackedIndexBytes - total_index_bytes ||
            total_index_bytes / 2 > UINT32_MAX ||
            rebased_offset < INT32_MIN || rebased_offset > INT32_MAX) {
            return reject(Reject::OutputOverflow);
        }
        total_stream_bytes += draw.streams[0].descriptor_size;
        total_index_bytes += draw.index_bytes.size;
    }
    for (auto& stream : result.batch.vertices) stream.reserve(total_stream_bytes);
    result.batch.indices.reserve(total_index_bytes);
    result.batch.commands.reserve(draws.size());
    for (const Draw& draw : draws) {
        const std::size_t segment_vertices = result.batch.vertices[0].size() / 8;
        result.batch.commands.push_back({draw.index_count, 1,
                                         std::uint32_t(result.batch.indices.size() / 2),
                                         std::int32_t(std::int64_t(draw.vertex_offset) +
                                                      std::int64_t(segment_vertices)), 0});
        for (std::size_t i = 0; i < 3; ++i) {
            const Bytes bytes = draw.streams[i].bytes;
            result.batch.vertices[i].insert(result.batch.vertices[i].end(),
                                            bytes.data, bytes.data + bytes.size);
        }
        result.batch.indices.insert(result.batch.indices.end(), draw.index_bytes.data,
                                    draw.index_bytes.data + draw.index_bytes.size);
    }
    result.batch.draw_count = std::uint32_t(draws.size());
    return result;
}

bool MayAttemptRecord(const BuildResult& result, const RecordProof& proof) {
    return result && result.batch.draw_count != 0 && proof.complete_state_equal_at_record &&
           proof.no_intervening_feedback_or_barrier;
}

} // namespace iss::native_geometry
