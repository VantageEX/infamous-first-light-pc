#include "infamous_renderer/secondary_batch_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

#include "common/perf_stats.h"
#include "core/memory.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"

namespace iss::native_geometry {
namespace {

struct Probe {
    std::mutex mutex;
    FILE* json{};
    FILE* bytes{};
    std::atomic<std::uint64_t> first_ns{};
    std::uint64_t sequence{};
    std::uint64_t bytes_written{};
    std::array<std::uint32_t, 2> count{};
    std::uint32_t max_per_family{128};
    std::uint64_t start_ns{45'000'000'000ull};
    std::atomic<bool> enabled{};

    Probe() {
        const char* flag = std::getenv("ISS_SECONDARY_BATCH_PROBE");
        const char* json_path = std::getenv("ISS_SECONDARY_BATCH_PROBE_LOG");
        const char* bytes_path = std::getenv("ISS_SECONDARY_BATCH_PROBE_BYTES");
        if (!flag || flag[0] != '1' || !json_path || !bytes_path) return;
        if (const char* value = std::getenv("ISS_SECONDARY_BATCH_PROBE_START_SECONDS")) {
            start_ns = std::uint64_t(std::strtoull(value, nullptr, 10)) * 1'000'000'000ull;
        }
        if (const char* value = std::getenv("ISS_SECONDARY_BATCH_PROBE_MAX_PER_FAMILY")) {
            max_per_family = std::uint32_t(
                std::min<unsigned long>(256, std::strtoul(value, nullptr, 10)));
        }
        json = std::fopen(json_path, "wb");
        bytes = std::fopen(bytes_path, "wb");
        enabled = json && bytes && max_per_family;
        if (!enabled) std::fprintf(stderr, "SECONDARY_BATCH_PROBE open failed\n");
    }

    ~Probe() {
        if (json) std::fclose(json);
        if (bytes) std::fclose(bytes);
    }
};

Probe& State() {
    static Probe state;
    return state;
}

std::uint8_t Ownership(VideoCore::BufferCache& cache, Core::MemoryManager& memory,
                       std::uint64_t address, std::uint64_t size) {
    if (!address || !size) return 0;
    const bool mapped = memory.ClampRangeSize(address, size) == size;
    return (cache.HasGpuData(address, size) ? 1 : 0) |
           (cache.IsRegionCpuModified(address, size) ? 2 : 0) | (mapped ? 4 : 0);
}

struct Snapshot {
    std::uint64_t address{};
    std::uint64_t size{};
    std::uint8_t ownership{};
    std::vector<std::uint8_t> data{};
    std::uint64_t blob_offset{};
    bool copied{};
};

void Prepare(Snapshot& out, VideoCore::BufferCache& cache, Core::MemoryManager& memory,
             std::uint64_t address, std::uint64_t size, std::uint64_t max_size) {
    out.address = address;
    out.size = size;
    out.ownership = Ownership(cache, memory, address, size);
    if ((out.ownership & 4) == 0 || (out.ownership & 1) != 0 ||
        !size || size > max_size) return;
    out.data.resize(size);
    memory.CopySparseMemory(address, out.data.data(), size);
    out.copied = true;
}

void WriteSnapshot(FILE* json, Probe& state, Snapshot& snapshot) {
    if (snapshot.copied) {
        snapshot.blob_offset = state.bytes_written;
        if (std::fwrite(snapshot.data.data(), 1, snapshot.data.size(), state.bytes) !=
            snapshot.data.size()) {
            snapshot.copied = false;
            state.enabled = false;
        } else {
            state.bytes_written += snapshot.data.size();
        }
    }
    std::fprintf(json,
                 "{\"address\":\"0x%llx\",\"size\":%llu,\"ownership\":%u,"
                 "\"bytes_copied\":%s,\"blob_offset\":%llu,\"blob_size\":%zu}",
                 static_cast<unsigned long long>(snapshot.address),
                 static_cast<unsigned long long>(snapshot.size), snapshot.ownership,
                 snapshot.copied ? "true" : "false",
                 static_cast<unsigned long long>(snapshot.blob_offset),
                 snapshot.copied ? snapshot.data.size() : std::size_t{0});
}

} // namespace

void ProbeSecondaryDraw(const Vulkan::GraphicsPipeline& pipeline,
                        const AmdGpu::Regs& regs, Core::MemoryManager& memory,
                        VideoCore::BufferCache& cache, int queue,
                        std::uint32_t index_offset, std::uint64_t tick) {
    auto& state = State();
    if (!state.enabled.load(std::memory_order_relaxed) || queue != 0) return;
    const std::uint64_t now = Common::Perf::NowNs();
    std::uint64_t first = state.first_ns.load(std::memory_order_acquire);
    if (!first) {
        std::uint64_t zero = 0;
        state.first_ns.compare_exchange_strong(zero, now, std::memory_order_acq_rel);
        first = state.first_ns.load(std::memory_order_acquire);
    }
    if (now - first < state.start_ns) return;
    const auto& vs = pipeline.GetStage(Shader::SwStage::Vertex);
    const auto& ps = pipeline.GetStage(Shader::SwStage::Fragment);
    if (ps.pgm_hash != 0x1318DDD4) return;
    const std::size_t family = vs.pgm_hash == 0xC01162F7 ? 0 :
                               vs.pgm_hash == 0x47B0694B ? 1 : 2;
    if (family == 2) return;
    const auto& fetch = pipeline.GetFetchShader();
    if (fetch.attributes.size() != 1 || vs.buffers.empty() ||
        vs.buffers[0].IsSpecial()) return;

    std::scoped_lock lock{state.mutex};
    if (state.count[family] >= state.max_per_family) return;
    const auto mapping_before = memory.MappingEpoch();
    const auto buffer_before = cache.MappingGeneration();
    const auto host_before = VideoCore::CurrentHostWriteGen();
    const auto upload_before = VideoCore::CurrentUploadEpoch();
    const auto& attribute = fetch.attributes[0];
    if (attribute.sgpr_base + 1 >= vs.UserData().size()) return;
    if ((fetch.vertex_offset_sgpr >= 0 &&
         std::size_t(fetch.vertex_offset_sgpr) >= vs.UserData().size()) ||
        (fetch.instance_offset_sgpr >= 0 &&
         std::size_t(fetch.instance_offset_sgpr) >= vs.UserData().size())) return;
    const std::uint64_t descriptor_address =
        std::uint64_t(vs.UserData()[attribute.sgpr_base]) |
        (std::uint64_t(vs.UserData()[attribute.sgpr_base + 1]) << 32);
    std::array<std::uint32_t, 4> descriptor_raw{};
    Snapshot fetch_table;
    Prepare(fetch_table, cache, memory, descriptor_address + 4 * attribute.dword_offset,
            sizeof(descriptor_raw), sizeof(descriptor_raw));
    AmdGpu::Buffer sharp{};
    if (fetch_table.copied) {
        std::memcpy(&sharp, fetch_table.data.data(), sizeof(sharp));
        sharp.base_address += attribute.inst_offset;
        if (attribute.data_format) {
            sharp.data_format = attribute.data_format;
            sharp.num_format = attribute.num_format;
        }
        std::memcpy(descriptor_raw.data(), &sharp, sizeof(sharp));
    }
    Snapshot vertex, ssbo, indices;
    if (fetch_table.copied) {
        Prepare(vertex, cache, memory, sharp.base_address, sharp.GetSize(), 64 * 1024);
    }
    const auto ssbo_sharp = vs.buffers[0].GetSharp(vs);
    const auto ssbo_raw = std::bit_cast<std::array<std::uint32_t, 4>>(ssbo_sharp);
    Prepare(ssbo, cache, memory, ssbo_sharp.base_address, ssbo_sharp.GetSize(), 64 * 1024);
    const bool index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const std::uint32_t index_size = index16 ? 2 : 4;
    const std::uint64_t index_address = regs.index_base_address.Address<std::uint64_t>() +
                                        std::uint64_t(index_offset) * index_size;
    std::uint32_t vertex_offset = regs.index_offset;
    if (!vertex_offset && fetch.vertex_offset_sgpr != -1) {
        vertex_offset = vs.UserData()[fetch.vertex_offset_sgpr];
    }
    Prepare(indices, cache, memory, index_address,
            std::uint64_t(regs.num_indices) * index_size, 64 * 1024);
    const bool stable = mapping_before == memory.MappingEpoch() &&
        buffer_before == cache.MappingGeneration() &&
        host_before == VideoCore::CurrentHostWriteGen() &&
        upload_before == VideoCore::CurrentUploadEpoch();
    if (!stable) {
        fetch_table.copied = vertex.copied = ssbo.copied = indices.copied = false;
    }

    FILE* const json = state.json;
    std::fprintf(json,
                 "{\"event\":\"secondary_draw\",\"seq\":%llu,\"tick\":%llu,"
                 "\"vs\":\"0x%08llx\",\"ps\":\"0x%08llx\","
                 "\"key\":\"0x%016llx\",\"vs_permhash\":\"0x%016llx\","
                 "\"primitive\":%u,\"instances\":%u,\"index_count\":%u,"
                 "\"index_type\":%u,\"index_offset\":%u,\"vertex_offset\":%d,"
                 "\"first_instance\":%u,"
                 "\"mapping_epoch\":%llu,\"buffer_generation\":%llu,"
                 "\"host_generation\":%llu,\"upload_epoch\":%llu,\"stable\":%s,"
                 "\"fetch_code_size\":%u,\"attribute\":[%u,%u,%u,%u,%u,%u,%u,%u],"
                 "\"vsharp_raw\":[%u,%u,%u,%u],\"ssbo_raw\":[%u,%u,%u,%u],"
                 "\"fetch_table\":",
                 static_cast<unsigned long long>(state.sequence++),
                 static_cast<unsigned long long>(tick),
                 static_cast<unsigned long long>(vs.pgm_hash),
                 static_cast<unsigned long long>(ps.pgm_hash),
                 static_cast<unsigned long long>(std::hash<Vulkan::GraphicsPipelineKey>{}(
                     pipeline.GetGraphicsKey())),
                 static_cast<unsigned long long>(pipeline.GetGraphicsKey().stage_hashes[
                     std::uint32_t(Shader::SwStage::Vertex)]),
                 std::uint32_t(regs.primitive_type), regs.num_instances.NumInstances(),
                 regs.num_indices, regs.index_buffer_type.raw, index_offset,
                 static_cast<std::int32_t>(vertex_offset),
                 fetch.instance_offset_sgpr == -1 ? 0u :
                     vs.UserData()[fetch.instance_offset_sgpr],
                 static_cast<unsigned long long>(mapping_before),
                 static_cast<unsigned long long>(buffer_before),
                 static_cast<unsigned long long>(host_before),
                 static_cast<unsigned long long>(upload_before),
                 stable ? "true" : "false", fetch.size,
                 attribute.dest_vgpr, attribute.num_elements, attribute.sgpr_base,
                 attribute.dword_offset, attribute.instance_data, attribute.inst_offset,
                 attribute.data_format, attribute.num_format,
                 descriptor_raw[0], descriptor_raw[1], descriptor_raw[2], descriptor_raw[3],
                 ssbo_raw[0], ssbo_raw[1], ssbo_raw[2], ssbo_raw[3]);
    WriteSnapshot(json, state, fetch_table);
    std::fputs(",\"vertex\":", json); WriteSnapshot(json, state, vertex);
    std::fputs(",\"ssbo\":", json); WriteSnapshot(json, state, ssbo);
    std::fputs(",\"indices\":", json); WriteSnapshot(json, state, indices);
    std::fputs("}\n", json);
    std::fflush(json);
    std::fflush(state.bytes);
    ++state.count[family];
}

} // namespace iss::native_geometry
