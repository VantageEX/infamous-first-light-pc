// SPDX-License-Identifier: GPL-2.0-or-later
// Optional, bounded diagnostic trace. All entry points run on the command processor.
// It records translation order and resource declarations, NOT completed GPU accesses.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/perf_stats.h"
#include "core/debug_state.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs_shader.h"

namespace iss::command_trace {
extern const bool enabled = [] { const char* p = std::getenv("ISS_COMMAND_TRACE"); return p && p[0] == '1'; }();
namespace {
struct Trace {
    FILE* file{};
    u64 first_ns{}, sequence{}, submit{}, start_submit{}, bytes{};
    u32 frames{2};
    double delay_seconds{30};
    bool active{}, finished{};
    std::array<char, 1024 * 1024> buffer{};
    Trace() {
        const char* path = std::getenv("ISS_COMMAND_TRACE_LOG");
        const char* enabled = std::getenv("ISS_COMMAND_TRACE");
        if (!path || !enabled || enabled[0] != '1') { finished = true; return; }
        if (const char* p = std::getenv("ISS_COMMAND_TRACE_START_SECONDS"))
            delay_seconds = std::clamp(std::strtod(p, nullptr), 0.0, 300.0);
        if (const char* p = std::getenv("ISS_COMMAND_TRACE_FRAMES"))
            frames = std::clamp<u32>(std::strtoul(p, nullptr, 10), 1, 10);
        file = std::fopen(path, "wb");
        if (!file) { finished = true; std::fprintf(stderr, "COMMAND_TRACE open failed\n"); return; }
        std::setvbuf(file, buffer.data(), _IOFBF, buffer.size());
        std::fprintf(file, "{\"event\":\"header\",\"version\":1,\"requested_submits\":%u,\"delay_seconds\":%.3f,\"meaning\":\"CPU translation order; declared ranges, not actual shader accesses\"}\n", frames, delay_seconds);
        first_ns = Common::Perf::NowNs();
    }
    ~Trace() { if (file) std::fclose(file); }
    void End(const char* why) {
        if (!file || finished) return;
        std::fprintf(file, "{\"event\":\"end\",\"reason\":\"%s\",\"seq\":%llu,\"submit\":%llu}\n", why, sequence, submit);
        std::fclose(file); file = nullptr; active = false; finished = true;
        std::fprintf(stderr, "COMMAND_TRACE ended: %s, events=%llu\n", why, sequence);
    }
    void Prefix(const char* event, int queue) {
        std::fprintf(file, "{\"event\":\"%s\",\"seq\":%llu,\"submit\":%llu,\"flip\":%llu,\"ns\":%llu,\"queue\":%d", event, ++sequence, submit,
                     static_cast<u64>(DebugState.flip_count.load()), Common::Perf::NowNs(), queue);
    }
    void CheckLimit() {
        const auto pos = _ftelli64(file);
        if (pos >= 128 * 1024 * 1024 || sequence >= 500000) End("limit_truncated");
    }
};
Trace& State() { static Trace state; return state; }
}

bool Active() { return State().active; }

void Submit() {
    auto& s = State();
    if (s.finished) return;
    ++s.submit;
    if (s.active) {
        s.Prefix("submit_end", -1); std::fputs("}\n", s.file);
        std::fflush(s.file);
        if (s.submit - s.start_submit >= s.frames) { s.End("complete"); return; }
    } else if (double(Common::Perf::NowNs() - s.first_ns) / 1e9 >= s.delay_seconds) {
        s.start_submit = s.submit; s.active = true;
        s.Prefix("begin", -1); std::fputs("}\n", s.file);
        std::fprintf(stderr, "COMMAND_TRACE started, submit=%llu\n", s.submit);
    }
}

void Packet(int queue, u32 opcode, const u32* words, size_t count) {
    auto& s = State(); if (!s.active) return;
    static const bool pm4_map = [] {
        const char* value = std::getenv("ISS_PM4_MAP");
        return value && value[0] == '1';
    }();
    // Register writes and inline NOP payloads are represented by the actual shader
    // descriptors recorded at dispatch/draw. Retain every other packet as a boundary.
    if (!pm4_map && (opcode == 0x10 || opcode == 0x76 || opcode == 0x69 || opcode == 0x68)) return;
    const size_t captured = std::min<size_t>(count, pm4_map ? 512 : 9);
    s.Prefix("packet", queue);
    std::fprintf(s.file, ",\"opcode\":%u,\"address\":\"0x%llx\",\"word_count\":%zu,\"captured_words\":%zu,\"words\":[", opcode, reinterpret_cast<u64>(words), count, captured);
    for (size_t i = 0; i < captured; ++i)
        std::fprintf(s.file, "%s%u", i ? "," : "", words[i]);
    std::fputs("]}\n", s.file); s.CheckLimit();
}

void Stage(const char* kind, int queue, const Shader::Info& info, u64 code,
           u32 x, u32 y, u32 z, u64 tick, bool barrier, u64 args, u64 count_address) {
    auto& s = State(); if (!s.active) return;
    s.Prefix(kind, queue);
    std::fprintf(s.file, ",\"stage\":%u,\"hash\":\"0x%llx\",\"code\":\"0x%llx\",\"cached_code\":\"0x%llx\",\"groups\":[%u,%u,%u],\"tick\":%llu,\"needs_barrier\":%s,\"args\":\"0x%llx\",\"count_address\":\"0x%llx\",\"uses_dma\":%s,\"user_data\":[",
                 u32(info.hw_stage), info.pgm_hash, code, info.pgm_base, x, y, z, tick,
                 barrier ? "true" : "false", args, count_address, info.uses_dma ? "true" : "false");
    for (size_t i = 0; i < info.user_data.size(); ++i)
        std::fprintf(s.file, "%s%u", i ? "," : "", info.user_data[i]);
    std::fputs("],\"buffers\":[", s.file);
    for (size_t i = 0; i < info.buffers.size(); ++i) {
        const auto& desc = info.buffers[i];
        std::fprintf(s.file, "%s{\"index\":%zu,\"type\":%u,\"written\":%s", i ? "," : "", i, u32(desc.buffer_type), desc.is_written ? "true" : "false");
        if (!desc.IsSpecial()) {
            const auto sharp = desc.GetSharp(info);
            std::array<u32, 4> raw{}; std::memcpy(raw.data(), &sharp, sizeof(sharp));
            std::fprintf(s.file, ",\"base\":\"0x%llx\",\"size\":%llu,\"raw\":[%u,%u,%u,%u]", u64(sharp.base_address), u64(sharp.GetSize()), raw[0], raw[1], raw[2], raw[3]);
        }
        std::fputs("}", s.file);
    }
    std::fputs("],\"images\":[", s.file);
    for (size_t i = 0; i < info.images.size(); ++i) {
        const auto& desc = info.images[i]; const auto sharp = desc.GetSharp(info);
        std::fprintf(s.file, "%s{\"index\":%zu,\"base\":\"0x%llx\",\"written\":%s,\"atomic\":%s}", i ? "," : "", i, u64(sharp.Address()), desc.is_written ? "true" : "false", desc.is_atomic ? "true" : "false");
    }
    std::fputs("]}\n", s.file); s.CheckLimit();
}

// Narrow observation for the first native draw-batch candidate. The three fetch descriptors
// are read exactly as GetVertexInputs reads them later; this changes no binding or guest byte.
void VertexFetch(int queue, const Shader::Info& info,
                 const Shader::Gcn::FetchShaderData& fetch, u64 pipeline_identity,
                 u64 context_changes, u32 primitive_type, u32 instance_count,
                 u32 index_count, u64 index_address, u32 index_type, u32 index_offset,
                 const std::array<u8, 3>& attribute_ownership, u8 index_ownership,
                 u64 mapping_epoch, u64 buffer_mapping_generation) {
    auto& s = State();
    if (!s.active || info.pgm_hash != 0x0BEB7156 || fetch.Empty()) return;
    s.Prefix("vertex_fetch", queue);
    std::fprintf(s.file,
                 ",\"hash\":\"0x%llx\",\"pipeline_identity\":\"0x%llx\","
                 "\"context_changes\":%llu,\"primitive_type\":%u,"
                 "\"instance_count\":%u,\"index_count\":%u,"
                 "\"index_address\":\"0x%llx\",\"index_type\":%u,"
                 "\"index_offset\":%u,\"index_ownership\":%u,"
                 "\"mapping_epoch\":%llu,\"buffer_mapping_generation\":%llu,"
                 "\"fetch_code_size\":%u,\"attributes\":[",
                 static_cast<unsigned long long>(info.pgm_hash),
                 static_cast<unsigned long long>(pipeline_identity),
                 static_cast<unsigned long long>(context_changes), primitive_type,
                 instance_count, index_count,
                 static_cast<unsigned long long>(index_address), index_type, index_offset,
                 index_ownership, static_cast<unsigned long long>(mapping_epoch),
                 static_cast<unsigned long long>(buffer_mapping_generation), fetch.size);
    for (size_t i = 0; i < fetch.attributes.size(); ++i) {
        const auto& attr = fetch.attributes[i];
        const auto sharp = attr.GetSharp(info);
        std::array<u32, 4> raw{};
        std::memcpy(raw.data(), &sharp, sizeof(sharp));
        std::fprintf(s.file,
                     "%s{\"index\":%zu,\"sgpr_base\":%u,\"dword_offset\":%u,"
                     "\"instance_data\":%u,\"base\":\"0x%llx\",\"size\":%u,"
                     "\"stride\":%u,\"ownership\":%u,\"raw\":[%u,%u,%u,%u]}",
                     i ? "," : "", i, attr.sgpr_base, attr.dword_offset,
                     attr.instance_data, static_cast<unsigned long long>(sharp.base_address),
                     sharp.GetSize(), sharp.GetStride(),
                     i < attribute_ownership.size() ? attribute_ownership[i] : 0,
                     raw[0], raw[1], raw[2], raw[3]);
    }
    std::fputs("]}\n", s.file);
    s.CheckLimit();
}
} // namespace iss::command_trace
