// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_GBUFFER: the engine side (native_gbuffer.h). Runs on the engine's threads.
#include <atomic>
#include "native_gbuffer.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef ISS_SYSV
#define ISS_SYSV __attribute__((sysv_abi))
#endif

namespace iss::native_gbuffer {
namespace {
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u64 RoutineOffset = 0x1A2030; // generic material range (slot 6)
constexpr u64 GbufferVtable = 0xD0FAC8;
constexpr u32 SlotCount = 8192;
constexpr u32 ContextBase = 0xA000, ShBase = 0x2C00, UconfigBase = 0xC000, ConfigBase = 0x2000;
constexpr unsigned char Prologue[13] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                        0x56, 0x41, 0x55, 0x41, 0x54, 0x53};

struct CommandBuffer {
    u32* begin; // the current chunk (sce::Gnm::CommandBuffer m_beginptr)
    u32* end;
    u32* cursor;
    bool(ISS_SYSV* reserve)(CommandBuffer*, u32, void*);
    void* user;
};

using Render = u64(ISS_SYSV*)(u64 technique, u64* args);
Render original{};
u64 image_base{};
std::unique_ptr<Batch[]> slots;
std::atomic<u32> next_generation{1};

enum Reject : u32 { RejectChunk, RejectOpcode, RejectTarget, RejectCompute, RejectRegs,
                    RejectBounds, RejectEmpty, RejectCount };
constexpr const char* reject_names[RejectCount] = {"chunk", "opcode", "target", "compute",
                                                   "regs", "bounds", "empty"};
struct EngineStats {
    std::atomic<u64> calls{}, other{}, item_path{}, double_draw{}, batches{}, draws{};
    std::array<std::atomic<u64>, RejectCount> reject{};
    std::array<std::atomic<u64>, 256> opcode{};
};
EngineStats engine_stats;
std::atomic<u64> track_resets{}, track_unknown{}, links{}, span_hits{}, span_misses{}; std::atomic<u64> technique_hits{}, technique_misses{}; // ISS_NATIVE_RECORDER
// What ended a chain: 0..255 the first other opcode between the ranges, 256 a target register,
// 257 compute registers, 258 another packet type, 259 no recorded previous range, 260 another
// chunk, 261 not linked by the recorder (formats).
std::array<std::atomic<u64>, 262> chain_breaks{};
std::array<std::atomic<u64>, 0x21> target_breaks{}; // by DB register (0x20: a CB register)
// NOPs between chained ranges: 0 empty, 1 engine draw marker 'ISS' (diagnostics and the native
// item path only), 2 'ISG', 3 'ISH', 4 flip or debug payloads, 5 other.
std::array<std::atomic<u64>, 6> nop_kinds{};
bool allow_engine_markers{};

// Registers of the target range a chain may cross: they do not change which images are
// attached (render control and override, occlusion count control, depth bounds, clear values,
// screen scissor); the draws' own state follows them (templates, dynamic state).
bool ChainSafeTargetReg(u32 reg) {
    switch (reg - ContextBase) {
    case 0x00: case 0x01: case 0x03: case 0x04: case 0x08: case 0x09: case 0x0A: case 0x0B:
    case 0x0C: case 0x0D:
        return true;
    default:
        return false;
    }
}
std::array<std::atomic<u64>, 257> unknown_opcodes{}; // 256: a packet type other than 2 or 3

struct CpStats {
    std::array<u64, ReasonCount> ranges{}, draws{};
    u64 total{};
};
CpStats cp_stats; // command processor thread only

void ReportEngine() {
    auto& s = engine_stats;
    std::string text;
    for (u32 i = 0; i < RejectCount; ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s=%llu", reject_names[i],
                      (unsigned long long)s.reject[i].load(std::memory_order_relaxed));
        text += item;
    }
    std::string unknown;
    for (u32 i = 0; i < 257; ++i)
        if (const u64 n = unknown_opcodes[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", i, (unsigned long long)n);
            unknown += item;
        }
    text += " ] track_unknown_opcodes[" + unknown;
    std::string breaks;
    for (u32 i = 0; i < 262; ++i)
        if (const u64 n = chain_breaks[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", i, (unsigned long long)n);
            breaks += item;
        }
    text += " ] chain_breaks[" + breaks;
    std::string targets;
    for (u32 i = 0; i < 0x21; ++i)
        if (const u64 n = target_breaks[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %x=%llu", i, (unsigned long long)n);
            targets += item;
        }
    text += " ] target_breaks[" + targets;
    char kinds[160];
    std::snprintf(kinds, sizeof(kinds), " ] nop_kinds[ empty=%llu iss=%llu isg=%llu ish=%llu flip_debug=%llu other=%llu",
                  (unsigned long long)nop_kinds[0].load(), (unsigned long long)nop_kinds[1].load(),
                  (unsigned long long)nop_kinds[2].load(), (unsigned long long)nop_kinds[3].load(),
                  (unsigned long long)nop_kinds[4].load(), (unsigned long long)nop_kinds[5].load());
    text += kinds;
    std::string opcodes;
    for (u32 i = 0; i < 256; ++i)
        if (const u64 n = s.opcode[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %02x=%llu", i, (unsigned long long)n);
            opcodes += item;
        }
    std::fprintf(stderr,
                 "NATIVE_GBUFFER_ENGINE calls=%llu other_technique=%llu item_path=%llu "
                 "double_draw=%llu batches=%llu draws=%llu track_resets=%llu track_unknown=%llu links=%llu "
                 "span_starts=%llu/%llu technique_starts=%llu/%llu "
                 "reject[%s ] reject_opcodes[%s ]\n",
                 (unsigned long long)s.calls.load(), (unsigned long long)s.other.load(),
                 (unsigned long long)s.item_path.load(), (unsigned long long)s.double_draw.load(),
                 (unsigned long long)s.batches.load(), (unsigned long long)s.draws.load(),
                 (unsigned long long)track_resets.load(), (unsigned long long)track_unknown.load(),
                 (unsigned long long)links.load(), (unsigned long long)span_hits.load(),
                 (unsigned long long)span_misses.load(), (unsigned long long)technique_hits.load(),
                 (unsigned long long)technique_misses.load(),
                 text.c_str(), opcodes.c_str());
}

bool IsTargetReg(u32 reg) {
    return (reg >= ContextBase && reg < ContextBase + 0x20) ||
           (reg >= ContextBase + 0x318 && reg < ContextBase + 0x3A0);
}
bool RejectTargetWrite(u32 reg, u32 value) {
    // DB_RENDER_OVERRIDE without copy/force modes and the depth bounds are plain state.
    if (reg == ContextBase + 3) return (value & 0x78000000u) != 0;
    if (reg == ContextBase + 8 || reg == ContextBase + 9) return false;
    return IsTargetReg(reg);
}

// Reads back the commands the routine wrote, [first, end). The first pass validates every
// packet and collects the state registers; the second records the draws and state versions.
bool Scan(Batch& b, const u32* first, const u32* end) {
    b.state_regs.clear();
    b.state_values.clear();
    b.states.clear();
    b.draws.clear();
    b.final_regs.clear();
    const auto reject = [](Reject r) {
        engine_stats.reject[r].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    const auto base_of = [](u32 opcode) {
        return opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
             : opcode == 0x76 ? ShBase : UconfigBase;
    };
    u32 draws = 0;
    for (const u32* p = first; p < end;) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        if (type != 3) return reject(RejectOpcode);
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        const u32* next = p + 1 + count;
        if (next > end) return reject(RejectBounds);
        switch (opcode) {
        case 0x10: case 0x2A: case 0x2F:
            break;
        case 0x27:
            if (count != 5) return reject(RejectOpcode);
            ++draws;
            break;
        case 0x50: // DMA_DATA to 0x3022C: an explicit no-op for the command processor
            if (count < 6 || p[4] != 0x3022C) {
                engine_stats.opcode[opcode].fetch_add(1, std::memory_order_relaxed);
                return reject(RejectOpcode);
            }
            break;
        case 0x68: case 0x69: case 0x76: case 0x79: {
            if (count < 2) return reject(RejectOpcode);
            if (opcode == 0x76 && p[1] >= 0x200) return reject(RejectCompute);
            const u32 base = base_of(opcode);
            for (u32 i = 0; i < count - 1; ++i) {
                const u32 reg = base + p[1] + i;
                if (opcode == 0x69 && RejectTargetWrite(reg, p[2 + i])) return reject(RejectTarget);
                if (IsVsUser(reg) || IsPsUser(reg) ||
                    std::find(b.state_regs.begin(), b.state_regs.end(), reg) != b.state_regs.end())
                    continue;
                if (b.state_regs.size() == MaxStateRegs) return reject(RejectRegs);
                b.state_regs.push_back(reg);
            }
            break;
        }
        default:
            engine_stats.opcode[opcode].fetch_add(1, std::memory_order_relaxed);
            return reject(RejectOpcode);
        }
        p = next;
    }
    if (draws == 0) return reject(RejectEmpty);
    std::sort(b.state_regs.begin(), b.state_regs.end());
    const u32 nregs = u32(b.state_regs.size());

    std::array<u32, MaxStateRegs> values{};
    u64 inherit = nregs == 64 ? ~0ull : (1ull << nregs) - 1;
    bool dirty = true;
    Draw current{};
    current.vs_inherit = current.ps_inherit = (1u << UserWords) - 1;
    current.instances_inherit = current.index_type_inherit = true;
    b.draws.reserve(draws);
    for (const u32* p = first; p < end;) {
        const u32 header = *p;
        if ((header >> 30) == 2) { ++p; continue; }
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        switch (opcode) {
        case 0x2A:
            current.index_type = p[1];
            current.index_type_inherit = false;
            break;
        case 0x2F:
            current.instances = p[1];
            current.instances_inherit = false;
            break;
        case 0x27: {
            if (dirty) {
                b.states.push_back({inherit, u32(b.state_values.size())});
                b.state_values.insert(b.state_values.end(), values.begin(), values.begin() + nregs);
                dirty = false;
            }
            Draw d = current;
            d.packet = p;
            d.max_index = p[1];
            d.index_lo = p[2];
            d.index_hi = p[3];
            d.index_count = p[4];
            d.initiator = p[5];
            d.state = u32(b.states.size() - 1);
            b.draws.push_back(d);
            break;
        }
        case 0x68: case 0x69: case 0x76: case 0x79: {
            const u32 base = base_of(opcode);
            for (u32 i = 0; i < count - 1; ++i) {
                const u32 reg = base + p[1] + i, value = p[2 + i];
                if (IsVsUser(reg)) {
                    current.vs_user[reg - VsUser0] = value;
                    current.vs_inherit &= ~(1u << (reg - VsUser0));
                } else if (IsPsUser(reg)) {
                    current.ps_user[reg - PsUser0] = value;
                    current.ps_inherit &= ~(1u << (reg - PsUser0));
                } else {
                    const u32 k = u32(std::lower_bound(b.state_regs.begin(), b.state_regs.end(), reg) -
                                      b.state_regs.begin());
                    // Repeated setters do not create a new state; inherited zero stays distinct
                    // from an explicit write of zero.
                    dirty |= ((inherit >> k) & 1) != 0 || values[k] != value;
                    values[k] = value;
                    inherit &= ~(1ull << k);
                }
                auto it = std::find_if(b.final_regs.begin(), b.final_regs.end(),
                                       [reg](const auto& r) { return r.first == reg; });
                if (it == b.final_regs.end()) b.final_regs.emplace_back(reg, value);
                else it->second = value;
            }
            break;
        }
        default:
            break;
        }
        p += 1 + count;
    }
    b.has_instances = !current.instances_inherit;
    b.final_instances = current.instances;
    b.has_index_type = !current.index_type_inherit;
    b.final_index_type = current.index_type;
    return true;
}


// ISS_NATIVE_RECORDER: what each command buffer of the engine wrote since its last baseline
// (native_gbuffer.h, RangeStart). A command buffer is written by one thread at a time; the
// table of them is shared because a job may continue one on another thread.
RecordFn recorder{};
LinkFn linker{};
bool all_techniques{};

struct Baseline {
    std::atomic<u64> sequence{}; // odd while written
    std::array<u32, TrackedWords> words{};
    u32 index_type{}, instances{};
};
Baseline baseline;

using Reserve = bool(ISS_SYSV*)(CommandBuffer*, u32, void*);
struct Track {
    RangeStart state;
    u32* begin{};
    const u32* scanned{};
    Reserve original{};
    u64 baseline_sequence{}; ///< of the baseline loaded last
    // The last recorded range of this command buffer, for chains.
    const Batch* last{};
    u32 last_generation{};
    const u32* last_end{};
    u32* last_begin{};
};

// Whether [p, end) holds only register writes the command processor applies plainly (no
// render target register, whose writes it reads size hints from).
bool OnlyPlainRegisterWrites(const u32* p, const u32* end) {
    while (p < end) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        const auto stop = [](u32 why) {
            chain_breaks[why].fetch_add(1, std::memory_order_relaxed);
            return false;
        };
        if (type != 3) return stop(258);
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        if (p + 1 + count > end) return stop(258);
        switch (opcode) {
        case 0x68: case 0x79: case 0x2A: case 0x2F: // registers, index type, instance count
            break;
        case 0x10: {
            const u32 word = count > 0 ? p[1] : 0;
            const u32 kind = count == 1 && word == 0 ? 0
                           : (word & 0xFFFFFF00u) == 0x49535300u ? 1
                           : (word & 0xFFFFFF00u) == 0x49534700u ? 2
                           : (word & 0xFFFFFF00u) == 0x49534800u ? 3
                           : (word & 0xFFFF0000u) == 0x68750000u ? 4 : 5;
            nop_kinds[kind].fetch_add(1, std::memory_order_relaxed);
            // SDK hints and debug markers (0x68750001..0x6875000E; the command processor
            // reads debug markers only with guest markers on, which keeps native ranges off);
            // never a flip marker.
            if (kind == 0 || (kind == 1 && allow_engine_markers && (word & 255) != 1) ||
                (kind == 4 && word <= 0x6875000Eu))
                break;
            return stop(opcode);
        }
        case 0x76: // the compute registers go to the queue's compute state instead
            if (p[1] >= 0x200) return stop(257);
            break;
        case 0x69:
            for (u32 i = 0; i + 1 < count; ++i) {
                const u32 reg = ContextBase + p[1] + i;
                if (IsTargetReg(reg) && !ChainSafeTargetReg(reg)) {
                    target_breaks[std::min<u32>(reg - ContextBase, 0x20)].fetch_add(
                        1, std::memory_order_relaxed);
                    return stop(256);
                }
            }
            break;
        default:
            return stop(opcode);
        }
        p += 1 + count;
    }
    return true;
}
std::mutex tracks_mutex;
std::unordered_map<CommandBuffer*, std::unique_ptr<Track>> tracks;
// The engine's chunk callback (one function for all its command buffers): the wrapper always
// calls it, also for a command buffer it does not follow (a copy of a followed one).
std::atomic<Reserve> engine_reserve{};

int TrackedIndex(u32 reg) {
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        if (reg >= TrackedFirst[r] && reg < TrackedFirst[r] + TrackedCount[r])
            return int(base + reg - TrackedFirst[r]);
        base += TrackedCount[r];
    }
    return -1;
}

// Registers at command buffer starts (PublishSpanStart), by address.
std::mutex span_mutex;
std::unordered_map<const u32*, std::unique_ptr<RangeStart>> span_starts;

std::unordered_map<u64, std::unique_ptr<RangeStart>> technique_starts;

bool LoadTechniqueStart(RangeStart& state, u64 technique) {
    std::scoped_lock lock{span_mutex};
    const auto it = technique_starts.find(technique);
    if (it == technique_starts.end()) return false;
    state = *it->second;
    return true;
}

bool LoadSpanStart(RangeStart& state, const u32* span) {
    std::scoped_lock lock{span_mutex};
    const auto it = span_starts.find(span);
    if (it == span_starts.end()) return false;
    state = *it->second;
    return true;
}

void LoadBaseline(RangeStart& state) {
    for (;;) {
        const u64 before = baseline.sequence.load(std::memory_order_acquire);
        if (before & 1) continue;
        state.words = baseline.words;
        state.index_type = baseline.index_type;
        state.instances = baseline.instances;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (baseline.sequence.load(std::memory_order_relaxed) == before) break;
    }
    state.known = baseline.sequence.load(std::memory_order_relaxed) != 0;
}

// Applies the register writes of [p, end) (whole packets the engine wrote).
void Apply(RangeStart& state, const u32* p, const u32* end) {
    const auto base_of = [](u32 opcode) {
        return opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
             : opcode == 0x76 ? ShBase : UconfigBase;
    };
    while (p < end) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        if (type != 3) {
            state.known = false;
            unknown_opcodes[256].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        if (p + 1 + count > end) { state.known = false; return; }
        switch (opcode) {
        case 0x68: case 0x69: case 0x76: case 0x79: {
            const u32 base = base_of(opcode);
            for (u32 i = 0; i + 1 < count; ++i) {
                const int index = TrackedIndex(base + p[1] + i);
                if (index >= 0) state.words[index] = p[2 + i];
            }
            break;
        }
        case 0x2A: state.index_type = p[1]; break;
        case 0x2F: state.instances = p[1]; break;
        // State the tracking does not follow: cleared, loaded from memory or indexed.
        case 0x12: case 0x28: case 0x5F: case 0x60: case 0x61: case 0x62: case 0x63:
        case 0x64: case 0x65: case 0x66: case 0x6A: case 0x9B: case 0x80:
            state.known = false;
            unknown_opcodes[opcode].fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
        }
        p += 1 + count;
    }
}

Track* FindTrack(CommandBuffer* dcb) {
    std::scoped_lock lock{tracks_mutex};
    const auto it = tracks.find(dcb);
    return it != tracks.end() ? it->second.get() : nullptr;
}

// ISS_NATIVE_GBUFFER_SEGMENTS=1 (with the recorder's chunk wrapper): a range that continues
// into the next chunk of its command buffer is split there, one marked range per chunk (the
// long item by item ranges otherwise cross chunks and are all left to interpretation).
bool SegmentsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_GBUFFER_SEGMENTS");
        return value && value[0] == '1';
    }();
    return enabled;
}
struct OpenRange {
    CommandBuffer* dcb{};
    u32* marker{};
    u32 generation{};
    RangeStart start;
    Track* track{};
    u64 technique{};
};
thread_local OpenRange* open_range = nullptr;
std::atomic<u64> segments{};
u32* OpenMarker(CommandBuffer* dcb, u32& generation);
void FinishRange(u32* marker, u32 generation, const u32* cursor, u32* chunk_begin,
                 Track* track, const RangeStart& start, u64 technique);

// The engine's chunk callback: the end of the chunk is applied before it is left.
bool ISS_SYSV ReserveWrapper(CommandBuffer* dcb, u32 dwords, void* user) {
    Track* t = FindTrack(dcb);
    if (!t) return engine_reserve.load(std::memory_order_acquire)(dcb, dwords, user);
    if (t->begin == dcb->begin && t->scanned >= dcb->begin && t->scanned <= dcb->cursor)
        Apply(t->state, t->scanned, dcb->cursor);
    else
        t->state.known = false;
    OpenRange* open = open_range;
    if (open && (open->dcb != dcb || open->track != t)) open = nullptr;
    u32* const old_begin = dcb->begin;
    const u32* const old_cursor = dcb->cursor;
    // Room for the next segment's marker too.
    const bool result = t->original(dcb, dwords + (open ? MarkerDwords : 0), user);
    if (open && result && dcb->begin != old_begin && open->marker >= old_begin &&
        open->marker < old_cursor) {
        // The segment ends where the old chunk was left; the next starts this chunk, from the
        // registers the old one ended with.
        FinishRange(open->marker, open->generation, old_cursor, old_begin, t, open->start,
                    open->technique);
        open->start = t->state;
        open->marker = OpenMarker(dcb, open->generation);
        if ((segments.fetch_add(1, std::memory_order_relaxed) & 0x3FFF) == 0)
            std::fprintf(stderr, "NATIVE_GBUFFER_SEGMENTS splits=%llu\n",
                         (unsigned long long)segments.load(std::memory_order_relaxed));
    }
    t->begin = dcb->begin;
    t->scanned = dcb->cursor;
    return result;
}

// The track of the command buffer, brought up to its cursor.
Track* Follow(CommandBuffer* dcb, u64 technique) {
    if (!(dcb->begin <= dcb->cursor && dcb->cursor <= dcb->end)) return nullptr;
    Track* t = FindTrack(dcb);
    if (!t) {
        std::scoped_lock lock{tracks_mutex};
        t = (tracks[dcb] = std::make_unique<Track>()).get();
    }
    // ISS_NATIVE_RECORDER_WRAP=1: the engine's chunk callback is wrapped so that a chunk's end
    // is applied before it is left; otherwise a new chunk starts again from the baseline.
    static const bool wrap = [] {
        const char* value = std::getenv("ISS_NATIVE_RECORDER_WRAP");
        return value && value[0] == '1';
    }();
    const bool wrapped = dcb->reserve == &ReserveWrapper;
    const bool continuous = (wrapped || !wrap) && t->begin == dcb->begin &&
                            t->scanned >= dcb->begin && t->scanned <= dcb->cursor;
    // Unknown since a packet the tracking does not follow, or since before the first baseline:
    // the latest baseline again, then what this chunk wrote.
    if (continuous && !t->state.known &&
        baseline.sequence.load(std::memory_order_acquire) == t->baseline_sequence) {
        // Still unknown, and no newer baseline to restart from.
        t->scanned = dcb->cursor;
        return t;
    }
    if (!continuous || !t->state.known) {
        // New, reset or reinitialized command buffer: the baseline, then what this chunk wrote.
        track_resets.fetch_add(1, std::memory_order_relaxed);
        if (wrap && !wrapped) {
            Reserve expected = nullptr;
            engine_reserve.compare_exchange_strong(expected, dcb->reserve);
            if (dcb->reserve != engine_reserve.load(std::memory_order_relaxed)) return nullptr;
            t->original = dcb->reserve;
            dcb->reserve = &ReserveWrapper;
        }
        t->baseline_sequence = baseline.sequence.load(std::memory_order_acquire);
        if (TechStartEnabled() && LoadTechniqueStart(t->state, technique)) {
            technique_hits.fetch_add(1, std::memory_order_relaxed);
        } else if (LoadSpanStart(t->state, dcb->begin)) {
            if (TechStartEnabled()) technique_misses.fetch_add(1, std::memory_order_relaxed);
            span_hits.fetch_add(1, std::memory_order_relaxed);
        } else {
            span_misses.fetch_add(1, std::memory_order_relaxed);
            LoadBaseline(t->state);
        }
        t->begin = dcb->begin;
        t->scanned = dcb->begin;
    }
    Apply(t->state, t->scanned, dcb->cursor);
    t->scanned = dcb->cursor;
    return t;
}

// ISS_ITEM_PATH_CENSUS: whether an item draw runs inside a hooked range call.
thread_local u32 in_range_hook = 0;
struct RangeHookScope {
    RangeHookScope() { ++in_range_hook; }
    ~RangeHookScope() { --in_range_hook; }
};

u64 ISS_SYSV RangeHook(u64 technique, u64* args) {
    const RangeHookScope scope;
    auto& s = engine_stats;
    if ((s.calls.fetch_add(1, std::memory_order_relaxed) & 0x3FFF) == 0x3FFF) ReportEngine();
    if (!all_techniques && *reinterpret_cast<const u64*>(technique) != image_base + GbufferVtable) {
        s.other.fetch_add(1, std::memory_order_relaxed);
        return original(technique, args);
    }
    const auto* t = reinterpret_cast<const u8*>(technique);
    // technique+0x1F clear: item by item (0x1A1960); +0x124: the doubled draw with altered state.
    static const bool item_path = [] {
        const char* value = std::getenv("ISS_NATIVE_ITEM_PATH");
        return value && value[0] == '1';
    }();
    if (t[0x1F] == 0) {
        s.item_path.fetch_add(1, std::memory_order_relaxed);
        // ISS_NATIVE_ITEM_PATH=1: item by item ranges too (their packets are checked the same).
        if (!item_path) return original(technique, args);
    }
    if (t[0x124] != 0) {
        s.double_draw.fetch_add(1, std::memory_order_relaxed);
        return original(technique, args);
    }
    auto* dcb = reinterpret_cast<CommandBuffer*>(args[0] + 8);
    if (static_cast<u64>(dcb->end - dcb->cursor) < MarkerDwords &&
        !dcb->reserve(dcb, MarkerDwords, dcb->user)) {
        return original(technique, args);
    }
    Track* track = recorder ? Follow(dcb, technique) : nullptr;
    OpenRange open{dcb, nullptr, 0, {}, track, technique};
    if (track) open.start = track->state;
    open.marker = OpenMarker(dcb, open.generation);
    // Segments need the wrapper (it is installed by Follow) and a single open range per thread.
    const bool split = SegmentsEnabled() && track && dcb->reserve == &ReserveWrapper && !open_range;
    if (split) open_range = &open;
    u32* const end_before = dcb->end;
    const u64 result = original(technique, args);
    if (split) open_range = nullptr;
    const u32* const cursor = dcb->cursor;
    if (track) {
        // The range's own writes (a chunk change inside it was applied by the wrapper).
        if (track->begin == dcb->begin && track->scanned >= dcb->begin && track->scanned <= cursor) {
            Apply(track->state, track->scanned, cursor);
            track->scanned = cursor;
        }
    }
    // A range that continued into another chunk (unsplit) is left to interpretation.
    if ((!split && dcb->end != end_before) || cursor < open.marker + MarkerDwords ||
        open.marker < dcb->begin || open.marker >= dcb->end) {
        s.reject[RejectChunk].fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    FinishRange(open.marker, open.generation, cursor, dcb->begin, track, open.start, technique);
    return result;
}

// A range marker at the cursor (a new batch slot, not ready until FinishRange publishes it).
u32* OpenMarker(CommandBuffer* dcb, u32& generation) {
    u32* const marker = dcb->cursor;
    generation = next_generation.fetch_add(1, std::memory_order_relaxed);
    Batch& b = slots[generation % SlotCount];
    b.ready.store(0, std::memory_order_release);
    b.generation = generation;
    b.recording = nullptr;
    marker[0] = (3u << 30) | ((MarkerDwords - 2) << 16) | (0x10u << 8); // type 3 NOP
    marker[1] = MarkerMagic | MarkerBegin;
    marker[2] = generation % SlotCount;
    marker[3] = generation;
    dcb->cursor = marker + MarkerDwords;
    return marker;
}

// The range from the marker to `cursor`, in the chunk starting at `chunk_begin`: scanned,
// recorded and linked, then published.
void FinishRange(u32* marker, u32 generation, const u32* cursor, u32* chunk_begin,
                 Track* track, const RangeStart& start, u64 technique) {
    auto& s = engine_stats;
    Batch& b = slots[generation % SlotCount];
    b.technique = technique;
    if (!Scan(b, marker + MarkerDwords, cursor)) return;
    b.marker = marker;
    b.end = cursor;
    s.batches.fetch_add(1, std::memory_order_relaxed);
    s.draws.fetch_add(b.draws.size(), std::memory_order_relaxed);
    if (track) {
        if (start.known) b.recording = recorder(b, start);
        else track_unknown.fetch_add(1, std::memory_order_relaxed);
        if (b.recording && linker) {
            if (!track->last || track->last->generation != track->last_generation)
                chain_breaks[259].fetch_add(1, std::memory_order_relaxed);
            else if (track->last_begin != chunk_begin || track->last_end > marker)
                chain_breaks[260].fetch_add(1, std::memory_order_relaxed);
            else if (OnlyPlainRegisterWrites(track->last_end, marker)) {
                linker(*track->last, b);
                links.fetch_add(1, std::memory_order_relaxed);
            }
        }
        track->last = b.recording ? &b : nullptr;
        track->last_generation = generation;
        track->last_end = cursor;
        track->last_begin = chunk_begin;
    }
    b.ready.store(generation, std::memory_order_release);
}
} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_GBUFFER");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool RecorderEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_RECORDER");
        return Enabled() && value && value[0] == '1';
    }();
    return enabled;
}

void SetRecorder(RecordFn fn, LinkFn link) {
    static const bool chains = [] {
        const char* value = std::getenv("ISS_NATIVE_RECORDER_CHAINS");
        return !value || value[0] != '0';
    }();
    linker = RecorderEnabled() && chains ? link : nullptr;
    recorder = RecorderEnabled() ? fn : nullptr;
}

void PublishBaseline(const u32* reg_array, u32 index_type, u32 instances) {
    baseline.sequence.fetch_add(1, std::memory_order_acq_rel);
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(&baseline.words[base], reg_array + TrackedFirst[r], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    baseline.index_type = index_type;
    baseline.instances = instances;
    baseline.sequence.fetch_add(1, std::memory_order_release);
}

bool TechStartEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_RECORDER_TECH_START");
        return value && value[0] == '1';
    }();
    return enabled;
}

void PublishTechniqueStart(u64 technique, const u32* reg_array, u32 index_type, u32 instances) {
    auto state = std::make_unique<RangeStart>();
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(&state->words[base], reg_array + TrackedFirst[r], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    state->index_type = index_type;
    state->instances = instances;
    state->known = true;
    std::scoped_lock lock{span_mutex};
    if (technique_starts.size() >= 4096) technique_starts.clear();
    technique_starts[technique] = std::move(state);
}

void PublishSpanStart(const u32* span, const u32* reg_array, u32 index_type, u32 instances) {
    auto state = std::make_unique<RangeStart>();
    u32 base = 0;
    for (u32 r = 0; r < 3; ++r) {
        std::memcpy(&state->words[base], reg_array + TrackedFirst[r], TrackedCount[r] * 4);
        base += TrackedCount[r];
    }
    state->index_type = index_type;
    state->instances = instances;
    state->known = true;
    std::scoped_lock lock{span_mutex};
    if (span_starts.size() >= 4096) span_starts.clear();
    span_starts[span] = std::move(state);
}

u32 LatestGeneration() {
    return next_generation.load(std::memory_order_relaxed);
}

// ISS_ITEM_PATH_CENSUS=1 (diagnosis): who calls the engine's item by item draw (eboot 0x1A1960,
// tech, ctx, item), by exact return address and technique vtable.
namespace {
using ItemDraw = u64(ISS_SYSV*)(u64 technique, u64 ctx, u64 item);
ItemDraw item_original{};
std::mutex item_census_mutex;
std::unordered_map<u64, u64> item_census;
std::atomic<u64> item_census_calls{};
u64 ISS_SYSV ItemCensusHook(u64 technique, u64 ctx, u64 item) {
    const u64 ret = reinterpret_cast<u64>(__builtin_return_address(0)) - image_base;
    const u64 vtable = *reinterpret_cast<const u64*>(technique) - image_base;
    {
        std::scoped_lock lock{item_census_mutex};
        ++item_census[(ret << 32) | (vtable & 0x7FFFFFFF) | (in_range_hook ? 0x80000000u : 0)];
    }
    if ((item_census_calls.fetch_add(1, std::memory_order_relaxed) & 0x7FFFF) == 0x7FFFF) {
        std::vector<std::pair<u64, u64>> top;
        {
            std::scoped_lock lock{item_census_mutex};
            for (const auto& [key, n] : item_census) top.emplace_back(n, key);
        }
        std::sort(top.rbegin(), top.rend());
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 20); ++i) {
            char line[64];
            std::snprintf(line, sizeof(line), " ret=%llx/vt=%llx:%llu",
                          (unsigned long long)(top[i].second >> 32),
                          (unsigned long long)(top[i].second & 0xFFFFFFFF),
                          (unsigned long long)top[i].first);
            text += line;
        }
        std::fprintf(stderr, "ITEM_PATH_CENSUS calls=%llu[%s ]\n",
                     (unsigned long long)item_census_calls.load(), text.c_str());
    }
    return item_original(technique, ctx, item);
}

// Jumps from `offset` (starting with Prologue) to `hook`; the trampoline runs the original.
void* PatchPrologue(u64 base, u64 offset, const void* hook) {
    auto* code = reinterpret_cast<unsigned char*>(base + offset);
    if (std::memcmp(code, Prologue, sizeof(Prologue)) != 0) return nullptr;
    auto* trampoline = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return nullptr;
    std::memcpy(trampoline, Prologue, sizeof(Prologue));
    const unsigned char jump_back[6] = {0xff, 0x25, 0, 0, 0, 0};
    const auto back = reinterpret_cast<std::uintptr_t>(code + sizeof(Prologue));
    std::memcpy(trampoline + 13, jump_back, 6);
    std::memcpy(trampoline + 19, &back, 8);
    unsigned char patch[13] = {0x48, 0xb8};
    const auto target = reinterpret_cast<std::uintptr_t>(hook);
    std::memcpy(patch + 2, &target, 8);
    patch[10] = 0xff;
    patch[11] = 0xe0;
    patch[12] = 0x90;
    DWORD old, ignored;
    if (!VirtualProtect(code, sizeof(patch), PAGE_EXECUTE_READWRITE, &old)) return nullptr;
    std::memcpy(code, patch, sizeof(patch));
    VirtualProtect(code, sizeof(patch), old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof(patch));
    return trampoline;
}
} // namespace

int Install(u64 base) {
    if (!Enabled()) return 0;
    if (const char* census = std::getenv("ISS_ITEM_PATH_CENSUS"); census && census[0] == '1') {
        image_base = base;
        item_original = reinterpret_cast<ItemDraw>(
            PatchPrologue(base, 0x1A1960, reinterpret_cast<const void*>(&ItemCensusHook)));
        std::fprintf(stderr, "ITEM_PATH_CENSUS %s\n", item_original ? "installed" : "refused");
    }
    auto* code = reinterpret_cast<unsigned char*>(base + RoutineOffset);
    if (std::memcmp(code, Prologue, sizeof(Prologue)) != 0) {
        std::fprintf(stderr, "NATIVE_GBUFFER rejected: eboot+0x%llx prologue mismatch\n",
                     (unsigned long long)RoutineOffset);
        return -1;
    }
    image_base = base;
    slots = std::make_unique<Batch[]>(SlotCount);
    if (const char* value = std::getenv("ISS_NATIVE_ALL_TECHNIQUES")) all_techniques = value[0] == '1';
    // Engine draw markers only feed diagnostics unless the native item path reads them.
    const auto flag = [](const char* name) {
        const char* value = std::getenv(name);
        return value && value[0] == '1';
    };
    allow_engine_markers = !flag("ISS_NATIVE_ITEMS") && !flag("ISS_NATIVE_VERIFY");
    const auto refuse = [](const char* step) {
        std::fprintf(stderr, "NATIVE_GBUFFER rejected: %s failed (error %lu)\n", step, GetLastError());
        return -1;
    };
    auto* trampoline = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return refuse("VirtualAlloc");
    std::memcpy(trampoline, Prologue, sizeof(Prologue));
    const unsigned char jump_back[6] = {0xff, 0x25, 0, 0, 0, 0};
    const auto back = reinterpret_cast<std::uintptr_t>(code + sizeof(Prologue));
    std::memcpy(trampoline + 13, jump_back, 6);
    std::memcpy(trampoline + 19, &back, 8);
    original = reinterpret_cast<Render>(trampoline);
    unsigned char patch[13] = {0x48, 0xb8};
    const auto hook = reinterpret_cast<std::uintptr_t>(&RangeHook);
    std::memcpy(patch + 2, &hook, 8);
    patch[10] = 0xff;
    patch[11] = 0xe0;
    patch[12] = 0x90;
    DWORD old, ignored;
    if (!VirtualProtect(code, sizeof(patch), PAGE_EXECUTE_READWRITE, &old))
        return refuse("VirtualProtect");
    std::memcpy(code, patch, sizeof(patch));
    if (!VirtualProtect(code, sizeof(patch), old, &ignored) ||
        !FlushInstructionCache(GetCurrentProcess(), code, sizeof(patch)))
        return refuse("restore protection");
    std::fprintf(stderr, "NATIVE_GBUFFER installed at eboot+0x%llx (technique vtable 0x%llx)\n",
                 (unsigned long long)RoutineOffset, (unsigned long long)GbufferVtable);
    return 1;
}

const Batch* Find(const u32* marker) {
    if (!slots) return nullptr;
    const u32 slot = marker[2], generation = marker[3];
    if (slot >= SlotCount || generation % SlotCount != slot) return nullptr;
    const Batch& b = slots[slot];
    if (b.ready.load(std::memory_order_acquire) != generation || b.marker != marker) return nullptr;
    return &b;
}

void NoteRange(Reason reason, u32 draws) {
    auto& s = cp_stats;
    ++s.ranges[reason];
    s.draws[reason] += draws;
    if ((++s.total & 0xFFF) != 1) return;
    static const char* const names[ReasonCount] = {"native", "partial", "learn", "context",
                                                   "missing", "mismatch", "verify"};
    std::string text;
    for (u32 i = 0; i < ReasonCount; ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s=%llu/%llu", names[i],
                      (unsigned long long)s.ranges[i], (unsigned long long)s.draws[i]);
        text += item;
    }
    std::fprintf(stderr, "NATIVE_GBUFFER ranges/draws%s\n", text.c_str());
    ReportEngine();
}

} // namespace iss::native_gbuffer
