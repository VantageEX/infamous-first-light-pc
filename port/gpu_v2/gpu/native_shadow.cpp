// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_SHADOW: the engine side (native_shadow.h). Runs on the engine's threads.
#include <atomic>
#include "native_shadow.h"
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#ifndef ISS_SYSV
#define ISS_SYSV __attribute__((sysv_abi))
#endif

namespace iss::native_shadow {
namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u64 RoutineOffset = 0x43D590; // depth technique, render item range (slot 6)
constexpr u64 ShadowVtable = 0xD68648;  // the technique of the 2048/1024 shadow maps
constexpr u32 SlotCount = 8192;
constexpr u32 ContextBase = 0xA000, ShBase = 0x2C00, UconfigBase = 0xC000, ConfigBase = 0x2000;
constexpr unsigned char Prologue[13] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                        0x56, 0x41, 0x55, 0x41, 0x54, 0x53};

// The engine's command buffer, at context + 8 (native/engine_draw.cpp).
struct CommandBuffer {
    u64 reserved;
    u32* end;
    u32* cursor;
    bool(ISS_SYSV* reserve)(CommandBuffer*, u32, void*);
    void* user;
};

using Render = u64(ISS_SYSV*)(u64 technique, u64* args);
Render original{};
using RenderBuckets = u64(ISS_SYSV*)(u64 list, u64 context, u32 first, u32 end);
RenderBuckets original_buckets{};
thread_local bool capturing_outer{};
u64 image_base{};
std::unique_ptr<Batch[]> slots;
std::atomic<u32> next_generation{1};

enum Reject : u32 { RejectChunk, RejectOpcode, RejectTarget, RejectCompute, RejectRegs,
                    RejectBounds, RejectEmpty, RejectCount };
constexpr const char* reject_names[RejectCount] = {"chunk", "opcode", "target", "compute",
                                                   "regs", "bounds", "empty"};
struct EngineStats {
    std::atomic<u64> calls{}, other{}, item_path{}, batches{}, draws{};
    std::array<std::atomic<u64>, RejectCount> reject{};
    std::array<std::atomic<u64>, 256> opcode{};
    std::atomic<u32> dma_dst_lo{}; // the last rejected DMA_DATA destination
};
EngineStats engine_stats;

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
    std::string opcodes;
    for (u32 i = 0; i < 256; ++i) {
        if (const u64 n = s.opcode[i].load(std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), " %02x=%llu", i, (unsigned long long)n);
            opcodes += item;
        }
    }
    std::fprintf(stderr,
                 "NATIVE_SHADOW_ENGINE calls=%llu other_technique=%llu item_path=%llu "
                 "batches=%llu draws=%llu reject[%s ] reject_opcodes[%s ] last_dma_dst=0x%x\n",
                 (unsigned long long)s.calls.load(), (unsigned long long)s.other.load(),
                 (unsigned long long)s.item_path.load(), (unsigned long long)s.batches.load(),
                 (unsigned long long)s.draws.load(), text.c_str(), opcodes.c_str(),
                 s.dma_dst_lo.load());
}

bool IsTargetReg(u32 reg) {
    // DB surface registers and the CB color targets: the range must keep the render targets.
    return (reg >= ContextBase && reg < ContextBase + 0x20) ||
           (reg >= ContextBase + 0x318 && reg < ContextBase + 0x3A0);
}
bool RejectTargetWrite(u32 reg, u32 value) {
    // DB_RENDER_OVERRIDE is state, not an address. Copy/force modes remain
    // excluded; DB_DEPTH_BOUNDS_MIN/MAX are plain dynamic depth bounds.
    if (reg == ContextBase + 3) return (value & 0x78000000u) != 0;
    if (reg == ContextBase + 8 || reg == ContextBase + 9) return false;
    return IsTargetReg(reg);
}

// Reads back the commands the routine wrote, [first, end). The first pass collects the state
// registers and validates every packet; the second records the draws and state versions.
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
    const auto is_draw_user = [](u32 reg) {
        return reg >= VsUserData2 && reg < VsUserData2 + DrawUserWords;
    };
    // Pass 1.
    u32 draws = 0;
    for (const u32* p = first; p < end;) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) {
            ++p;
            continue;
        }
        if (type != 3) return reject(RejectOpcode);
        const u32 count = ((header >> 16) & 0x3FFF) + 1; // payload dwords
        const u32 opcode = (header >> 8) & 0xFF;
        const u32* next = p + 1 + count;
        if (next > end) return reject(RejectBounds);
        switch (opcode) {
        case 0x10: // NOP
        case 0x2A: // INDEX_TYPE
        case 0x2F: // NUM_INSTANCES
            break;
        case 0x27: // DRAW_INDEX_2
            if (count != 5) return reject(RejectOpcode);
            ++draws;
            break;
        case 0x50: // DMA_DATA to 0x3022C: an explicit no-op for the command processor
            if (count < 6 || p[4] != 0x3022C) {
                engine_stats.opcode[opcode].fetch_add(1, std::memory_order_relaxed);
                engine_stats.dma_dst_lo.store(count >= 4 ? p[4] : 0, std::memory_order_relaxed);
                return reject(RejectOpcode);
            }
            break;
        case 0x68: case 0x69: case 0x76: case 0x79: { // SET_*_REG
            if (count < 2) return reject(RejectOpcode);
            const u32 base = opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
                           : opcode == 0x76 ? ShBase : UconfigBase;
            if (opcode == 0x76 && p[1] >= 0x200) return reject(RejectCompute);
            for (u32 i = 0; i < count - 1; ++i) {
                const u32 reg = base + p[1] + i;
                if (opcode == 0x69 && RejectTargetWrite(reg, p[2 + i])) return reject(RejectTarget);
                if (is_draw_user(reg) ||
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

    // Pass 2.
    std::array<u32, MaxStateRegs> values{};
    u64 inherit = nregs == 64 ? ~0ull : (1ull << nregs) - 1;
    bool dirty = true;
    Draw current{};
    current.user_inherit = (1u << DrawUserWords) - 1;
    current.instances_inherit = true;
    current.index_type_inherit = true;
    b.draws.reserve(draws);
    for (const u32* p = first; p < end;) {
        const u32 header = *p;
        if ((header >> 30) == 2) {
            ++p;
            continue;
        }
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
            const u32 base = opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
                           : opcode == 0x76 ? ShBase : UconfigBase;
            for (u32 i = 0; i < count - 1; ++i) {
                const u32 reg = base + p[1] + i, value = p[2 + i];
                if (is_draw_user(reg)) {
                    current.user[reg - VsUserData2] = value;
                    current.user_inherit &= ~(1u << (reg - VsUserData2));
                } else {
                    const u32 k = u32(std::lower_bound(b.state_regs.begin(), b.state_regs.end(), reg) -
                                      b.state_regs.begin());
                    // Repeated setters do not create a new draw state. Inherited
                    // zero is still distinct from an explicit write of zero.
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

#include "native_shadow_capture.inl"

u64 ISS_SYSV RenderHook(u64 technique, u64* args) {
    if (capturing_outer) return original(technique, args);
    auto& s = engine_stats;
    if ((s.calls.fetch_add(1, std::memory_order_relaxed) & 0x3FFF) == 0x3FFF) ReportEngine();
    if (*reinterpret_cast<const u64*>(technique) != image_base + ShadowVtable) {
        s.other.fetch_add(1, std::memory_order_relaxed);
        return original(technique, args);
    }
    // technique+0x1F: the instanced branch (0 draws item by item through DrawItem).
    if (*reinterpret_cast<const unsigned char*>(technique + 0x1F) == 0) {
        s.item_path.fetch_add(1, std::memory_order_relaxed);
        return original(technique, args);
    }
    auto* dcb = reinterpret_cast<CommandBuffer*>(args[0] + 8);
    if (static_cast<u64>(dcb->end - dcb->cursor) < MarkerDwords &&
        !dcb->reserve(dcb, MarkerDwords, dcb->user)) {
        return original(technique, args);
    }
    u32* const marker = dcb->cursor;
    const u32 generation = next_generation.fetch_add(1, std::memory_order_relaxed);
    Batch& b = slots[generation % SlotCount];
    b.ready.store(0, std::memory_order_release);
    marker[0] = (3u << 30) | ((MarkerDwords - 2) << 16) | (0x10u << 8); // type 3 NOP
    marker[1] = MarkerMagic | MarkerBegin;
    marker[2] = generation % SlotCount;
    marker[3] = generation;
    dcb->cursor = marker + MarkerDwords;
    u32* const end_before = dcb->end;
    std::optional<DirectCapture> capture;
    if (DirectCaptureEnabled()) capture.emplace(b, marker + MarkerDwords, end_before);
    const u64 result = original(technique, args);
    const u32* const cursor = dcb->cursor;
    // A range that continued into another chunk of the command buffer is left to interpretation.
    if (dcb->end != end_before || cursor < marker + MarkerDwords) {
        s.reject[RejectChunk].fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    if (FinishCapture(capture ? &*capture : nullptr, b, marker + MarkerDwords, cursor)) {
        b.marker = marker;
        b.end = cursor;
        s.batches.fetch_add(1, std::memory_order_relaxed);
        s.draws.fetch_add(b.draws.size(), std::memory_order_relaxed);
        b.ready.store(generation, std::memory_order_release);
    }
    return result;
}

#include "native_shadow_outer.inl"
} // namespace

void CaptureIndexed(const u32* packet, u32 count, u64 address, u32 initiator) {
    if (active_capture) active_capture->DrawIndexed(packet, count, address, initiator);
}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ISS_NATIVE_SHADOW");
        return value && value[0] == '1';
    }();
    return enabled;
}

int Install(u64 base, const unsigned char* code) {
    if (!Enabled()) return 0;
    if (!code || std::memcmp(code, Prologue, sizeof(Prologue)) != 0) {
        std::fprintf(stderr, "NATIVE_SHADOW rejected: eboot+0x%llx prologue mismatch\n",
                     (unsigned long long)RoutineOffset);
        return -1;
    }
    image_base = base;
    slots = std::make_unique<Batch[]>(SlotCount);
    const auto refuse = [](const char* step) {
        std::fprintf(stderr, "NATIVE_SHADOW rejected: %s failed (error %lu)\n", step, GetLastError());
        return -1;
    };
    auto* trampoline = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return refuse("VirtualAlloc");
    // The 13 prologue bytes run from the trampoline, which jumps back after them; the entry
    // becomes `movabs rax, hook; jmp rax` (RAX carries no argument).
    std::memcpy(trampoline, Prologue, sizeof(Prologue));
    const unsigned char jump_back[6] = {0xff, 0x25, 0, 0, 0, 0};
    const auto back = reinterpret_cast<std::uintptr_t>(code + sizeof(Prologue));
    std::memcpy(trampoline + 13, jump_back, 6);
    std::memcpy(trampoline + 19, &back, 8);
    original = reinterpret_cast<Render>(trampoline);
    unsigned char patch[13] = {0x48, 0xb8};
    const auto hook = reinterpret_cast<std::uintptr_t>(&RenderHook);
    std::memcpy(patch + 2, &hook, 8);
    patch[10] = 0xff;
    patch[11] = 0xe0;
    patch[12] = 0x90;
    auto* target = const_cast<unsigned char*>(code);
    DWORD old, ignored;
    if (!VirtualProtect(target, sizeof(patch), PAGE_EXECUTE_READWRITE, &old))
        return refuse("VirtualProtect");
    std::memcpy(target, patch, sizeof(patch));
    if (!VirtualProtect(target, sizeof(patch), old, &ignored) ||
        !FlushInstructionCache(GetCurrentProcess(), target, sizeof(patch))) {
        return refuse("restore protection");
    }
    std::fprintf(stderr, "NATIVE_SHADOW installed at eboot+0x%llx (technique vtable 0x%llx)\n",
                 (unsigned long long)RoutineOffset, (unsigned long long)ShadowVtable);
    if (const char* outer = std::getenv("ISS_NATIVE_SHADOW_OUTER"); outer && outer[0] == '1')
        std::fprintf(stderr, "NATIVE_SHADOW_OUTER install=%s offset=0x198bc0\n",
                     InstallBuckets() ? "ok" : "rejected");
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
    std::fprintf(stderr, "NATIVE_SHADOW ranges/draws%s\n", text.c_str());
    ReportEngine();
}

} // namespace iss::native_shadow
