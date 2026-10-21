// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_GBUFFER_CENSUS=1: what the engine's G-buffer material ranges write (observation only).
//
// The G-buffer technique (vtable 0xD0FAC8) draws its items through the generic material range
// routine (eboot 0x1A2030, slot 6): material binds (0x448640), instance records of 212 bytes and
// the draw helper 0x1A1770, or item by item (0x1A1960) when technique+0x1F is clear. A hook on
// the routine reads back the commands each range wrote and reports, every 4096 ranges:
// - the branches taken (instanced/item, double draw technique+0x124, extra descriptor +0x135);
// - draws per range and ranges that crossed a command buffer chunk;
// - the packet opcodes in the ranges (constant engine sync included);
// - the registers set, by how many draws set them (per draw or per range state).
// Native G-buffer contract input; nothing is changed.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifndef ISS_SYSV
#define ISS_SYSV __attribute__((sysv_abi))
#endif

namespace iss::native_gbuffer {
namespace {
using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u64 RoutineOffset = 0x1A2030;
constexpr u64 GbufferVtable = 0xD0FAC8;
constexpr unsigned char Prologue[13] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41,
                                        0x56, 0x41, 0x55, 0x41, 0x54, 0x53};
constexpr u32 ContextBase = 0xA000, ShBase = 0x2C00, UconfigBase = 0xC000, ConfigBase = 0x2000;

struct CommandBuffer {
    u64 reserved;
    u32* end;
    u32* cursor;
    bool(ISS_SYSV* reserve)(CommandBuffer*, u32, void*);
    void* user;
};
using Render = u64(ISS_SYSV*)(u64 technique, u64* args);
Render original{};
u64 image_base{};

// Register slots: context 0x400, SH 0x400, uconfig 0x400, config 0x400.
constexpr u32 RegSlots = 0x1000;
u32 Slot(u32 reg) {
    if (reg >= ContextBase && reg < ContextBase + 0x400) return reg - ContextBase;
    if (reg >= ShBase && reg < ShBase + 0x400) return 0x400 + reg - ShBase;
    if (reg >= UconfigBase && reg < UconfigBase + 0x400) return 0x800 + reg - UconfigBase;
    if (reg >= ConfigBase && reg < ConfigBase + 0x400) return 0xC00 + reg - ConfigBase;
    return ~0u;
}
u32 Reg(u32 slot) {
    return slot < 0x400 ? ContextBase + slot : slot < 0x800 ? ShBase + slot - 0x400
         : slot < 0xC00 ? UconfigBase + slot - 0x800 : ConfigBase + slot - 0xC00;
}

struct Stats {
    std::atomic<u64> calls{}, other{}, item{}, instanced{}, double_draw{}, extra_desc{},
        chunk{}, empty{}, ranges{}, draws{}, max_draws{}, packets{}, bytes{};
    std::array<std::atomic<u64>, 256> opcodes{};
    std::array<std::atomic<u64>, RegSlots> reg_writes{};  // writes
    std::array<std::atomic<u64>, RegSlots> reg_ranges{};  // ranges setting it at least once
    std::array<std::atomic<u64>, 16> draw_hist{};         // draws per range, log2 buckets
    std::array<std::atomic<u64>, 8> dma_dst{};            // first DMA_DATA destinations
};
Stats stats;

void Report() {
    auto& s = stats;
    const u64 ranges = std::max<u64>(s.ranges.load(), 1), draws = std::max<u64>(s.draws.load(), 1);
    std::fprintf(stderr,
                 "GBUFFER_CENSUS calls=%llu other_technique=%llu item_path=%llu instanced=%llu "
                 "double_draw=%llu extra_desc=%llu chunk=%llu empty=%llu ranges=%llu draws=%llu "
                 "draws_per_range=%.2f max_draws=%llu dwords_per_draw=%.1f\n",
                 (unsigned long long)s.calls.load(), (unsigned long long)s.other.load(),
                 (unsigned long long)s.item.load(), (unsigned long long)s.instanced.load(),
                 (unsigned long long)s.double_draw.load(), (unsigned long long)s.extra_desc.load(),
                 (unsigned long long)s.chunk.load(), (unsigned long long)s.empty.load(),
                 (unsigned long long)s.ranges.load(), (unsigned long long)s.draws.load(),
                 double(s.draws.load()) / ranges, (unsigned long long)s.max_draws.load(),
                 double(s.bytes.load()) / 4.0 / draws);
    std::string text;
    for (u32 i = 0; i < 256; ++i)
        if (const u64 n = s.opcodes[i].load()) {
            char item[40];
            std::snprintf(item, sizeof(item), " %02x=%.2f", i, double(n) / draws);
            text += item;
        }
    std::fprintf(stderr, "GBUFFER_CENSUS opcodes_per_draw%s\n", text.c_str());
    text.clear();
    for (u32 i = 0; i < 16; ++i)
        if (const u64 n = s.draw_hist[i].load()) {
            char item[40];
            std::snprintf(item, sizeof(item), " <%u:%llu", 1u << i, (unsigned long long)n);
            text += item;
        }
    std::fprintf(stderr, "GBUFFER_CENSUS draws_per_range%s\n", text.c_str());
    std::vector<std::pair<u64, u32>> regs;
    for (u32 i = 0; i < RegSlots; ++i)
        if (const u64 n = s.reg_writes[i].load()) regs.emplace_back(n, i);
    std::sort(regs.rbegin(), regs.rend());
    text.clear();
    for (std::size_t i = 0; i < std::min<std::size_t>(regs.size(), 96); ++i) {
        char item[64];
        // writes per draw / share of ranges that set it
        std::snprintf(item, sizeof(item), " %x:%.2f/%.0f%%", Reg(regs[i].second),
                      double(regs[i].first) / draws,
                      100.0 * double(s.reg_ranges[regs[i].second].load()) / ranges);
        text += item;
    }
    std::fprintf(stderr, "GBUFFER_CENSUS regs=%zu writes_per_draw/ranges%s\n", regs.size(),
                 text.c_str());
    text.clear();
    for (const auto& d : s.dma_dst)
        if (const u64 v = d.load()) {
            char item[32];
            std::snprintf(item, sizeof(item), " 0x%llx", (unsigned long long)v);
            text += item;
        }
    std::fprintf(stderr, "GBUFFER_CENSUS dma_dst%s\n", text.c_str());
}

void Scan(const u32* first, const u32* end) {
    auto& s = stats;
    std::array<u8, RegSlots / 8> seen{};
    u32 draws = 0;
    for (const u32* p = first; p < end;) {
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) { ++p; continue; }
        if (type != 3) break;
        const u32 count = ((header >> 16) & 0x3FFF) + 1;
        const u32 opcode = (header >> 8) & 0xFF;
        if (p + 1 + count > end) break;
        s.opcodes[opcode].fetch_add(1, std::memory_order_relaxed);
        s.packets.fetch_add(1, std::memory_order_relaxed);
        if (opcode == 0x27 || opcode == 0x2D || opcode == 0x26 || opcode == 0x2E) ++draws;
        if (opcode == 0x50 && count >= 4) {
            for (auto& d : s.dma_dst) {
                u64 expected = 0;
                if (d.load(std::memory_order_relaxed) == p[4]) break;
                if (d.compare_exchange_strong(expected, p[4])) break;
            }
        }
        if ((opcode == 0x68 || opcode == 0x69 || opcode == 0x76 || opcode == 0x79) && count >= 2) {
            const u32 base = opcode == 0x68 ? ConfigBase : opcode == 0x69 ? ContextBase
                           : opcode == 0x76 ? ShBase : UconfigBase;
            for (u32 i = 0; i < count - 1; ++i) {
                const u32 slot = Slot(base + p[1] + i);
                if (slot == ~0u) continue;
                s.reg_writes[slot].fetch_add(1, std::memory_order_relaxed);
                if (!(seen[slot >> 3] & (1u << (slot & 7)))) {
                    seen[slot >> 3] |= u8(1u << (slot & 7));
                    s.reg_ranges[slot].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        p += 1 + count;
    }
    s.bytes.fetch_add(u64(end - first) * 4, std::memory_order_relaxed);
    if (!draws) { s.empty.fetch_add(1, std::memory_order_relaxed); return; }
    s.ranges.fetch_add(1, std::memory_order_relaxed);
    s.draws.fetch_add(draws, std::memory_order_relaxed);
    u64 max = s.max_draws.load(std::memory_order_relaxed);
    while (draws > max && !s.max_draws.compare_exchange_weak(max, draws)) {}
    u32 bucket = 0;
    while (bucket < 15 && (1u << (bucket + 1)) <= draws) ++bucket;
    s.draw_hist[bucket].fetch_add(1, std::memory_order_relaxed);
}

u64 ISS_SYSV RangeHook(u64 technique, u64* args) {
    auto& s = stats;
    if ((s.calls.fetch_add(1, std::memory_order_relaxed) & 0xFFF) == 0xFFF) Report();
    if (*reinterpret_cast<const u64*>(technique) != image_base + GbufferVtable) {
        s.other.fetch_add(1, std::memory_order_relaxed);
        return original(technique, args);
    }
    const auto* t = reinterpret_cast<const u8*>(technique);
    if (t[0x1F] == 0) s.item.fetch_add(1, std::memory_order_relaxed);
    else s.instanced.fetch_add(1, std::memory_order_relaxed);
    if (t[0x124]) s.double_draw.fetch_add(1, std::memory_order_relaxed);
    if (t[0x135] != 0xFF) s.extra_desc.fetch_add(1, std::memory_order_relaxed);
    auto* dcb = reinterpret_cast<CommandBuffer*>(args[0] + 8);
    const u32* const before = dcb->cursor;
    u32* const end_before = dcb->end;
    const u64 result = original(technique, args);
    if (dcb->end != end_before || dcb->cursor < before) {
        s.chunk.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    Scan(before, dcb->cursor);
    return result;
}
} // namespace

int InstallCensus(u64 base) {
    const char* flag = std::getenv("ISS_GBUFFER_CENSUS");
    if (!flag || flag[0] != '1') return 0;
    if (const char* native = std::getenv("ISS_NATIVE_GBUFFER"); native && native[0] == '1') {
        std::fputs("GBUFFER_CENSUS skipped: ISS_NATIVE_GBUFFER owns eboot+0x1a2030\n", stderr);
        return 0;
    }
    auto* code = reinterpret_cast<unsigned char*>(base + RoutineOffset);
    if (std::memcmp(code, Prologue, sizeof(Prologue)) != 0) {
        std::fprintf(stderr, "GBUFFER_CENSUS rejected: eboot+0x%llx prologue mismatch\n",
                     (unsigned long long)RoutineOffset);
        return -1;
    }
    image_base = base;
    auto* trampoline = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return -1;
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
    if (!VirtualProtect(code, sizeof(patch), PAGE_EXECUTE_READWRITE, &old)) return -1;
    std::memcpy(code, patch, sizeof(patch));
    if (!VirtualProtect(code, sizeof(patch), old, &ignored) ||
        !FlushInstructionCache(GetCurrentProcess(), code, sizeof(patch)))
        return -1;
    std::fprintf(stderr, "GBUFFER_CENSUS installed at eboot+0x%llx (technique vtable 0x%llx)\n",
                 (unsigned long long)RoutineOffset, (unsigned long long)GbufferVtable);
    return 1;
}
} // namespace iss::native_gbuffer
