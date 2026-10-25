// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_HEAP_CENSUS: see guest_heap_census.h.
#include "guest_heap_census.h"

#include <windows.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace iss::heap_census {
namespace {
using u64 = std::uint64_t;

constexpr u64 InsertOffset = 0x58b60;  // insert(heap, node): a new block or free piece
constexpr u64 ReleaseOffset = 0x5a790; // release(heap, node): a block becomes free and merges
// Node fields (ghidra): +0x28 address, +0x30 size, +0x3a kind (0 free, 1 block, 2 slab).
constexpr u64 NodeAddress = 0x28, NodeSize = 0x30, NodeKind = 0x3a;

// insert: mov rdx,[rdi]; test rdx,rdx; jz +0x5d (-> +0x65); mov rcx,[rsi+0x28]; nop dword [rax]
constexpr unsigned char InsertEntry[16] = {0x48, 0x8b, 0x17, 0x48, 0x85, 0xd2, 0x74, 0x5d,
                                           0x48, 0x8b, 0x4e, 0x28, 0x0f, 0x1f, 0x40, 0x00};
// release: push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; push rax
constexpr unsigned char ReleaseEntry[14] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57,
                                            0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50};

struct Counters {
    std::atomic<u64> heap{}, live{}, peak{}, lowest{~0ull}, highest{}, allocs{}, frees{};
};
Counters heaps[2];
std::atomic<bool> installed{};
using Routine = void (*)(void*, void*);
Routine insert_original{}, release_original{};

Counters* Find(void* heap) {
    const u64 key = reinterpret_cast<u64>(heap);
    for (Counters& c : heaps) {
        u64 expected = 0;
        if (c.heap.load(std::memory_order_relaxed) == key ||
            c.heap.compare_exchange_strong(expected, key, std::memory_order_relaxed) ||
            expected == key)
            return &c;
    }
    return nullptr;
}

template <typename T>
T Field(const void* node, u64 offset) {
    T v;
    std::memcpy(&v, static_cast<const unsigned char*>(node) + offset, sizeof(T));
    return v;
}

void InsertHook(void* heap, void* node) {
    if (Field<std::uint8_t>(node, NodeKind) != 0) {
        if (Counters* c = Find(heap)) {
            const u64 address = Field<u64>(node, NodeAddress), size = Field<u64>(node, NodeSize);
            const u64 live = c->live.fetch_add(size, std::memory_order_relaxed) + size;
            u64 peak = c->peak.load(std::memory_order_relaxed);
            while (live > peak && !c->peak.compare_exchange_weak(peak, live)) {}
            u64 low = c->lowest.load(std::memory_order_relaxed);
            while (address < low && !c->lowest.compare_exchange_weak(low, address)) {}
            u64 high = c->highest.load(std::memory_order_relaxed);
            while (address + size > high && !c->highest.compare_exchange_weak(high, address + size)) {}
            c->allocs.fetch_add(1, std::memory_order_relaxed);
        }
    }
    insert_original(heap, node);
}

void ReleaseHook(void* heap, void* node) {
    if (Field<std::uint8_t>(node, NodeKind) != 0) {
        if (Counters* c = Find(heap)) {
            c->live.fetch_sub(Field<u64>(node, NodeSize), std::memory_order_relaxed);
            c->frees.fetch_add(1, std::memory_order_relaxed);
        }
    }
    release_original(heap, node);
}

// Writes `movabs rax, target; jmp rax` (12 bytes) at `at` and fills up to `length` with nops.
// RAX carries no argument at either entry.
bool Patch(unsigned char* at, unsigned length, const void* target) {
    unsigned char patch[16];
    std::memset(patch, 0x90, sizeof(patch));
    patch[0] = 0x48;
    patch[1] = 0xb8;
    const auto t = reinterpret_cast<u64>(target);
    std::memcpy(patch + 2, &t, 8);
    patch[10] = 0xff;
    patch[11] = 0xe0;
    DWORD old, ignored;
    if (!VirtualProtect(at, length, PAGE_EXECUTE_READWRITE, &old)) return false;
    std::memcpy(at, patch, length);
    return VirtualProtect(at, length, old, &ignored) &&
           FlushInstructionCache(GetCurrentProcess(), at, length);
}

// An absolute jump (jmp [rip+0]; dq target), 14 bytes.
unsigned char* JumpTo(unsigned char* p, u64 target) {
    const unsigned char jmp[6] = {0xff, 0x25, 0, 0, 0, 0};
    std::memcpy(p, jmp, 6);
    std::memcpy(p + 6, &target, 8);
    return p + 14;
}
} // namespace

void Install(u64 image_base) {
    if (const char* p = std::getenv("ISS_HEAP_CENSUS"); !p || p[0] != '1') return;
    auto* insert = reinterpret_cast<unsigned char*>(image_base + InsertOffset);
    auto* release = reinterpret_cast<unsigned char*>(image_base + ReleaseOffset);
    if (std::memcmp(insert, InsertEntry, sizeof(InsertEntry)) != 0 ||
        std::memcmp(release, ReleaseEntry, sizeof(ReleaseEntry)) != 0) {
        std::fprintf(stderr, "HEAP_CENSUS off: allocator entry bytes differ (eboot is not US 1.07)\n");
        return;
    }
    auto* code = static_cast<unsigned char*>(
        VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!code) {
        std::fprintf(stderr, "HEAP_CENSUS off: VirtualAlloc failed\n");
        return;
    }
    // insert's trampoline: its first 12 bytes, the short jz rewritten as a jz over an absolute
    // jump to its target, then back to +0x10 (the nop at +0xc is dropped).
    unsigned char* p = code;
    std::memcpy(p, InsertEntry, 6);                 // mov rdx,[rdi]; test rdx,rdx
    p += 6;
    *p++ = 0x75;                                   // jnz over the taken branch
    *p++ = 14;
    p = JumpTo(p, image_base + InsertOffset + 8 + 0x5d);
    std::memcpy(p, InsertEntry + 8, 4);             // mov rcx,[rsi+0x28]
    p += 4;
    p = JumpTo(p, image_base + InsertOffset + 0x10);
    insert_original = reinterpret_cast<Routine>(code);
    // release's trampoline: its 14 bytes of pushes, then back.
    unsigned char* r = code + 128;
    std::memcpy(r, ReleaseEntry, sizeof(ReleaseEntry));
    JumpTo(r + sizeof(ReleaseEntry), image_base + ReleaseOffset + sizeof(ReleaseEntry));
    release_original = reinterpret_cast<Routine>(r);
    if (!Patch(insert, 16, reinterpret_cast<const void*>(&InsertHook)) ||
        !Patch(release, 14, reinterpret_cast<const void*>(&ReleaseHook))) {
        std::fprintf(stderr, "HEAP_CENSUS off: patching failed\n");
        return;
    }
    installed.store(true, std::memory_order_release);
    std::fprintf(stderr, "HEAP_CENSUS on (insert eboot+0x%llx, release eboot+0x%llx)\n",
                 (unsigned long long)InsertOffset, (unsigned long long)ReleaseOffset);
}

bool Read(Heap out[2]) {
    if (!installed.load(std::memory_order_acquire)) return false;
    for (int i = 0; i < 2; ++i) {
        const Counters& c = heaps[i];
        const u64 low = c.lowest.load(std::memory_order_relaxed);
        out[i] = {c.live.load(std::memory_order_relaxed), c.peak.load(std::memory_order_relaxed),
                  low == ~0ull ? 0 : low, c.highest.load(std::memory_order_relaxed),
                  c.allocs.load(std::memory_order_relaxed), c.frees.load(std::memory_order_relaxed)};
    }
    return true;
}

} // namespace iss::heap_census
