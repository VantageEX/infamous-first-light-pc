// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3's stutter report (v3_stutter.h).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "common/fault_monitor.h"

#include "v3_stutter.h"

namespace V3::Stutter {

std::atomic<uint64_t> g_faults{0};
std::atomic<uint64_t> g_presented{0};

namespace {
std::mutex g_faults_mutex;
struct RegionFaults {
    uint32_t reads, writes, cp; // cp: of either kind, on the command processor's thread
};
std::unordered_map<uint64_t, RegionFaults> g_frame_faults; // MiB -> faults this frame
std::mutex g_sites_mutex;
std::vector<Site*> g_sites;
struct Stack {
    uint32_t count;
    uint16_t depth;
    void* frames[20];
};
std::unordered_map<ULONG, Stack> g_frame_stacks; // under g_faults_mutex; key: the frames' hash

const char* BaseName(const char* path);

// "module+0xoffset", the module's file name only.
std::string FrameName(void* address) {
    HMODULE module = nullptr;
    char path[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module)) {
        GetModuleFileNameA(module, path, sizeof(path));
    }
    char out[MAX_PATH + 32];
    std::snprintf(out, sizeof(out), "%s+%#llx", BaseName(path),
                  static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) -
                                                  reinterpret_cast<uintptr_t>(module)));
    return out;
}

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("TLG_V3_STUTTER");
        return !(v && v[0] == '0');
    }();
    return on;
}

const char* BaseName(const char* path) {
    const char* slash = std::strrchr(path, '/');
    const char* back = std::strrchr(path, '\\');
    if (back > slash) slash = back;
    return slash ? slash + 1 : path;
}
} // namespace

void RecordFault(uintptr_t address, bool write, bool cp_thread) {
    if (!Enabled()) return;
    std::scoped_lock lock{g_faults_mutex};
    RegionFaults& region = g_frame_faults[address >> 20];
    ++(write ? region.writes : region.reads);
    region.cp += cp_thread;
}

void RecordCpWriteStack() {
    if (!Enabled()) return;
    static std::atomic<uint32_t> calls{0};
    if (calls.fetch_add(1, std::memory_order_relaxed) % 16 != 0) return;
    Stack stack{};
    ULONG hash = 0;
    stack.depth = RtlCaptureStackBackTrace(0, 20, stack.frames, &hash);
    std::scoped_lock lock{g_faults_mutex};
    auto [it, inserted] = g_frame_stacks.try_emplace(hash, stack);
    ++it->second.count;
}

void Site::Register(Site* site) {
    std::scoped_lock lock{g_sites_mutex};
    g_sites.push_back(site);
}

void OnPresent() {
    g_presented.fetch_add(1, std::memory_order_relaxed);
    if (!Enabled()) return;
    using Clock = std::chrono::steady_clock;
    static Clock::time_point last_present{};
    static double average_ms = 0.0;
    static uint64_t last_faults = 0, frames = 0, reported = 0;
    const auto now = Clock::now();
    const bool first = last_present == Clock::time_point{};
    const double ms = first ? 0.0 : std::chrono::duration<double, std::milli>(now - last_present).count();
    last_present = now;
    ++frames;

    // Every site's calls in this frame (the present thread is the only reader of `last`).
    struct Delta {
        uint64_t count;
        Site* site;
    };
    std::vector<Delta> deltas;
    {
        std::scoped_lock lock{g_sites_mutex};
        deltas.reserve(g_sites.size());
        for (Site* site : g_sites) {
            const uint64_t hits = site->hits.load(std::memory_order_relaxed);
            if (hits != site->last) deltas.push_back({hits - site->last, site});
            site->last = hits;
        }
    }
    const uint64_t faults = g_faults.load(std::memory_order_relaxed);
    const uint64_t frame_faults = faults - last_faults;
    last_faults = faults;
    std::unordered_map<uint64_t, RegionFaults> regions;
    std::unordered_map<ULONG, Stack> stacks;
    {
        std::scoped_lock lock{g_faults_mutex};
        regions.swap(g_frame_faults);
        stacks.swap(g_frame_stacks);
    }
    // The heap's write faults widened to their block or kept to one page (GPU-modified block).
    static uint64_t last_widened = 0, last_kept = 0;
    const uint64_t widened = Common::FaultMonitor::heap_block_widened.load(std::memory_order_relaxed);
    const uint64_t kept = Common::FaultMonitor::heap_block_kept.load(std::memory_order_relaxed);
    const uint64_t frame_widened = widened - last_widened, frame_kept = kept - last_kept;
    last_widened = widened;
    last_kept = kept;
    if (first) return;

    const bool stutter = frames > 120 && ms > 33.0 && ms > 2.0 * average_ms;
    // The average follows the frame rate (about a second), not the stutters themselves.
    average_ms = average_ms == 0.0 ? ms : average_ms + (std::min(ms, 3.0 * average_ms) - average_ms) * 0.05;
    if (!stutter) return;
    // Bursts (a loading zone) stay readable: at most 400 reports a run.
    if (++reported > 400) return;
    std::sort(deltas.begin(), deltas.end(), [](const Delta& a, const Delta& b) { return a.count > b.count; });
    std::string sites;
    for (size_t i = 0; i < deltas.size() && i < 6; ++i) {
        char part[220];
        std::snprintf(part, sizeof(part), " | %s:%d x%llu \"%.60s\"", BaseName(deltas[i].site->file),
                      deltas[i].site->line, static_cast<unsigned long long>(deltas[i].count),
                      deltas[i].site->format ? deltas[i].site->format : "");
        sites += part;
    }
    std::vector<std::pair<uint64_t, RegionFaults>> busiest(regions.begin(), regions.end());
    const auto total = [](const RegionFaults& f) { return f.reads + f.writes; };
    std::sort(busiest.begin(), busiest.end(),
              [&](const auto& a, const auto& b) { return total(a.second) > total(b.second); });
    // Each region as address:r<reads>w<writes>[cp<on the command processor>].
    std::string where;
    for (size_t i = 0; i < busiest.size() && i < 5; ++i) {
        const RegionFaults& f = busiest[i].second;
        char part[80];
        int n = std::snprintf(part, sizeof(part), " %#llx:r%uw%u", static_cast<unsigned long long>(busiest[i].first << 20),
                              f.reads, f.writes);
        if (f.cp) std::snprintf(part + n, sizeof(part) - n, "cp%u", f.cp);
        where += part;
    }
    std::fprintf(stderr,
                 "V3 stutter: frame %.1f ms (avg %.1f) at frame %llu, %llu faults (heap widened %llu, kept %llu;"
                 " MiB:%s)%s\n",
                 ms, average_ms, static_cast<unsigned long long>(frames), static_cast<unsigned long long>(frame_faults),
                 static_cast<unsigned long long>(frame_widened), static_cast<unsigned long long>(frame_kept),
                 where.c_str(), sites.c_str());
    // The command processor's most common write-fault stack this frame (sampled 1 in 16).
    const Stack* top = nullptr;
    for (const auto& [hash, stack] : stacks) {
        if (!top || stack.count > top->count) top = &stack;
    }
    if (top && top->count >= 4) {
        std::string frames;
        for (uint16_t i = 0; i < top->depth; ++i) frames += " " + FrameName(top->frames[i]);
        std::fprintf(stderr, "V3 stutter: CP write-fault stack x%u of %zu:%s\n", top->count * 16, stacks.size(),
                     frames.c_str());
    }
}

} // namespace V3::Stutter
