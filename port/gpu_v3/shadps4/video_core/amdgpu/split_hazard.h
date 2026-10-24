// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <unordered_map>
#include "common/logging/log.h"
#include "common/types.h"

namespace AmdGpu::SplitHazard {

// SHADPS4_SPLIT_HAZARD_STATS=1 (diagnostic, CP thread only): sizes the run-ahead window of a
// front/back split of the command processor. A front thread would do each draw's pipeline lookup,
// which reads guest memory (shader code, tables behind user-data pointers), before the back thread
// has executed earlier CP-side guest writes (WRITE_DATA, DMA_DATA CPU paths, EOP/EOS fences).
// Counts draws whose shader-code page or user-data-pointer page was written by the CP within the
// last N draws. User-data pointers are detected heuristically (an SGPR pair holding a 4-40 GB
// address), so nested tables are missed: this is a lower bound.
inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_SPLIT_HAZARD_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}

struct State {
    std::unordered_map<u64, u64> written; // page -> draw index at the write
    u64 draws{};
    u64 writes{};
    u64 write_bytes{};
    std::array<u64, 5> hits{}; // distance <=1, <=8, <=64, <=512, older
    u64 reads{};
};

inline State& GetState() {
    static State state;
    return state;
}

inline void NoteWrite(u64 address, u64 size) {
    if (!Enabled() || size == 0) {
        return;
    }
    auto& s = GetState();
    ++s.writes;
    s.write_bytes += size;
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        s.written[page] = s.draws;
    }
}

inline void CheckPage(State& s, u64 address) {
    if (address < (4ULL << 30) || address >= (40ULL << 30)) {
        return;
    }
    ++s.reads;
    const auto it = s.written.find(address >> 12);
    if (it == s.written.end()) {
        return;
    }
    const u64 distance = s.draws - it->second;
    const int bucket = distance <= 1 ? 0 : distance <= 8 ? 1 : distance <= 64 ? 2 : distance <= 512 ? 3 : 4;
    ++s.hits[bucket];
}

/// Called per draw with the PS/VS program address words and user data.
template <typename Program>
void NoteDraw(const Program& ps, const Program& vs) {
    if (!Enabled()) {
        return;
    }
    auto& s = GetState();
    ++s.draws;
    for (const Program* p : {&ps, &vs}) {
        CheckPage(s, reinterpret_cast<u64>(p->template Address<const u8*>()));
        const auto& ud = p->user_data;
        for (size_t i = 0; i + 1 < ud.size(); ++i) {
            CheckPage(s, u64(ud[i]) | (u64(ud[i + 1]) << 32));
        }
    }
    if ((s.draws & 0x7ffff) == 0) {
        const u64 any = s.hits[0] + s.hits[1] + s.hits[2] + s.hits[3] + s.hits[4];
        LOG_INFO(Render, "SPLIT_HAZARD: draws {} cp_writes {} ({} KB) pointer/code pages checked {} "
                 "written-by-CP hits {} (draws back <=1 {}, <=8 {}, <=64 {}, <=512 {}, older {})",
                 s.draws, s.writes, s.write_bytes / 1024, s.reads, any, s.hits[0], s.hits[1],
                 s.hits[2], s.hits[3], s.hits[4]);
        s.writes = s.write_bytes = s.reads = 0;
        s.hits = {};
        if (s.written.size() > (1u << 20)) {
            s.written.clear();
        }
    }
}

} // namespace AmdGpu::SplitHazard
