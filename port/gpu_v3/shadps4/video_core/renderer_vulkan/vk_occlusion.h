// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

/**
 * Real occlusion queries for the guest's ZPASS_DONE / PixelPipeStatDump protocol
 * (SHADPS4_OCCLUSION_QUERIES=1, default off; 2026-10-02).
 *
 * The guest brackets a few draws with two dumps of the Z-pass counters: one into the "begin" u64 of
 * every counter pair of a result block and one into the "end" u64 (block base + 8), and later reads
 * end - begin on the CPU (bit 63 = value valid). Upstream answers every dump with an always-rising
 * fake count, so every query reads "visible" (inFAMOUS Second Son: headlight glows show through
 * walls; measured 2026-10-02: 8 queries per frame, fixed result blocks 0x100 apart, zeroed by the
 * guest before every begin; the glows vanish when the count is frozen).
 *
 * Here every draw between a begin dump and its end dump is wrapped in one Vulkan occlusion query
 * (begin and end recorded right around the draw, so both are in the draw's render pass instance and
 * command buffer). The result is NOT waited for: at the end dump the command processor publishes
 * the last RESOLVED count of the same result block (one or more frames old) at the same point in
 * the command stream where the fake was written, so guest-visible timing is unchanged. Results are
 * collected later on the same thread (Poll: when the tick of the query's commands is complete).
 *
 * Results are kept per QUERY, not per result block (2026-10-18, the headlamp flicker): the game
 * runs 12 queries per frame on 8 blocks - blocks 0x000-0x300 twice (two different draws), then
 * 0x400-0x700 - and a per-block "last resolved count" handed each of the two queries whichever of
 * them the GPU had finished last (published 0 <-> nonzero 30-180 times per 4 s on three blocks;
 * timing-dependent, so GPU load and SHADPS4_PARTICLE_LATE_READ changed how often). A query is
 * identified by its block and its first draw's signature (pipeline, depth and colour target), so
 * each one publishes its own previous result. SHADPS4_OCCLUSION_PER_BLOCK=1 restores per block.
 *
 * Threading: every method runs on the command processor thread that executes draws (the back
 * thread with SHADPS4_CP_SPLIT). The recorder thread only sees the pool handle and slot numbers
 * captured in recorded commands.
 */
class OcclusionQueries {
public:
    static constexpr u32 PoolSize = 4096;
    static constexpr u32 NoSlot = ~0u;
    static constexpr u32 MaxDrawsPerQuery = 64;
    static constexpr u32 MaxFailedPolls = 256;
    static constexpr u64 ValidBit = 0x8000000000000000ULL;

    static bool Requested() {
        static const bool on = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_QUERIES");
            return value && value[0] == '1';
        }();
        return on;
    }

    /// SHADPS4_OCCLUSION_PER_BLOCK=1: results keyed by result block only (the pre-10-18 behaviour).
    static bool PerBlock() {
        static const bool on = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_PER_BLOCK");
            return value && value[0] == '1';
        }();
        return on;
    }

    /// SHADPS4_OCCLUSION_SCALE=<n> (diagnostic): published counts are multiplied by n, to tell a
    /// guest that tests "any sample" from one that uses the magnitude.
    static u64 Scale() {
        static const u64 scale = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_SCALE");
            const long long n = value ? std::atoll(value) : 1;
            return static_cast<u64>(n > 0 ? n : 1);
        }();
        return scale;
    }

    OcclusionQueries(const Instance& instance_, Scheduler& scheduler_)
        : instance{instance_}, scheduler{scheduler_} {
        pool = instance.GetDevice()
                   .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                       .queryType = vk::QueryType::eOcclusion,
                       .queryCount = PoolSize,
                   })
                   .value;
        instance.GetDevice().resetQueryPool(*pool, 0, PoolSize);
        if (instance.IsOcclusionQueryPreciseSupported()) {
            flags = vk::QueryControlFlagBits::ePrecise;
        }
        LOG_INFO(Render_Vulkan, "OcclusionQ: real queries enabled, pool={} precise={}", PoolSize,
                 instance.IsOcclusionQueryPreciseSupported());
    }

    /// A query is open: draws executed now belong to it.
    [[nodiscard]] bool Open() const noexcept {
        return open;
    }

    [[nodiscard]] vk::QueryPool Pool() const noexcept {
        return *pool;
    }

    [[nodiscard]] vk::QueryControlFlags Flags() const noexcept {
        return flags;
    }

    /// One counter dump of the guest at `results` (`pairs` counter pairs, 16 bytes apart). Decides
    /// begin or end, writes the guest-visible values.
    void Dump(u64* results, s32 pairs) {
        const u64 addr = reinterpret_cast<u64>(results);
        Poll();
        // SHADPS4_OCCLUSION_TRACE=1 (2026-10-18, the headlamp flicker): the order of the guest's
        // dumps for 400 dumps after the 3,000th (block, begin / end, presented frame, the last
        // resolved count of the block).
        // =N: start after the Nth dump (1 = 3,000).
        static const u64 trace = [] {
            const char* v = std::getenv("SHADPS4_OCCLUSION_TRACE");
            const u64 n = v ? std::strtoull(v, nullptr, 10) : 0;
            return n == 1 ? u64{3000} : n;
        }();
        if (trace != 0) [[unlikely]] {
            static u64 n = 0;
            if (++n > trace && n <= trace + 400) {
                const u64 block = addr & ~u64{8};
                const auto it = last_count_block.find(block);
                LOG_INFO(Render_Vulkan, "OcclusionTrace: #{} {:#x} {} frame {} last {} sig {:#x}", n,
                         block, (addr & 8) ? "end" : "begin",
                         Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed),
                         it != last_count_block.end() ? static_cast<s64>(it->second) : -1,
                         (addr & 8) ? cur.sig : u64{0});
            }
        }
        if (open && addr == cur.base + 8) {
            // End dump: close the query; publish the last resolved count of the same query.
            const auto key = KeyOf(cur);
            Close();
            u64 count = 0;
            if (const auto it = last_count.find(key); it != last_count.end()) {
                count = it->second * Scale();
            } else {
                ++stats.unresolved_at_end;
            }
            // Per block (2026-10-18, the headlamp flicker): the begin word is ours (ValidBit) until
            // the guest zeroes the block for its next query - zero here means the guest moved on
            // before this end dump, so it read this block while it held no valid result.
            BlockStats& bs = block_stats[key];
            ++bs.ends;
            bs.late += results[-1] == 0 ? 1 : 0;
            if (bs.ends > 1 && (count == 0) != (bs.last_published == 0)) {
                ++bs.flips;
            }
            bs.zero += count == 0 ? 1 : 0;
            bs.last_published = count;
            // The per-render-backend counters are summed by the guest: all of it goes in pair 0.
            results[0] = (count & ~ValidBit) | ValidBit;
            for (s32 i = 1; i < pairs; ++i) {
                results[i * 2] = ValidBit;
            }
            return;
        }
        if (addr & 8) {
            // An end dump whose begin is not the open query (result blocks are 16-byte aligned,
            // the end u64 is the odd one): publish what is known and leave the state alone.
            ++stats.orphan_end;
            u64 count = 0;
            if (const auto it = last_count_block.find(addr - 8); it != last_count_block.end()) {
                count = it->second;
            }
            results[0] = (count & ~ValidBit) | ValidBit;
            for (s32 i = 1; i < pairs; ++i) {
                results[i * 2] = ValidBit;
            }
            return;
        }
        if (open) {
            // A second begin while a query is open: the open one ends here.
            ++stats.begin_while_open;
            Close();
        }
        open = true;
        cur = Query{};
        cur.base = addr;
        cur.frame = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        for (s32 i = 0; i < pairs; ++i) {
            results[i * 2] = ValidBit;
        }
    }

    /// Slot for the next draw of the open query, or NoSlot (the draw is then not counted).
    /// Call only while Open().
    [[nodiscard]] u32 DrawSlot(u64 draw_sig) {
        if (cur.draws++ == 0) {
            cur.sig = draw_sig; // the query's identity (with its block): its first draw
        }
        if (cur.slots.size() >= MaxDrawsPerQuery) {
            ++stats.capped_draws;
            cur.incomplete = true;
            return NoSlot;
        }
        if (in_flight >= PoolSize - 1) {
            ++stats.exhausted;
            cur.incomplete = true;
            return NoSlot;
        }
        const u32 slot = next;
        next = (next + 1) % PoolSize;
        ++in_flight;
        // Slots are handed out in ring order and released in the same order (Poll resolves the
        // oldest query first), so this slot's previous use has been read back.
        instance.GetDevice().resetQueryPool(*pool, slot, 1);
        cur.slots.push_back(slot);
        return slot;
    }

private:
    struct Query {
        u64 base{};
        u64 sig{};   ///< the first draw's signature (KeyOf)
        u32 draws{}; ///< draws seen while open (counted or not)
        u64 tick{};
        u64 frame{};
        /// A draw of the query was not counted: its sum is a lower bound, not published.
        bool incomplete{};
        u32 failed_polls{};
        boost::container::small_vector<u32, 4> slots;
    };

    void Close() {
        open = false;
        // The query's draw commands belong to the command buffer that the next submit signals with
        // this tick (or to an earlier one).
        cur.tick = scheduler.CurrentTick();
        ++stats.queries;
        stats.draws += cur.slots.size();
        if (cur.slots.empty()) {
            ++stats.empty;
        }
        pending.push_back(std::move(cur));
        cur = Query{};
    }

    void Poll() {
        while (!pending.empty()) {
            Query& query = pending.front();
            u64 sum = 0;
            if (!query.slots.empty()) {
                if (!scheduler.IsFreeMain(query.tick)) {
                    break;
                }
                bool ready = true;
                for (const u32 slot : query.slots) {
                    u64 value = 0;
                    const auto result = instance.GetDevice().getQueryPoolResults(
                        *pool, slot, 1, sizeof(value), &value, sizeof(value),
                        vk::QueryResultFlagBits::e64);
                    if (result != vk::Result::eSuccess) {
                        ready = false;
                        break;
                    }
                    sum += value;
                }
                if (!ready && ++query.failed_polls < MaxFailedPolls) {
                    ++stats.not_ready;
                    break;
                }
                in_flight -= static_cast<u32>(query.slots.size());
                if (!ready) {
                    // The tick is complete and the result still is not there (a command buffer
                    // that never ran): give the slots back, keep the old count.
                    ++stats.dropped;
                    pending.pop_front();
                    continue;
                }
            }
            if (query.incomplete) {
                ++stats.incomplete;
                pending.pop_front();
                continue;
            }
            last_count[KeyOf(query)] = sum;
            last_count_block[query.base] = sum;
            ++stats.resolved;
            if (sum == 0) {
                ++stats.zero;
            }
            const u64 latency =
                Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed) -
                query.frame;
            stats.latency_sum += latency;
            stats.latency_max = std::max(stats.latency_max, latency);
            pending.pop_front();
        }
        if (stats.resolved >= 4096) {
            LOG_INFO(Render_Vulkan,
                     "OcclusionQ: real: queries={} draws={} empty={} resolved={} zero={} "
                     "latency_frames avg={:.2f} max={} unresolved_at_end={} not_ready={} "
                     "exhausted={} capped_draws={} begin_while_open={} orphan_end={} incomplete={} "
                     "dropped={} pending={} blocks={}",
                     stats.queries, stats.draws, stats.empty, stats.resolved, stats.zero,
                     static_cast<double>(stats.latency_sum) / static_cast<double>(stats.resolved),
                     stats.latency_max, stats.unresolved_at_end, stats.not_ready, stats.exhausted,
                     stats.capped_draws, stats.begin_while_open, stats.orphan_end,
                     stats.incomplete, stats.dropped, pending.size(),
                     last_count.size());
            std::string counts;
            for (const auto& [base, count] : last_count_block) {
                counts += fmt::format(" {:#x}:{}", base, count);
            }
            counts += fmt::format(" | queries tracked {}", last_count.size());
            LOG_INFO(Render_Vulkan, "OcclusionQ: real: last counts (block:samples){}", counts);
            std::string blocks;
            for (const auto& [key, b] : block_stats) {
                blocks += fmt::format(" {:#x}/{:04x}: ends {} late {} zero {} flips {} |",
                                      key.first, key.second & 0xffff, b.ends, b.late, b.zero,
                                      b.flips);
            }
            LOG_INFO(Render_Vulkan, "OcclusionQ: real: per query (block/signature; late = the guest "
                                    "re-zeroed the block before the end dump; flips = published "
                                    "0 <-> nonzero){}",
                     blocks);
            block_stats.clear();
            stats = Stats{};
        }
    }

    struct Stats {
        u64 queries{};
        u64 draws{};
        u64 empty{};
        u64 resolved{};
        u64 zero{};
        u64 latency_sum{};
        u64 latency_max{};
        u64 unresolved_at_end{};
        u64 not_ready{};
        u64 exhausted{};
        u64 capped_draws{};
        u64 begin_while_open{};
        u64 orphan_end{};
        u64 incomplete{};
        u64 dropped{};
    };

    struct BlockStats {
        u64 ends{};
        u64 late{};
        u64 zero{};
        u64 flips{};
        u64 last_published{};
    };
    std::map<std::pair<u64, u64>, BlockStats> block_stats; ///< per query (KeyOf)

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueQueryPool pool;
    vk::QueryControlFlags flags{};
    bool open{};
    Query cur;
    std::deque<Query> pending;
    /// Last resolved count per query (block, first draw's signature).
    std::map<std::pair<u64, u64>, u64> last_count;
    /// ... and per block (orphan ends, the log).
    std::unordered_map<u64, u64> last_count_block;

    static std::pair<u64, u64> KeyOf(const Query& query) {
        return {query.base, PerBlock() ? u64{0} : query.sig};
    }
    u32 next{};
    u32 in_flight{};
    Stats stats;
};

} // namespace Vulkan
