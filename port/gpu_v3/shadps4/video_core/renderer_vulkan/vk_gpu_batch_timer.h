// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <source_location>
#include <string>
#include <vector>
#include <thread>

#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

/// Per-window counters of GPU pipeline breaks, bumped on the CP (diagnostic, same flag).
struct GpuBreakCounters {
    std::atomic<u64> render_begins{0};
    std::atomic<u64> buffer_barrier_calls{0};
    std::atomic<u64> buffer_barriers{0};
    std::atomic<u64> image_barrier_calls{0};
    std::atomic<u64> image_barriers{0};
};
inline GpuBreakCounters& GpuBreaks() {
    static GpuBreakCounters counters;
    return counters;
}

/// Per source location (same flag, diagnostic): render-pass ends, how many of those were followed
/// by a BeginRendering with the identical state (a pure restart), and image barriers by kind.
class BreakSites {
public:
    /// Hazard kinds; the same four again (+4) when the barrier also changes layout.
    enum Kind : u32 { WriteToRead, WriteToWrite, ReadToWrite, ReadToRead, NumKinds = 8 };

    u32 Site(const std::source_location& loc) {
        std::scoped_lock lk{mutex};
        return SiteLocked(loc);
    }
    void End(u32 site) {
        std::scoped_lock lk{mutex};
        ++entries[site].ends;
    }
    void Resumed(u32 site) {
        std::scoped_lock lk{mutex};
        ++entries[site].resumed;
    }
    void Image(const std::source_location& loc, vk::ImageLayout old_layout,
               vk::ImageLayout new_layout, vk::AccessFlags2 src, vk::AccessFlags2 dst) {
        constexpr auto writes =
            vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eMemoryWrite |
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        const bool src_w = static_cast<bool>(src & writes);
        const bool dst_w = static_cast<bool>(dst & writes);
        const u32 kind = (src_w ? (dst_w ? WriteToWrite : WriteToRead)
                          : dst_w ? ReadToWrite
                                  : ReadToRead) +
                         (old_layout != new_layout ? 4u : 0u);
        std::scoped_lock lk{mutex};
        ++entries[SiteLocked(loc)].img[kind];
    }
    /// Top sites by activity since the last call, then clears the counts.
    std::string Report() {
        std::scoped_lock lk{mutex};
        std::vector<const Entry*> order;
        for (const auto& e : entries) {
            order.push_back(&e);
        }
        const auto weight = [](const Entry* e) {
            u64 w = e->ends;
            for (const u64 v : e->img) {
                w += v;
            }
            return w;
        };
        std::ranges::sort(order,
                          [&](const Entry* a, const Entry* b) { return weight(a) > weight(b); });
        std::string out;
        for (size_t i = 0; i < order.size() && i < 14; ++i) {
            const Entry* e = order[i];
            if (weight(e) == 0) {
                break;
            }
            const char* base = e->file;
            for (const char* p = e->file; *p; ++p) {
                if (*p == '/' || *p == '\\') {
                    base = p + 1;
                }
            }
            out += fmt::format(" | {}:{} ends={} resumed={} same[w2r/w2w/r2w/r2r]={}/{}/{}/{} "
                               "relayout[w2r/w2w/r2w/r2r]={}/{}/{}/{}",
                               base, e->line, e->ends, e->resumed, e->img[0], e->img[1],
                               e->img[2], e->img[3], e->img[4], e->img[5], e->img[6], e->img[7]);
        }
        for (auto& e : entries) {
            e.ends = e.resumed = 0;
            e.img = {};
        }
        return out;
    }

private:
    struct Entry {
        const char* file;
        u32 line;
        u64 ends = 0;
        u64 resumed = 0;
        std::array<u64, NumKinds> img{};
    };
    u32 SiteLocked(const std::source_location& loc) {
        for (u32 i = 0; i < entries.size(); ++i) {
            if (entries[i].line == loc.line() &&
                std::strcmp(entries[i].file, loc.file_name()) == 0) {
                return i;
            }
        }
        entries.push_back(Entry{loc.file_name(), loc.line()});
        return static_cast<u32>(entries.size() - 1);
    }
    std::mutex mutex;
    std::vector<Entry> entries;
};
inline BreakSites& GpuBreakSites() {
    static BreakSites sites;
    return sites;
}

/**
 * SHADPS4_GPU_BATCH_TIMING=1 (with SHADPS4_ASYNC_SUBMIT=1): GPU timestamps at the start and end
 * of every main-queue batch, written by the recording worker. A reader thread waits for each
 * batch to complete and logs, every 10 s: GPU busy share of wall time, mean batch duration,
 * idle gaps between consecutive batches (other submissions on the queue fall into the gaps) and
 * GPU time per draw. Diagnostic only.
 */
class GpuBatchTimer {
public:
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_GPU_BATCH_TIMING");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }

    GpuBatchTimer(const Instance& instance_, MasterSemaphore& semaphore_)
        : instance{instance_}, semaphore{semaphore_} {
        const auto device = instance.GetDevice();
        auto [result, pool] = device.createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = Slots * 2,
        });
        ASSERT(result == vk::Result::eSuccess);
        query_pool = pool;
        device.resetQueryPool(query_pool, 0, Slots * 2);
        period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
        reader = std::jthread([this](std::stop_token stop) { ReadLoop(stop); });
    }

    ~GpuBatchTimer() {
        reader.request_stop();
        if (reader.joinable()) {
            reader.join();
        }
        instance.GetDevice().destroyQueryPool(query_pool);
    }

    /// Recording worker, at the start of a batch's command buffer.
    void Begin(vk::CommandBuffer cmdbuf) {
        const u64 slot = write_seq % Slots;
        cmdbuf.resetQueryPool(query_pool, u32(slot * 2), 2);
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, query_pool, u32(slot * 2));
        began = true;
    }

    /// Recording worker, just before ending a batch submitted with `signal_value`.
    void End(vk::CommandBuffer cmdbuf, u64 signal_value, u32 draws, u32 dispatches,
             bool asc_wait = false) {
        if (!began) {
            return;
        }
        const u64 slot = write_seq % Slots;
        batch_asc_wait[slot] = asc_wait;
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, query_pool,
                              u32(slot * 2 + 1));
        signal_values[slot] = signal_value;
        batch_draws[slot] = draws;
        batch_dispatches[slot] = dispatches;
        submit_ns[slot] = NowNs();
        written.store(++write_seq, std::memory_order_release);
        began = false;
    }

    static u64 NowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    /// Readback batch mix (with SHADPS4_READBACK_LATENCY): one timed readback copy. The main-queue
    /// batches `done_tick + 1 .. last_ahead` were not known complete when the copy was flushed;
    /// `own_tick` is the batch holding the copy (0 when it was a separate queue-order submission);
    /// `producer` is the newest batch that GPU-wrote the copied bytes (0 unknown).
    /// By faulting guest function (SHADPS4_BATCH_MIX_SS): control-block reader, mesh snap,
    /// exposure sum, anything else, and faults with no guest RIP (CP / offloaded paths).
    enum MixClass : u32 {
        MixArena, MixPool, MixOther, MixExposure, MixMemset, MixWriteFault, NumMixClasses
    };
    struct MixRequest {
        u64 done_tick;
        u64 last_ahead;
        u64 own_tick;
        u64 producer;
        u64 flush_ns;
        u32 cls;
        u64 label_tick; ///< Newest graphics label batch at the flush (0 unknown).
    };
    void RequestMix(const MixRequest& request) {
        std::scoped_lock lk{mix_mutex};
        if (mix_pending.size() < 4096) {
            mix_pending.push_back(request);
        }
    }

private:
    static constexpr u64 Slots = 1024;
    static constexpr u64 TickSlots = 8192;

    /// Per completed batch, indexed by its signal value (reader thread only).
    struct TickInfo {
        u64 tick = 0;
        u64 duration_ns = 0;
        u64 submit_ns = 0;
        u32 draws = 0;
        u32 dispatches = 0;
    };

    /// Batch kinds: 0 graphics only, 1 dispatches only, 2 both, 3 neither (copies, uploads).
    static u32 Kind(const TickInfo& t) {
        if (t.draws != 0) {
            return t.dispatches != 0 ? 2 : 0;
        }
        return t.dispatches != 0 ? 1 : 3;
    }

    struct MixAccum {
        u64 jobs = 0, own_jobs = 0, ahead_batches = 0, untimed = 0;
        std::array<u64, 4> ahead_ns{};    ///< GPU time of batches ahead, by kind.
        std::array<u64, 4> ahead_count{}; ///< Batches ahead, by kind.
        u64 before_producer_ns = 0;       ///< Ahead and older than the producer batch.
        u64 fenced_ns = 0;                ///< Ahead, at or before the newest graphics label.
        u64 tail_ns = 0;                  ///< Ahead, after it (overtakable by an ASC stream).
        u64 producer_ahead = 0;           ///< Producer among the batches ahead.
        u64 producer_own = 0;             ///< Producer is the copy's own batch.
        u64 producer_done = 0;            ///< Producer known complete at flush.
        u64 age_weighted = 0;             ///< Sum of (flush - submit) us x GPU ns, batches ahead.
        u64 own_ns = 0, own_draws = 0, own_dispatches = 0, own_timed = 0;
        std::array<u64, 8> ahead_hist{};  ///< Jobs by batches ahead: 0,1,2,3,4-7,8-15,16-31,32+.
    };

    void ResolveMix(u64 processed_tick) {
        std::vector<MixRequest> ready;
        {
            std::scoped_lock lk{mix_mutex};
            for (size_t i = 0; i < mix_pending.size();) {
                const MixRequest& r = mix_pending[i];
                if (std::max(r.own_tick, r.last_ahead) <= processed_tick) {
                    ready.push_back(r);
                    mix_pending[i] = mix_pending.back();
                    mix_pending.pop_back();
                } else {
                    ++i;
                }
            }
        }
        for (const MixRequest& r : ready) {
            MixAccum& a = mix[r.cls];
            ++a.jobs;
            u64 n = 0;
            // At most 256 batches; an older known-done value only means the poll was stale.
            const u64 first = std::max(r.done_tick + 1, r.last_ahead > 255 ? r.last_ahead - 255 : 1);
            for (u64 t = first; t <= r.last_ahead; ++t) {
                const TickInfo& info = ticks[t % TickSlots];
                ++n;
                if (info.tick != t) {
                    ++a.untimed;
                    continue;
                }
                const u32 k = Kind(info);
                a.ahead_ns[k] += info.duration_ns;
                ++a.ahead_count[k];
                if (r.producer != 0 && t < r.producer) {
                    a.before_producer_ns += info.duration_ns;
                }
                if (r.label_tick != 0) {
                    (t <= r.label_tick ? a.fenced_ns : a.tail_ns) += info.duration_ns;
                }
                if (r.flush_ns > info.submit_ns) {
                    a.age_weighted += (r.flush_ns - info.submit_ns) / 1000 * info.duration_ns;
                }
            }
            a.ahead_batches += n;
            ++a.ahead_hist[n < 4 ? n : n < 8 ? 4 : n < 16 ? 5 : n < 32 ? 6 : 7];
            if (r.producer != 0) {
                if (r.producer <= r.done_tick) {
                    ++a.producer_done;
                } else if (r.own_tick != 0 && r.producer >= r.own_tick) {
                    ++a.producer_own;
                } else {
                    ++a.producer_ahead;
                }
            }
            if (r.own_tick != 0) {
                ++a.own_jobs;
                const TickInfo& own = ticks[r.own_tick % TickSlots];
                if (own.tick == r.own_tick) {
                    ++a.own_timed;
                    a.own_ns += own.duration_ns;
                    a.own_draws += own.draws;
                    a.own_dispatches += own.dispatches;
                }
            }
        }
    }

    void LogMix() {
        static constexpr std::array<const char*, NumMixClasses> names = {
            "ctrl_3f3ed0", "snap_3fed00", "other", "exposure_171610", "hle_memset",
            "write_fault"};
        for (u32 c = 0; c < NumMixClasses; ++c) {
            MixAccum& a = mix[c];
            if (a.jobs == 0) {
                continue;
            }
            const double j = double(a.jobs);
            u64 ahead_total = 0;
            for (const u64 v : a.ahead_ns) {
                ahead_total += v;
            }
            const double ot = a.own_timed ? double(a.own_timed) : 1.0;
            LOG_INFO(Render_Vulkan,
                     "BatchMix[{}]: jobs={} own_batch_jobs={} ahead_batches={:.2f} untimed={:.2f} "
                     "ahead_gpu_us={:.1f} gfx_only_us={:.1f} disp_only_us={:.1f} mixed_us={:.1f} "
                     "neither_us={:.1f} n_gfx/disp/mixed/neither={:.2f}/{:.2f}/{:.2f}/{:.2f} "
                     "before_producer_us={:.1f} producer_done/ahead/own={}/{}/{} "
                     "ahead_age_ms_gpu_weighted={:.2f} own_us={:.1f} own_draws={:.1f} "
                     "own_disp={:.1f} hist[0,1,2,3,4-7,8-15,16-31,32+]={}/{}/{}/{}/{}/{}/{}/{} "
                     "fenced_us={:.1f} tail_us={:.1f}",
                     names[c], a.jobs, a.own_jobs, double(a.ahead_batches) / j,
                     double(a.untimed) / j, double(ahead_total) / j / 1000.0,
                     double(a.ahead_ns[0]) / j / 1000.0, double(a.ahead_ns[1]) / j / 1000.0,
                     double(a.ahead_ns[2]) / j / 1000.0, double(a.ahead_ns[3]) / j / 1000.0,
                     double(a.ahead_count[0]) / j, double(a.ahead_count[1]) / j,
                     double(a.ahead_count[2]) / j, double(a.ahead_count[3]) / j,
                     double(a.before_producer_ns) / j / 1000.0, a.producer_done,
                     a.producer_ahead, a.producer_own,
                     ahead_total ? double(a.age_weighted) / double(ahead_total) / 1000.0 : 0.0,
                     double(a.own_ns) / ot / 1000.0, double(a.own_draws) / ot,
                     double(a.own_dispatches) / ot, a.ahead_hist[0], a.ahead_hist[1],
                     a.ahead_hist[2], a.ahead_hist[3], a.ahead_hist[4], a.ahead_hist[5],
                     a.ahead_hist[6], a.ahead_hist[7], double(a.fenced_ns) / j / 1000.0,
                     double(a.tail_ns) / j / 1000.0);
            a = MixAccum{};
        }
    }

    void ReadLoop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:GpuBatchTimer");
        u64 read_seq = 0;
        u64 prev_end = 0;
        u64 busy = 0, gaps = 0, gap_count = 0, batches = 0, draws = 0, max_batch = 0;
        // ASC I1 split (design notes/asc-i1-relax-design-2026-09-28.md step 0b): the gap before a
        // batch that waited an ASC value (I1) vs before one that did not.
        u64 gaps_asc = 0, gap_count_asc = 0, batches_asc = 0;
        auto window_start = std::chrono::steady_clock::now();
        while (!stop.stop_requested()) {
            if (read_seq >= written.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const u64 slot = read_seq % Slots;
            const u64 value = signal_values[slot];
            while (!stop.stop_requested() && semaphore.QueryCounter() < value) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            std::array<u64, 2> ts{};
            const auto result = instance.GetDevice().getQueryPoolResults(
                query_pool, u32(slot * 2), 2, sizeof(ts), ts.data(), sizeof(u64),
                vk::QueryResultFlagBits::e64);
            ++read_seq;
            if (result != vk::Result::eSuccess || ts[1] < ts[0]) {
                continue;
            }
            const u64 start = u64(double(ts[0]) * period_ns);
            const u64 end = u64(double(ts[1]) * period_ns);
            const u64 duration = end - start;
            ticks[value % TickSlots] = TickInfo{value, duration, submit_ns[slot],
                                                batch_draws[slot], batch_dispatches[slot]};
            ResolveMix(value);
            busy += duration;
            max_batch = std::max(max_batch, duration);
            batches_asc += batch_asc_wait[slot];
            if (prev_end != 0 && start > prev_end && start - prev_end < 50'000'000) {
                gaps += start - prev_end;
                ++gap_count;
                if (batch_asc_wait[slot]) {
                    gaps_asc += start - prev_end;
                    ++gap_count_asc;
                }
            }
            prev_end = end;
            ++batches;
            draws += batch_draws[slot];
            const auto now = std::chrono::steady_clock::now();
            const double wall_ns =
                double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - window_start)
                           .count());
            if (wall_ns >= 10e9) {
                // This port: a warning, so it shows at renderer v3's default log level.
                LOG_WARNING(Render_Vulkan,
                         "GpuBatchTiming: batches={} busy_pct={:.1f} batch_us={:.1f} "
                         "max_batch_us={:.1f} gap_us={:.1f} draws_per_batch={:.1f} "
                         "gpu_us_per_draw={:.3f} render_begins={} buf_barrier_calls={} buf_barriers={} "
                         "img_barrier_calls={} img_barriers={}",
                         batches, 100.0 * double(busy) / wall_ns,
                         double(busy) / double(batches) / 1000.0, double(max_batch) / 1000.0,
                         gap_count ? double(gaps) / double(gap_count) / 1000.0 : 0.0,
                         double(draws) / double(batches),
                         draws ? double(busy) / double(draws) / 1000.0 : 0.0,
                         GpuBreaks().render_begins.exchange(0),
                         GpuBreaks().buffer_barrier_calls.exchange(0),
                         GpuBreaks().buffer_barriers.exchange(0),
                         GpuBreaks().image_barrier_calls.exchange(0),
                         GpuBreaks().image_barriers.exchange(0));
                LOG_INFO(Render_Vulkan, "GpuBreakSites:{}", GpuBreakSites().Report());
                if (batches_asc != 0) {
                    LOG_INFO(Render_Vulkan,
                             "GpuBatchI1: batches={} with_asc_wait={} gap_ms_asc_wait={:.1f} "
                             "gap_ms_other={:.1f} gap_us_asc_wait={:.1f} gap_us_other={:.1f} "
                             "(per 10 s window)",
                             batches, batches_asc, double(gaps_asc) / 1e6,
                             double(gaps - gaps_asc) / 1e6,
                             gap_count_asc ? double(gaps_asc) / double(gap_count_asc) / 1000.0 : 0.0,
                             gap_count > gap_count_asc
                                 ? double(gaps - gaps_asc) / double(gap_count - gap_count_asc) /
                                       1000.0
                                 : 0.0);
                }
                gaps_asc = gap_count_asc = batches_asc = 0;
                LogMix();
                busy = gaps = gap_count = batches = draws = max_batch = 0;
                window_start = now;
            }
        }
    }

    std::array<u8, Slots> batch_asc_wait{};
    const Instance& instance;
    MasterSemaphore& semaphore;
    vk::QueryPool query_pool;
    double period_ns = 1.0;
    std::array<u64, Slots> signal_values{};
    std::array<u32, Slots> batch_draws{};
    std::array<u32, Slots> batch_dispatches{};
    std::array<u64, Slots> submit_ns{};
    std::array<TickInfo, TickSlots> ticks{};
    std::array<MixAccum, NumMixClasses> mix{};
    std::mutex mix_mutex;
    std::vector<MixRequest> mix_pending;
    u64 write_seq = 0;
    bool began = false;
    std::atomic<u64> written{0};
    std::jthread reader;
};

} // namespace Vulkan
