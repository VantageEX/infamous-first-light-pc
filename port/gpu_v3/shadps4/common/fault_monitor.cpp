// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#if defined(__LLVM_INSTR_PROFILE_GENERATE)
#include <profile/instr_prof_interface.h>
#endif

#include <immintrin.h>
#include <cstring>
#include <cwchar>
#include <map>
#include <tuple>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif
#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "video_core/buffer_cache/submit_epoch.h"

namespace Common::FaultMonitor {

std::atomic<u64> guest_read_faults{0};
std::atomic<u64> guest_read_unhandled{0};
std::atomic<u64> guest_read_ns{0};
std::atomic<u64> guest_write_faults{0};
std::atomic<u64> guest_write_unhandled{0};
std::atomic<u64> guest_write_ns{0};
std::atomic<u64> jit_patch_faults{0};
std::atomic<u64> jit_patch_ns{0};
std::atomic<u64> presented_frames{0};
std::atomic<u64> sync_upload_bytes{0};
std::atomic<u64> sync_upload_calls{0};
std::atomic<u64> frame_last_ns{0};
std::atomic<u64> frame_max_ns{0};
std::atomic<u64> frame_over_10ms{0};
std::atomic<u64> frame_over_17ms{0};
std::atomic<u64> frame_over_33ms{0};
std::atomic<u64> frame_slow_ns{0};
std::atomic<u64> front_idle_ns{0};
std::atomic<u64> front_wait_ticks{0};
std::atomic<u64> back_idle_ticks{0};
std::atomic<u64> back_idle_total_ticks{0};
std::atomic<u64> rf_guest_ns{0};
std::atomic<u64> range_fragments{0};
std::atomic<u64> range_adds{0};
std::atomic<u64> range_subtracts{0};
std::atomic<u64> gpu_submits{0};
std::atomic<u64> gpu_waits{0};
std::atomic<u64> gpu_wait_ns{0};
std::atomic<u64> download_calls{0};
std::atomic<u64> download_bytes{0};
std::atomic<u64> heap_block_widened{0};
std::atomic<u64> heap_block_kept{0};
std::atomic<u64> invalidate_settles{0};
std::atomic<u64> invalidate_settle_ns{0};
std::atomic<u64> draw_count{0};
std::atomic<u64> draw_pipeline_ns{0};
std::atomic<u64> draw_bindres_ns{0};
std::atomic<u64> draw_beginrender_ns{0};
std::atomic<u64> draw_finddepth_ns{0};
std::atomic<u64> draw_bindbuf_ns{0};
std::atomic<u64> draw_bindtex_ns{0};
std::atomic<u64> bindbuf_find_ns{0};
std::atomic<u64> bindbuf_obtain_ns{0};
std::atomic<u64> obtain_query_ns{0};
std::atomic<u64> obtain_stream_ns{0};
std::atomic<u64> obtain_stream_calls{0};
std::atomic<u64> obtain_stream_bytes{0};
std::atomic<u64> obtain_sync_ns{0};
std::atomic<u64> obtain_sync_calls{0};
std::atomic<u64> upload_scan_ns{0};
std::atomic<u64> upload_copy_ns{0};
std::atomic<u64> upload_mark_ns{0};
std::atomic<u64> upload_written_calls{0};
std::atomic<u64> protect_write_calls{0};
std::atomic<u64> protect_write_ns{0};
std::atomic<u64> protect_read_calls{0};
std::atomic<u64> protect_read_ns{0};
std::atomic<u64> bindtex_findimage_ns{0};
std::atomic<u64> bindtex_view_ns{0};
std::atomic<u64> bindtex_sampler_ns{0};
std::atomic<u64> barrier_calls{0};
std::atomic<u64> barrier_partial_calls{0};
std::atomic<u64> barrier_subres_iters{0};
std::atomic<u64> barrier_emitted{0};
std::atomic<u64> bindtex_findtex_ns{0};
std::atomic<u64> bindtex_transit_ns{0};
std::atomic<u64> cp_wait_ns{0};
std::atomic<u64> cp_active_ns{0};
std::atomic<u64> dispatch_count{0};
std::atomic<u64> cp_read_faults{0};
std::atomic<u64> cp_read_ns{0};
std::atomic<u64> cp_write_faults{0};
std::atomic<u64> cp_write_ns{0};
std::atomic<u64> upload_calls{0};
std::atomic<u64> upload_empty_calls{0};
std::atomic<u64> upload_managers{0};
std::atomic<u64> upload_ranges{0};
std::atomic<u64> upload_skipped{0};
std::atomic<u64> upload_bytes{0};
std::atomic<u64> upload_fastpath{0};
std::atomic<u64> upload_memohit{0};
std::atomic<u64> dispatch_ns{0};
std::atomic<u64> draw_pre_ns{0};
std::atomic<u64> draw_vtxbuf_ns{0};
std::atomic<u64> draw_descwrite_ns{0};
std::atomic<u64> draw_dynstate_ns{0};
std::atomic<u64> draw_schedbegin_ns{0};
std::atomic<u64> draw_emit_ns{0};
std::atomic<u64> stream_map_ns{0};
std::atomic<u64> stream_valid_ns{0};
std::atomic<u64> stream_copy_ns{0};
std::atomic<u64> stream_commit_ns{0};
std::atomic<u64> waitregmem_count{0};
std::atomic<u64> waitregmem_ns{0};
std::atomic<u64> volabel_count{0};
std::atomic<u64> volabel_ns{0};
std::atomic<u64> present_count{0};
std::atomic<u64> present_ns{0};
std::atomic<u64> flip_pending_total{0};
std::atomic<u64> present_acquire_ns{0};
std::atomic<u64> present_flush_ns{0};
std::atomic<u64> present_lock_ns{0};
std::atomic<u64> present_khr_ns{0};
std::atomic<u64> renderframe_queue_ns{0};
std::atomic<u64> renderframe_fence_ns{0};
std::atomic<u64> renderframe_count{0};

namespace {

// Short enough to see when the decay starts, long enough that a multi-hour run stays small:
// 240 lines per hour at the default.
//
// Overridable per run with SHADPS4_FAULT_INTERVAL_MS, clamped to [250, 600000]. Both REFERENCE.md
// and PLAN.md documented this override for a while before it existed - a run that set the variable
// silently got 15 s anyway. The startup log line is what makes the override checkable rather than
// assumed.
constexpr long DEFAULT_REPORT_MS = 15000;

std::chrono::milliseconds ReportInterval() {
    static const std::chrono::milliseconds value = [] {
        long ms = DEFAULT_REPORT_MS;
        if (const char* env = std::getenv("SHADPS4_FAULT_INTERVAL_MS")) {
            char* end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end != env && parsed > 0) {
                ms = std::clamp(parsed, 250L, 600000L);
            }
        }
        return std::chrono::milliseconds(ms);
    }();
    return value;
}

std::mutex page_mutex;
std::unordered_map<u64, u64> read_fault_pages;

} // namespace

void RecordReadFaultPage(u64 addr) {
    std::scoped_lock lk(page_mutex);
    ++read_fault_pages[addr >> 12];
}

std::atomic<u64> gfx_submits{0};
std::atomic<u64> asc_submits{0};
std::atomic<u64> deferred_syncs{0};
std::atomic<u64> deferred_pages{0};

bool WriteFaultPagesEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_WRITE_FAULT_PAGES");
        return value && value[0] == '1';
    }();
    return enabled;
}

namespace {
struct WritePage {
    u32 faults{};
    u32 frames{};
    u64 last_frame{~0ULL};
};
std::mutex write_page_mutex;
std::unordered_map<u64, WritePage> write_fault_pages;
/// 2026-10-08 (zero-copy successor sizing, notes/zero-copy-build-2026-10-07.md sec. 5): per
/// frame, the distinct 4 KB pages and 64 KB / 256 KB / 1 MB blocks write-faulted inside the
/// per-frame heap span [0x248000000, 0x269e00000): what protecting at a coarser grain would
/// leave (faults ~ blocks) and upload (blocks x size). Under write_page_mutex.
struct SpanFrameBlocks {
    u64 frame{~0ULL};
    std::unordered_set<u64> sets[4]; // 4 KB pages, 64 KB, 256 KB, 1 MB blocks
    u64 sums[4]{};
    u64 frames{};
};
SpanFrameBlocks span_blocks;
/// 2026-10-13: readback-class write faults (page held GPU-written bytes) by page, all threads /
/// Game:Main (SHADPS4_WRITE_FAULT_PAGES + SHADPS4_GAME_MAIN_CENSUS). Under write_page_mutex.
std::unordered_map<u64, std::pair<u32, u32>> rb_fault_pages;
} // namespace

void RecordWriteFaultPage(u64 addr) {
    const u64 frame = presented_frames.load(std::memory_order_relaxed);
    std::scoped_lock lk(write_page_mutex);
    auto& page = write_fault_pages[addr >> 12];
    ++page.faults;
    if (span_blocks.frame != frame) {
        if (span_blocks.frame != ~0ULL) {
            for (int k = 0; k < 4; ++k) {
                span_blocks.sums[k] += span_blocks.sets[k].size();
                span_blocks.sets[k].clear();
            }
            ++span_blocks.frames;
        }
        span_blocks.frame = frame;
    }
    if (addr >= 0x248000000ULL && addr < 0x269e00000ULL) {
        span_blocks.sets[0].insert(addr >> 12);
        span_blocks.sets[1].insert(addr >> 16);
        span_blocks.sets[2].insert(addr >> 18);
        span_blocks.sets[3].insert(addr >> 20);
    }
    if (page.last_frame != frame) {
        page.last_frame = frame;
        ++page.frames;
    }
}

namespace {

/// Summarise and reset the window. "hot" = faulted in at least 90% of the window's frames.
std::string WriteFaultPageReport(u64 frames, u64 d_gfx, u64 d_asc) {
    std::vector<std::pair<u64, WritePage>> pages;
    u64 sb_sums[4]{};
    u64 sb_frames = 0;
    {
        std::scoped_lock lk(write_page_mutex);
        pages.assign(write_fault_pages.begin(), write_fault_pages.end());
        write_fault_pages.clear();
        for (int k = 0; k < 4; ++k) {
            sb_sums[k] = span_blocks.sums[k];
            span_blocks.sums[k] = 0;
        }
        sb_frames = span_blocks.frames;
        span_blocks.frames = 0;
    }
    if (frames == 0) {
        return "no frames";
    }
    u64 total = 0, hot_faults = 0, hot_frames = 0;
    u64 bucket_pages[4]{}, bucket_faults[4]{};
    std::vector<u64> hot;
    for (const auto& [page, st] : pages) {
        total += st.faults;
        const double frac = static_cast<double>(st.frames) / static_cast<double>(frames);
        const int b = frac >= 0.9 ? 3 : frac >= 0.5 ? 2 : frac >= 0.1 ? 1 : 0;
        bucket_pages[b]++;
        bucket_faults[b] += st.faults;
        if (b == 3) {
            hot.push_back(page);
            hot_faults += st.faults;
            hot_frames += st.frames;
        }
    }
    std::sort(hot.begin(), hot.end());
    std::vector<std::pair<u64, u64>> spans; // first page, count
    for (const u64 page : hot) {
        if (!spans.empty() && spans.back().first + spans.back().second == page) {
            ++spans.back().second;
        } else {
            spans.emplace_back(page, 1);
        }
    }
    std::sort(spans.begin(), spans.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    const double f = static_cast<double>(frames);
    std::string out = fmt::format(
        "frames={} faults/frame={:.1f} distinct_pages={} pages/frame(<10%,10-50,50-90,>=90% of "
        "frames)={}/{}/{}/{} fault_share={:.1f}/{:.1f}/{:.1f}/{:.1f}% hot_refaults_per_page_frame="
        "{:.2f} hot_spans={} gfx_submits/frame={:.2f} asc_submits/frame={:.2f} top_spans:",
        frames, total / f, pages.size(), bucket_pages[0], bucket_pages[1], bucket_pages[2],
        bucket_pages[3], total ? 100.0 * bucket_faults[0] / total : 0.0,
        total ? 100.0 * bucket_faults[1] / total : 0.0,
        total ? 100.0 * bucket_faults[2] / total : 0.0,
        total ? 100.0 * bucket_faults[3] / total : 0.0,
        hot_frames ? static_cast<double>(hot_faults) / hot_frames : 0.0, spans.size(), d_gfx / f,
        d_asc / f);
    for (size_t i = 0; i < std::min<size_t>(6, spans.size()); ++i) {
        out += fmt::format(" {:#x}+{}p", spans[i].first << 12, spans[i].second);
    }
    // Where the written pages live: 16 MB guest regions by fault count (plan b sizing).
    std::unordered_map<u64, std::pair<u64, u64>> regions; // region -> pages, faults
    for (const auto& [page, st] : pages) {
        auto& r = regions[page >> 12];
        ++r.first;
        r.second += st.faults;
    }
    std::vector<std::pair<u64, std::pair<u64, u64>>> by_region(regions.begin(), regions.end());
    std::sort(by_region.begin(), by_region.end(),
              [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    const double sbf = static_cast<double>(std::max<u64>(sb_frames, 1));
    out += fmt::format(" | span per frame: pages={:.1f} blocks64K={:.1f} blocks256K={:.1f} "
                       "blocks1M={:.1f} (frames {})",
                       sb_sums[0] / sbf, sb_sums[1] / sbf, sb_sums[2] / sbf, sb_sums[3] / sbf,
                       sb_frames);
    {
        std::vector<std::pair<u64, std::pair<u32, u32>>> rb;
        {
            std::scoped_lock lk(write_page_mutex);
            rb.assign(rb_fault_pages.begin(), rb_fault_pages.end());
            rb_fault_pages.clear();
        }
        std::sort(rb.begin(), rb.end(),
                  [](const auto& x, const auto& y) { return x.second.first > y.second.first; });
        u64 rb_total = 0;
        std::unordered_map<u64, u64> rb_regions;
        for (const auto& [pg, c] : rb) {
            rb_total += c.first;
            rb_regions[pg >> 12] += c.first;
        }
        out += fmt::format(" | readback-class faults/frame={:.2f} pages={} top_pages(faults/frame, Game:Main):",
                           rb_total / f, rb.size());
        for (size_t i = 0; i < std::min<size_t>(12, rb.size()); ++i) {
            out += fmt::format(" {:#x}:{:.2f}/{:.2f}", rb[i].first << 12, rb[i].second.first / f,
                               rb[i].second.second / f);
        }
        std::vector<std::pair<u64, u64>> rr(rb_regions.begin(), rb_regions.end());
        std::sort(rr.begin(), rr.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        out += " rb_regions16M:";
        for (size_t i = 0; i < std::min<size_t>(6, rr.size()); ++i) {
            out += fmt::format(" {:#x}:{:.2f}", rr[i].first << 24, rr[i].second / f);
        }
    }
    out += fmt::format(" regions16M={} top_regions:", by_region.size());
    for (size_t i = 0; i < std::min<size_t>(8, by_region.size()); ++i) {
        out += fmt::format(" {:#x}:{}p/{:.0f}f", by_region[i].first << 24,
                           by_region[i].second.first, by_region[i].second.second / f);
    }
    return out;
}

} // namespace

namespace {

/// Top read-faulting pages, plus how concentrated the distribution is. If a handful of pages
/// account for most faults, a targeted fix is possible; if it is spread thin, it is not.
std::string TopReadFaultPages(size_t top_n) {
    std::vector<std::pair<u64, u64>> pages;
    u64 total = 0;
    {
        std::scoped_lock lk(page_mutex);
        pages.assign(read_fault_pages.begin(), read_fault_pages.end());
    }
    for (const auto& [page, count] : pages) {
        total += count;
    }
    if (pages.empty() || total == 0) {
        return "no read faults";
    }
    const size_t n = std::min(top_n, pages.size());
    std::partial_sort(pages.begin(), pages.begin() + n, pages.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::string out = fmt::format("{} distinct pages, ", pages.size());
    u64 top_sum = 0;
    for (size_t i = 0; i < n; i++) {
        top_sum += pages[i].second;
    }
    out += fmt::format("top {} = {:.1f}% |", n, 100.0 * top_sum / total);
    for (size_t i = 0; i < n; i++) {
        out += fmt::format(" {:#x}={:.1f}%", pages[i].first << 12,
                           100.0 * pages[i].second / static_cast<double>(total));
    }
    return out;
}

std::mutex stop_mutex;
std::condition_variable stop_cv;
bool stop_requested = false;

struct Sample {
    u64 read;
    u64 read_unhandled;
    u64 read_ns;
    u64 write;
    u64 write_unhandled;
    u64 write_ns;
    u64 jit;
    u64 jit_ns;
    u64 frames;
    u64 fragments;
    u64 adds;
    u64 subtracts;
    u64 submits;
    u64 waits;
    u64 wait_ns;
    u64 wrm_count;
    u64 wrm_ns;
    u64 vo_count;
    u64 vo_ns;
    u64 pres_count;
    u64 pres_ns;
    u64 pending;
    u64 acq_ns;
    u64 flush_ns;
    u64 lock_ns;
    u64 khr_ns;
    u64 rf_count;
    u64 rf_queue_ns;
    u64 rf_fence_ns;
    u64 dl_calls;
    u64 dl_bytes;
    u64 draws;
    u64 d_pipeline_ns;
    u64 d_bindres_ns;
    u64 d_begin_ns;
    u64 d_finddepth_ns;
    u64 d_bindbuf_ns;
    u64 d_bindtex_ns;
    u64 bb_find_ns;
    u64 bb_obtain_ns;
    u64 ob_query_ns;
    u64 ob_stream_ns;
    u64 ob_stream_calls;
    u64 ob_stream_bytes;
    u64 ob_sync_ns;
    u64 ob_sync_calls;
    u64 up_scan_ns;
    u64 up_copy_ns;
    u64 up_mark_ns;
    u64 up_written_calls;
    u64 prot_w_calls;
    u64 prot_w_ns;
    u64 prot_r_calls;
    u64 prot_r_ns;
    u64 bt_findimage_ns;
    u64 bt_view_ns;
    u64 bt_sampler_ns;
    u64 bar_calls;
    u64 bar_partial;
    u64 bar_iters;
    u64 bar_emitted;
    u64 bt_findtex_ns;
    u64 bt_transit_ns;
    u64 cp_wait_ns_v;
    u64 cp_active_ns_v;
    u64 disp_count;
    u64 disp_ns;
    u64 d_pre_ns;
    u64 d_vtxbuf_ns;
    u64 d_descwrite_ns;
    u64 d_dynstate_ns;
    u64 d_schedbegin_ns;
    u64 d_emit_ns;
    u64 sc_map_ns;
    u64 sc_valid_ns;
    u64 sc_copy_ns;
    u64 sc_commit_ns;
    u64 cp_read;
    u64 cp_read_ns_v;
    u64 cp_write;
    u64 cp_write_ns_v;
    u64 up_calls;
    u64 up_empty;
    u64 up_managers;
    u64 up_ranges;
    u64 up_skipped;
    u64 up_bytes;
    u64 up_fastpath;
    u64 up_memohit;
};

Sample Read() {
    return Sample{
        guest_read_faults.load(std::memory_order_relaxed),
        guest_read_unhandled.load(std::memory_order_relaxed),
        guest_read_ns.load(std::memory_order_relaxed),
        guest_write_faults.load(std::memory_order_relaxed),
        guest_write_unhandled.load(std::memory_order_relaxed),
        guest_write_ns.load(std::memory_order_relaxed),
        jit_patch_faults.load(std::memory_order_relaxed),
        jit_patch_ns.load(std::memory_order_relaxed),
        presented_frames.load(std::memory_order_relaxed),
        range_fragments.load(std::memory_order_relaxed),
        range_adds.load(std::memory_order_relaxed),
        range_subtracts.load(std::memory_order_relaxed),
        gpu_submits.load(std::memory_order_relaxed),
        gpu_waits.load(std::memory_order_relaxed),
        gpu_wait_ns.load(std::memory_order_relaxed),
        waitregmem_count.load(std::memory_order_relaxed),
        waitregmem_ns.load(std::memory_order_relaxed),
        volabel_count.load(std::memory_order_relaxed),
        volabel_ns.load(std::memory_order_relaxed),
        present_count.load(std::memory_order_relaxed),
        present_ns.load(std::memory_order_relaxed),
        flip_pending_total.load(std::memory_order_relaxed),
        present_acquire_ns.load(std::memory_order_relaxed),
        present_flush_ns.load(std::memory_order_relaxed),
        present_lock_ns.load(std::memory_order_relaxed),
        present_khr_ns.load(std::memory_order_relaxed),
        renderframe_count.load(std::memory_order_relaxed),
        renderframe_queue_ns.load(std::memory_order_relaxed),
        renderframe_fence_ns.load(std::memory_order_relaxed),
        download_calls.load(std::memory_order_relaxed),
        download_bytes.load(std::memory_order_relaxed),
        draw_count.load(std::memory_order_relaxed),
        draw_pipeline_ns.load(std::memory_order_relaxed),
        draw_bindres_ns.load(std::memory_order_relaxed),
        draw_beginrender_ns.load(std::memory_order_relaxed),
        draw_finddepth_ns.load(std::memory_order_relaxed),
        draw_bindbuf_ns.load(std::memory_order_relaxed),
        draw_bindtex_ns.load(std::memory_order_relaxed),
        bindbuf_find_ns.load(std::memory_order_relaxed),
        bindbuf_obtain_ns.load(std::memory_order_relaxed),
        obtain_query_ns.load(std::memory_order_relaxed),
        obtain_stream_ns.load(std::memory_order_relaxed),
        obtain_stream_calls.load(std::memory_order_relaxed),
        obtain_stream_bytes.load(std::memory_order_relaxed),
        obtain_sync_ns.load(std::memory_order_relaxed),
        obtain_sync_calls.load(std::memory_order_relaxed),
        upload_scan_ns.load(std::memory_order_relaxed),
        upload_copy_ns.load(std::memory_order_relaxed),
        upload_mark_ns.load(std::memory_order_relaxed),
        upload_written_calls.load(std::memory_order_relaxed),
        protect_write_calls.load(std::memory_order_relaxed),
        protect_write_ns.load(std::memory_order_relaxed),
        protect_read_calls.load(std::memory_order_relaxed),
        protect_read_ns.load(std::memory_order_relaxed),
        bindtex_findimage_ns.load(std::memory_order_relaxed),
        bindtex_view_ns.load(std::memory_order_relaxed),
        bindtex_sampler_ns.load(std::memory_order_relaxed),
        barrier_calls.load(std::memory_order_relaxed),
        barrier_partial_calls.load(std::memory_order_relaxed),
        barrier_subres_iters.load(std::memory_order_relaxed),
        barrier_emitted.load(std::memory_order_relaxed),
        bindtex_findtex_ns.load(std::memory_order_relaxed),
        bindtex_transit_ns.load(std::memory_order_relaxed),
        cp_wait_ns.load(std::memory_order_relaxed),
        cp_active_ns.load(std::memory_order_relaxed),
        dispatch_count.load(std::memory_order_relaxed),
        dispatch_ns.load(std::memory_order_relaxed),
        draw_pre_ns.load(std::memory_order_relaxed),
        draw_vtxbuf_ns.load(std::memory_order_relaxed),
        draw_descwrite_ns.load(std::memory_order_relaxed),
        draw_dynstate_ns.load(std::memory_order_relaxed),
        draw_schedbegin_ns.load(std::memory_order_relaxed),
        draw_emit_ns.load(std::memory_order_relaxed),
        stream_map_ns.load(std::memory_order_relaxed),
        stream_valid_ns.load(std::memory_order_relaxed),
        stream_copy_ns.load(std::memory_order_relaxed),
        stream_commit_ns.load(std::memory_order_relaxed),
        cp_read_faults.load(std::memory_order_relaxed),
        cp_read_ns.load(std::memory_order_relaxed),
        cp_write_faults.load(std::memory_order_relaxed),
        cp_write_ns.load(std::memory_order_relaxed),
        upload_calls.load(std::memory_order_relaxed),
        upload_empty_calls.load(std::memory_order_relaxed),
        upload_managers.load(std::memory_order_relaxed),
        upload_ranges.load(std::memory_order_relaxed),
        upload_skipped.load(std::memory_order_relaxed),
        upload_bytes.load(std::memory_order_relaxed),
        upload_fastpath.load(std::memory_order_relaxed),
        upload_memohit.load(std::memory_order_relaxed),
    };
}

std::ofstream OpenCsv() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char name[64];
    std::strftime(name, sizeof(name), "fault_rate_%Y%m%d_%H%M%S.csv", &tm);

    const auto path = FS::GetUserPath(FS::PathType::LogDir) / name;
    std::ofstream csv(path);
    if (csv) {
        csv << "elapsed_s,fps,read_per_s,write_per_s,jit_per_s,"
               "faults_per_frame,read_us_each,write_us_each,handler_pct_of_wall,"
               "read_total,read_unhandled,write_total,write_unhandled,jit_total,frames_total,"
               "range_fragments,range_adds_per_s,range_subs_per_s,"
               "submits_per_frame,waits_per_frame,gpu_wait_ms_per_frame,"
               "waitregmem_per_frame,waitregmem_ms_per_frame,volabel_per_frame,"
               "volabel_ms_per_frame,present_ms_each,present_ms_per_frame,avg_flip_pending,"
               "acquire_ms,flush_ms,submitlock_ms,presentkhr_ms,"
               "rf_queue_ms,rf_fence_ms,download_per_frame,download_kb_per_frame,"
               "draws_per_frame,pipeline_us_each,bindres_us_each,beginrender_us_each,"
               "finddepth_us_each,draw_ms_per_frame,bindbuf_us_each,bindtex_us_each,"
               "bb_find_us_each,bb_obtain_us_each,"
               "ob_query_us_each,ob_stream_us_each,ob_sync_us_each,"
               "ob_stream_per_draw,ob_sync_per_draw,ob_stream_kb_per_draw,"
               "up_scan_us_each,up_copy_us_each,up_mark_us_each,up_written_per_draw,"
               "protw_per_draw,protw_us_each,protr_per_draw,protr_us_each,"
               "sc_map_us_each,sc_valid_us_each,sc_copy_us_each,sc_commit_us_each,"
               "bt_findimage_us_each,bt_view_us_each,bt_sampler_us_each,"
               "d_pre_us_each,d_vtxbuf_us_each,d_descwrite_us_each,d_dynstate_us_each,"
               "d_schedbegin_us_each,d_emit_us_each,"
               "cp_wait_ms_per_frame,cp_active_ms_per_frame,disp_per_frame,disp_us_each,"
               "bt_findtex_us_each,bt_transit_us_each,"
               "bar_calls_per_draw,bar_partial_pct,bar_iters_per_call,bar_emitted_per_call,"
               "cpfault_ms_per_frame,cpfault_pct_of_handler,"
               "up_empty_pct,up_managers_per_call,up_ranges_per_call,"
               "up_skipped_pct,up_kb_per_call,"
               "up_fastpath_pct,"
               "up_memohit_pct,"
               "frame_max_ms,frames_over_10ms,frames_over_17ms,frames_over_33ms,frame_slow_ms,"
               "front_idle_ms,front_wait_ms,back_idle_ms,rf_guest_ms,upload_kb_per_frame,uploads_per_frame\n";
        csv.flush();
        LOG_INFO(Render, "FaultMonitor: rate CSV at {}", FS::PathToUTF8String(path));
    } else {
        LOG_WARNING(Render, "FaultMonitor: could not open {}", FS::PathToUTF8String(path));
    }
    return csv;
}

} // namespace (ReportLoop helpers; gmc census below is public)

namespace {
struct GmCensus {
    std::array<std::atomic<u64>, 16> wait_ticks{}, wait_n{};
    std::array<std::atomic<u64>, 4> place{}; // [game main P, game main E, other P, other E]
    /// 2026-10-15: P / E samples by thread kind (PlaceKind: 0 Game:Main, 1 JobWorker, 2 other guest,
    /// 3 LeanWorker, 4 A2Worker, 5 front).
    std::array<std::atomic<u64>, 12> place_kind{};
    std::array<std::atomic<u64>, 16> gnm_all_ticks{}, gnm_all_n{}; // gnm waits, every thread
    std::mutex rip_mutex;
    std::unordered_map<u64, u32> rip_eboot; // 64 B buckets of the eboot offset
    u64 rip_samples{}, rip_eboot_n{}, rip_guest_other{}, rip_host{}, rip_samples_window{};
    std::unordered_map<u64, u32> rip_host_buckets; // 4 KB buckets of host addresses
    bool sampler_started{};
    // gmc3: delay() callers (first two eboot return addresses) and episodes
    std::unordered_map<u64, u32> delay_callers; // key: c1 offset << 32 | c2 offset
    struct Ep {
        u64 episodes{}, samples{};
    };
    std::unordered_map<u64, Ep> delay_eps; // by c1 offset
    u64 last_c1{~0ULL}, run{};
    u64 win32u_n{}, win32u_in_gpu_wait{};
    u64 win32u_lo{}, win32u_hi{};
    std::mutex gpu_mutex;
    struct GpuPath {
        u64 n{}, ticks{};
    };
    std::map<std::array<u64, 4>, GpuPath> gpu_paths;
    std::array<std::atomic<u64>, 8> gm_fault_n{}, gm_fault_ticks{}, all_fault_n{}, all_fault_ticks{};
    std::mutex page_mutex;
    u64 frame{~0ULL};
    std::unordered_set<u64> gm_pages, all_pages;
    u64 gm_pages_sum{}, all_pages_sum{}, frames{};
    void* thread{};
    // SHADPS4_JOB_CENSUS (2026-10-12): Game:Main's main-loop step and the job threads' split.
    // Main loop 0x29c8d0 (one caller): the qword below its RSP while it is inside a call holds
    // that call's return address = the step. Found once by a stack scan, then one read per sample.
    u64 loop_slot{}, loop_slot_finds{}, loop_misses{}, stack_lo{}, stack_hi{};
    std::unordered_map<u32, u64> steps; // step (eboot offset of the return address) -> samples
    std::unordered_map<u32, u64> steps_host; // of those, samples with Game:Main's RIP in host code
    /// r19c (2026-10-14 evening): (step, host RIP page) -> samples and the last exact RIP seen there,
    /// so the emulator functions behind each step's host time can be symbolized offline.
    std::map<std::pair<u32, u64>, std::pair<u64, u64>> steps_host_pages;
    u64 step_samples{};
    std::atomic<u32> nthreads{0};
    std::array<void*, 16> threads{};
    std::array<u64, 16> last_cyc{};
    u32 rr{};
    u64 job_samples{}, job_idle{}, job_busy_cyc{};
    u64 job_span_samples{}, job_span_idle{}, job_span_busy_cyc{};
    std::unordered_map<u64, u64> job_class, job_class_span; // 1 game, 2 delay, 3 other guest, else host 4 KB page
    u64 last_frames{};
    std::array<std::atomic<u64>, 2> nrec_cyc{}, nrec_n{}; // [Game:Main, other threads]
} g_gm;

constexpr u64 kLoopLo = 0x80029c8d0ULL, kLoopHi = 0x80029ce40ULL;   // main loop body
constexpr u64 kCaveLo = 0x8003ee160ULL, kCaveHi = 0x8003ee190ULL;   // V1 cave (main-loop depth)
// Steps from the render-job kick (0x190090 returns to 0x29cd07) to the join's return (0x29cd3b).
inline bool InRenderSpan(u32 step) {
    return step >= 0x29cd07 && step <= 0x29cd3b;
}
} // namespace

bool JobCensus() {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_JOB_CENSUS");
        return v && v[0] == '1' && GameMainCensus();
    }();
    return on;
}

void NoteGuestThread(const char* name) {
#ifdef _WIN32
    if (!JobCensus() || !name || std::strncmp(name, "JobWorker", 9) != 0) {
        return;
    }
    HANDLE h{};
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                    THREAD_QUERY_LIMITED_INFORMATION | THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT,
                    FALSE, 0);
    static std::mutex mu;
    std::scoped_lock lk{mu};
    const u32 n = g_gm.nthreads.load(std::memory_order_relaxed);
    if (h && n < g_gm.threads.size()) {
        g_gm.threads[n] = h;
        g_gm.nthreads.store(n + 1, std::memory_order_release);
    }
#endif
}

void NoteNativeRecCycles(u64 cycles) {
    thread_local u64 acc = 0, n = 0;
    acc += cycles;
    if ((++n & 255) == 0) {
        const u32 i = t_is_game_main ? 0 : 1;
        g_gm.nrec_cyc[i].fetch_add(acc, std::memory_order_relaxed);
        g_gm.nrec_n[i].fetch_add(256, std::memory_order_relaxed);
        acc = 0;
    }
}

bool GameMainCensus() {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_GAME_MAIN_CENSUS");
        return v && v[0] == '1';
    }();
    return on;
}

u64 CensusTicks() {
    return __rdtsc();
}

void MarkGameMainThread() {
    t_is_game_main = true;
#ifdef _WIN32
    HANDLE h{};
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                    THREAD_QUERY_LIMITED_INFORMATION | THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT,
                    FALSE, 0);
    g_gm.thread = h;
    if (GameMainCensus() && h && !g_gm.sampler_started) {
        g_gm.sampler_started = true;
        // ~1 kHz RIP sampler: the thread is suspended only between SuspendThread and
        // ResumeThread; nothing here takes a lock or allocates while it is suspended.
        std::thread([h] {
            timeBeginPeriod(1);
            HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr,
                                                  CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                                  TIMER_ALL_ACCESS);
            if (HMODULE w = GetModuleHandleA("win32u.dll")) {
                MEMORY_BASIC_INFORMATION mbi{};
                // the whole image: walk regions with the same allocation base
                u64 a = reinterpret_cast<u64>(w);
                g_gm.win32u_lo = a;
                while (VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof(mbi)) &&
                       mbi.AllocationBase == w) {
                    a += mbi.RegionSize;
                }
                g_gm.win32u_hi = a;
            }
            for (;;) {
                if (timer) {
                    LARGE_INTEGER due{};
                    due.QuadPart = -5000; // 0.5 ms
                    SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
                    WaitForSingleObject(timer, INFINITE);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                const u32 in_gpu = g_gm_in_gpu_wait.load(std::memory_order_relaxed);
                if (SuspendThread(h) == DWORD(-1)) {
                    continue;
                }
                CONTEXT ctx{};
                ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
                const BOOL ok = GetThreadContext(h, &ctx);
                std::array<u64, 64> stack{};
                const bool in_delay = ok && ctx.Rip >= 0x80053b2b0ULL && ctx.Rip < 0x80053b360ULL;
                if (in_delay) {
                    // the guest stack is ordinary committed memory; copied while suspended
                    std::memcpy(stack.data(), reinterpret_cast<const void*>(ctx.Rsp), sizeof(stack));
                }
                // SHADPS4_JOB_CENSUS: the main-loop step (one qword at the found slot) or, until the
                // slot is known, up to 16 KB of stack (clamped to the stack's region) to find it.
                const bool job = JobCensus();
                u64 slot_val = 0;
                u32 scan_n = 0;
                static std::array<u64, 2048> big{};
                if (job && ok) {
                    const u64 slot = g_gm.loop_slot;
                    if (slot != 0) {
                        if (ctx.Rsp < slot && slot + 8 <= g_gm.stack_hi) {
                            slot_val = *reinterpret_cast<const u64*>(slot);
                        }
                    } else if (ctx.Rsp >= g_gm.stack_lo && ctx.Rsp < g_gm.stack_hi) {
                        scan_n = static_cast<u32>(
                            std::min<u64>(big.size(), (g_gm.stack_hi - ctx.Rsp) / 8));
                        std::memcpy(big.data(), reinterpret_cast<const void*>(ctx.Rsp), scan_n * 8ULL);
                    }
                }
                ResumeThread(h);
                if (!ok) {
                    continue;
                }
                u32 step = 0;
                if (job) {
                    if (ctx.Rsp < g_gm.stack_lo || ctx.Rsp >= g_gm.stack_hi) {
                        MEMORY_BASIC_INFORMATION mbi{};
                        if (VirtualQuery(reinterpret_cast<void*>(ctx.Rsp), &mbi, sizeof(mbi))) {
                            g_gm.stack_lo = reinterpret_cast<u64>(mbi.BaseAddress);
                            g_gm.stack_hi = g_gm.stack_lo + mbi.RegionSize;
                            g_gm.loop_slot = 0;
                        }
                    }
                    const bool loop_ret = (slot_val >= kLoopLo && slot_val < kLoopHi) ||
                                          (slot_val >= kCaveLo && slot_val < kCaveHi);
                    if (ctx.Rip >= kLoopLo && ctx.Rip < kLoopHi) {
                        step = 1; // in the loop body itself
                    } else if (loop_ret) {
                        step = static_cast<u32>(slot_val - 0x800000000ULL);
                        g_gm.loop_misses = 0;
                    } else if (g_gm.loop_slot != 0 && ++g_gm.loop_misses > 4000) {
                        g_gm.loop_slot = 0; // ~2 s without the loop: find it again
                        g_gm.loop_misses = 0;
                    }
                    // the outermost return address into the loop body is its current call's slot
                    for (u32 i = scan_n; i-- > 0;) {
                        if (big[i] >= kLoopLo && big[i] < kLoopHi) {
                            g_gm.loop_slot = ctx.Rsp + 8ULL * i;
                            ++g_gm.loop_slot_finds;
                            step = static_cast<u32>(big[i] - 0x800000000ULL);
                            break;
                        }
                    }
                    const u32 nt = g_gm.nthreads.load(std::memory_order_acquire);
                    if (nt != 0) {
                        // one job thread per tick, round robin (~0.5 ms ticks)
                        const u32 i = g_gm.rr++ % nt;
                        HANDLE jh = static_cast<HANDLE>(g_gm.threads[i]);
                        ULONG64 cyc = 0;
                        QueryThreadCycleTime(jh, &cyc);
                        const u64 prev = g_gm.last_cyc[i];
                        g_gm.last_cyc[i] = cyc;
                        if (prev != 0 && SuspendThread(jh) != DWORD(-1)) {
                            CONTEXT jc{};
                            jc.ContextFlags = CONTEXT_CONTROL;
                            const BOOL jok = GetThreadContext(jh, &jc);
                            ResumeThread(jh);
                            if (jok) {
                                const u64 d = cyc - prev;
                                // visited every nt ticks: idle if it ran < 25% of that interval
                                const bool idle = d < 437'500ULL * nt / 4;
                                const u64 rip = jc.Rip;
                                const u64 key = rip >= 0x80053b2b0ULL && rip < 0x80053b360ULL ? 2
                                                : rip >= 0x800000000ULL && rip < 0x8087a8000ULL ? 1
                                                : rip >= 0x8087a8000ULL && rip < 0x1000000000ULL ? 3
                                                : (rip >> 12) << 12;
                                const bool span = InRenderSpan(step);
                                std::scoped_lock lk{g_gm.rip_mutex};
                                ++g_gm.job_samples;
                                g_gm.job_busy_cyc += d;
                                if (span) {
                                    ++g_gm.job_span_samples;
                                    g_gm.job_span_busy_cyc += d;
                                }
                                if (idle) {
                                    ++g_gm.job_idle;
                                    g_gm.job_span_idle += span;
                                } else {
                                    ++g_gm.job_class[key];
                                    if (span) {
                                        ++g_gm.job_class_span[key];
                                    }
                                }
                            }
                        }
                    }
                    std::scoped_lock lk{g_gm.rip_mutex};
                    ++g_gm.steps[step];
                    ++g_gm.step_samples;
                    // 2026-10-14: emulator (host) samples per step - Game:Main's RIP outside guest
                    // code (eboot / guest modules live at 0x800000000+): fault handling, HLE, waits.
                    if (ctx.Rip < 0x800000000ULL || ctx.Rip >= 0x1000000000ULL) {
                        ++g_gm.steps_host[step];
                        auto& hp = g_gm.steps_host_pages[{step, (ctx.Rip >> 12) << 12}];
                        ++hp.first;
                        hp.second = ctx.Rip;
                    }
                }
                const u64 rip = ctx.Rip;
                u64 c1 = 0, c2 = 0;
                if (in_delay) {
                    for (const u64 v : stack) {
                        if (v >= 0x800000000ULL && v < 0x800cc8000ULL &&
                            !(v >= 0x80053b2b0ULL && v < 0x80053b360ULL)) {
                            if (c1 == 0) {
                                c1 = v - 0x800000000ULL;
                            } else {
                                c2 = v - 0x800000000ULL;
                                break;
                            }
                        }
                    }
                }
                std::scoped_lock lk{g_gm.rip_mutex};
                ++g_gm.rip_samples;
                const u64 cur = in_delay ? c1 : ~0ULL;
                if (cur != g_gm.last_c1) {
                    if (g_gm.last_c1 != ~0ULL && g_gm.run != 0) {
                        auto& ep = g_gm.delay_eps[g_gm.last_c1];
                        ++ep.episodes;
                        ep.samples += g_gm.run;
                    }
                    g_gm.last_c1 = cur;
                    g_gm.run = 0;
                }
                if (in_delay) {
                    ++g_gm.run;
                    ++g_gm.delay_callers[c1 << 32 | (c2 & 0xffffffffULL)];
                }
                if (rip >= g_gm.win32u_lo && rip < g_gm.win32u_hi) {
                    ++g_gm.win32u_n;
                    g_gm.win32u_in_gpu_wait += in_gpu != 0;
                }
                if (rip >= 0x800000000ULL && rip < 0x8087a8000ULL) {
                    ++g_gm.rip_eboot_n;
                    ++g_gm.rip_eboot[(rip - 0x800000000ULL) >> 6];
                } else if (rip >= 0x8087a8000ULL && rip < 0x1000000000ULL) {
                    ++g_gm.rip_guest_other;
                } else {
                    ++g_gm.rip_host;
                    ++g_gm.rip_host_buckets[rip >> 12];
                }
            }
        }).detach();
    }
#endif
}

std::atomic<u32> g_gm_in_gpu_wait{0};

void NoteGameMainGpuWait(u64 ticks) {
    void* frames[4]{};
    const u32 n = RtlCaptureStackBackTrace(1, 4, frames, nullptr);
    std::array<u64, 4> key{};
    const u64 exe = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
    for (u32 i = 0; i < n && i < 4; ++i) {
        key[i] = reinterpret_cast<u64>(frames[i]) - exe;
    }
    if (n == 0) {
        key = {0xA000u + t_gpu_wait_tag, t_gpu_wait_early, 0, 0}; // m1: the site tag
    }
    std::scoped_lock lk{g_gm.gpu_mutex};
    auto& p = g_gm.gpu_paths[key];
    ++p.n;
    p.ticks += ticks;
}

void NoteGnmWait(u32 kind, u64 ticks) {
    g_gm.gnm_all_ticks[kind & 15].fetch_add(ticks, std::memory_order_relaxed);
    g_gm.gnm_all_n[kind & 15].fetch_add(1, std::memory_order_relaxed);
    if (t_is_game_main) {
        NoteGameMainWait(kind, ticks);
    }
}

void SampleGuestPlacement() {
    if (!GameMainCensus()) {
        return;
    }
    thread_local u32 calls = 0;
    if ((++calls & 63) != 0) {
        return;
    }
#ifdef _WIN32
    const bool p = Common::IsPerformanceProcessor(Common::CurrentProcessor());
    g_gm.place[(t_is_game_main ? 0 : 2) + (p ? 0 : 1)].fetch_add(1, std::memory_order_relaxed);
    // 2026-10-15: the game's job threads apart from the other guest threads (name read once).
    thread_local int kind = -1;
    if (kind < 0) {
        kind = 2;
        PWSTR desc = nullptr;
        if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &desc)) && desc != nullptr) {
            if (std::wcsncmp(desc, L"JobWorker", 9) == 0) {
                kind = 1;
            }
            LocalFree(desc);
        }
    }
    g_gm.place_kind[(t_is_game_main ? 0 : kind) * 2 + (p ? 0 : 1)].fetch_add(1, std::memory_order_relaxed);
#endif
}

void SampleEmuPlacement(u32 kind) {
    if (!GameMainCensus()) {
        return;
    }
    thread_local u32 calls = 0;
    if ((++calls & 15) != 0) {
        return;
    }
#ifdef _WIN32
    const bool p = Common::IsPerformanceProcessor(Common::CurrentProcessor());
    g_gm.place_kind[std::min<u32>(kind, 5) * 2 + (p ? 0 : 1)].fetch_add(1, std::memory_order_relaxed);
#endif
}

void NoteGameMainWait(u32 kind, u64 ticks) {
    g_gm.wait_ticks[kind & 15].fetch_add(ticks, std::memory_order_relaxed);
    g_gm.wait_n[kind & 15].fetch_add(1, std::memory_order_relaxed);
}

void NoteWriteFaultCensus(u64 addr, u64 ticks, u8 cls) {
    const u32 c = cls & 7;
    g_gm.all_fault_n[c].fetch_add(1, std::memory_order_relaxed);
    g_gm.all_fault_ticks[c].fetch_add(ticks, std::memory_order_relaxed);
    if (t_is_game_main) {
        g_gm.gm_fault_n[c].fetch_add(1, std::memory_order_relaxed);
        g_gm.gm_fault_ticks[c].fetch_add(ticks, std::memory_order_relaxed);
    }
    const u64 frame = presented_frames.load(std::memory_order_relaxed);
    std::scoped_lock lk{g_gm.page_mutex};
    if (frame != g_gm.frame) {
        if (g_gm.frame != ~0ULL) {
            g_gm.gm_pages_sum += g_gm.gm_pages.size();
            g_gm.all_pages_sum += g_gm.all_pages.size();
            ++g_gm.frames;
        }
        g_gm.frame = frame;
        g_gm.gm_pages.clear();
        g_gm.all_pages.clear();
    }
    g_gm.all_pages.insert(addr >> 12);
    if ((cls & 2) != 0 && WriteFaultPagesEnabled()) {
        std::scoped_lock lk2(write_page_mutex);
        auto& e = rb_fault_pages[addr >> 12];
        ++e.first;
        e.second += t_is_game_main ? 1 : 0;
    }
    if (t_is_game_main) {
        g_gm.gm_pages.insert(addr >> 12);
    }
}

void GameMainCensusMaybeLog() {
    if (!GameMainCensus()) {
        return;
    }
    static u64 last_tsc = 0, last_cpu_100ns = 0;
    const u64 now = __rdtsc();
    if (last_tsc == 0) {
        last_tsc = now;
        return;
    }
    // SHADPS4_GAME_MAIN_CENSUS_WINDOW_S (2026-10-12, play telemetry): window length, default 10 s.
    static const double window = [] {
        const char* v = std::getenv("SHADPS4_GAME_MAIN_CENSUS_WINDOW_S");
        return v ? std::clamp(std::atof(v), 0.5, 600.0) : 10.0;
    }();
    const double sec = double(now - last_tsc) / 3.5e9;
    if (sec < window) {
        return;
    }
    last_tsc = now;
    double cpu_ms = -1.0;
#ifdef _WIN32
    if (g_gm.thread) {
        FILETIME c, e, k, u;
        if (GetThreadTimes(static_cast<HANDLE>(g_gm.thread), &c, &e, &k, &u)) {
            const u64 t = ((u64(k.dwHighDateTime) << 32) | k.dwLowDateTime) +
                          ((u64(u.dwHighDateTime) << 32) | u.dwLowDateTime);
            if (last_cpu_100ns != 0) {
                cpu_ms = double(t - last_cpu_100ns) / 1e4 / sec;
            }
            last_cpu_100ns = t;
        }
    }
#endif
    const auto ms = [&](std::atomic<u64>& a) { return double(a.exchange(0)) / 3.5e6 / sec; };
    const auto per = [&](std::atomic<u64>& a) { return double(a.exchange(0)) / sec; };
    std::string waits;
    static constexpr const char* names[] = {"equeue", "cond", "evflag", "sema", "sleep", "mutex", "join",
                                            "gnm SubmitDone idle-wait", "gnm submit idle-wait",
                                            "gnm DingDong idle-wait", "SendCommand fault-prepare",
                                            "SendCommand retry-settle", "SendCommand fault-finish",
                                            "SendCommand other"};
    double wait_total = 0.0;
    for (u32 i = 0; i < 14; ++i) {
        const double n = per(g_gm.wait_n[i]);
        const double m = ms(g_gm.wait_ticks[i]);
        wait_total += m;
        waits += fmt::format(" {} {:.1f} ms/s ({:.0f}/s)", names[i], m, n);
    }
    {
        static constexpr const char* sc_names[] = {"prepare", "settle", "finish", "other"};
        waits += " | q13 SendCommand queue/run ms/s:";
        for (u32 i = 0; i < 4; ++i) {
            waits += fmt::format(" {} {:.1f}/{:.1f}", sc_names[i], ms(g_sc_queue_ticks[i]),
                                 ms(g_sc_run_ticks[i]));
        }
    }
    std::string gm_cls, all_cls;
    double gm_fault_ms = 0.0, gm_fault_n = 0.0, all_fault_ms = 0.0, all_fault_n = 0.0;
    static constexpr const char* cls_names[] = {"none", "buffer", "readback", "buffer+readback",
                                                "texture", "buffer+texture", "readback+texture",
                                                "all"};
    for (u32 i = 0; i < 8; ++i) {
        const double gn = per(g_gm.gm_fault_n[i]), gm = ms(g_gm.gm_fault_ticks[i]);
        const double an = per(g_gm.all_fault_n[i]), am = ms(g_gm.all_fault_ticks[i]);
        gm_fault_ms += gm;
        gm_fault_n += gn;
        all_fault_ms += am;
        all_fault_n += an;
        if (an > 0.0) {
            gm_cls += fmt::format(" {} {:.0f}/s {:.1f}us", cls_names[i], gn, gn > 0 ? gm * 1e3 / gn : 0.0);
            all_cls += fmt::format(" {} {:.0f}/s {:.1f}us", cls_names[i], an, an > 0 ? am * 1e3 / an : 0.0);
        }
    }
    double gm_pages = 0.0, all_pages = 0.0;
    {
        std::scoped_lock lk{g_gm.page_mutex};
        if (g_gm.frames != 0) {
            gm_pages = double(g_gm.gm_pages_sum) / double(g_gm.frames);
            all_pages = double(g_gm.all_pages_sum) / double(g_gm.frames);
        }
        g_gm.gm_pages_sum = g_gm.all_pages_sum = g_gm.frames = 0;
    }
    LOG_INFO(Render,
             "GameMain census: window {:.1f}s | Game:Main CPU {:.0f} ms/s, blocked {:.0f} ms/s | "
             "waits{} | total timed waits {:.0f} ms/s | write faults on Game:Main {:.0f}/s = "
             "{:.1f} ms/s ({:.1f} us each), by owner:{} | all guest threads {:.0f}/s = {:.1f} ms/s "
             "({:.1f} us each), by owner:{} | distinct faulting pages per frame: Game:Main {:.0f}, "
             "all {:.0f} | Game:Main rest (CPU - write faults) {:.0f} ms/s",
             sec, cpu_ms, cpu_ms >= 0 ? 1000.0 - cpu_ms : -1.0, waits, wait_total, gm_fault_n,
             gm_fault_ms, gm_fault_n > 0 ? gm_fault_ms * 1e3 / gm_fault_n : 0.0, gm_cls,
             all_fault_n, all_fault_ms, all_fault_n > 0 ? all_fault_ms * 1e3 / all_fault_n : 0.0,
             all_cls, gm_pages, all_pages, cpu_ms >= 0 ? cpu_ms - gm_fault_ms : -1.0);
    {
        std::string g;
        for (u32 i = 7; i < 10; ++i) {
            const double n = per(g_gm.gnm_all_n[i]);
            g += fmt::format(" {} {:.1f} ms/s ({:.0f}/s)", names[i], ms(g_gm.gnm_all_ticks[i]), n);
        }
        LOG_INFO(Render, "GameMain census gnm idle waits, all threads:{}", g);
    }
    {
        const double gp = double(g_gm.place[0].exchange(0)), ge = double(g_gm.place[1].exchange(0));
        const double op = double(g_gm.place[2].exchange(0)), oe = double(g_gm.place[3].exchange(0));
        LOG_INFO(Render,
                 "GameMain census placement (every 64th guest mutex lock): Game:Main P {:.1f}% E "
                 "{:.1f}% (n {:.0f}) | other guest threads P {:.1f}% E {:.1f}% (n {:.0f})",
                 100.0 * gp / std::max(gp + ge, 1.0), 100.0 * ge / std::max(gp + ge, 1.0), gp + ge,
                 100.0 * op / std::max(op + oe, 1.0), 100.0 * oe / std::max(op + oe, 1.0), op + oe);
        static constexpr const char* kinds[6] = {"Game:Main", "JobWorker", "other guest",
                                                 "LeanWorker", "A2Worker", "front"};
        std::string k;
        for (u32 i = 0; i < 6; ++i) {
            const double kp = double(g_gm.place_kind[2 * i].exchange(0));
            const double ke = double(g_gm.place_kind[2 * i + 1].exchange(0));
            k += fmt::format(" | {} P {:.1f}% (n {:.0f})", kinds[i], 100.0 * kp / std::max(kp + ke, 1.0),
                             kp + ke);
        }
        LOG_INFO(Render, "GameMain census placement by thread kind:{}", k);
    }
    {
        std::vector<std::pair<u32, u64>> top, host;
        u64 n = 0, eb = 0, go = 0, ho = 0;
        {
            std::scoped_lock lk{g_gm.rip_mutex};
            n = g_gm.rip_samples;
            eb = g_gm.rip_eboot_n;
            go = g_gm.rip_guest_other;
            ho = g_gm.rip_host;
            for (const auto& [b, c] : g_gm.rip_eboot) {
                top.emplace_back(c, b);
            }
            for (const auto& [b, c] : g_gm.rip_host_buckets) {
                host.emplace_back(c, b);
            }
            g_gm.rip_eboot.clear();
            g_gm.rip_host_buckets.clear();
            g_gm.rip_samples_window = g_gm.rip_samples;
            g_gm.rip_samples = g_gm.rip_eboot_n = g_gm.rip_guest_other = g_gm.rip_host = 0;
        }
        std::sort(top.rbegin(), top.rend());
        std::sort(host.rbegin(), host.rend());
        std::string s, hs;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 40); ++i) {
            s += fmt::format(" {:#x}:{:.1f}%", top[i].second << 6, 100.0 * top[i].first / std::max<u64>(n, 1));
        }
        for (size_t i = 0; i < std::min<size_t>(host.size(), 12); ++i) {
            HMODULE mod{};
            char name[64] = "?";
            const u64 addr = host[i].second << 12;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(addr), &mod)) {
                char path[MAX_PATH];
                if (GetModuleFileNameA(mod, path, MAX_PATH)) {
                    const char* base = std::strrchr(path, '\\');
                    std::snprintf(name, sizeof(name), "%s", base ? base + 1 : path);
                }
            }
            hs += fmt::format(" {}+{:#x}:{:.1f}%", name,
                              addr - reinterpret_cast<u64>(mod), 100.0 * host[i].first / std::max<u64>(n, 1));
        }
        LOG_INFO(Render,
                 "GameMain census RIP: samples {} | eboot {:.1f}% other guest modules {:.1f}% host "
                 "{:.1f}% | eboot top 64-byte buckets (offset from 0x800000000):{}",
                 n, 100.0 * eb / std::max<u64>(n, 1), 100.0 * go / std::max<u64>(n, 1),
                 100.0 * ho / std::max<u64>(n, 1), s);
        LOG_INFO(Render, "GameMain census RIP host top 4 KB pages:{}", hs);
    }
    {
        std::vector<std::pair<u32, u64>> callers;
        std::vector<std::pair<u64, std::pair<u64, u64>>> eps; // samples, (c1, episodes)
        u64 n = 0, w = 0, wg = 0;
        {
            std::scoped_lock lk{g_gm.rip_mutex};
            n = std::max<u64>(1, g_gm.rip_samples_window);
            for (const auto& [k, c] : g_gm.delay_callers) {
                callers.emplace_back(c, k);
            }
            for (const auto& [c1, ep] : g_gm.delay_eps) {
                eps.push_back({ep.samples, {c1, ep.episodes}});
            }
            w = g_gm.win32u_n;
            wg = g_gm.win32u_in_gpu_wait;
            g_gm.delay_callers.clear();
            g_gm.delay_eps.clear();
            g_gm.win32u_n = g_gm.win32u_in_gpu_wait = 0;
        }
        std::sort(callers.rbegin(), callers.rend());
        std::sort(eps.rbegin(), eps.rend());
        std::string cs, es;
        for (size_t i = 0; i < std::min<size_t>(callers.size(), 14); ++i) {
            cs += fmt::format(" {:#x}<-{:#x}:{:.1f}%", callers[i].second >> 32,
                              callers[i].second & 0xffffffffULL, 100.0 * callers[i].first / double(n));
        }
        for (size_t i = 0; i < std::min<size_t>(eps.size(), 10); ++i) {
            const double episodes = double(eps[i].second.second);
            es += fmt::format(" {:#x}: {:.1f}% {:.0f}/s {:.2f} ms each", eps[i].second.first,
                              100.0 * eps[i].first / double(n), episodes / sec,
                              episodes > 0 ? double(eps[i].first) * 0.5 / episodes : 0.0);
        }
        LOG_INFO(Render,
                 "GameMain census delay(): callers (return address <- its caller, offsets, share of "
                 "Game:Main wall samples):{} | episodes by caller (0.5 ms samples):{} | win32u samples "
                 "{:.1f}%, inside MasterSemaphore::Wait {:.1f}% of them",
                 cs, es, 100.0 * w / double(n), 100.0 * wg / double(std::max<u64>(w, 1)));
        std::vector<std::pair<u64, std::pair<std::array<u64, 4>, u64>>> paths;
        {
            std::scoped_lock lk{g_gm.gpu_mutex};
            for (const auto& [k, p] : g_gm.gpu_paths) {
                paths.push_back({p.ticks, {k, p.n}});
            }
            g_gm.gpu_paths.clear();
        }
        std::sort(paths.rbegin(), paths.rend());
        std::string ps;
        for (size_t i = 0; i < std::min<size_t>(paths.size(), 8); ++i) {
            const auto& k = paths[i].second.first;
            ps += fmt::format(" [{:#x} {:#x} {:#x} {:#x}] {:.1f} ms/s {:.0f}/s", k[0], k[1], k[2], k[3],
                              double(paths[i].first) / 3.5e6 / sec, double(paths[i].second.second) / sec);
        }
        LOG_INFO(Render, "GameMain census GPU waits on Game:Main (MasterSemaphore::Wait, shadPS4.exe "
                         "offsets of the 4 callers; m1: [0xA000+site, early ASC overlap] - site 1 fault "
                         "dl main sem, 2 fault dl readback queue, 3 fault dl fast lane, 4 buffer dl sync, 5 "
                         "fault verify, 6 DownloadBufferMemory sync, 7 texture dl sync, 0 untagged):{}",
                 ps);
    }
    if (JobCensus()) {
        const u64 frames_now = presented_frames.load(std::memory_order_relaxed);
        const double fr = double(frames_now - g_gm.last_frames) / sec;
        g_gm.last_frames = frames_now;
        const double frame_ms = fr > 0 ? 1000.0 / fr : 0.0;
        const auto host_name = [](u64 addr) {
            HMODULE mod{};
            char name[64] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(addr), &mod)) {
                char path[MAX_PATH];
                if (GetModuleFileNameA(mod, path, MAX_PATH)) {
                    const char* base = std::strrchr(path, '\\');
                    std::snprintf(name, sizeof(name), "%s", base ? base + 1 : path);
                }
            }
            return fmt::format("{}+{:#x}", name, addr - reinterpret_cast<u64>(mod));
        };
        std::vector<std::pair<u64, u32>> steps;
        std::unordered_map<u32, u64> steps_host;
        std::map<std::pair<u32, u64>, std::pair<u64, u64>> steps_host_pages;
        u64 ns = 0, finds = 0, slot = 0, js = 0, ji = 0, jss = 0, jsi = 0;
        double busy_ms = 0, span_busy_ms = 0;
        std::vector<std::pair<u64, u64>> cls, cls_span;
        {
            std::scoped_lock lk{g_gm.rip_mutex};
            for (const auto& [s, c] : g_gm.steps) {
                steps.emplace_back(c, s);
            }
            steps_host = g_gm.steps_host;
            g_gm.steps_host.clear();
            steps_host_pages.swap(g_gm.steps_host_pages);
            ns = std::max<u64>(1, g_gm.step_samples);
            finds = g_gm.loop_slot_finds;
            slot = g_gm.loop_slot;
            js = g_gm.job_samples;
            ji = g_gm.job_idle;
            jss = g_gm.job_span_samples;
            jsi = g_gm.job_span_idle;
            busy_ms = double(g_gm.job_busy_cyc) / 3.5e6 / sec;
            span_busy_ms = double(g_gm.job_span_busy_cyc) / 3.5e6 / sec;
            for (const auto& [k, c] : g_gm.job_class) {
                cls.emplace_back(c, k);
            }
            for (const auto& [k, c] : g_gm.job_class_span) {
                cls_span.emplace_back(c, k);
            }
            g_gm.steps.clear();
            g_gm.step_samples = 0;
            g_gm.job_samples = g_gm.job_idle = g_gm.job_span_samples = g_gm.job_span_idle = 0;
            g_gm.job_busy_cyc = g_gm.job_span_busy_cyc = 0;
            g_gm.job_class.clear();
            g_gm.job_class_span.clear();
        }
        std::sort(steps.rbegin(), steps.rend());
        double span = 0.0;
        std::string ss;
        for (const auto& [c, s] : steps) {
            const double share = double(c) / double(ns);
            if (InRenderSpan(static_cast<u32>(s))) {
                span += share;
            }
        }
        for (size_t i = 0; i < std::min<size_t>(steps.size(), 24); ++i) {
            const double share = double(steps[i].first) / double(ns);
            ss += fmt::format(" {:#x}:{:.1f}%={:.2f}ms", steps[i].second, 100.0 * share, share * frame_ms);
        }
        // 2026-10-14: the emulator (host-code) part of the top steps, ms per frame.
        std::string hs;
        for (size_t i = 0; i < std::min<size_t>(steps.size(), 12); ++i) {
            const auto it = steps_host.find(steps[i].second);
            const double h = it == steps_host.end() ? 0.0 : double(it->second) / double(ns);
            hs += fmt::format(" {:#x}:{:.2f}ms", steps[i].second, h * frame_ms);
        }
        LOG_INFO(Render, "JobCensus steps host: emulator-code ms per frame inside each step:{}", hs);
        // r19c: the host pages behind the four steps with the most host time (ms per frame, a sample
        // RIP for llvm-symbolizer: module offset + the linked base 0x700000000000 for shadPS4.exe).
        {
            std::vector<std::pair<u64, u32>> by_host;
            for (const auto& [st, c] : steps_host) {
                by_host.emplace_back(c, st);
            }
            std::sort(by_host.rbegin(), by_host.rend());
            for (size_t i = 0; i < std::min<size_t>(by_host.size(), 4); ++i) {
                const u32 st = by_host[i].second;
                std::vector<std::tuple<u64, u64, u64>> pages; // samples, page, rip
                for (const auto& [key, v] : steps_host_pages) {
                    if (key.first == st) {
                        pages.emplace_back(v.first, key.second, v.second);
                    }
                }
                std::sort(pages.rbegin(), pages.rend());
                std::string ps;
                for (size_t k = 0; k < std::min<size_t>(pages.size(), 10); ++k) {
                    const auto& [c, page, rip] = pages[k];
                    ps += fmt::format(" {}:{:.3f}ms(rip {})", host_name(page),
                                      double(c) / double(ns) * frame_ms, host_name(rip));
                }
                LOG_INFO(Render, "JobCensus step host pages {:#x} ({:.2f} ms/frame host):{}", st,
                         double(by_host[i].first) / double(ns) * frame_ms, ps);
            }
        }
        LOG_INFO(Render,
                 "JobCensus steps: window {:.1f}s fps {:.1f} | Game:Main main-loop steps (return address of "
                 "the loop's current call; 1 = loop body, 0 = not in the loop; share of wall = ms per "
                 "frame):{} | render span (kick 0x29cd07 .. join 0x29cd3b) {:.1f}% = {:.2f} ms/frame | "
                 "loop slot {:#x}, finds {}",
                 sec, fr, ss, 100.0 * span, span * frame_ms, slot, finds);
        const auto split = [&](std::vector<std::pair<u64, u64>>& v, u64 busy_samples) {
            std::map<std::string, u64> mods;
            std::vector<std::pair<u64, u64>> pages;
            u64 game = 0, dly = 0, guest = 0;
            for (const auto& [c, k] : v) {
                if (k == 1) {
                    game = c;
                } else if (k == 2) {
                    dly = c;
                } else if (k == 3) {
                    guest = c;
                } else {
                    const std::string n = host_name(k);
                    mods[n.substr(0, n.find('+'))] += c;
                    pages.emplace_back(c, k);
                }
            }
            std::sort(pages.rbegin(), pages.rend());
            const double b = double(std::max<u64>(1, busy_samples));
            std::string s = fmt::format("game code {:.1f}% delay() spin {:.1f}% other guest {:.1f}% |",
                                        100.0 * game / b, 100.0 * dly / b, 100.0 * guest / b);
            for (const auto& [n, c] : mods) {
                s += fmt::format(" {} {:.1f}%", n, 100.0 * c / b);
            }
            s += " | host pages:";
            for (size_t i = 0; i < std::min<size_t>(pages.size(), 14); ++i) {
                s += fmt::format(" {}:{:.1f}%", host_name(pages[i].second), 100.0 * pages[i].first / b);
            }
            return s;
        };
        std::sort(cls.rbegin(), cls.rend());
        std::sort(cls_span.rbegin(), cls_span.rend());
        const u32 nt = g_gm.nthreads.load(std::memory_order_relaxed);
        LOG_INFO(Render,
                 "JobCensus jobs: {} JobWorker threads, busy {:.0f} ms/s = {:.2f} ms/frame (all threads, "
                 "QueryThreadCycleTime at 3.5 GHz), samples {} idle {:.1f}% | busy samples: {}",
                 nt, busy_ms, fr > 0 ? busy_ms / fr : 0.0, js, js ? 100.0 * ji / js : 0.0,
                 split(cls, js - ji));
        LOG_INFO(Render,
                 "JobCensus jobs during the render span: samples {} idle {:.1f}%, busy {:.0f} ms/s = "
                 "{:.2f} ms/frame | busy samples: {}",
                 jss, jss ? 100.0 * jsi / jss : 0.0, span_busy_ms, fr > 0 ? span_busy_ms / fr : 0.0,
                 split(cls_span, jss - jsi));
        const double gm_cyc = double(g_gm.nrec_cyc[0].exchange(0));
        const double ot_cyc = double(g_gm.nrec_cyc[1].exchange(0));
        const double gm_n = double(g_gm.nrec_n[0].exchange(0));
        const double ot_n = double(g_gm.nrec_n[1].exchange(0));
        LOG_INFO(Render,
                 "JobCensus NativeRec (NativeRecDraw + NativeRecShader, rdtsc): Game:Main {:.2f} ms/frame "
                 "({:.0f} calls/frame, {:.0f} cycles each) | other threads {:.2f} ms/frame ({:.0f} "
                 "calls/frame, {:.0f} cycles each)",
                 fr > 0 ? gm_cyc / 3.5e6 / sec / fr : 0.0, fr > 0 ? gm_n / sec / fr : 0.0,
                 gm_n > 0 ? gm_cyc / gm_n : 0.0, fr > 0 ? ot_cyc / 3.5e6 / sec / fr : 0.0,
                 fr > 0 ? ot_n / sec / fr : 0.0, ot_n > 0 ? ot_cyc / ot_n : 0.0);
    }
}

namespace {
#if defined(__clang__)
// Keep this collection hook out of profiles in both training and profile-use builds.
__attribute__((no_profile_instrument_function))
#endif
void ReportLoop() {
    Common::SetCurrentThreadName("shadPS4:FaultMonitor");
    LOG_INFO(Render, "FaultMonitor: reporting every {} ms",
             static_cast<long long>(ReportInterval().count()));

    std::ofstream csv = OpenCsv();
    const auto started = std::chrono::steady_clock::now();
    auto prev_time = started;
    Sample prev = Read();
#if defined(__LLVM_INSTR_PROFILE_GENERATE)
    bool pgo_reset = false;
    bool pgo_dumped = false;
    double pgo_reset_elapsed = 0.0;
#endif

    while (true) {
        std::unique_lock lk(stop_mutex);
        if (stop_cv.wait_for(lk, ReportInterval(), [] { return stop_requested; })) {
            break;
        }
        lk.unlock();

        const auto now = std::chrono::steady_clock::now();
        const Sample cur = Read();

        const double dt = std::chrono::duration<double>(now - prev_time).count();
        // Thread-wait columns: rdtsc ticks to ms with this interval's own tick rate.
        static u64 prev_tsc = __rdtsc();
        const u64 tsc_now = __rdtsc();
        const double ms_per_tick =
            tsc_now > prev_tsc + 1000000 ? dt * 1e3 / static_cast<double>(tsc_now - prev_tsc)
                                         : 0.0; // 0 on the first row (no interval yet)
        prev_tsc = tsc_now;
        const double per_s = dt > 0.0 ? 1.0 / dt : 0.0;
        const double front_idle_ms =
            static_cast<double>(front_idle_ns.exchange(0, std::memory_order_relaxed)) / 1e6 * per_s;
        const double front_wait_ms =
            static_cast<double>(front_wait_ticks.exchange(0, std::memory_order_relaxed)) *
            ms_per_tick * per_s;
        const double back_idle_ms =
            static_cast<double>(back_idle_ticks.exchange(0, std::memory_order_relaxed)) *
            ms_per_tick * per_s;
        const double rf_guest_ms =
            static_cast<double>(rf_guest_ns.exchange(0, std::memory_order_relaxed)) / 1e6 * per_s;
        const double elapsed = std::chrono::duration<double>(now - started).count();
        if (dt <= 0.0) {
            continue;
        }

        const u64 d_read = cur.read - prev.read;
        const u64 d_write = cur.write - prev.write;
        const u64 d_jit = cur.jit - prev.jit;
        const u64 d_frames = cur.frames - prev.frames;
        const u64 d_read_ns = cur.read_ns - prev.read_ns;
        const u64 d_write_ns = cur.write_ns - prev.write_ns;
        const u64 d_jit_ns = cur.jit_ns - prev.jit_ns;
        const u64 d_adds = cur.adds - prev.adds;
        const u64 d_subs = cur.subtracts - prev.subtracts;
        const u64 d_submits = cur.submits - prev.submits;
        const u64 d_waits = cur.waits - prev.waits;
        const u64 d_wait_ns = cur.wait_ns - prev.wait_ns;

        // The decisive split: if submits/frame stays flat while gpu_wait_ms/frame grows, the GPU
        // side is getting slower rather than being asked to do more.
        const double submits_per_frame =
            d_frames ? static_cast<double>(d_submits) / static_cast<double>(d_frames) : 0.0;
        const double waits_per_frame =
            d_frames ? static_cast<double>(d_waits) / static_cast<double>(d_frames) : 0.0;
        const double wait_ms_per_frame =
            d_frames ? static_cast<double>(d_wait_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;

        const u64 d_wrm = cur.wrm_count - prev.wrm_count;
        const u64 d_wrm_ns = cur.wrm_ns - prev.wrm_ns;
        const u64 d_vo = cur.vo_count - prev.vo_count;
        const u64 d_vo_ns = cur.vo_ns - prev.vo_ns;
        const double wrm_per_frame =
            d_frames ? static_cast<double>(d_wrm) / static_cast<double>(d_frames) : 0.0;
        const double wrm_ms_per_frame =
            d_frames ? static_cast<double>(d_wrm_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        const double vo_per_frame =
            d_frames ? static_cast<double>(d_vo) / static_cast<double>(d_frames) : 0.0;
        const double vo_ms_per_frame =
            d_frames ? static_cast<double>(d_vo_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;

        const u64 d_pres = cur.pres_count - prev.pres_count;
        const u64 d_pres_ns = cur.pres_ns - prev.pres_ns;
        const u64 d_pending = cur.pending - prev.pending;
        const double present_ms_each = d_pres ? static_cast<double>(d_pres_ns) / d_pres / 1e6 : 0.0;
        const double present_ms_per_frame =
            d_frames ? static_cast<double>(d_pres_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        const double avg_pending =
            d_pres ? static_cast<double>(d_pending) / static_cast<double>(d_pres) : 0.0;

        const auto per_present = [&](u64 now, u64 before) {
            return d_pres ? static_cast<double>(now - before) / d_pres / 1e6 : 0.0;
        };
        const double acq_ms = per_present(cur.acq_ns, prev.acq_ns);
        const double flush_ms = per_present(cur.flush_ns, prev.flush_ns);
        const double lock_ms = per_present(cur.lock_ns, prev.lock_ns);
        const double khr_ms = per_present(cur.khr_ns, prev.khr_ns);

        const u64 d_rf = cur.rf_count - prev.rf_count;
        const auto per_rf = [&](u64 now, u64 before) {
            return d_rf ? static_cast<double>(now - before) / d_rf / 1e6 : 0.0;
        };
        const double rf_queue_ms = per_rf(cur.rf_queue_ns, prev.rf_queue_ns);
        const double rf_fence_ms = per_rf(cur.rf_fence_ns, prev.rf_fence_ns);

        // Readback downloads: one submit-and-fence-wait each, so this is the bulk of
        // submits_per_frame. The kB column is the cost side of widening them.
        const u64 d_dl = cur.dl_calls - prev.dl_calls;
        const u64 d_dl_bytes = cur.dl_bytes - prev.dl_bytes;
        const double dl_per_frame =
            d_frames ? static_cast<double>(d_dl) / static_cast<double>(d_frames) : 0.0;
        const double dl_kb_per_frame =
            d_frames ? static_cast<double>(d_dl_bytes) / static_cast<double>(d_frames) / 1024.0
                     : 0.0;

        // Per-draw CPU, measured rather than inferred from a stack scan. The *_us_each columns are
        // microseconds per draw call; draw_ms_per_frame is what all three cost per rendered frame.
        const u64 d_draws = cur.draws - prev.draws;
        const u64 d_pipe = cur.d_pipeline_ns - prev.d_pipeline_ns;
        const u64 d_bind = cur.d_bindres_ns - prev.d_bindres_ns;
        const u64 d_begin = cur.d_begin_ns - prev.d_begin_ns;
        const u64 d_fdepth = cur.d_finddepth_ns - prev.d_finddepth_ns;
        const auto per_draw_us = [&](u64 ns) {
            return d_draws ? static_cast<double>(ns) / static_cast<double>(d_draws) / 1000.0 : 0.0;
        };
        const double draws_per_frame =
            d_frames ? static_cast<double>(d_draws) / static_cast<double>(d_frames) : 0.0;
        const double pipeline_us = per_draw_us(d_pipe);
        const double bindres_us = per_draw_us(d_bind);
        const double beginrender_us = per_draw_us(d_begin);
        const double finddepth_us = per_draw_us(d_fdepth);
        const double bindbuf_us = per_draw_us(cur.d_bindbuf_ns - prev.d_bindbuf_ns);
        const double bindtex_us = per_draw_us(cur.d_bindtex_ns - prev.d_bindtex_ns);
        const double bb_find_us = per_draw_us(cur.bb_find_ns - prev.bb_find_ns);
        const double bb_obtain_us = per_draw_us(cur.bb_obtain_ns - prev.bb_obtain_ns);
        // All six are normalised per DRAW, not per call, so they add up against bb_obtain_us_each
        // directly. ob_stream_per_draw / ob_sync_per_draw say how the bindings split between the
        // two halves, which is what decides where a cut is worth making.
        const double ob_query_us = per_draw_us(cur.ob_query_ns - prev.ob_query_ns);
        const double ob_stream_us = per_draw_us(cur.ob_stream_ns - prev.ob_stream_ns);
        const double ob_sync_us = per_draw_us(cur.ob_sync_ns - prev.ob_sync_ns);
        const double ob_stream_per_draw =
            d_draws ? static_cast<double>(cur.ob_stream_calls - prev.ob_stream_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        const double ob_sync_per_draw =
            d_draws ? static_cast<double>(cur.ob_sync_calls - prev.ob_sync_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        const double ob_stream_kb_per_draw =
            d_draws ? static_cast<double>(cur.ob_stream_bytes - prev.ob_stream_bytes) /
                          static_cast<double>(d_draws) / 1024.0
                    : 0.0;
#if defined(__LLVM_INSTR_PROFILE_GENERATE)
        // 2026-10-14: or > 20 FPS - with the lean workers drawing (r9 / r12) the back's draw counter
        // reads ~190 per frame in gameplay, so the reset (and the dump) never happened.
        const double pgo_fps = dt > 0.0 ? static_cast<double>(d_frames) / dt : 0.0;
        if (!pgo_reset && elapsed >= 180.0 && (draws_per_frame > 1000.0 || pgo_fps > 20.0)) {
            __llvm_profile_reset_counters();
            pgo_reset = true;
            pgo_reset_elapsed = elapsed;
            LOG_INFO(Render, "PGO: counters reset at t={:.1f}s draws/frame={:.0f}", elapsed,
                     draws_per_frame);
        } else if (pgo_reset && !pgo_dumped && elapsed - pgo_reset_elapsed >= 120.0) {
            const int dump_result = __llvm_profile_dump();
            pgo_dumped = true;
            LOG_INFO(Render, "PGO: profile dumped at t={:.1f}s reset={:.1f}s rc={}", elapsed,
                     pgo_reset_elapsed, dump_result);
        }
#endif
        // Phases of ForEachUploadRange, per draw, so they add against ob_sync_us_each.
        const double up_scan_us = per_draw_us(cur.up_scan_ns - prev.up_scan_ns);
        const double up_copy_us = per_draw_us(cur.up_copy_ns - prev.up_copy_ns);
        const double up_mark_us = per_draw_us(cur.up_mark_ns - prev.up_mark_ns);
        const double up_written_per_draw =
            d_draws ? static_cast<double>(cur.up_written_calls - prev.up_written_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        // Protection changes are per draw for rate, but per CALL for cost - a VirtualProtect is
        // a syscall and its individual cost is the thing worth knowing.
        const u64 d_protw = cur.prot_w_calls - prev.prot_w_calls;
        const u64 d_protr = cur.prot_r_calls - prev.prot_r_calls;
        const double protw_per_draw =
            d_draws ? static_cast<double>(d_protw) / static_cast<double>(d_draws) : 0.0;
        const double protr_per_draw =
            d_draws ? static_cast<double>(d_protr) / static_cast<double>(d_draws) : 0.0;
        const double protw_us =
            d_protw ? static_cast<double>(cur.prot_w_ns - prev.prot_w_ns) / d_protw / 1000.0 : 0.0;
        const double protr_us =
            d_protr ? static_cast<double>(cur.prot_r_ns - prev.prot_r_ns) / d_protr / 1000.0 : 0.0;
        const double bt_findimage_us = per_draw_us(cur.bt_findimage_ns - prev.bt_findimage_ns);
        const double bt_view_us = per_draw_us(cur.bt_view_ns - prev.bt_view_ns);
        const double bt_sampler_us = per_draw_us(cur.bt_sampler_ns - prev.bt_sampler_ns);
        const u64 d_bar = cur.bar_calls - prev.bar_calls;
        const double bar_per_draw =
            d_draws ? static_cast<double>(d_bar) / static_cast<double>(d_draws) : 0.0;
        const double bar_partial_pct =
            d_bar ? static_cast<double>(cur.bar_partial - prev.bar_partial) / d_bar * 100.0 : 0.0;
        const double bar_iters_per_call =
            d_bar ? static_cast<double>(cur.bar_iters - prev.bar_iters) / d_bar : 0.0;
        const double bar_emitted_per_call =
            d_bar ? static_cast<double>(cur.bar_emitted - prev.bar_emitted) / d_bar : 0.0;
        const double bt_findtex_us = per_draw_us(cur.bt_findtex_ns - prev.bt_findtex_ns);
        const double bt_transit_us = per_draw_us(cur.bt_transit_ns - prev.bt_transit_ns);
        const double d_pre_us = per_draw_us(cur.d_pre_ns - prev.d_pre_ns);
        const double d_vtxbuf_us = per_draw_us(cur.d_vtxbuf_ns - prev.d_vtxbuf_ns);
        const double d_descwrite_us = per_draw_us(cur.d_descwrite_ns - prev.d_descwrite_ns);
        const double d_dynstate_us = per_draw_us(cur.d_dynstate_ns - prev.d_dynstate_ns);
        const double d_schedbegin_us = per_draw_us(cur.d_schedbegin_ns - prev.d_schedbegin_ns);
        const double d_emit_us = per_draw_us(cur.d_emit_ns - prev.d_emit_ns);
        // Per FRAME, not per draw: this is frame-level work that a per-draw column would hide.
        const auto per_frame_ms = [&](u64 ns) {
            return d_frames ? static_cast<double>(ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        };
        const double cp_wait_ms = per_frame_ms(cur.cp_wait_ns_v - prev.cp_wait_ns_v);
        const double cp_active_ms = per_frame_ms(cur.cp_active_ns_v - prev.cp_active_ns_v);
        const u64 d_disp = cur.disp_count - prev.disp_count;
        const double disp_per_frame =
            d_frames ? static_cast<double>(d_disp) / static_cast<double>(d_frames) : 0.0;
        const double disp_us =
            d_disp ? static_cast<double>(cur.disp_ns - prev.disp_ns) / d_disp / 1000.0 : 0.0;
        const double sc_map_us = per_draw_us(cur.sc_map_ns - prev.sc_map_ns);
        const double sc_valid_us = per_draw_us(cur.sc_valid_ns - prev.sc_valid_ns);
        const double sc_copy_us = per_draw_us(cur.sc_copy_ns - prev.sc_copy_ns);
        const double sc_commit_us = per_draw_us(cur.sc_commit_ns - prev.sc_commit_ns);
        const double draw_ms_per_frame =
            d_frames ? static_cast<double>(d_pipe + d_bind + d_begin) /
                           static_cast<double>(d_frames) / 1e6
                     : 0.0;

        const double read_rate = static_cast<double>(d_read) / dt;
        const double write_rate = static_cast<double>(d_write) / dt;
        const double jit_rate = static_cast<double>(d_jit) / dt;
        const double fps = static_cast<double>(d_frames) / dt;

        // Does the framerate fall because there are MORE faults per frame, or because each
        // fault costs MORE? These two columns separate them.
        const double per_frame =
            d_frames ? static_cast<double>(d_read + d_write) / static_cast<double>(d_frames) : 0.0;
        const double read_us = d_read ? static_cast<double>(d_read_ns) / d_read / 1000.0 : 0.0;
        const double write_us = d_write ? static_cast<double>(d_write_ns) / d_write / 1000.0 : 0.0;
        // Handler time as a share of one thread's wall time over the interval.
        const double handler_pct =
            static_cast<double>(d_read_ns + d_write_ns + d_jit_ns) / (dt * 1e9) * 100.0;

        // Of that handler time, how much was billed to the command-processor thread - the one
        // whose activity IS the frame. CP nanoseconds come off the frame; guest-thread
        // nanoseconds run concurrently and cost nothing.
        const u64 d_cp_ns = (cur.cp_read_ns_v - prev.cp_read_ns_v) +
                            (cur.cp_write_ns_v - prev.cp_write_ns_v);
        const u64 d_all_ns = d_read_ns + d_write_ns + d_jit_ns;
        const double cpfault_ms_per_frame =
            d_frames ? static_cast<double>(d_cp_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        const double cpfault_pct =
            d_all_ns ? static_cast<double>(d_cp_ns) / static_cast<double>(d_all_ns) * 100.0 : 0.0;

        // Shape of the upload scan: is it walking structure to find nothing?
        const u64 d_up_calls = cur.up_calls - prev.up_calls;
        const double up_empty_pct =
            d_up_calls ? static_cast<double>(cur.up_empty - prev.up_empty) /
                             static_cast<double>(d_up_calls) * 100.0
                       : 0.0;
        const double up_managers_per_call =
            d_up_calls ? static_cast<double>(cur.up_managers - prev.up_managers) /
                             static_cast<double>(d_up_calls)
                       : 0.0;
        const double up_ranges_per_call =
            d_up_calls ? static_cast<double>(cur.up_ranges - prev.up_ranges) /
                             static_cast<double>(d_up_calls)
                       : 0.0;
        const u64 d_up_managers = cur.up_managers - prev.up_managers;
        const double up_skipped_pct =
            d_up_managers ? static_cast<double>(cur.up_skipped - prev.up_skipped) /
                                static_cast<double>(d_up_managers) * 100.0
                          : 0.0;
        const double up_kb_per_call =
            d_up_calls ? static_cast<double>(cur.up_bytes - prev.up_bytes) /
                             static_cast<double>(d_up_calls) / 1024.0
                       : 0.0;
        const u64 d_up_fast = cur.up_fastpath - prev.up_fastpath;
        const double up_fastpath_pct =
            (d_up_calls + d_up_fast) ? static_cast<double>(d_up_fast) /
                                           static_cast<double>(d_up_calls + d_up_fast) * 100.0
                                     : 0.0;
        const u64 d_up_memo = cur.up_memohit - prev.up_memohit;
        const u64 d_up_total = d_up_calls + d_up_fast + d_up_memo;
        const double up_memohit_pct =
            d_up_total ? static_cast<double>(d_up_memo) / static_cast<double>(d_up_total) * 100.0
                       : 0.0;

        LOG_INFO(Render,
                 "FaultRate: t={:.0f}s fps={:.1f} read={:.0f}/s write={:.0f}/s jit={:.0f}/s "
                 "| per_frame={:.0f} read={:.2f}us write={:.2f}us handler={:.1f}% of wall "
                 "| ranges frags={} add={:.0f}/s sub={:.0f}/s "
                 "| gpu submits/frame={:.1f} waits/frame={:.1f} wait={:.1f}ms/frame "
                 "| waitregmem {:.1f}/frame {:.1f}ms/frame volabel {:.1f}/frame {:.1f}ms/frame "
                 "| present {:.2f}ms each {:.1f}ms/frame pending={:.2f} "
                 "| acquire={:.2f}ms flush={:.2f}ms submitlock={:.2f}ms presentKHR={:.2f}ms "
                 "| renderframe queue={:.2f}ms fence={:.2f}ms "
                 "| downloads {:.1f}/frame {:.1f}kB/frame "
                 "| draws {:.0f}/frame pipeline={:.2f}us bindres={:.2f}us begin={:.2f}us "
                 "(finddepth={:.2f}us bindbuf={:.2f}us [find={:.2f} obtain={:.2f}] bindtex={:.2f}us) = {:.1f}ms/frame "
                 "| obtain query={:.2f}us stream={:.2f}us sync={:.2f}us "
                 "({:.2f} stream + {:.2f} sync per draw, {:.1f}kB/draw copied) "
                 "| upload scan={:.2f}us copy={:.2f}us mark={:.2f}us ({:.2f} written/draw) "
                 "| protect w={:.3f}/draw {:.2f}us each r={:.3f}/draw {:.2f}us each "
                 "| streamcopy map={:.2f}us valid={:.2f}us copy={:.2f}us commit={:.2f}us "
                 "| bindtex findimage={:.2f}us view={:.2f}us sampler={:.2f}us "
                 "| drawtail pre={:.2f} vtxbuf={:.2f} descwrite={:.2f} dynstate={:.2f} "
                 "schedbegin={:.2f} emit={:.2f}us "
                 "| CP wait={:.1f}ms/frame active={:.1f}ms/frame dispatch {:.0f}/frame {:.2f}us each "
                 "| view split findtex={:.2f}us transit={:.2f}us "
                 "| barriers {:.2f}/draw {:.0f}% partial, {:.1f} iters {:.2f} emitted per call "
                 "| totals read={} write={} jit={} frames={} unhandled(r/w)={}/{}",
                 elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                 handler_pct, cur.fragments, static_cast<double>(d_adds) / dt,
                 static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                 wait_ms_per_frame, wrm_per_frame, wrm_ms_per_frame, vo_per_frame, vo_ms_per_frame,
                 present_ms_each, present_ms_per_frame, avg_pending, acq_ms, flush_ms, lock_ms,
                 khr_ms, rf_queue_ms, rf_fence_ms, dl_per_frame, dl_kb_per_frame, draws_per_frame,
                 pipeline_us, bindres_us, beginrender_us, finddepth_us, bindbuf_us, bb_find_us,
                 bb_obtain_us, bindtex_us,
                 draw_ms_per_frame, ob_query_us, ob_stream_us, ob_sync_us, ob_stream_per_draw,
                 ob_sync_per_draw, ob_stream_kb_per_draw, up_scan_us, up_copy_us, up_mark_us,
                 up_written_per_draw, protw_per_draw, protw_us, protr_per_draw, protr_us, sc_map_us,
                 sc_valid_us, sc_copy_us, sc_commit_us, bt_findimage_us, bt_view_us, bt_sampler_us,
                 d_pre_us, d_vtxbuf_us, d_descwrite_us, d_dynstate_us, d_schedbegin_us, d_emit_us,
                 cp_wait_ms, cp_active_ms, disp_per_frame, disp_us, bt_findtex_us, bt_transit_us,
                 bar_per_draw, bar_partial_pct, bar_iters_per_call, bar_emitted_per_call,
                 cur.read,
                 cur.write, cur.jit, cur.frames, cur.read_unhandled, cur.write_unhandled);

        if (d_read > 0) {
            LOG_INFO(Render, "FaultPages: {}", TopReadFaultPages(6));
        }
        GameMainCensusMaybeLog();
        if (VideoCore::g_epoch_defer_upload && d_frames > 0) {
            static u64 prev_epoch = 0, prev_syncs = 0, prev_pages = 0;
            const u64 epoch = VideoCore::g_submit_epoch.load(std::memory_order_relaxed);
            const u64 syncs = deferred_syncs.load(std::memory_order_relaxed);
            const u64 pages = deferred_pages.load(std::memory_order_relaxed);
            const double f = static_cast<double>(d_frames);
            static std::array<u64, 256> prev_reasons{};
            std::vector<std::pair<u64, u32>> reasons;
            for (u32 i = 0; i < 256; ++i) {
                const u64 v = VideoCore::g_submit_epoch_reasons[i].load(std::memory_order_relaxed);
                if (v != prev_reasons[i]) {
                    reasons.emplace_back(v - prev_reasons[i], i);
                }
                prev_reasons[i] = v;
            }
            std::sort(reasons.rbegin(), reasons.rend());
            std::string top;
            for (size_t i = 0; i < std::min<size_t>(5, reasons.size()); ++i) {
                top += fmt::format(" op{:#x}={:.1f}", reasons[i].second, reasons[i].first / f);
            }
            LOG_INFO(Render,
                     "EpochDefer: epochs/frame={:.1f} deferred_syncs/frame={:.1f} "
                     "deferred_pages/frame={:.1f} faults/frame={:.1f} bumps/frame by opcode:{}",
                     (epoch - prev_epoch) / f, (syncs - prev_syncs) / f, (pages - prev_pages) / f,
                     d_write / f, top);
            prev_epoch = epoch;
            prev_syncs = syncs;
            prev_pages = pages;
        }
        {
            static u64 prev_w = 0, prev_k = 0, prev_s = 0, prev_sns = 0;
            const u64 w = heap_block_widened.load(std::memory_order_relaxed);
            const u64 k = heap_block_kept.load(std::memory_order_relaxed);
            const u64 sn = invalidate_settles.load(std::memory_order_relaxed);
            const u64 sns = invalidate_settle_ns.load(std::memory_order_relaxed);
            if (d_frames > 0 && (w != prev_w || k != prev_k || sn != prev_s)) {
                const double f = static_cast<double>(d_frames);
                LOG_INFO(Render,
                         "HeapWriteBlock: t={:.0f}s widened/frame={:.1f} kept_single/frame={:.1f} "
                         "settles/frame={:.2f} settle_ms/frame={:.3f} settle_thread_ms_pct_of_wall={:.2f}% "
                         "write_faults/frame={:.1f} (totals widened={} kept={} settles={})",
                         elapsed, (w - prev_w) / f, (k - prev_k) / f, (sn - prev_s) / f,
                         (sns - prev_sns) / f / 1e6,
                         dt > 0 ? (sns - prev_sns) / (dt * 1e9) * 100.0 : 0.0, d_write / f, w, k,
                         sn);
            }
            prev_w = w;
            prev_k = k;
            prev_s = sn;
            prev_sns = sns;
        }
        if (WriteFaultPagesEnabled()) {
            static u64 prev_gfx = 0, prev_asc = 0;
            const u64 gfx = gfx_submits.load(std::memory_order_relaxed);
            const u64 asc = asc_submits.load(std::memory_order_relaxed);
            LOG_INFO(Render, "WriteFaultPages: {}",
                     WriteFaultPageReport(d_frames, gfx - prev_gfx, asc - prev_asc));
            prev_gfx = gfx;
            prev_asc = asc;
        }

        if (csv) {
            csv << fmt::format(
                "{:.1f},{:.2f},{:.1f},{:.1f},{:.1f},{:.1f},{:.4f},{:.4f},{:.2f},{},{},{},{},{},{},"
                "{},{:.1f},{:.1f},{:.2f},{:.2f},{:.3f},{:.2f},{:.3f},{:.2f},{:.3f},"
                "{:.3f},{:.3f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.2f},{:.1f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.4f},{:.3f},{:.4f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.2f},{:.2f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.1f},{:.2f},{:.3f},{:.3f},{:.1f},{:.2f},{:.3f},{:.3f},{:.2f},{:.1f},{:.2f},{:.2f},{:.2f},{},{},{},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.2f}\n",
                elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                handler_pct, cur.read, cur.read_unhandled, cur.write, cur.write_unhandled, cur.jit,
                cur.frames, cur.fragments, static_cast<double>(d_adds) / dt,
                static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                wait_ms_per_frame, wrm_per_frame, wrm_ms_per_frame, vo_per_frame, vo_ms_per_frame,
                present_ms_each, present_ms_per_frame, avg_pending, acq_ms, flush_ms, lock_ms,
                khr_ms, rf_queue_ms, rf_fence_ms, dl_per_frame, dl_kb_per_frame, draws_per_frame,
                pipeline_us, bindres_us, beginrender_us, finddepth_us, draw_ms_per_frame, bindbuf_us,
                bindtex_us, bb_find_us, bb_obtain_us, ob_query_us, ob_stream_us, ob_sync_us,
                ob_stream_per_draw, ob_sync_per_draw, ob_stream_kb_per_draw, up_scan_us, up_copy_us,
                up_mark_us, up_written_per_draw, protw_per_draw, protw_us, protr_per_draw, protr_us,
                sc_map_us, sc_valid_us, sc_copy_us, sc_commit_us, bt_findimage_us, bt_view_us,
                bt_sampler_us, d_pre_us, d_vtxbuf_us, d_descwrite_us, d_dynstate_us,
                d_schedbegin_us, d_emit_us, cp_wait_ms, cp_active_ms, disp_per_frame, disp_us,
                bt_findtex_us, bt_transit_us, bar_per_draw, bar_partial_pct, bar_iters_per_call,
                bar_emitted_per_call, cpfault_ms_per_frame, cpfault_pct, up_empty_pct,
                up_managers_per_call, up_ranges_per_call, up_skipped_pct, up_kb_per_call, up_fastpath_pct, up_memohit_pct,
                static_cast<double>(frame_max_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
                frame_over_10ms.exchange(0, std::memory_order_relaxed),
                frame_over_17ms.exchange(0, std::memory_order_relaxed),
                frame_over_33ms.exchange(0, std::memory_order_relaxed),
                static_cast<double>(frame_slow_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
                front_idle_ms, front_wait_ms, back_idle_ms, rf_guest_ms,
                d_frames ? static_cast<double>(sync_upload_bytes.exchange(0, std::memory_order_relaxed)) /
                               1024.0 / static_cast<double>(d_frames)
                         : 0.0,
                d_frames ? static_cast<double>(sync_upload_calls.exchange(0, std::memory_order_relaxed)) /
                               static_cast<double>(d_frames)
                         : 0.0);
            csv.flush(); // a killed process must not lose the curve
        }

        prev = cur;
        prev_time = now;
    }
}

// Joined at static destruction, which requests the stop and wakes the thread immediately.
struct Reporter {
    Reporter() : thread(ReportLoop) {
        LOG_INFO(Render, "FaultMonitor: timers={} details={} (disabled columns are unavailable)",
                 SHADPS4_FAULT_TIMERS != 0, DetailCountersEnabled);
    }
    ~Reporter() {
        {
            std::scoped_lock lk(stop_mutex);
            stop_requested = true;
        }
        stop_cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }
    std::thread thread;
};

} // namespace

void Start() {
    static Reporter reporter;
    (void)reporter;
}

} // namespace Common::FaultMonitor
