// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/perf_stats.h"

// Optional, bounded present-side sampling. Counters are process-wide deltas observed at a
// present boundary, not exclusive time owned by that frame. No per-packet or fault-path hooks.
namespace Common::Perf::CausalFrame {
constexpr size_t RingSize = 16;
constexpr u64 MaxRows = 4096;
constexpr u64 MaxBytes = 4 * 1024 * 1024;
constexpr double DefaultSlowMs = 50.0;

inline constexpr std::array<Counter, 21> Fields{
    GpuBusyNs, GpuSpinNs, AscBusyNs, RecorderBusyNs, RecorderSyncNs,
    AsyncUploadBusyNs, AsyncUploadWaitNs, ReadbackStallNs, ReadbackQueueNs,
    ReadbackSyncNs, ReadbackAsyncWaitNs, GpuWaitNs, BufferUploadNs,
    ImageRefreshNs, GpuProfileBusyNs, GpuProfileReadbackNs,
    ReadbackProducerDone, ReadbackProducerSubmitted, ReadbackProducerQueued,
    ReadbackProducerCurrent, ReadbackReads,
};

struct Row {
    u64 seq{};
    u64 guest_frame{};
    u64 present_start_ns{};
    u64 acquire_end_ns{};
    u64 flush_start_ns{};
    u64 flush_end_ns{};
    u64 present_end_ns{};
    u64 interval_ns{};
    std::array<u64, Fields.size()> delta{};
};

class Capture {
public:
    Capture() {
        const char* path = std::getenv("ISS_CAUSAL_FRAME_CSV");
        if (!path || !*path) return;
        file = std::fopen(path, "wb");
        if (!file) return;
        if (const char* value = std::getenv("ISS_CAUSAL_SLOW_MS"); value && *value) {
            char* end{};
            const double requested = std::strtod(value, &end);
            if (end != value && *end == '\0' && std::isfinite(requested)) {
                slow_ms = std::clamp(requested, 10.0, 1000.0);
            }
        }
        slow_ns = static_cast<u64>(slow_ms * 1'000'000.0);
        std::setvbuf(file, nullptr, _IOFBF, 64 * 1024);
        static constexpr char Header[] =
            "schema,event,clock,thread,owner,qid,producer,producer_generation,producer_tick,"
            "present_frame,guest_frame,present_start_ns,present_end_ns,present_interval_ns,trigger_ms,"
            "acquire_ns,record_ns,flush_ns,present_call_ns,cp_work_ns,cp_spin_ns,asc_work_ns,"
            "recorder_work_ns,recorder_wait_ns,upload_worker_work_ns,upload_worker_wait_ns,"
            "readback_stall_ns,readback_queue_ns,readback_sync_ns,readback_wait_ns,"
            "gpu_wait_ns,buffer_upload_ns,image_refresh_ns,gpu_completed_work_ns,"
            "gpu_completed_readback_ns,producer_done,producer_submitted,producer_queued,"
            "producer_current,readback_reads\n";
        std::fwrite(Header, 1, sizeof(Header) - 1, file);
        bytes = sizeof(Header) - 1;
    }
    ~Capture() { if (file) std::fclose(file); }
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void Observe(u64 guest_frame, u64 start, u64 acquired, u64 flush_start,
                 u64 flushed, u64 end) {
        if (!file || truncated) return;
        Row row{};
        row.seq = ++sequence;
        row.guest_frame = guest_frame;
        row.present_start_ns = start;
        row.acquire_end_ns = acquired;
        row.flush_start_ns = flush_start;
        row.flush_end_ns = flushed;
        row.present_end_ns = end;
        row.interval_ns = previous_end ? end - previous_end : 0;
        previous_end = end;
        for (size_t i = 0; i < Fields.size(); ++i) {
            const u64 total = Total(Fields[i]);
            row.delta[i] = row.seq == 1 ? 0 : total - previous[i];
            previous[i] = total;
        }
        ring[(row.seq - 1) % RingSize] = row;
        if (row.interval_ns < slow_ns) return;
        const u64 first = row.seq > RingSize ? row.seq - RingSize + 1 : 1;
        for (u64 seq = first; seq <= row.seq && !truncated; ++seq) {
            if (seq > last_written) Write(ring[(seq - 1) % RingSize]);
        }
        std::fflush(file); // Slow frames are rare; retain evidence if the process crashes.
    }

private:
    void Write(const Row& row) {
        if (rows >= MaxRows || bytes + 1024 >= MaxBytes) {
            static constexpr char Marker[] = "1,truncated,steady,present,present,all,mixed,,,0,0\n";
            std::fwrite(Marker, 1, sizeof(Marker) - 1, file);
            truncated = true;
            return;
        }
        char line[1024];
        int n = std::snprintf(line, sizeof(line),
            "1,frame,steady,present,present,all,mixed,,,%llu,%llu,%llu,%llu,%llu,%.3f,"
            "%llu,%llu,%llu,%llu",
            static_cast<unsigned long long>(row.seq),
            static_cast<unsigned long long>(row.guest_frame),
            static_cast<unsigned long long>(row.present_start_ns),
            static_cast<unsigned long long>(row.present_end_ns),
            static_cast<unsigned long long>(row.interval_ns),
            slow_ms,
            static_cast<unsigned long long>(row.acquire_end_ns - row.present_start_ns),
            static_cast<unsigned long long>(row.flush_start_ns - row.acquire_end_ns),
            static_cast<unsigned long long>(row.flush_end_ns - row.flush_start_ns),
            static_cast<unsigned long long>(row.present_end_ns - row.flush_end_ns));
        for (u64 value : row.delta) {
            if (n < 0 || static_cast<size_t>(n) >= sizeof(line)) break;
            const int written = std::snprintf(line + n, sizeof(line) - n, ",%llu",
                                              static_cast<unsigned long long>(value));
            if (written < 0) return;
            n += written;
        }
        if (n < 0 || static_cast<size_t>(n + 1) >= sizeof(line)) return;
        line[n++] = '\n';
        if (bytes + n > MaxBytes) { truncated = true; return; }
        std::fwrite(line, 1, n, file);
        bytes += n;
        ++rows;
        last_written = row.seq;
    }
    std::FILE* file{};
    std::array<Row, RingSize> ring{};
    std::array<u64, Fields.size()> previous{};
    u64 sequence{};
    u64 previous_end{};
    u64 last_written{};
    u64 rows{};
    u64 bytes{};
    bool truncated{};
    double slow_ms{DefaultSlowMs};
    u64 slow_ns{50'000'000};
};

inline void Observe(u64 guest_frame, u64 start, u64 acquired, u64 flush_start,
                    u64 flushed, u64 end) {
    static Capture capture;
    capture.Observe(guest_frame, start, acquired, flush_start, flushed, end);
}
} // namespace Common::Perf::CausalFrame
