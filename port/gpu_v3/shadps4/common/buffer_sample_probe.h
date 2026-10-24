// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>

#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/types.h"

namespace Common {

/// Explicit ObtainBuffer caller class for the probe's size/caller rows. Only the cache-buffer
/// path of Rasterizer::BindBuffers supplies Uniform or Storage; every other caller is Other.
/// It is an argument rather than a thread-local scope so nested fault work cannot inherit it.
enum class BufferSampleCaller : u8 {
    Other = 0,
    Uniform,
    Storage,
};

/// SHADPS4_BUFFER_SAMPLE, read once at startup: Enabled() runs per ObtainBuffer call, and a
/// function-local static would add a guard check to each.
inline const bool g_buffer_sample_enabled = [] {
    const char* env = std::getenv("SHADPS4_BUFFER_SAMPLE");
    return env != nullptr && std::atoi(env) != 0;
}();

/// A low-frequency, runtime-opt-in probe for the ordinary BufferCache path.
///
/// The probe deliberately keeps its counters thread-local. ObtainBuffer is normally called on
/// the command processor thread, and a fixed-size local window avoids adding atomics to the hot
/// path. A window is emitted after 16,384 sampled calls so FaultMonitor's settled interval can
/// bound the corresponding log lines. A fixed-seed PRNG samples about one call in 128 without
/// locking onto a repeating binding pattern. Clock overhead and preemption are included.
class BufferSampleProbe {
public:
    using Clock = std::chrono::steady_clock;

    enum class Kind : u8 {
        ReadOnly,
        ReadOnlyTexel,
        Written,
    };

    enum class UploadPath : u8 {
        Walk,
        Coarse,
        Memo,
    };

    static bool Enabled() noexcept {
        return g_buffer_sample_enabled;
    }

    static bool ShouldSample() noexcept {
        if (!Enabled()) {
            return false;
        }
        auto& state = RandomState();
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state & (SampleEvery - 1)) == 0;
    }

    BufferSampleProbe(bool active, bool is_written, bool is_texel, u64 size,
                      BufferSampleCaller caller = BufferSampleCaller::Other)
        : active{active},
          kind{is_written ? Kind::Written
                          : (is_texel ? Kind::ReadOnlyTexel : Kind::ReadOnly)},
          caller{caller},
          size{size},
          start{active ? Clock::now() : Clock::time_point{}} {}

    BufferSampleProbe(const BufferSampleProbe&) = delete;
    BufferSampleProbe& operator=(const BufferSampleProbe&) = delete;

    ~BufferSampleProbe() {
        Finish();
    }

    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    [[nodiscard]] Clock::time_point Now() const noexcept {
        return active ? Clock::now() : Clock::time_point{};
    }

    [[nodiscard]] u64 SinceNs(Clock::time_point start) const noexcept {
        if (!active) {
            return 0;
        }
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }

    void AddFindNs(u64 ns) noexcept {
        if (active) {
            find_ns += ns;
        }
    }

    void AddSyncNs(u64 ns) noexcept {
        if (active) {
            sync_ns += ns;
        }
    }

    void AddUploadNs(u64 ns) noexcept {
        if (active) {
            upload_ns += ns;
        }
    }

    void AddTexelNs(u64 ns) noexcept {
        if (active) {
            texel_ns += ns;
        }
    }

    void AddRangeAddNs(u64 ns) noexcept {
        if (active) {
            range_add_ns += ns;
        }
    }

    void RecordStream() noexcept {
        if (active) {
            stream = true;
        }
    }

    void RecordBufferIdFallback(bool fallback) noexcept {
        if (active && fallback) {
            buffer_id_fallback = true;
        }
    }

    void RecordUpload(UploadPath path, u64 bytes, u64 ranges) noexcept {
        if (!active) {
            return;
        }
        upload_bytes += bytes;
        upload_ranges += ranges;
        switch (path) {
        case UploadPath::Memo:
            ++upload_memo;
            break;
        case UploadPath::Coarse:
            ++upload_coarse;
            break;
        case UploadPath::Walk:
            ++upload_walk;
            break;
        }
    }

private:
    struct KindStats {
        u64 calls = 0;
        u64 call_bytes = 0;
        u64 span_ns = 0;
        u64 find_ns = 0;
        u64 sync_ns = 0;
        u64 upload_ns = 0;
        u64 texel_ns = 0;
        u64 range_add_ns = 0;
        u64 stream_calls = 0;
        u64 fallback_calls = 0;
        u64 upload_memo = 0;
        u64 upload_coarse = 0;
        u64 upload_walk = 0;
        u64 upload_bytes = 0;
        u64 upload_ranges = 0;
    };

    /// One Kind x Caller x request-size group. The timings are nested (upload inside sync inside
    /// span) and must not be added together as exclusive work. "dirty" is upload_bytes > 0.
    struct SizeStats {
        u64 calls = 0;
        u64 call_bytes = 0;
        u64 stream_calls = 0;
        u64 span_ns = 0;
        u64 sync_ns = 0;
        u64 upload_ns = 0;
        u64 upload_bytes = 0;
        u64 upload_ranges = 0;
        u64 dirty_calls = 0;
        u64 dirty_span_ns = 0;
        u64 dirty_sync_ns = 0;
        u64 dirty_upload_ns = 0;
    };

    static constexpr size_t NumKinds = 3;
    static constexpr size_t NumCallers = 3;
    static constexpr size_t NumSizeBuckets = 4;

    using SizeGroups =
        std::array<std::array<std::array<SizeStats, NumSizeBuckets>, NumCallers>, NumKinds>;

    struct Counters {
        u64 window_count = 0;
        u64 total_count = 0;
        u64 sequence = 0;
        std::array<KindStats, NumKinds> kinds{};
        SizeGroups groups{};
    };

    static constexpr u64 SampleEvery = 128;
    static constexpr u64 WindowSamples = 16'384;

    static Counters& ThreadCounters() noexcept {
        thread_local Counters counters;
        return counters;
    }

    static u32& RandomState() noexcept {
        static thread_local u32 state = 0xA57A'6123u;
        return state;
    }

    static constexpr size_t Index(Kind value) noexcept {
        return static_cast<size_t>(value);
    }

    static const char* KindName(Kind value) noexcept {
        switch (value) {
        case Kind::ReadOnly:
            return "ro";
        case Kind::ReadOnlyTexel:
            return "ro_texel";
        case Kind::Written:
            return "written";
        }
        return "unknown";
    }

    static constexpr size_t Index(BufferSampleCaller value) noexcept {
        const auto index = static_cast<size_t>(value);
        return index < NumCallers ? index : 0;
    }

    static const char* CallerName(BufferSampleCaller value) noexcept {
        switch (value) {
        case BufferSampleCaller::Other:
            return "other";
        case BufferSampleCaller::Uniform:
            return "uniform";
        case BufferSampleCaller::Storage:
            return "storage";
        }
        return "unknown";
    }

    static constexpr size_t SizeBucket(u64 bytes) noexcept {
        if (bytes <= 4 * 1024) {
            return 0;
        }
        if (bytes <= 16 * 1024) {
            return 1;
        }
        if (bytes <= 64 * 1024) {
            return 2;
        }
        return 3;
    }

    static const char* SizeBucketName(size_t bucket) noexcept {
        switch (bucket) {
        case 0:
            return "le4k";
        case 1:
            return "4k_16k";
        case 2:
            return "16k_64k";
        case 3:
            return "gt64k";
        }
        return "unknown";
    }

    void Finish() noexcept {
        if (!active || finished) {
            return;
        }
        finished = true;
        auto& counters = ThreadCounters();
        auto& stats = counters.kinds[Index(kind)];
        ++counters.window_count;
        ++counters.total_count;
        ++stats.calls;
        stats.call_bytes += size;
        const u64 span_ns = SinceNs(start);
        stats.span_ns += span_ns;
        auto& group = counters.groups[Index(kind)][Index(caller)][SizeBucket(size)];
        ++group.calls;
        group.call_bytes += size;
        group.span_ns += span_ns;
        group.sync_ns += sync_ns;
        group.upload_ns += upload_ns;
        group.upload_bytes += upload_bytes;
        group.upload_ranges += upload_ranges;
        if (stream) {
            ++group.stream_calls;
        }
        if (upload_bytes > 0) {
            ++group.dirty_calls;
            group.dirty_span_ns += span_ns;
            group.dirty_sync_ns += sync_ns;
            group.dirty_upload_ns += upload_ns;
        }
        if (stream) {
            ++stats.stream_calls;
        }
        if (buffer_id_fallback) {
            ++stats.fallback_calls;
        }
        stats.find_ns += find_ns;
        stats.sync_ns += sync_ns;
        stats.upload_ns += upload_ns;
        stats.texel_ns += texel_ns;
        stats.range_add_ns += range_add_ns;
        stats.upload_memo += upload_memo;
        stats.upload_coarse += upload_coarse;
        stats.upload_walk += upload_walk;
        stats.upload_bytes += upload_bytes;
        stats.upload_ranges += upload_ranges;
        if (counters.window_count >= WindowSamples) {
            ++counters.sequence;
            Report(counters);
            const u64 total_count = counters.total_count;
            const u64 sequence = counters.sequence;
            counters = {};
            counters.total_count = total_count;
            counters.sequence = sequence;
        }
    }

    static void Report(const Counters& counters) noexcept {
        for (size_t i = 0; i < counters.kinds.size(); ++i) {
            const auto& stats = counters.kinds[i];
            const auto mean_us = [count = stats.calls](u64 ns) {
                return count == 0 ? 0.0
                                  : static_cast<double>(ns) / (1000.0 * static_cast<double>(count));
            };
            LOG_INFO(Render,
                     "BufferSample seq={} window_count={} total_count={} kind={} calls={} "
                     "call_bytes={} span={:.3f}us find={:.3f}us sync={:.3f}us upload={:.3f}us "
                     "texel_alias={:.3f}us gpu_range_add={:.3f}us stream={} buffer_fallback={} "
                     "upload_path=memo:{},coarse:{},walk:{} uploaded_bytes={} upload_ranges={} "
                     "sample_every={} steady_clock_mark_overhead_included=true",
                     counters.sequence, counters.window_count, counters.total_count,
                     KindName(static_cast<Kind>(i)), stats.calls, stats.call_bytes,
                     mean_us(stats.span_ns), mean_us(stats.find_ns), mean_us(stats.sync_ns),
                     mean_us(stats.upload_ns), mean_us(stats.texel_ns), mean_us(stats.range_add_ns),
                     stats.stream_calls, stats.fallback_calls, stats.upload_memo,
                     stats.upload_coarse, stats.upload_walk, stats.upload_bytes,
                     stats.upload_ranges, SampleEvery);
        }
        // Cumulative frame count, read once per window so consecutive reports give an approximate
        // sampled rate per frame. Integer totals keep the rows exactly aggregable.
        const u64 frames = FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        for (size_t k = 0; k < NumKinds; ++k) {
            for (size_t c = 0; c < NumCallers; ++c) {
                for (size_t b = 0; b < NumSizeBuckets; ++b) {
                    const auto& group = counters.groups[k][c][b];
                    if (group.calls == 0) {
                        continue;
                    }
                    LOG_INFO(Render,
                             "BufferSizeSample seq={} window_count={} total_count={} "
                             "sample_every={} presented_frames={} kind={} caller={} size={} "
                             "calls={} call_bytes={} stream={} span_ns={} sync_ns={} "
                             "upload_ns={} uploaded_bytes={} upload_ranges={} dirty_calls={} "
                             "dirty_span_ns={} dirty_sync_ns={} dirty_upload_ns={} "
                             "nested_timings=true",
                             counters.sequence, counters.window_count, counters.total_count,
                             SampleEvery, frames, KindName(static_cast<Kind>(k)),
                             CallerName(static_cast<BufferSampleCaller>(c)), SizeBucketName(b),
                             group.calls, group.call_bytes, group.stream_calls, group.span_ns,
                             group.sync_ns, group.upload_ns, group.upload_bytes,
                             group.upload_ranges, group.dirty_calls, group.dirty_span_ns,
                             group.dirty_sync_ns, group.dirty_upload_ns);
                }
            }
        }
    }

    bool active = false;
    bool finished = false;
    Kind kind = Kind::ReadOnly;
    BufferSampleCaller caller = BufferSampleCaller::Other;
    u64 size = 0;
    Clock::time_point start{};
    bool stream = false;
    bool buffer_id_fallback = false;
    u64 find_ns = 0;
    u64 sync_ns = 0;
    u64 upload_ns = 0;
    u64 texel_ns = 0;
    u64 range_add_ns = 0;
    u64 upload_memo = 0;
    u64 upload_coarse = 0;
    u64 upload_walk = 0;
    u64 upload_bytes = 0;
    u64 upload_ranges = 0;

};

} // namespace Common
