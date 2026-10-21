// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_OCCLUSION_DIAG=1: what the guest's occlusion queries return, query by query (included by
// vk_rasterizer.cpp, command processor thread only). Each query is recorded at its end dump as
// counted by the GPU (samples read back once its command buffer completes) or emulated (all
// visible, a huge count) with the reason. Every 5 s OCCLUSION_DIAG reports, per frame, the
// counted/emulated mix, and per query address how often it switched between counted and
// emulated, or dipped to 0 samples and back, within a few frames: the jumps the guest sees.


namespace occlusion_diag {
const bool enabled = [] {
    const char* p = std::getenv("ISS_OCCLUSION_DIAG");
    return p && p[0] == '1';
}();

enum Reason : u32 { RCounted, ROff, RUnmapped, RTooMany, RUnaligned, RNotCounted, RNum };
constexpr const char* reason_names[RNum] = {"counted", "off", "unmapped", "too_many_open",
                                            "unaligned", "not_counted"};
struct Event {
    u64 frame;
    u32 reason;
    s64 samples; ///< -1 until read back (counted), -2 emulated
};
struct Track {
    std::array<Event, 64> events{};
    u32 next{}, size{};
    u32 pairs{};
};
struct Pending {
    VAddr address;
    u64 frame;
    u64 tick;
    u32 num_queries;
    std::array<u32, 16> queries;
};
std::unordered_map<VAddr, Track> tracks;
std::vector<Pending> pending;
std::array<u64, RNum> reasons{};
u64 last_report_ns{}, last_report_frame{};

u64 Frame() {
    return Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
}

void Note(VAddr address, u32 reason, u32 pairs) {
    ++reasons[reason];
    auto& t = tracks[address];
    t.pairs = pairs;
    t.events[t.next] = Event{Frame(), reason, reason == RCounted ? -1 : -2};
    t.next = (t.next + 1) % t.events.size();
    t.size = std::min<u32>(t.size + 1, t.events.size());
}

void NoteCounted(VAddr address, u32 pairs, std::span<const u32> queries, u64 tick) {
    Note(address, RCounted, pairs);
    Pending p{address, Frame(), tick, static_cast<u32>(std::min<size_t>(queries.size(), 16)), {}};
    std::copy_n(queries.begin(), p.num_queries, p.queries.begin());
    if (pending.size() < 8192) pending.push_back(p);
}

void Collect(const Instance& instance, Scheduler& scheduler, vk::QueryPool pool) {
    for (auto it = pending.begin(); it != pending.end();) {
        if (!scheduler.IsFree(it->tick)) {
            ++it;
            continue;
        }
        s64 samples = 0;
        for (u32 i = 0; i < it->num_queries; ++i) {
            u64 value = 0;
            const auto result = instance.GetDevice().getQueryPoolResults(
                pool, it->queries[i], 1, sizeof(value), &value, sizeof(value),
                vk::QueryResultFlagBits::e64);
            if (result == vk::Result::eSuccess) samples += static_cast<s64>(value);
        }
        auto& t = tracks[it->address];
        for (auto& e : t.events) {
            if (e.frame == it->frame && e.reason == RCounted && e.samples == -1) {
                e.samples = samples;
                break;
            }
        }
        it = pending.erase(it);
    }
}

void Report() {
    const u64 now = Common::Perf::NowNs();
    if (now - last_report_ns < 5'000'000'000ull) return;
    last_report_ns = now;
    const u64 frame = Frame();
    const double frames = double(frame > last_report_frame ? frame - last_report_frame : 1);
    last_report_frame = frame;
    std::fprintf(stderr, "OCCLUSION_DIAG frames=%.0f addresses=%zu per_frame:", frames,
                 tracks.size());
    for (u32 r = 0; r < RNum; ++r) {
        std::fprintf(stderr, " %s=%.1f", reason_names[r], reasons[r] / frames);
        reasons[r] = 0;
    }
    // Per address over its last 64 queries: switches counted<->emulated and dips to 0 samples.
    struct Row {
        VAddr address;
        u32 pairs, n, counted, switches, zero_dips, zeros;
        s64 min, max;
        double mean;
    };
    std::vector<Row> rows;
    u64 total_switches = 0, total_dips = 0, total_zero = 0, total_counted_read = 0;
    for (auto& [address, t] : tracks) {
        std::vector<Event> ev;
        for (u32 i = 0; i < t.size; ++i) ev.push_back(t.events[i]);
        std::sort(ev.begin(), ev.end(), [](const Event& a, const Event& b) {
            return a.frame < b.frame;
        });
        Row row{address, t.pairs, static_cast<u32>(ev.size()), 0, 0, 0, 0, INT64_MAX, -1, 0};
        u64 sum = 0;
        for (size_t i = 0; i < ev.size(); ++i) {
            const auto& e = ev[i];
            if (e.reason == RCounted && e.samples >= 0) {
                ++row.counted;
                sum += static_cast<u64>(e.samples);
                row.min = std::min(row.min, e.samples);
                row.max = std::max(row.max, e.samples);
                if (e.samples == 0) ++row.zeros;
            }
            if (i > 0 && ev[i - 1].frame + 4 >= e.frame) {
                const bool a = ev[i - 1].reason == RCounted, b = e.reason == RCounted;
                if (a != b) ++row.switches;
                if (i + 1 < ev.size() && a && b && ev[i + 1].reason == RCounted &&
                    ev[i - 1].samples > 0 && e.samples == 0 && ev[i + 1].samples > 0) {
                    ++row.zero_dips;
                }
            }
        }
        row.mean = row.counted ? double(sum) / row.counted : 0;
        if (row.min == INT64_MAX) row.min = -1;
        total_switches += row.switches;
        total_dips += row.zero_dips;
        total_zero += row.zeros;
        total_counted_read += row.counted;
        rows.push_back(row);
    }
    std::fprintf(stderr, " | window: switches=%llu zero_dips=%llu zero_results=%llu/%llu\n",
                 (unsigned long long)total_switches, (unsigned long long)total_dips,
                 (unsigned long long)total_zero, (unsigned long long)total_counted_read);
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.switches + a.zero_dips > b.switches + b.zero_dips;
    });
    for (size_t i = 0; i < std::min<size_t>(rows.size(), 10); ++i) {
        const auto& r = rows[i];
        if (r.switches + r.zero_dips == 0) break;
        std::fprintf(stderr,
                     "OCCLUSION_DIAG_TOP %llx pairs=%u last=%u counted=%u switches=%u "
                     "zero_dips=%u zeros=%u samples min=%lld mean=%.0f max=%lld\n",
                     (unsigned long long)r.address, r.pairs, r.n, r.counted, r.switches,
                     r.zero_dips, r.zeros, (long long)r.min, r.mean, (long long)r.max);
    }
    // Addresses not queried for a while are forgotten.
    for (auto it = tracks.begin(); it != tracks.end();) {
        const auto& t = it->second;
        const auto& last = t.events[(t.next + t.events.size() - 1) % t.events.size()];
        it = frame - last.frame > 600 ? tracks.erase(it) : std::next(it);
    }
}
} // namespace occlusion_diag
