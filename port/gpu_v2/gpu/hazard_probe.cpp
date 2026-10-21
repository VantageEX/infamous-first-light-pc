#include "hazard_probe.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>

#include "common/perf_stats.h"

namespace iss::hazard_probe {
namespace {

constexpr std::uint64_t DelayNs = 30'000'000'000ULL;
constexpr std::uint32_t MaxDispatches = 3'000;
constexpr std::size_t MaxBytes = 32U * 1024U * 1024U;

struct DispatchState {
    bool active{};
    std::uint32_t id{};
    std::uint64_t shader_hash{};
    std::uint64_t permutation_hash{};
    std::uint64_t tick{};
    std::uint32_t queue{};
    std::uint32_t stage{};
    bool inherited_needs_barrier{};
};
thread_local DispatchState current;

struct FileState {
    std::mutex mutex;
    std::FILE* file{};
    std::size_t bytes{};
    std::uint32_t rows{};
    bool attempted{};
    bool truncated{};
};
FileState& Output() {
    static FileState output;
    return output;
}

std::atomic<std::uint64_t> first_dispatch_ns{0};
std::atomic<std::uint32_t> next_dispatch_id{0};

struct Row {
    const char* event{};
    std::uint32_t descriptor_index{};
    std::uint32_t buffer_type{};
    std::array<std::uint32_t, 4> raw_vsharp{};
    std::uint64_t effective_address{};
    std::uint64_t effective_size{};
    std::uint64_t backing_id{};
    std::uint64_t buffer_offset{};
    std::uint64_t tracked_size{};
    bool is_written{};
    bool check_performed{};
    bool hit{};
    bool needs_before{};
    bool needs_after{};
    bool flushed{};
    std::uint32_t flags{};
};

void Write(const Row& row) noexcept {
    if (!current.active) {
        return;
    }
    const std::uint64_t now = Common::Perf::NowNs();
    const std::uint64_t thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
    char line[512];
    const int length = std::snprintf(
        line, sizeof(line),
        "%s,%llu,%u,%llu,%u,%llu,0x%llx,0x%llx,%u,%u,%u,"
        "0x%08x,0x%08x,0x%08x,0x%08x,0x%llx,%llu,0x%llx,%llu,%llu,"
        "%u,%u,%u,%u,%u,%u,%u\n",
        row.event, static_cast<unsigned long long>(now), current.id,
        static_cast<unsigned long long>(thread), current.queue,
        static_cast<unsigned long long>(current.tick),
        static_cast<unsigned long long>(current.shader_hash),
        static_cast<unsigned long long>(current.permutation_hash), current.stage,
        row.descriptor_index, row.buffer_type, row.raw_vsharp[0], row.raw_vsharp[1],
        row.raw_vsharp[2], row.raw_vsharp[3],
        static_cast<unsigned long long>(row.effective_address),
        static_cast<unsigned long long>(row.effective_size),
        static_cast<unsigned long long>(row.backing_id),
        static_cast<unsigned long long>(row.buffer_offset),
        static_cast<unsigned long long>(row.tracked_size), row.is_written ? 1U : 0U,
        row.check_performed ? 1U : 0U, row.hit ? 1U : 0U,
        row.needs_before ? 1U : 0U, row.needs_after ? 1U : 0U,
        row.flushed ? 1U : 0U, row.flags);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(line)) {
        return;
    }

    auto& out = Output();
    std::scoped_lock lock{out.mutex};
    if (!out.attempted) {
        out.attempted = true;
        out.file = std::fopen(std::getenv("ISS_GPU_HAZARD_LOG"), "wb");
        if (out.file) {
            std::setvbuf(out.file, nullptr, _IOFBF, 1 << 20);
            static constexpr char Header[] =
                "event,ns,dispatch,thread,queue,tick,shader_hash,permutation_hash,stage,"
                "descriptor_index,buffer_type,raw_v0,raw_v1,raw_v2,raw_v3,effective_address,"
                "effective_size,backing_id,buffer_offset,tracked_size,is_written,checked,hit,"
                "needs_before,needs_after,flushed,flags\n";
            std::fputs(Header, out.file);
            out.bytes = sizeof(Header) - 1;
        }
    }
    if (!out.file) {
        return;
    }
    if (out.bytes + static_cast<std::size_t>(length) > MaxBytes) {
        if (!out.truncated) {
            std::fputs("truncated\n", out.file);
            std::fflush(out.file);
            out.truncated = true;
        }
        return;
    }
    std::fwrite(line, 1, static_cast<std::size_t>(length), out.file);
    out.bytes += static_cast<std::size_t>(length);
    ++out.rows;
    if ((out.rows & 511U) == 0) {
        std::fflush(out.file);
    }
}

} // namespace

bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* flag = std::getenv("ISS_GPU_HAZARD_DIAG");
        const char* path = std::getenv("ISS_GPU_HAZARD_LOG");
        return flag && flag[0] == '1' && path && path[0] != '\0';
    }();
    return enabled;
}

void BeginDispatch(std::uint64_t shader_hash, std::uint64_t permutation_hash,
                   std::uint64_t tick, std::uint32_t queue, std::uint32_t stage,
                   bool inherited_needs_barrier) noexcept {
    if (!Enabled()) [[likely]] {
        return;
    }
    if (current.active) {
        AbortDispatch(0xffffU); // A prior dispatch exited without End/Abort.
    }
    if (next_dispatch_id.load(std::memory_order_relaxed) >= MaxDispatches) {
        return;
    }
    const std::uint64_t now = Common::Perf::NowNs();
    std::uint64_t unset = 0;
    first_dispatch_ns.compare_exchange_strong(unset, now, std::memory_order_relaxed);
    if (now - first_dispatch_ns.load(std::memory_order_relaxed) < DelayNs) {
        return;
    }
    const std::uint32_t id = next_dispatch_id.fetch_add(1, std::memory_order_relaxed);
    if (id >= MaxDispatches) {
        return;
    }
    current = {.active = true, .id = id, .shader_hash = shader_hash,
               .permutation_hash = permutation_hash, .tick = tick,
               .queue = queue, .stage = stage,
               .inherited_needs_barrier = inherited_needs_barrier};
    Write({.event = "begin", .needs_before = inherited_needs_barrier,
           .needs_after = inherited_needs_barrier});
}

void Buffer(const BufferCheck& check) noexcept {
    if (!current.active) [[likely]] {
        return;
    }
    Write({.event = "buffer", .descriptor_index = check.descriptor_index,
           .buffer_type = check.buffer_type, .raw_vsharp = check.raw_vsharp,
           .effective_address = check.effective_address,
           .effective_size = check.effective_size, .backing_id = check.backing_id,
           .buffer_offset = check.buffer_offset, .tracked_size = check.tracked_size,
           .is_written = check.is_written, .check_performed = check.check_performed,
           .hit = check.hit, .needs_before = check.needs_before,
           .needs_after = check.needs_after,
           .flags = check.is_special ? 1U : 0U});
}

void EndDispatch(bool needs_barrier, bool flushed) noexcept {
    if (!current.active) [[likely]] {
        return;
    }
    Write({.event = "end", .needs_before = current.inherited_needs_barrier,
           .needs_after = needs_barrier, .flushed = flushed});
    if (current.id + 1 == MaxDispatches) {
        auto& out = Output();
        std::scoped_lock lock{out.mutex};
        if (out.file) std::fflush(out.file);
    }
    current.active = false;
}

void AbortDispatch(std::uint32_t reason) noexcept {
    if (!current.active) [[likely]] {
        return;
    }
    Write({.event = "abort", .needs_before = current.inherited_needs_barrier,
           .flags = reason});
    if (current.id + 1 == MaxDispatches) {
        auto& out = Output();
        std::scoped_lock lock{out.mutex};
        if (out.file) std::fflush(out.file);
    }
    current.active = false;
}

} // namespace iss::hazard_probe
