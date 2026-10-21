// SPDX-License-Identifier: GPL-2.0-or-later
#include "srt_walker_capture.h"

#include <algorithm>
#include <limits>
#include <set>

#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"

namespace iss::g4 {
namespace {
struct WatchedPhysicalRange {
    std::uint64_t address{};
    std::uint64_t size{};
};

// The shadow backing is fault-free, but it is not necessarily the newest copy:
// GPU writes can leave the result only in an arena. Check every VA alias because
// ownership is tracked by guest VA while the backing is shared by physical bytes.
// This is a rejection test, not a readback; a protected C worker must never try
// to resolve a read fault by waiting for a G control that pauses C itself.
std::optional<Core::MemoryManager::BackingSpan> CpuCurrentBacking(
    Core::MemoryManager& memory, VideoCore::BufferCache& buffers,
    std::uint64_t address, std::size_t size) {
    if (address == 0 || size == 0 ||
        address > std::numeric_limits<std::uint64_t>::max() - size) return std::nullopt;
    const auto generation = buffers.MappingGeneration();
    const auto epoch = memory.MappingEpoch();
    if (generation == 0 || epoch == 0) return std::nullopt;
    const auto backing = memory.DescribeBacking(address, size);
    const auto aliases = memory.EnumerateBackingAliases(address, size);
    if (!backing || backing->size != size || !aliases) return std::nullopt;
    bool found_original = false;
    for (const auto& alias : *aliases) {
        if (alias.size == 0 || alias.physical_start < backing->physical_start ||
            alias.physical_start - backing->physical_start > backing->size ||
            alias.size > backing->size -
                             (alias.physical_start - backing->physical_start) ||
            alias.guest_address > std::numeric_limits<std::uint64_t>::max() - alias.size ||
            buffers.HasGpuData(alias.guest_address, alias.size)) return std::nullopt;
        found_original |= alias.guest_address == address &&
                          alias.physical_start == backing->physical_start &&
                          alias.size == size;
    }
    if (!found_original || buffers.MappingGeneration() != generation ||
        memory.MappingEpoch() != epoch) return std::nullopt;
    return backing;
}

class PageWatchLease final : public SrtReadWatchLease {
public:
    PageWatchLease(Core::MemoryManager& memory_, VideoCore::PageManager& pages_,
                   VideoCore::BufferCache& buffers_, std::uint64_t mapping_generation_,
                   std::uint64_t native_epoch_)
        : memory{memory_}, pages{pages_}, buffers{buffers_},
          mapping_generation{mapping_generation_}, native_epoch{native_epoch_} {}

    ~PageWatchLease() override {
        for (auto token : tokens) pages.ReleaseSrtWriteWatch(token);
    }

    bool Covers(std::span<const SrtReadSnapshot> reads) const override {
        if (!Valid()) return false;
        for (const auto& read : reads) {
            const auto backing = memory.DescribeBacking(read.guest_address, read.bytes.size());
            if (!backing || backing->size != read.bytes.size()) return false;
            const bool covered = std::any_of(physical.begin(), physical.end(), [&](const auto& r) {
                return backing->physical_start >= r.address &&
                       backing->physical_start - r.address <= r.size &&
                       backing->size <= r.size - (backing->physical_start - r.address);
            });
            if (!covered) return false;
        }
        return Valid();
    }

    bool Valid() const override {
        if (buffers.MappingGeneration() != mapping_generation ||
            memory.MappingEpoch() != native_epoch) return false;
        for (auto token : tokens) {
            if (!pages.CheckSrtWriteWatch(token)) return false;
        }
        return buffers.MappingGeneration() == mapping_generation &&
               memory.MappingEpoch() == native_epoch;
    }

    bool PublishIfValid(const std::function<bool()>& publish) const override {
        return memory.WithMappingReadLock([&](u64 epoch) {
            if (epoch != native_epoch || buffers.MappingGeneration() != mapping_generation)
                return false;
            return pages.WithSrtWriteWatchesLocked(tokens, [&] {
                if (buffers.MappingGeneration() != mapping_generation) return false;
                return publish();
            });
        });
    }

    Core::MemoryManager& memory;
    VideoCore::PageManager& pages;
    VideoCore::BufferCache& buffers;
    std::uint64_t mapping_generation;
    std::uint64_t native_epoch;
    std::vector<VideoCore::PageManager::SrtWatchToken> tokens;
    std::vector<WatchedPhysicalRange> physical;
};

} // namespace

TrackedSrtMemoryProvider::TrackedSrtMemoryProvider(Core::MemoryManager& memory_,
                                                   VideoCore::PageManager& pages_,
                                                   VideoCore::BufferCache& buffers_) noexcept
    : memory{memory_}, pages{pages_}, buffers{buffers_} {}

bool TrackedSrtMemoryProvider::Version(std::uint64_t address, std::size_t size,
                                      SrtMemoryVersion& out) {
    const auto generation = buffers.MappingGeneration();
    const auto native_epoch = memory.MappingEpoch();
    const auto backing = CpuCurrentBacking(memory, buffers, address, size);
    if (!backing || backing->size != size || generation == 0 || native_epoch == 0 ||
        buffers.MappingGeneration() != generation ||
        memory.MappingEpoch() != native_epoch) return false;
    out.mapping_generation = generation;
    out.native_mapping_epoch = native_epoch;
    out.physical_start = backing->physical_start;
    out.cpu_versions.clear(); // the active watch, rather than RegionManager, covers hot pages
    return true;
}

bool TrackedSrtMemoryProvider::CopySparse(std::uint64_t address,
                                          std::span<std::byte> out) {
    const auto before = CpuCurrentBacking(memory, buffers, address, out.size());
    if (!before) return false;
    const auto generation = buffers.MappingGeneration();
    const auto epoch = memory.MappingEpoch();
    memory.CopySparseMemory(address, reinterpret_cast<u8*>(out.data()), out.size());
    const auto after = CpuCurrentBacking(memory, buffers, address, out.size());
    return after && after->physical_start == before->physical_start &&
           after->size == before->size &&
           buffers.MappingGeneration() == generation && memory.MappingEpoch() == epoch;
}

std::shared_ptr<SrtReadWatchLease> TrackedSrtMemoryProvider::ArmWatchSet(
    std::span<const SrtReadSnapshot> reads) {
    if (reads.empty()) return {};
    const auto generation = buffers.MappingGeneration();
    const auto native_epoch = memory.MappingEpoch();
    if (generation == 0 || native_epoch == 0) return {};
    auto lease = std::make_shared<PageWatchLease>(memory, pages, buffers, generation,
                                                  native_epoch);
    std::set<std::uint64_t> guest_pages;
    for (const auto& read : reads) {
        if (read.guest_address == 0 || read.bytes.empty() ||
            read.guest_address > std::numeric_limits<std::uint64_t>::max() - read.bytes.size())
            return {};
        const auto backing = memory.DescribeBacking(read.guest_address, read.bytes.size());
        const auto aliases = memory.EnumerateBackingAliases(read.guest_address, read.bytes.size());
        if (!backing || backing->size != read.bytes.size() || !aliases) return {};
        const bool includes_original = std::any_of(aliases->begin(), aliases->end(),
            [&](const auto& alias) {
                return alias.guest_address == read.guest_address &&
                       alias.physical_start == backing->physical_start &&
                       alias.size >= read.bytes.size();
            });
        if (!includes_original) return {};
        lease->physical.push_back({backing->physical_start, backing->size});
        for (const auto& alias : *aliases) {
            if (alias.size == 0 || alias.guest_address >
                std::numeric_limits<std::uint64_t>::max() - alias.size) return {};
            const auto first = VideoCore::PageManager::GetPageAddr(alias.guest_address);
            const auto last = VideoCore::PageManager::GetPageAddr(
                alias.guest_address + alias.size - 1);
            for (auto page = first; page <= last; page += 4096) guest_pages.insert(page);
        }
    }
    for (auto page : guest_pages) {
        VideoCore::PageManager::SrtWatchToken token{};
        if (!pages.ArmSrtWriteWatch(page, token)) return {};
        lease->tokens.push_back(token);
    }
    return lease->Valid() ? lease : std::shared_ptr<SrtReadWatchLease>{};
}

} // namespace iss::g4
