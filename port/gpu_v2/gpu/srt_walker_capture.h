// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <memory>
#include <span>
#include <vector>

#include "g4_fallback_packet.h"
#include "shader_recompiler/ir/passes/srt.h"

namespace Core { class MemoryManager; }
namespace VideoCore { class PageManager; class BufferCache; }

namespace iss::g4 {

inline constexpr std::uint32_t SrtCaptureSchema = 1;

struct SrtMemoryVersion {
    std::uint64_t mapping_generation{};
    std::uint64_t native_mapping_epoch{};
    std::uint64_t physical_start{};
    std::vector<std::uint64_t> cpu_versions{};
    friend bool operator==(const SrtMemoryVersion&, const SrtMemoryVersion&) = default;
};

// An active watch protects every VA alias of every physical byte in the read set, and
// invalidates on direct backing writes and remaps. Its validity must survive until the
// packet's revalidate/publish boundary; the caller retains the returned result until then.
class SrtReadWatchLease {
public:
    virtual ~SrtReadWatchLease() = default;
    virtual bool Covers(std::span<const SrtReadSnapshot> reads) const = 0;
    virtual bool Valid() const = 0;
    virtual bool PublishIfValid(const std::function<bool()>& publish) const = 0;
};

// A provider must cover remap and physical backing changes for every range it reports.
// CopySparse must read the same guest backing represented by Version, with no guest fault.
class SrtMemoryProvider {
public:
    virtual ~SrtMemoryProvider() = default;
    // Global monotonic epoch for *all* guest writes and backing changes, including hot
    // pages and aliases. It must be synchronized with the packet publication boundary.
    // No epoch is a hard failure, never a request to guess from per-region versions.
    virtual std::optional<std::uint64_t> GlobalEpoch() { return std::nullopt; }
    virtual std::shared_ptr<SrtReadWatchLease> ArmWatchSet(
        std::span<const SrtReadSnapshot> reads) { return {}; }
    virtual bool Version(std::uint64_t address, std::size_t size,
                         SrtMemoryVersion& out) = 0;
    virtual bool CopySparse(std::uint64_t address, std::span<std::byte> out) = 0;
};

// Concrete provider for this port. It is conservative: ambiguous aliases, unavailable
// pages, a changed map, or a failed write watch all leave the result Unknown.
class TrackedSrtMemoryProvider final : public SrtMemoryProvider {
public:
    TrackedSrtMemoryProvider(Core::MemoryManager& memory,
                             VideoCore::PageManager& pages,
                             VideoCore::BufferCache& buffers) noexcept;
    bool Version(std::uint64_t address, std::size_t size,
                 SrtMemoryVersion& out) override;
    bool CopySparse(std::uint64_t address, std::span<std::byte> out) override;
    std::shared_ptr<SrtReadWatchLease> ArmWatchSet(
        std::span<const SrtReadSnapshot> reads) override;

private:
    Core::MemoryManager& memory;
    VideoCore::PageManager& pages;
    VideoCore::BufferCache& buffers;
};

struct SrtCaptureResult {
    std::vector<std::uint32_t> flattened{};
    std::vector<SrtReadSnapshot> reads{};
    std::vector<SrtMemoryVersion> read_versions{};
    std::uint64_t capture_epoch{};
    std::shared_ptr<SrtReadWatchLease> watch_lease{};
    SrtSnapshotProof proof{};
};

struct SrtExtraRead {
    std::uint64_t guest_address{};
    std::uint32_t size{};
};

// A capture walker is generated alongside the fast walker; it reads every guest pointer
// and dword through ReadCapturedSrt. The normal walker remains unchanged.
[[nodiscard]] SrtCaptureResult CaptureSrtSnapshot(
    const Shader::PersistentSrtInfo& info,
    std::span<const std::uint32_t> user_data, SrtMemoryProvider& provider,
    std::span<const SrtExtraRead> extra_reads = {});

// Call at the packet's revalidate/publish boundary, before any Vulkan emission.
[[nodiscard]] bool RevalidateSrtSnapshot(const SrtCaptureResult& capture,
                                         SrtMemoryProvider& provider);

// JIT-only entry point; do not call outside an active capture session.
extern "C" std::uint64_t PS4_SYSV_ABI ReadCapturedSrt(std::uint64_t address,
                                                        std::uint32_t size) noexcept;

} // namespace iss::g4
