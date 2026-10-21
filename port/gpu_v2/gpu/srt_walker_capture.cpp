// SPDX-License-Identifier: GPL-2.0-or-later
#include "srt_walker_capture.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>


namespace iss::g4 {
namespace {

constexpr std::size_t MaxSrtWords = 64 * 1024;
constexpr std::size_t MaxSrtReads = 64 * 1024;
constexpr std::size_t MaxExtraReadBytes = 128 * 1024;

struct CaptureSession {
    SrtMemoryProvider& provider;
    SrtCaptureResult& result;
    bool failed{};

    bool CaptureRange(std::uint64_t address, std::uint32_t size) {
        if (failed || size == 0 || address == 0 ||
            address > std::numeric_limits<std::uint64_t>::max() - size ||
            result.reads.size() == MaxSrtReads) {
            failed = true;
            return false;
        }
        SrtMemoryVersion before{}, after{};
        if (!provider.Version(address, size, before) || before.mapping_generation == 0) {
            failed = true;
            return false;
        }
        SrtReadSnapshot read{};
        read.guest_address = address;
        read.memory_generation = before.mapping_generation;
        read.bytes.resize(size);
        if (!provider.CopySparse(address, read.bytes) ||
            !provider.Version(address, size, after) || after != before) {
            failed = true;
            return false;
        }
        result.reads.push_back(std::move(read));
        result.read_versions.push_back(std::move(before));
        return true;
    }

    std::uint64_t Read(std::uint64_t address, std::uint32_t size) {
        if ((size != 4 && size != 8) || !CaptureRange(address, size)) return 0;
        std::uint64_t value{};
        std::memcpy(&value, result.reads.back().bytes.data(), size);
        return value;
    }
};

thread_local CaptureSession* active_capture{};

} // namespace

extern "C" std::uint64_t PS4_SYSV_ABI ReadCapturedSrt(std::uint64_t address,
                                                        std::uint32_t size) noexcept {
    if (!active_capture) return 0;
    try {
        return active_capture->Read(address, size);
    } catch (...) {
        active_capture->failed = true;
        return 0;
    }
}

bool RevalidateSrtSnapshot(const SrtCaptureResult& capture,
                           SrtMemoryProvider& provider) {
    if (capture.proof.coverage != SrtCoverage::Complete ||
        capture.reads.size() != capture.read_versions.size()) return false;
    if (capture.proof.no_guest_reads && capture.reads.empty()) return true;
    try {
        if (capture.capture_epoch != 0) {
            const auto epoch_before = provider.GlobalEpoch();
            if (!epoch_before || *epoch_before != capture.capture_epoch) return false;
        } else if (!capture.watch_lease || !capture.watch_lease->Valid() ||
                   !capture.watch_lease->Covers(capture.reads)) {
            return false;
        }
        for (std::size_t i = 0; i < capture.reads.size(); ++i) {
            const auto& read = capture.reads[i];
            SrtMemoryVersion current{};
            if (!provider.Version(read.guest_address, read.bytes.size(), current) ||
                current != capture.read_versions[i]) return false;
            std::vector<std::byte> bytes(read.bytes.size());
            if (!provider.CopySparse(read.guest_address, bytes) || bytes != read.bytes ||
                !provider.Version(read.guest_address, read.bytes.size(), current) ||
                current != capture.read_versions[i]) return false;
        }
        if (capture.capture_epoch != 0) {
            const auto epoch_after = provider.GlobalEpoch();
            if (!epoch_after || *epoch_after != capture.capture_epoch) return false;
        } else if (!capture.watch_lease->Valid()) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SrtCaptureResult CaptureSrtSnapshot(const Shader::PersistentSrtInfo& info,
                                    std::span<const std::uint32_t> user_data,
                                    SrtMemoryProvider& provider,
                                    std::span<const SrtExtraRead> extra_reads) {
    SrtCaptureResult result{};
    if (active_capture || user_data.size() > FallbackUserDataWords ||
        info.flattened_bufsize_dw < FallbackUserDataWords ||
        info.flattened_bufsize_dw > MaxSrtWords) return result;
    if (info.walker_func &&
        (!info.capture_walker_func || info.capture_schema != SrtCaptureSchema ||
         info.capture_walker_func_size == 0 || info.capture_relocations.empty())) return result;
    std::size_t extra_bytes{};
    for (const auto extra : extra_reads) {
        if (!extra.guest_address || !extra.size ||
            extra.guest_address > std::numeric_limits<std::uint64_t>::max() - extra.size ||
            extra.size > MaxExtraReadBytes - extra_bytes) return result;
        extra_bytes += extra.size;
    }

    std::array<std::uint32_t, FallbackUserDataWords> owned_ud{};
    std::copy(user_data.begin(), user_data.end(), owned_ud.begin());
    const auto run = [&]() {
        SrtCaptureResult pass{};
        pass.flattened.resize(info.flattened_bufsize_dw);
        std::copy(user_data.begin(), user_data.end(), pass.flattened.begin());
        CaptureSession session{provider, pass};
        active_capture = &session;
        try {
            if (info.capture_walker_func) {
                info.capture_walker_func(owned_ud.data(), pass.flattened.data());
            }
            for (const auto extra : extra_reads) {
                if (!session.CaptureRange(extra.guest_address, extra.size)) break;
            }
        } catch (...) {
            session.failed = true;
        }
        active_capture = nullptr;
        if (session.failed) pass.proof = {};
        return std::pair{std::move(pass), !session.failed};
    };

    const auto epoch_before = provider.GlobalEpoch();
    auto [first, first_ok] = run();
    if (!first_ok) return first;
    if (!info.capture_walker_func && extra_reads.empty()) {
        first.proof = {SrtCoverage::Complete, true, true, true, true};
        return first; // user data was copied; no guest read exists
    }
    const auto epoch_after = provider.GlobalEpoch();
    if (epoch_before && *epoch_before != 0 && epoch_after == epoch_before) {
        first.capture_epoch = *epoch_before;
        first.proof = {SrtCoverage::Complete, true, true, true};
        if (RevalidateSrtSnapshot(first, provider)) return first;
        first.proof = {};
    }

    // An exploratory read set is not a proof. Arm *all* physical aliases, then rerun
    // the walker. A dynamic pointer that escapes the watched set forces another pass.
    std::vector<SrtReadSnapshot> discovered = first.reads;
    auto watch = provider.ArmWatchSet(discovered);
    if (!watch) return first;
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        auto [pass, ok] = run();
        if (!ok) return pass;
        if (!watch->Valid()) {
            discovered.insert(discovered.end(), pass.reads.begin(), pass.reads.end());
            watch = provider.ArmWatchSet(discovered);
            if (!watch) return pass;
            continue;
        }
        if (!watch->Covers(pass.reads)) {
            discovered.insert(discovered.end(), pass.reads.begin(), pass.reads.end());
            auto expanded = provider.ArmWatchSet(discovered);
            if (!expanded) return pass;
            watch = std::move(expanded);
            continue;
        }
        pass.watch_lease = watch;
        pass.proof = {SrtCoverage::Complete, true, true, true};
        if (RevalidateSrtSnapshot(pass, provider)) return pass;
        pass.proof = {};
        if (watch->Valid()) return pass; // bytes changed without an invalidation
        discovered.insert(discovered.end(), pass.reads.begin(), pass.reads.end());
        watch = provider.ArmWatchSet(discovered);
        if (!watch) return pass;
    }
    return first;
}

} // namespace iss::g4
