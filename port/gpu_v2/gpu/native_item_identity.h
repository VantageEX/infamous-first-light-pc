// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace iss::native_item {
struct Identity {
    std::uint64_t technique{}, geometry{}, material{}, item{};
    bool operator==(const Identity&) const = default;
};
inline bool Flag(const char* name) {
    const auto* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}
inline bool Enabled() {
    static const bool enabled = Flag("ISS_ENGINE_MARKERS") && Flag("ISS_NATIVE_ITEM_RANGES") &&
        (Flag("ISS_NATIVE_ITEMS") || Flag("ISS_NATIVE_VERIFY"));
    return enabled;
}
inline thread_local const Identity* current{};
// Construct only around synchronous rasterizer recording, never across coroutine suspension.
struct Scope {
    explicit Scope(const Identity* identity) { current = identity; }
    ~Scope() { current = nullptr; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
enum class Reject { None, Disabled, Entry, Truncated, MissingEnd, IdentityMismatch,
                    Nested, MultipleDraws, NoDraw, Unsupported, Budget };
struct Match {
    Identity identity{};
    const std::uint32_t* draw{};
    Reject reject{Reject::Entry};
};
inline void NoteScan(Reject reject) {
    static thread_local std::array<std::uint64_t, 12> counts{};
    static thread_local std::uint64_t total{};
    ++counts[static_cast<unsigned>(reject)];
    if ((++total & 4095) == 1) {
        std::fprintf(stderr, "NATIVE_ITEM_RANGES scans=%llu counts=", static_cast<unsigned long long>(total));
        for (const auto n : counts) std::fprintf(stderr, "%llu,", static_cast<unsigned long long>(n));
        std::fputc('\n', stderr);
    }
}
inline Identity ReadIdentity(const std::uint32_t* p) {
    Identity id;
    std::memcpy(&id.technique, p + 2, 8);
    std::memcpy(&id.geometry, p + 4, 8);
    std::memcpy(&id.material, p + 6, 8);
    std::memcpy(&id.item, p + 8, 8);
    return id;
}
// The span must be the remaining words of ONE graphics DCB, starting at the entry NOP.
// Whitelist only linear state commands. Unknown commands and all control flow fail closed.
inline Match Scan(std::span<const std::uint32_t> words, bool enabled) {
    Match result;
    const auto fail = [&](Reject why) { result.draw = nullptr; result.reject = why; return result; };
    if (!enabled) return fail(Reject::Disabled);
    if (words.size() < 10) return fail(Reject::Truncated);
    if (words[0] != 0xc0081000u || words[1] != 0x49535301u) return fail(Reject::Entry);
    result.identity = ReadIdentity(words.data());
    constexpr std::size_t MaxWords = 4096;
    std::size_t pos = 10;
    while (pos < words.size()) {
        if (pos >= MaxWords) return fail(Reject::Budget);
        const auto h = words[pos];
        if ((h >> 30) == 2) { ++pos; continue; }
        if ((h >> 30) != 3 || (h & 1)) return fail(Reject::Unsupported);
        const std::size_t size = ((h >> 16) & 0x3fff) + 2;
        if (size > words.size() - pos) return fail(Reject::Truncated);
        if (size > MaxWords - pos) return fail(Reject::Budget);
        const auto* p = words.data() + pos;
        const auto opcode = (h >> 8) & 255;
        if (opcode == 0x10) {
            if ((p[1] & 0xffffff00u) == 0x49535300u) {
                if (size != 10) return fail(Reject::Unsupported);
                const auto kind = p[1] & 255;
                if (kind == 5) {
                    if (!(ReadIdentity(p) == result.identity)) return fail(Reject::IdentityMismatch);
                    if (!result.draw) return fail(Reject::NoDraw);
                    result.reject = Reject::None;
                    return result;
                }
                if (kind != 4) return fail(Reject::Nested);
            } else if (p[1] != 0 && p[1] != 0x6875000du) {
                return fail(Reject::Unsupported); // CP NOPs can carry flip/debug actions.
            }
        } else if (opcode == 0x27) {
            if (size != 6) return fail(Reject::Unsupported);
            if (result.draw) return fail(Reject::MultipleDraws);
            result.draw = p;
        } else {
            switch (opcode) {
            case 0x26: case 0x2a: case 0x2f: // index base/type and instance count
            case 0x68: case 0x69: case 0x76: case 0x79: // direct register writes
                break;
            default: return fail(Reject::Unsupported);
            }
        }
        pos += size;
    }
    return fail(Reject::MissingEnd);
}
} // namespace iss::native_item
