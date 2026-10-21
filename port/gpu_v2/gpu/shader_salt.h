// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
// ISS_SHADER_SALT=<hex> (measurement only): every SPIR-V module handed to the driver differs from
// earlier runs (generator word xor the salt, id bound raised), so neither the driver's own disk
// cache nor the port's caches recognise it: a run measures compiles as a player who never saw
// the shaders. The SPIR-V stays valid (the generator word is informational, a larger bound only
// reserves ids). The port's own cache keeps the unsalted words.
#include <cstdlib>
#include <span>
#include <vector>
#include "common/types.h"

namespace iss::shader_salt {
inline u32 Value() {
    static const u32 salt = [] {
        const char* p = std::getenv("ISS_SHADER_SALT");
        return p && *p ? u32(std::strtoul(p, nullptr, 16)) : 0u;
    }();
    return salt;
}
/// The words to create the module from: `code` itself, or a salted copy kept in `storage`.
inline std::span<const u32> Apply(std::span<const u32> code, std::vector<u32>& storage) {
    const u32 salt = Value();
    if (!salt || code.size() < 5) return code;
    storage.assign(code.begin(), code.end());
    storage[2] ^= salt;
    storage[3] += 1 + (salt & 0xF);
    return storage;
}
} // namespace iss::shader_salt
