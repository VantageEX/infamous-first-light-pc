// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>
#include "common/types.h"

// Immutable, profile-matched recipes. Only initialized/consumed during startup;
// no guest state, driver binaries or runtime speculative reads live here.
namespace iss::pipeline_library {
bool Open(std::span<const u8> profile, const std::filesystem::path& primary);
bool Read(const std::string& filename, std::vector<u8>& out);
void ForEachKey(const std::function<void(std::vector<u8>&&)>& callback);
/// Every recipe file (name, bytes) of the open library (ISS_AOT_RECIPES prepares the stages).
void ForEachFile(const std::function<void(const std::string&, const std::vector<u8>&)>& callback);
/// The keys ForEachKey will hand out (the loading screen's total).
size_t NumKeys();
void Finish(u32 accepted, u32 rejected);
void Discard();
void LogMiss(const char* kind, u64 key, u64 guest_hash, u64 duration_ns);
} // namespace iss::pipeline_library
