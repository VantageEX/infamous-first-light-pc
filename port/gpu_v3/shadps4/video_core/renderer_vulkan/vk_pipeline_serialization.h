// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <string_view>
#include <unordered_set>

#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

// The on-disk cache is named per shader binary version (cache_storage.cpp Open).
u32 GetShaderBinaryVersion();

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata);
void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata);
void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash, size_t perm_idx,
                        bool precompiled = false);
void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
// SHADPS4_PRECOMPILE: the game's first dispatch of a pre-compiled program, matched or not.
void NotePrecompiledUse(const Program& program, const Shader::StageSpecialization& live, bool hit);
// SHADPS4_PRECOMPILE sidecars in the cache folder (vk_pipeline_precompile.cpp): one program hash
// per line, hex. Written synchronously (flushed), so a crash right after keeps the line.
// retired: programs retired after a live miss (fix A); their guessed metas are never loaded.
inline constexpr std::string_view PrecompileRetiredFile = "precompile-retired.txt";
std::unordered_set<u64> ReadPrecompileHashes(std::string_view file_name);
void AppendPrecompileHash(std::string_view file_name, u64 hash);
// `precompiled` (optional): set when the meta carries the SHADPS4_PRECOMPILE marker.
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx,
                    bool* precompiled = nullptr);
// True when two translations of one program give the same descriptor bindings (buffer kinds and
// order, images, samplers, fmasks). Permutations share one Info, so they must agree on these.
bool SameBindingLayout(const Shader::Info& a, const Shader::Info& b);
// Boot passes on the main thread (vk_pipeline_precompile.cpp): window title progress + event pump
// at most every 250 ms; total == ~0ull puts the title back. one_time: "Preparing shaders (one
// time)" (stages are rebuilt this boot) instead of "Loading shaders". fraction >= 0: the whole
// boot's progress (0..1) for the SHADPS4_BOOT_SCREEN screen.
void ShowBootProgress(std::string_view what, u64 done, u64 total, bool one_time = true,
                      float fraction = -1.0f);
// SHADPS4_BOOT_SCREEN=1 (vk_presenter.cpp): draws and presents one boot frame (headline, note,
// progress bar) while a Presenter is building its rasterizer; a no-op otherwise.
void PresentBootScreen(std::string_view headline, std::string_view detail, float fraction,
                       bool one_time);
// 1.1 (user 10-21): keeps re-presenting that last boot frame (events pumped) until min_shown has
// passed since the first one, so the version line on it can be read; a no-op without boot frames.
void HoldBootScreen(std::string_view headline, std::string_view detail, bool one_time,
                    std::chrono::milliseconds min_shown);
// SHADPS4_CACHE_REBUILD seed stamp (vk_pipeline_precompile.cpp): the game file's OrbShdr crc32 and
// code length of a shader, from the game folder's shader index (built on first use, main thread).
bool GameShaderIdentity(u64 pgm_hash, u32* crc32, u32* code_length);
// SHADPS4_CACHE_REBUILD: SameBindingLayout plus every field of each buffer / image / sampler /
// fmask resource and the flattened user-data size (study sec. 5 finding 2: the rebuilt
// translation must bind exactly as the cached Info it is bound through).
bool SameResourceFields(const Shader::Info& a, const Shader::Info& b);

} // namespace Vulkan
