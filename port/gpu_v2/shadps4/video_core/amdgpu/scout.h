// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

/**
 * Scout (Tuning::scout): a thread reading the graphics submissions ahead of the command
 * processor. It replays the register writes of the command stream in a copy of the registers
 * and, at every draw, precomputes what depends only on the registers and on guest memory at
 * that point (ScoutClient), into entries the command processor finds at the same draws.
 *
 * Every entry is checked against the command processor's own state before use (program
 * addresses, user data): a mismatch, or a conditional execution the scout decided otherwise
 * (CondExec reads memory), stops the use of the submission's entries.
 */

/// The precomputed lookup of one shader stage of a draw (PipelineCache).
struct ScoutStage {
    bool valid{};
    /// The permutation was found too (memo of the program, vertex attributes checked): the
    /// command processor can take the whole lookup (Tuning::scout_fast).
    bool resolved{};
    u8 hw_stage{};
    u8 sw_stage{};
    u8 num_image_keys{};    ///< Tuning::bind_prefetch, see image_keys.
    u8 num_buffer_ranges{}; ///< Tuning::bind_prefetch, see buffer_addresses.
    VAddr pgm_base{};
    u64 ud_hash{};  ///< XXH3 of the user data registers of the stage.
    u64 memo_key{}; ///< Permutation memo key of the stage (PipelineCache::SpecializationKey).
    std::array<u32, NUM_USER_DATA> user_data{}; ///< The user data registers of the stage.
    const void* program{}; ///< The program (PipelineCache, stable once added).
    u64 params_hash{};     ///< Hash of the shader (its binary info).
    u32 code_dwords{};     ///< Size of the shader code.
    u32 perm_idx{};        ///< The permutation, when resolved.
    Shader::Backend::Bindings start{};
    const u32* flat{}; ///< Flattened user data (scout arena of the submission).
    u32 flat_size{};
    /// Tuning::bind_prefetch: the lookups of the stage's resources, from its flattened user
    /// data: texture memo keys of the image resources in order (first binding), and the guest
    /// ranges of the buffers read only. The command processor prefetches their memo entries a
    /// draw ahead, and takes the texture keys with the stage (the same flattened user data).
    static constexpr u32 MaxImageKeys = 8;
    static constexpr u32 MaxBufferRanges = 8;
    std::array<u64, MaxImageKeys> image_keys{};
    std::array<VAddr, MaxBufferRanges> buffer_addresses{};
    std::array<u32, MaxBufferRanges> buffer_sizes{};
    /// The runtime info (no default constructor: kept as bytes).
    alignas(Shader::RuntimeInfo) std::array<std::byte, sizeof(Shader::RuntimeInfo)> runtime_info{};

    void SetRuntimeInfo(const Shader::RuntimeInfo& info) noexcept {
        std::memcpy(runtime_info.data(), &info, sizeof(info));
    }
    Shader::RuntimeInfo GetRuntimeInfo() const noexcept {
        return *reinterpret_cast<const Shader::RuntimeInfo*>(runtime_info.data());
    }
};

struct ScoutDraw {
    static constexpr u32 MaxStages = 6;
    const void* packet{}; ///< The draw packet (its header address in the command buffer).
    u16 vertex_sgpr_offset{};   ///< Indirect draws: the user data the vertex offset goes to.
    u16 instance_sgpr_offset{}; ///< Indirect draws: the user data the instance offset goes to.
    u32 num_stages{};
    std::array<ScoutStage, MaxStages> stages{};
};

class Scout;

/// Dword arena of a submission: blocks never move, the entries point into them. The blocks
/// come from the scout's pool and go back to it with the submission.
class ScoutArena {
public:
    static constexpr u32 BlockDwords = 64 * 1024;
    /// Blocks kept for reuse by the scout (16 MB), the others are freed.
    static constexpr size_t MaxPooledBlocks = 64;
    using Block = std::unique_ptr<u32[]>;

    explicit ScoutArena(Scout& owner_) : owner{&owner_} {}

    u32* Allocate(u32 dwords);
    /// Gives the blocks back to the pool (its lock held by the caller).
    void Reset(std::vector<Block>& pool);

private:
    Scout* owner;
    std::vector<Block> blocks;
    u32 used{BlockDwords};
};

/// A graphics submission scouted (scout thread) and consumed (command processor thread).
class ScoutSubmission {
public:
    static constexpr u32 ChunkBits = 7;
    static constexpr u32 ChunkSize = 1u << ChunkBits;
    static constexpr u32 MaxChunks = 2048;
    static constexpr u32 MaxConditions = 4096;
    using Chunk = std::unique_ptr<ScoutDraw[]>;

    explicit ScoutSubmission(Scout& scout, u64 seq, std::span<const u32> dcb);
    ~ScoutSubmission();

    u64 Seq() const noexcept {
        return seq;
    }

    // Scout thread.
    ScoutDraw* NewDraw(const void* packet);
    void Publish();
    void RecordCondition(const void* packet, bool skip);
    void Abandon();
    ScoutArena& Arena() noexcept {
        return arena;
    }

    // Command processor thread.
    /// The entry of the draw packet, null when the scout did not reach it, decided another
    /// path, or was abandoned. Called once per draw packet, in packet order.
    const ScoutDraw* Next(const void* packet) noexcept;
    /// The entry Next returns at the next draw packet, when already published (not checked
    /// against the packet: for prefetching only).
    const ScoutDraw* Peek() const noexcept;
    /// A conditional execution: must match the scout's decision for its entries to be used.
    void CheckCondition(const void* packet, bool skip) noexcept;

private:
    Scout& scout;
    const u64 seq;
    const std::span<const u32> dcb;
    ScoutArena arena;
    // Entries, published in packet order.
    std::array<std::atomic<ScoutDraw*>, MaxChunks> chunks{};
    std::vector<Chunk> chunk_storage;
    u32 num_draws{};
    std::atomic<u32> published{};
    std::atomic<bool> abandoned{};
    u32 cursor{}; ///< Command processor: next entry.
    // Conditional executions, in packet order.
    struct Condition {
        const void* packet;
        bool skip;
    };
    std::unique_ptr<Condition[]> conditions;
    u32 num_conditions{};
    std::atomic<u32> published_conditions{};
    u32 condition_cursor{};
    bool desynced{};

    friend class Scout;
};

class ScoutClient {
public:
    virtual ~ScoutClient() = default;

    /// Precomputes the stages of a draw from the registers at that point. Scout thread.
    virtual void ScoutDraw(const Regs& regs, ScoutDraw& draw, ScoutArena& arena) = 0;
};

class Scout {
public:
    explicit Scout(ScoutClient& client);
    ~Scout();

    /// A graphics submission (dcb only, no constant engine), in submission order. Guest thread.
    void Submit(u64 seq, std::span<const u32> dcb);

    /// The scout is off for this submission: its registers go out of date, the next scouted
    /// submission starts from a copy of the command processor's (Begin).
    void Skip(u64 seq);

    /// Command processor: the scouted submission to consume, null when it has none. The
    /// registers are those of the command processor before the submission: copied when the
    /// scout needs them again (turned on, or late enough to have skipped a submission).
    std::shared_ptr<ScoutSubmission> Begin(u64 seq, const Regs& cp_regs);

    /// Command processor: done with the submission (its memory writes are done).
    void Finish(u64 seq);

    /// Scout thread: whether guest memory in the range may be written by a command processor
    /// packet it has not processed yet (so reading it now would read stale data).
    bool MayBeWrittenLater(VAddr address, u64 size) const noexcept;

    // Pools (any thread).
    ScoutArena::Block TakeBlock();
    ScoutSubmission::Chunk TakeChunk();
    void ReturnChunk(ScoutSubmission::Chunk chunk);
    void ReturnBlocks(std::vector<ScoutArena::Block>& blocks);
    std::vector<ScoutArena::Block>& BlockPool() noexcept {
        return block_pool;
    }
    std::mutex& PoolMutex() noexcept {
        return pool_mutex;
    }

private:
    void Run(std::stop_token stoken);
    void Walk(ScoutSubmission& sub, std::span<const u32> dcb, u32 depth);
    void NoteWrite(VAddr address, u64 size);

    ScoutClient& client;
    Regs regs{}; ///< The command processor's registers, replayed from a copy of them.
    bool regs_valid{};              ///< Scout thread.
    std::atomic<bool> want_regs{true}; ///< Begin copies the command processor's registers.
    std::unique_ptr<Regs> regs_copy; ///< Under mutex: the registers before regs_copy_seq.
    u64 regs_copy_seq{};
    u64 current_seq{};
    struct WrittenRange {
        VAddr begin;
        VAddr end;
        u64 seq;
    };
    std::vector<WrittenRange> written; ///< Scout thread only.

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<ScoutSubmission>> pending; ///< To scout.
    std::deque<std::shared_ptr<ScoutSubmission>> live;    ///< Not finished by the processor.
    std::atomic<u64> cp_seq{};          ///< The submission the command processor is on.
    std::atomic<u64> cp_finished_seq{}; ///< Submissions up to it are done.

    std::mutex pool_mutex;
    std::vector<ScoutSubmission::Chunk> chunk_pool;
    std::vector<ScoutArena::Block> block_pool;

    std::jthread thread;
};

} // namespace AmdGpu
