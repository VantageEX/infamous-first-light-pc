// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_GBUFFER=1: the G-buffer drawn from the engine's material ranges instead of their PM4.
//
// The G-buffer technique (vtable 0xD0FAC8) draws its instanced items through the generic material
// range routine (eboot 0x1A2030, slot 6; contract out/experimental/20261015-gbuffer/
// NATIVE-GBUFFER-CONTRACT.md). A hook on it, on the engine's thread, writes a marker NOP before
// the commands of the range, lets the routine run unchanged and reads back what it wrote: each
// draw's geometry, the pointers of the per draw resource tables (vertex and pixel stage user
// data, copied by the engine for every draw) and the render state versions of the range. The
// command processor, at the marker, records the draws itself and jumps over the range, or
// interprets it and learns the pipelines of its render states (native_gbuffer_rasterizer.inl).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace iss::native_gbuffer {

constexpr std::uint32_t MarkerMagic = 0x49534700u; // 'ISG' and the marker kind
constexpr std::uint32_t MarkerBegin = 1;
constexpr std::uint32_t MarkerDwords = 4;          // NOP header, magic, slot, generation
constexpr std::uint32_t MaxStateRegs = 64;
constexpr std::uint32_t UserWords = 16;
constexpr std::uint32_t PsUser0 = 0x2C0C;          // SPI_SHADER_USER_DATA_PS_0
constexpr std::uint32_t VsUser0 = 0x2C4C;          // SPI_SHADER_USER_DATA_VS_0

inline bool IsPsUser(std::uint32_t reg) { return reg >= PsUser0 && reg < PsUser0 + UserWords; }
inline bool IsVsUser(std::uint32_t reg) { return reg >= VsUser0 && reg < VsUser0 + UserWords; }

// One draw, as the registers stand at its DRAW_INDEX_2. Words the range had not set yet at that
// draw come from the registers before the range (inherit bits).
struct Draw {
    const std::uint32_t* packet{};                  // the DRAW_INDEX_2 packet
    std::array<std::uint32_t, UserWords> vs_user{}, ps_user{};
    std::uint32_t vs_inherit{}, ps_inherit{};       // bit per user word
    std::uint32_t index_lo{}, index_hi{}, index_count{}, max_index{}, initiator{};
    std::uint32_t instances{}, index_type{};
    bool instances_inherit{}, index_type_inherit{};
    std::uint32_t state{};                          // index into Batch::states
};

struct State {
    std::uint64_t inherit{};
    std::uint32_t first_value{}; // index into Batch::state_values
};

struct Recording; // ISS_NATIVE_RECORDER: the range recorded by the engine thread (rasterizer)

struct Batch {
    std::atomic<std::uint32_t> ready{}; // generation, published last
    std::uint32_t generation{};
    Recording* recording{};             // set before ready, null when not recorded
    const std::uint32_t* marker{};
    const std::uint32_t* end{};         // the first packet after the range
    std::uint64_t technique{};          // the engine technique object that wrote the range
    std::vector<std::uint32_t> state_regs; // sorted
    std::vector<std::uint32_t> state_values;
    std::vector<State> states;
    std::vector<Draw> draws;
    // Final value of every register the range writes (absolute word index, value).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> final_regs;
    std::uint32_t final_instances{}, final_index_type{};
    bool has_instances{}, has_index_type{};
};

bool Enabled();
int Install(std::uint64_t image_base);

// ISS_NATIVE_RECORDER=1: the engine thread that wrote a range records it into Vulkan right away
// (native_recorder.inl), from the registers it knows at the range start: those its command
// buffer wrote since the last baseline the command processor published, the baseline for the
// others. The command processor checks them against its own before using the recording.
// Registers followed: graphics context, vertex/pixel shader, VGT user config.
constexpr std::uint32_t TrackedFirst[3] = {0xA000, 0x2C00, 0xC240};
constexpr std::uint32_t TrackedCount[3] = {0x400, 0x80, 0x20};
constexpr std::uint32_t TrackedWords = 0x400 + 0x80 + 0x20;
struct RangeStart {
    std::array<std::uint32_t, TrackedWords> words{};
    std::uint32_t index_type{}, instances{};
    bool known{}; // false: a packet the tracking does not follow (the range is not recorded)
};
using RecordFn = Recording* (*)(const Batch& batch, const RangeStart& start);
/// Links a recorded range to the previous one of its command buffer when only register writes
/// (no target register, no other packet) lie between them: the command processor may then
/// execute the whole chain at the first one and jump to the end of the last.
using LinkFn = void (*)(const Batch& previous, const Batch& next);
bool RecorderEnabled();
void SetRecorder(RecordFn fn, LinkFn link);
/// Command processor: its registers at a range start (reg_array of AmdGpu::Regs).
void PublishBaseline(const std::uint32_t* reg_array, std::uint32_t index_type,
                     std::uint32_t instances);
std::uint32_t LatestGeneration();
/// Command processor: its registers at the start of a command buffer it processes. The engine
/// thread that writes the command buffer at that address next starts from them (a frame draws
/// much like the previous one), instead of the latest baseline.
void PublishSpanStart(const std::uint32_t* span, const std::uint32_t* reg_array,
                      std::uint32_t index_type, std::uint32_t instances);
/// ISS_NATIVE_RECORDER_TECH_START=1, command processor: its registers at the first range of a
/// command buffer, by the technique that wrote the range. An engine thread whose tracking of a
/// command buffer starts over (a new chunk) starts from them when its first range is of that
/// technique, then applies what the chunk wrote: command buffer addresses are reused by other
/// passes from frame to frame, techniques draw their pass alike.
bool TechStartEnabled();
void PublishTechniqueStart(std::uint64_t technique, const std::uint32_t* reg_array,
                           std::uint32_t index_type, std::uint32_t instances);
const Batch* Find(const std::uint32_t* marker);
enum Reason : std::uint32_t { Native, Partial, Learn, Context, Missing, Mismatch, Verify,
                              ReasonCount };
void NoteRange(Reason reason, std::uint32_t draws);

} // namespace iss::native_gbuffer
