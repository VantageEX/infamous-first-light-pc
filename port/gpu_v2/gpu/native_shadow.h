// SPDX-License-Identifier: GPL-2.0-or-later
// ISS_NATIVE_SHADOW=1: the shadow maps drawn from the engine's batches instead of their PM4.
//
// The engine draws every instanced shadow caster through one routine, the depth technique's
// "render item range" method (eboot 0x43D590, vtable 0xD68648 slot 6; evidence
// claude-shadow-native-mapping-20261015.md). A hook on it, on the engine's thread, writes a
// marker NOP before the commands the routine writes, lets the routine run unchanged and then
// reads back what it wrote: each draw's geometry, instance records and render state, and the
// final value of every register the range sets. The command processor, at the marker, records
// the draws itself and jumps over the range, or interprets the range as usual and learns the
// pipelines of its render states for the next frames (native_shadow_rasterizer.inl).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace iss::native_shadow {

constexpr std::uint32_t MarkerMagic = 0x49534800u; // 'ISH' and the marker kind
constexpr std::uint32_t MarkerBegin = 1;
constexpr std::uint32_t MarkerDwords = 4;          // NOP header, magic, slot, generation
constexpr std::uint32_t MaxStateRegs = 64;
constexpr std::uint32_t VsUserData2 = 0x2C4E;      // SPI_SHADER_USER_DATA_VS_2 (absolute word)
constexpr std::uint32_t DrawUserWords = 6;         // VS user data 2..7: fetch V# list, SSBO V#

// One draw of a range, as the registers stand at its DRAW_INDEX_2. Values the range had not
// set yet at that draw come from the registers before the range (inherit bits).
struct Draw {
    const std::uint32_t* packet{};                  // the DRAW_INDEX_2 packet
    std::array<std::uint32_t, DrawUserWords> user{};
    std::uint32_t user_inherit{};                   // bit per user word
    std::uint32_t index_lo{}, index_hi{}, index_count{}, max_index{}, initiator{};
    std::uint32_t instances{}, index_type{};
    bool instances_inherit{}, index_type_inherit{};
    std::uint32_t state{};                          // index into Batch::states
};

// The registers the range writes besides the per draw data (Batch::state_regs): one version per
// change, with their values (inherited from before the range where the bit is set).
struct State {
    std::uint64_t inherit{};
    std::uint32_t first_value{}; // index into Batch::state_values
};

struct Batch {
    std::atomic<std::uint32_t> ready{}; // generation, published last
    const std::uint32_t* marker{};      // the marker NOP
    const std::uint32_t* end{};         // the first packet after the range
    std::vector<std::uint32_t> state_regs;
    std::vector<std::uint32_t> state_values;
    std::vector<State> states;
    std::vector<Draw> draws;
    // Final value of every register the range writes (absolute word index, value).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> final_regs;
    // The last NUM_INSTANCES and INDEX_TYPE of the range, when it has one.
    std::uint32_t final_instances{}, final_index_type{};
    bool has_instances{}, has_index_type{};
};

bool Enabled();
// Hooks the engine routine (from iss_gpu_initialize, before the game renders). 1 installed,
// 0 disabled, -1 refused.
int Install(std::uint64_t image_base, const unsigned char* code);
// The batch a marker refers to, when it is complete and still describes that marker.
const Batch* Find(const std::uint32_t* marker);
// Command processor statistics: reason 0 native, else why the range was interpreted.
enum Reason : std::uint32_t { Native, Partial, Learn, Context, Missing, Mismatch, Verify,
                              ReasonCount };
void NoteRange(Reason reason, std::uint32_t draws);

} // namespace iss::native_shadow
