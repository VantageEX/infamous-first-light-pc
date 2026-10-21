// SPDX-License-Identifier: GPL-2.0-or-later
// Linux: renderer v2 behind this port's GPU library interface (gpu/tlggpu.h). The
// runtime (src/runtime_*.c) calls the tlggpu_* functions; they drive the renderer through its own
// host interface (iss_gpu.h), whose callbacks are the runtime's memory functions.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <imgui.h>
#include <SDL3/SDL.h>
#include "iss_gpu.h"
#include "tlggpu.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "sdl_window.h"
#include "tlg_overlay.h"
#include "tlg_png.h"
#include "tlg_settings.h"

extern "C" {
// runtime_memory.c
void* runtime_memory_backing_view(uintptr_t address, uint64_t size);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
void runtime_memory_visit(void (*visit)(uint64_t address, uint64_t size, uint8_t* backing));
uint64_t runtime_memory_epoch(void);
uint64_t runtime_memory_lock_shared(void);
void runtime_memory_unlock_shared(void);
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
// runtime_kernel.c
uint64_t runtime_process_time_us(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
// runtime.c
void runtime_report(void);
// bridge.cpp
void iss_gpu_install_image(uint64_t image_base);
}

extern Frontend::WindowSDL* g_window;
namespace {
std::string g_title, g_serial, g_user_dir;
std::atomic<uint64_t> g_flips{0};

uint8_t* Backing(uint64_t address, uint64_t size) {
    return static_cast<uint8_t*>(runtime_memory_backing_view(address, size));
}
uint64_t Clamp(uint64_t address, uint64_t size) {
    return runtime_memory_clamp(address, size);
}
void VisitMappings(void (*visit)(uint64_t address, uint64_t size, uint8_t* backing)) {
    runtime_memory_visit(visit);
}
int Protect(uint64_t address, uint64_t size, uint32_t permissions) {
    runtime_memory_gpu_protect(address, size, (permissions & 1) != 0, (permissions & 2) != 0);
    return 1;
}
} // namespace

[[noreturn]] void IssHostExit() {
    runtime_report();
    std::fflush(nullptr);
    std::_Exit(0);
}

extern "C" void tlggpu_register_kernel(void) {
    // The renderer registers the kernel event queues itself (bridge.cpp, its init thread).
}

extern "C" void tlggpu_init_headless(const TlgGpuConfig*) {}

extern "C" int tlggpu_init(const TlgGpuConfig* config) {
    g_title = config->title ? config->title : "inFAMOUS Second Son";
    g_serial = config->serial ? config->serial : "CUSA00223";
    g_user_dir = config->user_dir ? config->user_dir : ".";
    // Mappings made from now on reach the renderer; the existing ones are replayed now (they
    // are registered again, with the renderer, when it starts: MemoryManager::SetRasterizer).
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) { iss_gpu_map(address, size, Backing(address, size)); },
        [](uintptr_t address, uint64_t size) {
            if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->UnmapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) {
            if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->InvalidateMemory(address, size);
        });
    static IssGpuCallbacks callbacks{};
    callbacks.struct_size = sizeof(callbacks);
    callbacks.sdk_version = config->sdk_version;
    callbacks.psf_attributes = config->psf_attributes;
    callbacks.width = static_cast<uint32_t>(config->width);
    callbacks.height = static_cast<uint32_t>(config->height);
    callbacks.serial = g_serial.c_str();
    callbacks.title = g_title.c_str();
    callbacks.user_dir = g_user_dir.c_str();
    callbacks.tsc_frequency = runtime_tsc_frequency;
    callbacks.process_time_us = runtime_process_time_us;
    callbacks.errno_location = runtime_errno;
    callbacks.backing = Backing;
    callbacks.clamp = Clamp;
    callbacks.visit_mappings = VisitMappings;
    callbacks.protect = Protect;
    callbacks.image_base = 0; // tlggpu_set_guest_image
    callbacks.mapping_epoch = runtime_memory_epoch;
    callbacks.mapping_lock_shared = runtime_memory_lock_shared;
    callbacks.mapping_unlock_shared = runtime_memory_unlock_shared;
    TlgSettings::Load(); // tlg.ini: the overlay, the menu and the input settings
    // The launcher's choices (backports_table.py): TLG_RENDERER=2 picked this renderer, and
    // TLG_NATIVE_PASSES sets ISS_NATIVE_PASSES (off by default: NVIDIA's Linux driver).
    const char* native = std::getenv("ISS_NATIVE_PASSES");
    std::printf("GPU: native passes %s (TLG_NATIVE_PASSES)\n", native && native[0] == '1' ? "on" : "off");
    std::printf("GPU: renderer v2 (gpu_v2)\n");
    return iss_gpu_initialize(&callbacks) == 0 ? 0 : -1;
}

extern "C" void tlggpu_set_guest_image(uintptr_t base, uint64_t) {
    iss_gpu_install_image(base);
}

extern "C" uintptr_t iss_gpu_resolve_nid(const char* nid, size_t length);
extern "C" uintptr_t tlggpu_resolve(const char* scoped_nid) {
    // "NID#library...": the renderer's symbols by NID, as this port's own GPU library does.
    const char* hash = std::strchr(scoped_nid, '#');
    return iss_gpu_resolve_nid(scoped_nid, hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid));
}

extern "C" int tlggpu_handle_fault(void* ucontext, void* address) {
    return iss_gpu_fault(ucontext, address) ? 1 : 0;
}

extern "C" int tlggpu_cpu_write(void*, const void*, int, uint64_t) {
    return 0; // the runtime writes normally (faults reach tlggpu_handle_fault)
}

extern "C" void tlggpu_frame_counters(TlgGpuFrameCounters* out) {
    *out = {};
}
namespace Vulkan { extern std::atomic<u64> g_iss_total_draws; }
extern "C" uint64_t tlggpu_total_draws(void) {
    return Vulkan::g_iss_total_draws.load(std::memory_order_relaxed);
}
extern "C" unsigned tlggpu_symbol_count(void) {
    return 1; // nonzero: the renderer provides GnmDriver/VideoOut
}
extern "C" void tlggpu_set_wait_hooks(TlgGpuWaitBegin, TlgGpuWaitEnd) {}
extern "C" void tlggpu_note_open(const char*) {}
extern "C" void tlggpu_dump_guest_writes(void*) {}

// This port's overlay (gpu/ours/tlg_overlay.cpp): text entry, save dialogs, notices, guides.
extern "C" int tlggpu_text_input_begin(const char* initial, const char* prompt) {
    if (!g_window) return 0;
    g_window->BeginTextInput(initial ? initial : "", prompt ? prompt : "Text");
    return 1;
}
extern "C" int tlggpu_text_input_poll(char* out, uint64_t size) {
    if (!g_window) return 2;
    std::string text;
    const int state = g_window->PollTextInput(text);
    if (size) {
        const size_t n = std::min<size_t>(text.size(), size - 1);
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return state;
}
static TlgOverlay::SaveRow SaveRowOf(const TlgSaveEntry& entry) {
    TlgOverlay::SaveRow row;
    row.title = entry.title ? entry.title : "";
    row.subtitle = entry.subtitle ? entry.subtitle : "";
    row.detail = entry.detail ? entry.detail : "";
    row.mtime = entry.mtime;
    row.bytes = entry.bytes;
    row.is_new = entry.is_new != 0;
    return row;
}
static TlgPng::Image SavePicture(const char* path) {
    TlgPng::Image picture;
    if (path) TlgPng::LoadRgba(path, picture);
    return picture;
}
extern "C" int tlggpu_save_list_begin(int kind, const TlgSaveEntry* entries, int count, int focus) {
    if (!g_window || count <= 0 || !entries) return 0;
    std::vector<TlgOverlay::SaveRow> rows;
    std::vector<TlgPng::Image> pictures;
    for (int i = 0; i < count; ++i) {
        rows.push_back(SaveRowOf(entries[i]));
        pictures.push_back(SavePicture(entries[i].icon));
    }
    TlgOverlay::BeginSaveList(kind, std::move(rows), std::move(pictures), focus);
    return 1;
}
extern "C" int tlggpu_save_list_poll(void) { return TlgOverlay::PollSaveList(); }
extern "C" int tlggpu_save_message_begin(int kind, int code, uint64_t value, const char* text,
                                         int buttons, const TlgSaveEntry* entry) {
    if (!g_window) return 0;
    TlgOverlay::SaveRow row;
    if (entry) row = SaveRowOf(*entry);
    TlgOverlay::BeginSaveMessage(kind, code, value, text ? text : "", buttons, entry ? &row : nullptr,
                                 SavePicture(entry ? entry->icon : nullptr));
    return 1;
}
extern "C" int tlggpu_save_message_poll(void) { return TlgOverlay::PollSaveMessage(); }
extern "C" void tlggpu_save_dialog_close(void) { TlgOverlay::CloseSaveDialog(); }
extern "C" void tlggpu_save_progress(int kind, int active, int percent, const char* text) {
    if (g_window) TlgOverlay::SaveProgress(kind, active, percent, text);
}
extern "C" float tlggpu_sensor_aim_scale(void) {
    return float(TlgSettings::Get().sensor_aim.load()) / 100.0f;
}
extern "C" void tlggpu_graffiti_guide(int force) {
    if (g_window) TlgOverlay::ShowGraffitiGuide(force != 0);
}
extern "C" void tlggpu_save_toast(int kind) {
    if (g_window) TlgOverlay::SaveToast(kind);
}
extern "C" int tlggpu_overlay_captures_input(void) { return TlgOverlay::CapturesInput() ? 1 : 0; }
extern "C" void tlggpu_overlay_notice(const char* text, float seconds) {
    TlgOverlay::Notice(text ? text : "", seconds);
}
extern "C" void tlggpu_input_settings(TlgGpuInputSettings* out) {
    const auto& s = TlgSettings::Get();
    out->mouse_enabled = s.mouse_enabled ? 1 : 0;
    out->invert_y = s.mouse_invert_y ? 1 : 0;
    out->sensitivity = s.mouse_sensitivity;
    out->smoothing = s.mouse_smoothing;
    out->accel = s.mouse_accel;
    out->deadzone = s.mouse_deadzone;
    std::snprintf(out->key_map, sizeof(out->key_map), "%s", s.key_map.c_str());
}
extern "C" void tlggpu_mouse_pad_hud(int active, float x, float y, int touching, int clicking) {
    TlgOverlay::SetMousePad(active != 0, x, y, touching != 0, clicking != 0);
}
// tlggpu_mouse_take: window.cpp (the game's mouse, as in this port's renderer).

// The Windows save dialog (save_dialog_ui.cpp, XInput) is left out: the window and the
// presenter find no dialog up.
namespace iss::save_ui {
bool Active() { return false; }
void Draw(float, float, float) {}
void LoadFonts(ImGuiIO&) {}
bool OnEvent(const SDL_Event&) { return false; }
} // namespace iss::save_ui

