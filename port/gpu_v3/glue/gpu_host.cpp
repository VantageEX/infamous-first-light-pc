// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3 behind this port's GPU library interface (gpu/tlggpu.h). The runtime
// (src/runtime_*.c) calls the tlggpu_* functions; they start the fork's video core as its
// emulator would (the GnmDriver/VideoOut libraries make the presenter and the rasterizer) over the
// runtime's memory (bridge.cpp), and hand it this port's window, overlay and save dialogs.
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <imgui.h>
#include <SDL3/SDL.h>
#include "tlggpu.h"
#include "common/elf_info.h"
#include "common/fault_monitor.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "common/types.h"
#include "core/debug_state.h"
#include "core/libraries/kernel/equeue.h"
#include "core/loader/symbols_resolver.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "sdl_window.h"
#include "tlg_overlay.h"
#include "tlg_png.h"
#include "tlg_settings.h"
#include "v3_glue.h"
#include "v3_stutter.h"

extern "C" {
// runtime_memory.c
void runtime_memory_visit(void (*visit)(uint64_t address, uint64_t size, uint8_t* backing));
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
// runtime_kernel.c
uint64_t runtime_process_time_us(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
// runtime.c
void runtime_report(void);
}

namespace Libraries::GnmDriver {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
}
namespace Libraries::VideoOut {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
}
namespace Libraries::Kernel {
extern boost::asio::io_context io_context;
}

extern Frontend::WindowSDL* g_window;

// ElfInfo's fields are its emulator's to set (common/elf_info.h: friend class Core::Emulator).
namespace Core {
class Emulator {
public:
    static void Configure() {
        auto& info = Common::ElfInfo::Instance();
        info.initialized = true;
        info.game_serial = V3::g_host.serial;
        info.title = V3::g_host.title;
        info.sdk_ver = V3::g_host.sdk_version;
        info.psf_attributes.raw = V3::g_host.psf_attributes;
        info.app_ver = "01.00";
    }
};
} // namespace Core

namespace {
Core::Loader::SymbolsResolver g_resolver;
std::mutex g_ready_mutex;
std::condition_variable g_ready_cv;
int g_ready = 0; // 0 starting, 1 running, -1 failed

Vulkan::Rasterizer* Rasterizer() {
    return Core::Memory::Instance()->GetRasterizer();
}

void MapForGpu(uint64_t address, uint64_t size, uint8_t*) {
    if (auto* r = Rasterizer()) r->MapMemory(address, size);
}

// The fork keeps its files (pipeline cache, shader dumps, logs) under one user folder: this
// port's, beside the saves (TLG_GPU_USER_DIR), so renderer v3's cache is its own.
void SetPaths(const std::filesystem::path& user) {
    namespace FS = Common::FS;
    // SetUserPath takes only folders that exist.
    const auto set = [](FS::PathType type, const std::filesystem::path& path) {
        std::error_code error;
        std::filesystem::create_directories(path, error);
        FS::SetUserPath(type, path);
    };
    const auto v3 = user / "v3";
    for (int i = 0; i <= static_cast<int>(FS::PathType::CustomModulesDir); ++i) {
        set(static_cast<FS::PathType>(i), v3 / std::to_string(i));
    }
    set(FS::PathType::UserDir, v3);
    set(FS::PathType::CacheDir, user / "cache" / "v3");
}

// The window's thread: the window answers from its first instant while the libraries (the
// renderer, its pipeline cache) start on a thread of their own, as renderer v2 does it.
void WindowThread() {
    try {
        g_window = new Frontend::WindowSDL(V3::g_host.width, V3::g_host.height, V3::g_host.title.c_str());
        std::atomic<int> init_state{0};
        std::string init_error;
        std::thread init([&] {
            try {
                Common::SetCurrentThreadName("v3:Init");
                Libraries::Kernel::RegisterEventQueue(&g_resolver);
                static auto work = boost::asio::make_work_guard(Libraries::Kernel::io_context);
                std::thread([] {
                    Common::SetCurrentThreadName("v3:KernelService");
                    Libraries::Kernel::io_context.run();
                }).detach();
                Libraries::GnmDriver::RegisterLib(&g_resolver);
                Libraries::VideoOut::RegisterLib(&g_resolver);
                // The mappings made before the rasterizer existed.
                runtime_memory_visit(MapForGpu);
                // The fork's warm-up screen is over (its steps fed the loading screen).
                TlgOverlay::SetLoadingCompile(false);
                TlgOverlay::SetLoading(false, 0, 0);
                // This port's first-start shader compilation (the job file), before the game.
                V3::PrecompileJobs();
                init_state.store(1, std::memory_order_release);
            } catch (const std::exception& e) {
                init_error = e.what();
                init_state.store(-1, std::memory_order_release);
            }
        });
        while (init_state.load(std::memory_order_acquire) == 0) {
            if (!g_window->PollEvents()) {
                std::fputs("V3: window closed while loading\n", stderr);
                runtime_report();
                std::fflush(nullptr);
                std::_Exit(0);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        init.join();
        if (init_state.load() < 0) throw std::runtime_error(init_error);
        {
            std::scoped_lock lock{g_ready_mutex};
            g_ready = 1;
        }
        g_ready_cv.notify_all();
        for (;;) {
            if (!g_window->PollEvents()) {
                std::fputs("V3: window closed\n", stderr);
                runtime_report();
                std::fflush(nullptr);
                std::_Exit(0);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "GPU initialization failed (renderer v3): %s\n", e.what());
        {
            std::scoped_lock lock{g_ready_mutex};
            g_ready = -1;
        }
        g_ready_cv.notify_all();
    }
}
} // namespace

extern "C" void tlggpu_register_kernel(void) {
    // The renderer registers the kernel event queues itself (WindowThread's init).
}

extern "C" void tlggpu_init_headless(const TlgGpuConfig*) {}

extern "C" int tlggpu_init(const TlgGpuConfig* config) {
    auto& host = V3::g_host;
    host.title = config->title ? config->title : "inFAMOUS First Light";
    host.serial = config->serial ? config->serial : "CUSA00897";
    host.user_dir = config->user_dir ? config->user_dir : ".";
    host.sdk_version = config->sdk_version;
    host.psf_attributes = config->psf_attributes;
    host.width = config->width > 0 ? config->width : 1920;
    host.height = config->height > 0 ? config->height : 1080;
    host.tsc_frequency = runtime_tsc_frequency;
    host.process_time_us = runtime_process_time_us;
    host.errno_location = runtime_errno;
    MemoryPatcher::g_game_serial = host.serial;
    Core::Emulator::Configure();
    SetPaths(std::filesystem::u8path(host.user_dir));
    // The fork's frame and fault statistics, every 15 s to the log and a CSV (its LogDir).
    Common::FaultMonitor::Start();
    V3::StartThreadCpuReport(); // TLG_V3_THREAD_CPU=1
    V3::StartProfiler();        // TLG_V3_PROFILE=1
    // Mappings made from now on reach the rasterizer once it exists (WindowThread replays the
    // earlier ones).
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) { MapForGpu(address, size, nullptr); },
        [](uintptr_t address, uint64_t size) {
            if (auto* r = Rasterizer()) r->UnmapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) {
            if (auto* r = Rasterizer()) r->InvalidateMemory(address, size);
        });
    TlgSettings::Load(); // tlg.ini: the overlay, the menu and the input settings
    std::printf("GPU: renderer v3 (gpu_v3: axylpup second-son-v1.1.1)\n");
    std::fflush(stdout);
    std::thread(WindowThread).detach();
    std::unique_lock lock{g_ready_mutex};
    g_ready_cv.wait(lock, [] { return g_ready != 0; });
    return g_ready == 1 ? 0 : -1;
}

extern "C" void tlggpu_set_guest_image(uintptr_t base, uint64_t) {
    V3::g_host.image_base = base;
    MemoryPatcher::g_eboot_address = base;
}

extern "C" uintptr_t tlggpu_resolve(const char* scoped_nid) {
    // "NID#library...": the libraries' symbols by NID, as this port's own GPU library does.
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_resolver.GetSymbols()) {
        if (symbol.nid_name.size() == length && std::memcmp(symbol.nid_name.data(), scoped_nid, length) == 0) {
            return symbol.virtual_address;
        }
    }
    return 0;
}

// TLG_V3_FAULT_HISTOGRAM=1: the guest's write faults counted per 1 MiB of address space, the
// busiest printed every 15 s (where the game's per-frame heap lies in this runtime's layout).
namespace {
std::mutex g_histogram_mutex;
std::unordered_map<uint64_t, uint64_t> g_histogram;
bool HistogramEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("TLG_V3_FAULT_HISTOGRAM");
        if (!(v && v[0] == '1')) return false;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(15));
                std::vector<std::pair<uint64_t, uint64_t>> top;
                {
                    std::scoped_lock lock{g_histogram_mutex};
                    top.assign(g_histogram.begin(), g_histogram.end());
                    g_histogram.clear();
                }
                std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
                std::string line;
                for (size_t i = 0; i < top.size() && i < 16; ++i) {
                    line += fmt::format(" {:#x}:{}", top[i].first << 20, top[i].second);
                }
                std::fprintf(stderr, "V3 write faults per MiB (15 s):%s\n", line.c_str());
            }
        }).detach();
        return true;
    }();
    return on;
}
} // namespace

extern "C" int tlggpu_handle_fault(void* ucontext, void* address) {
    V3::Stutter::g_faults.fetch_add(1, std::memory_order_relaxed);
    const bool write = Common::IsWriteError(ucontext);
    V3::Stutter::RecordFault(reinterpret_cast<uintptr_t>(address), write, Common::FaultMonitor::t_is_cp_thread);
    if (write && Common::FaultMonitor::t_is_cp_thread) V3::Stutter::RecordCpWriteStack();
    if (HistogramEnabled() && write) {
        std::scoped_lock lock{g_histogram_mutex};
        ++g_histogram[reinterpret_cast<uintptr_t>(address) >> 20];
    }
    return Core::Signals::Instance()->DispatchAccessViolation(ucontext, address) ? 1 : 0;
}

// The runtime's large host writes into guest memory (file reads, the game's memcpy/memset): the
// range is invalidated in one call (CPU-modified, its pages unprotected), so the caller's copy that
// follows takes no page fault per 4 KB (texture streaming: whole MiB, 256 faults each, a frame's
// worth in a 50 ms stutter). Smaller writes take their faults as before. The runtime still writes.
extern "C" int tlggpu_cpu_write(void* dst, const void*, int, uint64_t size) {
    constexpr uint64_t MinBytes = 16 * 1024;
    if (size >= MinBytes) {
        if (auto* r = Rasterizer()) r->InvalidateMemory(reinterpret_cast<uintptr_t>(dst), size);
    }
    return 0;
}

extern "C" void tlggpu_frame_counters(TlgGpuFrameCounters* out) {
    // Only the frames presented: the freeze report (src/runtime_freeze.c) needs nothing else.
    *out = {};
    out->shown = out->presented = DebugState.GetFrameNum();
}
extern "C" uint64_t tlggpu_total_draws(void) {
    return 0;
}
extern "C" unsigned tlggpu_symbol_count(void) {
    return static_cast<unsigned>(g_resolver.GetSize());
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
