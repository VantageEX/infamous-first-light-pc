// Adapted from bloodborne_pc@4d4d20812c0af23a11c8b8d40f5e06a9054925ef, GPL-2.0-or-later.
// bbport: the game window. Created by the VideoOut driver on first open; the
// event pump runs on the port's window thread (see window.cpp).
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include "common/types.h"

struct SDL_Window;
union SDL_Event;

namespace Frontend {

enum class WindowSystemType : u8 { Headless, Windows, X11, Wayland, Metal };

struct WindowSystemInfo {
    void* display_connection = nullptr;
    void* render_surface = nullptr;
    float render_surface_scale = 1.0f;
    WindowSystemType type = WindowSystemType::Headless;
};

class WindowSDL {
public:
    WindowSDL(s32 width, s32 height, const char* title);
    ~WindowSDL();
    s32 GetWidth() const { return width.load(std::memory_order_relaxed); }
    s32 GetHeight() const { return height.load(std::memory_order_relaxed); }
    SDL_Window* GetSDLWindow() const { return window; }
    WindowSystemInfo GetWindowInfo() const { return window_info; }
    bool IsOpen() const { return is_open.load(std::memory_order_relaxed); }
    /// Processes pending window events. Returns false once the user closed the window.
    bool PollEvents();
    /// The loading screen (window thread, before the renderer presents): the step (bridge.cpp
    /// g_preload_phase), its count and total (0 = unknown length), in this port's look.
    void DrawLoading(int phase, unsigned done, unsigned total);
    /// Keyboard text entry for the system IME dialog; typed text shows in the title bar.
    void BeginTextInput(const std::string& initial, const std::string& prompt);
    /// 0 while typing, 1 confirmed (Enter), 2 cancelled (Escape); text is UTF-8.
    int PollTextInput(std::string& text);

private:
    std::atomic<s32> width, height;
    std::atomic<bool> is_open{true};
    std::mutex text_mutex;
    bool text_requested{}, text_active{};
    int text_state{};
    std::string text, text_prompt, base_title;
    void UpdateTextTitle();
    int display_mode = -1; ///< applied TlgSettings display mode (window thread)
    void ApplyDisplayMode(int mode);
    // Mouse for the game (tlggpu_mouse_take): relative mode while focused, the menu closed and
    // not released with Esc. Window thread only.
    bool focused{}, mouse_captured{}, mouse_released{};
    u32 mouse_swallow{}; ///< buttons whose release belongs to the click that took the mouse
    float wheel_x{}, wheel_y{};
    u64 mouse_test_start{}, mouse_test_last{}; ///< TLG_MOUSE_TEST: synthetic motion and clicks
    bool HandleMouse(const SDL_Event& event);
    void UpdateMouseCapture();
    void MouseTest();
    SDL_Window* window{};
    WindowSystemInfo window_info{};
};

} // namespace Frontend
