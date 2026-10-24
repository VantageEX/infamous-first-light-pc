// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3's window: renderer v2's (gpu_v2/gpu/window.cpp: the game's mouse, the launcher's
// display mode, text entry, a size that fits the screen, the app id, Alt+Enter, the F10 stutter
// mark) without v2's loading art and diagnostics.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef _WIN32
#include <windows.h>
#endif
#include <SDL3/SDL.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "sdl_window.h"
#include "tlg_graphics.h"
#include "tlg_overlay.h"
#include "tlg_settings.h"
#include "tlggpu.h"

namespace {
std::mutex mouse_mutex;
TlgGpuMouse mouse_state{}; // window thread adds, the pad (runtime_pad.c) takes

u32 MouseBit(u8 button) {
    return button >= 1 && button <= 32 ? 1u << (button - 1) : 0;
}
} // namespace

extern "C" void tlggpu_mouse_take(TlgGpuMouse* out) {
    std::scoped_lock lock{mouse_mutex};
    *out = mouse_state;
    mouse_state.dx = mouse_state.dy = 0.0f;
    mouse_state.pressed = 0;
    mouse_state.wheel_up = mouse_state.wheel_down = mouse_state.wheel_left = mouse_state.wheel_right = 0;
}

namespace Frontend {

WindowSDL::WindowSDL(s32 width_, s32 height_, const char* title) : width{width_}, height{height_} {
    // tlg: the desktop shows this name and matches the app id with the launcher's .desktop entry
    // (Alt+Tab, dock). The launcher sets TLG_APP_NAME / TLG_APP_ID per game.
    {
        const char* app_name = std::getenv("TLG_APP_NAME");
        const char* app_id = std::getenv("TLG_APP_ID");
        SDL_SetAppMetadata(app_name && *app_name ? app_name : "inFAMOUS Second Son", nullptr,
                           app_id && *app_id ? app_id : "infamous-pc");
        SDL_SetHint(SDL_HINT_APP_ID, app_id && *app_id ? app_id : "infamous-pc");
    }
    // Gamepads are sampled by runtime_pad.c; their events are pumped here with the window's.
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        UNREACHABLE_MSG("Failed to initialize SDL video: {}", SDL_GetError());
    }
    // tlg: the window opens at the game's size (1920x1080) unless the screen is smaller: on a
    // 1366x768 laptop it was larger than the screen and the picture was cut. It then opens at the
    // largest same-shape size that fits the usable area (taskbar and title bar kept clear); it can
    // still be maximized, resized or made fullscreen.
    SDL_Rect usable{};
    if (width_ > 0 && height_ > 0 && SDL_GetDisplayUsableBounds(SDL_GetPrimaryDisplay(), &usable) &&
        usable.w > 0 && usable.h > 0) {
        const double fit = std::min({1.0, usable.w * 0.95 / width_, (usable.h - 40) * 0.95 / height_});
        if (fit < 1.0) {
            const s32 w = std::max<s32>(640, s32(width_ * fit)), h = std::max<s32>(360, s32(height_ * fit));
            LOG_INFO(Frontend, "Window {}x{} does not fit the screen ({}x{} usable): opening at {}x{}", width_,
                     height_, usable.w, usable.h, w, h);
            width_ = w;
            height_ = h;
        }
    }
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, title);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width_);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height_);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_VULKAN_BOOLEAN, true);
    // ISS_HYBRID_BENCH_HIDDEN=1: v2's unattended benchmarks run without a visible window.
    if (const char* hidden = std::getenv("ISS_HYBRID_BENCH_HIDDEN"); hidden && hidden[0] == '1') {
        SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_HIDDEN_BOOLEAN, true);
    }
    base_title = title;
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    ASSERT_MSG(window, "Failed to create window: {}", SDL_GetError());
    // tlg: the game's own icon (run.py converts sce_sys/icon0.png to BMP, SDL's built-in format).
    if (const char* icon_path = std::getenv("TLG_WINDOW_ICON")) {
        if (SDL_Surface* icon = SDL_LoadBMP(icon_path)) {
            SDL_SetWindowIcon(window, icon);
            SDL_DestroySurface(icon);
        }
    }

    const char* driver = SDL_GetCurrentVideoDriver();
    const SDL_PropertiesID wp = SDL_GetWindowProperties(window);
#ifdef _WIN32
    if (driver && !std::strcmp(driver, "windows")) {
        window_info.type = WindowSystemType::Windows;
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    } else
#endif
    if (driver && !std::strcmp(driver, "x11")) {
        window_info.type = WindowSystemType::X11;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr);
        window_info.render_surface = reinterpret_cast<void*>(SDL_GetNumberProperty(wp, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
    } else if (driver && !std::strcmp(driver, "wayland")) {
        window_info.type = WindowSystemType::Wayland;
        window_info.display_connection = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr);
        window_info.render_surface = SDL_GetPointerProperty(wp, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr);
    } else {
        UNREACHABLE_MSG("Unsupported SDL video driver {}", driver ? driver : "(none)");
    }
    window_info.render_surface_scale = SDL_GetWindowDisplayScale(window);
    // tlg: TLG_FULLSCREEN=1 (older launchers) still asks for fullscreen.
    if (const char* fullscreen = std::getenv("TLG_FULLSCREEN"); fullscreen && fullscreen[0] == '1' &&
        TlgSettings::Get().display_mode == TlgGraphics::Windowed) {
        TlgSettings::Get().display_mode = TlgGraphics::Borderless;
    }
    ApplyDisplayMode(TlgSettings::Get().display_mode);
    // tlg: no system cursor over the game; the in-game menu draws its own (TlgOverlay::SetOpen).
    // (ps4_core: UpdateMouseCapture shows it again while Esc keeps the game's mouse released.)
    SDL_HideCursor();
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    width = w;
    height = h;
    LOG_INFO(Frontend, "Window {}x{} on {}", w, h, driver);
    focused = (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) != 0;
    if (const char* test = std::getenv("TLG_MOUSE_TEST"); test && test[0] == '1') {
        mouse_test_start = SDL_GetTicksNS();
        std::printf("Mouse: TLG_MOUSE_TEST synthesizes motion, clicks and wheel notches\n");
    }
}

// The game's mouse. Taken (relative mode) while the window has focus, the port's menu, text
// box and save list are closed, and the player did not release it with Esc; a click in the
// window takes it again. Returns true for events the game must not see.
bool WindowSDL::HandleMouse(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        focused = true;
        return false;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        focused = mouse_test_start != 0; // the test runs unattended
        return false;
    case SDL_EVENT_KEY_DOWN:
        // Esc is free in the game's layout; the menu (open) and the text/save boxes take it first.
        if (event.key.key == SDLK_ESCAPE && !event.key.repeat && mouse_captured) {
            mouse_released = true;
            return true;
        }
        return false;
    case SDL_EVENT_MOUSE_MOTION:
        if (mouse_captured) {
            std::scoped_lock lock{mouse_mutex};
            mouse_state.dx += event.motion.xrel;
            mouse_state.dy += event.motion.yrel;
        }
        return mouse_captured;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        const u32 bit = MouseBit(event.button.button);
        const bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
        if (!down && (mouse_swallow & bit)) {
            mouse_swallow &= ~bit;
            return true;
        }
        if (!mouse_captured) {
            // A click takes the mouse back after Esc; that click is not a game button.
            if (down && mouse_released && TlgSettings::Get().mouse_enabled && focused) {
                mouse_released = false;
                mouse_swallow |= bit;
                return true;
            }
            return false;
        }
        std::scoped_lock lock{mouse_mutex};
        if (down) {
            mouse_state.held |= bit;
            mouse_state.pressed |= bit;
        } else {
            mouse_state.held &= ~bit;
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        if (!mouse_captured) {
            return false;
        }
        const float flip = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1.0f : 1.0f;
        wheel_x += event.wheel.x * flip;
        wheel_y += event.wheel.y * flip;
        std::scoped_lock lock{mouse_mutex};
        for (; wheel_y >= 1.0f; wheel_y -= 1.0f) ++mouse_state.wheel_up;
        for (; wheel_y <= -1.0f; wheel_y += 1.0f) ++mouse_state.wheel_down;
        for (; wheel_x >= 1.0f; wheel_x -= 1.0f) ++mouse_state.wheel_right;
        for (; wheel_x <= -1.0f; wheel_x += 1.0f) ++mouse_state.wheel_left;
        return true;
    }
    default:
        return false;
    }
}

void WindowSDL::UpdateMouseCapture() {
    const bool want = TlgSettings::Get().mouse_enabled && focused && !mouse_released &&
                      !TlgOverlay::CapturesInput();
    if (want == mouse_captured) {
        return;
    }
    if (!SDL_SetWindowRelativeMouseMode(window, want) && want) {
        LOG_WARNING(Frontend, "Mouse capture: {}", SDL_GetError());
    }
    mouse_captured = want;
    // tlg: the system cursor stays hidden over the game, but Esc gives the pointer back with a
    // visible cursor so the player can reach other windows; a click captures it again.
    if (want) {
        SDL_HideCursor();
    } else if (mouse_released) {
        SDL_ShowCursor();
    }
    wheel_x = wheel_y = 0.0f;
    {
        std::scoped_lock lock{mouse_mutex};
        mouse_state = TlgGpuMouse{};
        mouse_state.captured = want;
    }
    std::printf("Mouse: %s\n", want ? "captured for the camera and touchpad (Esc releases)"
                                     : mouse_released ? "released (click in the window to capture)"
                                                      : "released");
}

// TLG_MOUSE_TEST=1: an 8 s loop of synthetic input through the real event path: 2 s right at
// 1500 counts/s, 1 s still, 2 s left, 1 s up at 1200 counts/s, then a right click (R1, call
// Trico) at 6.5 s and a wheel notch up at 7.5 s.
void WindowSDL::MouseTest() {
    const u64 now = SDL_GetTicksNS();
    const u64 last = mouse_test_last ? mouse_test_last : now;
    mouse_test_last = now;
    const double t = double((now - mouse_test_start) % 8000000000ull) / 1e9;
    const double prev = t - double(now - last) / 1e9;
    const float dt = float(now - last) / 1e9f;
    SDL_Event e{};
    const float vx = t < 2.0 ? 1500.0f : t >= 3.0 && t < 5.0 ? -1500.0f : 0.0f;
    const float vy = t >= 5.0 && t < 6.0 ? -1200.0f : 0.0f;
    if (vx != 0.0f || vy != 0.0f) {
        e.type = SDL_EVENT_MOUSE_MOTION;
        e.motion.windowID = SDL_GetWindowID(window);
        e.motion.xrel = vx * dt;
        e.motion.yrel = vy * dt;
        SDL_PushEvent(&e);
    }
    const auto crossed = [&](double at) { return prev < at && t >= at; };
    for (const auto& [at, down] : {std::pair{6.5, true}, std::pair{6.7, false}}) {
        if (crossed(at)) {
            e = SDL_Event{};
            e.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
            e.button.windowID = SDL_GetWindowID(window);
            e.button.button = SDL_BUTTON_RIGHT;
            e.button.down = down;
            SDL_PushEvent(&e);
        }
    }
    if (crossed(7.5)) {
        e = SDL_Event{};
        e.type = SDL_EVENT_MOUSE_WHEEL;
        e.wheel.windowID = SDL_GetWindowID(window);
        e.wheel.y = 1.0f;
        e.wheel.direction = SDL_MOUSEWHEEL_NORMAL;
        SDL_PushEvent(&e);
    }
}

// Windowed, borderless (desktop-sized, no mode change) or exclusive fullscreen at the desktop
// mode. Window thread only (SDL). The swapchain follows the size events.
void WindowSDL::ApplyDisplayMode(int mode) {
    display_mode = mode;
    if (mode == TlgGraphics::Windowed) {
        SDL_SetWindowFullscreen(window, false);
    } else {
        const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window));
        const bool exclusive = mode == TlgGraphics::Fullscreen && desktop;
        if (!SDL_SetWindowFullscreenMode(window, exclusive ? desktop : nullptr)) {
            LOG_WARNING(Frontend, "Fullscreen mode: {}", SDL_GetError());
        }
        SDL_SetWindowFullscreen(window, true);
    }
    SDL_SyncWindow(window);
    std::printf("Window: %s\n", TlgGraphics::DisplayModeNames[mode]);
}

WindowSDL::~WindowSDL() {
    SDL_DestroyWindow(window);
}

void WindowSDL::BeginTextInput(const std::string& initial, const std::string& prompt) {
    std::scoped_lock lock{text_mutex};
    text = initial;
    text_prompt = prompt;
    text_state = 0;
    text_requested = true;
}

int WindowSDL::PollTextInput(std::string& out) {
    std::scoped_lock lock{text_mutex};
    out = text;
    return text_state;
}

void WindowSDL::UpdateTextTitle() {
    const std::string title = text_active ? base_title + " \u2014 " + text_prompt + ": " + text + "_  (Enter = OK, Esc = cancel)"
                                          : base_title;
    SDL_SetWindowTitle(window, title.c_str());
    TlgOverlay::SetTextEntry(text_active, text_prompt, text);
}

bool WindowSDL::PollEvents() {
    {
        std::scoped_lock lock{text_mutex};
        if (text_requested) { // SDL text input must be toggled from the window thread
            text_requested = false;
            text_active = true;
            SDL_StartTextInput(window);
            UpdateTextTitle();
        }
    }
    if (!text_active) {
        TlgOverlay::UpdateTextInput(window);
    }
    if (const int mode = TlgSettings::Get().display_mode; mode != display_mode) {
        ApplyDisplayMode(mode);
    }
    if (mouse_test_start) {
        focused = true;
        MouseTest();
    }
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (text_active && (event.type == SDL_EVENT_TEXT_INPUT || event.type == SDL_EVENT_KEY_DOWN)) {
            std::scoped_lock lock{text_mutex};
            if (event.type == SDL_EVENT_TEXT_INPUT) {
                text += event.text.text;
            } else if (event.key.key == SDLK_BACKSPACE && !text.empty()) {
                size_t cut = text.size() - 1; // drop one UTF-8 code point
                while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
                text.erase(cut);
            } else if (event.key.key == SDLK_RETURN || event.key.key == SDLK_KP_ENTER || event.key.key == SDLK_ESCAPE) {
                text_state = event.key.key == SDLK_ESCAPE ? 2 : 1;
                text_active = false;
                SDL_StopTextInput(window);
            }
            UpdateTextTitle();
            continue;
        }
        // The controller finishes the text dialog too: Cross (A) accepts, Circle (B) cancels.
        if (text_active && event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN &&
            (event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH ||
             event.gbutton.button == SDL_GAMEPAD_BUTTON_EAST)) {
            std::scoped_lock lock{text_mutex};
            text_state = event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH ? 1 : 2;
            text_active = false;
            SDL_StopTextInput(window);
            UpdateTextTitle();
            continue;
        }
        if (TlgOverlay::HandleEvent(event) || HandleMouse(event)) {
            if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                is_open = false;
            }
            continue;
        }
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
            // Alt+Enter: windowed <-> borderless, kept in the settings like the menu's choice.
            if (event.key.key == SDLK_RETURN && (event.key.mod & SDL_KMOD_ALT)) {
                auto& mode = TlgSettings::Get().display_mode;
                mode = mode == TlgGraphics::Windowed ? TlgGraphics::Borderless : TlgGraphics::Windowed;
            }
            // F10: a stutter mark in the game log.
            if (event.key.key == SDLK_F10) {
                std::fprintf(stderr, "PERF_MARK tick_ms=%llu\n", (unsigned long long)SDL_GetTicks());
                std::fflush(stderr);
            }
        }
        switch (event.type) {
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED: {
            int w = 0, h = 0;
            SDL_GetWindowSizeInPixels(window, &w, &h);
            width = w;
            height = h;
            break;
        }
        case SDL_EVENT_QUIT:
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
            is_open = false;
            break;
        default:
            break;
        }
    }
    UpdateMouseCapture();
    return is_open;
}


// Renderer v3: the boot screen is the presenter's (glue/overrides/vk_presenter.cpp, this port's
// overlay over black); no art is drawn on the window here.
void WindowSDL::DrawLoading(int, unsigned, unsigned) {}

} // namespace Frontend
