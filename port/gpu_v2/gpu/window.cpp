// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v2's window: this port's window (gpu/shim/window.cpp: the game's mouse, the launcher's
// display mode, text entry, a size that fits the screen, the app id) plus what v2 adds: the loading
// screen while the shaders preload, Alt+Enter, the F10 stutter mark and the hidden bench window.
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
#include <objidl.h>
namespace Gdiplus { using std::min; using std::max; }
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#endif
#include <SDL3/SDL.h>
#include "common/assert.h"
#include "common/logging/log.h"
#include "runtime_timeline.h"
#include "sdl_window.h"
#include "tlg_graphics.h"
#include "tlg_overlay.h"
#include "tlg_settings.h"
#include "tlggpu.h"
#include "visual_capture.h"

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
            // F10: a stutter mark in the timeline (and a visual capture when it is on).
            if (event.key.key == SDLK_F10) {
                iss::timeline::Emit("user_stutter_mark", 0, 0, SDL_GetTicks());
                std::fprintf(stderr, "PERF_MARK tick_ms=%llu\n", (unsigned long long)SDL_GetTicks());
                std::fflush(stderr);
                if (iss::visual_capture::Enabled()) iss::visual_capture::Start(SDL_GetTicksNS());
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


// The loading screen while the renderer starts and the shaders preload (RPCS3 style): the game's
// own art (sce_sys/pic1.png, the crow emblem; pic0.png if missing, from ISS_GAME_DIR, as shadPS4's launcher
// shows it), a shade at the bottom, the status and a progress bar. Drawn with GDI+ straight on the
// window until the game's first frame replaces it (ISS_LOADING_SCREEN=0 off).
#ifndef _WIN32
// Linux: the window title shows the loading status (bridge.cpp); no art is drawn.
void WindowSDL::DrawLoading(int,unsigned,unsigned){}
#else
namespace {
std::wstring Widen(const std::string& text){
    if(text.empty())return {};
    const int n=MultiByteToWideChar(CP_UTF8,0,text.data(),int(text.size()),nullptr,0);
    std::wstring out(size_t(n),L'\0');
    MultiByteToWideChar(CP_UTF8,0,text.data(),int(text.size()),out.data(),n);
    return out;
}
struct LoadingArt{
    bool tried{},enabled{true};
    ULONG_PTR token{};
    std::unique_ptr<Gdiplus::Bitmap> image;
    std::unique_ptr<Gdiplus::Bitmap> scaled; ///< the art fitted to the current client size
    int scaled_w{},scaled_h{};
};
LoadingArt& Art(){static LoadingArt art;return art;}
}

namespace {
// The loading screen's words, pt-BR or en-US as the launcher's language (TLG_UI_LANG).
bool Portuguese(){
    static const bool pt=[]{const char* l=std::getenv("TLG_UI_LANG");return !(l&&!std::strncmp(l,"en",2));}();
    return pt;
}
const wchar_t* L2(const wchar_t* en,const wchar_t* pt){return Portuguese()?pt:en;}
}

// Before the renderer can present (v2's first-run steps: indexing and compiling the game's compute
// shaders, then reading the cache), drawn with GDI+ straight on the window in the look of this
// port's "Compiling shaders" screen (tlg_overlay.cpp LoadingScreen): the game's art dimmed behind a
// centred block with the step, its bar and count, the time left and a note. Phase 6 (graphics
// precompile) is the overlay's own screen, so this one is not drawn then (ISS_LOADING_SCREEN=0 off).
void WindowSDL::DrawLoading(int phase,unsigned done,unsigned total){
    auto& art=Art();
    if(!art.tried){
        art.tried=true;
        if(const char* p=std::getenv("ISS_LOADING_SCREEN");p&&p[0]=='0')art.enabled=false;
        Gdiplus::GdiplusStartupInput input;
        if(art.enabled&&Gdiplus::GdiplusStartup(&art.token,&input,nullptr)!=Gdiplus::Ok)art.enabled=false;
        if(art.enabled){
            if(const char* dir=std::getenv("ISS_GAME_DIR")){
                for(const char* name:{"pic1.png","pic0.png"}){
                    const std::wstring path=Widen(dir)+L"\\sce_sys\\"+Widen(name);
                    auto bitmap=std::make_unique<Gdiplus::Bitmap>(path.c_str());
                    if(bitmap->GetLastStatus()==Gdiplus::Ok&&bitmap->GetWidth()>0){art.image=std::move(bitmap);break;}
                }
            }
            std::fprintf(stderr,"LOADING_SCREEN art=%d\n",art.image?1:0);
        }
    }
    if(!art.enabled)return;
    HWND hwnd=static_cast<HWND>(window_info.render_surface);
    if(!hwnd)return;
    RECT client{};GetClientRect(hwnd,&client);
    const int w=client.right-client.left,h=client.bottom-client.top;
    if(w<=0||h<=0)return;
    using namespace Gdiplus;
    // The art scaled to cover the window and dimmed, kept until the size changes.
    if(!art.scaled||art.scaled_w!=w||art.scaled_h!=h){
        art.scaled=std::make_unique<Bitmap>(w,h,PixelFormat32bppPARGB);
        Graphics g(art.scaled.get());
        g.Clear(Color(255,11,14,20));
        if(art.image){
            g.SetInterpolationMode(InterpolationModeHighQualityBicubic);
            const double iw=art.image->GetWidth(),ih=art.image->GetHeight();
            const double scale=std::max(w/iw,h/ih);
            const REAL dw=REAL(iw*scale),dh=REAL(ih*scale);
            g.DrawImage(art.image.get(),RectF((w-dw)/2,(h-dh)/2,dw,dh));
            SolidBrush dim(Color(200,11,14,20));
            g.FillRectangle(&dim,0,0,w,h);
        }
        art.scaled_w=w;art.scaled_h=h;
    }
    // The time left: per step, from when its count started to move.
    static int eta_phase=-1;
    static ULONGLONG eta_start=0;
    static unsigned eta_from=0;
    if(phase!=eta_phase){eta_phase=phase;eta_start=GetTickCount64();eta_from=done;}
    const wchar_t* title=
        phase==5?L2(L"Indexing the game's shaders...",L"Indexando os shaders do jogo..."):
        phase==4?L2(L"Compiling compute shaders...",L"Compilando shaders de compute..."):
        phase==1?L2(L"Loading shader cache...",L"Carregando o cache de shaders..."):
        phase==0?L2(L"Starting...",L"Iniciando..."):
                 L2(L"Preparing graphics...",L"Preparando os gráficos...");
    const bool first_run=phase==4||phase==5;
    const wchar_t* note=first_run
        ?L2(L"Only on the first start on this computer and after updates: then the game runs without compiling shaders.",
            L"Só na primeira abertura neste computador e depois de atualizações: depois o jogo roda sem compilar shaders.")
        :L2(L"The first start after an update takes longer.",L"A primeira abertura depois de uma atualização demora mais.");
    Bitmap frame(w,h,PixelFormat32bppPARGB);
    {
        Graphics g(&frame);
        g.DrawImage(art.scaled.get(),0,0,w,h);
        g.SetSmoothingMode(SmoothingModeAntiAlias);
        g.SetTextRenderingHint(TextRenderingHintAntiAliasGridFit);
        // The overlay's block: 40% of the width, centred; ImGui's dark style colours.
        const REAL bw=REAL(w)*0.40f,x=(REAL(w)-bw)/2;
        const REAL base=std::max<REAL>(13.f,REAL(h)*0.018f);
        FontFamily family(L"Segoe UI");
        Font big(&family,base*1.6f,FontStyleRegular,UnitPixel),font(&family,base,FontStyleRegular,UnitPixel);
        SolidBrush text(Color(255,255,255,255)),dim_text(Color(255,128,128,128));
        const REAL bar_h=base*1.55f;
        REAL y=REAL(h)*0.5f-base*4.0f;
        g.DrawString(title,-1,&big,PointF(x,y),&text);
        y+=base*2.4f;
        if(total){
            SolidBrush frame_bg(Color(138,41,74,122)),fill(Color(255,230,179,0));
            g.FillRectangle(&frame_bg,x,y,bw,bar_h);
            const REAL part=bw*REAL(std::min(done,total))/REAL(total);
            if(part>0)g.FillRectangle(&fill,x,y,part,bar_h);
            wchar_t count[64];
            swprintf(count,64,L"%u / %u (%u%%)",done,total,unsigned(static_cast<unsigned long long>(std::min(done,total))*100/total));
            StringFormat centre;centre.SetAlignment(StringAlignmentCenter);centre.SetLineAlignment(StringAlignmentCenter);
            g.DrawString(count,-1,&font,RectF(x,y,bw,bar_h),&centre,&text);
            y+=bar_h+base*0.6f;
            const ULONGLONG ms=GetTickCount64()-eta_start;
            if(first_run&&done>eta_from&&done<total&&ms>3000){
                const unsigned left=unsigned(double(ms)/1000.0*double(total-done)/double(done-eta_from))+1;
                wchar_t eta[64];
                swprintf(eta,64,L2(L"About %u:%02u left",L"Faltam cerca de %u:%02u"),left/60,left%60);
                g.DrawString(eta,-1,&font,PointF(x,y),&text);
                y+=base*1.5f;
            }
        }else{
            // Unknown length: a segment sweeping across the bar.
            SolidBrush frame_bg(Color(138,41,74,122)),fill(Color(255,230,179,0));
            g.FillRectangle(&frame_bg,x,y,bw,bar_h*0.35f);
            const double t=double(GetTickCount64()%1600)/1600.0;
            const REAL seg=bw*0.25f,sx=x+REAL(t)*(bw+seg)-seg;
            const REAL left=std::max(x,sx),right=std::min(x+bw,sx+seg);
            if(right>left)g.FillRectangle(&fill,left,y,right-left,bar_h*0.35f);
            y+=bar_h*0.35f+base*0.8f;
        }
        g.DrawString(note,-1,&font,RectF(x,y,bw,base*4.0f),nullptr,&dim_text);
    }
    HDC dc=GetDC(hwnd);
    if(dc){
        Graphics out(dc);
        out.SetCompositingMode(CompositingModeSourceCopy);
        out.DrawImage(&frame,0,0,w,h);
        ReleaseDC(hwnd,dc);
    }
}

#endif

} // namespace Frontend
