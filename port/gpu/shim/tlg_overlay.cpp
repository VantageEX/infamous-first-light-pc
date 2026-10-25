// TLG_RENDERER_HOSTED: built into renderer v2 or v3 (gpu_v2, gpu_v3), whose own presenters, fixes
// and switches stand in for this port's renderer's.
#ifdef TLG_RENDERER_HOSTED
// Those builds have no perf trace (gpu/shim/tlg_perf_trace.cpp).
namespace TlgPerf {
inline bool Enabled() { return false; }
inline bool Mark(unsigned) { return false; }
} // namespace TlgPerf
#else
#include "tlg_perf_trace.h"
#endif
// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_overlay.h"
#include "../tlggpu.h" // tlg_mouse_camera_ready

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <array>
#include <map>
#include <mutex>
#include <vector>

#include <SDL3/SDL.h>
#include "tlg_settings.h"
#include "tlg_orbs.h"
#include "tlg_actors.h"
#include "tlg_graphics.h"
#include "tlg_perf_stats.h"
#include "tlg_trophies.h"
#include "tlg_sound.h"
#include "../../src/actor_hook.h"

extern TlgActorSlot tlg_actor_slots[2];
extern "C" void runtime_restart(void); // src/probe.c: starts run.py again, ends this process
#include "common/elf_info.h"
#ifndef TLG_RENDERER_HOSTED
#include "common/backports.h"
#include "shader_recompiler/recompiler_toggles.h"
#endif
#include "imgui.h"
#include "imgui_internal.h" // RegisterUserTexture (save pictures)
#include "imgui_impl_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#ifndef TLG_RENDERER_HOSTED
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#endif

// DejaVu Sans (Cyrillic), embedded (third_party/fonts, Bitstream Vera license).
#ifdef _WIN32
asm(".section .rdata,\"dr\"\n"
    ".balign 16\n"
    ".global tlg_font_ttf\n"
    "tlg_font_ttf:\n"
    ".incbin \"" TLG_FONT_PATH "\"\n"
    ".global tlg_font_ttf_end\n"
    "tlg_font_ttf_end:\n"
    ".text\n");
#else
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden tlg_font_ttf\n"
    ".global tlg_font_ttf\n"
    "tlg_font_ttf:\n"
    ".incbin \"" TLG_FONT_PATH "\"\n"
    ".hidden tlg_font_ttf_end\n"
    ".global tlg_font_ttf_end\n"
    "tlg_font_ttf_end:\n"
    ".previous\n");
#endif
extern "C" const unsigned char tlg_font_ttf[];
extern "C" const unsigned char tlg_font_ttf_end[];

namespace TlgOverlay {
namespace {
// This overlay's own ImGui context: renderer v2's presenter composes the game image with ImGui
// in its own context, so each entry point here makes this one current and restores the other.
ImGuiContext* overlay_context = nullptr;
#ifdef TLG_RENDERER_HOSTED
struct ContextScope {
    ImGuiContext* previous = ImGui::GetCurrentContext();
    ContextScope() {
        if (overlay_context) ImGui::SetCurrentContext(overlay_context);
    }
    // Without another context (renderer v3 has none) the overlay's stays current.
    ~ContextScope() {
        if (previous) ImGui::SetCurrentContext(previous);
    }
};
#else
struct ContextScope {}; // this port's renderer: the overlay's context is the only one
#endif
} // namespace

namespace {

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
std::atomic<bool> menu_open{false};
bool r3_down = false, l2_down = false;
std::atomic<bool> loading{false};
std::atomic<u32> loading_done{0}, loading_total{0};
std::atomic<bool> loading_compile{false}; // first-run shader compilation (SetLoadingCompile)
std::chrono::steady_clock::time_point loading_compile_started{};
std::atomic<u32> background_done{0}, background_total{0};
bool dirty = false; // settings changed while open: saved on close
int requested_tab = -1; // switch on the next frame (L1/R1, Q/E, menu opened)
// The game's text dialog (SetTextEntry), guarded by imgui_mutex.
bool text_entry_active = false;
std::string text_entry_prompt, text_entry_text;
// The game's save dialogs (sceSaveDataDialog): the list, a message the player answers
// (overwrite?, no space...), the "Saving..." panel and the "Game saved" toast. The save's text
// comes from the game; every other word is the port's, in the menu language. Guarded by
// imgui_mutex; the game's thread polls the atomics.
using SaveClock = std::chrono::steady_clock;
std::atomic<bool> choice_active{false};  // the list
std::atomic<int> choice_result{-1};      // -1 open, -2 cancelled, else the chosen index
std::atomic<bool> message_active{false};
std::atomic<int> message_result{-1};     // -1 open, 1 yes/OK, 2 no/cancel
int save_kind = 1;                       // 1 save, 2 load, 3 delete
std::vector<SaveRow> save_rows;
int choice_index = 0, choice_first = 0;  // selected row, first row shown (the list scrolls)
int message_code = 0, message_buttons = 0, message_focus = 1; // focus: 0 no/cancel, 1 yes/OK
u64 message_value = 0;
std::string message_text;
bool message_has_entry = false;
SaveRow message_entry;
bool progress_on = false;
int progress_kind = 1, progress_percent = -1;
std::string progress_text;
SaveClock::time_point progress_since{}, progress_until{}; // until: a panel shown for a moment
SaveClock::time_point toast_since{}, toast_until{};
int toast_kind = 0;
// The mouse (display pixels) and the last frame's rectangles: the rows, the message's buttons.
ImVec2 save_mouse{-1.0f, -1.0f};
std::vector<ImRect> save_row_rects;
ImRect save_button_rects[2];
// Save pictures: decoded on the game's thread, made textures on the present thread (null =
// none). Closed dialogs' textures wait in choice_retired until the backend has destroyed them.
std::vector<TlgPng::Image> choice_pictures;
std::vector<ImTextureData*> choice_textures, choice_retired;
// Trophy icons (tlg_trophies.cpp): pictures waiting to become textures, then the textures, kept
// for the session (one per trophy, 25 small ones); the new trophies waiting for their toast.
std::vector<std::pair<int, TlgPng::Image>> trophy_pictures;
std::map<int, ImTextureData*> trophy_textures;
std::vector<int> trophy_queue;
int trophy_toast = -1;
std::chrono::steady_clock::time_point trophy_toast_since;
float trophy_toast_seconds = 6.0f;
std::atomic<bool> choice_retiring{false};
float base_scale = 1.0f;
// The style at scale 1 (Init): each new scale starts from it. Scaling the scaled style again
// truncated its sizes a little each time; after a few minimise / restore cycles (0.6 and back)
// WindowBorderHoverPadding reached 0 and ImGui's assert closed the game.
ImGuiStyle base_style;
// Faces beside the default Segoe UI (Init): Semibold for titles and buttons, Light for the menu's
// labels (the launcher's Font.Light), and Windows' icon font for the menu's glyphs (none: no icons).
// Without Segoe (Linux, the Steam Deck) they are the port's Open Sans and its subset of Fluent UI
// System Icons (port/fonts, as in the launcher's fonts.py); icons_fluent then maps the code points.
ImFont* font_bold = nullptr;
ImFont* font_light = nullptr;
ImFont* font_icons = nullptr;
bool icons_fluent = false;

// The game's frame behind the panels (SetBackdrop): the view the presenter gave for this present,
// and a descriptor per view the overlay has sampled (ForgetBackdrop drops one). imgui_mutex.
struct BackdropSet {
    VkImageView view;
    VkImageLayout layout;
    VkDescriptorSet set;
};
std::vector<BackdropSet> backdrop_sets;
VkImageView backdrop_view = VK_NULL_HANDLE; // this present's frame (null: none)
VkImageLayout backdrop_layout = VK_IMAGE_LAYOUT_UNDEFINED;
ImRect backdrop_where;                       // where the frame is on the swapchain image
ImTextureRef backdrop_texture;
bool backdrop_ready = false;                 // backdrop_texture holds this present's frame

// The game behind a panel, frosted as the launcher's glass: the frame drawn 25 times along a
// spiral (radius 44 launcher pixels), layer i at 1/(i+1) so that the layers average to a blur.
// False when this present has no frame: the caller then draws a more opaque glass.
bool Frost(ImDrawList* dl, ImVec2 a, ImVec2 b, float rounding) {
    const float rw = backdrop_where.GetWidth(), rh = backdrop_where.GetHeight();
    if (!backdrop_ready || rw < 1.0f || rh < 1.0f) return false;
    constexpr int Taps = 25;
    const float radius = 44.0f * base_scale;
    for (int i = 0; i < Taps; ++i) {
        float dx = 0.0f, dy = 0.0f;
        if (i) {
            const float r = radius * std::sqrt(float(i) / float(Taps - 1)), angle = float(i) * 2.39996f;
            dx = std::cos(angle) * r;
            dy = std::sin(angle) * r;
        }
        const ImVec2 uv0((a.x + dx - backdrop_where.Min.x) / rw, (a.y + dy - backdrop_where.Min.y) / rh);
        const ImVec2 uv1((b.x + dx - backdrop_where.Min.x) / rw, (b.y + dy - backdrop_where.Min.y) / rh);
        dl->AddImageRounded(backdrop_texture, a, b, uv0, uv1, IM_COL32(255, 255, 255, int(255.0f / float(i + 1) + 0.5f)),
                            rounding);
    }
    return true;
}

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;
// The last present intervals, oldest first from frame_history_next (the overlay's graph).
std::array<float, 240> frame_history{};
size_t frame_history_next = 0, frame_history_count = 0;
// The last 2048 intervals for the lows, percentiles and the histogram, in no order.
std::array<float, 2048> frame_long{};
size_t frame_long_next = 0, frame_long_count = 0;

// Colours of the launcher (launcher/fl_launcher/qml/theme.js): grey see-through panels like its
// pickers, white text, its blue accent, small corner radii.
namespace Theme {
constexpr ImVec4 Glass{28.0f / 255, 28.0f / 255, 30.0f / 255, 1.0f};   // panels, at GlassAlpha
constexpr ImVec4 Popover{36.0f / 255, 36.0f / 255, 40.0f / 255, 1.0f}; // dialogs, at PopoverAlpha
constexpr float GlassAlpha = 0.78f, PopoverAlpha = 0.97f;
constexpr ImVec4 Text{1.0f, 1.0f, 1.0f, 1.0f};
constexpr ImVec4 Muted{1.0f, 1.0f, 1.0f, 0.64f};
constexpr ImVec4 Faint{1.0f, 1.0f, 1.0f, 0.38f};
constexpr ImVec4 Line{1.0f, 1.0f, 1.0f, 0.1f};
constexpr ImVec4 Ink{0.0f, 0.0f, 0.0f, 1.0f}; // text on a white (focused) button
constexpr ImVec4 Accent{0x1f / 255.0f, 0x8f / 255.0f, 1.0f, 1.0f};
constexpr ImVec4 Good{0x2a / 255.0f, 0xd1 / 255.0f, 0x6a / 255.0f, 1.0f};
constexpr ImVec4 Warn{1.0f, 0xc3 / 255.0f, 0x5a / 255.0f, 1.0f};
constexpr ImVec4 Bad{1.0f, 0x6b / 255.0f, 0x5a / 255.0f, 1.0f};
constexpr float PanelRadius = 6.0f, ItemRadius = 4.0f;

constexpr ImVec4 White(float alpha) { return {1.0f, 1.0f, 1.0f, alpha}; }
constexpr ImVec4 Fade(const ImVec4& c, float alpha) { return {c.x, c.y, c.z, c.w * alpha}; }
inline ImU32 U32(const ImVec4& c, float alpha = 1.0f) {
    return ImGui::ColorConvertFloat4ToU32(Fade(c, std::clamp(alpha, 0.0f, 1.0f)));
}
} // namespace Theme

// The overlay's own software cursor, on its own context: SetOpen and the dialogs run from the
// input handler, where renderer v2's ImGui context is current (its present thread draws it), and
// ImGui::GetIO() there reached v2's context - the menu opened without a cursor (issue #4).
void SetDrawCursor(bool value) {
    if (overlay_context) overlay_context->IO.MouseDrawCursor = value;
}

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    SetDrawCursor(value);
    if (value) {
        requested_tab = TlgSettings::Get().menu_tab;
    }
    if (!value && dirty) {
        dirty = false;
        TlgSettings::Save();
    }
}

ImGuiKey KeyFromSdl(SDL_Keycode key) {
    switch (key) {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_Q: return ImGuiKey_Q;
    case SDLK_E: return ImGuiKey_E;
    case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_RCTRL: return ImGuiKey_RightCtrl;
    case SDLK_LSHIFT: return ImGuiKey_LeftShift;
    case SDLK_RSHIFT: return ImGuiKey_RightShift;
    case SDLK_LALT: return ImGuiKey_LeftAlt;
    case SDLK_RALT: return ImGuiKey_RightAlt;
    default: return ImGuiKey_None;
    }
}

ImGuiKey KeyFromGamepad(u8 button) {
    switch (button) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    case SDL_GAMEPAD_BUTTON_BACK: return ImGuiKey_GamepadBack;
    default: return ImGuiKey_None;
    }
}

float PixelDensity(SDL_WindowID id) {
    SDL_Window* window = SDL_GetWindowFromID(id);
    const float density = window ? SDL_GetWindowPixelDensity(window) : 1.0f;
    return density > 0.0f ? density : 1.0f;
}

// Marks the settings dirty when a widget changed them.
template <typename T>
void Store(std::atomic<T>& target, T value, bool changed) {
    if (changed) {
        target = value;
        dirty = true;
    }
}

// Controller button names in the help text follow TLG_GLYPHS=xbox|nintendo (upstream 9155cdc;
// a port's runner can pick it from `tlg-probe --gamepad`); unset is PlayStation. Read once: the
// environment is fixed for a run.
static const char* PadText(const char* playstation, const char* xbox, const char* nintendo) {
    static const char* style = std::getenv("TLG_GLYPHS");
    if (!style) return playstation;
    return !std::strcmp(style, "xbox") ? xbox : !std::strcmp(style, "nintendo") ? nintendo : playstation;
}

// The note for the item under the mouse or the gamepad/keyboard focus, shown in the footer.
const char* footer_help = nullptr;

void Help(const char* text) {
    if (text && (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) || ImGui::IsItemFocused())) {
        footer_help = text;
    }
}

// ---- The menu's rows, drawn as the launcher's settings (qml/SettingRow.qml, ToggleRow.qml,
// Picker.qml): the label (Segoe UI Light 25) and its description (19, muted) on the left, the
// control on the right, a faint line under the row, the row lit when hovered or focused. Sizes
// are launcher pixels on a 1080-line stage, times base_scale. A slider row is changed with the
// arrows after Enter / A (the arrows move between rows otherwise), a list row opens its list. ----
float L(float px) { return px * base_scale; }
ImFont* LightFont() { return font_light ? font_light : ImGui::GetFont(); }

// UTF-8 of one code point (the icon font's private-use glyphs).
std::string Utf8(unsigned code) {
    std::string out;
    if (code < 0x80) {
        out += char(code);
    } else if (code < 0x800) {
        out += char(0xc0 | (code >> 6));
        out += char(0x80 | (code & 0x3f));
    } else {
        out += char(0xe0 | (code >> 12));
        out += char(0x80 | ((code >> 6) & 0x3f));
        out += char(0x80 | (code & 0x3f));
    }
    return out;
}

// Segoe Fluent Icons / MDL2 code points (launcher/fl_launcher/qml/theme.js).
namespace Icon {
constexpr unsigned Chart = 0xE9D2, Display = 0xE7F4, Speed = 0xEC4A, Gamepad = 0xE7FC, Chevron = 0xE70D,
                   Check = 0xE73E, Bolt = 0xE945, Save = 0xE74E, Warning = 0xE7BA;
}

// The icon font's text for a Segoe code point: the same icon in Fluent UI System Icons (theme.js
// iconFluent) when that is the font. Fluent's glyphs sit smaller in their em, hence `scale`.
std::string IconText(unsigned code) {
    if (!icons_fluent) return Utf8(code);
    switch (code) {
    case Icon::Chart: code = 0xF33D; break;
    case Icon::Display: code = 0xF35A; break;
    case Icon::Speed: code = 0xF4C8; break;
    case Icon::Gamepad: code = 0xEEB9; break;
    case Icon::Chevron: code = 0xF2A4; break;
    case Icon::Check: code = 0xF295; break;
    case Icon::Bolt: code = 0xE619; break;
    case Icon::Save: code = 0xF680; break;
    case Icon::Warning: code = 0xF86A; break;
    }
    return Utf8(code);
}
float IconScale() { return icons_fluent ? 1.15f : 1.0f; }

// An icon of `size` launcher pixels centred at `c`; nothing without the icon font.
void DrawIcon(ImDrawList* dl, unsigned code, ImVec2 c, float size, ImU32 color) {
    if (!font_icons) return;
    const std::string text = IconText(code);
    size *= IconScale();
    const ImVec2 s = font_icons->CalcTextSizeA(L(size), FLT_MAX, 0.0f, text.c_str());
    dl->AddText(font_icons, L(size), ImVec2(c.x - s.x / 2, c.y - s.y / 2), color, text.c_str());
}

ImVec2 TextSize(ImFont* font, float px, const char* text, float wrap = 0.0f) {
    return font->CalcTextSizeA(L(px), FLT_MAX, wrap, text);
}

// The row edited with the arrows (a slider after Enter / A) and the slider dragged with the mouse.
ImGuiID editing_row = 0, dragging_row = 0;

struct RowState {
    bool activated, hovered, focused, editing;
    ImGuiID id;
    ImRect bb;
    float control_x, mid; // the control's left edge and the row's vertical centre
    ImDrawList* dl;
};

// One row: its frame, label and description; the control is drawn by the caller in
// [control_x, control_x + control_w] at height mid. Info rows take no focus.
RowState RowFrame(const char* label, const char* help, float control_w, bool interactive = true) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float width = ImGui::GetContentRegionAvail().x;
    const float inset = L(20);
    const float text_w = std::max(L(160), width - inset * 2 - control_w - L(28));
    const ImVec2 label_size = TextSize(LightFont(), 25, label, text_w);
    const ImVec2 help_size = help && *help ? TextSize(ImGui::GetFont(), 19, help, text_w) : ImVec2();
    const float text_h = label_size.y + (help_size.y > 0 ? L(4) + help_size.y : 0.0f);
    const float h = std::max(L(66), text_h + L(24));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    RowState r{};
    r.dl = dl;
    ImGui::PushID(label);
    if (interactive) {
        r.activated = ImGui::InvisibleButton("##row", ImVec2(width, h));
        r.id = ImGui::GetItemID();
        r.hovered = ImGui::IsItemHovered();
        r.focused = ImGui::IsItemFocused();
    } else {
        ImGui::Dummy(ImVec2(width, h));
    }
    ImGui::PopID();
    r.editing = interactive && editing_row == r.id;
    if (r.editing && !r.focused) {
        editing_row = 0; // the focus moved on
        r.editing = false;
    }
    r.bb = ImRect(pos, ImVec2(pos.x + width, pos.y + h));
    if (r.focused || r.hovered) {
        dl->AddRectFilled(r.bb.Min, r.bb.Max, ImGui::GetColorU32(Theme::White(r.focused ? 0.11f : 0.06f)), L(6));
    } else {
        dl->AddLine(ImVec2(pos.x + inset, r.bb.Max.y - 0.5f), ImVec2(r.bb.Max.x - inset, r.bb.Max.y - 0.5f),
                    ImGui::GetColorU32(Theme::Line));
    }
    r.mid = pos.y + h / 2;
    const float y0 = r.mid - text_h / 2;
    dl->AddText(LightFont(), L(25), ImVec2(pos.x + inset, y0), ImGui::GetColorU32(Theme::Text), label, nullptr, text_w);
    if (help_size.y > 0) {
        dl->AddText(ImGui::GetFont(), L(19), ImVec2(pos.x + inset, y0 + label_size.y + L(4)),
                    ImGui::GetColorU32(Theme::Muted), help, nullptr, text_w);
    }
    r.control_x = r.bb.Max.x - inset - control_w;
    return r;
}

// The launcher's switch: 64 x 34, blue when on, a white knob.
void DrawToggle(ImDrawList* dl, float x, float mid, bool on) {
    const ImVec2 a(x, mid - L(17)), b(x + L(64), mid + L(17));
    dl->AddRectFilled(a, b, ImGui::GetColorU32(on ? Theme::Accent : Theme::White(0.28f)), L(17));
    const float knob = on ? b.x - L(30) : a.x + L(4);
    dl->AddRectFilled(ImVec2(knob, a.y + L(4)), ImVec2(knob + L(26), a.y + L(30)), ImGui::GetColorU32(Theme::Text), L(13));
}

void CheckRow(const char* label, std::atomic<bool>& value, const char* help = nullptr) {
    const RowState r = RowFrame(label, help, L(64));
    bool v = value;
    if (r.activated) {
        v = !v;
        Store(value, v, true);
    }
    DrawToggle(r.dl, r.control_x, r.mid, v);
}

// A toggle over one bit of a mask (the overlay's items).
void BitRow(const char* label, std::atomic<unsigned>& mask, unsigned bit) {
    const RowState r = RowFrame(label, nullptr, L(64));
    unsigned m = mask;
    if (r.activated) {
        m ^= bit;
        mask = m;
        dirty = true;
    }
    DrawToggle(r.dl, r.control_x, r.mid, (m & bit) != 0);
}

// The current value and a chevron; Enter / A / a click opens the launcher's list (Picker.qml):
// 60-pixel items, the current one checked.
void ChoiceRow(const char* label, std::atomic<int>& target, const char* const* labels, const int* values,
               int count, const char* help = nullptr) {
    int current = 0;
    for (int i = 0; i < count; ++i) {
        if ((values ? values[i] : i) == target.load()) current = i;
    }
    const float value_w = std::min(TextSize(LightFont(), 25, labels[current]).x, ImGui::GetContentRegionAvail().x * 0.42f);
    const RowState r = RowFrame(label, help, value_w + L(30));
    const ImVec2 value_size = TextSize(LightFont(), 25, labels[current]);
    ImGui::PushClipRect(ImVec2(r.control_x, r.bb.Min.y), ImVec2(r.control_x + value_w, r.bb.Max.y), true);
    r.dl->AddText(LightFont(), L(25), ImVec2(r.control_x, r.mid - value_size.y / 2), ImGui::GetColorU32(Theme::White(0.9f)),
                  labels[current]);
    ImGui::PopClipRect();
    DrawIcon(r.dl, Icon::Chevron, ImVec2(r.control_x + value_w + L(20), r.mid + L(1)), 14, ImGui::GetColorU32(Theme::Muted));
    char popup[64];
    std::snprintf(popup, sizeof(popup), "##list%08x", r.id);
    if (r.activated) {
        ImGui::OpenPopup(popup);
    }
    ImGui::SetNextWindowPos(ImVec2(r.bb.Max.x, r.bb.Max.y + L(6)), ImGuiCond_Appearing, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(L(480), 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(L(12), L(12)));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, L(6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0, 0, 0, 0));
    if (ImGui::BeginPopup(popup)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (int i = 0; i < count; ++i) {
            ImGui::PushID(i);
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            const bool picked = ImGui::Selectable("##item", false, 0, ImVec2(0.0f, L(60)));
            if (i == current && ImGui::IsWindowAppearing()) ImGui::SetItemDefaultFocus();
            const bool focused = ImGui::IsItemFocused(), hovered = ImGui::IsItemHovered();
            const ImVec2 end(pos.x + ImGui::GetContentRegionAvail().x, pos.y + L(60));
            if (focused || hovered) {
                dl->AddRectFilled(pos, end, ImGui::GetColorU32(Theme::White(focused ? 0.12f : 0.06f)), L(4));
            }
            const ImVec2 size = TextSize(LightFont(), 26, labels[i]);
            dl->AddText(LightFont(), L(26), ImVec2(pos.x + L(18), pos.y + (L(60) - size.y) / 2),
                        ImGui::GetColorU32(Theme::Text), labels[i]);
            if (i == current) {
                DrawIcon(dl, Icon::Check, ImVec2(end.x - L(30), pos.y + L(30)), 18, ImGui::GetColorU32(Theme::Text));
            }
            if (picked) {
                Store(target, values ? values[i] : i, true);
            }
            ImGui::PopID();
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(3);
}

// A slider: a 260-pixel track (blue up to the value) with a white knob and the value's text.
// Dragged with the mouse; Enter / A starts the arrows (left / right step by 1/100 of the range,
// a log slider by 5%), Enter / A or moving away stops them. Returns the new value when changed.
bool SliderControl(const RowState& r, float& v, float lo, float hi, const char* text, bool log_scale, float step) {
    const float track_w = L(260), track_x = r.control_x, value_x = track_x + track_w + L(18);
    const auto to_t = [&](float x) {
        return log_scale ? std::log(x / lo) / std::log(hi / lo) : (x - lo) / (hi - lo);
    };
    const auto from_t = [&](float t) {
        t = std::clamp(t, 0.0f, 1.0f);
        return log_scale ? lo * std::pow(hi / lo, t) : lo + (hi - lo) * t;
    };
    bool changed = false;
    const ImGuiIO& io = ImGui::GetIO();
    if (r.activated && io.MouseDown[0] == false && dragging_row != r.id) {
        editing_row = editing_row == r.id ? 0 : r.id; // Enter / A
    }
    if (ImGui::IsItemActivated() && io.MouseDown[0] && io.MousePos.x >= track_x - L(14) &&
        io.MousePos.x <= track_x + track_w + L(14)) {
        dragging_row = r.id;
    }
    if (dragging_row == r.id) {
        if (io.MouseDown[0]) {
            const float nv = from_t((io.MousePos.x - track_x) / track_w);
            changed = nv != v;
            v = nv;
        } else {
            dragging_row = 0;
        }
    }
    if (editing_row == r.id) {
        int dir = 0;
        for (const ImGuiKey key : {ImGuiKey_LeftArrow, ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadLStickLeft}) {
            ImGui::SetKeyOwner(key, r.id);
            if (ImGui::IsKeyPressed(key, ImGuiInputFlags_Repeat, r.id)) dir = -1;
        }
        for (const ImGuiKey key : {ImGuiKey_RightArrow, ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadLStickRight}) {
            ImGui::SetKeyOwner(key, r.id);
            if (ImGui::IsKeyPressed(key, ImGuiInputFlags_Repeat, r.id)) dir = 1;
        }
        if (dir) {
            const float nv = log_scale ? std::clamp(v * (dir > 0 ? 1.05f : 1.0f / 1.05f), lo, hi)
                                       : std::clamp(v + dir * step, lo, hi);
            changed = nv != v;
            v = nv;
        }
    }
    const float t = std::clamp(to_t(v), 0.0f, 1.0f);
    const float y = r.mid - L(3);
    r.dl->AddRectFilled(ImVec2(track_x, y), ImVec2(track_x + track_w, y + L(6)), ImGui::GetColorU32(Theme::White(0.18f)), L(3));
    r.dl->AddRectFilled(ImVec2(track_x, y), ImVec2(track_x + track_w * t, y + L(6)), ImGui::GetColorU32(Theme::Accent), L(3));
    const ImVec2 knob(track_x + track_w * t, r.mid);
    if (editing_row == r.id) {
        r.dl->AddCircle(knob, L(17), ImGui::GetColorU32(Theme::Accent), 0, L(2.5f));
    }
    r.dl->AddCircleFilled(knob, L(11), ImGui::GetColorU32(Theme::Text));
    const ImVec2 size = TextSize(LightFont(), 25, text);
    r.dl->AddText(LightFont(), L(25), ImVec2(value_x + L(90) - size.x, r.mid - size.y / 2),
                  ImGui::GetColorU32(editing_row == r.id ? Theme::Accent : Theme::Text), text);
    return changed;
}

constexpr float SliderWidth = 260 + 18 + 90;

void FloatRow(const char* label, std::atomic<float>& value, float lo, float hi, const char* format, const char* help,
              ImGuiSliderFlags flags = 0) {
    const RowState r = RowFrame(label, help, L(SliderWidth));
    float v = value;
    char text[32];
    std::snprintf(text, sizeof(text), format, v);
    if (SliderControl(r, v, lo, hi, text, (flags & ImGuiSliderFlags_Logarithmic) != 0, (hi - lo) / 100.0f)) {
        Store(value, std::clamp(v, lo, hi), true);
    }
}

void SliderRow(const char* label, std::atomic<int>& value, int lo, int hi, const char* format, const char* help) {
    const RowState r = RowFrame(label, help, L(SliderWidth));
    float v = float(value.load());
    char text[32];
    std::snprintf(text, sizeof(text), format, value.load());
    const float step = std::max(1.0f, std::round((hi - lo) / 50.0f));
    if (SliderControl(r, v, float(lo), float(hi), text, false, step)) {
        Store(value, std::clamp(int(std::lround(v)), lo, hi), true);
    }
}

// A 0.1x to 100x multiplier (log slider; 1x is the middle of the track).
void MultiplierRow(const char* label, std::atomic<float>& value, const char* help) {
    const RowState r = RowFrame(label, help, L(SliderWidth));
    float v = value;
    char text[32];
    std::snprintf(text, sizeof(text), "%.2fx", v);
    if (SliderControl(r, v, 0.1f, 100.0f, text, true, 0.0f)) {
        value = TlgOrbs::ClampMultiplier(v);
        dirty = true;
    }
}

// Text on the right, wrapped within 45% of the row.
void InfoRow(const char* label, const char* text) {
    const float max_w = ImGui::GetContentRegionAvail().x * 0.45f;
    const float w = std::min(TextSize(ImGui::GetFont(), 21, text).x, max_w);
    const RowState r = RowFrame(label, nullptr, w, false);
    const ImVec2 size = TextSize(ImGui::GetFont(), 21, text, w);
    r.dl->AddText(ImGui::GetFont(), L(21), ImVec2(r.bb.Max.x - L(20) - size.x, r.mid - size.y / 2),
                  ImGui::GetColorU32(Theme::Muted), text, nullptr, w);
}

// The launcher's pill (PillButton.qml): white with black text when focused or pressed, frosted
// otherwise. `size` is its height; returns true when pressed.
bool Pill(const char* label, float size = 56, float font = 22, float min_w = 150) {
    const ImVec2 text = TextSize(ImGui::GetFont(), font, label);
    const ImVec2 bb_size(std::max(L(min_w), text.x + L(size)), L(size));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton(label, bb_size);
    const bool lit = ImGui::IsItemFocused() || ImGui::IsItemActive();
    const ImVec4 fill = lit ? Theme::Text : ImGui::IsItemHovered() ? Theme::White(0.24f) : Theme::White(0.14f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + bb_size.x, pos.y + bb_size.y), ImGui::GetColorU32(fill), bb_size.y / 2);
    ImFont* face = font_bold ? font_bold : ImGui::GetFont();
    const ImVec2 t = face->CalcTextSizeA(L(font), FLT_MAX, 0.0f, label);
    dl->AddText(face, L(font), ImVec2(pos.x + (bb_size.x - t.x) / 2, pos.y + (bb_size.y - t.y) / 2),
                ImGui::GetColorU32(lit ? Theme::Ink : Theme::Text), label);
    return pressed;
}

// A row whose control is a pill button (Restore).
bool ButtonRow(const char* label, const char* help, const char* button) {
    const float w = std::max(L(150), TextSize(ImGui::GetFont(), 22, button).x + L(56));
    const RowState r = RowFrame(label, help, w);
    const ImVec2 a(r.control_x, r.mid - L(28)), b(r.control_x + w, r.mid + L(28));
    r.dl->AddRectFilled(a, b, ImGui::GetColorU32(r.focused ? Theme::Text : Theme::White(r.hovered ? 0.24f : 0.14f)), L(28));
    ImFont* face = font_bold ? font_bold : ImGui::GetFont();
    const ImVec2 t = face->CalcTextSizeA(L(22), FLT_MAX, 0.0f, button);
    r.dl->AddText(face, L(22), ImVec2(a.x + (w - t.x) / 2, r.mid - t.y / 2),
                  ImGui::GetColorU32(r.focused ? Theme::Ink : Theme::Text), button);
    return r.activated;
}

// Upper case of a UTF-8 string: ASCII and Russian letters (the section titles).
std::string Upper(const char* text) {
    std::string out;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p;) {
        if (*p < 0x80) {
            out += char(std::toupper(*p++));
            continue;
        }
        if ((p[0] & 0xe0) == 0xc0 && p[1]) {
            unsigned code = ((p[0] & 0x1f) << 6) | (p[1] & 0x3f);
            if (code >= 0x430 && code <= 0x44f) code -= 0x20;
            else if (code == 0x451) code = 0x401;
            out += Utf8(code);
            p += 2;
            continue;
        }
        out += char(*p++);
    }
    return out;
}

// A section's title: small faint capitals with the launcher's letter spacing.
void SectionTitle(const char* text) {
    const std::string caps = Upper(text);
    ImGui::Dummy(ImVec2(0.0f, L(ImGui::GetCursorPosY() > L(10) ? 22 : 6)));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    pos.x += L(20);
    ImFont* font = ImGui::GetFont();
    for (const char* p = caps.c_str(); *p;) {
        const char* next = p + 1;
        while (*next && (u8(*next) & 0xc0) == 0x80) ++next;
        const std::string glyph(p, next);
        dl->AddText(font, L(17), pos, ImGui::GetColorU32(Theme::Faint), glyph.c_str());
        pos.x += font->CalcTextSizeA(L(17), FLT_MAX, 0.0f, glyph.c_str()).x + L(1.5f);
        p = next;
    }
    ImGui::Dummy(ImVec2(0.0f, L(17) + L(10)));
}

// The rows of a section (no table any more: each row draws itself).
bool BeginRows(const char*) { return true; }
void EndRows() {}

const char* UpscalerLabel(int upscaler) {
    switch (upscaler) {
    case TlgSettings::UpscalerFsr3: return "FSR 3.1";
    case TlgSettings::UpscalerFsr4: return "FSR 4";
    case TlgSettings::UpscalerFsr411: return "FSR 4.1.1";
    case TlgSettings::UpscalerTaa: return "TAA";
    case TlgSettings::UpscalerDlss: return "DLSS";
    default: return "";
    }
}


// The interface language: TLG_UI_LANG (the launcher passes its own, ru or en), ISS_UI_LANG, else
// the system locale (LANG/LC_*). Russian when it starts with "ru", English otherwise.
bool UiRussian() {
    static const bool ru = [] {
        for (const char* name : {"TLG_UI_LANG", "ISS_UI_LANG", "LC_MESSAGES", "LANG"}) {
            if (const char* v = std::getenv(name); v && *v) return (v[0] | 0x20) == 'r' && (v[1] | 0x20) == 'u';
        }
        return false;
    }();
    return ru;
}
const char* TL(const char* en, const char* ru) { return UiRussian() ? ru : en; }

// A count's word: English one/many, Russian one (1, 21), few (2-4, 22-24) or many (5-20, 25...).
const char* Plural(int n, const char* en_one, const char* en_many, const char* ru_one, const char* ru_few,
                   const char* ru_many) {
    if (!UiRussian()) return n == 1 ? en_one : en_many;
    const int tens = n % 100, ones = n % 10;
    return tens / 10 == 1 ? ru_many : ones == 1 ? ru_one : ones >= 2 && ones <= 4 ? ru_few : ru_many;
}
const Vulkan::Instance* overlay_instance = nullptr; // for the overlay's measurements

// The performance overlay's text with the current settings (also the menu's preview).
std::string OverlayText(unsigned exclude = 0) {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const unsigned items = s.overlay_items & ~exclude;
    if ((items & (P::Gpu | P::Cpu | P::Ram | P::Vram)) && overlay_instance) {
        P::Start(*overlay_instance);
    }
    P::Sample sample = P::Latest();
    sample.fps = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
    sample.frame_ms = frame_ms_avg;
    sample.upscaler = UpscalerLabel(s.upscaler);
    return P::Format(sample, items, s.overlay_layout == 0);
}

constexpr const char* Unsupported = "Unavailable: game version or code signature mismatch.";


void ImageTab() {
    namespace G = TlgGraphics;
    auto& s = TlgSettings::Get();
    SectionTitle(TL("Display", "Экран"));
    if (BeginRows("##display")) {
        static const char* const modes[] = {TL("Windowed", "Оконный"), TL("Borderless window", "Окно без рамки"), TL("Fullscreen", "Полный экран")};
        ChoiceRow(TL("Display mode", "Режим экрана"), s.display_mode, G::DisplayModeCount == 3 ? modes : G::DisplayModeNames.data(),
                  nullptr, G::DisplayModeCount, TL("Windowed, borderless window or exclusive fullscreen.", "Окно, окно без рамки или эксклюзивный полный экран."));
        CheckRow("VSync", s.vsync, TL("Waits for the monitor: no tearing, a little more input delay.", "Ждёт монитор: без разрывов кадра, но задержка ввода чуть больше."));
        EndRows();
    }
    const int iw = s.image_width, ih = s.image_height, ww = s.window_width, wh = s.window_height;
    const bool upscaling = iw < ww && ih < wh;
    const bool downscaling = iw > ww || ih > wh;
    const bool rcas_blocked = s.rcas && !s.rcas_applied;
    // tlg: only spatial filters on the final image. The temporal upscalers (FSR 3/4, TAA, DLSS) were
    // built for GoW3's pipeline and make TLG's image shake, so they are not offered here.
    SectionTitle(TL("Scaling and sharpening", "Масштабирование и резкость"));
    if (BeginRows("##scaling")) {
        CheckRow(TL("Upscale with FSR 1", "Увеличение через FSR 1"), s.fsr1,
                 TL("Enlarges the game image when it is smaller than the window (spatial filter, no shimmer).", "Увеличивает картинку игры, когда она меньше окна (пространственный фильтр, без дрожания)."));
        ImGui::BeginDisabled(rcas_blocked);
        CheckRow(TL("Sharpening (RCAS)", "Резкость (RCAS)"), s.rcas,
                 rcas_blocked && downscaling ? TL("No effect: the game's image is larger than the window.", "Не действует: картинка игры больше окна.")
                 : rcas_blocked              ? TL("Needs FSR 1: the game's image is smaller than the window.", "Нужен FSR 1: картинка игры меньше окна.")
                                             : TL("Sharpens the final image.", "Делает итоговую картинку чётче."));
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!s.rcas || rcas_blocked);
        SliderRow(TL("Sharpening strength", "Сила резкости"), s.rcas_strength, 0, 100, "%d%%", TL("How strong the sharpening is.", "Насколько сильно повышается резкость."));
        ImGui::EndDisabled();
        EndRows();
    }
    SectionTitle(TL("Draw distance and detail", "Дальность прорисовки и детализация"));
    if (BeginRows("##view")) {
        ImGui::BeginDisabled(!s.view_supported);
        FloatRow(TL("LOD distance", "Дальность LOD"), s.lod_scale, 0.25f, 4.0f, "%.2fx",
                 s.view_supported ? TL("How far objects keep their detailed models (every LOD band). Above 1x costs CPU and GPU; "
                                       "the game still lowers it on its own in heavy scenes.",
                                       "Как далеко объекты сохраняют детальные модели (все ступени LOD). Выше 1x нагружает CPU и GPU; "
                                       "в тяжёлых сценах игра всё равно снижает её сама.")
                                  : TL(Unsupported, "Недоступно: другая версия игры или не совпала сигнатура кода."));
        FloatRow(TL("Regions' LOD_ADJUST", "LOD_ADJUST регионов"), s.region_lod, 0.0f, 1.0f, "%.2f",
                 TL("How much the map regions' own LOD changes count: 1 as in the game, 0 ignored.",
                    "Насколько учитываются собственные изменения LOD у регионов карты: 1 — как в игре, 0 — не учитываются."));
        FloatRow(TL("Regions' FADE_DISTANCE", "FADE_DISTANCE регионов"), s.region_fade, 0.25f, 4.0f, "%.2fx",
                 TL("Distance over which a region's lighting, haze and LOD settings blend in (multiplier).",
                    "Расстояние, на котором свет, дымка и LOD региона плавно сменяются при входе в него (множитель)."));
        FloatRow(TL("Near clip (NEAR_CLIP)", "Ближняя плоскость (NEAR_CLIP)"), s.near_clip, 1.0f, 100.0f, "%.1f",
                 TL("Default near plane of the cameras (the game uses 10). Lower shows objects closer to the camera, "
                    "higher reduces depth flicker far away. Applies to the next camera (load, cutscene).",
                    "Ближняя плоскость отсечения камер (в игре 10). Меньше — объекты видны ближе к камере, "
                    "больше — меньше мерцания глубины вдали. Действует со следующей камеры (загрузка, ролик)."));
        FloatRow(TL("Far clip (FAR_CLIP)", "Дальняя плоскость (FAR_CLIP)"), s.far_clip, 1000.0f, 1.0e7f, "%.0f",
                 TL("Default far plane of the cameras (the game uses 10000000, unlimited). Lower cuts the view; "
                    "applies to the next camera.",
                    "Дальняя плоскость отсечения камер (в игре 10000000, без ограничения). Меньше — обрезает обзор; "
                    "действует со следующей камеры."),
                 ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        FloatRow(TL("Texture LOD bias (MipLodBias)", "Смещение LOD текстур (MipLodBias)"), s.mip_lod_bias, -3.0f, 3.0f,
                 "%+.2f",
                 TL("Negative: sharper textures at a distance (more shimmer, a little more GPU). Positive: blurrier.",
                    "Отрицательное: текстуры вдали чётче (больше мерцания, чуть больше нагрузка на GPU). Положительное: размытее."));
        if (ButtonRow(TL("Restore the game's values", "Вернуть значения игры"),
                      TL("Every setting above back to the game's own.", "Все настройки выше — как в игре."),
                      TL("Restore", "Сбросить"))) {
            s.lod_scale = 1.0f;
            s.region_lod = 1.0f;
            s.region_fade = 1.0f;
            s.near_clip = 10.0f;
            s.far_clip = 1.0e7f;
            s.mip_lod_bias = 0.0f;
            const float bands[6] = {400.0f, 750.0f, 1500.0f, 3500.0f, 6000.0f, 15000.0f};
            for (int b = 0; b < 6; ++b) s.lod_bands[b] = bands[b];
            dirty = true;
        }
        EndRows();
    }
    SectionTitle(TL("Information", "Сведения"));
    if (BeginRows("##image_info")) {
        char text[96];
        std::snprintf(text, sizeof(text), "%d x %d", iw, ih);
        InfoRow(TL("Game image", "Картинка игры"), text);
        std::snprintf(text, sizeof(text), "%d x %d", ww, wh);
        InfoRow(TL("Windowed", "Окно"), text);
        InfoRow("FSR 1", !s.fsr1 ? TL("off", "выкл.") : upscaling ? TL("active", "работает") : TL("no effect (image not smaller than window)", "не действует (картинка не меньше окна)"));
        InfoRow(TL("Sharpening", "Резкость"), s.rcas_applied ? TL("applied", "применяется") : TL("not applied", "не применяется"));
        EndRows();
    }
}

void FixesSection();

void PerformanceTab() {
    if (TlgPerf::Enabled()) {
        SectionTitle(TL("Performance capture", "Запись производительности"));
        static int category = 0;
        const char* labels[] = {TL("Absorb smoke", "Поглощение дыма"), TL("Use power", "Применение силы"),
                                TL("Movement", "Передвижение"), TL("Other", "Другое")};
        ImGui::Combo(TL("Event", "Событие"), &category, labels, 4);
        if (ImGui::Button(TL("Mark performance event", "Отметить событие"))) {
            const bool accepted = TlgPerf::Mark(static_cast<unsigned>(category + 1));
            TlgOverlay::Notice(accepted ? TL("Event marked", "Событие отмечено") :
                                          TL("Capture queue full", "Очередь записи заполнена"), 2.0f);
        }
    }
    namespace G = TlgGraphics;
    auto& s = TlgSettings::Get();
    // tlg: GoW3's render-resolution and engine-frame-rate patches and the readback shortcuts are not
    // offered: they do not apply to TLG, and the readback ones crash it (it needs precise readbacks).
    SectionTitle(TL("Live", "Действуют сразу"));
    if (BeginRows("##live")) {
        // A label per value: the list draws one row per value (60 had none, the list read past
        // the labels and crashed, and 60 showed as "120": issue #13).
        static const char* const fps_labels[] = {"30", "60", "120", "240", TL("Unlimited", "Без ограничения")};
        static_assert(std::size(fps_labels) == G::FpsLimits.size());
        ChoiceRow(TL("Frame rate limit", "Ограничение FPS"), s.fps_limit, fps_labels, G::FpsLimits.data(), int(G::FpsLimits.size()),
                  TL("Caps the presented frames; never above the game's own rate.", "Ограничивает число выводимых кадров; не выше скорости самой игры."));
        static const char* const queued_labels[] = {TL("1 (lowest latency)", "1 (меньше задержка)"), "2", TL("Unlimited", "Без ограничения")};
        static_assert(std::size(queued_labels) == G::FramesQueued.size());
        ChoiceRow(TL("Frames queued", "Кадров в очереди"), s.frames_queued, queued_labels, G::FramesQueued.data(),
                  int(G::FramesQueued.size()), TL("More frames queued: steadier FPS, slower response to input.", "Больше кадров в очереди: ровнее FPS, но медленнее отклик."));
        CheckRow(TL("Compile new shaders in the background", "Компилировать новые шейдеры в фоне"), s.async_shaders,
                 TL("Fewer stutters in new areas; an object or effect may appear a moment later.", "Меньше подтормаживаний в новых местах; объект или эффект может появиться на мгновение позже."));
        EndRows();
    }
    FixesSection();
}


// tlg: the fixes (common/backports.h) and the A/B switches. Live ones change at once; the others
// are only recorded in tlg.ini (bpNN=0/1, wqm=0/1, ...) and apply on the next start, when the
// launcher turns the keys into TLG_BPnn / TLG_<NAME>. The table is iterated, so a reordered
// backports.h still lists every entry.
#ifdef TLG_RENDERER_HOSTED
// Renderers v2 and v3 have their own fixes and switches (ISS_* / SHADPS4_* variables): none of this port's.
void FixesSection() {
    SectionTitle(TL("Fixes", "Исправления"));
#ifdef TLG_RENDERER_V3
    InfoRow(TL("Renderer v3", "Рендер v3"), TL("its own fixes apply", "действуют его собственные исправления"));
#else
    InfoRow(TL("Renderer v2", "Рендер v2"), TL("its own fixes apply", "действуют его собственные исправления"));
#endif
}
#else
void FixesSection() {
    namespace B = Common::Backports;
    auto& s = TlgSettings::Get();
    SectionTitle(TL("Fixes", "Исправления"));
    ImGui::TextDisabled("%s", TL("Marked (next start) are saved now and apply the next time the game starts.",
                                 "Пункты с пометкой «(след. запуск)» сохраняются сразу, а действуют со следующего запуска игры."));
    if (ImGui::BeginTable("##fixes", 4, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 0.50f);
        ImGui::TableSetupColumn("on", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("hits", ImGuiTableColumnFlags_WidthStretch, 0.14f);
        ImGui::TableSetupColumn("source", ImGuiTableColumnFlags_WidthStretch, 0.36f);
        for (std::size_t i = 0; i < B::kEntries.size(); ++i) {
            const B::Entry& e = B::kEntries[i];
            const B::Id id = static_cast<B::Id>(i);
            const bool live = B::IsLive(id);
            const bool running = B::On(id);
            const int pref = i < std::size_t(TlgSettings::kBackportPrefs) ? s.backport_pref[i].load() : -1;
            bool shown = live || pref < 0 ? running : pref != 0;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%s  %s", e.id, e.what);
            ImGui::TableNextColumn();
            ImGui::PushID(e.id);
            if (ImGui::Checkbox("##on", &shown)) {
                if (live) {
                    B::SetLive(id, shown);
                }
                if (i < std::size_t(TlgSettings::kBackportPrefs)) {
                    s.backport_pref[i] = shown ? 1 : 0;
                    dirty = true;
                }
            }
            Help(live ? TL("Applies at once.", "Действует сразу.")
                      : TL("Applies on next start.", "Действует со следующего запуска."));
            ImGui::PopID();
            if (!live) {
                ImGui::SameLine();
                const bool differs = shown != running;
                ImGui::TextColored(differs ? Theme::Warn : Theme::Faint, "%s",
                                   TL("(next start)", "(след. запуск)"));
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%llu", static_cast<unsigned long long>(B::Hits(id)));
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", e.source);
        }
        ImGui::EndTable();
    }

    // A/B switches read once at start (recompiler or device creation): next start only.
    SectionTitle(TL("A/B switches (next start)", "Переключатели A/B (след. запуск)"));
    struct Extra { const char* name; bool (*current)(); };
    static const Extra extras[] = {
        {"WQM", Shader::Toggles::WholeQuadMode},
        {"LDS_SMALL", Shader::Toggles::SmallGroupLdsBarriers},
        {"LDS_MULTI", Shader::Toggles::MultiWaveLdsBarriers},
        {"LDS_READLANE", Shader::Toggles::LdsUniformReadLane},
        {"LANE_DIVERGENCE", Shader::Toggles::LaneDivergence},
        {"RFL_ALL", Shader::Toggles::ReadFirstLaneAll},
        {"BLOCK1D", nullptr},
        {"BC_TINY", nullptr},
        {"NO_MSAA", nullptr},
        {"DYNAMIC_STRIDE", Shader::Toggles::DynamicBufferStride},
    };
    static_assert(sizeof(extras) / sizeof(extras[0]) == TlgSettings::kExtraPrefs);
    if (BeginRows("##extras")) {
        static const char* const choices[] = {TL("Default", "По умолчанию"), TL("Off", "Выкл."), TL("On", "Вкл.")};
        static const int values[] = {-1, 0, 1};
        for (int e = 0; e < TlgSettings::kExtraPrefs; ++e) {
            char label[64];
            std::snprintf(label, sizeof(label), "%s%s", extras[e].name,
                          extras[e].current ? (extras[e].current() ? TL("  (running: on)", "  (сейчас: вкл.)")
                                                                    : TL("  (running: off)", "  (сейчас: выкл.)"))
                                            : "");
            ChoiceRow(label, s.extra_pref[e], choices, values, 3,
                      TL("Read once at start; saved in tlg.ini and applied on the next start.",
                         "Читается один раз при запуске; сохраняется в tlg.ini и применяется при следующем запуске."));
        }
        EndRows();
    }
}
#endif

void OpenGuide(); // the graffiti guide (below, with the save dialogs)

// ---- Keys (the Controls tab): every PS4 button and stick direction of the keyboard map
// (src/runtime_pad.c, the TLG_KEY_MAP syntax) with its keys and mouse buttons. A row waits for
// the next key or mouse button (Esc cancels) and gives it to that action alone. The result goes
// to key_map, which the pad thread applies on its next sample; the menu saves it on close. ----
struct KeyAction {
    const char* id;
    const char* en;
    const char* ru;
    const char* keys; ///< the defaults (runtime_pad.c key_map_default), '|' between keys
};
constexpr KeyAction key_actions[] = {
    {"move_up", "Move forward", "Вперёд", "W"},
    {"move_down", "Move back", "Назад", "S"},
    {"move_left", "Move left", "Влево", "A"},
    {"move_right", "Move right", "Вправо", "D"},
    {"cross", "Cross", "Крестик", "Space"},
    {"circle", "Circle", "Кружок", "Left Shift"},
    {"square", "Square", "Квадрат", "E"},
    {"triangle", "Triangle", "Треугольник", "Q"},
    {"l1", "L1", "L1", "1"},
    {"r1", "R1", "R1", "3"},
    {"l2", "L2", "L2", "R|Mouse Right"},
    {"r2", "R2", "R2", "F|Mouse Left"},
    {"l3", "L3", "L3", "Z"},
    {"r3", "R3", "R3", "C|Mouse Middle"},
    {"options", "Options", "Options", "Return"},
    {"touchpad_left", "Touchpad, left side", "Тачпад, левая сторона", "Tab"},
    {"touchpad_right", "Touchpad, right side", "Тачпад, правая сторона", "Backspace"},
    {"up", "D-pad up", "Крестовина вверх", "I"},
    {"down", "D-pad down", "Крестовина вниз", "K"},
    {"left", "D-pad left", "Крестовина влево", "J"},
    {"right", "D-pad right", "Крестовина вправо", "L"},
    {"look_up", "Look up", "Камера вверх", "Up"},
    {"look_down", "Look down", "Камера вниз", "Down"},
    {"look_left", "Look left", "Камера влево", "Left"},
    {"look_right", "Look right", "Камера вправо", "Right"},
};
constexpr int key_action_count = int(sizeof(key_actions) / sizeof(key_actions[0]));
int key_capture = -1; ///< the action waiting for a key, -1 none
std::chrono::steady_clock::time_point key_capture_since;

std::string Trimmed(std::string text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.erase(text.begin());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) text.pop_back();
    return text;
}

bool SameName(const std::string& a, const std::string& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
           });
}

// Each action's keys ('|' between them, empty for none): the defaults, then key_map's entries.
std::vector<std::string> CurrentKeys() {
    std::vector<std::string> keys(key_action_count);
    for (int i = 0; i < key_action_count; ++i) keys[i] = key_actions[i].keys;
    const std::string map = TlgSettings::KeyMap();
    for (size_t at = 0; at <= map.size();) {
        const size_t end = std::min(map.find(',', at), map.size());
        const std::string entry = map.substr(at, end - at);
        at = end + 1;
        const size_t eq = entry.find('=');
        if (eq == std::string::npos) continue;
        std::string id = Trimmed(entry.substr(0, eq));
        const std::string value = Trimmed(entry.substr(eq + 1));
        if (SameName(id, "touchpad")) id = "touchpad_left";
        for (int i = 0; i < key_action_count; ++i) {
            if (SameName(id, key_actions[i].id)) keys[i] = SameName(value, "none") ? "" : value;
        }
    }
    return keys;
}

// key_map from `keys`: the actions whose keys differ from their defaults.
void StoreKeys(const std::vector<std::string>& keys) {
    std::string map;
    for (int i = 0; i < key_action_count; ++i) {
        if (keys[i] == key_actions[i].keys) continue;
        map += (map.empty() ? "" : ",") + std::string(key_actions[i].id) + "=" + (keys[i].empty() ? "none" : keys[i]);
    }
    TlgSettings::SetKeyMap(map);
    dirty = true;
}

// Gives action `index` the key `name` alone and takes it from the others.
void AssignKey(int index, const std::string& name) {
    std::vector<std::string> keys = CurrentKeys();
    for (std::string& list : keys) {
        std::string kept;
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find('|', at), list.size());
            const std::string key = Trimmed(list.substr(at, end - at));
            at = end + 1;
            if (!key.empty() && !SameName(key, name)) kept += (kept.empty() ? "" : "|") + key;
        }
        list = kept;
    }
    keys[index] = name;
    StoreKeys(keys);
}

// A key's name as the menu shows it: the mouse in words, the rest as SDL names it.
std::string KeyLabel(const std::string& keys) {
    static const char* const mouse[][3] = {
        {"Mouse Left", "Left mouse button", "Левая кнопка мыши"},
        {"Mouse Right", "Right mouse button", "Правая кнопка мыши"},
        {"Mouse Middle", "Middle mouse button", "Средняя кнопка мыши"},
        {"Mouse X1", "Mouse button 4", "Кнопка мыши 4"},
        {"Mouse X2", "Mouse button 5", "Кнопка мыши 5"},
        {"Wheel Up", "Wheel up", "Колесо вверх"},
        {"Wheel Down", "Wheel down", "Колесо вниз"},
        {"Wheel Left", "Wheel left", "Колесо влево"},
        {"Wheel Right", "Wheel right", "Колесо вправо"},
    };
    if (keys.empty()) return TL("none", "нет");
    std::string out;
    for (size_t at = 0; at <= keys.size();) {
        const size_t end = std::min(keys.find('|', at), keys.size());
        std::string key = Trimmed(keys.substr(at, end - at));
        at = end + 1;
        for (const auto& m : mouse) {
            if (SameName(key, m[0])) key = TL(m[1], m[2]);
        }
        out += (out.empty() ? "" : " / ") + key;
    }
    return out;
}

// The key or mouse button an event gives the waiting action ("" for none); Esc and a gamepad
// button cancel. Insert, F10 and Esc stay the overlay's; ',', '=' and '|' cannot be written to
// key_map. Mouse presses count from a moment after the click that opened the wait.
bool CaptureKey(const SDL_Event& event) {
    if (key_capture < 0) return false;
    std::string name;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN: {
        if (event.key.repeat) return true;
        if (event.key.key == SDLK_ESCAPE) {
            key_capture = -1;
            return true;
        }
        if (event.key.key == SDLK_INSERT || event.key.key == SDLK_F10) return true;
        const char* text = SDL_GetScancodeName(event.key.scancode);
        name = text ? text : "";
        if (name.find_first_of(",=|") != std::string::npos) return true;
        break;
    }
    case SDL_EVENT_KEY_UP:
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        if (std::chrono::steady_clock::now() - key_capture_since < std::chrono::milliseconds(250)) return true;
        static const char* const buttons[] = {"", "Mouse Left", "Mouse Middle", "Mouse Right", "Mouse X1", "Mouse X2"};
        if (event.button.button < 1 || event.button.button > 5) return true;
        name = buttons[event.button.button];
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP:
        return true;
    case SDL_EVENT_MOUSE_WHEEL:
        name = event.wheel.y > 0 ? "Wheel Up" : event.wheel.y < 0 ? "Wheel Down" : event.wheel.x > 0 ? "Wheel Right" : "Wheel Left";
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        key_capture = -1;
        return true;
    default:
        return false;
    }
    if (!name.empty()) {
        AssignKey(key_capture, name);
        key_capture = -1;
    }
    return true;
}

void KeysSection() {
    SectionTitle(TL("Keys", "Клавиши"));
    const char* env = std::getenv("TLG_KEY_MAP");
    if (env && env[0]) {
        if (BeginRows("##keys_env")) {
            InfoRow(TL("Set by TLG_KEY_MAP", "Задано через TLG_KEY_MAP"), env);
            EndRows();
        }
        return;
    }
    const std::vector<std::string> keys = CurrentKeys();
    if (BeginRows("##keys")) {
        for (int i = 0; i < key_action_count; ++i) {
            const bool waiting = key_capture == i;
            const std::string value = waiting ? TL("Press a key...", "Нажмите клавишу...") : KeyLabel(keys[i]);
            const float value_w = std::min(TextSize(LightFont(), 25, value.c_str()).x, L(420));
            const RowState r = RowFrame(TL(key_actions[i].en, key_actions[i].ru),
                                        waiting ? TL("Esc cancels.", "Esc — отмена.") : nullptr, value_w);
            const ImVec2 size = TextSize(LightFont(), 25, value.c_str());
            r.dl->AddText(LightFont(), L(25), ImVec2(r.control_x, r.mid - size.y / 2),
                          ImGui::GetColorU32(waiting ? Theme::Accent : keys[i] == key_actions[i].keys ? Theme::White(0.9f) : Theme::Text),
                          value.c_str());
            if (r.activated) {
                key_capture = i;
                key_capture_since = std::chrono::steady_clock::now();
            }
        }
        EndRows();
    }
    if (ButtonRow(TL("Default keys", "Клавиши по умолчанию"),
                  TL("Every action gets its keys back.", "Возвращает всем действиям их клавиши."), TL("Reset", "Сбросить"))) {
        TlgSettings::SetKeyMap("");
        key_capture = -1;
        dirty = true;
    }
}

void ControlsTab() {
    auto& s = TlgSettings::Get();
    // Mouse (src/runtime_pad.c): the camera settings apply at once. The buttons and the touchpad
    // mode are fixed unless TLG_KEY_MAP / TLG_MOUSE_PAD_* (or tlg.ini key_map) change them.
    SectionTitle(TL("Mouse", "Мышь"));
    if (BeginRows("##mouse")) {
        CheckRow(TL("Mouse turns the camera", "Мышь управляет камерой"), s.mouse_enabled,
                 TL("The game window takes the mouse while it has focus. Esc releases it; a click in the window takes it back.",
                    "Окно игры захватывает мышь, пока оно в фокусе. Esc отпускает её, щелчок по окну захватывает снова."));
        ImGui::BeginDisabled(!s.mouse_enabled);
        const bool direct_ready = tlg_mouse_camera_ready() != 0;
        if (direct_ready) {
            CheckRow(TL("Mouse turns the camera directly", "Мышь поворачивает камеру напрямую"), s.mouse_direct,
                     TL("1:1, with no speed limit, as in PC games. Off: the mouse acts as the right stick (the game's own turn speed caps it).",
                        "1:1, без ограничения скорости, как в играх для ПК. Выкл.: мышь работает как правый стик (скорость поворота ограничена игрой)."));
        }
        const bool direct = direct_ready && s.mouse_direct;
        FloatRow(TL("Sensitivity", "Чувствительность"), s.mouse_sensitivity, 0.1f, 5.0f, "%.2f",
                 TL("How far the camera turns for the same mouse movement.", "Насколько поворачивается камера при одном и том же движении мыши."));
        CheckRow(TL("Invert Y axis", "Инвертировать ось Y"), s.mouse_invert_y,
                 TL("Moving the mouse forward looks down.", "Движение мыши от себя опускает взгляд."));
        // The stick's filters: the direct camera takes the motion as it is.
        ImGui::BeginDisabled(direct);
        FloatRow(TL("Smoothing", "Сглаживание"), s.mouse_smoothing, 0.0f, 0.9f, "%.2f",
                 TL("Evens out the camera motion; higher values feel softer and a little slower.",
                    "Делает движение камеры ровнее; при больших значениях оно мягче и чуть медленнее."));
        FloatRow(TL("Acceleration", "Ускорение"), s.mouse_accel, 0.0f, 2.0f, "%.2f",
                 TL("Fast movements turn more than slow ones. 0 is linear.", "Быстрые движения поворачивают сильнее медленных. 0 — линейно."));
        FloatRow(TL("Anti-dead zone", "Компенсация мёртвой зоны"), s.mouse_deadzone, 0.0f, 0.5f, "%.2f",
                 TL("Small mouse movements skip the game's stick dead zone. Raise it if slow movements do nothing.",
                    "Мелкие движения мыши перескакивают мёртвую зону стика в игре. Увеличьте, если медленные движения ничего не делают."));
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        InfoRow(TL("Touchpad mode", "Режим тачпада"), TL("hold Left Ctrl or Mouse X1", "удерживайте левый Ctrl или кнопку мыши X1"));
        InfoRow(TL("In touchpad mode", "В режиме тачпада"),
                TL("left touch/click/drag, middle touch, right two fingers", "левая — касание/нажатие/свайп, средняя — касание, правая — два пальца"));
        InfoRow(TL("Release the mouse", "Отпустить мышь"), "Esc");
        EndRows();
    }
    KeysSection();
    SectionTitle(TL("Sensor mode (graffiti)", "Режим сенсора (граффити)"));
    if (BeginRows("##sensor")) {
        SliderRow(TL("Graffiti aiming speed", "Скорость прицела граффити"), s.sensor_aim, 25, 200, "%d%%",
                  TL("How fast the stick moves the graffiti cursor in sensor mode (100% = default).",
                     "Как быстро стик двигает курсор граффити в режиме сенсора (100% — по умолчанию)."));
        EndRows();
    }
    if (ButtonRow(TL("Graffiti without a gyro", "Граффити без гироскопа"),
                  TL("Shows the steps again (the same screen as the first time).",
                     "Снова показывает шаги (тот же экран, что и в первый раз)."),
                  TL("Show", "Показать"))) {
        SetOpen(false);
        OpenGuide();
    }
}

ImVec2 DrawOverlay(ImDrawList* dl, ImVec2 at, float k); // the square styles (below, with the overlay)
void TrophiesTab(); // below, with the save dialogs' painter

void OverlayTab() {
    namespace P = TlgPerfStats;
    using namespace TlgSettings;
    auto& s = Get();
    if (BeginRows("##overlay_on")) {
        CheckRow(TL("Show overlay (F10)", "Показывать оверлей (F10)"), s.show_fps, TL("The performance overlay in a corner of the screen.", "Оверлей производительности в углу экрана."));
        EndRows();
    }
    ImGui::BeginDisabled(!s.show_fps);
    SectionTitle(TL("Style", "Вид"));
    if (BeginRows("##style")) {
        static const char* const styles[] = {TL("Card (earlier versions)", "Карточка (как раньше)"), TL("Compact", "Компактный"),
                                             TL("Strip", "Лента"), TL("Graph and histogram", "График и гистограмма"),
                                             TL("Telemetry", "Телеметрия"), TL("Mini", "Мини")};
        static_assert(std::size(styles) == OverlayStyleCount);
        ChoiceRow(TL("Overlay style", "Стиль оверлея"), s.overlay_style, styles, nullptr, OverlayStyleCount,
                  TL("Colours: green within the frame rate limit (60 without one), amber up to twice its frame time, red beyond.",
                     "Цвета: зелёный — в пределах лимита FPS (без лимита — 60), жёлтый — кадр до двух раз дольше, красный — дольше."));
        EndRows();
    }
    // The overlay as it will look, in a card under the style: the square styles drawn live at
    // their real size, the card's text.
    {
        ImGui::Dummy(ImVec2(0.0f, L(12)));
        const float w = ImGui::GetContentRegionAvail().x - L(40);
        const ImVec2 pos(ImGui::GetCursorScreenPos().x + L(20), ImGui::GetCursorScreenPos().y);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float content_h;
        if (s.overlay_style == OverlayCard) {
            const std::string text = OverlayText();
            const char* shown = text.empty() ? TL("(nothing selected)", "(ничего не выбрано)") : text.c_str();
            const float size = 19.0f * s.overlay_scale / 100.0f;
            content_h = TextSize(ImGui::GetFont(), size, shown, w - L(32)).y;
            dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + content_h + L(56)), ImGui::GetColorU32(Theme::White(0.04f)), L(10));
            dl->AddText(ImGui::GetFont(), L(size), ImVec2(pos.x + L(16), pos.y + L(38)), ImGui::GetColorU32(Theme::Text), shown,
                        nullptr, w - L(32));
        } else {
            dl->ChannelsSplit(2);
            dl->ChannelsSetCurrent(1);
            dl->PushClipRect(ImVec2(pos.x + L(16), pos.y), ImVec2(pos.x + w - L(16), pos.y + L(2000)), true);
            const ImVec2 size = DrawOverlay(dl, ImVec2(pos.x + L(16), pos.y + L(38)), base_scale * s.overlay_scale / 100.0f);
            dl->PopClipRect();
            content_h = size.y > 0.0f ? size.y : L(24);
            dl->ChannelsSetCurrent(0);
            dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + content_h + L(56)), ImGui::GetColorU32(Theme::White(0.04f)), L(10));
            if (size.y <= 0.0f) {
                dl->AddText(ImGui::GetFont(), L(19), ImVec2(pos.x + L(16), pos.y + L(38)), ImGui::GetColorU32(Theme::Text),
                            TL("(nothing selected)", "(ничего не выбрано)"));
            }
            dl->ChannelsMerge();
        }
        dl->AddText(ImGui::GetFont(), L(16), ImVec2(pos.x + L(16), pos.y + L(10)), ImGui::GetColorU32(Theme::Faint),
                    TL("Preview", "Предпросмотр"));
        ImGui::Dummy(ImVec2(0.0f, content_h + L(56)));
    }
    SectionTitle(TL("Position and size", "Положение и размер"));
    if (BeginRows("##appearance")) {
        SliderRow(TL("Size", "Размер"), s.overlay_scale, 50, 300, "%d%%", TL("Size of the overlay.", "Размер оверлея."));
        static const char* const places[] = {TL("Top left", "Слева сверху"), TL("Top right", "Справа сверху"),
                                             TL("Bottom left", "Слева снизу"), TL("Bottom right", "Справа снизу"),
                                             TL("Top centre", "Сверху по центру"), TL("Bottom centre", "Снизу по центру")};
        static_assert(std::size(places) == OverlayPositions);
        ChoiceRow(TL("Position", "Положение"), s.overlay_corner, places, nullptr, OverlayPositions, TL("Where the overlay sits.", "Где на экране оверлей."));
        SliderRow(TL("Distance from the side", "Отступ сбоку"), s.overlay_offset_x, 0, OverlayOffsetMaxX, "%d px",
                  TL("Pixels of a 1080-line screen from the left or right edge (not for the centre).",
                     "Пиксели экрана 1080p от левого или правого края (по центру не действует)."));
        SliderRow(TL("Distance from the top or bottom", "Отступ сверху или снизу"), s.overlay_offset_y, 0, OverlayOffsetMaxY, "%d px",
                  TL("Pixels of a 1080-line screen from the top or bottom edge.", "Пиксели экрана 1080p от верхнего или нижнего края."));
        SliderRow(TL("Background opacity", "Непрозрачность фона"), s.overlay_opacity, 0, 100, "%d%%", TL("0% is no background.", "0% — без фона."));
        if (s.overlay_style == OverlayCard) {
            static const char* const layouts[] = {TL("One line", "В одну строку"), TL("One item per line", "Каждый пункт с новой строки")};
            ChoiceRow(TL("Layout", "Расположение"), s.overlay_layout, layouts, nullptr, 2, TL("Items side by side or stacked.", "Пункты в ряд или столбиком."));
        } else if (s.overlay_style == OverlayMini) {
            static const char* const lines[] = {TL("One line", "Одна строка"), TL("Two lines and a graph", "Две строки и график")};
            ChoiceRow(TL("Lines", "Строки"), s.overlay_layout, lines, nullptr, 2, TL("The mini overlay's size.", "Размер мини-оверлея."));
        }
        EndRows();
    }
    SectionTitle(TL("What to show", "Что показывать"));
    static const char* const names[] = {"FPS", TL("Frametime", "Время кадра"), TL("Upscaler", "Апскейлер"),
                                        TL("GPU (name and use)", "GPU (название и загрузка)"), TL("CPU (game)", "CPU (игра)"),
                                        TL("RAM (game / total)", "RAM (игра / всего)"),
                                        TL("VRAM (used / available)", "VRAM (занято / доступно)"),
                                        TL("Frametime graph and statistics", "График времени кадра и статистика")};
    for (unsigned item = 0; item < P::NumItems; ++item) {
        BitRow(names[item], s.overlay_items, 1u << item);
    }
    ImGui::EndDisabled();
    SectionTitle(TL("About", "О программе"));
    if (BeginRows("##about")) {
        const auto& elf = Common::ElfInfo::Instance();
        char text[160];
        std::snprintf(text, sizeof(text), "%.*s v%.*s", int(elf.GameSerial().size()), elf.GameSerial().data(),
                      int(elf.AppVer().size()), elf.AppVer().data());
        InfoRow(TL("Game", "Игра"), text);
        const u32 total = background_total;
        if (total) {
            std::snprintf(text, sizeof(text), TL("loading %u%%", "загрузка %u%%"), u32(u64(background_done) * 100 / total));
        } else {
            std::snprintf(text, sizeof(text), "%s", TL("ready", "готов"));
        }
        InfoRow(TL("Shader cache", "Кэш шейдеров"), text);
        EndRows();
    }
}

// Startup options changed since this launch (applied by "Apply and restart").
int RestartChanges() {
    const auto& s = TlgSettings::Get();
    return (s.render_resolution != s.startup_render_resolution) + (s.engine_fps != s.startup_engine_fps) +
           (s.deferred_readback != s.startup_deferred_readback) + (s.stale_readback != s.startup_stale_readback);
}

// The Insert menu as the launcher's settings panel (qml/GameView.qml): a dimmed game behind a
// rounded glass panel, the sections with their icons on the left, a thin line, the section's rows
// on the right, the restart notice and the buttons under them.
bool SectionButton(const char* label, unsigned icon, bool current) {
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(ImGui::GetContentRegionAvail().x, L(64));
    const bool pressed = ImGui::InvisibleButton(label, size);
    const bool focused = ImGui::IsItemFocused(), hovered = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (current || focused || hovered) {
        dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                          ImGui::GetColorU32(Theme::White(current || focused ? 0.11f : 0.06f)), L(6));
    }
    const ImU32 color = ImGui::GetColorU32(current ? Theme::Text : Theme::White(0.7f));
    DrawIcon(dl, icon, ImVec2(pos.x + L(30), pos.y + size.y / 2), 24, color);
    const ImVec2 t = TextSize(LightFont(), 25, label);
    dl->AddText(LightFont(), L(25), ImVec2(pos.x + (font_icons ? L(60) : L(16)), pos.y + (size.y - t.y) / 2), color, label);
    ImGui::Dummy(ImVec2(0.0f, L(6) - ImGui::GetStyle().ItemSpacing.y));
    return pressed;
}

void Menu() {
    auto& s = TlgSettings::Get();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 size(std::min(L(1240), viewport->WorkSize.x * 0.94f), std::min(L(740), viewport->WorkSize.y * 0.92f));
    const ImVec2 at(viewport->WorkPos.x + (viewport->WorkSize.x - size.x) / 2,
                    viewport->WorkPos.y + (viewport->WorkSize.y - size.y) / 2);
    // The game dimmed behind the panel (no drawn shadow: stepped layers read as a halo).
    ImDrawList* back = ImGui::GetBackgroundDrawList();
    back->AddRectFilled(viewport->WorkPos, ImVec2(viewport->WorkPos.x + viewport->WorkSize.x, viewport->WorkPos.y + viewport->WorkSize.y),
                        Theme::U32(Theme::Ink, 0.42f));
    // The game frosted behind the panel, then the launcher's tint over it (more opaque without).
    const bool frosted = Frost(back, at, ImVec2(at.x + size.x, at.y + size.y), L(18));
    ImGui::SetNextWindowPos(at, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, L(18));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(18 / 255.0f, 18 / 255.0f, 22 / 255.0f, frosted ? 0.66f : 0.9f));
    ImGui::PushStyleColor(ImGuiCol_Border, Theme::White(0.07f));
    const bool shown = ImGui::Begin("##menu", nullptr,
                                    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
                                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
    if (!shown) {
        ImGui::End();
        return;
    }
    bool keep_open = true;
    footer_help = nullptr;
    constexpr int TabCount = 5;
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput) {
        const int step = ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_E, false) ? 1
                         : ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_Q, false)
                             ? -1
                             : 0;
        if (step) {
            requested_tab = (s.menu_tab + step + TabCount) % TabCount;
        }
    }
    bool reset_scroll = false; // a new section starts at its top
    if (requested_tab >= 0) {
        s.menu_tab = requested_tab;
        dirty = true;
        requested_tab = -1;
        editing_row = 0;
        reset_scroll = true;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ---- the sections: the game, its four sections, the keys --------------------------------
    const float side_w = L(236), side_x = L(24), top = L(28);
    ImGui::SetCursorPos(ImVec2(side_x, top));
    ImGui::BeginChild("##sections", ImVec2(side_w, size.y - top * 2), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoBackground);
    {
        static const std::string game = [] {
            const std::string_view title = Common::ElfInfo::Instance().Title();
            const char* app_name = std::getenv("TLG_APP_NAME");
            return std::string(!title.empty() ? title : app_name && *app_name ? app_name : "inFAMOUS First Light");
        }();
        ImDrawList* sdl = ImGui::GetWindowDrawList();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        sdl->AddRectFilled(ImVec2(p.x + L(8), p.y + L(4)), ImVec2(p.x + L(52), p.y + L(48)), ImGui::GetColorU32(Theme::White(0.08f)), L(8));
        DrawIcon(sdl, Icon::Bolt, ImVec2(p.x + L(30), p.y + L(26)), 22, ImGui::GetColorU32(Theme::Text));
        ImFont* bold = font_bold ? font_bold : ImGui::GetFont();
        const float text_x = p.x + L(66), text_w = side_w - L(70);
        ImGui::PushClipRect(ImVec2(text_x, p.y), ImVec2(text_x + text_w, p.y + L(56)), true);
        sdl->AddText(bold, L(20), ImVec2(text_x, p.y + L(2)), ImGui::GetColorU32(Theme::Text), game.c_str());
        sdl->AddText(ImGui::GetFont(), L(16), ImVec2(text_x, p.y + L(28)), ImGui::GetColorU32(Theme::Muted),
                     TL("PC port · menu", "ПК-порт · меню"));
        ImGui::PopClipRect();
        ImGui::Dummy(ImVec2(0.0f, L(74)));
        static const char* const tabs[TabCount] = {TL("Overlay", "Оверлей"), TL("Image", "Изображение"),
                                                   TL("Performance", "Производительность"), TL("Controls", "Управление"),
                                                   TL("Trophies", "Трофеи")};
        static const unsigned icons[TabCount] = {Icon::Chart, Icon::Display, Icon::Speed, Icon::Gamepad, Icon::Check};
        for (int tab = 0; tab < TabCount; ++tab) {
            if (SectionButton(tabs[tab], icons[tab], s.menu_tab == tab) && s.menu_tab != tab) {
                requested_tab = tab;
            }
        }
        const char* keys = TL(PadText("L1 / R1 or Q / E: sections\nInsert or R3+L2: close",
                                      "LB / RB or Q / E: sections\nInsert or RS+LT: close",
                                      "L / R or Q / E: sections\nInsert or RS+ZL: close"),
                              PadText("L1 / R1 или Q / E — разделы\nInsert или R3+L2 — закрыть",
                                      "LB / RB или Q / E — разделы\nInsert или RS+LT — закрыть",
                                      "L / R или Q / E — разделы\nInsert или RS+ZL — закрыть"));
        const ImVec2 k = TextSize(ImGui::GetFont(), 16, keys, side_w - L(16));
        const ImVec2 kp(ImGui::GetWindowPos().x + L(8), ImGui::GetWindowPos().y + ImGui::GetWindowHeight() - k.y - L(4));
        sdl->AddText(ImGui::GetFont(), L(16), kp, ImGui::GetColorU32(Theme::Faint), keys, nullptr, side_w - L(16));
    }
    ImGui::EndChild();
    const float line_x = at.x + side_x + side_w + L(22);
    dl->AddLine(ImVec2(line_x, at.y + top), ImVec2(line_x, at.y + size.y - top), ImGui::GetColorU32(Theme::Line));

    // ---- the section's rows, then the restart notice and the buttons -------------------------
    const float content_x = side_x + side_w + L(52) - L(20), content_y = L(22);
    const float footer_h = L(56) + L(28);
    const float content_w = size.x - content_x - L(40) + L(20);
    ImGui::SetCursorPos(ImVec2(content_x, content_y));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, L(6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(L(10), 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    if (reset_scroll) ImGui::SetNextWindowScroll(ImVec2(0.0f, 0.0f));
    ImGui::BeginChild("##rows", ImVec2(content_w, size.y - content_y - footer_h - L(10)), ImGuiChildFlags_NavFlattened,
                      ImGuiWindowFlags_NoBackground);
    switch (s.menu_tab) {
    case 1: ImageTab(); break;
    case 2: PerformanceTab(); break;
    case 3: ControlsTab(); break;
    case 4: TrophiesTab(); break;
    default: OverlayTab(); break;
    }
    ImGui::Dummy(ImVec2(0.0f, L(12)));
    // More rows below: they fade into the panel instead of being cut mid-line.
    const bool more_below = ImGui::GetScrollY() < ImGui::GetScrollMaxY() - 1.0f;
    const ImVec2 rows_min = ImGui::GetWindowPos(), rows_max(rows_min.x + ImGui::GetWindowWidth(), rows_min.y + ImGui::GetWindowHeight());
    ImGui::EndChild();
    if (more_below) {
        const ImU32 clear = ImGui::GetColorU32(ImVec4(18 / 255.0f, 18 / 255.0f, 22 / 255.0f, 0.0f));
        const ImU32 shade = ImGui::GetColorU32(ImVec4(18 / 255.0f, 18 / 255.0f, 22 / 255.0f, 0.92f));
        dl->AddRectFilledMultiColor(ImVec2(rows_min.x, rows_max.y - L(56)), rows_max, clear, clear, shade, shade);
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    const float footer_y = size.y - footer_h;
    ImGui::SetCursorPos(ImVec2(content_x + L(20), footer_y));
    if (const int changes = RestartChanges()) {
        char text[200];
        std::snprintf(text, sizeof(text), "%d %s", changes,
                      Plural(changes, "change needs a restart (unsaved progress is lost).",
                             "changes need a restart (unsaved progress is lost).",
                             "изменение требует перезапуска (несохранённый прогресс пропадёт).",
                             "изменения требуют перезапуска (несохранённый прогресс пропадёт).",
                             "изменений требуют перезапуска (несохранённый прогресс пропадёт)."));
        const float note_w = content_w - L(520);
        const ImVec2 t = TextSize(ImGui::GetFont(), 18, text, note_w);
        dl->AddText(ImGui::GetFont(), L(18), ImVec2(at.x + content_x + L(20), at.y + footer_y + (L(56) - t.y) / 2),
                    ImGui::GetColorU32(Theme::Warn), text, nullptr, note_w);
        ImGui::SetCursorPos(ImVec2(content_x + L(20) + note_w + L(16), footer_y));
        ImGui::BeginDisabled(!std::getenv("TLG_RESTART_COMMAND"));
        if (Pill(TL("Apply and restart", "Применить и перезапустить"), 56, 22, 200)) {
            // The new launch confirms the options once the game shows; run.py restores the
            // previous ones if it never does (restart_unconfirmed).
            s.restart_unconfirmed = true;
            TlgSettings::Save();
            runtime_restart();
        }
        ImGui::EndDisabled();
    }
    const char* close = TL("Close", "Закрыть");
    const float close_w = std::max(L(200), TextSize(ImGui::GetFont(), 22, close).x + L(56));
    ImGui::SetCursorPos(ImVec2(size.x - L(40) - close_w, footer_y));
    if (Pill(close, 56, 22, 200)) {
        keep_open = false;
    }
    ImGui::End();
    if (!keep_open) {
        SetOpen(false);
    }
}

// Enemy health bar from the damage hook's records (tlg_actors.cpp).
// Only the copied values are read, never the game's memory: an actor may be gone by now.
void HealthDisplay() {
    struct Seen {
        uint32_t seq = 0;
        std::chrono::steady_clock::time_point at{};
    };
    static Seen seen[2];
    const auto& s = TlgSettings::Get();
    if (!s.actor_watch_supported.load()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    TlgActorSlot slot[2];
    for (int i = 0; i < 2; ++i) {
        std::memcpy(&slot[i], const_cast<const TlgActorSlot*>(&tlg_actor_slots[i]), sizeof(slot[i]));
        if (slot[i].seq != seen[i].seq) {
            seen[i] = {slot[i].seq, now};
        }
    }
    const auto recent = [&](int i, int seconds) {
        return seen[i].seq != 0 && now - seen[i].at < std::chrono::seconds(seconds);
    };
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw = ImGui::GetForegroundDrawList();
    const float scale = base_scale;
    if (s.enemy_health_bar && recent(1, 4) && TlgActors::Valid(slot[1].health, slot[1].max_health)) {
        // Minimal: a thin green bar centered on the screen's top edge; it shrinks toward the center
        // so it stays symmetric, with the health in numbers below.
        const ImVec2 display = ImGui::GetIO().DisplaySize;
        const float width = 260.0f * scale, height = 5.0f * scale, center = display.x * 0.5f;
        const float top = 36.0f * scale;
        const float fraction = TlgActors::Fraction(slot[1].health, slot[1].max_health);
        draw->AddRectFilled(ImVec2(center - width * 0.5f - scale, top - scale),
                            ImVec2(center + width * 0.5f + scale, top + height + scale),
                            IM_COL32(0, 0, 0, 140), 2 * scale);
        const float half = width * 0.5f * fraction;
        draw->AddRectFilled(ImVec2(center - half, top), ImVec2(center + half, top + height),
                            IM_COL32(70, 190, 80, 220), 2 * scale);
        char text[48];
        std::snprintf(text, sizeof(text), "%.0f / %.0f", std::ceil(slot[1].health), slot[1].max_health);
        const float font = ImGui::GetFontSize() * 0.9f;
        const ImVec2 size = ImGui::GetFont()->CalcTextSizeA(font, FLT_MAX, 0.0f, text);
        const ImVec2 at{center - size.x * 0.5f, top + height + 4.0f * scale};
        draw->AddText(ImGui::GetFont(), font, ImVec2(at.x + scale, at.y + scale), IM_COL32(0, 0, 0, 160), text);
        draw->AddText(ImGui::GetFont(), font, at, IM_COL32(225, 235, 220, 220), text);
    }
}

// The game's text dialog: what is typed, and how to finish (keyboard or controller).

// ---- The dialogs over the game (saves, messages, the graffiti guide, text entry, notices):
// drawn as the launcher's panels (the menus' Canvas mock-up). The layout is in launcher pixels
// of a 1920x1080 stage, times k: the launcher's radii (panel 18, card 10, picture 6), its glass,
// Segoe UI Light titles, white and frosted pills. ----
constexpr float kPanelRadius = 18.0f, kItemRadius = 10.0f, kPictureRadius = 6.0f;

struct SavePainter {
    ImDrawList* dl;
    ImFont* font;
    float k;
    ImVec2 at; // the panel's top-left corner, in display pixels

    ImVec2 P(float x, float y) const { return ImVec2(at.x + x * k, at.y + y * k); }
    ImFont* Face(bool bold) const { return bold && font_bold ? font_bold : font; }
    float Width(float size, const std::string& text, bool bold = false) const {
        return Face(bold)->CalcTextSizeA(size * k, FLT_MAX, 0.0f, text.c_str()).x / k;
    }
    float LightWidth(float size, const std::string& text) const {
        return (font_light ? font_light : font)->CalcTextSizeA(size * k, FLT_MAX, 0.0f, text.c_str()).x / k;
    }
    // The height of `text` wrapped at `width` stage pixels.
    float Height(float size, const std::string& text, float width) const {
        return font->CalcTextSizeA(size * k, FLT_MAX, width * k, text.c_str()).y / k;
    }
    // The text cut with "..." to fit `width` stage pixels.
    std::string Fit(float size, const std::string& text, float width, bool bold = false) const {
        if (Width(size, text, bold) <= width) return text;
        std::string cut = text;
        while (!cut.empty()) {
            while (!cut.empty() && (u8(cut.back()) & 0xc0) == 0x80) cut.pop_back(); // UTF-8 tail
            if (!cut.empty()) cut.pop_back();
            if (Width(size, cut + "...", bold) <= width) return cut + "...";
        }
        return "...";
    }
    void Text(float x, float y, float size, const ImVec4& color, const std::string& text, bool bold = false,
              float alpha = 1.0f, float wrap = 0.0f) const {
        const ImU32 col = Theme::U32(color, alpha);
        dl->AddText(Face(bold), size * k, P(x, y), col, text.c_str(), nullptr, wrap * k);
        if (bold && !font_bold) dl->AddText(font, size * k, P(x + 0.55f, y), col, text.c_str(), nullptr, wrap * k);
    }
    // Segoe UI Light: the launcher's titles and labels.
    void TextLight(float x, float y, float size, const ImVec4& color, const std::string& text, float alpha = 1.0f) const {
        dl->AddText(font_light ? font_light : font, size * k, P(x, y), Theme::U32(color, alpha), text.c_str());
    }
    void TextRight(float right, float y, float size, const ImVec4& color, const std::string& text,
                   bool bold = false) const {
        Text(right - Width(size, text, bold), y, size, color, text, bold);
    }
    void Box(float x0, float y0, float x1, float y1, const ImVec4& fill, float rounding, float alpha = 1.0f) const {
        dl->AddRectFilled(P(x0, y0), P(x1, y1), Theme::U32(fill, alpha), rounding * k);
    }
    void Frame(float x0, float y0, float x1, float y1, const ImVec4& color, float rounding, float thickness,
               float alpha = 1.0f) const {
        dl->AddRect(P(x0, y0), P(x1, y1), Theme::U32(color, alpha), rounding * k, 0, thickness * k);
    }
    void Line(float x0, float y0, float x1, float y1, const ImVec4& color, float thickness, float alpha = 1.0f) const {
        dl->AddLine(P(x0, y0), P(x1, y1), Theme::U32(color, alpha), thickness * k);
    }
    // An icon of the icon font centred at (x, y); `fallback` draws it without the font.
    void Glyph(unsigned code, float x, float y, float size, const ImVec4& color, float alpha = 1.0f) const {
        if (!font_icons) return;
        const std::string text = IconText(code);
        size *= IconScale();
        const ImVec2 s = font_icons->CalcTextSizeA(size * k, FLT_MAX, 0.0f, text.c_str());
        const ImVec2 c = P(x, y);
        dl->AddText(font_icons, size * k, ImVec2(c.x - s.x / 2, c.y - s.y / 2), Theme::U32(color, alpha), text.c_str());
    }
    // The launcher's focus ring: white, 2 pixels, a 2-pixel gap outside the item.
    void Ring(float x0, float y0, float x1, float y1, float rounding) const {
        Frame(x0 - 3, y0 - 3, x1 + 3, y1 + 3, Theme::White(0.95f), rounding + 3, 2.0f);
    }
    // A controller button: 0 confirm (bottom face button), 1 back (right one), centred at (x, y):
    // Cross / Circle, or the letters of TLG_GLYPHS=xbox (A / B) and nintendo (B / A). `light`: on a
    // white pill.
    void Button(float x, float y, int which, bool light = false) const {
        const ImVec2 c = P(x, y);
        const ImVec4 ground = light ? Theme::Ink : Theme::Text;
        dl->AddCircleFilled(c, 15.0f * k, Theme::U32(ground, 0.08f));
        dl->AddCircle(c, 15.0f * k, Theme::U32(ground, 0.3f), 0, 1.5f * k);
        const ImVec4& mark = which == 0 ? Theme::Accent : Theme::Bad;
        if (const char* letter = PadText(nullptr, which == 0 ? "A" : "B", which == 0 ? "B" : "A")) {
            Text(x - Width(18.0f, letter, true) / 2, y - 12.5f, 18.0f, which == 0 ? Theme::Good : Theme::Bad, letter, true);
            return;
        }
        if (which == 0) {
            Line(x - 6, y - 6, x + 6, y + 6, mark, 2.6f);
            Line(x + 6, y - 6, x - 6, y + 6, mark, 2.6f);
        } else {
            dl->AddCircle(c, 7.0f * k, Theme::U32(mark), 0, 2.6f * k);
        }
    }
    // The panel: the launcher's glass and a faint border; the credit under its corner.
    void Panel(float w, float h) const {
        const bool frosted = Frost(dl, P(0, 0), P(w, h), kPanelRadius * k);
        Box(0, 0, w, h, ImVec4(18 / 255.0f, 18 / 255.0f, 22 / 255.0f, 1.0f), kPanelRadius, frosted ? 0.66f : 0.93f);
        Frame(0, 0, w, h, Theme::White(0.07f), kPanelRadius, 1.0f);
        const std::string credit = "@hayatog-dev";
        Text(w - 8.0f - Width(15.0f, credit), h + 14.0f, 15.0f, Theme::Faint, credit, false, 0.7f);
    }
    // A dialog's header: the launcher's 64-pixel tile with an icon, the title (36 Light) and a
    // line under it (19, muted). `icon` 0: the tile is drawn by the caller (`draw_tile`).
    void Header(float x, float y, unsigned icon, const std::string& title, const std::string& sub,
                const ImVec4& icon_color = Theme::Text) const {
        Box(x, y, x + 64, y + 64, Theme::White(0.08f), kItemRadius);
        if (icon) Glyph(icon, x + 32, y + 32, 30, icon_color);
        TextLight(x + 84, y + (sub.empty() ? 10 : -2), 36.0f, Theme::Text, title);
        if (!sub.empty()) Text(x + 84, y + 44, 19.0f, Theme::Muted, sub);
    }
    // A save's picture (16:9) or a placeholder; `is_new` draws the dashed "+" slot.
    void Picture(float x, float y, float w, float h, ImTextureData* tex, bool is_new) const {
        if (tex) {
            dl->AddImageRounded(tex->GetTexRef(), P(x, y), P(x + w, y + h), ImVec2(0, 0), ImVec2(1, 1),
                                IM_COL32_WHITE, kPictureRadius * k);
            return;
        }
        if (is_new) {
            for (float t = 0; t < 1.0f; t += 0.08f) { // dashes around the slot
                Line(x + w * t, y, x + w * (t + 0.04f), y, Theme::White(0.28f), 2.0f);
                Line(x + w * t, y + h, x + w * (t + 0.04f), y + h, Theme::White(0.28f), 2.0f);
            }
            for (float t = 0; t < 1.0f; t += 0.14f) {
                Line(x, y + h * t, x, y + h * (t + 0.07f), Theme::White(0.28f), 2.0f);
                Line(x + w, y + h * t, x + w, y + h * (t + 0.07f), Theme::White(0.28f), 2.0f);
            }
            Line(x + w / 2, y + h / 2 - 15, x + w / 2, y + h / 2 + 15, Theme::Muted, 2.4f);
            Line(x + w / 2 - 15, y + h / 2, x + w / 2 + 15, y + h / 2, Theme::Muted, 2.4f);
            return;
        }
        Box(x, y, x + w, y + h, Theme::White(0.06f), kPictureRadius);
    }
    // The width of a pill for `label` (with its controller button when `glyph` >= 0).
    float PillWidth(const char* label, int, float min_w = 160) const {
        return std::max(min_w, Width(22.0f, label, true) + 60.0f);
    }
    // The launcher's pill (PillButton.qml): white with black text when focused, frosted
    // otherwise, 60 pixels high. (`glyph`, the controller button, is told by the key hints.)
    void Pill(float x0, float y0, float x1, const char* label, int, bool focused) const {
        Box(x0, y0, x1, y0 + 60, focused ? Theme::Text : Theme::White(0.14f), 30.0f);
        const float text_w = Width(22.0f, label, true);
        Text(x0 + (x1 - x0 - text_w) / 2, y0 + 30 - 15, 22.0f, focused ? Theme::Ink : Theme::Text, label, true);
    }
    // Kept for the callers that size their buttons by corners (46 high in the old layout).
    void ActionButton(float x0, float y0, float x1, float, const char* label, int glyph, bool focused) const {
        Pill(x0, y0, x1, label, glyph, focused);
    }
};

// A painter for a w x h panel centred on the display (kept inside it on narrow windows).
SavePainter SavePainterFor(float w, float h) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k = std::min({viewport->WorkSize.y / 1080.0f, viewport->WorkSize.x * 0.94f / w,
                              viewport->WorkSize.y * 0.94f / (h + 40.0f)});
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    p.at = ImVec2(viewport->WorkPos.x + (viewport->WorkSize.x - w * k) * 0.5f,
                  viewport->WorkPos.y + (viewport->WorkSize.y - h * k) * 0.5f);
    return p;
}

void SaveBackdrop(float alpha) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::GetForegroundDrawList()->AddRectFilled(
        viewport->WorkPos, ImVec2(viewport->WorkPos.x + viewport->WorkSize.x, viewport->WorkPos.y + viewport->WorkSize.y),
        Theme::U32(Theme::Ink, alpha));
}

std::string SaveWhen(s64 mtime) {
    if (mtime <= 0) return {};
    const time_t t = time_t(mtime);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    // As the launcher's Continue card (saves.py when): today / yesterday at the time, else the
    // day and month (and the year when it is not this one).
    std::tm now{};
    const time_t now_t = std::time(nullptr);
#ifdef _WIN32
    localtime_s(&now, &now_t);
#else
    localtime_r(&now_t, &now);
#endif
    std::tm yesterday = now;
    yesterday.tm_mday -= 1;
    std::mktime(&yesterday);
    const auto same_day = [](const std::tm& a, const std::tm& b) {
        return a.tm_year == b.tm_year && a.tm_yday == b.tm_yday;
    };
    static const char* const ru_months[] = {"января", "февраля", "марта", "апреля", "мая", "июня",
                                            "июля", "августа", "сентября", "октября", "ноября", "декабря"};
    static const char* const en_months[] = {"January", "February", "March", "April", "May", "June",
                                            "July", "August", "September", "October", "November", "December"};
    char clock[16], out[80];
    std::snprintf(clock, sizeof(clock), "%02d:%02d", tm.tm_hour, tm.tm_min);
    const bool ru = UiRussian();
    if (same_day(tm, now)) {
        std::snprintf(out, sizeof(out), ru ? "сегодня в %s" : "today at %s", clock);
    } else if (same_day(tm, yesterday)) {
        std::snprintf(out, sizeof(out), ru ? "вчера в %s" : "yesterday at %s", clock);
    } else if (tm.tm_year == now.tm_year) {
        if (ru) std::snprintf(out, sizeof(out), "%d %s, %s", tm.tm_mday, ru_months[tm.tm_mon % 12], clock);
        else std::snprintf(out, sizeof(out), "%s %d, %s", en_months[tm.tm_mon % 12], tm.tm_mday, clock);
    } else {
        if (ru) std::snprintf(out, sizeof(out), "%d %s %d, %s", tm.tm_mday, ru_months[tm.tm_mon % 12], tm.tm_year + 1900, clock);
        else std::snprintf(out, sizeof(out), "%s %d %d, %s", en_months[tm.tm_mon % 12], tm.tm_mday, tm.tm_year + 1900, clock);
    }
    return out;
}

std::string SaveAgo(s64 mtime) {
    const s64 seconds = s64(std::time(nullptr)) - mtime;
    if (mtime <= 0 || seconds < 0) return {};
    char out[64];
    if (seconds < 60) return TL("just now", "только что");
    if (seconds < 3600) {
        const int m = int(seconds / 60);
        std::snprintf(out, sizeof(out), "%d %s", m,
                      Plural(m, "minute ago", "minutes ago", "минуту назад", "минуты назад", "минут назад"));
    } else if (seconds < 86400) {
        const int h = int(seconds / 3600);
        std::snprintf(out, sizeof(out), "%d %s", h,
                      Plural(h, "hour ago", "hours ago", "час назад", "часа назад", "часов назад"));
    } else {
        const int d = int(seconds / 86400);
        if (d == 1) return TL("yesterday", "вчера");
        std::snprintf(out, sizeof(out), "%d %s", d,
                      Plural(d, "day ago", "days ago", "день назад", "дня назад", "дней назад"));
    }
    return out;
}

std::string SaveSize(u64 bytes) {
    if (!bytes) return {};
    char out[32];
    if (bytes < 1024 * 1024) {
        std::snprintf(out, sizeof(out), "%llu %s", (unsigned long long)((bytes + 1023) / 1024), TL("KB", "КБ"));
    } else {
        std::snprintf(out, sizeof(out), "%.1f %s", double(bytes) / (1024.0 * 1024.0), TL("MB", "МБ"));
        if (UiRussian()) {
            for (char* c = out; *c; ++c) if (*c == '.') *c = ',';
        }
    }
    return out;
}

const char* SaveTitle(int kind) {
    return kind == 2 ? TL("Load game", "Загрузка игры") : kind == 3 ? TL("Delete save", "Удаление сохранения")
                                                         : TL("Save game", "Сохранение игры");
}

ImTextureData* SaveTexture(size_t i) {
    return i < choice_textures.size() ? choice_textures[i] : nullptr;
}

// One save as the launcher's card: picture 192 x 108, the save's main line (26) and the line
// under it (19, muted), when it was saved on the right. The game titles every save with its own
// name, so the subtitle (the mission and its percent) is the main line when there is one.
constexpr float kCardH = 136.0f;
void SaveCard(const SavePainter& p, float x0, float y0, float x1, const SaveRow& row, ImTextureData* tex,
              bool selected) {
    const float y1 = y0 + kCardH, mid = (y0 + y1) / 2;
    p.Box(x0, y0, x1, y1, Theme::White(selected ? 0.13f : 0.04f), kItemRadius);
    if (selected) p.Ring(x0, y0, x1, y1, kItemRadius);
    p.Picture(x0 + 18, y0 + 14, 192, 108, tex, row.is_new);
    const float text_x = x0 + 18 + 192 + 24;
    if (row.is_new) {
        p.Text(text_x, mid - 32, 26.0f, Theme::Text, TL("New save", "Новое сохранение"));
        p.Text(text_x, mid + 6, 19.0f, Theme::Muted, TL("Create a new save", "Создать новое сохранение"));
        return;
    }
    std::string ago = SaveAgo(row.mtime);
    const std::string when = SaveWhen(row.mtime), size = SaveSize(row.bytes);
    if (!size.empty()) ago = ago.empty() ? size : ago + " · " + size;
    const float right = x1 - 22;
    const float right_w = std::max(p.Width(21.0f, when), p.Width(17.0f, ago));
    const float text_w = right - text_x - (right_w > 0 ? right_w + 24 : 0);
    const std::string main = !row.subtitle.empty() ? row.subtitle : row.title;
    std::string line = !row.subtitle.empty() ? row.title : std::string();
    if (!row.detail.empty()) line = line.empty() ? row.detail : row.detail + " · " + line;
    for (char& c : line) if (c == '\n' || c == '\r') c = ' ';
    p.Text(text_x, mid - 32, 26.0f, Theme::Text, p.Fit(26.0f, main, text_w));
    p.Text(text_x, mid + 6, 19.0f, Theme::Muted, p.Fit(19.0f, line, text_w));
    p.TextRight(right, mid - 30, 21.0f, Theme::Text, when);
    p.TextRight(right, mid + 6, 17.0f, Theme::Faint, ago);
}

void SaveListBox() {
    const int count = int(save_rows.size());
    const int shown = std::min(count, 4);
    const float w = 1000, gap = 12, header_h = 124, footer_h = 108, pad = 24;
    const float list_h = shown * kCardH + std::max(shown - 1, 0) * gap + pad * 2;
    const float h = header_h + list_h + footer_h;
    SaveBackdrop(0.55f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    // Header: the launcher's tile with a floppy, the title, the game and the number of saves.
    const int saves = count - int(std::count_if(save_rows.begin(), save_rows.end(), [](const SaveRow& r) { return r.is_new; }));
    char sub[200];
    static const std::string game = [] {
        const std::string_view title = Common::ElfInfo::Instance().Title();
        const char* app_name = std::getenv("TLG_APP_NAME");
        return std::string(!title.empty() ? title : app_name && *app_name ? app_name : "");
    }();
    std::snprintf(sub, sizeof(sub), "%s%s%d %s", game.c_str(), game.empty() ? "" : " · ", saves,
                  Plural(saves, "save", "saves", "сохранение", "сохранения", "сохранений"));
    p.Header(40, 30, Icon::Save, SaveTitle(save_kind), sub);
    p.Line(40, header_h, w - 40, header_h, Theme::Line, 1.0f);
    // The cards, scrolled so the selected one is shown.
    if (choice_index < choice_first) choice_first = choice_index;
    if (choice_index >= choice_first + shown) choice_first = choice_index - shown + 1;
    choice_first = std::clamp(choice_first, 0, std::max(count - shown, 0));
    save_row_rects.assign(count, ImRect());
    for (int i = choice_first; i < choice_first + shown; ++i) {
        const float y = header_h + pad + (i - choice_first) * (kCardH + gap);
        SaveCard(p, 28, y, w - 28, save_rows[i], SaveTexture(size_t(i)), i == choice_index);
        save_row_rects[i] = ImRect(p.P(28, y), p.P(w - 28, y + kCardH));
    }
    if (count > shown) { // scroll bar
        const float track = list_h - pad * 2, y0 = header_h + pad;
        const float knob = track * shown / count, top = y0 + track * choice_first / count;
        p.Box(w - 16, y0, w - 12, y0 + track, Theme::Line, 2.0f);
        p.Box(w - 16, top, w - 12, top + knob, Theme::Faint, 2.0f);
    }
    // Footer: the pills (the action white, Back frosted) and what the keys do.
    const float fy = header_h + list_h;
    p.Line(40, fy, w - 40, fy, Theme::Line, 1.0f);
    const char* confirm = save_kind == 2 ? TL("Load", "Загрузить") : save_kind == 3 ? TL("Delete", "Удалить")
                                                                                  : TL("Save here", "Сохранить сюда");
    const char* back = TL("Back", "Назад");
    float x = 40;
    const float by = fy + 22;
    const float cw = p.PillWidth(confirm, 0, 260);
    p.Pill(x, by, x + cw, confirm, 0, true);
    x += cw + 14;
    const float bw = p.PillWidth(back, 1, 160);
    p.Pill(x, by, x + bw, back, 1, false);
    x += bw + 28;
    const std::string hint = TL(PadText("Cross / Enter: choose · Circle / Esc: back", "A / Enter: choose · B / Esc: back",
                                        "B / Enter: choose · A / Esc: back"),
                                PadText("Крестик / Enter — выбрать · Кружок / Esc — назад",
                                        "A / Enter — выбрать · B / Esc — назад", "B / Enter — выбрать · A / Esc — назад"));
    p.TextRight(w - 40, by + 19, 17.0f, Theme::Faint, p.Fit(17.0f, hint, w - 40 - x));
}

// The message's words for a system message type (sceSaveDataDialog), the game's own text or an
// error code. warn: drawn with the warning sign.
struct SaveMessageWords {
    std::string title, body;
    int buttons; // 0 OK, 1 yes/no, 3 OK/cancel
    bool warn;
};

SaveMessageWords SaveMessageFor(int kind, int code, u64 value, const std::string& text) {
    switch (code) {
    case 1:
        return {TL("No saved data", "Нет сохранений"),
                TL("There is no saved data to load.", "Загружать нечего: сохранений нет."), 0, false};
    case 2:
        if (kind == 2) return {TL("Load this save?", "Загрузить это сохранение?"),
                               TL("Progress that was not saved will be lost.", "Несохранённый прогресс будет потерян."), 1, false};
        if (kind == 3) return {TL("Delete this save?", "Удалить это сохранение?"),
                               TL("This cannot be undone.", "Это действие нельзя отменить."), 1, true};
        return {TL("Save the game?", "Сохранить игру?"), TL("A new save will be created.", "Будет создано новое сохранение."), 1, false};
    case 3:
        return {TL("Overwrite this save?", "Перезаписать это сохранение?"),
                TL("The current progress will replace this save. This cannot be undone.",
                   "Текущий прогресс заменит это сохранение. Это действие нельзя отменить."), 1, true};
    case 4:
    case 8:
        return {TL("Not enough space", "Недостаточно места"),
                TL("Free some disk space and try again.", "Освободите место на диске и попробуйте снова."), 0, true};
    case 6:
        return {TL("This save is damaged", "Сохранение повреждено"),
                TL("It cannot be used.", "Его нельзя использовать."), 0, true};
    case 10:
        return {TL("A damaged save was found", "Найдено повреждённое сохранение"),
                TL("It was deleted.", "Оно удалено."), 0, true};
    case 11:
        return {TL("A damaged save was found", "Найдено повреждённое сохранение"),
                TL("A new save was created in its place.", "Вместо него создано новое."), 0, true};
    case 13:
        return {TL("A damaged save was found", "Найдено повреждённое сохранение"),
                TL("It was restored.", "Оно восстановлено."), 0, true};
    case 14:
        return {TL("The saves are too large", "Сохранения слишком большие"),
                TL("Delete some saves and try again.", "Удалите несколько сохранений и попробуйте снова."), 0, true};
    case 100:
        return {SaveTitle(kind), text, 0, false};
    case 101: {
        char body[96];
        std::snprintf(body, sizeof(body), TL("Error code: 0x%08X", "Код ошибки: 0x%08X"), unsigned(value));
        return {TL("Save data error", "Ошибка данных сохранения"), body, 0, true};
    }
    default:
        return {SaveTitle(kind), text, 0, false};
    }
}

void SaveMessageBox() {
    const SaveMessageWords words = SaveMessageFor(save_kind, message_code, message_value, message_text);
    const int buttons = message_buttons >= 0 ? message_buttons : words.buttons;
    const float w = 640, pad_x = 36, pad_y = 34, inner = w - pad_x * 2;
    SavePainter p = SavePainterFor(w, 400); // k first: the text heights depend on it
    const float title_x = words.warn ? 44.0f : 0.0f;
    const float body_h = words.body.empty() ? 0.0f : p.Height(20.0f, words.body, inner);
    const float card_h = message_has_entry ? 116.0f : 0.0f;
    const float h = pad_y + 40 + (card_h ? 18 + card_h : 0) + (body_h ? 14 + body_h : 0) + 28 + 60 + 30;
    SaveBackdrop(0.55f);
    p = SavePainterFor(w, h);
    p.Panel(w, h);

    float y = pad_y;
    if (words.warn) p.Glyph(Icon::Warning, pad_x + 16, y + 20, 28, Theme::Warn); // a warning sign
    p.Text(pad_x + title_x, y, 30.0f, Theme::Text, p.Fit(30.0f, words.title, inner - title_x));
    y += 40;
    if (message_has_entry) {
        y += 18;
        p.Box(pad_x, y, w - pad_x, y + card_h, Theme::White(0.05f), kItemRadius);
        p.Picture(pad_x + 14, y + 14, 156, 88, SaveTexture(0), message_entry.is_new);
        const float tx = pad_x + 14 + 156 + 18, tw = w - pad_x - 16 - tx;
        const std::string main = message_entry.is_new ? std::string(TL("New save", "Новое сохранение"))
                                 : !message_entry.subtitle.empty() ? message_entry.subtitle : message_entry.title;
        std::string when = SaveWhen(message_entry.mtime);
        const std::string ago = SaveAgo(message_entry.mtime);
        if (!when.empty() && !ago.empty()) when += " · " + ago;
        p.Text(tx, y + 20, 23.0f, Theme::Text, p.Fit(23.0f, main, tw));
        p.Text(tx, y + 58, 18.0f, Theme::Muted, p.Fit(18.0f, when, tw));
        y += card_h;
    }
    if (body_h) {
        y += 14;
        p.Text(pad_x, y, 20.0f, Theme::Muted, words.body, false, 1.0f, inner);
        y += body_h;
    }
    y += 28;
    // The pills, left to right: [yes/OK] [no/cancel]; the focused one is white.
    const char* yes = buttons == 1 ? TL("Yes", "Да") : "OK";
    const char* no = buttons == 1 ? TL("No", "Нет") : TL("Cancel", "Отмена");
    const bool two = buttons == 1 || buttons == 3;
    if (!two) message_focus = 1;
    save_button_rects[0] = save_button_rects[1] = ImRect();
    float x = pad_x;
    for (int b = 1; b >= (two ? 0 : 1); --b) {
        const char* label = b == 1 ? yes : no;
        const float bw = p.PillWidth(label, b == 1 ? 0 : 1, 160);
        p.Pill(x, y, x + bw, label, b == 1 ? 0 : 1, b == message_focus);
        save_button_rects[b] = ImRect(p.P(x, y), p.P(x + bw, y + 60));
        x += bw + 14;
    }
}

void SaveProgressBox(SaveClock::time_point now) {
    const float w = 760, h = 226;
    SaveBackdrop(0.4f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    const char* title = progress_kind == 2 ? TL("Loading...", "Загрузка...")
                        : progress_kind == 3 ? TL("Deleting...", "Удаление...")
                                             : TL("Saving...", "Сохранение...");
    p.Header(40, 34, 0, title, TL("Do not close the game.", "Не закрывайте игру."));
    // A spinning arc in the tile.
    const float t = std::chrono::duration<float>(now - progress_since).count();
    const ImVec2 c = p.P(40 + 32, 34 + 32);
    p.dl->AddCircle(c, 15.0f * p.k, Theme::U32(Theme::Line), 0, 3.0f * p.k);
    p.dl->PathArcTo(c, 15.0f * p.k, t * 6.0f, t * 6.0f + 4.4f, 24);
    p.dl->PathStroke(Theme::U32(Theme::Accent), 0, 3.0f * p.k);
    const float bx0 = 40, bx1 = w - 40, by = 140;
    p.Box(bx0, by, bx1, by + 8, Theme::White(0.1f), 4.0f);
    if (progress_percent >= 0) {
        const float f = std::clamp(progress_percent, 0, 100) / 100.0f;
        if (f > 0) p.Box(bx0, by, bx0 + (bx1 - bx0) * f, by + 8, Theme::Accent, 4.0f);
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%d%%", std::clamp(progress_percent, 0, 100));
        p.TextRight(bx1, by + 24, 19.0f, Theme::Text, pct, true);
    } else { // no figure: a segment sweeps the bar
        const float span = (bx1 - bx0) * 0.3f, phase = std::fmod(t * 0.8f, 1.0f);
        const float s0 = bx0 + (bx1 - bx0 + span) * phase - span;
        p.Box(std::max(s0, bx0), by, std::min(s0 + span, bx1), by + 8, Theme::Accent, 4.0f);
    }
    if (!progress_text.empty()) {
        std::string text = progress_text;
        for (char& ch : text) if (ch == '\n' || ch == '\r') ch = ' ';
        p.Text(bx0, by + 24, 19.0f, Theme::Muted, p.Fit(19.0f, text, bx1 - bx0 - 90));
    }
}

// "Game saved" as the launcher's toast (qml/Toast.qml): a card at the bottom right, a tile with
// the green check, the words and the time.
void SaveToastBox(SaveClock::time_point now) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const char* text = toast_kind == 2 ? TL("Save loaded", "Сохранение загружено")
                       : toast_kind == 3 ? TL("Save deleted", "Сохранение удалено")
                                         : TL("Game saved", "Игра сохранена");
    const float in = std::chrono::duration<float>(now - toast_since).count();
    const float out = std::chrono::duration<float>(toast_until - now).count();
    const float alpha = std::clamp(std::min(in / 0.15f, out / 0.4f), 0.0f, 1.0f);
    const float slide = (1.0f - std::clamp(in / 0.35f, 0.0f, 1.0f)) * 60.0f; // slides in from the right
    const float k = viewport->WorkSize.y / 1080.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    char clock[16] = "";
    const std::time_t t = std::time(nullptr);
    if (const std::tm* local = std::localtime(&t)) std::strftime(clock, sizeof(clock), "%H:%M", local);
    const float w = std::max(560.0f, 22 + 68 + 20 + std::max(p.Width(23.0f, text, true), p.Width(20.0f, clock)) + 28), h = 104;
    p.at = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - (40 + w - slide) * k,
                  viewport->WorkPos.y + viewport->WorkSize.y - (40 + h) * k);
    const bool frosted = alpha > 0.98f && Frost(p.dl, p.P(0, 0), p.P(w, h), 16.0f * k);
    p.Box(0, 0, w, h, ImVec4(32 / 255.0f, 32 / 255.0f, 35 / 255.0f, 1.0f), 16.0f, (frosted ? 0.72f : 0.95f) * alpha);
    p.Box(22, 18, 90, 86, Theme::White(0.08f), kItemRadius, alpha);
    if (font_icons) {
        p.Glyph(Icon::Check, 56, 52, 32, Theme::Good, alpha);
    } else {
        p.Line(42, 52, 52, 62, Theme::Good, 3.0f, alpha);
        p.Line(52, 62, 71, 41, Theme::Good, 3.0f, alpha);
    }
    p.Text(110, 22, 23.0f, Theme::Text, text, true, alpha);
    p.Text(110, 56, 20.0f, Theme::Muted, clock, false, alpha);
}

// ---- The graffiti guide: how to paint without a gyro (sensor mode). Shown in the middle of the
// screen the first time sensor mode is used, closed with its button; the menu's Controls tab
// opens it again. imgui_mutex guards it. ----
std::atomic<bool> guide_active{false};
ImRect guide_button_rect;

void GuideBox() {
    const bool ru = UiRussian();
    const char* sticks = PadText("L3+R3", "LS+RS", "LS+RS");
    const char* shake = PadText(ru ? "✕ (крестик)" : "✕ (cross)", "A", "B");
    const char* spray = PadText("R2", "RT", "ZR");
    char steps[4][320];
    std::snprintf(steps[0], sizeof(steps[0]), TL("Tap %s together: sensor mode turns on and the can is held sideways.",
                                                 "Нажмите %s одновременно: включится режим сенсора, и баллончик повернётся набок."),
                  sticks);
    std::snprintf(steps[1], sizeof(steps[1]), TL("Hold %s to shake the can.", "Удерживайте %s, чтобы встряхнуть баллончик."), shake);
    std::snprintf(steps[2], sizeof(steps[2]), TL("Hold %s to spray and aim with the left stick (a small push is slow).",
                                                 "Удерживайте %s, чтобы распылять краску, и цельтесь левым стиком (лёгкий наклон — медленнее)."),
                  spray);
    std::snprintf(steps[3], sizeof(steps[3]), TL("Tap %s again to walk again.", "Снова нажмите %s, чтобы вернуться к ходьбе."), sticks);
    const char* keys = TL("Keyboard: hold T or G (can sideways), Y shakes, F sprays, the arrow keys aim.",
                          "Клавиатура: удерживайте T или G (баллончик набок), Y — встряхнуть, F — распылять, стрелки — прицел.");
    const char* again = TL("The Insert menu, Controls section, shows this again.",
                           "Это окно снова открывается в меню Insert, раздел «Управление».");
    const char* ok = TL("Got it", "Понятно");
    const float w = 1000, pad = 40, text_x = pad + 64, text_w = w - pad - text_x;
    SavePainter p = SavePainterFor(w, 700); // k first: the text heights depend on it
    float step_h[4], h = 30 + 64 + 30 + 1 + 28;
    for (int i = 0; i < 4; ++i) {
        step_h[i] = std::max(44.0f, p.Height(22.0f, steps[i], text_w));
        h += step_h[i] + 16;
    }
    const float keys_h = p.Height(19.0f, keys, w - pad * 2);
    h += 10 + keys_h + 28 + 1 + 22 + 60 + 26;
    SaveBackdrop(0.55f);
    p = SavePainterFor(w, h);
    p.Panel(w, h);
    p.Header(pad, 30, Icon::Gamepad, TL("Graffiti without a gyro", "Граффити без гироскопа"),
             TL("No motion sensor on your controller? Use sensor mode:",
                "На геймпаде нет датчика движения? Включите режим сенсора:"));
    float y = 30 + 64 + 30;
    p.Line(pad, y, w - pad, y, Theme::Line, 1.0f);
    y += 1 + 28;
    for (int i = 0; i < 4; ++i) {
        const ImVec2 c = p.P(pad + 22, y + 22);
        p.dl->AddCircleFilled(c, 21.0f * p.k, Theme::U32(Theme::Accent, 0.16f));
        p.dl->AddCircle(c, 21.0f * p.k, Theme::U32(Theme::Accent), 0, 2.0f * p.k);
        char number[4];
        std::snprintf(number, sizeof(number), "%d", i + 1);
        p.Text(pad + 22 - p.Width(20.0f, number, true) / 2, y + 22 - 14, 20.0f, Theme::Text, number, true);
        const float th = p.Height(22.0f, steps[i], text_w);
        p.Text(text_x, y + (step_h[i] - th) / 2, 22.0f, Theme::Text, steps[i], false, 1.0f, text_w);
        y += step_h[i] + 16;
    }
    y += 10;
    p.Text(pad, y, 19.0f, Theme::Muted, keys, false, 1.0f, w - pad * 2);
    y += keys_h + 28;
    p.Line(pad, y, w - pad, y, Theme::Line, 1.0f);
    y += 1 + 22;
    const float bw = p.PillWidth(ok, 0, 200), x0 = w - pad - bw;
    const float again_w = x0 - pad - 24;
    p.Text(pad, y + 30 - p.Height(17.0f, again, again_w) / 2, 17.0f, Theme::Faint, again, false, 1.0f, again_w);
    p.Pill(x0, y, w - pad, ok, 0, true);
    guide_button_rect = ImRect(p.P(x0, y), p.P(w - pad, y + 60));
}

// The game's text dialog as the launcher's panel: the prompt, the text in a field with the
// caret, what the keys and buttons do.
void TextEntryBox() {
    const float w = 900, pad = 40;
    const char* keys = TL("Keyboard: type, Backspace to erase, Enter = OK, Esc = cancel",
                          "Клавиатура: ввод текста, Backspace — стереть, Enter — OK, Esc — отмена");
    const char* pad_keys = TL(PadText("Controller: Cross (A) = OK, Circle (B) = cancel", "Controller: A = OK, B = cancel",
                                      "Controller: B = OK, A = cancel"),
                              PadText("Геймпад: крестик (A) — OK, кружок (B) — отмена", "Геймпад: A — OK, B — отмена",
                                      "Геймпад: B — OK, A — отмена"));
    const float h = 34 + 30 + 18 + 80 + 22 + 26 + 6 + 26 + 30;
    SavePainter p = SavePainterFor(w, h);
    // Above the middle, so the game's keyboard prompt below stays visible.
    p.at.y = ImGui::GetMainViewport()->WorkPos.y + ImGui::GetMainViewport()->WorkSize.y * 0.42f - h * p.k / 2;
    p.Panel(w, h);
    float y = 34;
    p.Text(pad, y, 22.0f, Theme::Muted, p.Fit(22.0f, text_entry_prompt, w - pad * 2));
    y += 30 + 18;
    p.Box(pad, y, w - pad, y + 80, Theme::White(0.06f), kItemRadius);
    p.Frame(pad, y, w - pad, y + 80, Theme::Accent, kItemRadius, 2.0f, 0.6f);
    const std::string shown = p.Fit(32.0f, text_entry_text, w - pad * 2 - 60);
    p.TextLight(pad + 22, y + 18, 32.0f, Theme::Text, shown);
    const float caret_x = pad + 22 + p.LightWidth(32.0f, shown) + 3;
    if (std::fmod(ImGui::GetTime(), 1.0) < 0.6) p.Box(caret_x, y + 20, caret_x + 2.5f, y + 60, Theme::Accent, 1.0f);
    y += 80 + 22;
    p.Text(pad, y, 17.0f, Theme::Faint, keys);
    y += 26 + 6;
    p.Text(pad, y, 17.0f, Theme::Faint, pad_keys);
}

// imgui_mutex held.
void OpenGuide() {
    guide_button_rect = ImRect();
    guide_active = true;
    SetDrawCursor(true);
}

void CloseGuide() {
    guide_active = false;
    SetDrawCursor(menu_open);
    auto& s = TlgSettings::Get();
    if (!s.graffiti_guide_seen.exchange(true)) {
        TlgSettings::Save();
    }
}

// Present thread, imgui_mutex held, after NewFrame: whichever save dialog is up.
void SaveDialogs() {
    const auto now = SaveClock::now();
    if (guide_active) {
        GuideBox();
        return;
    }
    if (choice_active) {
        SaveListBox();
    } else if (message_active) {
        SaveMessageBox();
    } else if (progress_on || now < progress_until) {
        SaveProgressBox(now);
    }
    if (now < toast_until && !choice_active && !message_active) {
        SaveToastBox(now);
    }
}

// "Compiling shaders" in the launcher's look (renderer v2's own loading screen too): a centred
// block, 40% of the width, with the step in Light, the blue bar, the count, the time left and a note.
// A trophy's icon as a texture; its picture is decoded now (imgui_mutex held) and becomes a
// texture before the next frame, so the first call returns null.
ImTextureData* TrophyTexture(int id, int* decode_budget) {
    if (const auto it = trophy_textures.find(id); it != trophy_textures.end()) return it->second;
    for (const auto& pending : trophy_pictures) {
        if (pending.first == id) return nullptr;
    }
    if (decode_budget && (*decode_budget)-- <= 0) return nullptr;
    TlgPng::Image picture;
    TlgTrophies::Icon(id, picture); // a failed one stays empty and is not tried again
    trophy_pictures.emplace_back(id, std::move(picture));
    return nullptr;
}

std::string TrophyDate(int64_t time) {
    const std::time_t t = std::time_t(time);
    char text[32] = "";
    if (const std::tm* local = std::localtime(&t)) std::strftime(text, sizeof(text), "%d.%m.%Y", local);
    return text;
}

ImVec4 GradeColor(TlgTrophies::Grade grade) {
    switch (grade) {
    case TlgTrophies::Platinum: return ImVec4(0.80f, 0.88f, 1.0f, 1.0f);
    case TlgTrophies::Gold: return ImVec4(1.0f, 0.80f, 0.35f, 1.0f);
    case TlgTrophies::Silver: return ImVec4(0.80f, 0.82f, 0.86f, 1.0f);
    default: return ImVec4(0.86f, 0.56f, 0.36f, 1.0f);
    }
}

const char* GradeName(TlgTrophies::Grade grade) {
    switch (grade) {
    case TlgTrophies::Platinum: return TL("Platinum", "Платиновый");
    case TlgTrophies::Gold: return TL("Gold", "Золотой");
    case TlgTrophies::Silver: return TL("Silver", "Серебряный");
    default: return TL("Bronze", "Бронзовый");
    }
}

// A trophy's name: the game's, or its number where this game is not described.
std::string TrophyName(int id) {
    if (const TlgTrophies::Info* info = TlgTrophies::Describe(id)) return info->name;
    char text[32];
    std::snprintf(text, sizeof(text), "%s %d", TL("Trophy No.", "Трофей №"), id);
    return text;
}

// The trophies of the pack, a row each: the icon (dimmed until unlocked), the name, the grade
// and how to get it, the unlock date on the right.
void TrophiesTab() {
    const int count = TlgTrophies::Count();
    const std::vector<TlgTrophies::Unlocked> unlocked = TlgTrophies::List();
    SectionTitle(TL("Trophies", "Трофеи"));
    if (BeginRows("##trophies")) {
        char text[64];
        std::snprintf(text, sizeof(text), "%zu / %d", unlocked.size(), count);
        InfoRow(TL("Unlocked on this PC", "Получено на этом ПК"),
                count ? text : TL("no trophy data in the game files", "в файлах игры нет данных о трофеях"));
        CheckRow(TL("Trophy sound", "Звук трофея"), TlgSettings::Get().trophy_sound,
                 TL("Your own sounds: data/sounds/trophy.mp3 and trophy-platinum.mp3 (or .wav).",
                    "Свои звуки: data/sounds/trophy.mp3 и trophy-platinum.mp3 (или .wav)."));
        EndRows();
    }
    if (!count) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float icon = L(76), row_h = L(100), inset = L(20);
    const float width = ImGui::GetContentRegionAvail().x - inset * 2;
    const ImVec2 origin(ImGui::GetCursorScreenPos().x + inset, ImGui::GetCursorScreenPos().y + L(8));
    int budget = 3; // icons decoded per frame
    for (int id = 0; id < count; ++id) {
        const float x = origin.x, y = origin.y + float(id) * row_h;
        const TlgTrophies::Unlocked* got = nullptr;
        for (const auto& u : unlocked) {
            if (u.id == id) got = &u;
        }
        const TlgTrophies::Info* info = TlgTrophies::Describe(id);
        if (ImTextureData* tex = TrophyTexture(id, &budget)) {
            dl->AddImageRounded(tex->GetTexRef(), ImVec2(x, y), ImVec2(x + icon, y + icon), ImVec2(0, 0), ImVec2(1, 1),
                                IM_COL32(255, 255, 255, got ? 255 : 70), L(8));
        } else {
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + icon, y + icon), ImGui::GetColorU32(Theme::White(0.06f)), L(8));
        }
        const float tx = x + icon + L(18), date_w = L(150);
        const float text_w = width - icon - L(18) - date_w;
        ImGui::PushClipRect(ImVec2(tx, y), ImVec2(tx + text_w, y + row_h), true);
        dl->AddText(LightFont(), L(25), ImVec2(tx, y + L(2)), ImGui::GetColorU32(got ? Theme::Text : Theme::White(0.75f)),
                    TrophyName(id).c_str());
        if (info) {
            const char* grade = GradeName(info->grade);
            dl->AddText(ImGui::GetFont(), L(16), ImVec2(tx, y + L(34)), ImGui::GetColorU32(GradeColor(info->grade)), grade);
            dl->AddText(ImGui::GetFont(), L(18), ImVec2(tx, y + L(54)), ImGui::GetColorU32(Theme::Muted), TL(info->en, info->ru));
        }
        ImGui::PopClipRect();
        const std::string when = got ? TrophyDate(got->time) : std::string(TL("locked", "не получен"));
        const ImVec2 ws = TextSize(ImGui::GetFont(), 17, when.c_str());
        if (got) DrawIcon(dl, Icon::Check, ImVec2(x + width - ws.x - L(16), y + L(16)), 16, ImGui::GetColorU32(Theme::Good));
        dl->AddText(ImGui::GetFont(), L(17), ImVec2(x + width - ws.x, y + L(6)), ImGui::GetColorU32(got ? Theme::Text : Theme::Faint),
                    when.c_str());
        if (id + 1 < count) {
            dl->AddLine(ImVec2(tx, y + row_h - L(10)), ImVec2(x + width, y + row_h - L(10)), ImGui::GetColorU32(Theme::Line));
        }
    }
    ImGui::Dummy(ImVec2(0.0f, L(8) + float(count) * row_h));
}

// "Trophy unlocked": the new trophy's icon and number in the top-left corner for 6 seconds, one
// after another when several come at once, 2.5 seconds each while more wait (a save loaded
// after this update: the game unlocks the ones it already gave again, new to this PC).
void TrophyToast() {
    const auto now = std::chrono::steady_clock::now();
    const float t = std::chrono::duration<float>(now - trophy_toast_since).count();
    if (trophy_toast >= 0 && t >= trophy_toast_seconds) trophy_toast = -1;
    if (trophy_toast < 0 && !trophy_queue.empty()) {
        trophy_toast = trophy_queue.front();
        trophy_queue.erase(trophy_queue.begin());
        trophy_toast_since = now;
        trophy_toast_seconds = trophy_queue.empty() ? 6.0f : 2.5f;
        if (TlgSettings::Get().trophy_sound) {
            const TlgTrophies::Info* info = TlgTrophies::Describe(trophy_toast);
            TlgSound::PlayTrophy(info && info->grade == TlgTrophies::Platinum);
        }
    }
    if (trophy_toast < 0) return;
    const float shown = std::chrono::duration<float>(now - trophy_toast_since).count();
    const float alpha = std::clamp(std::min(shown / 0.25f, (trophy_toast_seconds - shown) / 0.5f), 0.0f, 1.0f);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), base_scale,
                  ImVec2(viewport->WorkPos.x + 40 * base_scale, viewport->WorkPos.y + 36 * base_scale)};
    const float w = 420, h = 104;
    p.Box(0, 0, w, h, Theme::Popover, 14.0f, Theme::PopoverAlpha * alpha);
    p.Frame(0, 0, w, h, Theme::White(0.08f), 14.0f, 1.0f, alpha);
    if (ImTextureData* tex = TrophyTexture(trophy_toast, nullptr)) {
        p.dl->AddImageRounded(tex->GetTexRef(), p.P(16, 16), p.P(88, 88), ImVec2(0, 0), ImVec2(1, 1),
                              IM_COL32(255, 255, 255, int(255 * alpha)), 6.0f * p.k);
    } else {
        p.Box(16, 16, 88, 88, Theme::White(0.08f), 6.0f, alpha);
    }
    const TlgTrophies::Info* info = TlgTrophies::Describe(trophy_toast);
    p.Text(106, 18, 17.0f, info ? GradeColor(info->grade) : Theme::Muted,
           info ? std::string(TL("Trophy unlocked · ", "Трофей получен · ")) + GradeName(info->grade)
                : std::string(TL("Trophy unlocked", "Трофей получен")),
           false, alpha);
    p.Text(106, 44, 22.0f, Theme::Text, p.Fit(22.0f, TrophyName(trophy_toast), w - 122), true, alpha);
}

void LoadingScreen() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k = viewport->WorkSize.y / 1080.0f;
    const float w = std::max(640.0f, viewport->WorkSize.x / k * 0.4f);
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    p.at = ImVec2(viewport->WorkPos.x + (viewport->WorkSize.x - w * k) / 2, viewport->WorkPos.y + viewport->WorkSize.y / 2 - 90 * k);
    const u32 done = loading_done, total = loading_total;
    const bool compile = loading_compile;
    const char* title = compile ? TL("Compiling shaders...", "Компиляция шейдеров...")
                        : total ? TL("Loading shader cache...", "Загрузка кэша шейдеров...")
                                : TL("Preparing graphics...", "Подготовка графики...");
    p.TextLight(0, 0, 40.0f, Theme::Text, title);
    float y = 70;
    p.Box(0, y, w, y + 10, Theme::White(0.1f), 5.0f);
    if (total) {
        const float f = std::min(1.0f, float(done) / float(total));
        if (f > 0) p.Box(0, y, w * f, y + 10, Theme::Accent, 5.0f);
        char count[64];
        std::snprintf(count, sizeof(count), "%u / %u (%u%%)", done, total, u32(u64(std::min(done, total)) * 100 / total));
        p.Text(0, y + 24, 20.0f, Theme::Text, count);
        if (compile && done > 0 && done < total) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - loading_compile_started).count();
            const u32 left = u32(elapsed * double(total - done) / double(done)) + 1;
            char eta[64];
            std::snprintf(eta, sizeof(eta), TL("About %u:%02u left", "Осталось около %u:%02u"), left / 60, left % 60);
            p.TextRight(w, y + 24, 20.0f, Theme::Muted, eta);
        }
    } else { // no figure yet: a segment sweeps the bar
        const float span = w * 0.25f, phase = std::fmod(float(ImGui::GetTime()) * 0.6f, 1.0f);
        const float s0 = (w + span) * phase - span;
        p.Box(std::max(s0, 0.0f), y, std::min(s0 + span, w), y + 10, Theme::Accent, 5.0f);
    }
    y += 24 + 40;
    p.Text(0, y, 19.0f, Theme::Muted,
           compile ? TL("Only on the first start on this computer and after updates: then the game runs without compiling shaders.",
                        "Только при первом запуске на этом компьютере и после обновлений: дальше игра работает без компиляции шейдеров.")
                   : TL("The first start after an update takes longer.", "Первый запуск после обновления занимает больше времени."),
           false, 1.0f, w);
}

// Frame time statistics over the last frame_long intervals (half a minute at 60 FPS).
struct FrameStats {
    float min_fps = 0.0f, avg_fps = 0.0f, max_fps = 0.0f, low1_fps = 0.0f, low01_fps = 0.0f;
    float max_ms = 0.0f, p50_ms = 0.0f, p90_ms = 0.0f, p99_ms = 0.0f;
    float jitter_ms = 0.0f; ///< the mean change from one frame to the next
    int stutters = 0;       ///< frames over twice the average and over 25 ms
};

FrameStats ComputeFrameStats() {
    FrameStats out;
    const size_t n = frame_long_count;
    if (n == 0) {
        return out;
    }
    std::array<float, frame_long.size()> sorted{};
    const size_t first = (frame_long_next + frame_long.size() - n) % frame_long.size();
    double total = 0.0, change = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sorted[i] = frame_long[(first + i) % frame_long.size()];
        total += sorted[i];
        if (i) change += std::fabs(sorted[i] - sorted[i - 1]);
    }
    const float avg_ms = float(total / n);
    for (size_t i = 0; i < n; ++i) {
        out.stutters += sorted[i] > 2.0f * avg_ms && sorted[i] > 25.0f;
    }
    std::sort(sorted.begin(), sorted.begin() + n);
    const auto at = [&](float q) { return sorted[std::min(n - 1, size_t(n * q))]; };
    out.max_ms = sorted[n - 1];
    out.p50_ms = at(0.5f);
    out.p90_ms = at(0.9f);
    out.p99_ms = at(0.99f);
    out.avg_fps = 1000.0f / avg_ms;
    out.min_fps = 1000.0f / out.max_ms;
    out.max_fps = 1000.0f / std::max(sorted[0], 0.1f);
    // 1% and 0.1% lows: the frame times 99% and 99.9% of the frames stay under.
    out.low1_fps = 1000.0f / out.p99_ms;
    out.low01_fps = 1000.0f / at(0.999f);
    out.jitter_ms = n > 1 ? float(change / (n - 1)) : 0.0f;
    return out;
}

ImVec4 FpsColor(float fps) {
    return fps >= 55.0f ? Theme::Good : fps >= 28.0f ? Theme::Warn : Theme::Bad;
}

// Where an overlay of `size` display pixels sits: the chosen corner or the centre of the top or
// bottom edge, the chosen distance from the edges (a centred one ignores the side one), on screen.
ImVec2 OverlayPosition(ImVec2 size) {
    const auto& s = TlgSettings::Get();
    const ImGuiViewport* v = ImGui::GetMainViewport();
    const int place = s.overlay_corner;
    const bool centre = place >= 4, right = place == 1 || place == 3, bottom = place == 2 || place == 3 || place == 5;
    const float mx = s.overlay_offset_x * base_scale, my = s.overlay_offset_y * base_scale;
    float x = centre ? v->WorkPos.x + (v->WorkSize.x - size.x) / 2
              : right ? v->WorkPos.x + v->WorkSize.x - mx - size.x
                      : v->WorkPos.x + mx;
    float y = bottom ? v->WorkPos.y + v->WorkSize.y - my - size.y : v->WorkPos.y + my;
    x = std::clamp(x, v->WorkPos.x, std::max(v->WorkPos.x, v->WorkPos.x + v->WorkSize.x - size.x));
    y = std::clamp(y, v->WorkPos.y, std::max(v->WorkPos.y, v->WorkPos.y + v->WorkSize.y - size.y));
    return ImVec2(std::round(x), std::round(y));
}

// The performance overlay as the mock-up's card: the FPS large (Light, coloured by the rate) with
// the frame time on its right, the other items in one muted line, the frame time curve (guides
// at 60 and 30 FPS, stutters in red) and its statistics; the launcher's glass behind. The
// settings' scale, position, opacity and items apply.
void CardOverlay() {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const unsigned items = s.overlay_items;
    const bool big_fps = items & P::Fps, frame_time = items & P::FrameTime, graph = items & P::Graph;
    const std::string text = OverlayText(P::Fps | P::FrameTime);
    if (text.empty() && !big_fps && !frame_time && !graph) {
        return;
    }
    const float k = base_scale * s.overlay_scale / 100.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    const float fps = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
    char fps_text[16], ms_text[32];
    std::snprintf(fps_text, sizeof(fps_text), "%.0f", fps);
    std::snprintf(ms_text, sizeof(ms_text), "%.1f %s", frame_ms_avg, TL("ms", "мс"));
    if (UiRussian()) {
        for (char* c = ms_text; *c; ++c) if (*c == '.') *c = ',';
    }
    const float pad_x = 22, pad_y = 18, graph_w = 396, graph_h = 64;
    float w = graph ? graph_w : 0.0f;
    if (big_fps || frame_time) w = std::max(w, p.LightWidth(44.0f, fps_text) + 60 + p.Width(17.0f, ms_text) + 24);
    if (!text.empty()) w = std::max(w, std::min(p.Width(16.0f, text), 560.0f));
    const float text_h = text.empty() ? 0.0f : p.Height(16.0f, text, w);
    float h = pad_y * 2;
    if (big_fps || frame_time) h += 44 + 8;
    if (!text.empty()) h += text_h + 8;
    if (graph) h += graph_h + 8 + 22;
    h -= 8;
    w += pad_x * 2;
    p.at = OverlayPosition(ImVec2(w * k, h * k));
    const float opacity = s.overlay_opacity / 100.0f;
    if (opacity > 0.0f) {
        p.Box(0, 0, w, h, ImVec4(28 / 255.0f, 28 / 255.0f, 30 / 255.0f, 1.0f), 14.0f, 0.78f * opacity);
        p.Frame(0, 0, w, h, Theme::White(0.07f), 14.0f, 1.0f, opacity);
    }
    float y = pad_y;
    if (big_fps || frame_time) {
        float x = pad_x;
        if (big_fps) {
            p.TextLight(x, y - 6, 44.0f, FpsColor(fps), fps_text);
            x += p.LightWidth(44.0f, fps_text) + 10;
            p.Text(x, y + 20, 17.0f, Theme::Muted, "FPS");
        }
        if (frame_time) p.TextRight(w - pad_x, y + 20, 17.0f, Theme::Muted, ms_text);
        y += 44 + 8;
    }
    if (!text.empty()) {
        p.Text(pad_x, y, 16.0f, Theme::Muted, text, false, 1.0f, w - pad_x * 2);
        y += text_h + 8;
    }
    if (!graph) return;
    // The curve: the last present intervals over at least 40 ms (at most 100).
    const float gx = pad_x, gy = y;
    p.Box(gx, gy, gx + graph_w, gy + graph_h, Theme::White(0.05f), 4.0f);
    float top_ms = 40.0f;
    for (size_t i = 0; i < frame_history_count; ++i) top_ms = std::max(top_ms, frame_history[i] * 1.15f);
    top_ms = std::min(top_ms, 100.0f);
    const auto y_of = [&](float ms) { return gy + graph_h - std::min(ms, top_ms) / top_ms * graph_h; };
    for (const float ms : {1000.0f / 60.0f, 1000.0f / 30.0f}) { // dashed guides
        const float ly = y_of(ms);
        for (float x = gx; x < gx + graph_w; x += 7) p.Line(x, ly, std::min(x + 3, gx + graph_w), ly, Theme::White(0.12f), 1.0f);
    }
    const FrameStats stats = ComputeFrameStats();
    if (frame_history_count >= 2) {
        const size_t n = frame_history_count;
        const size_t first = (frame_history_next + frame_history.size() - n) % frame_history.size();
        float avg = 0.0f;
        for (size_t i = 0; i < n; ++i) avg += frame_history[i];
        avg /= float(n);
        std::vector<ImVec2> points(n);
        for (size_t i = 0; i < n; ++i) {
            const float ms = frame_history[(first + i) % frame_history.size()];
            points[i] = p.P(gx + graph_w * float(i) / float(frame_history.size() - 1), y_of(ms));
        }
        p.dl->AddPolyline(points.data(), int(n), Theme::U32(Theme::Accent), ImDrawFlags_None, 2.0f * k);
        for (size_t i = 0; i < n; ++i) {
            const float ms = frame_history[(first + i) % frame_history.size()];
            if (ms > 2.0f * avg && ms > 25.0f) p.dl->AddCircleFilled(points[i], 3.5f * k, Theme::U32(Theme::Bad));
        }
    }
    y += graph_h + 8;
    char line[160];
    std::snprintf(line, sizeof(line), "%s %.0f     %s %.0f     1%% %.0f     %s %d", TL("min", "мин"), stats.min_fps,
                  TL("avg", "сред"), stats.avg_fps, stats.low1_fps, TL("stutters", "фризы"), stats.stutters);
    p.Text(pad_x, y, 15.0f, Theme::Faint, line);
}

// ---- The square overlay styles (the Canvas mock-ups A to E in the launcher's colours): drawn
// in stage pixels from the panel's top-left corner, each returns the panel's size. Frames are
// coloured against the budget: green within the target's frame time, amber up to twice it, red
// beyond. ----

#if defined(TLG_RENDERER_V3)
constexpr const char* RendererName = "V3";
#elif defined(TLG_RENDERER_V2)
constexpr const char* RendererName = "V2";
#else
constexpr const char* RendererName = "V1";
#endif

// The overlay's numbers for one frame.
struct OverlayData {
    TlgPerfStats::Sample sample;
    FrameStats stats;
    unsigned items = 0;
    float fps = 0.0f, ms = 0.0f;
    int target = 60;                   ///< the FPS the colours compare with: the frame rate limit, else 60
    float budget_ms = 1000.0f / 60.0f; ///< the target's frame time
};

OverlayData CollectOverlayData() {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    OverlayData d;
    d.items = s.overlay_items;
    if ((d.items & (P::Gpu | P::Cpu | P::Ram | P::Vram)) && overlay_instance) {
        P::Start(*overlay_instance);
    }
    d.sample = P::Latest();
    d.ms = frame_ms_avg;
    d.fps = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
    d.sample.upscaler = UpscalerLabel(s.upscaler);
    d.stats = ComputeFrameStats();
    d.target = s.fps_limit > 0 ? s.fps_limit.load() : 60;
    d.budget_ms = 1000.0f / float(d.target);
    return d;
}

ImVec4 FrameColor(float ms, float budget_ms) {
    return ms <= budget_ms * 1.05f ? Theme::Good : ms <= budget_ms * 2.0f ? Theme::Warn : Theme::Bad;
}

ImVec4 RateColor(float fps, int target) {
    return fps >= target * 0.95f ? Theme::Good : fps >= target * 0.5f ? Theme::Warn : Theme::Bad;
}

// A number with `decimals` places, with a decimal comma in Russian.
std::string Num(float value, int decimals = 0) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    if (UiRussian()) {
        for (char* c = text; *c; ++c) if (*c == '.') *c = ',';
    }
    return text;
}

std::string Ms(float value) { return Num(value, 1) + " " + TL("ms", "мс"); }
std::string Percent(float value) { return value < 0.0f ? "n/a" : Num(value) + "%"; }
std::string Gigabytes(double used, double total) {
    if (used < 0.0) return "n/a";
    return total < 0.0 ? Num(float(used), 1) + "G" : Num(float(used), 1) + "/" + Num(float(total), 1) + "G";
}
float Fraction(double used, double total) { return used < 0.0 || total <= 0.0 ? 0.0f : float(used / total); }

// "NVIDIA GeForce RTX 4070 SUPER" as "RTX 4070 SUPER".
std::string ShortGpuName(std::string name) {
    for (const char* word : {"NVIDIA ", "GeForce ", "AMD ", "Radeon(TM) ", "Intel(R) ", "(R)", "(TM)"}) {
        for (size_t at; (at = name.find(word)) != std::string::npos;) name.erase(at, std::strlen(word));
    }
    return name;
}

// The i-th of the last n present intervals, oldest first.
float RecentFrame(size_t n, size_t i) {
    return frame_history[(frame_history_next + frame_history.size() - n + i) % frame_history.size()];
}

// The graphs' top: three frame budgets, at least 40 ms and at most 100.
float GraphTop(const OverlayData& d) { return std::clamp(d.budget_ms * 3.0f, 40.0f, 100.0f); }

// Dashed guides at 60 and 30 FPS over a graph whose top is `top` ms, labelled on the left.
void Guides(const SavePainter& p, float x, float y, float w, float h, float top, const OverlayData& d, bool labels) {
    for (const int rate : {60, 30}) {
        const float ms = 1000.0f / rate;
        if (ms >= top) continue;
        const float ly = y + h - ms / top * h;
        const ImVec4 color = FrameColor(ms, d.budget_ms);
        for (float lx = x; lx < x + w; lx += 6) p.Line(lx, ly, std::min(lx + 3, x + w), ly, color, 1.0f, 0.75f);
        if (labels) {
            const std::string text = std::to_string(rate);
            p.Box(x, ly - 13, x + p.Width(11.0f, text) + 6, ly, Theme::Ink, 0.0f, 0.6f);
            p.Text(x + 3, ly - 13, 11.0f, color, text);
        }
    }
}

// The last frames as bars (the newest on the right) coloured against the budget.
void FrameBars(const SavePainter& p, float x, float y, float w, float h, const OverlayData& d, bool labels) {
    p.Box(x, y, x + w, y + h, Theme::White(0.05f), 0.0f);
    const float top = GraphTop(d), step = 3.0f; // a 2-pixel bar and a 1-pixel gap
    const size_t n = std::min(frame_history_count, size_t(w / step));
    float max_ms = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float ms = RecentFrame(n, i);
        const float bx = x + w - float(n - i) * step;
        const float bh = std::max(1.0f, std::min(ms, top) / top * h);
        p.Box(bx, y + h - bh, bx + step - 1.0f, y + h, FrameColor(ms, d.budget_ms), 0.0f);
        max_ms = std::max(max_ms, ms);
    }
    Guides(p, x, y, w, h, top, d, labels);
    p.Line(x, y + h, x + w, y + h, Theme::White(0.2f), 1.0f);
    if (labels && n) {
        const std::string text = TL("max ", "макс ") + Num(max_ms, 1);
        const float tw = p.Width(11.0f, text) + 6;
        p.Box(x + w - tw, y, x + w, y + 13, Theme::Ink, 0.0f, 0.6f);
        p.Text(x + w - tw + 3, y, 11.0f, Theme::Muted, text);
    }
}

// The frame time curve over a filled area, the target's budget shaded, stutters marked.
void FrameCurve(const SavePainter& p, float x, float y, float w, float h, const OverlayData& d) {
    p.Box(x, y, x + w, y + h, Theme::White(0.05f), 0.0f);
    const float top = GraphTop(d);
    const auto y_of = [&](float ms) { return y + h - std::min(ms, top) / top * h; };
    if (d.budget_ms < top) p.Box(x, y_of(d.budget_ms), x + w, y + h, Theme::Good, 0.0f, 0.07f);
    Guides(p, x, y, w, h, top, d, true);
    const size_t n = frame_history_count;
    if (n >= 2) {
        const float step = w / float(frame_history.size() - 1);
        const float x0 = x + w - float(n - 1) * step;
        std::vector<ImVec2> points(n);
        size_t max_i = 0;
        for (size_t i = 0; i < n; ++i) {
            const float ms = RecentFrame(n, i), px = x0 + float(i) * step;
            p.Box(px, y_of(ms), px + step, y + h, Theme::Text, 0.0f, 0.07f);
            points[i] = p.P(px, y_of(ms));
            if (ms > RecentFrame(n, max_i)) max_i = i;
        }
        p.dl->AddPolyline(points.data(), int(n), Theme::U32(Theme::Text, 0.9f), ImDrawFlags_None, 1.5f * p.k);
        for (size_t i = 0; i < n; ++i) {
            const float ms = RecentFrame(n, i);
            if (ms > 2.0f * d.ms && ms > 25.0f) {
                const float px = x0 + float(i) * step, py = y_of(ms);
                p.Box(px - 2.5f, py - 2.5f, px + 2.5f, py + 2.5f, FrameColor(ms, d.budget_ms), 0.0f);
            }
        }
        const std::string text = Num(RecentFrame(n, max_i), 1);
        const float tw = p.Width(11.0f, text) + 6;
        const float tx = std::clamp(x0 + float(max_i) * step - tw / 2, x + 22, x + w - tw);
        p.Box(tx, y, tx + tw, y + 13, Theme::Ink, 0.0f, 0.6f);
        p.Text(tx + 3, y, 11.0f, FrameColor(RecentFrame(n, max_i), d.budget_ms), text);
    }
    p.Line(x, y + h, x + w, y + h, Theme::White(0.2f), 1.0f);
}

// How the frame times spread (the long window), on the curve's scale: a bar per 1/30 of it.
void FrameHistogram(const SavePainter& p, float x, float y, float w, float h, const OverlayData& d) {
    p.Box(x, y, x + w, y + h, Theme::White(0.05f), 0.0f);
    constexpr int rows = 30;
    const float top = GraphTop(d), row_h = h / rows;
    std::array<int, rows> counts{};
    for (size_t i = 0; i < frame_long_count; ++i) {
        ++counts[std::min(rows - 1, int(frame_long[i] / top * rows))];
    }
    const int peak = *std::max_element(counts.begin(), counts.end());
    for (int r = 0; peak && r < rows; ++r) {
        if (!counts[r]) continue;
        const float bw = std::max(1.0f, float(counts[r]) / float(peak) * (w - 2));
        const float ry = y + h - float(r + 1) * row_h;
        p.Box(x, ry + 0.5f, x + bw, ry + row_h - 0.5f, FrameColor((r + 0.5f) * top / rows, d.budget_ms), 0.0f);
    }
    p.Line(x, y + h, x + w, y + h, Theme::White(0.2f), 1.0f);
}

// Frame pacing: the change from the previous frame, up for slower and down for faster.
void FramePacing(const SavePainter& p, float x, float y, float w, float h, const OverlayData& d) {
    p.Box(x, y, x + w, y + h, Theme::White(0.04f), 0.0f);
    const float mid = y + h / 2, step = 3.0f;
    const size_t n = std::min(frame_history_count, size_t(w / step) + 1);
    for (size_t i = 1; i < n; ++i) {
        const float change = RecentFrame(n, i) - RecentFrame(n, i - 1), size = std::fabs(change);
        const float bh = std::min(size, 12.0f) / 12.0f * (h / 2 - 1), bx = x + w - float(n - i) * step;
        const ImVec4& color = size > 8.0f ? Theme::Bad : size > 3.0f ? Theme::Warn : Theme::Muted;
        if (change > 0) p.Box(bx, mid - bh, bx + step - 1, mid, color, 0.0f);
        else p.Box(bx, mid, bx + step - 1, mid + bh, color, 0.0f);
    }
    p.Line(x, mid, x + w, mid, Theme::White(0.2f), 1.0f);
    const std::string text = TL("pacing ±", "ровность ±") + Ms(d.stats.jitter_ms);
    const float tw = p.Width(11.0f, text) + 6;
    p.Box(x + w - tw, y, x + w, y + 13, Theme::Ink, 0.0f, 0.6f);
    p.Text(x + w - tw + 3, y, 11.0f, Theme::Muted, text);
}

// A thin track filled up to `fraction` (blue, amber from 90%).
void Track(const SavePainter& p, float x0, float y, float x1, float h, float fraction) {
    p.Box(x0, y, x1, y + h, Theme::White(0.14f), 0.0f);
    if (fraction > 0.0f) {
        p.Box(x0, y, x0 + (x1 - x0) * std::min(fraction, 1.0f), y + h, fraction >= 0.9f ? Theme::Warn : Theme::Accent, 0.0f);
    }
}

struct MeterItem {
    const char* label;
    std::string value, short_value;
    float fraction;
};

std::vector<MeterItem> Meters(const OverlayData& d) {
    namespace P = TlgPerfStats;
    const auto& s = d.sample;
    std::vector<MeterItem> out;
    if (d.items & P::Gpu) out.push_back({"GPU", Percent(s.gpu_percent), Percent(s.gpu_percent), s.gpu_percent / 100.0f});
    if (d.items & P::Cpu) out.push_back({"CPU", Percent(s.cpu_percent), Percent(s.cpu_percent), s.cpu_percent / 100.0f});
    if (d.items & P::Vram) {
        out.push_back({"VRAM", Gigabytes(s.vram_used_gb, s.vram_total_gb), Gigabytes(s.vram_used_gb, -1.0),
                       Fraction(s.vram_used_gb, s.vram_total_gb)});
    }
    if (d.items & P::Ram) {
        out.push_back({"RAM", Gigabytes(s.ram_used_gb, s.ram_total_gb), Gigabytes(s.ram_used_gb, -1.0),
                       Fraction(s.ram_used_gb, s.ram_total_gb)});
    }
    return out;
}

// A meter: the label and the value over a thin track (21 pixels high).
void Meter(const SavePainter& p, float x, float y, float w, const MeterItem& m) {
    p.Text(x, y, 13.0f, Theme::Muted, m.label);
    p.TextRight(x + w, y, 13.0f, Theme::Text, m.value);
    Track(p, x, y + 18, x + w, 3, m.fraction);
}

struct CellItem {
    std::string label, value;
    ImVec4 color;
};

// A row of equal cells, the label small over the value (34 pixels high).
void Cells(const SavePainter& p, float x, float y, float w, const std::vector<CellItem>& cells) {
    const float gap = 3.0f, cw = (w - gap * float(cells.size() - 1)) / float(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        const float cx = x + float(i) * (cw + gap);
        p.Box(cx, y, cx + cw, y + 34, Theme::White(0.06f), 0.0f);
        p.Text(cx + 5, y + 2, 11.0f, Theme::Muted, cells[i].label);
        p.Text(cx + 5, y + 14, 16.0f, cells[i].color, cells[i].value);
    }
}

// A grid of one-line cells, the label on the left and the value on the right (23 pixels a row).
float GridCells(const SavePainter& p, float x, float y, float w, int columns, const std::vector<CellItem>& cells) {
    const float gap = 3.0f, cw = (w - gap * float(columns - 1)) / float(columns);
    for (size_t i = 0; i < cells.size(); ++i) {
        const float cx = x + float(i % columns) * (cw + gap), cy = y + float(i / columns) * 23;
        p.Box(cx, cy, cx + cw, cy + 20, Theme::White(0.06f), 0.0f);
        p.Text(cx + 5, cy + 3, 12.0f, Theme::Muted, cells[i].label);
        p.TextRight(cx + cw - 5, cy + 1, 15.0f, cells[i].color, cells[i].value);
    }
    const size_t rows = (cells.size() + columns - 1) / columns;
    return rows ? float(rows) * 23 - 3 : 0.0f;
}

// The label muted and the value after it; returns their width.
float Pair(const SavePainter& p, float x, float y, float size, const std::string& label, const std::string& value,
           const ImVec4& color) {
    p.Text(x, y, size, Theme::Muted, label);
    const float lw = p.Width(size, label) + 4;
    p.Text(x + lw, y, size, color, value);
    return lw + p.Width(size, value);
}

float PairWidth(const SavePainter& p, float size, const std::string& label, const std::string& value) {
    return p.Width(size, label) + 4 + p.Width(size, value);
}

void PairRight(const SavePainter& p, float right, float y, float size, const std::string& label,
               const std::string& value, const ImVec4& color) {
    Pair(p, right - PairWidth(p, size, label, value), y, size, label, value, color);
}

using Chip = std::pair<std::string, bool>; // text, accent

// What runs: the renderer, the game's image and the window, the upscaler, VSync, the limit and
// (with `gpu`) the graphics card.
std::vector<Chip> InfoChips(const OverlayData& d, bool gpu) {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    std::vector<Chip> out{{RendererName, true}};
    const int iw = s.image_width, ih = s.image_height, ww = s.window_width, wh = s.window_height;
    if (iw > 0 && ih > 0) {
        std::string size = std::to_string(iw) + "×" + std::to_string(ih);
        if (ww > 0 && wh > 0 && (ww != iw || wh != ih)) size += " › " + std::to_string(ww) + "×" + std::to_string(wh);
        out.push_back({size, false});
    }
    if (d.items & P::Upscaler) out.push_back({d.sample.upscaler.empty() ? TL("Native", "Нативно") : d.sample.upscaler, false});
    out.push_back({s.vsync ? "VSync" : TL("VSync off", "VSync выкл"), false});
    if (s.fps_limit > 0) out.push_back({TL("limit ", "лимит ") + std::to_string(s.fps_limit.load()), false});
    if (gpu && (d.items & P::Gpu) && !d.sample.gpu_name.empty()) out.push_back({ShortGpuName(d.sample.gpu_name), false});
    return out;
}

// The chips in rows `width` wide; returns their height.
float Chips(const SavePainter& p, float x, float y, float width, const std::vector<Chip>& chips) {
    float cx = x, cy = y;
    for (const auto& [text, accent] : chips) {
        const float w = p.Width(12.0f, text, accent) + 10.0f;
        if (cx > x && cx + w > x + width) {
            cx = x;
            cy += 22.0f;
        }
        p.Frame(cx, cy, cx + w, cy + 18, accent ? Theme::Accent : Theme::White(0.22f), 0.0f, 1.0f);
        p.Text(cx + 5, cy + 1.5f, 12.0f, accent ? Theme::Accent : Theme::Muted, text, accent);
        cx += w + 4.0f;
    }
    return chips.empty() ? 0.0f : cy - y + 18.0f;
}

// The large FPS (Light, coloured against the target) and the frame time beside it, 40 pixels
// high; returns the right edge.
float Headline(const SavePainter& p, float x, float y, const OverlayData& d, float size, bool jitter) {
    namespace P = TlgPerfStats;
    if (d.items & P::Fps) {
        const std::string fps = Num(d.fps);
        p.TextLight(x, y - size * 0.16f, size, RateColor(d.fps, d.target), fps);
        x += p.LightWidth(size, fps) + 4;
        p.Text(x, y + size * 0.48f, 13.0f, Theme::Muted, "FPS");
        x += p.Width(13.0f, "FPS") + 12;
    }
    if (d.items & P::FrameTime) {
        const std::string ms = Ms(d.ms);
        p.Text(x, y + (jitter ? 1 : size * 0.3f), 18.0f, Theme::Text, ms);
        float w = p.Width(18.0f, ms);
        if (jitter) {
            const std::string spread = "±" + Ms(d.stats.jitter_ms);
            p.Text(x, y + 22, 13.0f, Theme::Muted, spread);
            w = std::max(w, p.Width(13.0f, spread));
        }
        x += w;
    }
    return x;
}

std::vector<CellItem> RateCells(const OverlayData& d) {
    const FrameStats& st = d.stats;
    return {{TL("min", "мин"), Num(st.min_fps), RateColor(st.min_fps, d.target)},
            {TL("avg", "сред"), Num(st.avg_fps), RateColor(st.avg_fps, d.target)},
            {TL("max", "макс"), Num(st.max_fps), RateColor(st.max_fps, d.target)}};
}

CellItem StutterCell(const OverlayData& d) {
    return {TL("stutters", "фризы"), std::to_string(d.stats.stutters), d.stats.stutters ? Theme::Warn : Theme::Good};
}

// A: the FPS, frame time and lows over the bar graph, five statistics, four meters, the chips.
ImVec2 CompactStyle(const SavePainter& p, const OverlayData& d) {
    namespace P = TlgPerfStats;
    const float W = 320, pad = 8, inner = W - 2 * pad, gap = 6;
    const bool head = d.items & (P::Fps | P::FrameTime), graph = d.items & P::Graph;
    float y = pad;
    if (head || graph) {
        if (head) Headline(p, pad, y, d, 40.0f, true);
        if (graph) {
            PairRight(p, W - pad, y + 1, 14.0f, "1%", Num(d.stats.low1_fps), RateColor(d.stats.low1_fps, d.target));
            PairRight(p, W - pad, y + 21, 14.0f, "0.1%", Num(d.stats.low01_fps), RateColor(d.stats.low01_fps, d.target));
        }
        y += 40 + gap;
    }
    if (graph) {
        FrameBars(p, pad, y, inner, 56, d, true);
        y += 56 + gap;
        std::vector<CellItem> cells = RateCells(d);
        cells.push_back({std::string("P99 ") + TL("ms", "мс"), Num(d.stats.p99_ms, 1), FrameColor(d.stats.p99_ms, d.budget_ms)});
        cells.push_back(StutterCell(d));
        Cells(p, pad, y, inner, cells);
        y += 34 + gap;
    }
    const std::vector<MeterItem> meters = Meters(d);
    const float mw = (inner - 12) / 2;
    for (size_t i = 0; i < meters.size(); ++i) {
        Meter(p, pad + float(i % 2) * (mw + 12), y + float(i / 2) * 27, mw, meters[i]);
    }
    y += float((meters.size() + 1) / 2) * 27;
    y += Chips(p, pad, y, inner, InfoChips(d, true)) + gap;
    return ImVec2(W, y - gap + pad);
}

// B: everything in one row, 46 pixels high, cells parted by thin lines.
ImVec2 StripStyle(const SavePainter& p, const OverlayData& d) {
    namespace P = TlgPerfStats;
    const float H = 46, pad = 8, top = 6, low = 25;
    float x = pad;
    bool first = true;
    const auto next = [&] {
        if (!first) {
            x += pad + 2;
            p.Line(x, 0, x, H, Theme::White(0.1f), 1.0f);
            x += pad + 2;
        }
        first = false;
    };
    if (d.items & (P::Fps | P::FrameTime)) {
        next();
        if (d.items & P::Fps) {
            const std::string fps = Num(d.fps);
            p.TextLight(x, 1, 34.0f, RateColor(d.fps, d.target), fps);
            x += p.LightWidth(34.0f, fps) + 6;
        }
        float w = 0.0f;
        if (d.items & P::Fps) {
            p.Text(x, top, 12.0f, Theme::Muted, "FPS");
            w = p.Width(12.0f, "FPS");
        }
        if (d.items & P::FrameTime) {
            const std::string ms = Ms(d.ms);
            p.Text(x, (d.items & P::Fps) ? low - 2 : 13, 15.0f, Theme::Text, ms);
            w = std::max(w, p.Width(15.0f, ms));
        }
        x += w;
    }
    if (d.items & P::Graph) {
        next();
        FrameBars(p, x, 7, 168, 32, d, false);
        x += 168;
        next();
        const std::string lows[2][2] = {{"1%", Num(d.stats.low1_fps)}, {"0.1%", Num(d.stats.low01_fps)}};
        const ImVec4 lows_color[2] = {RateColor(d.stats.low1_fps, d.target), RateColor(d.stats.low01_fps, d.target)};
        const std::string worst[2][2] = {{TL("max", "макс"), Num(d.stats.max_ms, 1)},
                                         {TL("stutters", "фризы"), std::to_string(d.stats.stutters)}};
        const ImVec4 worst_color[2] = {FrameColor(d.stats.max_ms, d.budget_ms), d.stats.stutters ? Theme::Warn : Theme::Good};
        float w = 0.0f;
        for (int r = 0; r < 2; ++r) w = std::max(w, Pair(p, x, r ? low : top, 14.0f, lows[r][0], lows[r][1], lows_color[r]));
        x += w + 12;
        w = 0.0f;
        for (int r = 0; r < 2; ++r) w = std::max(w, Pair(p, x, r ? low : top, 14.0f, worst[r][0], worst[r][1], worst_color[r]));
        x += w;
    }
    const std::vector<MeterItem> meters = Meters(d);
    for (size_t i = 0; i < meters.size(); i += 2) {
        next();
        float w = 0.0f;
        for (size_t j = i; j < std::min(i + 2, meters.size()); ++j) {
            const float ry = j == i ? top : low;
            p.Text(x, ry, 13.0f, Theme::Muted, meters[j].label);
            Track(p, x + 40, ry + 6, x + 70, 4, meters[j].fraction);
            p.Text(x + 76, ry, 14.0f, Theme::Text, meters[j].short_value);
            w = std::max(w, 76 + p.Width(14.0f, meters[j].short_value));
        }
        x += w;
    }
    next();
    const std::vector<Chip> chips = InfoChips(d, false);
    std::string first_line = std::string(RendererName), second_line;
    for (size_t i = 1; i < chips.size(); ++i) {
        std::string& line = i == 1 ? first_line : second_line;
        line += (line.empty() ? "" : i == 1 ? "  " : " · ") + chips[i].first;
    }
    p.Text(x, top, 13.0f, Theme::Accent, RendererName, true);
    p.Text(x + p.Width(13.0f, RendererName, true), top, 13.0f, Theme::Muted, first_line.substr(std::strlen(RendererName)));
    p.Text(x, low, 13.0f, Theme::Muted, second_line);
    x += std::max(p.Width(13.0f, first_line) + 1, p.Width(13.0f, second_line));
    return ImVec2(x + pad, H);
}

// C: the frame time curve with its histogram on the same scale beside it, the percentiles.
ImVec2 HistogramStyle(const SavePainter& p, const OverlayData& d) {
    namespace P = TlgPerfStats;
    const float W = 360, pad = 8, inner = W - 2 * pad, gap = 6;
    const bool head = d.items & (P::Fps | P::FrameTime), graph = d.items & P::Graph;
    float y = pad;
    if (head || graph) {
        if (head) Headline(p, pad, y, d, 40.0f, false);
        if (graph) {
            PairRight(p, W - pad, y + 1, 14.0f, "0.1%", Num(d.stats.low01_fps), RateColor(d.stats.low01_fps, d.target));
            PairRight(p, W - pad - PairWidth(p, 14.0f, "0.1%", Num(d.stats.low01_fps)) - 12, y + 1, 14.0f, "1%",
                      Num(d.stats.low1_fps), RateColor(d.stats.low1_fps, d.target));
            const float over = d.ms - d.budget_ms;
            const std::string budget = TL("budget ", "бюджет ") + Ms(d.budget_ms) + " · ";
            const std::string excess = (over > 0 ? "+" : "") + Num(over, 1);
            const float ew = p.Width(13.0f, excess);
            p.Text(W - pad - ew, y + 22, 13.0f, over > 0 ? Theme::Warn : Theme::Good, excess);
            p.TextRight(W - pad - ew, y + 22, 13.0f, Theme::Muted, budget);
        }
        y += 40 + gap;
    }
    if (graph) {
        const float hw = 52, gw = inner - hw - 4, gh = 92;
        FrameCurve(p, pad, y, gw, gh, d);
        FrameHistogram(p, pad + gw + 4, y, hw, gh, d);
        y += gh + gap;
        const std::string ms = TL(" ms", " мс");
        Cells(p, pad, y, inner,
              {{"P50" + ms, Num(d.stats.p50_ms, 1), FrameColor(d.stats.p50_ms, d.budget_ms)},
               {"P90" + ms, Num(d.stats.p90_ms, 1), FrameColor(d.stats.p90_ms, d.budget_ms)},
               {"P99" + ms, Num(d.stats.p99_ms, 1), FrameColor(d.stats.p99_ms, d.budget_ms)},
               {TL("max", "макс") + ms, Num(d.stats.max_ms, 1), FrameColor(d.stats.max_ms, d.budget_ms)},
               StutterCell(d)});
        y += 34 + gap;
    }
    const std::vector<MeterItem> meters = Meters(d);
    if (!meters.empty()) {
        const float cw = inner / float(meters.size());
        for (size_t i = 0; i < meters.size(); ++i) {
            Pair(p, pad + float(i) * cw, y, 13.0f, meters[i].label, meters[i].short_value, Theme::Text);
        }
        y += 17 + gap;
    }
    y += Chips(p, pad, y, inner, InfoChips(d, true)) + gap;
    return ImVec2(W, y - gap + pad);
}

// D: as much as is measured: the bars, the pacing, nine statistics, a meter per line, the chips.
ImVec2 TelemetryStyle(const SavePainter& p, const OverlayData& d) {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const float W = 380, pad = 8, inner = W - 2 * pad, gap = 6;
    const bool head = d.items & (P::Fps | P::FrameTime), graph = d.items & P::Graph;
    float y = pad;
    if (head || graph) {
        if (head) Headline(p, pad, y, d, 40.0f, false);
        const std::string target = TL("target ", "цель ") + std::to_string(d.target) + " · " + Ms(d.budget_ms);
        p.TextRight(W - pad, y + 1, 13.0f, Theme::Muted, target);
        const float load = d.budget_ms > 0.0f ? d.ms / d.budget_ms : 0.0f;
        const std::string load_text = Num(load * 100.0f) + "%";
        const float lw = p.Width(13.0f, load_text);
        const ImVec4& load_color = load > 1.05f ? Theme::Warn : Theme::Good;
        p.Text(W - pad - lw, y + 21, 13.0f, load_color, load_text);
        p.Box(W - pad - lw - 86, y + 27, W - pad - lw - 6, y + 31, Theme::White(0.14f), 0.0f);
        p.Box(W - pad - lw - 86, y + 27, W - pad - lw - 86 + 80 * std::min(load, 1.0f), y + 31, load_color, 0.0f);
        y += 40 + gap;
    }
    if (graph) {
        FrameBars(p, pad, y, inner, 56, d, true);
        y += 56 + 3;
        FramePacing(p, pad, y, inner, 24, d);
        y += 24 + gap;
        std::vector<CellItem> cells = RateCells(d);
        cells.push_back({"1%", Num(d.stats.low1_fps), RateColor(d.stats.low1_fps, d.target)});
        cells.push_back({"0.1%", Num(d.stats.low01_fps), RateColor(d.stats.low01_fps, d.target)});
        cells.push_back(StutterCell(d));
        cells.push_back({"P50", Num(d.stats.p50_ms, 1), FrameColor(d.stats.p50_ms, d.budget_ms)});
        cells.push_back({"P90", Num(d.stats.p90_ms, 1), FrameColor(d.stats.p90_ms, d.budget_ms)});
        cells.push_back({"P99", Num(d.stats.p99_ms, 1), FrameColor(d.stats.p99_ms, d.budget_ms)});
        y += GridCells(p, pad, y, inner, 3, cells) + gap;
    }
    // The busiest threads (with the CPU item), in percent of one core.
    if ((d.items & P::Cpu) && !d.sample.threads.empty()) {
        p.Text(pad, y, 11.0f, Theme::Muted, TL("threads, % of a core", "потоки, % ядра"));
        y += 15;
        for (const P::ThreadLoad& t : d.sample.threads) {
            p.Text(pad, y, 13.0f, Theme::Muted, p.Fit(13.0f, t.name, 150.0f));
            Track(p, pad + 156, y + 7, W - pad - 50, 4, t.percent / 100.0f);
            p.TextRight(W - pad, y, 14.0f, Theme::Text, Num(t.percent) + "%");
            y += 20;
        }
        y += gap - 2;
    }
    const std::vector<MeterItem> meters = Meters(d);
    for (const MeterItem& m : meters) {
        p.Text(pad, y, 13.0f, Theme::Muted, m.label);
        Track(p, pad + 46, y + 7, W - pad - 120, 4, m.fraction);
        p.TextRight(W - pad, y, 14.0f, Theme::Text, m.value);
        y += 20;
    }
    if (!meters.empty()) y += gap - 2;
    // What the renderer does: GPU memory tracking faults and pipelines made, per second.
    {
        const auto rate = [](float value) { return value < 0.0f ? std::string("n/a") : Num(value); };
        float x = pad;
        x += Pair(p, x, y, 13.0f, TL("memory faults/s", "ошибок памяти/с"), rate(d.sample.faults_per_second),
                  Theme::Text) + 14;
        x += Pair(p, x, y, 13.0f, TL("pipelines/s", "конвейеров/с"), rate(d.sample.pipelines_per_second),
                  d.sample.pipelines_per_second > 0.5f ? Theme::Warn : Theme::Text) + 6;
        p.Text(x, y, 13.0f, Theme::Muted, "(" + std::to_string(d.sample.pipelines) + ")");
        y += 17 + gap;
    }
    std::vector<Chip> chips = InfoChips(d, true);
    chips.push_back({TL("queue ", "очередь ") + std::to_string(s.frames_queued.load()), false});
    y += Chips(p, pad, y, inner, chips) + gap;
    return ImVec2(W, y - gap + pad);
}

// E: one line (the FPS, the frame time, a small bar graph, the 1% low) or two lines and a graph.
ImVec2 MiniStyle(const SavePainter& p, const OverlayData& d) {
    namespace P = TlgPerfStats;
    const float pad = 6;
    if (TlgSettings::Get().overlay_layout != 1) {
        float x = pad;
        if (d.items & P::Fps) {
            const std::string fps = Num(d.fps);
            p.TextLight(x, 0, 26.0f, RateColor(d.fps, d.target), fps);
            x += p.LightWidth(26.0f, fps) + 8;
        }
        if (d.items & P::FrameTime) {
            const std::string ms = Ms(d.ms);
            p.Text(x, 8, 14.0f, Theme::Text, ms);
            x += p.Width(14.0f, ms) + 8;
        }
        if (d.items & P::Graph) {
            FrameBars(p, x, 7, 84, 18, d, false);
            x += 84 + 8;
            x += Pair(p, x, 8, 14.0f, "1%", Num(d.stats.low1_fps), RateColor(d.stats.low1_fps, d.target)) + 8;
        }
        return ImVec2(x - 8 + pad, 32);
    }
    const float W = 250, inner = W - 2 * pad;
    float y = 4;
    if (d.items & (P::Fps | P::FrameTime | P::Graph)) {
        float x = pad;
        if (d.items & P::Fps) {
            const std::string fps = Num(d.fps);
            p.TextLight(x, y - 4, 26.0f, RateColor(d.fps, d.target), fps);
            x += p.LightWidth(26.0f, fps) + 8;
        }
        if (d.items & P::FrameTime) p.Text(x, y + 4, 14.0f, Theme::Text, Ms(d.ms));
        if (d.items & P::Graph) {
            const std::string low01 = Num(d.stats.low01_fps), low1 = Num(d.stats.low1_fps);
            PairRight(p, W - pad, y + 4, 13.0f, "0.1%", low01, RateColor(d.stats.low01_fps, d.target));
            PairRight(p, W - pad - PairWidth(p, 13.0f, "0.1%", low01) - 10, y + 4, 13.0f, "1%", low1,
                      RateColor(d.stats.low1_fps, d.target));
        }
        y += 28;
    }
    if (d.items & P::Graph) {
        FrameBars(p, pad, y, inner, 24, d, false);
        y += 24 + 5;
    }
    float x = pad;
    for (const MeterItem& m : Meters(d)) {
        if (m.label == std::string("RAM")) continue;
        x += Pair(p, x, y, 12.0f, m.label, m.short_value, Theme::Text) + 10;
    }
    p.TextRight(W - pad, y, 12.0f, Theme::Accent, RendererName, true);
    y += 16;
    return ImVec2(W, y + 4);
}

// Draws the overlay in its square style with the top-left corner at `at`; returns its size in
// display pixels (zero when nothing is shown). The panel is drawn first as a rectangle of 4
// vertices (PrimRect: a, (b.x, a.y), b, (a.x, b.y)) and given its size once the items are laid out.
ImVec2 DrawOverlay(ImDrawList* dl, ImVec2 at, float k) {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const OverlayData d = CollectOverlayData();
    if (!(d.items & P::AllItems)) {
        return {};
    }
    const SavePainter p{dl, ImGui::GetFont(), k, at};
    const float opacity = s.overlay_opacity / 100.0f;
    const int panel = dl->VtxBuffer.Size;
    if (opacity > 0.0f) p.Box(0, 0, 1, 1, Theme::Glass, 0.0f, Theme::GlassAlpha * opacity);
    const bool placeholder = dl->VtxBuffer.Size == panel + 4; // none when the fill's alpha rounds to 0
    ImVec2 size;
    switch (s.overlay_style) {
    case TlgSettings::OverlayStrip: size = StripStyle(p, d); break;
    case TlgSettings::OverlayHistogram: size = HistogramStyle(p, d); break;
    case TlgSettings::OverlayTelemetry: size = TelemetryStyle(p, d); break;
    case TlgSettings::OverlayMini: size = MiniStyle(p, d); break;
    default: size = CompactStyle(p, d); break;
    }
    if (placeholder) {
        ImDrawVert* v = dl->VtxBuffer.Data + panel;
        const ImVec2 a = p.P(0, 0), b = p.P(size.x, size.y);
        v[0].pos = a;
        v[1].pos = ImVec2(b.x, a.y);
        v[2].pos = b;
        v[3].pos = ImVec2(a.x, b.y);
        p.Frame(0, 0, size.x, size.y, Theme::White(0.08f), 0.0f, 1.0f, opacity);
    }
    return ImVec2(size.x * k, size.y * k);
}

// The performance overlay: the card of earlier versions, or a square style drawn at the screen's
// corner and moved to its place once its size is known.
void FpsCounter() {
    const auto& s = TlgSettings::Get();
    if (s.overlay_style == TlgSettings::OverlayCard) {
        CardOverlay();
        return;
    }
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 origin = ImGui::GetMainViewport()->WorkPos;
    const int first = dl->VtxBuffer.Size;
    const ImVec2 size = DrawOverlay(dl, origin, base_scale * s.overlay_scale / 100.0f);
    if (size.x <= 0.0f) {
        return;
    }
    const ImVec2 at = OverlayPosition(size);
    const ImVec2 shift(at.x - origin.x, at.y - origin.y);
    for (int i = first; i < dl->VtxBuffer.Size; ++i) {
        dl->VtxBuffer[i].pos.x += shift.x;
        dl->VtxBuffer[i].pos.y += shift.y;
    }
}

std::string notice_text;
std::chrono::steady_clock::time_point notice_until;

void NoticeBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k = viewport->WorkSize.y / 1080.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    const float w = std::min(p.Width(23.0f, notice_text) + 56, viewport->WorkSize.x / k - 80), h = 72;
    p.at = ImVec2(viewport->WorkPos.x + (viewport->WorkSize.x - w * k) / 2, viewport->WorkPos.y + 40 * k);
    const bool frosted = Frost(p.dl, p.P(0, 0), p.P(w, h), 16.0f * k);
    p.Box(0, 0, w, h, ImVec4(32 / 255.0f, 32 / 255.0f, 35 / 255.0f, 1.0f), 16.0f, frosted ? 0.72f : 0.95f);
    p.Text(28, 21, 23.0f, Theme::Text, p.Fit(23.0f, notice_text, w - 56));
}

// The mouse's touchpad (touchpad mode): the pad as the game sees it and the finger on it.
std::atomic<bool> mouse_pad_on{false}, mouse_pad_touch{false}, mouse_pad_click{false};
std::atomic<float> mouse_pad_x{0.5f}, mouse_pad_y{0.5f};

void MousePadBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float w = 240.0f * base_scale, h = w * 942.0f / 1919.0f;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 24.0f * base_scale,
                                   viewport->WorkPos.y + viewport->WorkSize.y - 24.0f * base_scale),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.45f);
    ImGui::SetNextWindowSize(ImVec2(w + 16.0f * base_scale, h + 16.0f * base_scale));
    ImGui::Begin("##mousepad", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings);
    const ImVec2 origin = ImVec2(ImGui::GetWindowPos().x + 8.0f * base_scale, ImGui::GetWindowPos().y + 8.0f * base_scale);
    ImDrawList* list = ImGui::GetWindowDrawList();
    const bool click = mouse_pad_click, touch = mouse_pad_touch;
    const float radius = Theme::ItemRadius * base_scale;
    list->AddRectFilled(origin, ImVec2(origin.x + w, origin.y + h), Theme::U32(Theme::Ink, 0.35f), radius);
    list->AddRect(origin, ImVec2(origin.x + w, origin.y + h), click ? Theme::U32(Theme::Warn) : Theme::U32(Theme::Muted),
                  radius, 0, 2.0f * base_scale);
    const ImVec2 finger(origin.x + std::clamp(mouse_pad_x.load(), 0.0f, 1.0f) * w,
                        origin.y + std::clamp(mouse_pad_y.load(), 0.0f, 1.0f) * h);
    const float r = 7.0f * base_scale;
    if (touch) {
        list->AddCircleFilled(finger, r, click ? Theme::U32(Theme::Warn) : Theme::U32(Theme::Accent));
    } else {
        list->AddCircle(finger, r, Theme::U32(Theme::White(0.86f)), 0, 2.0f * base_scale);
    }
    ImGui::End();
}

void BackgroundLoading() {
    const u32 done = background_done, total = std::max(background_total.load(), 1u);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k = viewport->WorkSize.y / 1080.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    char text[96];
    std::snprintf(text, sizeof(text), TL("Loading shader cache %u%%", "Загрузка кэша шейдеров %u%%"), u32(u64(done) * 100 / total));
    const float w = p.Width(17.0f, text) + 40, h = 54;
    p.at = ImVec2(viewport->WorkPos.x + 24 * k, viewport->WorkPos.y + viewport->WorkSize.y - (24 + h) * k);
    p.Box(0, 0, w, h, ImVec4(28 / 255.0f, 28 / 255.0f, 30 / 255.0f, 1.0f), 12.0f, 0.78f);
    p.Text(20, 10, 17.0f, Theme::Muted, text);
    p.Box(20, 38, w - 20, 42, Theme::White(0.12f), 2.0f);
    p.Box(20, 38, 20 + (w - 40) * std::min(1.0f, float(done) / float(total)), 42, Theme::Accent, 2.0f);
}

} // namespace

void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count) {
    std::scoped_lock lock{imgui_mutex};
    if (initialized) {
        return;
    }
    IMGUI_CHECKVERSION();
    ImGuiContext* const previous_context = ImGui::GetCurrentContext();
    overlay_context = ImGui::CreateContext();
    ImGui::SetCurrentContext(overlay_context);
#ifdef TLG_RENDERER_HOSTED
    // The renderer's own context stays current for its presenter.
    struct Restore { ImGuiContext* c; ~Restore() { ImGui::SetCurrentContext(c); } } restore{previous_context};
#else
    (void)previous_context;
#endif
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // window positions are not kept
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.BackendPlatformName = "tlg";

    // The launcher's look: grey see-through panels with a faint border, small radii, white
    // text, blue accent.
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = Theme::PanelRadius;
    style.ChildRounding = Theme::PanelRadius;
    style.FrameRounding = Theme::ItemRadius;
    style.PopupRounding = Theme::PanelRadius;
    style.GrabRounding = Theme::ItemRadius;
    style.TabRounding = Theme::ItemRadius;
    style.ScrollbarRounding = Theme::ItemRadius;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.WindowPadding = ImVec2(16.0f, 12.0f);
    style.FramePadding = ImVec2(10.0f, 6.0f);
    style.ItemSpacing = ImVec2(10.0f, 8.0f);
    const auto white = [](float alpha) { return ImVec4(1.0f, 1.0f, 1.0f, alpha); };
    const auto accent = [](float alpha) {
        return ImVec4(Theme::Accent.x, Theme::Accent.y, Theme::Accent.z, alpha);
    };
    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = Theme::Text;
    c[ImGuiCol_TextDisabled] = Theme::Muted;
    c[ImGuiCol_WindowBg] = Theme::Fade(Theme::Glass, Theme::GlassAlpha);
    c[ImGuiCol_ChildBg] = white(0.03f);
    c[ImGuiCol_PopupBg] = Theme::Fade(Theme::Popover, Theme::PopoverAlpha);
    c[ImGuiCol_PlotHistogram] = accent(1.0f); // progress bars (loading screen)
    c[ImGuiCol_PlotHistogramHovered] = accent(1.0f);
    c[ImGuiCol_TextSelectedBg] = accent(0.35f);
    c[ImGuiCol_Border] = white(0.1f);
    c[ImGuiCol_FrameBg] = white(0.07f);
    c[ImGuiCol_FrameBgHovered] = white(0.12f);
    c[ImGuiCol_FrameBgActive] = white(0.16f);
    c[ImGuiCol_TitleBg] = ImVec4(0.05f, 0.05f, 0.07f, 0.95f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.1f, 0.97f);
    c[ImGuiCol_CheckMark] = accent(1.0f);
    c[ImGuiCol_SliderGrab] = accent(0.85f);
    c[ImGuiCol_SliderGrabActive] = accent(1.0f);
    c[ImGuiCol_Button] = white(0.08f);
    c[ImGuiCol_ButtonHovered] = white(0.16f);
    c[ImGuiCol_ButtonActive] = accent(0.6f);
    c[ImGuiCol_Header] = white(0.08f);
    c[ImGuiCol_HeaderHovered] = white(0.14f);
    c[ImGuiCol_HeaderActive] = accent(0.45f);
    c[ImGuiCol_Separator] = white(0.1f);
    c[ImGuiCol_Tab] = white(0.05f);
    c[ImGuiCol_TabHovered] = white(0.14f);
    c[ImGuiCol_TabSelected] = accent(0.55f);
    c[ImGuiCol_TabSelectedOverline] = accent(1.0f);
    c[ImGuiCol_ScrollbarBg] = white(0.02f);
    c[ImGuiCol_ScrollbarGrab] = white(0.16f);
    c[ImGuiCol_NavCursor] = accent(1.0f);
    c[ImGuiCol_TableHeaderBg] = white(0.05f);
    c[ImGuiCol_TableBorderStrong] = white(0.08f);
    c[ImGuiCol_TableBorderLight] = white(0.05f);
    base_style = style;

    // The launcher's font (Segoe UI, every Windows has it; Cyrillic included), with the built-in
    // DejaVu Sans merged in for the glyphs it lacks (✕); else DejaVu alone. Semibold Segoe UI is
    // the save dialogs' bold. Glyphs are rasterized on demand (dynamic atlas), so no ranges.
    const auto add_file = [&io](const std::string& path, float size) -> ImFont* {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return nullptr;
        std::fclose(f);
        return io.Fonts->AddFontFromFileTTF(path.c_str(), size);
    };
    const auto add_dejavu = [&io](bool merge) {
        ImFontConfig font_config;
        font_config.FontDataOwnedByAtlas = false;
        font_config.MergeMode = merge;
        return io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(tlg_font_ttf),
                                              int(tlg_font_ttf_end - tlg_font_ttf), 18.0f, &font_config);
    };
    ImFont* font = nullptr;
#ifdef _WIN32
    // FL_FONTS=bundled: the port's faces on Windows too (the Linux look, as in the launcher).
    const char* fl_fonts = std::getenv("FL_FONTS");
    const bool bundled = fl_fonts && std::strcmp(fl_fonts, "bundled") == 0;
    // getenv, not GetEnvironmentVariableA: renderer v2 builds this file without <windows.h>.
    const char* windir = std::getenv("WINDIR");
    const std::string fonts = std::string(windir && *windir ? windir : "C:\\Windows") + "\\Fonts\\";
    for (const char* name : {"segoeui.ttf", "segoeuisl.ttf"}) {
        if (bundled) break;
        if ((font = add_file(fonts + name, 19.0f))) {
            add_dejavu(true);
            break;
        }
    }
    if (font) {
        font_bold = add_file(fonts + "segoeuisb.ttf", 19.0f);
        if (font_bold) add_dejavu(true);
        font_light = add_file(fonts + "segoeuil.ttf", 25.0f);
        if (font_light) add_dejavu(true);
    }
    // Segoe Fluent Icons (Windows 11), else Segoe MDL2 Assets (Windows 10): the launcher's glyphs.
    for (const char* name : {"SegoeIcons.ttf", "segmdl2.ttf"}) {
        if (bundled) break;
        if ((font_icons = add_file(fonts + name, 24.0f))) break;
    }
#endif
    // The port's own faces (run.py sets TLG_FONT_DIR to port/fonts): Open Sans, Cyrillic included,
    // and the launcher's Fluent icons.
    if (const char* dir = std::getenv("TLG_FONT_DIR"); dir && *dir) {
        const std::string own = std::string(dir) + "/";
        if (!font && (font = add_file(own + "OpenSans-Regular.ttf", 19.0f))) {
            add_dejavu(true);
            if ((font_bold = add_file(own + "OpenSans-SemiBold.ttf", 19.0f))) add_dejavu(true);
            if ((font_light = add_file(own + "OpenSans-Light.ttf", 25.0f))) add_dejavu(true);
        }
        if (!font_icons && (font_icons = add_file(own + "FluentSystemIcons-Subset.ttf", 24.0f))) {
            icons_fluent = true;
        }
    }
    if (!font) {
        font = add_dejavu(false);
    }
    io.FontDefault = font;

    const vk::Instance vk_instance = instance.GetInstance();
    ImGui_ImplVulkan_LoadFunctions(
        instance.ApiVersion(),
        [](const char* name, void* user) {
            return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
                *static_cast<const vk::Instance*>(user), name);
        },
        const_cast<vk::Instance*>(&vk_instance));

    const VkFormat color_format = static_cast<VkFormat>(format);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = instance.ApiVersion();
    info.Instance = vk_instance;
    info.PhysicalDevice = instance.GetPhysicalDevice();
    info.Device = instance.GetDevice();
    info.QueueFamily = instance.GetGraphicsQueueFamilyIndex();
    info.Queue = instance.GetGraphicsQueue();
    // Font atlas plus the save list pictures (up to 33, and a closed list's until freed).
    info.DescriptorPoolSize = 128;
    info.MinImageCount = std::max(image_count, 2u);
    info.ImageCount = std::max(image_count, 2u);
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    if (!ImGui_ImplVulkan_Init(&info)) {
        std::printf("Overlay: ImGui Vulkan backend init failed\n");
        ImGui::DestroyContext();
        return;
    }
    overlay_instance = &instance;
    initialized = true;
    std::printf("Overlay: menu ready (Insert or R3+L2)\n");
}

void UpdateTextInput(SDL_Window* window) {
    if (!overlay_context) return; // before Init
    bool want = false;
    {
        std::scoped_lock lock{imgui_mutex};
        want = initialized && menu_open && overlay_context->IO.WantTextInput;
    }
    if (want != SDL_TextInputActive(window)) {
        if (want) {
            SDL_StartTextInput(window);
        } else {
            SDL_StopTextInput(window);
        }
    }
}

// imgui_mutex held. The backend frees a texture some frames after its last use.
void RetireChoiceTextures() {
    for (ImTextureData* tex : choice_textures) {
        if (tex) {
            tex->WantDestroyNextFrame = true;
            choice_retired.push_back(tex);
        }
    }
    choice_textures.clear();
    choice_pictures.clear();
    choice_retiring = !choice_retired.empty();
}

// imgui_mutex held, before NewFrame: new pictures become textures, destroyed ones are freed.
void UpdateChoiceTextures() {
    for (auto& [id, picture] : trophy_pictures) {
        if (trophy_textures.count(id)) continue;
        if (picture.rgba.empty()) {
            trophy_textures[id] = nullptr; // no icon: not decoded again
            continue;
        }
        ImTextureData* tex = IM_NEW(ImTextureData)();
        tex->Create(ImTextureFormat_RGBA32, picture.width, picture.height);
        std::memcpy(tex->GetPixels(), picture.rgba.data(), picture.rgba.size());
        ImGui::RegisterUserTexture(tex);
        trophy_textures[id] = tex;
    }
    trophy_pictures.clear();
    if (!choice_pictures.empty()) {
        choice_textures.assign(choice_pictures.size(), nullptr);
        for (size_t i = 0; i < choice_pictures.size(); ++i) {
            const TlgPng::Image& picture = choice_pictures[i];
            if (picture.rgba.empty()) {
                continue;
            }
            ImTextureData* tex = IM_NEW(ImTextureData)();
            tex->Create(ImTextureFormat_RGBA32, picture.width, picture.height);
            std::memcpy(tex->GetPixels(), picture.rgba.data(), picture.rgba.size());
            ImGui::RegisterUserTexture(tex);
            choice_textures[i] = tex;
        }
        choice_pictures.clear();
    }
    std::erase_if(choice_retired, [](ImTextureData* tex) {
        if (tex->Status != ImTextureStatus_Destroyed) {
            return false;
        }
        ImGui::UnregisterUserTexture(tex);
        IM_DELETE(tex);
        return true;
    });
    choice_retiring = !choice_retired.empty();
}

// The guide takes the buttons, keys and mouse until it is closed (its button, cross/A, Enter,
// Esc, circle/B); the game's input is held neutral meanwhile.
bool HandleGuideEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    bool close = false;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN: {
        const SDL_Keycode key = event.key.key;
        close = key == SDLK_RETURN || key == SDLK_KP_ENTER || key == SDLK_SPACE || key == SDLK_ESCAPE;
        break;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        close = event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH || event.gbutton.button == SDL_GAMEPAD_BUTTON_EAST;
        break;
    case SDL_EVENT_MOUSE_MOTION: {
        const float density = PixelDensity(event.motion.windowID);
        overlay_context->IO.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const ImVec2 at = overlay_context->IO.MousePos;
        close = event.button.button == SDL_BUTTON_LEFT && guide_button_rect.GetWidth() > 0 &&
                guide_button_rect.Contains(at);
        break;
    }
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_WHEEL:
        return true;
    default:
        return false;
    }
    if (close) {
        CloseGuide();
    }
    return true;
}

void FinishSaveDialog() {
    RetireChoiceTextures();
    SetDrawCursor(menu_open);
}

// The save list and messages take the keys, buttons and mouse that answer them; the rest stays
// with the game (held neutral by CapturesInput).
bool HandleSaveEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    int move = 0, finish = 0; // finish: 1 accept, 2 back
    bool moved = false, click = false;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN: {
        const SDL_Keycode key = event.key.key;
        move = key == SDLK_UP || key == SDLK_W || key == SDLK_LEFT || key == SDLK_A ? -1
               : key == SDLK_DOWN || key == SDLK_S || key == SDLK_RIGHT || key == SDLK_D ? 1
                                                                                      : 0;
        finish = key == SDLK_RETURN || key == SDLK_KP_ENTER || key == SDLK_SPACE ? 1
                 : key == SDLK_ESCAPE || key == SDLK_BACKSPACE                   ? 2
                                                                                 : 0;
        break;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: {
        const u8 button = event.gbutton.button;
        move = button == SDL_GAMEPAD_BUTTON_DPAD_UP || button == SDL_GAMEPAD_BUTTON_DPAD_LEFT ? -1
               : button == SDL_GAMEPAD_BUTTON_DPAD_DOWN || button == SDL_GAMEPAD_BUTTON_DPAD_RIGHT ? 1
                                                                                                  : 0;
        finish = button == SDL_GAMEPAD_BUTTON_SOUTH ? 1 : button == SDL_GAMEPAD_BUTTON_EAST ? 2 : 0;
        break;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        const float density = PixelDensity(event.motion.windowID);
        save_mouse = ImVec2(event.motion.x * density, event.motion.y * density);
        overlay_context->IO.AddMousePosEvent(save_mouse.x, save_mouse.y);
        moved = true;
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        click = event.button.button == SDL_BUTTON_LEFT;
        finish = event.button.button == SDL_BUTTON_RIGHT ? 2 : 0;
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        move = event.wheel.y > 0 ? -1 : event.wheel.y < 0 ? 1 : 0;
        break;
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        return true;
    default:
        return false;
    }
    if (message_active) {
        const bool two = (message_buttons >= 0 ? message_buttons
                          : SaveMessageFor(save_kind, message_code, message_value, message_text).buttons) % 2 == 1;
        if (two && move) message_focus = move < 0 ? 0 : 1;
        for (int b = 0; b < 2 && (moved || click); ++b) {
            if (save_button_rects[b].GetWidth() > 0 && save_button_rects[b].Contains(save_mouse)) {
                message_focus = b;
                finish = click ? 1 : finish;
            }
        }
        if (finish) {
            message_result = !two || (finish == 1 && message_focus == 1) ? 1 : 2;
            message_active = false;
            FinishSaveDialog();
        }
        return true;
    }
    const int count = int(save_rows.size());
    if (move && count) {
        choice_index = (choice_index + move + count) % count;
    }
    for (int i = 0; i < int(save_row_rects.size()) && (moved || click); ++i) {
        if (save_row_rects[i].GetWidth() > 0 && save_row_rects[i].Contains(save_mouse)) {
            choice_index = i;
            finish = click ? 1 : finish;
        }
    }
    if (finish) {
        choice_result = finish == 1 && count ? choice_index : -2;
        choice_active = false;
        FinishSaveDialog();
    }
    return true;
}

bool HandleEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    std::scoped_lock lock{imgui_mutex};
#ifdef TLG_RENDERER_V3
    // Renderer v3 has no ImGui context of its own: the input goes to the overlay's. Not in v2:
    // ImGui's current context is one for the process, and v2's present thread draws its own
    // context's frame meanwhile (switching it here broke v2's NewFrame: the font atlas assert).
    const ContextScope context_scope;
#endif
    if (!initialized) {
        return false;
    }
    if (guide_active && !menu_open) {
        return HandleGuideEvent(event);
    }
    if ((choice_active || message_active) && !menu_open) {
        return HandleSaveEvent(event);
    }
    ImGuiIO& io = overlay_context->IO; // not made current: other threads draw with ImGui
    const bool is_open = menu_open;
    if (!is_open) {
        key_capture = -1; // the menu closed while a key row waited
    } else if (CaptureKey(event)) {
        return true;
    }
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = event.type == SDL_EVENT_KEY_DOWN;
        static const bool capture_key = [] {
            const char* env = std::getenv("TLG_FRAME_CAPTURE_KEY");
            return env && *env == '1';
        }();
        if (down && !event.key.repeat && capture_key && event.key.key == SDLK_F11) {
#ifndef TLG_RENDERER_HOSTED
            Vulkan::FrameCapture::Request();
#endif
            return true;
        }
        // F12 arms the frame recorder (vk_frame_record.h) by creating its trigger file, for
        // effects that only show while the player holds a button.
        static const char* record_trigger = std::getenv("TLG_FRAME_RECORD_TRIGGER");
        if (down && !event.key.repeat && record_trigger && *record_trigger &&
            event.key.key == SDLK_F12) {
            if (std::FILE* f = std::fopen(record_trigger, "wb")) {
                std::fclose(f);
                std::printf("Frame record: F12 created the trigger\n");
            }
            return true;
        }
        // tlg: F10 shows or hides the performance overlay (FPS counter) at any time.
        if (down && !event.key.repeat && event.key.key == SDLK_F10) {
            auto& settings = TlgSettings::Get();
            settings.show_fps = !settings.show_fps;
            return true;
        }
        if (down && !event.key.repeat &&
            (event.key.key == SDLK_INSERT || (is_open && event.key.key == SDLK_ESCAPE))) {
            SetOpen(event.key.key == SDLK_INSERT ? !is_open : false);
            return true;
        }
        if (!is_open) {
            return false;
        }
        io.AddKeyEvent(ImGuiMod_Ctrl, (event.key.mod & SDL_KMOD_CTRL) != 0);
        io.AddKeyEvent(ImGuiMod_Shift, (event.key.mod & SDL_KMOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiMod_Alt, (event.key.mod & SDL_KMOD_ALT) != 0);
        if (const ImGuiKey key = KeyFromSdl(event.key.key); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        const bool down = event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        const u8 button = event.gbutton.button;
        // R3+L2 opens the menu: L3+R3 is a game action in God of War III.
        if (button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
            r3_down = down;
            if (down && l2_down) {
                SetOpen(!is_open);
                return true;
            }
        }
        if (!is_open) {
            return false;
        }
        if (const ImGuiKey key = KeyFromGamepad(button); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
            const bool down = event.gaxis.value > 16000;
            if (down && !l2_down && r3_down) {
                l2_down = true;
                SetOpen(!is_open);
                return true;
            }
            l2_down = down;
        }
        return false;
    }
    case SDL_EVENT_TEXT_INPUT: {
        // Typed characters (Ctrl+click on a slider, a text field): key events alone erase but
        // do not type. SDL sends them while text input is on (UpdateTextInput).
        if (!is_open) {
            return false;
        }
        io.AddInputCharactersUTF8(event.text.text);
        return true;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        if (!is_open) {
            return false;
        }
        const float density = PixelDensity(event.motion.windowID);
        io.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (!is_open) {
            return false;
        }
        const int button = event.button.button == SDL_BUTTON_LEFT    ? 0
                           : event.button.button == SDL_BUTTON_RIGHT  ? 1
                           : event.button.button == SDL_BUTTON_MIDDLE ? 2
                                                                      : -1;
        if (button >= 0) {
            io.AddMouseButtonEvent(button, event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        if (!is_open) {
            return false;
        }
        io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);
        return true;
    default:
        return false;
    }
}

// After "Apply and restart", the new options are kept once the game has shown about ten
// seconds of frames after loading (run.py restores the previous ones otherwise).
void ConfirmRestart() {
    static u32 frames = 0;
    auto& s = TlgSettings::Get();
    if (!s.restart_unconfirmed || loading || ++frames < 600) {
        return;
    }
    s.restart_unconfirmed = false;
    TlgSettings::Save();
    std::puts("Settings: graphics options confirmed after restart");
}

void SetBackgroundLoading(u32 done, u32 total) {
    background_done = done;
    background_total = total;
}

void SetLoadingCompile(bool compile) {
    if (compile && !loading_compile) {
        loading_compile_started = std::chrono::steady_clock::now();
    }
    loading_compile = compile;
}

void SetLoading(bool active, u32 done, u32 total) {
    loading_done = done;
    loading_total = total;
    loading = active;
}

void Notice(const std::string& text, float seconds) {
    std::scoped_lock lock{imgui_mutex};
    notice_text = text;
    notice_until = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(static_cast<int>(seconds * 1000.0f));
}

static bool NoticeActive() {
    return std::chrono::steady_clock::now() < notice_until;
}

void SetMousePad(bool active, float x, float y, bool touching, bool clicking) {
    mouse_pad_x = x;
    mouse_pad_y = y;
    mouse_pad_touch = touching;
    mouse_pad_click = clicking;
    mouse_pad_on = active;
}

bool Visible() {
    const auto& s = TlgSettings::Get();
    const auto now = std::chrono::steady_clock::now();
    return initialized && (loading || background_total || menu_open || text_entry_active || choice_active ||
                           message_active || guide_active || progress_on || now < progress_until || now < toast_until ||
                           NoticeActive() || mouse_pad_on || trophy_toast >= 0 || !trophy_queue.empty() ||
                           choice_retiring || s.show_fps ||
                           (s.actor_watch_supported && s.enemy_health_bar));
}

bool CapturesInput() {
    return menu_open || text_entry_active || choice_active || message_active || guide_active;
}

void ShowGraffitiGuide(bool force) {
    std::scoped_lock lock{imgui_mutex};
    if (!force && TlgSettings::Get().graffiti_guide_seen) {
        return;
    }
    OpenGuide();
}

void BeginSaveList(int kind, std::vector<SaveRow> rows, std::vector<TlgPng::Image> pictures, int focus) {
    std::scoped_lock lock{imgui_mutex};
    RetireChoiceTextures();
    choice_pictures = std::move(pictures);
    choice_pictures.resize(rows.size());
    save_kind = kind;
    save_rows = std::move(rows);
    save_row_rects.clear();
    choice_index = focus >= 0 && focus < int(save_rows.size()) ? focus : 0;
    choice_first = 0;
    choice_result = -1;
    message_active = false;
    progress_on = false;
    choice_active = true;
    overlay_context->IO.MouseDrawCursor = true;
}

int PollSaveList() {
    return choice_result;
}

void BeginSaveMessage(int kind, int code, u64 value, const std::string& text, int buttons,
                      const SaveRow* entry, TlgPng::Image picture) {
    std::scoped_lock lock{imgui_mutex};
    RetireChoiceTextures();
    message_has_entry = entry != nullptr;
    if (entry) {
        message_entry = *entry;
        choice_pictures.clear();
        choice_pictures.push_back(std::move(picture));
    }
    save_kind = kind;
    message_code = code;
    message_value = value;
    message_text = text;
    message_buttons = buttons;
    // Deleting starts on "No"; everything else on the answer the player came for.
    message_focus = kind == 3 ? 0 : 1;
    save_button_rects[0] = save_button_rects[1] = ImRect();
    message_result = -1;
    choice_active = false;
    progress_on = false;
    message_active = true;
    overlay_context->IO.MouseDrawCursor = true;
}

int PollSaveMessage() {
    return message_result;
}

void CloseSaveDialog() {
    std::scoped_lock lock{imgui_mutex};
    if (choice_active.exchange(false)) {
        choice_result = -2;
    }
    if (message_active.exchange(false)) {
        message_result = 2;
    }
    FinishSaveDialog();
}

void SaveProgress(int kind, int active, int percent, const char* text) {
    std::scoped_lock lock{imgui_mutex};
    const auto now = SaveClock::now();
    if (!active) {
        progress_on = false;
        return;
    }
    if (!progress_on && now >= progress_until) {
        progress_since = now;
    }
    progress_kind = kind;
    progress_percent = percent;
    if (text) {
        progress_text = text;
    }
    if (active == 2) {
        progress_until = now + std::chrono::milliseconds(1200);
    } else {
        progress_on = true;
    }
}

void ShowTrophy(int id) {
    TlgPng::Image picture;
    TlgTrophies::Icon(id, picture); // decoded on the caller's thread, outside the lock
    const TlgTrophies::Info* info = TlgTrophies::Describe(id);
    TlgSound::Prepare(info && info->grade == TlgTrophies::Platinum); // the file and the audio stream, here too
    std::scoped_lock lock{imgui_mutex};
    if (!trophy_textures.count(id)) trophy_pictures.emplace_back(id, std::move(picture));
    trophy_queue.push_back(id);
}

void SaveToast(int kind) {
    std::scoped_lock lock{imgui_mutex};
    const auto now = SaveClock::now();
    // A dialog's "saved" and the save's own unmount come together: one toast.
    if (now < toast_until + std::chrono::seconds(3)) {
        toast_until = std::max(toast_until, now + std::chrono::milliseconds(2600));
        return;
    }
    toast_kind = kind;
    toast_since = now;
    toast_until = now + std::chrono::milliseconds(2600);
}

void SetTextEntry(bool active, const std::string& prompt, const std::string& text) {
    std::scoped_lock lock{imgui_mutex};
    text_entry_active = active;
    text_entry_prompt = prompt;
    text_entry_text = text;
}

void SetBackdrop(vk::ImageView view, vk::ImageLayout layout, vk::Rect2D where) {
    std::scoped_lock lock{imgui_mutex};
    backdrop_view = static_cast<VkImageView>(view);
    backdrop_layout = static_cast<VkImageLayout>(layout);
    backdrop_where = ImRect(float(where.offset.x), float(where.offset.y), float(where.offset.x + s32(where.extent.width)),
                            float(where.offset.y + s32(where.extent.height)));
}

void ForgetBackdrop(vk::ImageView view) {
    const ContextScope context_scope;
    std::scoped_lock lock{imgui_mutex};
    for (auto it = backdrop_sets.begin(); it != backdrop_sets.end();) {
        if (it->view == static_cast<VkImageView>(view)) {
            ImGui_ImplVulkan_RemoveTexture(it->set);
            it = backdrop_sets.erase(it);
        } else {
            ++it;
        }
    }
    if (backdrop_view == static_cast<VkImageView>(view)) backdrop_view = VK_NULL_HANDLE;
}

void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent) {
    const ContextScope context_scope;
    // Present interval for the FPS readout (measured also while nothing is drawn).
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - last_present).count();
    last_present = now;
    if (ms > 0.0f && ms < 1000.0f) {
        frame_ms_avg = frame_ms_avg == 0.0f ? ms : frame_ms_avg * 0.95f + ms * 0.05f;
        frame_history[frame_history_next] = ms;
        frame_history_next = (frame_history_next + 1) % frame_history.size();
        frame_history_count = std::min(frame_history_count + 1, frame_history.size());
        frame_long[frame_long_next] = ms;
        frame_long_next = (frame_long_next + 1) % frame_long.size();
        frame_long_count = std::min(frame_long_count + 1, frame_long.size());
    }
    ConfirmRestart();
    std::unique_lock lock{imgui_mutex};
    // This present's frame for the frosted panels: its descriptor, made once per view.
    backdrop_ready = false;
    if (backdrop_view != VK_NULL_HANDLE && initialized) {
        auto it = std::find_if(backdrop_sets.begin(), backdrop_sets.end(), [](const BackdropSet& b) {
            return b.view == backdrop_view && b.layout == backdrop_layout;
        });
        if (it == backdrop_sets.end()) {
            const VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(backdrop_view, backdrop_layout);
            if (set != VK_NULL_HANDLE) it = backdrop_sets.insert(backdrop_sets.end(), {backdrop_view, backdrop_layout, set});
        }
        if (it != backdrop_sets.end()) {
            backdrop_texture = ImTextureRef(ImTextureID(it->set));
            backdrop_ready = true;
        }
    }
    backdrop_view = VK_NULL_HANDLE; // each present gives its own
    if (!Visible()) {
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(extent.width), float(extent.height));
    io.DeltaTime = ms > 0.0f && ms < 1000.0f ? ms / 1000.0f : 1.0f / 60.0f;
    // UI scale follows the display height (1080p = 1).
    const float scale = std::max(float(extent.height) / 1080.0f, 0.6f);
    if (std::abs(scale - base_scale) > 0.01f) {
        ImGuiStyle& style = ImGui::GetStyle();
        style = base_style;
        style.ScaleAllSizes(scale);
        style.FontScaleMain = scale;
        base_scale = scale;
    }

    UpdateChoiceTextures();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (loading) {
        LoadingScreen();
    }
    if (menu_open) {
        Menu();
    }
    {
        // The overlay asks the sampler for the busiest threads only while the telemetry style
        // shows them (in the game or in the menu's preview).
        const auto& settings = TlgSettings::Get();
        TlgPerfStats::want_threads.store((settings.show_fps || menu_open) &&
                                             settings.overlay_style == TlgSettings::OverlayTelemetry &&
                                             (settings.overlay_items & TlgPerfStats::Cpu),
                                         std::memory_order_relaxed);
    }
    if (TlgSettings::Get().show_fps && !menu_open) {
        FpsCounter();
    }
    if (!loading && !menu_open) {
        HealthDisplay();
    }
    if (background_total && !loading && !menu_open) {
        BackgroundLoading();
    }
    if (text_entry_active) {
        TextEntryBox();
    }
    TrophyToast();
    if (!menu_open) {
        SaveDialogs();
    }
    if (NoticeActive() && !menu_open) {
        NoticeBox();
    }
    if (mouse_pad_on && !menu_open) {
        MousePadBox();
    }
    ImGui::Render();

    const vk::RenderingAttachmentInfo attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    {
        // Font atlas uploads submit to the graphics queue themselves.
        std::scoped_lock submit_lock{Vulkan::Scheduler::submit_mutex};
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdbuf);
    }
    cmdbuf.endRendering();
}

} // namespace TlgOverlay
