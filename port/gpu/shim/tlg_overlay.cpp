#include "tlg_perf_trace.h"
// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_overlay.h"

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
#include <mutex>
#include <vector>

#include <SDL3/SDL.h>
#include "tlg_settings.h"
#include "tlg_orbs.h"
#include "tlg_actors.h"
#include "tlg_graphics.h"
#include "tlg_perf_stats.h"
#include "../../src/actor_hook.h"

extern TlgActorSlot tlg_actor_slots[2];
extern "C" void runtime_restart(void); // src/probe.c: starts run.py again, ends this process
#include "common/elf_info.h"
#include "common/backports.h"
#include "shader_recompiler/recompiler_toggles.h"
#include "imgui.h"
#include "imgui_internal.h" // RegisterUserTexture (save pictures)
#include "imgui_impl_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_frame_capture.h"

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
std::atomic<bool> choice_retiring{false};
float base_scale = 1.0f;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;
// The last present intervals, oldest first from frame_history_next (the overlay's graph).
std::array<float, 240> frame_history{};
size_t frame_history_next = 0, frame_history_count = 0;

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

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    ImGui::GetIO().MouseDrawCursor = value;
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

void Checkbox(const char* label, std::atomic<bool>& value) {
    bool v = value;
    const bool changed = ImGui::Checkbox(label, &v);
    Store(value, v, changed);
}

// A combo over `count` labels storing an index or a value from `values`.
void Choice(const char* label, std::atomic<int>& target, const char* const* labels, const int* values,
            int count) {
    int current = 0;
    for (int i = 0; i < count; ++i) {
        if ((values ? values[i] : i) == target.load()) current = i;
    }
    if (ImGui::Combo(label, &current, labels, count)) {
        Store(target, values ? values[current] : current, true);
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

// Two-column rows: the name on the left, the control on the right.
bool BeginRows(const char* id) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) {
        return false;
    }
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 0.52f);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.48f);
    return true;
}

void Row(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::PushID(label);
}

void CheckRow(const char* label, std::atomic<bool>& value, const char* help = nullptr) {
    Row(label);
    Checkbox("##value", value);
    Help(help);
    ImGui::PopID();
}

void ChoiceRow(const char* label, std::atomic<int>& target, const char* const* labels, const int* values,
               int count, const char* help = nullptr) {
    Row(label);
    Choice("##value", target, labels, values, count);
    Help(help);
    ImGui::PopID();
}

// A 0.1x to 100x multiplier with a 1x button.
void MultiplierRow(const char* label, std::atomic<float>& value, const char* help) {
    Row(label);
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("1x").x + style.FramePadding.x * 2 + style.ItemSpacing.x));
    float v = value;
    if (ImGui::SliderFloat("##value", &v, 0.1f, 100.0f, "%.2fx",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        value = TlgOrbs::ClampMultiplier(v);
        dirty = true;
    }
    Help(help);
    ImGui::SameLine();
    if (ImGui::SmallButton("1x")) {
        value = 1.0f;
        dirty = true;
    }
    Help(help);
    ImGui::PopID();
}

void FloatRow(const char* label, std::atomic<float>& value, float lo, float hi, const char* format, const char* help,
              ImGuiSliderFlags flags = 0) {
    Row(label);
    float v = value;
    if (ImGui::SliderFloat("##value", &v, lo, hi, format, ImGuiSliderFlags_AlwaysClamp | flags)) {
        Store(value, std::clamp(v, lo, hi), true);
    }
    Help(help);
    ImGui::PopID();
}

void InfoRow(const char* label, const char* text) {
    Row(label);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", text);
    ImGui::PopTextWrapPos();
    ImGui::PopID();
}

void SliderRow(const char* label, std::atomic<int>& value, int lo, int hi, const char* format, const char* help) {
    Row(label);
    int v = value;
    if (ImGui::SliderInt("##value", &v, lo, hi, format, ImGuiSliderFlags_AlwaysClamp)) {
        Store(value, std::clamp(v, lo, hi), true);
    }
    Help(help);
    ImGui::PopID();
}

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
    ImGui::SeparatorText(TL("Display", "Экран"));
    if (BeginRows("##display")) {
        static const char* const modes[] = {TL("Windowed", "Оконный"), TL("Borderless window", "Окно без рамки"), TL("Fullscreen", "Полный экран")};
        ChoiceRow(TL("Display mode", "Режим экрана"), s.display_mode, G::DisplayModeCount == 3 ? modes : G::DisplayModeNames.data(),
                  nullptr, G::DisplayModeCount, TL("Windowed, borderless window or exclusive fullscreen.", "Окно, окно без рамки или эксклюзивный полный экран."));
        CheckRow("VSync", s.vsync, TL("Waits for the monitor: no tearing, a little more input delay.", "Ждёт монитор: без разрывов кадра, но задержка ввода чуть больше."));
        ImGui::EndTable();
    }
    const int iw = s.image_width, ih = s.image_height, ww = s.window_width, wh = s.window_height;
    const bool upscaling = iw < ww && ih < wh;
    const bool downscaling = iw > ww || ih > wh;
    const bool rcas_blocked = s.rcas && !s.rcas_applied;
    // tlg: only spatial filters on the final image. The temporal upscalers (FSR 3/4, TAA, DLSS) were
    // built for GoW3's pipeline and make TLG's image shake, so they are not offered here.
    ImGui::SeparatorText(TL("Scaling and sharpening", "Масштабирование и резкость"));
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
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Draw distance and detail", "Дальность прорисовки и детализация"));
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
        Row(TL("Restore the game's values", "Вернуть значения игры"));
        if (ImGui::SmallButton(TL("Restore", "Сбросить"))) {
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
        ImGui::PopID();
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Information", "Сведения"));
    if (BeginRows("##image_info")) {
        char text[96];
        std::snprintf(text, sizeof(text), "%d x %d", iw, ih);
        InfoRow(TL("Game image", "Картинка игры"), text);
        std::snprintf(text, sizeof(text), "%d x %d", ww, wh);
        InfoRow(TL("Windowed", "Окно"), text);
        InfoRow("FSR 1", !s.fsr1 ? TL("off", "выкл.") : upscaling ? TL("active", "работает") : TL("no effect (image not smaller than window)", "не действует (картинка не меньше окна)"));
        InfoRow(TL("Sharpening", "Резкость"), s.rcas_applied ? TL("applied", "применяется") : TL("not applied", "не применяется"));
        ImGui::EndTable();
    }
}

void FixesSection();

void PerformanceTab() {
    if (TlgPerf::Enabled()) {
        ImGui::SeparatorText(TL("Performance capture", "Запись производительности"));
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
    ImGui::SeparatorText(TL("Live", "Действуют сразу"));
    if (BeginRows("##live")) {
        static const char* const fps_labels[] = {"30", "120", "240", TL("Unlimited", "Без ограничения")};
        ChoiceRow(TL("Frame rate limit", "Ограничение FPS"), s.fps_limit, fps_labels, G::FpsLimits.data(), int(G::FpsLimits.size()),
                  TL("Caps the presented frames. The game runs at 30 FPS (as on PS4).", "Ограничивает число выводимых кадров. Сама игра идёт в 30 FPS (как на PS4)."));
        static const char* const queued_labels[] = {TL("1 (lowest latency)", "1 (меньше задержка)"), "2", TL("Unlimited", "Без ограничения")};
        ChoiceRow(TL("Frames queued", "Кадров в очереди"), s.frames_queued, queued_labels, G::FramesQueued.data(),
                  int(G::FramesQueued.size()), TL("More frames queued: steadier FPS, slower response to input.", "Больше кадров в очереди: ровнее FPS, но медленнее отклик."));
        CheckRow(TL("Compile new shaders in the background", "Компилировать новые шейдеры в фоне"), s.async_shaders,
                 TL("Fewer stutters in new areas; an object or effect may appear a moment later.", "Меньше подтормаживаний в новых местах; объект или эффект может появиться на мгновение позже."));
        ImGui::EndTable();
    }
    FixesSection();
}


// tlg: the fixes (common/backports.h) and the A/B switches. Live ones change at once; the others
// are only recorded in tlg.ini (bpNN=0/1, wqm=0/1, ...) and apply on the next start, when the
// launcher turns the keys into TLG_BPnn / TLG_<NAME>. The table is iterated, so a reordered
// backports.h still lists every entry.
void FixesSection() {
    namespace B = Common::Backports;
    auto& s = TlgSettings::Get();
    ImGui::SeparatorText(TL("Fixes", "Исправления"));
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
    ImGui::SeparatorText(TL("A/B switches (next start)", "Переключатели A/B (след. запуск)"));
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
        ImGui::EndTable();
    }
}

void OpenGuide(); // the graffiti guide (below, with the save dialogs)

void ControlsTab() {
    auto& s = TlgSettings::Get();
    // Mouse (src/runtime_pad.c): the camera settings apply at once. The buttons and the touchpad
    // mode are fixed unless TLG_KEY_MAP / TLG_MOUSE_PAD_* (or tlg.ini key_map) change them.
    ImGui::SeparatorText(TL("Mouse", "Мышь"));
    if (BeginRows("##mouse")) {
        CheckRow(TL("Mouse turns the camera", "Мышь управляет камерой"), s.mouse_enabled,
                 TL("The game window takes the mouse while it has focus. Esc releases it; a click in the window takes it back.",
                    "Окно игры захватывает мышь, пока оно в фокусе. Esc отпускает её, щелчок по окну захватывает снова."));
        ImGui::BeginDisabled(!s.mouse_enabled);
        FloatRow(TL("Sensitivity", "Чувствительность"), s.mouse_sensitivity, 0.1f, 5.0f, "%.2f",
                 TL("How far the camera turns for the same mouse movement.", "Насколько поворачивается камера при одном и том же движении мыши."));
        CheckRow(TL("Invert Y axis", "Инвертировать ось Y"), s.mouse_invert_y,
                 TL("Moving the mouse forward looks down.", "Движение мыши от себя опускает взгляд."));
        FloatRow(TL("Smoothing", "Сглаживание"), s.mouse_smoothing, 0.0f, 0.9f, "%.2f",
                 TL("Evens out the camera motion; higher values feel softer and a little slower.",
                    "Делает движение камеры ровнее; при больших значениях оно мягче и чуть медленнее."));
        FloatRow(TL("Acceleration", "Ускорение"), s.mouse_accel, 0.0f, 2.0f, "%.2f",
                 TL("Fast movements turn more than slow ones. 0 is linear.", "Быстрые движения поворачивают сильнее медленных. 0 — линейно."));
        FloatRow(TL("Anti-dead zone", "Компенсация мёртвой зоны"), s.mouse_deadzone, 0.0f, 0.5f, "%.2f",
                 TL("Small mouse movements skip the game's stick dead zone. Raise it if slow movements do nothing.",
                    "Мелкие движения мыши перескакивают мёртвую зону стика в игре. Увеличьте, если медленные движения ничего не делают."));
        ImGui::EndDisabled();
        InfoRow(TL("Left / right / middle", "Левая / правая / средняя"), "R2 / L2 / R3");
        InfoRow(TL("Touchpad mode", "Режим тачпада"), TL("hold Left Ctrl or Mouse X1", "удерживайте левый Ctrl или кнопку мыши X1"));
        InfoRow(TL("In touchpad mode", "В режиме тачпада"),
                TL("left touch/click/drag, middle touch, right two fingers", "левая — касание/нажатие/свайп, средняя — касание, правая — два пальца"));
        InfoRow(TL("Release the mouse", "Отпустить мышь"), "Esc");
        const char* env = std::getenv("TLG_KEY_MAP");
        const std::string remap = env && env[0] ? std::string(env) : s.key_map;
        if (!remap.empty()) {
            InfoRow(TL("Remapped", "Переназначено"), remap.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Sensor mode (graffiti)", "Режим сенсора (граффити)"));
    if (BeginRows("##sensor")) {
        SliderRow(TL("Graffiti aiming speed", "Скорость прицела граффити"), s.sensor_aim, 25, 200, "%d%%",
                  TL("How fast the stick moves the graffiti cursor in sensor mode (100% = default).",
                     "Как быстро стик двигает курсор граффити в режиме сенсора (100% — по умолчанию)."));
        ImGui::EndTable();
    }
    if (ImGui::Button(TL("How to do the graffiti without a gyro", "Как рисовать граффити без гироскопа"))) {
        SetOpen(false);
        OpenGuide();
    }
    Help(TL("Shows the steps again (the same screen as the first time).",
            "Снова показывает шаги (тот же экран, что и в первый раз)."));
}

void OverlayTab() {
    namespace P = TlgPerfStats;
    auto& s = TlgSettings::Get();
    if (BeginRows("##overlay_on")) {
        CheckRow(TL("Show overlay (F10)", "Показывать оверлей (F10)"), s.show_fps, TL("The performance overlay in a corner of the screen.", "Оверлей производительности в углу экрана."));
        ImGui::EndTable();
    }
    ImGui::BeginDisabled(!s.show_fps);
    ImGui::SeparatorText(TL("What to show", "Что показывать"));
    if (ImGui::BeginTable("##overlay_items", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableNextColumn();
        static const char* const names[] = {"FPS", TL("Frametime", "Время кадра"), "Upscaler", TL("GPU (name and use)", "GPU (название и загрузка)"), TL("CPU (game)", "CPU (игра)"),
                                            TL("RAM (game / total)", "RAM (игра / всего)"), TL("VRAM (used / available)", "VRAM (занято / доступно)"),
                                            TL("Frametime graph", "График времени кадра")};
        for (unsigned item = 0; item < P::NumItems; ++item) {
            const unsigned bit = 1u << item;
            bool on = (s.overlay_items & bit) != 0;
            if (ImGui::Checkbox(names[item], &on)) {
                s.overlay_items = on ? (s.overlay_items | bit) : (s.overlay_items & ~bit);
                dirty = true;
            }
        }
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", TL("Preview", "Предпросмотр"));
        ImGui::BeginChild("##preview", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
        ImGui::SetWindowFontScale(s.overlay_scale / 100.0f);
        const std::string text = OverlayText();
        ImGui::TextUnformatted(text.empty() ? TL("(nothing selected)", "(ничего не выбрано)") : text.c_str());
        ImGui::SetWindowFontScale(1.0f);
        ImGui::EndChild();
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Appearance", "Внешний вид"));
    if (BeginRows("##appearance")) {
        SliderRow(TL("Font size", "Размер шрифта"), s.overlay_scale, 50, 300, "%d%%", TL("Size of the overlay text.", "Размер текста оверлея."));
        static const char* const corners[] = {TL("Top left", "Слева сверху"), TL("Top right", "Справа сверху"), TL("Bottom left", "Слева снизу"), TL("Bottom right", "Справа снизу")};
        ChoiceRow(TL("Corner", "Угол"), s.overlay_corner, corners, nullptr, 4, TL("Where the overlay sits.", "В каком углу экрана оверлей."));
        SliderRow(TL("Background opacity", "Непрозрачность фона"), s.overlay_opacity, 0, 100, "%d%%", TL("0% is no background.", "0% — без фона."));
        static const char* const layouts[] = {TL("One line", "В одну строку"), TL("One item per line", "Каждый пункт с новой строки")};
        ChoiceRow(TL("Layout", "Расположение"), s.overlay_layout, layouts, nullptr, 2, TL("Items side by side or stacked.", "Пункты в ряд или столбиком."));
        ImGui::EndTable();
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText(TL("About", "О программе"));
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
        ImGui::EndTable();
    }
}

// Startup options changed since this launch (applied by "Apply and restart").
int RestartChanges() {
    const auto& s = TlgSettings::Get();
    return (s.render_resolution != s.startup_render_resolution) + (s.engine_fps != s.startup_engine_fps) +
           (s.deferred_readback != s.startup_deferred_readback) + (s.stale_readback != s.startup_stale_readback);
}

void Menu() {
    auto& s = TlgSettings::Get();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(760.0f * base_scale, viewport->WorkSize.x * 0.95f),
                                    std::min(680.0f * base_scale, viewport->WorkSize.y * 0.9f)),
                             ImGuiCond_Appearing);
    bool keep_open = true;
    // tlg: the game's own title (param.sfo), not a fixed one.
    static const std::string heading = [] {
        const std::string_view title = Common::ElfInfo::Instance().Title();
        const char* app_name = std::getenv("TLG_APP_NAME");
        return std::string(!title.empty() ? title : app_name && *app_name ? app_name : "The Last Guardian") +
               PadText("  (Insert / R3+L2)", "  (Insert / RS+LT)", "  (Insert / RS+ZL)");
    }();
    // The launcher's popover fill, so the game's own menus do not show through the text.
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Theme::Fade(Theme::Popover, Theme::PopoverAlpha));
    const bool shown = ImGui::Begin(heading.c_str(), &keep_open, ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleColor();
    if (!shown) {
        ImGui::End();
        return;
    }
    footer_help = nullptr;
    constexpr int TabCount = 4;
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
    const float footer = ImGui::GetFrameHeightWithSpacing() * 2.0f + ImGui::GetTextLineHeightWithSpacing() * 2.0f;
    if (ImGui::BeginTabBar("##tabs")) {
        static const char* const tabs[TabCount] = {TL("Overlay", "Оверлей"), TL("Image", "Изображение"), TL("Performance", "Производительность"),
                                                   TL("Controls", "Управление")};
        for (int tab = 0; tab < TabCount; ++tab) {
            const ImGuiTabItemFlags flags = requested_tab == tab ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(tabs[tab], nullptr, flags)) {
                continue;
            }
            if (s.menu_tab != tab && requested_tab < 0) {
                s.menu_tab = tab;
                dirty = true;
            }
            if (ImGui::BeginChild("##body", ImVec2(0, -footer))) {
                switch (tab) {
                case 1: ImageTab(); break;
                case 2: PerformanceTab(); break;
                case 3: ControlsTab(); break;
                default: OverlayTab(); break;
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (requested_tab >= 0) {
            s.menu_tab = requested_tab;
            dirty = true;
            requested_tab = -1;
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", footer_help ? footer_help
                              : TL(PadText("L1 / R1 or Q / E: switch tabs. Settings are saved in tlg.ini.",
                                           "LB / RB or Q / E: switch tabs. Settings are saved in tlg.ini.",
                                           "L / R or Q / E: switch tabs. Settings are saved in tlg.ini."),
                                   PadText("L1 / R1 или Q / E: смена вкладки. Настройки сохраняются в tlg.ini.",
                                           "LB / RB или Q / E: смена вкладки. Настройки сохраняются в tlg.ini.",
                                           "L / R или Q / E: смена вкладки. Настройки сохраняются в tlg.ini.")));
    ImGui::PopTextWrapPos();
    if (const int changes = RestartChanges()) {
        ImGui::TextColored(Theme::Warn, "%d %s", changes,
                           Plural(changes, "change needs a restart (unsaved progress is lost).",
                                  "changes need a restart (unsaved progress is lost).",
                                  "изменение требует перезапуска (несохранённый прогресс пропадёт).",
                                  "изменения требуют перезапуска (несохранённый прогресс пропадёт).",
                                  "изменений требуют перезапуска (несохранённый прогресс пропадёт)."));
        ImGui::SameLine();
        ImGui::BeginDisabled(!std::getenv("TLG_RESTART_COMMAND"));
        if (ImGui::Button(TL("Apply and restart", "Применить и перезапустить"))) {
            // The new launch confirms the options once the game shows; run.py restores the
            // previous ones if it never does (restart_unconfirmed).
            s.restart_unconfirmed = true;
            TlgSettings::Save();
            runtime_restart();
        }
        ImGui::EndDisabled();
    } else {
        ImGui::NewLine();
    }
    const float close_width = ImGui::CalcTextSize(TL("Close", "Закрыть")).x + ImGui::GetStyle().FramePadding.x * 2;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - close_width);
    if (ImGui::Button(TL("Close", "Закрыть"))) {
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
void TextEntryBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.42f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(viewport->WorkSize.x * 0.32f, 0.0f),
                                        ImVec2(viewport->WorkSize.x * 0.8f, FLT_MAX));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, Theme::Fade(Theme::Popover, Theme::PopoverAlpha));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f * base_scale, 16.0f * base_scale));
    ImGui::Begin("##text_entry", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextColored(Theme::Muted, "%s", text_entry_prompt.c_str());
    ImGui::Separator();
    ImGui::SetWindowFontScale(1.4f);
    ImGui::Text("%s", text_entry_text.c_str());
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextColored(Theme::Accent, "_");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::TextColored(Theme::Muted, "%s", TL("Keyboard: type, Backspace to erase, Enter = OK, Esc = cancel",
                                              "Клавиатура: ввод текста, Backspace — стереть, Enter — OK, Esc — отмена"));
    ImGui::TextColored(Theme::Muted, "%s", TL(PadText("Controller: Cross (A) = OK, Circle (B) = cancel", "Controller: A = OK, B = cancel", "Controller: B = OK, A = cancel"),
                                 PadText("Геймпад: крестик (A) — OK, кружок (B) — отмена", "Геймпад: A — OK, B — отмена", "Геймпад: B — OK, A — отмена")));
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

// ---- Save dialogs: drawing. The layout is in design pixels of a 1280x720 screen, times k. ----
// The colours and radii are the launcher's (Theme); its sizes are on a 1080-line stage, so a
// launcher pixel is kStage design pixels here.
constexpr float kStage = 720.0f / 1080.0f;
constexpr float kPanelRadius = Theme::PanelRadius * kStage, kItemRadius = Theme::ItemRadius * kStage;

// Semibold Segoe UI for titles and buttons (Init), else bold is drawn twice a hair apart.
ImFont* font_bold = nullptr;

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
    // The text cut with "..." to fit `width` design pixels.
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
    // Text at (x, y) design pixels.
    void Text(float x, float y, float size, const ImVec4& color, const std::string& text, bool bold = false,
              float alpha = 1.0f, float wrap = 0.0f) const {
        const ImU32 col = Theme::U32(color, alpha);
        dl->AddText(Face(bold), size * k, P(x, y), col, text.c_str(), nullptr, wrap * k);
        if (bold && !font_bold) dl->AddText(font, size * k, P(x + 0.55f, y), col, text.c_str(), nullptr, wrap * k);
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
    // The launcher's focus ring: a white outline a little outside the item.
    void Ring(float x0, float y0, float x1, float y1, float rounding) const {
        const float gap = 3.0f * kStage;
        Frame(x0 - gap, y0 - gap, x1 + gap, y1 + gap, Theme::White(0.95f), rounding + gap, 3.0f * kStage);
    }
    // A controller button: 0 confirm (bottom face button), 1 back (right one), centred at (x, y):
    // Cross / Circle, or the letters of TLG_GLYPHS=xbox (A / B) and nintendo (B / A). `light`: on a
    // white button.
    void Button(float x, float y, int which, bool light = false) const {
        const ImVec2 c = P(x, y);
        const ImVec4 ground = light ? Theme::Ink : Theme::Text;
        dl->AddCircleFilled(c, 11.0f * k, Theme::U32(ground, 0.08f));
        dl->AddCircle(c, 11.0f * k, Theme::U32(ground, 0.28f), 0, 1.0f * k);
        const ImVec4& mark = which == 0 ? Theme::Accent : Theme::Bad;
        if (const char* letter = PadText(nullptr, which == 0 ? "A" : "B", which == 0 ? "B" : "A")) {
            Text(x - Width(13.0f, letter, true) / 2, y - 8.5f, 13.0f, which == 0 ? Theme::Good : Theme::Bad, letter, true);
            return;
        }
        if (which == 0) {
            Line(x - 4, y - 4, x + 4, y + 4, mark, 2.0f);
            Line(x + 4, y - 4, x - 4, y + 4, mark, 2.0f);
        } else {
            dl->AddCircle(c, 5.0f * k, Theme::U32(mark), 0, 2.0f * k);
        }
    }
    // The panel itself (the launcher's popover): a soft shadow, fill, faint border; then the
    // discreet credit under its right corner.
    void Panel(float w, float h) const {
        for (int i = 3; i >= 1; --i) {
            const float spread = 4.0f * i;
            Box(-spread, 12 - spread, w + spread, h + 12 + spread, Theme::Ink, kPanelRadius + spread, 0.14f);
        }
        Box(0, 0, w, h, Theme::Popover, kPanelRadius, Theme::PopoverAlpha);
        Frame(0, 0, w, h, Theme::Line, kPanelRadius, 1.0f);
        const std::string credit = "@hayatog-dev";
        Text(w - 6.0f - Width(11.0f, credit), h + 10.0f, 11.0f, Theme::Faint, credit, false, 0.7f);
    }
    // A save's picture (16:9) or a placeholder; `is_new` draws the dashed "+" slot.
    void Picture(float x, float y, float w, float h, ImTextureData* tex, bool is_new) const {
        if (tex) {
            dl->AddImageRounded(tex->GetTexRef(), P(x, y), P(x + w, y + h), ImVec2(0, 0), ImVec2(1, 1),
                                IM_COL32_WHITE, kItemRadius * k);
            Frame(x, y, x + w, y + h, Theme::Line, kItemRadius, 1.0f);
            return;
        }
        if (is_new) {
            for (float t = 0; t < 1.0f; t += 0.1f) { // dashes around the slot
                Line(x + w * t, y, x + w * (t + 0.05f), y, Theme::Faint, 1.5f);
                Line(x + w * t, y + h, x + w * (t + 0.05f), y + h, Theme::Faint, 1.5f);
            }
            for (float t = 0; t < 1.0f; t += 0.18f) {
                Line(x, y + h * t, x, y + h * (t + 0.09f), Theme::Faint, 1.5f);
                Line(x + w, y + h * t, x + w, y + h * (t + 0.09f), Theme::Faint, 1.5f);
            }
            Line(x + w / 2, y + h / 2 - 11, x + w / 2, y + h / 2 + 11, Theme::Muted, 2.0f);
            Line(x + w / 2 - 11, y + h / 2, x + w / 2 + 11, y + h / 2, Theme::Muted, 2.0f);
            return;
        }
        Box(x, y, x + w, y + h, Theme::White(0.06f), kItemRadius);
        Frame(x, y, x + w, y + h, Theme::Line, kItemRadius, 1.0f);
    }
    // A dialog button: the focused one white with black text (the launcher's Play button), the
    // others faint.
    void ActionButton(float x0, float y0, float x1, float y1, const char* label, int glyph, bool focused) const {
        Box(x0, y0, x1, y1, focused ? Theme::Text : Theme::White(0.08f), kItemRadius);
        if (!focused) Frame(x0, y0, x1, y1, Theme::Line, kItemRadius, 1.0f);
        const float mid = (y0 + y1) / 2;
        Button(x0 + 32, mid, glyph, focused);
        Text(x0 + 52, mid - 10, 15.0f, focused ? Theme::Ink : Theme::Text, label, true);
    }
};

SavePainter SavePainterFor(float w, float h) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    // 720p design scaled to the display, kept inside the screen on narrow windows.
    const float k = std::min(viewport->WorkSize.y / 720.0f, viewport->WorkSize.x * 0.94f / w);
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
    char out[64];
    if (UiRussian()) {
        std::snprintf(out, sizeof(out), "%02d.%02d.%04d · %02d:%02d", tm.tm_mday, tm.tm_mon + 1,
                      tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
    } else {
        static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        const int hour = tm.tm_hour % 12 ? tm.tm_hour % 12 : 12;
        std::snprintf(out, sizeof(out), "%s %d, %d · %d:%02d %s", months[tm.tm_mon % 12], tm.tm_mday,
                      tm.tm_year + 1900, hour, tm.tm_min, tm.tm_hour < 12 ? "AM" : "PM");
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

// One save as a card: picture, the game's title and text, when it was saved.
void SaveCard(const SavePainter& p, float x0, float y0, float x1, const SaveRow& row, ImTextureData* tex,
              bool selected, float picture_w) {
    const float picture_h = picture_w * 9.0f / 16.0f, h = picture_h + 24.0f;
    p.Box(x0, y0, x1, y0 + h, Theme::White(selected ? 0.12f : 0.04f), kItemRadius);
    if (selected) {
        p.Ring(x0, y0, x1, y0 + h, kItemRadius);
    }
    p.Picture(x0 + 16, y0 + 12, picture_w, picture_h, tex, row.is_new);
    const float text_x = x0 + 16 + picture_w + 18;
    std::string when, ago;
    if (row.is_new) {
        p.Text(text_x, y0 + h / 2 - 20, 17.0f, Theme::Text, TL("New save", "Новое сохранение"), true);
        p.Text(text_x, y0 + h / 2 + 4, 13.5f, Theme::Muted, TL("Create a new save", "Создать новое сохранение"));
        return;
    }
    when = SaveWhen(row.mtime);
    ago = SaveAgo(row.mtime);
    const std::string size = SaveSize(row.bytes);
    if (!size.empty()) ago = ago.empty() ? size : ago + " · " + size;
    const float right = x1 - 16;
    const float right_w = std::max(p.Width(14.0f, when, true), p.Width(12.5f, ago));
    const float text_w = right - text_x - (right_w > 0 ? right_w + 20 : 0);
    std::string line = row.subtitle;
    if (!row.detail.empty()) line = line.empty() ? row.detail : line + " · " + row.detail;
    for (char& c : line) if (c == '\n' || c == '\r') c = ' ';
    p.Text(text_x, y0 + h / 2 - 20, 17.0f, Theme::Text, p.Fit(17.0f, row.title, text_w, true), true);
    p.Text(text_x, y0 + h / 2 + 4, 13.5f, Theme::Muted, p.Fit(13.5f, line, text_w));
    p.TextRight(right, y0 + h / 2 - 18, 14.0f, Theme::Text, when, true);
    p.TextRight(right, y0 + h / 2 + 5, 12.5f, Theme::Faint, ago);
}

void SaveListBox() {
    const int count = int(save_rows.size());
    const int shown = std::min(count, 4);
    const float w = 880, row_h = 96, gap = 10, header_h = 78, footer_h = 54, pad = 18;
    const float list_h = shown * row_h + std::max(shown - 1, 0) * gap + pad * 2;
    const float h = header_h + list_h + footer_h;
    SaveBackdrop(0.6f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);

    // Header: a floppy, the title, the game and the number of saves.
    const float ix = 28, iy = 25;
    p.Frame(ix, iy, ix + 26, iy + 26, Theme::Text, 4.0f, 1.8f);
    p.Frame(ix + 6, iy, ix + 20, iy + 9, Theme::Text, 1.0f, 1.8f);
    p.Frame(ix + 6, iy + 15, ix + 20, iy + 26, Theme::Text, 1.0f, 1.8f);
    p.Text(70, 17, 24.0f, Theme::Text, SaveTitle(save_kind), true);
    const int saves = count - int(std::count_if(save_rows.begin(), save_rows.end(), [](const SaveRow& r) { return r.is_new; }));
    char sub[200];
    const char* game = std::getenv("TLG_APP_NAME");
    std::snprintf(sub, sizeof(sub), "%s%s%d %s", game && *game ? game : "", game && *game ? " · " : "", saves,
                  Plural(saves, "save", "saves", "сохранение", "сохранения", "сохранений"));
    p.Text(70, 48, 13.0f, Theme::Faint, sub);
    p.Line(0, header_h, w, header_h, Theme::Line, 1.0f);

    // Rows, scrolled so the selected one is shown.
    if (choice_index < choice_first) choice_first = choice_index;
    if (choice_index >= choice_first + shown) choice_first = choice_index - shown + 1;
    choice_first = std::clamp(choice_first, 0, std::max(count - shown, 0));
    save_row_rects.assign(count, ImRect());
    for (int i = choice_first; i < choice_first + shown; ++i) {
        const float y = header_h + pad + (i - choice_first) * (row_h + gap);
        SaveCard(p, 20, y, w - 20, save_rows[i], SaveTexture(size_t(i)), i == choice_index, 128.0f);
        save_row_rects[i] = ImRect(p.P(20, y), p.P(w - 20, y + row_h));
    }
    if (count > shown) { // scroll bar
        const float track = list_h - pad * 2, y0 = header_h + pad;
        const float knob = track * shown / count, top = y0 + track * choice_first / count;
        p.Box(w - 11, y0, w - 8, y0 + track, Theme::Line, 2.0f);
        p.Box(w - 11, top, w - 8, top + knob, Theme::Faint, 2.0f);
    }

    // Footer: what the buttons do.
    const float fy = header_h + list_h;
    p.Line(0, fy, w, fy, Theme::Line, 1.0f);
    const char* confirm = save_kind == 2 ? TL("Load", "Загрузить") : save_kind == 3 ? TL("Delete", "Удалить")
                                                                                  : TL("Save here", "Сохранить сюда");
    const char* back = TL("Back", "Назад");
    float x = 40;
    p.Button(x, fy + footer_h / 2, 0);
    p.Text(x + 18, fy + footer_h / 2 - 9, 13.5f, Theme::Muted, confirm);
    x += 18 + p.Width(13.5f, confirm) + 34;
    p.Button(x, fy + footer_h / 2, 1);
    p.Text(x + 18, fy + footer_h / 2 - 9, 13.5f, Theme::Muted, back);
    x += 18 + p.Width(13.5f, back) + 30;
    const std::string hint = TL("Enter / click: choose · Esc / right click: back",
                                "Enter / щелчок: выбрать · Esc / правая кнопка: назад");
    p.TextRight(w - 28, fy + footer_h / 2 - 8, 12.5f, Theme::Faint, p.Fit(12.5f, hint, w - 28 - x));
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
    const float w = 560, pad = 28, inner = w - pad * 2;
    // Height: title, the save's card, the wrapped text, the buttons.
    ImFont* font = ImGui::GetFont();
    const float k0 = std::min(ImGui::GetMainViewport()->WorkSize.y / 720.0f,
                              ImGui::GetMainViewport()->WorkSize.x * 0.94f / w);
    const float body_h = words.body.empty() ? 0.0f
        : font->CalcTextSizeA(14.5f * k0, FLT_MAX, inner * k0, words.body.c_str()).y / k0;
    const float card_h = message_has_entry ? 87.0f : 0.0f;
    float h = pad + 30 + 20 + (card_h ? card_h + 20 : 0) + (body_h ? body_h + 20 : 0) + 46 + pad;
    SaveBackdrop(0.6f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);

    float y = pad;
    if (words.warn) { // triangle
        const ImVec2 a = p.P(pad + 13, y + 2), b = p.P(pad + 26, y + 26), c = p.P(pad, y + 26);
        p.dl->AddTriangle(a, b, c, Theme::U32(Theme::Warn), 2.0f * p.k);
        p.Line(pad + 13, y + 11, pad + 13, y + 18, Theme::Warn, 2.0f);
        p.dl->AddCircleFilled(p.P(pad + 13, y + 22), 1.4f * p.k, Theme::U32(Theme::Warn));
    } else {
        p.dl->AddCircle(p.P(pad + 13, y + 14), 12.0f * p.k, Theme::U32(Theme::Accent), 0, 2.0f * p.k);
        p.Text(pad + 13 - p.Width(16.0f, "?", true) / 2, y + 3.5f, 16.0f, Theme::Accent, "?", true);
    }
    p.Text(pad + 40, y + 1, 22.0f, Theme::Text, p.Fit(22.0f, words.title, inner - 40, true), true);
    y += 30 + 20;
    if (message_has_entry) {
        p.Box(pad, y, w - pad, y + card_h, Theme::White(0.04f), kItemRadius);
        p.Frame(pad, y, w - pad, y + card_h, Theme::Line, kItemRadius, 1.0f);
        p.Picture(pad + 14, y + 12, 112, 63, SaveTexture(0), message_entry.is_new);
        const float tx = pad + 14 + 112 + 16, tw = w - pad - 14 - tx;
        std::string line = message_entry.subtitle;
        if (!message_entry.detail.empty()) line = line.empty() ? message_entry.detail : line + " · " + message_entry.detail;
        for (char& c : line) if (c == '\n' || c == '\r') c = ' ';
        std::string when = SaveWhen(message_entry.mtime);
        const std::string ago = SaveAgo(message_entry.mtime);
        if (!when.empty() && !ago.empty()) when += " (" + ago + ")";
        p.Text(tx, y + 16, 16.0f, Theme::Text,
               p.Fit(16.0f, message_entry.is_new ? std::string(TL("New save", "Новое сохранение")) : message_entry.title, tw, true),
               true);
        p.Text(tx, y + 39, 13.0f, Theme::Muted, p.Fit(13.0f, line, tw));
        p.Text(tx, y + 59, 12.5f, Theme::Faint, p.Fit(12.5f, when, tw));
        y += card_h + 20;
    }
    if (body_h) {
        p.Text(pad, y, 14.5f, Theme::Muted, words.body, false, 1.0f, inner);
        y += body_h + 20;
    }

    // Buttons, right-aligned: [no/cancel] [yes/OK]; the focused one is white.
    const char* yes = buttons == 1 ? TL("Yes", "Да") : "OK";
    const char* no = buttons == 1 ? TL("No", "Нет") : TL("Cancel", "Отмена");
    const bool two = buttons == 1 || buttons == 3;
    if (!two) message_focus = 1;
    float right = w - pad;
    save_button_rects[0] = save_button_rects[1] = ImRect();
    for (int b = 1; b >= (two ? 0 : 1); --b) {
        const char* label = b == 1 ? yes : no;
        const float bw = 22 + 20 + 10 + p.Width(15.0f, label, true) + 22, x0 = right - bw;
        p.ActionButton(x0, y, right, y + 46, label, b == 1 ? 0 : 1, b == message_focus);
        save_button_rects[b] = ImRect(p.P(x0, y), p.P(right, y + 46));
        right = x0 - 12;
    }
}

void SaveProgressBox(SaveClock::time_point now) {
    const float w = 520, h = 140;
    SaveBackdrop(0.45f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    // A spinning arc.
    const float t = std::chrono::duration<float>(now - progress_since).count();
    const ImVec2 c = p.P(28 + 13, 26 + 13);
    p.dl->AddCircle(c, 11.0f * p.k, Theme::U32(Theme::Line), 0, 2.4f * p.k);
    p.dl->PathArcTo(c, 11.0f * p.k, t * 6.0f, t * 6.0f + 4.4f, 24);
    p.dl->PathStroke(Theme::U32(Theme::Accent), 0, 2.4f * p.k);
    const char* title = progress_kind == 2 ? TL("Loading...", "Загрузка...")
                        : progress_kind == 3 ? TL("Deleting...", "Удаление...")
                                             : TL("Saving...", "Сохранение...");
    p.Text(66, 19, 21.0f, Theme::Text, title, true);
    p.Text(66, 47, 13.0f, Theme::Faint, TL("Do not close the game.", "Не закрывайте игру."));
    const float bx0 = 28, bx1 = w - 28, by = 84;
    p.Box(bx0, by, bx1, by + 6, Theme::White(0.1f), 3.0f);
    if (progress_percent >= 0) {
        const float f = std::clamp(progress_percent, 0, 100) / 100.0f;
        if (f > 0) p.Box(bx0, by, bx0 + (bx1 - bx0) * f, by + 6, Theme::Accent, 3.0f);
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%d%%", std::clamp(progress_percent, 0, 100));
        p.TextRight(bx1, by + 16, 13.0f, Theme::Text, pct, true);
    } else { // no figure: a segment sweeps the bar
        const float span = (bx1 - bx0) * 0.3f, phase = std::fmod(t * 0.8f, 1.0f);
        const float s0 = bx0 + (bx1 - bx0 + span) * phase - span;
        p.Box(std::max(s0, bx0), by, std::min(s0 + span, bx1), by + 6, Theme::Accent, 3.0f);
    }
    if (!progress_text.empty()) {
        std::string text = progress_text;
        for (char& ch : text) if (ch == '\n' || ch == '\r') ch = ' ';
        p.Text(bx0, by + 16, 13.0f, Theme::Muted, p.Fit(13.0f, text, bx1 - bx0 - 60));
    }
}

void SaveToastBox(SaveClock::time_point now) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const char* text = toast_kind == 2 ? TL("Save loaded", "Сохранение загружено")
                       : toast_kind == 3 ? TL("Save deleted", "Сохранение удалено")
                                         : TL("Game saved", "Игра сохранена");
    const float in = std::chrono::duration<float>(now - toast_since).count();
    const float out = std::chrono::duration<float>(toast_until - now).count();
    const float alpha = std::clamp(std::min(in / 0.15f, out / 0.4f), 0.0f, 1.0f);
    const float k = viewport->WorkSize.y / 720.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    const float w = 18 + 22 + 12 + p.Width(14.0f, text, true) + 20, h = 46;
    p.at = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - (32 + w) * k,
                  viewport->WorkPos.y + viewport->WorkSize.y - (28 + h) * k);
    p.Box(0, 0, w, h, Theme::Popover, kPanelRadius, Theme::PopoverAlpha * alpha);
    p.Frame(0, 0, w, h, Theme::Line, kPanelRadius, 1.0f, alpha);
    p.Line(18, 24, 24, 30, Theme::Good, 2.4f, alpha); // a check mark
    p.Line(24, 30, 36, 17, Theme::Good, 2.4f, alpha);
    p.Text(52, 13, 14.0f, Theme::Text, text, true, alpha);
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
    const char* again = TL("The Insert menu, Controls tab, shows this again.",
                           "Это окно снова открывается в меню Insert, вкладка «Управление».");
    const float w = 640, pad = 30, inner = w - pad * 2, text_x = pad + 40;
    ImFont* font = ImGui::GetFont();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k0 = std::min(viewport->WorkSize.y / 720.0f, viewport->WorkSize.x * 0.94f / w);
    auto height = [&](const char* text, float size, float width) {
        return font->CalcTextSizeA(size * k0, FLT_MAX, width * k0, text).y / k0;
    };
    float step_h[4], h = pad + 30 + 8 + 20 + 22;
    for (int i = 0; i < 4; ++i) {
        step_h[i] = std::max(28.0f, height(steps[i], 15.0f, inner - 40));
        h += step_h[i] + 12;
    }
    const float keys_h = height(keys, 13.0f, inner);
    h += 8 + keys_h + 22 + 46 + pad;
    SaveBackdrop(0.6f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    float y = pad;
    p.Text(pad, y, 23.0f, Theme::Text, TL("Graffiti without a gyro", "Граффити без гироскопа"), true);
    y += 30 + 8;
    p.Text(pad, y, 14.0f, Theme::Muted, TL("No motion sensor on your controller? Use sensor mode:",
                                           "На геймпаде нет датчика движения? Включите режим сенсора:"));
    y += 20 + 22;
    for (int i = 0; i < 4; ++i) {
        const ImVec2 c = p.P(pad + 13, y + 11);
        p.dl->AddCircleFilled(c, 13.0f * p.k, Theme::U32(Theme::Accent, 0.16f));
        p.dl->AddCircle(c, 13.0f * p.k, Theme::U32(Theme::Accent), 0, 1.5f * p.k);
        char number[4];
        std::snprintf(number, sizeof(number), "%d", i + 1);
        p.Text(pad + 13 - p.Width(14.0f, number, true) / 2, y + 1.5f, 14.0f, Theme::Text, number, true);
        p.Text(text_x, y + 1, 15.0f, Theme::Text, steps[i], false, 1.0f, inner - 40);
        y += step_h[i] + 12;
    }
    y += 8;
    p.Text(pad, y, 13.0f, Theme::Muted, keys, false, 1.0f, inner);
    y += keys_h + 22;
    const char* ok = TL("Got it", "Понятно");
    const float bw = 22 + 20 + 10 + p.Width(15.0f, ok, true) + 22, x0 = w - pad - bw;
    const float again_w = x0 - pad - 20;
    p.Text(pad, y + 23 - height(again, 12.5f, again_w) / 2, 12.5f, Theme::Faint, again, false, 1.0f, again_w);
    p.ActionButton(x0, y, w - pad, y + 46, ok, 0, true);
    guide_button_rect = ImRect(p.P(x0, y), p.P(w - pad, y + 46));
}

// imgui_mutex held.
void OpenGuide() {
    guide_button_rect = ImRect();
    guide_active = true;
    ImGui::GetIO().MouseDrawCursor = true;
}

void CloseGuide() {
    guide_active = false;
    ImGui::GetIO().MouseDrawCursor = menu_open;
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

void LoadingScreen() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 0.4f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing);
    const u32 done = loading_done, total = loading_total;
    const bool compile = loading_compile;
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextUnformatted(compile ? TL("Compiling shaders...", "Компиляция шейдеров...")
                           : total ? TL("Loading shader cache...", "Загрузка кэша шейдеров...")
                                   : TL("Preparing graphics...", "Подготовка графики..."));
    ImGui::SetWindowFontScale(1.0f);
    if (total) {
        char count[64];
        std::snprintf(count, sizeof(count), "%u / %u (%u%%)", done, total,
                      u32(u64(done) * 100 / total));
        ImGui::ProgressBar(float(done) / float(total), ImVec2(-1.0f, 0.0f), count);
    }
    if (compile && total && done > 0 && done < total) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                             loading_compile_started)
                                   .count();
        const u32 left = u32(elapsed * double(total - done) / double(done)) + 1;
        ImGui::Text(TL("About %u:%02u left", "Осталось около %u:%02u"), left / 60, left % 60);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", compile ? TL("Only on the first start on this computer and after updates: "
                                           "then the game runs without compiling shaders.",
                                           "Только при первом запуске на этом компьютере и после "
                                           "обновлений: дальше игра работает без компиляции шейдеров.")
                                      : TL("The first start after an update takes longer.",
                                           "Первый запуск после обновления занимает больше времени."));
    ImGui::PopTextWrapPos();
    ImGui::End();
}

// Frame time statistics over the graph's window.
struct FrameStats {
    float min_fps = 0.0f, avg_fps = 0.0f, low1_fps = 0.0f, max_ms = 0.0f;
    int stutters = 0; ///< frames over twice the average and over 25 ms
};

FrameStats ComputeFrameStats() {
    FrameStats out;
    if (frame_history_count == 0) {
        return out;
    }
    std::array<float, frame_history.size()> sorted{};
    double total = 0.0;
    for (size_t i = 0; i < frame_history_count; ++i) {
        sorted[i] = frame_history[i];
        total += frame_history[i];
        out.max_ms = std::max(out.max_ms, frame_history[i]);
    }
    const float avg_ms = float(total / frame_history_count);
    for (size_t i = 0; i < frame_history_count; ++i) {
        out.stutters += frame_history[i] > 2.0f * avg_ms && frame_history[i] > 25.0f;
    }
    std::sort(sorted.begin(), sorted.begin() + frame_history_count);
    // 1% low: the frame time 99% of the frames stay under.
    const float low1_ms = sorted[std::min(frame_history_count - 1, size_t(frame_history_count * 0.99f))];
    out.avg_fps = 1000.0f / avg_ms;
    out.min_fps = 1000.0f / out.max_ms;
    out.low1_fps = 1000.0f / low1_ms;
    return out;
}

ImVec4 FpsColor(float fps) {
    return fps >= 55.0f ? Theme::Good : fps >= 28.0f ? Theme::Warn : Theme::Bad;
}

// The frame time curve (a "net graph"): the last present intervals, guides at 60 and 30 FPS,
// stutters marked in red.
void FrameGraph(float width, float height, float text_scale) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 end{origin.x + width, origin.y + height};
    const float rounding = Theme::ItemRadius * base_scale;
    draw->AddRectFilled(origin, end, ImGui::GetColorU32(ImVec4(1, 1, 1, 0.05f)), rounding);
    // Scale: at least 40 ms, else the slowest frame plus a margin (at most 100 ms).
    float top_ms = 40.0f;
    for (size_t i = 0; i < frame_history_count; ++i) {
        top_ms = std::max(top_ms, frame_history[i] * 1.15f);
    }
    top_ms = std::min(top_ms, 100.0f);
    const auto y_of = [&](float ms) { return end.y - std::min(ms, top_ms) / top_ms * height; };
    const ImU32 guide = ImGui::GetColorU32(ImVec4(1, 1, 1, 0.16f));
    const ImU32 label = ImGui::GetColorU32(Theme::Muted);
    ImFont* font = ImGui::GetFont();
    const float label_size = ImGui::GetFontSize() * 0.62f;
    for (const auto& [ms, text] : {std::pair{1000.0f / 60.0f, "60"}, std::pair{1000.0f / 30.0f, "30"}}) {
        const float y = y_of(ms);
        draw->AddLine(ImVec2(origin.x + 4.0f, y), ImVec2(end.x - 4.0f, y), guide, 1.0f);
        draw->AddText(font, label_size, ImVec2(end.x - label_size * 1.4f, y - label_size - 1.0f), label, text);
    }
    if (frame_history_count >= 2) {
        const size_t n = frame_history_count;
        const size_t first = (frame_history_next + frame_history.size() - n) % frame_history.size();
        float avg = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            avg += frame_history[i];
        }
        avg /= float(n);
        std::vector<ImVec2> points(n);
        for (size_t i = 0; i < n; ++i) {
            const float ms = frame_history[(first + i) % frame_history.size()];
            points[i] = ImVec2(origin.x + 4.0f + (width - 8.0f) * float(i) / float(frame_history.size() - 1),
                               y_of(ms));
        }
        // a soft fill under the curve, then the curve
        const ImU32 fill = ImGui::GetColorU32(ImVec4(Theme::Accent.x, Theme::Accent.y, Theme::Accent.z, 0.14f));
        for (size_t i = 1; i < n; ++i) {
            draw->AddQuadFilled(points[i - 1], points[i], ImVec2(points[i].x, end.y - 1.0f),
                                ImVec2(points[i - 1].x, end.y - 1.0f), fill);
        }
        draw->AddPolyline(points.data(), int(n), ImGui::GetColorU32(Theme::Accent), ImDrawFlags_None,
                          1.6f * base_scale);
        const ImU32 stutter = ImGui::GetColorU32(Theme::Bad);
        for (size_t i = 0; i < n; ++i) {
            const float ms = frame_history[(first + i) % frame_history.size()];
            if (ms > 2.0f * avg && ms > 25.0f) {
                draw->AddCircleFilled(points[i], 2.6f * base_scale, stutter);
            }
        }
    }
    ImGui::Dummy(ImVec2(width, height));
    (void)text_scale;
}

void FpsCounter() {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const unsigned items = s.overlay_items;
    const bool big_fps = items & P::Fps, graph = items & P::Graph;
    const std::string text = OverlayText(P::Fps);
    if (text.empty() && !big_fps && !graph) {
        return;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 14.0f * base_scale;
    const int corner = s.overlay_corner;
    const bool right = corner == 1 || corner == 3, bottom = corner >= 2;
    ImGui::SetNextWindowPos(ImVec2(right ? viewport->WorkPos.x + viewport->WorkSize.x - pad : viewport->WorkPos.x + pad,
                                   bottom ? viewport->WorkPos.y + viewport->WorkSize.y - pad : viewport->WorkPos.y + pad),
                            ImGuiCond_Always, ImVec2(right ? 1.0f : 0.0f, bottom ? 1.0f : 0.0f));
    ImGui::SetNextWindowBgAlpha(s.overlay_opacity / 100.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Theme::PanelRadius * base_scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f * base_scale, 10.0f * base_scale));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    const float font_scale = s.overlay_scale / 100.0f;
    const FrameStats stats = graph ? ComputeFrameStats() : FrameStats{};
    if (big_fps) {
        const float fps = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
        ImGui::SetWindowFontScale(font_scale * 2.1f);
        ImGui::PushStyleColor(ImGuiCol_Text, FpsColor(fps));
        ImGui::Text("%.0f", fps);
        ImGui::PopStyleColor();
        ImGui::SameLine(0.0f, 6.0f * base_scale);
        ImGui::SetWindowFontScale(font_scale * 0.85f);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetFontSize() * 1.35f);
        ImGui::TextColored(Theme::Muted, "FPS");
    }
    ImGui::SetWindowFontScale(font_scale);
    if (!text.empty()) {
        ImGui::TextUnformatted(text.c_str());
    }
    if (graph) {
        ImGui::Dummy(ImVec2(0.0f, 2.0f * base_scale));
        FrameGraph(250.0f * base_scale * font_scale, 64.0f * base_scale * font_scale, font_scale);
        ImGui::SetWindowFontScale(font_scale * 0.8f);
        ImGui::TextColored(Theme::Muted, "%s %.0f   %s %.0f   1%% %.0f   %s %d", TL("min", "мин"), stats.min_fps,
                           TL("avg", "сред"), stats.avg_fps, stats.low1_fps, TL("stutters", "фризы"),
                           stats.stutters);
    }
    ImGui::SetWindowFontScale(1.0f);
    ImGui::End();
    ImGui::PopStyleVar(3);
}

std::string notice_text;
std::chrono::steady_clock::time_point notice_until;

void NoticeBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + 40.0f * base_scale),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowBgAlpha(Theme::GlassAlpha);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f * base_scale, 12.0f * base_scale));
    ImGui::Begin("##notice", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::SetWindowFontScale(1.3f);
    ImGui::TextUnformatted(notice_text.c_str());
    ImGui::End();
    ImGui::PopStyleVar();
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
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + pad,
                                   viewport->WorkPos.y + viewport->WorkSize.y - pad),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("##background_loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextDisabled(TL("Loading shader cache %u%%", "Загрузка кэша шейдеров %u%%"), u32(u64(done) * 100 / total));
    ImGui::End();
}

} // namespace

void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count) {
    std::scoped_lock lock{imgui_mutex};
    if (initialized) {
        return;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
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
    char windir[260] = {};
    const DWORD n = GetEnvironmentVariableA("WINDIR", windir, sizeof(windir));
    const std::string fonts = std::string(n ? windir : "C:\\Windows") + "\\Fonts\\";
    for (const char* name : {"segoeui.ttf", "segoeuisl.ttf"}) {
        if ((font = add_file(fonts + name, 19.0f))) {
            add_dejavu(true);
            break;
        }
    }
    if (font) {
        font_bold = add_file(fonts + "segoeuisb.ttf", 19.0f);
        if (font_bold) add_dejavu(true);
    }
#endif
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
    bool want = false;
    {
        std::scoped_lock lock{imgui_mutex};
        want = initialized && menu_open && ImGui::GetIO().WantTextInput;
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
        ImGui::GetIO().AddMousePosEvent(event.motion.x * density, event.motion.y * density);
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const ImVec2 at = ImGui::GetIO().MousePos;
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
    ImGui::GetIO().MouseDrawCursor = menu_open;
}

// The save list and messages take the keys, buttons and mouse that answer them; the rest stays
// with the game (held neutral by CapturesInput).
bool HandleSaveEvent(const SDL_Event& event) {
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
        ImGui::GetIO().AddMousePosEvent(save_mouse.x, save_mouse.y);
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
    std::scoped_lock lock{imgui_mutex};
    if (!initialized) {
        return false;
    }
    if (guide_active && !menu_open) {
        return HandleGuideEvent(event);
    }
    if ((choice_active || message_active) && !menu_open) {
        return HandleSaveEvent(event);
    }
    ImGuiIO& io = ImGui::GetIO();
    const bool is_open = menu_open;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = event.type == SDL_EVENT_KEY_DOWN;
        static const bool capture_key = [] {
            const char* env = std::getenv("TLG_FRAME_CAPTURE_KEY");
            return env && *env == '1';
        }();
        if (down && !event.key.repeat && capture_key && event.key.key == SDLK_F11) {
            Vulkan::FrameCapture::Request();
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
                           NoticeActive() || mouse_pad_on ||
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
    ImGui::GetIO().MouseDrawCursor = true;
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
    ImGui::GetIO().MouseDrawCursor = true;
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

void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent) {
    // Present interval for the FPS readout (measured also while nothing is drawn).
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - last_present).count();
    last_present = now;
    if (ms > 0.0f && ms < 1000.0f) {
        frame_ms_avg = frame_ms_avg == 0.0f ? ms : frame_ms_avg * 0.95f + ms * 0.05f;
        frame_history[frame_history_next] = ms;
        frame_history_next = (frame_history_next + 1) % frame_history.size();
        frame_history_count = std::min(frame_history_count + 1, frame_history.size());
    }
    ConfirmRestart();
    if (!Visible()) {
        return;
    }
    std::scoped_lock lock{imgui_mutex};
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(extent.width), float(extent.height));
    io.DeltaTime = ms > 0.0f && ms < 1000.0f ? ms / 1000.0f : 1.0f / 60.0f;
    // UI scale follows the display height (1080p = 1).
    const float scale = std::max(float(extent.height) / 1080.0f, 0.75f);
    if (std::abs(scale - base_scale) > 0.01f) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(scale / base_scale);
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
