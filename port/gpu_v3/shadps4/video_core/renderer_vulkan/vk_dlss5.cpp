// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// DLSS 5 Neural Rendering (vk_dlss5.h). Facts this rests on, read from the fork's source
// (github.com/Skynizz/optiscaler-dlss5 6bfc4389, OptiScaler/dlssnr; LATER-PLAN 9, 2026-10-18):
// - the model (NGX feature 18) has a native Vulkan surface: NVSDK_NGX_VULKAN_Init_Ext /
//   CreateFeature / EvaluateFeature / ReleaseFeature, reached through the forwarder's
//   dlssnr_vk_probe / _init / _create / _evaluate / _release;
// - inputs colour, depth, motion vectors, one output; the tuning is read at create;
// - multi-pass = one feature per pass on the previous answer, a new pass used from the next frame
//   (DlssNr_Menu.cpp "Model passes"); never rebuild every frame (the driver stops answering);
// - the device needs VK_NVX_binary_import and VK_NVX_image_view_handle (plus push descriptors and
//   buffer device address, which the emulator enables anyway).

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <initializer_list>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/libraries/videodec/video_activity.h"
#include "video_core/host_shaders/dlss5_compose_comp.h"
#include "video_core/host_shaders/dlss5_compose_frag.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::Dlss5 {

namespace {

constexpr const char* ForwarderName = "nvngx.dll_dlssnr.dll";
constexpr const char* ModelName = "nvngx_dlssnr.dll";
constexpr const char* SettingsName = "dlss5-settings.ini";
constexpr int NgxSuccess = 1;
constexpr int NgxFail = static_cast<int>(0xBAD00000u);
constexpr int NgxSdkVersion = 0x15;
constexpr int FeatureId = 18;
constexpr u64 RetireAfter = 64; ///< frames before a replaced feature / image is freed
constexpr u32 TimingSlots = 8;

// Bisect switches for the round 2+ device-lost crash (2026-10-18; read once):
//   SHADPS4_DLSS5_RESET=1   every model run starts a new history (gameplay and video).
//   SHADPS4_DLSS5_PATH=1    round 1's gameplay path (e491fade ran 20k+ frames without a crash):
//                           a blit in, one model pass, its answer straight to DLSS 4.5. None of
//                           our compute passes run (no true colour; Blend / Layers / Compare do
//                           nothing) and there is no video lane.
//   SHADPS4_DLSS5_PATH=2    our passes kept, the model's own images apart: the encode writes a
//                           scratch image that a copy moves into the model's input, the answer is
//                           copied out before the compose, and the model's images keep their
//                           contents between frames (no per-frame discard).
//   SHADPS4_DLSS5_NOMODEL=1 neither DLL is loaded: every model run is a plain copy, so our passes
//                           run alone (a test that needs no unsigned code).
//   SHADPS4_DLSS5_COMPUTE=1 our passes as compute dispatches again (rounds 2-3; they crashed even
//                           with no model - NOMODEL test 2026-10-18) instead of full-screen draws.
struct Switches {
    bool reset = false;
    int path = 0;
    bool no_model = false;
    bool compute = false;
};

const Switches& Test() {
    static const Switches t = [] {
        const auto on = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] == '1';
        };
        Switches x;
        x.reset = on("SHADPS4_DLSS5_RESET");
        x.no_model = on("SHADPS4_DLSS5_NOMODEL");
        x.compute = on("SHADPS4_DLSS5_COMPUTE");
        if (const char* v = std::getenv("SHADPS4_DLSS5_PATH"); v && (v[0] == '1' || v[0] == '2')) {
            x.path = v[0] - '0';
        }
        return x;
    }();
    return t;
}

const std::vector<const char*> g_extensions = {"VK_NVX_binary_import",
                                               "VK_NVX_image_view_handle"};

// ---- the NGX parameter block -------------------------------------------------------------

// The NGX parameter block as the model sees it. The SDK declares eight Set overloads, eight Get
// overloads and Reset (nvsdk_ngx_params.h; order ULL, float, double, uint, int, ID3D11Resource*,
// ID3D12Resource*, void*), and the model, built with MSVC, calls them through the vtable. MSVC
// groups overloads in REVERSE declaration order, so the slots are the ones below - the fork
// measured exactly this on the driver's own block (resource setter 0 / getter 8, uint written
// through 3 = Set(int), float setter 6). Distinct names give this order on any compiler; SelfTest()
// checks the slots the forwarder writes through before anything uses the block. No virtual
// destructor: it would add slots the model does not expect.
class NgxParameters final {
public:
    enum class Kind : u8 { U64, F32, F64, U32, I32, Ptr };
    struct Value {
        Kind kind;
        union {
            u64 u64v;
            float f32;
            double f64;
            u32 u32v;
            s32 i32;
            void* ptr;
        };
    };

    // Slots 0-7: Set(void*), Set(ID3D12Resource*), Set(ID3D11Resource*), Set(int),
    // Set(unsigned int), Set(double), Set(float), Set(unsigned long long).
    virtual void SetPtr(const char* name, void* v) {
        Value x{};
        x.kind = Kind::Ptr;
        x.ptr = v;
        Put(name, x);
    }
    virtual void SetD3D12(const char* name, void* v) {
        SetPtr(name, v);
    }
    virtual void SetD3D11(const char* name, void* v) {
        SetPtr(name, v);
    }
    virtual void SetI32(const char* name, int v) {
        Value x{};
        x.kind = Kind::I32;
        x.i32 = v;
        Put(name, x);
    }
    virtual void SetU32(const char* name, unsigned int v) {
        Value x{};
        x.kind = Kind::U32;
        x.u32v = v;
        Put(name, x);
    }
    virtual void SetF64(const char* name, double v) {
        Value x{};
        x.kind = Kind::F64;
        x.f64 = v;
        Put(name, x);
    }
    virtual void SetF32(const char* name, float v) {
        Value x{};
        x.kind = Kind::F32;
        x.f32 = v;
        Put(name, x);
    }
    virtual void SetU64(const char* name, unsigned long long v) {
        Value x{};
        x.kind = Kind::U64;
        x.u64v = v;
        Put(name, x);
    }

    // Slots 8-15: the Gets in the same order; 16: Reset.
    virtual int GetPtr(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetD3D12(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetD3D11(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetI32(const char* name, int* out) const {
        return Number(name, out);
    }
    virtual int GetU32(const char* name, unsigned int* out) const {
        return Number(name, out);
    }
    virtual int GetF64(const char* name, double* out) const {
        return Number(name, out);
    }
    virtual int GetF32(const char* name, float* out) const {
        return Number(name, out);
    }
    virtual int GetU64(const char* name, unsigned long long* out) const {
        return Number(name, out);
    }
    virtual void Reset() {
        values.clear();
    }

    const Value* Find(const char* name) const {
        if (!name) {
            return nullptr;
        }
        const auto it = values.find(name);
        return it == values.end() ? nullptr : &it->second;
    }
    void Clear() {
        values.clear();
    }

private:
    void Put(const char* name, const Value& v) {
        if (name) {
            values[name] = v;
        }
    }

    template <typename T>
    int Number(const char* name, T* out) const {
        const Value* v = Find(name);
        if (!v || !out) {
            NoteMissing(name);
            return NgxFail;
        }
        switch (v->kind) {
        case Kind::U64:
            *out = static_cast<T>(v->u64v);
            break;
        case Kind::F32:
            *out = static_cast<T>(v->f32);
            break;
        case Kind::F64:
            *out = static_cast<T>(v->f64);
            break;
        case Kind::U32:
            *out = static_cast<T>(v->u32v);
            break;
        case Kind::I32:
            *out = static_cast<T>(v->i32);
            break;
        case Kind::Ptr:
            if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                *out = static_cast<T>(reinterpret_cast<uintptr_t>(v->ptr));
                break;
            }
            return NgxFail;
        }
        return NgxSuccess;
    }

    int Pointer(const char* name, void** out) const {
        const Value* v = Find(name);
        if (!v || !out) {
            NoteMissing(name);
            return NgxFail;
        }
        if (v->kind == Kind::Ptr) {
            *out = v->ptr;
        } else if (v->kind == Kind::U64) {
            *out = reinterpret_cast<void*>(static_cast<uintptr_t>(v->u64v));
        } else {
            return NgxFail;
        }
        return NgxSuccess;
    }

    // What the model asks for that nobody set: the first answer to "why did create fail".
    void NoteMissing(const char* name) const {
        if (!name || missing.size() >= 64 || !missing.insert(name).second) {
            return;
        }
        LOG_INFO(Render_Vulkan, "DLSS 5: the model asked for {} (not set)", name);
    }

    std::unordered_map<std::string, Value> values;
    mutable std::unordered_set<std::string> missing;
};

// ---- the NGX Vulkan resource (nvsdk_ngx_vk.h layout) --------------------------------------

struct NgxImageViewInfoVk {
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned int width;
    unsigned int height;
};
struct NgxBufferInfoVk {
    VkBuffer buffer;
    unsigned int size;
};
struct NgxResourceVk {
    union {
        NgxImageViewInfoVk image;
        NgxBufferInfoVk buffer;
    } resource;
    int type; ///< 0 = image view
    bool read_write;
};
static_assert(sizeof(NgxImageViewInfoVk) == 48 && sizeof(NgxResourceVk) == 56);

NgxResourceVk MakeResource(const Dlss::EvalImage& img, bool read_write) {
    NgxResourceVk r{};
    r.resource.image.view = static_cast<VkImageView>(img.view);
    r.resource.image.image = static_cast<VkImage>(img.image);
    r.resource.image.range = VkImageSubresourceRange{
        .aspectMask = img.depth ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT}
                                : VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT},
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    r.resource.image.format = static_cast<VkFormat>(img.format);
    r.resource.image.width = img.width;
    r.resource.image.height = img.height;
    r.type = 0;
    r.read_write = read_write;
    return r;
}

// ---- the forwarder ------------------------------------------------------------------------

using PFN_Probe = int (*)(const wchar_t*);
using PFN_Init = int (*)(const wchar_t*, const wchar_t*, void*, void*, void*, int);
using PFN_Create = void* (*)(void*, void*, unsigned int, unsigned int, int, float, int, float, float,
                             float, int, int);
using PFN_Evaluate = int (*)(void*, void*, void*, void*, void*, void*, void*, unsigned int,
                             unsigned int, unsigned int, unsigned int, int, int, float, int, float,
                             float, float, int, float, float);
using PFN_Release = void (*)(void*);
using PFN_SetFloatSlot = void (*)(int);
using PFN_ProbeFloat = void (*)(void*, const char*, float, int);
using PFN_Populate = int (*)(void*);

struct Forwarder {
#ifdef _WIN32
    HMODULE module{};
#endif
    PFN_Probe probe{};
    PFN_Init init{};
    PFN_Create create{};
    PFN_Evaluate evaluate{};
    PFN_Release release{};
    PFN_SetFloatSlot set_float_slot{};
    PFN_ProbeFloat probe_float{};
    PFN_Populate populate{}; ///< optional (our forwarder build); null = skipped
    const int* last_init{};
    const int* last_create{};
};

struct OwnedImage {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format;
    bool used = false; ///< PATH=2: written before (its contents are kept from then on)
};

constexpr vk::ImageUsageFlags OwnedUsage =
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc |
    vk::ImageUsageFlagBits::eColorAttachment; // our passes draw into them
constexpr vk::Format OwnedFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};
constexpr int MaxPasses = 3;

/// One chain of model features and its images: gameplay (1-3 passes) or video (1 pass).
struct Lane {
    std::array<void*, MaxPasses> features{};
    std::array<u32, MaxPasses> feature_gen{~0u, ~0u, ~0u};
    std::array<u64, MaxPasses> built_at{}; ///< g.applies when created
    std::unique_ptr<OwnedImage> input;     ///< the frame for the model: encoded true colour
    std::array<std::unique_ptr<OwnedImage>, MaxPasses> out; ///< each pass's answer
    std::unique_ptr<OwnedImage> composed;  ///< what goes on (DLSS 4.5 / the post pass)
    std::unique_ptr<OwnedImage> encoded;   ///< PATH=2: the encode's own image (copied to input)
    std::unique_ptr<OwnedImage> answer;    ///< PATH=2: the last pass's answer, copied out
    u32 width = 0;
    u32 height = 0;
    u32 failed_gen = ~0u; ///< the model refused pass 1 with these settings; wait for a change
    u32 eval_failures = 0;
    bool was_off = true;
    /// A later pass was refused / failed: run fewer until the settings or the pass count change
    /// (never rebuild it every frame - the fork measured that the driver stops answering).
    u32 extra_failed_gen = ~0u;
    int extra_failed_passes = 0;
    int pass_cap = MaxPasses;
    u64 last_apply = 0;                        ///< g.applies of this lane's previous run
    std::array<bool, MaxPasses> fresh{};       ///< the pass has not been evaluated yet
};

struct ComposeParams {
    u32 mode;
    u32 src_linear;
    u32 compare;
    float split;
    float transfer;
    float colour;
    float max_ratio;
    float saturation;
};
static_assert(sizeof(ComposeParams) == 32);

struct ComposePipeline {
    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline; ///< compute (SHADPS4_DLSS5_COMPUTE=1)
    // The default: dlss5_compose.frag as a full-screen draw, built like the post-process pass
    // (push descriptors of combined image samplers, fragment push constants, dynamic rendering).
    vk::UniqueDescriptorSetLayout gfx_set_layout;
    vk::UniquePipelineLayout gfx_layout;
    vk::UniquePipeline gfx_pipeline;
};

// The model state below is used by the thread that records the DLSS resolve (gameplay) and by
// the presenter's recording (video), which may be another thread: every use holds g_model_mutex
// (the fork's "one lock" rule). The settings and the status are read by the panel and the window
// thread too (under g_mutex).
struct Context {
    Forwarder fwd;
    NgxParameters params;
    std::filesystem::path dir;
    vk::Device device;
    float timestamp_period = 0.0f;

    u64 applies = 0; ///< every lane's runs: the clock retired objects wait on
    std::deque<std::pair<u64, void*>> retired_features;
    vk::PhysicalDevice physical_device;
    u32 features_alive = 0;         ///< created and not yet released
    std::deque<s64> build_times_ms; ///< recent model builds (the rebuild guard)
    s64 builds_paused_until_ms = 0;
    std::deque<std::pair<u64, std::unique_ptr<OwnedImage>>> retired_images;

    Lane game;
    Lane video;
    std::unique_ptr<OwnedImage> video_depth;  ///< R32F, one flat value
    std::unique_ptr<OwnedImage> video_motion; ///< RG16F, zero
    float video_depth_value = -1.0f;
    u64 game_frames = 0;
    u64 video_frames = 0;

    std::unique_ptr<ComposePipeline> compose;

    VkQueryPool query_pool = VK_NULL_HANDLE;
    bool slot_written[TimingSlots]{};
    float window_sum = 0.0f;
    float window_max = 0.0f;
    u32 window_count = 0;
};

Context g;
std::mutex g_model_mutex; ///< g (the model, its features and images)
std::mutex g_mutex;       ///< g_settings, g_status
std::mutex g_file_mutex;  ///< dlss5-settings.ini
Settings g_settings;
Status g_status;
std::atomic<u32> g_generation{1};
std::atomic<State> g_state{State::Off};

void SetStatus(State state, std::string message) {
    std::scoped_lock lock{g_mutex};
    g_status.state = state;
    g_status.message = std::move(message);
    g_state.store(state, std::memory_order_release);
}

void SetMessage(std::string message) {
    std::scoped_lock lock{g_mutex};
    g_status.message = std::move(message);
}

std::filesystem::path ExeDir() {
#ifdef _WIN32
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return {};
        }
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    return std::filesystem::path{buf}.parent_path();
#else
    return {};
#endif
}

bool HasBoth(const std::filesystem::path& dir) {
    std::error_code ec;
    return !dir.empty() && std::filesystem::is_regular_file(dir / ForwarderName, ec) &&
           std::filesystem::is_regular_file(dir / ModelName, ec);
}

// ---- settings file (key = value, one per line) ---------------------------------------------

void LoadSettingsFile() {
    std::ifstream in{g.dir / SettingsName};
    if (!in) {
        return;
    }
    Settings s = DefaultSettings();
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos || line.starts_with('#')) {
            continue;
        }
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
        value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
        const float f = static_cast<float>(std::atof(value.c_str()));
        const int i = std::atoi(value.c_str());
        if (key == "enabled") {
            s.enabled = i != 0;
        } else if (key == "style") {
            s.style = std::clamp(i, 0, 2);
        } else if (key == "intensity") {
            s.intensity = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "local_structure") {
            s.local_structure = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "local_tone") {
            s.local_tone = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "skin_follows") {
            s.skin_follows = i != 0;
        } else if (key == "skin_structure") {
            s.skin_structure = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "auto_mask") {
            s.auto_mask = i != 0;
        } else if (key == "preset") {
            s.preset = std::clamp(i, 0, 15);
        } else if (key == "ui_correction") {
            s.ui_correction = i != 0;
        } else if (key == "mv_mode") {
            s.mv_mode = std::clamp(i, 0, 2);
        } else if (key == "video") {
            s.video = i != 0;
        } else if (key == "video_depth") {
            s.video_depth = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "video_fresh") {
            s.video_fresh = i != 0;
        } else if (key == "passes") {
            s.passes = std::clamp(i, 1, MaxPasses);
        } else if (key == "transfer") {
            s.transfer = std::clamp(f, 0.0f, 1.5f);
        } else if (key == "colour") {
            s.colour = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "saturation") {
            s.saturation = std::clamp(f, 0.5f, 1.5f);
        } else if (key == "max_ratio") {
            s.max_ratio = std::clamp(f, 1.1f, 8.0f);
        } else if (key == "compare") {
            s.compare = i != 0;
        } else if (key == "split") {
            s.split = std::clamp(f, 0.0f, 1.0f);
        }
    }
    std::scoped_lock lock{g_mutex};
    g_settings = s;
}

void SaveSettingsFile(const Settings& s) {
    if (g.dir.empty()) {
        return;
    }
    std::scoped_lock lock{g_file_mutex};
    std::ofstream out{g.dir / SettingsName, std::ios::trunc};
    if (!out) {
        return;
    }
    out << "# DLSS 5 panel settings (Page Up in game). Deleting this file restores the defaults.\n"
        << "enabled = " << (s.enabled ? 1 : 0) << "\n"
        << "style = " << s.style << "\n"
        << "intensity = " << s.intensity << "\n"
        << "local_structure = " << s.local_structure << "\n"
        << "local_tone = " << s.local_tone << "\n"
        << "skin_follows = " << (s.skin_follows ? 1 : 0) << "\n"
        << "skin_structure = " << s.skin_structure << "\n"
        << "auto_mask = " << (s.auto_mask ? 1 : 0) << "\n"
        << "preset = " << s.preset << "\n"
        << "ui_correction = " << (s.ui_correction ? 1 : 0) << "\n"
        << "mv_mode = " << s.mv_mode << "\n"
        << "video = " << (s.video ? 1 : 0) << "\n"
        << "video_depth = " << s.video_depth << "\n"
        << "video_fresh = " << (s.video_fresh ? 1 : 0) << "\n"
        << "passes = " << s.passes << "\n"
        << "transfer = " << s.transfer << "\n"
        << "colour = " << s.colour << "\n"
        << "saturation = " << s.saturation << "\n"
        << "max_ratio = " << s.max_ratio << "\n"
        << "compare = " << (s.compare ? 1 : 0) << "\n"
        << "split = " << s.split << "\n";
}

// ---- start-up -----------------------------------------------------------------------------

template <typename T>
bool Resolve(T& out, const char* name) {
#ifdef _WIN32
    out = reinterpret_cast<T>(GetProcAddress(g.fwd.module, name));
#endif
    if (!out) {
        LOG_ERROR(Render_Vulkan, "DLSS 5: the forwarder has no {}", name);
    }
    return out != nullptr;
}

// Proves the vtable slots the forwarder writes through land on the right setters of our block,
// and finds the float one (the forwarder's own block probe; see the NgxParameters comment).
bool SelfTest() {
    NgxParameters& p = g.params;
    p.Clear();
    void** vt = *reinterpret_cast<void***>(&p);
    using SetU64 = void (*)(void*, const char*, unsigned long long);
    using SetU32 = void (*)(void*, const char*, unsigned int);
    reinterpret_cast<SetU64>(vt[0])(&p, "selftest.ptr", 0x1122334455667788ull);
    reinterpret_cast<SetU32>(vt[3])(&p, "selftest.uint", 7u);
    const auto* ptr = p.Find("selftest.ptr");
    const auto* uint = p.Find("selftest.uint");
    const bool ptr_ok = ptr && ptr->kind == NgxParameters::Kind::Ptr &&
                        reinterpret_cast<uintptr_t>(ptr->ptr) == 0x1122334455667788ull;
    const bool uint_ok = uint && uint->kind == NgxParameters::Kind::I32 && uint->i32 == 7;
    int float_slot = -1;
    for (int slot = 0; slot < 8 && float_slot < 0; ++slot) {
        char name[32];
        std::snprintf(name, sizeof(name), "selftest.float%d", slot);
        g.fwd.probe_float(&p, name, 1.25f, slot);
        const auto* f = p.Find(name);
        if (f && f->kind == NgxParameters::Kind::F32 && f->f32 == 1.25f) {
            float_slot = slot;
        }
    }
    p.Clear();
    LOG_INFO(Render_Vulkan, "DLSS 5: parameter block slots: 0 {}, 3 {}, float {}",
             ptr_ok ? "pointer (ok)" : "WRONG", uint_ok ? "int (ok)" : "WRONG", float_slot);
    // 6 is where MSVC puts Set(float) in this declaration; anything else means the model would
    // call the wrong getters too.
    if (!ptr_ok || !uint_ok || float_slot != 6) {
        return false;
    }
    g.fwd.set_float_slot(float_slot);
    return true;
}

// ---- images, features, timing ---------------------------------------------------------------

std::unique_ptr<OwnedImage> MakeImage(const Instance& instance, u32 width, u32 height,
                                      vk::Format format, const char* name) {
    auto t = std::make_unique<OwnedImage>();
    t->format = format;
    t->image = VideoCore::UniqueImage(g.device, instance.GetAllocator());
    t->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent{width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = OwnedUsage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    SetObjectName(g.device, static_cast<vk::Image>(t->image), "{}", name);
    // For SHADPS4_ADDR_BIND_REPORT's device-fault dump (which object held a faulting address).
    LOG_INFO(Render_Vulkan, "DLSS 5: {} {}x{} = image {:#x}", name, width, height,
             reinterpret_cast<u64>(static_cast<VkImage>(t->image.image)));
    t->view = Check<"create dlss5 view">(g.device.createImageViewUnique({
        .image = t->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = ColorRange,
    }));
    return t;
}

Dlss::EvalImage AsEval(const OwnedImage& img, u32 width, u32 height) {
    return Dlss::EvalImage{
        .image = img.image,
        .view = img.view.get(),
        .format = img.format,
        .layout = vk::ImageLayout::eGeneral,
        .usage = OwnedUsage,
        .width = width,
        .height = height,
        .depth = false,
    };
}

void Retire(std::unique_ptr<OwnedImage>& img) {
    if (img) {
        g.retired_images.emplace_back(g.applies, std::move(img));
    }
}

void RetireFeature(Lane& lane, int pass) {
    if (lane.features[pass]) {
        g.retired_features.emplace_back(g.applies, lane.features[pass]);
        lane.features[pass] = nullptr;
    }
}

void RetireLane(Lane& lane) {
    for (int k = 0; k < MaxPasses; ++k) {
        RetireFeature(lane, k);
        Retire(lane.out[k]);
    }
    Retire(lane.input);
    Retire(lane.composed);
    Retire(lane.encoded);
    Retire(lane.answer);
    lane.width = 0;
    lane.height = 0;
}

void FreeRetired(bool all) {
    while (!g.retired_features.empty() &&
           (all || g.retired_features.front().first + RetireAfter <= g.applies)) {
        if (g.fwd.release) { // null in the no-model test
            g.fwd.release(g.retired_features.front().second);
        }
        --g.features_alive;
        g.retired_features.pop_front();
    }
    while (!g.retired_images.empty() &&
           (all || g.retired_images.front().first + RetireAfter <= g.applies)) {
        g.retired_images.pop_front();
    }
}

/// The lane's images at w x h: input, `passes` answers, composed (made / remade as needed).
void EnsureImages(const Instance& instance, Lane& lane, u32 w, u32 h, int passes,
                  const char* name) {
    if (lane.input && (lane.width != w || lane.height != h)) {
        RetireLane(lane);
    }
    if (!lane.input) {
        lane.input = MakeImage(instance, w, h, OwnedFormat, "DLSS 5 input");
        lane.composed = MakeImage(instance, w, h, OwnedFormat, "DLSS 5 composed");
        lane.width = w;
        lane.height = h;
        LOG_INFO(Render_Vulkan, "DLSS 5: {} images {}x{}", name, w, h);
    }
    if (Test().path == 2 && !lane.encoded) {
        lane.encoded = MakeImage(instance, w, h, OwnedFormat, "DLSS 5 encoded");
        lane.answer = MakeImage(instance, w, h, OwnedFormat, "DLSS 5 answer");
    }
    for (int k = 0; k < passes; ++k) {
        if (!lane.out[k]) {
            lane.out[k] = MakeImage(instance, w, h, OwnedFormat, "DLSS 5 pass");
        }
    }
}

void EnsureCompose() {
    if (g.compose) {
        return;
    }
    auto c = std::make_unique<ComposePipeline>();
    c->sampler = Check<"create dlss5 sampler">(g.device.createSamplerUnique(vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eNearest,
        .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 0.0f,
    }));
    const vk::Sampler sampler = c->sampler.get();
    const std::array<vk::DescriptorSetLayoutBinding, 5> bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 3,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 4,
         .descriptorType = vk::DescriptorType::eSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute,
         .pImmutableSamplers = &sampler},
    }};
    c->set_layout = Check<"create dlss5 set layout">(g.device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange push{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(ComposeParams),
    };
    c->layout = Check<"create dlss5 layout">(g.device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &c->set_layout.get(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push,
    }));
    const auto module = Compile(HostShaders::DLSS5_COMPOSE_COMP, vk::ShaderStageFlagBits::eCompute,
                                g.device);
    ASSERT(module);
    c->pipeline = Check<"create dlss5 pipeline">(g.device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = module,
                    .pName = "main",
                },
                .layout = c->layout.get(),
            }));
    g.device.destroyShaderModule(module);

    const std::array<vk::DescriptorSetLayoutBinding, 3> gfx_bindings{{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eFragment},
    }};
    c->gfx_set_layout = Check<"create dlss5 draw set layout">(g.device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
        .bindingCount = static_cast<u32>(gfx_bindings.size()),
        .pBindings = gfx_bindings.data(),
    }));
    const vk::PushConstantRange gfx_push{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(ComposeParams),
    };
    c->gfx_layout = Check<"create dlss5 draw layout">(g.device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &c->gfx_set_layout.get(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &gfx_push,
    }));
    const auto vs = Compile(HostShaders::FS_TRI_VERT, vk::ShaderStageFlagBits::eVertex, g.device);
    const auto fs =
        Compile(HostShaders::DLSS5_COMPOSE_FRAG, vk::ShaderStageFlagBits::eFragment, g.device);
    ASSERT(vs && fs);
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex, .module = vs, .pName = "main"},
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment, .module = fs, .pName = "main"},
    };
    const vk::Format color_format = OwnedFormat;
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport_state{
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const vk::PipelineRasterizationStateCreateInfo raster{
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const vk::PipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend{
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };
    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    c->gfx_pipeline = Check<"create dlss5 draw pipeline">(g.device.createGraphicsPipelineUnique(
        {}, vk::GraphicsPipelineCreateInfo{
                .pNext = &rendering,
                .stageCount = static_cast<u32>(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &raster,
                .pMultisampleState = &multisample,
                .pColorBlendState = &blend,
                .pDynamicState = &dynamic,
                .layout = c->gfx_layout.get(),
            }));
    g.device.destroyShaderModule(vs);
    g.device.destroyShaderModule(fs);
    g.compose = std::move(c);
}

/// One pass of dlss5_compose.comp (SHADPS4_DLSS5_COMPUTE=1). `src` is sampled in `src_layout`;
/// the three storage images are in GENERAL. Unused bindings get valid views anyway (the shader
/// declares them all).
void DispatchCompute(vk::CommandBuffer cmd, const ComposeParams& params, vk::ImageView src,
                     vk::ImageLayout src_layout, vk::ImageView original, vk::ImageView model,
                     vk::ImageView dst, u32 w, u32 h) {
    const vk::DescriptorImageInfo src_info{.imageView = src, .imageLayout = src_layout};
    const vk::DescriptorImageInfo original_info{.imageView = original,
                                                .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo model_info{.imageView = model,
                                             .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo dst_info{.imageView = dst,
                                           .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 4> writes{{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &src_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &original_info},
        {.dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &model_info},
        {.dstBinding = 3,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &dst_info},
    }};
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, g.compose->pipeline.get());
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, g.compose->layout.get(), 0, writes);
    cmd.pushConstants(g.compose->layout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(ComposeParams), &params);
    cmd.dispatch((w + 7) / 8, (h + 7) / 8, 1);
}

/// One pass of dlss5_compose.frag: a full-screen triangle into `dst` (GENERAL), recorded exactly
/// like the post-process pass that runs in the same command buffers (pipeline, viewport, scissor,
/// push descriptors, push constants, then the rendering). `src` is sampled in `src_layout`,
/// `original` / `model` in GENERAL; the encode modes read `src` only, so every binding gets it
/// (dst must not be bound while it is drawn into).
void DrawPass(vk::CommandBuffer cmd, const ComposeParams& params, vk::ImageView src,
              vk::ImageLayout src_layout, vk::ImageView original, vk::ImageView model,
              vk::ImageView dst, u32 w, u32 h) {
    const bool encode = params.mode <= 1;
    const vk::Sampler sampler = g.compose->sampler.get();
    const vk::DescriptorImageInfo src_info{
        .sampler = sampler, .imageView = src, .imageLayout = src_layout};
    const vk::DescriptorImageInfo original_info{
        .sampler = sampler,
        .imageView = encode ? src : original,
        .imageLayout = encode ? src_layout : vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo model_info{
        .sampler = sampler,
        .imageView = encode ? src : model,
        .imageLayout = encode ? src_layout : vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 3> writes{{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &src_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &original_info},
        {.dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .pImageInfo = &model_info},
    }};
    const vk::RenderingAttachmentInfo attachment{
        .imageView = dst,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = vk::AttachmentLoadOp::eDontCare, // every pixel is drawn
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering{
        .renderArea{.extent{.width = w, .height = h}},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    };
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, g.compose->gfx_pipeline.get());
    cmd.setViewport(0, vk::Viewport{.x = 0.0f,
                                    .y = 0.0f,
                                    .width = static_cast<float>(w),
                                    .height = static_cast<float>(h),
                                    .minDepth = 0.0f,
                                    .maxDepth = 1.0f});
    cmd.setScissor(0, vk::Rect2D{.extent{.width = w, .height = h}});
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, g.compose->gfx_layout.get(), 0,
                             writes);
    cmd.pushConstants(g.compose->gfx_layout.get(), vk::ShaderStageFlagBits::eFragment, 0,
                      sizeof(ComposeParams), &params);
    cmd.beginRendering(rendering);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
}

/// One of our passes: a full-screen draw (default) or the old compute dispatch.
void Dispatch(vk::CommandBuffer cmd, const ComposeParams& params, vk::ImageView src,
              vk::ImageLayout src_layout, vk::ImageView original, vk::ImageView model,
              vk::ImageView dst, u32 w, u32 h) {
    if (Test().compute) {
        DispatchCompute(cmd, params, src, src_layout, original, model, dst, w, h);
    } else {
        DrawPass(cmd, params, src, src_layout, original, model, dst, w, h);
    }
}

/// Everything written before is visible to everything after (one global barrier: these passes
/// are a handful per frame).
void FullBarrier(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

/// The images about to be fully rewritten: discard into GENERAL after all earlier use.
void ToGeneral(vk::CommandBuffer cmd, std::initializer_list<const OwnedImage*> images) {
    std::array<vk::ImageMemoryBarrier2, MaxPasses + 4> barriers{};
    u32 n = 0;
    for (const OwnedImage* img : images) {
        if (!img || n >= barriers.size()) {
            continue;
        }
        barriers[n++] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead |
                             vk::AccessFlagBits2::eTransferWrite |
                             vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eColorAttachmentRead,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = img->image,
            .subresourceRange = ColorRange,
        };
    }
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = n,
        .pImageMemoryBarriers = barriers.data(),
    });
}

/// PATH=2: the model's images keep their contents once written (GENERAL to GENERAL after all
/// earlier use); only the first use discards from UNDEFINED.
void KeepGeneral(vk::CommandBuffer cmd, std::initializer_list<OwnedImage*> images) {
    std::array<vk::ImageMemoryBarrier2, MaxPasses + 4> barriers{};
    u32 n = 0;
    for (OwnedImage* img : images) {
        if (!img || n >= barriers.size()) {
            continue;
        }
        barriers[n++] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = img->used ? vk::AccessFlagBits2::eMemoryWrite
                                       : vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = img->used ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = img->image,
            .subresourceRange = ColorRange,
        };
        img->used = true;
    }
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = n,
        .pImageMemoryBarriers = barriers.data(),
    });
}

/// A whole-image copy between two same-size images, both in GENERAL (barriers: the caller's).
void CopyWhole(vk::CommandBuffer cmd, vk::Image src, vk::Image dst, u32 w, u32 h) {
    const vk::ImageSubresourceLayers layers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const vk::ImageCopy region{
        .srcSubresource = layers,
        .dstSubresource = layers,
        .extent = vk::Extent3D{w, h, 1},
    };
    cmd.copyImage(src, vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
}

s64 NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Device-local VRAM in use / budget in MB (VK_EXT_memory_budget, which the instance enables),
/// also put into the status for the panel.
std::pair<u32, u32> UpdateVram() {
    if (!g.physical_device) {
        return {0, 0};
    }
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props{.pNext = &budget};
    g.physical_device.getMemoryProperties2(&props);
    u64 used = 0;
    u64 total = 0;
    for (u32 i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
        if (props.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            used += budget.heapUsage[i];
            total += budget.heapBudget[i];
        }
    }
    const std::pair<u32, u32> mb{static_cast<u32>(used >> 20), static_cast<u32>(total >> 20)};
    std::scoped_lock lock{g_mutex};
    g_status.vram_used_mb = mb.first;
    g_status.vram_budget_mb = mb.second;
    g_status.features_alive = g.features_alive;
    return mb;
}

/// The rebuild guard (2026-10-18: at cutscene exits the two lanes rebuilt each other ~40 times in
/// 10 s; VRAM ran over budget - Streamline warned on every frame for the rest of the session and
/// the game fell to ~8 FPS). More than 12 builds in 10 s pause new builds for 10 s; the frame
/// goes through untouched meanwhile. False = do not build now.
bool BuildAllowed() {
    const s64 now = NowMs();
    if (now < g.builds_paused_until_ms) {
        return false;
    }
    while (!g.build_times_ms.empty() && now - g.build_times_ms.front() > 10000) {
        g.build_times_ms.pop_front();
    }
    if (g.build_times_ms.size() >= 12) {
        g.builds_paused_until_ms = now + 10000;
        g.build_times_ms.clear();
        LOG_WARNING(Render_Vulkan, "DLSS 5: 12 model builds in 10 s - new builds paused for 10 s "
                                   "(each one holds VRAM until it is released)");
        SetMessage("Paused 10 s: too many rebuilds in a row (they fill the graphics memory).");
        return false;
    }
    g.build_times_ms.push_back(now);
    return true;
}

ComposeParams MakeParams(u32 mode, const Settings& s, bool src_linear) {
    return ComposeParams{
        .mode = mode,
        .src_linear = src_linear ? 1u : 0u,
        .compare = s.compare ? 1u : 0u,
        .split = s.split,
        .transfer = s.transfer,
        .colour = s.colour,
        .max_ratio = s.max_ratio,
        .saturation = s.saturation,
    };
}

void* CreateFeature(vk::CommandBuffer cmd, const Settings& s, u32 w, u32 h) {
    if (Test().no_model) {
        static uintptr_t next = 0x1000;
        next += 0x10;
        return reinterpret_cast<void*>(next); // a stand-in, never dereferenced
    }
    const float skin = s.skin_follows ? -1.0f : s.skin_structure;
    return g.fwd.create(static_cast<VkCommandBuffer>(cmd), &g.params, w, h, s.preset, s.intensity,
                        s.style, s.local_structure, s.local_tone, skin, s.auto_mask ? 1 : 0,
                        s.ui_correction ? 1 : 0);
}

/// One model run. False = refused (logged by the caller).
bool Evaluate(vk::CommandBuffer cmd, void* feature, const Settings& s, const Dlss::EvalImage& color,
              const Dlss::EvalImage& depth, const Dlss::EvalImage& motion,
              const Dlss::EvalImage& output, float mv_x, float mv_y, bool reset, int* result) {
    if (Test().no_model) { // the "model": a copy (both images are ours, in GENERAL)
        CopyWhole(cmd, color.image, output.image, output.width, output.height);
        *result = 1;
        return true;
    }
    const float skin = s.skin_follows ? -1.0f : s.skin_structure;
    NgxResourceVk res_color = MakeResource(color, false);
    NgxResourceVk res_depth = MakeResource(depth, false);
    NgxResourceVk res_motion = MakeResource(motion, false);
    NgxResourceVk res_output = MakeResource(output, true);
    *result = g.fwd.evaluate(static_cast<VkCommandBuffer>(cmd), feature, &g.params, &res_color,
                             &res_depth, &res_motion, &res_output, color.width, color.height,
                             depth.width, depth.height, /*depth inverted*/ 1, reset ? 1 : 0,
                             s.intensity, s.style, s.local_structure, s.local_tone, skin,
                             s.auto_mask ? 1 : 0, mv_x, mv_y);
    // The block outlives this call and the four resources live on this stack: a later create
    // must not find them.
    for (const char* name : {"DLSSNR.Color", "DLSSNR.Depth", "DLSSNR.MVec", "DLSSNR.Output"}) {
        g.params.SetPtr(name, nullptr);
    }
    return *result == 1;
}

void ReadTiming() {
    if (g.query_pool == VK_NULL_HANDLE) {
        return;
    }
    // The oldest slot: written TimingSlots - 1 gameplay frames ago, long executed.
    const u32 slot = static_cast<u32>((g.game_frames + 1) % TimingSlots);
    if (!g.slot_written[slot]) {
        return;
    }
    u64 data[4]{};
    const VkResult r = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
        static_cast<VkDevice>(g.device), g.query_pool, slot * 2, 2, sizeof(data), data,
        sizeof(u64) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if ((r != VK_SUCCESS && r != VK_NOT_READY) || data[1] == 0 || data[3] == 0) {
        return;
    }
    g.slot_written[slot] = false;
    const float ms =
        static_cast<float>(data[2] - data[0]) * g.timestamp_period / 1'000'000.0f;
    g.window_sum += ms;
    g.window_max = std::max(g.window_max, ms);
    if (++g.window_count >= 120) {
        const float avg = g.window_sum / static_cast<float>(g.window_count);
        {
            std::scoped_lock lock{g_mutex};
            g_status.gpu_ms_avg = avg;
            g_status.gpu_ms_max = g.window_max;
        }
        if ((g.game_frames / 120) % 5 == 0) {
            LOG_INFO(Render_Vulkan, "DLSS 5: GPU time {:.2f} ms avg, {:.2f} ms max ({} frames)", avg,
                     g.window_max, g.game_frames);
        }
        g.window_sum = 0.0f;
        g.window_max = 0.0f;
        g.window_count = 0;
    }
}

void SetRunning(bool running, int passes) {
    std::scoped_lock lock{g_mutex};
    g_status.running = running;
    g_status.video_running = false; // a gameplay frame: not a video frame
    g_status.passes_running = running ? passes : 0;
    g_status.frames = g.game_frames;
    g_status.width = g.game.width;
    g_status.height = g.game.height;
}

void SetVideoRunning(bool running) {
    std::scoped_lock lock{g_mutex};
    g_status.video_running = running;
    g_status.running = false;
    g_status.video_frames = g.video_frames;
}

/// Builds / rebuilds the lane's pass-`k` feature for this generation. Pass 0 is used the frame
/// it is built; a later pass only from the next frame (the fork's rule: building one and running
/// it in the same command buffer hung the GPU in multi-pass). False = refused.
bool EnsureFeature(vk::CommandBuffer cmd, Lane& lane, int k, const Settings& s, u32 gen, u32 w,
                   u32 h, const char* name) {
    if (lane.features[k] && lane.feature_gen[k] == gen) {
        return true;
    }
    if (!BuildAllowed()) {
        return false; // paused, not refused: no failure bookkeeping
    }
    RetireFeature(lane, k);
    lane.features[k] = CreateFeature(cmd, s, w, h);
    if (lane.features[k]) {
        ++g.features_alive;
    }
    if (!lane.features[k]) {
        const int code = g.fwd.last_create ? *g.fwd.last_create : 0;
        LOG_ERROR(Render_Vulkan, "DLSS 5: the model would not build {} pass {} ({:#x}) at {}x{}",
                  name, k + 1, static_cast<u32>(code), w, h);
        if (k == 0) {
            lane.failed_gen = gen;
            SetMessage(fmt::format("The model refused these settings (code {:#x}). Change a "
                                   "setting or press Defaults.",
                                   static_cast<u32>(code)));
        } else {
            // A later pass only: keep the passes that work (review 2).
            lane.extra_failed_gen = gen;
            lane.extra_failed_passes = s.passes;
            lane.pass_cap = k;
        }
        return false;
    }
    lane.feature_gen[k] = gen;
    lane.built_at[k] = g.applies;
    lane.fresh[k] = true;
    lane.eval_failures = 0;
    const auto [vram_used, vram_budget] = UpdateVram();
    LOG_INFO(Render_Vulkan,
             "DLSS 5: {} pass {} up at {}x{} (style {}, network {}, intensity {:.2f}, structure "
             "{:.2f}, tone {:.2f}, skin {:.2f}, auto mask {}) | model copies alive {}, VRAM {} / "
             "{} MB",
             name, k + 1, w, h, s.style, s.preset, s.intensity, s.local_structure, s.local_tone,
             s.skin_follows ? -1.0f : s.skin_structure, s.auto_mask, g.features_alive, vram_used,
             vram_budget);
    return true;
}

void EvalFailed(Lane& lane, u32 gen, int result, const char* name) {
    if (++lane.eval_failures <= 8) {
        LOG_ERROR(Render_Vulkan, "DLSS 5: {} evaluate failed ({:#x})", name,
                  static_cast<u32>(result));
    }
    if (lane.eval_failures >= 8) {
        for (int k = 0; k < MaxPasses; ++k) {
            RetireFeature(lane, k);
        }
        lane.failed_gen = gen;
        SetMessage(fmt::format("The model stopped (evaluate {:#x}). Change a setting to retry.",
                               static_cast<u32>(result)));
    }
}

/// SHADPS4_DLSS5_PATH=1: round 1's gameplay path (e491fade), for the crash bisect: the game's
/// frame blitted into the model's input as it is (red / blue swapped), one pass, the answer
/// straight to DLSS 4.5. None of our compute passes run.
std::optional<Dlss::EvalImage> ApplyRound1(vk::CommandBuffer cmdbuf, Lane& lane, const Settings& s,
                                           const Dlss::EvalImage& color,
                                           const Dlss::EvalImage& motion,
                                           const Dlss::EvalImage& depth, bool reset, u32 gen,
                                           u32 w, u32 h) {
    const std::array<vk::ImageMemoryBarrier2, 3> before{{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = color.layout,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .image = color.image,
            .subresourceRange = ColorRange,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = lane.input->image,
            .subresourceRange = ColorRange,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined, // the model writes all of it
            .newLayout = vk::ImageLayout::eGeneral,
            .image = lane.out[0]->image,
            .subresourceRange = ColorRange,
        },
    }};
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(before.size()),
        .pImageMemoryBarriers = before.data(),
    });
    vk::ImageBlit blit{};
    blit.srcSubresource = vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0,
        .layerCount = 1};
    blit.srcOffsets[1] = vk::Offset3D{static_cast<s32>(w), static_cast<s32>(h), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1] = blit.srcOffsets[1];
    cmdbuf.blitImage(color.image, vk::ImageLayout::eTransferSrcOptimal, lane.input->image,
                     vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
    const std::array<vk::ImageMemoryBarrier2, 2> after_copy{{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = color.layout,
            .image = color.image,
            .subresourceRange = ColorRange,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = lane.input->image,
            .subresourceRange = ColorRange,
        },
    }};
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(after_copy.size()),
        .pImageMemoryBarriers = after_copy.data(),
    });
    const u32 slot = static_cast<u32>(g.game_frames % TimingSlots);
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.resetQueryPool(vk::QueryPool{g.query_pool}, slot * 2, 2);
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2);
    }
    const float mv_x = s.mv_mode == 0 ? -static_cast<float>(w) : s.mv_mode == 1 ? -1.0f : 0.0f;
    const float mv_y = s.mv_mode == 0 ? -static_cast<float>(h) : s.mv_mode == 1 ? -1.0f : 0.0f;
    int result = 0;
    const bool ok = Evaluate(cmdbuf, lane.features[0], s, AsEval(*lane.input, w, h), depth, motion,
                             AsEval(*lane.out[0], w, h), mv_x, mv_y, reset || lane.fresh[0],
                             &result);
    lane.fresh[0] = false;
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2 + 1);
        g.slot_written[slot] = true;
    }
    if (!ok) {
        EvalFailed(lane, gen, result, "gameplay");
        SetRunning(false, 0);
        return std::nullopt;
    }
    lane.eval_failures = 0;
    const vk::ImageMemoryBarrier2 written{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = lane.out[0]->image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &written,
    });
    ++g.game_frames;
    if (g.game_frames == 1 || (g.game_frames % 1800) == 0) {
        LOG_INFO(Render_Vulkan, "DLSS 5: running ({} gameplay frames, round 1 path)",
                 g.game_frames);
    }
    SetRunning(true, 1);
    return AsEval(*lane.out[0], w, h);
}

} // namespace

Settings DefaultSettings() {
    return Settings{};
}

bool Wanted() {
    static const bool wanted = [] {
        if (Dlss::Mode() != 1) {
            return false;
        }
        const char* env = std::getenv("SHADPS4_DLSS5");
        if (env && env[0] == '0') {
            LOG_INFO(Render_Vulkan, "DLSS 5: off (SHADPS4_DLSS5=0)");
            return false;
        }
        std::vector<std::filesystem::path> candidates;
        if (const char* dir = std::getenv("SHADPS4_DLSS5_DIR"); dir && dir[0]) {
            candidates.emplace_back(dir);
        }
        const auto exe = ExeDir();
        candidates.push_back(exe / "dlss5");
        candidates.push_back(exe / ".." / ".." / "dlss5");
        for (const auto& c : candidates) {
            if (HasBoth(c)) {
                std::error_code ec;
                g.dir = std::filesystem::weakly_canonical(c, ec);
                if (ec) {
                    g.dir = c;
                }
                break;
            }
        }
        if (g.dir.empty()) {
            LOG_INFO(Render_Vulkan, "DLSS 5: not found - DLSS 4.5 alone (put {} and {} in {})",
                     ModelName, ForwarderName, Common::FS::PathToUTF8String(exe / "dlss5"));
            SetStatus(State::NotFound, "Not found: put nvngx_dlssnr.dll and nvngx.dll_dlssnr.dll "
                                       "in the dlss5 folder next to shadPS4.exe.");
            return false;
        }
        {
            std::scoped_lock lock{g_mutex};
            g_status.folder = Common::FS::PathToUTF8String(g.dir);
        }
        LoadSettingsFile();
        LOG_INFO(Render_Vulkan, "DLSS 5: found in {} (unofficial model - private use)",
                 Common::FS::PathToUTF8String(g.dir));
        SetStatus(State::Starting, "Starting...");
        return true;
    }();
    return wanted;
}

const std::vector<const char*>& DeviceExtensions() {
    return g_extensions;
}

void Disable(const char* why) {
    if (Wanted() && g_state.load(std::memory_order_acquire) == State::Starting) {
        LOG_WARNING(Render_Vulkan, "DLSS 5: off - {}", why);
        SetStatus(State::Failed, std::string{"Off: "} + why);
    }
}

#ifdef _WIN32
/// Loads the forwarder and the model and starts it on our device. False = failed (logged, the
/// status says why).
static bool LoadModel(vk::Instance instance, vk::PhysicalDevice physical_device,
                      vk::Device device) {
    const auto fail = [](const std::string& why) {
        LOG_ERROR(Render_Vulkan, "DLSS 5: {} - DLSS 4.5 alone", why);
        SetStatus(State::Failed, why);
    };
    const std::filesystem::path forwarder = g.dir / ForwarderName;
    const std::filesystem::path model = g.dir / ModelName;
    g.fwd.module = LoadLibraryExW(forwarder.wstring().c_str(), nullptr,
                                  LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g.fwd.module) {
        fail(fmt::format("could not load {} (error {})", Common::FS::PathToUTF8String(forwarder),
                         GetLastError()));
        return false;
    }
    bool ok = Resolve(g.fwd.probe, "dlssnr_vk_probe") && Resolve(g.fwd.init, "dlssnr_vk_init") &&
              Resolve(g.fwd.create, "dlssnr_vk_create") &&
              Resolve(g.fwd.evaluate, "dlssnr_vk_evaluate") &&
              Resolve(g.fwd.release, "dlssnr_vk_release") &&
              Resolve(g.fwd.set_float_slot, "dlssnr_call_set_float_slot") &&
              Resolve(g.fwd.probe_float, "dlssnr_call_probe_float") &&
              Resolve(g.fwd.last_init, "dlssnr_vk_last_init") &&
              Resolve(g.fwd.last_create, "dlssnr_vk_last_create");
    if (!ok) {
        fail("the forwarder is missing entry points (wrong build?)");
        return false;
    }
    g.fwd.populate = reinterpret_cast<PFN_Populate>(GetProcAddress(g.fwd.module, "dlssnr_vk_populate"));
    if (!SelfTest()) {
        fail("the parameter block layout check failed");
        return false;
    }
    LOG_INFO(Render_Vulkan, "DLSS 5: loading the model ({})", Common::FS::PathToUTF8String(model));
    const int probe = g.fwd.probe(model.wstring().c_str());
    if (probe != 15) {
        fail(fmt::format("the model's Vulkan entry points are incomplete (probe {})", probe));
        return false;
    }
    std::error_code ec;
    const std::filesystem::path data = g.dir / "ngx-data";
    std::filesystem::create_directories(data, ec);
    const int init = g.fwd.init(model.wstring().c_str(), data.wstring().c_str(),
                                static_cast<VkInstance>(instance),
                                static_cast<VkPhysicalDevice>(physical_device),
                                static_cast<VkDevice>(device), NgxSdkVersion);
    if (init != 1) {
        fail(fmt::format("the model would not start on this GPU (init {:#x})",
                         static_cast<u32>(init)));
        return false;
    }
    if (g.fwd.populate) {
        const int populated = g.fwd.populate(&g.params);
        LOG_INFO(Render_Vulkan, "DLSS 5: the model filled its parameter defaults ({:#x})",
                 static_cast<u32>(populated));
    } else {
        LOG_WARNING(Render_Vulkan, "DLSS 5: the forwarder has no dlssnr_vk_populate (an upstream "
                                   "build?) - the parameter block starts empty");
    }
    return true;
}
#endif

void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
#ifdef _WIN32
    if (!Wanted() || g_state.load(std::memory_order_acquire) != State::Starting) {
        return;
    }
    std::scoped_lock model_lock{g_model_mutex};
    const Switches& t = Test();
    if (t.reset || t.path != 0 || t.no_model || t.compute) {
        LOG_WARNING(Render_Vulkan,
                    "DLSS 5: test switches - reset every run {}, path {}, no model {}, compute "
                    "passes {}",
                    t.reset, t.path, t.no_model, t.compute);
    }
    if (t.no_model) {
        LOG_WARNING(Render_Vulkan, "DLSS 5: NO-MODEL test (SHADPS4_DLSS5_NOMODEL=1) - nothing "
                                   "loaded, every model run is a copy");
    } else if (!LoadModel(instance, physical_device, device)) {
        return;
    }
    g.device = device;
    g.physical_device = physical_device;
    g.timestamp_period = physical_device.getProperties().limits.timestampPeriod;
    if (g.timestamp_period > 0.0f) {
        const VkQueryPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = TimingSlots * 2,
        };
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateQueryPool(static_cast<VkDevice>(device), &info,
                                                            nullptr, &g.query_pool) != VK_SUCCESS) {
            g.query_pool = VK_NULL_HANDLE;
        }
    }
    LOG_INFO(Render_Vulkan, "DLSS 5: ready (End = on / off, Page Up = panel)");
    SetStatus(State::Ready, "Ready");
#else
    (void)instance;
    (void)physical_device;
    (void)device;
#endif
}

void Shutdown() {
    if (g_state.load(std::memory_order_acquire) != State::Ready) {
        return;
    }
    std::scoped_lock model_lock{g_model_mutex};
    RetireLane(g.game);
    RetireLane(g.video);
    Retire(g.video_depth);
    Retire(g.video_motion);
    FreeRetired(true);
    g.compose.reset();
    if (g.query_pool != VK_NULL_HANDLE) {
        VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyQueryPool(static_cast<VkDevice>(g.device),
                                                         g.query_pool, nullptr);
        g.query_pool = VK_NULL_HANDLE;
    }
    SetStatus(State::Failed, "Shut down");
    LOG_INFO(Render_Vulkan, "DLSS 5: shut down after {} gameplay / {} video frames", g.game_frames,
             g.video_frames);
}

Status GetStatus() {
    std::scoped_lock lock{g_mutex};
    return g_status;
}

Settings GetSettings() {
    std::scoped_lock lock{g_mutex};
    return g_settings;
}

void SetSettings(const Settings& settings, bool rebuild, bool save) {
    bool tuning_changed = rebuild;
    {
        std::scoped_lock lock{g_mutex};
        const Settings& o = g_settings;
        // What the model reads when a feature is built: a change rebuilds it.
        tuning_changed |= o.style != settings.style || o.intensity != settings.intensity ||
                          o.local_structure != settings.local_structure ||
                          o.local_tone != settings.local_tone ||
                          o.skin_follows != settings.skin_follows ||
                          o.skin_structure != settings.skin_structure ||
                          o.auto_mask != settings.auto_mask || o.preset != settings.preset ||
                          o.ui_correction != settings.ui_correction ||
                          o.mv_mode != settings.mv_mode;
        g_settings = settings;
    }
    if (tuning_changed) {
        g_generation.fetch_add(1, std::memory_order_acq_rel);
    }
    if (save) {
        SaveSettingsFile(settings);
    }
}

bool ToggleEnabled() {
    Settings s;
    {
        std::scoped_lock lock{g_mutex};
        g_settings.enabled = !g_settings.enabled;
        s = g_settings;
    }
    SaveSettingsFile(s); // on / off only: the features stay as built
    return s.enabled;
}

std::optional<Dlss::EvalImage> Apply(const Instance& instance, vk::CommandBuffer cmdbuf,
                                     const Dlss::EvalImage& color, const Dlss::EvalImage& motion,
                                     const Dlss::EvalImage& depth, bool reset) {
    if (g_state.load(std::memory_order_acquire) != State::Ready) {
        return std::nullopt;
    }
    std::scoped_lock model_lock{g_model_mutex};
    Lane& lane = g.game;
    ++g.applies;
    FreeRetired(false);
    ReadTiming();
    // The video lane's model goes once no video frame came for 600 runs (~5 s of gameplay): it
    // holds VRAM. Both lanes' models may live at once - round 3's "one lane at a time" rested on
    // a wrong crash theory (the crash was our compute passes), and at cutscene exits the lanes
    // then rebuilt each other ~40 times in 10 s (2026-10-18 runs 13:38 / 14:07).
    if (g.video.features[0] && g.applies - g.video.last_apply > 600) {
        for (int k = 0; k < MaxPasses; ++k) {
            RetireFeature(g.video, k);
        }
    }
    if ((g.applies & 127) == 0) {
        UpdateVram();
    }

    const u32 gen = g_generation.load(std::memory_order_acquire);
    const Settings s = GetSettings();
    if (!s.enabled || gen == lane.failed_gen) {
        lane.was_off = true;
        SetRunning(false, 0);
        return std::nullopt;
    }
    const u32 w = color.width;
    const u32 h = color.height;
    const int wanted = Test().path == 1 ? 1 : std::clamp(s.passes, 1, MaxPasses);
    EnsureImages(instance, lane, w, h, wanted, "gameplay");
    EnsureCompose();
    for (int k = wanted; k < MaxPasses; ++k) {
        RetireFeature(lane, k); // fewer passes than before
    }
    // A later pass that failed stays off until the settings or the pass count change.
    if (lane.extra_failed_gen != gen || lane.extra_failed_passes != s.passes) {
        lane.extra_failed_gen = ~0u;
        lane.pass_cap = MaxPasses;
    }
    // Pass 0 runs the frame it is built; pass k is built only once pass k-1 existed before this
    // frame, and used from the frame after it is built (the fork's rule against multi-pass hangs).
    int passes = 0;
    for (int k = 0; k < std::min(wanted, lane.pass_cap); ++k) {
        if (k > 0 && lane.built_at[k - 1] == g.applies) {
            break;
        }
        if (!EnsureFeature(cmdbuf, lane, k, s, gen, w, h, "gameplay")) {
            break;
        }
        if (k > 0 && lane.built_at[k] == g.applies) {
            break;
        }
        passes = k + 1;
    }
    if (passes == 0) {
        SetRunning(false, 0);
        return std::nullopt;
    }
    // A new history after a gap (a cutscene, DLSS 5 or DLSS off, a loading screen) or a new pass 1.
    const bool gap = lane.last_apply + 1 != g.applies;
    lane.last_apply = g.applies;
    if (lane.built_at[0] == g.applies || lane.was_off || gap) {
        reset = true;
        lane.was_off = false;
    }

    if (Test().reset) {
        reset = true;
    }
    if (Test().path == 1) {
        return ApplyRound1(cmdbuf, lane, s, color, motion, depth, reset, gen, w, h);
    }
    const bool apart = Test().path == 2;

    const u32 slot = static_cast<u32>(g.game_frames % TimingSlots);
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.resetQueryPool(vk::QueryPool{g.query_pool}, slot * 2, 2);
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2);
    }
    if (apart) {
        ToGeneral(cmdbuf, {lane.encoded.get(), lane.answer.get(), lane.composed.get()});
        KeepGeneral(cmdbuf, {lane.input.get(), lane.out[0].get(), lane.out[1].get(),
                             lane.out[2].get()});
    } else {
        ToGeneral(cmdbuf, {lane.input.get(), lane.out[0].get(), lane.out[1].get(),
                           lane.out[2].get(), lane.composed.get()});
    }
    // The game's frame -> the model's input in true colour (the game keeps red / blue swapped).
    const OwnedImage& enc = apart ? *lane.encoded : *lane.input;
    Dispatch(cmdbuf, MakeParams(0, s, false), color.view, color.layout, enc.view.get(),
             enc.view.get(), enc.view.get(), w, h);
    FullBarrier(cmdbuf);
    if (apart) {
        CopyWhole(cmdbuf, lane.encoded->image, lane.input->image, w, h);
        FullBarrier(cmdbuf);
    }

    // The game's vectors are screen UV, current minus previous (vk_dlss.cpp): the model wants
    // the offset to the previous position in pixels -> -size (DLSS SR's NGX scale, study 10-14).
    const float mv_x = s.mv_mode == 0 ? -static_cast<float>(w) : s.mv_mode == 1 ? -1.0f : 0.0f;
    const float mv_y = s.mv_mode == 0 ? -static_cast<float>(h) : s.mv_mode == 1 ? -1.0f : 0.0f;
    for (int k = 0; k < passes; ++k) {
        const OwnedImage& in = k == 0 ? *lane.input : *lane.out[k - 1];
        int result = 0;
        const bool pass_reset = reset || lane.fresh[k];
        lane.fresh[k] = false;
        if (!Evaluate(cmdbuf, lane.features[k], s, AsEval(in, w, h), depth, motion,
                      AsEval(*lane.out[k], w, h), mv_x, mv_y, pass_reset, &result)) {
            if (k > 0) {
                // A later pass: drop it (and the ones after) and compose what ran (review 2).
                LOG_ERROR(Render_Vulkan, "DLSS 5: gameplay pass {} evaluate failed ({:#x}) - "
                                         "running {} pass(es)",
                          k + 1, static_cast<u32>(result), k);
                for (int j = k; j < MaxPasses; ++j) {
                    RetireFeature(lane, j);
                }
                lane.extra_failed_gen = gen;
                lane.extra_failed_passes = s.passes;
                lane.pass_cap = k;
                passes = k;
                break;
            }
            EvalFailed(lane, gen, result, "gameplay");
            SetRunning(false, 0);
            return std::nullopt;
        }
        FullBarrier(cmdbuf);
    }
    lane.eval_failures = 0;
    const OwnedImage* answer = lane.out[passes - 1].get();
    if (apart) {
        CopyWhole(cmdbuf, answer->image, lane.answer->image, w, h);
        FullBarrier(cmdbuf);
        answer = lane.answer.get();
    }
    // Composed once against the game's frame (detail builds up over passes, colour does not
    // compound), back in the game's channel order for DLSS 4.5.
    Dispatch(cmdbuf, MakeParams(2, s, false), enc.view.get(), vk::ImageLayout::eGeneral,
             enc.view.get(), answer->view.get(), lane.composed->view.get(), w, h);
    FullBarrier(cmdbuf);
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2 + 1);
        g.slot_written[slot] = true;
    }
    ++g.game_frames;
    if (g.game_frames == 1 || (g.game_frames % 1800) == 0) {
        const auto [vram_used, vram_budget] = UpdateVram();
        LOG_INFO(Render_Vulkan,
                 "DLSS 5: running ({} gameplay frames, {} passes) | model copies alive {}, VRAM "
                 "{} / {} MB",
                 g.game_frames, passes, g.features_alive, vram_used, vram_budget);
    }
    SetRunning(true, passes);
    return AsEval(*lane.composed, w, h);
}

bool VideoFrame(u64 resolves_at_last_flip) {
    // Diagnostics for the next round: video frames, and game frames without the T2x resolve
    // (an in-engine scene that bypasses the hook would show as the second kind).
    // A kind counts once it held for 3 flips in a row: a game that shows one frame twice (a 30 FPS
    // scene at 60 flips) must not flip-flop between kinds, nor DLSS 5 between the two paths.
    static int last_kind = 0; // 0 game-rendered, 1 video, 2 no resolve and no video
    static int candidate = 0;
    static int streak = 0;
    static u64 run = 0;
    // DLSS 4.5 up and on: otherwise no frame has a T2x replacement and every flip during a
    // decode (an in-world TV) would look like a video frame.
    if (g_state.load(std::memory_order_acquire) != State::Ready ||
        !Dlss::g_ready.load(std::memory_order_relaxed) ||
        !Dlss::g_user_on.load(std::memory_order_relaxed)) {
        return false;
    }
    const bool resolved = g_resolves.load(std::memory_order_acquire) != resolves_at_last_flip;
    const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    const bool decoding =
        now_ms - Libraries::Videodec::g_last_decode_ms.load(std::memory_order_relaxed) < 250;
    // A video frame only after 500 ms without the T2x pass, and never within 500 ms of one: in the
    // city an in-world TV decodes all the time, and a stall (a few flips without a resolve) made
    // game frames count as video - both lanes then took turns and rebuilt the model every frame
    // (NOMODEL tests 2026-10-18: 14 flips at full speed, 12 under the validation layer with a
    // 250 ms rule). A cutscene video lasts many seconds; DLSS 5 joins it half a second in.
    static s64 last_resolve_ms = 0;
    if (resolved) {
        last_resolve_ms = now_ms;
    }
    const bool quiet = now_ms - last_resolve_ms >= 500;
    const int seen = resolved ? 0 : (decoding && quiet) ? 1 : 2;
    streak = seen == candidate ? streak + 1 : 1;
    candidate = seen;
    const int kind = streak >= 3 ? candidate : last_kind;
    if (kind != last_kind) {
        static const char* const names[] = {"game-rendered", "video cutscene",
                                            "no T2x resolve, no video"};
        LOG_INFO(Render_Vulkan, "DLSS 5: frames now {} (after {} {} flips)", names[kind], run,
                 names[last_kind]);
        last_kind = kind;
        run = 0;
    }
    ++run;
    if (kind != 1 || !quiet) {
        return false;
    }
    // SHADPS4_DLSS5_VIDEO=0: never the video path, whatever the panel says (a bisect launcher
    // that leaves the saved settings alone).
    static const bool video_env_off = [] {
        const char* v = std::getenv("SHADPS4_DLSS5_VIDEO");
        return v && v[0] == '0';
    }();
    const Settings s = GetSettings();
    return s.enabled && s.video && !video_env_off;
}

std::optional<vk::ImageView> ApplyVideo(const Instance& instance, vk::CommandBuffer cmdbuf,
                                        vk::ImageView frame_view, u32 width, u32 height,
                                        bool linear) {
    if (g_state.load(std::memory_order_acquire) != State::Ready || Test().path == 1) {
        return std::nullopt; // PATH=1: round 1 had no video lane
    }
    std::scoped_lock model_lock{g_model_mutex};
    Lane& lane = g.video;
    ++g.applies;
    FreeRetired(false);
    if ((g.applies & 127) == 0) {
        UpdateVram();
    }
    const u32 gen = g_generation.load(std::memory_order_acquire);
    const Settings s = GetSettings();
    if (gen == lane.failed_gen) {
        SetVideoRunning(false);
        return std::nullopt;
    }
    const u32 w = width;
    const u32 h = height;
    const bool resized = !lane.input || lane.width != w || lane.height != h;
    EnsureImages(instance, lane, w, h, 1, "video");
    EnsureCompose();
    // No depth or motion in a video: a flat depth and zero motion, made at the frame size.
    if (resized || !g.video_depth || g.video_depth_value != s.video_depth) {
        Retire(g.video_depth);
        Retire(g.video_motion);
        g.video_depth = MakeImage(instance, w, h, vk::Format::eR32Sfloat, "DLSS 5 video depth");
        g.video_motion = MakeImage(instance, w, h, vk::Format::eR16G16Sfloat, "DLSS 5 video motion");
        ToGeneral(cmdbuf, {g.video_depth.get(), g.video_motion.get()});
        const vk::ClearColorValue depth_value{std::array{s.video_depth, 0.0f, 0.0f, 0.0f}};
        const vk::ClearColorValue zero{std::array{0.0f, 0.0f, 0.0f, 0.0f}};
        cmdbuf.clearColorImage(g.video_depth->image, vk::ImageLayout::eGeneral, depth_value,
                               ColorRange);
        cmdbuf.clearColorImage(g.video_motion->image, vk::ImageLayout::eGeneral, zero, ColorRange);
        g.video_depth_value = s.video_depth;
    }
    if (!EnsureFeature(cmdbuf, lane, 0, s, gen, w, h, "video")) {
        SetVideoRunning(false);
        return std::nullopt;
    }
    const bool apart = Test().path == 2;
    if (apart) {
        ToGeneral(cmdbuf, {lane.encoded.get(), lane.answer.get(), lane.composed.get()});
        KeepGeneral(cmdbuf, {lane.input.get(), lane.out[0].get()});
    } else {
        ToGeneral(cmdbuf, {lane.input.get(), lane.out[0].get(), lane.composed.get()});
    }
    const OwnedImage& enc = apart ? *lane.encoded : *lane.input;
    Dispatch(cmdbuf, MakeParams(1, s, linear), frame_view, vk::ImageLayout::eShaderReadOnlyOptimal,
             enc.view.get(), enc.view.get(), enc.view.get(), w, h);
    FullBarrier(cmdbuf);
    if (apart) {
        CopyWhole(cmdbuf, lane.encoded->image, lane.input->image, w, h);
        FullBarrier(cmdbuf);
    }
    // History kept across a video (round 2 reset every frame: the model redrew its detail from
    // scratch each frame and the intro flickered); a new one when a video starts (a gap in the
    // lane's runs), on a new feature, or every frame with `video_fresh`. Zero motion: a moving
    // shot may ghost a little.
    const bool gap = lane.last_apply + 1 != g.applies;
    lane.last_apply = g.applies;
    const bool reset = s.video_fresh || gap || lane.fresh[0] || Test().reset;
    lane.fresh[0] = false;
    int result = 0;
    if (!Evaluate(cmdbuf, lane.features[0], s, AsEval(*lane.input, w, h),
                  AsEval(*g.video_depth, w, h), AsEval(*g.video_motion, w, h),
                  AsEval(*lane.out[0], w, h), 0.0f, 0.0f, reset, &result)) {
        EvalFailed(lane, gen, result, "video");
        SetVideoRunning(false);
        return std::nullopt;
    }
    lane.eval_failures = 0;
    FullBarrier(cmdbuf);
    const OwnedImage* answer = lane.out[0].get();
    if (apart) {
        CopyWhole(cmdbuf, answer->image, lane.answer->image, w, h);
        FullBarrier(cmdbuf);
        answer = lane.answer.get();
    }
    Dispatch(cmdbuf, MakeParams(3, s, false), enc.view.get(), vk::ImageLayout::eGeneral,
             enc.view.get(), answer->view.get(), lane.composed->view.get(), w, h);
    // The post pass samples it. Written by our draw (or the compute dispatch with
    // SHADPS4_DLSS5_COMPUTE=1): both writers are in the source scope.
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = lane.composed->image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_read,
    });
    ++g.video_frames;
    if (g.video_frames == 1 || (g.video_frames % 1800) == 0) {
        LOG_INFO(Render_Vulkan, "DLSS 5: video frames {} ({}x{})", g.video_frames, w, h);
    }
    SetVideoRunning(true);
    return lane.composed->view.get();
}

} // namespace Vulkan::Dlss5
