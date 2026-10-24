// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3: what the fork's video core takes from its emulator, here from this port's runtime.
// Guest memory is the runtime's (src/runtime_memory.c): one pool (a page-file section on Windows,
// a memfd on Linux) with a second view that carries no page protection, the guest's views of it
// in the PS4 ranges, the GPU's page protection through runtime_memory_gpu_protect. Faults reach
// the fork's page manager through SignalDispatch (gpu_host.cpp tlggpu_handle_fault). The rest are
// the few kernel, debug, settings and UI entry points the video core calls; game patches of the
// fork's (MemoryPatcher: Second Son's live resize, quality ladder) are neutral here.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif
#include <boost/asio/io_context.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/rdtsc.h"
#include "common/release_mode.h"
#include "common/scm_rev.h"
#include "common/switch_mutex.h"
#include "core/address_space.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/loader/symbols_resolver.h"
#include "core/memory.h"
#include "core/signals.h"
#include "imgui/imgui_texture.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/texture_manager.h"
#include "sdl_window.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "v3_glue.h"

extern "C" {
// runtime_memory.c
unsigned char* runtime_memory_backing_base(uint64_t* size);
int runtime_memory_backing_span(uintptr_t address, uintptr_t* start, uintptr_t* end, uint64_t* phys);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size);
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
// runtime_kernel.c
uint64_t runtime_process_time_us(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
}

// The window (glue/window.cpp makes it, the presenter draws into it).
Frontend::WindowSDL* g_window = nullptr;

namespace V3 {
HostConfig g_host{};
} // namespace V3

// ---- memory ------------------------------------------------------------------------------------
namespace Core {

// The PS4 ranges the runtime reserves (runtime_memory.c SYSTEM_RESERVED_*, USER_*).
constexpr VAddr SystemReservedMin = 0x700000000ULL, SystemReservedMax = 0x800000000ULL;
constexpr VAddr UserMin = 0x1000000000ULL, UserMax = 0xfc00000000ULL;

bool GlueIsValidMapping(VAddr virtual_addr, u64 size) {
    if (size == 0) {
        return runtime_memory_clamp(virtual_addr, 1) == 1;
    }
    return runtime_memory_clamp(virtual_addr, size) == size;
}

struct AddressSpace::Impl {};

AddressSpace::AddressSpace() {
    u64 size = 0;
    backing_base = runtime_memory_backing_base(&size);
    system_reserved_base = reinterpret_cast<u8*>(SystemReservedMin);
    system_reserved_size = SystemReservedMax - SystemReservedMin;
    user_base = reinterpret_cast<u8*>(UserMin);
    user_size = UserMax - UserMin;
}
AddressSpace::~AddressSpace() = default;

u64 AddressSpace::BackingSizeBytes() const noexcept {
    u64 size = 0;
    runtime_memory_backing_base(&size);
    return size;
}

void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    const auto bits = static_cast<u32>(perms);
    runtime_memory_gpu_protect(virtual_addr, size, (bits & static_cast<u32>(MemoryPermission::Read)) != 0,
                               (bits & static_cast<u32>(MemoryPermission::Write)) != 0);
}

void ReleaseDriverVaHole() {} // DLSS's address space hole: no DLSS here

MemoryManager::MemoryManager() {}
MemoryManager::~MemoryManager() = default;

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    return runtime_memory_clamp(virtual_addr, size);
}
u64 MemoryManager::TryClampRangeSize(VAddr virtual_addr, u64 size) {
    return runtime_memory_clamp(virtual_addr, size);
}

void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // Backed spans are copied from the pool's unprotected view; holes read as zero.
    u8* base = impl.BackingBase();
    while (size) {
        uintptr_t start = 0, end = 0;
        u64 phys = 0;
        if (base && runtime_memory_backing_span(source, &start, &end, &phys)) {
            const u64 n = std::min<u64>(size, end - source);
            std::memcpy(dest, base + phys + (source - start), n);
            source += n;
            dest += n;
            size -= n;
            continue;
        }
        const u64 n = std::min<u64>(size, 0x4000 - (source & 0x3fff));
        std::memset(dest, 0, n);
        source += n;
        dest += n;
        size -= n;
    }
}

bool MemoryManager::TryReadBacking(VAddr virtual_addr, u8* dest, u64 size) {
    return runtime_memory_read_backing(virtual_addr, dest, size) != 0;
}
bool MemoryManager::TryReadGuest(VAddr virtual_addr, u8* dest, u64 size) {
    return runtime_memory_read_backing(virtual_addr, dest, size) != 0;
}
bool MemoryManager::FullyBacked(VAddr virtual_addr, u64 size) {
    uintptr_t start = 0, end = 0;
    u64 phys = 0;
    while (size) {
        if (!runtime_memory_backing_span(virtual_addr, &start, &end, &phys)) {
            return false;
        }
        const u64 n = std::min<u64>(size, end - virtual_addr);
        virtual_addr += n;
        size -= n;
    }
    return true;
}
bool MemoryManager::BackingSpan(VAddr virtual_addr, VAddr& span_start, VAddr& span_end, u64& backing_out) {
    uintptr_t start = 0, end = 0;
    u64 phys = 0;
    if (!runtime_memory_backing_span(virtual_addr, &start, &end, &phys)) {
        return false;
    }
    span_start = start;
    span_end = end;
    backing_out = phys;
    return true;
}

bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    return runtime_memory_write_backing(reinterpret_cast<uintptr_t>(address), data, size) != 0;
}

void MemoryManager::ExcludeFromDownloads(VAddr addr) {
    const u32 n = num_download_exclusions.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (download_exclusions[i].load(std::memory_order_relaxed) == addr) {
            return;
        }
    }
    if (n < MaxDownloadExclusions) {
        download_exclusions[n].store(addr, std::memory_order_relaxed);
        num_download_exclusions.store(n + 1, std::memory_order_release);
    }
}

bool MemoryManager::TryWriteBackingDownload(void* address, const void* data, u64 size) {
    // The excluded 4-byte words (a command processor read pointer the front owns) keep their
    // value: the pieces between them are written.
    const auto begin = reinterpret_cast<VAddr>(address);
    const auto* bytes = static_cast<const u8*>(data);
    VAddr at = begin;
    const VAddr end = begin + size;
    const u32 n = num_download_exclusions.load(std::memory_order_acquire);
    while (at < end) {
        VAddr next_skip = end;
        for (u32 i = 0; i < n; ++i) {
            const VAddr word = download_exclusions[i].load(std::memory_order_relaxed);
            if (word + 4 > at && word < next_skip) {
                next_skip = std::max(word, at);
            }
        }
        if (next_skip > at &&
            !runtime_memory_write_backing(at, bytes + (at - begin), next_skip - at)) {
            return false;
        }
        if (next_skip >= end) {
            break;
        }
        download_exclusion_hits.fetch_add(1, std::memory_order_relaxed);
        at = std::min<VAddr>(end, next_skip + 4);
    }
    return true;
}

void MemoryManager::InvalidateMemory(VAddr addr, u64 size) const {
    if (rasterizer) {
        rasterizer->InvalidateMemory(addr, size);
    }
}

// Faults: the runtime's exception handler (gpu_host.cpp tlggpu_handle_fault) asks here.
SignalDispatch::SignalDispatch() = default;
SignalDispatch::~SignalDispatch() = default;
bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}
bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core

// ---- the event queues' file handles --------------------------------------------------------------
namespace Core::FileSys {
int HandleTable::CreateHandle() {
    std::scoped_lock lock{m_mutex};
    for (size_t i = 0; i < m_files.size(); ++i) {
        if (!m_files[i]) {
            m_files[i] = new File{};
            return static_cast<int>(i);
        }
    }
    m_files.push_back(new File{});
    return static_cast<int>(m_files.size() - 1);
}
void HandleTable::DeleteHandle(int d) {
    std::scoped_lock lock{m_mutex};
    if (d >= 0 && static_cast<size_t>(d) < m_files.size()) {
        delete m_files[d];
        m_files[d] = nullptr;
    }
}
File* HandleTable::GetFile(int d) {
    std::scoped_lock lock{m_mutex};
    return d >= 0 && static_cast<size_t>(d) < m_files.size() ? m_files[d] : nullptr;
}
} // namespace Core::FileSys

// ---- the libraries' symbols (GnmDriver, VideoOut, the event queues) ------------------------------
namespace Core::Loader {
// nid_name holds the NID (LIB_FUNCTION's first argument): tlggpu_resolve looks it up.
void SymbolsResolver::AddSymbol(const SymbolResolver& s, u64 virtual_addr) {
    m_symbols.push_back(SymbolRecord{GenerateName(s), s.name, virtual_addr});
}
std::string SymbolsResolver::GenerateName(const SymbolResolver& s) {
    return fmt::format("{}#{}#{}#{}#{}", s.name, s.library, s.library_version, s.module,
                       SymbolTypeToS(s.type));
}
} // namespace Core::Loader

// ---- kernel ------------------------------------------------------------------------------------
namespace Libraries::Kernel {
boost::asio::io_context io_context; // the event queues' timers (gpu_host.cpp runs it)
thread_local Pthread* g_curthread = nullptr; // the runtime's threads are not the fork's
void KernelSignalRequest() {} // a work guard keeps io_context running
s32* PS4_SYSV_ABI __Error() {
    return V3::g_host.errno_location ? V3::g_host.errno_location() : nullptr;
}
u64 PS4_SYSV_ABI sceKernelGetTscFrequency() {
    return V3::g_host.tsc_frequency();
}
u64 PS4_SYSV_ABI sceKernelReadTsc() {
    return Common::FencedRDTSC();
}
u64 PS4_SYSV_ABI sceKernelGetProcessTime() {
    return V3::g_host.process_time_us();
}
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) {
        return static_cast<s32>(0x80020016);
    }
    *ver = static_cast<s32>(V3::g_host.sdk_version);
    return 0;
}
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel

// ---- common ------------------------------------------------------------------------------------
namespace Common {
extern const char g_scm_desc[] = "renderer v3 (axylpup second-son-v1.1.1, 528aea02)";

// The fork's guest mutex helpers (core/libraries/kernel/sync/mutex.cpp).
void FutexWait(std::atomic<u32>* address, u32 value) {
#ifdef _WIN32
    WaitOnAddress(address, &value, sizeof(value), INFINITE);
#else
    while (address->load(std::memory_order_acquire) == value) {
        address->wait(value);
    }
#endif
}
void FutexWakeOne(std::atomic<u32>* address) {
#ifdef _WIN32
    WakeByAddressSingle(address);
#else
    address->notify_one();
#endif
}
bool FlushProcessWriteBuffersAll() {
#ifdef _WIN32
    FlushProcessWriteBuffers();
    return true;
#else
    return false;
#endif
}
void ReportMutexReentry(const char* name, void*, void*) {
    static std::atomic<u64> events{0};
    const u64 n = events.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 20 || (n & (n - 1)) == 0) {
        LOG_ERROR(Kernel, "SwitchMutex re-entry #{} ({})", n, name);
    }
}
#ifdef _WIN32
void SrwLockExclusive(void** lock) {
    AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}
bool SrwTryLockExclusive(void** lock) {
    return TryAcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock)) != 0;
}
void SrwUnlockExclusive(void** lock) {
    ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}
#else
void SrwLockExclusive(void**) {}
bool SrwTryLockExclusive(void**) {
    return false;
}
void SrwUnlockExclusive(void**) {}
#endif

namespace ReleaseMode {
void ShowOutOfMemoryNotice(unsigned long long bytes) {
    std::fprintf(stderr, "V3: out of GPU memory (%llu MiB asked)\n", bytes >> 20);
}
} // namespace ReleaseMode
} // namespace Common

// ---- the fork's Second Son patches: neutral ------------------------------------------------------
namespace MemoryPatcher {
EXPORT uintptr_t g_eboot_address = 0;
std::string g_game_serial;
bool LiveResizeAvailable() {
    return false;
}
void OnSubmitDone() {}
bool TakeLivePurge(u64*) {
    return false;
}
void RequeueLivePurge() {}
bool InLiveResizeWindow() {
    return false;
}
bool IsRenderShaped(u32, u32) {
    return false;
}
s64 MsSinceLiveResize() {
    return INT64_MAX;
}
bool MovieFitRect(float&, float&) {
    return false;
}
int GameQualityLevel() {
    return -1;
}
} // namespace MemoryPatcher

// ---- debug state (the fork's debugger: off) ------------------------------------------------------
static DebugStateType::DebugStateImpl g_debug_state;
DebugStateType::DebugStateImpl& DebugState = g_debug_state;
namespace DebugStateType {
void DebugStateImpl::PauseGuestThreads() {}
void DebugStateImpl::PushQueueDump(QueueDump) {}
void DebugStateImpl::PushRegsDump(uintptr_t, uintptr_t, const AmdGpu::Regs&) {}
void DebugStateImpl::PushRegsDumpCompute(uintptr_t, uintptr_t, const CsState&) {}
void DebugStateImpl::CollectShader(const std::string&, Shader::LogicalStage, vk::ShaderModule,
                                   std::span<const u32>, std::span<const u32>, std::span<const u32>,
                                   bool) {}
} // namespace DebugStateType

// ---- settings ------------------------------------------------------------------------------------
std::shared_ptr<EmulatorSettingsImpl> EmulatorSettingsImpl::s_instance;
std::mutex EmulatorSettingsImpl::s_mutex;
EmulatorSettingsImpl::EmulatorSettingsImpl() = default;
EmulatorSettingsImpl::~EmulatorSettingsImpl() = default;
std::shared_ptr<EmulatorSettingsImpl> EmulatorSettingsImpl::GetInstance() {
    std::scoped_lock lock{s_mutex};
    if (!s_instance) {
        s_instance = std::make_shared<EmulatorSettingsImpl>();
    }
    return s_instance;
}

// ---- the fork's ImGui interface: this port's overlay draws instead -------------------------------
namespace ImGui {
RefCountedTexture::~RefCountedTexture() = default;
namespace Core {
void Initialize(const Vulkan::Instance&, const Frontend::WindowSDL&, u32, vk::Format,
                const vk::AllocationCallbacks*) {}
void OnSurfaceFormatChange(vk::Format) {}
void Shutdown(const vk::Device&) {}
bool MustKeepDrawing() {
    return false;
}
namespace TextureManager {
void Submit() {}
bool HasPendingUploads() {
    return false;
}
} // namespace TextureManager
} // namespace Core
} // namespace ImGui

// ---- RenderDoc and screenshots (F12: this port's overlay has its own) ----------------------------
namespace VideoCore {
static std::atomic<u32> screenshot_requests{0}, overlay_screenshot_requests{0};
void LoadRenderDoc() {}
void StartCapture() {}
void EndCapture() {}
void TriggerCapture() {}
void OnFrameBoundary() {}
void SetOutputDir(const std::filesystem::path&, const std::string&) {}
bool IsRenderDocLoaded() {
    return false;
}
void RequestScreenshot(ScreenshotRequest request) {
    if (request == ScreenshotRequest::GameOnly) {
        ++screenshot_requests;
    } else {
        ++overlay_screenshot_requests;
    }
}
u32 ConsumeGameOnlyScreenshotRequests() {
    return screenshot_requests.exchange(0);
}
u32 ConsumeWithOverlaysScreenshotRequests() {
    return overlay_screenshot_requests.exchange(0);
}
ScreenshotRequests ConsumeScreenshotRequests() {
    return {ConsumeGameOnlyScreenshotRequests(), ConsumeWithOverlaysScreenshotRequests()};
}
} // namespace VideoCore
