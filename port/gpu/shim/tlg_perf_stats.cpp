// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_perf_stats.h"
#include "tlg_profile.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "video_core/renderer_vulkan/vk_instance.h"

#ifndef _WIN32
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

namespace TlgPerfStats {

namespace {

std::mutex mutex;
Sample latest;
std::once_flag started;

// The fault and pipeline rates since the last sample, `seconds` ago.
void CountRates(Sample& s, double seconds) {
    static uint64_t last_faults = 0, last_pipelines = 0;
    static bool counted = false;
    const uint64_t faults = fault_count.load(std::memory_order_relaxed);
    const uint64_t pipelines = pipeline_count.load(std::memory_order_relaxed);
    if (counted && seconds > 0.0) {
        s.faults_per_second = float(double(faults - last_faults) / seconds);
        s.pipelines_per_second = float(double(pipelines - last_pipelines) / seconds);
    }
    s.pipelines = pipelines;
    last_faults = faults;
    last_pipelines = pipelines;
    counted = true;
}

// The busiest of `rows` (at least 1% of a core), at most four, without the "shadPS4:" prefix.
std::vector<ThreadLoad> Busiest(std::vector<ThreadLoad> rows) {
    std::sort(rows.begin(), rows.end(), [](const ThreadLoad& a, const ThreadLoad& b) { return a.percent > b.percent; });
    std::vector<ThreadLoad> out;
    for (ThreadLoad& row : rows) {
        if (out.size() == 4 || row.percent < 1.0f) break;
        if (row.name.rfind("shadPS4:", 0) == 0) row.name.erase(0, 8);
        out.push_back(std::move(row));
    }
    return out;
}

#ifdef _WIN32
uint64_t Ticks(const FILETIME& time) {
    return (uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

// This process's use of the GPU's 3D engines, as Task Manager shows it ("GPU Engine" counters,
// any vendor). Negative until the second sample or without the counters.
class GpuUsage {
public:
    GpuUsage() {
        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) {
            query = nullptr;
            return;
        }
        if (PdhAddEnglishCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter) !=
            ERROR_SUCCESS) {
            PdhCloseQuery(query);
            query = nullptr;
            return;
        }
        PdhCollectQueryData(query);
        pid = L"pid_" + std::to_wstring(GetCurrentProcessId()) + L"_";
    }

    float Percent() {
        if (!query || PdhCollectQueryData(query) != ERROR_SUCCESS) {
            return -1.0f;
        }
        DWORD size = 0, count = 0;
        if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count,
                                         nullptr) != PDH_MORE_DATA) {
            return -1.0f;
        }
        buffer.resize(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, &size, &count,
                                         items) != ERROR_SUCCESS) {
            return -1.0f;
        }
        double total = 0.0;
        for (DWORD i = 0; i < count; ++i) {
            const wchar_t* name = items[i].szName;
            if (items[i].FmtValue.CStatus == PDH_CSTATUS_VALID_DATA && wcsstr(name, pid.c_str()) &&
                wcsstr(name, L"engtype_3D")) {
                total += items[i].FmtValue.doubleValue;
            }
        }
        return std::clamp(float(total), 0.0f, 100.0f);
    }

private:
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
    std::wstring pid;
    std::vector<unsigned char> buffer;
};

// Each thread's kernel and user time since the last call (a system thread snapshot), named by its
// description (SetThreadDescription, pthread_setname_np) or its id.
class ThreadTimes {
public:
    std::vector<ThreadLoad> Loads(uint64_t wall_delta) {
        std::vector<ThreadLoad> rows;
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return rows;
        std::unordered_map<DWORD, uint64_t> now;
        THREADENTRY32 entry{.dwSize = sizeof(entry)};
        const DWORD process = GetCurrentProcessId();
        for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != process) continue;
            HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
            if (!thread) continue;
            FILETIME created, ended, kernel, user;
            if (GetThreadTimes(thread, &created, &ended, &kernel, &user)) {
                const uint64_t total = Ticks(kernel) + Ticks(user);
                now[entry.th32ThreadID] = total;
                const auto before = last.find(entry.th32ThreadID);
                if (before != last.end() && wall_delta && total >= before->second) { // an id reused: no sample
                    rows.push_back({Name(thread, entry.th32ThreadID),
                                    float(double(total - before->second) * 100.0 / double(wall_delta))});
                }
            }
            CloseHandle(thread);
        }
        CloseHandle(snapshot);
        last = std::move(now);
        return rows;
    }

private:
    static std::string Name(HANDLE thread, DWORD id) {
        using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
        static const auto get_description = reinterpret_cast<GetDescription>(
            reinterpret_cast<void (*)()>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
        PWSTR wide = nullptr;
        std::string name;
        if (get_description && SUCCEEDED(get_description(thread, &wide)) && wide) {
            const int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
            if (n > 1) {
                name.resize(size_t(n - 1));
                WideCharToMultiByte(CP_UTF8, 0, wide, -1, name.data(), n, nullptr, nullptr);
            }
            LocalFree(wide);
        }
        return name.empty() ? "tid " + std::to_string(id) : name;
    }

    std::unordered_map<DWORD, uint64_t> last;
};

void Run(const Vulkan::Instance* instance) {
    GpuUsage gpu;
    ThreadTimes threads;
    FILETIME created, ended, kernel, user, now;
    uint64_t last_process = 0, last_wall = 0;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    while (true) {
        Sample s;
        s.gpu_name = std::string(instance->GetModelName());
        GetSystemTimeAsFileTime(&now);
        if (GetProcessTimes(GetCurrentProcess(), &created, &ended, &kernel, &user)) {
            const uint64_t process = Ticks(kernel) + Ticks(user), wall = Ticks(now);
            if (last_wall) {
                s.cpu_percent = CpuPercent(process - last_process, wall - last_wall, cores);
                CountRates(s, double(wall - last_wall) / 1e7);
                if (want_threads.load(std::memory_order_relaxed)) s.threads = Busiest(threads.Loads(wall - last_wall));
            }
            last_process = process;
            last_wall = wall;
        }
        PROCESS_MEMORY_COUNTERS memory{};
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) {
            s.ram_used_gb = double(memory.WorkingSetSize) / GiB;
        }
        MEMORYSTATUSEX system{.dwLength = sizeof(MEMORYSTATUSEX)};
        if (GlobalMemoryStatusEx(&system)) {
            s.ram_total_gb = double(system.ullTotalPhys) / GiB;
        }
        if (instance->CanReportMemoryUsage()) {
            s.vram_used_gb = double(instance->GetDeviceMemoryUsage()) / GiB;
            #ifdef TLG_RENDERER_HOSTED
            s.vram_total_gb = double(instance->GetTotalMemoryBudget()) / GiB;
#else
            s.vram_total_gb = double(instance->GetDeviceMemoryBudgetNow()) / GiB;
#endif
        }
        s.gpu_percent = gpu.Percent();
        {
            std::scoped_lock lock{mutex};
            latest = std::move(s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
#else
// tlg: Linux measurements. CPU and RAM from /proc, VRAM from VK_EXT_memory_budget, GPU load from
// NVML (NVIDIA, loaded at run time) or the amdgpu sysfs counter; anything else stays "n/a".
class GpuUsage {
public:
    GpuUsage() {
        for (int card = 0; card < 8 && !sysfs; ++card) {
            char path[96];
            std::snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/gpu_busy_percent", card);
            if (std::FILE* f = std::fopen(path, "r")) { std::fclose(f); sysfs_path = path; sysfs = true; }
        }
        if (sysfs) return;
        nvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!nvml) return;
        using Init = int (*)();
        using Handle = int (*)(unsigned, void**);
        auto init = reinterpret_cast<Init>(dlsym(nvml, "nvmlInit_v2"));
        auto handle = reinterpret_cast<Handle>(dlsym(nvml, "nvmlDeviceGetHandleByIndex_v2"));
        rates = reinterpret_cast<Rates>(dlsym(nvml, "nvmlDeviceGetUtilizationRates"));
        if (!init || !handle || !rates || init() != 0 || handle(0, &device) != 0) rates = nullptr;
    }
    float Percent() {
        if (sysfs) {
            std::FILE* f = std::fopen(sysfs_path.c_str(), "r");
            int value = -1;
            if (f) { if (std::fscanf(f, "%d", &value) != 1) value = -1; std::fclose(f); }
            return value < 0 ? -1.0f : float(value);
        }
        struct { unsigned gpu, memory; } u{};
        return rates && rates(device, &u) == 0 ? float(u.gpu) : -1.0f;
    }
private:
    using Rates = int (*)(void*, void*);
    bool sysfs = false;
    std::string sysfs_path;
    void* nvml = nullptr;
    void* device = nullptr;
    Rates rates = nullptr;
};

uint64_t ProcessTicks() { // utime + stime in clock ticks
    std::FILE* f = std::fopen("/proc/self/stat", "r");
    if (!f) return 0;
    char buffer[1024];
    const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
    std::fclose(f);
    buffer[n] = 0;
    const char* p = std::strrchr(buffer, ')'); // the command name may contain spaces
    unsigned long long utime = 0, stime = 0;
    if (!p || std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2)
        return 0;
    return utime + stime;
}

// Each thread's utime + stime since the last call (/proc/self/task), named by its comm.
class ThreadTimes {
public:
    std::vector<ThreadLoad> Loads(uint64_t wall_ticks) {
        std::vector<ThreadLoad> rows;
        DIR* dir = opendir("/proc/self/task");
        if (!dir) return rows;
        std::unordered_map<long, uint64_t> now;
        while (const dirent* e = readdir(dir)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            const long tid = std::atol(e->d_name);
            char path[64], buffer[512];
            std::snprintf(path, sizeof(path), "/proc/self/task/%ld/stat", tid);
            std::FILE* f = std::fopen(path, "r");
            if (!f) continue;
            const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, f);
            std::fclose(f);
            buffer[n] = 0;
            const char* open = std::strchr(buffer, '(');
            const char* close = std::strrchr(buffer, ')');
            unsigned long long utime = 0, stime = 0;
            if (!open || !close || close < open ||
                std::sscanf(close + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2)
                continue;
            const uint64_t total = utime + stime;
            now[tid] = total;
            const auto before = last.find(tid);
            if (before != last.end() && wall_ticks && total >= before->second) { // an id reused: no sample
                rows.push_back({std::string(open + 1, close),
                                float(double(total - before->second) * 100.0 / double(wall_ticks))});
            }
        }
        closedir(dir);
        last = std::move(now);
        return rows;
    }

private:
    std::unordered_map<long, uint64_t> last;
};

void Run(const Vulkan::Instance* instance) {
    GpuUsage gpu;
    ThreadTimes threads;
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    const long hz = sysconf(_SC_CLK_TCK), page = sysconf(_SC_PAGESIZE);
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    uint64_t last_process = 0;
    auto last_wall = std::chrono::steady_clock::now();
    while (true) {
        Sample s;
        s.gpu_name = std::string(instance->GetModelName());
        const uint64_t process = ProcessTicks();
        const auto wall = std::chrono::steady_clock::now();
        if (last_process && hz > 0) {
            const double seconds = std::chrono::duration<double>(wall - last_wall).count();
            const uint64_t wall_ticks = uint64_t(seconds * double(hz));
            s.cpu_percent = CpuPercent(process - last_process, wall_ticks, cores);
            CountRates(s, seconds);
            if (want_threads.load(std::memory_order_relaxed)) s.threads = Busiest(threads.Loads(wall_ticks));
        }
        last_process = process;
        last_wall = wall;
        if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
            unsigned long long size = 0, resident = 0;
            if (std::fscanf(f, "%llu %llu", &size, &resident) == 2) s.ram_used_gb = double(resident) * double(page) / GiB;
            std::fclose(f);
        }
        struct sysinfo info{};
        if (sysinfo(&info) == 0) s.ram_total_gb = double(info.totalram) * double(info.mem_unit) / GiB;
        if (instance->CanReportMemoryUsage()) {
            s.vram_used_gb = double(instance->GetDeviceMemoryUsage()) / GiB;
#ifdef TLG_RENDERER_HOSTED
            s.vram_total_gb = double(instance->GetTotalMemoryBudget()) / GiB;
#else
            s.vram_total_gb = double(instance->GetDeviceMemoryBudgetNow()) / GiB;
#endif
        }
        s.gpu_percent = gpu.Percent();
        {
            std::scoped_lock lock{mutex};
            latest = std::move(s);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
#endif

} // namespace

void Start(const Vulkan::Instance& instance) {
    // ponytail: never joined; a thread that ended while the game ran crashed in ntdll
    // (vk_pipeline_serialization.cpp, StopWarmUp). It sleeps between samples.
    std::call_once(started, [&] { std::thread(Run, &instance).detach(); });
    TlgProfile::Start(); // the profiling session's sensors, with TLG_SESSION_DIR
}

Sample Latest() {
    std::scoped_lock lock{mutex};
    return latest;
}

} // namespace TlgPerfStats
