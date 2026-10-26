// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the profiling session's sensors (tlg_profile.h, docs/profiling.md).

#include "tlg_profile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tlg_perf_stats.h"

#ifdef TLG_PROFILE
#include <tracy/Tracy.hpp>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

namespace TlgProfile {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t NowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

// The readings of one sample, written together; each name is also a Tracy plot, whose name must
// stay at one address for the whole run (Names keeps every one ever used).
class Writer {
public:
    explicit Writer(std::FILE* file) : file(file) {}
    void Add(const std::string& name, double value) {
        std::fprintf(file, "%llu,%s,%.4g\n", (unsigned long long)time, name.c_str(), value);
#ifdef TLG_PROFILE
        TracyPlot(Intern(name), value);
#endif
    }
    void Flush() { std::fflush(file); }
    uint64_t time = 0;

private:
#ifdef TLG_PROFILE
    const char* Intern(const std::string& name) {
        auto [it, added] = names.try_emplace(name, nullptr);
        if (added) it->second = storage.emplace_back(name).c_str();
        return it->second;
    }
    std::map<std::string, const char*> names;
    std::deque<std::string> storage;
#endif
    std::FILE* file;
};

#ifndef _WIN32
// One number from a sysfs or procfs file; false when it is missing.
bool ReadNumber(const std::string& path, double& value) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    const bool ok = std::fscanf(f, "%lf", &value) == 1;
    std::fclose(f);
    return ok;
}

std::string ReadLine(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return {};
    char line[256] = {};
    if (!std::fgets(line, sizeof(line), f)) line[0] = 0;
    std::fclose(f);
    std::string text = line;
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
    return text;
}

// pp_dpm_sclk / pp_dpm_mclk: "1: 1600Mhz *"; the line with the star is the clock now.
double CurrentDpmMhz(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return -1;
    char line[128];
    double mhz = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (!std::strchr(line, '*')) continue;
        const char* colon = std::strchr(line, ':');
        if (colon) mhz = std::atof(colon + 1);
    }
    std::fclose(f);
    return mhz;
}

// The machine: cores, the amdgpu device and its hwmon, the CPU's temperature sensor, a battery.
struct Machine {
    int cores = 0;
    std::string gpu;       // /sys/class/drm/cardN/device
    std::string gpu_hwmon; // its hwmon directory
    std::string cpu_temp;  // k10temp / zenpower / coretemp temp1_input
    std::string battery;   // /sys/class/power_supply/BAT*

    Machine() {
        cores = int(sysconf(_SC_NPROCESSORS_CONF));
        for (int card = 0; card < 8 && gpu.empty(); ++card) {
            const std::string device = "/sys/class/drm/card" + std::to_string(card) + "/device";
            double busy;
            if (ReadNumber(device + "/gpu_busy_percent", busy)) gpu = device;
        }
        if (!gpu.empty()) {
            const std::string hwmon = FirstEntry(gpu + "/hwmon", "hwmon");
            if (!hwmon.empty()) gpu_hwmon = gpu + "/hwmon/" + hwmon;
        }
        if (DIR* dir = opendir("/sys/class/hwmon")) {
            while (dirent* entry = readdir(dir)) {
                const std::string base = std::string("/sys/class/hwmon/") + entry->d_name;
                const std::string name = ReadLine(base + "/name");
                if (name == "k10temp" || name == "zenpower" || name == "coretemp") {
                    cpu_temp = base + "/temp1_input";
                    break;
                }
            }
            closedir(dir);
        }
        const std::string bat = FirstEntry("/sys/class/power_supply", "BAT");
        if (!bat.empty()) battery = "/sys/class/power_supply/" + bat;
    }

    // The name of the first entry of `dir_path` that starts with `prefix`, or "".
    static std::string FirstEntry(const std::string& dir_path, const char* prefix) {
        std::string found;
        if (DIR* dir = opendir(dir_path.c_str())) {
            while (dirent* entry = readdir(dir)) {
                if (std::strncmp(entry->d_name, prefix, std::strlen(prefix)) == 0) {
                    found = entry->d_name;
                    break;
                }
            }
            closedir(dir);
        }
        return found;
    }
};

// Each core's busy share since the last call, from /proc/stat.
class CoreLoad {
public:
    void Sample(Writer& w) {
        std::FILE* f = std::fopen("/proc/stat", "r");
        if (!f) return;
        char line[512];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "cpu", 3) != 0) break;
            char name[16];
            unsigned long long v[8] = {};
            if (std::sscanf(line, "%15s %llu %llu %llu %llu %llu %llu %llu %llu", name, &v[0], &v[1], &v[2], &v[3],
                            &v[4], &v[5], &v[6], &v[7]) < 5)
                continue;
            const unsigned long long idle = v[3] + v[4], total = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
            auto& last = previous[name];
            if (last.second && total > last.second) {
                const double busy = 100.0 * (1.0 - double(idle - last.first) / double(total - last.second));
                w.Add(std::strcmp(name, "cpu") == 0 ? "cpu busy %" : std::string(name) + " busy %", busy);
            }
            last = {idle, total};
        }
        std::fclose(f);
    }

private:
    std::map<std::string, std::pair<unsigned long long, unsigned long long>> previous;
};

// Every thread's running and runnable-but-waiting time since the last call (schedstat, ns),
// summed by thread name, as a share of the interval.
class Threads {
public:
    void Sample(Writer& w, uint64_t interval_ns) {
        std::map<std::string, std::pair<double, double>> by_name;
        std::map<long, std::pair<unsigned long long, unsigned long long>> now;
        if (DIR* dir = opendir("/proc/self/task")) {
            while (dirent* entry = readdir(dir)) {
                const long tid = std::atol(entry->d_name);
                if (tid <= 0) continue;
                const std::string base = "/proc/self/task/" + std::to_string(tid);
                std::FILE* f = std::fopen((base + "/schedstat").c_str(), "r");
                if (!f) continue;
                unsigned long long run = 0, wait = 0;
                const bool ok = std::fscanf(f, "%llu %llu", &run, &wait) == 2;
                std::fclose(f);
                if (!ok) continue;
                now[tid] = {run, wait};
                const auto last = previous.find(tid);
                if (last == previous.end() || !interval_ns) continue;
                auto& sum = by_name[ReadLine(base + "/comm")];
                sum.first += double(run - last->second.first);
                sum.second += double(wait - last->second.second);
            }
            closedir(dir);
        }
        previous = std::move(now);
        for (const auto& [name, times] : by_name) {
            w.Add("thread run % " + name, 100.0 * times.first / double(interval_ns));
            w.Add("thread wait % " + name, 100.0 * times.second / double(interval_ns));
        }
    }

private:
    std::map<long, std::pair<unsigned long long, unsigned long long>> previous;
};

void SampleMachine(Writer& w, const Machine& m) {
    for (int core = 0; core < m.cores; ++core) {
        double khz;
        if (ReadNumber("/sys/devices/system/cpu/cpu" + std::to_string(core) + "/cpufreq/scaling_cur_freq", khz))
            w.Add("cpu" + std::to_string(core) + " MHz", khz / 1000.0);
    }
    double value;
    if (!m.gpu.empty()) {
        if (ReadNumber(m.gpu + "/gpu_busy_percent", value)) w.Add("gpu busy %", value);
        if (ReadNumber(m.gpu + "/mem_busy_percent", value)) w.Add("gpu memory busy %", value);
        const double sclk = CurrentDpmMhz(m.gpu + "/pp_dpm_sclk"), mclk = CurrentDpmMhz(m.gpu + "/pp_dpm_mclk");
        if (sclk >= 0) w.Add("gpu MHz", sclk);
        if (mclk >= 0) w.Add("gpu memory MHz", mclk);
    }
    if (!m.gpu_hwmon.empty()) {
        if (ReadNumber(m.gpu_hwmon + "/power1_average", value) || ReadNumber(m.gpu_hwmon + "/power1_input", value))
            w.Add("soc power W", value / 1e6);
        if (ReadNumber(m.gpu_hwmon + "/temp1_input", value)) w.Add("gpu temp C", value / 1000.0);
        if (ReadNumber(m.gpu_hwmon + "/freq1_input", value)) w.Add("gpu sclk MHz", value / 1e6);
    }
    if (!m.cpu_temp.empty() && ReadNumber(m.cpu_temp, value)) w.Add("cpu temp C", value / 1000.0);
    if (!m.battery.empty()) {
        if (ReadNumber(m.battery + "/power_now", value)) w.Add("battery W", value / 1e6);
        if (ReadNumber(m.battery + "/capacity", value)) w.Add("battery %", value);
    }
}
#else
// Every thread's CPU time since the last call (GetThreadTimes), summed by thread name, as a share
// of the interval. Windows keeps no runnable-but-waiting time per thread.
class Threads {
public:
    void Sample(Writer& w, uint64_t interval_ns) {
        using Describe = HRESULT(WINAPI*)(HANDLE, PWSTR*);
        static const auto describe = reinterpret_cast<Describe>(
            reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
        std::map<std::string, double> by_name;
        std::map<DWORD, unsigned long long> now;
        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return;
        THREADENTRY32 entry{sizeof(entry)};
        const DWORD process = GetCurrentProcessId();
        for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID != process) continue;
            const HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
            if (!thread) continue;
            FILETIME created, ended, kernel, user;
            std::string name = "thread " + std::to_string(entry.th32ThreadID);
            if (GetThreadTimes(thread, &created, &ended, &kernel, &user)) {
                const unsigned long long total =
                    (ULARGE_INTEGER{{kernel.dwLowDateTime, kernel.dwHighDateTime}}.QuadPart +
                     ULARGE_INTEGER{{user.dwLowDateTime, user.dwHighDateTime}}.QuadPart) * 100;
                PWSTR text = nullptr;
                if (describe && SUCCEEDED(describe(thread, &text)) && text && *text) {
                    char utf8[128];
                    if (WideCharToMultiByte(CP_UTF8, 0, text, -1, utf8, sizeof(utf8), nullptr, nullptr) > 1) name = utf8;
                }
                if (text) LocalFree(text);
                now[entry.th32ThreadID] = total;
                const auto last = previous.find(entry.th32ThreadID);
                if (last != previous.end() && interval_ns) by_name[name] += double(total - last->second);
            }
            CloseHandle(thread);
        }
        CloseHandle(snapshot);
        previous = std::move(now);
        for (const auto& [name, ns] : by_name) w.Add("thread run % " + name, 100.0 * ns / double(interval_ns));
    }

private:
    std::map<DWORD, unsigned long long> previous;
};
#endif

void Run(std::string dir) {
    std::FILE* file = std::fopen((dir + "/sensors.csv").c_str(), "w");
    if (!file) {
        std::fprintf(stderr, "Profile: cannot write %s/sensors.csv\n", dir.c_str());
        return;
    }
    std::setvbuf(file, nullptr, _IOFBF, 1 << 20);
    std::fputs("time_ns,name,value\n", file);
    const char* period_text = std::getenv("TLG_SESSION_SAMPLE_MS");
    const int period_ms = period_text && std::atoi(period_text) > 0 ? std::atoi(period_text) : 20;
    std::printf("Profile: sensors every %d ms to %s/sensors.csv\n", period_ms, dir.c_str());
    Writer w{file};
#ifndef _WIN32
    const Machine machine;
    CoreLoad cores;
#endif
    Threads threads;
    uint64_t last_threads = 0, last_overlay = 0;
    for (auto next = Clock::now();; next += std::chrono::milliseconds(period_ms)) {
        std::this_thread::sleep_until(next);
        w.time = NowNs();
#ifndef _WIN32
        SampleMachine(w, machine);
        cores.Sample(w);
#endif
        if (w.time - last_threads >= 100'000'000ull) { // the threads every 100 ms
            threads.Sample(w, last_threads ? w.time - last_threads : 0);
            last_threads = w.time;
        }
        if (w.time - last_overlay >= 500'000'000ull) { // the overlay's own sample is 500 ms
            const TlgPerfStats::Sample s = TlgPerfStats::Latest();
            if (s.fps > 0) w.Add("fps", s.fps);
            if (s.gpu_percent >= 0) w.Add("overlay gpu %", s.gpu_percent);
            if (s.cpu_percent >= 0) w.Add("process cpu %", s.cpu_percent);
            if (s.vram_used_gb >= 0) w.Add("vram GB", s.vram_used_gb);
            if (s.ram_used_gb >= 0) w.Add("ram GB", s.ram_used_gb);
            if (s.faults_per_second >= 0) w.Add("gpu memory faults /s", s.faults_per_second);
            if (s.pipelines_per_second >= 0) w.Add("pipelines made /s", s.pipelines_per_second);
            last_overlay = w.time;
            w.Flush();
        }
    }
}

std::once_flag started;

} // namespace

void Start() {
    const char* dir = std::getenv("TLG_SESSION_DIR");
    if (!dir || !*dir) return;
    // Never joined, as TlgPerfStats' thread: it sleeps between samples until the process ends.
    std::call_once(started, [dir] { std::thread(Run, std::string(dir)).detach(); });
}

} // namespace TlgProfile
