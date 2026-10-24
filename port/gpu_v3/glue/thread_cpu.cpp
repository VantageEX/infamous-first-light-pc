// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3 diagnostics: TLG_V3_THREAD_CPU=1 prints, every 10 s (the GPU timer's window,
// SHADPS4_GPU_BATCH_TIMING=1), the frame rate and the busiest threads of the process by name with
// their share of one core over the interval (which thread holds the frame
// rate back: a game thread, the command processor front or back, a worker).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif
#include "v3_glue.h"
#include "v3_stutter.h"

namespace V3 {

void StartThreadCpuReport() {
#ifdef _WIN32
    const char* on = std::getenv("TLG_V3_THREAD_CPU");
    if (!(on && on[0] == '1')) return;
    std::thread([] {
        using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
        const auto get_description = reinterpret_cast<GetDescription>(
            reinterpret_cast<void (*)()>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
        std::unordered_map<DWORD, unsigned long long> last;
        auto last_time = std::chrono::steady_clock::now();
        uint64_t last_frames = Stutter::g_presented.load(std::memory_order_relaxed);
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            const auto now = std::chrono::steady_clock::now();
            const double wall_100ns = std::chrono::duration<double>(now - last_time).count() * 1e7;
            const uint64_t frames = Stutter::g_presented.load(std::memory_order_relaxed);
            const double fps = double(frames - last_frames) * 1e7 / wall_100ns;
            last_frames = frames;
            last_time = now;
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snapshot == INVALID_HANDLE_VALUE) continue;
            struct Row {
                double percent;
                std::string name;
            };
            std::vector<Row> rows;
            std::unordered_map<std::string, double> by_name;
            THREADENTRY32 entry{.dwSize = sizeof(entry)};
            const DWORD process = GetCurrentProcessId();
            for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
                if (entry.th32OwnerProcessID != process) continue;
                HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
                if (!thread) continue;
                FILETIME created, exited, kernel, user;
                if (GetThreadTimes(thread, &created, &exited, &kernel, &user)) {
                    const unsigned long long total =
                        (ULARGE_INTEGER{{kernel.dwLowDateTime, kernel.dwHighDateTime}}.QuadPart) +
                        (ULARGE_INTEGER{{user.dwLowDateTime, user.dwHighDateTime}}.QuadPart);
                    auto it = last.find(entry.th32ThreadID);
                    const unsigned long long before = it == last.end() ? total : it->second;
                    last[entry.th32ThreadID] = total;
                    std::string name = "tid " + std::to_string(entry.th32ThreadID);
                    PWSTR wide = nullptr;
                    if (get_description && SUCCEEDED(get_description(thread, &wide)) && wide) {
                        if (*wide) {
                            const int n = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
                            std::string utf8(size_t(n > 0 ? n - 1 : 0), '\0');
                            WideCharToMultiByte(CP_UTF8, 0, wide, -1, utf8.data(), n, nullptr, nullptr);
                            name = utf8;
                        }
                        LocalFree(wide);
                    }
                    rows.push_back({100.0 * double(total - before) / wall_100ns, name});
                }
                CloseHandle(thread);
            }
            CloseHandle(snapshot);
            std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.percent > b.percent; });
            double sum = 0;
            for (const auto& r : rows) sum += r.percent;
            std::string line;
            for (size_t i = 0; i < rows.size() && i < 14; ++i) {
                char part[160];
                std::snprintf(part, sizeof(part), " %s %.0f%%", rows[i].name.c_str(), rows[i].percent);
                line += part;
            }
            std::fprintf(stderr, "V3 thread CPU (10 s, %.1f FPS, %% of a core; process %.0f%%):%s\n", fps, sum,
                         line.c_str());
        }
    }).detach();
#endif
}

} // namespace V3
