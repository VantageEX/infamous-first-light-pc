// SPDX-License-Identifier: GPL-2.0-or-later
// Renderer v3 diagnostics: TLG_V3_PROFILE=1, a sampling profiler of the busiest threads. Every
// second it picks the threads that used over 40% of a core in the last one; about every
// millisecond it suspends each, reads its registers and walks its stack with the modules' own
// unwind tables (.pdata, read from their images in memory: no lock, nothing allocated while a
// thread is stopped), and resumes it. Every 10 s it prints, per thread, the functions taking the
// most samples, by their own code (self) and with their callees (incl), as module+offset of the
// function's start: tools/v3_profile.py names them with llvm-symbolizer.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_map>
#include <cstring>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif
#include "v3_glue.h"

namespace V3 {

#ifdef _WIN32
namespace {

struct Module {
    uintptr_t base, end;
    const RUNTIME_FUNCTION* pdata;
    size_t count;
    std::string name;
};

std::vector<Module> LoadModules() {
    std::vector<Module> modules;
    HMODULE handles[1024];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), handles, sizeof(handles), &needed)) return modules;
    for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 1024; ++i) {
        const auto base = reinterpret_cast<uintptr_t>(handles[i]);
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        char path[MAX_PATH] = "?";
        GetModuleFileNameA(handles[i], path, sizeof(path));
        const char* slash = std::max(std::strrchr(path, '\\'), std::strrchr(path, '/'));
        modules.push_back({base, base + nt->OptionalHeader.SizeOfImage,
                           reinterpret_cast<const RUNTIME_FUNCTION*>(base + dir.VirtualAddress),
                           dir.Size / sizeof(RUNTIME_FUNCTION), slash ? slash + 1 : path});
    }
    std::sort(modules.begin(), modules.end(), [](const Module& a, const Module& b) { return a.base < b.base; });
    return modules;
}

const Module* FindModule(const std::vector<Module>& modules, uintptr_t address) {
    auto it = std::upper_bound(modules.begin(), modules.end(), address,
                               [](uintptr_t a, const Module& m) { return a < m.base; });
    if (it == modules.begin()) return nullptr;
    --it;
    return address < it->end ? &*it : nullptr;
}

// The function entry covering `rva` (sorted .pdata, binary search); a chained entry is followed
// to its primary one by RtlVirtualUnwind itself, the start reported is the primary's.
const RUNTIME_FUNCTION* FindFunction(const Module& m, DWORD rva) {
    size_t lo = 0, hi = m.count;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (rva < m.pdata[mid].BeginAddress) hi = mid;
        else if (rva >= m.pdata[mid].EndAddress) lo = mid + 1;
        else return &m.pdata[mid];
    }
    return nullptr;
}

constexpr int MaxFrames = 32;

// One sample: up to MaxFrames function starts (0: an address with no unwind entry, the walk ends).
int Walk(const std::vector<Module>& modules, CONTEXT& context, uintptr_t* starts) {
    int n = 0;
    while (n < MaxFrames && context.Rip != 0) {
        const Module* m = FindModule(modules, context.Rip);
        const RUNTIME_FUNCTION* f = m ? FindFunction(*m, DWORD(context.Rip - m->base)) : nullptr;
        if (!f) {
            starts[n++] = context.Rip | 1; // odd: a raw address (leaf function or JIT code)
            // A leaf function: its return address is at the top of the stack.
            if (n != 1) break;
            context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
            context.Rsp += 8;
            continue;
        }
        starts[n++] = m->base + f->BeginAddress;
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, m->base, context.Rip, const_cast<PRUNTIME_FUNCTION>(f), &context,
                         &handler_data, &frame, nullptr);
    }
    return n;
}

std::string Name(const std::vector<Module>& modules, uintptr_t address) {
    const Module* m = FindModule(modules, address & ~uintptr_t{1});
    char out[MAX_PATH + 32];
    if (!m) {
        std::snprintf(out, sizeof(out), "?%#llx", static_cast<unsigned long long>(address));
    } else {
        std::snprintf(out, sizeof(out), "%s+%#llx%s", m->name.c_str(),
                      static_cast<unsigned long long>((address & ~uintptr_t{1}) - m->base),
                      address & 1 ? "~" : "");
    }
    return out;
}

struct Profile {
    std::string name;
    uint64_t samples = 0;
    std::unordered_map<uintptr_t, uint64_t> self, incl;
};

std::string ThreadName(HANDLE thread, DWORD tid) {
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetDescription>(
        reinterpret_cast<void (*)()>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
    std::string name = "tid " + std::to_string(tid);
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
    return name;
}

void Print(const std::vector<Module>& modules, const Profile& p) {
    const auto top = [&](const std::unordered_map<uintptr_t, uint64_t>& counts, size_t limit) {
        std::vector<std::pair<uintptr_t, uint64_t>> rows(counts.begin(), counts.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string line;
        for (size_t i = 0; i < rows.size() && i < limit; ++i) {
            char part[MAX_PATH + 64];
            std::snprintf(part, sizeof(part), " %s %.1f%%", Name(modules, rows[i].first).c_str(),
                          100.0 * double(rows[i].second) / double(p.samples));
            line += part;
        }
        return line;
    };
    std::fprintf(stderr, "V3 profile %s (%llu samples) self:%s\n", p.name.c_str(),
                 static_cast<unsigned long long>(p.samples), top(p.self, 25).c_str());
    std::fprintf(stderr, "V3 profile %s incl:%s\n", p.name.c_str(), top(p.incl, 40).c_str());
}

void Run() {
    std::vector<Module> modules = LoadModules();
    std::unordered_map<DWORD, unsigned long long> last_cpu;
    std::unordered_map<DWORD, Profile> profiles;
    struct Target {
        DWORD tid;
        HANDLE handle;
    };
    std::vector<Target> targets;
    const DWORD self_tid = GetCurrentThreadId();
    auto last_pick = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    auto last_print = std::chrono::steady_clock::now();
    uintptr_t starts[MaxFrames];
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_pick >= std::chrono::seconds(1)) {
            const double wall_100ns = std::chrono::duration<double>(now - last_pick).count() * 1e7;
            last_pick = now;
            for (auto& t : targets) CloseHandle(t.handle);
            targets.clear();
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snapshot != INVALID_HANDLE_VALUE) {
                THREADENTRY32 entry{.dwSize = sizeof(entry)};
                const DWORD process = GetCurrentProcessId();
                for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
                    if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self_tid) continue;
                    HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                               FALSE, entry.th32ThreadID);
                    if (!thread) continue;
                    FILETIME created, exited, kernel, user;
                    bool keep = false;
                    if (GetThreadTimes(thread, &created, &exited, &kernel, &user)) {
                        const unsigned long long total =
                            ULARGE_INTEGER{{kernel.dwLowDateTime, kernel.dwHighDateTime}}.QuadPart +
                            ULARGE_INTEGER{{user.dwLowDateTime, user.dwHighDateTime}}.QuadPart;
                        auto it = last_cpu.find(entry.th32ThreadID);
                        keep = it != last_cpu.end() && double(total - it->second) > 0.4 * wall_100ns;
                        last_cpu[entry.th32ThreadID] = total;
                    }
                    if (keep) {
                        Profile& p = profiles[entry.th32ThreadID];
                        if (p.name.empty()) p.name = ThreadName(thread, entry.th32ThreadID);
                        targets.push_back({entry.th32ThreadID, thread});
                    } else {
                        CloseHandle(thread);
                    }
                }
                CloseHandle(snapshot);
            }
        }
        if (now - last_print >= std::chrono::seconds(10)) {
            last_print = now;
            std::vector<const Profile*> order;
            for (const auto& [tid, p] : profiles) {
                if (p.samples > 0) order.push_back(&p);
            }
            std::sort(order.begin(), order.end(), [](const Profile* a, const Profile* b) { return a->samples > b->samples; });
            for (size_t i = 0; i < order.size() && i < 4; ++i) Print(modules, *order[i]);
            profiles.clear();
            modules = LoadModules(); // a module loaded since (no thread is stopped here)
        }
        for (const Target& t : targets) {
            if (SuspendThread(t.handle) == DWORD(-1)) continue;
            CONTEXT context{};
            context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            int n = 0;
            if (GetThreadContext(t.handle, &context)) n = Walk(modules, context, starts);
            ResumeThread(t.handle);
            if (n == 0) continue;
            Profile& p = profiles[t.tid];
            ++p.samples;
            ++p.self[starts[0]];
            // Inclusive: each function once per sample (recursion counts once).
            for (int i = 0; i < n; ++i) {
                bool seen = false;
                for (int j = 0; j < i && !seen; ++j) seen = starts[j] == starts[i];
                if (!seen) ++p.incl[starts[i]];
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace
#endif

void StartProfiler() {
#ifdef _WIN32
    const char* on = std::getenv("TLG_V3_PROFILE");
    if (!(on && on[0] == '1')) return;
    std::thread([] {
        SetThreadDescription(GetCurrentThread(), L"V3 profiler");
        Run();
    }).detach();
#endif
}

} // namespace V3
