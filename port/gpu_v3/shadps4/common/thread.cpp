// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <vector>
#include <cstring>
#include <string>
#include <thread>

#include "core/libraries/kernel/threads/pthread.h"

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <bit>
#include <intrin.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // CPU time to grant.
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    // Determine the timebase for converting time to ticks.
    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        // Should not matter since preemptible is false, but needs to be >= computation regardless.
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // Not implemented
}

#endif

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _WIN32

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !_WIN32, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
    // TODO
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    // Do Nothing on MinGW
}

void SetThreadName(void* thread, const char* name) {
    // Do Nothing on MinGW
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

#ifdef _WIN32
namespace {
// Efficiency class per logical processor index from the CPU set list (P-cores report the
// highest class on hybrid parts). Index 0..63 only; enough for this machine (20 threads).
struct CpuClasses {
    std::array<u8, 64> cls{};
    std::array<u8, 64> core{};
    std::array<ULONG, 64> ids{};
    u8 max_class{};
    size_t count{};
};
const CpuClasses& GetCpuClasses() {
    static const CpuClasses classes = [] {
        CpuClasses c{};
        ULONG len = 0;
        GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
        std::vector<u8> buf(len);
        auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data());
        if (!GetSystemCpuSetInformation(info, len, &len, GetCurrentProcess(), 0)) {
            return c;
        }
        for (ULONG off = 0; off < len;) {
            auto* e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
            if (e->Type == CpuSetInformation && e->CpuSet.LogicalProcessorIndex < 64) {
                const auto idx = e->CpuSet.LogicalProcessorIndex;
                c.cls[idx] = e->CpuSet.EfficiencyClass;
                c.core[idx] = e->CpuSet.CoreIndex;
                c.ids[idx] = e->CpuSet.Id;
                c.max_class = std::max<u8>(c.max_class, e->CpuSet.EfficiencyClass);
                c.count = std::max<size_t>(c.count, idx + 1);
            }
            off += e->Size;
        }
        return c;
    }();
    return classes;
}

/// The process affinity mask now (2026-10-15: a harness or the user may restrict it after start,
/// e.g. the weaker-CPU arms; every placement below stays inside it).
u64 ProcessMaskNow() {
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) || proc == 0) {
        return ~u64{0};
    }
    return static_cast<u64>(proc);
}

// Whole performance cores handed to single threads (SHADPS4_CP_QOS=3, SHADPS4_HOT_CORES): the
// process default CPU set is everything except the reserved cores (both logical processors).
std::mutex g_reserved_mutex;
std::array<bool, 64> g_reserved_core{}; // by CoreIndex, performance cores only

/// Gives the calling thread the `rank`-th performance core counted from the last one (0 = last)
/// and keeps the rest of the process off it. `first` receives its first logical processor.
bool ReservePerformanceCore(int rank, int& first, size_t& others_count) {
    const auto& c = GetCpuClasses();
    const u64 allowed = ProcessMaskNow();
    std::vector<int> firsts;
    for (size_t i = 0; i < c.count; ++i) {
        // 2026-10-15: only a P-core whose first logical processor is inside the process mask
        // (10-14 W4E arm: the CP landed on logical 10, outside a 4-E-core mask).
        if (c.cls[i] == c.max_class &&
            (i == 0 || c.core[i - 1] != c.core[i] || c.cls[i - 1] != c.max_class) &&
            ((allowed >> i) & 1) != 0) {
            firsts.push_back(int(i));
        }
    }
    first = -1;
    if (rank < 0 || size_t(rank) >= firsts.size()) {
        return false;
    }
    first = firsts[firsts.size() - 1 - size_t(rank)];
    std::scoped_lock lock{g_reserved_mutex};
    g_reserved_core[c.core[first]] = true;
    std::vector<ULONG> others;
    for (size_t i = 0; i < c.count; ++i) {
        if (!(c.cls[i] == c.max_class && g_reserved_core[c.core[i]]) && ((allowed >> i) & 1) != 0) {
            others.push_back(c.ids[i]);
        }
    }
    others_count = others.size();
    const ULONG mine = c.ids[first];
    return SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(), ULONG(others.size())) &&
           SetThreadSelectedCpuSets(GetCurrentThread(), &mine, 1);
}
} // namespace

void ApplyCriticalThreadPriority(HotThread which) {
    // SHADPS4_CRIT_PRIO (2026-10-15, default off): bit 1 = the recorder lanes and the front, bit 2 =
    // Game:Main, at THREAD_PRIORITY_ABOVE_NORMAL. Nothing is lowered (the reverted POOL_AUTO helper
    // drop put below-normal workers on the back's waits). Trace s1201 (6c/12t mask, b15g): the CP
    // waited 31% of its wall time on the recorders while they were ready-but-not-running 25% of
    // theirs (the front 33%): lean and job threads held the cores the critical chain needed.
    static const unsigned crit = [] {
        const char* value = std::getenv("SHADPS4_CRIT_PRIO");
        return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 0)) : 0u;
    }();
    // Requires SHADPS4_SPINLOCK_PARK (Opus review 10-15): with pure-pause SpinLocks an above-normal
    // waiter never gives its core to a preempted normal-priority holder (region / page locks are
    // taken by guest faults, the front and the recorders alike).
    static const bool park = [] {
        const char* value = std::getenv("SHADPS4_SPINLOCK_PARK");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    if ((crit & (which == HotThread::GameMain ? 2u : 1u)) != 0 && !park) {
        LOG_WARNING(Common, "SHADPS4_CRIT_PRIO={} ignored: needs SHADPS4_SPINLOCK_PARK", crit);
    } else if ((crit & (which == HotThread::GameMain ? 2u : 1u)) != 0) {
        const bool prio_ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        LOG_INFO(Common, "SHADPS4_CRIT_PRIO={}: {} above normal, ok {}", crit,
                 which == HotThread::Recorder ? "recorder"
                 : which == HotThread::Front  ? "front"
                                              : "Game:Main",
                 prio_ok);
    }
}

void ApplyHotThreadPlacement(HotThread which) {
    ApplyCriticalThreadPriority(which);
    static const unsigned mask = [] {
        const char* value = std::getenv("SHADPS4_HOT_CORES");
        return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 0)) : 0u;
    }();
    const unsigned bit = which == HotThread::Recorder ? 1u : which == HotThread::Front ? 2u : 4u;
    if ((mask & bit) == 0) {
        return;
    }
    THREAD_POWER_THROTTLING_STATE throttling{};
    throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &throttling,
                         sizeof(throttling));
    int first = -1;
    size_t others = 0;
    const int rank = which == HotThread::Recorder ? 1 : which == HotThread::Front ? 2 : 3;
    const bool ok = ReservePerformanceCore(rank, first, others);
    LOG_INFO(Common, "SHADPS4_HOT_CORES={}: {} on logical {}, {} logical for the rest, ok {}", mask,
             which == HotThread::Recorder ? "recorder"
             : which == HotThread::Front  ? "front"
                                          : "Game:Main",
             first, others, ok);
}

namespace {
/// Every logical processor of the given class kind (true: performance) minus reserved P-cores.
std::vector<ULONG> ClassCpuSets(bool performance) {
    const auto& c = GetCpuClasses();
    const u64 allowed = ProcessMaskNow();
    std::vector<ULONG> ids;
    std::scoped_lock lock{g_reserved_mutex};
    for (size_t i = 0; i < c.count; ++i) {
        const bool p = c.cls[i] == c.max_class;
        if (p == performance && !(p && g_reserved_core[c.core[i]]) && ((allowed >> i) & 1) != 0) {
            ids.push_back(c.ids[i]);
        }
    }
    return ids;
}
} // namespace

void ApplyGuestThreadPlacement(void* handle, const char* name) {
    static const unsigned mask = [] {
        const char* value = std::getenv("SHADPS4_GUEST_PCORE");
        return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 0)) : 0u;
    }();
    if (mask == 0 || name == nullptr) {
        return;
    }
    const bool job = std::strncmp(name, "JobWorker", 9) == 0;
    const bool main = std::strcmp(name, "Game:Main") == 0;
    if (main) {
        // Review RISK: SHADPS4_HOT_CORES bit 2 already gave Game:Main a whole reserved core.
        const char* hot = std::getenv("SHADPS4_HOT_CORES");
        if (hot && (std::strtoul(hot, nullptr, 0) & 4u) != 0) {
            return;
        }
    }
    if (!((job && (mask & 1)) || (main && (mask & 2)))) {
        return;
    }
    const HANDLE h = static_cast<HANDLE>(handle);
    THREAD_POWER_THROTTLING_STATE throttling{};
    throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0;
    SetThreadInformation(h, ThreadPowerThrottling, &throttling, sizeof(throttling));
    const auto ids = ClassCpuSets(true);
    const bool ok = !ids.empty() && SetThreadSelectedCpuSets(h, ids.data(), ULONG(ids.size()));
    LOG_INFO(Common, "SHADPS4_GUEST_PCORE={}: {} on {} performance logical processors, ok {}", mask,
             name, ids.size(), ok);
}

u32 AutoPoolWorkers(bool lean) {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_POOL_AUTO");
        return v != nullptr && v[0] == '1';
    }();
    if (!on) {
        return 0;
    }
    const u32 n = static_cast<u32>(std::popcount(ProcessMaskNow()));
    // lean: 6 logical -> 3, 12 -> 5, 20 -> 7 (measured best: 6 -> 3, 12 -> 4); A2: 1 below 16.
    const u32 w = lean ? std::clamp<u32>(n / 4 + 2, 2, 8) : (n < 16 ? 1u : 2u);
    LOG_INFO(Common, "SHADPS4_POOL_AUTO: {} logical processors -> {} {} workers", n, w,
             lean ? "lean" : "A2");
    return w;
}

void ApplyHelperThreadPlacement(const char* name) {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_HELPER_PLACE");
        // Not implied by SHADPS4_POOL_AUTO (b15f did, b15g reverted): below-normal workers sit on
        // the back's wait paths (Opus review 10-15: starvation risk on < 16 logical), and
        // SHADPS4_SPINLOCK_PARK=1 alone recovered more at 4c/8t (57.9 vs 54.4 FPS, s1182 / s1172).
        return value ? value[0] - '0' : 0;
    }();
    if (mode <= 0) {
        return;
    }
    bool ok = false;
    size_t n = 0;
    if (mode == 1) {
        const auto ids = ClassCpuSets(false);
        n = ids.size();
        ok = !ids.empty() &&
             SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), ULONG(ids.size()));
    } else if (mode == 2) {
        ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    LOG_INFO(Common, "SHADPS4_HELPER_PLACE={}: {} ({} efficiency logical processors), ok {}", mode,
             name, n, ok);
}

void ProcessMemory(u64& private_bytes, u64& working_set) {
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                                sizeof(pmc))) {
        private_bytes = pmc.PrivateUsage;
        working_set = pmc.WorkingSetSize;
    } else {
        private_bytes = working_set = 0;
    }
}

u64 PhysicalRamBytes() {
    if (const char* sim = std::getenv("SHADPS4_SIM_RAM_MB")) {
        return std::strtoull(sim, nullptr, 10) << 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
}

std::string DescribeSystemMemory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return {};
    }
    u64 private_bytes = 0, working_set = 0;
    ProcessMemory(private_bytes, working_set);
    // ullTotalPageFile is the commit limit (RAM + page files), ullAvailPageFile what is left of it.
    return fmt::format("Windows: RAM free {} of {} MB, commit free {} of limit {} MB, this process "
                       "{} MB private",
                       ms.ullAvailPhys >> 20, ms.ullTotalPhys >> 20, ms.ullAvailPageFile >> 20,
                       ms.ullTotalPageFile >> 20, private_bytes >> 20);
}

CommitCensus WalkCommitted() {
    CommitCensus c{};
    u64 cur_base = 0, cur_bytes = 0;
    const auto flush = [&] {
        if (cur_bytes == 0) {
            return;
        }
        ++c.host_blocks;
        for (int i = 0; i < 8; ++i) {
            if (cur_bytes > c.top_bytes[i]) {
                for (int j = 7; j > i; --j) {
                    c.top_bytes[j] = c.top_bytes[j - 1];
                    c.top_base[j] = c.top_base[j - 1];
                }
                c.top_bytes[i] = cur_bytes;
                c.top_base[i] = cur_base;
                break;
            }
        }
        cur_bytes = 0;
    };
    u64 a = 0x10000;
    MEMORY_BASIC_INFORMATION mbi{};
    while (a < 0x7fffffff0000ull &&
           VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof(mbi)) == sizeof(mbi)) {
        const u64 base = reinterpret_cast<u64>(mbi.BaseAddress);
        const u64 size = mbi.RegionSize;
        if (mbi.State == MEM_COMMIT) {
            const bool guest = base >= 0x200000000ull && base < 0x1000000000ull;
            if (mbi.Type == MEM_PRIVATE) {
                (guest ? c.private_guest : c.private_host) += size;
                if (!guest) {
                    const u64 alloc = reinterpret_cast<u64>(mbi.AllocationBase);
                    if (alloc != cur_base) {
                        flush();
                        cur_base = alloc;
                    }
                    cur_bytes += size;
                }
            } else if (mbi.Type == MEM_MAPPED) {
                (guest ? c.mapped_guest : c.mapped_host) += size;
            } else {
                c.image += size;
            }
        }
        a = base + size;
    }
    flush();
    // Heaps: the default process heap carries the CRT's malloc / new (ours); a driver or other
    // module heap shows up among the others. HeapSummary needs Windows 10.
    HANDLE heaps[256];
    const DWORD n = GetProcessHeaps(256, heaps);
    const HANDLE def = GetProcessHeap();
    for (DWORD i = 0; i < n && i < 256; ++i) {
        HEAP_SUMMARY hs{};
        hs.cb = sizeof(hs);
        if (HeapSummary(heaps[i], 0, &hs)) {
            (heaps[i] == def ? c.default_heap : c.other_heaps) += hs.cbCommitted;
        }
    }
    c.heaps = n;
    return c;
}

void ApplyCommandProcessorPlacement() {
    const char* value = std::getenv("SHADPS4_CP_QOS");
    int mode = value ? value[0] - '0' : 0;
    if (value && (value[0] == 'a' || value[0] == 'A')) {
        // =auto (1.1): mode 3 with enough logical processors in the process mask, else off. A
        // reserved P-core costs a CPU with few cores more than it gives: on 6 cores / 12 threads
        // =0 beat =3 by 11.6% (s1404, fire, A B B A), on 16 (12 P + 4 E threads) by ~8% (s1421).
        // SHADPS4_CP_QOS_AUTO_MIN moves the line.
        const char* min_value = std::getenv("SHADPS4_CP_QOS_AUTO_MIN");
        const int min_logical = min_value ? std::atoi(min_value) : 20;
        const int logical = std::popcount(ProcessMaskNow());
        mode = logical >= min_logical ? 3 : 0;
        LOG_INFO(Common, "SHADPS4_CP_QOS=auto: {} logical processors (line {}) -> mode {}",
                 logical, min_logical, mode);
    }
    if (mode <= 0) {
        return;
    }
    PROCESS_POWER_THROTTLING_STATE throttling{};
    throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0; // always high QoS
    const bool proc_ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                               &throttling, sizeof(throttling));
    THREAD_POWER_THROTTLING_STATE thread_throttling{};
    thread_throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    thread_throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    thread_throttling.StateMask = 0;
    const bool thread_ok = SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling,
                                                &thread_throttling, sizeof(thread_throttling));
    const bool prio_ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    bool sets_ok = true;
    size_t p_cores = 0;
    if (mode == 3) {
        // A whole P-core for this thread: the last performance core's first logical processor
        // for us, and every other thread (process default) kept off both of its siblings.
        int first = -1;
        sets_ok = ReservePerformanceCore(0, first, p_cores);
        if (first >= 0) {
            LOG_INFO(Common, "SHADPS4_CP_QOS=3: CP on logical {} (core {}), {} logical for the rest",
                     first, GetCpuClasses().core[first], p_cores);
        }
    } else if (mode >= 2) {
        const auto& c = GetCpuClasses();
        const u64 allowed = ProcessMaskNow();
        std::vector<ULONG> ids;
        for (size_t i = 0; i < c.count; ++i) {
            if (c.cls[i] == c.max_class && ((allowed >> i) & 1) != 0) {
                ids.push_back(c.ids[i]);
            }
        }
        p_cores = ids.size();
        sets_ok = !ids.empty() &&
                  SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), ULONG(ids.size()));
    }
    LOG_INFO(Common, "SHADPS4_CP_QOS={}: process throttling off {}, thread {}, priority {}, "
                     "P-core set {} ({} logical)",
             mode, proc_ok, thread_ok, prio_ok, sets_ok, p_cores);
}

void SampleCommandProcessorCore() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_CP_CORE_STATS");
        return value && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::array<u64, 64> hits{};
    static u64 samples = 0;
    const DWORD cpu = GetCurrentProcessorNumber();
    ++hits[cpu & 63];
    if (++samples % 200000 != 0) {
        return;
    }
    const auto& c = GetCpuClasses();
    u64 p = 0;
    std::string top;
    for (size_t i = 0; i < 64; ++i) {
        if (hits[i] && c.cls[i] == c.max_class) {
            p += hits[i];
        }
        if (hits[i] * 20 >= samples) {
            top += fmt::format(" cpu{}(c{}):{:.0f}%", i, c.cls[i], 100.0 * hits[i] / samples);
        }
    }
    LOG_INFO(Common, "CP_CORE_STATS: samples {} on P-cores {:.1f}%{}", samples,
             100.0 * p / samples, top);
    hits.fill(0);
    samples = 0;
}
namespace {
struct ThreadPlacementState {
    std::array<u32, 64> hits{};
    std::array<u32, 48> probes{};
    u32 calls{};
    u32 samples{};
    u32 migrations{};
    u32 num_probes{};
    u32 last_cpu{~0u};
    u64 window_tsc{};
    u64 probe_tsc{};
    u64 tsc_per_s{3'500'000'000ULL};
    LARGE_INTEGER window_qpc{};
    u64 cpu_100ns{};
};

u64 ThreadCpuTime100ns() {
    FILETIME creation, exit, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        return 0;
    }
    return ((u64(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
           ((u64(user.dwHighDateTime) << 32) | user.dwLowDateTime);
}

/// TSC ticks for 4096 iterations of a two-operation dependency chain (rotate, xor): the chain
/// runs at the core's real clock, the TSC at a fixed rate. Best of four; ~1 us each.
__declspec(noinline) u32 ClockProbeTicks() {
    static volatile u64 seed = 0x9E3779B97F4A7C15ULL;
    u64 best = ~0ULL;
    for (int rep = 0; rep < 4; ++rep) {
        u64 v = seed;
        const u64 t0 = __rdtsc();
        for (u32 i = 0; i < 4096; ++i) {
            v = std::rotl(v, 7) ^ i;
        }
        const u64 dt = __rdtsc() - t0;
        seed = v;
        best = std::min(best, dt);
    }
    return static_cast<u32>(std::min<u64>(best, 0xffffffffULL));
}

/// Busy share of every logical processor since the previous call (whole machine, all
/// processes), from the kernel's per-processor idle times. One caller only (the CP back).
std::string LogicalProcessorLoad() {
    struct ProcessorPerf {
        LARGE_INTEGER idle, kernel, user, dpc, interrupt;
        ULONG interrupts;
    };
    using QueryFn = LONG(__stdcall*)(ULONG, PVOID, ULONG, PULONG);
    static const QueryFn query = reinterpret_cast<QueryFn>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation"));
    static std::array<ProcessorPerf, 64> last{};
    std::array<ProcessorPerf, 64> now{};
    ULONG bytes = 0;
    // 8 = SystemProcessorPerformanceInformation
    if (!query || query(8, now.data(), ULONG(sizeof(now)), &bytes) < 0) {
        return {};
    }
    const size_t count = bytes / sizeof(ProcessorPerf);
    std::string out;
    for (size_t i = 0; i < count; ++i) {
        // kernel time includes idle time
        const double total = double((now[i].kernel.QuadPart - last[i].kernel.QuadPart) +
                                    (now[i].user.QuadPart - last[i].user.QuadPart));
        const double idle = double(now[i].idle.QuadPart - last[i].idle.QuadPart);
        out += fmt::format(" {}:{:.0f}", i, total > 0 ? 100.0 * (total - idle) / total : 0.0);
    }
    last = now;
    return out;
}
} // namespace

u32 CurrentProcessor() {
    return GetCurrentProcessorNumber();
}
bool IsPerformanceProcessor(u32 cpu) {
    const auto& c = GetCpuClasses();
    return cpu >= 64 || c.cls[cpu] == c.max_class;
}
u32 ClockProbe() {
    return ClockProbeTicks();
}

const bool g_thread_stats = [] {
    const char* value = std::getenv("SHADPS4_THREAD_STATS");
    return value && value[0] == '1';
}();

void SampleThreadPlacementSlow(const char* role, unsigned stride_log2) {
    thread_local ThreadPlacementState st;
    if ((++st.calls & ((1u << stride_log2) - 1)) != 0) {
        return;
    }
    const u32 cpu = GetCurrentProcessorNumber() & 63;
    ++st.hits[cpu];
    ++st.samples;
    st.migrations += st.last_cpu != ~0u && cpu != st.last_cpu;
    st.last_cpu = cpu;
    const u64 now = __rdtsc();
    if (st.window_tsc == 0) {
        st.window_tsc = now;
        st.probe_tsc = now;
        QueryPerformanceCounter(&st.window_qpc);
        st.cpu_100ns = ThreadCpuTime100ns();
        return;
    }
    if (now - st.probe_tsc > st.tsc_per_s / 4 && st.num_probes < st.probes.size()) {
        st.probes[st.num_probes++] = ClockProbeTicks();
        st.probe_tsc = __rdtsc();
    }
    if (now - st.window_tsc < st.tsc_per_s * 10) {
        return;
    }
    LARGE_INTEGER qpc, freq;
    QueryPerformanceCounter(&qpc);
    QueryPerformanceFrequency(&freq);
    const double wall_s = double(qpc.QuadPart - st.window_qpc.QuadPart) / double(freq.QuadPart);
    const u64 cpu_100ns = ThreadCpuTime100ns();
    const double cpu_s = double(cpu_100ns - st.cpu_100ns) * 1e-7;
    if (wall_s > 1.0) {
        st.tsc_per_s = static_cast<u64>(double(now - st.window_tsc) / wall_s);
    }
    const auto& c = GetCpuClasses();
    std::string top;
    for (size_t i = 0; i < 64; ++i) {
        if (u64(st.hits[i]) * 50 >= st.samples) {
            top += fmt::format(" cpu{}({}{}):{:.0f}%", i, c.cls[i] == c.max_class ? "P" : "E",
                               c.core[i], 100.0 * st.hits[i] / st.samples);
        }
    }
    std::sort(st.probes.begin(), st.probes.begin() + st.num_probes);
    const u32 probe_min = st.num_probes ? st.probes[0] : 0;
    const u32 probe_med = st.num_probes ? st.probes[st.num_probes / 2] : 0;
    // 8192 chain operations per probe: operations per TSC tick x TSC rate = core clock estimate
    // (exact only if each operation takes one cycle; compare between runs of the same EXE).
    const auto ghz = [&](u32 ticks) {
        return ticks ? 8192.0 / double(ticks) * double(st.tsc_per_s) * 1e-9 : 0.0;
    };
    LOG_INFO(Common,
             "ThreadStats: role={} window_s={:.2f} cpu_s={:.2f} ({:.1f}%) samples={} migrations={} "
             "clock_probe_ticks med={} min={} n={} (~{:.2f} / {:.2f} GHz) tsc_mhz={:.0f} on{}",
             role, wall_s, cpu_s, 100.0 * cpu_s / std::max(wall_s, 1e-9), st.samples,
             st.migrations, probe_med, probe_min, st.num_probes, ghz(probe_med), ghz(probe_min),
             double(st.tsc_per_s) * 1e-6, top);
    if (role[0] == 'b') {
        LOG_INFO(Common, "ThreadStats: logical processor busy % (all processes):{}",
                 LogicalProcessorLoad());
    }
    st.hits.fill(0);
    st.samples = 0;
    st.migrations = 0;
    st.num_probes = 0;
    st.window_tsc = now;
    st.window_qpc = qpc;
    st.cpu_100ns = cpu_100ns;
}
#else
u32 AutoPoolWorkers(bool) {
    return 0;
}
void ApplyGuestThreadPlacement(void*, const char*) {}
void ApplyHelperThreadPlacement(const char*) {}
void ApplyCommandProcessorPlacement() {}
void ProcessMemory(u64& private_bytes, u64& working_set) {
    private_bytes = working_set = 0;
}
CommitCensus WalkCommitted() {
    return {};
}
u64 PhysicalRamBytes() {
    const char* sim = std::getenv("SHADPS4_SIM_RAM_MB");
    return sim ? std::strtoull(sim, nullptr, 10) << 20 : 0;
}
std::string DescribeSystemMemory() {
    return {};
}
void ApplyHotThreadPlacement(HotThread) {}
void ApplyCriticalThreadPriority(HotThread) {}
void SampleCommandProcessorCore() {}
const bool g_thread_stats = false;
void SampleThreadPlacementSlow(const char*, unsigned) {}
u32 CurrentProcessor() {
    return 0;
}
bool IsPerformanceProcessor(u32) {
    return true;
}
u32 ClockProbe() {
    return 0;
}
#endif

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name = nullptr;
    if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || !name) {
        return "<unknown name>";
    }
    std::string result = Common::UTF16ToUTF8(name);
    LocalFree(name);
    return result;
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

u64 CurrentThreadCycles() {
#ifdef _WIN32
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return cycles;
#else
    return 0;
#endif
}

} // namespace Common
