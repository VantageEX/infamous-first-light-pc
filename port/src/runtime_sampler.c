/* The profiling session's guest sampler (docs/profiling.md). With TLG_SESSION_DIR set, a thread
 * looks at every guest thread TLG_GUEST_SAMPLE_HZ times a second (default 200, 0 = off) and writes
 * <dir>/guest_samples.bin: for a thread blocked in one of the runtime's waits, what it waits on
 * (no need to stop it); for a running one, its RIP and frame chain, read as the freeze report
 * reads them (runtime_freeze_sample: a signal on Linux, a short suspension on Windows). The
 * analyser turns the addresses into offsets in the game's image: where the game's own code spends
 * the CPU, per thread.
 *
 * The file (little endian): "TLGSMPL2", u64 image base, u64 image size, u32 rate (Hz), u32 0;
 * then records, each starting with a type byte:
 *   'T' u32 tid, u8 n, n bytes: a thread's name (before its first sample)
 *   'R' u64 time_ns, u32 tid, u8 flags, u8 n, u64 rip, n x u64 return addresses: running;
 *       flags bit 0: the instruction before RIP is a syscall, so the thread is in a system call
 *       (blocked in the OS, or just entering it: a wait the runtime did not mark)
 *   'W' u64 time_ns, u32 tid, u8 n, n bytes: blocked, the wait's kind ("mutex", "sleep"...)
 * time_ns is the steady clock of sensors.csv and the renderer v2 timeline.
 * <dir>/modules.txt lists the loaded modules ("base size path" a line, hex; Linux: a copy of
 * /proc/self/maps), so the analyser can put the samples outside the game's image (the runtime,
 * the renderer, the driver) on their module and its symbols. */
#define _GNU_SOURCE /* strnlen with -std=c11 (glibc) */
#include "runtime.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#define PSAPI_VERSION 2 /* EnumProcessModules... from kernel32 (K32*): no psapi.lib */
#include <psapi.h>
#endif

#define SAMPLE_THREADS 512
#define SAMPLE_FRAMES 12

static uint64_t steady_ns(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER now;
    if (!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    return (uint64_t)((__int128)now.QuadPart * 1000000000 / frequency.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
#endif
}

static void sleep_until_ns(uint64_t deadline) {
    uint64_t now = steady_ns();
    if (deadline <= now) return;
#ifdef _WIN32
    Sleep((DWORD)((deadline - now) / 1000000));
#else
    struct timespec nap = {(time_t)((deadline - now) / 1000000000u), (long)((deadline - now) % 1000000000u)};
    nanosleep(&nap, NULL);
#endif
}

static void put(FILE *f, const void *data, size_t size) { fwrite(data, 1, size, f); }
static void put_u8(FILE *f, uint8_t v) { put(f, &v, 1); }
static void put_u32(FILE *f, uint32_t v) { put(f, &v, 4); }
static void put_u64(FILE *f, uint64_t v) { put(f, &v, 8); }
static void put_text(FILE *f, const char *text) {
    size_t n = strnlen(text ? text : "", 255);
    put_u8(f, (uint8_t)n);
    put(f, text ? text : "", n);
}

typedef struct { FILE *file; unsigned hz; } Sampler;

static void *sampler(void *argument) {
    Sampler *s = argument;
    static RuntimeThreadInfo threads[SAMPLE_THREADS];
    static uint32_t named[SAMPLE_THREADS * 4];
    size_t named_count = 0;
    uint64_t period = 1000000000u / s->hz, next = steady_ns(), samples = 0;
    for (;;) {
        next += period;
        sleep_until_ns(next);
        const size_t count = runtime_thread_snapshot(threads, SAMPLE_THREADS);
        for (size_t i = 0; i < count; ++i) {
            const RuntimeThreadInfo *t = &threads[i];
            if (!t->record || t->finished || !t->os_tid) continue; /* guest threads only */
            const uint32_t tid = (uint32_t)t->os_tid;
            int known = 0;
            for (size_t k = 0; k < named_count && !known; ++k) known = named[k] == tid;
            if (!known) {
                if (named_count < sizeof(named) / sizeof(*named)) named[named_count++] = tid;
                put_u8(s->file, 'T'); put_u32(s->file, tid); put_text(s->file, t->name);
            }
            const char *kind = t->wait.depth ? t->wait.level[0].kind : NULL;
            const uint64_t now = steady_ns();
            if (kind) {
                put_u8(s->file, 'W'); put_u64(s->file, now); put_u32(s->file, tid); put_text(s->file, kind);
                continue;
            }
            uintptr_t rip, chain[SAMPLE_FRAMES];
            unsigned frames = 0;
            if (!runtime_freeze_sample(t->os_tid, &rip, chain, SAMPLE_FRAMES, &frames)) continue;
            unsigned char before[2] = {0};
            const int in_syscall = runtime_freeze_read(rip - 2, before, 2) && before[0] == 0x0f && before[1] == 0x05;
            put_u8(s->file, 'R'); put_u64(s->file, now); put_u32(s->file, tid); put_u8(s->file, (uint8_t)in_syscall);
            put_u8(s->file, (uint8_t)frames);
            put_u64(s->file, rip);
            for (unsigned f = 0; f < frames; ++f) put_u64(s->file, chain[f]);
        }
        if (++samples % (s->hz ? s->hz : 1) == 0) fflush(s->file); /* about once a second */
    }
    return NULL;
}

static void write_modules(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/modules.txt", dir);
    FILE *out = fopen(path, "w");
    if (!out) return;
#ifdef _WIN32
    HMODULE modules[1024];
    DWORD needed = 0;
    HANDLE process = GetCurrentProcess();
    if (EnumProcessModules(process, modules, sizeof(modules), &needed)) {
        for (DWORD i = 0; i < needed / sizeof(HMODULE) && i < 1024; ++i) {
            MODULEINFO info;
            char name[MAX_PATH];
            if (GetModuleInformation(process, modules[i], &info, sizeof(info)) &&
                GetModuleFileNameA(modules[i], name, sizeof(name)))
                fprintf(out, "%llx %lx %s\n", (unsigned long long)(uintptr_t)info.lpBaseOfDll, (unsigned long)info.SizeOfImage, name);
        }
    }
#else
    FILE *maps = fopen("/proc/self/maps", "r");
    char line[1024];
    while (maps && fgets(line, sizeof(line), maps)) fputs(line, out);
    if (maps) fclose(maps);
#endif
    fclose(out);
}

void runtime_sampler_start(void) {
    const char *dir = getenv("TLG_SESSION_DIR");
    const char *rate = getenv("TLG_GUEST_SAMPLE_HZ");
    const long hz = rate && *rate ? strtol(rate, NULL, 10) : 200;
    if (!dir || !*dir || hz <= 0) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/guest_samples.bin", dir);
    static Sampler s;
    s.file = fopen(path, "wb");
    if (!s.file) { printf("Profile: cannot write %s\n", path); return; }
    setvbuf(s.file, NULL, _IOFBF, 1 << 20);
    s.hz = (unsigned)(hz > 2000 ? 2000 : hz);
    uintptr_t base; uint64_t size;
    runtime_freeze_image(&base, &size);
    write_modules(dir);
    put(s.file, "TLGSMPL2", 8); put_u64(s.file, base); put_u64(s.file, size); put_u32(s.file, s.hz); put_u32(s.file, 0);
    pthread_t thread;
    if (pthread_create(&thread, NULL, sampler, &s)) { puts("Profile: guest sampler not started"); return; }
    pthread_detach(thread);
    printf("Profile: guest code sampled %u times a second to %s\n", s.hz, path);
}
