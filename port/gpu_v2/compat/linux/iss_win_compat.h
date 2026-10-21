// SPDX-License-Identifier: GPL-2.0-or-later
// Linux: the subset of the Win32 API renderer v2 uses (pc-port/gpu), over POSIX.
// Force-included on Linux; <windows.h> and <tlhelp32.h> resolve to this header.
#pragma once
#ifndef _WIN32

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

using DWORD = std::uint32_t;
using BOOL = int;
using BYTE = std::uint8_t;
using WORD = std::uint16_t;
using UINT = unsigned int;
using LONG = long;
using LONGLONG = long long;
using ULONG_PTR = std::uintptr_t;
using SIZE_T = std::size_t;
using HANDLE = void*;
using HMODULE = void*;
using LPCSTR = const char*;
using LPSTR = char*;
using LPVOID = void*;
using LPCVOID = const void*;
using PDWORD = DWORD*;
using ULONG_PTR = std::uintptr_t;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#ifndef MAX_PATH
#define MAX_PATH 260
#endif
#define WINAPI

#define PAGE_NOACCESS 0x01
#define PAGE_READONLY 0x02
#define PAGE_READWRITE 0x04
#define PAGE_EXECUTE 0x10
#define PAGE_EXECUTE_READ 0x20
#define PAGE_EXECUTE_READWRITE 0x40
#define MEM_COMMIT 0x1000
#define MEM_RESERVE 0x2000
#define MEM_RELEASE 0x8000
#define GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT 0x2
#define GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS 0x4

typedef union _LARGE_INTEGER {
    struct {
        DWORD LowPart;
        LONG HighPart;
    };
    LONGLONG QuadPart;
} LARGE_INTEGER;

typedef struct _FILETIME {
    DWORD dwLowDateTime;
    DWORD dwHighDateTime;
} FILETIME;

namespace IssWinCompat {
inline int ToProt(DWORD protect) {
    switch (protect & 0xff) {
    case PAGE_NOACCESS:
        return PROT_NONE;
    case PAGE_READONLY:
        return PROT_READ;
    case PAGE_READWRITE:
        return PROT_READ | PROT_WRITE;
    case PAGE_EXECUTE:
        return PROT_EXEC;
    case PAGE_EXECUTE_READ:
        return PROT_READ | PROT_EXEC;
    default:
        return PROT_READ | PROT_WRITE | PROT_EXEC;
    }
}
} // namespace IssWinCompat

// The renderer patches the game's code: write access, then back to read/execute.
inline BOOL VirtualProtect(LPVOID address, SIZE_T size, DWORD protect, PDWORD old) {
    const std::uintptr_t page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(address) & ~(page - 1);
    const std::uintptr_t end = (reinterpret_cast<std::uintptr_t>(address) + size + page - 1) & ~(page - 1);
    if (old) {
        *old = PAGE_EXECUTE_READ; // the code pages these calls restore
    }
    return mprotect(reinterpret_cast<void*>(start), end - start, IssWinCompat::ToProt(protect)) == 0;
}
inline LPVOID VirtualAlloc(LPVOID address, SIZE_T size, DWORD, DWORD protect) {
    void* p = mmap(address, size, IssWinCompat::ToProt(protect), MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline BOOL VirtualFree(LPVOID address, SIZE_T size, DWORD) {
    return munmap(address, size ? size : static_cast<SIZE_T>(sysconf(_SC_PAGESIZE))) == 0;
}
inline BOOL FlushInstructionCache(HANDLE, LPCVOID, SIZE_T) {
    return TRUE; // x86: coherent
}
inline HANDLE GetCurrentProcess() {
    return reinterpret_cast<HANDLE>(-1);
}
inline HANDLE GetCurrentThread() {
    return reinterpret_cast<HANDLE>(-2);
}
inline DWORD GetCurrentThreadId() {
    return static_cast<DWORD>(syscall(SYS_gettid));
}
inline DWORD GetCurrentProcessId() {
    return static_cast<DWORD>(getpid());
}
inline BOOL CloseHandle(HANDLE) {
    return TRUE;
}
inline void Sleep(DWORD ms) {
    usleep(static_cast<useconds_t>(ms) * 1000);
}
inline BOOL SwitchToThread() {
    return sched_yield() == 0;
}
inline BOOL QueryPerformanceCounter(LARGE_INTEGER* value) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    value->QuadPart = static_cast<LONGLONG>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    return TRUE;
}
inline BOOL QueryPerformanceFrequency(LARGE_INTEGER* value) {
    value->QuadPart = 1000000000LL;
    return TRUE;
}
// Process CPU times in 100 ns units, as Windows reports them.
inline BOOL GetProcessTimes(HANDLE, FILETIME* creation, FILETIME* exit_time, FILETIME* kernel,
                            FILETIME* user) {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const auto put = [](FILETIME* t, std::uint64_t v) {
        if (t) {
            t->dwLowDateTime = static_cast<DWORD>(v);
            t->dwHighDateTime = static_cast<DWORD>(v >> 32);
        }
    };
    put(creation, 0);
    put(exit_time, 0);
    put(kernel, (usage.ru_stime.tv_sec * 1000000ULL + usage.ru_stime.tv_usec) * 10);
    put(user, (usage.ru_utime.tv_sec * 1000000ULL + usage.ru_utime.tv_usec) * 10);
    return TRUE;
}
inline void GetCurrentThreadStackLimits(ULONG_PTR* low, ULONG_PTR* high) {
    pthread_attr_t attr;
    void* base = nullptr;
    size_t size = 0;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
        pthread_attr_getstack(&attr, &base, &size);
        pthread_attr_destroy(&attr);
    }
    *low = reinterpret_cast<ULONG_PTR>(base);
    *high = reinterpret_cast<ULONG_PTR>(base) + size;
}
// Module of an address (diagnostics: the name of the library a pointer belongs to).
inline BOOL GetModuleHandleExA(DWORD, LPCSTR address, HMODULE* module) {
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_fbase) {
        return FALSE;
    }
    *module = info.dli_fbase;
    return TRUE;
}
inline DWORD GetModuleFileNameA(HMODULE module, LPSTR name, DWORD size) {
    Dl_info info{};
    if (!size || !module || !dladdr(module, &info) || !info.dli_fname) {
        if (size) {
            name[0] = 0;
        }
        return 0;
    }
    std::snprintf(name, size, "%s", info.dli_fname);
    return static_cast<DWORD>(std::strlen(name));
}
inline HMODULE LoadLibraryW(const wchar_t*) {
    return nullptr; // winmm, XInput: Windows only
}
inline void* GetProcAddress(HMODULE, LPCSTR) {
    return nullptr;
}
inline void* _aligned_malloc(size_t size, size_t alignment) {
    void* p = nullptr;
    return posix_memalign(&p, alignment < sizeof(void*) ? sizeof(void*) : alignment, size) == 0 ? p
                                                                                                 : nullptr;
}
inline void _aligned_free(void* p) {
    std::free(p);
}
inline int localtime_s(std::tm* out, const std::time_t* time) {
    return localtime_r(time, out) ? 0 : errno;
}
#define _ftelli64 ftello
#define _fseeki64 fseeko
#define _ReturnAddress() __builtin_return_address(0)
#define _AddressOfReturnAddress() \
    (static_cast<void*>(static_cast<char*>(__builtin_frame_address(0)) + sizeof(void*)))


using USHORT = unsigned short;
using ULONG = unsigned long;
using LPDWORD = DWORD*;
#define INVALID_HANDLE_VALUE (reinterpret_cast<HANDLE>(static_cast<std::intptr_t>(-1)))
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 0x1
#define CREATE_ALWAYS 2
#define OPEN_EXISTING 3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define FILE_FLAG_SEQUENTIAL_SCAN 0x08000000
#define MOVEFILE_REPLACE_EXISTING 0x1
#define MOVEFILE_WRITE_THROUGH 0x8
#define FILE_MAP_READ 0x4

inline DWORD GetLastError() {
    return static_cast<DWORD>(errno);
}
inline std::uint64_t GetTickCount64() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}
inline long SetThreadDescription(HANDLE, const wchar_t* name) {
    char narrow[16]{};
    for (int i = 0; i < 15 && name[i]; ++i) {
        narrow[i] = static_cast<char>(name[i] < 128 ? name[i] : '?');
    }
    pthread_setname_np(pthread_self(), narrow);
    return 0;
}
// 100 ns units since 1601-01-01, as Windows counts.
inline void GetSystemTimeAsFileTime(FILETIME* out) {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const std::uint64_t v = (static_cast<std::uint64_t>(ts.tv_sec) + 11644473600ULL) * 10000000ULL +
                            ts.tv_nsec / 100;
    out->dwLowDateTime = static_cast<DWORD>(v);
    out->dwHighDateTime = static_cast<DWORD>(v >> 32);
}

#include <execinfo.h>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <sys/stat.h>
inline USHORT RtlCaptureStackBackTrace(ULONG skip, ULONG count, void** frames, ULONG* hash) {
    void* all[128];
    const int n = backtrace(all, 128);
    USHORT out = 0;
    for (int i = static_cast<int>(skip) + 1; i < n && out < count; ++i) {
        frames[out++] = all[i];
    }
    if (hash) {
        *hash = 0;
    }
    return out;
}
using _invalid_parameter_handler = void (*)(const wchar_t*, const wchar_t*, const wchar_t*, unsigned,
                                            std::uintptr_t);
inline _invalid_parameter_handler _set_invalid_parameter_handler(_invalid_parameter_handler) {
    return nullptr; // no CRT invalid parameter checks here
}

// Files: a HANDLE holds fd + 1 (0 stays "no handle").
namespace IssWinCompat {
inline HANDLE FromFd(int fd) {
    return fd < 0 ? INVALID_HANDLE_VALUE : reinterpret_cast<HANDLE>(static_cast<std::intptr_t>(fd) + 1);
}
inline int ToFd(HANDLE h) {
    return static_cast<int>(reinterpret_cast<std::intptr_t>(h) - 1);
}
inline std::string Narrow(const wchar_t* path) {
    return std::filesystem::path(path).string();
}
struct Mapping {
    int fd;
    std::size_t size;
};
} // namespace IssWinCompat
inline HANDLE CreateFileW(const wchar_t* path, DWORD access, DWORD, void*, DWORD disposition, DWORD,
                          HANDLE) {
    int flags = (access & GENERIC_WRITE) ? ((access & GENERIC_READ) ? O_RDWR : O_WRONLY) : O_RDONLY;
    if (disposition == CREATE_ALWAYS) {
        flags |= O_CREAT | O_TRUNC;
    }
    return IssWinCompat::FromFd(open(IssWinCompat::Narrow(path).c_str(), flags | O_CLOEXEC, 0644));
}
inline BOOL WriteFile(HANDLE file, const void* data, DWORD size, LPDWORD written, void*) {
    const ssize_t n = write(IssWinCompat::ToFd(file), data, size);
    if (written) {
        *written = n < 0 ? 0 : static_cast<DWORD>(n);
    }
    return n == static_cast<ssize_t>(size);
}
inline BOOL FlushFileBuffers(HANDLE file) {
    return fsync(IssWinCompat::ToFd(file)) == 0;
}
inline BOOL MoveFileExW(const wchar_t* from, const wchar_t* to, DWORD) {
    return rename(IssWinCompat::Narrow(from).c_str(), IssWinCompat::Narrow(to).c_str()) == 0;
}
inline BOOL GetFileSizeEx(HANDLE file, LARGE_INTEGER* size) {
    struct stat st{};
    if (fstat(IssWinCompat::ToFd(file), &st) != 0) {
        return FALSE;
    }
    size->QuadPart = st.st_size;
    return TRUE;
}
inline HANDLE CreateFileMappingW(HANDLE file, void*, DWORD, DWORD, DWORD, const wchar_t*) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size)) {
        return nullptr;
    }
    return new IssWinCompat::Mapping{IssWinCompat::ToFd(file), static_cast<std::size_t>(size.QuadPart)};
}
inline void* MapViewOfFile(HANDLE mapping, DWORD, DWORD, DWORD, SIZE_T) {
    auto* m = static_cast<IssWinCompat::Mapping*>(mapping);
    void* p = mmap(nullptr, m->size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    return p == MAP_FAILED ? nullptr : p;
}
// The size is not known here: the views this code maps are released with the process.
inline BOOL UnmapViewOfFile(const void*) {
    return TRUE;
}
inline HANDLE CreateFileW(const char* path, DWORD access, DWORD share, void* security,
                          DWORD disposition, DWORD flags, HANDLE templ) {
    const std::wstring wide = std::filesystem::path(path).wstring();
    return CreateFileW(wide.c_str(), access, share, security, disposition, flags, templ);
}
inline BOOL MoveFileExW(const char* from, const char* to, DWORD flags) {
    return rename(from, to) == 0 || flags == ~0u;
}
#endif // !_WIN32
