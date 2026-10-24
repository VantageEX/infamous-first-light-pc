// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/range_lock.h"
#include "common/readback_offload.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <windows.h>
#include "common/path_util.h"
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

#ifdef _WIN64
// SHADPS4_READ_FAULT_RIP=1 (diagnostic, default off): one line per guest read fault with the
// faulting instruction, the eboot return-address candidates found on the guest stack (a scan, so
// stale slots appear too), the thread and the first 32 bytes around the address after the
// download. Joined offline by fault address with the CP's "ReadProducerFault" lines
// (SHADPS4_READ_PRODUCER_STATS=1). The decrypted eboot code segment is written once to
// log/eboot_code_800000000.bin for offline disassembly.
static bool ReadFaultRipEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADPS4_READ_FAULT_RIP");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

/// The faulting 4 KiB page as guest memory held it before the download (backing view, no
/// protection); has_pre false if the page lacks backing.
struct ReadFaultPrePage {
    std::array<u8, 4096> bytes;
    bool has_pre;
};
static thread_local ReadFaultPrePage t_read_fault_pre{};

static void SnapshotReadFaultPage(VAddr addr) {
    t_read_fault_pre.has_pre = Core::Memory::Instance()->TryReadBacking(
        addr & ~VAddr{4095}, t_read_fault_pre.bytes.data(), t_read_fault_pre.bytes.size());
}

static void LogReadFaultRip(void* context, VAddr addr) {
    constexpr u64 CodeBegin = 0x800000000ULL; // eboot segment 0 (R_X), REFERENCE.md
    constexpr u64 CodeEnd = 0x800cc8000ULL;
    constexpr u64 MaxLines = 200000;
    static std::atomic<u64> seq{0};
    const u64 n = seq.fetch_add(1, std::memory_order_relaxed);
    if (n >= MaxLines) {
        return;
    }
    static std::once_flag dump_once;
    std::call_once(dump_once, [] {
        const auto path =
            Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "eboot_code_800000000.bin";
        std::FILE* file = _wfopen(path.c_str(), L"wb");
        if (file != nullptr) {
            std::fwrite(reinterpret_cast<const void*>(CodeBegin), 1, CodeEnd - CodeBegin, file);
            std::fclose(file);
        }
        LOG_INFO(Render, "ReadFaultRip: eboot code dump {} ({})", path.string(),
                 file != nullptr ? "ok" : "failed");
    });
    const CONTEXT* ctx = static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord;
    std::array<u64, 8> frames{};
    size_t num_frames = 0;
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(ctx->Rsp), &mbi, sizeof(mbi)) != 0 &&
        mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
        const u64 region_end = reinterpret_cast<u64>(mbi.BaseAddress) + mbi.RegionSize;
        for (u64 slot = ctx->Rsp; slot + 8 <= region_end && slot < ctx->Rsp + 2048 * 8 &&
                                  num_frames < frames.size();
             slot += 8) {
            const u64 value = *reinterpret_cast<const u64*>(slot);
            if (value >= CodeBegin && value < CodeEnd) {
                frames[num_frames++] = value;
            }
        }
    }
    // Frame-pointer chain: guest functions that keep rbp (e.g. eboot+0x3fed00) put their return
    // address at [rbp+8]. Each step must stay inside the committed stack region found above.
    std::string rbp_chain;
    if (mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
        const u64 region_begin = reinterpret_cast<u64>(mbi.BaseAddress);
        const u64 region_end = region_begin + mbi.RegionSize;
        u64 frame = ctx->Rbp;
        for (int depth = 0; depth < 6; ++depth) {
            if (frame < ctx->Rsp || frame < region_begin || frame + 16 > region_end ||
                (frame & 7) != 0) {
                break;
            }
            const u64 ret = *reinterpret_cast<const u64*>(frame + 8);
            rbp_chain += fmt::format(" {:#x}", ret);
            const u64 next = *reinterpret_cast<const u64*>(frame);
            if (next <= frame) {
                break;
            }
            frame = next;
        }
    }
    thread_local std::string thread_name = [] {
        std::string name;
        PWSTR desc = nullptr;
        if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &desc)) && desc != nullptr) {
            for (const wchar_t* c = desc; *c != 0; ++c) {
                name.push_back(*c < 128 && *c > 32 ? static_cast<char>(*c) : '_');
            }
            LocalFree(desc);
        }
        return name.empty() ? std::string("?") : name;
    }();
    std::string stack;
    for (size_t i = 0; i < num_frames; ++i) {
        stack += fmt::format(" {:#x}", frames[i]);
    }
    std::string bytes = "unreadable";
    const auto* data = reinterpret_cast<const u8*>(addr & ~VAddr{31});
    MEMORY_BASIC_INFORMATION data_mbi{};
    if (VirtualQuery(data, &data_mbi, sizeof(data_mbi)) != 0 && data_mbi.State == MEM_COMMIT &&
        (data_mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ |
                             PAGE_EXECUTE_READWRITE)) != 0 &&
        (data_mbi.Protect & PAGE_GUARD) == 0) {
        bytes.clear();
        for (size_t i = 0; i < 32; ++i) {
            bytes += fmt::format("{:02x}", data[i]);
        }
    }
    // Bytes the download changed: in the page, and in the 96 bytes from the 32-byte block.
    s64 page_diff = -1;
    s64 near_diff = -1;
    u32 dword_pre = 0;
    u32 dword_post = 0;
    if (t_read_fault_pre.has_pre) {
        std::array<u8, 4096> post{};
        if (Core::Memory::Instance()->TryReadBacking(addr & ~VAddr{4095}, post.data(),
                                                     post.size())) {
            page_diff = 0;
            near_diff = 0;
            const size_t near_begin = (addr & 4095) & ~size_t{31};
            const size_t at = std::min<size_t>(addr & 4095, 4092);
            std::memcpy(&dword_pre, t_read_fault_pre.bytes.data() + at, 4);
            std::memcpy(&dword_post, post.data() + at, 4);
            for (size_t i = 0; i < post.size(); ++i) {
                const bool diff = post[i] != t_read_fault_pre.bytes[i];
                page_diff += diff;
                near_diff += diff && i >= near_begin && i < near_begin + 96;
            }
        }
    }
    const u64 exe_base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
    LOG_INFO(Render,
             "ReadFaultRip: n={} addr={:#x} rip={:#x} exe_off={:#x} thread={} tid={} rax={:#x} "
             "rbx={:#x} rcx={:#x} rdx={:#x} rsi={:#x} rdi={:#x} rbp={:#x} r8={:#x} r9={:#x} "
             "r10={:#x} r11={:#x} r12={:#x} r13={:#x} r14={:#x} r15={:#x} rsp={:#x} stack:{} rbpchain:{} "
             "data32@{:#x}={} page_diff={} near96_diff={} dword_pre={:#x} dword_post={:#x}",
             n, addr, ctx->Rip, ctx->Rip - exe_base, thread_name, GetCurrentThreadId(), ctx->Rax,
             ctx->Rbx, ctx->Rcx, ctx->Rdx, ctx->Rsi, ctx->Rdi, ctx->Rbp, ctx->R8, ctx->R9, ctx->R10,
             ctx->R11, ctx->R12, ctx->R13, ctx->R14, ctx->R15, ctx->Rsp, stack, rbp_chain,
             addr & ~VAddr{31},
             bytes, page_diff, near_diff, dword_pre, dword_post);
}
#endif

/// See PageManager::SetDeferredProtectSink. Only write-protect (track, !is_read) requests go
/// here; unprotects and read protection always reach the OS immediately.
static thread_local std::vector<PageManager::ProtectRange>* t_protect_sink = nullptr;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void OnMap(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void OnMap(VAddr address, size_t size) {
        // No-op (the protection shadow follows AddressSpace::RemapEpoch).
    }

    void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto start = Common::FaultMonitor::Now();
        if (Common::IsWriteError(context)) {
            // Batch-mix diagnostic: the write-fault RIP too (low bit set marks a write).
            if (Common::ReadbackOffload::LatencyEnabled()) [[unlikely]] {
                Common::ReadbackOffload::t_read_fault_rip =
                    static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord->Rip | 1ULL << 63;
            }
            // A fault taken by the CP itself is part of the command stream it is executing, so
            // only guest threads' faults may be deferred (submit_epoch.h).
            const bool census = Common::FaultMonitor::GameMainCensus();
            const u64 c0 = census ? Common::FaultMonitor::CensusTicks() : 0;
            Common::FaultMonitor::t_fault_class = 0;
            const bool handled = rasterizer->InvalidateMemory(
                addr, 8,
                Common::FaultMonitor::t_is_cp_thread || Common::FaultMonitor::t_acting_cp
                    ? VideoCore::InvalidateOrigin::CpWriteFault
                    : VideoCore::InvalidateOrigin::GuestWriteFault);
            if (census) {
                Common::FaultMonitor::NoteWriteFaultCensus(
                    addr, Common::FaultMonitor::CensusTicks() - c0,
                    Common::FaultMonitor::t_fault_class);
            }
            Common::ReadbackOffload::t_read_fault_rip = 0;
            Common::FaultMonitor::CountGuestWriteFault(handled,
                                                       Common::FaultMonitor::SinceNs(start));
            if (handled && Common::FaultMonitor::WriteFaultPagesEnabled()) {
                Common::FaultMonitor::RecordWriteFaultPage(addr);
            }
            return handled;
        } else {
            // Read faults only exist under Precise readbacks - it is the only mode that
            // read-protects guest pages holding GPU-modified data.
            if (ReadFaultRipEnabled()) [[unlikely]] {
                SnapshotReadFaultPage(addr);
            }
            if (Common::ReadbackOffload::LatencyEnabled()) [[unlikely]] {
                Common::ReadbackOffload::t_read_fault_rip =
                    static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord->Rip;
            }
            const bool handled = rasterizer->ReadMemory(addr, 8);
            Common::ReadbackOffload::t_read_fault_rip = 0;
            Common::FaultMonitor::CountGuestReadFault(handled, Common::FaultMonitor::SinceNs(start),
                                                      addr);
            if (handled && ReadFaultRipEnabled()) [[unlikely]] {
                LogReadFaultRip(context, addr);
            }
            return handled;
        }
    }
#endif

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // Acquire locks for the range of pages
        auto lk = LockPages(page, page_end);
        OsPending os_pending;

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                if constexpr (track && !is_read) {
                    if (t_protect_sink) {
                        t_protect_sink->push_back({range_begin << PM_PAGE_BITS, range_bytes});
                        range_bytes = 0;
                        potential_range_bytes = 0;
                        return;
                    }
                }
                if (os_lock_mode) {
                    os_pending.push_back({range_begin, range_begin + (range_bytes >> PM_PAGE_BITS)});
                } else {
                    Protect(range_begin << PM_PAGE_BITS, range_bytes, perms);
                }
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        // SHADPS4_WATCH_MAPCHECK=0: skip this log-only check (a locked hash lookup per call; the
        // warning was never seen in the forest logs). SHADPS4_LEAN=1 implies it.
        static const bool map_check = [] {
            const char* value = std::getenv("SHADPS4_WATCH_MAPCHECK");
            return !::Common::Lean() && (!value || value[0] != '0');
        }();
        if (map_check && !rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
        ApplyOsPending(lk, os_pending);
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        // The stripes the mask's page span touches; without SHADPS4_PAGE_LOCK_PAGES that is the
        // region's one lock, as before.
        auto lk = LockPages(base_page + start_range.first, base_page + end_range.second);
        OsPending os_pending;
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                if constexpr (track && !is_read) {
                    if (t_protect_sink) {
                        t_protect_sink->push_back({range_begin << PM_PAGE_BITS, range_bytes});
                        range_bytes = 0;
                        potential_range_bytes = 0;
                        return;
                    }
                }
                if (os_lock_mode) {
                    os_pending.push_back({range_begin, range_begin + (range_bytes >> PM_PAGE_BITS)});
                } else {
                    Protect((range_begin << PM_PAGE_BITS), range_bytes, perms);
                }
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
        ApplyOsPending(lk, os_pending);
    }

    /// Applies the current watcher state of every page in the range to the OS protection.
    void ReconcileProtection(VAddr addr, u64 size) {
        RENDERER_TRACE;
        const size_t first = addr >> PM_PAGE_BITS;
        const size_t end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        if (end <= first) {
            return;
        }
        if (os_lock_mode) {
            ReconcileOs(first, end, 1);
            return;
        }
        auto lk = LockPages(first, end);
        const auto perms_of = [&](size_t page) {
            const auto perms = cached_pages[page].Perms();
            // Write-only is not expressible on the CPU; the stricter state is always safe.
            return perms == Core::MemoryPermission::Write ? Core::MemoryPermission::None : perms;
        };
        size_t run = first;
        auto perms = perms_of(first);
        for (size_t page = first + 1; page < end; ++page) {
            const auto next = perms_of(page);
            if (next != perms) {
                Protect(run << PM_PAGE_BITS, (page - run) << PM_PAGE_BITS, perms);
                run = page;
                perms = next;
            }
        }
        Protect(run << PM_PAGE_BITS, (end - run) << PM_PAGE_BITS, perms);
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};

    /**
     * SHADPS4_PAGE_OS_LOCK=1 (design B of notes/page-lock-hold-design-2026-09-28.md): the page
     * locks (`locks`, the "count locks") are never held across a protection syscall. Watcher
     * counts still change only under them; every syscall is made by ReconcileOs under a second
     * stripe array `os_locks` (one per 4 MB region), which snapshots Perms(counts) under the
     * count locks after taking the OS stripe, releases the count locks, then calls the OS.
     * Why the OS state stays right: syscalls on a stripe are serialized by its OS lock and each
     * applies counts read after that lock was taken, so a stale snapshot can never land after a
     * newer one; every count change is followed by its own ReconcileOs on the same thread before
     * the caller returns (or, for deferred write protects, by the recorder's ReconcileProtection
     * before its copy), so the last reconcile of a page always reads its latest counts.
     * Lock order: region lock -> OS stripes -> count stripes; count holders never take an OS
     * stripe. Measured 2026-09-28 (a33 fire, back stacks): the CP spun 4.3% of its wall time on
     * count locks held across VirtualProtect by fault handlers, the recorder and the Protector
     * (7.1% with SHADPS4_FAULT_WATCH_OUTSIDE=1, which moves the region-lock wait onto them).
     * SHADPS4_PAGE_OS_VERIFY=1 (diagnostic): after each syscall, still under the OS stripe,
     * VirtualQuery the run's first and last page and count mismatches with the snapshot. False
     * positives: guest mprotect (MemoryManager::Protect), SHADPS4_TIMESTAMP_READER (do not combine),
     * pages AddressSpace skips as unmapped that VirtualQuery reports committed. Costly: fire ran
     * at ~18 FPS with it (VirtualQuery against concurrent VirtualProtect; measured 2026-09-28).
     */
    static bool EnvIs(const char* name, char value) {
        const char* env = std::getenv(name);
        return env && env[0] == value;
    }
    const bool os_lock_mode = EnvIs("SHADPS4_PAGE_OS_LOCK", '1');
    const bool os_verify = os_lock_mode && EnvIs("SHADPS4_PAGE_OS_VERIFY", '1');
    const std::unique_ptr<LockType[]> os_locks =
        os_lock_mode ? std::make_unique<LockType[]>(NUM_ADDRESS_LOCKS) : nullptr;
    std::atomic<u64> os_reconciles{0};
    std::atomic<u64> os_syscalls{0};
    std::atomic<u64> os_verified{0};
    std::atomic<u64> os_mismatches{0};

    /// Page ranges [first, end) whose protection a watcher update must reconcile.
    using OsPending = boost::container::small_vector<std::pair<size_t, size_t>, 4>;

    static Core::MemoryPermission ExpressiblePerms(Core::MemoryPermission perms) {
        // Write-only is not expressible on the CPU; the stricter state is always safe.
        return perms == Core::MemoryPermission::Write ? Core::MemoryPermission::None : perms;
    }

    /// Releases the count locks, then reconciles what the update left pending.
    template <typename Guard>
    void ApplyOsPending(Guard& lk, const OsPending& pending) {
        if (pending.empty()) {
            return;
        }
        lk.unlock();
        // One reconcile over the whole span (ranges are pushed in ascending order): gap pages
        // get Perms(counts) as well, which is their state already.
        ReconcileOs(pending.front().first, pending.back().second, 0);
    }

    /**
     * SHADPS4_PAGE_OS_SHADOW (needs PAGE_OS_LOCK): a byte per page remembering the protection
     * last applied to it (0 = unknown), read and written only under the page's OS stripe by
     * ReconcileOs. Invalidation: AddressSpace bumps RemapEpoch around every Map / MapFile / Unmap
     * (a partial unmap re-maps neighbouring pages read-write, SplitRegion) and ExternalProtect
     * (MemoryManager's own protects); ValidateStripes clears a stripe's shadow whenever the epoch
     * moved since the stripe was last validated or a change is in progress (Opus review
     * 2026-09-28: without it a skip could leave a re-mapped page writable).
     * =1 counts, per caller (0 = watcher updates: fault handlers, CP, Protector; 1 = the
     * recorder's deferred protects), the runs whose syscall would change nothing. =2 also skips
     * those syscalls. Measured 2026-09-28 (a35, =1): 67-68% of the recorder's runs unchanged
     * (fire and forest), 0.2% of watcher runs. 2^28 bytes committed; only touched pages use RAM.
     * Soundness (inferred): a skip decided before a remap begins is equivalent to a syscall made
     * just before it (today's behaviour); any shadow written before a remap ends is discarded at
     * the next ValidateStripes. Other bypasses (TIMESTAMP_READER) are diagnostic only.
     */
    const bool os_shadow_mode =
        os_lock_mode && (EnvIs("SHADPS4_PAGE_OS_SHADOW", '1') || EnvIs("SHADPS4_PAGE_OS_SHADOW", '2'));
    const bool os_shadow_skip = os_lock_mode && EnvIs("SHADPS4_PAGE_OS_SHADOW", '2');
    u8* const os_shadow = [this]() -> u8* {
#ifdef _WIN32
        if (os_shadow_mode) {
            return static_cast<u8*>(VirtualAlloc(nullptr, NUM_ADDRESS_PAGES,
                                                 MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        }
#endif
        return nullptr;
    }();
    struct ShadowStats {
        std::atomic<u64> runs{0}, pages{0}, same_runs{0}, same_pages{0}, calls{0};
    };
    std::array<ShadowStats, 2> shadow_stats{};
    /// RemapEpoch value each OS stripe's shadow was validated at (~0 = never trust); only
    /// touched under that stripe.
    const std::unique_ptr<u64[]> os_stripe_epoch =
        os_shadow_mode ? std::make_unique<u64[]>(NUM_ADDRESS_LOCKS) : nullptr;
    std::atomic<u64> shadow_invalidations{0};

    /// Under the OS stripes of [first, end): drop the shadow of every stripe whose validation is
    /// older than the address space's last remap, or while a remap is in progress.
    void ValidateStripes(size_t first, size_t end) {
        auto& address_space = Core::Memory::Instance()->GetAddressSpace();
        // Active first, then the epoch (pairs with EndRemap's order).
        const bool active = address_space.RemapsActive() != 0;
        const u64 epoch = address_space.RemapEpoch();
        const size_t s_end = Common::DivCeil(end, PAGES_PER_LOCK);
        for (size_t stripe = first / PAGES_PER_LOCK; stripe < s_end; ++stripe) {
            if (!active && os_stripe_epoch[stripe] == epoch) {
                continue;
            }
            std::memset(os_shadow + stripe * PAGES_PER_LOCK, 0, PAGES_PER_LOCK);
            os_stripe_epoch[stripe] = active ? ~0ULL : epoch;
            shadow_invalidations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// A protection change made outside the page manager (MemoryManager): bracketed by the remap
    /// epoch so ReconcileOs never trusts a shadow this call may have overwritten.
    void ExternalProtect(VAddr address, u64 size, Core::MemoryPermission perms) {
        auto& address_space = Core::Memory::Instance()->GetAddressSpace();
        if (!os_shadow) {
            address_space.Protect(address, size, perms);
            return;
        }
        address_space.BeginRemap();
        address_space.Protect(address, size, perms);
        address_space.EndRemap();
    }

    /// Under the OS stripe. True when every page of the run already has `perms` applied.
    bool ShadowSame(size_t first, size_t end, Core::MemoryPermission perms) const {
        const u8 value = static_cast<u8>(static_cast<u32>(perms) + 1);
        for (size_t page = first; page < end; ++page) {
            if (os_shadow[page] != value) {
                return false;
            }
        }
        return true;
    }

    void CountShadow(int caller, size_t first, size_t end, Core::MemoryPermission perms,
                     bool same) {
        const u8 value = static_cast<u8>(static_cast<u32>(perms) + 1);
        std::memset(os_shadow + first, value, end - first);
        auto& st = shadow_stats[caller];
        st.runs.fetch_add(1, std::memory_order_relaxed);
        st.pages.fetch_add(end - first, std::memory_order_relaxed);
        if (same) {
            st.same_runs.fetch_add(1, std::memory_order_relaxed);
            st.same_pages.fetch_add(end - first, std::memory_order_relaxed);
        }
    }

    void ReconcileOs(size_t first, size_t end, int caller) {
        RENDERER_TRACE;
        Common::RangeLockGuard<LockType*> os_lk(os_locks.get() + first / PAGES_PER_LOCK,
                                                os_locks.get() +
                                                    Common::DivCeil(end, PAGES_PER_LOCK));
        if (os_shadow) [[unlikely]] {
            ValidateStripes(first, end);
        }
        // Run starts and their perms; a run ends where the next starts (or at `end`).
        boost::container::small_vector<std::pair<size_t, Core::MemoryPermission>, 8> runs;
        {
            auto lk = LockPages(first, end);
            auto perms = ExpressiblePerms(cached_pages[first].Perms());
            runs.push_back({first, perms});
            for (size_t page = first + 1; page < end; ++page) {
                const auto next = ExpressiblePerms(cached_pages[page].Perms());
                if (next != perms) {
                    runs.push_back({page, next});
                    perms = next;
                }
            }
        }
        for (size_t i = 0; i < runs.size(); ++i) {
            const size_t run_end = i + 1 < runs.size() ? runs[i + 1].first : end;
            // Still under the OS stripe: the shadow changes only under it (see os_shadow_mode).
            const bool same = os_shadow && ShadowSame(runs[i].first, run_end, runs[i].second);
            if (!(same && os_shadow_skip)) {
                Protect(runs[i].first << PM_PAGE_BITS, (run_end - runs[i].first) << PM_PAGE_BITS,
                        runs[i].second);
            }
            if (os_shadow) [[unlikely]] {
                CountShadow(caller, runs[i].first, run_end, runs[i].second, same);
            }
#ifdef _WIN32
            if (os_verify) [[unlikely]] {
                VerifyPage(runs[i].first, runs[i].second);
                VerifyPage(run_end - 1, runs[i].second);
            }
#endif
        }
        os_lk.unlock();
        if (os_shadow) [[unlikely]] {
            const u64 calls =
                shadow_stats[caller].calls.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((calls & ((1ULL << 20) - 1)) == 0) {
                const auto& st = shadow_stats[caller];
                const u64 r = st.runs.load(std::memory_order_relaxed);
                const u64 pg = st.pages.load(std::memory_order_relaxed);
                const u64 sr = st.same_runs.load(std::memory_order_relaxed);
                const u64 sp = st.same_pages.load(std::memory_order_relaxed);
                LOG_INFO(Render,
                         "PageOsShadow: caller={} calls={} runs={} pages={} unchanged_runs={} "
                         "({:.1f}%) unchanged_pages={} ({:.1f}%) pages/run={:.1f} skip={} "
                         "stripe_invalidations={}",
                         caller == 0 ? "watch" : "recorder", calls, r, pg, sr,
                         100.0 * double(sr) / double(std::max<u64>(r, 1)), sp,
                         100.0 * double(sp) / double(std::max<u64>(pg, 1)),
                         double(pg) / double(std::max<u64>(r, 1)), os_shadow_skip,
                         shadow_invalidations.load(std::memory_order_relaxed));
            }
        }
        os_syscalls.fetch_add(runs.size(), std::memory_order_relaxed);
        const u64 n = os_reconciles.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & ((1ULL << 22) - 1)) == 0) {
            LOG_INFO(Render,
                     "PageOsLock: reconciles={} syscalls={} verified_pages={} mismatches={}", n,
                     os_syscalls.load(std::memory_order_relaxed),
                     os_verified.load(std::memory_order_relaxed),
                     os_mismatches.load(std::memory_order_relaxed));
        }
    }

#ifdef _WIN32
    /// SHADPS4_PAGE_OS_VERIFY: the OS protection of `page` must equal `perms` on the read/write
    /// axis (guard and execute bits ignored; pages the address space does not commit skipped).
    void VerifyPage(size_t page, Core::MemoryPermission perms) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<LPCVOID>(page << PM_PAGE_BITS), &info, sizeof(info)) ==
                0 ||
            info.State != MEM_COMMIT) {
            return;
        }
        const DWORD raw = info.Protect & 0xff;
        const bool read = raw == PAGE_READONLY || raw == PAGE_READWRITE ||
                          raw == PAGE_EXECUTE_READ || raw == PAGE_EXECUTE_READWRITE ||
                          raw == PAGE_WRITECOPY || raw == PAGE_EXECUTE_WRITECOPY;
        const bool write = raw == PAGE_READWRITE || raw == PAGE_EXECUTE_READWRITE ||
                           raw == PAGE_WRITECOPY || raw == PAGE_EXECUTE_WRITECOPY;
        const bool want_read = True(perms & Core::MemoryPermission::Read);
        const bool want_write = True(perms & Core::MemoryPermission::Write);
        os_verified.fetch_add(1, std::memory_order_relaxed);
        if (read != want_read || write != want_write) {
            const u64 bad = os_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
            if (bad <= 8) {
                LOG_CRITICAL(Render,
                             "PageOsVerify: page {:#x} OS protect {:#x} (read={} write={}) != "
                             "snapshot read={} write={}",
                             page << PM_PAGE_BITS, info.Protect, read, write, want_read,
                             want_write);
            }
        }
    }
#endif

    /**
     * SHADPS4_PAGE_LOCK_PAGES=<n> (power of two, 16..NUM_PAGES_PER_REGION): lock stripes of n pages
     * instead of `locks`' one per 4 MB region. Measured 2026-09-25 (a19): the CP spun 0.86% of its
     * time in LockRange while re-protecting CPU-dirty pages, behind fault handlers, the Protector
     * and ReconcileProtection holding the same region. Every page's counts and protection still
     * change only under the stripe that covers it. Costs NUM_ADDRESS_PAGES / n extra locks
     * (n=64: 2^22 x 4 B = 16 MB on Windows), allocated only when the flag is set.
     *
     * Fine stripes are taken in ascending order, iteratively: every holder takes its whole span in
     * one LockPages call and takes no other stripe while holding it, so the order alone rules out
     * deadlock. (LockRange/TryLockRange recurse once per lock, and an image span can cover
     * hundreds of fine stripes - on a guest thread's stack when a fault handler untracks it.)
     *
     * SHADPS4_PAGE_LOCK_STATS=1: on the CP thread, try the whole span first and count the
     * acquisitions where that failed (the ones that would spin), with the time spent getting them.
     */
    static u32 FineLockBits() {
        const char* value = std::getenv("SHADPS4_PAGE_LOCK_PAGES");
        if (!value) {
            return 0;
        }
        const u64 pages = std::strtoull(value, nullptr, 10);
        if (pages < 16 || pages > PAGES_PER_LOCK || !std::has_single_bit(pages)) {
            LOG_WARNING(Render, "SHADPS4_PAGE_LOCK_PAGES={} ignored (power of two, 16-{})", value,
                        PAGES_PER_LOCK);
            return 0;
        }
        if (pages == PAGES_PER_LOCK) {
            return 0;
        }
        LOG_INFO(Render, "PageLock: {} pages per lock stripe ({} locks)", pages,
                 NUM_ADDRESS_PAGES / pages);
        return static_cast<u32>(std::countr_zero(pages));
    }

    static bool LockStatsEnabled() {
        const char* value = std::getenv("SHADPS4_PAGE_LOCK_STATS");
        return value && value[0] == '1';
    }

    using LockGuard = Common::RangeLockGuard<LockType*>;

    /// Locks every stripe covering pages [first, end).
    LockGuard LockPages(size_t first, size_t end) {
        if (lock_slow_path) [[unlikely]] {
            return LockPagesSlow(first, end);
        }
        return LockGuard(locks.data() + first / PAGES_PER_LOCK,
                         locks.data() + Common::DivCeil(end, PAGES_PER_LOCK));
    }

    static bool TryLockOrdered(LockType* begin, LockType* end) {
        for (LockType* it = begin; it != end; ++it) {
            if (!it->try_lock()) {
                while (it != begin) {
                    (--it)->unlock();
                }
                return false;
            }
        }
        return true;
    }

    static void LockOrdered(LockType* begin, LockType* end) {
        for (; begin != end; ++begin) {
            begin->lock();
        }
    }

    LockGuard LockPagesSlow(size_t first, size_t end) {
        const bool fine = fine_locks != nullptr;
        LockType* begin = locks.data() + first / PAGES_PER_LOCK;
        LockType* last = locks.data() + Common::DivCeil(end, PAGES_PER_LOCK);
        if (fine) {
            begin = fine_locks.get() + (first >> fine_lock_bits);
            last = fine_locks.get() + ((end + (1ULL << fine_lock_bits) - 1) >> fine_lock_bits);
        }
        const bool count = lock_stats && Common::FaultMonitor::t_is_cp_thread;
        if (count) {
            // Both tries either take the whole span or release what they took.
            if (fine ? TryLockOrdered(begin, last) : Common::TryLockRange(begin, last) == last) {
                CountCpLock(false, 0);
                return LockGuard(begin, last, std::adopt_lock);
            }
        }
        const auto start =
            count ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (fine) {
            LockOrdered(begin, last);
        } else {
            Common::LockRange(begin, last);
        }
        if (count) {
            CountCpLock(true, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - start)
                                                   .count()));
        }
        return LockGuard(begin, last, std::adopt_lock);
    }

    /// CP thread only (t_is_cp_thread), so plain counters. Logged every 512 presented frames.
    void CountCpLock(bool spun, u64 spin_ns) {
        ++cp_locks;
        cp_spins += spun;
        cp_spin_ns += spin_ns;
        if ((cp_locks & 0xfff) != 0) {
            return;
        }
        const u64 frames = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        if (cp_stats_frames == 0 || frames < cp_stats_frames + 512) {
            if (cp_stats_frames == 0 && frames != 0) {
                // Start the first window at the first presented frame, not at boot.
                cp_stats_frames = frames;
                cp_locks = cp_spins = cp_spin_ns = 0;
            }
            return;
        }
        const double n = static_cast<double>(frames - cp_stats_frames);
        LOG_INFO(Render,
                 "PageLockSpin: stripe_pages={} frames={} cp_locks/frame={:.1f} "
                 "cp_spins/frame={:.2f} ({:.2f}% of locks) cp_spin_us/frame={:.1f}",
                 fine_locks ? (1ULL << fine_lock_bits) : PAGES_PER_LOCK, frames - cp_stats_frames,
                 cp_locks / n, cp_spins / n, 100.0 * cp_spins / cp_locks, cp_spin_ns / n / 1000.0);
        cp_stats_frames = frames;
        cp_locks = cp_spins = cp_spin_ns = 0;
    }

    const u32 fine_lock_bits = FineLockBits();
    const std::unique_ptr<LockType[]> fine_locks =
        fine_lock_bits != 0 ? std::make_unique<LockType[]>(NUM_ADDRESS_PAGES >> fine_lock_bits)
                            : nullptr;
    const bool lock_stats = LockStatsEnabled();
    // The one flag the default path reads: either option routes locking through LockPagesSlow.
    const bool lock_slow_path = fine_lock_bits != 0 || lock_stats;
    u64 cp_locks = 0;
    u64 cp_spins = 0;
    u64 cp_spin_ns = 0;
    u64 cp_stats_frames = 0;
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

void PageManager::SetDeferredProtectSink(std::vector<ProtectRange>* sink) {
    t_protect_sink = sink;
}

void PageManager::ReconcileProtection(std::span<const ProtectRange> ranges) const {
    for (const ProtectRange& range : ranges) {
        impl->ReconcileProtection(range.addr, range.size);
    }
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::ExternalProtect(VAddr address, u64 size, Core::MemoryPermission perms) const {
#ifdef ENABLE_USERFAULTFD
    Core::Memory::Instance()->GetAddressSpace().Protect(address, size, perms);
#else
    impl->ExternalProtect(address, size, perms);
#endif
}

bool PageManager::AnyReadProtected(VAddr addr, u64 size) const {
    if (size == 0) {
        return false;
    }
    const u64 end = addr + size;
    if (end < addr || end > (u64{1} << Impl::ADDRESS_BITS)) {
        return true; // outside the tracked space: treat as not safely readable
    }
    for (u64 page = addr >> PM_PAGE_BITS, last = (end - 1) >> PM_PAGE_BITS; page <= last; ++page) {
        if (impl->cached_pages[page].num_read_watchers != 0) {
            return true;
        }
    }
    return false;
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

} // namespace VideoCore

#ifdef _WIN32
namespace AmdGpu {
// SHADPS4_TIMESTAMP_READER=1 (diagnostic, see pm4_cmds.h NoteGpuTimestamp): after a guest GPU-timestamp
// label write, a one-shot PAGE_GUARD is ORed into that page's current protection (the emulator's own
// protection is restored by Windows when the guard fires). The first access raises a guard violation,
// logged here with the faulting RIP when it is guest code (eboot 0x800000000..0x800cc8000).
static LONG CALLBACK TimestampGuardHandler(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != STATUS_GUARD_PAGE_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const u64 rip = ep->ContextRecord->Rip;
    const u64 addr = ep->ExceptionRecord->ExceptionInformation[1];
    const bool write = ep->ExceptionRecord->ExceptionInformation[0] == 1;
    static std::atomic<u64> guest{0}, other{0};
    if (rip >= 0x800000000ULL && rip < 0x800cc8000ULL) {
        const u64 n = guest.fetch_add(1, std::memory_order_relaxed);
        if (n < 400 || (n & 0x3ff) == 0) {
            LOG_INFO(Render, "TimestampReader: n={} exe_off={:#x} addr={:#x} write={} tid={} other={}",
                     n, rip - 0x800000000ULL, addr, write, GetCurrentThreadId(), other.load());
        }
    } else {
        other.fetch_add(1, std::memory_order_relaxed);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

void ArmTimestampReaderProbe(const void* address) {
    static const bool enabled = [] {
        const char* v = std::getenv("SHADPS4_TIMESTAMP_READER");
        return v && v[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [] { AddVectoredExceptionHandler(1, TimestampGuardHandler); });
    void* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(address) & ~uintptr_t{4095});
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(page, &mbi, sizeof(mbi)) == 0 || (mbi.Protect & PAGE_GUARD) ||
        (mbi.Protect & (PAGE_NOACCESS | PAGE_EXECUTE)) != 0) {
        return;
    }
    DWORD old{};
    VirtualProtect(page, 4096, mbi.Protect | PAGE_GUARD, &old);
}
} // namespace AmdGpu
#endif
