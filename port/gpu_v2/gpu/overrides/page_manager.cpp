// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <set>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/memory_patcher.h"
#include "common/perf_stats.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/perf_sampler.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "srt_write_watch.h"
#include "repeat_census.h"
#include "feedback_join.h"
#include "video_core/buffer_cache/buffer_cache.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

namespace {
/// Deferred protection (PageManager::DeferProtect): page ranges [first, end) whose new write
/// protection the calling thread leaves to RefreshDeferredProtect.
thread_local std::vector<std::pair<u64, u64>>* t_deferred_protect = nullptr;

/// Late protection (PageManager::BeginLateProtect): the protections decided in the region lock
/// section of the calling thread, and the page ranges [first, end) it keeps locked until they
/// are applied.
struct LateProtect {
    struct Run {
        VAddr address;
        u64 size;
        Core::MemoryPermission perms;
    };
    std::vector<Run> runs;
    std::vector<std::pair<u64, u64>> locked;
    bool active{};
};
thread_local LateProtect t_late_protect;
} // Anonymous namespace

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;
        /// The protection of the page does not match its watchers yet (deferred protection).
        u8 protect_pending;
        iss::g4::SrtWriteWatchState srt_watch{};

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

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    ASSERT_MSG(num_read_watchers == 0, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        std::scoped_lock reserve_lock{reserve_mutex};
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    // ISS_SRT_DIRECT_FAST=1: no SRT write watch is ever armed (arming fails, which its callers,
    // the G4 snapshot paths, treat as an unusable snapshot), so a direct guest write has none
    // to invalidate and takes no lock (two global mutexes per fence or readback write).
    static bool SrtDirectFast() {
        static const bool enabled = [] {
            const char* value = std::getenv("ISS_SRT_DIRECT_FAST");
            const bool on = value && value[0] == '1';
            std::fprintf(stderr, "PORT_TUNING srt_direct_fast=%d\n", on ? 1 : 0);
            return on;
        }();
        return enabled;
    }

    bool ArmSrtWriteWatch(VAddr address, PageManager::SrtWatchToken& token) {
        token = {};
        if (SrtDirectFast()) return false;
        std::scoped_lock direct_gate{srt_direct_write_mutex};
        const VAddr aligned = PageManager::GetPageAddr(address);
        if (aligned >= (VAddr{1} << ADDRESS_BITS)) return false;
        if (!rasterizer->IsMapped(aligned, PM_PAGE_SIZE)) {
            // A CPU-only alias can write the same physical SRT bytes. It is safe to
            // watch only a whole, non-executable data page with known backing.
            const auto backing = Core::Memory::Instance()->DescribeBacking(aligned, PM_PAGE_SIZE);
            if (!backing || backing->size != PM_PAGE_SIZE) return false;
#ifdef _WIN64
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(aligned), &mbi, sizeof(mbi)) ||
                mbi.State != MEM_COMMIT ||
                (mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                PAGE_EXECUTE_WRITECOPY | PAGE_GUARD | PAGE_NOACCESS))) return false;
#else
            return false;
#endif
        }
        {
            std::scoped_lock reserve_lock{reserve_mutex};
            EnsurePages(aligned, aligned + PM_PAGE_SIZE);
        }
        const u64 page = aligned >> PM_PAGE_BITS;
        auto* state = cached_pages.find(page);
        auto* lock = locks.find(page);
        if (!state || !lock) return false;
        std::scoped_lock page_lock{*lock};
        u64 watch_generation{};
        if (!state->srt_watch.Arm(state->num_write_watchers, watch_generation)) return false;
        const auto old_perms = state->Perms();
        ++state->num_write_watchers;
        state->protect_pending = 0;
        if (state->Perms() != old_perms) {
            try {
                Protect(aligned, PM_PAGE_SIZE, state->Perms());
            } catch (...) {
                [[maybe_unused]] const bool released = state->srt_watch.Release(watch_generation);
                --state->num_write_watchers;
                return false;
            }
        }
        {
            std::scoped_lock active_lock{srt_active_mutex};
            srt_active_pages.insert(page);
        }
        token = {aligned, watch_generation};
        return true;
    }

    bool CheckSrtWriteWatch(PageManager::SrtWatchToken token) {
        if (!token.generation || token.page != PageManager::GetPageAddr(token.page) ||
            token.page >= (VAddr{1} << ADDRESS_BITS)) return false;
        const u64 page = token.page >> PM_PAGE_BITS;
        auto* state = cached_pages.find(page);
        auto* lock = locks.find(page);
        if (!state || !lock) return false;
        std::scoped_lock page_lock{*lock};
        return state->srt_watch.Check(token.generation);
    }

    void ReleaseSrtWriteWatch(PageManager::SrtWatchToken token) {
        if (!token.generation || token.page != PageManager::GetPageAddr(token.page) ||
            token.page >= (VAddr{1} << ADDRESS_BITS)) return;
        const u64 page = token.page >> PM_PAGE_BITS;
        auto* state = cached_pages.find(page);
        auto* lock = locks.find(page);
        if (!state || !lock) return;
        std::scoped_lock page_lock{*lock};
        if (!state->srt_watch.Release(token.generation)) return;
        const auto old_perms = state->Perms();
        --state->num_write_watchers;
        if (!state->srt_watch.Active()) {
            std::scoped_lock active_lock{srt_active_mutex};
            srt_active_pages.erase(page);
        }
        if (state->Perms() != old_perms) Protect(token.page, PM_PAGE_SIZE, state->Perms());
    }

    // Also used by the write-fault handler. Invalidate all leases of a page at once,
    // restore their write-protection contribution, and bump the generation before retry.
    bool InvalidateSrtPage(u64 page) {
        auto* state = cached_pages.find(page);
        auto* lock = locks.find(page);
        if (!state || !lock) return false;
        std::scoped_lock page_lock{*lock};
        const auto old_perms = state->Perms();
        const auto removed = state->srt_watch.Invalidate();
        if (!removed) return false;
        state->num_write_watchers -= removed;
        {
            std::scoped_lock active_lock{srt_active_mutex};
            srt_active_pages.erase(page);
        }
        if (state->Perms() != old_perms) Protect(page << PM_PAGE_BITS, PM_PAGE_SIZE,
                                                 state->Perms());
        return true;
    }

    void InvalidateSrtWriteWatches(VAddr address, u64 size) {
        if (size == 0 || address >= (VAddr{1} << ADDRESS_BITS) ||
            size > (VAddr{1} << ADDRESS_BITS) - address) return;
        const u64 first = address >> PM_PAGE_BITS;
        const u64 last = (address + size - 1) >> PM_PAGE_BITS;
        for (u64 page = first; page <= last; ++page) InvalidateSrtPage(page);
    }

    void InvalidateAllSrtWriteWatchesLocked() {
        std::vector<u64> pages;
        {
            std::scoped_lock active_lock{srt_active_mutex};
            pages.assign(srt_active_pages.begin(), srt_active_pages.end());
        }
        for (u64 page : pages) InvalidateSrtPage(page);
    }

    void InvalidateAllSrtWriteWatches() {
        std::scoped_lock direct_gate{srt_direct_write_mutex};
        InvalidateAllSrtWriteWatchesLocked();
    }

    void WithSrtDirectWrite(const std::function<void()>& write) {
        if (SrtDirectFast()) {
            write();
            return;
        }
        std::scoped_lock direct_gate{srt_direct_write_mutex};
        InvalidateAllSrtWriteWatchesLocked();
        write();
    }

    bool WithSrtWriteWatchesLocked(std::span<const PageManager::SrtWatchToken> tokens,
                                   const std::function<bool()>& publish) {
        if (tokens.empty()) return false;
        std::scoped_lock direct_gate{srt_direct_write_mutex};
        std::vector<PageManager::SrtWatchToken> sorted{tokens.begin(), tokens.end()};
        std::sort(sorted.begin(), sorted.end(), [](auto a, auto b) { return a.page < b.page; });
        std::vector<std::unique_lock<MutexTraits::Entry>> held;
        held.reserve(sorted.size());
        for (size_t i = 0; i < sorted.size(); ++i) {
            const auto token = sorted[i];
            if (!token.generation || token.page >= (VAddr{1} << ADDRESS_BITS) ||
                token.page != PageManager::GetPageAddr(token.page) ||
                (i && sorted[i - 1].page == token.page)) return false;
            auto* lock = locks.find(token.page >> PM_PAGE_BITS);
            if (!lock) return false;
            held.emplace_back(*lock);
        }
        for (auto token : sorted) {
            auto* state = cached_pages.find(token.page >> PM_PAGE_BITS);
            if (!state || !state->srt_watch.Check(token.generation)) return false;
        }
        return publish();
    }

    /// Applies the protections of the late protection section of the calling thread and
    /// unlocks its pages (the section stays begun).
    void ApplyLateProtect() {
        auto& late = t_late_protect;
        for (const auto& run : late.runs) {
            Protect(run.address, run.size, run.perms);
        }
        for (const auto& [first, end] : late.locked) {
            for (u64 page = first; page != end; ++page) {
                if (auto* lock = locks.find(page)) {
                    lock->unlock();
                }
            }
        }
        if (!late.runs.empty()) {
            Common::Perf::Add(Common::Perf::LateProtects, late.runs.size());
        }
        late.runs.clear();
        late.locked.clear();
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        if (t_late_protect.active && !t_late_protect.locked.empty()) [[unlikely]] {
            // Never expected in a region lock section: page locks are taken in order.
            ApplyLateProtect();
        }
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            std::scoped_lock reserve_lock{reserve_mutex};
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            locks[page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // A deferred protection not applied yet is applied now.
            const bool was_pending = state->protect_pending != 0;
            state->protect_pending = 0;

            // If the page must be (un)protected
            if (new_perms != old_perms || was_pending) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        // Protection added after an upload (against writes, and against reads of a range the
        // GPU is about to write) can be left to another thread (deferred protection, see
        // PageManager::DeferProtect); protection removed is applied at once.
        const bool defer = t_deferred_protect != nullptr && write_op == PageOp::Track &&
                           (read_op == PageOp::None || read_op == PageOp::Track);
        // Late protection (PageManager::BeginLateProtect): the protections are applied once the
        // caller released its region lock, the pages stay locked until then. Pages kept locked by
        // an earlier call of the section are released first: page locks are always taken in
        // address order by one call, never while holding others.
        auto& late = t_late_protect;
        if (late.active && !late.locked.empty()) {
            ApplyLateProtect();
        }
        const auto release_pending = [&] {
            if (range_pages > 0) {
                if (defer) {
                    t_deferred_protect->emplace_back(range_begin, range_begin + range_pages);
                } else if (late.active) {
                    late.runs.push_back({range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS,
                                         perms});
                } else {
                    Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                }
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            locks[base_page + page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            bool must_protect = new_perms != old_perms;
            if (defer) {
                if (must_protect) {
                    state->protect_pending = 1;
                }
            } else if (state->protect_pending != 0) {
                // A deferred protection not applied yet is applied now.
                state->protect_pending = 0;
                must_protect = true;
            }

            // If the page must be (un)protected
            if (must_protect) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        if (late.active) {
            // Unlocked by ApplyLateProtect, once the protections are applied.
            late.locked.emplace_back(base_page + page_start, base_page + page_end);
            return;
        }
        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    /// Applies the deferred protections of the pages [page_start, page_end): the protection
    /// their watchers need now, unless another change of their watchers applied it already.
    u64 RefreshDeferredProtect(u64 page_start, u64 page_end) {
        // Runs of consecutive pending pages needing the same protection: their pages stay locked
        // until it is applied, then are unlocked.
        Core::MemoryPermission perms{};
        u64 run_begin = 0;
        u64 run_pages = 0;
        u64 num_protected = 0;
        const auto flush_run = [&] {
            if (run_pages > 0) {
                Protect(run_begin << PM_PAGE_BITS, run_pages << PM_PAGE_BITS, perms);
                num_protected += run_pages;
                for (u64 page = run_begin; page != run_begin + run_pages; ++page) {
                    locks[page].unlock();
                }
                run_pages = 0;
            }
        };
        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                flush_run();
                continue;
            }
            locks[page].lock();
            if (state->protect_pending == 0) {
                // Applied already by another change of its watchers.
                locks[page].unlock();
                flush_run();
                continue;
            }
            state->protect_pending = 0;
            const auto page_perms = state->Perms();
            if (run_pages > 0 && page_perms != perms) {
                flush_run();
            }
            if (run_pages == 0) {
                run_begin = page;
                perms = page_perms;
            }
            ++run_pages;
        }
        flush_run();
        return num_protected;
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
    std::mutex reserve_mutex;
    std::mutex srt_direct_write_mutex;
    std::mutex srt_active_mutex;
    std::set<u64> srt_active_pages;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

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
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            InvalidateSrtPage(addr >> PageManager::PM_PAGE_BITS);
            rasterizer->InvalidateMemory(addr, 1,
                                         rasterizer->IsGpuCommandProcessorThreadId(ptid));

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

// ISS_FAULT_BURST_LOG (default on): a guest frame with more than 1000 write faults (each a
// Windows exception, ~10 us) is summarized on stderr: the 1 MB regions, the guest code writing,
// whether images live there. The periodic 45 ms frames of 2026-10-17 had ~2875 faults each.
struct FaultBurst {
    std::mutex mutex;
    u64 frame{~0ull};
    std::vector<std::pair<u64, u64>> faults; // address, rip
    u64 image_pages{}, srt_pages{};
    std::vector<u32> threads; // per fault, parallel to faults
    std::vector<u64> callers; // per fault: [rsp] at the fault (the leaf memcpy's return address)
    u32 reported{};
};
static FaultBurst g_fault_burst;
static void NoteWriteFault(u64 addr, u64 rip, Vulkan::Rasterizer* rasterizer, bool srt,
                           u64 caller) {
    static const bool enabled = [] {
        const char* p = std::getenv("ISS_FAULT_BURST_LOG");
        return !p || p[0] != '0';
    }();
    if (!enabled) return;
    auto& b = g_fault_burst;
    const u64 frame = Common::Perf::GuestFrameCounter().load(std::memory_order_relaxed);
    std::scoped_lock lock{b.mutex};
    if (frame != b.frame) {
        if (b.faults.size() > 1000 && b.reported < 40) {
            ++b.reported;
            std::unordered_map<u64, std::pair<u64, u64>> regions; // 1 MB -> count, min page
            std::unordered_map<u64, u64> rips;
            for (const auto& [a, r] : b.faults) {
                auto& reg = regions[a >> 20];
                if (reg.first++ == 0 || a < reg.second) reg.second = a;
                ++rips[r];
            }
            std::vector<std::pair<u64, u64>> top;
            for (const auto& [k, v] : regions) top.emplace_back(v.first, k);
            std::sort(top.rbegin(), top.rend());
            std::vector<std::pair<u64, u64>> top_rips;
            for (const auto& [k, v] : rips) top_rips.emplace_back(v, k);
            std::sort(top_rips.rbegin(), top_rips.rend());
            const u64 base = MemoryPatcher::g_eboot_address;
            std::string text;
            char item[128];
            for (size_t i = 0; i < std::min<size_t>(top.size(), 6); ++i) {
                const u64 region = top[i].second << 20;
                const bool images = false;
                std::snprintf(item, sizeof(item), " %llx:%llu%s", (unsigned long long)region,
                              (unsigned long long)top[i].first, images ? "(img)" : "");
                text += item;
            }
            text += " | rip";
            for (size_t i = 0; i < std::min<size_t>(top_rips.size(), 4); ++i) {
                const u64 r = top_rips[i].second;
                std::snprintf(item, sizeof(item), " %llx:%llu",
                              (unsigned long long)(r >= base && base ? r - base : r),
                              (unsigned long long)top_rips[i].first);
                text += item;
            }
            {
                // The order: page deltas of the first faults, the distinct threads, the callers.
                const u64 eboot = MemoryPatcher::g_eboot_address;
                std::string order;
                std::unordered_map<u32, u32> per_thread;
                std::unordered_map<u64, u32> per_caller;
                for (const u32 t : b.threads) ++per_thread[t];
                for (const u64 c : b.callers) ++per_caller[c];
                for (size_t i = 0; i < std::min<size_t>(b.faults.size(), 40); ++i) {
                    char d[32];
                    const s64 delta = i ? (s64(b.faults[i].first >> 12) - s64(b.faults[i - 1].first >> 12)) : 0;
                    std::snprintf(d, sizeof(d), " %lld/t%u", (long long)delta, b.threads[i] % 100);
                    order += d;
                }
                std::string callers;
                for (const auto& [c, n] : per_caller) {
                    if (callers.size() > 300) break;
                    char d[64];
                    std::snprintf(d, sizeof(d), " %llx%s:%u",
                                  (unsigned long long)(eboot && c >= eboot ? c - eboot : c),
                                  eboot && c >= eboot ? "(eboot)" : "", n);
                    callers += d;
                }
                std::fprintf(stderr, "FAULT_BURST_ORDER threads=%zu first=%llx%s | callers%s\n",
                             per_thread.size(),
                             (unsigned long long)(b.faults.empty() ? 0 : b.faults[0].first),
                             order.c_str(), callers.c_str());
            }
            std::fprintf(stderr, "FAULT_BURST frame=%llu faults=%zu images=%llu srt=%llu regions%s\n",
                         (unsigned long long)b.frame, b.faults.size(),
                         (unsigned long long)b.image_pages, (unsigned long long)b.srt_pages,
                         text.c_str());
        }
        b.faults.clear();
        b.threads.clear();
        b.callers.clear();
        b.image_pages = b.srt_pages = 0;
        b.frame = frame;
    }
    if (b.faults.size() < 16384) {
        b.faults.emplace_back(addr, rip);
        b.threads.push_back(u32(GetCurrentThreadId()));
        b.callers.push_back(caller);
    }
    b.srt_pages += srt;
    if (rasterizer && rasterizer->GetTextureCache().HasImagesInPage(addr)) ++b.image_pages;
}

struct SignalImpl : public PageManager::Impl {
    inline static SignalImpl* active{};
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        active = this;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    ~SignalImpl() override { active = nullptr; }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        Common::Perf::ScopedTimer protect_timer{Common::Perf::ProtectCalls,
                                                Common::Perf::ProtectNs};
        if (iss::repeat_census::enabled) [[unlikely]]
            iss::repeat_census::Note(iss::repeat_census::Protect, address, size, u32(perms));
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        const bool is_write = Common::IsWriteError(context);
        const bool handled_srt = is_write && active &&
                                 active->InvalidateSrtPage(addr >> PageManager::PM_PAGE_BITS);
        const auto is_gpu_thread =
            rasterizer->PrepareDrawPipeFault(is_write);
        if (is_write) {
            const bool handled = rasterizer->InvalidateMemory(addr, size, is_gpu_thread);
            if (handled || handled_srt) {
                Common::Perf::Add(Common::Perf::FaultWrites);
                u64 caller = 0;
#ifdef _WIN64
                if (const auto* ctx = static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord) {
                    // The first return address into the eboot's code on the stack (the game's
                    // call goes through the HLE libc to the host memcpy).
                    const u64 eboot = MemoryPatcher::g_eboot_address;
                    const auto* sp = reinterpret_cast<const u64*>(ctx->Rsp);
                    for (u32 i = 0; i < 96 && eboot; ++i) {
                        const u64 v = sp[i];
                        if (v > eboot + 0x1000 && v < eboot + 0x1000000) {
                            caller = v;
                            break;
                        }
                    }
                    // Once: the code addresses on the faulting stack, by module (who writes).
                    static std::atomic<u32> dumped{};
                    if (!caller && g_fault_burst.faults.size() > 1500 &&
                        dumped.fetch_add(1, std::memory_order_relaxed) < 3) {
                        std::string text;
                        for (u32 i = 0; i < 160 && text.size() < 900; ++i) {
                            const u64 v = sp[i];
                            HMODULE module{};
                            if (v < 0x10000 ||
                                !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                    reinterpret_cast<LPCSTR>(v), &module))
                                continue;
                            char name[MAX_PATH]{};
                            GetModuleFileNameA(module, name, MAX_PATH);
                            const char* base = std::strrchr(name, '\\');
                            char item[96];
                            std::snprintf(item, sizeof(item), " %s+%llx", base ? base + 1 : name,
                                          (unsigned long long)(v - reinterpret_cast<u64>(module)));
                            text += item;
                        }
                        HMODULE rip_module{};
                        char rip_name[MAX_PATH]{};
                        const u64 rip = reinterpret_cast<u64>(Common::GetRip(context));
                        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                               reinterpret_cast<LPCSTR>(rip), &rip_module))
                            GetModuleFileNameA(rip_module, rip_name, MAX_PATH);
                        std::fprintf(stderr, "FAULT_BURST_STACK rip=%s+%llx thread=%lu stack%s\n",
                                     std::strrchr(rip_name, '\\') ? std::strrchr(rip_name, '\\') + 1 : rip_name,
                                     (unsigned long long)(rip - reinterpret_cast<u64>(rip_module)),
                                     GetCurrentThreadId(), text.c_str());
                    }
                }
#endif
                NoteWriteFault(addr, reinterpret_cast<u64>(Common::GetRip(context)), rasterizer,
                               handled_srt, caller);
            }
            return handled || handled_srt;
        } else {
            const u64 rip = reinterpret_cast<u64>(Common::GetRip(context));
            iss::feedback_join::ReadTicket feedback_ticket{};
#ifdef _WIN64
            if (iss::feedback_join::enabled) {
                const auto* registers = static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord;
                feedback_ticket = iss::feedback_join::BeginRead(
                    rip, addr, registers->R15, registers->Rax,
                    rasterizer->GetBufferCache().MappingGeneration());
            }
#endif
            const bool handled = rasterizer->ReadMemory(addr, size, is_gpu_thread, rip);
            if (feedback_ticket.id) {
                iss::feedback_join::EndRead(feedback_ticket, handled,
                    rasterizer->GetBufferCache().MappingGeneration());
            }
            if (handled) {
                Common::Perf::Add(Common::Perf::FaultReads);
                Core::PerfSampler::RecordReadback(addr, rip);
            }
            return handled;
        }
        return false;
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->InvalidateSrtWriteWatches(address, size);
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->InvalidateSrtWriteWatches(address, size);
    impl->OnUnmap(address, size);
}

bool PageManager::ArmSrtWriteWatch(VAddr address, SrtWatchToken& token) const {
    return impl->ArmSrtWriteWatch(address, token);
}

bool PageManager::CheckSrtWriteWatch(SrtWatchToken token) const {
    return impl->CheckSrtWriteWatch(token);
}

void PageManager::ReleaseSrtWriteWatch(SrtWatchToken token) const {
    impl->ReleaseSrtWriteWatch(token);
}

void PageManager::InvalidateSrtWriteWatches(VAddr address, u64 size) const {
    impl->InvalidateSrtWriteWatches(address, size);
}

void PageManager::InvalidateAllSrtWriteWatches() const {
    impl->InvalidateAllSrtWriteWatches();
}

void PageManager::WithSrtDirectWrite(const std::function<void()>& write) const {
    impl->WithSrtDirectWrite(write);
}

bool PageManager::WithSrtWriteWatchesLocked(
    std::span<const SrtWatchToken> tokens,
    const std::function<bool()>& publish) const {
    return impl->WithSrtWriteWatchesLocked(tokens, publish);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

void PageManager::DeferProtect(std::vector<std::pair<u64, u64>>* deferred_pages) {
    t_deferred_protect = deferred_pages;
}

u64 PageManager::RefreshDeferredProtect(u64 first_page, u64 end_page) const {
    return impl->RefreshDeferredProtect(first_page, end_page);
}

bool PageManager::BeginLateProtect() const {
    auto& late = t_late_protect;
    if (late.active || !Common::Perf::GetTuning().late_protect.load(std::memory_order_relaxed)) {
        return false;
    }
    late.active = true;
    return true;
}

void PageManager::EndLateProtect() const {
    impl->ApplyLateProtect();
    t_late_protect.active = false;
}

} // namespace VideoCore
