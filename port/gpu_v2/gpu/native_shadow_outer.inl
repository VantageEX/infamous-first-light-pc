// Static ABI evidence: out/experimental/20261015-shadow-stages/engine-evidence.
// Aggregate consecutive eligible technique groups inside an engine job/range.
using RenderGroups = u64(ISS_SYSV*)(u64* args, u64 mask);
RenderGroups original_groups{};
Render original_technique{};
struct OuterStats {
    std::atomic<u64> scopes{}, groups{}, batches{}, draws{}, chunks{}, rejected{};
};
OuterStats outer_stats;
struct OuterRange {
    CommandBuffer* dcb{};
    u32* marker{};
    const u32* end_before{};
    Batch* batch{};
    u32 generation{}, groups{};
    std::optional<DirectCapture> capture;
    static u32 MaxGroups() {
        static const u32 count = [] {
            const char* p = std::getenv("ISS_NATIVE_SHADOW_OUTER_MAX_GROUPS");
            const int n = p ? std::atoi(p) : 16;
            return u32(std::clamp(n, 1, 256));
        }();
        return count;
    }
    bool Begin(u64 context) {
        dcb = reinterpret_cast<CommandBuffer*>(context + 8);
        if (static_cast<u64>(dcb->end - dcb->cursor) < MarkerDwords &&
            !dcb->reserve(dcb, MarkerDwords, dcb->user)) return false;
        marker = dcb->cursor;
        generation = next_generation.fetch_add(1, std::memory_order_relaxed);
        batch = &slots[generation % SlotCount];
        batch->ready.store(0, std::memory_order_release);
        marker[0] = (3u << 30) | ((MarkerDwords - 2) << 16) | (0x10u << 8);
        marker[1] = MarkerMagic | MarkerBegin;
        marker[2] = generation % SlotCount;
        marker[3] = generation;
        dcb->cursor = marker + MarkerDwords;
        end_before = dcb->end;
        if (DirectCaptureEnabled()) capture.emplace(*batch, marker + MarkerDwords, end_before);
        groups = 0;
        capturing_outer = true;
        return true;
    }
    void Finish() {
        if (!batch) return;
        capturing_outer = false;
        if (dcb->end != end_before || dcb->cursor < marker + MarkerDwords) ++outer_stats.chunks;
        else if (FinishCapture(capture ? &*capture : nullptr, *batch,
                               marker + MarkerDwords, dcb->cursor)) {
            batch->marker = marker; batch->end = dcb->cursor;
            outer_stats.draws += batch->draws.size(); ++outer_stats.batches;
            batch->ready.store(generation, std::memory_order_release);
        } else ++outer_stats.rejected;
        capture.reset(); batch = nullptr;
    }
    ~OuterRange() { Finish(); }
};
thread_local OuterRange* outer_range{};
struct OuterScope {
    OuterRange range;
    OuterScope() { outer_range = &range; ++outer_stats.scopes; }
    ~OuterScope() {
        range.Finish(); outer_range = nullptr;
        if ((outer_stats.scopes.load() & 255) == 1)
            std::fprintf(stderr, "NATIVE_SHADOW_OUTER scopes=%llu groups=%llu batches=%llu draws=%llu chunks=%llu rejected=%llu\n",
                outer_stats.scopes.load(), outer_stats.groups.load(), outer_stats.batches.load(),
                outer_stats.draws.load(), outer_stats.chunks.load(), outer_stats.rejected.load());
    }
};
u64 ISS_SYSV RenderTechniqueHook(u64 technique, u64* args) {
    auto* range = outer_range;
    if (!range) return original_technique(technique, args);
    const bool eligible = *reinterpret_cast<const u64*>(technique) == image_base + ShadowVtable &&
                          *reinterpret_cast<const unsigned char*>(technique + 0x1F) != 0;
    if (!eligible || (range->batch && range->dcb != reinterpret_cast<CommandBuffer*>(args[0] + 8)))
        range->Finish();
    if (eligible && !range->batch) range->Begin(args[0]);
    const u64 result = original_technique(technique, args);
    if (eligible && range->batch) {
        ++outer_stats.groups;
        if (++range->groups >= OuterRange::MaxGroups() || range->dcb->end != range->end_before)
            range->Finish();
    }
    return result;
}
u64 ISS_SYSV RenderGroupsHook(u64* args, u64 mask) {
    if (outer_range) return original_groups(args, mask);
    OuterScope scope;
    return original_groups(args, mask); // preserve range mutation and original RAX
}
u64 ISS_SYSV RenderBucketsHook(u64 list, u64 context, u32 first, u32 end) {
    if (outer_range || first != 39 || end != 40) return original_buckets(list, context, first, end);
    OuterScope scope;
    return original_buckets(list, context, first, end);
}
template <class Function>
bool InstallOuterHook(u64 offset, Function hook, Function& original_function) {
    auto* target = reinterpret_cast<unsigned char*>(image_base + offset);
    if (std::memcmp(target, Prologue, sizeof(Prologue)) != 0) return false;
    auto* trampoline = static_cast<unsigned char*>(VirtualAlloc(
        nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!trampoline) return false;
    std::memcpy(trampoline, Prologue, sizeof(Prologue));
    const unsigned char jump[6] = {0xff, 0x25, 0, 0, 0, 0};
    const auto back = reinterpret_cast<std::uintptr_t>(target + sizeof(Prologue));
    std::memcpy(trampoline + 13, jump, 6);
    std::memcpy(trampoline + 19, &back, 8);
    original_function = reinterpret_cast<Function>(trampoline);
    unsigned char patch[13] = {0x48, 0xb8};
    const auto hook_address = reinterpret_cast<std::uintptr_t>(hook);
    std::memcpy(patch + 2, &hook_address, 8);
    patch[10] = 0xff; patch[11] = 0xe0; patch[12] = 0x90;
    DWORD old, ignored;
    if (!VirtualProtect(target, sizeof(patch), PAGE_EXECUTE_READWRITE, &old)) return false;
    std::memcpy(target, patch, sizeof(patch));
    return VirtualProtect(target, sizeof(patch), old, &ignored) &&
           FlushInstructionCache(GetCurrentProcess(), target, sizeof(patch));
}
bool InstallBuckets() {
    // Patch the inner dispatcher first: an unsuccessful outer installation then
    // leaves it inert (outer_range is null), preserving the original execution.
    return InstallOuterHook(0x448260, &RenderTechniqueHook, original_technique) &&
           InstallOuterHook(0x198670, &RenderGroupsHook, original_groups) &&
           InstallOuterHook(0x198BC0, &RenderBucketsHook, original_buckets);
}
