// SPDX-License-Identifier: GPL-2.0-or-later
// GPU black box (gpu_blackbox.h). Included by overrides/vk_scheduler.cpp inside namespace Vulkan.
//
// Markers: VK_AMD_buffer_marker writes, before each marked command, its number at top of pipe
// (the GPU reached it: everything before it started) and at bottom of pipe (everything before it
// completed). Slots 0..6 are the schedulers (queues) in their submission order; slot 7 the draws
// inside native segments (key = segment << 12 | draw). After a hang the commands between the
// "completed" and the "started" numbers are the ones in flight.

// Includes: at the top of overrides/vk_scheduler.cpp (this file is inside namespace Vulkan).

namespace GpuCrashDiag {
namespace box {

constexpr u32 Slots = 8;
constexpr u32 NativeSlot = 7;
constexpr u64 ItemRing = 1ull << 16;
constexpr u64 NativeRing = 1ull << 17;
constexpr u64 EventRing = 1ull << 14;

struct Item {
    u64 seq;
    Entry entry;
    Context ctx;
    u32 slot;
    bool paused;
    double time;
};
struct NativeItem {
    u32 key; ///< segment << 12 | draw
    Entry entry;
    Context ctx;
};
struct Ev {
    std::atomic<u64> stamp; ///< event number + 1 once written
    int category;
    double time;
    char text[184];
};

/// Images the texture cache deleted (uid, address, when), for the use-after-free check.
struct Deleted {
    std::atomic<u64> stamp;
    u64 uid;
    u64 address;
    double time;
};
constexpr u64 DeletedRing = 1ull << 14;

struct Marks {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    volatile u32* mapped{};
};
Marks g_marks;
std::once_flag g_marks_once;
Item* g_items{};
NativeItem* g_native{};
Ev* g_events{};
std::atomic<u64> g_seq{1};
std::atomic<u64> g_native_next{};
std::atomic<u32> g_segment{};
std::atomic<u64> g_event_next{};
std::array<std::atomic<const void*>, NativeSlot> g_owner{};
std::array<std::atomic<const char*>, NativeSlot> g_owner_name{};
std::array<std::atomic<u64>, Slots> g_recorded{};
std::array<std::atomic<u64>, Slots> g_submitted{};
std::atomic<int> g_stall_reports{};
std::atomic<bool> g_crashed{};
std::once_flag g_watch_once;
std::once_flag g_crash_once;
const auto g_start = std::chrono::steady_clock::now();

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
}

std::string WallClock() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &t);
    const int ms = int(std::chrono::duration_cast<std::chrono::milliseconds>(
                           now.time_since_epoch()).count() % 1000);
    char text[64];
    std::snprintf(text, sizeof(text), "%04d-%02d-%02d %02d:%02d:%02d.%03d", local.tm_year + 1900,
                  local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec, ms);
    return text;
}

bool Enabled() {
    static const bool on = [] {
        const char* p = std::getenv("ISS_GPU_BLACKBOX");
        return !(p && p[0] == '0');
    }();
    return on;
}

bool Ready() {
    const Instance* instance = g_instance.load(std::memory_order_relaxed);
    const auto& dld = VULKAN_HPP_DEFAULT_DISPATCHER;
    if (!Enabled() || !instance || !g_iss_amd_buffer_marker || !dld.vkCmdWriteBufferMarkerAMD)
        return false;
    std::call_once(g_marks_once, [instance, &dld] {
        const VkDevice device = instance->GetDevice();
        const VkBufferCreateInfo buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = Slots * 8,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer buffer{};
        if (dld.vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) return;
        VkMemoryRequirements requirements{};
        dld.vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties properties{};
        dld.vkGetPhysicalDeviceMemoryProperties(instance->GetPhysicalDevice(), &properties);
        // Host visible and coherent, in system memory if possible (read after the device is lost).
        constexpr VkMemoryPropertyFlags wanted =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        u32 type = ~0u;
        for (u32 pass = 0; pass < 2 && type == ~0u; ++pass)
            for (u32 i = 0; i < properties.memoryTypeCount && type == ~0u; ++i) {
                const auto flags = properties.memoryTypes[i].propertyFlags;
                if (!(requirements.memoryTypeBits & (1u << i)) || (flags & wanted) != wanted) continue;
                if (pass == 0 && (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
                type = i;
            }
        VkDeviceMemory memory{};
        const VkMemoryAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = type,
        };
        void* mapped{};
        if (type == ~0u || dld.vkAllocateMemory(device, &allocate_info, nullptr, &memory) != VK_SUCCESS ||
            dld.vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS ||
            dld.vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            std::fprintf(stderr, "GPU_BLACKBOX off: no host visible marker buffer\n");
            return;
        }
        std::memset(mapped, 0, Slots * 8);
        g_items = new Item[ItemRing]{};
        g_native = new NativeItem[NativeRing]{};
        g_marks = {buffer, memory, static_cast<volatile u32*>(mapped)};
        std::fprintf(stderr, "GPU_BLACKBOX on (memory type %u): reports in the run folder\n", type);
    });
    return g_marks.mapped != nullptr;
}

Ev* Events() {
    static Ev* events = new Ev[EventRing]{};
    return events;
}

Deleted* DeletedImages() {
    static Deleted* deleted = new Deleted[DeletedRing]{};
    return deleted;
}
std::atomic<u64> g_deleted_next{};

/// When the image `uid` was deleted (seconds since the start), or a negative value.
double DeletedAt(u64 uid) {
    if (!uid) return -1.0;
    Deleted* deleted = DeletedImages();
    const u64 last = g_deleted_next.load(std::memory_order_relaxed);
    for (u64 n = last; n > (last > DeletedRing ? last - DeletedRing : 0); --n) {
        const Deleted& d = deleted[(n - 1) & (DeletedRing - 1)];
        if (d.stamp.load(std::memory_order_acquire) == n && d.uid == uid) return d.time;
    }
    return -1.0;
}

/// The images a command uses, and any of them deleted before the report (use after free).
std::string ImagesOf(const Context& c, double recorded_at, bool& freed) {
    std::string out;
    char item[160];
    const auto one = [&](const char* what, u64 uid, u64 address) {
        if (!uid) return;
        const double at = DeletedAt(uid);
        // Deleted after the command was recorded is normal (freed once the GPU passed it);
        // deleted BEFORE means the command uses freed memory.
        if (at >= 0.0 && at < recorded_at) freed = true;
        if (at >= 0.0)
            std::snprintf(item, sizeof(item), " %s uid %llu@%llx [APAGADA %.3f s, %s do comando]",
                          what, (unsigned long long)uid, (unsigned long long)address, at,
                          at < recorded_at ? "ANTES" : "depois");
        else
            std::snprintf(item, sizeof(item), " %s uid %llu@%llx", what, (unsigned long long)uid,
                          (unsigned long long)address);
        out += item;
    };
    for (u32 i = 0; i < 8; ++i) one("tex", c.tex_uid[i], c.tex_addr[i]);
    for (u32 i = 0; i < 8; ++i) one("cb", c.rt_uid[i], c.color[i]);
    one("db", c.depth_uid, c.depth);
    return out;
}

void WriteMarks(VkCommandBuffer cmd, u32 slot, u32 value) {
    const auto& dld = VULKAN_HPP_DEFAULT_DISPATCHER;
    dld.vkCmdWriteBufferMarkerAMD(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_marks.buffer, slot * 8,
                                  value);
    dld.vkCmdWriteBufferMarkerAMD(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_marks.buffer,
                                  slot * 8 + 4, value);
}

u32 SlotOf(const void* owner) {
    for (u32 i = 0; i < NativeSlot; ++i) {
        const void* current = g_owner[i].load(std::memory_order_acquire);
        if (current == owner) return i;
        if (!current) {
            const void* expected = nullptr;
            if (g_owner[i].compare_exchange_strong(expected, owner) || expected == owner) return i;
        }
    }
    return NativeSlot - 1; // more schedulers than slots: they share the last one
}

const char* KindText(Kind kind) {
    switch (kind) {
    case Kind::Draw: return "desenho";
    case Kind::DrawIndexed: return "desenho indexado";
    case Kind::DrawIndirect: return "desenho INDIRETO";
    case Kind::DrawIndexedIndirect: return "desenho indexado INDIRETO";
    case Kind::Dispatch: return "dispatch (compute)";
    case Kind::DispatchIndirect: return "dispatch INDIRETO (compute)";
    case Kind::Detile: return "detile de imagem";
    case Kind::Tile: return "tile de imagem";
    case Kind::NativeSegment: return "SEGMENTO NATIVO";
    case Kind::NativeDraw: return "desenho nativo";
    }
    return "?";
}

const char* PathText(u32 path) {
    switch (path) {
    case 0: return "CP (caminho original)";
    case 1: return "NATIVO (segmento v1)";
    case 2: return "NATIVO (desenho do segmento)";
    }
    return "?";
}

std::string Targets(const Context& c) {
    std::string out;
    char item[96];
    for (u32 i = 0; i < 8; ++i) {
        if (!c.color[i]) continue;
        std::snprintf(item, sizeof(item), " cb%u=%llx(fmt %u%s)", i, (unsigned long long)c.color[i],
                      c.color_format[i], (c.blend_mask >> i) & 1 ? ",BLEND" : "");
        out += item;
    }
    if (c.depth) {
        std::snprintf(item, sizeof(item), " db=%llx", (unsigned long long)c.depth);
        out += item;
    }
    return out.empty() ? " (sem alvos)" : out;
}

std::string Describe(const Entry& e, const Context& c) {
    char text[512];
    switch (e.kind) {
    case Kind::Dispatch:
    case Kind::DispatchIndirect:
        std::snprintf(text, sizeof(text), "%s cs %llx grupos %ux%ux%u args %llx", KindText(e.kind),
                      (unsigned long long)e.hashes[0], e.dims[0], e.dims[1], e.dims[2],
                      (unsigned long long)e.address);
        return text;
    case Kind::Detile:
    case Kind::Tile:
        std::snprintf(text, sizeof(text), "%s %ux%ux%u em %llx", KindText(e.kind), e.dims[0],
                      e.dims[1], e.dims[2], (unsigned long long)e.address);
        return text;
    case Kind::NativeSegment:
        std::snprintf(text, sizeof(text), "%s id %llu com %u desenhos, area %ux%u%s",
                      KindText(e.kind), (unsigned long long)e.address, e.count, e.dims[0], e.dims[1],
                      Targets(c).c_str());
        return text;
    default:
        std::snprintf(text, sizeof(text), "%s vs %llx ps %llx n %u inst/stride %u area %ux%u args %llx%s",
                      KindText(e.kind), (unsigned long long)e.hashes[0],
                      (unsigned long long)e.hashes[1], e.count, e.instances, e.dims[0], e.dims[1],
                      (unsigned long long)e.address, Targets(c).c_str());
        return text;
    }
}

const Item* ItemAt(u64 seq) {
    if (!g_items || seq == 0) return nullptr;
    const Item& it = g_items[seq & (ItemRing - 1)];
    return it.seq == seq ? &it : nullptr;
}

/// The full 64-bit number of a marker value (32 bits) near the last number given out.
u64 Widen(u32 value) {
    const u64 last = g_seq.load(std::memory_order_relaxed);
    u64 seq = (last & ~0xFFFFFFFFull) | value;
    if (seq > last) seq -= 1ull << 32;
    return value ? seq : 0;
}

std::string JsonText(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out;
}

std::string OutputDir() {
    if (const char* p = std::getenv("SHADPS4_FRAME_CSV"); p && *p) {
        std::string dir = p;
        const size_t cut = dir.find_last_of("/\\");
        if (cut != std::string::npos) return dir.substr(0, cut);
    }
    return ".";
}

void Dump(const char* reason, const char* file_stem) {
    if (!g_marks.mapped) return;
    const double now = Now();
    const bool paused = iss::pause_guard::active.load(std::memory_order_relaxed);
    std::string text, json;
    char line[1024];
    const auto add = [&](const char* format, ...) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(line, sizeof(line), format, args);
        va_end(args);
        text += line;
        text += '\n';
    };
    add("CAIXA-PRETA DA GPU: %s", reason);
    add("hora %s | %.2f s desde o inicio do jogo | pause: %s", WallClock().c_str(), now,
        paused ? "SIM (nativo desligado)" : "nao");
    json += "{\"reason\":\"" + JsonText(reason) + "\",\"wall\":\"" + WallClock() + "\"";
    std::snprintf(line, sizeof(line), ",\"time\":%.3f,\"paused\":%s,\"queues\":[", now,
                  paused ? "true" : "false");
    json += line;

    // The command in flight on each queue.
    std::vector<std::pair<u32, u64>> culprits; // slot, first seq in flight
    bool first_queue = true;
    add("");
    add("== Filas (marcadores: iniciado / concluido / enviado) ==");
    for (u32 s = 0; s < NativeSlot; ++s) {
        const void* owner = g_owner[s].load(std::memory_order_relaxed);
        if (!owner) continue;
        const char* name = g_owner_name[s].load(std::memory_order_relaxed);
        const u64 started = Widen(g_marks.mapped[s * 2]);
        const u64 done = Widen(g_marks.mapped[s * 2 + 1]);
        const u64 submitted = g_submitted[s].load(std::memory_order_relaxed);
        const bool stuck = submitted > done;
        add("fila %u (%s): iniciado #%llu, concluido ate #%llu, enviado ate #%llu%s", s,
            name ? name : "?", (unsigned long long)started, (unsigned long long)done,
            (unsigned long long)submitted, stuck ? "  <-- TRABALHO PENDENTE" : "");
        std::snprintf(line, sizeof(line),
                      "%s{\"slot\":%u,\"name\":\"%s\",\"started\":%llu,\"done\":%llu,\"submitted\":%llu}",
                      first_queue ? "" : ",", s, name ? name : "?", (unsigned long long)started,
                      (unsigned long long)done, (unsigned long long)submitted);
        json += line;
        first_queue = false;
        if (stuck && done) culprits.emplace_back(s, done);
    }
    json += "],\"culprits\":[";
    const u32 native_value = g_marks.mapped[NativeSlot * 2];
    const u32 native_done = g_marks.mapped[NativeSlot * 2 + 1];
    bool first_culprit = true;
    for (const auto& [slot, done] : culprits) {
        const u64 started = Widen(g_marks.mapped[slot * 2]);
        add("");
        add("== PROVAVEL CULPADO na fila %u: comandos #%llu..#%llu em andamento ==", slot,
            (unsigned long long)done, (unsigned long long)std::max(done, started));
        for (u64 seq = done; seq <= std::max(done, started) && seq < done + 16; ++seq) {
            const Item* it = ItemAt(seq);
            if (!it) continue;
            add(">> #%llu tick %llu [%s]%s %s", (unsigned long long)seq,
                (unsigned long long)it->entry.tick, PathText(it->ctx.path),
                it->paused ? " (no pause)" : "", Describe(it->entry, it->ctx).c_str());
            bool freed = false;
            const std::string images = ImagesOf(it->ctx, it->time, freed);
            add("   imagens:%s", images.empty() ? " (nenhuma registrada)" : images.c_str());
            if (freed) add("   !!! USA IMAGEM APAGADA PELO CACHE (memoria liberada): provavel page fault");
            std::snprintf(line, sizeof(line),
                          "%s{\"slot\":%u,\"seq\":%llu,\"tick\":%llu,\"path\":%u,\"kind\":%u,"
                          "\"vs\":\"%llx\",\"ps\":\"%llx\",\"count\":%u,\"args\":\"%llx\","
                          "\"width\":%u,\"height\":%u,\"color0\":\"%llx\",\"depth\":\"%llx\","
                          "\"blend\":%u,\"paused\":%s,\"text\":\"%s\"}",
                          first_culprit ? "" : ",", slot, (unsigned long long)seq,
                          (unsigned long long)it->entry.tick, it->ctx.path, u32(it->entry.kind),
                          (unsigned long long)it->entry.hashes[0],
                          (unsigned long long)it->entry.hashes[1], it->entry.count,
                          (unsigned long long)it->entry.address, it->entry.dims[0],
                          it->entry.dims[1], (unsigned long long)it->ctx.color[0],
                          (unsigned long long)it->ctx.depth, it->ctx.blend_mask,
                          it->paused ? "true" : "false",
                          JsonText(Describe(it->entry, it->ctx)).c_str());
            json += line;
            first_culprit = false;
            if (it->entry.kind == Kind::NativeSegment) {
                const u32 segment = u32(it->entry.address);
                add("   dentro do segmento %u: marcador nativo iniciado %u.%u, concluido %u.%u",
                    segment, native_value >> 12, native_value & 0xFFF, native_done >> 12,
                    native_done & 0xFFF);
                const u64 last = g_native_next.load(std::memory_order_relaxed);
                for (u64 n = last > NativeRing ? last - NativeRing : 0; n < last; ++n) {
                    const NativeItem& ni = g_native[n & (NativeRing - 1)];
                    if ((ni.key >> 12) != (segment & 0xFFFFF)) continue;
                    const bool flight = ni.key == native_value || ni.key == native_done;
                    add("   %s desenho %u: %s", flight ? ">>" : "  ", ni.key & 0xFFF,
                        Describe(ni.entry, ni.ctx).c_str());
                }
            }
        }
        add("");
        add("-- os 40 comandos anteriores na fila %u --", slot);
        int freed_before = 0;
        for (u64 seq = done > 40 ? done - 40 : 1; seq < done; ++seq) {
            const Item* it = ItemAt(seq);
            if (!it || it->slot != slot) continue;
            bool freed = false;
            const std::string images = ImagesOf(it->ctx, it->time, freed);
            freed_before += freed;
            add("   #%llu tick %llu [%s]%s %s%s", (unsigned long long)seq,
                (unsigned long long)it->entry.tick, PathText(it->ctx.path),
                it->paused ? " (no pause)" : "", Describe(it->entry, it->ctx).c_str(),
                freed ? "  !!! usa imagem apagada" : "");
            if (freed) add("      imagens:%s", images.c_str());
        }
        add("checagem de memoria liberada: %d dos 40 comandos anteriores usam imagem apagada",
            freed_before);
    }
    if (culprits.empty()) add("(nenhuma fila com trabalho pendente marcado: o travamento pode ser fora dos comandos marcados)");

    // The diary, by category (newest last): pause and others, compiles, images.
    json += "],\"events\":[";
    Ev* events = Events();
    const u64 last = g_event_next.load(std::memory_order_relaxed);
    bool first_event = true;
    static constexpr const char* titles[] = {"pause e outros", "compilacoes de shader/pipeline",
                                             "imagens criadas (+) e apagadas (-)"};
    static constexpr u64 limits[] = {60, 40, 120};
    const u64 oldest = last > EventRing ? last - EventRing : 0;
    for (int cat = 0; cat < 3; ++cat) {
        std::vector<u64> picked;
        for (u64 n = last; n > oldest && picked.size() < limits[cat]; --n) {
            const Ev& ev = events[(n - 1) & (EventRing - 1)];
            if (ev.stamp.load(std::memory_order_acquire) != n || ev.category != cat) continue;
            picked.push_back(n - 1);
        }
        add("");
        add("== Diario: %s (s desde o inicio; entre parenteses, quanto antes do relatorio) ==",
            titles[cat]);
        for (auto it = picked.rbegin(); it != picked.rend(); ++it) {
            const Ev& ev = events[*it & (EventRing - 1)];
            add("%9.3f  (-%6.2f s)  %s", ev.time, now - ev.time, ev.text);
            std::snprintf(line, sizeof(line), "%s{\"time\":%.3f,\"category\":%d,\"text\":\"%s\"}",
                          first_event ? "" : ",", ev.time, cat, JsonText(ev.text).c_str());
            json += line;
            first_event = false;
        }
    }
    json += "]}\n";

    const std::string dir = OutputDir();
    const std::string txt_path = dir + "\\" + file_stem + ".txt";
    const std::string json_path = dir + "\\" + file_stem + ".json";
    if (FILE* f = std::fopen(txt_path.c_str(), "wb")) {
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
    }
    if (FILE* f = std::fopen(json_path.c_str(), "wb")) {
        std::fwrite(json.data(), 1, json.size(), f);
        std::fclose(f);
    }
    std::fprintf(stderr, "GPU_BLACKBOX %s: %s\n", reason, txt_path.c_str());
    std::fputs(text.c_str(), stderr);
    std::fflush(stderr);
}

/// Watchdog: a queue whose submitted work stops advancing for 1 s (the driver resets at 2 s).
void Watch() {
    std::array<u32, NativeSlot> last{};
    std::array<double, NativeSlot> since{};
    std::array<u32, NativeSlot> reported{};
    // ISS_GPU_BLACKBOX_TEST_S=N: one report N s after the first mark, without a hang (to check
    // the report itself).
    // The test report stalls the game ~350 ms (measured 2026-10-19, at the same second in every
    // scene): only in a diagnostic session, not from the play flags.
    static const double test_at = [] {
        const char* p = std::getenv("ISS_GPU_BLACKBOX_TEST_S");
        const char* diagnostic = std::getenv("ISS_DIAGNOSTIC_SESSION");
        return p && diagnostic && diagnostic[0] == '1' ? std::atof(p) : 0.0;
    }();
    const double first = Now();
    bool tested = test_at <= 0.0;
    while (!g_crashed.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const double now = Now();
        if (!tested && now - first >= test_at) {
            tested = true;
            Dump("TESTE: retrato da caixa-preta (sem travamento)", "gpu-blackbox-test");
        }
        for (u32 s = 0; s < NativeSlot; ++s) {
            if (!g_owner[s].load(std::memory_order_relaxed)) continue;
            const u64 submitted = g_submitted[s].load(std::memory_order_relaxed);
            const u32 done = g_marks.mapped[s * 2 + 1];
            if (u32(submitted) == done || done != last[s]) {
                last[s] = done;
                since[s] = now;
                continue;
            }
            if (now - since[s] < 1.0 || reported[s] == done) continue;
            reported[s] = done;
            const int n = g_stall_reports.fetch_add(1) + 1;
            if (n > 5) continue;
            char stem[32];
            std::snprintf(stem, sizeof(stem), "gpu-stall-%d", n);
            Dump("GPU parada ha mais de 1 s (antes do reset do driver)", stem);
        }
    }
}

} // namespace box

bool Blackbox() {
    return box::Ready();
}

u32 NewNativeSegment() {
    return (box::g_segment.fetch_add(1, std::memory_order_relaxed) + 1) & 0xFFFFF;
}

void MarkBox(const void* owner, const CommandRecorder& cmdbuf, u64 tick, const Entry& entry,
             const Context& ctx) {
    if (!box::Ready()) return;
    std::call_once(box::g_watch_once, [] { std::thread(box::Watch).detach(); });
    const u64 seq = box::g_seq.fetch_add(1, std::memory_order_relaxed);
    const u32 slot = box::SlotOf(owner);
    box::Item& item = box::g_items[seq & (box::ItemRing - 1)];
    item.entry = entry;
    item.entry.seq = seq;
    item.entry.tick = tick;
    item.ctx = ctx;
    item.slot = slot;
    item.paused = iss::pause_guard::active.load(std::memory_order_relaxed);
    item.time = box::Now();
    item.seq = seq;
    cmdbuf.Record([slot, seq](vk::CommandBuffer c) {
        box::g_recorded[slot].store(seq, std::memory_order_relaxed);
        box::WriteMarks(static_cast<VkCommandBuffer>(c), slot, u32(seq));
    });
}

void MarkNativeDraw(VkCommandBuffer cmd, u32 segment, u32 draw, const Entry& entry,
                    const Context& ctx) {
    if (!box::Ready()) return;
    const u32 key = (segment << 12) | (draw & 0xFFF);
    const u64 n = box::g_native_next.fetch_add(1, std::memory_order_relaxed);
    box::NativeItem& item = box::g_native[n & (box::NativeRing - 1)];
    item.entry = entry;
    item.ctx = ctx;
    item.key = key;
    box::WriteMarks(cmd, box::NativeSlot, key);
}

void NoteSubmit(const void* owner, const char* name) {
    if (!box::g_marks.mapped) return;
    const u32 slot = box::SlotOf(owner);
    box::g_owner_name[slot].store(name, std::memory_order_relaxed);
    box::g_submitted[slot].store(box::g_recorded[slot].load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
}

void BlackboxEventV(int category, const char* format, va_list args) {
    if (!box::Enabled()) return;
    box::Ev* events = box::Events();
    const u64 n = box::g_event_next.fetch_add(1, std::memory_order_relaxed);
    box::Ev& ev = events[n & (box::EventRing - 1)];
    ev.stamp.store(0, std::memory_order_relaxed);
    ev.category = category;
    ev.time = box::Now();
    std::vsnprintf(ev.text, sizeof(ev.text), format, args);
    ev.stamp.store(n + 1, std::memory_order_release);
}

void BlackboxImageDeleted(u64 uid, u64 address) {
    if (!box::Enabled()) return;
    box::Deleted* deleted = box::DeletedImages();
    const u64 n = box::g_deleted_next.fetch_add(1, std::memory_order_relaxed);
    box::Deleted& d = deleted[n & (box::DeletedRing - 1)];
    d.stamp.store(0, std::memory_order_relaxed);
    d.uid = uid;
    d.address = address;
    d.time = box::Now();
    d.stamp.store(n + 1, std::memory_order_release);
}

void BlackboxCrash(const char* where) {
    if (!box::g_marks.mapped) return;
    std::call_once(box::g_crash_once, [where] {
        box::g_crashed.store(true);
        char reason[128];
        std::snprintf(reason, sizeof(reason), "DEVICE LOST (GPU travou, driver resetou) em %s", where);
        box::Dump(reason, "gpu-crash");
        // The device is lost: end the process now (no assert dialog, no more submissions to a
        // resetting driver, which can cascade into resets that need a reboot).
        if (const char* p = std::getenv("ISS_GPU_BLACKBOX_KEEP_ALIVE"); !(p && p[0] == '1')) {
            std::fflush(nullptr);
            std::_Exit(3);
        }
    });
}

} // namespace GpuCrashDiag
