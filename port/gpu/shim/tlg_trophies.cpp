// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_trophies.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

#include "../tlggpu.h"
#include "common/elf_info.h"
#include "tlg_overlay.h"

namespace TlgTrophies {

namespace {

struct Entry {
    std::string name;
    uint64_t offset, size;
};

std::mutex mutex;
bool attached = false;
std::string pack, store;
std::vector<Entry> entries;
std::vector<Unlocked> unlocked;
int count = 0;

uint32_t Big32(const unsigned char* p) {
    return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
}
uint64_t Big64(const unsigned char* p) {
    return uint64_t(Big32(p)) << 32 | Big32(p + 4);
}

// The pack's index: magic DCA24D00, then (big endian) version, file size, entry count and entry
// size at 0x10 and 0x14; entries from 0x60: a 32-byte name, the offset and the size.
void ReadIndex() {
    std::FILE* f = std::fopen(pack.c_str(), "rb");
    if (!f) return;
    unsigned char head[0x60];
    if (std::fread(head, 1, sizeof(head), f) == sizeof(head) && Big32(head) == 0xDCA24D00u) {
        const uint32_t n = Big32(head + 0x10), size = Big32(head + 0x14);
        std::vector<unsigned char> table(size_t(n) * size);
        if (n < 4096 && size >= 48 && size <= 1024 && std::fread(table.data(), 1, table.size(), f) == table.size()) {
            for (uint32_t i = 0; i < n; ++i) {
                const unsigned char* e = table.data() + size_t(i) * size;
                entries.push_back({std::string(reinterpret_cast<const char*>(e), strnlen(reinterpret_cast<const char*>(e), 32)),
                                   Big64(e + 32), Big64(e + 40)});
                int id;
                if (std::sscanf(entries.back().name.c_str(), "TROP%3d.PNG", &id) == 1) count = std::max(count, id + 1);
            }
        }
    }
    std::fclose(f);
}

} // namespace

void Attach(const char* pack_path, const char* store_path) {
    std::scoped_lock lock{mutex};
    if (attached) return;
    attached = true;
    pack = pack_path ? pack_path : "";
    store = store_path ? store_path : "";
    ReadIndex();
    if (std::FILE* f = std::fopen(store.c_str(), "r")) {
        int id;
        long long time;
        while (std::fscanf(f, "%d %lld", &id, &time) == 2) {
            bool known = false;
            for (const Unlocked& u : unlocked) known |= u.id == id;
            if (!known) unlocked.push_back({id, time});
        }
        std::fclose(f);
    }
    std::printf("Trophies: %d in the pack, %zu unlocked on this PC\n", count, unlocked.size());
}

bool Unlock(int id) {
    std::scoped_lock lock{mutex};
    for (const Unlocked& u : unlocked) {
        if (u.id == id) return false;
    }
    const int64_t now = int64_t(std::time(nullptr));
    unlocked.push_back({id, now});
    if (std::FILE* f = std::fopen(store.c_str(), "a")) {
        std::fprintf(f, "%d %lld\n", id, static_cast<long long>(now));
        std::fclose(f);
    }
    return true;
}

int Count() {
    std::scoped_lock lock{mutex};
    return count;
}

std::vector<Unlocked> List() {
    std::scoped_lock lock{mutex};
    return unlocked;
}

bool Icon(int id, TlgPng::Image& out) {
    char name[16];
    std::snprintf(name, sizeof(name), "TROP%03d.PNG", id);
    uint64_t offset = 0, size = 0;
    {
        std::scoped_lock lock{mutex};
        for (const Entry& e : entries) {
            if (e.name == name) {
                offset = e.offset;
                size = e.size;
            }
        }
    }
    if (!size || size > (16u << 20)) return false;
    std::vector<unsigned char> data(size);
    std::FILE* f = std::fopen(pack.c_str(), "rb");
    if (!f) return false;
    const bool read = std::fseek(f, long(offset), SEEK_SET) == 0 && std::fread(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    return read && TlgPng::DecodeRgba(data.data(), data.size(), out);
}

// inFAMOUS First Light (CUSA00897): the trophies in the pack's order (TROP000 to TROP024), matched
// against their icons (the arenas' A, B and Г, the turret, the graduation cap, the fish...).
constexpr Info first_light[] = {
    {Platinum, "Noble Achievement", "Earn every other trophy", "Получите все остальные трофеи"},
    {Bronze, "Lost", "Story", "Сюжет"},
    {Silver, "Cracked", "Story", "Сюжет"},
    {Gold, "Betrayed", "Story", "Сюжет"},
    {Gold, "She's Ready", "Finish the story", "Пройдите сюжет"},
    {Silver, "Illuminated", "Collect every Neon Lumen", "Соберите все неоновые люмены"},
    {Bronze, "Blackout", "Destroy every police drone", "Уничтожьте все полицейские дроны"},
    {Silver, "Lickety Pronto", "Finish every Lumen race", "Пройдите все гонки за люменами"},
    {Bronze, "That's So Fetch", "Complete every neon graffiti", "Нарисуйте все неоновые граффити"},
    {Gold, "Unleashed", "Buy every upgrade", "Купите все улучшения"},
    {Bronze, "Novice", "Complete 25% of the challenges", "Выполните 25% испытаний"},
    {Silver, "Professional", "Complete 50% of the challenges", "Выполните 50% испытаний"},
    {Gold, "Expert", "Complete 75% of the challenges", "Выполните 75% испытаний"},
    {Gold, "Perfectionist", "Complete every challenge", "Выполните все испытания"},
    {Silver, "Alpha Rescue Diamond", "Score 500,000 in the Alpha rescue arena", "Наберите 500 000 очков на арене «Альфа», спасение"},
    {Silver, "Alpha Survival Diamond", "Score 500,000 in the Alpha survival arena", "Наберите 500 000 очков на арене «Альфа», выживание"},
    {Silver, "Beta Rescue Diamond", "Score 500,000 in the Beta rescue arena", "Наберите 500 000 очков на арене «Бета», спасение"},
    {Silver, "Beta Survival Diamond", "Score 500,000 in the Beta survival arena", "Наберите 500 000 очков на арене «Бета», выживание"},
    {Silver, "Gamma Survival Diamond", "Score 500,000 in the Gamma survival arena", "Наберите 500 000 очков на арене «Гамма», выживание"},
    {Silver, "Turret Syndrome", "Overload 25 turrets", "Перегрузите 25 турелей"},
    {Gold, "Great Responsibility", "Save 50 hostages", "Спасите 50 заложников"},
    {Gold, "Top of the Class", "Score 10,000,000 in the arenas in all", "Наберите на аренах 10 000 000 очков в сумме"},
    {Bronze, "Event Horizon", "Defeat 100 enemies with the Neon Singularity", "Победите 100 врагов неоновой сингулярностью"},
    {Silver, "...Bada Boom!", "Defeat 20 enemies in a row with homing missiles", "Победите 20 врагов подряд самонаводящимися ракетами"},
    {Bronze, "Fish in a Barrel", "Get 25 stasis and weak-point kills", "25 раз добейте врага в стазисе или в уязвимое место"},
};

const Info* Describe(int id) {
    static const bool first_light_game = Common::ElfInfo::Instance().GameSerial() == "CUSA00897";
    if (!first_light_game || id < 0 || id >= int(sizeof(first_light) / sizeof(first_light[0]))) return nullptr;
    return &first_light[id];
}

} // namespace TlgTrophies

extern "C" void tlggpu_trophy_attach(const char* pack_path, const char* store_path) {
    TlgTrophies::Attach(pack_path, store_path);
    // TLG_TROPHY_TEST=<id>[,<id>...]: those trophies' toasts at start, nothing recorded (checks
    // the overlay and the sounds).
    static bool tested = false;
    if (const char* test = std::getenv("TLG_TROPHY_TEST"); test && *test && !tested) {
        tested = true;
        for (const char* p = test; *p;) {
            char* end;
            const long id = std::strtol(p, &end, 10);
            if (end == p) break;
            TlgOverlay::ShowTrophy(int(id));
            p = *end == ',' ? end + 1 : end;
        }
    }
}

extern "C" int tlggpu_trophy_unlock(int id) {
    if (!TlgTrophies::Unlock(id)) return 0;
    TlgOverlay::ShowTrophy(id);
    return 1;
}
