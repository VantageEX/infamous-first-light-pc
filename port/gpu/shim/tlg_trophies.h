// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the game's trophies (src/runtime_services.c, sceNpTrophy*). The trophy pack
// (sce_sys/trophy/trophy00.trp) holds each trophy's icon as a plain PNG; its names and
// descriptions are encrypted with the console's key, so the overlay shows icons and numbers.
// The trophies unlocked on this PC are kept in <user>/trophies.txt, one "id unix-time" a line.

#pragma once

#include <cstdint>
#include <vector>

#include "tlg_png.h"

namespace TlgTrophies {

struct Unlocked {
    int id;
    int64_t time; ///< seconds since 1970
};

/// Opens the pack and reads the unlocked list (once; later calls do nothing).
void Attach(const char* pack_path, const char* store_path);
/// Records a trophy; false when it was already unlocked (the game unlocks some again).
bool Unlock(int id);
/// Trophies in the pack (its TROPnnn.PNG icons); 0 before Attach or without the pack.
int Count();
/// The unlocked trophies, in unlock order.
std::vector<Unlocked> List();
/// Decodes trophy `id`'s icon; false when the pack has none.
bool Icon(int id, TlgPng::Image& out);

enum Grade { Platinum, Gold, Silver, Bronze };
/// What a trophy is: the pack's own texts are encrypted, so inFAMOUS First Light's are kept
/// here (the names as the game has them, the requirements in our words, no story spoilers).
struct Info {
    Grade grade;
    const char* name;
    const char* en;
    const char* ru;
};
/// Trophy `id` of this game, or null (another game, an unknown id).
const Info* Describe(int id);

} // namespace TlgTrophies
