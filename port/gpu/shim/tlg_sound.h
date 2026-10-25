// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: the overlay's own sounds (the trophy toast). A file the player puts in <data>/sounds
// (TLG_SOUND_DIR overrides; trophy.mp3 / trophy.wav, trophy-platinum.mp3 / .wav) plays when there
// is one, else a chime made here. Its own SDL audio stream, beside the game's audio.

#pragma once

namespace TlgSound {

/// Plays the trophy sound once (any thread; a new one replaces one still playing).
void PlayTrophy(bool platinum);

} // namespace TlgSound
