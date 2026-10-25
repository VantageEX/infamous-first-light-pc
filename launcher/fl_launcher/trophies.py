# SPDX-License-Identifier: GPL-2.0-or-later
"""The trophies unlocked on this PC, for the launcher's small trophy card. The core keeps them in
<data>/user/trophies.txt, one "id unix-time" a line (port/gpu/shim/tlg_trophies.cpp); the icons
come from the game's trophy pack, sce_sys/trophy/trophy00.trp, where they are plain PNGs (the
names are encrypted there, so inFAMOUS First Light's are kept here, as in the core)."""
import datetime
import struct
from pathlib import Path

PACK = Path('sce_sys') / 'trophy' / 'trophy00.trp'
PACK_MAGIC = 0xDCA24D00

# inFAMOUS First Light (CUSA00897), in the pack's order (TROP000 to TROP024): grade, name.
NAMES = {'CUSA00897': [
    ('platinum', 'Noble Achievement'), ('bronze', 'Lost'), ('silver', 'Cracked'), ('gold', 'Betrayed'),
    ('gold', "She's Ready"), ('silver', 'Illuminated'), ('bronze', 'Blackout'), ('silver', 'Lickety Pronto'),
    ('bronze', "That's So Fetch"), ('gold', 'Unleashed'), ('bronze', 'Novice'), ('silver', 'Professional'),
    ('gold', 'Expert'), ('gold', 'Perfectionist'), ('silver', 'Alpha Rescue Diamond'),
    ('silver', 'Alpha Survival Diamond'), ('silver', 'Beta Rescue Diamond'), ('silver', 'Beta Survival Diamond'),
    ('silver', 'Gamma Survival Diamond'), ('silver', 'Turret Syndrome'), ('gold', 'Great Responsibility'),
    ('gold', 'Top of the Class'), ('bronze', 'Event Horizon'), ('silver', '...Bada Boom!'),
    ('bronze', 'Fish in a Barrel'),
]}


def pack_entries(path):
    """{name: (offset, size)} of a trophy pack: magic, then (big endian) the entry count and size
    at 0x10 and 0x14; entries from 0x60: a 32-byte name, the offset and the size."""
    with open(path, 'rb') as f:
        head = f.read(0x60)
        if len(head) < 0x60 or struct.unpack_from('>I', head, 0)[0] != PACK_MAGIC:
            raise ValueError('not a trophy pack')
        count, size = struct.unpack_from('>II', head, 0x10)
        if count > 4096 or not 48 <= size <= 1024:
            raise ValueError('bad trophy pack index')
        table = f.read(count * size)
    entries = {}
    for i in range(len(table) // size):
        raw = table[i * size:(i + 1) * size]
        name = raw[:32].split(b'\0', 1)[0].decode('ascii', 'replace')
        entries[name] = struct.unpack_from('>QQ', raw, 32)
    return entries


def icons(game_dir, cache_dir):
    """{id: icon file} for the pack's TROPnnn.PNG, copied once into cache_dir; {} without a pack."""
    pack = Path(game_dir) / PACK
    try:
        entries = pack_entries(pack)
    except (OSError, ValueError, struct.error):
        return {}
    cache = Path(cache_dir)
    found = {}
    with open(pack, 'rb') as f:
        for name, (offset, size) in entries.items():
            if not (name.startswith('TROP') and name.endswith('.PNG') and name[4:7].isdigit()):
                continue
            target = cache / name
            if not target.is_file() or target.stat().st_size != size:
                f.seek(offset)
                data = f.read(size)
                try:
                    cache.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                except OSError:
                    continue
            found[int(name[4:7])] = target
    return found


def unlocked(user_dir):
    """[(id, unix time)] in unlock order, each id once."""
    seen, out = set(), []
    try:
        lines = (Path(user_dir) / 'trophies.txt').read_text(encoding='utf-8').splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[0] not in seen:
            seen.add(parts[0])
            try:
                out.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    return out


def summary(game_dir, user_dir, cache_dir, title_id):
    """{'total', 'unlocked': [{'id', 'name', 'grade', 'icon' (path), 'date'}], newest first}."""
    files = icons(game_dir, cache_dir)
    names = NAMES.get(title_id, [])
    rows = []
    for trophy, time in reversed(unlocked(user_dir)):
        grade, name = names[trophy] if trophy < len(names) else ('', f'#{trophy}')
        rows.append({'id': trophy, 'name': name, 'grade': grade, 'icon': files.get(trophy),
                     'date': datetime.datetime.fromtimestamp(time).strftime('%d.%m.%Y')})
    return {'total': len(files), 'unlocked': rows}
