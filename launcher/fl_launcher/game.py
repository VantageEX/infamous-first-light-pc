# SPDX-License-Identifier: GPL-2.0-or-later
"""What the launcher reads from the player's game folder: param.sfo, the PlayGo languages, art."""
import struct
from dataclasses import dataclass, field
from pathlib import Path

# inFAMOUS First Light dumps the launcher accepts without a warning. The core itself boots any
# dump; other ids get a "not tested" note.
KNOWN_TITLES = {'CUSA00897', 'CUSA00575'}

# Core switches the game runs with; the advanced box overrides them. The core's own defaults are
# not what iss_pc ships with: its launcher sets these (port/launcher/infamous_run.py DEFAULT_ENV,
# DEFAULT_SETTINGS and build_env), and run.py alone does not.
FIRST_LIGHT_ENV = {
    # Precise GPU readbacks. The core defaults to relaxed, where the CPU never sees what the GPU
    # wrote: First Light's emissive exposure sticks at its -9.5 EV floor, so the neon dash trail
    # is ~1000 times too dim and the city stays dark after loading a save.
    'TLG_READBACKS': '2',
    'TLG_EAGER_READBACK': '0',     # on, smoke and glow flicker in precise mode (iss_pc)
    'TLG_WRITE_MERGE': '1',        # game memcpy/memset into GPU memory without a GPU wait
    'TLG_PIPELINE_BINARIES': '0',  # VK_KHR_pipeline_binary crashes on the NVIDIA 595 driver
    'TLG_GC_BUDGET_MB': '3500',    # texture cache budget (iss_pc: 5600 runs out of 6 GB of VRAM)
    'TLG_FRAME_STATS': '1',        # FPS in the log every 5 s
    # The body of First Light's particle lists (both buffer sets), read by the GPU only: their
    # read faults stay out of precise mode (a little FPS). Their first page (the list header)
    # and the emitter counters after them stay precise: without those part of the dash effects
    # vanish. Fixed addresses: the game allocates them the same way every start.
    'TLG_RELAXED_RANGES': '106e159000-106e258000,106ff59000-1070058000',
    # The light glare reads two occlusion queries a frame. Mode 3 gives it 0/0 (yellow screens
    # and buildings), 2 a negative count (white-outs), 4 (iss_pc's real queries, the core's
    # default) yellow flickers and host waits. 5 is this port's fixed step.
    'TLG_OCCLUSION_MODE': '5',
}

# Test builds (settings.TEST_BUILD) add the core statistics the developer reads in the reports;
# the advanced box still overrides them.
TEST_BUILD_ENV = {'TLG_READBACK_STATS': '1', 'TLG_BUFFER_STATS': '1'}

# PS4 system language ids (sceSystemServiceParamGetInt LANG), as the core's TLG_LANGUAGE takes them.
LANGUAGES = {
    'ja': 0, 'en': 1, 'fr': 2, 'es': 3, 'de': 4, 'it': 5, 'nl': 6, 'pt_PT': 7, 'ru': 8, 'ko': 9,
    'zh_TW': 10, 'zh_CN': 11, 'fi': 12, 'sv': 13, 'da': 14, 'no': 15, 'pl': 16, 'pt_BR': 17,
    'en_GB': 18, 'tr': 19, 'es_LA': 20, 'ar': 21, 'fr_CA': 22, 'cs': 23, 'hu': 24, 'el': 25,
    'ro': 26, 'th': 27, 'vi': 28, 'id': 29,
}
LANGUAGE_NAMES = {
    'ja': '日本語', 'en': 'English (US)', 'fr': 'Français', 'es': 'Español', 'de': 'Deutsch',
    'it': 'Italiano', 'nl': 'Nederlands', 'pt_PT': 'Português (Portugal)', 'ru': 'Русский',
    'ko': '한국어', 'zh_TW': '繁體中文', 'zh_CN': '简体中文', 'fi': 'Suomi', 'sv': 'Svenska',
    'da': 'Dansk', 'no': 'Norsk', 'pl': 'Polski', 'pt_BR': 'Português (Brasil)',
    'en_GB': 'English (UK)', 'tr': 'Türkçe', 'es_LA': 'Español (Latinoamérica)', 'ar': 'العربية',
    'fr_CA': 'Français (Canada)', 'cs': 'Čeština', 'hu': 'Magyar', 'el': 'Ελληνικά',
    'ro': 'Română', 'th': 'ไทย', 'vi': 'Tiếng Việt', 'id': 'Bahasa Indonesia',
}


def read_sfo(path):
    """{key: value} of a param.sfo (strings and 32-bit integers)."""
    data = Path(path).read_bytes()
    if data[:4] != b'\0PSF' or len(data) < 20:
        raise ValueError('not a param.sfo')
    key_table, data_table, count = struct.unpack_from('<III', data, 8)
    values = {}
    for i in range(count):
        key_offset, fmt, length, _, data_offset = struct.unpack_from('<HHIII', data, 20 + i * 16)
        start = key_table + key_offset
        key = data[start:data.index(b'\0', start)].decode('utf-8', 'replace')
        raw = data[data_table + data_offset:data_table + data_offset + length]
        if fmt == 0x0404:
            values[key] = struct.unpack('<I', raw[:4])[0]
        else:
            values[key] = raw.split(b'\0', 1)[0].decode('utf-8', 'replace')
    return values


def playgo_languages(path):
    """Language ids the package has PlayGo chunks for (sce_sys/playgo-chunk.dat): the chunk
    count is at 10, the attribute table offset at 192, and each 32-byte attribute holds a 64-bit
    language mask at 16 (bit 63 - id). Masks of 0 or all ones mean "every language"."""
    data = Path(path).read_bytes()
    if data[:4] != b'plgo' or len(data) < 200:
        raise ValueError('not a PlayGo chunk file')
    count = struct.unpack_from('<H', data, 10)[0]
    table = struct.unpack_from('<I', data, 192)[0]
    if table + count * 32 > len(data):
        raise ValueError('PlayGo chunk table out of the file')
    ids = set()
    for i in range(count):
        mask = struct.unpack_from('<Q', data, table + i * 32 + 16)[0]
        if mask & 0xffffffff and not mask >> 32:
            mask = int.from_bytes(mask.to_bytes(8, 'little'), 'big')
        if mask not in (0, (1 << 64) - 1):
            ids |= {63 - bit for bit in range(64) if mask >> bit & 1}
    return ids


@dataclass
class GameInfo:
    folder: Path
    title: str = ''
    title_id: str = ''
    version: str = ''
    languages: list = field(default_factory=list)   # LANGUAGES keys, empty when unknown
    icon: Path | None = None
    background: Path | None = None

    @property
    def known(self):
        return self.title_id in KNOWN_TITLES


def inspect(folder):
    """GameInfo of a game folder; raises ValueError with the reason when it is not one."""
    folder = Path(folder)
    if not (folder / 'eboot.bin').is_file():
        raise ValueError('no_eboot')
    sfo = folder / 'sce_sys' / 'param.sfo'
    if not sfo.is_file():
        raise ValueError('no_sfo')
    values = read_sfo(sfo)
    info = GameInfo(folder, title=str(values.get('TITLE', '')), title_id=str(values.get('TITLE_ID', '')),
                    version=str(values.get('APP_VER', values.get('VERSION', ''))))
    try:
        ids = playgo_languages(folder / 'sce_sys' / 'playgo-chunk.dat')
        info.languages = [key for key, code in LANGUAGES.items() if code in ids]
    except (OSError, ValueError, struct.error):
        pass
    for name, attr in (('icon0.png', 'icon'), ('pic1.png', 'background'), ('pic0.png', 'background')):
        art = folder / 'sce_sys' / name
        if art.is_file() and getattr(info, attr) is None:
            setattr(info, attr, art)
    return info


def renderer_v2_probe(port):
    """tlg-probe-v2.exe (renderer v2, port/gpu_v2) when this build has it: bin/ (packaged
    layouts that use it) or out/ (dev builds and this port's packages), else None."""
    for folder in ('bin', 'out'):
        probe = Path(port) / folder / 'tlg-probe-v2.exe'
        if probe.is_file():
            return probe
    return None


def renderer_env(renderer, game_dir, port):
    """The environment renderer v2 starts with, as iss_pc's launcher passes it
    (infamous_run.renderer_v2_env): its defaults file, the game folder its shader index reads,
    unified memory off on Windows (measured: 11-20 FPS with it on, ~40-45 off, RTX 3060);
    TLG_PROBE picks the v2 probe. Nothing for the core's own renderer.
    Native passes off, unlike iss_pc: in First Light they gained ~2 FPS and crashed in NVIDIA's
    driver (nvoglv64.dll) after 4 minutes, and their pause guard only knows Second Son's eboot."""
    probe = renderer_v2_probe(port) if renderer == '2' else None
    if probe is None:
        return {}
    env = {}
    try:
        for line in (Path(port) / 'gpu_v2' / 'defaults.env').read_text(encoding='utf-8').splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and key and not key.startswith('#'):
                env[key] = value
    except OSError:
        pass
    env.update({'TLG_PROBE': str(probe), 'ISS_GAME_DIR': str(game_dir),
                'ISS_NATIVE_PASSES': '0', 'ISS_UNIFIED_MEMORY': '0'})
    return env
