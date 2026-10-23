# SPDX-License-Identifier: GPL-2.0-or-later
"""Launcher settings (launcher.json) and the core's tlg.ini, both in the data folder."""
import json
import os
import sys
from pathlib import Path

# Packaged (PyInstaller onedir): <dist>/inFAMOUS First Light.exe beside <dist>/port and <dist>/data.
# Developing: the repository holds port/ and data/.
FROZEN = getattr(sys, 'frozen', False)
REPO = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[2]
PORT = REPO / 'port'
# A TEST_BUILD file beside the executable (or in the repository root) makes a build for testers:
# extra core statistics in the log and the report reminder in the interface.
TEST_BUILD = (REPO / 'TEST_BUILD').is_file()
VERSION_FILE = REPO / 'VERSION'


def writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / '.write-test'
        probe.write_bytes(b'')
        probe.unlink()
        return True
    except OSError:
        return False


def data_dir():
    """FL_DATA_DIR, else data/ beside the launcher executable (packaged) or in the repository.
    A package unzipped where it cannot write (Program Files) keeps its data in
    %LOCALAPPDATA%/inFAMOUS First Light PC instead (Linux: ~/.local/share)."""
    if os.environ.get('FL_DATA_DIR'):
        return Path(os.environ['FL_DATA_DIR'])
    beside = REPO / 'data'
    if not FROZEN or writable(beside):
        return beside
    if os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA')
    else:
        local = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(local) / 'inFAMOUS First Light PC' if local else beside


# tlg.ini keys the core reads at start (port/gpu/shim/tlg_settings.cpp) and their defaults.
# Resolution and engine frame rate patches exist only for Second Son, so they are left out.
CORE_DEFAULTS = {
    'display_mode': 'windowed',   # windowed / borderless / fullscreen
    'vsync': '1',
    'fps_limit': '60',            # 30, 60, 120, 240 or 0 (unlimited)
    'fsr1': '0',
    'rcas': '0',
    'show_fps': '0',
    'async_shaders': '1',
    'deferred_readback': '1',
    'stale_readback': '1',
}
DISPLAY_MODES = ('windowed', 'borderless', 'fullscreen')
FPS_LIMITS = ('30', '60', '120', '240', '0')

LAUNCHER_DEFAULTS = {
    'game_dir': '',
    'game_language': 'auto',
    'ui_language': 'auto',
    'extra_env': '',   # KEY=VALUE lines for testing core switches
    'renderer': '2',   # 2: renderer v2 (port/gpu_v2) when built, else 1: the core's own renderer
}


def _read_ini(path):
    values = {}
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError:
        return values
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def _write_ini(path, updates):
    """Rewrites the keys in `updates`, appends the missing ones and keeps every other line (the
    in-game menu writes the same file)."""
    path = Path(path)
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        lines = ['# inFAMOUS First Light settings (launcher and in-game menu)']
    done, out = set(), []
    for line in lines:
        key = line.split('=', 1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if key in updates:
            out.append(f'{key}={updates[key]}')
            done.add(key)
        else:
            out.append(line)
    out += [f'{key}={value}' for key, value in updates.items() if key not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')


class Settings:
    def __init__(self, folder=None):
        self.folder = Path(folder or data_dir())
        self.launcher_path = self.folder / 'launcher.json'
        self.ini_path = self.folder / 'tlg.ini'
        self.launcher = dict(LAUNCHER_DEFAULTS)
        try:
            stored = json.loads(self.launcher_path.read_text(encoding='utf-8'))
            self.launcher.update({k: v for k, v in stored.items() if k in LAUNCHER_DEFAULTS})
        except (OSError, ValueError):
            pass
        ini = _read_ini(self.ini_path)
        self.core = {key: ini.get(key, value) for key, value in CORE_DEFAULTS.items()}
        if self.core['display_mode'].lower() not in DISPLAY_MODES:
            self.core['display_mode'] = CORE_DEFAULTS['display_mode']
        if self.core['fps_limit'] not in FPS_LIMITS:
            self.core['fps_limit'] = CORE_DEFAULTS['fps_limit']

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.launcher_path.write_text(json.dumps(self.launcher, indent=2, ensure_ascii=False), encoding='utf-8')
        _write_ini(self.ini_path, self.core)


# Core switches that run.py takes from tlg.ini (scripts/graphics_options.py) and so would override
# the same variable in the environment: the launcher writes them to tlg.ini instead.
# (async_shaders has its own checkbox.)
INI_SWITCHES = {'TLG_DEFERRED_READBACK': 'deferred_readback', 'TLG_STALE_READBACK': 'stale_readback'}


def parse_env(text):
    """{KEY: value} from KEY=VALUE lines; blank lines and # comments are skipped."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if key.strip():
                values[key.strip()] = value.strip()
    return values
