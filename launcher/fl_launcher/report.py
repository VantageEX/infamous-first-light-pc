# SPDX-License-Identifier: GPL-2.0-or-later
"""The report a player sends to the developer: recent run logs, the system, the settings.
Full user paths stay out of it: the game, data, install and home folders are replaced by placeholders."""
import json
import os
import platform
import re
import time
import zipfile
from pathlib import Path

from .settings import REPO

RUN_LOGS = 3
GPU_LINE = re.compile(r'^(?:\[\s*[\d.]+\] )?(GPU \d+:|GPU: using)')


def _windows():
    text = platform.platform()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Windows NT\CurrentVersion') as key:
            build = winreg.QueryValueEx(key, 'CurrentBuild')[0]
            try:
                display = winreg.QueryValueEx(key, 'DisplayVersion')[0]
            except OSError:
                display = ''
        # ProductName still says "Windows 10" on Windows 11; the build tells them apart.
        name = 'Windows 11' if int(build) >= 22000 else 'Windows 10' if int(build) >= 10240 else 'Windows'
        text = f'{name} {display} (build {build}) · {text}'.replace('  ', ' ')
    except (ImportError, OSError, ValueError):
        pass
    return text


def _linux():
    """'SteamOS 3.7.13 · Linux-6.11...' from os-release, the kernel after it."""
    name = ''
    try:
        for line in Path('/etc/os-release').read_text(encoding='utf-8').splitlines():
            if line.startswith('PRETTY_NAME='):
                name = line.split('=', 1)[1].strip().strip('"')
    except OSError:
        pass
    return f'{name} · {platform.platform()}' if name else platform.platform()


def _cpu():
    if os.name != 'nt':
        try:
            for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or '?'
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
    except (ImportError, OSError):
        return platform.processor() or '?'


def _ram_gb():
    if os.name != 'nt':
        try:
            for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024 ** 2
        except (OSError, ValueError, IndexError):
            pass
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys / 1024 ** 3


def _private(text, game_dir, data_dir=None):
    """The text with the game, data, install and home folders replaced (any letter case, either
    slash), the longest first: the game or data folder may sit inside the install or home folder."""
    folders = [(str(game_dir or ''), f'<game>/{Path(game_dir).name}' if game_dir else ''),
               (str(data_dir or ''), '<data>'),
               (str(REPO), '<install>'),
               (str(Path.home()), '%USERPROFILE%' if os.name == 'nt' else '~')]
    for path, placeholder in sorted(folders, key=lambda folder: -len(folder[0])):
        if path:
            pattern = re.escape(str(path).replace('/', '\\')).replace(r'\\', r'[\\/]')
            text = re.sub(pattern, lambda _: placeholder, text, flags=re.IGNORECASE)
    return text


def system_text(logs):
    """system.txt: Windows, CPU, RAM and the GPU lines the core printed in the newest log."""
    ram = _ram_gb()
    lines = [f'Windows: {_windows()}' if os.name == 'nt' else f'Linux: {_linux()}',
             f'CPU: {_cpu()}',
             f'Logical processors: {os.cpu_count() or "?"}',
             f'RAM: {ram:.1f} GB' if ram else 'RAM: ?']
    gpu = []
    if logs:
        for line in logs[-1].read_text(encoding='utf-8', errors='replace').splitlines():
            if GPU_LINE.match(line):
                gpu.append(line.rstrip())
    lines += gpu or ['GPU: (no GPU lines in the newest run log)']
    return '\n'.join(lines) + '\n'


def session_header(version, game, core, renderer, user_env):
    """The first lines of a run log: the launcher's version, the system, the game and the
    settings the run uses, so a log alone tells what ran where."""
    ram = _ram_gb()
    system = _windows() if os.name == 'nt' else _linux()
    lines = [f"=== inFAMOUS First Light PC {version or '(no version)'} · {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
             f'System: {system}',
             f'CPU: {_cpu()} · {os.cpu_count() or "?"} logical processors · RAM {ram:.1f} GB' if ram
             else f'CPU: {_cpu()} · {os.cpu_count() or "?"} logical processors',
             f'Game: {game.title_id} {game.version}' if game else 'Game: (none)',
             f'Renderer: {renderer} · settings: ' + ', '.join(f'{key}={value}' for key, value in sorted(core.items()))]
    if user_env:
        lines.append("Player's variables: " + ', '.join(f'{key}={value}' for key, value in user_env.items()))
    return '\n'.join(lines)


def create(data_dir, launcher_settings, game=None, version_file=None):
    """Writes <data>/reports/report-YYYYMMDD-HHMMSS.zip and returns its path. `launcher_settings`
    is launcher.json's dict, `game` the GameInfo (or None), `version_file` the package's VERSION."""
    data_dir = Path(data_dir)
    game_dir = launcher_settings.get('game_dir', '')
    logs = sorted((data_dir / 'logs').glob('run-*.log'))[-RUN_LOGS:]
    out = data_dir / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    path = out / time.strftime('report-%Y%m%d-%H%M%S.zip')

    launcher = dict(launcher_settings)
    launcher['game_dir'] = Path(game_dir).name if game_dir else ''
    if game is not None:
        launcher['title_id'] = game.title_id
        launcher['game_version'] = game.version
    launcher['extra_env'] = _private(launcher.get('extra_env', ''), game_dir, data_dir)

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for log in logs:
            archive.writestr(f'logs/{log.name}', _private(log.read_text(encoding='utf-8', errors='replace'), game_dir, data_dir))
        archive.writestr('system.txt', _private(system_text(logs), game_dir, data_dir))
        ini = data_dir / 'tlg.ini'
        if ini.is_file():
            archive.writestr('tlg.ini', _private(ini.read_text(encoding='utf-8', errors='replace'), game_dir, data_dir))
        archive.writestr('launcher.json', json.dumps(launcher, indent=2, ensure_ascii=False))
        if version_file and Path(version_file).is_file():
            archive.writestr('VERSION', Path(version_file).read_text(encoding='utf-8', errors='replace'))
    return path
