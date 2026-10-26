"""Profiling sessions from the launcher (port/docs/profiling.md).

A session folder per start in <data>/sessions (port/scripts/profile_session.py fills it: the Tracy
capture, the renderer v2 timeline, the sensors, the guest samples); after the game the launcher adds
the run's log and packs the folder into <name>.zip beside it, the one file a tester sends.
Pad routes for benchmarks live in <data>/routes: recorded with TLG_PAD_RECORD (F9, or for a set
time from the start of gameplay), replayed with TLG_PAD_REPLAY (the replay presses Cross until
gameplay, then plays the route), so every benchmark run takes the same path.
"""
import os
import shutil
import time
import zipfile
from pathlib import Path

# Benchmark and route lengths offered, in seconds; 0: until the game is closed (routes: F9).
SECONDS = ('0', '60', '120', '180', '300')


def sessions_dir(data_dir):
    return Path(data_dir) / 'sessions'


def routes_dir(data_dir):
    return Path(data_dir) / 'routes'


def list_routes(data_dir):
    """The recorded routes, newest first (file names)."""
    folder = routes_dir(data_dir)
    if not folder.is_dir():
        return []
    files = [p for p in folder.glob('*.txt') if p.is_file() and p.stat().st_size > 0]
    return [p.name for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]


def profile_build(probe):
    """The profiling build beside a renderer's probe (tlg-probe-v2 -> tlg-probe-v2-profile)."""
    if not probe:
        return None
    probe = Path(probe)
    candidate = probe.with_name(probe.stem + '-profile' + probe.suffix)
    return candidate if candidate.is_file() else None


def start_env(data_dir, record, route, seconds, record_route, stamp=None):
    """The variables of a start (run.py and the runtime read them) and the session folder (None
    without a session). record: write a session; route: a file of routes_dir to replay; seconds:
    how long the session (or the route recording) runs, 0 until the game is closed; record_route:
    record a new route instead (F9 starts and stops it when seconds is 0)."""
    env = {}
    stamp = stamp or time.strftime('%Y%m%d-%H%M%S')
    seconds = int(seconds) if str(seconds).isdigit() else 0
    session = None
    if record_route:
        routes = routes_dir(data_dir)
        routes.mkdir(parents=True, exist_ok=True)
        env['TLG_PAD_RECORD'] = str(routes / f'route-{stamp}.txt')
        if seconds:
            env['TLG_PAD_RECORD_SECONDS'] = str(seconds)
    elif route and (routes_dir(data_dir) / route).is_file():
        env['TLG_PAD_REPLAY'] = str(routes_dir(data_dir) / route)
        env['TLG_PAD_REPLAY_AUTO'] = '1'
    if record:
        session = sessions_dir(data_dir) / (stamp + ('-benchmark' if 'TLG_PAD_REPLAY' in env else ''))
        session.mkdir(parents=True, exist_ok=True)
        env['TLG_SESSION_DIR'] = str(session)
        if seconds and not record_route:
            env['TLG_SESSION_SECONDS'] = str(seconds)
        if 'TLG_PAD_REPLAY' in env:
            (session / 'route.txt').write_bytes((routes_dir(data_dir) / route).read_bytes())
    return env, session


# Already compressed: stored as they are (zip's deflate would only cost time).
_STORED = {'.tracy', '.zip', '.png', '.jpg'}


def pack(session, log_path=None):
    """Adds the run's log to the session folder and packs it into <folder>.zip; the zip's path."""
    session = Path(session)
    if log_path and Path(log_path).is_file():
        shutil.copyfile(log_path, session / 'run.log')
    archive = session.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(session.rglob('*')):
            if path.is_file():
                kind = zipfile.ZIP_STORED if path.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED
                z.write(path, f'{session.name}/{path.relative_to(session).as_posix()}', compress_type=kind)
    return archive


def size_text(path):
    return f'{os.path.getsize(path) / 1e6:.0f} MB'
