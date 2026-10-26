#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""The profiling session (docs/profiling.md) for run.py.

TLG_SESSION_DIR=<folder> (or "auto": <data>/sessions/<date-time>) records one play session into
that folder:
  session.tracy   the Tracy capture (session-2.tracy... after an in-game restart) (tracy-capture, with the profiling build tlg-probe*-profile):
                  every thread's zones, the GPU's, locks, plots, messages
  timeline.csv    renderer v2's event timeline (ISS_TIMELINE), steady clock
  sensors.csv     clocks, loads, power, temperatures, threads (gpu/shim/tlg_profile.cpp)
  session.json    what ran: the time, the probe, the switches
  functions.bin   the game image's functions (its .eh_frame), for the guest samples' names:
                  "TLGFUNC1", u32 count, count x (u32 start, u32 end) image offsets
  guest_samples.bin, modules.txt  the guest sampler's (src/runtime_sampler.c)
Without the profiling build (or without tracy-capture) the session still gets the other files.
TLG_SESSION_CAPTURE=0: no tracy-capture; a Tracy viewer connects to the game (port 8086) instead.
TLG_SESSION_SECONDS=N: the game is stopped after N seconds (unattended runs).
"""
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

# The switches worth keeping with a session (the rest of the environment is the user's).
_KEPT = ('TLG_', 'ISS_', 'TRACY_', 'RADV_', 'MESA_', 'DXVK_', 'VKD3D_', 'SDL_')


def session_dir(env, data):
    """The session folder TLG_SESSION_DIR names, made; None without one."""
    value = env.get('TLG_SESSION_DIR', '').strip()
    if not value or value == '0':
        return None
    folder = Path(data) / 'sessions' / time.strftime('%Y%m%d-%H%M%S') if value in ('1', 'auto') else Path(value)
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def profile_probe(probe):
    """The profiling build beside `probe` (tlg-probe-v2.exe -> tlg-probe-v2-profile.exe), or None."""
    candidate = probe.with_name(probe.stem + '-profile' + probe.suffix)
    return candidate if candidate.is_file() else None


def write_functions(boot_linked, path):
    """functions.bin from the linked image run.py made (out/boot-linked.bin); False without it."""
    try:
        import isa_rewrite
        import red_zone
        segments, image = isa_rewrite.read_boot_linked(Path(boot_linked).read_bytes())
        functions = red_zone.find_functions(image, segments)
    except (OSError, ValueError, ImportError, struct.error):
        return False
    with open(path, 'wb') as f:
        f.write(b'TLGFUNC1' + struct.pack('<I', len(functions)))
        f.write(b''.join(struct.pack('<II', start, end) for start, end in functions))
    return True


def find_capture(probe):
    """tracy-capture beside the probe, else in tools/."""
    name = 'tracy-capture' + ('.exe' if os.name == 'nt' else '')
    for folder in (probe.parent, probe.parent.parent / 'tools', Path(__file__).resolve().parent.parent / 'tools'):
        if (folder / name).is_file():
            return folder / name
    return None


def start(env, data, probe, creationflags=0):
    """Prepares the session when TLG_SESSION_DIR asks for one: returns (probe to run, the capture
    process or None, the folder or None)."""
    folder = session_dir(env, data)
    if folder is None:
        return probe, None, None
    env['TLG_SESSION_DIR'] = str(folder)
    env['ISS_TIMELINE'] = '1'  # gpu_v2/defaults.env has it off
    env.setdefault('ISS_TIMELINE_LOG', str(folder / 'timeline.csv'))
    profiled = profile_probe(probe)
    capture = None
    if profiled is None:
        print(f'Profile: no profiling build beside {probe.name}: no Tracy capture (sensors and timeline only)', flush=True)
    elif env.get('TLG_SESSION_CAPTURE') == '0':
        # A Tracy viewer (or a capture started by hand) connects to the game itself.
        probe = profiled
        print('Profile: TLG_SESSION_CAPTURE=0: connect a Tracy viewer to this computer, port 8086', flush=True)
    else:
        probe = profiled
        tool = find_capture(probe)
        if tool is None:
            print('Profile: tracy-capture not found: no Tracy capture', flush=True)
        else:
            # "Apply and restart" starts the game again into the same folder: a file per start.
            tracy = folder / 'session.tracy'
            for n in range(2, 100):
                if not tracy.exists():
                    break
                tracy = folder / f'session-{n}.tracy'
            log = open(tracy.with_suffix('.log'), 'w', encoding='utf-8')
            # The game's client listens on 127.0.0.1:8086; the capture connects once it is up and
            # ends when the game exits.
            capture = subprocess.Popen([str(tool), '-o', str(tracy), '-a', '127.0.0.1'],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=creationflags)
            # The game waits at its exit until the capture has every event.
            env['TRACY_NO_EXIT'] = '1'
            capture.tracy = tracy
            print(f'Profile: Tracy capture to {tracy}', flush=True)
    write_functions(Path(data) / 'out' / 'boot-linked.bin', folder / 'functions.bin')
    (folder / 'session.json').write_text(json.dumps({
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'probe': probe.name,
        'tracy': capture is not None,
        'platform': sys.platform,
        'switches': {k: v for k, v in sorted(env.items()) if k.startswith(_KEPT)},
    }, indent=2), encoding='utf-8')
    print(f'Profile: session in {folder}', flush=True)
    return probe, capture, folder


def time_limit(env, folder):
    """TLG_SESSION_SECONDS: how long a session runs before the game is stopped (unattended tests,
    benchmarks); None without one."""
    value = env.get('TLG_SESSION_SECONDS', '')
    return float(value) if folder is not None and value.replace('.', '', 1).isdigit() and float(value) > 0 else None


def finish(capture, folder, timeout=120):
    """After the game: lets the capture write its file (it ends once the game has gone)."""
    if capture is None:
        return
    try:
        capture.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        capture.terminate()
        print('Profile: tracy-capture did not end; stopped', flush=True)
    tracy = capture.tracy
    if tracy.is_file():
        print(f'Profile: {tracy} ({tracy.stat().st_size / 1e6:.1f} MB)', flush=True)
    else:
        print(f'Profile: tracy-capture ended (code {capture.returncode}) without {tracy.name}; '
              f'see {tracy.with_suffix(".log").name}', flush=True)
