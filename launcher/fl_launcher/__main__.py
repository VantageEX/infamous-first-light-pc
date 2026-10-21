# SPDX-License-Identifier: GPL-2.0-or-later
import os
import sys
from pathlib import Path


def _std_streams():
    """A windowed packaged exe starts without sys.stdout/stderr even when its output goes to a
    pipe (port/run.py reads the scripts' output): take the process's own handles."""
    for name, number in (('stdout', -11), ('stderr', -12)):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if os.name == 'nt':
            import ctypes
            import msvcrt
            handle = ctypes.windll.kernel32.GetStdHandle(number)
            if handle not in (0, -1, None):
                try:
                    fd = msvcrt.open_osfhandle(handle, os.O_WRONLY)
                    stream = open(fd, 'w', encoding='utf-8', errors='replace', buffering=1)
                except OSError:
                    stream = None
        setattr(sys, name, stream or open(os.devnull, 'w'))


def run_script():
    """`<launcher> --script path args...`: the packaged launcher stands in for python.exe for
    port/run.py and its scripts (run.py calls sys.executable --script when sys.frozen is set)."""
    import runpy
    path = sys.argv[2]
    sys.argv = [path, *sys.argv[3:]]
    sys.path.insert(0, str(Path(path).resolve().parent))
    _std_streams()
    runpy.run_path(path, run_name='__main__')
    return 0


if len(sys.argv) > 2 and sys.argv[1] == '--script':
    sys.exit(run_script())
if len(sys.argv) > 1 and sys.argv[1] == '--run':
    # The in-game restart (run.py's TLG_RESTART_COMMAND: `<launcher> --run --after <pid>`) starts
    # port/run.py again, not a second launcher.
    sys.argv[1:2] = ['--script', str(Path(sys.executable).resolve().parent / 'port' / 'run.py')]
    sys.exit(run_script())

try:
    from .app import main
except ImportError:   # started as a plain script (the PyInstaller entry point)
    from fl_launcher.app import main

sys.exit(main())
