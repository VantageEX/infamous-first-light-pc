#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Prepares the game image and starts tlg-probe (tlg-probe.exe on Windows): the launcher's way in,
on Windows and on Linux (the Steam Deck) alike; run.sh is the upstream Linux script.

Same environment variables as run.sh (TLG_GAME_DIR, TLG_DATA_DIR, TLG_PATCHES, TLG_MODS_*,
TLG_USER_DIR, ...). Extra arguments go to tlg-probe.
The in-game "Apply and restart" runs this script again through TLG_RESTART_COMMAND;
`--after PID` waits for the previous game process to end first (its GPU device and memory).
"""
import ctypes
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

PORT = Path(__file__).resolve().parent
WINDOWS = os.name == 'nt'
EXE = '.exe' if WINDOWS else ''


def _data_dir(data):
    """The saves folder when ISS_USER_DIR is not set: the per-user pointer file of the launcher
    (launcher/userdata.py), else <data>/user."""
    try:
        sys.path.insert(0, str(PORT / 'launcher'))
        import userdata
        return userdata.user_dir(data)
    except ImportError:
        return data / 'user'


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def wait_for(pid):
    if not WINDOWS:
        # Linux restarts exec this script in the game's own process (probe.c runtime_restart):
        # only a different process is waited for.
        deadline = time.monotonic() + 30
        while pid != os.getpid() and time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.1)
        return
    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x00100000
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if handle:
        kernel32.WaitForSingleObject(handle, 30000)
        kernel32.CloseHandle(handle)


def no_console():
    """Started without a console (the launcher): console programs must not open their own."""
    if not WINDOWS:
        return 0
    return 0 if ctypes.windll.kernel32.GetConsoleWindow() else subprocess.CREATE_NO_WINDOW


def linux_libraries(env, probe):
    """Linux package: the libraries built with the port (SDL3, FFmpeg, ...) are in out/lib beside
    tlg-probe. The frozen launcher runs with PyInstaller's LD_LIBRARY_PATH (its _internal folder),
    which the game must not inherit: the one the launcher started with comes back."""
    frozen = getattr(sys, 'frozen', False)
    original = env.pop('LD_LIBRARY_PATH_ORIG', None) if frozen else env.get('LD_LIBRARY_PATH')
    if frozen and original is None:
        env.pop('LD_LIBRARY_PATH', None)
    bundled = probe.parent / 'lib'
    paths = [str(bundled)] if bundled.is_dir() else []
    if original:
        paths.append(original)
    if paths:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(paths)


def parse_cpu_features(output):
    """Instruction sets `tlg-probe --cpu-features` reports missing ([] when complete)."""
    for line in output.splitlines():
        if line.startswith('CPU: missing '):
            return [name.strip() for name in line[len('CPU: missing '):].split(',') if name.strip()]
    return []


def red_zone_wanted(env):
    """Windows red-zone protection (scripts/red_zone.py): default on Windows, TLG_RED_ZONE_REWRITE=0
    off; on Linux only =1 (tests)."""
    policy = env.get('ISS_RED_ZONE_REWRITE') or env.get('TLG_RED_ZONE_REWRITE', '')
    return policy != '0' if os.name == 'nt' else policy == '1'


def isa_rewrite_wanted(missing, env):
    """TLG_ISA_REWRITE=0 disables the ISA rewrite table; 1 or a list of sets ("sse4a", "movbe,bmi1")
    forces it (tlg-probe rewrites those sets); otherwise the CPU decides."""
    policy = env.get('ISS_ISA_REWRITE') or env.get('TLG_ISA_REWRITE', '')
    if policy:
        return policy != '0'
    return bool(missing)


def run_script(name, *args, capture=False, check=True):
    # Inside the packaged launcher executable (PyInstaller) the executable runs scripts itself.
    script = ['--script'] if getattr(sys, 'frozen', False) else []
    command = [sys.executable, *script, str(PORT / 'scripts' / name), *map(str, args)]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    if capture:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                                encoding='utf-8', env=env, creationflags=no_console())
        if not check:
            return result.returncode, result.stdout
        if result.returncode:
            sys.exit(result.returncode)
        return result.stdout
    # stdin given: the output handles are passed explicitly (a windowed launcher child would
    # get none otherwise).
    result = subprocess.run(command, stdin=subprocess.DEVNULL, env=env, creationflags=no_console())
    if result.returncode:
        sys.exit(result.returncode)
    return None


def build_job_file(game, jobs, out, env):
    """The first-start pipeline job file (`<user>/cache/<title id>-jobs.bin`, both renderers read
    it): made once from the game's own .xpps packages by tools/xpps (scripts/build_jobs.py, about
    a minute), so that the renderer compiles the game's graphics pipelines before it starts and
    not while it plays. TLG_BUILD_JOBS=0 skips it; a failure only costs the stutter it saves."""
    packages = game / 'art' / 'cache'
    if env.get('TLG_BUILD_JOBS') == '0' or jobs.is_file() or not any(packages.glob('*.xpps')):
        return
    jobs.parent.mkdir(parents=True, exist_ok=True)
    index = out / 'xpps-index.sqlite'
    temp = jobs.with_suffix('.tmp')
    print('Shader jobs: reading the game files for the first-start compilation (once)', flush=True)
    script = ['--script'] if getattr(sys, 'frozen', False) else []
    command = [sys.executable, *script, str(PORT / 'scripts' / 'build_jobs.py'), str(packages), str(temp),
               '--index', str(index), '--nice']
    started = time.monotonic()
    # One log line per tenth of each phase: the generator prints a PROGRESS line per package.
    shown = {}
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace',
                          env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, creationflags=no_console()) as child:
        for line in child.stdout:
            parts = line.split()
            if len(parts) == 4 and parts[0] == 'PROGRESS' and parts[2].isdigit() and parts[3].isdigit():
                tenth = int(parts[2]) * 10 // max(1, int(parts[3]))
                if shown.get(parts[1]) != tenth:
                    shown[parts[1]] = tenth
                    print(f'Shader jobs: {parts[1]} {tenth * 10}%', flush=True)
            elif line.strip():
                print(f'Shader jobs: {line.rstrip()}', flush=True)
    for leftover in (index, index.with_name(index.name + '.tmp')):
        leftover.unlink(missing_ok=True)
    if child.returncode or not temp.is_file():
        temp.unlink(missing_ok=True)
        print(f'Shader jobs: not built (exit {child.returncode}); the game compiles pipelines as it draws', flush=True)
        return
    temp.replace(jobs)
    print(f'Shader jobs: {jobs.name} in {time.monotonic() - started:.0f} s', flush=True)


def find_executable(name):
    """bin/ (packaged), else out/ (built with build.sh)."""
    for candidate in (PORT / 'bin' / name, PORT / 'out' / name):
        if candidate.is_file():
            return candidate
    return None


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--after':
        wait_for(int(args[1]))
        args = args[2:]
    # As run.sh: relative paths (TLG_GAME_DIR, TLG_DATA_DIR, fsr4_shaders/) are from the port.
    os.chdir(PORT)
    env = os.environ
    data = Path(env.get('TLG_DATA_DIR', PORT))
    out = data / 'out'
    out.mkdir(parents=True, exist_ok=True)
    env.setdefault('TLG_CONFIG', str(data / 'tlg.ini'))
    # Values this script derived on the previous start ("Apply and restart" inherits them) are
    # derived again from the new options.
    for key in env.pop('TLG_DERIVED', '').split(','):
        env.pop(key, None)
    derived = []
    sys.path.insert(0, str(PORT / 'scripts'))
    import graphics_options
    graphics, reverted = graphics_options.prepare(env['TLG_CONFIG'], out / 'graphics_state.json')
    if reverted:
        print('Graphics: the last restart never reached the game; previous options restored', flush=True)
    env.update(graphics_options.environment(graphics))
    if not env.get('TLG_FSR411_DIR') and not (PORT / 'fsr4_411').is_dir() and (data / 'fsr4_411').is_dir():
        env['TLG_FSR411_DIR'] = str(data / 'fsr4_411')

    probe = Path(env['TLG_PROBE']) if env.get('TLG_PROBE') else find_executable('tlg-probe' + EXE)
    if not probe or not probe.is_file():
        fail(f'tlg-probe{EXE} not found: build it with build.sh or use a packaged build.')
    # The overlay's own faces where Segoe UI is missing (Linux): Open Sans and the Fluent icons.
    if (PORT / 'fonts').is_dir():
        env.setdefault('TLG_FONT_DIR', str(PORT / 'fonts'))
    if not WINDOWS:
        linux_libraries(env, probe)
    # Development builds run from the MSYS2 tree: its CLANG64 DLLs (SDL3, FFmpeg, ...).
    elif not (probe.parent / 'SDL3.dll').is_file():
        clang64 = Path(env.get('MSYS2_ROOT', r'C:\msys64')) / 'clang64' / 'bin'
        if clang64.is_dir():
            env['PATH'] = f'{clang64}{os.pathsep}{env.get("PATH", "")}'
    # Instruction sets this CPU lacks (MOVBE, BMI1, LZCNT): the ISA rewrite covers them; exit 20
    # is a set nothing covers.
    features = subprocess.run([str(probe), '--cpu-features'], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              text=True, creationflags=no_console())
    if features.returncode == 20:
        print(features.stdout, end='', flush=True)
        sys.exit(20)
    missing = parse_cpu_features(features.stdout)

    game = Path(env.get('TLG_GAME_DIR', PORT.parent / 'game' / 'CUSA00223'))
    if not (game / 'eboot.bin').is_file():
        fail(f'No eboot.bin in {game} (set TLG_GAME_DIR).')
    original_game = game.resolve()
    game = Path(run_script('mods.py', game, '--out', out,
                           '--mods-dir', env.get('TLG_MODS_DIR', data / 'mods'),
                           '--config', env.get('TLG_MODS_CONFIG', data / 'mods.json'),
                           '--enabled', env.get('TLG_MODS_ENABLED', '1'), capture=True).strip())
    mod_view = game if game.resolve() != original_game else None
    try:
        run_script('prepare.py', game, '--out', out)
        run_script('link_libc.py', game, '--out', out)
        run_script('link_modules.py', game, '--out', out)
        # A failed rewrite prints its reason and the game starts without the table.
        isa_args = []
        red_zone = red_zone_wanted(env)
        if isa_rewrite_wanted(missing, env) or red_zone:
            code, output = run_script('isa_rewrite.py', out / 'boot-linked.bin', '--out', out,
                                      *(['--red-zone'] if red_zone else []), capture=True, check=False)
            print(output, end='', flush=True)
            if code == 0:
                isa_args = ['--isa-rewrite', str(out / 'isa-rewrite.bin')]
        run_script('content_profile.py', game, '--out', out, '--sku', env.get('TLG_CONTENT_SKU', 'full'))

        from patches import game_patchable, game_profile, game_title_id, patch_requirements, selected_patches
        title_id = game_title_id(game)
        profile = game_profile(title_id)
        # The switches this game was measured with (scripts/game_defaults.py), under the ones set.
        import game_defaults
        if defaults := game_defaults.apply(env, title_id):
            derived += defaults
            print(f'Game defaults for {title_id}: ' + ' '.join(f'{k}={env[k]}' for k in defaults), flush=True)
        patched = bool(profile) and (game_patchable(game) or bool(env.get('TLG_FORCE_PATCHES')))
        engine_fps = 60
        # The window shows the game's icon (BMP: the format SDL loads without extra libraries).
        try:
            from PIL import Image
            with Image.open(game / 'sce_sys' / 'icon0.png') as icon:
                icon.convert('RGBA').resize((64, 64), Image.LANCZOS).save(out / 'window_icon.bmp')
            env.setdefault('TLG_WINDOW_ICON', str(out / 'window_icon.bmp'))
        except (ImportError, OSError):
            pass
        # The temporal upscalers and motion vectors are not calibrated for this game yet; linear
        # image readbacks fix its corrupted textures.
        env.setdefault('TLG_UPSCALER', 'none')
        env.setdefault('TLG_READBACK_LINEAR', '1')
        if patched:
            # The patch notes give the direct memory and VBlank rate the selected patches need.
            names = selected_patches(profile[1], profile[2], env.get('TLG_PATCHES', ''), env.get('TLG_PATCHES_ONLY') == '1')
            # tlg.ini decides the resolution and frame rate patches (in-game menu, launcher).
            names = graphics_options.patch_names(names, graphics)
            env['TLG_PATCHES'] = ';'.join(names)
            env['TLG_PATCHES_ONLY'] = '1'
            engine_fps = graphics_options.applied_engine_fps(names)
            dmem, vblank = patch_requirements(profile[1], names, profile[2])
            for key, value in (('TLG_DMEM_MB', dmem), ('TLG_VBLANK_HZ', vblank)):
                if value and key not in env:
                    env[key] = str(value)
                    derived.append(key)
        run_script('patches.py', '--out', out, '--extra', env.get('TLG_PATCHES', ''), '--game-dir', game,
                   '--patches-dir', env.get('TLG_PATCHES_DIR', data / 'patches'),
                   '--patches-config', env.get('TLG_PATCHES_CONFIG', data / 'patches.json'))
        if 'TLG_VBLANK_HZ' not in env:
            env['TLG_VBLANK_HZ'] = '60'
            derived.append('TLG_VBLANK_HZ')
        # Without the frame rate patch the engine steps 1/60 s per frame: the present thread must
        # not flip faster (tlg_graphics.h PresentLimit), or the game speeds up.
        env['TLG_ENGINE_FPS'] = str(engine_fps)
        derived.append('TLG_ENGINE_FPS')
        print(f'Engine frame rate: {engine_fps} FPS'
              + ('' if engine_fps > 60 or graphics['engine_fps'] == '60' else
                 ' (the frame rate patch is not available for this game build)'), flush=True)
        env['TLG_DERIVED'] = ','.join(derived)

        # The in-game restart starts this script again once this process is gone.
        # GPU caches (shaders, pipelines) beside the saves (path_util.cpp reads it when
        # the GPU starts; set here too so the in-game restart and the tools agree).
        user_dir = Path(env.get('ISS_USER_DIR') or env.get('TLG_USER_DIR') or _data_dir(data)).resolve()
        user_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault('TLG_GPU_USER_DIR', str(user_dir))
        if title_id:
            build_job_file(game, user_dir / 'cache' / f'{title_id}-jobs.bin', out, env)
        try:
            sys.path.insert(0, str(PORT / 'launcher'))
            import savemanager
            note = savemanager.backup_before_play(user_dir)   # also copies to the extra backup folder
            if note:
                print(note, flush=True)
        except ImportError:
            from save_backup import backup
            try:
                if made := backup(user_dir):
                    print(f'Saves backed up to {made}', flush=True)
            except OSError as error:
                print(f'Save backup failed, starting anyway: {error}', flush=True)
        this = ['--run'] if getattr(sys, 'frozen', False) else [str(Path(__file__).resolve())]
        if WINDOWS:
            restart = [sys.executable, *this, '--after', str(os.getpid()), *args]
            env['TLG_RESTART_COMMAND'] = subprocess.list2cmdline(restart)
        else:
            # Linux: the game execs this through sh -c in its own process, so nothing to wait for.
            env['TLG_RESTART_COMMAND'] = shlex.join([sys.executable, *this, *args])
        command = [str(probe), str(out / 'boot-linked.bin'), '--content-profile', str(out / 'content.bin'),
                   '--patches', str(out / 'patches.bin'), '--app0', str(game),
                   '--user', str(user_dir),
                   '--timeout', env.get('TLG_TIMEOUT', '0'), *isa_args, *args]
        # No stdin: an inherited pipe (shells such as Git Bash) cost the game its console output.
        return subprocess.run(command, stdin=subprocess.DEVNULL, creationflags=no_console()).returncode
    finally:
        if mod_view:
            shutil.rmtree(mod_view, ignore_errors=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
