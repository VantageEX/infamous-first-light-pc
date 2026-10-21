#!/usr/bin/env bash
# Builds dist/inFAMOUS-First-Light-PC/ (and a .zip of it) after `cd port && bash build.sh`, in an
# MSYS2 CLANG64 shell. Layout:
#   inFAMOUS First Light.exe   the launcher, frozen with PyInstaller (no Python or Qt needed)
#   port/run.py, port/scripts/ the core's preparation scripts (the launcher exe runs them: --script)
#   port/out/tlg-probe.exe     the core, with every CLANG64 DLL it loads
#   data/                      created on first start: settings, logs, saves
# No game file ships: the player picks their own dump in the launcher.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
root=$PWD
[[ -f port/out/tlg-probe.exe ]] || { echo 'Build the core first: cd port && bash build.sh' >&2; exit 1; }
name=inFAMOUS-First-Light-PC
# TEST=1: the test build for players' reports (a TEST_BUILD file: the launcher turns the readback
# statistics on and asks for a report after playing).
if [[ ${TEST:-0} == 1 ]]; then name=$name-test; fi
exe_name='inFAMOUS First Light'

python=${WINPYTHON:-}
if [[ -z $python ]]; then
    for candidate in "${LOCALAPPDATA:-/c/Users/$USER/AppData/Local}"/Programs/Python/Python3*/python.exe /c/Python3*/python.exe; do
        [[ -x $candidate ]] && python=$candidate
    done
fi
[[ -n $python ]] || { echo 'Need a Windows Python 3 (python.org) or WINPYTHON=path\to\python.exe' >&2; exit 1; }
# A login MSYS2 shell drops USERPROFILE, and PyInstaller needs the home folder.
export USERPROFILE=${USERPROFILE:-$(cygpath -w "/c/Users/$(id -un)")}
pyenv=port/out/pyenv-fl
if [[ ! -x $pyenv/Scripts/python.exe ]]; then
    "$python" -m venv "$pyenv"
fi
"$pyenv/Scripts/python.exe" -m pip install -q --disable-pip-version-check pyinstaller iced-x86 PySide6-Essentials pillow

# The scripts the frozen launcher runs with --script import these from the standard library
# (PyInstaller only sees the launcher's own imports).
hidden=()
for module in argparse base64 collections hashlib json re shutil struct tempfile xml.etree.ElementTree \
              urllib.request runpy sqlite3 mmap dataclasses array bisect random time typing \
              ctypes.wintypes msvcrt zlib iced_x86 PIL.Image multiprocessing multiprocessing.pool; do
    hidden+=(--hidden-import "$module")
done
entry=port/out/fl_entry.py
# A plain import, so PyInstaller sees fl_launcher.__main__ (collect_submodules skips __main__);
# importing it runs the launcher, or a script with --script.
printf '%s\n' '# PyInstaller entry: the launcher package run as a program.' \
    'import fl_launcher.__main__  # noqa: F401' > "$entry"
"$pyenv/Scripts/python.exe" -m PyInstaller --noconfirm --clean --log-level WARN --windowed \
    --name "$exe_name" --icon "$(cygpath -w "$root/port/packaging/windows/app.ico")" \
    --paths "$(cygpath -w "$root/launcher")" --collect-submodules fl_launcher \
    --add-data "$(cygpath -w "$root/launcher/fl_launcher/qml");fl_launcher/qml" \
    --add-data "$(cygpath -w "$root/launcher/fl_launcher/assets");fl_launcher/assets" \
    "${hidden[@]}" --distpath port/out/pyi-dist --workpath port/out/pyi-work --specpath port/out \
    "$(cygpath -w "$root/$entry")"

dest=dist/$name
# Emptied rather than removed: an Explorer window open on it keeps the folder itself busy.
mkdir -p "$dest"
find "$dest" -mindepth 1 -delete
mkdir -p "$dest/port/out"
cp -r "port/out/pyi-dist/$exe_name/." "$dest/"
llvm-strip --strip-debug -o "$dest/port/out/tlg-probe.exe" port/out/tlg-probe.exe
# Renderer v2 (port/gpu_v2) when it was built (TLG_RENDERER_V2=1 build.sh): its probe and defaults.
if [[ -f port/out/tlg-probe-v2.exe ]]; then
    llvm-strip --strip-debug -o "$dest/port/out/tlg-probe-v2.exe" port/out/tlg-probe-v2.exe
    mkdir -p "$dest/port/gpu_v2"
    cp port/gpu_v2/defaults.env "$dest/port/gpu_v2/"
fi
# Every DLL tlg-probe.exe loads from the CLANG64 tree, by walking the import tables (names that
# are not in /clang64/bin are Windows' own).
declare -A shipped=()
queue=("$dest/port/out/tlg-probe.exe")
[[ -f "$dest/port/out/tlg-probe-v2.exe" ]] && queue+=("$dest/port/out/tlg-probe-v2.exe")
while ((${#queue[@]})); do
    binary=${queue[0]}
    queue=("${queue[@]:1}")
    while read -r dll; do
        key=${dll,,}
        [[ -n ${shipped[$key]:-} ]] && continue
        shipped[$key]=1
        source=$(find /clang64/bin -maxdepth 1 -iname "$dll" -print -quit)
        [[ -n $source ]] || continue
        cp -u "$source" "$dest/port/out/"
        queue+=("$source")
    done < <(llvm-readobj --coff-imports "$binary" | awk '$1 == "Name:" && tolower($2) ~ /\.dll$/ {print $2}')
done
echo "Packaged $(find "$dest/port/out" -maxdepth 1 -iname '*.dll' | wc -l) DLLs from /clang64/bin"
cp port/run.py "$dest/port/"
cp -r port/scripts port/patches "$dest/port/"
# The first-start shader jobs generator (scripts/build_jobs.py runs it).
mkdir -p "$dest/port/tools"
cp -r port/tools/xpps "$dest/port/tools/"
find "$dest" -name __pycache__ -prune -exec rm -r {} +
cp port/LICENSE "$dest/LICENSE.txt"
git rev-parse --short HEAD > "$dest/VERSION"
# The release tag (RELEASE=v0.3.2, else this port's vX.Y.Z tag on HEAD): the launcher compares it
# with the latest GitHub release to offer updates. A package without it never updates itself.
release=${RELEASE:-$(git tag --points-at HEAD | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)}
if [[ -n $release ]]; then
    echo "$release" > "$dest/RELEASE"
else
    echo 'No release tag (RELEASE=vX.Y.Z): this package will not offer updates' >&2
fi
if [[ ${TEST:-0} == 1 ]]; then
    echo 'Test build: play, then Log > Create report, and send the zip.' > "$dest/TEST_BUILD"
fi
# Nothing of the player's may ship.
if find "$dest" -iname 'eboot.bin' -o -iname '*.sfo' -o -iname 'param.sfo' | grep -q .; then
    echo 'A game file ended up in the package' >&2; exit 1
fi
rm -f "dist/$name.zip"
(cd dist && powershell -NoProfile -Command "Compress-Archive -Path '$name' -DestinationPath '$name.zip'")
du -sh "$dest" "dist/$name.zip"
