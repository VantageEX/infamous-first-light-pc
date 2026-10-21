#!/usr/bin/env bash
# Builds dist/$PORT_SLUG-windows-x86_64.zip after `bash build.sh` in an MSYS2 CLANG64 shell:
# "$PORT_NAME.exe" (the Qt Quick launcher) and "$PORT_NAME (classic).exe" (the Tk launcher),
# both frozen with PyInstaller (packaging/launcher.spec) so players need no Python, Qt or Tk,
# "Play $PORT_NAME.exe" (starts the game with the saved settings), out/tlg-probe.exe with
# the CLANG64 DLLs it loads, the preparation scripts and the player README. No game or console
# file may ship. Freezing uses a Windows Python 3.10+ (python.org or actions/setup-python;
# WINPYTHON overrides) and a private venv in out/pyenv.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
# shellcheck source=../port.env
source packaging/port.env
source ./msys2-env.sh
[[ -f out/tlg-probe.exe && -f out/tlg-gpu-capabilities.exe && -f out/tlg-play.exe ]] || { echo 'Build first: bash build.sh' >&2; exit 1; }
name=${TLG_PACKAGE_NAME:-$PORT_SLUG-windows-x86_64}

python=${WINPYTHON:-}
if [[ -z $python ]]; then
    for candidate in /c/Python3*/python.exe "${LOCALAPPDATA:-/c/Users/$USER/AppData/Local}"/Programs/Python/Python3*/python.exe \
                     /c/hostedtoolcache/windows/Python/3*/x64/python.exe; do
        [[ -x $candidate ]] && python=$candidate
    done
fi
[[ -n $python ]] || { echo 'Need a Windows Python 3 (python.org) or WINPYTHON=path\to\python.exe' >&2; exit 1; }
export USERPROFILE=${USERPROFILE:-$(cygpath -w "/c/Users/$(id -un)")}
if [[ ! -x out/pyenv/Scripts/python.exe ]]; then
    "$python" -m venv out/pyenv
fi
out/pyenv/Scripts/python.exe -m pip install -q --disable-pip-version-check pyinstaller iced-x86==1.21.0 "$PORT_QT_PACKAGE"
hidden=()
for module in argparse base64 collections hashlib json re shutil struct tempfile xml.etree.ElementTree \
              urllib.request runpy sqlite3 mmap multiprocessing multiprocessing.pool dataclasses array bisect random time typing ctypes.wintypes msvcrt iced_x86; do
    hidden+=("$module")
done
# One folder, two launchers sharing their libraries; native Windows paths go to the spec in the
# environment (it joins them with pathlib), the icon is the same .ico as before.
ISS_SPEC_NAME=$PORT_NAME ISS_SPEC_ROOT=$(cygpath -w "$PWD") ISS_SPEC_ICON=$(cygpath -w "$PWD/launcher/$PORT_ICON.ico") \
ISS_SPEC_QT=$PORT_LAUNCHER ISS_SPEC_TK=$PORT_LAUNCHER_CLASSIC ISS_SPEC_HIDDEN="${hidden[*]}" \
    out/pyenv/Scripts/python.exe -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath out/pyi-dist --workpath out/pyi-work "$(cygpath -w "$PWD/packaging/launcher.spec")"

dest=out/stage/$name
rm -rf -- out/stage
mkdir -p "$dest/out" "$dest/launcher" "$dest/system_lib"
cp -r "out/pyi-dist/$PORT_NAME/." "$dest/"
llvm-strip -o "$dest/Play $PORT_NAME.exe" out/tlg-play.exe
cp launcher/$PORT_ICON.ico launcher/$PORT_ICON.png "$dest/launcher/"
# The executables without debug information, where tlg_run.py looks for them (out/).
for exe in tlg-probe.exe tlg-gpu-capabilities.exe; do
    llvm-strip --strip-debug -o "$dest/out/$exe" "out/$exe"
done
# Renderer v2 (TLG_RENDERER_V2=1 build.sh): its probe and its defaults (gpu_v2/defaults.env, read by
# the launcher when the player picks it).
if [[ -f out/tlg-probe-v2.exe ]]; then
    llvm-strip --strip-debug -o "$dest/out/tlg-probe-v2.exe" out/tlg-probe-v2.exe
    mkdir -p "$dest/gpu_v2"
    cp gpu_v2/defaults.env gpu_v2/README.md "$dest/gpu_v2/"
fi
# Every DLL the executables load from the CLANG64 tree (SDL3, FFmpeg, Vulkan loader, ...), found by
# walking the import tables against /clang64/bin itself. ldd resolved them through PATH, where the
# runner's Strawberry Perl has its own libiconv-2.dll: that one was filtered out as "not CLANG64"
# and some packages shipped without libiconv-2.dll ("libiconv-2.dll was not found" at Play).
# Names not in /clang64/bin are Windows' own DLLs.
declare -A shipped=()
queue=("$dest/out/tlg-probe.exe" "$dest/out/tlg-gpu-capabilities.exe")
[[ -f "$dest/out/tlg-probe-v2.exe" ]] && queue+=("$dest/out/tlg-probe-v2.exe")
while ((${#queue[@]})); do
    binary=${queue[0]}
    queue=("${queue[@]:1}")
    while read -r dll; do
        key=${dll,,}
        [[ -n ${shipped[$key]:-} ]] && continue
        shipped[$key]=1
        source=$(find /clang64/bin -maxdepth 1 -iname "$dll" -print -quit)
        [[ -n $source ]] || continue
        cp -u "$source" "$dest/out/"
        queue+=("$source")
    done < <(llvm-readobj --coff-imports "$binary" | awk '$1 == "Name:" && tolower($2) ~ /\.dll$/ {print $2}')
done
echo "Packaged $(find "$dest/out" -maxdepth 1 -iname '*.dll' | wc -l) DLLs from /clang64/bin"
cp -r scripts "$dest/"
cp -r patches "$dest/"
# The shader list generator (scripts/build_jobs.py imports it from ../tools): plain files, like scripts/.
mkdir -p "$dest/tools"
cp -r tools/xpps "$dest/tools/"
find "$dest" -name __pycache__ -prune -exec rm -r {} +
cp LICENSE packaging/README-player.md "$dest/"
# The release name (vYYYY.MM.DD-<commit>, as the CI publishes it): the launcher shows it and compares it with the latest release.
if [[ -n ${GITHUB_SHA:-} ]]; then printf 'v%s-%s\n' "$(date -u +%Y.%m.%d)" "${GITHUB_SHA::7}" > "$dest/VERSION"; fi
printf '%s\r\n' 'Este jogo não precisa de módulos do firmware do console: a pasta fica vazia.' \
    'This game needs no console firmware modules: leave this folder empty.' > "$dest/system_lib/LEIA-ME.txt"
bash packaging/check_package.sh "$dest"
# Only these hashed package files may be replaced by the in-place updater.
if [[ ! -f "$dest/VERSION" ]]; then git rev-parse --short HEAD > "$dest/VERSION"; fi
out/pyenv/Scripts/python.exe launcher/update_install.py --manifest "$dest" "$(tr -d '\r\n' < "$dest/VERSION")"
mkdir -p dist
rm -f "dist/$name.zip"
(cd out/stage && powershell -NoProfile -Command "Compress-Archive -Path '$name' -DestinationPath '../../dist/$name.zip'")
du -sh "$dest" "dist/$name.zip"
