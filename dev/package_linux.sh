#!/usr/bin/env bash
# Builds dist/inFAMOUS-First-Light-PC-linux-x86_64/ (and a .tar.gz of it) after
# `cd port && TLG_RENDERER_V2=1 bash build.sh` on Linux (the GitHub Actions job: Ubuntu 24.04, so
# glibc 2.39: SteamOS 3.6 and newer). The same layout as the Windows package (dev/package.sh):
#   inFAMOUS First Light        the launcher, frozen with PyInstaller (no Python or Qt needed)
#   port/run.py, port/scripts/  the core's preparation scripts (the launcher runs them: --script)
#   port/out/tlg-probe          the core, its GPU library in out/gpu, renderer v2 beside it, and
#                               the libraries a desktop may lack in out/lib (run.py adds it to
#                               LD_LIBRARY_PATH)
#   port/fonts/                 Open Sans and the Fluent icons (Linux has no Segoe UI)
#   data/                       created on first start: settings, logs, saves
# A .tar.gz, not a .zip: it keeps the programs executable. No game file ships.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
[[ -f port/out/tlg-probe && -f port/out/gpu/libtlggpu.so ]] || { echo 'Build the core first: cd port && TLG_RENDERER_V2=1 bash build.sh' >&2; exit 1; }
name=inFAMOUS-First-Light-PC-linux-x86_64
if [[ ${TEST:-0} == 1 ]]; then name=$name-test; fi
# PROFILE=1: the profiling package for testers (port/docs/profiling.md): the profiling build of
# renderer v2 and tracy-capture beside the normal ones (TLG_PROFILE=1 build.sh,
# packaging/build_tracy_tools.sh).
if [[ ${PROFILE:-0} == 1 ]]; then
    name=$name-profiling
    [[ -f port/out/tlg-probe-v2-profile && -f port/out/gpu_v2_profile/libissgpu.so && -f port/out/tracy-capture ]] ||
        { echo 'Build the profiling build first: TLG_PROFILE=1 build.sh and packaging/build_tracy_tools.sh' >&2; exit 1; }
fi
exe_name='inFAMOUS First Light'

python=${PYTHON:-python3}
pyenv=port/out/pyenv-fl
if [[ ! -x $pyenv/bin/python ]]; then
    "$python" -m venv "$pyenv"
fi
"$pyenv/bin/python" -m pip install -q --disable-pip-version-check pyinstaller iced-x86 PySide6-Essentials pillow

# The scripts the frozen launcher runs with --script import these from the standard library
# (PyInstaller only sees the launcher's own imports).
hidden=()
for module in argparse base64 collections hashlib json re shutil shlex struct tempfile xml.etree.ElementTree \
              urllib.request runpy sqlite3 mmap dataclasses array bisect random time typing \
              zlib iced_x86 PIL.Image multiprocessing multiprocessing.pool; do
    hidden+=(--hidden-import "$module")
done
entry=port/out/fl_entry.py
printf '%s\n' '# PyInstaller entry: the launcher package run as a program.' \
    'import fl_launcher.__main__  # noqa: F401' > "$entry"
# Absolute paths: PyInstaller reads relative ones from the spec's folder (port/out).
"$pyenv/bin/python" -m PyInstaller --noconfirm --clean --log-level WARN \
    --name "$exe_name" --paths "$PWD/launcher" --collect-submodules fl_launcher \
    --add-data "$PWD/launcher/fl_launcher/qml:fl_launcher/qml" \
    --add-data "$PWD/launcher/fl_launcher/assets:fl_launcher/assets" \
    "${hidden[@]}" --distpath port/out/pyi-dist --workpath port/out/pyi-work --specpath port/out \
    "$PWD/$entry"

dest=dist/$name
rm -rf -- "$dest"
mkdir -p "$dest/port/out/gpu" "$dest/port/out/lib"
cp -r "port/out/pyi-dist/$exe_name/." "$dest/"
strip --strip-debug -o "$dest/port/out/tlg-probe" port/out/tlg-probe
strip --strip-debug -o "$dest/port/out/gpu/libtlggpu.so" port/out/gpu/libtlggpu.so
# Renderer v2 when it was built: its probe (rpath $ORIGIN/gpu_v2), its library and its defaults.
v2_bins=()
if [[ -f port/out/tlg-probe-v2 && -f port/out/gpu_v2/libissgpu.so ]]; then
    mkdir -p "$dest/port/out/gpu_v2" "$dest/port/gpu_v2"
    strip --strip-debug -o "$dest/port/out/tlg-probe-v2" port/out/tlg-probe-v2
    strip --strip-debug -o "$dest/port/out/gpu_v2/libissgpu.so" port/out/gpu_v2/libissgpu.so
    cp port/gpu_v2/defaults.env "$dest/port/gpu_v2/"
    v2_bins=(port/out/tlg-probe-v2 port/out/gpu_v2/libissgpu.so)
    if [[ ${PROFILE:-0} == 1 ]]; then
        # Only the debug sections go: the symbol tables name the port's code in the reports.
        mkdir -p "$dest/port/out/gpu_v2_profile"
        strip --strip-debug -o "$dest/port/out/tlg-probe-v2-profile" port/out/tlg-probe-v2-profile
        strip --strip-debug -o "$dest/port/out/gpu_v2_profile/libissgpu.so" port/out/gpu_v2_profile/libissgpu.so
        strip -o "$dest/port/out/tracy-capture" port/out/tracy-capture
        v2_bins+=(port/out/tlg-probe-v2-profile port/out/gpu_v2_profile/libissgpu.so port/out/tracy-capture)
    fi
fi
# Libraries to bundle: everything the core loads except the base system (glibc, the C++ runtime,
# the Vulkan loader and drivers, X11/Wayland, audio servers), which must come from the player's
# system to match its drivers (as port/packaging/linux/package.sh).
system='^(ld-linux|libc|libm|libdl|libpthread|librt|libmvec|libresolv|libutil|libstdc\+\+|libgcc_s|libvulkan|libGL|libEGL|libGLX|libGLdispatch|libX|libxcb|libwayland|libdrm|libasound|libpulse|libpipewire|libudev|libdbus|libsystemd|libz\.|libgomp|libcap|libffi)'
ldd port/out/tlg-probe port/out/gpu/libtlggpu.so "${v2_bins[@]}" | awk '/=> \// {print $1, $3}' | sort -u |
while read -r soname path; do
    [[ $soname == libtlggpu.so || $soname == libissgpu.so ]] && continue
    [[ $soname =~ $system ]] && continue
    cp -L "$path" "$dest/port/out/lib/$soname"
done
echo "Packaged $(find "$dest/port/out/lib" -name '*.so*' | wc -l) libraries into port/out/lib"
cp port/run.py "$dest/port/"
cp -r port/scripts port/patches port/fonts port/sounds "$dest/port/"
mkdir -p "$dest/port/tools"
cp -r port/tools/xpps "$dest/port/tools/"
find "$dest" -name __pycache__ -prune -exec rm -r {} +
cp port/LICENSE "$dest/LICENSE.txt"
cp dev/README-linux.txt "$dest/"
git rev-parse --short HEAD > "$dest/VERSION"
release=${RELEASE:-$(git tag --points-at HEAD | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)}
if [[ -n $release ]]; then
    echo "$release" > "$dest/RELEASE"
else
    echo 'No release tag (RELEASE=vX.Y.Z): this package will not offer updates' >&2
fi
if [[ ${TEST:-0} == 1 ]]; then
    echo 'Test build: play, then Log > Create report, and send the zip.' > "$dest/TEST_BUILD"
fi
if find "$dest" -iname 'eboot.bin' -o -iname '*.sfo' | grep -q .; then
    echo 'A game file ended up in the package' >&2; exit 1
fi
# Every program the core loads must resolve from the package and the base system alone.
missing=$(LD_LIBRARY_PATH="$dest/port/out/lib" ldd "$dest/port/out/tlg-probe" "$dest/port/out/gpu/libtlggpu.so" \
    ${v2_bins:+"$dest/port/out/tlg-probe-v2" "$dest/port/out/gpu_v2/libissgpu.so"} \
    $([[ ${PROFILE:-0} == 1 ]] && echo "$dest/port/out/tlg-probe-v2-profile $dest/port/out/gpu_v2_profile/libissgpu.so $dest/port/out/tracy-capture") |
    grep 'not found' || true)
[[ -z $missing ]] || { echo "Unresolved libraries:"$'\n'"$missing" >&2; exit 1; }
rm -f "dist/$name.tar.gz"
tar -C dist -czf "dist/$name.tar.gz" "$name"
du -sh "$dest" "dist/$name.tar.gz"
