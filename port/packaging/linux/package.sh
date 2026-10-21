#!/usr/bin/env bash
# Builds dist/$PORT_SLUG-linux-x86_64.tar.gz after `bash build.sh`: "$PORT_NAME" (the Qt Quick launcher)
# and "$PORT_NAME (classic)" (the Tk launcher), both
# frozen with PyInstaller so players need no Python, Qt or Tk, out/tlg-probe with libtlggpu.so, the
# libraries they need that a desktop may lack (out/lib: SDL3, FFmpeg, Zydis, fmt, ...), the
# preparation scripts and the player README. No game or console file may ship.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
# shellcheck source=../port.env
source packaging/port.env
[[ -f out/tlg-probe && -f out/gpu/libtlggpu.so ]] || { echo 'Build first: bash build.sh' >&2; exit 1; }
name=${TLG_PACKAGE_NAME:-$PORT_SLUG-linux-x86_64}
python=${PYTHON:-python3}
if [[ ! -x out/pyenv/bin/python ]]; then
    "$python" -m venv out/pyenv
fi
out/pyenv/bin/python -m pip install -q --disable-pip-version-check pyinstaller iced-x86==1.21.0 "$PORT_QT_PACKAGE"
# The scripts run inside the executable (--script): the standard modules they import come along,
# and iced-x86 for scripts/isa_rewrite.py (CPUs without MOVBE/BMI1/LZCNT).
hidden=()
for module in argparse base64 collections hashlib json re shutil struct tempfile xml.etree.ElementTree \
              urllib.request runpy sqlite3 mmap multiprocessing multiprocessing.pool dataclasses array bisect random time typing iced_x86; do
    hidden+=("$module")
done
# packaging/launcher.spec freezes both launchers into one folder (shared libraries) and trims Qt.
ISS_SPEC_NAME=$PORT_NAME ISS_SPEC_ROOT=$PWD ISS_SPEC_QT=$PORT_LAUNCHER ISS_SPEC_TK=$PORT_LAUNCHER_CLASSIC \
ISS_SPEC_HIDDEN="${hidden[*]}" \
    out/pyenv/bin/python -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath out/pyi-dist --workpath out/pyi-work packaging/launcher.spec

dest=out/stage/$name
rm -rf -- out/stage
mkdir -p "$dest/out/gpu" "$dest/out/lib" "$dest/launcher" "$dest/system_lib"
cp -r "out/pyi-dist/$PORT_NAME/." "$dest/"
cp launcher/$PORT_ICON.png "$dest/launcher/"
strip --strip-debug -o "$dest/out/tlg-probe" out/tlg-probe
strip --strip-debug -o "$dest/out/gpu/libtlggpu.so" out/gpu/libtlggpu.so
# Renderer v2 (TLG_RENDERER_V2=1 build.sh): its probe (rpath $ORIGIN/gpu_v2), its library and its
# defaults (gpu_v2/defaults.env, read by the launcher when the player picks it).
v2_bins=()
if [[ -f out/tlg-probe-v2 && -f out/gpu_v2/libissgpu.so ]]; then
    mkdir -p "$dest/out/gpu_v2" "$dest/gpu_v2"
    strip --strip-debug -o "$dest/out/tlg-probe-v2" out/tlg-probe-v2
    strip --strip-debug -o "$dest/out/gpu_v2/libissgpu.so" out/gpu_v2/libissgpu.so
    cp gpu_v2/defaults.env gpu_v2/README.md "$dest/gpu_v2/"
    v2_bins=(out/tlg-probe-v2 out/gpu_v2/libissgpu.so)
fi
[[ -f out/tlg-gpu-capabilities ]] && strip -o "$dest/out/tlg-gpu-capabilities" out/tlg-gpu-capabilities
# Libraries to bundle: everything the two ELF files load except the base system (glibc, the C++
# runtime, the Vulkan loader and drivers, X11/Wayland, audio servers), which must come from the
# player's distribution to match its drivers.
system='^(ld-linux|libc|libm|libdl|libpthread|librt|libmvec|libresolv|libutil|libstdc\+\+|libgcc_s|libvulkan|libGL|libEGL|libGLX|libGLdispatch|libX|libxcb|libwayland|libdrm|libasound|libpulse|libpipewire|libudev|libdbus|libsystemd|libz\.|libgomp|libcap|libffi)'
ldd out/tlg-probe out/gpu/libtlggpu.so "${v2_bins[@]}" | awk '/=> \// {print $1, $3}' | sort -u | while read -r soname path; do
    [[ $soname == libtlggpu.so ]] && continue
    [[ $soname =~ $system ]] && continue
    cp -L "$path" "$dest/out/lib/$soname"
done
# SDL3 loads its video/audio backends at run time from the player's system: nothing else to copy.
cp -r scripts "$dest/"
cp -r patches "$dest/"
# The shader list generator (scripts/build_jobs.py imports it from ../tools): plain files, like scripts/.
mkdir -p "$dest/tools"
cp -r tools/xpps "$dest/tools/"
find "$dest" -name __pycache__ -prune -exec rm -r {} +
cp LICENSE packaging/README-player.md "$dest/"
# The release name (vYYYY.MM.DD-<commit>, as the CI publishes it): the launcher shows it and compares it with the latest release.
if [[ -n ${GITHUB_SHA:-} ]]; then printf 'v%s-%s\n' "$(date -u +%Y.%m.%d)" "${GITHUB_SHA::7}" > "$dest/VERSION"; fi
cp packaging/linux/install_desktop.sh packaging/port.env "$dest/"
cat > "$dest/system_lib/LEIA-ME.txt" <<'TXT'
Este jogo não precisa de módulos do firmware do console: a pasta fica vazia.
This game needs no console firmware modules: leave this folder empty.
TXT
bash packaging/check_package.sh "$dest"
mkdir -p dist
tar -C out/stage -czf "dist/$name.tar.gz" "$name"
du -sh "$dest" "dist/$name.tar.gz"
