#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Players' names work too (the README says ISS_LTO=OFF): ISS_LTO/ISS_PGO set TLG_LTO/TLG_PGO.
[[ -n ${ISS_LTO:-} && -z ${TLG_LTO:-} ]] && export TLG_LTO=$ISS_LTO
[[ -n ${ISS_PGO:-} && -z ${TLG_PGO:-} ]] && export TLG_PGO=$ISS_PGO
# The commit the probe logs on its first line (CI: GITHUB_SHA; local: git, plus "+dirty").
if [[ -z ${TLG_BUILD_COMMIT:-} ]]; then
    TLG_BUILD_COMMIT=${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
    TLG_BUILD_COMMIT=${TLG_BUILD_COMMIT:0:12}
    [[ -z ${GITHUB_SHA:-} ]] && ! git diff --quiet HEAD -- . 2>/dev/null && TLG_BUILD_COMMIT+="+dirty"
fi
export TLG_BUILD_COMMIT
mkdir -p out
# Windows: MSYS2 CLANG64 (packages in packaging/windows/README.md). The GPU library is static
# there and CMake links tlg-probe.exe as well; no PGO (GCC-only flags), LTO off by default.
case $(uname -s) in
MINGW*|MSYS*)
    # shellcheck source=msys2-env.sh
    source ./msys2-env.sh
    for tool in clang cmake ninja pkg-config; do
        command -v "$tool" >/dev/null || { echo "Need $tool from MSYS2 CLANG64 (packaging/windows/README.md)." >&2; exit 1; }
    done
    if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ||
          ! -f gpu/third_party/xbyak/xbyak/xbyak.h ]]; then
        git submodule update --init --recursive
    fi
    for patch in gpu/patches/fsr-vulkan/*.patch; do
        if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
            git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
        fi
    done
    cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTLG_PGO=off \
        -DTLG_LTO="${TLG_LTO:-OFF}" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ >/dev/null
    echo "GPU library and loader: LTO ${TLG_LTO:-OFF}"
    if ! ninja -C out/gpu tlg-probe tlg-gpu-capabilities tlg-play > out/gpu-build.log 2>&1; then
        grep -v '^\[' out/gpu-build.log | tail -40 >&2
        echo 'Build failed (full log: out/gpu-build.log)' >&2; exit 1
    fi
    cp out/gpu/tlg-probe.exe out/gpu/tlg-gpu-capabilities.exe out/gpu/tlg-play.exe out/
    echo "Built $PWD/out/tlg-probe.exe"
    # TLG_RENDERER_V2=1: renderer v2 (gpu_v2/README.md), linked into out/tlg-probe-v2.exe.
    if [[ ${TLG_RENDERER_V2:-${ISS_RENDERER_V2:-}} == 1 ]]; then
        cmake -S gpu_v2 -B out/gpu_v2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ >/dev/null
        if ! ninja -C out/gpu_v2 tlg-probe-v2 > out/gpu_v2-build.log 2>&1; then
            grep -v '^\[' out/gpu_v2-build.log | tail -60 >&2
            echo 'Renderer v2 build failed (full log: out/gpu_v2-build.log)' >&2; exit 1
        fi
        cp out/gpu_v2/tlg-probe-v2.exe out/
        echo "Built $PWD/out/tlg-probe-v2.exe (renderer v2)"
        # TLG_PROFILE=1: also the profiling build of renderer v2 (docs/profiling.md), the Tracy
        # client built in, as out/tlg-probe-v2-profile.exe (scripts/profile_session.py runs it).
        if [[ ${TLG_PROFILE:-} == 1 ]]; then
            cmake -S gpu_v2 -B out/gpu_v2_profile -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DTLG_PROFILE=ON >/dev/null
            if ! ninja -C out/gpu_v2_profile tlg-probe-v2 > out/gpu_v2_profile-build.log 2>&1; then
                grep -v '^\[' out/gpu_v2_profile-build.log | tail -60 >&2
                echo 'Profiling build failed (full log: out/gpu_v2_profile-build.log)' >&2; exit 1
            fi
            cp out/gpu_v2_profile/tlg-probe-v2.exe out/tlg-probe-v2-profile.exe
            echo "Built $PWD/out/tlg-probe-v2-profile.exe (renderer v2, profiling)"
        fi
    fi
    # TLG_RENDERER_V3=1: renderer v3 (gpu_v3/README.md), linked into out/tlg-probe-v3.exe.
    if [[ ${TLG_RENDERER_V3:-} == 1 ]]; then
        cmake -S gpu_v3 -B out/gpu_v3 -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ >/dev/null
        if ! ninja -C out/gpu_v3 tlg-probe-v3 > out/gpu_v3-build.log 2>&1; then
            grep -v '^\[' out/gpu_v3-build.log | tail -60 >&2
            echo 'Renderer v3 build failed (full log: out/gpu_v3-build.log)' >&2; exit 1
        fi
        cp out/gpu_v3/tlg-probe-v3.exe out/
        echo "Built $PWD/out/tlg-probe-v3.exe (renderer v3)"
    fi
    exit 0
    ;;
esac
if [[ -z ${CC:-} ]]; then
    CC=$(command -v cc || command -v gcc || true)
    if [[ -z $CC ]]; then
        for candidate in /nix/store/*-gcc-wrapper-*/bin/gcc; do
            if [[ -x $candidate ]]; then CC=$candidate; break; fi
        done
    fi
fi
if [[ -z ${CC:-} ]]; then echo 'Install GCC/Clang or set CC.' >&2; exit 1; fi
# Dependencies come from pkg-config (Vulkan loader/headers, SDL3). On NixOS the
# environment is provided by shell.nix; re-enter it automatically if needed.
if ! { command -v pkg-config >/dev/null && pkg-config --exists vulkan sdl3 && command -v cmake >/dev/null && command -v ninja >/dev/null; }; then
    if [[ -z ${TLG_IN_NIX_SHELL:-} ]] && command -v nix-shell >/dev/null; then
        exec env TLG_IN_NIX_SHELL=1 nix-shell shell.nix --run "bash build.sh $*"
    fi
    echo 'Need pkg-config with vulkan and sdl3, cmake and ninja (see shell.nix).' >&2; exit 1
fi
# FreeType headers: the system package, or a local copy in the workspace tools (no root needed).
if ! pkg-config --exists freetype2 && [[ -d ../../tools/freetype-devel/pkgconfig ]]; then
    export PKG_CONFIG_PATH="$(cd ../../tools/freetype-devel/pkgconfig && pwd)${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
read -r -a includes <<< "$(pkg-config --cflags vulkan sdl3 libavcodec libavutil libswscale freetype2)"
read -r -a libraries <<< "$(pkg-config --libs vulkan sdl3 libavcodec libavutil libswscale freetype2)"
# GPU library (shadPS4 video core + drivers), built by CMake into out/gpu/libtlggpu.so.
# TLG_PGO: generate (instrumented build that writes pgo/ while the game runs), use, off.
# Default: use the profile in pgo/ when there is one. TLG_LTO=OFF disables link-time optimization.
pgo=${TLG_PGO:-}
if [[ -z $pgo ]]; then
    if [[ -n $(find pgo -name '*.gcda' -print -quit 2>/dev/null) ]]; then pgo=use; else pgo=off; fi
fi
mkdir -p pgo
# Submodules (git clone --recursive, or: git submodule update --init) and this port's changes
# to FSR-Vulkan (gpu/patches/fsr-vulkan), applied to its working tree once.
if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ]]; then
    git submodule update --init --recursive
fi
for patch in gpu/patches/fsr-vulkan/*.patch; do
    if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
    fi
done
cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTLG_PGO="$pgo" \
    -DTLG_LTO="${TLG_LTO:-ON}" -DTLG_PGO_DIR="$PWD/pgo" >/dev/null
echo "GPU library: PGO $pgo, LTO ${TLG_LTO:-ON}"
# A failed GPU build must stop here: an older libtlggpu.so would otherwise be used silently.
if ! ninja -C out/gpu tlggpu > out/gpu-build.log 2>&1; then
    grep -v '^\[' out/gpu-build.log | tail -40 >&2
    echo 'GPU library build failed (full log: out/gpu-build.log)' >&2; exit 1
fi
# $ORIGIN/gpu: packaged copies keep the library next to the binary without patching it.
gpu=(-Lout/gpu -ltlggpu -Wl,-rpath,'$ORIGIN/gpu' -Wl,-rpath,"$PWD/out/gpu" -rdynamic)
runtime=(src/runtime*.c)
# Third-party decoders: compiled once, without this project's -Werror policy.
atrac9=(third_party/LibAtrac9/C/src/*.c)
if [[ ! -f out/libatrac9.a || -n $(find third_party/LibAtrac9/C/src -newer out/libatrac9.a -name '*.c') ]]; then
    rm -rf out/atrac9 && mkdir -p out/atrac9
    for source in "${atrac9[@]}"; do "$CC" -std=c99 -O2 -g -w -c "$source" -o "out/atrac9/$(basename "${source%.c}").o"; done
    ar rcs out/libatrac9.a out/atrac9/*.o
fi
# isa_emulate runs inside guest code: no XMM/YMM.
"$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -mgeneral-regs-only -c src/isa_emulate.c -o out/isa_emulate.o
# --test builds everything, then the unit tests; --test-only reuses the GPU library in out/ and
# builds and runs only the tests (CI, after the Build step with the same TLG_LTO/TLG_PGO).
if [[ ${1:-} != --test-only ]]; then
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -no-pie -DTLG_BUILD_COMMIT="\"$TLG_BUILD_COMMIT\"" "${includes[@]}" -I. -Isrc src/probe.c "${runtime[@]}" src/cpu_check.c src/vulkan_smoke.c out/isa_emulate.o src/isa_rewrite.c out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/tlg-probe
    echo "Built $PWD/out/tlg-probe"
    # TLG_RENDERER_V2=1 (ISS_RENDERER_V2): also renderer v2 (gpu_v2/README.md) and
    # the probe that uses it, out/tlg-probe-v2 (TLG_PROBE=tlg-probe-v2 runs it). Needs clang.
    if [[ ${TLG_RENDERER_V2:-${ISS_RENDERER_V2:-}} == 1 ]]; then
        cmake -S gpu_v2 -B out/gpu_v2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang >/dev/null
        if ! ninja -C out/gpu_v2 issgpu > out/gpu_v2-build.log 2>&1; then
            grep -v '^\[' out/gpu_v2-build.log | tail -40 >&2
            echo 'Renderer v2 build failed (full log: out/gpu_v2-build.log)' >&2; exit 1
        fi
        cs=(-Lout/gpu_v2 -lissgpu -Wl,-rpath,'$ORIGIN/gpu_v2' -Wl,-rpath,"$PWD/out/gpu_v2" -rdynamic)
        "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -no-pie -DTLG_BUILD_COMMIT="\"$TLG_BUILD_COMMIT\"" "${includes[@]}" -I. -Isrc src/probe.c "${runtime[@]}" src/cpu_check.c src/vulkan_smoke.c out/isa_emulate.o src/isa_rewrite.c out/libatrac9.a -lm "${cs[@]}" "${libraries[@]}" -o out/tlg-probe-v2
        echo "Built $PWD/out/tlg-probe-v2 (renderer v2)"
        # TLG_PROFILE=1: also the profiling build of renderer v2 (docs/profiling.md): the library
        # with the Tracy client in out/gpu_v2_profile, and the runtime compiled with TRACY_ENABLE
        # (its zones call the client there) as out/tlg-probe-v2-profile.
        if [[ ${TLG_PROFILE:-} == 1 ]]; then
            cmake -S gpu_v2 -B out/gpu_v2_profile -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DTLG_PROFILE=ON >/dev/null
            if ! ninja -C out/gpu_v2_profile issgpu > out/gpu_v2_profile-build.log 2>&1; then
                grep -v '^\[' out/gpu_v2_profile-build.log | tail -40 >&2
                echo 'Profiling build failed (full log: out/gpu_v2_profile-build.log)' >&2; exit 1
            fi
            ps=(-Lout/gpu_v2_profile -lissgpu -Wl,-rpath,'$ORIGIN/gpu_v2_profile' -Wl,-rpath,"$PWD/out/gpu_v2_profile" -rdynamic)
            "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -no-pie -DTRACY_ENABLE=1 -DTLG_PROFILE=1 -isystem third_party/tracy/public -DTLG_BUILD_COMMIT="\"$TLG_BUILD_COMMIT\"" "${includes[@]}" -I. -Isrc src/probe.c "${runtime[@]}" src/cpu_check.c src/vulkan_smoke.c out/isa_emulate.o src/isa_rewrite.c out/libatrac9.a -lm "${ps[@]}" "${libraries[@]}" -o out/tlg-probe-v2-profile
            echo "Built $PWD/out/tlg-probe-v2-profile (renderer v2, profiling)"
        fi
    fi
    # GPU check for run.sh (live_resolution=auto): links only the Vulkan loader.
    "$CC" -std=c11 -O2 -Wall -Wextra -Werror -Igpu/shadps4 tools/gpu_capabilities.c "${libraries[@]}" -o out/tlg-gpu-capabilities
fi
if [[ ${1:-} == --test || ${1:-} == --test-only ]]; then
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_pad.c "${libraries[@]}" -lm -o out/pad-test
    out/pad-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_pad_map.c "${libraries[@]}" -lm -o out/pad-map-test
    out/pad-map-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror tests/test_mouse_stick.c -lm -o out/mouse-stick-test
    out/mouse-stick-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -UNDEBUG tests/test_mouse_touch.c -Isrc -lm -o out/mouse-touch-test
    out/mouse-touch-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -UNDEBUG tests/test_rumble_gate.c -Isrc -o out/rumble-gate-test
    out/rumble-gate-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -UNDEBUG -pthread "${includes[@]}" -I. -Isrc tests/test_mouse_pad.c "${libraries[@]}" -lm -o out/mouse-pad-test
    out/mouse-pad-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_runtime.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/runtime-test
    out/runtime-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_file_mods.c -o out/file-mods-test
    out/file-mods-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_sema.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/sema-test
    out/sema-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -Isrc tests/test_rwlock_rp.c -o out/rwlock-rp-test
    out/rwlock-rp-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_freeze.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/freeze-test
    out/freeze-test
    "$CC" -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -Werror -I. -Isrc tests/test_content.c src/runtime_content.c -o out/content-test
    out/content-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_mp3.c src/runtime_mp3.c out/libatrac9.a -lm "${libraries[@]}" -o out/mp3-test
    out/mp3-test tests/data/sine.mp3
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_videodec.c "${libraries[@]}" -o out/videodec-test
    out/videodec-test tests/data/tiny.h264
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_services.c -o out/services-test
    out/services-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -mbmi -mlzcnt -mmovbe -I. -Isrc tests/test_isa_emulate.c out/isa_emulate.o -o out/isa-emulate-test
    out/isa-emulate-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -c src/isa_rewrite.c -I. -Isrc -o out/isa_rewrite.o
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -mbmi -mlzcnt -I. -Isrc tests/test_isa_stub.c out/isa_rewrite.o out/isa_emulate.o -o out/isa-stub-test
    out/isa-stub-test tests/data/isa_fixture.bin
    # Windows red-zone protection: patched fixture functions survive a Windows-like clobber below rsp.
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -I. -Isrc tests/test_red_zone.c out/isa_rewrite.o out/isa_emulate.o -o out/red-zone-test
    out/red-zone-test tests/data/red_zone_image.bin tests/data/red_zone_table.bin
    # SSE4A (EXTRQ/INSERTQ): the CPU is the oracle on AMD; elsewhere recorded AMD results.
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -msse4a -I. -Isrc tests/test_isa_sse4a.c out/isa_rewrite.o out/isa_emulate.o -o out/isa-sse4a-test
    out/isa-sse4a-test
    # F16C (VCVTPH2PS/VCVTPS2PH, #51): the CPU is the oracle where it has F16C; elsewhere recorded results.
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -mavx -mf16c -I. -Isrc tests/test_isa_f16c.c out/isa_rewrite.o out/isa_emulate.o -o out/isa-f16c-test
    out/isa-f16c-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -UNDEBUG -I. -Isrc tests/test_cpu_check.c -o out/cpu-check-test
    out/cpu-check-test
fi
