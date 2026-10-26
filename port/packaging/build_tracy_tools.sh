#!/usr/bin/env bash
# tracy-capture and tracy-csvexport for the profiling build (docs/profiling.md), from the Tracy
# release whose client is third_party/tracy (the capture's protocol must match it), into out/.
# The tarball is checked against its SHA-256. Tracy's CMake fetches its own dependencies
# (capstone, zstd, PPQSort, nlohmann/json) with CPM.
set -euo pipefail
cd "$(dirname "$0")/.."
version=0.14.1
sum=bf4af567e9c7524d07f3caa745fad02fb33bd5694f11910750382d1efbb251c1
work=out/tracy-tools
mkdir -p "$work"
archive="$work/tracy-$version.tar.gz"
[[ -f $archive ]] || curl -fsSL -o "$archive" "https://github.com/wolfpld/tracy/archive/refs/tags/v$version.tar.gz"
echo "$sum  $archive" | sha256sum -c -
rm -rf "$work/src"
mkdir -p "$work/src"
tar -xzf "$archive" -C "$work/src" --strip-components=1
exe=''
[[ ${OS:-} == Windows_NT ]] && exe=.exe
for tool in capture csvexport; do
    # No -march=native (the tools run on other machines) and no LTO; newer C++ libraries no longer
    # bring <atomic> and <cmath> in where Tracy's server code uses them.
    cmake -S "$work/src/$tool" -B "$work/build-$tool" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DNO_ISA_EXTENSIONS=ON -DNO_LTO=ON -DCMAKE_CXX_FLAGS='-include atomic -include cmath' >/dev/null
    ninja -C "$work/build-$tool" "tracy-$tool" > "$work/build-$tool.log" 2>&1 ||
        { tail -40 "$work/build-$tool.log" >&2; echo "tracy-$tool build failed" >&2; exit 1; }
    cp "$work/build-$tool/tracy-$tool$exe" out/
    echo "Built $PWD/out/tracy-$tool$exe"
done
