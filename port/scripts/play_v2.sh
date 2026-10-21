#!/usr/bin/env bash
# The game with renderer v2 (port/gpu_v2): its launcher's defaults
# (gpu_v2/defaults.env; a variable already set wins) and this port's runtime, launcher
# settings and saves. Build it first: TLG_RENDERER_V2=1 bash port/build.sh.
# ISS_NATIVE_PASSES=0 by default here: the native passes crash NVIDIA's Linux driver (595).
here=$(cd "$(dirname "$0")/.." && pwd)
: "${ISS_NATIVE_PASSES:=0}"
export ISS_NATIVE_PASSES
while IFS='=' read -r key value; do
    [[ -n $key && $key != \#* && -z ${!key+x} ]] && export "$key=$value"
done < "$here/gpu_v2/defaults.env"
export TLG_PROBE=tlg-probe-v2
# The renderer's first-run shader index and compute precompile read the game folder.
if [[ -z ${ISS_GAME_DIR:-} ]]; then
    ISS_GAME_DIR=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/infamous-launcher/settings.json'))).get('game_dir',''))" 2>/dev/null)
fi
export ISS_GAME_DIR
exec "$here/../run_infamous.sh" "$@"
