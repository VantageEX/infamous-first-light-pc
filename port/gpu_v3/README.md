# renderer v3 (experimental)

A third renderer for this port: axylpup's multithreaded Second Son build of shadPS4, running on this
port's runtime the way renderer v2 (`port/gpu_v2`) runs Castlesea's renderer.

## Status

Experimental: the launcher offers it as "v3 — experimental" with a notice across its window.
On the First Light dump (Ryzen 7 5700X, RTX 4070 SUPER, 1080p): 35-45 FPS in the streets, 60-100 high
up (a flip is presented when it is ready, not held to the next vblank: `TLG_V3_FLIP_IMMEDIATE`), few
stutters. The fork itself reaches 57-60 FPS in the same streets on the same machine, so the gap is in
this port's glue or runtime, not in the renderer; finding it is the next step. The fork's native
draw path (`SHADPS4_NATIVE_*`) is off in `defaults.env`: on this port it draws some buildings black
outside dynamic light and glowing boxes under the neon bubbles. No upscaler (FSR 1 / RCAS of the
launcher do not apply to it yet), Windows only.

Diagnostics: `TLG_V3_STUTTER` (on; long frames with their faults and busiest log sites),
`TLG_V3_THREAD_CPU=1` (frame rate and per-thread load every 10 s), `TLG_V3_PROFILE=1` (sampling
profiler; `tools/v3_profile.py` names its functions), `TLG_V3_LOG=info` (the fork's own reports).

## Origin and credit

- **axylpup**, [shadPS4 `second-son` branch](https://github.com/axylpup/shadPS4/tree/second-son), release
  `second-son-v1.1.1` (commit `528aea02`): `src/video_core`, `src/shader_recompiler`, `src/imgui`,
  `src/common`, the headers of `src/core` and its GnmDriver, VideoOut and kernel event queue sources →
  `shadps4/` (paths kept). Its multithreaded command processing (front/back split, lean workers,
  async recording) is why this renderer exists here: on the First Light dump it held 56-60 FPS where
  renderer v2 does 37-45 (Ryzen 7 5700X, RTX 4070 SUPER).
- **shadPS4** project (v0.17): the emulator that fork is built on.

GPL-2.0, as both. Left out: DLSS (Streamline SDK), RenderDoc, the emulator's fonts and NID table
(this port has its own overlay and loader).

## Changes made here

The first commit of `shadps4/` is the files as they are in `528aea02`; every change after it is this
port's (git log). The glue to this port's runtime (`port/gpu/tlggpu.h`) follows renderer v2's.
