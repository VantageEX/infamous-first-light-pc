# renderer v3 (in progress)

A third renderer for this port: axylpup's multithreaded Second Son build of shadPS4, running on this
port's runtime the way renderer v2 (`port/gpu_v2`) runs Castlesea's renderer.

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
