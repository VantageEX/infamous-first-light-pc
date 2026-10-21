# renderer v2 (experimental)

**pt-BR** — Um renderer alternativo (origem abaixo), rodando no Linux sobre o runtime deste port. A interface
que o jogador vê (FPS, menu Insert, "Compilando shaders", diálogos de save) é a deste port.

**en-US** — An alternative renderer (origin below), running on Linux on this port's runtime. What the player
sees (FPS counter, Insert menu, "Compiling shaders", save dialogs) is this port's own.

## Origin and credit

- **Castlesea**, [infamous-second-son-pc-port](https://github.com/Castlesea/infamous-second-son-pc-port)
  commit `8e167dc`: `pc-port/gpu` (native passes, unified memory, ahead-of-time compute shaders, real
  occlusion, the renderer changes) → `gpu/`.
- **weshayoub92**, [shadPS4-perf](https://github.com/weshayoub92/shadPS4-perf) commit `153a5aa7`,
  with Castlesea's `pc-port/third_party/shadPS4-perf-local.patch` applied → `shadps4/` (only the files
  this build uses).
- **shadPS4** project: the video core and shader recompiler both of them build on.
- `compat/stb_image*.h` (public domain, Sean Barrett) and `compat/nlohmann/json.hpp` (MIT, Niels Lohmann).

All GPL-2.0-or-later (except the two header libraries above).

## Changes made here

- `compat/`: the Win32 subset the renderer uses (memory protection, threads, files, SHA-256 through
  OpenSSL), over POSIX.
- `gpu/bridge.cpp`, `gpu/window.cpp`: Linux branches (X11/Wayland surface, no GDI+ loading art, no
  Windows thread priorities); the image hooks installed once this port's loader knows the image
  (`iss_gpu_install_image`).
- `gpu/gpu_host.cpp`: the renderer behind `port/gpu/tlggpu.h` (the runtime's GPU interface), with this
  port's overlay, settings and save dialogs (`port/gpu/shim/tlg_*.cpp`, built with
  `TLG_RENDERER_V2`).
- `gpu/precompile_jobs.inl`: this port's first-start job file (`<user>/cache/CUSA00223-jobs.bin`)
  compiled by this renderer before the game starts. Compute is the renderer's own AOT compile
  (`ISS_AOT_COMPUTE`, needs `ISS_GAME_DIR`).
- `gpu/overrides/vk_presenter.cpp`: the game image fills the window without ImGui docking; this
  port's overlay drawn over it.
- `shadps4/video_core/amdgpu/regs_shader.h`: this port's binary info search (stops at the first
  OrbShdr, checked against the code); `shader_recompiler/frontend/control_flow_graph.cpp`: BP03
  (shadPS4 c34ca21f).
- Static thread-local storage under ~45 KiB (glibc puts it in every thread's stack).

## Build and run

```sh
TLG_RENDERER_V2=1 bash port/build.sh       # also builds out/gpu_v2 and out/tlg-probe-v2[.exe] (clang)
port/scripts/play_v2.sh           # its launcher's defaults (defaults.env), this port's saves
```

`ISS_NATIVE_PASSES=0` is the default in `play_v2.sh`: the native passes crash NVIDIA's Linux
driver (595) in the pool thread's secondary command buffers.

Measured (RTX 2060, Linux, 11% save, scripted route): first start compiles the 9148 graphics records
of the job file in 14 s and 7952 compute shaders in 85 s; in game then 14 graphics pipelines and no
compute shader are compiled (1097 and 21 without the job file).

Windows: built with MSYS2 CLANG64 as a static library linked into `out/tlg-probe-v2.exe`
(`compat/mingw.h` for the MSVC intrinsics MinGW lacks); both CI packages ship it. On Windows the
launcher starts the game with renderer v2 by default, native passes on and unified memory off
(launcher switches "Renderer", "Native passes", "Unified memory"). Measured on two PCs with an RTX 3060
(Ryzen 5 5500 and 4600G, package v2026.10.21-744a854): 11-20 FPS in gameplay with unified memory on,
~40-45 FPS with it off and native passes on, about what Castlesea's release does on the same PC. With
unified memory on, the game's fences wait for the GPU about 28 times a frame. Castlesea's release has
it on too, but there its heap range (0x20000000) never matches the heap, which sits at 0x80000000.
`ISS_NATIVE_PASSES_BLEND_DENY` in `defaults.env` is Castlesea's pause-menu target moved to this
runtime's heap C (0x1068000000 instead of 0x200000000, same offset).

Not yet: mouse capture, the renderer's ImGui still composes the game image, native passes on NVIDIA.

## Versões suportadas / Supported versions

- **pt-BR:** os endereços do renderer v2 (ganchos, contador de pausa, vtables) são do eboot **CUSA00223 (EUA) com o update 1.07** (sha256 `45b7cc97…07ee5`). O log mostra `V2_EBOOT us_1_07=1`. Com outro eboot (base 1.00, outra região) o guarda de pausa se desliga (`V2_SITE pause_counter not found`) e os ganchos recusam pelo prólogo, em vez de ler lixo.
- **en-US:** renderer v2 addresses (hooks, pause counter, vtables) belong to the **CUSA00223 (US) update 1.07** eboot (sha256 `45b7cc97…07ee5`). The log prints `V2_EBOOT us_1_07=1`. With another eboot (base 1.00, other region) the pause guard turns off (`V2_SITE pause_counter not found`) and the hooks refuse on their prologue check instead of reading garbage.
