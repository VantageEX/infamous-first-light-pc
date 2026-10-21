# inFAMOUS First Light — native PC port

[![license](https://img.shields.io/badge/license-GPL--2.0-blue)](LICENSE)
[![platform](https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white)](#playing)
[![C++](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](port/gpu)
[![C](https://img.shields.io/badge/C-11-A8B9CC?logo=c&logoColor=black)](port/src)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-AC162C?logo=vulkan&logoColor=white)](port/gpu)
[![Qt](https://img.shields.io/badge/Qt%20Quick-PySide6-41CD52?logo=qt&logoColor=white)](launcher)
[![release](https://img.shields.io/github/v/release/VantageEX/infamous-first-light-pc?include_prereleases&label=release)](https://github.com/VantageEX/infamous-first-light-pc/releases)
[![downloads](https://img.shields.io/github/downloads/VantageEX/infamous-first-light-pc/total?label=downloads)](https://github.com/VantageEX/infamous-first-light-pc/releases)
[![status](https://img.shields.io/badge/status-early%20preview-orange)](#status)

An unofficial native Windows port of **inFAMOUS First Light** (PS4). The game's own x86-64 code runs directly on the PC's processor, a reimplemented PS4 runtime stands in for the console's system libraries, and the graphics are translated to Vulkan. It is not a general-purpose emulator.

> **You need your own copy of the game**, dumped from your own PS4. No game files, firmware, keys or Sony assets are part of this repository or its releases.

[Русская версия ниже](#русский)

| | |
|---|---|
| ![Fetch using her neon power in the city at night, 42 FPS on renderer v2](docs/screenshots/gameplay-city.jpg) | ![Fetch's neon dash close up, with the performance overlay, 41 FPS](docs/screenshots/gameplay-neon-dash.jpg) |

![The launcher: the game's own art, Play, the newest save to continue and the settings](docs/screenshots/launcher.jpg)

### ▶ Gameplay video

[![inFAMOUS First Light PC Port — Native Gameplay part 2 (YouTube)](https://img.youtube.com/vi/ISWGOfrwtqA/maxresdefault.jpg)](https://youtu.be/ISWGOfrwtqA)

<sub>Ryzen + RTX 4070 SUPER, 1080p. The launcher shows the art of the player's own game files.</sub>

## Status

**Early, playable.** The game boots, reaches the menus and gameplay, saves, and plays through the first missions; Fetch's neon powers, the dash trail, lights and particles render.

| | |
|---|---|
| Tested dump | CUSA00897 (EU), base game 1.00 |
| Frame rate | about 35–45 FPS with renderer v2 (the default), 15–25 with the classic one, on a Ryzen + RTX 4070 SUPER; goal: a steady 30 FPS on mid-range PCs |
| Known issues | the first start prepares and compiles shaders (2–3 minutes once); GPU readbacks cost frame rate (work in progress); the classic renderer flashes yellow for a moment while a save loads; some lights pulse in the boat cutscene |

## Playing

1. Download the [latest release](https://github.com/VantageEX/infamous-first-light-pc/releases/latest) and unzip it anywhere you can write to (not Program Files).
2. Start **inFAMOUS First Light.exe** and pick your game folder (the one with `eboot.bin` and `sce_sys`).
3. Press **Play**. F10 shows the performance overlay; Insert (or R3 + L2) opens the in-game menu.

From v0.3.2 the launcher checks for a new release when it starts and installs it with one click; your saves and settings stay.

Requirements: Windows 10/11 64-bit, a Vulkan 1.3 GPU, a CPU with AVX (Intel CPUs work: the AMD-only instructions the game uses are emulated).

**Dumping your game:** on your own PS4 with GoldHEN, dump the game (and its update, if any) with a homebrew dumper such as Itemzflow, and copy the folder to the PC.

### Controls

Xbox, PlayStation and other controllers work, and so does the keyboard. The touchpad and the motion sensor are simulated on controllers without them:

| In game | Xbox / other controller | Keyboard |
|---|---|---|
| Press the touchpad | View | Tab / Backspace |
| Touchpad swipe | hold View + right stick | hold Tab + arrows |
| Graffiti: controller on its side | L3 + R3 (sensor mode on/off) | hold T or G |
| Graffiti: shake / spray / aim | A / R2 / left stick | Y / F / arrows |

## Reporting performance

Reports from many different PCs are what the optimisation needs:

1. Play 10–15 minutes (the city, running, dashes, a fight). Test builds (with a `TEST_BUILD` file) log extra performance statistics.
2. In the launcher open **Log → Create report**: it writes a zip in `data/reports` (logs and the PC's specs, no personal paths).
3. Open a **[performance report](https://github.com/VantageEX/infamous-first-light-pc/issues/new?template=performance_report.yml)**, fill in the form and drag the zip into it. Crashes and glitches go to a **[bug report](https://github.com/VantageEX/infamous-first-light-pc/issues/new?template=bug_report.yml)**.

Never attach game files.

## Building

In an **MSYS2 CLANG64** shell:

```bash
pacman -S --needed git patch \
  mingw-w64-clang-x86_64-{toolchain,cmake,ninja,pkgconf,python} \
  mingw-w64-clang-x86_64-{vulkan-headers,vulkan-loader,vulkan-memory-allocator,glslang} \
  mingw-w64-clang-x86_64-{sdl3,fmt,boost,robin-map,xxhash,ffmpeg,zydis} \
  mingw-w64-clang-x86_64-{spirv-tools,spirv-headers,spirv-cross}
git clone --recursive <this repository> && cd <repository>
cd port && ISS_LTO=OFF TLG_RENDERER_V2=1 bash build.sh && cd ..
```

The launcher (PySide6): `py -m venv .venv`, `.venv\Scripts\pip install PySide6-Essentials pillow iced-x86`, then `launcher\start.bat`. A release folder: `bash dev/package.sh` (`TEST=1` for a test build). `bash dev/export_public.sh` shows exactly what a push publishes.

| Folder | What it holds |
|---|---|
| `port/` | The core: loader, PS4 runtime (`src/`), Vulkan renderers (`gpu/` classic, `gpu_v2/` renderer v2), game preparation scripts |
| `launcher/` | The launcher (PySide6 + Qt Quick) |
| `dev/` | Packaging, RenderDoc capture, debugging aids |
| `docs/iss_pc/` | The documentation of the project the core comes from |

## Credits and license

This port is a fork of [**iss_pc**](https://github.com/HayatoG/iss_pc) by **HayatoG**, the native PC port of inFAMOUS Second Son; its core (runtime, loader, GPU translation and many fixes) is the foundation of this project. The GPU translation comes from [**shadPS4**](https://github.com/shadps4-emu/shadPS4); iss_pc also credits [tlg_pc](https://github.com/HayatoG/tlg_pc), [gr2_pc](https://github.com/HayatoG/gr2_pc), [gow3_pc](https://github.com/FelipePrado0/gow3_pc), [bloodborne_pc](https://github.com/deadinside28/bloodborne_pc), [ShadowOfTheColossus-PC](https://github.com/AjncSystem/ShadowOfTheColossus-PC) and the other projects listed in [docs/iss_pc](docs/iss_pc/README.md). Renderer v2 (`port/gpu_v2`, by way of iss_pc) is **Castlesea**'s renderer from [infamous-second-son-pc-port](https://github.com/Castlesea/infamous-second-son-pc-port), on **weshayoub92**'s [shadPS4-perf](https://github.com/weshayoub92/shadPS4-perf). ATRAC9 audio via [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9).

Licensed under **GPL-2.0**, like the projects it builds on ([LICENSE](LICENSE)).

*inFAMOUS* is a trademark of Sony Interactive Entertainment. This project is not affiliated with Sony Interactive Entertainment or Sucker Punch Productions.

---

<a id="русский"></a>

# inFAMOUS First Light — нативный порт на ПК

Неофициальный нативный порт **inFAMOUS First Light** (PS4) на Windows. Собственный x86-64 код игры выполняется прямо на процессоре ПК, системные библиотеки консоли заменены переписанной средой выполнения, а графика переводится на Vulkan. Это не универсальный эмулятор.

> **Нужна ваша собственная копия игры**, снятая с вашей PS4. Ни файлов игры, ни прошивки, ни ключей, ни материалов Sony в репозитории и релизах нет.

Скриншоты — [в начале страницы](#infamous-first-light--native-pc-port): неоновая сила Проныры в ночном городе, неоновый рывок с оверлеем производительности и лаунчер. **▶ [Видео геймплея на YouTube](https://youtu.be/ISWGOfrwtqA)**.

## Состояние

**Ранняя версия, игра проходится.** Игра запускается, доходит до меню и геймплея, сохраняется, проходятся первые миссии. Неоновые способности Проныры, след рывка, свет и частицы отображаются.

| | |
|---|---|
| Проверенный дамп | CUSA00897 (EU), базовая игра 1.00 |
| FPS | около 35–45 на рендере v2 (по умолчанию), 15–25 на обычном, Ryzen + RTX 4070 SUPER; цель — стабильные 30 FPS на средних ПК |
| Известные проблемы | первый запуск готовит и компилирует шейдеры (2–3 минуты, один раз); чтение данных с видеокарты стоит FPS (в работе); на обычном рендере при загрузке сохранения на мгновение мелькает жёлтый; в катсцене с лодкой мигают некоторые огни |

## Как играть

1. Скачайте [последний релиз](https://github.com/VantageEX/infamous-first-light-pc/releases/latest) и распакуйте в любую папку, куда можно писать (не в Program Files).
2. Запустите **inFAMOUS First Light.exe** и выберите папку с игрой (в ней `eboot.bin` и `sce_sys`).
3. Нажмите **«Играть»**. F10 показывает оверлей производительности, Insert (или R3 + L2) открывает меню в игре.

Начиная с v0.3.2 лаунчер при запуске проверяет новые релизы и ставит их в один клик; сохранения и настройки остаются.

Нужно: Windows 10/11 64-бит, видеокарта с Vulkan 1.3, процессор с AVX (процессоры Intel тоже подходят: инструкции AMD, которые использует игра, эмулируются).

**Как снять дамп:** на своей PS4 с GoldHEN снимите игру (и обновление, если есть) homebrew-дампером, например Itemzflow, и скопируйте папку на ПК.

### Управление

Работают геймпады Xbox, PlayStation и другие, а также клавиатура. Тачпад и датчик движения эмулируются на геймпадах без них:

| В игре | Xbox / другой геймпад | Клавиатура |
|---|---|---|
| Нажать тачпад | View | Tab / Backspace |
| Свайп по тачпаду | зажать View + правый стик | зажать Tab + стрелки |
| Граффити: геймпад на боку | L3 + R3 (режим датчика вкл/выкл) | зажать T или G |
| Граффити: встряхнуть / распылять / целиться | A / R2 / левый стик | Y / F / стрелки |

## Отчёты о производительности

Для оптимизации нужны отчёты с как можно большего числа разных ПК:

1. Поиграйте 10–15 минут (город, бег, рывки, бой). Сборки для замеров (с файлом `TEST_BUILD`) пишут дополнительную статистику производительности.
2. В лаунчере откройте **«Журнал» → «Собрать отчёт»**: в папке `data/reports` появится архив (логи и характеристики ПК, без личных путей).
3. Откройте **[отчёт о производительности](https://github.com/VantageEX/infamous-first-light-pc/issues/new?template=performance_report.yml)**, заполните форму и перетащите в неё архив. Вылеты и графические ошибки — в **[сообщение об ошибке](https://github.com/VantageEX/infamous-first-light-pc/issues/new?template=bug_report.yml)**.

Файлы игры не прикладывайте.

## Сборка из исходников

Шаги выше, в разделе [Building](#building): MSYS2 CLANG64, `bash build.sh` в `port/`, лаунчер через `launcher\start.bat`, пакет через `bash dev/package.sh`.

## Авторы и лицензия

Этот порт — форк [**iss_pc**](https://github.com/HayatoG/iss_pc) от **HayatoG**, нативного порта inFAMOUS Second Son; его ядро (среда выполнения, загрузчик, перевод графики и множество исправлений) — основа этого проекта. Перевод графики взят из [**shadPS4**](https://github.com/shadps4-emu/shadPS4); остальные проекты, на которых построен iss_pc, перечислены в [docs/iss_pc](docs/iss_pc/README.md). Рендер v2 (`port/gpu_v2`, через iss_pc) — рендер **Castlesea** из [infamous-second-son-pc-port](https://github.com/Castlesea/infamous-second-son-pc-port) на основе [shadPS4-perf](https://github.com/weshayoub92/shadPS4-perf) от **weshayoub92**.

Лицензия **GPL-2.0**, как у проектов, на которых он основан ([LICENSE](LICENSE)).

*inFAMOUS* — товарный знак Sony Interactive Entertainment. Проект не связан с Sony Interactive Entertainment и Sucker Punch Productions.
