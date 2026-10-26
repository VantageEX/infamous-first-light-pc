# Profiling sessions

A profiling session records everything about one play session, so that one tester's time gives
enough data to study the game's performance on their machine (first of all the Steam Deck) at
length: what every thread did, what the GPU did, where the game's own code spends the CPU, the
machine's clocks, power and temperatures. One zip comes out of it; `tools/session_report.py` turns it
into a page.

- [For the tester](#for-the-tester) · [Для тестера](#для-тестера)
- [For the developer](#for-the-developer)

## For the tester

You need the **profiling package** (`inFAMOUS-First-Light-PC-linux-x86_64-profiling.tar.gz` for the
Steam Deck, or the `-profiling` zip for Windows). Unpack it like the normal package, in its own
folder; it uses its own `data` folder, so copy your `data/user` folder over if you want your saves.
Use **renderer v2** (launcher → Display → Renderer).

### 1. A free session (you play)

1. Launcher → **Profiling** → turn on **Record a session**. Leave **Benchmark route** on *None*.
2. Press **Play** and play normally for **5–10 minutes**, through what you want measured (flying
   over the city, fights, smoke, the places where the frame rate drops). The game can run a
   little slower than usual while it records.
3. Close the game the normal way (the menu, or Alt+F4). Wait until the launcher's log shows
   **Session recorded: …zip** (packing can take a minute).
4. Send that zip (**Sessions folder** opens it). It can be a few hundred MB.

### 2. A benchmark (the same route every time)

With a recorded route the game plays itself along the same path, so a change can be measured
before and after without you playing again.

1. **Record a route once**, with a gamepad: Profiling → pick a **Length** (e.g. 120 seconds) →
   turn on **Record a route at the next start** → **Play**. From the moment you control Fetch,
   play the route you want for that time (keep it simple and repeatable: start from the same
   save, fly along a street, a small fight). With *Until I close the game* as the length, **F9**
   starts and stops the recording instead.
2. **Run the benchmark**: Profiling → **Record a session** on → **Benchmark route**: the route →
   **Length** the same → **Play**, and do not touch anything. The game presses Cross through the
   menus, plays the route and closes after the length. The session's zip ends in `-benchmark`.
3. Send the zip, and keep the route (`data/routes/route-….txt`) for the next builds.

The benchmark starts from **Continue**, your latest save, and the game's autosaves move it. To
start every run at the same place, copy your `data/user` folder before recording the route and
copy it back before each benchmark.

Things that make the data better: the Steam Deck on its charger or always on battery (the same
every time), the same performance settings in its Quick Access menu, the game's own settings
unchanged between runs, nothing else running.

On the Steam Deck the Tracy part needs nothing extra. (Switches between threads and cores are not
recorded without root; the sensors record each thread's waiting for a free core instead.)

## Для тестера

Нужен **профилировочный пакет** (`inFAMOUS-First-Light-PC-linux-x86_64-profiling.tar.gz` для Steam
Deck или zip с `-profiling` для Windows). Распакуйте его как обычный, в отдельную папку; у него своя
папка `data`, поэтому скопируйте туда свою `data/user`, если нужны ваши сохранения. Рендер —
**v2** (лаунчер → Экран → Рендер).

### 1. Свободная сессия (играете вы)

1. Лаунчер → **Профилирование** → включите **«Записывать сессию»**. **«Маршрут бенчмарка»** —
   *«Нет: играть самому»*.
2. **«Играть»** и играйте как обычно **5–10 минут** там, что нужно измерить (полёт над городом,
   бои, дым, места, где падает FPS). Во время записи игра может идти чуть медленнее.
3. Закройте игру обычным способом. Дождитесь в журнале лаунчера строки **«Сессия записана: …zip»**
   (упаковка может занять минуту).
4. Пришлите этот zip (кнопка **«Папка сессий»** откроет папку). Он может весить несколько сотен МБ.

### 2. Бенчмарк (каждый раз один и тот же маршрут)

С записанным маршрутом игра сама проходит тот же путь, и изменение можно измерить до и после, не
играя заново.

1. **Один раз запишите маршрут**, геймпадом: Профилирование → **«Длительность»** (например,
   120 секунд) → включите **«Записать маршрут при следующем запуске»** → **«Играть»**. С момента,
   когда вы управляете Фетч, пройдите нужный маршрут за это время (простой и повторяемый: с одного
   сохранения, полёт вдоль улицы, небольшой бой). С длительностью *«Пока не закрою игру»* запись
   начинает и останавливает **F9**.
2. **Запуск бенчмарка**: Профилирование → **«Записывать сессию»** вкл → **«Маршрут бенчмарка»**:
   ваш маршрут → та же **«Длительность»** → **«Играть»** и ничего не трогайте. Игра сама нажимает
   Крест в меню, проходит маршрут и закрывается. Архив сессии оканчивается на `-benchmark`.
3. Пришлите zip, а маршрут (`data/routes/route-….txt`) сохраните для следующих сборок.

Бенчмарк начинается с **«Продолжить»**, то есть с последнего сохранения, а автосохранения игры его
сдвигают. Чтобы каждый прогон начинался в одном месте, скопируйте папку `data/user` перед записью
маршрута и возвращайте эту копию перед каждым бенчмарком.

Чтобы данные были точнее: Steam Deck всегда на зарядке (или всегда от батареи — одинаково каждый
раз), одни и те же настройки производительности в его быстром меню, настройки игры не меняются
между прогонами, больше ничего не запущено.

## For the developer

### What a session holds

| File | What | Written by |
|---|---|---|
| `session.tracy` | Tracy capture: zones of the renderer's threads (command processor, recorder, present, uploads, readbacks, pipeline builds), the runtime's waits (`wait: mutex`, `wait: condition variable`, `wait: file read`...), GPU zones per command processor session, plots (sensors, `CP spin %`, `flip queue`), frames (`Frame` presented, `Game flip`) | the profiling build + `tracy-capture` |
| `timeline.csv` | renderer v2's event timeline (`ISS_TIMELINE`), steady clock ns | renderer v2 |
| `sensors.csv` | `time_ns,name,value`: every core's clock and load, GPU load, clocks, power, temperature, CPU temperature, battery (Linux sysfs); each thread's run % and (Linux) run-queue wait % every 100 ms; FPS, VRAM, RAM every 500 ms | `gpu/shim/tlg_profile.cpp` |
| `guest_samples.bin` | 200 samples/s of every game thread: running (RIP, frame chain, in a system call or not) or blocked (the wait's kind) | `src/runtime_sampler.c` |
| `functions.bin`, `modules.txt` | the game image's functions (.eh_frame) and the loaded modules, to name addresses | `scripts/profile_session.py`, the sampler |
| `session.json`, `run.log`, `route.txt` | what ran, the run's log, the benchmark's route | run.py, the launcher |

### Building it

- Windows: `TLG_RENDERER_V2=1 TLG_PROFILE=1 bash build.sh` (adds `out/tlg-probe-v2-profile.exe`),
  `bash packaging/build_tracy_tools.sh` (`out/tracy-capture.exe`, `out/tracy-csvexport.exe`), then
  `PROFILE=1 bash dev/package.sh` for a tester's zip.
- Linux / Steam Deck: Actions → **Linux package** → Run workflow with **profiling** on (branch of
  your choice): the artifact `…-linux-x86_64-profiling`. Never attached to a release.
- The Tracy client is `third_party/tracy` (v0.14.1); `build_tracy_tools.sh` builds the tools from
  the same release (the protocol must match). The viewer: `windows-0.14.1.zip` from
  github.com/wolfpld/tracy releases (`tracy-profiler.exe`).

### Reading it

```
python tools/session_report.py session.zip                      # session-report.html beside it
python tools/session_report.py after.zip --compare before.zip   # side by side, with deltas
```

The page: frame rate, 1% low, percentiles, stutters; each frame's GPU time, command processor
busy time, present waits, readback waits and pipeline builds and **what it waited on** (a pipeline
build or readback over 30% of the frame, else GPU or command processor over 85%, else the game's
main thread running over 85% of its samples, else sync / pacing); the worst frames; threads; the
machine; the costliest zones; the game's threads, hottest functions (`game:xxxx` = the function at
that image offset; `module!name` = the port's code) and a flame graph per thread. For the details,
open `session.tracy` in the Tracy viewer (timeline, zone statistics, GPU zones, plots).

`--symbols <folder>` adds builds whose symbol tables name the port's code (the package that
recorded: `port/out` of its folder).

### Switches

| Variable | |
|---|---|
| `TLG_SESSION_DIR=<folder>` or `auto` | record a session there (`auto`: `data/sessions/<date-time>`) |
| `TLG_SESSION_SECONDS=N` | stop the game after N s (benchmarks, unattended runs) |
| `TLG_SESSION_CAPTURE=0` | no `tracy-capture`: a Tracy viewer connects to the game (port 8086) live |
| `TLG_SESSION_SAMPLE_MS=N` | sensor period (20 ms) |
| `TLG_GUEST_SAMPLE_HZ=N` | guest samples per second (200; 0 off) |
| `TLG_PAD_RECORD=<file>`, `TLG_PAD_RECORD_SECONDS=N` | record a pad route (F9, or N s from gameplay) |
| `TLG_PAD_REPLAY=<file>`, `TLG_PAD_REPLAY_AUTO=1` | replay it (Cross through the menus first) |

### Cost

The profiling build with a capture runs the CPU side a little slower (Tracy's zones are ~10–20 ns
each; the command processor's poll loop has no zones, `CP spin %` stands for it; per-draw zones
remain). Compare sessions with sessions, not with the normal build's frame rate.
