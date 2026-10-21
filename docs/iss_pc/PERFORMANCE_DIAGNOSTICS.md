# CPU/GPU stall diagnostics

This is optional instrumentation, not a performance patch. It does not disable particles,
change readback accuracy or replace the existing frame, regional and readback statistics.

## Build and enable

Build the Windows runtime with MSYS2 CLANG64 as described in
`port/packaging/windows/README.md`. Keep it separate from a playable release.
Set `TLG_PERF_TRACE=1` in the environment of the diagnostic game process. Use a separate
`ISS_USER_DIR` for experiments and preserve the original saves and caches. Never clear a
player's cache automatically. The direct `--play` launcher now prepares the first-start
shader job list just like the graphical launcher; cancellation prevents launch, generator
failure remains visible and playable. A job list existing on disk does not prove precompilation
succeeded: check runtime startup output as well.

For paired runs, `TLG_PERF_TRACE=0 TLG_PERF_FRAMES=1` records only the common frame
series, with detailed hooks disabled. Both variants include this minimal frame recorder;
the comparison measures the additional cost of detailed tracing, not the cost of all
instrumentation against a stock executable. The PowerShell helper accepts `-NoTrace` for
this variant and `-Python <interpreter>` for an isolated launcher environment. Normal
window closure finalizes output before the runtime's fast exit; crash/forced termination
can still leave an incomplete file.

When tracing is on, **Insert > Performance / Desempenho** provides categories and a
**Mark performance event / Marcar evento de desempenho** button. A marker records the user's
observation. It does not automatically recognize a smoke effect and may be later than the event
if opening the menu takes time. Mark before and after the action. No new gameplay key is intercepted.

## Outputs

Files are written below `<GPU user data>/perf-traces/session-<unique-id>/`:

- `frames.csv`: low-volume series of presented intervals, including startup. Its final comment
  records losses/truncation. Missing terminal comment means the session may have crashed or
  been terminated; never treat it as a complete benchmark.
- `capture-*.jsonl`: up to ten detailed captures, normally triggered by an interval >=100 ms,
  a marker, or absence of a presented frame for 100 ms after frames have begun.
  The last condition uses the consumer thread, so a stalled presentation thread need not
  resume before data can be saved. Pre-first-frame boot hangs remain visible only in legacy
  freeze diagnostics; they are not detected by this frame-based trigger.

The history aims for five seconds before and two after a trigger; bursts can reduce coverage.
Overlapping triggers extend a capture only to a bounded seven-second post-open deadline.
Producer rings are preallocated (128 producers x 4,096 events x 64 bytes, 32 MiB); the consumer
keeps up to 65,536 historical events (4 MiB), plus bounded drain/serialization working memory.
A full ring drops and counts events instead of blocking a game/fault thread. Producer slots
are session lifetime registrations, not an unbounded registry. Overflow and registration
failure are reported in the terminal capture status. Files use unique session directories;
no earlier capture is overwritten or deleted.
Detailed capture footers and the `frames.csv` terminal comment also break down losses into `dropped_ring`,
`dropped_unregistered` and `dropped_reentrant`; older captures have only the total.

Session data is capped at 128 MiB, including a frame series capped at 16 MiB. Quotas are
conservative diagnostics defaults; after a capture limit is reached a marker can still enter
the in-memory event stream but a new detailed capture is not promised. A consumer notice
reports the limit; restart a new controlled session if more detailed captures are required.
Hard termination can leave the final file incomplete. File-system errors stop the affected
output, not gameplay. Do not enable other profilers indiscriminately; measure their overhead.

## Report

No Python dependencies are required:

```text
python scripts/perf_trace_report.py capture-1-1.jsonl --out report --frames frames.csv
python scripts/perf_trace_report.py --frames frames.csv --out frames-only-report
```

This writes `report.json` and an interactive `report.html` (expand each stall to inspect its
supporting events). Use `--start-frame N --end-frame M` to calculate series metrics for a
controlled route, rather than menus/boot. Frame p99 uses nearest rank of positive recorded
intervals. Missing rows, losses, truncation or a missing terminal marker mark the series
incomplete. A p99 from a detailed capture is explicitly a biased capture-only statistic,
never a whole-session p99 or a global 1% low.

## Interpretation

Schema 1 uses numeric events with a session-relative monotonic timestamp, observed presentation
frame ID (0 before the first), local thread ID, operation ID and object/generation context.
Frame context on an asynchronous worker does not prove exclusive ownership by that frame.
Complete BEGIN/END pairs establish the duration of the recorded scope. Same-thread nested spans
have a parent operation; compiler queue operation IDs connect enqueue, worker begin/end and nested work; buffer readback request/apply events also share a request ID across threads. The time to apply start includes queue progress, not GPU execution alone; other cross-thread parentage remains unknown when unavailable.
The report unions overlapping intervals rather than adding them as elapsed wall time. It
labels proximity as association, not proof of cause. Missing endpoints remain incomplete.

Pipeline requests distinguish absence in RAM, known pending graphics compilation and lazy
program loading mode. They do **not** establish absence from the offline jobs list or driver
cache. Shader translation events and disk-program loads are separate. Vulkan creation feedback
(core in required Vulkan 1.3) is attached only while diagnostics are enabled and interpreted
only with a valid result and VALID flag. `APPLICATION_PIPELINE_CACHE_HIT` means the application's
cache helped; lack of that bit says nothing conclusive about an internal driver cache.
Base-pipeline acceleration is not a cache hit. Compile scope durations include nested work;
use their union, not a sum, and inspect the actual identity and feedback.

Texture creation, removal, refresh and uploads carry lifetime generation and an opaque
session-salted origin ID. Recreated origin is a churn candidate, not proof of identical texture
content. GC removals distinguish pressure from ordinary collection; other removal reasons remain unknown at the shared unregister site. Reported upload bytes are
logical guest image bytes, not a measurement of actual transfer bandwidth. Memory pressure
is observed at the existing throttled reporting point.

Windows faults use the existing portable `Common::GetRip` abstraction; only an instruction
offset inside the guest range is recorded. No guest stack walking, game memory, raw address,
texture/buffer content or personal path is serialized. Fault producers use fixed numeric events,
lock-free atomics and a monotonic clock, with no file I/O, allocation or blocking locks. Existing
`TLG_FAULT_PROFILE` remains a Linux-only profiler; it has not been copied into a Windows handler.

Reuse `TLG_READBACK_STATS=1` and `TLG_PERF_DIAG=1` in separately measured diagnostic runs.
`TLG_GPU_PROFILE=1` already provides delayed GPU timestamps: a separate run can help distinguish
GPU pass cost from CPU wait time without introducing a GPU-wide wait.

## Tests and rollout

The standalone recorder test does not need Vulkan or MSYS2:

```text
cmake -S port/tests/perf -B <build-dir>
cmake --build <build-dir> --config Debug
ctest --test-dir <build-dir> -C Debug --output-on-failure --timeout 15
python -m unittest discover -s port/tests -p test_cli_precompile.py -v
python -m unittest discover -s port/tests -p test_perf_trace_report.py -v
python -m unittest discover -s port/tests -p test_perf_build_revision.py -v
```

Tests use temporary synthetic data, not the game. Never publish local captures or game assets
with the contribution. Before using results to choose an optimization, run three paired
2–3-minute routes on the same instrumented build with tracing off/on, preserving resolution,
VSync, save, action sequence, driver, memory budget and warmed cache. Initial overhead goals
are <=2% FPS and <=5% frame p99; these are acceptance goals, not measured achievements.
Only then separately test a larger memory budget or different warmup policy. Compare visuals.

Coverage intentionally remains unknown for jobs-list membership, non-compiler cross-thread queue
relationships and some resource removal reasons; do not claim those were implemented. Gameplay
cost, overhead goals and improved FPS require measured runs and cannot be certified by unit tests.
