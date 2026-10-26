#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""The profiling session report (docs/profiling.md).

    python tools/session_report.py <session.zip or folder> [--compare <other>] [-o report.html]

Reads what a session recorded (scripts/profile_session.py) and writes one self-contained HTML page:
the frames (rate, lows, percentiles, stutters), what each frame waited on (the GPU, the command
processor, presentation, readbacks, pipeline builds, the game's own threads), the worst frames,
every thread's CPU and run-queue wait, the machine (clocks, loads, power, temperatures, memory),
the costliest profiler zones, and the game's own code (the hottest functions and a flame graph per
thread, from the guest samples). With --compare, a second session beside it, metric by metric.

The Tracy capture is read with tracy-csvexport (port/out, tools/, or --csvexport); without it the
report has everything but the zones, the GPU times and the per-frame breakdown.
"""
import argparse
import bisect
import collections
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PORT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORT.parent / 'launcher'))
csv.field_size_limit(1 << 24)

# Per-frame categories: the zones (substrings of their names) whose time inside a frame counts.
WAIT_PRESENT = ('Wait for the frame\'s GPU work', 'Wait for a free frame', 'Present acquire swapchain image')
COMPILE = ('Driver: create', 'Shader translate', 'Shader compile')


def log(text):
    print(text, file=sys.stderr, flush=True)


# ---- the session --------------------------------------------------------------------------------

def open_session(path, scratch):
    """The session's folder: `path` itself, or a zip unpacked into scratch."""
    path = Path(path)
    if path.is_dir():
        return path
    with zipfile.ZipFile(path) as z:
        z.extractall(scratch)
    folders = [p for p in Path(scratch).iterdir() if p.is_dir()]
    return folders[0] if len(folders) == 1 else Path(scratch)


def read_frames(timeline):
    """The presented game frames (present_end of a new game frame: a = reused, b = game frame) and
    the timeline's own counts. Steady clock ns."""
    presents, counts, readback_waits, misses = [], collections.Counter(), [], 0
    if not timeline.is_file():
        return presents, counts, readback_waits, misses
    with open(timeline, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 11:
                continue
            event = row[0]
            counts[event] += 1
            if event == 'present_end' and row[6] == '0' and row[7] == '1':
                presents.append(int(row[1]))
            elif event == 'readback_wait_end':
                readback_waits.append((int(row[1]), int(row[10])))
            elif event == 'pipeline_miss':
                misses += 1
    return presents, counts, readback_waits, misses


def read_sensors(path):
    """name -> [(time_ns, value)]."""
    series = collections.defaultdict(list)
    if not path.is_file():
        return series
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) == 3:
                try:
                    series[row[1]].append((int(row[0]), float(row[2])))
                except ValueError:
                    pass
    return series


# ---- the Tracy capture (tracy-csvexport) --------------------------------------------------------

def find_csvexport(given):
    name = 'tracy-csvexport' + ('.exe' if os.name == 'nt' else '')
    for candidate in ([Path(given)] if given else []) + [PORT / 'out' / name, PORT / 'tools' / name, PORT / 'bin' / name]:
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


class Capture:
    def __init__(self, tool, trace):
        self.tool, self.trace = tool, trace

    def rows(self, *args):
        process = subprocess.Popen([str(self.tool), *args, str(self.trace)], stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace')
        reader = csv.reader(process.stdout)
        header = next(reader, None)
        if header:
            for row in reader:
                yield dict(zip(header, row))
        process.wait()

    def clock_offset(self):
        """steady_ns - tracy_ns, from the TLG_CLOCK message (gpu/shim/tlg_profile.cpp)."""
        for row in self.rows('-m'):
            name = row.get('MessageName', '')
            if name.startswith('TLG_CLOCK '):
                return int(name.split()[1]) - int(row['total_ns'])
        return None

    def zone_stats(self):
        out = []
        for row in self.rows('-e'):
            try:
                out.append({'name': row['name'], 'file': row['src_file'].replace('\\', '/').split('/')[-1] + ':' + row['src_line'],
                            'self_ms': int(row['total_ns']) / 1e6, 'count': int(row['counts']),
                            'mean_us': int(row['mean_ns']) / 1e3, 'max_ms': int(row['max_ns']) / 1e6})
            except (KeyError, ValueError):
                pass
        return sorted(out, key=lambda z: -z['self_ms'])

    def events(self, pattern):
        """(name, start tracy ns, length ns, thread) of the zones whose name holds `pattern`."""
        for row in self.rows('-u', '-f', pattern):
            try:
                yield row['name'], int(row['ns_since_start']), int(row['exec_time_ns']), row['thread']
            except (KeyError, ValueError):
                continue

    def plots(self):
        """name -> [(tracy ns, value)] (csvexport -p reports every plot)."""
        series = collections.defaultdict(list)
        for row in self.rows('-u', '-p', '-f', '\x01no zone has this name'):
            if row.get('value') not in (None, ''):
                try:
                    series[row['name']].append((int(row['ns_since_start']), float(row['value'])))
                except ValueError:
                    pass
        return series

    def gpu(self):
        out = []
        for row in self.rows('-g'):
            try:
                out.append((int(row['Time from start of program']), int(row['GPU execution time'])))
            except (KeyError, ValueError):
                pass
        return out


# ---- per frame ----------------------------------------------------------------------------------

class Frames:
    """Sums, per frame, the time of intervals inside it (steady ns)."""

    def __init__(self, presents):
        self.starts = presents[:-1]
        self.ends = presents[1:]
        self.ms = [(b - a) / 1e6 for a, b in zip(self.starts, self.ends)]

    def add(self, sums, start, length):
        if not self.starts or length <= 0:
            return
        end = start + length
        i = max(bisect.bisect_right(self.starts, start) - 1, 0)
        while i < len(self.starts) and self.starts[i] < end:
            overlap = min(end, self.ends[i]) - max(start, self.starts[i])
            if overlap > 0:
                sums[i] += overlap / 1e6
            i += 1

    def series(self):
        return [0.0] * len(self.starts)


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]


def frame_summary(ms):
    if not ms:
        return {}
    total = sum(ms)
    worst = sorted(ms, reverse=True)
    low1 = worst[:max(1, len(ms) // 100)]
    median = percentile(ms, 50)
    return {
        'frames': len(ms), 'seconds': total / 1000.0, 'fps': 1000.0 * len(ms) / total if total else 0,
        'low1_fps': 1000.0 / (sum(low1) / len(low1)), 'p50_ms': median, 'p95_ms': percentile(ms, 95),
        'p99_ms': percentile(ms, 99), 'max_ms': worst[0], 'stutters': sum(1 for m in ms if m > 2 * median and m > 25),
        'under30_share': 100.0 * sum(1 for m in ms if m > 1000 / 30 + 0.5) / len(ms),
    }


# What a frame waited on: the first category that took most of it (heuristic; docs/profiling.md).
def classify(f, ms):
    if ms <= 0:
        return 'unknown'
    if f['compile'] > 0.3 * ms:
        return 'pipeline build'
    if f['readback'] > 0.3 * ms:
        return 'GPU readback'
    if f['gpu'] > 0.85 * ms:
        return 'GPU'
    if f['cp_busy'] > 0.85 * ms:
        return 'command processor'
    if f['main_run'] > 0.85:
        return 'game thread'
    return 'sync / pacing'


# ---- the game's code (guest samples) ------------------------------------------------------------

def read_functions(path):
    if not path.is_file():
        return [], []
    data = path.read_bytes()
    if data[:8] != b'TLGFUNC1':
        return [], []
    count, = struct.unpack_from('<I', data, 8)
    pairs = struct.unpack_from(f'<{2 * count}I', data, 12)
    return list(pairs[0::2]), list(pairs[1::2])


def read_modules(path):
    """[(base, size, file name)] (Windows: "base size path"; Linux: /proc/self/maps)."""
    modules = []
    if not path.is_file():
        return modules
    spans = {}
    for line in path.read_text(errors='replace').splitlines():
        parts = line.split()
        if len(parts) >= 3 and '-' not in parts[0]:
            path = ' '.join(parts[2:])
            modules.append((int(parts[0], 16), int(parts[1], 16), Path(path.replace('\\', '/')).name, path))
        elif len(parts) >= 6 and '-' in parts[0]:
            lo, hi = (int(x, 16) for x in parts[0].split('-'))
            name = Path(parts[5]).name
            a, b, _ = spans.get(name, (lo, hi, parts[5]))
            spans[name] = (min(a, lo), max(b, hi), parts[5])
    for name, (lo, hi, path) in spans.items():
        modules.append((lo, hi - lo, name, path))
    return sorted(modules)


def pe_exports(path):
    """[(rva, name)] of a DLL's exported functions (system DLLs keep no symbol table)."""
    data = Path(path).read_bytes()
    pe, = struct.unpack_from('<I', data, 0x3C)
    if data[pe:pe + 4] != b'PE\0\0':
        return []
    count, = struct.unpack_from('<H', data, pe + 6)
    optional_size, = struct.unpack_from('<H', data, pe + 20)
    optional = pe + 24
    if struct.unpack_from('<H', data, optional)[0] != 0x20B:  # PE32+ only
        return []
    export_rva, = struct.unpack_from('<I', data, optional + 112)
    sections = [struct.unpack_from('<8sIIII', data, optional + optional_size + 40 * i) for i in range(count)]

    def offset(rva):
        for _, vsize, vaddr, rsize, raw in sections:
            if vaddr <= rva < vaddr + max(vsize, rsize):
                return raw + rva - vaddr
        return None
    at = offset(export_rva) if export_rva else None
    if at is None:
        return []
    _, n_names, functions, names, ordinals = struct.unpack_from('<IIIII', data, at + 20)
    f_at, n_at, o_at = offset(functions), offset(names), offset(ordinals)
    out = []
    for i in range(n_names):
        name_at = offset(struct.unpack_from('<I', data, n_at + 4 * i)[0])
        ordinal, = struct.unpack_from('<H', data, o_at + 2 * i)
        rva, = struct.unpack_from('<I', data, f_at + 4 * ordinal)
        end = data.index(b'\0', name_at)
        out.append((rva, data[name_at:end].decode(errors='replace')))
    return sorted(out)


def elf_functions(path):
    """[(address, name)] of an ELF's function symbols (.symtab, else .dynsym)."""
    data = Path(path).read_bytes()
    if data[:4] != b'\x7fELF' or data[4] != 2:
        return []
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', data, 0x3A)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    out = []
    for kind in (2, 11):  # SHT_SYMTAB, SHT_DYNSYM
        for _, stype, _, _, offset, size, link, _, _, entsize in sections:
            if stype != kind or not entsize:
                continue
            strtab = sections[link]
            for i in range(size // entsize):
                name, info, _, _, value, _ = struct.unpack_from('<IBBHQQ', data, offset + i * entsize)
                if info & 0xF == 2 and value:  # STT_FUNC
                    end = data.index(b'\0', strtab[4] + name)
                    out.append((value, data[strtab[4] + name:end].decode(errors='replace')))
        if out:
            break
    return sorted(out)


class HostSymbols:
    """Names for addresses outside the game's image: the module, and the function when the
    module's file (with symbols) is found locally (--symbols folders)."""

    def __init__(self, modules, folders):
        self.modules = modules
        self.bases = [m[0] for m in modules]
        self.folders = [Path(f) for f in folders]
        self.tables = {}

    def table(self, name, path=''):
        """The module's functions: its file where it ran (the same machine), else a build of it in
        the symbol folders."""
        if name not in self.tables:
            table = []
            for candidate in ([Path(path)] if path else []) + [folder / name for folder in self.folders]:
                if candidate.is_file():
                    try:
                        if name.lower().endswith('.exe'):
                            from fl_launcher import symbols
                            table = symbols.coff_functions(candidate)
                        elif name.lower().endswith('.dll'):
                            table = pe_exports(candidate)
                        else:
                            table = elf_functions(candidate)
                    except (OSError, ValueError, struct.error, ImportError):
                        table = []
                    if table:
                        break
            self.tables[name] = ([a for a, _ in table], [n for _, n in table])
        return self.tables[name]

    def label(self, address):
        i = bisect.bisect_right(self.bases, address) - 1
        if i < 0 or address >= self.modules[i][0] + self.modules[i][1]:
            return f'[{address:#x}]'
        base, _, name, path = self.modules[i]
        rva = address - base
        starts, names = self.table(name, path)
        j = bisect.bisect_right(starts, rva) - 1
        return f'{name}!{names[j]}' if j >= 0 else f'{name}+{rva:#x}'


def read_samples(path, functions, host, frames):
    """Per thread: running and waiting sample counts, the hottest functions, folded stacks; and per
    frame the share of the main thread's samples that ran."""
    result = {'threads': {}, 'hz': 0, 'functions': collections.Counter(), 'image': [0, 0]}
    if not path.is_file():
        return result, []
    data = path.read_bytes()
    if data[:8] not in (b'TLGSMPL1', b'TLGSMPL2'):
        return result, []
    flagged = data[:8] == b'TLGSMPL2'  # a flags byte: bit 0, in a system call
    base, size, hz, _ = struct.unpack_from('<QQII', data, 8)
    result['hz'], result['image'] = hz, [base, size]
    starts, ends = functions
    names, labels = {}, {}

    def label(address):
        if address in labels:
            return labels[address]
        if base <= address < base + size:
            offset = address - base
            j = bisect.bisect_right(starts, offset) - 1
            text = f'game:{starts[j]:x}' if j >= 0 and offset < ends[j] else f'game+{offset:x}'
        else:
            text = host.label(address)
        labels[address] = text
        return text

    threads = {}
    main_run = [0] * len(frames.starts)
    main_all = [0] * len(frames.starts)
    p = 32
    while p < len(data):
        kind = data[p]
        p += 1
        if kind == 0x54:  # 'T'
            tid, n = struct.unpack_from('<IB', data, p)
            p += 5
            names[tid] = data[p:p + n].decode(errors='replace')
            p += n
            continue
        if kind == 0x52:  # 'R'
            ns, tid = struct.unpack_from('<QI', data, p)
            p += 12
            flags = data[p] if flagged else 0
            p += flagged
            n = data[p]
            p += 1
            rip, = struct.unpack_from('<Q', data, p)
            chain = struct.unpack_from(f'<{n}Q', data, p + 8)
            p += 8 + 8 * n
            # In a system call: blocked in the OS (a wait the runtime does not mark), named by the
            # first of the port's or the game's frames above it.
            state = None
            if flags & 1:
                caller = next((label(a) for a in chain if not label(a).startswith(('ntdll', 'KERNEL', 'kernelbase', 'libc', '['))), label(rip))
                state = 'system call (' + caller + ')'
        elif kind == 0x57:  # 'W'
            ns, tid, n = struct.unpack_from('<QIB', data, p)
            p += 13
            state = data[p:p + n].decode(errors='replace')
            p += n
        else:
            break
        name = names.get(tid, str(tid))
        t = threads.setdefault(name, {'run': 0, 'wait': collections.Counter(), 'self': collections.Counter(),
                                      'stacks': collections.Counter()})
        if state is None:
            t['run'] += 1
            leaf = label(rip)
            t['self'][leaf] += 1
            result['functions'][leaf] += 1
            stack = [label(a) for a in reversed(chain)] + [leaf]
            t['stacks'][';'.join(stack)] += 1
        else:
            t['wait'][state] += 1
        if name == 'main' and frames.starts:
            i = bisect.bisect_right(frames.starts, ns) - 1
            if 0 <= i < len(frames.starts) and ns < frames.ends[i]:
                main_all[i] += 1
                main_run[i] += state is None
    share = [r / a if a else 0.0 for r, a in zip(main_run, main_all)]
    result['threads'] = threads
    return result, share


# ---- the report ---------------------------------------------------------------------------------

def downsample(points, t0, bucket_ns=1_000_000_000):
    """[(seconds since t0, mean)] per bucket."""
    buckets = collections.defaultdict(list)
    for t, v in points:
        buckets[(t - t0) // bucket_ns].append(v)
    return [[k * bucket_ns / 1e9, sum(v) / len(v)] for k, v in sorted(buckets.items()) if k >= 0]


def analyse(folder, csvexport, symbol_folders):
    started = time.time()
    info = {}
    if (folder / 'session.json').is_file():
        info = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    log(f'{folder.name}: frames')
    presents, counts, readback_waits, misses = read_frames(folder / 'timeline.csv')
    frames = Frames(presents)
    t0 = presents[0] if presents else 0
    log(f'{folder.name}: sensors')
    sensors = read_sensors(folder / 'sensors.csv')
    if not t0 and sensors:
        t0 = min(points[0][0] for points in sensors.values() if points)
    per = {k: frames.series() for k in ('gpu', 'cp_idle', 'cp_busy', 'present_wait', 'readback', 'compile', 'main_run')}
    for t, length in readback_waits:  # the timeline's readback waits end at t
        frames.add(per['readback'], t - length, length)
    zones, plots, gpu_total = [], {}, 0.0
    capture_found = (folder / 'session.tracy').is_file()
    tool = find_csvexport(csvexport) if capture_found else None
    if tool:
        capture = Capture(tool, folder / 'session.tracy')
        offset = capture.clock_offset()
        log(f'{folder.name}: zones')
        zones = capture.zone_stats()[:60]
        if offset is not None:
            log(f'{folder.name}: GPU times')
            for start, length in capture.gpu():
                frames.add(per['gpu'], start + offset, length)
                gpu_total += length
            log(f'{folder.name}: waits')
            for name, start, length, _ in capture.events('wait'):
                if name == 'CP wait for work':
                    frames.add(per['cp_idle'], start + offset, length)
                elif name in WAIT_PRESENT:
                    frames.add(per['present_wait'], start + offset, length)
            for pattern in ('Present acquire', 'Driver:', 'Shader'):
                for name, start, length, _ in capture.events(pattern):
                    key = 'compile' if name.startswith(COMPILE) else 'present_wait'
                    frames.add(per[key], start + offset, length)
            log(f'{folder.name}: plots')
            raw = capture.plots()
            spin = sorted((t + offset, v) for t, v in raw.get('CP spin %', []))
            # The command processor's busy time in a frame: awake (not waiting for work) and not
            # spinning on a wait inside the guest's commands.
            spin_t = [t for t, _ in spin]
            for i, (a, b) in enumerate(zip(frames.starts, frames.ends)):
                lo, hi = bisect.bisect_left(spin_t, a), bisect.bisect_left(spin_t, b)
                share = sum(v for _, v in spin[lo:hi]) / (hi - lo) / 100.0 if hi > lo else 0.0
                per['cp_busy'][i] = max(frames.ms[i] - per['cp_idle'][i], 0.0) * (1.0 - share)
            for name in ('CP spin %', 'flip queue'):
                if name in raw:
                    plots[name] = downsample([(t + offset, v) for t, v in raw[name]], t0)
    log(f'{folder.name}: game code')
    modules = read_modules(folder / 'modules.txt')
    host = HostSymbols(modules, symbol_folders)
    samples, main_share = read_samples(folder / 'guest_samples.bin', read_functions(folder / 'functions.bin'), host, frames)
    if main_share:
        per['main_run'] = main_share
    breakdown = []
    for i, ms in enumerate(frames.ms):
        f = {k: per[k][i] for k in per}
        f['ms'] = ms
        f['class'] = classify(f, ms)
        breakdown.append(f)
    classes = collections.Counter(f['class'] for f in breakdown)
    summary = frame_summary(frames.ms)
    if breakdown:
        for k in ('gpu', 'cp_busy', 'present_wait', 'readback', 'compile'):
            summary[f'{k}_ms'] = sum(f[k] for f in breakdown) / len(breakdown)
        summary['main_run_share'] = 100.0 * sum(f['main_run'] for f in breakdown) / len(breakdown)
    summary['pipeline_misses'] = misses
    summary['readback_waits'] = len(readback_waits)
    # Sensors: charts (1 s means) and per-name statistics.
    charts = {name: downsample(points, t0) for name, points in sensors.items() if not name.startswith('thread ')}
    threads_cpu = []
    by_thread = collections.defaultdict(dict)
    for name, points in sensors.items():
        if name.startswith('thread run % ') or name.startswith('thread wait % '):
            kind = 'run' if name.startswith('thread run') else 'wait'
            thread = name.split(' % ', 1)[1]
            values = [v for _, v in points]
            by_thread[thread][kind] = (sum(values) / len(values), max(values))
    for thread, d in by_thread.items():
        run = d.get('run', (0, 0))
        wait = d.get('wait', (None, None))
        threads_cpu.append({'name': thread, 'run_avg': run[0], 'run_max': run[1], 'wait_avg': wait[0], 'wait_max': wait[1]})
    threads_cpu.sort(key=lambda t: -t['run_avg'])
    guest = []
    for name, t in samples['threads'].items():
        total = t['run'] + sum(t['wait'].values())
        guest.append({'name': name, 'samples': total, 'run_share': 100.0 * t['run'] / total if total else 0,
                      'waits': [[k, 100.0 * v / total] for k, v in t['wait'].most_common(4)],
                      'top': [[k, 100.0 * v / max(t['run'], 1)] for k, v in t['self'].most_common(12)],
                      'stacks': [[k, v] for k, v in t['stacks'].most_common(3000)]})
    guest.sort(key=lambda g: -g['run_share'] * g['samples'])
    worst = sorted(range(len(breakdown)), key=lambda i: -breakdown[i]['ms'])[:25]
    log(f'{folder.name}: done in {time.time() - started:.0f} s')
    return {
        'name': folder.name, 'info': info, 'summary': summary, 'classes': classes.most_common(),
        'frames': [[round((frames.starts[i] - t0) / 1e9, 4), round(f['ms'], 3), f['class'],
                    round(f['gpu'], 2), round(f['cp_busy'], 2), round(f['present_wait'], 2),
                    round(f['readback'], 2), round(f['compile'], 2), round(f['main_run'], 2)]
                   for i, f in enumerate(breakdown)],
        'worst': [[round((frames.starts[i] - t0) / 1e9, 3), breakdown[i]] for i in worst],
        'charts': charts, 'plots': plots, 'zones': zones, 'threads': threads_cpu[:40], 'guest': guest[:24],
        'functions': [[k, 100.0 * v / max(sum(samples['functions'].values()), 1)] for k, v in samples['functions'].most_common(40)],
        'timeline_counts': counts.most_common(40), 'tracy': bool(tool), 'capture_found': capture_found,
        'sample_hz': samples['hz'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session')
    parser.add_argument('--compare', help='a second session (the baseline: "before")')
    parser.add_argument('-o', '--out', help='the report (default: <session>-report.html beside it)')
    parser.add_argument('--csvexport', help='tracy-csvexport')
    parser.add_argument('--symbols', action='append', default=[],
                        help='folders with the builds that recorded (their symbols name the host code)')
    a = parser.parse_args()
    symbol_folders = a.symbols + [str(PORT / 'out'), str(PORT / 'bin'), str(PORT / 'out' / 'gpu_v2_profile')]
    symbol_folders += [str(p) for p in sorted((PORT.parent / 'dist' / 'symbols').glob('*'))[::-1]]
    with tempfile.TemporaryDirectory() as scratch:
        sessions = []
        for i, path in enumerate([a.session] + ([a.compare] if a.compare else [])):
            folder = open_session(path, Path(scratch) / str(i))
            sessions.append(analyse(folder, a.csvexport, symbol_folders))
    template = (Path(__file__).with_name('session_report.html')).read_text(encoding='utf-8')
    data = json.dumps({'sessions': sessions}, separators=(',', ':'))
    page = template.replace('/*SESSION_DATA*/null', data.replace('</', '<\\/'))
    source = Path(a.session)
    out = Path(a.out) if a.out else source.with_name(source.stem + '-report.html')
    out.write_text(page, encoding='utf-8')
    log(f'Report: {out}')


if __name__ == '__main__':
    main()
