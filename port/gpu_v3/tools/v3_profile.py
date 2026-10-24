#!/usr/bin/env python3
"""Names the functions in renderer v3's profile lines (TLG_V3_PROFILE=1, glue/profiler.cpp).

    python tools/v3_profile.py <game log> [tlg-probe-v3.exe]

Every "tlg-probe-v3.exe+0x..." in a "V3 profile" line becomes the function's name, through
llvm-symbolizer (MSYS2 CLANG64) on the exe the log came from (default out/tlg-probe-v3.exe).
"""
import re
import subprocess
import sys
from pathlib import Path

IMAGE_BASE = 0x140000000
EXE_NAME = 'tlg-probe-v3.exe'


def main():
    log = Path(sys.argv[1])
    exe = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / 'out' / EXE_NAME
    lines = [line for line in log.read_text(encoding='utf-8', errors='replace').splitlines()
             if line.startswith('V3 profile ')]
    pattern = re.compile(re.escape(EXE_NAME) + r'\+(0x[0-9a-f]+)~?')
    offsets = sorted({int(m.group(1), 16) for line in lines for m in pattern.finditer(line)})
    names = {}
    if offsets:
        query = '\n'.join(hex(IMAGE_BASE + o) for o in offsets) + '\n'
        out = subprocess.run(['llvm-symbolizer', f'--obj={exe}', '--demangle', '--no-inlines',
                              '--output-style=JSON'], input=query, capture_output=True, text=True,
                             check=True).stdout
        import json
        for line in out.splitlines():
            entry = json.loads(line)
            frames = entry.get('Symbol') or []
            name = frames[0].get('FunctionName', '?') if frames else '?'
            names[int(entry['Address'], 16) - IMAGE_BASE] = shorten(name)
    for line in lines:
        print(pattern.sub(lambda m: '[' + names.get(int(m.group(1), 16), m.group(0)) + ']', line))


def shorten(name):
    """Drops argument lists and template arguments, which make a line unreadable."""
    out, depth = [], 0
    for ch in name:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return ''.join(out).replace('  ', ' ') or name


if __name__ == '__main__':
    main()
