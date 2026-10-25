# SPDX-License-Identifier: GPL-2.0-or-later
"""Names for the core's crash stacks. The core prints the frames of a crash or a failed assert as
"<module>+0x<offset>" (src/probe.c, the renderers' assert handlers); the packaged tlg-probe*.exe
keep their COFF symbol table (only the debug sections are stripped), so the launcher can turn an
offset into "function+0x..." after the game exits and add that to the log."""
import bisect
import re
import struct
from pathlib import Path

FRAME = re.compile(r'\b(tlg-probe[\w-]*\.exe)\+0x([0-9a-fA-F]+)\b')
IMAGE_SCN_MEM_EXECUTE = 0x20000000


def coff_functions(path):
    """[(rva, name)] of the executable's code symbols, sorted; [] when it has no symbol table."""
    data = Path(path).read_bytes()
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        return []
    sections, symbols_at, symbol_count, optional_size = struct.unpack_from('<2xH4xIIH', data, pe + 4)
    if not symbols_at or not symbol_count:
        return []
    table = pe + 24 + optional_size
    rvas, code = [], set()
    for i in range(sections):
        _size, rva, _raw_size, _raw_at = struct.unpack_from('<IIII', data, table + i * 40 + 8)
        characteristics = struct.unpack_from('<I', data, table + i * 40 + 36)[0]
        rvas.append(rva)
        if characteristics & IMAGE_SCN_MEM_EXECUTE:
            code.add(i + 1)
    strings = symbols_at + symbol_count * 18
    out = []
    i = 0
    while i < symbol_count:
        at = symbols_at + i * 18
        raw, value, section, _type, storage, aux = struct.unpack_from('<8sIhHBB', data, at)
        i += 1 + aux
        if section not in code or storage not in (2, 3, 6):   # external, static, label
            continue
        if raw[:4] == b'\0\0\0\0':
            offset = struct.unpack_from('<I', raw, 4)[0]
            end = data.index(b'\0', strings + offset)
            name = data[strings + offset:end].decode('utf-8', 'replace')
        else:
            name = raw.split(b'\0', 1)[0].decode('utf-8', 'replace')
        if not name or name[0] in '.$' or name.startswith('__unnamed'):
            continue
        out.append((rvas[section - 1] + value, name))
    out.sort()
    return out


def symbolize(text, exe_dir):
    """Lines naming every frame "<tlg-probe exe>+0x<offset>" in `text` (each once, in order),
    "" when there is none or no symbols."""
    frames = []
    for match in FRAME.finditer(text):
        frame = (match[1], int(match[2], 16))
        if frame not in frames:
            frames.append(frame)
    if not frames:
        return ''
    tables = {}
    lines = []
    for module, offset in frames:
        if module not in tables:
            try:
                tables[module] = coff_functions(Path(exe_dir) / module)
            except (OSError, ValueError, struct.error):
                tables[module] = []
        table = tables[module]
        index = bisect.bisect_right(table, (offset, '￿')) - 1
        if index < 0:
            continue
        start, name = table[index]
        lines.append(f'  {module}+0x{offset:x} = {name}+0x{offset - start:x}')
    return '\n'.join(lines)
