# SPDX-License-Identifier: GPL-2.0-or-later
"""Makes the port's own trophy sounds, port/sounds/trophy.wav and trophy-platinum.wav (48 kHz,
16-bit stereo), played by the overlay's trophy toast (gpu/shim/tlg_sound.cpp) unless the player
puts theirs in data/sounds. Synthesised here, nothing sampled: soft glassy notes (a sine with a quiet
second and third partial that fade sooner), a little stereo spread and a short echo tail, at
about -11 dBFS so the game stays louder.

    python port/tools/make_trophy_sounds.py
"""
import math
import struct
import wave
from pathlib import Path

RATE = 48000
OUT = Path(__file__).resolve().parent.parent / 'sounds'


def hz(note):
    """A note name (A4, E5, C#6...) in Hz."""
    names = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}
    base = names[note[0]] + (1 if '#' in note else 0)
    octave = int(note[-1])
    return 440.0 * 2 ** ((base + 12 * (octave - 4)) / 12)


def bell(buffer, start, note, length, gain, pan=0.0, brightness=1.0):
    """Adds one soft note to the stereo buffer (lists of floats) at `start` seconds: a sine with
    a quiet second and third partial that die away sooner (a glassy mallet, nothing harsh)."""
    f = hz(note)
    first = int(start * RATE)
    count = int(length * RATE)
    left = math.cos((pan + 1) * math.pi / 4)
    right = math.sin((pan + 1) * math.pi / 4)
    for i in range(count):
        t = i / RATE
        attack = 0.5 - 0.5 * math.cos(math.pi * min(t / 0.012, 1.0))
        body = math.exp(-t * 4.0 / length)
        upper = math.exp(-t * 14.0 / length)
        v = (math.sin(2 * math.pi * f * t)
             + brightness * 0.18 * upper * math.sin(2 * math.pi * 2 * f * t)
             + brightness * 0.05 * upper * math.sin(2 * math.pi * 3 * f * t))
        v *= gain * attack * body
        at = first + i
        if at >= len(buffer[0]):
            break
        buffer[0][at] += v * left
        buffer[1][at] += v * right


def sparkle(buffer, start, length, gain, seed):
    """A few quick, quiet high pings (the platinum's glitter)."""
    state = seed
    for k in range(5):
        state = (state * 1103515245 + 12345) & 0x7fffffff
        note = ['D6', 'G6', 'B6'][state % 3]
        pan = ((state >> 8) % 160) / 100 - 0.8
        bell(buffer, start + k * length / 5, note, 0.4, gain, pan, 0.3)


def echo(buffer, delays=((0.071, 0.28), (0.113, 0.2), (0.167, 0.13))):
    """A short tail: delayed, quieter copies, crossed between the channels."""
    out = [list(buffer[0]), list(buffer[1])]
    for delay, gain in delays:
        d = int(delay * RATE)
        for c in (0, 1):
            src = buffer[1 - c]
            for i in range(d, len(src)):
                out[c][i] += src[i - d] * gain
    return out


def write(path, buffer, peak=0.28):
    top = max(max(abs(v) for v in ch) for ch in buffer) or 1.0
    scale = peak / top
    frames = len(buffer[0])
    fade = int(0.04 * RATE)
    data = bytearray()
    for i in range(frames):
        tail = min(1.0, (frames - i) / fade)
        for c in (0, 1):
            data += struct.pack('<h', int(max(-1.0, min(1.0, buffer[c][i] * scale * tail)) * 32767))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(data))
    print(f'{path} {frames / RATE:.2f} s')


def trophy():
    """A soft note, then one a fifth up that rings a little longer: about 0.8 s."""
    n = int(0.85 * RATE)
    buffer = [[0.0] * n, [0.0] * n]
    bell(buffer, 0.00, 'G4', 0.50, 0.6, -0.15, 0.8)
    bell(buffer, 0.10, 'D5', 0.72, 0.8, 0.1, 1.0)
    bell(buffer, 0.10, 'G3', 0.60, 0.25, 0.0, 0.3)
    return echo(buffer)


def platinum():
    """A rising arpeggio, a held top note with a little glitter, a soft low tail: about 1.5 s."""
    n = int(1.55 * RATE)
    buffer = [[0.0] * n, [0.0] * n]
    for k, note in enumerate(['G4', 'D5', 'G5', 'B5']):
        bell(buffer, k * 0.1, note, 0.8, 0.5, -0.3 + k * 0.2, 0.9)
    bell(buffer, 0.40, 'D6', 1.1, 0.5, 0.0, 0.8)
    bell(buffer, 0.40, 'G5', 1.1, 0.35, -0.2, 0.6)
    bell(buffer, 0.38, 'G3', 1.1, 0.3, 0.0, 0.3)
    sparkle(buffer, 0.5, 0.5, 0.06, 7)
    return echo(buffer)


if __name__ == '__main__':
    write(OUT / 'trophy.wav', trophy())
    write(OUT / 'trophy-platinum.wav', platinum())
