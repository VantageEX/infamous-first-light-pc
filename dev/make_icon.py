#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Draws the port's icon: a lightning bolt painted in three dry-brush strokes, magenta into
orange like First Light's title lettering, on a white rounded square. Original art drawn from
code (no game or Sony asset); the random seed is fixed, so it redraws the same.

Writes launcher/fl_launcher/assets/icon.png (1024 px) and port/packaging/windows/app.ico (16 to
256 px), the launcher window's and the executables' icon.

Usage: python dev/make_icon.py   (needs Pillow)
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1024
SCALE = 3  # drawn at 3x, then downsampled
SEED = 1405

MAGENTA = (200, 0, 105)
PINK = (238, 20, 95)
ORANGE = (255, 96, 0)
GOLD = (255, 160, 20)

# The bolt's three strokes in a 0..1 square: (start, end, width). Painted top to bottom, each a
# fast slash like the title's letters.
STROKES = [
    ((0.72, 0.07), (0.30, 0.55), 0.19, 0.25),
    ((0.24, 0.55), (0.74, 0.43), 0.12, 0.0),
    ((0.73, 0.42), (0.28, 0.95), 0.20, 0.92),
]


def brush_stroke(draw, rng, start, end, width, taper, size):
    """A dry-brush stroke: many bristles along the stroke, each with its own offset, length,
    thickness and ink; the edges fray and the end breaks up. `taper` (0..1) narrows the stroke
    towards its end, down to a point at 1."""
    (x0, y0), (x1, y1) = [(x * size, y * size) for x, y in (start, end)]
    length = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux
    half = width * size / 2
    bristles = 420
    for i in range(bristles):
        side = (i / (bristles - 1)) * 2 - 1           # -1..1 across the stroke
        edge = abs(side)
        offset = side * half + rng.gauss(0, half * 0.03)
        # the brush starts loaded (full) and runs dry towards the end, more so at the edges
        start_cut = rng.uniform(0, 0.04) + edge ** 4 * rng.uniform(0, 0.2)
        end_cut = rng.uniform(0, 0.05) + edge ** 3 * rng.uniform(0.0, 0.35)
        if edge > 0.6 and rng.random() < 0.35:         # an edge bristle that skips: dry brush
            end_cut = rng.uniform(0.25, 0.75)
        a = start_cut * length
        b = (1 - end_cut) * length
        if b <= a:
            continue
        thick = max(1, round(rng.uniform(1.2, 2.6) * size / 1024 * 3))
        ink = 255 if edge < 0.7 else round(255 * rng.uniform(0.55, 1.0))
        # a slight wobble so the bristles do not look ruled
        wobble = rng.uniform(-0.006, 0.006) * size
        points = []
        steps = 8
        for k in range(steps + 1):
            t = a + (b - a) * k / steps
            bend = math.sin(k / steps * math.pi) * wobble
            narrow = 1 - taper * (t / length) ** 1.3
            off = offset * narrow + bend
            points.append((x0 + ux * t + nx * off, y0 + uy * t + ny * off))
        draw.line(points, fill=ink, width=thick, joint='curve')
    # streaks thrown off along the stroke's path (the title's speed lines)
    for _ in range(18):
        t = rng.uniform(-0.05, 1.05) * length
        o = (rng.choice((-1, 1)) * rng.uniform(0.55, 0.85)) * half * (1 - taper * max(0, t / length))
        run = rng.uniform(0.04, 0.16) * length
        sx, sy = x0 + ux * t + nx * o, y0 + uy * t + ny * o
        draw.line((sx, sy, sx + ux * run, sy + uy * run), fill=round(rng.uniform(150, 255)),
                  width=max(1, round(rng.uniform(0.8, 1.6) * size / 1024 * 3)))


def dry_brush(rng, size):
    """Streaks of thinner paint along the strokes' direction, and a ragged edge: a texture the
    paint is multiplied by (255 = full ink)."""
    small = size // 4
    noise = Image.effect_noise((small * 2, small * 2), 90).filter(ImageFilter.GaussianBlur(1))
    # stretch the noise along the strokes (about 52 degrees from horizontal) into streaks
    streaks = noise.resize((small * 2, small // 6), Image.BILINEAR).resize((small * 2, small * 2), Image.BICUBIC)
    streaks = streaks.rotate(-52, resample=Image.BICUBIC, expand=False)
    box = (small // 2, small // 2, small // 2 + small, small // 2 + small)
    streaks = streaks.crop(box).resize((size, size), Image.BICUBIC)
    # mostly full ink, with darker gaps where the noise dips
    return streaks.point(lambda v: 255 if v > 118 else round(150 + (v / 118) * 105))


def gradient(size):
    """Magenta at the top into orange and gold at the bottom, slightly diagonal."""
    image = Image.new('RGB', (size, size))
    stops = [(0.0, MAGENTA), (0.35, PINK), (0.75, ORANGE), (1.0, GOLD)]
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            t = min(1.0, max(0.0, y / size * 0.85 + x / size * 0.15))
            for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
                if t <= p1:
                    u = (t - p0) / (p1 - p0)
                    pixels[x, y] = tuple(round(a + (b - a) * u) for a, b in zip(c0, c1))
                    break
    return image


def draw():
    rng = random.Random(SEED)
    big = SIZE * SCALE
    plate = Image.new('L', (big, big), 0)
    ImageDraw.Draw(plate).rounded_rectangle((0, 0, big - 1, big - 1), radius=big * 0.22, fill=255)
    icon = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    icon.paste(Image.new('RGBA', (big, big), (255, 255, 255, 255)), (0, 0), plate)

    paint = Image.new('L', (big, big), 0)
    pen = ImageDraw.Draw(paint)
    for start, end, width, taper in STROKES:
        brush_stroke(pen, rng, start, end, width, taper, big)
    paint = paint.filter(ImageFilter.GaussianBlur(big * 0.0008))
    paint = ImageChops.multiply(paint, dry_brush(rng, big))

    colour = gradient(256).resize((big, big), Image.BICUBIC).convert('RGBA')
    # a soft glow of the same colours under the paint, as the title has
    glow = paint.filter(ImageFilter.GaussianBlur(big * 0.025)).point(lambda v: round(v * 0.3))
    icon = Image.composite(colour, icon, ImageChops.multiply(glow, plate))
    icon = Image.composite(colour, icon, paint)
    return icon.resize((SIZE, SIZE), Image.LANCZOS)


def main():
    icon = draw()
    png = ROOT / 'launcher' / 'fl_launcher' / 'assets' / 'icon.png'
    png.parent.mkdir(parents=True, exist_ok=True)
    icon.save(png, optimize=True)
    ico = ROOT / 'port' / 'packaging' / 'windows' / 'app.ico'
    icon.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f'{png.relative_to(ROOT)}, {ico.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
