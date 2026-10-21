#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Draws the port's icon: a neon lightning bolt in the spirit of First Light's pink and violet
neon, on a white rounded square. Original art, drawn from code (no game or Sony asset).

Writes launcher/fl_launcher/assets/icon.png (1024 px) and port/packaging/windows/app.ico (16 to
256 px), the launcher window's and the executables' icon.

Usage: python dev/make_icon.py   (needs Pillow)
"""
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SIZE = 1024
SCALE = 4  # drawn at 4x, then downsampled: smooth edges

PINK = (255, 38, 150)
MAGENTA = (240, 60, 230)
VIOLET = (138, 63, 252)
CORE = (255, 236, 250)

# The bolt, in a 0..1 square: slanted like a brush stroke, a wide top and a sharp tip.
BOLT = [(0.62, 0.06), (0.25, 0.55), (0.47, 0.55), (0.35, 0.95), (0.77, 0.39),
        (0.55, 0.39), (0.71, 0.06)]


def scaled(points, size, inset=0.0):
    """Points in pixels, pulled towards the bolt's centre by `inset` (0..1) for inner strokes."""
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return [((cx + (x - cx) * (1 - inset)) * size, (cy + (y - cy) * (1 - inset)) * size)
            for x, y in points]


def gradient(size, top, bottom, angle_mix=0.35):
    """A diagonal gradient from `top` to `bottom`."""
    image = Image.new('RGB', (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(0, size, 1):
            t = min(1.0, max(0.0, (y / size) * (1 - angle_mix) + (x / size) * angle_mix))
            pixels[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(top, bottom))
    return image


def mask(points, size, inset=0.0, blur=0.0):
    m = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m).polygon(scaled(points, size, inset), fill=255)
    return m.filter(ImageFilter.GaussianBlur(blur)) if blur else m


def draw():
    big = SIZE * SCALE
    # white rounded square with a faint inner shade so it reads on a white desktop
    base = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    plate = Image.new('L', (big, big), 0)
    ImageDraw.Draw(plate).rounded_rectangle((0, 0, big - 1, big - 1), radius=big * 0.22, fill=255)
    white = Image.new('RGBA', (big, big), (255, 255, 255, 255))
    shade = Image.new('RGBA', (big, big), (236, 232, 244, 255))
    vignette = Image.new('L', (big, big), 0)
    ImageDraw.Draw(vignette).ellipse((-big * 0.25, -big * 0.25, big * 1.25, big * 1.25), fill=255)
    vignette = ImageChops.invert(vignette.filter(ImageFilter.GaussianBlur(big * 0.08)))
    face = Image.composite(shade, white, vignette)
    base.paste(face, (0, 0), plate)

    # neon glow: wide soft pink, then a tighter magenta halo
    for colour, inset, blur, alpha in ((PINK, -0.10, big * 0.05, 0.55), (MAGENTA, -0.03, big * 0.018, 0.75)):
        glow_mask = mask(BOLT, big, inset, blur).point(lambda v: round(v * alpha))
        base = Image.composite(Image.new('RGBA', (big, big), colour + (255,)), base,
                               ImageChops.multiply(glow_mask, plate))
    # the bolt: pink to violet
    body = gradient(big // 8, PINK, VIOLET).resize((big, big), Image.BICUBIC).convert('RGBA')
    base = Image.composite(body, base, mask(BOLT, big))
    # the neon tube: a bright pale line just inside the bolt's edge, as on a neon sign
    tube = Image.new('L', (big, big), 0)
    outline = scaled(BOLT, big, inset=0.16)
    ImageDraw.Draw(tube).line(outline + outline[:1], fill=255, width=round(big * 0.016), joint='curve')
    tube = ImageChops.multiply(tube.filter(ImageFilter.GaussianBlur(big * 0.003)), mask(BOLT, big))
    tube_glow = tube.filter(ImageFilter.GaussianBlur(big * 0.012)).point(lambda v: min(255, v * 2))
    base = Image.composite(Image.new('RGBA', (big, big), (255, 170, 235, 255)), base,
                           ImageChops.multiply(tube_glow, mask(BOLT, big)))
    base = Image.composite(Image.new('RGBA', (big, big), CORE + (255,)), base, tube)
    return base.resize((SIZE, SIZE), Image.LANCZOS)


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
