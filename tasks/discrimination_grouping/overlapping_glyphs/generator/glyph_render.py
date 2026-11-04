"""Portable presentation helpers for overlapping glyphs.

Backgrounds and query frames use the final 448 x 448 canvas. Outline coordinates
and line widths use the canonical 448 x 448 foreground; render them before the
foreground is resized to 368 x 368. The functions do not read files, fonts or
other project modules.
"""

from __future__ import annotations

import colorsys
import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

SIZE = 448
BACKGROUND_STYLES = ('paper', 'blobs', 'bands', 'dots', 'grid', 'waves')
FRAME_SIDES = ('bottom', 'top', 'left', 'right')


def procedural_background(seed: int, style: str | None = None) -> tuple[Image.Image, dict]:
    """Light seeded patterns independent of glyph identities, materials and Q/A."""
    if style is not None and style not in BACKGROUND_STYLES:
        raise ValueError(f'Unknown background style: {style!r}')
    rng = np.random.default_rng(seed)
    if style is None:
        style = BACKGROUND_STYLES[int(rng.integers(len(BACKGROUND_STYLES)))]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    tones = np.array(
        [(249, 245, 234), (237, 247, 249), (245, 238, 249), (238, 248, 237), (250, 239, 234), (244, 245, 247)], float
    )
    base = tones[int(rng.integers(len(tones)))]
    tint = tones[int(rng.integers(len(tones)))] - 18
    field = np.broadcast_to(base, (SIZE, SIZE, 3)).copy()
    params = {}
    if style == 'paper':
        coarse = ndi.gaussian_filter(rng.normal(size=(SIZE, SIZE)), 24)
        coarse /= max(float(coarse.std()), 1e-6)
        field += coarse[..., None] * 1.8 + rng.normal(0, 1.6, (SIZE, SIZE, 1))
    elif style == 'blobs':
        for _ in range(7):
            cx, cy = rng.uniform(-40, SIZE + 40, 2)
            rx, ry = rng.uniform(55, 170, 2)
            alpha = np.exp(-0.5 * (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)) * 0.43
            color = tones[int(rng.integers(len(tones)))] - rng.uniform(4, 13)
            field = field * (1 - alpha[..., None]) + color * alpha[..., None]
    elif style == 'bands':
        angle = rng.uniform(0, np.pi)
        spacing = rng.uniform(38, 75)
        phase = rng.uniform(0, spacing)
        a = ((xx * np.cos(angle) + yy * np.sin(angle) + phase) % spacing) < spacing * rng.uniform(0.23, 0.45)
        field[a] = 0.62 * base + 0.38 * tint
        params = dict(angle=float(angle), spacing=float(spacing))
    elif style == 'waves':
        angle = rng.uniform(0, np.pi)
        spacing = rng.uniform(39, 61)
        u = xx * np.cos(angle) + yy * np.sin(angle)
        v = -xx * np.sin(angle) + yy * np.cos(angle)
        w = (u + 18 * np.sin(v / rng.uniform(40, 90)) + rng.uniform(0, spacing)) % spacing
        alpha = np.exp(-(((w - spacing / 2) / rng.uniform(1.8, 3.2)) ** 2)) * 0.42
        field = field * (1 - alpha[..., None]) + tint * alpha[..., None]
        params = dict(angle=float(angle), spacing=float(spacing))
    else:
        spacing = float(rng.uniform(24, 42))
        ox, oy = rng.uniform(0, spacing, 2)
        dx = np.minimum((xx + ox) % spacing, spacing - (xx + ox) % spacing)
        dy = np.minimum((yy + oy) % spacing, spacing - (yy + oy) % spacing)
        if style == 'dots':
            alpha = np.exp(-(dx * dx + dy * dy) / (2 * rng.uniform(1, 1.8) ** 2)) * 0.58
        else:
            alpha = np.maximum(np.exp(-dx * dx / 1.1), np.exp(-dy * dy / 1.1)) * 0.34
        field = field * (1 - alpha[..., None]) + tint * alpha[..., None]
        params = dict(spacing=spacing, offset=[float(ox), float(oy)])
    im = Image.fromarray(np.clip(field, 217, 255).astype(np.uint8))
    return im, dict(seed=int(seed), style=style, base_rgb=base.astype(int).tolist(), parameters=params)


def draw_query_frames(im: Image.Image, colors: list[str], side: str, seed: int, palette: dict) -> tuple[list, tuple]:
    """Mutate ``im`` with an ordered frame strip; return inclusive boxes and strip.

    The caller keeps the foreground viewport clear of this strip. Colors must
    already be in answer order; this function never selects or reorders targets.
    """
    if im.size != (SIZE, SIZE) or im.mode not in ('RGB', 'RGBA'):
        raise ValueError('Query frames require an RGB or RGBA 448 x 448 image')
    if side not in FRAME_SIDES:
        raise ValueError(f'Unknown frame side: {side!r}')
    if not 1 <= len(colors) <= 8:
        raise ValueError('Query frame count must be between 1 and 8')
    for color in colors:
        if color not in palette:
            raise ValueError(f'Unknown query color: {color!r}')
    n = len(colors)
    gap = 20 if n <= 6 else 8
    box_size = min(49, (SIZE - 36 - (n - 1) * gap) // n)
    width = n * box_size + (n - 1) * gap
    rng = random.Random(seed)
    start = rng.randint(18, SIZE - 18 - width)
    cross = rng.randint(10, 18)
    if side in ('bottom', 'right'):
        cross = SIZE - cross - box_size
    horizontal = side in ('top', 'bottom')
    x0, y0 = (start, cross) if horizontal else (cross, start)
    # A quiet local backing prevents decorative grid cells from resembling frames.
    strip = (
        (x0 - 6, y0 - 6, x0 + width + 5, y0 + box_size + 5)
        if horizontal
        else (x0 - 6, y0 - 6, x0 + box_size + 5, y0 + width + 5)
    )
    d = ImageDraw.Draw(im)
    boxes = []
    d.rounded_rectangle(strip, radius=5, fill=(251, 251, 248))
    for j, color in enumerate(colors):
        x = x0 + j * (box_size + gap) if horizontal else x0
        y = y0 if horizontal else y0 + j * (box_size + gap)
        b = max(4, box_size // 8)
        d.rectangle((x, y, x + box_size - 1, y + box_size - 1), fill=tuple(palette[color]))
        d.rectangle((x + b, y + b, x + box_size - 1 - b, y + box_size - 1 - b), fill='white')
        boxes.append([x, y, x + box_size - 1, y + box_size - 1])
    return boxes, strip


def _sample_color(rng: random.Random) -> tuple[list[int], list[float]]:
    hsv = (rng.random(), rng.uniform(0.45, 0.85), rng.uniform(0.45, 0.85))
    rgb = [round(c * 255) for c in colorsys.hsv_to_rgb(*hsv)]
    return rgb, list(hsv)


def sample_outline(rng: random.Random, center: tuple) -> dict:
    """Sample one hollow colored contour; widths are 2.3--4.1 final 368 px.

    Returned coordinates and widths are JSON serializable and canonical 448 px.
    The contour is fit inside the canvas; its color is independent of glyph colors.
    """
    family = rng.choice(['circle', 'ellipse', 'rectangle', 'triangle', 'pentagon'])
    rx = rng.uniform(32, 94)
    ry = rx if family == 'circle' else rng.uniform(26, 85)
    angle = rng.uniform(0, 2 * math.pi)
    if family == 'rectangle':
        pts = np.array([[-rx, -ry], [rx, -ry], [rx, ry], [-rx, ry]])
    else:
        count = {'triangle': 3, 'pentagon': 5}.get(family, 72)
        a = np.linspace(0, 2 * math.pi, count, endpoint=False)
        pts = np.column_stack((rx * np.cos(a), ry * np.sin(a)))
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    pts = pts @ rot.T
    cx = float(np.clip(center[0] + rng.uniform(-24, 24), 10 - pts[:, 0].min(), 437 - pts[:, 0].max()))
    cy = float(np.clip(center[1] + rng.uniform(-24, 24), 10 - pts[:, 1].min(), 437 - pts[:, 1].max()))
    pts += np.array([cx, cy])
    width = rng.uniform(2.3, 4.1)
    rgb, hsv = _sample_color(rng)
    return dict(
        kind='polygon',
        family=family,
        points=pts.tolist(),
        hollow=True,
        line_width_final_px=width,
        line_width_canonical_px=width / (368 / 448),
        color_rgb=rgb,
        color_hsv=hsv,
    )


def _filled_mask(shape: dict, size: int = SIZE) -> np.ndarray:
    im = Image.new('L', (size, size))
    d = ImageDraw.Draw(im)
    if shape['kind'] == 'ellipse':
        d.ellipse(shape['bbox'], fill=255)
    else:
        d.polygon([tuple(p) for p in shape['points']], fill=255)
    return np.asarray(im) > 127


def raster_outlines(shapes: list[dict], size: int = SIZE) -> np.ndarray:
    """Return a bool union mask with shape (size, size).

    ``size`` selects the raster extent; coordinates are not rescaled. Use 448
    for canonical shapes, then downsample the completed foreground. Each stroke
    is the inward distance band of its filled shape, leaving interiors empty.
    A shape with ``contained_in`` is clipped to that shape's filled area.
    """
    result = np.zeros((size, size), bool)
    for shape in shapes:
        fill = _filled_mask(shape, size)
        if shape.get('hollow', False):
            distance = ndi.distance_transform_edt(np.pad(fill, 1))[1:-1, 1:-1]
            mask = fill & (distance <= shape['line_width_canonical_px'])
        else:
            mask = fill
        # a contained shape stays wholly within its container
        if 'contained_in' in shape:
            mask &= _filled_mask(shape['contained_in'], size)
        result |= mask
    return result


def draw_outlines(foreground: Image.Image, shapes: list[dict]) -> Image.Image:
    """Return an RGBA copy with opaque colored contours and transparent interiors.

    Draw in canonical coordinates before resizing to the final scene viewport.
    The input foreground is not mutated; later contours overpaint earlier ones.
    """
    if foreground.mode != "RGBA" or foreground.width != foreground.height:
        raise ValueError("Outline foreground must be a square RGBA image")
    out = foreground.copy()
    for shape in shapes:
        mask = raster_outlines([shape], size=foreground.width)
        layer = Image.new('RGBA', foreground.size, tuple(shape.get('color_rgb', [163, 169, 170])) + (255,))
        layer.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
        out = Image.alpha_composite(out, layer)
    return out
