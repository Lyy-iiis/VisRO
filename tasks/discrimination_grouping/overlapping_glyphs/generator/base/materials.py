"""Seeded glyph materials; only Pillow, NumPy and SciPy are required.

Render each glyph separately before scene compositing. This module takes masks
only: fonts, identity, rotation, stroke scaling and scene order belong to callers.
"""
from __future__ import annotations
from visro.seeding import seed_text

import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .rope_renderer import render_rope

# recorded as materials_version in every item's metadata
LABEL = seed_text(2259522025802877623524870799437105)
ID_MATERIALS = ("plain", "furry", "beads")
OOD_MATERIALS = ("braided", "relief")


def _rgba(mask, rgb, shade=1.0):
    alpha = np.asarray(mask, dtype=np.uint8)
    out = np.empty((*alpha.shape, 4), dtype=np.uint8)
    shading = np.asarray(shade)[..., None]
    out[..., :3] = np.clip(np.asarray(rgb) * shading, 0, 255)
    out[..., 3] = alpha
    return Image.fromarray(out, "RGBA")


def _tint(rgb, factor, lift=0.0):
    return tuple(int(v) for v in np.clip(np.asarray(rgb) * factor + lift, 0, 255))


def _furry(mask, rgb, seed, stroke):
    rng = np.random.default_rng(seed)
    binary = np.asarray(mask) > 127
    dist = ndi.distance_transform_edt(binary)
    signed = dist - ndi.distance_transform_edt(~binary)
    gy, gx = np.gradient(ndi.gaussian_filter(signed, max(.6, stroke / 30)))
    shade = .62 + .30 * np.minimum(dist / max(1, .27 * stroke), 1)
    shade += ndi.gaussian_filter(rng.normal(size=dist.shape), max(.5, stroke / 34)) * .12
    width, height = mask.size
    canvas = _rgba(mask, rgb, shade).resize((width * 2, height * 2), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    scale = max(.18, stroke / 60.0)

    def fiber(x, y, angle, length, bend, color, pen):
        vx, vy = math.cos(angle), math.sin(angle)
        points = []
        for t in (0., .2, .4, .6, .8, 1.):
            u = bend * math.sin(math.pi * t * .8)
            points.append((2 * (x + vx * length * t - vy * u),
                           2 * (y + vy * length * t + vx * u)))
        draw.line(points, fill=(*color, 245), width=pen, joint="curve")

    edge = np.argwhere(binary & (dist <= max(1, .054 * stroke)))
    count = min(6500, max(64, int(len(edge) * 1.35 / max(1, scale))))
    for index in rng.integers(0, len(edge), size=count):
        y, x = edge[index]
        angle = math.atan2(-gy[y, x], -gx[y, x]) + rng.normal(0, .48)
        fiber(x, y, angle, rng.uniform(5, 17) * scale,
              rng.uniform(-4, 4) * scale, _tint(rgb, rng.uniform(.72, 1.27), rng.uniform(0, 10)),
              max(1, round(rng.choice((1, 2, 2, 3)) * scale)))
    inside = np.argwhere(binary)
    count = min(14500, max(100, int(len(inside) / max(1, scale * scale) * .38)))
    for index in rng.integers(0, len(inside), size=count):
        y, x = inside[index]
        angle = -.9 + .55 * math.sin(x / (58 * scale)) + .45 * math.cos(y / (81 * scale))
        angle += rng.normal(0, .30)
        if dist[y, x] < 8 * scale:
            angle = math.atan2(-gy[y, x], -gx[y, x]) + rng.normal(0, .55)
        fiber(x, y, angle, rng.uniform(4, 12) * scale,
              rng.uniform(-2.7, 2.7) * scale, _tint(rgb, rng.uniform(.72, 1.25), rng.uniform(0, 10)),
              max(1, round(rng.choice((1, 2, 2)) * scale)))
    return canvas.resize(mask.size, Image.Resampling.LANCZOS)


def _beads(mask, rgb, seed, stroke):
    rng = np.random.default_rng(seed)
    dist = ndi.distance_transform_edt(np.asarray(mask) > 127)
    width, height = mask.size
    canvas = Image.new("RGBA", (width * 2, height * 2))
    bead = max(1.35, stroke * rng.uniform(.095, .112))
    step_x, step_y = bead * 2.18, bead * 1.91
    origin_x, origin_y = rng.uniform(0, step_x), rng.uniform(0, step_y)
    for row, y0 in enumerate(np.arange(origin_y, height, step_y)):
        for x0 in np.arange(origin_x, width, step_x):
            x = x0 + (row % 2) * step_x / 2 + rng.uniform(-.28, .28) * bead
            y = y0 + rng.uniform(-.25, .25) * bead
            xi, yi = round(x), round(y)
            if not (0 <= xi < width and 0 <= yi < height) or dist[yi, xi] < bead * .45:
                continue
            radius = bead * rng.uniform(.91, 1.08)
            rad = math.ceil(radius * 2 + 2)
            yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1] / (radius * 2)
            r2 = xx * xx + yy * yy
            z = np.sqrt(np.maximum(0, 1 - r2))
            light = np.clip(-.42 * xx - .50 * yy + .77 * z, 0, 1)
            spec = np.exp(-((xx + .32) ** 2 + (yy + .40) ** 2) / .045) * .22
            shade = (.40 + .72 * light) * rng.uniform(.90, 1.10)
            color = np.asarray(rgb)[None, None, :] * shade[..., None]
            color += (255 - color) * spec[..., None]
            patch = np.empty((*r2.shape, 4), dtype=np.uint8)
            patch[..., :3] = np.clip(color, 0, 255)
            patch[..., 3] = np.clip((1 - np.sqrt(r2)) * radius * 2 + .5, 0, 1) * 255
            canvas.alpha_composite(Image.fromarray(patch, "RGBA"),
                                   (round(x * 2) - rad, round(y * 2) - rad))
    return canvas.resize(mask.size, Image.Resampling.LANCZOS)


def _relief(mask, rgb, seed, stroke):
    rng = np.random.default_rng(seed)
    dist = ndi.distance_transform_edt(np.asarray(mask) > 127)
    bevel = max(1.2, stroke * rng.uniform(.14, .22))
    height = ndi.gaussian_filter(np.minimum(dist, bevel), max(.45, stroke / 86))
    gy, gx = np.gradient(height)
    nx, ny, nz = -gx, -gy, .85
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    angle = rng.uniform(-.15, .15)
    lx, ly = -.42 * math.cos(angle) + .58 * math.sin(angle), -.58 * math.cos(angle)
    light = np.clip((nx * lx + ny * ly + nz * .70) / norm, 0, 1)
    texture = ndi.gaussian_filter(rng.normal(size=dist.shape), max(.4, stroke / 100)) * .025
    return _rgba(mask, rgb, .50 + .66 * light + texture)


def render_material(mask: Image.Image, rgb: tuple, material: str, seed: int) -> Image.Image:
    """Return same-size RGBA. Effects scale with stroke width, not canvas size.

    Furry/beaded boundaries can extend slightly beyond the input silhouette.
    Input canvas margins must therefore accommodate roughly 0.35 stroke widths.
    Empty masks return transparent images. Unknown material names raise ValueError.
    """
    if material not in ID_MATERIALS + OOD_MATERIALS:
        raise ValueError(f"Unknown glyph material: {material!r}")
    if len(rgb) != 3 or any(not 0 <= channel <= 255 for channel in rgb):
        raise ValueError("rgb must contain three channels in [0, 255]")
    mask = mask.convert("L")
    bounds = mask.getbbox()
    if bounds is None:
        return Image.new("RGBA", mask.size)
    if material == "plain":
        return _rgba(mask, rgb)
    tight = mask.crop(bounds)
    # Zero padding makes the estimate correct even when a stroke fills its bbox.
    distance = ndi.distance_transform_edt(np.pad(np.asarray(tight) > 127, 1))
    values = distance[distance > 0]
    if not len(values):
        return _rgba(mask, rgb)
    stroke = max(2.0, 2 * float(np.percentile(values, 80)))
    padding = max(4, math.ceil(.36 * stroke + 3))
    roi_box = (bounds[0] - padding, bounds[1] - padding,
               bounds[2] + padding, bounds[3] + padding)
    roi = mask.crop(roi_box)  # Pillow supplies zeros outside the input canvas.
    if material == "braided":
        rendered = render_rope(roi, rgb, seed)
    else:
        rendered = {"furry": _furry, "beads": _beads, "relief": _relief}[material](roi, rgb, seed, stroke)
    result = Image.new("RGBA", mask.size)
    result.paste(rendered, (roi_box[0], roi_box[1]))
    return result
