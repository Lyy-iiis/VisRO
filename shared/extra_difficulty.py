"""Helpers shared by the xhard/xxhard tiers of several tasks: the tier image size,
final-resolution readability and difference checks, heightmap proposals and the
record helper.  The 3D tasks' xhard/xxhard generators themselves live in each
task's generator/xhard.py.

These checks screen out hidden evidence and tiny raster differences; they are
not a substitute for human or model calibration of the nominal difficulty.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from shared.drawing import isometric as iso

EXTRA = ("xhard", "xxhard")
VERSION = "diverse-scenes"
MIN_SURFACE = 8


def resolve_size(size, difficulty):
    if size is None:
        return 588 if difficulty in EXTRA else 448
    if difficulty in EXTRA and not 448 <= size <= 588:
        raise ValueError("xhard/xxhard require a size between 448 and 588 pixels")
    return size


def base_rung(index, difficulty=None):
    """The L1–L5 rung of a tier's index (medium and hard alternate two rungs by index parity)."""
    if difficulty is None:
        return index % 5 + 1
    rungs = {"easy": (1,), "medium": (2, 3), "hard": (4, 5)}[difficulty]
    return rungs[index % len(rungs)]


def rgb(image, background=(255, 255, 255)):
    out = Image.new("RGB", image.size, background)
    out.paste(image, (0, 0), image if image.mode == "RGBA" else None)
    return np.asarray(out).astype(np.int16)


def difference(a, b, contrast=64, area=48, core=12, background=(255, 255, 255)):
    delta = np.max(np.abs(rgb(a, background) - rgb(b, background)), axis=2)
    mask = delta >= contrast
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    cores = np.bincount(labels[inner], minlength=n + 1)
    eligible = [k for k in range(1, n + 1) if areas[k] >= area and cores[k] >= core]
    best = (
        max(eligible, key=lambda k: areas[k])
        if eligible
        else (int(np.argmax(areas[1:])) + 1 if n else 0)
    )
    return dict(
        readable=bool(eligible),
        contrast_pixels=int(mask.sum()),
        component_pixels=int(areas[best]) if best else 0,
        core_pixels=int(cores[best]) if best else 0,
    )


def difference3d(a, b, **kwargs):
    # The renderer's face shades differ by about 32–54 levels. Evaluate these
    # visible interior patches rather than requiring a black/white transition.
    return difference(a, b, contrast=32, **kwargs)


def surface_counts(H, px, margin=6, info=None, mode="surface"):
    """Exposed final-pixel interiors with production placement and outlines.

    surface: one label per column top or empty base cell.
    cube: one label per cube; returns both raw and robust areas per cube.
    """
    if info is None:
        info = iso.placement(H, px, margin)
    ss = info["ss"]
    origin, hw, hh, ch = (info[k] for k in ("origin", "hw", "hh", "ch"))
    nr, nc = len(H), len(H[0])
    image = Image.new("I", (px * ss, px * ss), 0)
    d = ImageDraw.Draw(image)
    cubes = [(i, j, k) for j in range(nr) for i in range(nc) for k in range(H[j][i])]
    labels = {cube: n + 1 for n, cube in enumerate(cubes)}
    if mode == "surface":
        for j in range(nr):
            for i in range(nc):
                top, _, _ = iso.cube_polys(i, j, -1, origin, hw, hh, ch)
                d.polygon(top, fill=1 + j * nc + i)
        iso.draw_grid_floor(d, nr, nc, origin, hw, hh, ch, color=0, w=ss, pad=1)
    for i, j, k in sorted(cubes, key=lambda c: (c[0] + c[1], c[2])):
        polys = iso.cube_polys(i, j, k, origin, hw, hh, ch)
        for face, poly in enumerate(polys):
            label = (
                labels[i, j, k]
                if mode == "cube"
                else (1 + j * nc + i if face == 0 and k == H[j][i] - 1 else 0)
            )
            d.polygon(poly, fill=label, outline=0)
        for poly in polys:
            d.line(poly + [poly[0]], fill=0, width=ss)
    arr = np.asarray(image)
    raw = np.bincount(
        arr.ravel(), minlength=(len(cubes) if mode == "cube" else nr * nc) + 1
    )
    sub = arr.reshape(px, ss, px, ss).transpose(0, 2, 1, 3)
    lo, hi = sub.min(axis=(2, 3)), sub.max(axis=(2, 3))
    ids = np.where(lo == hi, lo, 0)
    padded = np.pad(ids, 1)
    inner = ids.copy()
    for dy in range(3):
        for dx in range(3):
            inner[padded[dy : dy + px, dx : dx + px] != ids] = 0
    counts = np.bincount(inner.ravel(), minlength=len(raw))
    if mode == "cube":
        return {c: (int(raw[n]), int(counts[n])) for c, n in labels.items()}
    return counts[1:].reshape(nr, nc).tolist()


def readable(H, px, margin=6, info=None):
    counts = surface_counts(H, px, margin, info)
    return min(map(min, counts)) >= MIN_SURFACE


def stepped_map(rng, rows, cols, height, holes=False):
    """Random descending terraces, with empty cells exposed at the front.

    The weak bias only proposes candidates; final-pixel guards remain mandatory.
    """
    H = []
    cutoff = rng.randint(rows + cols - 4, rows + cols - 2) if holes else rows + cols
    for j in range(rows):
        row = []
        for i in range(cols):
            value = max(
                1, min(height, height - (i + j) // 2 + rng.choice((-1, 0, 0, 1)))
            )
            if j:
                value = min(value, H[j - 1][i])
            if i:
                value = min(value, row[i - 1])
            if holes and i + j >= cutoff:
                value = 0
            row.append(value)
        H.append(row)
    return H


def diverse_map(rng, nr, nc, height, holes=False):
    """Propose varied heightmaps without imposing monotonic rows and columns.

 Build from front to back. The diagonal bound prevents a nearer top from
 covering the same projected cell; the adjacent bound prevents both nearer
 sides from covering it. These proposal bounds do not replace pixel checks.
 """
    slope_i = rng.uniform(0.15, 0.65)
    slope_j = rng.uniform(0.15, 0.65)
    peak = rng.uniform(height - 1.6, height + 0.2)
    floor = 0 if holes else 1
    H = [[floor] * nc for _ in range(nr)]
    for j in reversed(range(nr)):
        for i in reversed(range(nc)):
            right = H[j][i + 1] if i + 1 < nc else floor
            down = H[j + 1][i] if j + 1 < nr else floor
            diag = H[j + 1][i + 1] if j + 1 < nr and i + 1 < nc else floor
            lower = max(floor, diag, min(right, down))
            target = max(floor + 0.15, peak - slope_i * i - slope_j * j)
            choices = list(range(lower, height + 1))
            weights = [math.exp(-0.5 * ((v - target) / 0.8) ** 2) for v in choices]
            H[j][i] = rng.choices(choices, weights)[0]
    return H


def record(task, idx, tier, question, answer, meta, options=()):
    return dict(
        id=f"{task}_{idx:06d}",
        task_type="spatial_perception",
        subtype=task,
        level="high",
        difficulty=tier,
        question=question,
        ans_type="choice" if options else "blank",
        options=list(options),
        answer=str(answer),
        cot=meta.pop("_cot", ""),
        meta=dict(generator_version=VERSION, resolution=meta.pop("_size"), **meta),
    )
