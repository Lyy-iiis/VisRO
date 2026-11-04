"""Base generator of find_the_different, in the style of BabyVision's "Find the different" task.

Renders a regular R x C grid of an IDENTICAL real icon, with exactly ONE cell
subtly altered (the odd-one-out). The answer is that cell's 1-indexed (row, col).

Icon diversity
--------------
Icons are drawn from `iconlib` (~16k real icons: Material Design Icons,
game-icons.net, OpenMoji) instead of a handful of hand-coded families, so the
*object* in each problem is rarely repeated. Two visual styles:
  * silhouette  (MDI + game-icons, tinted one color)
  * color       (OpenMoji, full color, multi-part)

Representation level: **fine** (local visual acuity). The bottleneck is purely
perceptual -- you must inspect small/low-contrast/cluttered local detail and
compare it across many near-identical tiles. Nothing in the question text or any
shortcut formula reveals the answer; you have to actually look at the pixels.

How correctness is guaranteed
-----------------------------
The base icon is rendered ONCE into a tile bitmap. Every NORMAL cell is the SAME
bitmap pasted at grid positions, so they are byte-for-byte pixel identical. The
odd cell is a SEPARATELY altered tile pasted at its position. Therefore there is
provably exactly one differing cell, and its pixel bbox is known exactly. Ground
truth (row, col) is the paste index -- never hand-typed. The alteration is
accepted only if it changes the FINAL (downsampled) tile by a perceptible-but-
bounded number of pixels, so the odd cell is always genuinely (and uniquely)
different.

ACUITY KNOB (the core fine-grained difficulty axis)
---------------------------------------------------
`difficulty` controls grid density (easy 4x4 .. hard up to 8x8 -> more clutter,
smaller tiles) and alteration `strength` (easy = a big, high-contrast change;
hard = a tiny, low-contrast change). Smaller strength + denser grid = harder.

Beyond that nominal label, each record carries a MEASURED `meta.difficulty_score`
in [0,1] (higher = harder) = 0.5*search_load + 0.5*subtlety, where search_load
grows with the number of cells (4x4->0 .. 8x8->1) and subtlety grows as the
visible changed-pixel area of the odd cell shrinks. So a dense grid with a tiny
change scores near 1; a sparse grid with an obvious change scores near 0. Use this
to sort/filter by real difficulty independent of the generation knob.

Alterations (iconlib.alter): part removed (a real SVG sub-element for color icons,
or a circle/square/triangle/line-shaped erase), an edge protrusion added,
rotation, mirror, recolor / hue-shift. Pure pose changes (translate/scale) are
excluded (same shape -> not a valid cue). Exactly one is applied to the odd tile.
The exclusion is also enforced on the rendered tiles: a
mirror/rotation of an (almost) symmetric icon that is merely off-centre in its
viewBox comes out as a TRANSLATED copy of the normal tile, so a candidate whose
difference some integer shift of the normal tile explains (the shifted residual
is smaller and no longer passes the readability floor) is refused
(`translation_explains`).  The odd cell must also still show an icon
(`visible_pixels`), so "a total of n icons" in the question stays true.
Question template 1's worked example is a cell outside every grid, so it can
never be the answer (see QUESTION_TEMPLATES).

Each problem is a pure function of (seed, idx, difficulty):
random.Random(f"{seed}:{idx}:{difficulty}").  The tier is part of the key, so the
same index draws a different icon and alteration in every tier.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image

from shared.iconlib import pixdiff
import shared.extra_difficulty as extra

BG = (255, 255, 255)

# dark-ish tints for silhouette grids (whole grid shares one)
SIL_COLORS = [
    (30, 30, 30), (25, 60, 120), (120, 35, 35), (30, 95, 55),
    (70, 45, 105), (140, 80, 20), (20, 90, 100),
]

SS = 4  # supersampling for crisp tiles


GENERATOR_VERSION = "readable-odd"


def derive_rng(seed: int, idx: int, difficulty: str) -> random.Random:
    """Per-problem stream.  The tier is part of the key (see module docstring)."""
    return random.Random(f"{seed}:{idx}:{difficulty}")


def diff_params(diff, rng):
    if diff == "easy":
        rows, cols = rng.randint(4, 5), rng.randint(4, 5)
        strength = rng.uniform(0.7, 1.0)
    elif diff == "medium":
        rows, cols = rng.randint(5, 6), rng.randint(5, 7)
        strength = rng.uniform(0.35, 0.6)
    elif diff == "hard":
        rows, cols = rng.randint(6, 8), rng.randint(6, 8)
        strength = rng.uniform(0.10, 0.30)
    elif diff == "xhard":
        rows, cols, strength = 9, 9, rng.uniform(.3, .5)
    elif diff == "xxhard":
        rows, cols, strength = 10, 10, rng.uniform(.25, .45)
    else:
        raise ValueError(diff)
    return rows, cols, strength


# Every template states the full answer convention: rows counted from 1 at the
# top, columns from 1 at the left, answer written (row,column).  The
# item noun is always "icons" (a naive plural of the icon name gives "balconys",
# "human childs").
# Template 1's worked example is row 12, column 15 (an in-grid cell such as
# (2,3) would be the true answer of ~1/(rows*cols) of its items), outside every grid (the largest is 10x10),
# so it can never equal or share a component with an answer; the coordinates
# are also called (a,b) like templates 2 and 3, not (x,y).

QUESTION_TEMPLATES = [
    "The image shows {n} icons in {r} rows and {c} columns, and one differs from the others. Answer its "
    "position as (row,column), counting from 1 from the top and from the left.",
    "In which row and column is the icon that differs from the others? Answer (a,b): row a from the top, "
    "column b from the left, counting from 1.",
    "Find the icon that is different from the others. Answer (row,column), counting from 1 from the top and "
    "from the left.",
]

# visible_pixels: a pixel shows ink when some channel is >= this far from white.
VISIBLE_DELTA = 32
# translation_explains searches integer shifts up to this fraction of the tile.
SHIFT_FRACTION = 1 / 3


def _on_white(tile) -> np.ndarray:
    return extra.rgb(tile)  # int16 HxWx3, composited exactly as it is pasted


def visible_pixels(tile) -> int:
    """Pixels of the white-composited tile that visibly differ from white."""
    a = _on_white(tile)
    return int((np.max(255 - a, axis=2) >= VISIBLE_DELTA).sum())


def _shift_white(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.full_like(a, 255)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def translation_explains(base_tile, odd_tile, contrast=64) -> bool:
    """True if the odd tile is (up to an unreadable residual) a shifted copy.

    For every integer shift (dx, dy) != 0 within SHIFT_FRACTION of the tile,
    compare the odd tile with the shifted base tile.  If some shift leaves FEWER
    high-contrast pixels than no shift at all, and what is left no longer passes
    the readability floor, the visible difference is a change of position, not
    of the icon -- refused, like the excluded translate/scale alterations.
    """
    a = _on_white(odd_tile)
    b = _on_white(base_tile)
    c0 = int((np.max(np.abs(a - b), axis=2) >= contrast).sum())
    radius = max(3, int(a.shape[0] * SHIFT_FRACTION))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            s = _shift_white(b, dx, dy)
            if int((np.max(np.abs(a - s), axis=2) >= contrast).sum()) >= c0:
                continue
            shifted = Image.fromarray(s.astype(np.uint8), "RGB")
            if not extra.difference(shifted, Image.fromarray(a.astype(np.uint8), "RGB"))["readable"]:
                return True
    return False


def make_problem(lib, rng, size, diff):
    size = extra.resolve_size(size, diff)
    rows, cols, strength = diff_params(diff, rng)

    # style mix: 60% silhouette, 40% color
    style = "silhouette" if rng.random() < 0.6 else "color"
    color = rng.choice(SIL_COLORS) if style == "silhouette" else None
    # layout: fit the whole R x C grid inside the canvas (cell bounded by BOTH
    # axes, so dense non-square grids never spill off-canvas)
    margin = int(size * 0.04)
    avail = size - 2 * margin
    cell = min(avail // cols, avail // rows)
    tile = int(cell * 0.86)
    pad = (cell - tile) // 2
    grid_w, grid_h = cell * cols, cell * rows
    left = (size - grid_w) // 2
    top = (size - grid_h) // 2

    S = tile * SS
    # Readability is evaluated on the FINAL white-composited tile, with the same
    # floor on every tier: tiny anti-aliasing changes are invalid.
    odd_tile = None
    cap = .08 if diff == 'xxhard' else (.12 if diff == 'xhard' else 1.0)
    for _icon_attempt in range(160):
        entry = lib.pick(rng, style=style)
        base_full = lib.render(entry, S, color=color)
        base_tile = base_full.resize((tile, tile), Image.LANCZOS)
        base_visible = visible_pixels(base_tile)
        for attempt in range(24):
            s_eff = min(1.0, strength * (1.0 + .25 * (attempt % 5)))
            cand_full, method, desc = lib.alter(entry, base_full, rng, s_eff, color=color)
            cand_tile = cand_full.resize((tile, tile), Image.LANCZOS)
            metrics = extra.difference(base_tile, cand_tile)
            if (metrics['readable'] and metrics['contrast_pixels'] <= cap * tile * tile
                    and visible_pixels(cand_tile) >= max(48, 0.1 * base_visible)
                    and not translation_explains(base_tile, cand_tile)):
                odd_tile = cand_tile
                break
        if odd_tile is not None:
            break
    if odd_tile is None:
        raise RuntimeError(f'could not build readable {diff} odd-one-out at {size}')

    img = Image.new("RGB", (size, size), BG)
    odd_r = rng.randint(1, rows)
    odd_c = rng.randint(1, cols)
    odd_bbox = None
    for r in range(rows):
        for c in range(cols):
            x = left + c * cell + pad
            y = top + r * cell + pad
            if (r + 1, c + 1) == (odd_r, odd_c):
                img.paste(odd_tile, (x, y), odd_tile)
                odd_bbox = [x, y, x + tile, y + tile]
            else:
                img.paste(base_tile, (x, y), base_tile)

    n = rows * cols
    plural = "icons"
    qt = rng.choice(QUESTION_TEMPLATES)
    question = qt.format(n=n, r=rows, c=cols)
    answer = f"({odd_r},{odd_c})"
    cot = (
        f"The image shows {n} {plural} arranged in {rows} rows and {cols} "
        f"columns. All but one are identical. Scanning the grid, the icon in row "
        f"{odd_r}, column {odd_c} is the odd one out: {desc}. The answer is {answer}."
    )

    # measured difficulty: combine search load (how many cells to scan) with how
    # subtle the one change is (visible changed px as a fraction of a tile).
    # Many cells + a tiny change => hard; few cells + a big change => easy.
    # higher difficulty_score = harder. (pixdiff compares as-displayed on white.)
    odd_diff_px = pixdiff(base_tile, odd_tile)
    odd_diff_frac_tile = odd_diff_px / float(tile * tile)
    n_cells = rows * cols
    search_load = max(0.0, min(1.0, (n_cells - 16) / (64 - 16)))  # 4x4->0 .. 8x8->1
    subtlety = 1.0 - min(1.0, odd_diff_frac_tile / 0.15)          # 1=subtle,0=obvious
    difficulty_score = round(0.5 * search_load + 0.5 * subtlety, 3)

    meta = {
        "generator_version": GENERATOR_VERSION,
        "readability": {"min_contrast": 64, "min_component_pixels": 48, "min_core_pixels": 12, **metrics},
        "icon_id": entry["id"], "icon_set": entry["set"], "style": style,
        "icon_color": color, "difficulty": diff, "rows": rows, "cols": cols,
        "odd_row": odd_r, "odd_col": odd_c, "alteration": method,
        "alteration_desc": desc, "strength": round(strength, 3),
        "odd_bbox": odd_bbox, "cell_px": cell, "tile_px": tile,
        "grid_origin": [left, top], "size": size,
        "visible_px": {"delta": VISIBLE_DELTA, "normal": base_visible,
                       "odd": visible_pixels(odd_tile)},
        "n_cells": n_cells, "odd_diff_px": odd_diff_px,
        "odd_diff_frac_tile": round(odd_diff_frac_tile, 5),
        "search_load": round(search_load, 3), "subtlety": round(subtlety, 3),
        "difficulty_score": difficulty_score,
    }
    return img, question, answer, cot, meta

