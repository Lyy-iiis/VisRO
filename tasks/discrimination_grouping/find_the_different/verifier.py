"""Independent verifier for find_the_different.

Uses only the delivered PNG and the record.  It does not import the generator,
iconlib or extra_difficulty: the layout rule, the readability floor, the
translation test and the question forms are re-implemented here from their
written definitions.

Checks, in order:
  form       answer is the canonical "(r,c)" (ASCII digits, no leading zero,
             equal to meta.odd_row/odd_col); the question is one of the three
             declared templates (each states 1-based rows from the top, columns
             from the left, and the (row,column) format) and its numbers match
             the grid.
  example    no "(p,q)" written in the question (template 1's format
             illustration) can be an answer: it may not equal the gold, and
             neither number may be a row or column index of this grid.
  tier       canvas size, grid shape, area cap and the nominal strength band
             belong to the record's tier; meta.applied_strength is the base
             strength times one of the generator's retry factors (1..2) and
             iconlib.alter's boosts (1, 1.7, 2.6), capped at 1, and the recorded
             alteration parameters (rotation angle, stamp radius, colour shift)
             follow from it.
  layout     margin/cell/tile/origin re-derived from (size, rows, cols) equal
             meta; every pixel outside the R*C tile boxes is pure white, so
             nothing can hide between tiles and the grid is exactly R x C.
  gold       all R*C tile crops are hashed; there must be exactly one crop that
             differs from the others, all R*C-1 others byte-identical, and its
             1-based (row, col) must equal the answer and meta.
  evidence   on the delivered pixels the odd tile differs from a normal tile by
             a component of >=48 px (8-connected) with >=12 px left after a 3x3
             erosion, at per-channel contrast >=64; for xhard/xxhard the number
             of contrast>=64 pixels is <=12% / 8% of tile^2.
  not pose   no integer shift (up to a third of the tile) of the normal tile
             both reduces the contrast>=64 difference and leaves no component
             that passes the floor -- i.e. the change is not just a position
             change inside the cell.
  content    every tile visibly shows an icon (>=48 px at >=32 from white; the
             odd tile also >=10% of a normal tile), so "n icons" is true.
  key        meta.readability / visible_px / tile_sha256 match the
             recomputation; the alteration is "mirror" exactly when the odd
             tile is the normal tile flipped left-right; core_key is sha256 of
             (task, icon_family, alteration_key) and alteration_key follows
             from meta.alteration_params.  (icon_family, the artwork class of
             icon_id, needs the icon library and is checked by rebuilding.)
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re

import numpy as np
from scipy import ndimage

TASK = "find_the_different"

CORRECTNESS = ("Decodes the R x C grid from the PNG with the documented layout rule, requires every pixel "
               "outside the tile boxes to be white, hashes all R*C tile crops and locates the single crop that "
               "differs from the rest; its 1-based (row from top, column from left) must equal the answer and meta.")
UNIQUENESS = ("The other R*C-1 tiles must be byte-identical, so no second cell can be the answer; the one "
              "difference must pass the 64/48/12 readability floor on the delivered pixels, must not be "
              "explained by shifting the normal tile, and the odd cell must still show an icon.")

TIERS = {
    #          size  rows      cols      cap (fraction of tile^2)
    "easy":   (448, {4, 5}, {4, 5}, None),
    "medium": (448, {5, 6}, {5, 6, 7}, None),
    "hard":   (448, {6, 7, 8}, {6, 7, 8}, None),
    "xhard":  (588, {9}, {9}, 0.12),
    "xxhard": (588, {10}, {10}, 0.08),
}

# Nominal (base) strength band of each tier, as drawn by the generator.
STRENGTH = {"easy": (0.70, 1.00), "medium": (0.35, 0.60), "hard": (0.10, 0.30),
            "xhard": (0.30, 0.50), "xxhard": (0.25, 0.45)}
RETRY_FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0)     # generator: 1 + 0.25 * (attempt % 5)
ALTER_BOOSTS = (1.0, 1.7, 2.6)                  # iconlib.alter
BASE_ROUNDING = 0.0005                          # meta.strength is rounded to 3 decimals

CONTRAST, AREA, CORE = 64, 48, 12
VISIBLE = 32
SHIFT_FRACTION = 1 / 3
EIGHT = np.ones((3, 3), bool)

_N = r"([0-9]+)"
QUESTION_FORMS = [
    re.compile(
        rf"The image shows {_N} icons in {_N} rows and {_N} columns, and one differs from the others\. Answer its "
        rf"position as \(row,column\), counting from 1 from the top and from the left\."),
    re.compile(
        r"In which row and column is the icon that differs from the others\? Answer \(a,b\): row a from the top, "
        r"column b from the left, counting from 1\."),
    re.compile(
        r"Find the icon that is different from the others\. Answer \(row,column\), counting from 1 from the top "
        r"and from the left\."),
]
PAIR_IN_TEXT = re.compile(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")
CANONICAL_ANSWER = re.compile(r"\(([1-9][0-9]*),([1-9][0-9]*)\)")


def layout(size: int, rows: int, cols: int):
    """The generator's documented layout: 4% margin, square cells, 86% tiles, centred."""
    margin = int(size * 0.04)
    avail = size - 2 * margin
    cell = min(avail // cols, avail // rows)
    tile = int(cell * 0.86)
    pad = (cell - tile) // 2
    left = (size - cell * cols) // 2
    top = (size - cell * rows) // 2
    return cell, tile, pad, left, top


def floor(a: np.ndarray, b: np.ndarray):
    """(readable, contrast_pixels, best_component, its_core) for two RGB crops."""
    mask = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) >= CONTRAST
    lab, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return False, 0, 0, 0
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    cores = np.bincount(lab[ndimage.binary_erosion(mask, structure=EIGHT)], minlength=n + 1)
    good = [k for k in range(1, n + 1) if areas[k] >= AREA and cores[k] >= CORE]
    best = max(good, key=lambda k: areas[k]) if good else int(np.argmax(areas[1:])) + 1
    return bool(good), int(mask.sum()), int(areas[best]), int(cores[best])


def visible(t: np.ndarray) -> int:
    return int(((255 - t.astype(np.int16)).max(axis=2) >= VISIBLE).sum())


def shift_explains(odd: np.ndarray, normal: np.ndarray):
    """The (dx, dy) of a shift of `normal` that explains the difference, or None."""
    h, w = normal.shape[:2]
    rad = max(3, int(h * SHIFT_FRACTION))
    padded = np.full((h + 2 * rad, w + 2 * rad, 3), 255, np.int16)
    padded[rad:rad + h, rad:rad + w] = normal
    o = odd.astype(np.int16)
    base = int((np.abs(o - normal.astype(np.int16)).max(axis=2) >= CONTRAST).sum())
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx == 0 and dy == 0:
                continue
            win = padded[rad - dy:rad - dy + h, rad - dx:rad - dx + w]   # `normal` moved by (+dx, +dy)
            if int((np.abs(o - win).max(axis=2) >= CONTRAST).sum()) >= base:
                continue
            if not floor(o, win)[0]:
                return dx, dy
    return None


def alteration_key(p: dict) -> list:
    m = p.get("method")
    if m == "remove_element":
        return [m, p["leaf_index"]]
    if m in ("erase", "add"):
        return [m, p["shape"], min(5, int(p["cx"] * 6)), min(5, int(p["cy"] * 6))]
    if m == "rotate":
        return [m, int(math.copysign(1, p["degrees"])), round(abs(p["degrees"]) / 10)]
    if m == "recolor":
        return [m, p["rgb"]]
    if m == "hue_shift":
        return [m, int(math.copysign(1, p["dh"]))]
    return [m]


def strength_problems(m: dict, tier: str) -> list[str]:
    """Base strength in the tier band; applied strength and parameters follow from it."""
    out = []
    base, applied = m.get("strength"), m.get("applied_strength")
    lo, hi = STRENGTH[tier]
    if not isinstance(base, (int, float)) or not lo <= base <= hi:
        return [f"meta.strength {base!r} is outside the {tier} band [{lo}, {hi}]"]
    params = m.get("alteration_params") or {}
    method = params.get("method")
    if method == "erase_fallback":
        return [] if applied is None else ["erase_fallback carries an applied strength"]
    if not isinstance(applied, (int, float)):
        return ["meta.applied_strength is missing"]
    tol = BASE_ROUNDING * max(RETRY_FACTORS) * max(ALTER_BOOSTS) + 1e-9
    if not any(abs(applied - min(1.0, min(1.0, base * f) * b)) <= tol
               for f in RETRY_FACTORS for b in ALTER_BOOSTS):
        out.append(f"meta.applied_strength {applied} is not base {base} x retry factor x alter boost")
    s = float(applied)
    S = int(m.get("tile_px", 0)) * int((m.get("render") or {}).get("supersample", 0))
    try:
        if method == "rotate":
            ok = abs(abs(params["degrees"]) - (12 + s * 28)) <= 0.002
        elif method == "erase":
            ok = S > 0 and params["r"] == round(max(2, int((0.05 + s * 0.10) * S)) / S, 4)
        elif method == "add":
            ok = S > 0 and params["r"] == round(max(2, int((0.05 + s * 0.09) * S)) / S, 4)
        elif method == "recolor":
            shift = int(20 + s * 90)
            col = m.get("icon_color") or (30, 30, 30)
            ok = all(v in (min(255, max(0, c + shift)), min(255, max(0, c - shift)))
                     for v, c in zip(params["rgb"], col))
        elif method == "hue_shift":
            ok = abs(params["dh"]) == int(10 + s * 60)
        else:                                            # remove_element, mirror: strength unused
            ok = True
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        out.append(f"{method} parameters do not follow from applied strength {s}")
    return out


def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def verify(record: dict, image) -> list[str]:
    problems: list[str] = []
    m = record.get("meta") or {}
    tier = record.get("tier")
    if tier not in TIERS:
        return [f"unknown tier {tier!r}"]
    size, rows_ok, cols_ok, cap = TIERS[tier]

    # ---- form
    answer = str(record.get("answer", ""))
    mt = CANONICAL_ANSWER.fullmatch(answer)
    if not mt:
        return [f"answer {record.get('answer')!r} is not the canonical '(row,col)' (ASCII digits, no leading zero)"]
    gold = (int(mt.group(1)), int(mt.group(2)))
    if answer != f"({m.get('odd_row')},{m.get('odd_col')})":
        problems.append(f"answer {answer!r} is not the canonical string of meta odd cell")
    if record.get("answer_type") != "text" or record.get("choices"):
        problems.append("answer_type must be 'text' with no choices")
    try:
        rows, cols = int(m["rows"]), int(m["cols"])
    except (KeyError, TypeError, ValueError):
        return ["meta lacks rows/cols"]
    q = str(record.get("question", ""))
    hits = [f.fullmatch(q) for f in QUESTION_FORMS]
    hit = next((h for h in hits if h), None)
    if hit is None:
        problems.append("question does not match a declared template")
    elif hits[0]:
        n, r, c = map(int, hit.groups())
        if (n, r, c) != (rows * cols, rows, cols):
            problems.append(f"question numbers {(n, r, c)} do not match the {rows}x{cols} grid")

    # ---- the question's format example can never be an answer
    for px in PAIR_IN_TEXT.finditer(q):
        ex = (int(px.group(1)), int(px.group(2)))
        if ex == gold:
            problems.append("answer equals the question's format example (copyable)")
        elif any(1 <= v <= max(rows, cols) for v in ex):
            problems.append(f"format example {ex} shares a component with a possible answer of the {rows}x{cols} grid")

    # ---- tier
    if m.get("difficulty") != tier or record.get("difficulty", tier) != tier:
        problems.append(f"meta.difficulty {m.get('difficulty')!r} / difficulty is not the tier {tier!r}")
    problems += strength_problems(m, tier)
    if image.size != (size, size) or m.get("size") != size:
        problems.append(f"image {image.size} / meta.size {m.get('size')} is not the {tier} canvas {size}")
        return problems
    if rows not in rows_ok or cols not in cols_ok:
        problems.append(f"{rows}x{cols} grid is outside the {tier} definition")
    if not (1 <= gold[0] <= rows and 1 <= gold[1] <= cols):
        return problems + [f"answer {gold} outside the {rows}x{cols} grid"]
    if (m.get("odd_row"), m.get("odd_col")) != gold:
        problems.append(f"meta odd cell {(m.get('odd_row'), m.get('odd_col'))} != answer {gold}")

    # ---- layout
    cell, tile, pad, left, top = layout(size, rows, cols)
    if (m.get("cell_px"), m.get("tile_px"), list(m.get("grid_origin") or [])) != (cell, tile, [left, top]):
        problems.append("meta geometry differs from the layout rule")
    img = np.asarray(image.convert("RGB"))
    covered = np.zeros(img.shape[:2], bool)
    crops = {}
    for i in range(rows):
        for j in range(cols):
            x, y = left + j * cell + pad, top + i * cell + pad
            crops[i + 1, j + 1] = img[y:y + tile, x:x + tile]
            covered[y:y + tile, x:x + tile] = True
    if not (img[~covered] == 255).all():
        problems.append(f"{int((img[~covered] != 255).any(axis=-1).sum())} non-white pixels outside the tile boxes")

    # ---- gold from pixels
    groups = collections.defaultdict(list)
    for k, crop in crops.items():
        groups[hashlib.sha256(crop.tobytes()).hexdigest()].append(k)
    sizes = sorted(len(v) for v in groups.values())
    if sizes != [1, rows * cols - 1]:
        return problems + [f"tile groups {sizes[:6]}: not exactly one tile differing from {rows * cols - 1} identical tiles"]
    odd = next(v[0] for v in groups.values() if len(v) == 1)
    if odd != gold:
        return problems + [f"the differing tile is at {odd}, not at the answer {gold}"]
    x0, y0 = left + (gold[1] - 1) * cell + pad, top + (gold[0] - 1) * cell + pad
    if list(m.get("odd_bbox") or []) != [x0, y0, x0 + tile, y0 + tile]:
        problems.append("meta odd_bbox differs from the decoded odd tile")
    sha = {("odd" if len(v) == 1 else "normal"): h for h, v in groups.items()}
    ts = m.get("tile_sha256") or {}
    if (ts.get("odd"), ts.get("normal")) != (sha["odd"], sha["normal"]):
        problems.append("meta.tile_sha256 does not match the delivered tiles")

    # ---- evidence on delivered pixels
    a = crops[gold]
    b = next(v for k, v in crops.items() if k != gold)
    ok, cpx, comp, core = floor(a, b)
    if not ok:
        problems.append(f"difference below the readability floor (component {comp} px, core {core} px)")
    if cap is not None and cpx > cap * tile * tile:
        problems.append(f"{cpx} contrast pixels exceed the {tier} cap {cap} * {tile}^2")
    rd = m.get("readability") or {}
    if (rd.get("contrast_pixels"), rd.get("component_pixels"), rd.get("core_pixels"), rd.get("readable")) != (cpx, comp, core, ok):
        problems.append("meta.readability does not match the pixels")

    # ---- not a pose change
    sh = shift_explains(a, b)
    if sh is not None:
        problems.append(f"difference is explained by shifting the normal tile by {sh}")

    # ---- every cell shows an icon
    vo, vn = visible(a), visible(b)
    if vn < AREA or vo < max(AREA, 0.1 * vn):
        problems.append(f"a cell shows no icon (visible px odd {vo}, normal {vn})")
    vp = m.get("visible_px") or {}
    if (vp.get("delta"), vp.get("odd"), vp.get("normal")) != (VISIBLE, vo, vn):
        problems.append("meta.visible_px does not match the pixels")

    # ---- dedup key
    params = m.get("alteration_params") or {}
    try:
        akey = alteration_key(params)
    except (KeyError, TypeError, ValueError):
        akey = None
    if akey is None or akey != m.get("alteration_key"):
        problems.append("meta.alteration_key does not follow from meta.alteration_params")
    expected_method = "erase" if params.get("method") == "erase_fallback" else params.get("method")
    if expected_method != m.get("alteration"):
        problems.append("meta.alteration disagrees with alteration_params")
    # A mirrored odd tile is exactly the normal tile flipped left-right (LANCZOS from a 4x render
    # is symmetric); no other alteration produces an exact flip.  Ties the claim to the pixels.
    flipped = bool(np.array_equal(a, b[:, ::-1]))
    if (params.get("method") == "mirror") != flipped:
        problems.append("the odd tile is the flipped normal tile but the alteration is not 'mirror'" if flipped
                        else "alteration 'mirror' recorded but the odd tile is not the flipped normal tile")
    icon, family = m.get("icon_id"), m.get("icon_family")
    if not (isinstance(icon, str) and isinstance(family, str) and family.split("/")[0] == icon.split("/")[0]
            and family <= icon):
        problems.append("meta.icon_family is not an artwork-class name for icon_id (same set, smallest id)")
    if record.get("core_key") != _digest([TASK, family, m.get("alteration_key")]):
        problems.append("core_key is not sha256(task, icon_family, alteration_key)")
    if (m.get("render") or {}).get("path") != "cairosvg-live":
        problems.append("item was not rendered on the pinned cairosvg-live path")
    return problems
