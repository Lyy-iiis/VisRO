"""Independent verifier for tangram_reconstruction.

Imports nothing from the generator (generator/tangram.py, generator/polytan.py,
shared/puzzles).  Everything is re-derived from the question text, the
decoded PNG and the record's meta:

  1. question  exact match with the tier's template restated here; the allowed turns K
               are read from the template that matched (easy: k=0 only, else 0..3).
  2. layout    every grid line the meta layout declares (target lattice, each piece
               panel) must be present in the pixels at its declared position, and the
               pixels just outside each declared lattice must be white.
  3. decode    each triangle of each lattice is classified from its interior pixels
               (pixels >= 1.5 px from every triangle edge, so grid lines and outlines
               are excluded); all interior pixels must carry one fill colour (gray
               target / blue fixed / orange piece / white) -- this is also the
               legibility gate.  The decoded target, fixed and pieces must equal meta.
  4. labels    axis numbers 0..w / 0..h and piece numbers 1..n are re-rendered with the
               pinned font at their declared spots and must match the image exactly;
               they must sit next to the grid line / panel they name and overlap no
               drawing.
  4b. image    the whole question image is re-drawn from the DECODED puzzle with the
               drawing rules restated here (fills, 2-px outlines, lattices with
               diagonals, labels) and must equal the PNG pixel for pixel, so no extra
               mark, text, decoy or missing outline can pass.
  5. contract  tier parameters (area, free pieces, fixed hint, span, min piece size,
               connectivity) checked on the decoded geometry, and no piece may be made
               only of whole diamonds (those are the only shapes whose turns k and k+2
               can coincide physically with different triangles, or that fit half a
               diagonal off the lattice).
  6. answer    Algorithm X exact cover over every (piece, orientation in K, translation)
               of the DECODED board; exactly one assignment may exist, and the answer
               must be that assignment written as compact JSON [[x,y,k],...] with the
               smallest k per placement.  At easy the cover must also be unique when
               k ranges over 0..3.  The question's format pattern must contain no digit
               and no part of the answer.
  7. key       the content key is recomputed from the decoded geometry (8 board
               symmetries, rotation-canonical pieces) and must equal core_key.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import hashlib
import json
import math
from functools import lru_cache

import numpy as np

CORRECTNESS = ("The target, blue fixed and orange piece triangles are decoded from the PNG (one fill colour per "
               "triangle interior, inside grid lines that must appear exactly where meta's layout says) and must "
               "equal meta; the whole image must equal an independent re-render of the decoded puzzle; the answer "
               "is recomputed as the exact cover of the decoded board and must be the record's answer string, with "
               "labels re-rendered and matched and the question equal to its template (digit-free format pattern).")
UNIQUENESS = ("An independent Algorithm-X enumeration of every placement the question allows (k=0 only at easy, "
              "0-3 clockwise turns elsewhere, no reflections) finds exactly one piece-to-triangles assignment, so "
              "with k fixed to the smallest equivalent turn exactly one answer string is valid; congruent pieces "
              "would give two and are refused, pieces made only of whole diamonds (k/k+2 and half-diagonal "
              "aliases) are refused, and easy is also unique if rotations were allowed.")

TIERS = ("easy", "medium", "hard", "xhard", "xxhard")
# area, free pieces, fixed hint, span limit, minimum piece size -- per the tier table
EXPECT = {
    "easy":   dict(area=18, free=2, hint=True,  span=6,  min_size=4),
    "medium": dict(area=24, free=3, hint=True,  span=8,  min_size=4),
    "hard":   dict(area=28, free=3, hint=True,  span=11, min_size=4),
    "xhard":  dict(area=44, free=6, hint=False, span=11, min_size=4),
    "xxhard": dict(area=64, free=8, hint=False, span=11, min_size=4),
}
SIZE = 448
FONT = (_ROOT / 'assets/fonts/DejaVuSans.ttf')
FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"

GRAY, BLUE, ORANGE, WHITE = (200, 203, 210), (119, 171, 224), (219, 148, 93), (255, 255, 255)
LATTICE_RGB = (175, 181, 188)
LABEL_RGB = (25, 30, 40)
DEEP = 1.5  # px an interior pixel must keep from every triangle edge

# The two question templates (under 256 tokens for LLaVA/vicuna with BOS and for PaliGemma).  Restated here, not imported from the generator.


TEMPLATES = {
    (0,): (
        'Cover the gray target with every numbered orange piece once, no gaps or overlaps, '
        'leaving blue triangles uncovered; count grid squares (4 triangles each), not pixels. '
        'Place each piece exactly as drawn (no turning or flipping). Answer a JSON array without '
        'spaces, one [x,y,0] per piece in number order: put the top-left of its grid-aligned '
        'bounding box (corner triangles included) at grid point (x,y), x right, y down.'
    ),
    (0, 1, 2, 3): (
        'Cover the gray target with every numbered orange piece once, no gaps or overlaps, '
        'leaving blue triangles uncovered; count grid squares (4 triangles each), not pixels. '
        'Pieces may turn but not flip. Answer a JSON array without spaces, one [x,y,k] per piece '
        'in number order: turn it k clockwise quarter-turns (0-3, smallest k if equal), then put '
        'the top-left of its grid-aligned bounding box (corner triangles included) at grid point '
        '(x,y), x right, y down.'
    ),
}
TIER_TURNS = {"easy": (0,), "medium": (0, 1, 2, 3), "hard": (0, 1, 2, 3), "xhard": (0, 1, 2, 3), "xxhard": (0, 1, 2, 3)}


# --------------------------------------------------------------------------- #
# Geometry restated from the question text
# --------------------------------------------------------------------------- #
def cw(cells):
    """One clockwise quarter turn with y pointing down: (x,y,t) -> (-y-1, x, t+1)."""
    return [(-y - 1, x, (t + 1) % 4) for x, y, t in cells]


def at_origin(cells):
    cells = list(cells)
    mx = min(c[0] for c in cells)
    my = min(c[1] for c in cells)
    return frozenset((x - mx, y - my, t) for x, y, t in cells)


def turned(piece, k):
    cells = [tuple(c) for c in piece]
    for _ in range(k):
        cells = cw(cells)
    return at_origin(cells)


def across(c):
    """The triangle sharing c's square side (N of (x,y) <-> S of (x,y-1), E <-> W of (x+1,y))."""
    x, y, t = c
    return [(x, y - 1, 2), (x + 1, y, 3), (x, y + 1, 0), (x - 1, y, 1)][t]


def adjacent(c):
    x, y, t = c
    return [(x, y, (t + 1) % 4), (x, y, (t + 3) % 4), across(c)]


def made_of_diamonds(cells):
    """Every triangle has its across-the-side partner in the set: a union of 45-degree diamonds.

    A half-diagonal shift maps the triangle lattice onto itself only for such unions (any square
    side on the boundary would move off the grid lines), so they are exactly the pieces whose
    turns k and k+2 can show the same physical shape with different triangles, or that fit the
    target half a diagonal off the lattice.
    """
    s = set(cells)
    return bool(s) and all(across(c) in s for c in s)


def is_connected(cells):
    cells = set(cells)
    if not cells:
        return False
    stack, seen = [next(iter(cells))], set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack += [n for n in adjacent(c) if n in cells and n not in seen]
    return seen == cells


def extent(cells):
    return max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1


# --------------------------------------------------------------------------- #
# Exact cover (Algorithm X over placements; independent of the generator's solver)
# --------------------------------------------------------------------------- #
def all_covers(free, pieces, turns, cap=64):
    """Every assignment piece -> (smallest k, covered triangles) that tiles `free`."""
    free = frozenset(free)
    rows = {}
    for i, p in enumerate(pieces):
        orients = {}
        for k in sorted(turns):
            o = turned(p, k)
            orients.setdefault(o, k)  # keep the smallest k for each distinct placed shape
        for o, k in orients.items():
            w, h = extent(o)
            xs = [c[0] for c in free]
            ys = [c[1] for c in free]
            for dx in range(min(xs), max(xs) - w + 2):
                for dy in range(min(ys), max(ys) - h + 2):
                    cells = frozenset((x + dx, y + dy, t) for x, y, t in o)
                    if cells <= free:
                        rows[(i, k, dx, dy)] = [("piece", i)] + [("cell", c) for c in cells]
    cols = {("piece", i): set() for i in range(len(pieces))}
    cols.update({("cell", c): set() for c in free})
    for r, cs in rows.items():
        for c in cs:
            cols[c].add(r)
    out = []

    def select(r):
        removed = []
        for j in rows[r]:
            for i in cols[j]:
                for k in rows[i]:
                    if k != j:
                        cols[k].discard(i)
            removed.append(cols.pop(j))
        return removed

    def deselect(r, removed):
        for j in reversed(rows[r]):
            cols[j] = removed.pop()
            for i in cols[j]:
                for k in rows[i]:
                    if k != j:
                        cols[k].add(i)

    def search(partial):
        if len(out) >= cap:
            return
        if not cols:
            out.append(sorted(partial))
            return
        c = min(cols, key=lambda c: len(cols[c]))
        for r in sorted(cols[c]):
            partial.append(r)
            removed = select(r)
            search(partial)
            deselect(r, removed)
            partial.pop()

    if sum(len(p) for p in pieces) == len(free):
        search([])
    return out


def cover_answer(cover):
    """[(i, k, dx, dy)] -> canonical answer string (pose = bbox top-left of the placed piece)."""
    return json.dumps([[dx, dy, k] for _, k, dx, dy in sorted(cover)], separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Image decoding
# --------------------------------------------------------------------------- #
_FILLS = {GRAY: "gray", BLUE: "blue", ORANGE: "orange", WHITE: "white"}


def decode_panel(arr, x0, y0, cell, w, h):
    """{(x,y,t): colour name or None} for every triangle of a w x h lattice at (x0,y0)."""
    jj, ii = np.mgrid[y0:y0 + h * cell + 1, x0:x0 + w * cell + 1]
    u = (ii - x0) / cell
    v = (jj - y0) / cell
    sx = np.clip(np.floor(u).astype(int), 0, w - 1)
    sy = np.clip(np.floor(v).astype(int), 0, h - 1)
    fu, fv = u - sx, v - sy
    side = np.stack([fv, 1 - fu, 1 - fv, fu])           # distance to N, E, S, W square side (cell units)
    t = np.argmin(side, axis=0)
    d = np.minimum.reduce([side.min(axis=0) * cell,
                           np.abs(fu - fv) * cell / math.sqrt(2),
                           np.abs(fu + fv - 1) * cell / math.sqrt(2)])
    deep = d >= DEEP
    px = arr[y0:y0 + h * cell + 1, x0:x0 + w * cell + 1].astype(np.int64)
    rgb = (px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2]
    tri = ((sy * w + sx) * 4 + t)[deep]
    pairs = np.unique((tri << 24) | rgb[deep])           # distinct (triangle, colour) pairs
    n_col = np.bincount(pairs >> 24, minlength=w * h * 4)
    colour = {int(p >> 24): int(p & 0xFFFFFF) for p in pairs}
    fills = {(r << 16) | (g << 8) | b: name for (r, g, b), name in _FILLS.items()}
    out = {}
    for y in range(h):
        for x in range(w):
            for tt in range(4):
                k = (y * w + x) * 4 + tt
                out[(x, y, tt)] = fills.get(colour[k]) if n_col[k] == 1 else None
    return out


def lattice_problems(arr, x0, y0, cell, w, h, what):
    probs = []
    H, W = arr.shape[:2]
    if x0 - 3 < 0 or y0 - 3 < 0 or x0 + w * cell + 3 >= W or y0 + h * cell + 3 >= H:
        return [f"{what}: declared lattice leaves the canvas"]
    for c in range(w + 1):
        for r in range(h):
            if tuple(arr[y0 + r * cell + cell // 2, x0 + c * cell]) != LATTICE_RGB:
                probs.append(f"{what}: vertical grid line {c} missing at row {r}")
    for r in range(h + 1):
        for c in range(w):
            if tuple(arr[y0 + r * cell, x0 + c * cell + cell // 2]) != LATTICE_RGB:
                probs.append(f"{what}: horizontal grid line {r} missing at column {c}")
    for r in range(h):
        yy = y0 + r * cell + cell // 2
        for xx in (x0 - 3, x0 + w * cell + 3):
            if tuple(arr[yy, xx]) != WHITE:
                probs.append(f"{what}: drawing continues past the declared lattice (x={xx})")
    for c in range(w):
        xx = x0 + c * cell + cell // 2
        for yy in (y0 - 3, y0 + h * cell + 3):
            if tuple(arr[yy, xx]) != WHITE:
                probs.append(f"{what}: drawing continues past the declared lattice (y={yy})")
    return probs[:4]


@lru_cache(None)
def _font(size):
    from PIL import ImageFont
    if hashlib.sha256(FONT.read_bytes()).hexdigest() != FONT_SHA256:
        raise RuntimeError("pinned font changed")
    return ImageFont.truetype(str(FONT), size)


def label_problems(arr, layout, tgt, panels):
    from PIL import Image, ImageDraw
    probs = []
    labels = layout.get("labels") or []
    w, h = tgt["w"], tgt["h"]
    want = [str(c) for c in range(w + 1)] + [str(r) for r in range(h + 1)] + [str(i + 1) for i in range(len(panels))]
    if [lab["text"] for lab in labels] != want:
        return [f"labels {[lab['text'] for lab in labels]} != expected {want}"]
    rects = [(tgt["x"], tgt["y"], tgt["x"] + w * tgt["cell"], tgt["y"] + h * tgt["cell"])]
    rects += [(p["x"], p["y"], p["x"] + p["w"] * p["cell"], p["y"] + p["h"] * p["cell"]) for p in panels]
    boxes = []
    scratch = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(scratch)
    for n, lab in enumerate(labels):
        f = _font(int(lab["size"]))
        box = d.textbbox((lab["x"], lab["y"]), lab["text"], font=f)
        x0, y0, x1, y1 = (int(math.floor(box[0])), int(math.floor(box[1])), int(math.ceil(box[2])), int(math.ceil(box[3])))
        if x0 < 0 or y0 < 0 or x1 > SIZE or y1 > SIZE:
            probs.append(f"label {lab['text']} leaves the canvas")
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if n <= w:                                   # column number c sits above grid line x = c
            gx = tgt["x"] + n * tgt["cell"]
            ok = abs(cx - gx) <= tgt["cell"] / 2 and y1 <= tgt["y"]
        elif n <= w + h + 1:                         # row number r sits left of grid line y = r
            gy = tgt["y"] + (n - w - 1) * tgt["cell"]
            ok = abs(cy - gy) <= tgt["cell"] / 2 and x1 <= tgt["x"]
        else:                                        # piece number i sits above panel i
            p = panels[n - w - h - 2]
            ok = y1 <= p["y"] and p["y"] - y1 <= 12 and p["x"] - 2 <= x0 <= p["x"] + p["w"] * p["cell"]
        if not ok:
            probs.append(f"label {lab['text']} is not next to what it names")
        for r in rects:
            if x0 < r[2] and r[0] < x1 and y0 < r[3] and r[1] < y1:
                probs.append(f"label {lab['text']} overlaps a drawing")
        for b in boxes:
            if x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1:
                probs.append(f"label {lab['text']} overlaps another label")
        boxes.append((x0, y0, x1, y1))
        d.text((lab["x"], lab["y"]), lab["text"], fill=LABEL_RGB, font=f)
        if not np.array_equal(np.asarray(scratch)[y0:y1, x0:x1], arr[y0:y1, x0:x1]):
            probs.append(f"label {lab['text']} pixels differ from the expected glyphs")
    return probs[:4]


# --------------------------------------------------------------------------- #
# Whole-image re-render from the DECODED puzzle (drawing rules restated, no generator code)
# --------------------------------------------------------------------------- #
EDGE_RGB = (40, 40, 44)          # outline of every filled region (only edges used by one triangle)
DIAGONAL_RGB = (206, 209, 214)   # square diagonals of every lattice
SOLUTION_PALETTE = ((219, 148, 93), (170, 210, 160), (230, 180, 210), (240, 210, 130), (198, 150, 210),
                    (150, 205, 205), (225, 160, 150), (190, 190, 120))


def _fill_region(d, cells, cell, ox, oy, fill):
    """Fill each triangle (vertices in half-cell units), then stroke the region's boundary edges 2 px wide."""
    def px(v):
        return (ox + v[0] * cell / 2.0, oy + v[1] * cell / 2.0)

    uses = {}
    for x, y, t in sorted(cells):
        corners = ((2 * x, 2 * y), (2 * x + 2, 2 * y), (2 * x + 2, 2 * y + 2), (2 * x, 2 * y + 2))
        vs = (corners[t], corners[(t + 1) % 4], (2 * x + 1, 2 * y + 1))
        d.polygon([px(v) for v in vs], fill=fill)
        for i in range(3):
            e = frozenset((vs[i], vs[(i + 1) % 3]))
            uses[e] = uses.get(e, 0) + 1
    for e, n in uses.items():
        if n == 1:
            a, b = sorted(e)
            d.line([px(a), px(b)], fill=EDGE_RGB, width=2)


def _lattice(d, x, y, w, h, cell):
    for c in range(w + 1):
        d.line((x + c * cell, y, x + c * cell, y + h * cell), fill=LATTICE_RGB, width=1)
    for r in range(h + 1):
        d.line((x, y + r * cell, x + w * cell, y + r * cell), fill=LATTICE_RGB, width=1)
    for r in range(h):
        for c in range(w):
            d.line((x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell), fill=DIAGONAL_RGB, width=1)
            d.line((x + (c + 1) * cell, y + r * cell, x + c * cell, y + (r + 1) * cell), fill=DIAGONAL_RGB, width=1)


def rerender_question(target, fixed, pieces, lay):
    """The question image the decoded puzzle implies: target (+ blue fixed) on its lattice with
    axis numbers, then each piece panel (number, fill, lattice).  Everything else is white."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (SIZE, SIZE), "white")
    d = ImageDraw.Draw(im)
    tg = lay["target"]
    labels = lay["labels"]
    _fill_region(d, target, tg["cell"], tg["x"], tg["y"], GRAY)
    if fixed:
        _fill_region(d, fixed, tg["cell"], tg["x"], tg["y"], BLUE)
    _lattice(d, tg["x"], tg["y"], tg["w"], tg["h"], tg["cell"])
    n_axis = tg["w"] + tg["h"] + 2
    for lab in labels[:n_axis]:
        d.text((lab["x"], lab["y"]), lab["text"], fill=LABEL_RGB, font=_font(int(lab["size"])))
    for lab, p, pl in zip(labels[n_axis:], pieces, lay["pieces"]):
        d.text((lab["x"], lab["y"]), lab["text"], fill=LABEL_RGB, font=_font(int(lab["size"])))
        _fill_region(d, p, pl["cell"], pl["x"], pl["y"], ORANGE)
        _lattice(d, pl["x"], pl["y"], pl["w"], pl["h"], pl["cell"])
    return np.asarray(im)


def _centroid(c):
    x, y, t = c
    corners = ((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1))
    a, b = corners[t], corners[(t + 1) % 4]
    return ((a[0] + b[0] + x + .5) / 3, (a[1] + b[1] + y + .5) / 3)


# --------------------------------------------------------------------------- #
# Content key, recomputed
# --------------------------------------------------------------------------- #
def content_key(target, fixed, pieces, turns):
    mirror_t = {0: 0, 1: 3, 2: 2, 3: 1}

    def sym(cells, g):
        cells = [tuple(c) for c in cells]
        if g >= 4:
            cells = [(-x - 1, y, mirror_t[t]) for x, y, t in cells]
        for _ in range(g % 4):
            cells = cw(cells)
        return cells

    def shift(cells, ref):
        mx = min(c[0] for c in ref)
        my = min(c[1] for c in ref)
        return sorted([x - mx, y - my, t] for x, y, t in cells)

    variants = []
    for g in range(8):
        tg = sym(target, g)
        ps = []
        for p in pieces:
            q = sym(p, g)
            ps.append(min(shift(turned(q, r), turned(q, r)) for r in range(4)) if len(turns) > 1 else shift(q, q))
        body = dict(target=shift(tg, tg), fixed=shift(sym(fixed, g), tg) if fixed else [], pieces=sorted(ps))
        variants.append(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    puzzle = min(variants)
    blob = json.dumps(dict(puzzle=puzzle, task="tangram_reconstruction"), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
def format_problems(question: str, gold: str) -> list[str]:
    """The question carries no example answer: apart from the stated k range and the square's four triangles,
    no digit may appear (so no x, y or k value, triple or prefix of any gold), and not the gold itself."""
    rest = question.replace("[x,y,0]", "").replace("(0-3, smallest k if equal)", "").replace("(4 triangles each)", "")
    probs = []
    if any(ch.isdigit() for ch in rest):
        probs.append("question contains a number that a real answer could share")
    if gold in question or any(json.dumps(t, separators=(",", ":")) in question for t in json.loads(gold)):
        probs.append("question contains the answer or part of it")
    return probs


# --------------------------------------------------------------------------- #
def verify(record: dict, image) -> list[str]:
    probs: list[str] = []
    tier = record.get("tier")
    if tier not in EXPECT:
        return [f"unknown tier {tier!r}"]
    e = EXPECT[tier]
    m = record.get("meta") or {}
    for key in ("target", "fixed", "pieces", "planted_poses", "layout", "tier_params"):
        if key not in m:
            return [f"meta lacks {key}"]
    if record.get("answer_type") != "text" or record.get("choices") is not None:
        probs.append("answer_type must be text without choices")

    # 1. question -> rule
    q = record.get("question", "")
    turns = next((k for k, tpl in TEMPLATES.items() if q == tpl), None)
    if turns is None:
        return probs + ["question text is not one of the two declared templates"]
    if turns != TIER_TURNS[tier]:
        probs.append(f"question states turns {turns} but tier {tier} uses {TIER_TURNS[tier]}")
    if list(m["tier_params"].get("turns", [])) != list(TIER_TURNS[tier]):
        probs.append("meta tier_params.turns disagree with the tier")

    # 2-3. image -> structure
    if image.size != (SIZE, SIZE):
        return probs + [f"image is {image.size}, expected {(SIZE, SIZE)}"]
    arr = np.asarray(image.convert("RGB"))
    lay = m["layout"]
    tg = lay["target"]
    panels = lay["pieces"]
    if len(panels) != len(m["pieces"]):
        return probs + ["layout piece panels do not match the piece count"]
    probs += lattice_problems(arr, tg["x"], tg["y"], tg["cell"], tg["w"], tg["h"], "target")
    for i, p in enumerate(panels):
        probs += lattice_problems(arr, p["x"], p["y"], p["cell"], p["w"], p["h"], f"piece {i + 1}")
    if probs:
        return probs
    board = decode_panel(arr, tg["x"], tg["y"], tg["cell"], tg["w"], tg["h"])
    bad = [c for c, v in board.items() if v not in ("gray", "blue", "white")]
    target = {c for c, v in board.items() if v in ("gray", "blue")}
    fixed = {c for c, v in board.items() if v == "blue"}
    pieces = []
    for i, p in enumerate(panels):
        dec = decode_panel(arr, p["x"], p["y"], p["cell"], p["w"], p["h"])
        bad += [(i + 1,) + c for c, v in dec.items() if v not in ("orange", "white")]
        pieces.append({c for c, v in dec.items() if v == "orange"})
    if bad:
        return probs + [f"{len(bad)} triangle(s) without one clean fill colour, e.g. {bad[:3]}"]
    if not target or any(not p for p in pieces):
        return probs + ["empty target or piece decoded from the image"]
    if (target != set(map(tuple, m["target"])) or fixed != set(map(tuple, m["fixed"]))
            or [set(map(tuple, p)) for p in m["pieces"]] != pieces):
        probs.append("geometry decoded from the image differs from meta")
    if extent(target) != (tg["w"], tg["h"]) or any(extent(p) != (pl["w"], pl["h"]) for p, pl in zip(pieces, panels)):
        probs.append("declared lattice is not the bounding box of what is drawn")
    probs += label_problems(arr, lay, tg, panels)
    # 4b. whole image: nothing may be drawn that the decoded puzzle does not explain
    try:
        expected = rerender_question(target, fixed, pieces, lay)
        extra = int((expected != arr).any(axis=2).sum())
    except (KeyError, TypeError, ValueError, OSError) as exc:
        extra, probs = 0, probs + [f"layout cannot be re-rendered: {exc!r}"]
    if extra:
        probs.append(f"image has {extra} px the decoded puzzle does not explain (extra marks, text, or missing "
                     "outline/diagonals)")

    # 5. tier contract on the decoded geometry
    if len(target) != e["area"]:
        probs.append(f"target has {len(target)} triangles, tier says {e['area']}")
    if max(extent(target)) > e["span"]:
        probs.append("target exceeds the tier span")
    if not is_connected(target):
        probs.append("target is not connected")
    if bool(fixed) != e["hint"] or (fixed and (not is_connected(fixed) or len(fixed) < e["min_size"])):
        probs.append("fixed hint disagrees with the tier")
    if len(pieces) != e["free"]:
        probs.append(f"{len(pieces)} pieces, tier says {e['free']}")
    for i, p in enumerate(pieces):
        if len(p) < e["min_size"] or not is_connected(p) or at_origin(p) != frozenset(p):
            probs.append(f"piece {i + 1} is undersized, disconnected or not drawn at its bbox origin")
        if made_of_diamonds(p):
            probs.append(f"piece {i + 1} is made only of whole diamonds (turns k and k+2 / half-diagonal "
                         "placements alias)")

    # 6. answer: recompute and prove uniqueness under the stated rule
    free = target - fixed
    covers = all_covers(free, pieces, turns)
    if len(covers) != 1:
        return probs + [f"{len(covers)} valid assemblies under the stated turns {list(turns)}"]
    gold = cover_answer(covers[0])
    if record.get("answer") != gold:
        probs.append(f"answer {record.get('answer')!r} != unique assembly {gold}")
    probs += format_problems(q, gold)
    if json.dumps(m["planted_poses"], separators=(",", ":")) != gold:
        probs.append("meta planted_poses differ from the unique assembly")
    if tier == "easy":
        any_k = all_covers(free, pieces, (0, 1, 2, 3))
        if len(any_k) != 1:
            probs.append(f"easy: {len(any_k)} assemblies if rotations were allowed")

    # 7. content key
    if record.get("core_key") != content_key(target, fixed, pieces, turns):
        probs.append("core_key is not the content key of the decoded puzzle")
    return probs
