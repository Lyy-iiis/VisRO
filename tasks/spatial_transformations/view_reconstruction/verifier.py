"""Independent verifier for view_reconstruction.

Imports nothing from the generator, its renderer or extra_difficulty.py (only
visro.paths.ROOT, to find the DejaVu font asset used for the text on the image):
the projections, the reading conventions, the layout arithmetic, the
isometric rasteriser, the drawing of the view panels / arrows / letters /
legends, the legibility measure and the render-difference measure are
re-implemented here from the conventions the question states:

  heightmap H[j][i]: j = depth row (0 = back), i = column (0 = left)
  top   : H > 0, back row at the top
  front : bar_i = max_j H[j][i], i left to right
  side  : physical right-side camera, bar_k = max_i H[N-1-k][i] (front row left, back right)
  views are max(H_answer) levels tall
  isometric: x = ox + (i - j) * hw, y = oy + (i + j) * hh - k * ch; the +j face
  (FRONT) faces screen lower-left, the +i face (SIDE) lower-right.

Image checks, from weakest to strongest assumption about the drawing:
  * each view panel is DECODED from its pixels (every cell's inset interior must
    be >= 90% fill colour or >= 90% white) and each panel label by template IoU;
  * the arrows' direction is read from their red-pixel profile;
  * the whole canvas is RE-DRAWN from the verified answer (its projections), the
    option heightmaps and the documented layout -- labels, panels, BACK arrows,
    divider, option renders, slot frames, the letters A-D and the FRONT/SIDE
    legends -- and must equal the PNG pixel for pixel, so nothing on the image
    (a letter, a legend, an arrow label, a stray mark in blank space) can differ
    from what the question describes.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from visro.paths import ROOT

CORRECTNESS = (
    "Decodes every view panel from the PNG (each cell from its inset interior, and its TOP/FRONT/SIDE label by "
    "template match) and requires it to equal both meta.view_grids and this verifier's own physical-camera "
    "projection of meta.heightmap; re-draws the whole canvas (view labels, panels of the answer's projections, BACK "
    "arrows, option renders from its own isometric rasteriser, slot frames, letters A-D, FRONT/SIDE legends) and "
    "requires the PNG to match it pixel for pixel; then projects each option onto the decoded views and requires "
    "exactly one match, at the gold letter; the question must be the template for the shown views with no example "
    "answer letter, and meta.rationale, meta.option_breaks, meta.view_partitions and core_key must equal their "
    "recomputed values."
)
UNIQUENESS = (
    "Every other option fails at least one decoded view; none matches under any mirrored or flipped reading of the "
    "views or a shifted reading that ignores empty edge rows/columns; every shown view splits the options 2|2 (two "
    "views are needed); every column top and empty floor cell shows >= 8 interior pixels; and no single-column "
    "misreading that renders near-identically (same camera) can make a distractor fit the views or the answer "
    "break them."
)

LETTERS = "ABCD"
VIEWS = ("top", "front", "side")
TIER_RUNGS = {"easy": ("L1",), "medium": ("L2", "L3"), "hard": ("L4", "L5"), "xhard": ("L6",), "xxhard": ("L7",)}
# rung -> (rows, cols, height cap, number of views, design, edit kind)
RUNG_SPEC = {
    "L1": ((2, 4), (4, 7), 2, 2, "square", "cells"),   # footprint 4x4, 3x5, 2x6 or 2x7 (FOOTPRINTS)
    "L2": ((2, 3), (3, 3), 3, 2, "square", "cells"),
    "L3": ((3, 3), (3, 4), 3, 3, "tetra", "cells"),
    "L4": ((3, 3), (3, 4), 3, 3, "square", "cells"),
    "L5": ((3, 4), (3, 4), 4, 3, "square", "cells"),
    "L6": ((4, 4), (4, 4), 4, 3, "square", "move"),
    "L7": ((4, 4), (5, 5), 5, 3, "square", "move"),
}
TIER_SIZE = {"easy": 588, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}   # easy at 588 px
# rungs whose footprint is one of an explicit list (not every rows x cols combination)
FOOTPRINTS = {"L1": ((4, 4), (3, 5), (2, 6), (2, 7))}
# rungs whose four options must each touch every edge of the floor (easy): the floor is then the
# bounding box of every option, so a structure is never shown shifted on a larger floor
FULL_SPAN = {"L1"}
PALETTES = [(205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160)]
VIEW_FILL = (82, 100, 122)
ARROW_RED = (200, 60, 30)
EDGE = (20, 20, 20)
MIN_SURFACE = 8
SHADES = (1.0, 0.62, 0.80)          # top, FRONT, SIDE face shades of the release renderer
FLOOR_COLOR = (150, 156, 168)
LEGEND_H = 20
# isometric metrics at supersample 1 (half tile width / height, cube height)
HW, HH, CH = 34, 17, 40

# The question (every one of the four possible strings is at most 240 tokens under the
# LLaVA/vicuna and PaliGemma tokenizers).
VIEW_PARTS = {"top": "top view from above with the back row at the top",
                 "front": "front view from the front",
                 "side": "side view from the right with the front row on the left"}

CHOICE_RANGE = "A-D"   # names the whole choice set; it is not an example answer


def expected_question(views):
    shown = [VIEW_PARTS[v] for v in views]
    listed = ", ".join(shown[:-1]) + (", and " if len(shown) == 3 else " and ") + shown[-1]
    grid = "" if "top" in views else " (columns from the front view, rows from the side view)"
    which = "all three views" if len(views) == 3 else "both views"
    return (f"The top row shows the {listed}; a square is filled if any cube is in its line of sight, floor at the "
            f"bottom. Options A-D show cube structures on the full floor grid{grid}, FRONT faces lower left and SIDE "
            f"faces lower right. Which option matches {which}? Answer with its letter.")


def example_letters(question: str) -> list:
    """Letters the question shows as an example answer: every standalone A-D token
    outside the choice-range marker "A-D".  Must be empty: a format example may
    never equal or share a component with a gold."""
    return re.findall(r"(?<![A-Za-z])[A-D](?![A-Za-z])", str(question or "").replace(CHOICE_RANGE, " "))


def expected_rationale(views, letter, breaks):
    others = ", ".join(f"{LETTERS[k]} fails {'+'.join(b)}" for k, b in enumerate(breaks) if b)
    return (f"Projecting every option onto the shown views ({', '.join(views)}): only {letter} "
            f"reproduces all of them; {others}.")


def expected_core_key(H):
    """sha256 of the canonical JSON of {task, answer heightmap} (the release's content key)."""
    value = {"task": "view_reconstruction", "answer_heightmap": [[int(x) for x in r] for r in H]}
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------- projections
def silhouette(H, view):
    """(kind, data): top -> occupancy rows (back first); front/side -> bar heights left to right."""
    A = np.asarray(H, dtype=int)
    if view == "top":
        return tuple(tuple(int(x) for x in r) for r in (A > 0))
    if view == "front":
        return tuple(int(A[:, i].max()) for i in range(A.shape[1]))
    return tuple(int(A[j, :].max()) for j in reversed(range(A.shape[0])))


def as_grid(H, view, frame):
    s = silhouette(H, view)
    if view == "top":
        return [list(r) for r in s]
    return [[1 if level <= b else 0 for b in s] for level in range(frame, 0, -1)]


def grid_to_sil(grid, view):
    """Inverse of as_grid for a well-formed panel (bars must be bottom-filled)."""
    g = np.asarray(grid, dtype=int)
    if view == "top":
        return tuple(tuple(int(x) for x in r) for r in g)
    out = []
    for c in range(g.shape[1]):
        col = g[::-1, c]
        h = int(col.sum())
        if h and not col[:h].all():
            return None  # floating cell: not a silhouette of stacked cubes
        out.append(h)
    return tuple(out)


def trimmed(view, sil):
    """Silhouette with empty edge rows/columns removed (a shifted reading of the panel)."""
    if view == "top":
        A = np.asarray(sil, dtype=int)
        r, c = np.flatnonzero(A.any(axis=1)), np.flatnonzero(A.any(axis=0))
        return tuple(tuple(int(x) for x in row) for row in A[r[0]:r[-1] + 1, c[0]:c[-1] + 1]) if len(r) else ()
    b = list(sil)
    while b and b[0] == 0:
        b.pop(0)
    while b and b[-1] == 0:
        b.pop()
    return tuple(b)


def misreadings(view, sil):
    def lr(s):
        return tuple(tuple(r[::-1]) for r in s) if view == "top" else tuple(s[::-1])
    out = {sil, lr(sil)}
    if view == "top":
        ud = tuple(sil[::-1])
        out |= {ud, lr(ud)}
    return out


# ----------------------------------------------------------------------------- layout
def layout(size):
    zone = int(size * 0.30)
    sw = (size - 12) // 2
    sh = (size - zone - 12) // 2
    slots = []
    for k in range(4):
        r, c = divmod(k, 2)
        x0 = 6 + c * sw + (2 if c else 0)
        y0 = zone + 4 + r * sh + (2 if r else 0)
        slots.append((x0, y0, x0 + sw - 4, y0 + sh - 4))
    px = min(min(x1 - x0, y1 - y0 - LEGEND_H) for x0, y0, x1, y1 in slots) - 6
    boxes = [(x0 + (x1 - x0 - px) // 2, y0 + (y1 - LEGEND_H - y0 - px) // 2, px) for x0, y0, x1, y1 in slots]
    return zone, slots, boxes


# ----------------------------------------------------------------------------- isometric rasteriser
def _pt(i, j, k, o, hw, hh, ch):
    return (o[0] + (i - j) * hw, o[1] + (i + j) * hh - k * ch)


def camera(H, px, margin, ss):
    N, M = len(H), len(H[0])

    def bbox(hw, hh, ch):
        pts = [_pt(i, j, 0, (0, 0), hw, hh, ch) for i, j in ((0, 0), (M + 1, 0), (0, N + 1), (M + 1, N + 1))]
        for j in range(N):
            for i in range(M):
                if H[j][i] > 0:
                    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        pts.append(_pt(i + di, j + dj, H[j][i], (0, 0), hw, hh, ch))
                        pts.append(_pt(i + di, j + dj, 0, (0, 0), hw, hh, ch))
        xs, ys = zip(*pts)
        return min(xs), min(ys), max(xs), max(ys)

    hw, hh, ch = HW * ss, HH * ss, CH * ss
    x0, y0, x1, y1 = bbox(hw, hh, ch)
    canvas = px * ss
    s = min(1.0, (canvas - 2 * margin * ss) / max(x1 - x0, y1 - y0, 1))
    hw, hh, ch = max(2, int(hw * s)), max(1, int(hh * s)), max(2, int(ch * s))
    x0, y0, x1, y1 = bbox(hw, hh, ch)
    return dict(o=((canvas - (x1 - x0)) / 2 - x0, (canvas - (y1 - y0)) / 2 - y0), hw=hw, hh=hh, ch=ch, ss=ss)


def faces(i, j, k, cam):
    """(top, front(+j), side(+i)) polygons of the cube at column (i, j), level k."""
    o, hw, hh, ch = cam["o"], cam["hw"], cam["hh"], cam["ch"]
    P = lambda di, dj, dk: _pt(i + di, j + dj, k + dk, o, hw, hh, ch)  # noqa: E731
    top = [P(0, 0, 1), P(1, 0, 1), P(1, 1, 1), P(0, 1, 1)]
    front = [P(0, 1, 1), P(1, 1, 1), P(1, 1, 0), P(0, 1, 0)]
    side = [P(1, 0, 1), P(1, 1, 1), P(1, 1, 0), P(1, 0, 0)]
    return top, front, side


def painter(H):
    N, M = len(H), len(H[0])
    cols = sorted(((i, j) for j in range(N) for i in range(M)), key=lambda c: c[0] + c[1])
    return [(i, j, k) for i, j in cols for k in range(H[j][i])]


def rasterise(H, px, cam, base, shades, floor_color, floor_pad):
    ss = cam["ss"]
    N, M = len(H), len(H[0])
    img = Image.new("RGB", (px * ss, px * ss), (255, 255, 255))
    d = ImageDraw.Draw(img)
    o, hw, hh, ch = cam["o"], cam["hw"], cam["hh"], cam["ch"]
    for i in range(M + floor_pad + 1):
        d.line([_pt(i, 0, 0, o, hw, hh, ch), _pt(i, N + floor_pad, 0, o, hw, hh, ch)], fill=floor_color, width=ss)
    for j in range(N + floor_pad + 1):
        d.line([_pt(0, j, 0, o, hw, hh, ch), _pt(M + floor_pad, j, 0, o, hw, hh, ch)], fill=floor_color, width=ss)
    tone = [tuple(max(0, min(255, int(c * f))) for c in base) for f in shades]
    for i, j, k in painter(H):
        polys = faces(i, j, k, cam)
        for poly, col in zip(polys, tone):
            d.polygon(poly, fill=col, outline=EDGE)
        for poly in polys:
            d.line(poly + [poly[0]], fill=EDGE, width=ss)
    return img.resize((px, px), Image.LANCZOS) if ss != 1 else img


def surface_pixels(H, px, cam):
    """Interior final-size pixels showing each column top / empty floor cell (N x M)."""
    ss = cam["ss"]
    N, M = len(H), len(H[0])
    lab = Image.new("I", (px * ss, px * ss), 0)
    d = ImageDraw.Draw(lab)
    o, hw, hh, ch = cam["o"], cam["hw"], cam["hh"], cam["ch"]
    for j in range(N):
        for i in range(M):
            d.polygon(faces(i, j, -1, cam)[0], fill=1 + j * M + i)
    for i in range(M + 2):
        d.line([_pt(i, 0, 0, o, hw, hh, ch), _pt(i, N + 1, 0, o, hw, hh, ch)], fill=0, width=ss)
    for j in range(N + 2):
        d.line([_pt(0, j, 0, o, hw, hh, ch), _pt(M + 1, j, 0, o, hw, hh, ch)], fill=0, width=ss)
    for i, j, k in sorted(painter(H), key=lambda c: (c[0] + c[1], c[2])):
        polys = faces(i, j, k, cam)
        for f, poly in enumerate(polys):
            d.polygon(poly, fill=(1 + j * M + i) if (f == 0 and k == H[j][i] - 1) else 0, outline=0)
        for poly in polys:
            d.line(poly + [poly[0]], fill=0, width=ss)
    a = np.asarray(lab).reshape(px, ss, px, ss).transpose(0, 2, 1, 3)
    lo, hi = a.min(axis=(2, 3)), a.max(axis=(2, 3))
    ids = np.where(lo == hi, lo, 0)
    pad = np.pad(ids, 1)
    keep = ids.copy()
    for dy in range(3):
        for dx in range(3):
            keep[pad[dy:dy + px, dx:dx + px] != ids] = 0
    cnt = np.bincount(keep.ravel(), minlength=N * M + 1)
    return cnt[1:].reshape(N, M)


def readably_different(a, b, contrast=32, area=24, core=6):
    """A connected changed region of >= area px (contrast >= 32) with >= core eroded px."""
    delta = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max(axis=2)
    mask = delta >= contrast
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return False
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    cores = np.bincount(lab[inner], minlength=n + 1)
    return bool(((areas[1:] >= area) & (cores[1:] >= core)).any())


def neighbours(O, hcap):
    N, M = len(O), len(O[0])
    for j in range(N):
        for i in range(M):
            for v in range(hcap + 1):
                if v != O[j][i]:
                    X = [list(r) for r in O]
                    X[j][i] = v
                    yield X
    for j in range(N):
        for i in range(M):
            if O[j][i] == 0:
                continue
            for dj in (1, -1):
                tj, ti = j + dj, i + dj
                if 0 <= tj < N and 0 <= ti < M and O[tj][ti] == 0:
                    for nh in (O[j][i] - 1, O[j][i] + 1):
                        if 1 <= nh <= hcap:
                            X = [list(r) for r in O]
                            X[j][i], X[tj][ti] = 0, nh
                            yield X


# ----------------------------------------------------------------------------- pixel decoding
LABELS = {"top": "TOP VIEW", "front": "FRONT VIEW", "side": "SIDE VIEW"}
_TEMPLATES: dict = {}


def _font(size):
    """DejaVu Sans Bold at `size` (every text on the image uses it)."""
    return ImageFont.truetype(str(ROOT / "assets" / "fonts" / "DejaVuSans-Bold.ttf"), size)


def label_masks(vslot):
    """Dark-pixel masks of the three view labels, centred in a view slot (rows 0..21)."""
    if vslot not in _TEMPLATES:
        f = _font(14)
        out = {}
        for v, text in LABELS.items():
            im = Image.new("L", (vslot, 22), 255)
            d = ImageDraw.Draw(im)
            d.text(((vslot - d.textlength(text, font=f)) / 2, 4), text, fill=0, font=f)
            out[v] = np.asarray(im) < 160
        _TEMPLATES[vslot] = out
    return _TEMPLATES[vslot]


def decode_label(arr, x0, vslot):
    """(best view, IoU per view) for the label strip of the slot starting at x0."""
    crop = arr[0:22, x0:x0 + vslot].astype(int)
    mask = crop.max(axis=2) < 160
    ious = {}
    for v, t in label_masks(vslot).items():
        union = int((mask | t).sum())
        ious[v] = int((mask & t).sum()) / union if union else 0.0
    return max(ious, key=ious.get), ious


PANEL_INSET, PANEL_FRAC, PANEL_TOL = 3, 0.9, 6


def decode_panel(arr, box, rows, cols):
    """Grid read from a panel drawn as rows x cols square cells with a 4 px pad.

    Each cell is classified from its whole interior inset by PANEL_INSET px (not
    from one pixel): filled when >= 90% of those pixels are the fill colour,
    empty when >= 90% are white; anything else makes the panel undecodable.
    """
    x0, y0, x1, y1 = box
    cw, chh = (x1 - x0 - 8) / cols, (y1 - y0 - 8) / rows
    if cw != chh or cw != int(cw) or cw < 10:
        return None, f"panel box {box} is not {rows}x{cols} square cells of >= 10 px"
    c = int(cw)
    if y1 > arr.shape[0] or x1 > arr.shape[1]:
        return None, f"panel box {box} leaves the image"
    grid = []
    for r in range(rows):
        row = []
        for q in range(cols):
            cx0, cy0 = x0 + 4 + q * c, y0 + 4 + r * c
            sub = arr[cy0 + PANEL_INSET: cy0 + c - PANEL_INSET + 1, cx0 + PANEL_INSET: cx0 + c - PANEL_INSET + 1]
            sub = sub.astype(int)
            fill = float((np.abs(sub - np.asarray(VIEW_FILL)).max(axis=2) <= PANEL_TOL).mean())
            white = float((np.abs(sub - 255).max(axis=2) <= PANEL_TOL).mean())
            if fill >= PANEL_FRAC:
                row.append(1)
            elif white >= PANEL_FRAC:
                row.append(0)
            else:
                return None, f"panel cell ({r},{q}) is neither filled nor empty ({fill:.0%} fill, {white:.0%} white)"
        grid.append(row)
    return grid, None


# ----------------------------------------------------------------------------- canvas re-draw
# The documented delivery layout of the question image (independent of the
# generator's code): view labels in DejaVu Sans Bold 14 (60,60,66) centred at
# y=4 in slots of (size-16)//n_views px from x=8; each panel's square cells are
# max(10, min((slot-56)//cols, (zone-76)//rows)) px with a 4 px white pad,
# centred in the slot at y=38, filled cells (82,100,122) with a 2 px (40,40,40)
# outline, empty cells white with a 1 px (220,220,224) outline; the BACK arrow
# (200,60,30) runs up 10 px left of the top panel (label above its head) and
# right 9 px under the side panel (label under its head), in Bold 11; a 2 px
# (230,230,234) divider at y=zone; four slots, each with its option render, a
# radius-8 (225,225,230) frame, its letter A-D in Bold 18 (30,30,30) at
# (x0+7, y0+4), and the legend "FRONT <down-left arrow> <down-right arrow>
# SIDE" in Bold 11 (200,60,30) centred at the slot's bottom (y1-LEGEND_H+4).
LABEL_INK = (60, 60, 66)
DIVIDER = (230, 230, 234)
SLOT_FRAME = (225, 225, 230)
LETTER_INK = (30, 30, 30)


def panel_cell(size, nviews, rows, cols):
    zone = int(size * 0.30)
    vslot = (size - 16) // nviews
    return max(10, min((vslot - 56) // cols, (zone - 76) // rows))


def draw_panel(grid, cell):
    R, C = len(grid), len(grid[0])
    img = Image.new("RGB", (C * cell + 8, R * cell + 8), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(R):
        for q in range(C):
            box = [4 + q * cell, 4 + r * cell, 4 + (q + 1) * cell, 4 + (r + 1) * cell]
            if grid[r][q]:
                d.rectangle(box, fill=VIEW_FILL, outline=(40, 40, 40), width=2)
            else:
                d.rectangle(box, fill=(255, 255, 255), outline=(220, 220, 224), width=1)
    return img


def draw_back_arrow(d, view, pbox, font):
    """The BACK arrow of the top view (up, left of the panel) or the side view
    (right, under the panel); returns its [shaft+head, label] boxes."""
    x0, y0, x1, y1 = pbox
    tw = d.textlength("BACK", font=font)
    if view == "top":
        ax = x0 - 10
        d.line([(ax, y1), (ax, y0 + 10)], fill=ARROW_RED, width=2)
        d.polygon([(ax - 4, y0 + 10), (ax + 4, y0 + 10), (ax, y0 + 2)], fill=ARROW_RED)
        tx, ty = max(2, ax + 8 - tw), y0 - 14
        d.text((tx, ty), "BACK", fill=ARROW_RED, font=font)
        boxes = [(ax - 4, y0 + 2, ax + 4, y1), (tx, ty, tx + tw, ty + 12)]
    else:
        ay = y1 + 9
        d.line([(x0, ay), (x1 - 8, ay)], fill=ARROW_RED, width=2)
        d.polygon([(x1 - 8, ay - 4), (x1 - 8, ay + 4), (x1, ay)], fill=ARROW_RED)
        tx = max(x0, x1 - tw)
        d.text((tx, ay + 5), "BACK", fill=ARROW_RED, font=font)
        boxes = [(x0, ay - 4, x1, ay + 4), (tx, ay + 5, tx + tw, ay + 17)]
    return [[round(x) for x in b] for b in boxes]


def draw_legend(d, cx, y, font):
    """'FRONT' + an arrow pointing down-left, a gap, an arrow pointing down-right + 'SIDE', centred at cx."""
    twf = d.textlength("FRONT", font=font)
    tws = d.textlength("SIDE", font=font)
    gap, aw = 18, 10
    total = twf + 4 + aw + gap + aw + 4 + tws
    x = cx - total / 2
    d.text((x, y), "FRONT", fill=ARROW_RED, font=font)
    ax = x + twf + 4
    d.line([(ax + aw, y + 2), (ax + 2, y + 10)], fill=ARROW_RED, width=2)
    d.polygon([(ax, y + 12), (ax + 7, y + 10), (ax + 2, y + 5)], fill=ARROW_RED)
    ax2 = ax + aw + gap
    d.line([(ax2, y + 2), (ax2 + aw - 2, y + 10)], fill=ARROW_RED, width=2)
    d.polygon([(ax2 + aw, y + 12), (ax2 + aw - 7, y + 10), (ax2 + aw - 2, y + 5)], fill=ARROW_RED)
    d.text((ax2 + aw + 4, y), "SIDE", fill=ARROW_RED, font=font)


def panel_boxes_for(size, views, shapes):
    """{view: [x0, y0, x1, y1]} of each view panel; shapes[v] = (rows, cols)."""
    vslot = (size - 16) // len(views)
    out = {}
    for n, v in enumerate(views):
        rows, cols = shapes[v]
        cell = panel_cell(size, len(views), rows, cols)
        w, h = cols * cell + 8, rows * cell + 8
        px = 8 + n * vslot + (vslot - w) // 2
        out[v] = [px, 38, px + w, 38 + h]
    return out


def draw_canvas(size, views, grids, renders):
    """The whole question image from the shown grids and the four option renders.
    Returns (image, panel_boxes, arrow_boxes)."""
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    zone, slots, boxes = layout(size)
    f14, f11, f18 = _font(14), _font(11), _font(18)
    vslot = (size - 16) // len(views)
    pboxes = panel_boxes_for(size, views, {v: (len(grids[v]), len(grids[v][0])) for v in views})
    aboxes = {}
    for n, v in enumerate(views):
        x0 = 8 + n * vslot
        label = LABELS[v]
        d.text((x0 + (vslot - d.textlength(label, font=f14)) / 2, 4), label, fill=LABEL_INK, font=f14)
        px, py, px1, py1 = pboxes[v]
        canvas.paste(draw_panel(grids[v], (px1 - px - 8) // len(grids[v][0])), (px, py))
        if v in ("top", "side"):
            aboxes[v] = draw_back_arrow(d, v, (px, py, px1, py1), f11)
    d.line([(8, zone), (size - 8, zone)], fill=DIVIDER, width=2)
    for k, ren in enumerate(renders):
        x0, y0, x1, y1 = slots[k]
        ox, oy, _ = boxes[k]
        canvas.paste(ren, (ox, oy))
        d.rounded_rectangle([x0, y0, x1, y1], radius=8, outline=SLOT_FRAME, width=2)
        d.text((x0 + 7, y0 + 4), LETTERS[k], fill=LETTER_INK, font=f18)
        draw_legend(d, (x0 + x1) // 2, y1 - LEGEND_H + 4, f11)
    return canvas, pboxes, aboxes


def arrow_points(arr, box, direction):
    """True when the red arrow inside `box` has its head at the stated end."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    sub = arr[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2].astype(int)
    red = (np.abs(sub - np.asarray(ARROW_RED)).max(axis=2) <= 40)
    if red.sum() < 10:
        return False
    if direction == "right":
        prof = red.sum(axis=0)
        cols = np.flatnonzero(prof)
        span = cols[-1] - cols[0] + 1
        head, tail = prof[cols[-1] - span // 6: cols[-1] + 1], prof[cols[0]: cols[0] + span // 6]
    else:  # up
        prof = red.sum(axis=1)
        rows = np.flatnonzero(prof)
        span = rows[-1] - rows[0] + 1
        head, tail = prof[rows[0]: rows[0] + span // 6], prof[rows[-1] - span // 6: rows[-1] + 1]
    return head.max() > tail.max() + 2


# ----------------------------------------------------------------------------- verify
def verify(record: dict, image) -> list[str]:
    try:
        return _verify(record, image)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        return [f"record malformed: {exc!r}"]


def _verify(record: dict, image) -> list[str]:  # noqa: C901 - one linear checklist
    P = []
    try:
        m = record["meta"]
        tier = record["tier"]
        views = list(m["views"])
        opts = [[list(map(int, r)) for r in o] for o in m["option_heightmaps"]]
        H = [list(map(int, r)) for r in m["heightmap"]]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"record lacks a required field: {exc!r}"]

    # ---- record form
    if record.get("answer_type") != "choice" or record.get("choices") != list(LETTERS):
        P.append("answer_type/choices must be choice/A-D")
    ans = record.get("answer")
    if ans not in LETTERS:
        return P + [f"answer {ans!r} is not a letter A-D"]
    if record.get("question") != expected_question(views):
        P.append("question text differs from the template for the shown views")
    shown_letters = example_letters(record.get("question"))
    if shown_letters:
        P.append(f"the question shows example answer letter(s) {shown_letters}"
                 + (" equal to the gold" if ans in shown_letters else ""))
    rung = m.get("rung")
    if rung not in TIER_RUNGS.get(tier, ()):
        return P + [f"rung {rung!r} not allowed in tier {tier!r}"]
    rows, cols, cap, nviews, design, edit = RUNG_SPEC[rung]
    if len(views) != nviews or len(set(views)) != nviews or any(v not in VIEWS for v in views):
        return P + [f"views {views} do not fit {rung}"]
    if views != [v for v in VIEWS if v in views]:
        P.append("views not in canonical top/front/side order")
    if m.get("design") != design or m.get("edit_kind") != edit or m.get("height_cap") != cap:
        P.append("design / edit kind / height cap differ from the tier definition")
    if len(opts) != 4:
        return P + ["need exactly four options"]
    N, M = len(H), len(H[0])
    if not (rows[0] <= N <= rows[1] and cols[0] <= M <= cols[1]) or m.get("grid") != [N, M]:
        P.append(f"footprint {N}x{M} outside {rung} ranges")
    elif rung in FOOTPRINTS and (N, M) not in FOOTPRINTS[rung]:
        P.append(f"footprint {N}x{M} is not one of {rung}'s footprints {FOOTPRINTS[rung]}")
    for k, O in enumerate(opts):
        if len(O) != N or any(len(r) != M for r in O):
            return P + [f"option {LETTERS[k]} has a different footprint"]
        if min(map(min, O)) < 0 or max(map(max, O)) > cap:
            P.append(f"option {LETTERS[k]} heights outside 0..{cap}")
    if len({json.dumps(O) for O in opts}) != 4:
        P.append("options are not pairwise distinct")
    if rung in FULL_SPAN:
        for k, O in enumerate(opts):
            occ = [[h > 0 for h in r] for r in O]
            if not (any(occ[0]) and any(occ[-1]) and any(r[0] for r in occ) and any(r[-1] for r in occ)):
                P.append(f"option {LETTERS[k]} leaves an edge row or column of the floor empty ({rung} options span the floor)")
    if edit == "move" and len({sum(map(sum, O)) for O in opts}) != 1:
        P.append("move tier options do not have equal cube counts")
    ci = LETTERS.index(ans)
    if m.get("correct_index") != ci or opts[ci] != H:
        P.append("gold letter, meta.correct_index and meta.heightmap disagree")
    frame = max(map(max, H))
    if frame < 1:
        return P + ["answer heightmap is empty"]
    if m.get("hmax") != frame:
        P.append("meta.hmax is not the answer's tallest column")
    want_grids = {v: as_grid(H, v, frame) for v in views}
    for v in views:
        if m["view_grids"].get(v) != want_grids[v]:
            P.append(f"meta.view_grids[{v}] is not the projection of meta.heightmap")
    if m.get("side_axis") != "front_row_left_back_row_right" or (
            "side" in views and m.get("conventions", {}).get("side") != {"arrow": "right", "label": "BACK"}):
        P.append("side view convention is not the physical one")
    if record.get("core_key") != expected_core_key(H):
        P.append("core_key is not the digest of the answer heightmap")

    # ---- image: size, view panels decoded from pixels, labels, arrows
    size = TIER_SIZE[tier]
    if image.size != (size, size) or m.get("resolution") != size:
        return P + [f"image is {image.size}, tier {tier} needs {size}x{size}"]
    arr = np.asarray(image.convert("RGB"))
    zone, slots, boxes = layout(size)
    shapes = {v: ((N, M) if v == "top" else (frame, M if v == "front" else N)) for v in views}
    pboxes = panel_boxes_for(size, views, shapes)
    if m.get("panel_boxes") != pboxes:
        P.append("meta.panel_boxes differ from the layout")
    decoded = {}
    vslot = (size - 16) // len(views)
    for n, v in enumerate(views):
        best, ious = decode_label(arr, 8 + n * vslot, vslot)
        if best != v or ious[v] < 0.8:
            P.append(f"view slot {n} label reads as {best!r} (IoU {ious[best]:.2f}), meta says {v!r}")
        box = pboxes[v]
        if not (8 + n * vslot <= box[0] and box[2] <= 8 + (n + 1) * vslot and box[3] < zone):
            P.append(f"{v} panel does not fit its slot")
            continue
        grid, err = decode_panel(arr, box, *shapes[v])
        if err:
            P.append(f"{v} panel: {err}")
            continue
        sil = grid_to_sil(grid, v)
        if sil is None:
            P.append(f"{v} panel has a floating cell")
            continue
        decoded[v] = sil
        if grid != m["view_grids"][v]:
            P.append(f"{v} panel pixels differ from meta.view_grids")
        if sil != silhouette(H, v):
            P.append(f"{v} panel pixels are not the projection of the answer")
    for v, direction in (("top", "up"), ("side", "right")):
        if v in views:
            ab = (m.get("arrow_boxes") or {}).get(v)
            if not ab or not arrow_points(arr, ab[0], direction):
                P.append(f"{v} view arrow does not point {direction}")
            elif any(b[3] >= zone for b in ab):
                P.append(f"{v} arrow leaves the view zone")

    # ---- image: option crops == independent re-render of meta heightmaps
    base = tuple(m.get("palette", ()))
    if base not in [tuple(p) for p in PALETTES] or (edit == "move" and base != PALETTES[0]):
        P.append(f"palette {base} not allowed")
        base = PALETTES[0]
    shades, floor_color = SHADES, FLOOR_COLOR
    if (m.get("floor_pad") != 0 or m.get("option_margin") != 6 or m.get("supersample") != 2
            or tuple(m.get("shades", ())) != SHADES or tuple(m.get("floor_color", ())) != FLOOR_COLOR):
        P.append("option render parameters differ from the release spec")
    if [list(b) for b in boxes] != m.get("option_boxes"):
        P.append("option boxes differ from the layout")
    cams, renders = [], []
    for k, O in enumerate(opts):
        x, y, px = boxes[k]
        cam = camera(O, px, 6, 2)
        ren = rasterise(O, px, cam, base, shades, floor_color, 0)
        cams.append(cam)
        renders.append(ren)
        crop = arr[y:y + px, x:x + px].astype(int)
        diff = np.abs(crop - np.asarray(ren, dtype=int)).max(axis=2)
        if (diff > 0).any():
            P.append(f"option {LETTERS[k]} pixels differ from its heightmap ({int((diff > 0).sum())} px)")

    # ---- image: the whole canvas == a re-draw from the verified answer and options
    canvas, _, want_arrows = draw_canvas(size, views, want_grids, renders)
    if (m.get("arrow_boxes") or {}) != want_arrows:
        P.append("meta.arrow_boxes differ from the layout")
    off = np.abs(arr.astype(int) - np.asarray(canvas, dtype=int)).max(axis=2) > 0
    for x, y, px in boxes:          # option crops are reported above, per option
        off[y:y + px, x:x + px] = False
    if off.any():
        where = []
        if off[:zone + 2].any():
            where.append(f"view zone {int(off[:zone + 2].sum())} px")
        for k, (x0, y0, x1, y1) in enumerate(slots):
            sub = off[y0:y1 + 1, x0:x1 + 1]
            if sub.any():
                letter_px = int(sub[:30, :30].sum())
                legend_px = int(sub[-LEGEND_H - 2:].sum())
                where.append(f"slot {LETTERS[k]} {int(sub.sum())} px (letter {letter_px}, legend {legend_px})")
        P.append("image differs from the re-drawn layout (labels, panels, arrows, letters, legends, blank "
                 "space) in " + ", ".join(where or [f"{int(off.sum())} px"]))

    if any(v not in decoded for v in views):
        return P + ["could not decode every view panel"]

    # ---- correctness: exactly one option matches the decoded views
    sils = [{v: silhouette(O, v) for v in views} for O in opts]
    match = [k for k in range(4) if all(sils[k][v] == decoded[v] for v in views)]
    if match != [ci]:
        P.append(f"options matching the decoded views: {[LETTERS[k] for k in match]}, gold {ans}")
    breaks = [[v for v in views if sils[k][v] != decoded[v]] for k in range(4)]
    if m.get("option_breaks") != breaks:
        P.append("meta.option_breaks differ from the recomputed broken views")
    if m.get("rationale") != expected_rationale(views, ans, breaks):
        P.append("meta.rationale differs from the recomputed answer and broken views")

    # ---- uniqueness / robustness
    for v in views:
        groups = {}
        for k in range(4):
            groups.setdefault(sils[k][v], []).append(k)
        if sorted(map(len, groups.values())) != [2, 2]:
            P.append(f"the {v} view does not split the options 2|2 (a single view would decide)")
    splits = {v: tuple(sorted(tuple(k for k in range(4) if sils[k][v] == s) for s in set(sils[k][v] for k in range(4))))
              for v in views}
    if m.get("view_partitions") != {v: [list(g) for g in splits[v]] for v in views}:
        P.append("meta.view_partitions differ from the recomputed splits")
    distinct = len(set(splits.values()))
    want = {("square", 2): 2, ("square", 3): 2, ("tetra", 3): 3}.get((design, len(views)))
    if distinct != want:
        P.append(f"view partitions form {distinct} distinct splits; {design} with {len(views)} views needs {want}")
    for k in range(4):
        if k != ci and all(sils[k][v] in misreadings(v, decoded[v]) for v in views):
            P.append(f"option {LETTERS[k]} matches the views under a mirrored/flipped reading")
        if k != ci and all(trimmed(v, sils[k][v]) == trimmed(v, decoded[v]) for v in views):
            P.append(f"option {LETTERS[k]} matches the views under a shifted reading (empty edges ignored)")
    for k, O in enumerate(opts):
        px = boxes[k][2]
        surf = surface_pixels(O, px, cams[k])
        if int(surf.min()) < MIN_SURFACE:
            P.append(f"option {LETTERS[k]}: a column top or floor cell shows {int(surf.min())} < {MIN_SURFACE} px")
    for a, b in combinations(range(4), 2):
        if not readably_different(renders[a], renders[b]):
            P.append(f"options {LETTERS[a]} and {LETTERS[b]} render nearly identically")
    # Near-invisible single-column misreadings: none may make a distractor fit
    # the views, and none may change what the answer shows in a shown view.
    for k, O in enumerate(opts):
        px = boxes[k][2]
        for T in neighbours(O, cap + 1):
            fits = all(silhouette(T, v) == decoded[v] for v in views)
            if (k == ci and fits) or (k != ci and not fits):
                continue
            img = rasterise(T, px, cams[k], base, shades, floor_color, 0)
            if not readably_different(renders[k], img):
                role = "the answer" if k == ci else "a distractor"
                P.append(f"option {LETTERS[k]} ({role}) has a near-invisible single-column twin that "
                         f"{'breaks' if k == ci else 'matches'} the views")
                break
    return P
