"""metro_map -- the same board, question and answer in several visual idioms.

The question type, the graph, the six ringed candidates and the answer never
change here; only the picture's idiom does.  The idioms (STYLES) are modeled on
the conventions of real metro maps, as in BabyVision's "Metro map" images.

Contracts kept in EVERY style, because they are what makes the tiers set valid
rather than pretty:
  * station pitch >= 21px: no frame is narrower than the bottom_strip span,
    and crop framing only enlarges the pitch;
  * the six candidate names are drawn at 29pt, first, by the stated right/left
    convention, and re-stroked where they must sit on track;
  * a non-candidate name is never drawn over track -- it is dropped instead;
  * markers are painted last, so text can never erase a station;
  * the grey candidate ring (105,105,105) is drawn under the strokes in every
    style, so pair localisation stays a 1-in-15 guess everywhere.

This module renders; it does not generate.  Boards come from metro_core, so the
answer is the one metro_core computed.
"""
from __future__ import annotations

import dataclasses
import functools
import math
import random

import networkx as nx
from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

from . import metro_board


VERSION = "metro_map_render"
CAND_RING = (105, 105, 105)
INK = (30, 30, 30)
BIG_PT_FRAC = 0.0655         # 29 pt at size 448 -> cap height 21px = 1.5 patch

PALETTES = {
    # red-orange, yellow, teal, purple, green, blue
    "classic": [(228, 87, 46), (255, 191, 0), (0, 168, 168), (155, 89, 182),
                (46, 160, 87), (52, 120, 220)],
    # official-map hues
    "official_cn": [(226, 35, 45), (0, 92, 170), (0, 150, 80), (120, 50, 150),
                    (250, 140, 30), (0, 160, 200)],
    # muted hues: steel blue, sage, orange, pink, tan, magenta
    "pastel": [(74, 128, 170), (70, 150, 112), (232, 120, 48), (232, 140, 184),
               (196, 170, 110), (176, 60, 150)],
}


@dataclasses.dataclass(frozen=True)
class Style:
    name: str
    role: str = "id"                  # proposed split: id (train) | ood (eval-only)
    corners: str = "sharp"            # sharp | rounded
    track: str = "solid"              # solid | dashed
    station: str = "dot"              # dot | tick | dash | dot_on_line | ring_outline
    interchange: str = "ring"         # ring | capsule | arrow_pill | ring_arrows
    termini: str = "none"             # none | badge | block
    legend: str = "bottom_strip"      # bottom_strip | boxed_bottom_right | top_right_list | top_strip | none
    title: str = "plain"              # plain | boxed | subtitle | none
    line_name: str = "L{n}"           # "L{n}" | "Line {n}"
    labels: str = "horizontal"        # horizontal | diagonal
    underlays: tuple = ()             # river | roads | watermark | compass
    background: tuple = (250, 250, 250)
    palette: str = "classic"
    stroke_px: int = 7
    framing: str = "full"             # full | crop
    note: str = ""


# Map frame (top fraction, bottom fraction) per legend layout.  bottom_strip is
# the reference frame (0.11 / 0.185 -> 317px span, pitch 21.13px).  Layouts
# that empty the bottom strip move the map down and gain pitch (24.2px).
FRAME = {
    "bottom_strip": (0.11, 0.185),
    "boxed_bottom_right": (0.11, 0.185),
    "boxed_bottom_left": (0.11, 0.185),
    "top_right_list": (0.13, 0.05),
    "top_strip": (0.13, 0.05),
    "none": (0.05, 0.05),
}

# Five idioms are used in the in-distribution tiers; four are held out for the
# xxhard (out-of-distribution) tier.
STYLES = [
    Style("classic_current", role="id",
          note="the exact current renderer"),
    Style("rounded_only", role="id", corners="rounded",
          note="rounded bends only"),
    Style("thin_stroke", role="id", corners="rounded", stroke_px=5,
          note="5px stroke"),
    Style("tick_boxed_river", role="id", corners="rounded", station="tick",
          legend="boxed_bottom_right", title="boxed", line_name="Line {n}",
          underlays=("river",), note="tick markers, boxed legend, river band"),
    Style("dashed_track", role="id", corners="rounded", track="dashed",
          station="dot_on_line", legend="none", title="plain", line_name="Line {n}",
          note="dashed strokes, no legend"),
    Style("dash_badges_toplist", role="ood", corners="rounded",
          station="dash", termini="badge", legend="top_right_list", title="plain",
          line_name="Line {n}", stroke_px=9, underlays=("river",),
          note="white dashes on a 9px stroke, numbered termini, legend top-right"),
    Style("pastel_palette", role="ood", corners="rounded",
          legend="boxed_bottom_right", title="boxed", line_name="Line {n}",
          palette="pastel", note="palette swap, gated by Lab dE"),
    Style("grey_ground", role="ood", corners="rounded", station="ring_outline",
          termini="badge", legend="boxed_bottom_right", title="none", line_name="Line {n}",
          palette="official_cn", background=(236, 236, 236), underlays=("compass",),
          note="grey ground, colour-outlined stations, numbered termini, compass"),
    Style("crop_watermark", role="ood", corners="rounded",
          station="tick", termini="block", legend="none", title="none", line_name="Line {n}",
          underlays=("watermark",), framing="crop", stroke_px=8,
          note="cropped window, lines leave the frame, district watermark, no legend"),
]


# ------------------------------------------------------------------ nuisances
# Everything that does not bear on the answer is drawn per image: the title's
# wording, place, colour and size; where the legend sits; how lines are named;
# which colour each line gets; stroke width; the ground tint; the small label
# size.  A style keeps only what defines it (markers, corners, track, palette
# family, underlays, framing).  Fixed decoration is a cue a model can learn
# and a human never needs, so it is randomised, and every draw is recorded.
TITLE_TEXTS = ("METRO MAP", "Metro Map", "SUBWAY MAP", "Subway Map", "City Metro", "Transit Map",
               "METRO", "Rail Map", "Metro Network", "Line Map", "City Subway")
TITLE_COLOURS = ((60, 60, 60), (30, 30, 30), (95, 95, 95), (40, 70, 120), (120, 40, 40),
                 (30, 90, 70), (80, 50, 110))
LEGEND_CHOICES = {
    "bottom_strip": ("bottom_strip", "boxed_bottom_right", "boxed_bottom_left", "top_strip"),
    "boxed_bottom_right": ("boxed_bottom_right", "boxed_bottom_left", "bottom_strip"),
    "top_right_list": ("top_right_list", "top_strip", "boxed_bottom_right", "boxed_bottom_left"),
    "top_strip": ("top_strip", "top_right_list", "bottom_strip"),
    "none": ("none",),
}
LINE_NAME_FORMATS = ("L{n}", "Line {n}", "{n}")
GROUNDS = ((250, 250, 250), (252, 250, 246), (247, 249, 251), (245, 245, 245), (253, 251, 248), (248, 248, 250))


@dataclasses.dataclass(frozen=True)
class Nuisance:
    title_text: str
    title_pos: str            # tl | tc | tr
    title_colour: tuple
    title_pt: int
    legend: str
    line_name: str
    palette_order: tuple      # line index -> colour index in the style's palette
    stroke_px: int
    background: tuple
    small_pt: int


def sample_nuisance(style: Style, n_lines: int, rng: random.Random) -> Nuisance:
    legend = rng.choice(LEGEND_CHOICES[style.legend])
    order = list(range(len(PALETTES[style.palette])))
    rng.shuffle(order)
    if style.background == (250, 250, 250):
        ground = rng.choice(GROUNDS)
    else:                                   # a deliberately tinted ground keeps its family
        g = style.background[0] + rng.randint(-4, 4)
        ground = (g, g, g)
    return Nuisance(
        title_text=rng.choice(TITLE_TEXTS),
        title_pos=rng.choice(("tl", "tl", "tc", "tr")),
        title_colour=rng.choice(TITLE_COLOURS),
        title_pt=rng.randint(18, 24),
        legend=legend,
        line_name=rng.choice(LINE_NAME_FORMATS),
        palette_order=tuple(order[:max(n_lines, 1)] + order[max(n_lines, 1):]),
        stroke_px=max(5, style.stroke_px + rng.choice((-1, 0, 0, 1))),
        background=ground,
        small_pt=rng.choice((10, 11, 11, 12)),
    )


# ------------------------------------------------------------------ helpers
@functools.lru_cache(maxsize=None)
def font(size: int):
    return load_bundled_font(size, "sans-bold")


def _lab(rgb):
    def f(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (f(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def g_(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = g_(x), g_(y), g_(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def de(a, b):
    return math.dist(_lab(a), _lab(b))


def erased_markers(img, centres):
    """A marker with no pure-white pixel left has been painted out."""
    W, H = img.size
    p = img.load()
    n = 0
    for c, (x, y) in centres.items():
        ok = False
        xi, yi = int(round(x)), int(round(y))
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                u, v = xi + dx, yi + dy
                if 0 <= u < W and 0 <= v < H and p[u, v] == (255, 255, 255):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            n += 1
    return n


def palette_gate(palette, background, n_lines):
    cols = palette[:n_lines]
    pair = min(de(a, b) for i, a in enumerate(cols) for b in cols[i + 1:])
    bg = min(de(c, background) for c in cols)
    return round(pair, 1), round(bg, 1)


def _unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


@dataclasses.dataclass
class Board:
    lines: list
    cell_lines: dict
    names: dict
    cands: list
    A: tuple
    B: tuple
    path: list
    answer: int


def repick_decoys_near_route(board: Board, rng: random.Random) -> Board:
    """Four decoys inside the route's bounding box grown by three cells, kept
    >= 4 cells (Chebyshev) apart where the box allows (then 3, then 2), and
    renamed to 3-letter names so the 29pt big-label convention still holds."""
    xs = [c[0] for c in board.path]
    ys = [c[1] for c in board.path]
    x0, x1, y0, y1 = min(xs) - 3, max(xs) + 3, min(ys) - 3, max(ys) + 3
    pool = sorted(c for c in board.cell_lines
                  if x0 <= c[0] <= x1 and y0 <= c[1] <= y1 and c not in (board.A, board.B))
    rng.shuffle(pool)
    cands = [board.A, board.B]
    for sep in (metro_board.CAND_MIN_SEP, 3, 2):
        for c in pool:
            if len(cands) >= metro_board.N_CANDIDATES:
                break
            if c not in cands and all(metro_board.cheb(c, q) >= sep for q in cands):
                cands.append(c)
        if len(cands) >= metro_board.N_CANDIDATES:
            break
    names = dict(board.names)
    used = set(names.values())
    for c in cands[2:]:
        if c not in board.cands:
            names[c] = metro_board.short_name(rng, used)
    return dataclasses.replace(board, cands=cands, names=names)


# ------------------------------------------------------------------ frame
def make_frame(style: Style, size: int, board: Board, rng: random.Random, nuis: "Nuisance | None" = None):
    """Returns (px, sc, extra) where px maps a cell to pixels.  For crop framing
    the window is searched so that the route and all six candidates are inside
    with a margin, no station centre sits on the frame edge, and as many
    stations as possible fall outside (lines visibly leave the picture)."""
    cells = list(board.cell_lines)
    gx0, gx1 = min(c[0] for c in cells), max(c[0] for c in cells)
    gy0, gy1 = min(c[1] for c in cells), max(c[1] for c in cells)
    gw, gh = max(1, gx1 - gx0), max(1, gy1 - gy0)
    top, bot = FRAME[nuis.legend if nuis is not None else style.legend]
    margin = int(size * 0.085)
    map_top, map_bot = int(size * top), size - int(size * bot)
    # size - 2*margin - 55: a 29pt 3-letter name (55px) to the right of the
    # rightmost column must still fit; the 317px span is exactly that.
    # Layouts without a bottom strip are taller but keep the same span, so the
    # pitch is 21.13px in every full-frame style and only the framing moves.
    span = min(size - 2 * margin - 55, map_bot - map_top)
    sc = span / max(gw, gh)
    ox = margin + (size - 2 * margin - gw * sc) / 2
    oy = map_top + ((map_bot - map_top) - gh * sc) / 2
    extra = {"map_box": (margin, map_top, size - margin, map_bot)}
    if style.framing == "crop":
        # The route and all six candidates stay inside with a margin (the
        # question contract).  No marker may be cut by the frame: a station
        # centre must be farther from every edge than its own marker radius.
        # That radius is per marker class, NOT a blanket band -- a 14px band on
        # a 24px lattice always catches one row and no window ever qualifies.
        must = set(board.path) | set(board.cands)
        lw = style.stroke_px
        inter_r = max(7, int(lw * 0.95))
        cand_r = inter_r + max(4, int(lw * 0.55))
        plain_r = {"dot": max(3, int(lw * 0.42)), "tick": (lw + 4) / 2, "dash": (lw - 1) / 2,
                   "dot_on_line": max(3, int(lw * 0.32)), "ring_outline": 4}[style.station]
        rad = {c: (cand_r if c in board.cands else inter_r if len(ls) >= 2 else plain_r) + 2
               for c, ls in board.cell_lines.items()}
        pad = 30
        best = None
        for k in (1.5, 1.4, 1.3, 1.2, 1.1):
            s = sc * k
            lo_x = max(pad - (c[0] - gx0) * s for c in must)
            hi_x = min(size - pad - (c[0] - gx0) * s for c in must)
            lo_y = max(pad - (c[1] - gy0) * s for c in must)
            hi_y = min(size - pad - (c[1] - gy0) * s for c in must)
            if lo_x > hi_x or lo_y > hi_y:
                continue
            xs = [lo_x + i * 2 for i in range(int((hi_x - lo_x) // 2) + 1)]
            ys = [lo_y + i * 2 for i in range(int((hi_y - lo_y) // 2) + 1)]
            for x0 in xs:
                for y0 in ys:
                    outside, bad = 0, False
                    for c in cells:
                        X, Y = x0 + (c[0] - gx0) * s, y0 + (c[1] - gy0) * s
                        r = rad[c]
                        cut_x = -r < X < r or size - r < X < size + r
                        cut_y = -r < Y < r or size - r < Y < size + r
                        if cut_x or cut_y:
                            bad = True
                            break
                        if X < 0 or X > size or Y < 0 or Y > size:
                            outside += 1
                    if bad:
                        continue
                    if best is None or outside > best[0]:
                        best = (outside, s, x0, y0)
            if best is not None:
                break
        if best is None:
            raise RuntimeError("crop framing found no window; route too large for a crop")
        _, sc, ox, oy = best
        extra = {"map_box": (0, 0, size, size), "crop_scale": round(sc / (span / max(gw, gh)), 3),
                 "stations_outside": best[0]}

    def px(c):
        return (ox + (c[0] - gx0) * sc, oy + (c[1] - gy0) * sc)
    return px, sc, extra


# ------------------------------------------------------------------ geometry
def _bez(a, c, b, t):
    return ((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t * t * b[0],
            (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t * t * b[1])


def line_geometry(cells, px, style: Style, sc):
    """Dense pixel polyline for one line, plus where each station sits on it
    and the local track direction there (for ticks and dashes)."""
    pts = [px(c) for c in cells]
    pos, direction = {}, {}
    if len(pts) == 1:
        return pts, {cells[0]: pts[0]}, {cells[0]: (1.0, 0.0)}
    for i, c in enumerate(cells):
        p = pts[i]
        u = _unit(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) if i > 0 else None
        v = _unit(pts[i + 1][0] - p[0], pts[i + 1][1] - p[1]) if i < len(pts) - 1 else None
        if u and v:
            direction[c] = _unit(u[0] + v[0], u[1] + v[1])
        else:
            direction[c] = u or v
    if style.corners != "rounded" or len(pts) < 3:
        return pts, {c: pts[i] for i, c in enumerate(cells)}, direction
    out = [pts[0]]
    pos[cells[0]] = pts[0]
    R = 0.45 * sc
    for i in range(1, len(pts) - 1):
        p, q, r = pts[i - 1], pts[i], pts[i + 1]
        u = _unit(q[0] - p[0], q[1] - p[1])
        v = _unit(r[0] - q[0], r[1] - q[1])
        if abs(u[0] - v[0]) < 1e-6 and abs(u[1] - v[1]) < 1e-6:
            out.append(q)
            pos[cells[i]] = q
            continue
        d = min(R, 0.45 * math.dist(p, q), 0.45 * math.dist(q, r))
        a = (q[0] - u[0] * d, q[1] - u[1] * d)
        b = (q[0] + v[0] * d, q[1] + v[1] * d)
        seg = [_bez(a, q, b, t / 8) for t in range(9)]
        out.extend(seg)
        pos[cells[i]] = _bez(a, q, b, 0.5)      # the station sits on the curve
    out.append(pts[-1])
    pos[cells[-1]] = pts[-1]
    return out, pos, direction


def _draw_poly(d, pts, col, w, track, resets=()):
    if track == "solid":
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=col, width=w)
        for p in pts:
            r = w / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)
        return
    # dashed: each run between two consecutive stations gets its
    # own symmetric pattern that starts AND ends with a dash, so the ~7px of
    # track visible between 1-cell neighbours is always painted.  A
    # free-running pattern put a 5px gap on that run and erased the
    # connection; restarting only at the near station still left the far end
    # in a gap.
    on, off = 9.0, 5.0
    reset = {(round(x, 3), round(y, 3)) for x, y in resets}
    runs, cur = [], [pts[0]]
    for q in pts[1:]:
        cur.append(q)
        if (round(q[0], 3), round(q[1], 3)) in reset:
            runs.append(cur)
            cur = [q]
    if len(cur) > 1:
        runs.append(cur)
    for run in runs:
        L = sum(math.dist(a, b) for a, b in zip(run, run[1:]))
        if L <= 0:
            continue
        n = max(1, int(round((L + off) / (on + off))))
        dash = (L - (n - 1) * off) / n
        pattern = []
        pos = 0.0
        for k in range(n):
            pattern.append((pos, pos + dash))
            pos += dash + off
        # walk the run and paint the [start, end) intervals of the pattern
        walked = 0.0
        for a, b in zip(run, run[1:]):
            seg = math.dist(a, b)
            if seg <= 0:
                continue
            for s0, s1 in pattern:
                lo, hi = max(s0, walked), min(s1, walked + seg)
                if hi > lo:
                    t0, t1 = (lo - walked) / seg, (hi - walked) / seg
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                            (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=col, width=w)
            walked += seg
        for q in run:
            r = w / 2
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col)


def _hits(p, q, box):
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 4))
    for i in range(n + 1):
        t = i / n
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            return True
    return False


def _hits_mask(p, q, box, alpha):
    """Like _hits, but a sample only counts where the sprite actually has ink
    (a rotated name's axis-aligned box is mostly empty corners)."""
    n = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 3))
    w, h = int(box[2] - box[0]), int(box[3] - box[1])
    for i in range(n + 1):
        t = i / n
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if box[0] <= x < box[2] and box[1] <= y < box[3]:
            for dx in (-3, 0, 3):
                for dy in (-3, 0, 3):
                    u, v = int(x - box[0]) + dx, int(y - box[1]) + dy
                    if 0 <= u < w and 0 <= v < h and alpha[u, v] > 40:
                        return True
    return False


def _overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


# ------------------------------------------------------------------ underlays
def draw_underlays(img, d, style: Style, size, map_box, spos, rng):
    x0, y0, x1, y1 = map_box
    if "river" in style.underlays:
        # one smooth band across the map, under everything
        col = (204, 227, 244)
        if rng.random() < 0.5:
            pts = [(x0 - 20, rng.uniform(y0, y1))] + \
                  [(x0 + (x1 - x0) * k / 4 + rng.uniform(-15, 15), rng.uniform(y0, y1)) for k in (1, 2, 3)] + \
                  [(x1 + 20, rng.uniform(y0, y1))]
        else:
            pts = [(rng.uniform(x0, x1), y0 - 20)] + \
                  [(rng.uniform(x0, x1), y0 + (y1 - y0) * k / 4 + rng.uniform(-15, 15)) for k in (1, 2, 3)] + \
                  [(rng.uniform(x0, x1), y1 + 20)]
        # smooth by midpoint subdivision
        for _ in range(2):
            sm = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                sm.append(((3 * a[0] + b[0]) / 4, (3 * a[1] + b[1]) / 4))
                sm.append(((a[0] + 3 * b[0]) / 4, (a[1] + 3 * b[1]) / 4))
            sm.append(pts[-1])
            pts = sm
        w = 26
        for a, b in zip(pts, pts[1:]):
            d.line([a, b], fill=col, width=w)
        for p in pts:
            d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=col)
        mid = pts[len(pts) // 2]
        d.text((mid[0] - 14, mid[1] - 6), "River", font=font(11), fill=(150, 180, 208))
    if "roads" in style.underlays:
        col = (222, 222, 222)
        for _ in range(7):
            p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            ang = rng.choice([0, 45, 90, 135])
            pts = [p]
            for _k in range(rng.randint(2, 4)):
                L = rng.uniform(60, 160)
                q = (pts[-1][0] + L * math.cos(math.radians(ang)), pts[-1][1] + L * math.sin(math.radians(ang)))
                pts.append(q)
                ang += rng.choice([-45, 45])
            for a, b in zip(pts, pts[1:]):
                d.line([a, b], fill=col, width=2)
    if "watermark" in style.underlays:
        # district names in big light grey text in the emptiest cells
        col = (226, 226, 226)
        f = font(24)
        words = [metro_board.gen_name(rng) + " District" for _ in range(3)]
        cells_x, cells_y = 3, 3
        density = {}
        for i in range(cells_x):
            for j in range(cells_y):
                cx0, cy0 = x0 + (x1 - x0) * i / cells_x, y0 + (y1 - y0) * j / cells_y
                cx1, cy1 = x0 + (x1 - x0) * (i + 1) / cells_x, y0 + (y1 - y0) * (j + 1) / cells_y
                density[(i, j)] = sum(1 for p in spos.values() if cx0 <= p[0] < cx1 and cy0 <= p[1] < cy1)
        W = img.size[0]
        for (i, j), w in zip(sorted(density, key=lambda k: density[k])[:2], words):
            cx = x0 + (x1 - x0) * (i + 0.5) / cells_x
            cy = y0 + (y1 - y0) * (j + 0.5) / cells_y
            tb = d.textbbox((0, 0), w, font=f)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            tx = max(4, min(W - tw - 4, cx - tw / 2))
            ty = max(4, min(img.size[1] - th - 4, cy - th / 2))
            d.text((tx - tb[0], ty - tb[1]), w, font=f, fill=col)
    if "compass" in style.underlays:
        cx, cy = x1 - 10, y0 + 6
        d.polygon([(cx, cy - 12), (cx - 5, cy + 6), (cx, cy + 2), (cx + 5, cy + 6)], fill=(90, 90, 90))
        d.text((cx - 4, cy + 7), "N", font=font(9), fill=(90, 90, 90))


# ------------------------------------------------------------------ chrome
def title_box(d, style: Style, size, margin, nuis: Nuisance):
    """Where the title will be drawn, so labels can keep off it; None if no title."""
    if style.title == "none" or nuis.legend in ("top_strip", "top_right_list"):
        return None
    tf = font(nuis.title_pt)
    tb = d.textbbox((0, 0), nuis.title_text, font=tf)
    tw = tb[2] - tb[0]
    x = {"tl": margin, "tc": (size - tw) / 2, "tr": size - margin - tw}[nuis.title_pos]
    y = int(size * 0.014)
    return (x - 8, y - 4, x + tw + 8, y + tb[3] + 6)


def draw_title(d, style: Style, size, margin, nuis: Nuisance):
    if style.title == "none" or nuis.legend in ("top_strip", "top_right_list"):
        return                                  # the top strip belongs to the legend
    tf = font(nuis.title_pt)
    text = nuis.title_text
    tb = d.textbbox((0, 0), text, font=tf)
    tw = tb[2] - tb[0]
    x = {"tl": margin, "tc": (size - tw) / 2, "tr": size - margin - tw}[nuis.title_pos]
    y = int(size * 0.014)
    if style.title == "boxed":
        d.rectangle([x - 6, y - 2, x + tw + 6, y + tb[3] + 4], outline=(120, 120, 120), width=2)
    d.text((x, y), text, font=tf, fill=nuis.title_colour)


def draw_legend(d, style: Style, size, margin, n_lines, palette, nuis: Nuisance):
    layout, line_name = nuis.legend, nuis.line_name
    lf = font(max(9, int(size * 0.024)))
    lw = nuis.stroke_px
    names = [line_name.format(n=i + 1) for i in range(n_lines)]
    transfer = "Transfer"
    if layout == "bottom_strip":
        ly = size - int(size * 0.085)
        sw = int(size * 0.04)
        slot = (size - 2 * margin) / (n_lines + 1.4)
        for li in range(n_lines):
            lx = margin + li * slot
            d.line([(lx, ly + 7), (lx + sw, ly + 7)], fill=palette[li], width=max(5, lw - 2))
            d.text((lx + sw + 4, ly), names[li], font=lf, fill=(50, 50, 50))
        ix = margin + n_lines * slot
        d.ellipse([ix, ly, ix + 15, ly + 15], fill=(255, 255, 255), outline=(70, 70, 70), width=3)
        d.text((ix + 20, ly), transfer, font=lf, fill=(50, 50, 50))
    elif layout in ("boxed_bottom_right", "boxed_bottom_left"):
        entries = names + [transfer]
        cols = 3 if len(entries) > 4 else 2
        rows = -(-len(entries) // cols)
        cw, rh = 66, 16
        bw, bh = cols * cw + 10, rows * rh + 10
        by0 = size - int(size * 0.185) + 8
        if layout == "boxed_bottom_right":
            bx1 = size - margin + 12
            bx0 = bx1 - bw
        else:
            bx0 = margin - 12
            bx1 = bx0 + bw
        by1 = by0 + bh
        d.rectangle([bx0, by0, bx1, by1], fill=(255, 255, 255), outline=(170, 170, 170), width=1)
        for k, e in enumerate(entries):
            cx = bx0 + 6 + (k % cols) * cw
            cy = by0 + 6 + (k // cols) * rh
            if k < n_lines:
                d.line([(cx, cy + 6), (cx + 16, cy + 6)], fill=palette[k], width=5)
            else:
                d.ellipse([cx + 2, cy, cx + 13, cy + 11], fill=(255, 255, 255), outline=(70, 70, 70), width=2)
            d.text((cx + 20, cy - 1), e, font=lf, fill=(50, 50, 50))
    elif layout in ("top_right_list", "top_strip"):
        if layout == "top_right_list":
            cols = 2
            cw, rh = 62, 14
            bx0 = size - margin - cols * cw
            by0 = 6
            for k in range(n_lines):
                cx = bx0 + (k % cols) * cw
                cy = by0 + (k // cols) * rh
                d.ellipse([cx, cy + 1, cx + 11, cy + 12], fill=palette[k])
                d.text((cx + 2, cy + 1), str(k + 1), font=font(8), fill=(255, 255, 255))
                d.text((cx + 15, cy), names[k], font=lf, fill=(50, 50, 50))
        else:
            ly = 12
            slot = (size - 2 * margin) / (n_lines + 1.4)
            for li in range(n_lines):
                lx = margin + li * slot
                d.line([(lx, ly + 7), (lx + 14, ly + 7)], fill=palette[li], width=5)
                d.text((lx + 18, ly), names[li], font=lf, fill=(50, 50, 50))
            ix = margin + n_lines * slot
            d.ellipse([ix, ly, ix + 15, ly + 15], fill=(255, 255, 255), outline=(70, 70, 70), width=3)
            d.text((ix + 20, ly), transfer, font=lf, fill=(50, 50, 50))
            d.line([(margin, ly + 24), (size - margin, ly + 24)], fill=(200, 200, 200), width=1)


# ------------------------------------------------------------------ markers
def draw_station(d, style: Style, p, dirn, col, lw):
    x, y = p
    if style.station == "dot":
        r = max(3, int(lw * 0.42))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(90, 90, 90), width=max(2, int(lw * 0.22)))
    elif style.station == "ring_outline":
        r = 4
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=col, width=2)
    elif style.station == "dot_on_line":
        r = max(3, int(lw * 0.32))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    elif style.station in ("tick", "dash"):
        px_, py_ = -dirn[1], dirn[0]
        half = (lw + 4) / 2 if style.station == "tick" else (lw - 1) / 2
        a = (x - px_ * half, y - py_ * half)
        b = (x + px_ * half, y + py_ * half)
        # widths 5 / 3, not 4 / 2: a 2px diagonal line rasterises with gaps
        # and the tick's white core broke into three blobs, two of which then
        # read as stations 8px apart
        if style.station == "tick":
            d.line([a, b], fill=(120, 120, 120), width=5)
        d.line([a, b], fill=(255, 255, 255), width=3 if style.station == "tick" else 4)


def draw_interchange(d, style: Style, p, dirn, lw):
    x, y = p
    r = max(7, int(lw * 0.95))
    ow = max(2, int(lw * 0.28))
    if style.interchange in ("ring", "ring_arrows"):
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=(70, 70, 70), width=ow)
        if style.interchange == "ring_arrows":
            _arrows(d, x, y, 4)
        return
    # capsule along the local direction of the first line;
    # arrow pill is the same capsule with the transfer glyph inside
    ux, uy = dirn
    half = 1.5 * r
    w = int(1.5 * r)
    a, b = (x - ux * half, y - uy * half), (x + ux * half, y + uy * half)
    for wid, colr in ((w + 2 * ow, (70, 70, 70)), (w, (255, 255, 255))):
        d.line([a, b], fill=colr, width=wid)
        for q in (a, b):
            d.ellipse([q[0] - wid / 2, q[1] - wid / 2, q[0] + wid / 2, q[1] + wid / 2], fill=colr)
    if style.interchange == "arrow_pill":
        _arrows(d, x, y, 5)


def _arrows(d, x, y, L):
    c = (70, 70, 70)
    d.line([(x - L, y - 2), (x + L, y - 2)], fill=c, width=1)
    d.polygon([(x + L, y - 2), (x + L - 3, y - 4), (x + L - 3, y)], fill=c)
    d.line([(x - L, y + 2), (x + L, y + 2)], fill=c, width=1)
    d.polygon([(x - L, y + 2), (x - L + 3, y), (x - L + 3, y + 4)], fill=c)


# ------------------------------------------------------------------ render
def render_styled(board: Board, style: Style, size: int, rng: random.Random):
    """One picture of `board` in `style`.  Returns (image, names_used, diag)."""
    names = dict(board.names)
    n_lines = len(board.lines)
    nuis = sample_nuisance(style, n_lines, random.Random(rng.random()))
    base_palette = PALETTES[style.palette]
    palette = [base_palette[nuis.palette_order[i % len(base_palette)]] for i in range(len(base_palette))]
    img = Image.new("RGB", (size, size), nuis.background)
    d = ImageDraw.Draw(img)
    margin = int(size * 0.085)
    repicked = False
    if style.framing == "crop":
        # metro_core spreads the four decoys over the whole board, so the six
        # candidates span the full grid and a crop has no slack: at best a row
        # or two leaves the frame.  For the crop idiom the decoys are re-picked
        # near the route (same >=4-cell separation rule); A, B and the answer
        # are untouched.
        board = repick_decoys_near_route(board, random.Random(rng.random()))
        names = dict(board.names)
        repicked = True
    px, sc, extra = make_frame(style, size, board, rng, nuis)
    lw = nuis.stroke_px

    # geometry first: dense polylines, station positions, local directions
    polys, spos, sdir = [], {}, {}
    for li, cells in enumerate(board.lines):
        pts, pos, dirn = line_geometry(cells, px, style, sc)
        polys.append(pts)
        for c in cells:
            spos.setdefault(c, pos[c])      # the first (lowest) line owns the position
            sdir.setdefault(c, dirn[c])
    tracks = [(a, b) for pts in polys for a, b in zip(pts, pts[1:])]
    inter_r = max(7, int(lw * 0.95))
    cand_r = inter_r + max(4, int(lw * 0.55))

    draw_underlays(img, d, style, size, extra["map_box"], spos, rng)

    # 0. candidate rings UNDER the strokes.  Drawn on top they are 11px in
    # radius against a 21px station pitch, so a ring swallows the whole visible
    # run of track to a 1-cell neighbour.  Under the strokes the ring is cut
    # where a track crosses it and the track stays continuous.
    for c in board.cands:
        p = spos[c]
        d.ellipse([p[0] - cand_r, p[1] - cand_r, p[0] + cand_r, p[1] + cand_r],
                  outline=CAND_RING, width=max(3, int(lw * 0.45)))
    # 1. strokes
    for li, pts in enumerate(polys):
        _draw_poly(d, pts, palette[li % len(palette)], lw, style.track,
                   resets=[spos[c] for c in board.lines[li]])

    # 1b. termini badges / blocks: placed like labels, never on track.  They
    # are placed AFTER the six candidate names (see the label loop): a
    # badge is decoration and yields, a candidate name is answer-bearing and
    # does not -- placed first, badges pinned names into overlapping them.
    placed, station_boxes = [], []
    for c in board.cell_lines:
        r = (cand_r if c in board.cands else inter_r) + 2
        p = spos[c]
        station_boxes.append((p[0] - r, p[1] - r, p[0] + r, p[1] + r))
    n_badges = 0
    centred_badges = {}            # plain terminus -> (box, label, colour): drawn after markers
    badge_boxes = []               # every badge/block box: a decorative name may not sit on one
    tbox = title_box(d, style, size, margin, nuis)
    if tbox is not None:           # the title is furniture too: names keep off it
        badge_boxes.append(tbox)
        placed.append(tbox)
    big_boxes = []                 # placed candidate names: nothing may sit on them

    def place_termini():
        nonlocal n_badges
        if style.termini != "none":
            bf = font(9)
            kf = font(9)
            for li, cells in enumerate(board.lines):
                col = palette[li % len(palette)]
                for end, nb in ((cells[0], cells[1]), (cells[-1], cells[-2])):
                    e, q = spos[end], spos[nb]
                    if not (0 <= e[0] <= size and 0 <= e[1] <= size):
                        continue                                    # off-frame terminus (crop)
                    is_cand, is_inter = end in board.cands, len(board.cell_lines[end]) >= 2
                    if style.termini == "badge":
                        if is_cand:
                            continue
                        bw = bh = 18
                        label = str(li + 1)
                        if not is_inter and end not in centred_badges:
                            box = (e[0] - 9, e[1] - 9, e[0] + 9, e[1] + 9)
                            centred_badges[end] = (box, label, col)
                            placed.append(box)
                            badge_boxes.append(box)
                            n_badges += 1
                            continue
                    else:
                        label = nuis.line_name.format(n=li + 1)
                        tb = d.textbbox((0, 0), label, font=kf)
                        bw, bh = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
                    ux, uy = _unit(e[0] - q[0], e[1] - q[1])
                    r_here = cand_r if is_cand else inter_r if is_inter else 5
                    for ang in (0, 45, -45, 90, -90):
                        a = math.radians(ang)
                        vx, vy = ux * math.cos(a) - uy * math.sin(a), ux * math.sin(a) + uy * math.cos(a)
                        dist = r_here + 3 + (bw * abs(vx) + bh * abs(vy)) / 2
                        cx, cy = e[0] + vx * dist, e[1] + vy * dist
                        box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
                        if not (0 <= box[0] and box[2] <= size and 0 <= box[1] and box[3] <= size):
                            continue
                        if any(_hits(u, v, box) for u, v in tracks):
                            continue
                        if any(_overlap(box, b) for b in station_boxes + placed + big_boxes):
                            continue
                        if style.termini == "badge":
                            d.ellipse(box, fill=col)
                            tb = d.textbbox((0, 0), label, font=bf)
                            d.text((cx - (tb[2] - tb[0]) / 2 - tb[0], cy - (tb[3] - tb[1]) / 2 - tb[1]),
                                   label, font=bf, fill=(255, 255, 255))
                        else:
                            d.rectangle(box, fill=col)
                            tb = d.textbbox((0, 0), label, font=kf)
                            d.text((cx - (tb[2] - tb[0]) / 2 - tb[0], cy - (tb[3] - tb[1]) / 2 - tb[1]),
                                   label, font=kf, fill=(255, 255, 255))
                        placed.append(box)
                        badge_boxes.append(box)
                        n_badges += 1
                        break


    # 2. labels (before markers)
    small = font(nuis.small_pt)
    big = font(max(11, int(size * BIG_PT_FRAC)))
    # the boxed legend's frame starts 8px below the map; the bottom strip's
    # swatches sit lower, at 91.5% of the height
    label_ymax = (size - int(size * 0.185) + 6 if nuis.legend in ("boxed_bottom_right", "boxed_bottom_left")
                  else size - int(size * 0.10) if nuis.legend == "bottom_strip" else size - 4)
    label_ymin = 2 if nuis.legend not in ("top_right_list", "top_strip") else int(size * 0.10)
    off = inter_r + 4
    # A candidate name clears the whole grey ring: a name at dot+11 on the
    # right, or ending AT the dot on the left, has its edge inside the 11px ring
    # and hides an arc of it.  Here both sides stand off by cand_r + 3.
    big_off = cand_r + 3
    cand_offsets = [(off, -0.5), (off, -1.0), (-1.0, -1.0), (-1.0, 0.0),
                    (off, 0.0), (0.0, off), (0.0, -1.5), (off, off), (-1.0, off)]
    big_offsets = [(big_off, -0.5), (-1.0, -0.5), (big_off, -1.0), (-1.0, -1.0),
                   (big_off, 0.0), (-1.0, 0.0), (big_off, -1.6), (-1.0, -1.6), (big_off, 0.6), (-1.0, 0.6)]
    order = list(board.cands) + sorted((c for c in board.cell_lines if c not in board.cands),
                                       key=lambda c: -len(board.cell_lines[c]))
    repair, dropped, on_track, n_diag = [], 0, 0, 0
    big_collisions = 0             # candidate names that still had to overlap another
    termini_placed = False
    halo = nuis.background
    for cell in order:
        nm = names[cell]
        isbig = cell in board.cands
        if not isbig and not termini_placed:
            place_termini()           # after every candidate name, before decoration
            termini_placed = True
        p = spos[cell]
        if p[0] < -5 or p[0] > size + 5 or p[1] < -5 or p[1] > size + 5:
            continue                                   # off-frame station (crop)
        f = big if isbig else small
        dirn = sdir[cell]
        diagonal = (style.labels == "diagonal" and not isbig
                    and abs(abs(dirn[0]) - abs(dirn[1])) < 0.2 and abs(dirn[0]) > 0.3)
        if diagonal:
            # rotated text along the track: rendered to a sprite, collision-tested
            # on its axis-aligned bbox (conservative), placed on either side
            tb = d.textbbox((0, 0), nm, font=f)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            sprite = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sprite)
            for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                sd.text((2 - tb[0] + ddx, 2 - tb[1] + ddy), nm, font=f, fill=halo + (255,))
            sd.text((2 - tb[0], 2 - tb[1]), nm, font=f, fill=INK + (255,))
            ang = -math.degrees(math.atan2(dirn[1], dirn[0]))
            if ang > 90 or ang < -90:
                ang += 180
            rot = sprite.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
            rw, rh = rot.size
            alpha = rot.split()[-1].load()
            nx_, ny_ = -dirn[1], dirn[0]
            best = None
            for sgn in (1, -1):
                for extra_off in (0, 4):
                    dist = off - 2 + th / 2 + extra_off
                    cx = p[0] + sgn * nx_ * dist
                    cy = p[1] + sgn * ny_ * dist
                    box = (cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2)
                    if box[0] < 2 or box[2] > size - 2 or box[1] < label_ymin or box[3] > label_ymax:
                        continue
                    ntrack = sum(1 for (u, v) in tracks if _hits_mask(u, v, box, alpha))
                    cost = sum(1 for b in placed if _overlap(box, b)) * 3
                    cost += 3 * sum(1 for b in station_boxes if _overlap(box, b)) + 3 * ntrack
                    if best is None or cost < best[0]:
                        best = (cost, box, ntrack)
            if best is None or best[2] > 0 or best[0] >= 3 or any(_overlap(best[1], b) for b in big_boxes):
                dropped += 1
                continue
            box = best[1]
            img.paste(rot, (int(box[0]), int(box[1])), rot)
            placed.append(box)
            n_diag += 1
            continue
        tb = d.textbbox((0, 0), nm, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        best_box, best_cost, best_tr = None, None, 0
        # left slots back off by the same stand-off as the right ones
        left_gap = big_off if isbig else 0
        for ax, ay in (big_offsets if isbig else cand_offsets):
            tx0 = p[0] + (ax if ax >= 0 else ax * tw - left_gap)
            ty0 = p[1] + (ay if ay >= 0 else ay * th)
            tx = max(2, min(size - tw - 2, tx0))
            ty = max(label_ymin, min(label_ymax - th, ty0))
            box = (tx, ty, tx + tw, ty + th)
            wt, wm = (8, 6) if isbig else (3, 3)
            ntrack = sum(1 for (u, v) in tracks if _hits(u, v, box))
            cost = sum(1 for b in placed if _overlap(box, b)) * 3
            # Two 29pt names on top of each other, or a name across a badge,
            # is unreadable, and at cost 3 it was CHEAPER than crossing a
            # 7px run of track (8).  For a candidate name both are now dearer
            # than any track hit; the least-bad slot still wins if every slot
            # collides, and that case is counted in the diagnostics.
            if isbig:
                cost += 50 * sum(1 for b in big_boxes if _overlap(box, b))
                cost += 30 * sum(1 for b in badge_boxes if _overlap(box, b))
            # a big name pushed off its convention slot by the canvas edge ends
            # up across its own dot and the marker then eats a letter ("Bi|n");
            # keep that as a last resort only
            if isbig and (abs(tx - tx0) > 4 or abs(ty - ty0) > 4):
                cost += 20
            cost += wm * sum(1 for b in station_boxes if _overlap(box, b))
            cost += wt * ntrack
            if best_cost is None or cost < best_cost:
                best_cost, best_box, best_tr = cost, box, ntrack
            if cost == 0:
                break
        if not isbig and (best_tr > 0 or any(_overlap(best_box, b) for b in badge_boxes + big_boxes)):
            dropped += 1                      # decoration never covers a name that matters
            continue
        tx, ty = best_box[0], best_box[1]
        placed.append(best_box)
        if isbig:
            if any(_overlap(best_box, b) for b in big_boxes):
                big_collisions += 1
            big_boxes.append(best_box)
        if isbig and best_tr > 0:
            repair.append(best_box)
            on_track += 1
        ax_, ay_ = tx - tb[0], ty - tb[1]
        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]:
            d.text((ax_ + ddx, ay_ + ddy), nm, font=f, fill=halo)
        d.text((ax_, ay_), nm, font=f, fill=INK)
    if not termini_placed:
        place_termini()
    # 2b. re-stroke runs buried under a candidate name
    for box in repair:
        for li, pts in enumerate(polys):
            col = palette[li % len(palette)]
            for u, v in zip(pts, pts[1:]):
                if _hits(u, v, box):
                    d.line([u, v], fill=col, width=max(3, lw // 2))

    # 3. markers LAST
    for cell, ls in board.cell_lines.items():
        p = spos[cell]
        col = palette[min(ls) % len(palette)]
        if cell in centred_badges:
            box, label, bcol = centred_badges[cell]
            d.ellipse(box, fill=bcol)
            tb = d.textbbox((0, 0), label, font=font(9))
            d.text((p[0] - (tb[2] - tb[0]) / 2 - tb[0], p[1] - (tb[3] - tb[1]) / 2 - tb[1]),
                   label, font=font(9), fill=(255, 255, 255))
        elif len(ls) >= 2:
            draw_interchange(d, style, p, sdir[cell], lw)
        else:
            draw_station(d, style, p, sdir[cell], col, lw)

    draw_title(d, style, size, margin, nuis)
    if nuis.legend != "none":
        draw_legend(d, style, size, margin, n_lines, palette, nuis)

    # ---- diagnostics
    visible = {c: p for c, p in spos.items() if 3 <= p[0] <= size - 3 and 3 <= p[1] <= size - 3}
    erased = erased_markers(img, visible)
    pair_de, bg_de = palette_gate(palette, nuis.background, n_lines)
    diag = {
        "style": style.name, "role": style.role,
        "pitch_px": round(sc, 2), "pitch_patches": round(sc / 14, 2),
        "n_stations": len(spos), "n_visible_stations": len(visible),
        "labels_dropped": dropped, "candidate_labels_on_track": on_track,
        "candidate_name_collisions": big_collisions,
        "diagonal_labels": n_diag, "termini_marks": n_badges,
        "erased_markers": erased, "palette_min_pair_de": pair_de, "palette_min_bg_de": bg_de,
        "frame": extra, "decoys_repicked": repicked,
        "nuisance": {f.name: (list(getattr(nuis, f.name)) if isinstance(getattr(nuis, f.name), tuple)
                              else getattr(nuis, f.name)) for f in dataclasses.fields(nuis)},
        "candidates": [list(c) for c in board.cands],
        # where the middle station of the gold route landed: a detail crop
        # around it shows the marker idiom at actual pixels
        "path_mid_px": [round(v, 1) for v in spos[board.path[len(board.path) // 2]]],
        # every station's drawn centre (rounded bends move a station onto the
        # arc apex, up to ~3px off the straight-line grid position)
        "station_px": {f"{c[0]},{c[1]}": [round(p[0], 1), round(p[1], 1)] for c, p in spos.items()},
    }
    if style.framing == "crop":
        # gate: the visible subgraph must give the same answer as the full graph
        G = nx.Graph()
        for cells in board.lines:
            for a, b in zip(cells, cells[1:]):
                if a in visible and b in visible:
                    G.add_edge(a, b)
        try:
            vis_ans = nx.shortest_path_length(G, board.A, board.B) - 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            vis_ans = None
        diag["crop_visible_answer"] = vis_ans
        diag["crop_gate_ok"] = (vis_ans == board.answer)
    return img, names, diag
