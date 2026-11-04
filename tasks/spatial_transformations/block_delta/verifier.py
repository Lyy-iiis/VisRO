"""Independent verifier for block_delta.

Nothing here imports the generator (generator, extra_difficulty,
shared/drawing/isometric.py).  The isometric projection and the painter are re-implemented from the
documented rendering contract:

    screen_x = ox + (i - j) * hw,   screen_y = oy + (i + j) * hh - k * ch
    (ss = 2 canvas, painter order: diagonal i+j ascending, then j, then k;
    faces top / left(+j) / right(+i) shaded 1.0 / 0.74 / 0.88 of the palette,
    black (20,20,20) outlines of width 2, floor grid (200,205,212) width 2 over
    (N+1) x (M+1) cells, LANCZOS down to the panel size)

and the record's meta only supplies the camera (origin, hw, hh, ch), the
palette and the layout -- the HEIGHTS are decoded from the PNG pixels.

Per record:
  1. form: tier <-> rung <-> answer kind <-> answer type; the question equals the
     declared template for that kind and camera; answer syntax.
  2. envelope: grid size, heights, edit counts (= added + removed), kinds
     present, tier minima (and on xhard/xxhard the descending-terrace shape
     and max(2, n//3) <= added <= min(n-2, 2n//3) for n edits).
  3. layout + chrome: image size and panel boxes; then EVERY pixel outside the
     two decode windows must equal an independently drawn reference of the
     documented chrome (white canvas, "BEFORE" on the left and "AFTER" on the
     right in DejaVuSans-Bold, the L1-L5 rounded frames, the separator line).
     So the label order the question relies on is checked, and no overlay,
     stray text or extra cube can hide outside the decode windows.
  4. DECODE (uniqueness proof): for each panel, a front-to-back
     analysis-by-synthesis search over every height of every floor-grid cell
     (the always-empty pad row/column included, heights from 0 up to "the
     whole stack is above the panel") keeps a partial assignment only if every
     display pixel whose LANCZOS support is already fully determined matches
     the PNG exactly.  Exactly one complete assignment may survive, and it must
     equal the meta heightmap.  The gold answer is recomputed from the two
     DECODED heightmaps.  Before decoding, the whole floor grid (pad row and
     column included) must lie inside the compared window: its front pad
     strip is always empty and unoccluded, so its grid lines, matched pixel
     for pixel, pin the camera that meta supplies (floor_outside_window).
     The decode compares only the display pixels that some stack can reach;
     the rest of the window (the corners below the floor rhombus and the
     strips beside it, 16-32% of the window) is the same for every heightmap.
     So, after the decode, the WHOLE window must equal the render of the
     decoded heightmap pixel for pixel (window_mismatch).  With step 3, every
     pixel of the image is verified.
  5. legibility at the delivered resolution: every column's top face keeps >= 8
     robust interior pixels, and every single-column height change (0..hmax+1)
     and every change of two screen-overlapping columns (nearer one lowered by
     1, farther one +-1) alters a connected patch of >= 24 px (eroded core
     >= 6) at channel contrast >= 20.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import functools
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CORRECTNESS = (
    "Decodes both panels' heightmaps from the PNG pixels (own iso projection and painter; meta supplies only "
    "camera, palette and layout), requires them to equal the meta heightmaps, and recomputes added = "
    "sum(max(0, after-before)) and removed = sum(max(0, before-after)) in the form the question asks "
    "(added / added+removed / 'added,removed'); the question must equal the declared template, contain no digit "
    "and carry only the placeholder format 'added,removed'; each decode window must equal the render of its "
    "decoded heightmap pixel for pixel, and every pixel outside the windows must equal the documented chrome "
    "(BEFORE label left, AFTER label right, frames, separator, white).")
UNIQUENESS = (
    "An exhaustive front-to-back decoder over every height of every floor cell (pad row/column included, up to "
    "'above the panel') must find exactly ONE heightmap per panel that reproduces the panel pixels exactly, so no "
    "other count is consistent with the image. The camera is pinned by the image: the whole floor grid, including "
    "its always-empty, unoccluded front pad strip, must lie inside the compared window. In addition every column "
    "top shows >= 8 robust px, and every single-column change and every change of two overlapping columns (nearer "
    "one lowered) alters a visible patch (contrast 20, >= 24 px).")

SS = 2
EDGE = (20, 20, 20)
FLOOR = (200, 205, 212)
WHITE = (255, 255, 255)
GRAY = (205, 208, 214)
PALETTES = ((205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160))

TIER_SPEC = {
    "easy": dict(size=448, rungs=("L1",), camera="per_panel"),
    "medium": dict(size=448, rungs=("L2", "L3"), camera="per_panel"),
    "hard": dict(size=448, rungs=("L4", "L5"), camera="per_panel"),
    "xhard": dict(size=588, rungs=("L6",), camera="shared"),
    "xxhard": dict(size=588, rungs=("L7",), camera="shared"),
}
RUNG_SPEC = {
    "L1": dict(rows=(3, 4), cols=(4,), hmax=2, edits=(1, 2), kind="added"),
    "L2": dict(rows=(3,), cols=(3,), hmax=3, edits=(2, 4), kind="total"),
    "L3": dict(rows=(3, 4), cols=(3, 4), hmax=3, edits=(3, 5), kind="total"),
    "L4": dict(rows=(3, 4), cols=(4,), hmax=4, edits=(3, 6), kind="total"),
    "L5": dict(rows=(4,), cols=(4,), hmax=4, edits=(4, 7), kind="pair"),
    "L6": dict(rows=(4,), cols=(5,), hmax=5, edits=(8, 10), kind="pair", min_each=2, min_changed=5),
    "L7": dict(rows=(5,), cols=(5,), hmax=6, edits=(12, 16), kind="pair", min_each=4, min_changed=8),
}

STEM = (
    'BEFORE (left) and AFTER (right) show the same unit-cube stacks on a floor grid; cubes were '
    'only added to or removed from stack tops. '
)
CAMERA = {
    "per_panel": "The AFTER view may be shifted or zoomed, so match stacks by floor cell. ",
    "shared": "Both panels use the same viewpoint and zoom. ",
}
RULE = "Compare heights cell by cell (empty cell = 0). "
ASK = {
    "added": "How many cubes were added? Answer with an integer.",
    "total": "How many cubes were added or removed in total? Answer with an integer.",
    "pair": "How many cubes were added and how many removed? Answer 'added,removed'.",
}
ANSWER_RE = {"added": r"[0-9]+", "total": r"[0-9]+", "pair": r"[0-9]+,[0-9]+"}
ANSWER_TYPE = {"added": "int", "total": "int", "pair": "token-list"}

# legibility thresholds (display pixels)
MIN_TOP = 8
CONTRAST, AREA, CORE = 20, 24, 6
DECODE_BUDGET = 20000


def expected_question(kind: str, camera: str) -> str:
    return STEM + CAMERA[camera] + RULE + ASK[kind]


def example_overlap(question: str, answer: str) -> list[str]:
    """Ways the question's format example could give away (part of) the answer.

    The only format example is the placeholder 'added,removed' (words, no
    numbers), so it can never equal a gold or share a component with one.  This
    is checked, not assumed: the question may contain no digit at all, and no
    quoted token (or its comma-separated components) may equal the answer or
    one of its components."""
    out = []
    if re.search(r"[0-9]", question.replace(RULE, "")):
        out.append("question contains a digit (only the placeholder format 'added,removed' is allowed)")
    gold = [p.strip() for p in str(answer).split(",")]
    for tok in re.findall(r"'([^']*)'", question):
        parts = [p.strip() for p in tok.split(",")]
        if tok.strip() == str(answer).strip():
            out.append(f"format example {tok!r} equals the answer")
        elif set(parts) & set(gold):
            out.append(f"format example {tok!r} shares a component with the answer {answer!r}")
    return out


# --------------------------------------------------------------------------- #
# geometry and painter (re-implemented from the rendering contract)
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, p: dict):
        self.ox, self.oy = float(p["origin"][0]), float(p["origin"][1])
        self.hw, self.hh, self.ch = int(p["hw"]), int(p["hh"]), int(p["ch"])

    def pt(self, i, j, k):
        return (self.ox + (i - j) * self.hw, self.oy + (i + j) * self.hh - k * self.ch)

    def faces(self, i, j, k):
        """(top, left, right) face polygons of the cube at cell (i, j), level k."""
        hw, hh, ch = self.hw, self.hh, self.ch
        x = self.ox + (i - j) * hw            # x of the back/front corners
        yb = self.oy + (i + j) * hh - k * ch  # back corner at level k
        t_back, t_right = (x, yb - ch), (x + hw, yb + hh - ch)
        t_front, t_left = (x, yb + 2 * hh - ch), (x - hw, yb + hh - ch)
        b_right, b_front, b_left = (x + hw, yb + hh), (x, yb + 2 * hh), (x - hw, yb + hh)
        return ([t_back, t_right, t_front, t_left],
                [t_left, t_front, b_front, b_left],
                [t_right, t_front, b_front, b_right])

    def cube_box(self, i, j, k, pad=2):
        """Pixel box (x0, y0, x1, y1) that the cube's fill and outlines can touch."""
        x = self.ox + (i - j) * self.hw
        yb = self.oy + (i + j) * self.hh - k * self.ch
        return (x - self.hw - pad, yb - self.ch - pad, x + self.hw + pad, yb + 2 * self.hh + pad)

    def hexagon(self, i, j, h):
        """Silhouette of a stack of height h >= 1 on cell (i, j)."""
        P = self.pt
        return [P(i, j, h), P(i + 1, j, h), P(i + 1, j, 0), P(i + 1, j + 1, 0), P(i, j + 1, 0), P(i, j + 1, h)]


def _shade(base, f):
    return tuple(max(0, min(255, int(c * f))) for c in base)


def _paint(canvas, cam, base, heights: dict, n_rows, n_cols, clip=None):
    """Floor grid + cubes of `heights` {(i, j): h} in painter order (diagonal
    i+j, then j, then level k).  With clip=(x0, y0, x1, y1) cubes that cannot
    touch that box are skipped, which leaves every pixel inside it unchanged."""
    # Same core calls as ImageDraw.line(xy, fill, width) and
    # ImageDraw.polygon(xy, fill=col, outline=EDGE) (fill pass, then a 1-px
    # outline pass when the inks differ), without per-call ink lookups.
    d = ImageDraw.Draw(canvas)
    core = d.draw
    e_ink, f_ink = d._getink(EDGE)[0], d._getink(FLOOR)[0]
    s_inks = [d._getink(_shade(base, f))[0] for f in (1.0, 0.74, 0.88)]
    n, m = n_rows + 1, n_cols + 1
    for i in range(m + 1):
        core.draw_lines([cam.pt(i, 0, 0), cam.pt(i, n, 0)], f_ink, SS)
    for j in range(n + 1):
        core.draw_lines([cam.pt(0, j, 0), cam.pt(m, j, 0)], f_ink, SS)
    for (i, j) in sorted(heights, key=lambda c: (c[0] + c[1], c[1])):
        for k in range(heights[i, j]):
            if clip is not None:
                x0, y0, x1, y1 = cam.cube_box(i, j, k)
                if x1 < clip[0] or x0 > clip[2] or y1 < clip[1] or y0 > clip[3]:
                    continue
            polys = cam.faces(i, j, k)
            for poly, s_ink in zip(polys, s_inks):
                core.draw_polygon(poly, s_ink, 1)
                if e_ink != s_ink:
                    core.draw_polygon(poly, e_ink, 0, 1)
            for poly in polys:
                core.draw_lines(poly + [poly[0]], e_ink, SS)


def render_panel(H, cam, base, panel):
    """The full panel for heightmap H (rows j, cols i)."""
    canvas = Image.new("RGB", (panel * SS, panel * SS), WHITE)
    heights = {(i, j): H[j][i] for j in range(len(H)) for i in range(len(H[0]))}
    _paint(canvas, cam, base, heights, len(H), len(H[0]))
    return canvas.resize((panel, panel), Image.LANCZOS)


def _render_box(heights, cam, base, n_rows, n_cols, box, panel):
    """Display pixels rows r0..r1-1, cols q0..q1-1 of the panel showing
    `heights`; renders only the ss crop holding their LANCZOS support (ss
    rows/cols 2r-5 .. 2r+6), aligned to even ss coordinates.  Byte-identical to
    the same pixels of a full render."""
    S = panel * SS
    r0, r1, q0, q1 = box
    sy0, sy1 = max(0, 2 * (r0 - 3)), min(S, 2 * (r1 + 3))
    sx0, sx1 = max(0, 2 * (q0 - 3)), min(S, 2 * (q1 + 3))
    canvas = Image.new("RGB", (sx1, sy1), WHITE)
    _paint(canvas, cam, base, heights, n_rows, n_cols, clip=(sx0, sy0, sx1, sy1))
    got = np.asarray(canvas.crop((sx0, sy0, sx1, sy1)).resize(((sx1 - sx0) // 2, (sy1 - sy0) // 2), Image.LANCZOS))
    oy, ox = r0 - sy0 // 2, q0 - sx0 // 2
    return got[oy:oy + (r1 - r0), ox:ox + (q1 - q0)]


def _poly_mask(shape, poly):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(poly, fill=1, outline=1)
    return np.asarray(im, dtype=bool)


def _bbox(poly, pad, S):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = max(0, int(np.floor(min(xs))) - pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    x1 = min(S, int(np.ceil(max(xs))) + pad + 1)
    y1 = min(S, int(np.ceil(max(ys))) + pad + 1)
    return x0, y0, x1, y1


def _box_count(m, r):
    """Number of True pixels in the (2r+1)^2 window around each pixel (zero outside)."""
    p = np.pad(m.astype(np.int32), r)
    sat = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.int32)
    sat[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return sat[k:, k:] - sat[:-k, k:] - sat[k:, :-k] + sat[:-k, :-k]


def _local_mask(poly, box, grow=0):
    """Polygon rasterised inside box (x0, y0, x1, y1), dilated (grow > 0) or
    eroded (grow < 0) by |grow| pixels in the Chebyshev metric."""
    x0, y0, x1, y1 = box
    m = _poly_mask((y1 - y0, x1 - x0), [(x - x0, y - y0) for x, y in poly])
    if grow > 0:
        return _box_count(m, grow) > 0
    if grow < 0:
        return _box_count(m, -grow) == (2 * -grow + 1) ** 2
    return m


# --------------------------------------------------------------------------- #
# exact decoder
# --------------------------------------------------------------------------- #
class _Decoder:
    """Exhaustive front-to-back analysis-by-synthesis decoder of one panel.

    Cells (pad row/column included) are decided in exactly the reverse of the
    painter order, so when a stack is decided every stack painted after it is
    already decided.  An ss pixel is FINAL when (a) a decided stack's draw
    calls touch it (they are opaque and not antialiased, and only decided
    stacks are painted later), or (b) no undecided cell can touch it at any
    height (its tallest silhouette, grown by 2 px for outlines, misses it).  A
    display pixel is compared with the PNG as soon as all ss pixels of its
    LANCZOS support (ss rows/cols 2r-5 .. 2r+6) are final.  Final pixels never
    change again, so a mismatch prunes the subtree.  Only display pixels near
    some cell's tallest silhouette (its box grown by 6 ss px) are ever
    compared, so a complete assignment reproduces every window pixel that a
    stack can reach.  The other window pixels (corners below the floor, strips
    beside it) render identically for every heightmap, so they play no part in
    uniqueness; check_image compares them afterwards (window_mismatch).

    Two exact reductions keep the search small: a height whose every touched
    pixel is already overwritten by decided nearer stacks renders exactly like
    height 0 under any completion ("hidden class", explored once; if it leads
    to a solution and the class has >= 2 heights the panel is ambiguous), and a
    candidate's check depends only on the heights of the decided cells that
    overlap its region, so it is memoised on that context."""

    GROW = 2

    def __init__(self, panel_img: np.ndarray, cam: Camera, base, n_rows, n_cols, window):
        self.img = panel_img
        self.cam, self.base = cam, base
        self.N, self.M = n_rows, n_cols
        self.panel = panel_img.shape[0]
        self.S = self.panel * SS
        self.r0, self.r1, self.c0, self.c1 = window
        S = self.S
        cells = [(i, j) for j in range(n_rows + 1) for i in range(n_cols + 1)]  # pad row/col included
        self.order = sorted(cells, key=lambda c: (-(c[0] + c[1]), -c[1]))     # reverse painter order
        # Cube level K cannot reach a window pixel once its lowest point (+2 px
        # of outline) is above ss row 2*r0 - 5, so every height >= cap renders
        # identically inside the window: cap stands for "cap or taller".
        top_ss = 2 * self.r0 - 5
        self.cap = {}
        for (i, j) in cells:
            K = 1
            while K < 60 and cam.oy + (i + j + 2) * cam.hh - K * cam.ch + 2 >= top_ss:
                K += 1
            self.cap[i, j] = K
        self.unk = np.zeros((S, S), np.int16)
        self.own = np.zeros((S, S), np.int16)
        self.maxbox, self.maxmask = {}, {}
        for c in cells:
            poly = cam.hexagon(c[0], c[1], self.cap[c])
            box = _bbox(poly, self.GROW + 1, S)
            m = _local_mask(poly, box, self.GROW)
            self.maxbox[c], self.maxmask[c] = box, m
            x0, y0, x1, y1 = box
            self.unk[y0:y1, x0:x1] += m
        # cells decided before c whose region can meet c's evaluation region
        def grown(b, g):
            return (b[0] - g, b[1] - g, b[2] + g, b[3] + g)
        def meets(a, b):
            return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
        self.context = {}
        for t, c in enumerate(self.order):
            reg = grown(self.maxbox[c], 14)
            self.context[c] = [d for d in self.order[:t] if meets(reg, grown(self.maxbox[d], 14))]
        self._own_cache = {}
        self._memo = {}
        self.checked = np.zeros((self.panel, self.panel), bool)
        self.heights: dict = {}
        self.solutions: list = []
        self.nodes = 0
        self.exhausted = False

    def _owned(self, c, h):
        """Exact set of ss pixels the draw calls of stack c@h touch."""
        key = (c, h)
        if key not in self._own_cache:
            cam, S = self.cam, self.S
            box = _bbox(cam.hexagon(c[0], c[1], h), 2, S)
            im = Image.new("L", (box[2], box[3]), 0)  # absolute coordinates: same rasterisation as the render
            d = ImageDraw.Draw(im)
            for k in range(h):
                polys = cam.faces(c[0], c[1], k)
                for poly in polys:
                    d.polygon(poly, fill=1, outline=1)
                for poly in polys:
                    d.line(poly + [poly[0]], fill=1, width=SS)
            m = np.asarray(im, dtype=bool)[box[1]:box[3], box[0]:box[2]].copy()
            self._own_cache[key] = (box, m)
        return self._own_cache[key]

    def _hidden(self, c, h):
        """c@h touches only pixels that decided (later-painted) stacks overwrite."""
        if h == 0:
            return True
        (a0, b0, a1, b1), m = self._owned(c, h)
        return bool(np.all(self.own[b0:b1, a0:a1][m] > 0))

    def _apply(self, c, h):
        x0, y0, x1, y1 = self.maxbox[c]
        self.unk[y0:y1, x0:x1] -= self.maxmask[c]
        if h:
            (a0, b0, a1, b1), m = self._owned(c, h)
            self.own[b0:b1, a0:a1] += m
        self.heights[c] = h

    def _revert(self, c, h):
        x0, y0, x1, y1 = self.maxbox[c]
        self.unk[y0:y1, x0:x1] += self.maxmask[c]
        if h:
            (a0, b0, a1, b1), m = self._owned(c, h)
            self.own[b0:b1, a0:a1] -= m
        del self.heights[c]

    def _check_new(self, c):
        """Compare the display pixels around cell c that just became final;
        return their coordinates, or None on a mismatch."""
        key = (c, self.heights[c], tuple(self.heights[d] for d in self.context[c]))
        if key in self._memo:
            return self._memo[key]
        res = self._check_new_raw(c)
        self._memo[key] = res
        return res

    def _check_new_raw(self, c):
        S = self.S
        x0, y0, x1, y1 = self.maxbox[c]
        r0 = max(self.r0, (y0 - 6) // 2)
        r1 = min(self.r1, (y1 + 6) // 2 + 1)
        q0 = max(self.c0, (x0 - 6) // 2)
        q1 = min(self.c1, (x1 + 6) // 2 + 1)
        if r0 >= r1 or q0 >= q1:
            return ()
        sy0, sy1 = max(0, 2 * (r0 - 3)), min(S, 2 * (r1 + 3))
        sx0, sx1 = max(0, 2 * (q0 - 3)), min(S, 2 * (q1 + 3))
        unknown = ~((self.own[sy0:sy1, sx0:sx1] > 0) | (self.unk[sy0:sy1, sx0:sx1] == 0))
        sat = np.zeros((unknown.shape[0] + 1, unknown.shape[1] + 1), np.int32)
        sat[1:, 1:] = unknown.cumsum(0).cumsum(1)
        rows, cols = np.arange(r0, r1), np.arange(q0, q1)
        ya = np.clip(2 * rows - 5, sy0, sy1) - sy0
        yb = np.clip(2 * rows + 7, sy0, sy1) - sy0
        xa = np.clip(2 * cols - 5, sx0, sx1) - sx0
        xb = np.clip(2 * cols + 7, sx0, sx1) - sx0
        cnt = sat[yb][:, xb] - sat[ya][:, xb] - sat[yb][:, xa] + sat[ya][:, xa]
        new = (cnt == 0) & ~self.checked[r0:r1, q0:q1]
        if not new.any():
            return ()
        ys, xs = np.nonzero(new)
        ys += r0
        xs += q0
        decided = {d: h for d, h in self.heights.items() if h}
        h = self.heights[c]
        if h:
            # cheap probe first: wrong heights almost always fail at the
            # centre of their own top face
            cx, cy = self.cam.pt(c[0] + 0.5, c[1] + 0.5, h)
            near = (np.abs(ys - cy / 2) <= 4) & (np.abs(xs - cx / 2) <= 6)
            if near.any():
                py, px = ys[near], xs[near]
                pbox = (int(py.min()), int(py.max()) + 1, int(px.min()), int(px.max()) + 1)
                pred = _render_box(decided, self.cam, self.base, self.N, self.M, pbox, self.panel)
                if np.any(pred[py - pbox[0], px - pbox[2]] != self.img[py, px]):
                    return None
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        pred = _render_box(decided, self.cam, self.base, self.N, self.M, box, self.panel)
        if np.any(pred[ys - box[0], xs - box[2]] != self.img[ys, xs]):
            return None
        return (ys, xs)

    def _dfs(self, t):
        if t == len(self.order):
            self.solutions.append(dict(self.heights))
            return
        c = self.order[t]
        hidden = [h for h in range(self.cap[c] + 1) if self._hidden(c, h)]   # a prefix 0..h_hid
        for h in [0] + [h for h in range(self.cap[c] + 1) if h not in hidden]:
            self.nodes += 1
            if self.nodes > DECODE_BUDGET:
                self.exhausted = True
                return
            before = len(self.solutions)
            self._apply(c, h)
            res = self._check_new(c)
            if res is not None:
                if len(res):
                    self.checked[res[0], res[1]] = True
                self._dfs(t + 1)
                if len(res):
                    self.checked[res[0], res[1]] = False
            self._revert(c, h)
            if h == 0 and len(hidden) > 1 and len(self.solutions) > before:
                twin = dict(self.solutions[before])  # same image with c at another hidden height
                twin[c] = hidden[1]
                self.solutions.append(twin)
            if len(self.solutions) >= 2 or self.exhausted:
                return

    def run(self):
        self._dfs(0)
        return self.solutions[:2]


def decode_panel(panel_img: np.ndarray, cam: Camera, base, n_rows, n_cols, window):
    """Returns (solutions as N x M lists [<= 2], info).  Solutions that put
    cubes on the pad row/column or a stack at its cap are reported in info."""
    dec = _Decoder(panel_img, cam, base, n_rows, n_cols, window)
    sols = dec.run()
    out, notes = [], []
    for s in sols:
        H = [[s[i, j] for i in range(n_cols)] for j in range(n_rows)]
        pads = {c: h for c, h in s.items() if (c[0] == n_cols or c[1] == n_rows) and h}
        capped = [c for c, h in s.items() if h >= dec.cap[c]]
        if pads:
            notes.append(f"solution with cubes on the pad row/column {pads}")
        if capped:
            notes.append(f"solution with stacks reaching above the panel at {capped}")
        out.append(H)
    return out, dict(nodes=dec.nodes, exhausted=dec.exhausted, notes=notes)


# --------------------------------------------------------------------------- #
# legibility
# --------------------------------------------------------------------------- #
def top_face_counts(H, cam, panel):
    """Robust display pixels of every column's top face: all 2x2 ss subpixels
    and the 3x3 display neighbourhood carry that face's label; outlines and the
    floor count as background."""
    S = panel * SS
    n_rows, n_cols = len(H), len(H[0])
    im = Image.new("I", (S, S), 0)
    d = ImageDraw.Draw(im)
    for (i, j) in sorted(((i, j) for j in range(n_rows) for i in range(n_cols)), key=lambda c: (c[0] + c[1], c[1])):
        h = H[j][i]
        for k in range(h):
            polys = cam.faces(i, j, k)
            for f, poly in enumerate(polys):
                lab = 1 + j * n_cols + i if (f == 0 and k == h - 1) else 0
                d.polygon(poly, fill=lab, outline=0)
            for poly in polys:
                d.line(poly + [poly[0]], fill=0, width=SS)
    a = np.asarray(im, dtype=np.int64)
    sub = a.reshape(panel, SS, panel, SS).transpose(0, 2, 1, 3)
    lo, hi = sub.min(axis=(2, 3)), sub.max(axis=(2, 3))
    ids = np.where(lo == hi, lo, 0)
    pad = np.pad(ids, 1)
    inner = ids.copy()
    for dy in range(3):
        for dx in range(3):
            inner[pad[dy:dy + panel, dx:dx + panel] != ids] = 0
    cnt = np.bincount(inner.ravel(), minlength=n_rows * n_cols + 1)
    return {(i, j): int(cnt[1 + j * n_cols + i]) for j in range(n_rows) for i in range(n_cols) if H[j][i] > 0}


def visible_patch(a: np.ndarray, b: np.ndarray):
    """(ok, area, core) of the largest connected patch with contrast >= CONTRAST."""
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) >= CONTRAST
    if not diff.any():
        return False, 0, 0
    lab, n = ndimage.label(diff, structure=np.ones((3, 3)))
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    er = ndimage.binary_erosion(diff, structure=np.ones((3, 3)))
    cores = np.bincount(lab[er], minlength=n + 1)
    good = [k for k in range(1, n + 1) if areas[k] >= AREA and cores[k] >= CORE]
    k = max(good, key=lambda q: areas[q]) if good else int(np.argmax(areas[1:])) + 1
    return bool(good), int(areas[k]), int(cores[k])


def _alternatives(H, hmax, cam, S):
    """Every single-column height 0..hmax+1, and every pair of columns whose
    silhouettes overlap on screen with the NEARER one lowered by 1 and the
    farther one changed by +-1.  (When the nearer column is raised, its new
    cube is painted last over the same pixels as in the single-column change,
    so that pair differs at least where the single change does; lowering the
    nearer column is the only way a second change can mask the first.)"""
    n_rows, n_cols = len(H), len(H[0])
    cells = [(i, j) for j in range(n_rows) for i in range(n_cols)]
    for (i, j) in cells:
        for v in range(hmax + 2):
            if v != H[j][i]:
                yield ((i, j, v),)
    box = {(i, j): _bbox(cam.hexagon(i, j, min(hmax + 1, H[j][i] + 1)), 0, S) for (i, j) in cells}
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            near, far = sorted((cells[a], cells[b]), key=lambda c: (c[0] + c[1], c[1]), reverse=True)
            (ia, ja), (ib, jb) = near, far
            ba, bb = box[near], box[far]
            if abs((ia - ja) - (ib - jb)) > 1 or ba[2] <= bb[0] or bb[2] <= ba[0] or ba[3] <= bb[1] or bb[3] <= ba[1]:
                continue  # these two stacks never overlap on screen
            va = H[ja][ia] - 1
            for db in (-1, 1):
                vb = H[jb][ib] + db
                if va >= 0 and 0 <= vb <= hmax + 1:
                    yield ((ia, ja, va), (ib, jb, vb))


def weakest_alternative(H, ref: np.ndarray, cam, base, hmax, window):
    """Brute force around H (whose exact panel pixels are `ref`); returns
    (n alternatives, n without a visible patch, weakest (area, core, change))."""
    panel = ref.shape[0]
    S = panel * SS
    n_rows, n_cols = len(H), len(H[0])
    n = weak = 0
    worst = None
    for change in _alternatives(H, hmax, cam, S):
        heights = {(i, j): H[j][i] for j in range(n_rows) for i in range(n_cols)}
        xs0, ys0, xs1, ys1 = S, S, 0, 0
        for i, j, v in change:
            # only the cubes between the two heights (and the lower top face,
            # which the lowest of them covers) are painted differently
            lo, hi = sorted((v, heights[i, j]))
            heights[i, j] = v
            bx0, by0 = cam.cube_box(i, j, hi - 1)[:2]
            bx1, by1 = cam.cube_box(i, j, lo)[2:]
            bx0, by0 = max(0, int(np.floor(bx0))), max(0, int(np.floor(by0)))
            bx1, by1 = min(S, int(np.ceil(bx1)) + 1), min(S, int(np.ceil(by1)) + 1)
            xs0, ys0, xs1, ys1 = min(xs0, bx0), min(ys0, by0), max(xs1, bx1), max(ys1, by1)
        r0, r1 = max(window[0], (ys0 - 6) // 2), min(window[1], (ys1 + 6) // 2 + 1)
        q0, q1 = max(window[2], (xs0 - 6) // 2), min(window[3], (xs1 + 6) // 2 + 1)
        n += 1
        if r0 >= r1 or q0 >= q1:
            weak += 1
            worst = (0, 0, change)
            continue
        alt = _render_box(heights, cam, base, n_rows, n_cols, (r0, r1, q0, q1), panel)
        ok, area, core = visible_patch(ref[r0:r1, q0:q1], alt)
        if not ok:
            weak += 1
        if worst is None or (area, core) < worst[:2]:
            worst = (area, core, change)
    return n, weak, worst


# --------------------------------------------------------------------------- #
# record checks
# --------------------------------------------------------------------------- #
def _counts(H1, H2):
    a, b = np.asarray(H1), np.asarray(H2)
    return int(np.maximum(b - a, 0).sum()), int(np.maximum(a - b, 0).sum())


def _gold(kind, added, removed):
    return {"added": str(added), "total": str(added + removed), "pair": f"{added},{removed}"}[kind]


def _is_grid(H):
    return (isinstance(H, list) and H and all(isinstance(r, list) and len(r) == len(H[0]) and r for r in H)
            and all(isinstance(v, int) and not isinstance(v, bool) for r in H for v in r))


def check_form(record) -> list[str]:
    P = []
    m = record.get("meta") or {}
    tier = record.get("tier")
    if tier not in TIER_SPEC:
        return [f"unknown tier {tier!r}"]
    spec = TIER_SPEC[tier]
    rung = m.get("rung")
    if rung not in spec["rungs"]:
        return [f"rung {rung!r} is not a {tier} rung {spec['rungs']}"]
    idx = (record.get("source") or {}).get("index")
    if len(spec["rungs"]) == 2 and isinstance(idx, int) and spec["rungs"][idx % 2] != rung:
        P.append(f"rung {rung} does not match index parity {idx} ({spec['rungs']})")
    rs = RUNG_SPEC[rung]
    kind = m.get("answer_kind")
    if kind != rs["kind"]:
        P.append(f"answer kind {kind!r} != {rs['kind']!r} for {rung}")
        return P
    if record.get("answer_type") != ANSWER_TYPE[kind]:
        P.append(f"answer_type {record.get('answer_type')!r} != {ANSWER_TYPE[kind]!r}")
    if m.get("camera") != spec["camera"]:
        P.append(f"camera {m.get('camera')!r} != {spec['camera']!r}")
    if record.get("question") != expected_question(kind, spec["camera"]):
        P.append("question text differs from the declared template for this answer kind / framing")
    P += example_overlap(str(record.get("question", "")), str(record.get("answer", "")))
    if not re.fullmatch(ANSWER_RE[kind], str(record.get("answer", ""))):
        P.append(f"answer {record.get('answer')!r} is not of the form {ANSWER_RE[kind]}")
    return P


def check_envelope(record) -> list[str]:
    m = record["meta"]
    rs = RUNG_SPEC[m["rung"]]
    H1, H2 = m.get("heightmap_before"), m.get("heightmap_after")
    if not (_is_grid(H1) and _is_grid(H2)):
        return ["heightmaps are not integer grids"]
    a, b = np.asarray(H1), np.asarray(H2)
    P = []
    if a.shape != b.shape or a.shape[0] not in rs["rows"] or a.shape[1] not in rs["cols"]:
        P.append(f"grid {a.shape}/{b.shape} outside {rs['rows']} x {rs['cols']}")
        return P
    if min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > rs["hmax"]:
        P.append(f"height outside 0..{rs['hmax']}")
    added, removed = _counts(H1, H2)
    if not rs["edits"][0] <= added + removed <= rs["edits"][1]:
        P.append(f"{added + removed} edits outside {rs['edits']}")
    if rs["kind"] == "added":
        if removed or not added:
            P.append("L1 must only add cubes")
        if not 4 <= a.sum() <= 12:
            P.append("L1 BEFORE must hold 4-12 cubes")
    elif not (added and removed):
        P.append("mixed rung without both additions and removals")
    if b.sum() == 0:
        P.append("AFTER is empty")
    if "min_each" in rs:
        n = added + removed
        if min(added, removed) < rs["min_each"] or int((a != b).sum()) < rs["min_changed"]:
            P.append("xhard/xxhard minimum change constraints violated")
        if not max(2, n // 3) <= added <= min(n - 2, 2 * n // 3):
            P.append(f"xhard/xxhard added share {added}/{n} outside [max(2,n//3), min(n-2,2n//3)]")
        for H in (a, b):
            if (np.diff(H, axis=0) > 0).any() or (np.diff(H, axis=1) > 0).any():
                P.append("xhard/xxhard map is not a descending terrace")
                break
    if (m.get("added"), m.get("removed")) != (added, removed):
        P.append(f"meta added/removed {m.get('added')},{m.get('removed')} != heightmaps {added},{removed}")
    return P


def floor_outside_window(cam, n_rows, n_cols, window, margin=3):
    """Corners of the floor grid (pad row/column included) that are not at
    least `margin` display px inside the compared window.

    The pad row and column (i = n_cols or j = n_rows) are the front strip of
    the floor.  They are always empty (the decode rejects cubes there) and
    nothing is painted in front of them: a stack's silhouette ends at its own
    cell's front edges.  Their grid lines are therefore always visible, and
    they fix the camera's origin and tile size.  This is how a viewer lines up
    the two separately framed L1-L5 panels.  Requiring the whole floor (a
    convex rhombus) to lie inside the window means the decode compared those
    lines pixel for pixel, so the camera taken from meta is the one the image
    shows, not an outside assumption."""
    r0, r1, c0, c1 = window
    bad = []
    for (i, j) in ((0, 0), (n_cols + 1, 0), (n_cols + 1, n_rows + 1), (0, n_rows + 1)):
        x, y = cam.pt(i, j, 0)
        x, y = x / SS, y / SS
        if not (c0 + margin <= x <= c1 - margin and r0 + margin <= y <= r1 - margin):
            bad.append((i, j, round(x, 1), round(y, 1)))
    return bad


def _camera_window(camera, panel):
    """Decode window (r0, r1, c0, c1) inside a panel, in panel pixels."""
    if camera == "per_panel":
        # L1-L5: the BEFORE/AFTER label text, the rounded frame and its corner
        # arcs are drawn over the pasted panel; keep clear of them.
        return (6, panel - 18, 12, panel - 12)
    return (0, panel, 0, panel)


def _window(tier, panel):
    return _camera_window(TIER_SPEC[tier]["camera"], panel)


# --------------------------------------------------------------------------- #
# chrome: everything outside the decode windows
# --------------------------------------------------------------------------- #
FONT_BOLD = (_ROOT / 'assets/fonts/DejaVuSans-Bold.ttf')
LABELS = ("BEFORE", "AFTER")  # left, right -- the order the question states


def layout(size):
    """Documented layout: (panel size, panel x positions, panel top y)."""
    panel = (size - 12) // 2
    return panel, (4, 8 + panel), (size - panel) // 2 + 10


@functools.lru_cache(maxsize=4)
def chrome_reference(camera: str, size: int):
    """(reference RGB array, mask of the pixels it governs) for a layout.

    Drawn here from the documented chrome, not taken from the generator:
      per_panel (L1-L5, 448 px): white canvas; per panel, the label centred
        at (x + (panel - textlength)/2, y - 14) in DejaVuSans-Bold 17, fill
        (60,60,66), then a rounded frame [x, y-18, x+panel, y+panel-14],
        radius 10, outline (225,225,230), width 2; then the separator from
        (size//2, 24) to (size//2, size-24), fill (230,230,234), width 2.
      shared (L6/L7, 588 px): white canvas; per panel, the label at
        (x + 14, y - 24) in DejaVuSans-Bold 18, fill (30,30,30); then the
        separator from (size//2, y-25) to (size//2, y+panel), fill
        (210,215,220), width 2.
    Panel content keeps a 24 px (L1-L5) margin on a white background, so the
    pixels outside the decode windows are exactly this drawing.  The mask is
    True outside both decode windows."""
    panel, xs, y = layout(size)
    img = Image.new("RGB", (size, size), WHITE)
    d = ImageDraw.Draw(img)
    if camera == "per_panel":
        f = ImageFont.truetype(str(FONT_BOLD), 17)
        for x, label in zip(xs, LABELS):
            tw = d.textlength(label, font=f)
            d.text((x + (panel - tw) / 2, y - 14), label, fill=(60, 60, 66), font=f)
            d.rounded_rectangle([x, y - 18, x + panel, y + panel - 14], radius=10, outline=(225, 225, 230), width=2)
        d.line([(size // 2, 24), (size // 2, size - 24)], fill=(230, 230, 234), width=2)
    else:
        f = ImageFont.truetype(str(FONT_BOLD), 18)
        for x, label in zip(xs, LABELS):
            d.text((x + 14, y - 24), label, fill=(30, 30, 30), font=f)
        d.line([(size // 2, y - 25), (size // 2, y + panel)], fill=(210, 215, 220), width=2)
    r0, r1, c0, c1 = _camera_window(camera, panel)
    outside = np.ones((size, size), bool)
    for x in xs:
        outside[y + r0:y + r1, x + c0:x + c1] = False
    ref = np.asarray(img)
    ref.setflags(write=False)
    outside.setflags(write=False)
    return ref, outside


def check_chrome(arr: np.ndarray, camera: str, size: int) -> list[str]:
    """Every pixel outside the decode windows must equal the documented chrome."""
    try:
        ref, outside = chrome_reference(camera, size)
    except OSError as exc:  # font missing: fail closed
        return [f"cannot draw the chrome reference ({exc}); label order unchecked"]
    bad = outside & (arr != ref).any(axis=2)
    if not bad.any():
        return []
    ys, xs = np.nonzero(bad)
    return [f"{int(bad.sum())} px outside the decode windows differ from the documented chrome (BEFORE left, "
            f"AFTER right, frames, separator, white) in box x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}"]


def window_mismatch(pimg: np.ndarray, H, cam, base, window) -> str:
    """'' when every pixel of the decode window equals the full render of H,
    else a description of the differing pixels (panel coordinates)."""
    r0, r1, c0, c1 = window
    ref = np.asarray(render_panel(H, cam, base, pimg.shape[0]))
    bad = (ref[r0:r1, c0:c1] != pimg[r0:r1, c0:c1]).any(axis=2)
    if not bad.any():
        return ""
    ys, xs = np.nonzero(bad)
    return (f"{int(bad.sum())} window px differ from the render of the decoded heightmap in box "
            f"x {c0 + xs.min()}..{c0 + xs.max()}, y {r0 + ys.min()}..{r0 + ys.max()} (panel px)")


def check_image(record, image) -> tuple[list[str], dict]:
    """Layout + decode + legibility.  Returns (problems, decoded info)."""
    m = record["meta"]
    tier = record["tier"]
    size = TIER_SPEC[tier]["size"]
    P = []
    if image.size != (size, size):
        return [f"image size {image.size} != {(size, size)}"], {}
    panel, xs, y = layout(size)
    xs = list(xs)
    if (m.get("panel_size"), list(m.get("panel_x", [])), m.get("panel_y"), m.get("resolution")) != (panel, xs, y, size):
        P.append("meta layout differs from the documented layout")
    base = tuple(m.get("palette", ()))
    if base not in PALETTES or (TIER_SPEC[tier]["camera"] == "shared" and base != GRAY):
        return P + [f"palette {base} not allowed"], {}
    arr = np.asarray(image.convert("RGB"))
    chrome = check_chrome(arr, TIER_SPEC[tier]["camera"], size)
    if chrome:
        # wrong labels, overlays or stray marks: the image is not the documented
        # question image, so the (costly) decode is skipped
        return P + chrome, {}
    H1, H2 = m["heightmap_before"], m["heightmap_after"]
    n_rows, n_cols = len(H1), len(H1[0])
    pb, pa = m.get("placement_before") or {}, m.get("placement_after") or {}
    if TIER_SPEC[tier]["camera"] == "shared" and pb != pa:
        P.append("shared-camera tier with two different placements")
    win = _window(tier, panel)
    hmax = RUNG_SPEC[m["rung"]]["hmax"]
    decoded, info = [], {"panels": []}
    for name, x, H, pl in (("BEFORE", xs[0], H1, pb), ("AFTER", xs[1], H2, pa)):
        try:
            if int(pl.get("ss", 0)) != SS or (pl.get("N"), pl.get("M")) != (n_rows, n_cols):
                P.append(f"{name}: placement grid/ss mismatch")
                continue
            cam = Camera(pl)
        except (KeyError, TypeError, ValueError):
            P.append(f"{name}: bad placement {pl!r}")
            continue
        if not (cam.hw > 0 and cam.hh > 0 and cam.ch > 2 * cam.hh):
            P.append(f"{name}: implausible camera {pl}")
            continue
        off = floor_outside_window(cam, n_rows, n_cols, win)
        if off:
            P.append(f"{name}: floor grid corners outside the compared window {off} (camera not pinned by the image)")
            continue
        pimg = arr[y:y + panel, x:x + panel]
        sols, dinfo = decode_panel(pimg, cam, base, n_rows, n_cols, win)
        pinfo = dict(decode_nodes=dinfo["nodes"], solutions=len(sols))
        info["panels"].append(pinfo)
        if dinfo["exhausted"]:
            P.append(f"{name}: decoder budget exhausted ({dinfo['nodes']} nodes)")
            continue
        if not sols:
            P.append(f"{name}: no heightmap reproduces the panel pixels")
            continue
        if len(sols) > 1:
            diff = [(i, j, sols[0][j][i], sols[1][j][i]) for j in range(n_rows) for i in range(n_cols)
                    if sols[0][j][i] != sols[1][j][i]]
            P.append(f"{name}: AMBIGUOUS, >= 2 heightmaps reproduce the panel exactly (cells (i,j,h,h') {diff[:4]})")
            continue
        if dinfo["notes"]:
            P.append(f"{name}: " + "; ".join(dinfo["notes"]))
            continue
        if sols[0] != H:
            P.append(f"{name}: decoded heightmap {sols[0]} != meta {H}")
            continue
        # The decode compares only display pixels some stack could reach; the
        # rest of the window (corners below the floor, strips beside it) is the
        # same for every heightmap.  Compare the WHOLE window with the render of
        # the decoded heightmap, so nothing foreign can sit there either.
        bad = window_mismatch(pimg, H, cam, base, win)
        if bad:
            P.append(f"{name}: {bad}")
            continue
        decoded.append(sols[0])
        tops = top_face_counts(H, cam, panel)
        low = {c: v for c, v in tops.items() if v < MIN_TOP}
        pinfo["min_top_face_px"] = min(tops.values()) if tops else None
        if low:
            P.append(f"{name}: column top faces with < {MIN_TOP} robust px: {low}")
        n_alt, n_weak, worst = weakest_alternative(H, pimg, cam, base, hmax, win)
        pinfo.update(alternatives=n_alt, weakest_area=worst[0] if worst else None,
                     weakest_core=worst[1] if worst else None)
        if n_weak:
            P.append(f"{name}: {n_weak}/{n_alt} alternative heightmaps differ by no visible patch "
                     f"(weakest {worst[2]} area {worst[0]} core {worst[1]})")
    info["decoded"] = decoded
    return P, info


def verify(record: dict, image) -> list[str]:
    """Problems with this record (empty == pass).  The cheap form, envelope and
    answer checks run first; when they already fail, the pixel decode is
    skipped (the driver rejects on the first problem)."""
    try:
        P = check_form(record)
        if any(s.startswith(("unknown tier", "rung ", "answer kind")) for s in P):
            return P
        P += check_envelope(record)
        if any(s.startswith(("heightmaps are not", "grid ")) for s in P):
            return P
        m = record["meta"]
        added, removed = _counts(m["heightmap_before"], m["heightmap_after"])
        gold = _gold(m["answer_kind"], added, removed)
        if record["answer"] != gold:
            P.append(f"answer {record['answer']!r} != recomputed {gold!r}")
        if P:
            return P
        P, info = check_image(record, image)
        if len(info.get("decoded", [])) == 2:
            dg = _gold(m["answer_kind"], *_counts(*info["decoded"]))
            if dg != record["answer"]:
                P.append(f"answer {record['answer']!r} != answer of the decoded panels {dg!r}")
        return P
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return [f"malformed record: {type(exc).__name__}: {exc}"]
