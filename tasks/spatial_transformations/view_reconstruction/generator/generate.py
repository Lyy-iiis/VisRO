"""Generator of view_reconstruction.

LEVEL = high (3D mental simulation). The top row shows 2-3 orthographic
silhouette views (TOP / FRONT / SIDE) of a hidden cube structure; four option
panels (A-D) below show candidate structures rendered isometrically on their
floor grid. Exactly one candidate is consistent with ALL shown views.

View conventions (physical cameras, identical on every tier, stated in the
question and drawn on the image):
  top   : looking down; N rows (depth j, back row j=0 at the TOP, arrow BACK)
          x M columns (i, left to right).
  front : camera in front (+j) looking back; column i shows a bar of height
          max_j H[j][i]; width M, i increasing to the right.
  side  : camera on the right (+i) looking left; column k shows the bar
          max_i H[N-1-k][i]: the FRONT row is at the LEFT, the BACK row at the
          RIGHT (arrow BACK). This is the physical right-side view, consistent
          with the face legend.
  Both lateral views are exactly max(H_answer) levels tall.

Exchangeable options
--------------------
If every distractor were an independent edit of the truth, the truth would be
the medoid of the four options and could be picked without the views. Instead
the four options are a lattice built from a prior sample X and edits A, B (C)
on pairwise disjoint cells:

  square : {X, X+A, X+B, X+A+B}
  tetra  : {X, X+A+B, X+A+C, X+B+C}

Every acceptance gate below is evaluated for EVERY choice of answer inside the
set, and only then is the answer drawn uniformly from the four options. So,
conditional on the option set, the answer is uniform: no statistic of the
option panel alone (medoid, volume, ink, prior likelihood, projection
agreement, ...) can beat 25% in expectation.

Gates (all symmetric in the four options):
  * partition : for every shown view, the four options' projections fall in
    two classes of two (so every shown view rules out exactly two of the three
    distractors, no single view suffices, and the options are pairwise
    distinguishable by the shown views); the design fixes how the classes of
    different views relate (square-2: two different splits; tetra: three
    different splits; square-3: one view's split differs from the other two).
    With four options and three views it is impossible for every answer choice
    to need all three views, so two views are always necessary and sufficient.
  * misreading : no option matches another option's shown views under any
    alternative reading (left/right mirror of any view, front/back flip or
    180-degree turn of the top view), nor under a shifted reading that ignores
    empty edge rows/columns of the view panels (trimmed_key).
  * legibility : every column top and every empty floor cell of every option
    shows >= MIN_SURFACE interior pixels at the final option size; every pair
    of option renders differs readably; and no option has a near-identical
    single-cell twin (rendered with the same camera) whose projections equal
    another option's, so a misread column can never turn a distractor into a
    consistent structure.

Each problem is a pure function of the RNG passed to make_item.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso
from shared.drawing.fonts import load_bundled_font

GENERATOR_VERSION = "view_reconstruction-exchangeable"


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
TIERS = ("easy", "medium", "hard", "xhard", "xxhard")
# Every rung except easy (see below) draws X from extra.diverse_map(holes=True),
# whose proposals keep column tops and floor cells visible at the final option size.
# design  : "square" = {X, X+A, X+B, X+A+B}; "tetra" = {X, X+A+B, X+A+C, X+B+C}
# edit    : "cells" = one lattice edit sets `edits` cells to new heights in
#           0..cap; "move" = one lattice edit moves one cube (equal volume)
# easy (L1): a 3-4 x 4 floor with 121 px options has only ~37k legal answers, too
# few for a 25,000-item cell. L1 keeps cap 2, two views, the
# square lattice, 2-3-cell edits, the 4-12-cube window of the prior sample and
# every gate, and gets its room from more floor shapes instead of bigger
# structures:
#   * one footprint per item from `footprints` = 4x4, 3x5, 2x6, 2x7 with
#     `footprint_weights` 11:22:14:42 (thousands of distinct legal 4-12-cube
#     full-span X per footprint, counting a 4x4 map and its mirror once), so
#     every structure is about equally likely; the three oblong footprints exist
#     in ONE orientation only, so none of their structures has a mirror image
#     (transpose) on an easy floor,
#   * the 588 px layout (170 px options, as xhard/xxhard): at 121 px these
#     floors leave few maps legible,
#   * X UNIFORM over diverse_map's support with `cubes` = 4..12 (prior=
#     "uniform", see uniform_support_map) instead of its sloped-terrain weights,
#   * every option touches all four edges of its floor (`full_span`): the floor
#     is the bounding box, so no item shows another structure shifted.
# Rungs without these keys sample X with diverse_map's default weights.
RUNGS = {
    1: dict(rows=(2, 4), cols=(4, 7), footprints=((4, 4), (3, 5), (2, 6), (2, 7)),
            footprint_weights=(11, 22, 14, 42), cap=2, n_views=2, design="square",
            edit="cells", edits=(2, 3), prior="uniform", cubes=(4, 12), full_span=True),
    2: dict(rows=(2, 3), cols=(3, 3), cap=3, n_views=2, design="square", edit="cells", edits=(1, 2)),
    3: dict(rows=(3, 3), cols=(3, 4), cap=3, n_views=3, design="tetra", edit="cells", edits=(1, 2)),
    4: dict(rows=(3, 3), cols=(3, 4), cap=3, n_views=3, design="square", edit="cells", edits=(1, 2)),
    5: dict(rows=(3, 4), cols=(3, 4), cap=4, n_views=3, design="square", edit="cells", edits=(1, 2)),
    6: dict(rows=(4, 4), cols=(4, 4), cap=4, n_views=3, design="square", edit="move"),
    7: dict(rows=(4, 4), cols=(5, 5), cap=5, n_views=3, design="square", edit="move"),
}
TIER_RUNGS = {"easy": (1,), "medium": (2, 3), "hard": (4, 5), "xhard": (6,), "xxhard": (7,)}
TIER_SIZE = {"easy": 588, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}   # easy at 588 px

VIEWS = ("top", "front", "side")
VIEW_PAIRS = [("top", "front"), ("top", "side"), ("front", "side")]
MIN_SURFACE = extra.MIN_SURFACE      # interior px per column top / empty cell
MARGIN = 6                           # option render margin (px)
SUPERSAMPLE = 2
# Face shades: top, FRONT (+j) and SIDE (+i).  Every face pair is >= 37 grey
# levels apart on all four palettes, so a column whose top is seen through a
# small window cannot be removed by a change below the readability floor.
SHADES = (1.0, 0.62, 0.80)
FLOOR_PAD = 0                        # floor grid = exactly the N x M footprint
FLOOR_COLOR = (150, 156, 168)        # floor grid lines
TWIN_AREA, TWIN_CORE = 24, 6         # difference3d thresholds (generator's floor)
ATTEMPTS = 60                        # prior samples per item before giving up
EDIT_PROPOSALS = 160                 # cell-edit proposals per prior sample
POOL = 6                             # kept edits per wanted break set


def rung_for(tier, idx):
    rungs = TIER_RUNGS[tier]
    return rungs[idx % len(rungs)]


# --------------------------------------------------------------------------- #
# Orthographic projections (physical cameras) and reading conventions
# --------------------------------------------------------------------------- #
# SINGLE SOURCE OF TRUTH for the conventions drawn on the panels.
#   top : row 0 of the top view is j=0, the BACK row -> arrow UP labelled BACK
#   side: the right-most column is j=0, the BACK row -> arrow RIGHT labelled BACK
CONVENTIONS = {
    "top": dict(arrow="up", label="BACK"),
    "side": dict(arrow="right", label="BACK"),
}
SIDE_AXIS = "front_row_left_back_row_right"
# Face legend under each iso OPTION: the FRONT face (+j, seen by the front
# view) faces screen lower-LEFT; the SIDE face (+i, seen by the side view)
# faces screen lower-RIGHT.
ISO_FACE_LEGEND = [("FRONT", "sw"), ("SIDE", "se")]


def bars(H, view):
    A = np.asarray(H)
    if view == "front":
        return A.max(axis=0)
    if view == "side":
        return A.max(axis=1)[::-1]
    raise ValueError(view)


def view_grid(H, view, frame):
    if view == "top":
        return (np.asarray(H) > 0).astype(int)
    b = bars(H, view)
    return (np.arange(frame, 0, -1)[:, None] <= b[None, :]).astype(int)


def view_key(H, view):
    """Hashable projection; equal keys <=> identical view panel (frame = max of answer)."""
    if view == "top":
        return tuple(tuple(int(x) for x in row) for row in (np.asarray(H) > 0).astype(int))
    return tuple(int(x) for x in bars(H, view))


def _flip_lr(k):
    return tuple(tuple(r[::-1]) for r in k) if k and isinstance(k[0], tuple) else tuple(k[::-1])


def readings(view, key):
    """Every reading of a shown view a confused solver might adopt (incl. the correct one)."""
    if view == "top":
        ud = tuple(key[::-1])
        return {key, _flip_lr(key), ud, _flip_lr(ud)}
    return {key, _flip_lr(key)}


def keys_of(H, views):
    return tuple(view_key(H, v) for v in views)


def trimmed_key(H, view):
    """Projection with empty edge rows/columns removed: what a solver who misses a
    faint empty edge cell of a view panel (a shifted reading) would compare."""
    if view == "top":
        A = (np.asarray(H) > 0).astype(int)
        r, c = np.flatnonzero(A.any(axis=1)), np.flatnonzero(A.any(axis=0))
        A = A[r[0]:r[-1] + 1, c[0]:c[-1] + 1] if len(r) else A[:0, :0]
        return tuple(tuple(int(x) for x in row) for row in A)
    b = [int(x) for x in bars(H, view)]
    while b and b[0] == 0:
        b.pop(0)
    while b and b[-1] == 0:
        b.pop()
    return tuple(b)


# --------------------------------------------------------------------------- #
# Prior and edits
# --------------------------------------------------------------------------- #
def sample_prior(rng, cfg, footprint=None):
    if footprint is not None:
        N, M = footprint          # easy: drawn once per item in build_lattice
    else:
        N = rng.randint(*cfg["rows"])
        M = rng.randint(*cfg["cols"])
    if cfg.get("prior") == "uniform":
        return uniform_support_map(rng, N, M, cfg["cap"], *cfg["cubes"], full_span=cfg.get("full_span", False))
    for _ in range(100):
        H = extra.diverse_map(rng, N, M, cfg["cap"], holes=True)
        if cfg["cap"] > 2 or 4 <= sum(map(sum, H)) <= 12:
            return H
    return None


def draw_footprint(rng, cfg):
    """One footprint per item from cfg["footprints"]: uniformly, or with integer weights
    cfg["footprint_weights"] (one rng.randrange call either way)."""
    fps = cfg["footprints"]
    w = cfg.get("footprint_weights")
    if not w:
        return fps[rng.randrange(len(fps))]
    r = rng.randrange(sum(w))
    for fp, wk in zip(fps, w):
        if r < wk:
            return fp
        r -= wk
    raise AssertionError("unreachable")


def occlusion_floor(H, j, i):
    """The lowest height cell (j, i) may have so that its top is not covered by the
    nearer columns: the diagonal front neighbour (j+1, i+1) and the lower of the two
    side front neighbours (j, i+1), (j+1, i) (cells off the grid count as 0). This is
    the lower bound extra.diverse_map uses when it builds a map front to back."""
    N, M = len(H), len(H[0])
    right = H[j][i + 1] if i + 1 < M else 0
    down = H[j + 1][i] if j + 1 < N else 0
    diag = H[j + 1][i + 1] if j + 1 < N and i + 1 < M else 0
    return max(diag, min(right, down))


def spans_floor(H):
    """True if the structure touches every edge of its floor: the first and last row and
    the first and last column of the top view each hold at least one column. Then the
    floor is the structure's bounding box, so no other footprint or position shows the
    same structure shifted (empty edge rows/columns are what a 'shifted' copy adds)."""
    occ = np.asarray(H) > 0
    return bool(occ[0].any() and occ[-1].any() and occ[:, 0].any() and occ[:, -1].any())


def uniform_support_map(rng, N, M, cap, lo, hi, full_span=False, tries=200000):
    """A heightmap drawn UNIFORMLY from {H : heights 0..cap, lo <= cubes <= hi, every
    H[j][i] >= occlusion_floor(H, j, i)[, spans_floor(H)]}, i.e. from
    extra.diverse_map(holes=True)'s whole support (which contains every map that passes
    the legibility gate), with equal probability for every map instead of diverse_map's
    sloped-terrain weights.

    Exact rejection sampling: every cell is drawn uniformly from 0..cap, front to back;
    a draw is abandoned as soon as it leaves the set (a cell below its occlusion floor,
    or more than `hi` cubes), so the accepted map is uniform on the set."""
    for _ in range(tries):
        H = [[0] * M for _ in range(N)]
        vol = 0
        ok = True
        for j in reversed(range(N)):
            for i in reversed(range(M)):
                v = rng.randrange(cap + 1)
                vol += v
                if v < occlusion_floor(H, j, i) or vol > hi:
                    ok = False
                    break
                H[j][i] = v
            if not ok:
                break
        if ok and vol >= lo and (not full_span or spans_floor(H)):
            return H
    return None


def apply_edit(H, edit):
    out = [row[:] for row in H]
    for (j, i), value in edit.items():
        out[j][i] = value
    return out


def propose_cells(X, rng, n_changes, cap):
    """{(j, i): new height} on n_changes distinct cells (heights stay in 0..cap)."""
    N, M = len(X), len(X[0])
    cells = [(j, i) for j in range(N) for i in range(M)]
    rng.shuffle(cells)
    edit = {}
    for (j, i) in cells:
        if len(edit) >= n_changes:
            break
        old = X[j][i]
        choices = sorted(c for c in {0, old - 1, old + 1, rng.randint(0, cap)}
                         if 0 <= c <= cap and c != old)
        if choices:
            edit[(j, i)] = rng.choice(choices)
    return edit or None


def all_moves(X, rng, cap):
    """Every one-cube move a -> b (volume preserved), shuffled."""
    cells = [(j, i) for j in range(len(X)) for i in range(len(X[0]))]
    moves = [{a: X[a[0]][a[1]] - 1, b: X[b[0]][b[1]] + 1}
             for a in cells for b in cells
             if a != b and X[a[0]][a[1]] > 0 and X[b[0]][b[1]] < cap]
    rng.shuffle(moves)
    return moves


def broken_views(Y, X, views):
    return frozenset(v for v in views if view_key(Y, v) != view_key(X, v))


# --------------------------------------------------------------------------- #
# Symmetric gates
# --------------------------------------------------------------------------- #
def option_px(size):
    _, slots = layout_slots(size)
    return min(min(x1 - x0, y1 - y0 - LEGEND_H) for x0, y0, x1, y1 in slots) - 6


def render_option(H, px, base, info=None):
    if info is None:
        info = iso.placement(H, px, MARGIN, SUPERSAMPLE)
    return iso.render_fixed(H, px, info, base_rgb=base, draw_floor=True,
                            floor_pad=FLOOR_PAD, floor_color=FLOOR_COLOR, shades=SHADES)


def view_splits(S, views):
    """Per view: the partition of option indices by projection (sorted tuple of tuples)."""
    out = {}
    for v in views:
        groups = {}
        for k, O in enumerate(S):
            groups.setdefault(view_key(O, v), []).append(k)
        out[v] = tuple(sorted(tuple(g) for g in groups.values()))
    return out


def design_of(splits):
    """Name of the lattice pattern the view partitions form, or None."""
    if any(sorted(map(len, p)) != [2, 2] for p in splits.values()):
        return None
    distinct = len(set(splits.values()))
    if len(splits) == 2:
        return "square" if distinct == 2 else None
    return {3: "tetra", 2: "square"}.get(distinct)


def twins(O, hcap):
    """Single-column misreadings: one cell set to another height, or one column
    shifted one step along the viewing diagonal with its height +-1."""
    N, M = len(O), len(O[0])
    for j in range(N):
        for i in range(M):
            for v in range(hcap + 1):
                if v != O[j][i]:
                    X = [r[:] for r in O]
                    X[j][i] = v
                    yield X
    for j in range(N):
        for i in range(M):
            h = O[j][i]
            if h == 0:
                continue
            for dj, di in ((1, 1), (-1, -1)):
                tj, ti = j + dj, i + di
                if not (0 <= tj < N and 0 <= ti < M) or O[tj][ti] != 0:
                    continue
                for nh in (h - 1, h + 1):
                    if 1 <= nh <= hcap:
                        X = [r[:] for r in O]
                        X[j][i] = 0
                        X[tj][ti] = nh
                        yield X


class Gates:
    """Caches per-option work while one item's candidate lattices are tried."""

    def __init__(self, views, px, base, cap, full_span=False):
        self.views, self.px, self.base, self.cap = views, px, base, cap
        self.full_span = full_span
        self._readable = {}
        self._render = {}
        self._twin_ok = {}

    def readable(self, O):
        k = str(O)
        if k not in self._readable:
            self._readable[k] = extra.readable(O, self.px, margin=MARGIN)
        return self._readable[k]

    def render(self, O):
        k = str(O)
        if k not in self._render:
            info = iso.placement(O, self.px, MARGIN, SUPERSAMPLE)
            self._render[k] = (render_option(O, self.px, self.base, info), info)
        return self._render[k]

    def twins_visible(self, O):
        k = str(O)
        if k not in self._twin_ok:
            own = keys_of(O, self.views)
            ref, info = self.render(O)
            ok = True
            for T in twins(O, self.cap + 1):
                if keys_of(T, self.views) == own:
                    continue
                img = iso.render_fixed(T, self.px, info, base_rgb=self.base, draw_floor=True,
                                       floor_pad=FLOOR_PAD, floor_color=FLOOR_COLOR, shades=SHADES)
                if not extra.difference3d(ref, img, area=TWIN_AREA, core=TWIN_CORE)["readable"]:
                    ok = False
                    break
            self._twin_ok[k] = ok
        return self._twin_ok[k]

    def check(self, S, design):
        views = self.views
        if len({str(O) for O in S}) != 4:
            return "options not distinct"
        if any(max(map(max, O)) < 2 or sum(map(sum, O)) < 3 for O in S):
            return "option too flat"
        if self.full_span and not all(spans_floor(O) for O in S):
            return "option leaves an edge row or column of the floor empty"
        splits = view_splits(S, views)
        got = design_of(splits)
        if got != design:
            return "view partition pattern"
        keys = [keys_of(O, views) for O in S]
        for a, b in combinations(range(4), 2):
            if all(keys[a][n] in readings(v, keys[b][n]) for n, v in enumerate(views)):
                return "matches another option under a misreading"
        trimmed = [tuple(trimmed_key(O, v) for v in views) for O in S]
        if len(set(trimmed)) != 4:
            return "matches another option under a shifted reading"
        if not all(self.readable(O) for O in S):
            return "hidden column top or floor cell"
        renders = [self.render(O)[0] for O in S]
        for a, b in combinations(range(4), 2):
            if not extra.difference3d(renders[a], renders[b], area=TWIN_AREA, core=TWIN_CORE)["readable"]:
                return "two option renders nearly identical"
        # Every single-column misreading of an option that changes what the
        # option shows in some shown view must be readably visible: otherwise a
        # solver could misread the answer as inconsistent or a distractor as
        # consistent. Checked for every option, i.e. for every possible answer.
        if not all(self.twins_visible(O) for O in S):
            return "near-invisible single-column change alters a view"
        return None


# --------------------------------------------------------------------------- #
# Lattice search
# --------------------------------------------------------------------------- #
def edit_pools(X, rng, cfg, views, wanted):
    """Edits of X bucketed by the set of shown views they change (only `wanted` sets)."""
    pools = {w: [] for w in wanted}
    if cfg["edit"] == "move":
        source = all_moves(X, rng, cfg["cap"])
    else:
        source = (propose_cells(X, rng, rng.randint(*cfg["edits"]), cfg["cap"])
                  for _ in range(EDIT_PROPOSALS))
    seen = set()
    for edit in source:
        if edit is None:
            continue
        key = tuple(sorted(edit.items()))
        if key in seen:
            continue
        seen.add(key)
        bs = broken_views(apply_edit(X, edit), X, views)
        if bs in pools and len(pools[bs]) < POOL:
            pools[bs].append(edit)
            if all(len(p) >= POOL for p in pools.values()):
                break
    return pools


def lattice_candidates(X, rng, cfg, views):
    """Yield (option list, design) lattices built on X; the gates decide."""
    design = cfg["design"]
    if design == "tetra":
        singles = [frozenset([v]) for v in views]
        pools = edit_pools(X, rng, cfg, views, singles)
        for ea in pools[singles[0]]:
            for eb in pools[singles[1]]:
                if set(ea) & set(eb):
                    continue
                for ec in pools[singles[2]]:
                    if set(ec) & (set(ea) | set(eb)):
                        continue
                    yield [X, apply_edit(apply_edit(X, ea), eb),
                           apply_edit(apply_edit(X, ea), ec),
                           apply_edit(apply_edit(X, eb), ec)]
        return
    if len(views) == 2:
        ua, ub = frozenset([views[0]]), frozenset([views[1]])
    else:
        u = views[rng.randrange(3)]
        ua, ub = frozenset([u]), frozenset(v for v in views if v != u)
    pools = edit_pools(X, rng, cfg, views, [ua, ub])
    for ea in pools[ua]:
        for eb in pools[ub]:
            if set(ea) & set(eb):
                continue
            A = apply_edit(X, ea)
            yield [X, A, apply_edit(X, eb), apply_edit(A, eb)]


def build_lattice(rng, cfg, px, base, reasons):
    # The shown views are drawn ONCE per item, before any search (redrawing the
    # pair on every prior sample would skew the mix, because front+side lattices
    # fail the gates more often).
    views = (list(rng.choice(VIEW_PAIRS)) if cfg["n_views"] == 2 else list(VIEWS))
    # easy: the footprint, like the view pair, is drawn once per item before the
    # search (weights cfg["footprint_weights"]), so the footprints follow the weights.
    # Other rungs draw nothing here.
    footprint = draw_footprint(rng, cfg) if "footprints" in cfg else None
    gates = Gates(views, px, base, cfg["cap"], full_span=cfg.get("full_span", False))
    for _attempt in range(ATTEMPTS):
        X = sample_prior(rng, cfg, footprint)
        if X is None:
            reasons["prior proposal"] += 1
            continue
        if not gates.readable(X):
            reasons["prior sample not legible"] += 1
            continue
        tried = 0
        for S in lattice_candidates(X, rng, cfg, views):
            tried += 1
            why = gates.check(S, cfg["design"])
            if why is None:
                return S, views
            reasons[why] += 1
            if tried >= 40:
                break
        if not tried:
            reasons["no edit pair"] += 1
    raise RuntimeError("no exchangeable lattice: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common(3)))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
PALETTES = [(205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160)]
VIEW_LABEL = {"top": "TOP VIEW", "front": "FRONT VIEW", "side": "SIDE VIEW"}
VIEW_FILL = (82, 100, 122)
ARROW_COLOR = (200, 60, 30)
LEGEND_H = 20  # strip reserved under each iso render for the face legend


def draw_convention_arrow(d, view, panel_box, fsmall):
    """Draw the CONVENTIONS arrow for this view OUTSIDE panel_box; return its boxes.
    up   : vertical arrow left of the panel, head at top, label above the head.
    right: horizontal arrow under the panel, head at right, label under the head.
    """
    conv = CONVENTIONS.get(view)
    if conv is None:
        return None
    x0, y0, x1, y1 = panel_box
    boxes = []
    tw = d.textlength(conv["label"], font=fsmall)
    if conv["arrow"] == "up":
        ax = x0 - 10
        d.line([(ax, y1), (ax, y0 + 10)], fill=ARROW_COLOR, width=2)
        d.polygon([(ax - 4, y0 + 10), (ax + 4, y0 + 10), (ax, y0 + 2)], fill=ARROW_COLOR)
        boxes.append((ax - 4, y0 + 2, ax + 4, y1))
        tx, ty = max(2, ax + 8 - tw), y0 - 14
        d.text((tx, ty), conv["label"], fill=ARROW_COLOR, font=fsmall)
        boxes.append((tx, ty, tx + tw, ty + 12))
    else:
        ay = y1 + 9
        d.line([(x0, ay), (x1 - 8, ay)], fill=ARROW_COLOR, width=2)
        d.polygon([(x1 - 8, ay - 4), (x1 - 8, ay + 4), (x1, ay)], fill=ARROW_COLOR)
        boxes.append((x0, ay - 4, x1, ay + 4))
        tx = max(x0, x1 - tw)
        d.text((tx, ay + 5), conv["label"], fill=ARROW_COLOR, font=fsmall)
        boxes.append((tx, ay + 5, tx + tw, ay + 17))
    return boxes


def _boxes_disjoint(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def render_grid_panel(g, cell, fill, pad=4):
    R, C = g.shape
    img = Image.new("RGB", (C * cell + 2 * pad, R * cell + 2 * pad), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(R):
        for c in range(C):
            box = [pad + c * cell, pad + r * cell, pad + (c + 1) * cell, pad + (r + 1) * cell]
            if g[r][c]:
                d.rectangle(box, fill=fill, outline=(40, 40, 40), width=2)
            else:
                d.rectangle(box, fill=(255, 255, 255), outline=(220, 220, 224), width=1)
    return img


def layout_slots(size):
    view_zone = int(size * 0.30)
    slot_w = (size - 12) // 2
    slot_h = (size - view_zone - 12) // 2
    slots = []
    for k in range(4):
        r, c = divmod(k, 2)
        x0 = 6 + c * slot_w + (2 if c else 0)
        y0 = view_zone + 4 + r * slot_h + (2 if r else 0)
        slots.append((x0, y0, x0 + slot_w - 4, y0 + slot_h - 4))
    return view_zone, slots


def option_boxes(size):
    """(x, y, px): where each option render is pasted."""
    _, slots = layout_slots(size)
    px = option_px(size)
    return [(x0 + (x1 - x0 - px) // 2, y0 + (y1 - LEGEND_H - y0 - px) // 2, px)
            for x0, y0, x1, y1 in slots]


def draw_face_legend(d, cx, y):
    """`FRONT <lower-left   lower-right> SIDE` centred at cx (from ISO_FACE_LEGEND)."""
    fs = load_bundled_font(11, "sans-bold")
    (lf, df), (ls, ds) = ISO_FACE_LEGEND
    assert (df, ds) == ("sw", "se")
    twf = d.textlength(lf, font=fs)
    tws = d.textlength(ls, font=fs)
    gap, aw = 18, 10
    total = twf + 4 + aw + gap + aw + 4 + tws
    x = cx - total / 2
    d.text((x, y), lf, fill=ARROW_COLOR, font=fs)
    ax = x + twf + 4
    d.line([(ax + aw, y + 2), (ax + 2, y + 10)], fill=ARROW_COLOR, width=2)
    d.polygon([(ax, y + 12), (ax + 7, y + 10), (ax + 2, y + 5)], fill=ARROW_COLOR)
    ax2 = ax + aw + gap
    d.line([(ax2, y + 2), (ax2 + aw - 2, y + 10)], fill=ARROW_COLOR, width=2)
    d.polygon([(ax2 + aw, y + 12), (ax2 + aw - 7, y + 10), (ax2 + aw - 2, y + 5)], fill=ARROW_COLOR)
    d.text((ax2 + aw + 4, y), ls, fill=ARROW_COLOR, font=fs)


def compose(views, shown, options, size, base):
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    view_zone, slots = layout_slots(size)
    f = load_bundled_font(14, "sans-bold")
    fsmall = load_bundled_font(11, "sans-bold")
    vslot = (size - 16) // len(views)
    panel_boxes, arrow_boxes = {}, {}
    for k, v in enumerate(views):
        g = shown[v]
        # margins reserved for the arrows: 20px left, 24px bottom, 16px top
        cell = max(10, min((vslot - 56) // g.shape[1], (view_zone - 76) // g.shape[0]))
        panel = render_grid_panel(g, cell, VIEW_FILL)
        x0 = 8 + k * vslot
        label = VIEW_LABEL[v]
        tw = d.textlength(label, font=f)
        d.text((x0 + (vslot - tw) / 2, 4), label, fill=(60, 60, 66), font=f)
        px = x0 + (vslot - panel.width) // 2
        py = 38
        canvas.paste(panel, (px, py))
        pbox = (px, py, px + panel.width, py + panel.height)
        panel_boxes[v] = list(pbox)
        aboxes = draw_convention_arrow(d, v, pbox, fsmall)
        if aboxes:
            for ab in aboxes:
                assert _boxes_disjoint(ab, pbox), f"convention arrow overlaps {v} panel"
                assert ab[3] < view_zone, f"convention arrow of {v} leaves the view zone"
            arrow_boxes[v] = [[round(x) for x in ab] for ab in aboxes]
    d.line([(8, view_zone), (size - 8, view_zone)], fill=(230, 230, 234), width=2)

    fl = load_bundled_font(18, "sans-bold")
    boxes = option_boxes(size)
    for k, opt in enumerate(options):
        x0, y0, x1, y1 = slots[k]
        ox, oy, px = boxes[k]
        canvas.paste(render_option(opt, px, base), (ox, oy))
        d.rounded_rectangle([x0, y0, x1, y1], radius=8, outline=(225, 225, 230), width=2)
        d.text((x0 + 7, y0 + 4), "ABCD"[k], fill=(30, 30, 30), font=fl)
        draw_face_legend(d, (x0 + x1) // 2, y1 - LEGEND_H + 4)
    return canvas, {"panel_boxes": panel_boxes, "arrow_boxes": arrow_boxes,
                    "option_boxes": [list(b) for b in boxes]}


# --------------------------------------------------------------------------- #
# Question
# --------------------------------------------------------------------------- #
# The question states every convention in few words, so that every question
# is at most 240 tokens (LLaVA/vicuna incl. BOS and PaliGemma; the three-view text
# is the longest of the four possible strings). The question depends only on the
# shown views, so it never changes the RNG stream, the image or the gold.
VIEW_PARTS = {"top": "top view from above with the back row at the top",
                 "front": "front view from the front",
                 "side": "side view from the right with the front row on the left"}
# The isometric pose of the options, stated in words (as well as by the FRONT/SIDE
# legend drawn under each option): the +j (FRONT) face points to
# screen lower-left and the +i (SIDE) face to screen lower-right.

# The question names the choice set only as the range "A-D" and gives no example
# answer letter (a single-letter example would equal the gold in 1/4 of items).


def build_question(views):
    shown = [VIEW_PARTS[v] for v in views]
    listed = ", ".join(shown[:-1]) + (", and " if len(shown) == 3 else " and ") + shown[-1]
    grid = "" if "top" in views else " (columns from the front view, rows from the side view)"
    which = "all three views" if len(views) == 3 else "both views"
    return (f"The top row shows the {listed}; a square is filled if any cube is in its line of sight, floor at the "
            f"bottom. Options A-D show cube structures on the full floor grid{grid}, FRONT faces lower left and SIDE "
            f"faces lower right. Which option matches {which}? Answer with its letter.")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def make_item(rng, tier, rung, size=None):
    """One question from `rng`. Returns (record, image, reasons)."""
    size = size or TIER_SIZE[tier]
    cfg = RUNGS[rung]
    px = option_px(size)
    base = PALETTES[0] if cfg["edit"] == "move" else PALETTES[rng.randrange(len(PALETTES))]
    reasons = Counter()
    S, views = build_lattice(rng, cfg, px, base, reasons)

    # The answer is drawn uniformly from the accepted set (every gate above was
    # evaluated for every possible answer), then the options are shuffled.
    truth = S[rng.randrange(4)]
    order = list(range(4))
    rng.shuffle(order)
    options = [S[o] for o in order]
    correct_index = next(k for k, O in enumerate(options) if O == truth)
    hmax = max(map(max, truth))
    shown = {v: view_grid(truth, v, hmax) for v in views}
    breaks = [[v for v in views if view_key(O, v) != view_key(truth, v)] for O in options]

    img, layout = compose(views, shown, options, size, base)
    letter = "ABCD"[correct_index]
    others = ", ".join(f"{'ABCD'[k]} fails {'+'.join(b)}" for k, b in enumerate(breaks) if b)
    cot = (f"Projecting every option onto the shown views ({', '.join(views)}): only {letter} "
           f"reproduces all of them; {others}.")
    surfaces = [extra.surface_counts(O, px, margin=MARGIN) for O in options]
    meta = dict(
        generator_version=GENERATOR_VERSION,
        tier=tier,
        rung=f"L{rung}",
        design=cfg["design"],
        edit_kind=cfg["edit"],
        grid=[len(truth), len(truth[0])],
        height_cap=cfg["cap"],
        resolution=size,
        heightmap=truth,
        hmax=hmax,
        views=views,
        view_grids={v: shown[v].tolist() for v in views},
        option_heightmaps=options,
        option_breaks=breaks,
        correct_index=correct_index,
        palette=list(base),
        shades=list(SHADES),
        floor_pad=FLOOR_PAD,
        floor_color=list(FLOOR_COLOR),
        option_size=px,
        option_margin=MARGIN,
        supersample=SUPERSAMPLE,
        min_surface_pixels=MIN_SURFACE,
        option_min_surface=[int(min(map(min, s))) for s in surfaces],
        conventions={v: CONVENTIONS[v] for v in views if v in CONVENTIONS},
        side_axis=SIDE_AXIS,
        view_partitions={v: [list(g) for g in p] for v, p in view_splits(options, views).items()},
        **layout,
    )
    rec = dict(
        task_type="spatial_perception",
        subtype="view_reconstruction",
        level="high",
        difficulty=tier,
        question=build_question(views),
        ans_type="choice",
        options=["A", "B", "C", "D"],
        answer=letter,
        cot=cot,
        meta=meta,
    )
    return rec, img, reasons
