"""Base generator of three_d_views, in the style of BabyVision's "3D Views" task.

LEVEL = high (3D mental model). Show an isometric stack of unit cubes and ask
which of four option panels (A-D) is its orthographic view (front / top / side).
You must build the 3D model from the iso drawing and project it onto a plane,
then match against rendered candidate grids -- the core high-level 3D skill.

`task_type=spatial_perception`, `subtype=3d_views`, `level=high`,
`ans_type=choice`, answer = the correct letter (A/B/C/D). Options are rendered
INTO the image.

Scene & projection
------------------
Gravity-filled heightmap H[j][i] over N(depth rows) x M(width cols) base, drawn
with the shared 2:1 isometric renderer (shared/drawing/isometric.py): increasing i -> front-right,
increasing j -> front-left, height k -> up. An arrow on the image marks the view
direction asked.

Orthographic projections use physical camera directions:
  top   : footprint H[j][i] > 0; rows run back to front, columns left to right.
  front : camera at +j looking along -j; bars max_j H[j][i], increasing i.
  side  : camera at +i looking along -i; bars max_i H[j][i], DECREASING j.

Every column top and every empty floor cell must have a visible interior patch
at the actual final figure size (259 px inside a 448 px question). This prevents
fully hidden cells from changing the answer while leaving the input unchanged.

The four options are the corners of two disjoint edits of a proposal scene
(H0, A, B, A+B), each edit realized in 3D, so every option is the view of a
physically valid scene.  All pairwise-distance sums tie (no unique Hamming
medoid).  The ASKED scene is a corner drawn uniformly at random
after the options are fixed, and every corner scene must pass the readability
gate, so options-only rules (fullest, staircase, reaches the frame top) are at
chance; a lateral frame is as tall as the tallest bar of any option.

Difficulty controls scene size/height and the size of each edit (easy 2-3 cells
or bar 0<->2, medium 1-2, hard 1).  xhard/xxhard use descending terraces with a
random staircase-shaped empty corner and 2/3 one-step corner edits
(terrace_scene / terrace_family, called from xhard.py).

Each problem is a pure function of (seed, idx):  random.Random(f"{seed}:{idx}").
"""
from __future__ import annotations

import itertools
import random

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso
from shared.drawing.fonts import load_bundled_font

GENERATOR_VERSION = "diverse-projections"
FIGURE_FRACTION = 0.58
FIGURE_COLOR = (244, 120, 70)
MIN_SURFACE_PIXELS = 8
VIEW_DIRECTIONS = {"front": (0, -1, 0), "side": (-1, 0, 0), "top": (0, 0, -1)}


def derive_rng(seed, idx):
    return random.Random(f"{seed}:{idx}")


# --------------------------------------------------------------------------- #
# Heightmap
# --------------------------------------------------------------------------- #
DIFFS = {
    "easy":   dict(rows=(3, 4), cols=(4, 4), hmax=2, fill=0.8, ddiff=(2, 3)),
    "medium": dict(rows=(3, 3), cols=(3, 4), hmax=3, fill=0.78, ddiff=(1, 2)),
    "hard":   dict(rows=(3, 4), cols=(3, 4), hmax=4, fill=0.74, ddiff=(1, 1)),
}


def make_heightmap(rng, diff):
    cfg = DIFFS[diff]
    N = rng.randint(*cfg["rows"])
    M = rng.randint(*cfg["cols"])
    hmax = cfg["hmax"]
    if diff == "easy":
        # Wider low stacks offer structural variety without taller cubes.
        for _ in range(100):
            candidate = extra.diverse_map(rng, N, M, hmax, holes=True)
            if 4 <= sum(map(sum, candidate)) <= 14:
                return candidate
        raise RuntimeError("could not propose a small readable heightmap")
    H = [[0] * M for _ in range(N)]
    for j in range(N):
        for i in range(M):
            if rng.random() < cfg["fill"]:
                r = rng.random()
                h = 1 if r < 0.4 else (2 if r < 0.75 else rng.randint(2, hmax))
                H[j][i] = min(h, hmax)
    if sum(sum(r) for r in H) < 2:
        H[0][0] = 2
    return H


# --------------------------------------------------------------------------- #
# Orthographic projections -> binary occupancy grids
# --------------------------------------------------------------------------- #
def proj_top(H):
    """Footprint grid, shape (N rows, M cols). 1 if any cube there."""
    N, M = len(H), len(H[0])
    g = np.zeros((N, M), dtype=int)
    for j in range(N):
        for i in range(M):
            g[j][i] = 1 if H[j][i] > 0 else 0
    return g


def proj_front(H, hmax):
    """Front view, shape (hmax rows top->bottom, M cols).

    column i has a bar of height max_j H[j][i]. Row 0 = top level.
    """
    N, M = len(H), len(H[0])
    bars = [max(H[j][i] for j in range(N)) for i in range(M)]
    g = np.zeros((hmax, M), dtype=int)
    for i in range(M):
        bh = bars[i]
        for lvl in range(bh):
            row = hmax - 1 - lvl   # bottom row = hmax-1
            g[row][i] = 1
    return g


def proj_side(H, hmax):
    """Side view, shape (hmax rows, N cols).

    Viewed from +i looking along -i: j decreases from left to right. Row 0 = top level.
    """
    N, M = len(H), len(H[0])
    bars = [max(H[j][i] for i in range(M)) for j in reversed(range(N))]
    g = np.zeros((hmax, N), dtype=int)
    for j in range(N):
        bh = bars[j]
        for lvl in range(bh):
            row = hmax - 1 - lvl
            g[row][j] = 1
    return g


def project(H, view, hmax):
    if view == "top":
        return proj_top(H)
    if view == "front":
        return proj_front(H, hmax)
    if view == "side":
        return proj_side(H, hmax)
    raise ValueError(view)


def grid_key(g):
    return (g.shape, g.tobytes())


def crop_to_ink(g):
    """The filled part of a grid, without its empty border rows and columns."""
    g = np.asarray(g)
    rows, cols = np.flatnonzero(g.any(axis=1)), np.flatnonzero(g.any(axis=0))
    if not len(rows):
        return g[:0, :0]
    return g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def translation_alias(truth, grid):
    """True when `grid` is the truth's silhouette moved to another place in the frame.

    The figure's floor is drawn one row and one column larger than the base
    (iso.draw_grid_floor pad=1) and nothing in the image ties an option frame
    to particular floor cells, so such a distractor shows the same view.
    """
    if np.array_equal(truth, grid):
        return False
    a, b = crop_to_ink(truth), crop_to_ink(grid)
    return a.shape == b.shape and bool(np.array_equal(a, b))


# --------------------------------------------------------------------------- #
# Final-resolution visibility and balanced, physically valid distractors
# --------------------------------------------------------------------------- #
FIGURE_MARGIN = 46


def figure_margin(N, M):
    """Margin (px) between the iso figure and its square, for an N x M base: 46 for every base.  The release
    adapter swaps this hook during easy builds (three_d_views: 38 px on a 4x5 base and 30 px on a 5x5
    base, so their cubes are drawn at the 4x4 size)."""
    return FIGURE_MARGIN


EASY_CUBES = (4, 14)


def easy_cube_range(view):
    """Cube count every corner scene of an easy item must have: 4-14 for every view.  The release adapter swaps
    this hook during easy builds (three_d_views: 8-14 for front/side views)."""
    return EASY_CUBES


def render_figure(H, size):
    return iso.render_stack(H, size=int(size * FIGURE_FRACTION),
                            base_rgb=FIGURE_COLOR, margin=figure_margin(len(H), len(H[0])), supersample=2)


def visible_surfaces(H, size):
    """Count exposed top/floor interiors at the ACTUAL figure resolution.

    Every base cell needs a visible patch: its column top if occupied, otherwise
    its floor square. Hidden floor squares can conceal extra columns, so checking
    only visible cubes is insufficient. Outline pixels and antialias boundaries
    do not count. Placement comes directly from the shared renderer.
    """
    _, info = render_figure(H, size)
    px, ss = int(size * FIGURE_FRACTION), info["ss"]
    origin, hw, hh, ch = (info[k] for k in ("origin", "hw", "hh", "ch"))
    rows, cols = len(H), len(H[0])
    mask = Image.new("I", (px * ss, px * ss), 0)
    d = ImageDraw.Draw(mask)
    for j in range(rows):
        for i in range(cols):
            floor, _, _ = iso.cube_polys(i, j, -1, origin, hw, hh, ch)
            d.polygon(floor, fill=1 + j * cols + i)
    iso.draw_grid_floor(d, rows, cols, origin, hw, hh, ch, color=0, w=ss, pad=1)
    cells = sorted(((i, j) for j in range(rows) for i in range(cols)),
                   key=lambda q: q[0] + q[1])
    for i, j in cells:
        for k in range(H[j][i]):
            polys = iso.cube_polys(i, j, k, origin, hw, hh, ch)
            for side, poly in enumerate(polys):
                label = 1 + j * cols + i if side == 0 and k == H[j][i] - 1 else 0
                d.polygon(poly, fill=label, outline=0)
            for poly in polys:
                d.line(poly + [poly[0]], fill=0, width=ss)
    arr = np.asarray(mask).reshape(px, ss, px, ss).transpose(0, 2, 1, 3)
    # A final pixel counts only if every supersample and its 3x3 neighborhood
    # belong to the same face. This excludes thin edges exposed by rasterization.
    lo, hi = arr.min(axis=(2, 3)), arr.max(axis=(2, 3))
    ids = np.where(lo == hi, lo, 0)
    padded = np.pad(ids, 1)
    core = ids.copy()
    for dy in range(3):
        for dx in range(3):
            core[padded[dy:dy + px, dx:dx + px] != ids] = 0
    counts = np.bincount(core.ravel(), minlength=rows * cols + 1)
    return counts[1:].reshape(rows, cols).tolist()


# --------------------------------------------------------------------------- #
# Option families whose truth is a uniformly random corner ()
# --------------------------------------------------------------------------- #
# If the truth were always the unedited scene (with the frame at max(H)), options-only rules could
# find it: a bar at max(H) can only be lowered, mostly full footprints mostly lose cells, and the
# xhard/xxhard terrace/diagonal-cut prior is broken by every edit.  The generator instead draws k
# disjoint edits of a proposal scene H0, realizes
# each edit in 3D, and asks about a scene chosen uniformly among the 2^k corner scenes.  The option
# set is fixed before that choice, so no rule that sees only the options can beat 1/2^k (up to the
# verifier's rejections).  Every corner scene must pass the same gates.

def view_lines(shape, view):
    """Base cells (j, i) behind each displayed column of a front/side view, in display order.

    The first cell of a line is its back-left one: the tallest of the line in a descending terrace.
    """
    N, M = shape
    if view == "front":
        return [[(j, i) for j in range(N)] for i in range(M)]
    return [[(j, i) for i in range(M)] for j in reversed(range(N))]


def new_height(rng, diff):
    """Height of a column that a top-view edit adds, from the tier's own height law."""
    hm = DIFFS[diff]["hmax"]
    if diff == "easy":
        return rng.randint(1, hm)
    r = rng.random()
    return min(hm, 1 if r < 0.4 else (2 if r < 0.75 else rng.randint(2, hm)))


def corner_scenes(H0, edits):
    """The 2^k scenes H0 + every subset of the edits (each edit is a list of (cell, new height))."""
    H0 = np.asarray(H0, dtype=int)
    out = []
    for bits in range(1 << len(edits)):
        W = H0.copy()
        for n, edit in enumerate(edits):
            if bits >> n & 1:
                for cell, v in edit:
                    W[cell] = v
        out.append(W)
    return out


def family_grids(scenes, view):
    """Option grids of the corner scenes in one frame: a lateral frame is as tall as the tallest
    bar of ANY option, so no option is told apart by reaching the frame top."""
    frame = max(int(W.max()) for W in scenes)
    return [project(W.tolist(), view, frame) for W in scenes], frame


def family_ok(grids):
    """Non-empty, distinct, and no option is another option shifted inside the frame."""
    if any(not g.any() for g in grids):
        return False
    if len({grid_key(g) for g in grids}) != len(grids):
        return False
    return not any(translation_alias(a, b) for a, b in itertools.combinations(grids, 2))


class SurfaceCache:
    """visible_surfaces per scene, computed once (corner scenes share their single-edit parents)."""

    def __init__(self, size):
        self.size, self.memo = size, {}

    def counts(self, W):
        W = np.asarray(W, dtype=int)
        key = (W.shape, W.tobytes())
        if key not in self.memo:
            self.memo[key] = visible_surfaces(W.tolist(), self.size)
        return self.memo[key]

    def readable(self, W):
        return min(map(min, self.counts(W))) >= MIN_SURFACE_PIXELS


def _first_readable(H0, options, ok):
    """The first candidate edit (in the given random order) whose single-edit scene is readable."""
    for edit in options:
        if ok(corner_scenes(H0, [edit])[1]):
            return edit
    return None


def option_family(H0, view, rng, diff, ok, k=2, tries=12):
    """easy/medium/hard: k disjoint edits of H0, realized in 3D -> (corner scenes, grids, frame).

    top:   each edit toggles lo..hi footprint cells (a cleared cell -> 0, an added cell gets a
           height from the tier's height law);
    front/side: each edit moves one displayed bar by lo..hi levels within 0..tier hmax (a lowered
           bar clips its line of cells, a raised bar raises one cell of the line).
    `ok(scene)` is the readability gate.  Edits are drawn only among those whose own scene passes it
    and every corner scene must pass it, so the gate never depends on which corner is asked.
    """
    H0 = np.asarray(H0, dtype=int)
    lo, hi = DIFFS[diff]["ddiff"]
    hm = DIFFS[diff]["hmax"]
    for _ in range(tries):
        edits = []
        if view == "top":
            cells = list(np.ndindex(H0.shape))
            rng.shuffle(cells)
            singles = ([(c, 0 if H0[c] else new_height(rng, diff))] for c in cells)
            pool = []
            sizes = [rng.randint(lo, hi) for _ in range(k)]
            for single in singles:
                if ok(corner_scenes(H0, [single])[1]):
                    pool.append(single[0])
                    if len(pool) == sum(sizes):
                        break
            else:
                return None
            start = 0
            for n in sizes:
                edits.append(pool[start:start + n])
                start += n
        else:
            lines = view_lines(H0.shape, view)
            bars = [max(int(H0[c]) for c in line) for line in lines]
            order = list(range(len(lines)))
            rng.shuffle(order)
            for c in order:
                options = []
                for v in range(hm + 1):
                    if lo <= abs(bars[c] - v) <= hi:
                        if v < bars[c]:
                            options.append([(cell, v) for cell in lines[c] if H0[cell] > v])
                        else:
                            options += [[(cell, v)] for cell in lines[c]]
                rng.shuffle(options)
                edit = _first_readable(H0, options, ok)
                if edit is not None:
                    edits.append(edit)
                    if len(edits) == k:
                        break
            else:
                return None
        scenes = corner_scenes(H0, edits)
        if not all(ok(W) for W in scenes):
            continue
        grids, frame = family_grids(scenes, view)
        if family_ok(grids):
            return scenes, grids, frame
    return None


# xhard / xxhard terraces ----------------------------------------------------------------------
TERRACE_MAX_HOLES = 8


def terrace_scene(rng, rows, cols, height, max_holes=TERRACE_MAX_HOLES):
    """A descending terrace (extra.stepped_map without holes: every cell >= 1, non-increasing along
    i and j) with a random staircase-shaped empty region grown cell by cell from the front-right
    corner (1..max_holes cells).  stepped_map(holes=True) empties exactly {i + j >= c}, which made
    every xhard/xxhard top-view answer one of two diagonal cuts."""
    H = extra.stepped_map(rng, rows, cols, height, holes=False)
    holes = set()
    for _ in range(rng.randint(1, max_holes)):
        free = [(j, i) for j in range(rows) for i in range(cols) if (j, i) not in holes
                and (j + 1 == rows or (j + 1, i) in holes) and (i + 1 == cols or (j, i + 1) in holes)]
        holes.add(rng.choice(free))
    for j, i in holes:
        H[j][i] = 0
    return H


def is_terrace(W):
    """Heights non-increasing along i and along j (so the footprint is a staircase at the back-left)."""
    W = np.asarray(W)
    return bool((np.diff(W, axis=0) <= 0).all() and (np.diff(W, axis=1) <= 0).all())


def terrace_family(H0, view, rng, k, height, tries=60):
    """xhard/xxhard: k one-step edits at the terrace's corners -> (corner scenes, grids, frame).

    top:   add or clear one footprint cell (an added cell gets a height between 1 and its back and
           left neighbours);
    front/side: move one displayed bar one level (lower: clip the line; raise: raise the line's
           back-left cell, its tallest).
    Every edit, and every combination of edits, must leave a descending terrace, so every option is
    a staircase footprint / monotone bar profile and the truth is not told apart by its shape.
    """
    H0 = np.asarray(H0, dtype=int)
    cands = []
    if view == "top":
        for j, i in np.ndindex(H0.shape):
            if H0[j, i]:
                cands.append([((j, i), 0)])
            else:
                cap = min(int(H0[j - 1, i]) if j else height, int(H0[j, i - 1]) if i else height)
                if cap >= 1:
                    cands.append([((j, i), rng.randint(1, cap))])
    else:
        for line in view_lines(H0.shape, view):
            b = max(int(H0[c]) for c in line)
            if b >= 1:
                cands.append([(c, b - 1) for c in line if H0[c] > b - 1])
            if b + 1 <= height:
                cands.append([(line[0], b + 1)])
    cands = [e for e in cands if is_terrace(corner_scenes(H0, [e])[1])]
    for _ in range(tries):
        rng.shuffle(cands)
        edits, used = [], set()
        for e in cands:
            cells = {c for c, _ in e}
            if cells & used:
                continue
            edits.append(e)
            used |= cells
            if len(edits) == k:
                break
        if len(edits) < k:
            return None
        scenes = corner_scenes(H0, edits)
        if not all(is_terrace(W) for W in scenes):
            continue
        grids, frame = family_grids(scenes, view)
        if family_ok(grids):
            return scenes, grids, frame
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def arrow_segment(view, fig_bbox):
    """Project the SAME viewing axis used by the orthographic projection."""
    x0, y0, x1, y1 = fig_bbox
    if view == "top":
        start = ((x0 + x1) / 2, y0 - 34)
        scale = 0.7
    elif view == "front":
        start = (x0 - 28, y1 + 8)
        scale = 1.2
    else:
        start = (x1 + 28, y1 + 8)
        scale = 1.2
    dx, dy = iso.iso_xy(*VIEW_DIRECTIONS[view], (0, 0))
    return start, (start[0] + scale * dx, start[1] + scale * dy)


def draw_arrow(draw, view, fig_bbox, color=(40, 110, 220)):
    import math
    (ax, ay), (bx, by) = arrow_segment(view, fig_bbox)
    draw.line([(ax, ay), (bx, by)], fill=color, width=5)
    ang = math.atan2(by - ay, bx - ax)
    for da in (math.radians(150), math.radians(-150)):
        hx, hy = bx + 13 * math.cos(ang + da), by + 13 * math.sin(ang + da)
        draw.line([(bx, by), (hx, hy)], fill=color, width=5)
    if view != "top":
        label = "FRONT" if view == "front" else "RIGHT"
        draw.text((ax - 18, ay + 8), label, font=load_bundled_font(10, "sans-bold"), fill=color)
        return dict(text=label, xy=[ax - 18, ay + 8], font_size=10, bold=True)
    return None


def render_grid_panel(g, cell, line=(40, 40, 40), fill=(244, 120, 70),
                      empty=(255, 255, 255), pad=6):
    """Render a binary occupancy grid as filled/empty unit squares."""
    R, C = g.shape
    W = C * cell + 2 * pad
    Hh = R * cell + 2 * pad
    img = Image.new("RGB", (W, Hh), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(R):
        for c in range(C):
            x0 = pad + c * cell
            y0 = pad + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            if g[r][c]:
                d.rectangle([x0, y0, x1, y1], fill=fill, outline=line, width=2)
            else:
                d.rectangle([x0, y0, x1, y1], fill=empty, outline=(220, 220, 224),
                            width=1)
    return img


def compose(H, view, grids, correct_index, size, rng, layout=None):
    """Compose the full question image: iso figure on top, A-D panels below.

    If `layout` is a dict it receives every placement an independent checker
    needs (figure offset, arrow, text labels, option panel boxes).
    """
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    # --- top: iso figure, scaled to upper ~58% ---
    fig_area_h = int(size * FIGURE_FRACTION)
    fig_img, info = render_figure(H, size)
    fx = (size - fig_area_h) // 2
    fy = 6
    canvas.paste(fig_img, (fx, fy))

    # figure bbox in canvas coords (approx from rendered content)
    arr = np.asarray(fig_img)
    nonwhite = np.where((arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250))
    if len(nonwhite[0]):
        fy0 = fy + int(nonwhite[0].min()); fy1 = fy + int(nonwhite[0].max())
        fx0 = fx + int(nonwhite[1].min()); fx1 = fx + int(nonwhite[1].max())
    else:
        fx0, fy0, fx1, fy1 = fx, fy, fx + fig_area_h, fy + fig_area_h
    arrow_label = draw_arrow(d, view, (fx0, fy0, fx1, fy1))
    if view == "top":
        d.text((fx0 - 8, fy1 + 9), "FRONT", fill=(40, 110, 220), font=load_bundled_font(10, "sans-bold"))
        arrow_label = dict(text="FRONT", xy=[fx0 - 8, fy1 + 9], font_size=10, bold=True)
    if layout is not None:
        (ax, ay), (bx, by) = arrow_segment(view, (fx0, fy0, fx1, fy1))
        layout.update(
            size=size,
            figure=dict(offset=[fx, fy], size=fig_area_h, bbox=[fx0, fy0, fx1, fy1],
                        color=list(FIGURE_COLOR), margin=figure_margin(len(H), len(H[0])), supersample=2),
            arrow=dict(segment=[[ax, ay], [bx, by]], color=[40, 110, 220], width=5, head=13),
            text_labels=[arrow_label],
        )

    # --- bottom: 4 option panels ---
    labels = ["A", "B", "C", "D"]
    R, C = grids[0].shape
    panel_zone_top = int(size * 0.60)
    panel_zone_h = size - panel_zone_top - 8
    slot_w = (size - 8) // 4
    # cell size so the largest dim fits in slot with margin
    cell = max(10, min((slot_w - 18) // max(C, 1), (panel_zone_h - 30) // max(R, 1)))

    f = load_bundled_font(18, "sans-bold")
    boxes, label_xy, slots = [], [], []
    for k in range(4):
        panel = render_grid_panel(grids[k], cell)
        sx = 4 + k * slot_w + (slot_w - panel.width) // 2
        sy = panel_zone_top + 22 + (panel_zone_h - 22 - panel.height) // 2
        # label
        lx = 4 + k * slot_w + 6
        d.text((lx, panel_zone_top), labels[k], fill=(30, 30, 30), font=f)
        canvas.paste(panel, (sx, sy))
        boxes.append([sx, sy, sx + panel.width, sy + panel.height])
        label_xy.append([lx, panel_zone_top])
        slots.append([4 + k * slot_w + 2, panel_zone_top + 18, 4 + (k + 1) * slot_w - 4, size - 6])
        # light box around the slot
        d.rounded_rectangle([4 + k * slot_w + 2, panel_zone_top + 18,
                             4 + (k + 1) * slot_w - 4, size - 6],
                            radius=8, outline=(225, 225, 230), width=2)

    # divider
    d.line([(8, panel_zone_top - 6), (size - 8, panel_zone_top - 6)],
           fill=(230, 230, 234), width=2)
    if layout is not None:
        layout["panels"] = dict(boxes=boxes, cell=cell, pad=6, label_xy=label_xy,
                                label_font_size=18, label_color=[30, 30, 30], slots=slots,
                                zone_top=panel_zone_top, divider_y=panel_zone_top - 6)
    return canvas


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def view_question(view, n=4):
    """The question for a view with n options (A.. the n-th letter); one wording for every tier."""
    letters = f"A-{'ABCDEFGH'[n - 1]}"
    if view == "top":
        return (f"Which option {letters} is the top view of this cube figure, looking straight down? Filled squares "
                "are where the figure is seen (the grid may extend past it); the top of each option is the back, "
                "opposite FRONT. Answer with the letter.")
    name = "front" if view == "front" else "right-side"
    return (f"Which option {letters} is the {name} view of this cube figure, in the direction of the arrow? Filled "
            "squares are where the figure is seen (the grid may extend past it); the bottom row is floor level. "
            "Answer with the letter.")


VIEW_Q = {v: view_question(v, 4) for v in ("front", "top", "side")}


def generate_one(seed, idx, size, diff):
    """One easy/medium/hard item: (record, image)."""
    rng = derive_rng(seed, idx)
    views = ["front", "top", "side"]

    view = rng.choice(views)
    for _attempt in range(2000):
        H0 = make_heightmap(rng, diff)
        if max(max(r) for r in H0) < 1:
            continue
        cache = SurfaceCache(size)
        if not cache.readable(H0):
            continue
        if diff == "easy":
            lo_c, hi_c = easy_cube_range(view)

            def ok(W):     # easy's cube range (4-14) holds for every corner scene, like readability
                return lo_c <= int(np.sum(W)) <= hi_c and cache.readable(W)
        else:
            ok = cache.readable
        fam = option_family(H0, view, rng, diff, ok)
        if fam is not None:
            scenes, family, frame = fam
            break
    else:
        raise RuntimeError(f"could not build a readable {diff}/{view} question at size={size}, idx={idx}")
    # every corner scene passed the gates, so which one is asked is a fair coin
    t = rng.randrange(len(scenes))
    H, truth, surfaces = scenes[t].tolist(), family[t], cache.counts(scenes[t])
    order = list(range(len(family)))
    rng.shuffle(order)
    grids = [family[k] for k in order]
    correct = order.index(t)
    hmax = frame if view != "top" else max(map(max, H))

    layout = {}
    img = compose(H, view, grids, correct, size, rng, layout)
    letter = ["A", "B", "C", "D"][correct]

    cot = (f"Build the 3D model from the isometric drawing, then take its {view} "
           f"view. Comparing the four candidate grids, only option {letter} matches "
           f"the true {view} silhouette of the figure; the others differ in at least "
           f"one square. Therefore the answer is {letter}.")

    rec = dict(
        id=f"3d_views_{idx:06d}",
        task_type="spatial_perception",
        subtype="3d_views",
        level="high",
        difficulty=diff,
        question=VIEW_Q[view],
        ans_type="choice",
        options=["A", "B", "C", "D"],
        answer=letter,
        cot=cot,
        meta=dict(
            generator_version=GENERATOR_VERSION,
            view=view,
            viewing_direction=list(VIEW_DIRECTIONS[view]),
            visible_surface_pixels=surfaces,
            min_surface_pixels=MIN_SURFACE_PIXELS,
            figure_size=int(size * FIGURE_FRACTION),
            heightmap=H,
            hmax=hmax,
            correct_index=correct,
            option_grids=[g.tolist() for g in grids],
            true_projection=truth.tolist(),
            base_rows=len(H),
            base_cols=len(H[0]),
            tier_hmax=DIFFS[diff]["hmax"],
            resolution=size,
            layout=layout,
        ),
    )
    return rec, img
