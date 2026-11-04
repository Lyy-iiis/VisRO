"""Base generator of count_3d_blocks, in the style of BabyVision's "Count 3D blocks" task.

LEVEL = high (3D mental model). The bottleneck is building a 3D model from a 2D
isometric drawing and reasoning about occlusion: counting the unit cubes in a
stack where back cubes may be hidden behind taller front ones.

Scene
-----
A gravity-filled heightmap H over an N(depth rows) x M(width cols) base. H[j][i]
is the stack height at base cell (i col, j row). Gravity-filled = a column of
height h occupies levels 0..h-1 with NO floating cubes, so the column holds
exactly h cubes. We render with a classic 2:1 isometric projection (shared/drawing/isometric.py),
3 face tones (top lightest, sides darker), black edges, thin grid floor, drawn
back-to-front (painter's algorithm) so occlusion is physically correct.

Two question variants (exact text: QUESTION_TOTAL / QUESTION_MINIMUM below)
---------------------
Both state the conventions the answer depends on: cubes stand on the squares
of the drawn floor grid, every cube rests directly on the grid or directly on
another cube, and the cubes need not form one connected piece.

total   : "...How many cubes make up this figure?" -> answer = sum(H).
          Used only when EVERY cube column's top is visible, so the count is
          determined (the release verifier additionally proves with an integer
          program that no hidden position on the drawn grid can hold a cube).

minimum : "...What is the minimum number of cubes a figure must contain to look
          exactly like this from this viewpoint?" -> answer = sum of the forced
          heights.  Visibility comes from a cube-ID z-buffer: each cube is
          rasterized back-to-front with a unique id and is "visible" iff it
          owns >= 1 subpixel.  A column's forced height is (highest visible
          cube level + 1) by the support rule, else 0.  This is the generator's
          own rule.  The release adds two gates: the adapter's depth-cue gate
          (tasks/spatial_transformations/count_3d_blocks/adapter.py) and the independent
          verifier (tasks/spatial_transformations/count_3d_blocks/verifier.py), which requires
          every structure with fewer cubes to differ from the image on >= 100
          robust pixels and every forced top to own >= 12 robust pixels.

Difficulty knob (see DIFFS; xhard/xxhard live in xhard.py)
---------------
easy   : base 3-4 rows x 4 cols, max height 2, 4..14 cubes, drawn exactly from
         every readable map, tilted toward hidden cubes (readable_map); every
         column top visible ('total' variant only).
medium : base 3-4 x 3-4, max height 3, a seeded hidden back column ('minimum').
hard   : base 3-4 x 3-4, max height 4, a seeded hidden back column behind a
         height-4 front L ('minimum', heavier occlusion).

Each problem is a pure function of (seed, idx):  random.Random(f"{seed}:{idx}").
The release adapter (tasks/spatial_transformations/count_3d_blocks/adapter.py) passes a seed
already derived from (task, tier, split, base seed, index).
"""
from __future__ import annotations

import functools
import random

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso


def derive_rng(seed, idx):
    return random.Random(f"{seed}:{idx}")


# --------------------------------------------------------------------------- #
# Heightmap generation
# --------------------------------------------------------------------------- #
DIFFS = {
    "easy":   dict(rows=(3, 4), cols=(4, 4), hmax=2, fill=0.85, force_hidden=False),
    "medium": dict(rows=(3, 4), cols=(3, 4), hmax=3, fill=0.78, force_hidden=True),
    "hard":   dict(rows=(3, 4), cols=(3, 4), hmax=4, fill=0.72, force_hidden=True),
}


# Easy proposal: bases 3x4/4x4, heights <= 2, 4-14 cubes and the 'total' question.  extra.diverse_map's
# proposal bound H >= max(diag, min(right, down)) forbids an EMPTY cell behind a height-1 neighbour (although
# a cube added there shows its whole top face) and leaves only 33,352 distinct figures, so easy draws EXACTLY
# from all readable maps (706,996 figures) with
#     P(map) ~ EASY_MU ** hidden * EASY_NU ** cubes,
# a maximum-entropy tilt that matches diverse_map's answer and hidden-cube distribution (mean answer about
# 11.5, mean hidden cubes about 1.5); effective size ~296k figures.
EASY_TOTAL = (4, 14)
EASY_MU = (8, 3)   # weight 8/3 per hidden cube (integer ratio: the sampler is exact)
EASY_NU = (4, 5)   # weight 4/5 per cube


def _readable(v, right, down, diag):
    """A column of height v reads unambiguously given its nearer neighbours (heights, 0 outside the
    grid): its top cube -- for an empty column, a cube that could be added there -- is not covered.
    The diagonal neighbour covers it when taller than max(v, 1); the right and down neighbours
    together cover it when both are."""
    return max(diag, min(right, down)) <= max(v, 1)


def _hidden_bottom(v, right, down, diag):
    """1 when a height-2 column's bottom cube shows no pixel (heights <= 2): a height-2 diagonal
    neighbour covers it, or the right and the down neighbour both do.  Equals the cube-ID z-buffer
    count on readable maps."""
    return int(v == 2 and (diag == 2 or (right >= 1 and down >= 1)))


def _factor(v, right, down, diag, hmax):
    """Integer weight of one cell: EASY_MU per hidden cube, EASY_NU per cube (common denominators)."""
    mu = EASY_MU[0] if _hidden_bottom(v, right, down, diag) else EASY_MU[1]
    return mu * EASY_NU[0] ** v * EASY_NU[1] ** (hmax - v)


def _neighbours(win, j, i, N, M):
    right = win[-1] if i + 1 < M else 0
    down = win[-M] if j + 1 < N else 0
    diag = win[-M - 1] if j + 1 < N and i + 1 < M else 0
    return right, down, diag


@functools.lru_cache(maxsize=None)
def _readable_table(N, M, hmax, lo, hi):
    """Cells front to back (diverse_map's order) and the exact weighted completion count
    count(k, window, s) of cells k.. (totals in [lo, hi]), given the last M + 1 values and total s."""
    if hmax != 2:
        raise ValueError("the hidden-cube rule is exact for heights <= 2 only")
    cells = tuple((j, i) for j in reversed(range(N)) for i in reversed(range(M)))

    @functools.lru_cache(maxsize=None)
    def count(k, win, s):
        if s > hi:
            return 0
        if k == len(cells):
            return 1 if s >= lo else 0
        nb = _neighbours(win, *cells[k], N, M)
        return sum(_factor(v, *nb, hmax) * count(k + 1, win[1:] + (v,), s + v)
                   for v in range(hmax + 1) if _readable(v, *nb))
    return cells, count


def readable_map(rng, shapes, hmax, total):
    """One draw from P(map) ~ EASY_MU**hidden * EASY_NU**cubes over every readable map (_readable) of the
    given (rows, cols) shapes with heights 0..hmax and `total` cubes.  Exact: one rng.randrange over the
    integer total weight, decoded shape first, then cell by cell."""
    pad = EASY_MU[1] * EASY_NU[1] ** hmax          # weight of an empty, non-hidden cell
    kmax = max(n * m for n, m in shapes)
    tabs = []
    for n, m in shapes:
        cells, count = _readable_table(n, m, hmax, *total)
        tabs.append((n, m, cells, count, pad ** (kmax - n * m)))
    r = rng.randrange(sum(t[3](0, (0,) * (t[1] + 1), 0) * t[4] for t in tabs))
    for N, M, cells, count, scale in tabs:
        z = count(0, (0,) * (M + 1), 0) * scale
        if r < z:
            break
        r -= z
    r //= scale
    win, s = (0,) * (M + 1), 0
    H = [[0] * M for _ in range(N)]
    for k, (j, i) in enumerate(cells):
        nb = _neighbours(win, j, i, N, M)
        for v in range(hmax + 1):
            if not _readable(v, *nb):
                continue
            f = _factor(v, *nb, hmax)
            c = f * count(k + 1, win[1:] + (v,), s + v)
            if r < c:
                r //= f
                break
            r -= c
        else:
            raise AssertionError("readable_map: rank outside the table")
        H[j][i] = v
        win, s = win[1:] + (v,), s + v
    return H


def make_heightmap(rng, diff):
    cfg = DIFFS[diff]
    if diff == "easy":
        # every readable 3x4 / 4x4 map with 4..14 cubes, with the EASY_MU / EASY_NU tilt (readable_map)
        shapes = [(n, m) for n in range(cfg["rows"][0], cfg["rows"][1] + 1)
                  for m in range(cfg["cols"][0], cfg["cols"][1] + 1)]
        return readable_map(rng, shapes, cfg["hmax"], EASY_TOTAL)
    N = rng.randint(*cfg["rows"])
    M = rng.randint(*cfg["cols"])
    hmax = cfg["hmax"]
    H = [[0] * M for _ in range(N)]
    for j in range(N):
        for i in range(M):
            if rng.random() < cfg["fill"]:
                # bias toward shorter stacks, occasional tall
                r = rng.random()
                if r < 0.45:
                    h = 1
                elif r < 0.78:
                    h = 2
                else:
                    h = rng.randint(min(3, hmax), hmax)
                H[j][i] = min(h, hmax)
    # ensure non-empty
    if sum(sum(row) for row in H) == 0:
        H[0][0] = 1

    if cfg["force_hidden"]:
        # try to seed a fully-hidden back column: pick a back cell (small i+j)
        # and ensure the diagonal front neighbor is tall enough to cover it.
        # Back cell (i,j); the front-covering cells are (i+1,j),(i,j+1),(i+1,j+1).
        candidates = [(i, j) for j in range(N) for i in range(M)
                      if i + 1 < M and j + 1 < N]
        rng.shuffle(candidates)
        for (i, j) in candidates:
            # set back cell to a modest height and the front-right/left/diag to tall
            back_h = rng.randint(1, max(1, hmax - 2))
            H[j][i] = back_h
            tall = hmax
            H[j][i + 1] = tall
            H[j + 1][i] = tall
            H[j + 1][i + 1] = tall
            break
    return H


# --------------------------------------------------------------------------- #
# Exact visibility via cube-ID z-buffer
# --------------------------------------------------------------------------- #
def visible_cubes(H, render_size=448, ss=2):
    """Return a dict {(i,j): set_of_visible_levels} computed by an ID z-buffer.

    Each cube is rasterized back-to-front with a unique RGB id. After all cubes
    are drawn, any id still present in the buffer = that cube is visible (owns
    at least one pixel). No floor, no antialias, hard edges -> exact pixel test.
    """
    N = len(H)
    M = len(H[0])

    # The camera of render_stack, so visibility matches the rendered image.
    info = iso.placement(H, render_size, margin=46, supersample=ss)
    origin, hw, hh, ch = info["origin"], info["hw"], info["hh"], info["ch"]
    canvas = render_size * ss

    img = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # assign unique ids
    cube_list = []
    for j in range(N):
        for i in range(M):
            for k in range(H[j][i]):
                cube_list.append((i, j, k))
    id_of = {}
    for idx, c in enumerate(cube_list):
        # id index starts at 1; encode in RGB
        n = idx + 1
        id_of[c] = (n & 255, (n >> 8) & 255, (n >> 16) & 255)

    # painter order
    cube_list.sort(key=lambda c: (c[0] + c[1], c[2]))
    for (i, j, k) in cube_list:
        top, left, right = iso.cube_polys(i, j, k, origin, hw, hh, ch)
        col = id_of[(i, j, k)]
        for poly in (top, left, right):
            draw.polygon(poly, fill=col)

    arr = np.asarray(img).astype(np.int64)
    ids = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    present = set(np.unique(ids).tolist())
    present.discard(0)

    vis = {}
    for (i, j, k), col in id_of.items():
        n = col[0] + (col[1] << 8) + (col[2] << 16)
        if n in present:
            vis.setdefault((i, j), set()).add(k)
    return vis


def minimum_count(H, size=448):
    """Provable minimum cubes consistent with the drawing.

    For each column, the minimum forced height = (highest VISIBLE cube level) + 1
    (gravity forces all cubes below a visible one). Fully hidden columns -> 0.
    `size` must be the rendered resolution (the camera scale depends on it).
    """
    vis = visible_cubes(H, render_size=size)
    total_min = 0
    per_col = {}
    for j in range(len(H)):
        for i in range(len(H[0])):
            if (i, j) in vis and vis[(i, j)]:
                forced = max(vis[(i, j)]) + 1
            else:
                forced = 0
            per_col[(i, j)] = forced
            total_min += forced
    return total_min, per_col, vis


def all_columns_visible(H, size=448):
    """True iff every non-empty column's TOP cube is visible -> exact count is
    uniquely determined from the drawing (height of each stack is recoverable).

    This ignores hidden EMPTY cells; the release verifier additionally proves
    with an integer program that no hidden cube can be added anywhere."""
    vis = visible_cubes(H, render_size=size)
    for j in range(len(H)):
        for i in range(len(H[0])):
            h = H[j][i]
            if h == 0:
                continue
            top_level = h - 1
            if (i, j) not in vis or top_level not in vis[(i, j)]:
                return False
    return True


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
PALETTES = [
    (205, 208, 214),   # light gray
    (205, 208, 214),
    (208, 211, 216),
    (200, 203, 210),
]


def render_question(H, size, rng):
    base = rng.choice(PALETTES)
    img, info = iso.render_stack(H, size=size, base_rgb=base, supersample=2)
    # rounded light frame
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([3, 3, size - 4, size - 4], radius=14,
                        outline=(225, 225, 230), width=2)
    info = dict(info, base_rgb=list(base))
    return img, info


QUESTION_TOTAL = (
    'How many unit cubes make up this figure? Every cube rests on the floor grid or on another '
    'cube. Answer with an integer.'
)
QUESTION_MINIMUM = (
    'This figure of unit cubes stands on the floor grid; no cube floats, some may be hidden, and '
    'the cubes need not be connected. What is the fewest cubes that would look exactly like this '
    'from this viewpoint? Answer with an integer.'
)


def build_qa(H, variant, total_min, per_col):
    total = sum(sum(r) for r in H)
    if variant == "total":
        q = QUESTION_TOTAL
        ans = total
        cot = (f"Each column is a solid stack from the floor up (no floating cubes). "
               f"Reading the stack heights and summing them gives {total} cubes.")
    else:
        q = QUESTION_MINIMUM
        ans = total_min
        cot = (f"Counting only the cubes whose presence is forced: in each column, the "
               f"highest visible cube and every cube under it (each cube must rest on "
               f"the grid or on another cube). Any cube above the highest visible one, "
               f"and every column with no visible cube, can be left out without changing "
               f"the picture, so the minimum is {total_min} cubes.")
    return q, str(ans), cot


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_one(seed, idx, size, diff):
    """One easy/medium/hard item: (record, image)."""
    rng = derive_rng(seed, idx)

    # build a heightmap and decide variant; retry to get a clean, valid sample
    for _attempt in range(40):
        H = make_heightmap(rng, diff)
        total = sum(sum(r) for r in H)
        if total < 2:
            continue
        total_min, per_col, vis = minimum_count(H, size)
        full_vis = all_columns_visible(H, size)

        if diff == "easy":
            # exact-count variant: require every column visible
            if not full_vis:
                continue
            variant = "total"
        else:
            # minimum variant: prefer maps where occlusion bites (min < total)
            variant = "minimum"
            if total_min == total and _attempt < 25:
                # no occlusion benefit yet; keep trying for a hidden column
                continue
        break
    else:
        if diff == "easy":
            # Never relabel a hidden-column scene as an easy 'total' question.
            raise RuntimeError("no fully visible easy heightmap in 40 attempts")
        variant = "minimum"

    total_min, per_col, vis = minimum_count(H, size)
    q, ans, cot = build_qa(H, variant, total_min, per_col)

    img, info = render_question(H, size, rng)
    witness = [[per_col[(i, j)] for i in range(len(H[0]))] for j in range(len(H))]

    rec = dict(
        id=f"count_3d_blocks_{idx:06d}",
        task_type="spatial_perception",
        subtype="count_3d_blocks",
        level="high",
        difficulty=diff,
        question=q,
        ans_type="blank",
        options=[],
        answer=ans,
        cot=cot,
        meta=dict(
            generator_version=extra.VERSION,
            resolution=size,
            variant=variant,
            heightmap=H,
            total_cubes=sum(sum(r) for r in H),
            minimum_cubes=total_min,
            per_column_forced={f"{i},{j}": per_col[(i, j)] for (i, j) in per_col},
            base_rows=len(H),
            base_cols=len(H[0]),
            occlusion=sum(sum(r) for r in H) - total_min,
            minimum_heightmap=witness,
            palette=info["base_rgb"],
            placement={k: info[k] for k in ("N", "M", "hw", "hh", "ch", "origin", "ss")},
        ),
    )
    return rec, img
