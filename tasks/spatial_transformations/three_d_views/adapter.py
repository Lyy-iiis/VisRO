"""three_d_views: pick the orthographic front / right-side / top view of an isometric cube stack.

Generator:
  easy/medium/hard   generator/generate.py  (DIFFS, make_heightmap, option_family,
                     final-resolution visibility guard, 448 px, 4 options)
  xhard/xxhard       generator/xhard.py  (terrace_scene / terrace_family in generate.py,
                     2^k option corners, 588 px, 4 / 8 options)
both rendered by shared/drawing/isometric.py.

The adapter calls generate.generate_one(seed, index, size, tier) or xhard.generate(seed, index, size,
tier) with the release seed (already derived from task/tier/split/base seed/index), so tiers and splits
never share a random stream.

  * Options: a proposal scene H0 gets k disjoint edits, each realized in 3D (option_family /
    terrace_family); the options are the 2^k corner views and the ASKED scene is a corner chosen
    uniformly at random.  Every corner scene must pass the same gates, so the option set carries no
    information about which option is the truth.  The lateral option frame is the tallest bar of
    any option.
  * xhard/xxhard terraces get a random staircase-shaped empty region (terrace_scene), and their
    edits sit at the terrace's corners, so every option is a staircase footprint / monotone profile.
  * The question names the option range (A-D or A-H) and states what an option panel encodes; the
    verifier proves uniqueness for cubes standing on the drawn floor grid.
  * option_family / terrace_family redraw the edits when one option is another's silhouette
    translated inside the frame.
  * compose / xhard.generate record the full layout (figure offset, arrow, text labels,
    panel boxes, cell size, letter positions) in meta so the verifier can redraw the image.
The remaining uniqueness gates (every distractor, and every single-cell change of the asked view,
must be refuted by a robust visual difference, not by a <= 30-level top/right face shade swap) are
enforced by the independent verifier.

Easy proposals (_easy_proposal), per view class (top; lateral = front and right-side views):
  * the proposal is drawn from the complete support of diverse_map(holes=True) on the class's base
    (easy_support: every cell at least as tall as its diagonal front neighbour and as the lower of
    its two front neighbours), which is exactly the set of readable 0-2 heightmaps;
  * top views: base 4x4, 3x5 or 4x5; within a cube count weight
    0.7^(2-stacks + partly hidden tops) * 0.1^(floor pieces - 1); every corner scene has 4-14 cubes;
  * front/side views: base 5x5; cube count uniform over 8-14; within a cube count weight
    W^(floor pieces - 1) with W = 1/5; every corner scene has 8-14 cubes;
  * a 4x5 figure is drawn with a 38 px margin and a 5x5 figure with 30 px instead of 46
    (generate.figure_margin), so every easy figure has the 4x4 cube size.
The adapter runs generate.generate_one with make_heightmap, figure_margin and easy_cube_range
swapped for the easy proposal, margin and per-class cube range for the duration of one easy build.
"""
from __future__ import annotations

import bisect
import contextlib
import itertools

import numpy as np

from visro import api
from visro.seeding import digest

from tasks.spatial_transformations.three_d_views.generator import generate, xhard

TASK = "three_d_views"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

# --------------------------------------------------------------------------- easy proposal
EASY_HMAX = 2
EASY_CUBES = (4, 14)        # cube counts easy_support enumerates; every class's range lies inside
# Per view class ("lateral" = front and right-side views, which share core_key's lateral class):
#   EASY_BASES      (rows N, cols M, weight) of the proposal's base;
#   EASY_CUBE_RANGE cube count every corner scene must have (swapped in for generate.easy_cube_range);
#   EASY_CUBE_LAW   weight of each cube count 4, 5, ..., 14 of the proposal, per base;
#   EASY_TILT       (num, den): weight (num/den)^(2-stacks + partly hidden tops) within a cube count;
#   EASY_COMPACT    (num, den): weight (num/den)^(4-connected floor pieces - 1) within a cube count.
# Top views keep diverse_map's cube law of the base (estimated on 10^6 proposals per base,
# SE <= 4 per 10,000; 4x5 uses the 4x4 row).  Front/side views use a
# 5x5 base, 8-14 cubes in every corner scene and a flat cube law over 8-14: smaller lateral corner scenes and
# narrower bases fall onto a few thousand recurring shapes (see the module doc).
EASY_BASES = {"top": ((4, 4, 1), (3, 5, 1), (4, 5, 1)), "lateral": ((5, 5, 1),)}
EASY_CUBE_RANGE = {"top": (4, 14), "lateral": (8, 14)}
_CUBE_LAW_44 = (89, 168, 284, 445, 640, 883, 1132, 1377, 1566, 1695, 1721)
EASY_CUBE_LAW = {"top": {(4, 4): _CUBE_LAW_44, (4, 5): _CUBE_LAW_44,
                         (3, 5): (124, 228, 365, 548, 761, 985, 1206, 1386, 1495, 1500, 1403)},
                 "lateral": {(5, 5): (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1)}}
EASY_TILT = {"top": (7, 10), "lateral": (1, 1)}
# W per class: top 1/10 keeps diverse_map's fragmentation of top scenes (mean floor pieces 1.52 vs 1.47);
# front/side 1/5 trades scattered floors for space on the 5x5 base (see the module doc).
EASY_COMPACT = {"top": (1, 10), "lateral": (1, 5)}
# Figure margin (px, generate.figure_margin) per easy base: 46 everywhere except 4x5 (38) and 5x5 (30), so their
# cubes have the 4x4 size (half-width 33 supersampled px; smallest robust visible face median/p10 45/41 px).
EASY_FIGURE_MARGIN = {(4, 5): 38, (5, 5): 30}
_EASY_SUPPORT: dict = {}


def _pieces(H, N: int, M: int) -> int:
    """Number of 4-connected pieces of the occupied floor cells of H (N x M lists)."""
    seen = [[False] * M for _ in range(N)]
    n = 0
    for j in range(N):
        for i in range(M):
            if H[j][i] and not seen[j][i]:
                n += 1
                seen[j][i] = True
                stack = [(j, i)]
                while stack:
                    a, b = stack.pop()
                    for x, y in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= x < N and 0 <= y < M and H[x][y] and not seen[x][y]:
                            seen[x][y] = True
                            stack.append((x, y))
    return n


def easy_support(N: int, M: int) -> dict:
    """{cubes: [((level, pieces), maps), ...]} for every map extra.diverse_map(holes=True) can propose on an
    N x M base with heights 0..EASY_HMAX and EASY_CUBES cubes.  diverse_map fills front to back (j, then i,
    descending) and gives each cell a height >= max(diagonal front neighbour, lower of the front-left and
    front-right neighbours) with positive weight on every such height, so these maps are exactly its
    support.  level = number of 2-stacks + number of partly hidden tops (an occupied cell with a taller
    front-left or front-right neighbour); pieces = number of 4-connected pieces of the occupied floor.
    Each map is bytes(row-major heights), in enumeration order (4x5: 292,937 maps; 5x5: 1,097,239; built once
    per process, about 20 s for 5x5)."""
    key = (int(N), int(M))
    if key not in _EASY_SUPPORT:
        lo_c, hi_c = EASY_CUBES
        H = [[0] * M for _ in range(N)]
        cells = [(j, i) for j in reversed(range(N)) for i in reversed(range(M))]
        out: dict = {}

        def rec(t, s):
            if s > hi_c:
                return
            if t == len(cells):
                if s >= lo_c:
                    level = 0
                    for j in range(N):
                        for i in range(M):
                            h = H[j][i]
                            if h:
                                level += (h == 2) + ((j + 1 < N and H[j + 1][i] > h) or (i + 1 < M and H[j][i + 1] > h))
                    group = (level, _pieces(H, N, M))
                    out.setdefault(s, {}).setdefault(group, []).append(bytes(v for r in H for v in r))
                return
            j, i = cells[t]
            right = H[j][i + 1] if i + 1 < M else 0
            down = H[j + 1][i] if j + 1 < N else 0
            diag = H[j + 1][i + 1] if j + 1 < N and i + 1 < M else 0
            for v in range(max(diag, min(right, down)), EASY_HMAX + 1):
                H[j][i] = v
                rec(t + 1, s + v)
            H[j][i] = 0

        rec(0, 0)
        _EASY_SUPPORT[key] = {c: sorted(groups.items()) for c, groups in out.items()}
    return _EASY_SUPPORT[key]


def _pick(rng, items, weights):
    """items[k] with probability weights[k] / sum (integer weights, exact; zero weights are never picked)."""
    cum = list(itertools.accumulate(int(w) for w in weights))
    return items[bisect.bisect_right(cum, rng.randrange(cum[-1]))]


def easy_class(view: str) -> str:
    return "top" if view == "top" else "lateral"


def easy_heightmap(rng, view: str) -> list:
    """The easy proposal scene for `view`: base (EASY_BASES), then cube count (EASY_CUBE_LAW), then a map of
    that cube count with weight EASY_TILT^level * EASY_COMPACT^(pieces - 1) among the base's easy_support, all per
    view class.  Top views draw exactly as in the law."""
    cls = easy_class(view)
    bases = EASY_BASES[cls]
    N, M = _pick(rng, [(n, m) for n, m, _ in bases], [w for _, _, w in bases])
    support = easy_support(N, M)
    c = _pick(rng, range(EASY_CUBES[0], EASY_CUBES[1] + 1), EASY_CUBE_LAW[cls][(N, M)])
    groups = support[c]
    e_top = max(e for (e, _), _ in groups)
    k_top = max(k for (_, k), _ in groups)
    (p, q), (u, w) = EASY_TILT[cls], EASY_COMPACT[cls]
    maps = _pick(rng, [m for _, m in groups],
                 [len(m) * p ** e * q ** (e_top - e) * u ** (k - 1) * w ** (k_top - k) for (e, k), m in groups])
    b = maps[rng.randrange(len(maps))]
    return [list(b[j * M:(j + 1) * M]) for j in range(N)]


@contextlib.contextmanager
def _easy_proposal(gen, view: str):
    """Run generate.generate_one for one easy item with make_heightmap replaced by easy_heightmap(view),
    figure_margin by EASY_FIGURE_MARGIN (falling back to the stock 46 px) and easy_cube_range by the view
    class's EASY_CUBE_RANGE.

    generate_one draws the view first and then calls make_heightmap(rng, "easy") for every proposal
    attempt; everything else (edits, gates, corner choice, rendering) runs unchanged.  The stock
    functions are put back afterwards, also on an exception."""
    stock = (gen.make_heightmap, gen.figure_margin, gen.easy_cube_range)
    if any(getattr(f, "_easy_1_3_0", False) for f in stock):
        raise RuntimeError("generate.make_heightmap / figure_margin / easy_cube_range is already overridden")

    def make_heightmap(rng, diff):
        if diff != "easy":
            raise RuntimeError(f"easy proposal called for {diff!r}")
        return easy_heightmap(rng, view)

    def figure_margin(N, M):
        return EASY_FIGURE_MARGIN.get((N, M), stock[1](N, M))

    def easy_cube_range(v):
        if v != view:
            raise RuntimeError(f"easy cube range asked for a {v!r} view during a {view!r} build")
        return EASY_CUBE_RANGE[easy_class(view)]

    for f in (make_heightmap, figure_margin, easy_cube_range):
        f._easy_1_3_0 = True
    gen.make_heightmap, gen.figure_margin, gen.easy_cube_range = make_heightmap, figure_margin, easy_cube_range
    try:
        yield
    finally:
        gen.make_heightmap, gen.figure_margin, gen.easy_cube_range = stock

_COMMON = ("the options are fixed first and the asked scene is one of their 2^k corner scenes chosen "
           "uniformly; lateral option frame = tallest bar of any option; gates on EVERY corner scene: every "
           "column top and empty floor cell keeps >= 8 robust interior px; no option is a translate of another; "
           "verifier: every scene the stacking rule allows (solid columns of any height on the drawn grid, base "
           "plus front padding strip) whose view has a distractor's shape, and every single-cell change of the "
           "asked view, renders with a robust difference (contrast >= 32, area >= 48 px, core >= 12 px) under "
           "the figure's own camera")
TIER_DEFINITIONS = {
    "easy": ("448 px (figure 259 px, drawn with a 46 px margin; 38 px on a 4x5 base and 30 px on a 5x5 base so "
             "every cube has the 4x4 size); view uniform over front/right-side/top; proposal scene: heights 0-2, "
             "a map from the support of extra.diverse_map(holes=True) (every cell >= max(diagonal front "
             "neighbour, lower front neighbour)); top views: base 4x4, 3x5 or 4x5 rows x cols (uniform), cube "
             "count from the diverse_map law of the base (4x5: the 4x4 law), then weight 0.7^(2-stacks + "
             "partly hidden tops) * 0.1^(4-connected floor pieces - 1) within that count; front/side views: base "
             "5x5, cube count uniform over 8-14, then weight (1/5)^(4-connected floor pieces - 1) within that "
             "count; 4 options "
             "A-D = the 2^2 corners of 2 disjoint edits of the proposal, realized in 3D: top view - each edit "
             "toggles 2-3 footprint cells (cleared -> 0, added -> height 1-2); front/side - each edit moves one "
             "bar between 0 and 2 (the only 2-3-level moves within heights 0-2; lowering clips the bar's line of "
             "cells, raising raises one cell of it); every corner scene has 4-14 cubes (top views) or 8-14 cubes "
             "(front/side views)"),
    "medium": ("448 px; proposal scene: base 3 x 3-4, each cell filled w.p. 0.78 with height 1 (p .40), 2 "
               "(p .35) else uniform 2-3; view uniform; 4 options A-D = the 2^2 corners of 2 disjoint edits: top "
               "- each toggles 1-2 cells (added height from the same law); front/side - each moves one bar 1-2 "
               "levels within 0-3"),
    "hard": ("448 px; proposal scene: base 3-4 x 3-4, each cell filled w.p. 0.74 with height 1 (p .40), 2 "
             "(p .35) else uniform 2-4; view uniform; 4 options A-D = the 2^2 corners of 2 disjoint edits: top - "
             "each toggles exactly 1 cell (added height from the same law); front/side - each moves one bar "
             "exactly 1 level within 0-4"),
    "xhard": ("588 px (figure 341 px); proposal scene: base 4 x 5 descending terrace (extra.stepped_map hmax 4 "
              "without holes) with a random staircase-shaped empty region of 1-8 cells grown from the front-right "
              "corner; view cycles front/top/side by index; 4 options A-D = the 2^2 corners of 2 disjoint "
              "one-step edits at the terrace's corners (top: add or clear one footprint cell; front/side: move "
              "one bar +-1 level within 0-4); every corner scene is a descending terrace, so every option is a "
              "staircase footprint / monotone bar profile; top views: every option's fill is 25-90%"),
    "xxhard": ("588 px; proposal scene: base 5 x 5 descending terrace (stepped_map hmax 5 without holes) with a "
               "random staircase-shaped empty region of 1-8 cells; view cycles by index; 8 options A-H in 2 rows "
               "of 4 = the 2^3 corners of 3 disjoint one-step corner edits (heights 0-5), every corner scene a "
               "descending terrace; top views: every option's fill is 25-90%"),
}
TIER_DEFINITIONS = {t: f"{d}; {_COMMON}" for t, d in TIER_DEFINITIONS.items()}

SIZES = {"easy": 448, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}


def _trim(H):
    """Drop empty border rows/columns: the same solid on a padded base is the same item."""
    H = np.asarray(H, dtype=int)
    rows = np.flatnonzero(H.any(axis=1))
    cols = np.flatnonzero(H.any(axis=0))
    if not len(rows):
        return []
    return H[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].tolist()


def core_key(heightmap, view) -> str:
    """Content key: the solid (translation-trimmed) and the asked view, option order ignored.

    (H, right-side) is the horizontal mirror image of (H transposed, front): same picture
    mirrored, same answer mirrored.  Both map to ("lateral", trim(H) as seen from the front).
    """
    t = _trim(heightmap)
    if view == "top":
        return digest(["three_d_views", "top", t])
    if view == "side":
        t = np.asarray(t, dtype=int).T.tolist() if t else t
    return digest(["three_d_views", "lateral", t])


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in SIZES:
        raise ValueError(f"unknown tier {tier!r}")
    size = SIZES[tier]
    if tier == "easy":
        # generate_one's first draw is the view (derive_rng(seed, idx).choice(views)); the proposal
        # depends on it, so read it from the same stream before the build and check it afterwards.
        view = generate.derive_rng(int(seed), int(index)).choice(["front", "top", "side"])
        with _easy_proposal(generate, view):
            rec, img = generate.generate_one(int(seed), int(index), size, tier)
        if rec["meta"]["view"] != view:
            raise RuntimeError(f"easy view {rec['meta']['view']!r} != predicted {view!r}")
    elif tier in ("xhard", "xxhard"):
        rec, img = xhard.generate(int(seed), int(index), size, tier)
    else:
        rec, img = generate.generate_one(int(seed), int(index), size, tier)
    m = rec["meta"]
    if rec.get("difficulty") != tier:
        raise api.Reject(f"generator returned tier {rec.get('difficulty')!r}")
    labels = list(rec["options"])
    meta = {
        "view": m["view"],
        "viewing_direction": list(m["viewing_direction"]),
        "heightmap": [[int(v) for v in row] for row in m["heightmap"]],
        "frame_height": int(m["hmax"]),
        "tier_hmax": int(m["tier_hmax"]),
        "option_grids": m["option_grids"],
        "labels": labels,
        "correct_index": int(m["correct_index"]),
        "true_projection": m["true_projection"],
        "resolution": size,
        "layout": m["layout"],
        "visible_surface_pixels": m["visible_surface_pixels"],
        "min_surface_pixels": int(m["min_surface_pixels"]),
        "generator_version": generate.GENERATOR_VERSION,
        "generator": ("generator/xhard.py:generate" if tier in ("xhard", "xxhard")
                      else "generator/generate.py:generate_one" + (" + adapter.py:easy_heightmap"
                                                                   if tier == "easy" else "")),
    }
    if "independent_edits" in m:
        meta["independent_edits"] = int(m["independent_edits"])
    return api.Item(
        question=rec["question"],
        answer=rec["answer"],
        image=img.convert("RGB"),
        meta=meta,
        core_key=core_key(meta["heightmap"], meta["view"]),
        answer_type="choice",
        choices=labels,
    )
