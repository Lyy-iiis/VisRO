"""count_3d_blocks: count the unit cubes in an isometric stack (total or minimum).

Generator:
  easy/medium/hard   generator/generate.py  (DIFFS, make_heightmap,
                     cube-ID z-buffer visibility, 448 px)
  xhard/xxhard       generator/xhard.py  (588 px, robust visibility,
                     pixel-exact witness, readable decrements)
both rendered by shared/drawing/isometric.py.

The adapter calls generate.generate_one(seed, index, size, tier) or
xhard.generate(seed, index, size, tier) with the release seed (already derived from task/tier/split/base seed/index), so tiers
and splits never share a random stream.  The generator stores palette, camera,
resolution and minimum_heightmap in meta and states the support rule in the
question.  The adapter adds two gates for the minimum tiers: occlusion >= 1 and
the depth-cue gate (depth_cue_lookalike).
"""
from __future__ import annotations

import os

# The integer programs and numpy are single-threaded workloads; OpenBLAS's spinning
# worker threads otherwise multiply CPU use by ~3x per process on a shared host.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


import numpy as np

from visro import api
from visro.seeding import digest
from shared.drawing import isometric as iso
from tasks.spatial_transformations.count_3d_blocks.generator import generate, xhard

TASK = "count_3d_blocks"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

_DEPTH = ("; rejected if a cheaper figure made by lowering one forced column, or by moving its top one step back and "
          "down, changes < 100 robust px")
TIER_DEFINITIONS = {
    "easy": ("448 px; base 3x4 or 4x4 (rows x cols); heights 0-2 with 4-14 cubes, drawn from every readable map (each "
             "column's top cube, or a cube added on an empty cell, is not covered by nearer columns) with weight "
             "(8/3)^hidden cubes * (4/5)^cubes; every non-empty column's top cube is visible; question 'how many "
             "cubes' (total)"),
    "medium": ("448 px; base 3-4 x 3-4; each cell filled w.p. 0.78 with height 1 (45%), 2 (33%) or 3 (22%); in one random "
               "2x2 block the back cell gets height 1 and its three front neighbours height 3; question 'minimum', "
               "redrawn up to 25 times until minimum < total, and minimum == total is rejected" + _DEPTH),
    "hard": ("448 px; base 3-4 x 3-4; each cell filled w.p. 0.72 with height 1 (45%), 2 (33%) or uniform 3-4 (22%); in one "
             "random 2x2 block the back cell gets height 1-2 and its three front neighbours height 4; question 'minimum', "
             "redrawn up to 25 times until minimum < total, and minimum == total is rejected" + _DEPTH),
    "xhard": ("588 px; base 4 x 5; each cell height uniform 1-5 w.p. 0.86 else 0; question 'minimum' with minimum >= 30 and "
              "total - minimum >= 2; every forced top >= 12 robust px, witness reproduces the image, every forced-top "
              "removal readable (<= 2500 redraws)" + _DEPTH),
    "xxhard": ("588 px; base 5 x 5; each cell height uniform 1-6 w.p. 0.86 else 0; question 'minimum' with minimum >= 45 "
               "and total - minimum >= 4; same forced-top, witness and removal gates as xhard (<= 2500 redraws)" + _DEPTH),
}

SIZES = {"easy": 448, "medium": 448, "hard": 448, "xhard": 588, "xxhard": 588}
VARIANT = {"easy": "total", "medium": "minimum", "hard": "minimum", "xhard": "minimum", "xxhard": "minimum"}
# Depth-cue gate (minimum tiers).  The projection is not true isometric: cube (i,j,k) and cube
# (i+1,j+1,k+1) are drawn 6 px apart, so lowering a column, or moving its top one step back and down,
# can give a cheaper structure that differs from the picture only by a sliver.  Such a candidate is
# rejected here when it changes fewer than DEPTH_CUE_PX robust pixels; the release verifier enforces
# the same 100-px margin exactly (over all structures, by integer programming).
DEPTH_CUE_PX = 100


def _trim(H):
    """Drop empty border rows/columns: the same figure on a padded grid is the same item."""
    rows = [j for j, r in enumerate(H) if any(r)]
    cols = [i for i in range(len(H[0])) if any(r[i] for r in H)]
    if not rows:
        return []
    return [[int(H[j][i]) for i in range(cols[0], cols[-1] + 1)] for j in range(rows[0], rows[-1] + 1)]


def canonical_figure(W):
    """Visible figure up to translation/padding and the left-right mirror (transpose).

    The count is invariant to the mirror i<->j (it swaps the left/right faces of
    the drawing), so a scene and its mirror are the same content.
    """
    a = _trim(W)
    b = _trim([list(r) for r in zip(*W)]) if W else []
    return min(a, b, key=lambda m: (len(m), len(m[0]) if m else 0, m))


def core_key(W) -> str:
    return digest([TASK, canonical_figure(W)])


def _cheaper_moves(W):
    """Cheaper structures one step from the witness: one column lowered, or its top moved one step back
    and down (column (i,j) of height h -> column (i-1,j-1) raised to h-1, column (i,j) lowered)."""
    N, M = len(W), len(W[0])
    for j in range(N):
        for i in range(M):
            h = W[j][i]
            for lo in range(h):
                X = [r[:] for r in W]
                X[j][i] = lo
                yield X
            if h >= 2 and i >= 1 and j >= 1 and W[j - 1][i - 1] < h - 1:
                for lo in range(h):
                    X = [r[:] for r in W]
                    X[j][i], X[j - 1][i - 1] = lo, h - 1
                    if sum(map(sum, X)) < sum(map(sum, W)):
                        yield X


def _changed_box(W, X, size, pl):
    """Final-pixel box holding every cube of the columns where X and W differ (nothing else can change)."""
    ss, origin, hw, hh, ch = pl["ss"], tuple(pl["origin"]), pl["hw"], pl["hh"], pl["ch"]
    xs, ys = [], []
    for j in range(len(W)):
        for i in range(len(W[0])):
            if W[j][i] != X[j][i]:
                for k in range(max(W[j][i], X[j][i])):
                    for poly in iso.cube_polys(i, j, k, origin, hw, hh, ch):
                        xs += [q[0] for q in poly]
                        ys += [q[1] for q in poly]
    return (max(0, int(min(xs) // ss) - 3), max(0, int(min(ys) // ss) - 3),
            min(size, int(max(xs) // ss) + 4), min(size, int(max(ys) // ss) + 4))


def depth_cue_lookalike(W, size, placement, palette):
    """(cubes, robust pixels changed) of the closest one-step cheaper structure if it changes fewer than
    DEPTH_CUE_PX robust pixels, else None.  Drawn with the generator's renderer (iso.draw_fixed) and camera.  A
    final pixel is robust when its 3x3 neighbourhood has one colour in the witness image; it counts as
    changed when any of its supersampled subpixels changes (so LANCZOS ringing around a moved edge is
    not counted, only real coverage changes)."""
    from PIL import Image
    ss = placement["ss"]
    im = iso.draw_fixed(W, size, placement, base_rgb=tuple(palette))
    ref = np.asarray(im, np.int16).reshape(size, ss, size, ss, 3)
    fin = np.asarray(im.resize((size, size), Image.Resampling.LANCZOS), np.int16)
    pad = np.pad(fin, ((1, 1), (1, 1), (0, 0)), mode="edge")
    robust = np.ones((size, size), bool)
    for dy in range(3):
        for dx in range(3):
            robust &= np.abs(pad[dy:dy + size, dx:dx + size] - fin).max(axis=2) <= 1
    best = None
    for X in _cheaper_moves(W):
        x0, y0, x1, y1 = _changed_box(W, X, size, placement)
        alt = np.asarray(iso.draw_fixed(X, size, placement, base_rgb=tuple(palette)).crop((x0 * ss, y0 * ss, x1 * ss, y1 * ss)), np.int16)
        alt = alt.reshape(y1 - y0, ss, x1 - x0, ss, 3)
        px = int(((alt != ref[y0:y1, :, x0:x1]).any(axis=(1, 3, 4)) & robust[y0:y1, x0:x1]).sum())
        if px < DEPTH_CUE_PX and (best is None or px < best[1]):
            best = (sum(map(sum, X)), px)
    return best


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    size = SIZES[tier]
    if tier in ("xhard", "xxhard"):
        rec, img = xhard.generate(int(seed), int(index), size, tier)
    else:
        rec, img = generate.generate_one(int(seed), int(index), size, tier)
    m = dict(rec["meta"])
    if rec["difficulty"] != tier:
        raise api.Reject(f"generator returned tier {rec['difficulty']}")
    if m["variant"] != VARIANT[tier]:
        raise api.Reject(f"{tier} produced a {m['variant']} question")
    if img.size != (size, size) or m.get("resolution") != size:
        raise api.Reject(f"image size {img.size} != {size}")
    W = m["minimum_heightmap"]
    if sum(map(sum, W)) != int(rec["answer"]):
        raise api.Reject("answer is not the witness sum")
    if VARIANT[tier] == "minimum" and int(rec["answer"]) >= m["total_cubes"]:
        # the generator only prefers occlusion for 25 redraws; the tier definition requires it
        raise api.Reject("minimum == total: no hidden cube")
    if VARIANT[tier] == "minimum":
        alt = depth_cue_lookalike(W, size, m["placement"], m["palette"])
        if alt:
            raise api.Reject(f"depth cue: {alt[0]} cubes look the same up to {alt[1]} robust px")
    meta = dict(m, tier=tier, cot=rec.get("cot", ""),
                generator=("generator/xhard.py:generate" if tier in ("xhard", "xxhard")
                           else "generator/generate.py:generate_one"),
                canonical_figure=canonical_figure(W))
    return api.Item(question=rec["question"], answer=str(rec["answer"]), image=img.convert("RGB"), meta=meta,
                    core_key=core_key(W), answer_type="int")
