"""Synthetic generator for a **Block Delta** (before/after change counting) task.

LEVEL = high (3D mental simulation). Two isometric cube-stack renders are shown
side by side ("BEFORE" left, "AFTER" right). Cubes were added to and/or removed
from column tops (gravity-respecting). The question asks how many cubes were
added / removed / changed in total; at rung L5 the answer is the two-number
format "added,removed".

Ground truth is exact by construction
-------------------------------------
H2 is produced from H1 by explicit per-column top pushes/pops, so
added = sum(max(0, H2-H1)) and removed = sum(max(0, H1-H2)) are computed from
the same heightmaps that are rendered. Nothing is ever labeled perceptually.

Determinism guard (both heightmaps, exact)
------------------------------------------
The answer must be uniquely readable from the two images, so each rendered
heightmap must be uniquely determined by its pixels:
  1. every non-empty column's TOP cube is visible (cube-ID z-buffer, the exact
     same projection as the render), and
  2. for every EMPTY cell, adding one cube there changes the rendered pixels
     (otherwise a hidden stack could exist there unseen).
Together these rule out any other gravity-filled heightmap rendering to the
same image (a taller stack over a visible top always changes pixels; a hidden
stack on an empty cell is caught by check 2; a hidden taller top is caught by
check 1).

Both checks run on a 224 px fill-only raster, not on the shown panel, so
displayed_readable() repeats the essential part on the delivered raster
(panel (size-12)//2 px, ss 2, outlines, LANCZOS): every column's top face keeps
>= 8 robust interior pixels, and an empty cell whose floor is hidden must show
a probe cube as a visible patch.  The release verifier
(tasks/spatial_transformations/block_delta/verifier.py) additionally decodes both panels from
the PNG and proves the decoded heightmaps unique.

Difficulty ladder (meta.rung; difficulty maps L1->easy, L2-3->medium,
L4-5->hard). Difficulty axis = scene size + number of edits + how much of the
structure is partially occluded (tops stay visible; bodies hide behind fronts).
  L1: small grid, 1-2 cubes ADDED only.        answer = # added
  L2: 2-4 mixed adds/removes.                  answer = # changed in total
  L3: larger scene, 3-5 mixed.                 answer = # changed in total
  L4: dense tall scene (heavy partial occlusion), 3-6 mixed. answer = total
  L5: large scene, 4-7 mixed.                  answer = "added,removed"

Each problem is a pure function of (seed, idx): random.Random(f"{seed}:{idx}").
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso
from shared.drawing.fonts import load_bundled_font


def derive_rng(seed, idx):
    return random.Random(f"{seed}:{idx}")


# --------------------------------------------------------------------------- #
# Exact pixel machinery (same projection as the render)
# --------------------------------------------------------------------------- #
def _camera(H, render_size, ss):
    """iso.placement's camera (margin 46) as (origin, hw, hh, ch, canvas px)."""
    info = iso.placement(H, render_size, supersample=ss)
    return info["origin"], info["hw"], info["hh"], info["ch"], render_size * ss


def id_render(H, render_size=224, ss=1):
    """Rasterize cubes with unique RGB ids; return (id_array, id_of)."""
    N, M = len(H), len(H[0])
    origin, hw, hh, ch, canvas = _camera(H, render_size, ss)
    img = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cubes = [(i, j, k) for j in range(N) for i in range(M) for k in range(H[j][i])]
    id_of = {}
    for n, c in enumerate(cubes, start=1):
        id_of[c] = (n & 255, (n >> 8) & 255, (n >> 16) & 255)
    cubes.sort(key=lambda c: (c[0] + c[1], c[2]))
    for (i, j, k) in cubes:
        for poly in iso.cube_polys(i, j, k, origin, hw, hh, ch):
            draw.polygon(poly, fill=id_of[(i, j, k)])
    arr = np.asarray(img).astype(np.int64)
    ids = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    return ids, id_of


def tops_visible(H):
    """True iff every non-empty column's top cube owns >=1 pixel."""
    ids, id_of = id_render(H)
    present = set(np.unique(ids).tolist())
    for j in range(len(H)):
        for i in range(len(H[0])):
            h = H[j][i]
            if h == 0:
                continue
            col = id_of[(i, j, h - 1)]
            if col[0] + (col[1] << 8) + (col[2] << 16) not in present:
                return False
    return True


def empty_cells_determined(H, render_size=224):
    """For every empty cell, adding one cube must change the rendered pixels.

    Rendered with the placement of the ORIGINAL H so masks are comparable even
    when the probe cube would enlarge the bbox (enlarging always changes pixels,
    and with a fixed placement the new cube simply draws outside the old mask,
    which also registers as a change)."""
    N, M = len(H), len(H[0])
    origin, hw, hh, ch, canvas = _camera(H, render_size, 1)

    def mask(Hx):
        img = Image.new("L", (canvas, canvas), 0)
        draw = ImageDraw.Draw(img)
        cubes = [(i, j, k) for j in range(N) for i in range(M)
                 for k in range(Hx[j][i])]
        cubes.sort(key=lambda c: (c[0] + c[1], c[2]))
        for (i, j, k) in cubes:
            for poly in iso.cube_polys(i, j, k, origin, hw, hh, ch):
                draw.polygon(poly, fill=255)
        return np.asarray(img)

    base = mask(H)
    for j in range(N):
        for i in range(M):
            if H[j][i] != 0:
                continue
            probe = [row[:] for row in H]
            probe[j][i] = 1
            if np.array_equal(mask(probe), base):
                return False
    return True


def uniquely_determined(H):
    return tops_visible(H) and empty_cells_determined(H)


def panel_size(size):
    return (size - 12) // 2


def displayed_readable(H, size):
    """Final-resolution guard for L1-L5, on the raster that is actually shown.

    uniquely_determined() above works on a 224 px fill-only raster with margin
    46 and accepts a top cube visible by one pixel; the shown panel is
    (size-12)//2 px, rendered at ss=2 with margin 24, black outlines and
    LANCZOS.  Here (the xhard/xxhard surface check, ported):
      * every non-empty column's TOP FACE keeps >= extra.MIN_SURFACE robust
        interior pixels in the shown panel, and
      * every empty cell either shows >= MIN_SURFACE robust floor pixels or,
        if its floor is hidden, a one-cube stack placed there would change a
        visible patch (contrast 20, >= 24 px, core >= 6) under every palette.
    """
    panel = panel_size(size)
    info = iso.placement(H, panel, margin=24)
    counts = extra.surface_counts(H, panel, info=info)
    hidden_floor = []
    for j, row in enumerate(H):
        for i, h in enumerate(row):
            if counts[j][i] >= extra.MIN_SURFACE:
                continue
            if h:
                return False
            hidden_floor.append((j, i))
    for base in PALETTES if hidden_floor else ():
        before = iso.render_fixed(H, panel, info, base_rgb=base)
        for j, i in hidden_floor:
            probe = [r[:] for r in H]
            probe[j][i] = 1
            after = iso.render_fixed(probe, panel, info, base_rgb=base)
            if not extra.difference(before, after, contrast=20, area=24, core=6)["readable"]:
                return False
    return True


def placement(H, size, margin=24):
    """The per-panel camera compose() uses for heightmap H (ss canvas units)."""
    return iso.placement(H, panel_size(size), margin=margin)


# --------------------------------------------------------------------------- #
# Scene + edit sampling per rung
# --------------------------------------------------------------------------- #
RUNGS = {
    1: dict(rows=(3, 4), cols=(4, 4), hmax=2, fill=0.7,  ops=(1, 2), kinds=("add",),
            answer="added"),
    2: dict(rows=(3, 3), cols=(3, 3), hmax=3, fill=0.7,  ops=(2, 4), kinds=("add", "remove"),
            answer="total"),
    3: dict(rows=(3, 4), cols=(3, 4), hmax=3, fill=0.72, ops=(3, 5), kinds=("add", "remove"),
            answer="total"),
    4: dict(rows=(3, 4), cols=(4, 4), hmax=4, fill=0.85, ops=(3, 6), kinds=("add", "remove"),
            answer="total", back_tall=True),
    5: dict(rows=(4, 4), cols=(4, 4), hmax=4, fill=0.8,  ops=(4, 7), kinds=("add", "remove"),
            answer="pair", back_tall=True),
}


def sample_heightmap(rng, cfg):
    N = rng.randint(*cfg["rows"])
    M = rng.randint(*cfg["cols"])
    if cfg["hmax"] == 2:
        for _ in range(100):
            candidate = extra.diverse_map(rng, N, M, 2, holes=True)
            if 4 <= sum(map(sum, candidate)) <= 12:
                return candidate
        raise RuntimeError("could not propose a small readable heightmap")
    H = [[0] * M for _ in range(N)]
    hmax = cfg["hmax"]
    grad = cfg.get("back_tall", False)
    for j in range(N):
        for i in range(M):
            if rng.random() >= cfg["fill"]:
                continue
            if grad:
                # taller toward the back (small i+j) so tops stay visible even
                # in dense scenes; noise keeps it irregular
                depth = (i + j) / max(1, (N - 1) + (M - 1))
                h = round(hmax - depth * (hmax - 1)) + rng.choice((-1, 0, 0, 1))
                H[j][i] = max(1, min(hmax, h))
            else:
                r = rng.random()
                H[j][i] = 1 if r < 0.5 else (2 if r < 0.85
                                             else rng.randint(2, hmax))
    if sum(sum(r) for r in H) < 3:
        H[0][0] = 1; H[-1][-1] = 2
    return H


def apply_ops(H1, rng, cfg):
    """Push/pop column tops. Returns (H2, added, removed) or None."""
    N, M = len(H1), len(H1[0])
    H2 = [row[:] for row in H1]
    n_ops = rng.randint(*cfg["ops"])
    if "remove" in cfg["kinds"]:
        n_add = rng.randint(0, n_ops)
        # guarantee at least one change of each present kind when mixed
        if n_ops >= 2:
            n_add = min(max(n_add, 1), n_ops - 1)
    else:
        n_add = n_ops
    n_rem = n_ops - n_add
    for _ in range(n_add):
        cells = [(i, j) for j in range(N) for i in range(M)
                 if H2[j][i] < cfg["hmax"]]
        if not cells:
            return None
        i, j = rng.choice(cells)
        H2[j][i] += 1
    for _ in range(n_rem):
        # only pop columns we haven't grown (keep per-column deltas one-signed
        # so added/removed are exactly the sums of positive/negative deltas)
        cells = [(i, j) for j in range(N) for i in range(M)
                 if H2[j][i] > 0 and H2[j][i] <= H1[j][i]]
        if not cells:
            return None
        i, j = rng.choice(cells)
        H2[j][i] -= 1
    added = sum(max(0, H2[j][i] - H1[j][i]) for j in range(N) for i in range(M))
    removed = sum(max(0, H1[j][i] - H2[j][i]) for j in range(N) for i in range(M))
    if added + removed == 0 or sum(sum(r) for r in H2) == 0:
        return None
    return H2, added, removed


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
PALETTES = [(205, 208, 214), (244, 150, 90), (150, 185, 235), (170, 210, 160)]


def compose(H1, H2, size, base):
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    panel = (size - 12) // 2
    f = load_bundled_font(17, "sans-bold")
    for x, (H, label) in zip((4, 8 + panel), ((H1, "BEFORE"), (H2, "AFTER"))):
        img, _ = iso.render_stack(H, size=panel, base_rgb=base, margin=24,
                                  supersample=2)
        y = (size - panel) // 2 + 10
        canvas.paste(img, (x, y))
        tw = d.textlength(label, font=f)
        d.text((x + (panel - tw) / 2, y - 14), label, fill=(60, 60, 66), font=f)
        d.rounded_rectangle([x, y - 18, x + panel, y + panel - 14], radius=10,
                            outline=(225, 225, 230), width=2)
    d.line([(size // 2, 24), (size // 2, size - 24)], fill=(230, 230, 234),
           width=2)
    return canvas


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
CAMERA_SENTENCE = {
    # L1-L5: compose() frames each panel on its own (render_stack recentres
    # and, for tall scenes, rescales), so the AFTER view can be shifted or
    # zoomed relative to BEFORE.  xhard/xxhard render both with one camera.
    "per_panel": "The AFTER view may be shifted or zoomed, so match stacks by floor cell. ",
    "shared": "Both panels use the same viewpoint and zoom. ",
}


def build_qa(answer_kind, added, removed, camera="per_panel"):
    base_q = ("BEFORE (left) and AFTER (right) show the same unit-cube stacks on a "
              "floor grid; cubes were only added to or removed from stack tops. "
              + CAMERA_SENTENCE[camera]
              + "Compare heights cell by cell (empty cell = 0). ")
    if answer_kind == "added":
        q = base_q + "How many cubes were added? Answer with an integer."
        ans = str(added)
        cot = (f"Comparing each stack's height between the two images, the "
               f"heights that grew account for {added} new cubes in total.")
    elif answer_kind == "total":
        q = base_q + "How many cubes were added or removed in total? Answer with an integer."
        ans = str(added + removed)
        cot = (f"Stack-by-stack comparison shows {added} cubes added and "
               f"{removed} removed, so {added + removed} cubes changed in "
               f"total.")
    else:  # pair
        q = base_q + "How many cubes were added and how many removed? Answer 'added,removed'."
        ans = f"{added},{removed}"
        cot = (f"Stack-by-stack comparison shows {added} cubes were added and "
               f"{removed} were removed.")
    return q, ans, cot


def difficulty_for(rung):
    return {1: "easy", 2: "medium", 3: "medium", 4: "hard", 5: "hard"}[rung]


def generate_one(seed, idx, size, difficulty=None):
    """One L1-L5 item: (record, image)."""
    rng = derive_rng(seed, idx)
    rung = extra.base_rung(idx, difficulty)
    cfg = RUNGS[rung]

    for _attempt in range(2000):
        H1 = sample_heightmap(rng, cfg)
        if not uniquely_determined(H1):
            continue
        res = apply_ops(H1, rng, cfg)
        if res is None:
            continue
        H2, added, removed = res
        if not uniquely_determined(H2):
            continue
        # The guard above is not evaluated on the shown raster; this one is.
        # It uses no randomness, so accepted scenes keep their RNG stream.
        if not (displayed_readable(H1, size) and displayed_readable(H2, size)):
            continue
        break
    else:
        raise RuntimeError(f"could not build a valid problem for idx={idx}")

    q, ans, cot = build_qa(cfg["answer"], added, removed, camera="per_panel")
    base = PALETTES[rng.randrange(len(PALETTES))]
    img = compose(H1, H2, size, base)
    panel = panel_size(size)

    rec = dict(
        id=f"block_delta_{idx:06d}",
        task_type="spatial_perception",
        subtype="block_delta",
        level="high",
        difficulty=difficulty_for(rung),
        question=q,
        ans_type="blank",
        options=[],
        answer=ans,
        cot=cot,
        meta=dict(
            generator_version=extra.VERSION,
            rung=f"L{rung}",
            sampling_attempts=_attempt + 1,
            heightmap_before=H1,
            heightmap_after=H2,
            added=added,
            removed=removed,
            answer_kind=cfg["answer"],
            per_column_delta={f"{i},{j}": H2[j][i] - H1[j][i]
                              for j in range(len(H1))
                              for i in range(len(H1[0]))
                              if H2[j][i] != H1[j][i]},
            # Everything an independent checker needs to re-render and
            # decode the panels: layout, palette and each panel's camera.
            resolution=size,
            panel_size=panel,
            panel_x=[4, 8 + panel],
            panel_y=(size - panel) // 2 + 10,
            palette=list(base),
            camera="per_panel",
            placement_before=placement(H1, size),
            placement_after=placement(H2, size),
        ),
    )
    return rec, img
