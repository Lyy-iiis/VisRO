"""viewpoint_rotation xhard / xxhard (rungs L6, L7): larger diverse_map references.

The reference is extra.diverse_map with holes (4 x 5, height <= 3, 20-28 cubes
for xhard; 5 x 5, height <= 4, 30-42 cubes for xxhard), chiral, without
rotational self-symmetry, and every column top / empty floor cell shows
>= extra.MIN_SURFACE interior pixels in the reference panel.  The options are
the truth, one mirror image and two edits (one-cube moves for xhard, column
height swaps for xxhard), each readably different from all four views of the
truth and accepted by generate.ReferenceGuard.  Everything is drawn in
PALETTES[0].

Each problem is a pure function of (seed, idx, tier):
random.Random(f"{seed}:{idx}:{tier}").
"""
from __future__ import annotations

import random

import numpy as np

from shared import extra_difficulty as extra

from .generate import (
    GUARD_VERSION, PALETTES, QUESTION, ReferenceGuard, canon, compose, cube_count,
    has_rot_symmetry, is_chiral, layout_slots, mirror, render_h, rot,
)


def generate(seed, idx, size, tier):
    """One xhard/xxhard item: (record, image)."""
    size = extra.resolve_size(size, tier)
    rng = random.Random(f"{seed}:{idx}:{tier}")
    high = tier == "xxhard"
    nr, nc = (5, 5) if high else (4, 5)
    lo, hi = (30, 42) if high else (20, 28)
    refpx, slots = layout_slots(size)
    px = min(min(x1 - x0, y1 - y0) - 6 for x0, y0, x1, y1 in slots)
    for attempt in range(4000):
        # All source heights are readable. Rotated candidates may occlude
        # columns; reject them against every rendered rotation of the known
        # source, rather than requiring the same face to remain visible at yaw.
        H = extra.diverse_map(rng, nr, nc, 4 if high else 3, holes=True)
        if (
            not lo <= cube_count(H) <= hi
            or not is_chiral(H)
            or has_rot_symmetry(H)
        ):
            continue
        if not extra.readable(H, refpx, margin=16):
            continue
        # Reference-side guard: heights the reference panel cannot pin down
        # must not make any distractor a second answer.
        guard = ReferenceGuard(H, size, PALETTES[0])
        if guard.variants is None:
            continue
        yaw = rng.choice((1, 2, 3))
        truth = rot(H, yaw)
        all_truth = [
            render_h(rot(H, k), px, PALETTES[0], 4, False) for k in range(4)
        ]
        candidates = [truth]
        kinds = ["truth"]
        seen = {canon(H)}
        for n in range(220):
            if len(candidates) == 4:
                break
            if len(candidates) == 1:
                D = rot(mirror(H), rng.randrange(4))
                kind = "mirror"
            elif high:
                a, b = rng.sample(list(np.ndindex((nr, nc))), 2)
                if H[a[0]][a[1]] == H[b[0]][b[1]]:
                    continue
                D = [r[:] for r in H]
                D[a[0]][a[1]], D[b[0]][b[1]] = D[b[0]][b[1]], D[a[0]][a[1]]
                D = rot(D, rng.randrange(4))
                kind = "height_swap"
            else:
                a = rng.choice([p for p in np.ndindex((nr, nc)) if H[p[0]][p[1]] > 1])
                b = rng.choice(list(np.ndindex((nr, nc))))
                if a == b or H[b[0]][b[1]] >= 3:
                    continue
                D = [r[:] for r in H]
                D[a[0]][a[1]] -= 1
                D[b[0]][b[1]] += 1
                D = rot(D, rng.randrange(4))
                kind = "move"
            key = canon(D)
            if key in seen:
                continue
            rendered = render_h(D, px, PALETTES[0], 4, False)
            others = all_truth + [
                render_h(x, px, PALETTES[0], 4, False) for x in candidates[1:]
            ]
            if not all(
                extra.difference3d(rendered, x, area=24, core=6)["readable"] for x in others
            ):
                continue
            if not guard.accepts(D):
                continue
            seen.add(key)
            candidates.append(D)
            kinds.append(kind)
        if len(candidates) != 4:
            continue
        order = list(range(4))
        rng.shuffle(order)
        options = [candidates[k] for k in order]
        kinds = [kinds[k] for k in order]
        correct = order.index(0)
        break
    else:
        raise RuntimeError(
            f"could not build readable {tier} viewpoint_rotation at {size}"
        )
    img = compose(H, options, size, PALETTES[0])
    rec = extra.record(
        "viewpoint_rotation",
        idx,
        tier,
        QUESTION,
        "ABCD"[correct],
        dict(
            _size=size,
            _cot=f'Only {"ABCD"[correct]} is a {yaw*90}-degree rotation of the reference. All candidates have the same cube count.',
            rung="L7" if high else "L6",
            heightmap=H,
            cube_count=cube_count(H),
            yaw_degrees=90 * yaw,
            option_heightmaps=options,
            option_kinds=kinds,
            correct_index=correct,
            reference_size=refpx,
            option_size=px,
            reference_surface_pixels=extra.surface_counts(H, refpx, margin=16),
            visible_surface_pixels=[
                extra.surface_counts(o, px, margin=4) for o in options
            ],
            min_surface_pixels=extra.MIN_SURFACE,
            equal_cube_counts=True,
            equal_height_histograms=high,
            source_fully_readable=True,
            all_rotations_raster_checked=True,
            palette_rgb=list(PALETTES[0]),
            reference_guard=GUARD_VERSION,
            reference_hidden_cells=len(guard.hidden),
            reference_variants=len(guard.variants),
        ),
        options="ABCD",
    )
    return rec, img
