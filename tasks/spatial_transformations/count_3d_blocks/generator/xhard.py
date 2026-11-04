"""count_3d_blocks xhard / xxhard: the 'minimum' question on 588 px scenes.

Every forced column top keeps >= 12 robust pixels, the minimum witness redraws
the question image byte for byte with the question's camera, and removing any
forced top cube changes a readable patch.

Each problem is a pure function of (seed, idx, tier):
random.Random(f"{seed}:{idx}:{tier}").
"""
from __future__ import annotations

import random

from PIL import ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso

from .generate import build_qa


def generate(seed, idx, size, tier):
    """One xhard/xxhard item: (record, image)."""
    size = extra.resolve_size(size, tier)
    rng = random.Random(f"{seed}:{idx}:{tier}")
    high = tier == "xxhard"
    nr, nc, hmax = (5, 5, 6) if high else (4, 5, 5)
    for attempt in range(2500):
        H = [
            [rng.randint(1, hmax) if rng.random() < 0.86 else 0 for _ in range(nc)]
            for _ in range(nr)
        ]
        img0, info = iso.render_stack(H, size=size, margin=46, supersample=2)
        seen = extra.surface_counts(H, size, info=info, mode="cube")
        per = {}
        valid = True
        for j in range(nr):
            for i in range(nc):
                levels = [k for k in range(H[j][i]) if seen[i, j, k][0] > 0]
                forced = max(levels) + 1 if levels else 0
                if forced and seen[i, j, forced - 1][1] < 12:
                    valid = False
                    break
                per[i, j] = forced
            if not valid:
                break
        if not valid:
            continue
        minimum = sum(per.values())
        total = sum(map(sum, H))
        if minimum < (45 if high else 30) or total - minimum < (4 if high else 2):
            continue
        witness = [[per[i, j] for i in range(nc)] for j in range(nr)]
        # The claimed minimum must reproduce the actual RGB question exactly,
        # with a fixed camera. Removing any forced top must visibly change it.
        witness_img = iso.render_fixed(witness, size, info)
        if img0.tobytes() != witness_img.tobytes():
            continue
        witnesses = []
        for j in range(nr):
            for i in range(nc):
                if not witness[j][i]:
                    continue
                probe = [row[:] for row in witness]
                probe[j][i] -= 1
                witnesses.append(
                    extra.difference3d(
                        witness_img,
                        iso.render_fixed(probe, size, info),
                        area=24,
                        core=6,
                    )
                )
        if not all(x["readable"] for x in witnesses):
            continue
        break
    else:
        raise RuntimeError(f"could not build readable {tier} count_3d_blocks at {size}")
    q, ans, cot = build_qa(H, "minimum", minimum, per)
    img = img0.copy()
    ImageDraw.Draw(img).rounded_rectangle(
        [3, 3, size - 4, size - 4], radius=14, outline=(225, 225, 230), width=2
    )
    rec = extra.record(
        "count_3d_blocks",
        idx,
        tier,
        q,
        ans,
        dict(
            _size=size,
            _cot=cot,
            variant="minimum",
            heightmap=H,
            total_cubes=total,
            minimum_cubes=minimum,
            minimum_heightmap=witness,
            occlusion=total - minimum,
            base_rows=nr,
            base_cols=nc,
            per_column_forced={f"{i},{j}": v for (i, j), v in per.items()},
            placement=info,
            palette=[205, 208, 214],  # render_stack's default base colour
            minimum_reproduces_image=True,
            forced_column_differences=witnesses,
            min_visible_cube_core_pixels=min(
                seen[i, j, k - 1][1] for (i, j), k in per.items() if k
            ),
        ),
    )
    return rec, img
