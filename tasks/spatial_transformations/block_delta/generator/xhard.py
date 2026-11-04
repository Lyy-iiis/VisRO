"""block_delta xhard / xxhard (rungs L6, L7): one shared camera for both panels.

BEFORE is a descending terrace (extra.stepped_map with an empty front corner);
AFTER applies terrace_ops' mixed, non-cancelling single-cube edits.  Both maps
are drawn with the camera of their envelope, every column top and empty cell
keeps >= extra.MIN_SURFACE robust pixels in both panels, and every changed
column alone alters a readable patch.

Each problem is a pure function of (seed, idx, tier):
random.Random(f"{seed}:{idx}:{tier}").
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from shared import extra_difficulty as extra
from shared.drawing import isometric as iso
from shared.drawing.fonts import load_bundled_font

from .generate import build_qa


def terrace_ops(H, rng, cfg):
    """Mixed, non-cancelling edits that keep a visible descending terrace."""
    A = [r[:] for r in H]
    nr, nc = len(A), len(A[0])
    directions = {}
    n = rng.randint(*cfg["ops"])
    na = rng.randint(max(2, n // 3), min(n - 2, 2 * n // 3))
    for delta in [1] * na + [-1] * (n - na):
        possible = []
        for j in range(nr):
            for i in range(nc):
                h = A[j][i] + delta
                if not 0 <= h <= cfg["hmax"] or directions.get((j, i), delta) != delta:
                    continue
                if j and h > A[j - 1][i]:
                    continue
                if i and h > A[j][i - 1]:
                    continue
                if j + 1 < nr and h < A[j + 1][i]:
                    continue
                if i + 1 < nc and h < A[j][i + 1]:
                    continue
                possible.append((j, i))
        if not possible:
            return None
        # Spread changes over columns before revisiting one.
        fresh = [p for p in possible if p not in directions]
        j, i = rng.choice(fresh or possible)
        A[j][i] += delta
        directions[j, i] = delta
    return A, na, n - na


def generate(seed, idx, size, tier):
    """One xhard/xxhard item: (record, image)."""
    size = extra.resolve_size(size, tier)
    rng = random.Random(f"{seed}:{idx}:{tier}")
    high = tier == "xxhard"
    nr, nc, hmax = (5, 5, 6) if high else (4, 5, 5)
    panel = (size - 12) // 2
    cfg = dict(ops=(12, 16) if high else (8, 10), kinds=("add", "remove"), hmax=hmax)
    for attempt in range(2500):
        H1 = extra.stepped_map(rng, nr, nc, hmax, holes=True)
        result = terrace_ops(H1, rng, cfg)
        if result is None:
            continue
        H2, added, removed = result
        changed = int(np.count_nonzero(np.asarray(H1) != H2))
        if min(added, removed) < (4 if high else 2) or changed < (8 if high else 5):
            continue
        envelope = np.maximum(H1, H2).tolist()
        info = iso.placement(envelope, panel, margin=16, supersample=2)
        counts = [extra.surface_counts(H, panel, info=info) for H in (H1, H2)]
        if min(min(map(min, c)) for c in counts) < extra.MIN_SURFACE:
            continue
        # Each actual changed column must alter a clear patch on its own.
        before = iso.render_fixed(H1, panel, info)
        weak = []
        for j, i in zip(*np.where(np.asarray(H1) != H2)):
            probe = [row[:] for row in H1]
            probe[j][i] = H2[j][i]
            weak.append(
                extra.difference3d(
                    before, iso.render_fixed(probe, panel, info), area=24, core=6
                )
            )
        if not all(x["readable"] for x in weak):
            continue
        break
    else:
        raise RuntimeError(f"could not build readable {tier} block_delta at {size}")
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    y = (size - panel) // 2 + 10
    for x, H, label in [(4, H1, "BEFORE"), (8 + panel, H2, "AFTER")]:
        img.paste(iso.render_fixed(H, panel, info), (x, y))
        d.text((x + 14, y - 24), label, fill=(30, 30, 30), font=load_bundled_font(18, "sans-bold"))
    d.line([(size // 2, y - 25), (size // 2, y + panel)], fill=(210, 215, 220), width=2)
    # build_qa states the per-stack (net) counting rule for every tier.
    q, ans, cot = build_qa("pair", added, removed, camera="shared")
    rec = extra.record(
        "block_delta",
        idx,
        tier,
        q,
        ans,
        dict(
            _size=size,
            _cot=cot,
            rung="L7" if high else "L6",
            heightmap_before=H1,
            heightmap_after=H2,
            added=added,
            removed=removed,
            answer_kind="pair",
            changed_columns=changed,
            visible_surface_pixels=counts,
            panel_size=panel,
            placement=info,
            min_surface_pixels=extra.MIN_SURFACE,
            changed_column_differences=weak,
            panel_x=[4, 8 + panel],
            panel_y=y,
            palette=[205, 208, 214],
            camera="shared",
            placement_before=info,
            placement_after=info,
        ),
    )
    return rec, img
