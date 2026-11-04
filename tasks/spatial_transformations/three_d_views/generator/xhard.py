"""three_d_views xhard / xxhard: descending terraces with 2^k corner options.

The proposal scene is terrace_scene (a descending terrace with a random
staircase-shaped empty corner); terrace_family makes k = 2 (xhard) or 3
(xxhard) one-step edits at the terrace's corners, and the asked scene is drawn
uniformly from the 2^k corner scenes, so the option set says nothing about
which option is the truth.  xxhard shows its 8 options A-H in two rows.

Each problem is a pure function of (seed, idx, tier):
random.Random(f"{seed}:{idx}:{tier}").
"""
from __future__ import annotations

import random

from PIL import ImageDraw

from shared import extra_difficulty as extra
from shared.drawing.fonts import load_bundled_font

from .generate import (
    FIGURE_FRACTION, GENERATOR_VERSION, MIN_SURFACE_PIXELS, VIEW_DIRECTIONS, compose,
    render_grid_panel, terrace_family, terrace_scene, view_question, visible_surfaces,
)


def generate(seed, idx, size, tier):
    """One xhard/xxhard item: (record, image)."""
    size = extra.resolve_size(size, tier)
    rng = random.Random(f"{seed}:{idx}:{tier}")
    high = tier == "xxhard"
    nr, nc, hmax = (5, 5, 5) if high else (4, 5, 4)
    view = ("front", "top", "side")[idx % 3]
    nbits = 3 if high else 2
    # a terrace with a random staircase-shaped empty corner (not stepped_map's fixed
    # diagonal cut), edits only at the terrace's corners, and the asked scene drawn uniformly from
    # the 2^k corner scenes, so the option set says nothing about which option is the truth.
    for attempt in range(2500):
        H0 = terrace_scene(rng, nr, nc, hmax)
        fam = terrace_family(H0, view, rng, nbits, hmax)
        if fam is None:
            continue
        scenes, family, frame = fam
        if view == "top" and not all(0.25 <= float(x.mean()) <= 0.90 for x in family):
            continue
        surfaces = []
        for W in scenes:
            counts = visible_surfaces(W.tolist(), size)
            if min(map(min, counts)) < MIN_SURFACE_PIXELS:
                break
            surfaces.append(counts)
        else:
            break
    else:
        raise RuntimeError(f"could not build readable {tier} three_d_views at {size}")
    t = rng.randrange(len(scenes))
    H, truth, counts = scenes[t].tolist(), family[t], surfaces[t]
    actual_h = frame if view != "top" else max(map(max, H))
    order = list(range(len(family)))
    rng.shuffle(order)
    grids = [family[k] for k in order]
    correct = order.index(t)
    labels = "ABCDEFGH"[: len(grids)]
    layout = {}
    img = compose(H, view, grids[:4], min(correct, 3), size, rng, layout)
    # Keep the same large reference. Reserve a separate option grid below it.
    d = ImageDraw.Draw(img)
    top = int(size * 0.60)
    d.rectangle((0, top - 5, size, size), fill="white")
    d.line([(8, top - 6), (size - 8, top - 6)], fill=(230, 230, 234), width=2)
    rows = 2 if high else 1
    sw = (size - 16) // 4
    sh = (size - top - 8) // rows
    boxes, label_xy = [], []
    for k, grid in enumerate(grids):
        rr, cc = divmod(k, 4)
        x = 8 + cc * sw
        y = top + rr * sh
        cell = min((sw - 16) // grid.shape[1], (sh - 28) // grid.shape[0])
        if cell < 12:
            raise ValueError("resolution too small for readable projection options")
        panel = render_grid_panel(grid, cell, pad=4)
        px = x + (sw - panel.width) // 2
        py = y + 24 + (sh - 24 - panel.height) // 2
        img.paste(panel, (px, py))
        d.text((x + 4, y), labels[k], fill=(20, 20, 20), font=load_bundled_font(16, "sans-bold"))
        boxes.append([px, py, px + panel.width, py + panel.height])
        label_xy.append([x + 4, y])
    # The earlier panel zone (slots, labels, divider) was painted over above.
    layout["panels"] = dict(boxes=boxes, cell=cell, pad=4, label_xy=label_xy,
                            label_font_size=16, label_color=[20, 20, 20], slots=[],
                            zone_top=top, divider_y=top - 6)
    q = view_question(view, len(grids))
    rec = extra.record(
        "three_d_views",
        idx,
        tier,
        q,
        labels[correct],
        dict(
            _size=size,
            _cot=f"The {view} projection matches only {labels[correct]}.",
            view=view,
            viewing_direction=list(VIEW_DIRECTIONS[view]),
            heightmap=H,
            hmax=actual_h,
            true_projection=truth.tolist(),
            option_grids=[x.tolist() for x in grids],
            correct_index=correct,
            visible_surface_pixels=counts,
            min_surface_pixels=MIN_SURFACE_PIXELS,
            figure_size=int(size * FIGURE_FRACTION),
            independent_edits=nbits,
            panel_boxes=boxes,
            tier_hmax=hmax,
            three_d_views_version=GENERATOR_VERSION,
            layout=layout,
        ),
        options=labels,
    )
    return rec, img
