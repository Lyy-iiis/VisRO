"""view_reconstruction: which isometric cube structure matches 2-3 orthographic views.

Generator: generator/generate.py:

* the SIDE view is the physical right-side view on every tier (front row at
  the left, back row at the right, arrow BACK);
* the four options are an exchangeable lattice ({X, X+A, X+B, X+A+B} or
  {X, X+A+B, X+A+C, X+B+C}) whose gates hold for every possible answer
  (including: no option matches another under a mirrored, flipped or
  shifted reading of the views), and the answer is drawn uniformly from it,
  so the option panel alone carries no information about the answer;
* every tier draws X from extra.diverse_map, draws the option floor grid as
  exactly the N x M footprint, and gates legibility at the final pixel size;
* the question names the shown views and their reading direction and the
  isometric pose of the options, and gives no example answer letter;
* the shown view pair is drawn once per item (delivered pairs are uniform);
  core_key is the answer structure alone, so the same answer is never
  delivered twice in a release (across cells, tiers and splits).

Easy: one footprint per item from 4x4, 3x5, 2x6, 2x7 (weights 11:22:14:42), the
588 px layout (170 px options, as xhard/xxhard), X uniform over diverse_map's
support (every map that can pass the legibility gate), cap 2, X with 4-12 cubes,
and every option touching all four edges of its floor; the oblong footprints
come in one orientation only, so no easy structure has its mirror image on an
easy floor.

All randomness comes from random.Random(seed), where `seed` is the driver's
derive_seed(task, tier, split, base seed, index); the rung inside a mixed tier
is index % 2 (generate.rung_for).
"""
from __future__ import annotations

import random

from visro import api
from visro.seeding import digest

from tasks.spatial_transformations.view_reconstruction.generator import generate as _GEN

TASK = "view_reconstruction"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

TIER_DEFINITIONS = {
    "easy": ("L1: 2 views (top+front, top+side or front+side, drawn uniformly once per item); footprint 4x4, "
             "3x5, 2x6 or 2x7 rows x cols (drawn once per item, weights 11:22:14:42); X uniform over the cap-2 "
             "heightmaps with 4-12 cubes in which no column is lower than its occlusion floor max(diagonal front "
             "neighbour, min(two side front neighbours)) (diverse_map's support; the legibility gate then "
             "applies); square lattice {X, X+A, X+B, X+A+B} where A changes only view 1 and B only view 2, each "
             "edit setting 2-3 cells to new heights 0-2; every option touches all four edges of its floor; "
             "588 px image, 170 px options, palette uniform from 4."),
    "medium": ("even index L2: 2 views (pair drawn uniformly once per item); 2-3 rows x 3 cols; "
               "X = diverse_map(cap 3, holes); square lattice, edits of 1-2 cells (heights 0-3). odd index L3: "
               "3 views; 3 rows x 3-4 cols; cap 3; tetra lattice {X, X+A+B, X+A+C, X+B+C} (each distractor fails "
               "exactly two views), edits of 1-2 cells (L3 differs from hard's L4 only in this lattice design); "
               "448 px image, 121 px options, palette uniform from 4."),
    "hard": ("even index L4: 3 views; 3 rows x 3-4 cols; X = diverse_map(cap 3, holes); square lattice in which "
             "one random view splits the options differently from the other two (distractors fail 1, 2 and 3 "
             "views), edits of 1-2 cells (heights 0-cap). odd index L5: the same with 3-4 rows x 3-4 cols "
             "(redrawn for every prior sample) and cap 4; 448 px image, 121 px options, palette uniform from 4."),
    "xhard": ("L6: 3 views; 4 x 4 footprint; X = diverse_map(cap 4, holes); square lattice of single-cube moves "
              "(all four options have equal volume) in which one random view splits the options differently "
              "from the other two (distractors fail 1, 2 and 3 views); 588 px image, 170 px options, grey "
              "palette (205,208,214)."),
    "xxhard": ("L7: 3 views; 4 x 5 footprint; X = diverse_map(cap 5, holes); square lattice of single-cube moves "
               "(all four options have equal volume) in which one random view splits the options differently "
               "from the other two (distractors fail 1, 2 and 3 views); 588 px image, 170 px options, grey "
               "palette (205,208,214)."),
}

_TIER_RUNGS = {"easy": (1,), "medium": (2, 3), "hard": (4, 5), "xhard": (6,), "xxhard": (7,)}
assert _TIER_RUNGS == {t: tuple(r) for t, r in _GEN.TIER_RUNGS.items()}, "adapter/generator tier map drifted"


def core_key(meta: dict) -> str:
    """Content key: the answer structure (heightmap) alone.

    Two items with the same answer are one question for a solver who has seen
    either: they share the answer and at least one view panel (or show the
    same views with other distractors), so a release never repeats an answer,
    whatever the views shown, the distractors, the letter order or the palette.
    Spatial mirrors are NOT
    folded: a mirrored structure has different view panels and a different
    isometric picture, i.e. it is a different question.
    """
    return digest({"task": TASK, "answer_heightmap": meta["heightmap"]})


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in _TIER_RUNGS:
        raise ValueError(f"unknown tier {tier!r}")
    rung = _GEN.rung_for(tier, index)
    rng = random.Random(seed)
    rec, image, reasons = _GEN.make_item(rng, tier, rung)
    meta = dict(rec["meta"])
    meta["rationale"] = rec["cot"]
    meta["generator"] = "generator/generate.py:make_item"
    # Gate rejections inside this item's search (prior samples / lattices tried).
    meta["search_rejects"] = dict(sorted(reasons.items()))
    return api.Item(
        question=rec["question"],
        answer=rec["answer"],
        image=image,
        meta=meta,
        core_key=core_key(meta),
        answer_type="choice",
        choices=["A", "B", "C", "D"],
    )
