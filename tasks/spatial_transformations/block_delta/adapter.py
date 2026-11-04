"""block_delta -- count the cubes added / removed between two isometric renders.

Generator: generator/generate.py (rungs L1-L5 for easy/medium/hard) and
generator/xhard.py (L6 = xhard, L7 = xxhard).  The adapter calls
`generate.generate_one(seed, index, size, tier)` or `xhard.generate(seed, index,
size, tier)` with the visro-derived seed, so the generator's RNG
(`Random(f"{seed}:{idx}")`, or `...:{tier}` for xhard/xxhard) is keyed by (task,
tier, split, base seed, index) and no two tiers or splits share a stream.
medium/hard alternate their two rungs by index parity
(`extra_difficulty.base_rung`).

  * question: names the two labelled panels, states the framing (per panel on
    L1-L5, shared on L6/L7) and that heights are compared cell by cell with an
    empty cell as 0; the pair answer is 'added,removed'.
  * gate: generate.displayed_readable() requires, on the shown panel, >= 8
    robust px of every column top face and, for empty cells whose floor is
    hidden, a visible probe cube.  It uses no randomness.
  * meta: palette, layout and each panel's camera are recorded so the verifier
    can re-render and DECODE the panels independently; the verifier also checks
    every pixel outside the decode windows against the documented chrome, that
    each whole decode window equals the render of the decoded heightmap, and
    that the question's only format example, the placeholder 'added,removed',
    can never match a gold.
  * core_key: the scene (before, after) canonicalised under the i<->j transpose
    (which mirrors the picture) and under time reversal (after, before), without
    palette or question kind, so a scene is never reused across tiers or splits.
"""
from __future__ import annotations

from visro import api
from visro.seeding import digest

from tasks.spatial_transformations.block_delta.generator import generate, xhard

TASK = "block_delta"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

TIER_DEFINITIONS = {
    "easy": ("L1: 448x448 image, two 218 px panels each framed on its own (render_stack margin 24, ss 2); "
             "grid 3-4 rows x 4 cols; BEFORE = diverse_map(height<=2, holes) with 4-12 cubes; 1-2 single-cube "
             "additions on cells below height 2, no removals; asks the number ADDED (int); 4 palettes."),
    "medium": ("L2 on even index: 3x3 grid, height<=3, fill 0.7, 2-4 single-cube edits with >=1 add and >=1 "
               "remove; L3 on odd index: 3-4 x 3-4 grid, height<=3, fill 0.72, 3-5 edits; every column changes "
               "in one direction; asks added+removed (int); 448 px, per-panel framing, 4 palettes."),
    "hard": ("L4 on even index: 3-4 rows x 4 cols, height<=4, fill 0.85, taller toward the back, 3-6 mixed edits, "
             "asks added+removed (int); L5 on odd index: 4x4, height<=4, fill 0.8, taller toward the back, 4-7 "
             "mixed edits, asks 'added,removed'; 448 px, per-panel framing, 4 palettes."),
    "xhard": ("L6: 588x588 image, two 288 px panels with ONE shared camera (envelope of both maps, margin 16); "
              "4x5 descending terrace (stepped_map height 5, empty front corner), n = 8-10 terrace-preserving "
              "single-cube edits with max(2,n//3) <= added <= min(n-2,2n//3), min(added,removed)>=2, >=5 changed "
              "columns; asks 'added,removed'; gray palette."),
    "xxhard": ("L7: as xhard with a 5x5 grid, height<=6, 12-16 edits, min(added,removed)>=4 and >=8 changed "
               "columns; asks 'added,removed'; 588 px, shared camera, gray palette."),
}

def _transpose(H):
    return [list(r) for r in zip(*H)]


def scene_key(H1, H2) -> str:
    """Content key of a (before, after) scene.  Swapping the i and j axes of
    both maps mirrors the rendered picture left-right (x = (i - j) * hw) and
    keeps every count; reversing time (H2, H1) shows the same two pictures in
    swapped panels (same total, added/removed swapped).  All four variants map
    to one key, so neither a mirror image nor a reversed scene can reappear in
    another tier or split."""
    variants = [[A, B] for A, B in ((H1, H2), (H2, H1))]
    variants += [[_transpose(A), _transpose(B)] for A, B in variants]
    return digest(["block_delta/scene", min(variants)])


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in TIER_DEFINITIONS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier in ("xhard", "xxhard"):
        rec, image = xhard.generate(int(seed), int(index), 588, tier)
    else:
        rec, image = generate.generate_one(int(seed), int(index), 448, tier)
    m = rec["meta"]
    H1, H2 = m["heightmap_before"], m["heightmap_after"]
    kind = m["answer_kind"]
    meta = {
        "tier": tier,
        "rung": m["rung"],
        "answer_kind": kind,
        "heightmap_before": H1,
        "heightmap_after": H2,
        "added": m["added"],
        "removed": m["removed"],
        "resolution": m["resolution"],
        "panel_size": m["panel_size"],
        "panel_x": m["panel_x"],
        "panel_y": m["panel_y"],
        "palette": m["palette"],
        "camera": m["camera"],
        "placement_before": m["placement_before"],
        "placement_after": m["placement_after"],
        "generator_version": m["generator_version"],
        "generator": ("generator/xhard.py:generate" if tier in ("xhard", "xxhard")
                      else "generator/generate.py:generate_one"),
    }
    for k in ("sampling_attempts", "changed_columns", "min_surface_pixels"):
        if k in m:
            meta[k] = m[k]
    return api.Item(
        question=rec["question"],
        answer=rec["answer"],
        image=image,
        meta=meta,
        core_key=scene_key(H1, H2),
        answer_type="token-list" if kind == "pair" else "int",
    )
