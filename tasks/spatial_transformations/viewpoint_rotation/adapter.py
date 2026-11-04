"""viewpoint_rotation -- which option is the reference cube structure seen after a yaw?

Generator: generator/generate.py (rungs L1-L5 for easy/medium/hard) and
generator/xhard.py (L6 = xhard, L7 = xxhard).  The adapter calls
`generate.generate_one(seed, index, size, tier)`, or `xhard.generate(seed, index,
size, tier)` for xhard/xxhard, with the visro-derived seed, so the generator's
RNG (`Random(f"{seed}:{idx}")`, or `...:{tier}` for xhard/xxhard) is keyed by
(task, tier, split, base seed, index) and no two tiers or splits share a stream.  medium/hard alternate their
two rungs by index parity (`extra_difficulty.base_rung`).

  * uniqueness: generate.ReferenceGuard computes the heights a hidden column of
    the REFERENCE could take at the delivered reference size and refuses any
    distractor that is a view of such a variant, in every tier; it compares
    every distractor with every view of every variant at the delivered option
    size, in place and silhouette-aligned, refuses references whose panel
    allows an unreadable change of two cells, and refuses distractors closer
    than half a cube top face of readable area (measured at face level) to an
    aligned view of any reference-consistent structure.
  * question: states the conventions the answer depends on (hidden cubes
    possible, yaw by 90/180/270 degrees, options scaled to fit and drawn
    without the floor grid) and shows no example answer.
  * distractors: medium/hard have exactly one mirror image; the rest are the
    rung's edits.
  * meta: palette, image size and panel sizes are recorded for every tier so
    the verifier can re-render and decode the panels.
  * core_key: the REFERENCE structure up to translation, yaw and mirror, so no
    reference repeats within a cell, across tiers or across splits.
  * easy floors: L1 draws a 5x5, 4x4 or 3x5 floor (2:1:1) and keeps a reference
    only if its occupied bounding box is the whole floor; 5x5 takes 4-8 cubes,
    4x4 and 3x5 exactly 9 (609,099 reference classes).  No easy key can equal a
    key of another tier.  The answer is shown at 90/270 deg, each edit at a
    random yaw.
"""
from __future__ import annotations

from visro import api
from visro.seeding import digest

from tasks.spatial_transformations.viewpoint_rotation.generator import generate, xhard

TASK = "viewpoint_rotation"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

_COMMON = ("reference panel with floor grid + four auto-fitted option panels A-D without floor; answer = the one "
           "option that is the reference yawed by 90/180/270 deg; reference chiral, no rotational self-symmetry; "
           "reference guard: the reference panel allows at most 64 structures (column heights 0..max+2 that change no "
           "readable reference pixel) and no unreadable change of two cells (grid or front pad row/column) that "
           "single-column changes cannot explain; each distractor readably different (difference3d contrast 32, "
           "area 24, core 6, delivered option size, in place and silhouette-aligned) from every view of every such "
           "structure, separated from the silhouette-aligned views of the reference by readable regions totalling "
           ">= half a cube top face, and from every view of every such structure (in place and silhouette-aligned) "
           "by >= half a cube top face of face-level difference (pixels both panels show as a face or the "
           "background, of different shade, at contrast >= 32, not within 1 px of an outline in both panels)")
TIER_DEFINITIONS = {
    "easy": ("L1: 448 px image (reference 179 px with floor grid, options 117 px); floor 5x5, 4x4 or 3x5 "
             "rows x cols (drawn 2:1:1, about a third of the items each) and the reference's occupied bounding box "
             "is the whole floor (cubes in its first and last row and in its first and last column; inner rows and "
             "columns may be empty); column height <= 2; 4-8 cubes on 5x5 (random proposals are trimmed to the "
             "bound, so nearly every reference has 8), exactly 9 on 4x4 and 3x5; yaw 90 or 270 deg; "
             "3 distractors = 2-3-cell height edits (cube count may change; an edit that happens to be the mirror "
             "image is labelled mirror), each shown at a random yaw; 4 palettes; "
             + _COMMON),
    "medium": ("L2 on even index: base 2-3 x 2-3, 5-8 cubes, height <= 3, exactly 1 mirror image + 2 two-cell "
               "height edits (count may change); L3 on odd index: base 3 x 3-4, 7-10 cubes (trimmed to the bound, "
               "mostly 10), height <= 3, exactly 1 mirror image + 2 count-preserving moves of 1-2 cubes; yaw "
               "90/180/270; 448 px, 4 palettes; " + _COMMON),
    "hard": ("L4 on even index: base 3 x 3-4, 7-12 cubes, height <= 3, proposal made left-right symmetric with "
             "one breaking edit, then trimmed by random one-cube removals to the bound (mostly 12 cubes; the trimming "
             "often breaks more of the symmetry); L5 on odd index: base 3-4 x 3-4, 10-14 cubes, height <= 4; "
             "both: exactly 1 mirror image + 2 one-cube moves (count preserved); yaw 90/180/270; 448 px, "
             "4 palettes; " + _COMMON),
    "xhard": ("L6: 588 px image (reference 235 px, options 159 px); base 4 x 5, 20-28 cubes, diverse_map height "
              "<= 3 with holes, every column top / empty floor cell >= 8 px in the reference; exactly 1 mirror "
              "image + 2 one-cube moves (source height > 1, destination height < 3), each readably different "
              "(difference3d area 24 core 6) from all 4 views of the truth; yaw 90/180/270; gray palette; " + _COMMON),
    "xxhard": ("L7: 588 px image; base 5 x 5, 30-42 cubes, diverse_map height <= 4 with holes, readable reference "
               "as xhard; exactly 1 mirror image + 2 column-height swaps (count and height multiset preserved), "
               "readably different from all 4 views of the truth; yaw 90/180/270; gray palette; " + _COMMON),
}

def _canon(a):
    """Structure up to translation and the four yaws: (shape, values) of the
    smallest yaw of the occupied extent."""
    import numpy as np

    a = np.asarray(a)
    nz = np.argwhere(a > 0)
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
    e = a[r0:r1 + 1, c0:c1 + 1]
    return min((list(np.rot90(e, k).shape), [int(v) for v in np.rot90(e, k).ravel()]) for k in range(4))


def core_key(H) -> str:
    """Reference identity: the reference structure up to translation, yaw and
    mirror.  The correct option is a yaw of the reference, so every item with
    this reference shares its answer structure, whatever the distractors."""
    import numpy as np

    a = np.asarray(H)
    return digest(["viewpoint_rotation/reference", min(_canon(a), _canon(a[:, ::-1]))])


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    extra_tier = tier in ("xhard", "xxhard")
    size = 588 if extra_tier else 448
    if extra_tier:
        rec, img = xhard.generate(seed, index, size, tier)
    else:
        rec, img = generate.generate_one(seed, index, size, tier)
    m = rec["meta"]
    ref_px, _ = generate.layout_slots(size)
    opt_px = generate.option_px(size)
    # xhard/xxhard draw everything in PALETTES[0]; rungs L1-L5 record their palette.
    palette = list(m.get("palette_rgb") or generate.PALETTES[0])
    if m.get("reference_size", ref_px) != ref_px or m.get("option_size", opt_px) != opt_px:
        raise api.Reject("panel size disagrees with the layout")
    meta = {
        "generator": "generator/xhard.py:generate" if extra_tier else "generator/generate.py:generate_one",
        "generator_version": m.get("generator_version"),
        "reference_guard": m["reference_guard"],
        "rung": m["rung"],
        "size": size,
        "palette_rgb": palette,
        "reference_px": ref_px,
        "option_px": opt_px,
        "heightmap": m["heightmap"],
        "option_heightmaps": m["option_heightmaps"],
        "option_kinds": m["option_kinds"],
        "correct_index": m["correct_index"],
        "yaw_degrees": m["yaw_degrees"],
        "cube_count": m["cube_count"],
        "option_cube_counts": [int(sum(map(sum, o))) for o in m["option_heightmaps"]],
        "reference_hidden_cells": m["reference_hidden_cells"],  # heights the reference cannot pin down
        "conventions": {
            "heightmap": "heightmap[j][i] = cubes stacked on floor cell (row j, column i); +i is drawn toward "
                         "the lower right, +j toward the lower left, so the cell with the largest i+j is nearest",
            "projection": "screen x = ox + (i - j) * hw, y = oy + (i + j) * hh - level * ch (2:1 isometric, "
                          "hw:hh:ch = 34:17:40); faces: top 1.0, left (+j side) 0.74, right (+i side) 0.88 of "
                          "the palette colour, dark outlines; reference panel draws the floor grid",
            "yaw": "option_heightmaps[correct_index] == numpy.rot90(heightmap, yaw_degrees // 90)",
            "layout": "reference centred at the top, options A B / C D in a 2x2 grid below; each panel "
                      "auto-fitted to its own structure",
        },
    }
    return api.Item(
        question=rec["question"],
        answer=rec["answer"],
        image=img,
        meta=meta,
        core_key=core_key(m["heightmap"]),
        answer_type="choice",
        choices=["A", "B", "C", "D"],
    )
