"""find_the_same -- find the pairs of identical pictures among near-twins and singles.

Generator: generator/generate.py, `make_problem` as it is: diff_params,
build_items (pairs pasted from ONE rendered tile, near-twins from iconlib.alter,
the 64/48/12 readability floor -- as placed and after the best small
translation -- against every existing tile, the near-twin nearness gate),
layout_geom, its own pairs_from_tiles == intended_pairs assertion and its
all-pairs readability assertion.

* Seed.  `seed` is visro's derived seed, keyed by (task, tier, split, base seed,
  index); tiers and splits never share a stream.
* Render path.  Every icon is rasterised from its SVG with cairosvg; the
  adapter refuses to run without cairosvg and checks that iconlib and the icon
  tree are this checkout's.  meta.render records the cairosvg version and the
  sha256 of every SVG the item draws.
* Icon pool per split: train/val draw from iconlib's "train" icons, eval/ood
  from the held-out "test" icons (ICON_SPLIT).
* Answer: pair-list, canonical "a-b,c-d"; meta keeps the geometry, item roles
  and readability numbers the independent verifier needs, as lists indexed by
  slot-1.
* Question: states that rotated / mirrored / edited copies are different, that
  identical items come only in pairs, and one answer format, "a-b" in letters,
  pairs separated by commas.  Pose-only
  distractors are rejected (two tiles must still meet the floor after the best
  translation within 2 px of their ink-box offset), and near-twins must change
  <= 55% of their pair's ink pixels on the delivered tile and be strictly
  nearest their own pair, so the item roles can be checked from the pixels.

CORE KEY
    sha256 over the sorted multiset of (icon_id, role) -- role is pair /
    near_twin / single.  Slot arrangement, tint, tile size and the exact
    alteration are nuisance: the same icons in the same roles are the same
    comparison problem, so they count as a duplicate across tiers too.
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import functools
import hashlib
import os
from pathlib import Path

# Before the generator imports numpy: OpenBLAS would otherwise start one
# spinning thread per core (240 on the release host) in every process, and
# nothing here uses BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from visro import api  # noqa: E402
from visro.seeding import digest  # noqa: E402

TASK = "find_the_same"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

ICON_SPLIT = {"train": "train", "val": "train", "eval": "test", "ood": "test"}

# Strength numbers measured on seed 20261011 train items (1,200 medium / 1,200 hard /
# 400 xhard / 400 xxhard).
_SHARED = ("silhouette (MDI/game-icons, p=0.6, one of 7 tints for the whole grid) or "
           "OpenMoji colour; true pairs = one rendered tile pasted twice; a near-twin = a "
           "pair's icon with one iconlib.alter change, which the generator requests at the "
           "tier's nominal strength s x1.0/1.5/2.2 and alter retries per method at "
           "x1.0/1.7/2.6 (capped at 1); silhouette: erase/add/rotate>=12deg/mirror/recolour "
           "in random order, s sets the erase/add radius, rotation angle and recolour shift "
           "(mirror ignores it; measured rank correlation of s with the realised change "
           "+0.08 to +0.11 per tier); colour: remove_element is always tried first -- it "
           "deletes one random SVG leaf and ignores s, a retry only redraws the leaf -- then "
           "erase/hue-shift/rotate/mirror in random order; 4,193 of 4,243 measured colour "
           "near-twins are remove_element, so s has no effect on colour grids outside that "
           "rare fallback; alter keeps a change of >=12 px (pixels whose summed absolute "
           "RGB difference is >18 on the 4x-supersampled render composited on white) and "
           "<=55% of the footprint (pixels with alpha >24 on that render) (remove_element "
           "has no upper cap; fallback: an erase at the centroid); the generator then keeps "
           "a near-twin only if, on the delivered tile, it changes <=55% of its pair's ink "
           "pixels (max-channel diff >=64 over ink >=32) and is strictly nearest its own "
           "pair by that measure; every non-matching pair of delivered tiles differs by a "
           "connected >=48 px patch of max-channel difference >=64 with >=12 px left after "
           "3x3 erosion, both as placed and after the best translation (fewest pixels "
           "differing by >=64 among (0,0) and the shifts within 2 px of the ink-box offset, "
           "for tiles whose ink boxes agree in size within 2 px); no pair members "
           "orthogonally adjacent")
TIER_DEFINITIONS = {
    "easy": ("448x448; M in {9,10,11} items (3x3 grid, tile 99 px, for M=9; else 4 columns, "
             "tile 73 px); 2-3 identical pairs; 0 near-twins; the rest distinct icons; "
             + _SHARED),
    "medium": ("448x448; M in {11,12,14} items (4 columns, tile 73 px for 11/12; 5 columns, "
               "tile 57 px for 14); 3-4 identical pairs; 1-2 near-twins; nominal strength "
               "s ~ U(0.4,0.7), silhouette only; M=14 with 4 pairs and 2 near-twins is also "
               "a hard configuration; " + _SHARED),
    "hard": ("448x448; M in {14,15,16} items (5 columns, tile 57 px); 4-5 identical pairs; "
             "2-3 near-twins; nominal strength s ~ U(0.18,0.40), silhouette only; M=14 with "
             "4 pairs and 2 near-twins is also a medium configuration; " + _SHARED),
    "xhard": ("588x588; M=20 items (5x4 grid, tile 75 px); 6 identical pairs; 6 near-twins; "
              "nominal strength s ~ U(0.30,0.50), silhouette only; " + _SHARED),
    "xxhard": ("588x588; M=25 items (5x5 grid, tile 75 px); 8 identical pairs; 8 near-twins; "
               "nominal strength s ~ U(0.25,0.45), silhouette only; " + _SHARED),
}

_GEN = None
_RENDER = None
_SVG_SHA: dict[str, str] = {}
_PATHS = None


@functools.lru_cache(maxsize=2)
def _library(split: str):
    import shared.iconlib as iconlib
    return iconlib.IconLibrary(default_split=split)


def _gen():
    """Import the generator once per process and pin iconlib to live cairosvg."""
    global _GEN, _RENDER
    if _GEN is not None:
        return _GEN
    from tasks.discrimination_grouping.find_the_same.generator import generate
    import shared.iconlib as iconlib
    if Path(iconlib.__file__).resolve().parent != (_ROOT / 'shared/iconlib'):
        raise RuntimeError(f"iconlib resolved to {iconlib.__file__}, not this checkout")
    if not iconlib.HAVE_CAIROSVG:
        raise RuntimeError("find_the_same needs cairosvg: the release renders every icon live")
    lib = _library("train")
    if Path(lib.root).resolve() != (_ROOT / 'assets/icons'):
        raise RuntimeError(f"icon tree {lib.root} is not this checkout's assets/icons")
    import cairosvg
    _RENDER = {"path": "cairosvg-live", "cairosvg": cairosvg.__version__}
    _GEN = generate
    return generate


def _svg_sha(lib, icon_id: str) -> str:
    if icon_id not in _SVG_SHA:
        global _PATHS
        if _PATHS is None:
            _PATHS = {e["id"]: e["path"] for e in lib.icons}
        _SVG_SHA[icon_id] = hashlib.sha256((Path(lib.root) / _PATHS[icon_id]).read_bytes()).hexdigest()[:16]
    return _SVG_SHA[icon_id]


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in api.TIERS or split not in api.SPLITS:
        raise ValueError(f"unknown tier/split {tier}/{split}")
    generate = _gen()
    icon_split = ICON_SPLIT[split]
    lib = _library(icon_split)
    # RuntimeError (no readable scene in 40 resamples) and AssertionError (gold
    # derived from tiles != intended groups, or an unreadable non-matching
    # pair) are the generator's own gates; the driver counts them as rejects.
    img, question, answer, m = generate.make_problem(lib, generate.derive_rng(int(seed), int(index)), None, tier)
    if m["difficulty"] != tier or img.size[0] != (588 if tier in ("xhard", "xxhard") else 448):
        raise api.Reject(f"generator returned {m['difficulty']} at {img.size}")
    M = int(m["M"])
    slots = [str(k) for k in range(1, M + 1)]
    ids = list(m["icon_ids"])
    meta = {
        "generator_version": m["generator_version"],
        "icon_split": icon_split,
        "size": img.size[0],
        "style": m["style"],
        "color": m["color"],
        "M": M, "rows": m["rows"], "cols": m["cols"],
        "tile_px": m["tile_px"], "cell_px": m["cell_px"],
        "pairs": [list(p) for p in m["pairs"]],
        "n_pairs": m["n_pairs"], "n_near_twin": m["n_near_twin"],
        # per slot, index = slot number - 1
        "slot_boxes": [list(m["slot_boxes"][s]) for s in slots],
        "icon_boxes": [list(m["icon_bbox"][s]) for s in slots],
        "icon_ids": ids,
        "item_kinds": list(m["item_kinds"]),
        "alterations": list(m["alterations"]),
        # per near-twin: [pixels changed vs its own pair, that pair's ink pixels]
        "twin_change": list(m["twin_change"]),
        "readability": m["readability"],
        "render": dict(_RENDER, icon_svg_sha256={i: _svg_sha(lib, i) for i in sorted(set(ids))}),
    }
    core = sorted([i, k] for i, k in zip(ids, m["item_kinds"]))
    return api.Item(
        question=question,
        answer=answer,
        image=img,
        meta=meta,
        core_key=digest({"task": TASK, "items": core}),
        answer_type="pair-list",
    )
