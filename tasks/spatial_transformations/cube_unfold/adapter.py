"""cube_unfold: which numbered isometric cubes are views of the cube a net folds into.

Generator: generator/generate.py.  The
adapter calls generate.generate_one(seed, index, size, tier) with the release
seed, which seeding.derive_seed keys by (task, tier, split, base seed, index),
so no two tiers or splits share a net, labels or palette.

Every tier's candidates come from the 24-rotation orbit of the folded cube
(positives) and tampered views verified to lie outside it (negatives).
"""
from __future__ import annotations

from visro import api
from visro.seeding import digest

from tasks.spatial_transformations.cube_unfold.generator import generate

TASK = "cube_unfold"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

_COMMON = ("net = one of the 11 hexomino cube nets (fixed drawing, random face order); candidates = distinct "
           "corner views (top/right/left) of the folded cube from its 24 rotations plus distinct negatives proven "
           "outside that orbit; >= 1 match and >= 1 non-match; answer = increasing comma list of matching numbers")
TIER_DEFINITIONS = {
    "easy": ("448 px; 4 candidates (1-2 match); symbols = colour + centred white dot (rotation-invariant), colours "
             "one of 3 six-colour palettes with pairwise CIEDE2000 >= 29; negatives = swap two shown faces or "
             "replace one with an unshown face; " + _COMMON),
    "medium": ("448 px; 6 candidates (1-3 match); colour + dot as easy; negatives = swap/replace as easy; "
               + _COMMON),
    "hard": ("588 px; 8-9 candidates (2-4 match); colour + one distinct orientation glyph per face (arrow, tri, ell, "
             "tee, flag, key; large: scale 0.44, strokes >= 0.24); negatives = swap, replace, or turn one glyph "
             "1-3 quarter turns (mode drawn 1:1:2); page must pass cube_readable (<= 40 redraws); " + _COMMON),
    "xhard": ("588 px; 10 candidates (2-5 match); colour + glyph as hard; every negative is a legal view with "
              "exactly one glyph turned 1-3 quarter turns (face arrangement always legal); cube_readable gate; "
              + _COMMON),
    "xxhard": ("588 px; 12 candidates (3-6 match); as xhard but all six faces the same colour (65,85,112), so "
               "faces are told apart only by their glyphs; cube_readable gate; " + _COMMON),
}
SIZES = {"easy": 448, "medium": 448, "hard": 588, "xhard": 588, "xxhard": 588}
assert generate.GENERATOR_VERSION == "physical-fold", generate.GENERATOR_VERSION


def core_key(meta: dict) -> str:
    """Content-canonical key, invariant to symbol relabelling and candidate order.

    Faces are renamed by the rank of their net cell (so palette, colour order
    and glyph assignment drop out) and the candidates become a sorted multiset
    of (top, right, left) views in those names; steps count only on glyph tiers.
    The tier is left out on purpose, so one puzzle cannot appear in two tiers.
    """
    cells = [tuple(c) for c in meta["net_cells"]]
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    norm = [(r - r0, c - c0) for r, c in cells]
    rank = {fid: sorted(norm).index(cell) for fid, cell in enumerate(norm)}
    oriented = bool(meta["oriented"])
    views = sorted(tuple((rank[v[s][0]], v[s][1] if oriented else 0) for s in ("top", "right", "left"))
                   for v in (c["view"] for c in meta["candidates"]))
    return digest(["cube_unfold", oriented, sorted(norm), views])


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    rec, image = generate.generate_one(seed, index, SIZES[tier], tier)
    if rec["difficulty"] != tier or image.size != (SIZES[tier], SIZES[tier]):
        raise api.Reject("generator returned the wrong tier or size")
    meta = dict(rec["meta"])
    meta["cot"] = rec["cot"]
    meta["generator"] = "generator/generate.py:generate_one"
    meta["view_convention"] = ("view {slot: [face, step]}: face = index into net_cells/face_colors/face_glyphs; "
                               "slots top/right/left have outward normals +Z/+Y/+X; step = right-hand quarter "
                               "turns of the symbol's up about the outward normal from slot_ref_up "
                               "(top +Y, sides +Z); net cell (r, c) folds with its drawn side out and glyph up = "
                               "north (-r)")
    return api.Item(question=rec["question"], answer=rec["answer"], image=image, meta=meta,
                    core_key=core_key(meta), answer_type="token-list")
