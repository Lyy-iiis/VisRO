"""reconstruction -- polyomino assembly, five tiers on the target-rigidity dial.

Generator: generator/ (tiers.TIERS, the target sampler, the carving sampler,
the tiling search, every gate and the renderer).  The item builder is
`decoys.build(rng, tiers.cfg_for(tier))`; it keeps every gate: gate 0 (all 2n
piece shapes distinct), gate R (target connected and spanning its full W x H
box), gate A (the true set is found inside solve_band), gate 1 (every wrong
option refuted under rotation-only inside refute_band), gate 2 (every wrong
option refuted again with mirroring allowed), signature-matched decoys, 2-2
balanced slots, distinct option sets.  Rendering is the generator's own
`render.compose` on the 840x532 "top4" canvas.

Selection rule: the first builder draw whose option panels get >= 21 px per
cell.

  * decoys: every decoy comes from other independent carvings of the same
    target into the same sizes, signature-matched per slot, with an acceptance
    weight that makes the true piece and its decoy exchangeable.  On
    xhard/xxhard the decoy also matches the true piece's outline-corner count
    and bays count, and a posterior-balance step (decoys.BALANCE) cancels what
    the tiling gates' own selection leaves on 20 picture-only features.
  * QUESTION: states that the top shape and the option pieces are drawn at
    different scales (target cells 18-36 px, panel cells >= 21 px) and that
    every piece must be used exactly once.
  * meta: every piece is stored normalised; the true placement is stored
    explicitly as `solution`, its pieces in the drawing order of the correct
    panel.
  * core_key: canonical under the 8 symmetries of the whole picture.
  * rng: random.Random(<visro derived seed>), so tiers and splits never share a
    stream.
"""
from __future__ import annotations

import random

from visro import api
from visro.seeding import digest

from tasks.planar_transformations.reconstruction.generator import decoys as _C, render as _R, tiers as _T

TASK = "reconstruction"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

# Canvas, layout and legibility floor (fixed).  840 * 532 = 446880 px, both sides multiples of 28.
W, H, FRAC = 840, 532, 0.30
PX_FLOOR = 21
MAX_DRAWS = 4000

QUESTION = (
    "Which option's coloured pieces, each used exactly once, tile the top shape exactly with no "
    'overlap or leftover cells? Pieces may be moved and turned by 90 degrees but not flipped; '
    'options may be drawn at another scale. Answer with the letter.'
)
CHOICES = ["A", "B", "C", "D"]


def _shapes_txt(tier):
    return "{" + ", ".join(f"{w}x{h}/{c}" for w, h, c in _T.shapes_for(tier)) + "}"


_KEY_TXT = {"sig": "(area, perimeter, sorted bbox) signature",
            "sig+corners": "(area, perimeter, sorted bbox) signature and outline-corner count",
            "sig+corners+bays": "(area, perimeter, sorted bbox) signature, outline-corner count and bays "
                                "count (connected empty regions inside the piece's bbox)"}


def _tier_line(tier):
    c = _T.TIERS[tier]
    lo, hi = c["refute_band"]
    return (f"target = W x H rectangle with W*H-cells boundary cells bitten off (connected, spans the full box); "
            f"(W x H/cells) drawn uniformly on every builder draw from {_shapes_txt(tier)} (the accepted mix is "
            f"skewed by the gates and the px floor); rho = cells/(W*H) = {c['rho']} +- 0.002; "
            f"{c['n_parts']} pieces of {c['min_size']}-{c['max_size']} cells, the true set being one grow_split "
            f"carving of the target; 4 options = true set + 3 sets mixing true pieces with decoys (2-2 balanced "
            f"slots, 2n distinct shapes), slot j's decoy being a piece with the true piece's "
            f"{_KEY_TXT[_C.DECOY_KEY[tier]]}, drawn from the other kept carvings among {_C.POOL + 1} attempted "
            f"independent grow_split carvings of the same target into the same sizes (failed carvings and "
            f"carvings with two rotation-equal pieces dropped), under the true/decoy exchangeability "
            f"weight; every wrong option refuted exhaustively under rotation-only in {lo}-{hi} anchor-search "
            f"nodes and again with mirroring (<= {c['mirror_cap']:.0e} nodes); truth found in "
            f"<= {c['solve_band'][1]} nodes; option cells >= {PX_FLOOR} px on the {W}x{H} top4 canvas "
            f"(checked per decoy attempt before the tiling gates)" + _balance_txt(tier) +
            "; the first builder draw that passes everything is kept")


def _balance_txt(tier):
    stages = _C.BALANCE.get(tier)
    if not stages:
        return ""
    ks = ", ".join(f"{st['keep']:.2f}" for st in stages)
    return (f"; posterior balance in {len(stages)} stage{'s' if len(stages) > 1 else ''} "
            f"(decoys.BALANCE[{tier!r}]): each stage is a fixed picture-only scorer -- per-slot linear "
            f"weights on {len(_C.FEATURES)} piece-feature differences, fitted by conditional-logit maximum "
            f"likelihood on {stages[0]['n']} builds made without the balance (stage 1 on them as drawn, "
            f"each later stage on them weighted by the earlier stages' keep probabilities) -- giving the "
            f"four options scores s_k; for each stage in turn, a gate-passing attempt whose answer option "
            f"scores s_t is kept with probability exp(min_k s_k - s_t) = min_k q_k / q_t "
            f"(q = softmax(s)), and a rejection ends the builder draw, so the kept population is the "
            f"unbalanced one reweighted by the product of those probabilities (mean keep per stage {ks} "
            f"on the calibration builds).  Guarantee: where a stage's q equals the posterior of the "
            f"answer given those features in the population the stage is applied to, the answer of the "
            f"items it keeps is independent of them.  The stages are fitted estimates (finite sample, "
            f"linear model), so this holds only approximately, and other features and scorers are not "
            f"covered: picture-only scorers are not exactly at chance")


TIER_DEFINITIONS = {t: _tier_line(t) for t in api.TIERS}
assert tuple(_T.ORDER) == api.TIERS


# --------------------------------------------------------------------------- #
# small polyomino helpers (local; used for meta and the dedup key)
# --------------------------------------------------------------------------- #
def _norm(cells):
    cells = [tuple(c) for c in cells]
    x0 = min(x for x, _ in cells)
    y0 = min(y for _, y in cells)
    return tuple(sorted((x - x0, y - y0) for x, y in cells))


def _rot(cells):
    return _norm([(-y, x) for x, y in cells])


def _mir(cells):
    return _norm([(-x, y) for x, y in cells])


def _canon_rot(cells):
    c, best = _norm(cells), None
    for _ in range(4):
        best = c if best is None or c < best else best
        c = _rot(c)
    return best


def _d4_key(target, pieces):
    """min over the 8 symmetries g of the whole picture of
    (g(target), sorted rotation-canonical g(piece)).  Rotating or mirroring the
    whole puzzle gives an isomorphic item under the no-flip rule."""
    best = None
    for mirrored in (False, True):
        for r in range(4):
            def g(cells):
                c = _norm(cells)
                if mirrored:
                    c = _mir(c)
                for _ in range(r):
                    c = _rot(c)
                return c
            k = (g(target), tuple(sorted(_canon_rot(g(p)) for p in pieces)))
            best = k if best is None or k < best else best
    return best


def _cells(cells):
    return [list(c) for c in sorted(tuple(c) for c in cells)]


def _legible(display):
    return _R.panel_px(display, W, H, FRAC) >= PX_FLOOR


def _draw(tier, rng):
    cfg = _T.cfg_for(tier)
    cfg["decoy_key"] = _C.DECOY_KEY[tier]
    cfg["balance"] = _C.BALANCE.get(tier)
    for draw in range(1, MAX_DRAWS + 1):
        r = _C.build(rng, cfg, display_ok=_legible)
        if r is None or not _legible(r[2]):
            continue
        return r, draw
    raise api.Reject(f"no builder draw with panel cells >= {PX_FLOOR}px in {MAX_DRAWS} draws")


def render_item(target, display):
    """(question image, slots, panel px, target px)."""
    target = [tuple(c) for c in target]
    display = [[[tuple(c) for c in p] for p in panel] for panel in display]
    return _R.compose(target, display, W, H, FRAC)


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    if tier not in _T.TIERS:
        raise ValueError(f"reconstruction: unknown tier {tier!r}")
    rng = random.Random(seed)
    (target, opts, display, ci, stats), draws = _draw(tier, rng)

    # true pieces (target coordinates) in the drawing order of the correct
    # panel, so solution piece j == panel colour j.  All 2n shapes are
    # rotation-distinct (gate 0), so the match is unique.
    shown = [_canon_rot(p) for p in display[ci]]
    by_shape = {_canon_rot(p): p for p in opts[ci]}
    if sorted(shown) != sorted(by_shape) or len(by_shape) != len(shown):
        raise api.Reject("correct panel is not a re-posing of the true pieces")
    solution = [sorted(tuple(c) for c in by_shape[s]) for s in shown]

    img, _slots, px, tp = render_item(target, display)
    if px < PX_FLOOR:
        raise api.Reject(f"panel px {px} < {PX_FLOOR}")
    w, h = stats["rect"]
    meta = dict(
        tier=tier,
        rules=dict(rotations=True, flips=False, grid="unit square cells"),
        rect=[w, h], cells=len(target), rho=stats["rho"], n_parts=stats["n_parts"],
        target=_cells(_norm(target)),
        # display[k][j] = piece j of panel k exactly as drawn (normalised), and
        # drawn in colour PIECE[j]
        display=[[_cells(_norm(p)) for p in panel] for panel in display],
        # options[k] = the solver-space piece set behind panel k (normalised)
        options=[[_cells(_norm(p)) for p in o] for o in opts],
        correct_index=ci,
        solution=[_cells(p) for p in solution],
        render=dict(canvas=[W, H], layout="top4", target_frac=FRAC, panel_px=px, target_px=tp,
                    target_fill=list(_R.TARGET_FILL), piece_colors=[list(c) for c in _R.PIECE]),
        draws=draws,
        generator="generator/decoys.py:build",
        stats=stats,
    )
    key = _d4_key(target, opts[ci])
    return api.Item(
        question=QUESTION, answer=CHOICES[ci], image=img, meta=meta,
        core_key=digest(["reconstruction", [list(map(list, key[0])), [list(map(list, p)) for p in key[1]]]]),
        answer_type="choice", choices=list(CHOICES),
    )
