"""Item builder of every tier: decoys carved from the target, exchangeable with
the true pieces.

WHY
---
A decoy made by moving one cell of the true piece in its slot (signature-
matched) only has to fit somewhere inside the target, while the true pieces
are carved out of it, so on the rigid tiers the two populations are visibly
different: the true pieces have fewer outline corners, less rotational
symmetry, more legal placements, more cells on their bbox border and longer
straight runs, so a picture-only scorer that never tiles anything could pick
the true option far above chance (0.25).

WHAT THIS BUILDER DOES INSTEAD
------------------------------
The true set P and every decoy come from the SAME sampler: independent
carvings of the same target into the same multiset of piece sizes, by the
same `split.grow_split` the true pieces come from.

  1. target   `tiling.bite_rect` on a rectangle drawn from cfg["shapes"]
              (cfg["rect"] when the tier has no shape list),
     sizes    `split.sample_sizes`.
  2. carvings K = cfg["pool"] + 1 independent `grow_split` carvings of that
     target into those sizes are attempted; failed carvings and carvings
     with two rotation-equal pieces are dropped (the true set must have n
     distinct shapes, so the pool must follow the same law); fewer than 2
     kept carvings end the draw.
  3. truth    carvings are tried in turn as the true set P; the other carvings
     form the decoy pool.  Slot j's decoy is drawn uniformly (with
     multiplicity) from pool pieces with the same class key whose shape is
     none of P's shapes.  The key is sig() = (area, perimeter, sorted bbox);
     on xhard/xxhard it is (sig, outline-corner count, bays count) -- see
     DECOY_KEY.
  4. exchangeability.  Drawing the decoy "from the class, minus the true
     shape" alone would make the true piece the MORE COMMON shape of its pair:
     P(true=A, decoy=B) = f(A) f(B) / (1 - f(A)).  So candidate P is accepted
     with probability  prod_j (1 - f(banned in class of slot j)),  which turns
     the pair law into f(A) f(B): symmetric in A and B.  A decoy-decoy clash
     (two slots drawing one shape) ends the draw for the same reason.
  5. gates    every gate is applied to the resulting options:
     gate 0 (2n distinct shapes), distinct option sets, gate A (truth found
     inside solve_band), gate 1 (rotation-only refutation inside
     refute_band), gate 2 (refuted again with mirroring), the 2-2 balanced
     slot groups, the answer shuffle and the per-panel re-posing and shuffle.
     The answer position, poses and piece order are drawn before gates 1/2 so
     the caller's legibility check (display_ok) can skip an attempt without
     running the tiling searches.
  6. balance  (xhard, xxhard only) the gate asymmetry explained below cannot
     be removed at its source, so its remainder is cancelled.  BALANCE[tier]
     is a list of stages; each holds a fixed per-slot linear scorer w on the
     20 picture-only piece features in FEATURES, which gives the four
     options scores s_k.  For each stage in turn a gate-passing attempt whose
     true option scores s_t is kept with probability exp(min_k s_k - s_t)
     (= min_k q_k / q_t, q = softmax(s)), and a rejection ENDS the builder
     draw.  So the kept population is exactly the unbalanced population
     (first gate-passing attempt of a draw) reweighted by that probability,
     which is the population the stages were fitted on; where q is the
     unbalanced posterior of the answer given the features, the kept items'
     answer is independent of those features.

WHY THE RIGID TIERS ALSO MATCH CORNERS AND BAYS
-----------------------------------------------
Before the gates the true piece and its decoy are exchangeable.  The gates
are not symmetric, and cannot be: with 4 options and 2-2 balanced slots each
decoy sits in TWO wrong options and each true piece in ONE, so rejecting a
wrong option that tiles (or needs too much search) removes smooth, easily
tiled decoys about twice as often as smooth true pieces; with sig-only
classes the true pieces end up with fewer outline corners than their decoys
on xhard (not at easy/medium/hard, where the gates reject far less).
Matching the corner count per slot removes the main channel of that
selection on the two rigid tiers.  The next one is bays (connected empty
regions inside the piece's bbox): unbalanced true pieces have fewer bays
than their decoys, so the decoys match the bays count too.  The rest goes
through correlated features (C4 order, neighbour-degree counts, bbox-border
cells, longest run, row/column profile, placements touching the target's
corners and edges).  That remainder is what step 6 cancels, on the two
rigid tiers only: on the other three a learned scorer is at chance on
unbalanced builds, and balancing on noise would only add signal.

Decoys are carved from the target, so "individually placeable inside the
target" holds by construction, and signature profiles stay identical across
the four options.
"""
from __future__ import annotations

import math

from . import split as S, tiling as F
from .base import generate as G
from .tiling import tile_search, sig

POOL = 40          # decoy-pool carvings per target (K = POOL + 1 carvings)

# Decoy class key per tier.  "sig" = (area, perimeter, sorted bbox).  On the
# two rigid tiers the tiling gates reject most decoy attempts (xhard: ~1/3 by
# gate 1 and ~3/4 of the rest by gate 2), and because every decoy sits in two
# refuted mixes but every true piece in only one, that rejection removes
# smooth decoys about twice as often as smooth true pieces, so the accepted
# true pieces would have fewer outline corners than their decoys.  Matching
# the outline-corner count as well closes that channel at its source.  The
# same selection also leaves the true pieces with fewer "bays" (connected
# empty regions inside the piece's bounding box), so the bays count is
# matched too.
DECOY_KEY = {"easy": "sig", "medium": "sig", "hard": "sig",
             "xhard": "sig+corners+bays", "xxhard": "sig+corners+bays"}


def outline_corners(cells):
    """Vertices of the piece outline (a pinch point counts twice)."""
    s = set(cells)
    n = 0
    for vx, vy in {(x + dx, y + dy) for x, y in s for dx in (0, 1) for dy in (0, 1)}:
        q = ((vx - 1, vy - 1) in s, (vx, vy - 1) in s, (vx - 1, vy) in s, (vx, vy) in s)
        k = sum(q)
        n += 1 if k in (1, 3) else (2 if k == 2 and q[0] == q[3] else 0)
    return n


def bays(cells):
    """Connected components (4-neighbour) of the empty cells inside the
    piece's bounding box."""
    s = set(G.normalize(cells))
    W = max(x for x, _ in s) + 1
    H = max(y for _, y in s) + 1
    empty = {(x, y) for x in range(W) for y in range(H)} - s
    seen, n = set(), 0
    for e in sorted(empty):
        if e in seen:
            continue
        n += 1
        stack = [e]
        seen.add(e)
        while stack:
            x, y = stack.pop()
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if q in empty and q not in seen:
                    seen.add(q)
                    stack.append(q)
    return n


def class_key(p, key="sig"):
    if key == "sig":
        return sig(p)
    if key == "sig+corners":
        return (sig(p), outline_corners(p))
    if key == "sig+corners+bays":
        return (sig(p), outline_corners(p), bays(p))
    raise ValueError(f"unknown decoy key {key!r}")


# --------------------------------------------------------------------------- #
# posterior balance (step 6)
# --------------------------------------------------------------------------- #
# Picture-only piece features: 9 basic shape statistics and 11 finer features
# that the two-stage rank balance does not control on its own.  Every feature is a
# function of the drawn piece and the drawn target only, and invariant under
# rotation, so it is the same number for the piece as carved and as shown.
FEATURES = ("corners", "c4", "logplace", "deg1", "deg2", "deg3", "deg4", "border", "maxrun",
            "n2x2", "mirsym", "inertia", "logcornerplace", "logedgeplace", "fullrows", "notches",
            "deg2bent", "diagpairs", "bays", "rowvar")

# Per tier, a list of stages applied in order.  A stage is a fixed per-slot
# linear scorer w on the 20 feature differences (true minus decoy).  Its
# option scores s_k = sum_j (+-) w.d_j / 2 (+ when panel k holds slot j's true
# piece) define the 4-way posterior q_k = softmax(s)_k that panel k is the
# answer.  A gate-passing build whose answer is panel t is kept with
# probability
#         min_k q_k / q_t  =  exp(min_k s_k - s_t)
# and a rejection ENDS the builder draw (a fresh target and carvings follow),
# so the kept population is exactly the unbalanced one reweighted by that
# probability.  Where q is the true posterior given the features, the kept
# items have P(answer = k | picture features) = q_k * min q / q_k = const,
# i.e. no function of those features beats chance.
#   stage 1: w fitted by maximum likelihood of the 4-way choice (conditional
#            logit, L2 0.05 on standardised weights) on 900 unbalanced builds
#            per tier.
#   stage 2: the same, on those builds weighted by their stage-1 keep
#            probability.  In xhard split-half tests
#            on the calibration builds (fit on 450, balance the other 450) a
#            20-feature scorer trained on the balanced half scored 0.274
#            after stage 1 and 0.263 after stage 2, and one trained on the
#            fitting half reweighted by its own keep 0.277 -> 0.249.
#   keep:    the stage's mean keep probability on its calibration population.
# Calibration builds: adapter _draw with BALANCE emptied, rng seeds
# 7100000-7100899 (xhard) and 7110000-7110899 (xxhard), i.e. records of
# decoys.build output.  Only the two rigid tiers leak
# (see the module docstring); the others are not balanced.
BALANCE = {
    "xhard": [
        dict(w=(0.0, -0.03855, -0.00684, 0.02497, -0.01392, 0.02873, -0.09378, 0.51268, -0.02135, 0.0,
                -0.04808, -0.98862, -0.02967, 0.11533, -0.08022, -0.30584, -0.01249, 0.0, 0.0, 0.22507),
             n=900, keep=0.54),     # stage 1: 900 unbalanced builds
        dict(w=(0.0, 0.03318, -0.10523, 0.00073, 0.00034, -0.00347, 0.03275, 0.04052, -0.02048, 0.0,
                0.04366, 0.0559, 0.16868, -0.01366, -0.0089, -0.02813, -0.00036, 0.0, 0.0, 0.05064),
             n=900, keep=0.94),     # stage 2: the same builds weighted by stage-1 keep (n_eff 711)
    ],
    "xxhard": [
        dict(w=(0.0, 0.02244, -0.03512, 0.01099, -0.00915, 0.03032, -0.20509, 0.46008, 0.03986, 0.0,
                -0.13852, -0.63944, 0.26569, -0.03547, 0.0039, -0.28064, -0.00549, 0.0, 0.0, 0.19683),
             n=900, keep=0.57),     # stage 1: 900 unbalanced builds
        dict(w=(0.0, 0.01079, -0.01386, -0.00735, 0.00271, -0.00057, -0.0402, 0.02421, -0.05832, 0.0,
                -0.01096, -0.05137, 0.09915, -0.01534, -0.01832, -0.06791, 0.00367, 0.0, 0.0, 0.07172),
             n=900, keep=0.95),     # stage 2: the same builds weighted by stage-1 keep (n_eff 736)
    ],
}


def target_context(target):
    """(target cells, cells with >= 2 outside neighbours (corners), cells
    with >= 1 outside neighbour (edge))"""
    tset = set(map(tuple, target))
    out = {c: sum((c[0] + dx, c[1] + dy) not in tset for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
           for c in tset}
    return tset, {c for c, k in out.items() if k >= 2}, {c for c, k in out.items() if k >= 1}


def piece_features(p, tset, tcorner=None, tedge=None):
    """The FEATURES of piece p (any pose) inside target tset."""
    if tcorner is None or tedge is None:
        tset, tcorner, tedge = target_context(tset)
    s = set(G.normalize(p))
    W = max(x for x, _ in s) + 1
    H = max(y for _, y in s) + 1

    def nb(x, y):
        return [(x + dx, y + dy) in s for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]

    deg = [0] * 5
    bent = 0
    for x, y in s:
        r, l, d, u = nb(x, y)
        deg[r + l + d + u] += 1
        if r + l + d + u == 2 and not ((r and l) or (d and u)):
            bent += 1
    border = sum(1 for x, y in s if x in (0, W - 1) or y in (0, H - 1))
    run = 0
    for y in range(H):
        r = 0
        for x in range(W):
            r = r + 1 if (x, y) in s else 0
            run = max(run, r)
    for x in range(W):
        r = 0
        for y in range(H):
            r = r + 1 if (x, y) in s else 0
            run = max(run, r)
    poses = G.orientations(frozenset(s), True)
    place = cplace = eplace = 0
    for o in poses:
        for tx, ty in tset:
            cells = [(tx + a, ty + b) for a, b in o]
            if all(c in tset for c in cells):
                place += 1
                cplace += any(c in tcorner for c in cells)
                eplace += any(c in tedge for c in cells)
    n2 = sum(1 for x, y in s if (x + 1, y) in s and (x, y + 1) in s and (x + 1, y + 1) in s)
    mirrored = G.normalize((-x, y) for x, y in s)
    mirsym = int(G.canon(mirrored, True) == G.canon(s, True))
    cx = sum(x for x, _ in s) / len(s)
    cy = sum(y for _, y in s) / len(s)
    inertia = sum((x - cx) ** 2 + (y - cy) ** 2 for x, y in s) / len(s)
    fullrows = (sum(1 for y in range(H) if all((x, y) in s for x in range(W))) +
                sum(1 for x in range(W) if all((x, y) in s for y in range(H))))
    empty = [(x, y) for x in range(W) for y in range(H) if (x, y) not in s]
    notches = sum(1 for x, y in empty if sum(nb(x, y)) >= 3)
    diag = sum(1 for x, y in s for dx in (1, -1) if (x + dx, y + 1) in s)
    rows = [sum(1 for x in range(W) if (x, y) in s) for y in range(H)]
    cols = [sum(1 for y in range(H) if (x, y) in s) for x in range(W)]
    rowvar = _var(rows) + _var(cols)
    return [outline_corners(s), 4 // len(poses), math.log1p(place),
            deg[1], deg[2], deg[3], deg[4], border, run,
            n2, mirsym, inertia, math.log1p(cplace), math.log1p(eplace), fullrows, notches,
            bent, diag, bays(s), rowvar]


def _var(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / len(v)


def option_scores(target, parts, decoy, grp, w):
    """Scores of the four UNSHUFFLED options (0 = the true set, k = slots with
    grp == k true, the rest decoys) under the per-slot scorer w."""
    ctx = target_context(target)
    z = [sum(a * (b - c) for a, b, c in zip(w, piece_features(p, *ctx), piece_features(q, *ctx)))
         for p, q in zip(parts, decoy)]
    s0 = sum(z) / 2
    return [s0] + [sum(z[j] / 2 if grp[j] == k else -z[j] / 2 for j in range(len(z))) for k in (1, 2, 3)]


def keep_probability(scores):
    """min_k q_k / q_true with q = softmax(scores), the true option first."""
    return math.exp(min(scores) - scores[0])


def carvings(rng, target, sizes, rot, k, key="sig"):
    """k independent grow_split carvings; each -> (parts, canons, class keys)."""
    out = []
    for _ in range(k):
        q = S.grow_split(rng, target, sizes)
        if q is None:
            continue
        cs = [G.canon(p, rot) for p in q]
        if len(set(cs)) != len(cs):
            continue
        out.append((list(q), cs, [class_key(p, key) for p in q]))
    return out


def decoy_classes(carv, i):
    """Decoy pool for candidate truth carv[i]: per slot, the pool pieces with
    the slot's class key and a shape that is not one of the truth's shapes,
    plus the exchangeability acceptance weight.  -> (weight, [entries per slot])."""
    parts, cs, sg = carv[i]
    pool = {}
    for k, (q, qc, qs) in enumerate(carv):
        if k == i:
            continue
        for p, c, s in zip(q, qc, qs):
            pool.setdefault(s, []).append((c, p))
    banned = set(cs)
    w, free = 1.0, []
    for s in sg:
        ent = pool.get(s, ())
        fr = [e for e in ent if e[0] not in banned]
        if not fr:
            return 0.0, None
        w *= len(fr) / len(ent)
        free.append(fr)
    return w, free


def balanced_groups(rng, n):
    """slot -> which option (1/2/3) carries the TRUE piece there; as even as
    possible, all three groups non-empty."""
    base = ([1, 2, 3] * ((n + 2) // 3))[:n]
    for _ in range(200):
        rng.shuffle(base)
        if len(set(base)) == 3:
            return list(base)
    return base


def propose(rng, cfg):
    """Steps 1-4: target, sizes, carvings, and the first carving accepted as
    the true set under the exchangeability weight.
    -> None | (target, (w, h, ncells), carvings, truth index, weight, free)"""
    rot = cfg["rotations"]
    shapes = cfg.get("shapes")
    if shapes:
        w, h, ncells = shapes[rng.randrange(len(shapes))]
    else:
        (w, h), ncells = cfg["rect"], cfg["cells"]
    target = F.bite_rect(rng, w, h, w * h - ncells)
    if target is None:
        return None
    sizes = S.sample_sizes(rng, ncells, cfg["n_parts"], cfg["min_size"], cfg["max_size"])
    carv = carvings(rng, target, sizes, rot, cfg.get("pool", POOL) + 1, cfg.get("decoy_key", "sig"))
    if len(carv) < 2:
        return None
    for i in range(len(carv)):
        wgt, free = decoy_classes(carv, i)
        if wgt > 0 and rng.random() < wgt:
            return target, (w, h, ncells), carv, i, wgt, free
    return None


def draw_decoys(rng, free):
    """One decoy per slot, uniform (with multiplicity) over the slot's free
    class.  None on a decoy-decoy clash: the caller ends the draw, because
    redrawing would bias which shape of a pair is the true one."""
    decoy, used = [], set()
    for fr in free:
        c, p = fr[rng.randrange(len(fr))]
        if c in used:
            return None
        used.add(c)
        decoy.append(G.normalize(p))
    return decoy


def build(rng, cfg, display_ok=None):
    """One builder draw -> (target, options, display, correct index, stats) or
    None.  `display_ok(display)` (the adapter's 21 px legibility floor) is
    checked on each attempt's display BEFORE the tiling gates: it depends only
    on the pieces' bounding boxes and random poses, which a true piece and its
    decoy share, so it cannot tell them apart, and it saves the gate searches
    of displays that would be thrown away."""
    rot = cfg["rotations"]
    prop = propose(rng, cfg)
    if prop is None:
        return None
    target, (w, h, ncells), carv, ci_carving, wgt, free = prop
    parts = carv[ci_carving][0]
    n = len(parts)

    # gate A -- the truth is findable (cheap, and independent of the decoys)
    slo, shi = cfg["solve_band"]
    ok, tnodes, tcap = tile_search(target, parts, rot, False, cap=shi)
    if not ok or tcap or tnodes < slo:
        return None

    lo, hi = cfg["refute_band"]
    skipped_px = 0
    for _attempt in range(cfg["decoy_attempts"]):
        decoy = draw_decoys(rng, free)
        if decoy is None:
            return None                  # decoy-decoy clash ends the draw

        # gate 0 -- all 2n shapes canonically distinct
        if len({G.canon(p, rot) for p in parts} |
               {G.canon(d, rot) for d in decoy}) != 2 * n:
            continue

        grp = balanced_groups(rng, n)
        opts = [list(parts)]
        for k in (1, 2, 3):
            opts.append([parts[j] if grp[j] == k else decoy[j] for j in range(n)])

        keys, dup = set(), False
        for o in opts:
            key = G.set_key(o, rot)
            if key in keys:
                dup = True
                break
            keys.add(key)
        if dup:
            continue

        # answer position, poses and piece order of every panel
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[o] for o in order]
        ci = order.index(0)
        display = []
        for pset in shuffled:
            shown = []
            for p in pset:
                o = G.orientations(p, rot) if rot else [G.normalize(p)]
                shown.append(o[rng.randrange(len(o))])
            rng.shuffle(shown)
            display.append(shown)
        if display_ok is not None and not display_ok(display):
            skipped_px += 1
            continue

        # gate 1 -- rotations-only uniqueness + refutation-work band
        work, bad = [], False
        for O in opts[1:]:
            solved, nodes, capped = tile_search(target, O, rot, False, cap=hi)
            if solved or capped or not (lo <= nodes <= hi):
                bad = True
                break
            work.append(nodes)
        if bad:
            continue

        # gate 2 -- uniqueness with MIRROR allowed
        mwork = []
        for O in opts[1:]:
            solved, nodes, capped = tile_search(target, O, rot, True, cap=cfg["mirror_cap"])
            if solved or capped:
                bad = True
                break
            mwork.append(nodes)
        if bad:
            continue

        # step 6 -- posterior balance: for each stage of BALANCE[tier] in
        # turn, keep with probability exp(min_k s_k - s_true); a rejection
        # ends the draw, so the kept population is the unbalanced one
        # reweighted by the keep probability (what the stages were fitted on)
        ranks, keeps = [], []
        for stage in cfg.get("balance") or ():
            sc = option_scores(target, parts, decoy, grp, stage["w"])
            kp = keep_probability(sc)
            ranks.append(1 + sum(1 for v in sc[1:] if v > sc[0] + 1e-9))
            keeps.append(round(kp, 4))
            if rng.random() >= kp:
                return None

        sigs = [sorted(sig(p) for p in o) for o in shuffled]
        stats = dict(cells=len(target), n_parts=n, rect=[w, h],
                     rho=round(ncells / (w * h), 4),
                     bites=w * h - ncells,
                     groups=grp,
                     hamming=[sum(1 for j in range(n) if grp[j] != k) for k in (1, 2, 3)],
                     refute_nodes=sorted(work),
                     refute_nodes_mirror=sorted(mwork),
                     solve_nodes=tnodes,
                     part_sizes=sorted(len(p) for p in parts),
                     decoy_rule="carved",
                     decoy_key=cfg.get("decoy_key", "sig"),
                     carvings=len(carv),
                     truth_carving=ci_carving,
                     accept_weight=round(wgt, 6),
                     decoy_attempt=_attempt,
                     attempts_below_px=skipped_px,
                     balance_rank=ranks or None,
                     balance_keep=keeps or None,
                     sig_profiles_identical=all(s == sigs[0] for s in sigs))
        return target, shuffled, display, ci, stats
    return None
