"""Maze detour-excess builders of the hard / xhard / xxhard boards.

    grid        21x21 cell 21 (corridor 19px thin / 18px medium)
    D freeze    [14,20], and per-slot D/R plans SHARED across tiers (closer/kth
                D=20, within R in {16,18,20}), so D carries zero tier bits and
                E_frac is compared at one radius
    X bands     {4,6}/{8,10,12}/{14,16} (X=16 lives only on within slots --
                closer/kth cap at X <= D-2-4 = 14)
    within      8 markers A..H, answer space 0..8
    balancing   key W in [3,4]; per-task record Wbar windows; per-slot E_frac
                sub-windows shared across tiers

X = d_corr(key, target) - L1(key, target), measured on the key candidate:
closer = the winner, kth = the k-th closest (the answer), within = the outside
marker whose straight-line distance lies (d in [R+2, R+4]).  X is even by grid
parity.  The separation floor (SEP[0] = 0.12 * diagonal = 3.82 cells Euclid at
this grid => L1 >= 4) caps X at d_key - 4; with the key at 18 (closer winner,
kth answer) that is 14, and only the within key (d up to R+4 = 24) reaches 16.

Uses board_law.py (board law, separation, marker rules, renderer) and
corridor_questions.py (question templates + their renderer).
"""
from __future__ import annotations

import math
import random

from . import board_law, corridor_questions, lattice, questions

CELL = 21                       # full-size render: 21*21 + stroke ~= 443 <= 448
GRID = (21, 21, CELL)
BANDS_X = {"hard": (4, 6), "xhard": (8, 12), "xxhard": (14, 16)}
D_FREEZE = (14, 20)             # the anti-sprawl commitment, both ends binding
K_LAW = (10, 12, 14)            # shared by every tier (visible => must not vary)
# W DISCIPLINE (W = walls crossed by a marker's straight segment).  At the
# X=14/L1=4 corner a key with W in [3,4] barely exists (a detour of 18 does not
# require the 4-edge straight line to be mostly walled), and decoy windows above
# the key's would make the key the unique min-W candidate.  So one uniform rule,
# shared by every tier and task:
#     key W in [2,4]  (floor: the key's straight line stays non-trivially
#                      blocked)  +  EVERY other marker W <= 4.
# This compresses record Wbar/Wmax into one band (K4) and makes min-W a label
# coin flip (wallx_min).
W_WIN = (2, 4)                  # key straight-segment wall crossings
W_CAP = 4                       # every non-key marker, every task, every tier
EFRAC_WIN = (0.14, 0.30)        # K5 pre-balancing, at r* = 18/20/R per task
MARGIN = 2                      # adjacent ranks differ by >= this many cells
TIERS = ("hard", "xhard", "xxhard")
N_WITHIN = 8                    # within uses 8 labelled markers (A..H)
WITHIN_LABELS = "ABCDEFGH"

# ---- per-slot plans, identical columns shared across tiers ------------------
D_CLOSER = 20                   # forced by xxhard: winner at D-2=18 carries X=14
D_KTH = 20
R_WITHIN = (16, 18, 20)         # slots 7/8/9
# At this grid the separation floor is 0.12*diag = 3.82 cells, so
# an L1=4 marker exists ONLY axis-aligned ((4,0): 4.00 cells, 84.0px) -- the
# diagonal shapes ((2,2)=2.83, (3,1)=3.16) are under the floor.
# No lattice displacement lies in (3.82, 4.0), therefore nothing can be
# pixel-NEARER than an X=14 key, and a closer 'misl' loser (needs e <= 0.95 *
# 84.0 = 79.8px < 80.3px floor) is geometrically impossible at xxhard.
# (X=14 answers sit at exactly 84.0px with no pixel-nearer candidate at any
# depth.)  The xxhard plan therefore replaces 'misl' with a third exact-pixel TIE, and the kth
# rank rotation with exact-tie decoys whose LABELS carry the rotation (the
# rank cue tie-breaks alphabetically; planned so it hits ~1/2 -- same device
# as the closer tie labels).
PLAN_CLOSER = {
    "hard":   (("cons", "A"), ("tie", "B"), ("misl", "B"), ("tie", "A")),
    "xhard":  (("cons", "A"), ("tie", "B"), ("misl", "B"), ("tie", "A")),
    "xxhard": (("cons", "A"), ("tie", "B"), ("tie", "B"), ("tie", "A")),
}
PLAN_X = {
    "hard":   {"closer": (4, 6, 4, 6),     "mc": (6, 4, 6),    "count": (4, 6, 6)},
    # xhard kth j4 has k=2 (answer depth <= 16): X=12 there forces L1=4, the
    # degenerate axis shape -- swapped with the k=3 slot (answer at 17: X=12
    # gives L1=5, where (4,1)=4.12 lives above the floor and (4,0) can be a
    # strictly nearer decoy)
    "xhard":  {"closer": (8, 10, 12, 10),  "mc": (10, 8, 12),  "count": (12, 8, 10)},
    "xxhard": {"closer": (14, 14, 14, 14), "mc": (14, 14, 14), "count": (16, 16, 14)},
}

# relax ladder (attempt-staged; stage recorded per sample and reported).
# The kth pixel-rank enforcement and the key W window are NEVER relaxed:
# letting them drop at stage 2 fails K2/K3/K4 (euclid/l1_rank_1 0.857 on
# xxhard -- the answer becomes the pixel-nearest marker on the relaxed kth
# records; W=2 keys push the Wbar AUC to the 0.75 line).  Satisfiability is
# bought with construction (subset-based rank roles, see kth_x) and attempts,
# not with discipline.
#   stage 0 (<300): everything exact (planned X, cue class, E_frac bin)
#   stage 1 (300-449): X anywhere in band; E_frac full window
#   stage 2 (450-599): closer class may fall back [want, tie, cons]; the
#                      within outside pool widens to [R+2, R+8]
#   at EVERY stage: kth erank exact per plan; key W in W_WIN; marker W <= W_CAP
MAX_ATT = 600


def l1(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def crossings(walls, a, b):
    """Closed walls crossed by the straight segment centre(a)->centre(b), in
    cell units."""
    ax, ay = a[0] + 0.5 + 1e-4, a[1] + 0.5 + 2e-4
    bx, by = b[0] + 0.5 - 1.3e-4, b[1] + 0.5 - 0.7e-4
    n = max(2, int(40 * max(abs(bx - ax), abs(by - ay))))
    seq = []
    for i in range(n + 1):
        t = i / n
        c = (int(ax + (bx - ax) * t), int(ay + (by - ay) * t))
        if not seq or seq[-1] != c:
            seq.append(c)
    w = 0
    for p, q in zip(seq, seq[1:]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        if abs(dx) + abs(dy) == 1:
            w += lattice.wall_key(p, q) in walls
        else:
            r1 = (lattice.wall_key(p, (p[0] + dx, p[1])) in walls) + \
                 (lattice.wall_key((p[0] + dx, p[1]), q) in walls)
            r2 = (lattice.wall_key(p, (p[0], p[1] + dy)) in walls) + \
                 (lattice.wall_key((p[0], p[1] + dy), q) in walls)
            w += min(r1, r2)
    return w


def _board(lat, rng, K):
    got = board_law.build_board(lat, rng, K, [1.0], focus_random=True)
    if got is None:
        return None
    walls, comps, carver = got
    if len(comps) != 1:
        return None
    return walls, comps[0], carver


def _xw(d, c, t, walls):
    dd = d[c]
    return dd - l1(c, t), crossings(walls, c, t)


def _x_ok(X, want, band, stage):
    if stage == 0:
        return X == want
    return band[0] <= X <= band[1]


def _w_ok(W, stage):
    lo, hi = W_WIN            # never widened, at any stage
    return lo <= W <= hi


def _ef_ok(ef, bin_, stage):
    lo, hi = bin_ if stage == 0 else EFRAC_WIN
    return lo <= ef <= hi


# Plan hooks: which marker-ok cells may host a start / the target.
# The defaults accept every cell, so hard / xhard / xxhard build exactly as
# before; the edge-entrance boards set START_OK to the border cells and
# TARGET_OK to the interior for the duration of one build.
START_OK = lambda lat, c: True      # noqa: E731
TARGET_OK = lambda lat, c: True     # noqa: E731
# within hook: at least this many NON-key outside markers must look inside
# by straight line (L1 <= R-2), so "count the starts within R as the crow
# flies" is off by a varying amount.  0 = detour_core's behaviour, untouched.
WITHIN_FOOL = 0
# closer hook: which start is nearer by straight lattice distance (L1):
# None = detour_core (prefer an L1 tie, else |X_l - X_w| <= 2, so the winner is
# never farther by L1 and "min-L1" read the answer 0.67); "win" = the winner
# is nearer, "lose" = the loser is nearer (X_l >= X_w + 3), "tie" = equal.
CLOSER_L1 = None
# kth hook: the answer's rank by straight lattice distance among the four
# (1 = nearest); None = detour_core (unplanned, and "k-th by L1" read it 0.43).
KTH_L1_RANK = None
# wall hooks: the key's straight segment crosses W_WIN[0]..W_WIN[1] walls
# (2..4 on detour_core's tiers) but the other markers only need W <= W_CAP, so
# "the start whose straight line crosses more walls" leans to the answer.
# These floors put the other markers under the same lower bound.  0 = detour_core.
CLOSER_LOSER_WMIN = 0
KTH_DECOY_WMIN = 0


def closer_x(lat, rng, K, X_want, band, want_cls, want_lab, efbin, stage):
    """Two starts, one component, D=20 fixed: winner at 18 (the key), loser at
    20, margin=2.  Winner X per plan, W in W_WIN; loser X-matched to +-2.  The
    euclid cue class is judged on PIXEL distance with the base renderer's 5% tie
    window (judging on L1 makes misl unsatisfiable at high X: at xxhard the misl loser needs X=16 / L1=4 diagonal
    against an axis-aligned winner, which pixels can express and L1 cannot)."""
    D = D_CLOSER
    diag = math.hypot(*lat.image_size())
    got = _board(lat, rng, K)
    if got is None:
        return None
    walls, comp, carver = got
    ex = board_law.excess(lat, walls)
    cells = [c for c in comp if board_law.marker_ok(lat, walls, c)]
    if len(cells) < 8:
        return None
    rng.shuffle(cells)
    tcells = [c for c in cells if TARGET_OK(lat, c)]
    cells = [c for c in cells if START_OK(lat, c)]
    r = D - MARGIN
    for target in tcells[:14]:
        d = lattice.bfs_dist(target, lat.neighbors, walls)
        E = sum(1 for _c, dd in d.items() if dd <= r)
        if not _ef_ok(E / len(lat.cells), efbin, stage):
            continue
        wins = []
        for c in cells:
            if d.get(c) == r and board_law.sep_ok(lat, diag, c, target):
                X, W = _xw(d, c, target, walls)
                if _x_ok(X, X_want, band, stage) and _w_ok(W, stage):
                    wins.append((c, X, W))
        rng.shuffle(wins)
        for win, X_w, W_w in wins[:10]:
            classes = [want_cls] if stage < 2 else [want_cls, "tie", "cons"]
            en = lattice.distance_px(lat, win, target)
            for cls in classes:
                ls = []
                for c in cells:
                    if d.get(c) != D or c == win:
                        continue
                    if not (board_law.sep_ok(lat, diag, c, target)
                            and board_law.sep_ok(lat, diag, c, win)):
                        continue
                    X_l, W_l = _xw(d, c, target, walls)
                    dX = X_l - X_w
                    if CLOSER_L1 is None:
                        if abs(dX) > 2:
                            continue      # loser stays X-matched to the key
                    elif CLOSER_L1 == "win":
                        if not (-2 <= dX <= 1):
                            continue      # L1_l - L1_w = 2 - dX >= 1
                    elif CLOSER_L1 == "tie":
                        if dX != 2:
                            continue
                    elif not (3 <= dX <= 6):
                        continue          # "lose": L1_l < L1_w
                    if W_l > W_CAP or W_l < CLOSER_LOSER_WMIN:
                        continue          # uniform marker W cap; plan floor
                    ef_ = lattice.distance_px(lat, c, target)
                    ratio_ = abs(en - ef_) / max(en, ef_, 1.0)
                    got_cls = ("tie" if ratio_ < board_law.TIE_EPS else
                               "cons" if en < ef_ else "misl")
                    if got_cls == cls:
                        ls.append((c, X_l, W_l))
                if not ls:
                    continue
                # prefer X_l = X_w+2, i.e. L1_loser = L1_winner --
                # the winner is otherwise STRICTLY min-L1 on every closer
                # record (L1_l - L1_w = 2 + X_w - X_l >= 0) and l1_rank_1
                # reads it; an L1 tie makes that cue a label coin-flip
                tie_l1 = [t for t in ls if t[1] == X_w + 2] if CLOSER_L1 is None else []
                loser, X_l, W_l = rng.choice(tie_l1 if tie_l1 else ls)
                other = "B" if want_lab == "A" else "A"
                starts = sorted([(want_lab, win), (other, loser)])
                ef = lattice.distance_px(lat, loser, target)
                ratio = abs(en - ef) / max(en, ef, 1.0)
                return dict(
                    walls=walls, starts=starts, exit=target,
                    answer=want_lab,
                    path=lattice.shortest_path(win, target, lat.neighbors, walls),
                    lam=board_law.lam(ex, [c for c in comp if d.get(c, 1e9) <= r]),
                    rstar=r, E=E, carver=carver, ncomp=1, sub=0,
                    euclid_ratio=round(ratio, 4), cue_class=cls,
                    dnear=r, dfar=D,
                    D=D, margin=MARGIN, X=X_w,
                    X_all=[X_w, X_l], W_all=[W_w, W_l],
                    L1_all=[l1(win, target), l1(loser, target)],
                    dists=[r, D], key_cell=list(win),
                    key_W=W_w, key_L1=l1(win, target))
    return None


def kth_x(lat, rng, K, X_want, band, k, want_lab, want_erank, efbin, stage,
          tie_lab=None):
    """Four starts at 4 distinct distances (gaps in {2,3}), deepest = D = 20.
    The answer (k-th closest) is the key: X per plan, W in W_WIN, so its L1
    collapses toward the separation floor at xxhard.  The answer's PIXEL rank
    among the four is enforced exactly at stage 0/1 (rotated {2,3} per plan --
    without it the answer pins at euclid rank 2); stage 1
    keeps the rank enforcement but frees X inside the band; only the decoy
    fallback of the relax branch is used from stage 2."""
    D = D_KTH
    diag = math.hypot(*lat.image_size())
    got = _board(lat, rng, K)
    if got is None:
        return None
    walls, comp, carver = got
    ex = board_law.excess(lat, walls)
    cells = [c for c in comp if board_law.marker_ok(lat, walls, c)]
    if len(cells) < 10:
        return None
    rng.shuffle(cells)
    tcells = [c for c in cells if TARGET_OK(lat, c)]
    cells = [c for c in cells if START_OK(lat, c)]
    for target in tcells[:12]:
        d = lattice.bfs_dist(target, lat.neighbors, walls)
        E = sum(1 for _c, dd in d.items() if dd <= D)
        if not _ef_ok(E / len(lat.cells), efbin, stage):
            continue
        for _try in range(4):
            gaps = [rng.choice((2, 3)) for _ in range(3)]
            window = [D - sum(gaps), D - gaps[1] - gaps[2], D - gaps[2], D]
            ans_d = window[k - 1]
            if stage == 0 and ans_d - X_want < 4:
                continue                  # plan needs L1 >= separation floor
            if ans_d - band[0] < 4:
                continue
            cands = []
            for c in cells:
                if d.get(c) == ans_d and board_law.sep_ok(lat, diag, c, target):
                    X, W = _xw(d, c, target, walls)
                    if _x_ok(X, X_want, band, stage) and _w_ok(W, stage):
                        cands.append((c, X, W))
            rng.shuffle(cands)
            # answers whose own pixel distance is at the geometric minimum
            # ((2,2) = 2.83px at the L1 floor) admit NO pixel-nearer decoy at
            # all; try the pixel-farther shapes first so the rank pattern has
            # room to exist
            cands.sort(key=lambda t: -lattice.distance_px(lat, t[0], target))
            js = sorted(jj for jj in range(4) if jj != k - 1)
            # Exact pixel-rank enforcement at EVERY stage, with the
            # (want_erank-1) "pixel-nearer" roles assigned over decoy-slot
            # SUBSETS.  At X=14 the answer sits AT the L1 floor and the only
            # cells that can beat it in pixels are other floor cells -- which
            # live DEEP (X = d - 4 large), not shallow, so a shallow-slots-only
            # rule would leave the rank unsatisfiable there.
            import itertools as _it
            if want_erank is not None:
                subsets = list(_it.combinations(js, want_erank - 1))
                subsets.sort(key=lambda S: -sum(S) if X_want >= 12 else sum(S))
            else:
                # Tie mode (xxhard): nothing can be pixel-nearer
                # than an X=14 answer (see PLAN_CLOSER note) -- instead,
                # exactly one decoy sits at the answer's EXACT pixel
                # distance (another axis arm of the target) and the
                # alphabetical tie-break carries the rotation via its
                # planned label; the remaining decoys are strictly farther.
                # Shallow slot first: an arm cell at window[0] needs only a
                # natural X = d - 4.
                subsets = [(j,) for j in js]
                subsets.sort(key=lambda S: sum(S))
            for ans_c, X_a, W_a in cands[:8]:
                e_a = lattice.distance_px(lat, ans_c, target)
                L1_a = ans_d - X_a
                got_picks = None
                tie_slot = None
                for S in subsets:
                    picks = {k - 1: ans_c}
                    ok = True
                    for j in js:
                        pool = [c for c in cells
                                if d.get(c) == window[j] and c != ans_c
                                and board_law.sep_ok(lat, diag, c, target)
                                and all(board_law.sep_ok(lat, diag, c, p)
                                        for p in picks.values())
                                and KTH_DECOY_WMIN <= crossings(walls, c, target) <= W_CAP]
                        e_of = {c: lattice.distance_px(lat, c, target)
                                for c in pool}
                        if want_erank is None:
                            if j in S:
                                pool = [c for c in pool
                                        if abs(e_of[c] - e_a) <= 1e-6]
                            else:
                                pool = [c for c in pool
                                        if e_of[c] > e_a + 1e-6]
                            if not pool:
                                ok = False
                                break
                            picks[j] = rng.choice(pool)
                            continue
                        want_near = j in S
                        pool = [c for c in pool
                                if abs(e_of[c] - e_a) > 1e-6
                                and ((e_of[c] < e_a) == want_near)]
                        if want_near:
                            # prefer an L1-floor nearer decoy: pixel-nearer
                            # does NOT imply L1-nearer ((3,2)=3.61px is inside
                            # (4,0)=4.0px but L1 5 > 4), and without an L1 tie
                            # the answer stays strictly min-L1 (l1_rank_1)
                            floor = [c for c in pool
                                     if l1(c, target) <= L1_a]
                            if floor:
                                pool = floor
                        if not pool:
                            ok = False
                            break
                        picks[j] = rng.choice(pool)
                    if ok and len(picks) == 4:
                        if KTH_L1_RANK is not None:
                            l1_a = l1(ans_c, target)
                            l1r = 1 + sum(1 for j_, c_ in picks.items() if j_ != k - 1 and l1(c_, target) < l1_a)
                            if l1r != KTH_L1_RANK:
                                continue
                        got_picks = picks
                        tie_slot = S[0] if want_erank is None else None
                        break
                if got_picks is None:
                    continue
                picks = got_picks
                decoy_placed = want_erank is not None and want_erank >= 2
                cells4 = [picks[j] for j in range(4)]
                Ws = [crossings(walls, c, target) for c in cells4]
                labels = ["A", "B", "C", "D"]
                rest = [x for x in labels if x != want_lab
                        and (tie_slot is None or x != tie_lab)]
                rng.shuffle(rest)
                lab_of_rank = [None] * 4
                lab_of_rank[k - 1] = want_lab
                if tie_slot is not None:
                    lab_of_rank[tie_slot] = tie_lab
                ri = 0
                for j in range(4):
                    if lab_of_rank[j] is None:
                        lab_of_rank[j] = rest[ri]
                        ri += 1
                starts = sorted(zip(lab_of_rank, cells4))
                Xs = [d[c] - l1(c, target) for c in cells4]
                return dict(
                    walls=walls, starts=starts, exit=target, answer=want_lab,
                    path=lattice.shortest_path(ans_c, target, lat.neighbors, walls),
                    lam=board_law.lam(ex, [c for c in comp if d.get(c, 1e9) <= D]),
                    rstar=D, E=E, carver=carver, ncomp=1, sub=0,
                    comp_sizes=[len(comp)], target_comp=len(comp),
                    kth=k, dists=window,
                    euclid_rank=corridor_questions.euclid_rank(lat, cells4, target),
                    D=D, margin=min(gaps), X=X_a,
                    X_all=Xs, W_all=Ws,
                    L1_all=[l1(c, target) for c in cells4],
                    decoy_strict=decoy_placed, key_cell=list(ans_c),
                    tie_lab=(tie_lab if tie_slot is not None else None),
                    n_pixel_ties=sum(
                        1 for c in cells4 if c != ans_c and abs(
                            lattice.distance_px(lat, c, target) - e_a) <= 1e-6),
                    key_W=W_a, key_L1=ans_d - X_a,
                    ans_erank=1 + sum(
                        1 for c in cells4
                        if c != ans_c and lattice.distance_px(lat, c, target)
                        < lattice.distance_px(lat, ans_c, target)))
    return None


def within_x(lat, rng, K, R, X_want, band, ans, efbin, stage):
    """Eight starts A..H, how many within R corridor steps (R printed).
    Inside markers at d <= R-2, outside at d in [R+2, R+6] (stage 2: R+8):
    every marker at least MARGIN=2 from the boundary.  The key is an outside
    marker at d in [R+2, R+4] with the planned X and W in W_WIN; at X >= 4 its
    L1 <= R, i.e. it LOOKS inside by straight line.  Up to two inside
    alpha-decoys (L1 >= R-4) are preferred: they break inflated-L1 balls at
    the other end.  Answer space 0..8; the key is outside, so answers <= 7."""
    diag = math.hypot(*lat.image_size())
    got = _board(lat, rng, K)
    if got is None:
        return None
    walls, comp, carver = got
    ex = board_law.excess(lat, walls)
    cells = [c for c in comp if board_law.marker_ok(lat, walls, c)]
    if len(cells) < 12:
        return None
    rng.shuffle(cells)
    out_hi = R + 6 if stage < 2 else R + 8
    for target in cells[:12]:
        d = lattice.bfs_dist(target, lat.neighbors, walls)
        E = sum(1 for _c, dd in d.items() if dd <= R)
        if not _ef_ok(E / len(lat.cells), efbin, stage):
            continue
        keys = []
        for c in cells:
            dd = d.get(c)
            if dd is None or not (R + 2 <= dd <= R + 4):
                continue
            if not board_law.sep_ok(lat, diag, c, target):
                continue
            X, W = _xw(d, c, target, walls)
            if _x_ok(X, X_want, band, stage) and _w_ok(W, stage):
                keys.append((c, X, W))
        rng.shuffle(keys)
        for key_c, X_k, W_k in keys[:6]:
            inside_pool = [c for c in cells
                           if c != target and 2 <= d.get(c, 1e9) <= R - 2
                           and board_law.sep_ok(lat, diag, c, target)
                           and crossings(walls, c, target) <= W_CAP]
            out_pool = [c for c in cells
                        if c != key_c and R + 2 <= d.get(c, -1) <= out_hi
                        and board_law.sep_ok(lat, diag, c, target)
                        and crossings(walls, c, target) <= W_CAP]
            if len(inside_pool) < ans or len(out_pool) < N_WITHIN - 1 - ans:
                continue
            n_fool = min(WITHIN_FOOL, N_WITHIN - 1 - ans)
            if n_fool and sum(1 for c in out_pool if l1(c, target) <= R - 2) < n_fool:
                continue
            for _pack in range(8):        # greedy packing, several reshuffles
                rng.shuffle(inside_pool)
                alpha = [c for c in inside_pool if l1(c, target) >= R - 4]
                ins = []
                for c in alpha[:2]:
                    if len(ins) < ans and all(board_law.sep_ok(lat, diag, c, p)
                                              for p in ins):
                        ins.append(c)
                for c in inside_pool:
                    if len(ins) >= ans:
                        break
                    if c not in ins and all(board_law.sep_ok(lat, diag, c, p)
                                            for p in ins):
                        ins.append(c)
                if len(ins) < ans:
                    continue
                rng.shuffle(out_pool)
                if n_fool:                # fooling outside markers first, then the rest
                    fool = [c for c in out_pool if l1(c, target) <= R - 2]
                    out_pool = fool + [c for c in out_pool if c not in fool]
                outs = [key_c]
                for c in out_pool:
                    if len(outs) >= N_WITHIN - ans:
                        break
                    if all(board_law.sep_ok(lat, diag, c, p) for p in outs + ins):
                        outs.append(c)
                if len(outs) < N_WITHIN - ans:
                    continue
                if n_fool and sum(1 for c in outs[1:] if l1(c, target) <= R - 2) < n_fool:
                    continue
                picks = ins + outs
                if not all(board_law.sep_ok(lat, diag, a, b)
                           for i_, a in enumerate(picks) for b in picks[i_ + 1:]):
                    continue
                Wp = [crossings(walls, c, target) for c in picks]
                rng.shuffle(picks)
                starts = list(zip(WITHIN_LABELS, picks))
                return dict(
                    walls=walls, starts=starts, exit=target, answer=str(ans),
                    path=[c for c in ins],
                    lam=board_law.lam(ex, [c for c in comp if d.get(c, 1e9) <= R]),
                    rstar=R, E=E, carver=carver, ncomp=1, sub=0,
                    comp_sizes=[len(comp)], target_comp=len(comp),
                    R=R, dists=[d[c] for c in picks],
                    euclid_rank=corridor_questions.euclid_rank(lat, picks, target),
                    D=R, margin=min(abs(d[c] - R) for c in picks), X=X_k,
                    X_all=[d[c] - l1(c, target) for c in picks],
                    W_all=Wp,
                    L1_all=[l1(c, target) for c in picks],
                    alpha_decoy=bool(ans >= 1 and alpha),
                    key_fools_l1=(l1(key_c, target) <= R - 1),
                    key_cell=list(key_c), key_W=W_k, key_L1=l1(key_c, target))
    return None


class StyleRng(random.Random):
    """Restricts two base-renderer style draws, by tuple identity, without touching
    the base code: thick gives a 16px corridor at cell=21 and a
    square target renders at 11px -- both far under even the deliberately
    breached feature line -> thin/medium and diamond/star only.  Same
    restriction at every tier: styles carry no tier information."""
    def choice(self, seq):
        if seq is questions.THICKNESSES:
            seq = ("thin", "medium")
        if seq is questions.TARGET_SHAPES:
            seq = ("diamond", "star")
        return super().choice(seq)
