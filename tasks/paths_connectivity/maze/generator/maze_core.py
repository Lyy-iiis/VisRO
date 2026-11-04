"""maze: five rungs on the detour-excess axis, five question forms.

Ladder:

    tier     rect grid                 hex        polar       boards                     forms
    easy     9x9  cell 44              8x8  29    10x6  36    K in {2,3,4}               all five
    medium   13x13 cell 33             10x10 23   14x8  27    K in {4,6,8}               all five
    hard     21x21 cell 21             --         --          detour_core's builders + bands, balanced plans
    xhard    21x21 / 23x23 (cell 19)   --         --          same; the 23 grid scaled (below)
    xxhard   21 / 23 / 25 (cell 17)    --         --          same; kth on 21 only      (OOD only)

Bigger grids (so the OOD set has a grid size never seen in training): the 448 canvas stays, so the cell shrinks (19 / 17 px).
On a 23 / 25 grid detour_core's depth structure scales with the side in
builder_config -- D 20 -> 22 / 24, R_WITHIN, the tf / mc freeze, the X bands
and plan values (rounded to even, and capped so the key's straight distance
clears the pixel separation floor of 12% of the diagonal, 3.9 / 4.3 cells
there: xxhard X = 14 on 23, 16 on 25, straight distance 6 in both) -- the
wall-crossing windows widen by 2, and xxhard's closer uses xhard's cue
rotation (its tie-heavy one needs two cells at one exact pixel distance on
the axis arms, rare on these grids).  xxhard kth stays on 21x21: four starts
at exact depths with planned pixel ranks build only half the time there.
On the 21x21 grid nothing changes: the boards are byte-identical to detour_core.

Hard and above use detour_core's builders, bands (X 4-6 / 8-12 / 14-16,
D = 20, R 16-20, E and W windows) and wording, driven by balanced plan digits
(per-slot plans alone would leak).  The floor: nothing at hard and above may
be easier than detour_core's -- the bands are untouched, the reachability forms
are fixed to depth 18-20, and hard is rect 21x21 only (a 143-cell hex board
is a smaller search than the 21x21 even at the same depth and detour).  The
plan changes only move labels, pixel ranks, answer counts and cue classes,
and two wall floors put the closer loser and the kth decoys (hard, xhard)
under the key's 2-wall minimum on the straight segment; every one of these
removes a cheap cue, none shortens a path.

Lattices (shares ~70 / 20 / 10):

  * polar carries only the two reachability forms (tf, mc): its cells grow
    with the ring, so "corridor steps" is not something the picture shows;
  * hex carries all five forms on easy / medium (uniform cells; the axis
    X = corridor - no-wall lattice distance is the same definition);
  * at hard, hex / polar carry only tf / mc: the distance forms need depth
    20, i.e. ~440 cells, and the largest hex that fits 448 has 143.  Those
    forms stay on the 21x21 rect, detour_core unchanged.  xhard and xxhard are
    rect only: a 143-cell hex cannot look harder than hard's (no visible
    difference), so the top rungs are carried by the 21x21 rect.
  * the lattice is the fastest digit of the (tier, form) occurrence and the
    plan rotations run on the quotient, so every answer meets every lattice
    equally (the lattice carries no information about the answer).

Wall width is sampled per record (WALL_PX: rect / hex 2-10 px, polar 3-10 px,
capped at 30% of the cell) instead of board_law's 2-4 px, which is too thin on
polar.  Cell sizes are the largest whose drawn content fits the
448 canvas; autocrop's 14 px pad is trimmed by the centred paste, exactly as
detour_core did with its 21x21 boards.

Edge entrances (as in a classic printed maze): on ~30% of
the easy / medium rect and hex boards of tf / mc / closer / kth, every start
is an opening in the outer wall with an arrow pointing in and its letter
beside it; on half of those (never tf) the target is an exit opening with an
arrow pointing out, the question says "the exit", and -- with nothing drawn
inside the maze -- the walls may be as thick as the corridor (BLOCK_WALL).
Distances are those of the border cell behind the opening, the same for
every start, so nothing about the answer changes.

Forms and their share of every tier, one question per image:

    tf       10%  "Is B reachable from A?"                     True / False
    mc       15%  "Which start reaches the target?"            A-D
    closer   25%  which of A, B is closer by corridor          A / B
    kth      25%  which start is the k-th closest              A-D
    within   25%  how many starts within R corridor steps      0-8

tf and mc live on boards with two / four components, and are graded on the
same bait as detour_core: the answer looks wrong by straight line.  A
True tf board hides its path behind a detour X in the tier's band; a False
board puts A and B a short straight line apart on either side of a wall
(the tier band is that straight-line distance, smaller = harder); an mc
winner is the only connected start AND sits at straight-line rank 2 / 3 / 4
among the four (harder = farther-looking).  True and False boards share one
component-size law, so neither the count nor the size of components tells
the answer.

Shortcut audit.  Every cheap rule -- pixel
rank, straight lattice distance, walls on the straight segment, region size,
label, K, lattice, edge flags -- is at chance on every form and tier, except
two structural residuals on easy
(the k-th nearest by lattice distance 0.43, the L1 count of within 0.27),
which are the small-X regime itself.

Every block is a pure function of (seed, split, block).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
import random
import re

from PIL import ImageDraw

from shared.drawing.fonts import load_bundled_font
from visro.seeding import seed_text

from . import board_law, corridor_questions, detour_core, lattice

# RNG_NAMESPACE is the fixed namespace of every board / style random stream: changing it changes every maze.
RNG_NAMESPACE = seed_text(686594245443073406512178434333687154852527223186426028121906)
RENDER_VERSION = "maze_render"
TRAIN_TIERS = ("easy", "medium", "hard", "xhard")
OOD_TIER = "xxhard"
TIERS = TRAIN_TIERS + (OOD_TIER,)
FORMS = ("tf", "mc", "closer", "kth", "within")
FORM_NAME = {"tf": "reachable_tf", "mc": "which_reaches", "closer": "which_is_closer",
             "kth": "kth_closest", "within": "within_radius"}
# 20-slot cycle: tf 2, mc 3, closer/kth/within 5 each = 10 / 15 / 25 / 25 / 25 %
FORM_CYCLE = ("closer", "kth", "within", "mc", "closer", "kth", "within", "tf", "closer", "kth",
              "within", "mc", "closer", "kth", "within", "tf", "closer", "kth", "within", "mc")
assert len(FORM_CYCLE) == 20 and FORM_CYCLE.count("tf") == 2 and FORM_CYCLE.count("mc") == 3

# ---- the small tiers: passage law, depth, and the axis bands (per lattice kind)
SMALL = {
    "easy":   dict(K=(2, 3, 4), D=8, R=(6, 8), W=(0, 4), ef=(0.05, 0.95),
                   X={"closer": (0, 2), "kth": (0, 2), "within": (0, 2), "tf": (0, 2)},
                   mc_D=(5, 8)),
    "medium": dict(K=(4, 6, 8), D=12, R=(10, 12), W=(0, 4), ef=(0.05, 0.95),
                   X={"closer": (2, 4), "kth": (2, 4), "within": (2, 4), "tf": (2, 4)},
                   mc_D=(8, 12)),
}
# Lattices per tier, (kind, w, h, cell).  Cell sizes are the largest whose
# drawn content (walls at the thickest menu width, without autocrop's 14 px
# pad) fits the 448 canvas: rect 13x13 at 33 is 438, hex 8x8 at 29 is 436,
# hex 11x13 at 21 is 429, polar 16x10 at 22 is 441.  The rect 21x21 of hard
# and above is detour_core unchanged (444 px at its 3 px walls).
LATTICES = {
    "easy":   {"rect": ("rect", 9, 9, 44),   "hex": ("hex", 8, 8, 29),    "polar": ("polar", 10, 6, 36)},
    "medium": {"rect": ("rect", 13, 13, 33), "hex": ("hex", 10, 10, 23),  "polar": ("polar", 14, 8, 27)},
    "hard":   {"rect": ("rect",) + tuple(detour_core.GRID), "hex": ("hex", 11, 13, 21), "polar": ("polar", 16, 10, 22)},
    # bigger rect grids: xhard also on 23x23, xxhard
    # on 21 / 23 / 25 so the OOD set has a grid size never seen in training.
    # Same 448 canvas, so the cell shrinks: 19 px (corridor 16) and 17 px
    # (corridor 14).  detour_core's depth structure scales with the side (see
    # builder_config); the plans and everything else are untouched.
    "xhard":  {"rect": ("rect",) + tuple(detour_core.GRID), "rect23": ("rect", 23, 23, 19)},
    "xxhard": {"rect": ("rect",) + tuple(detour_core.GRID), "rect23": ("rect", 23, 23, 19), "rect25": ("rect", 25, 25, 17)},
}
# Which lattices carry which question form.  Polar cells grow with the ring
# (inner wedges are a few px, outer ones a few dozen), so "how many corridor
# steps" is not a visual quantity there: polar carries only the two
# reachability forms.  Hex cells are uniform and carry everything on the
# small tiers; at hard and above the distance forms stay on the rect 21x21
# (detour_core's builders; the D = 20 depth needs ~440 cells, the largest hex
# that fits 448 has 143).
FORM_LATTICES = {
    "small": {"tf": ("rect", "hex", "polar"), "mc": ("rect", "hex", "polar"),
              "closer": ("rect", "hex"), "kth": ("rect", "hex"), "within": ("rect", "hex")},
    # hard: rect only (hard and above must be at least as
    # hard as detour_core's; a 143-cell hex / 160-cell polar board is a smaller
    # search than the 21x21 even at the same depth and detour)
    "hard":  {f: ("rect",) for f in FORMS},
    # xhard / xxhard: rect only (a 143-cell hex or 160-cell polar cannot look
    # harder than hard's), on one, two or three grid sizes.
    "xhard":  {f: ("rect", "rect23") for f in FORMS},
    # xxhard kth stays on 21x21: four starts at exact depths with planned
    # pixel ranks at X = 14-16 build on the bigger grids only half the time
    # (150-200 boards per case); the other four forms carry 23 and 25
    "xxhard": {**{f: ("rect", "rect23", "rect25") for f in FORMS}, "kth": ("rect",)},
}


# Edge entrances: on some easy / medium boards the
# starts are openings in the outer wall, the letter drawn outside the maze,
# as in a classic maze; the target stays inside.  rect and hex only (the
# polar rim is one circle) and not for within_radius (eight entrances).  The
# corridor distance of an entrance is that of the border cell behind it;
# every start is an entrance on such a board, so comparisons are unchanged.
# The cells are a little smaller so the outside arrows and letters fit the
# canvas: the arrow adds at most ~32 px per side (measured on generated
# boards: rect 424 / 441 px, hex 432 / 427 px at the thickest block wall).
P_EDGE = 0.30
P_EXIT = 0.50          # of the edge boards with a target: the target is an exit opening on the rim
EDGE_FORMS = ("tf", "mc", "closer", "kth")
# exit boards carry no marker inside the maze, so their walls may be as thick
# as the corridor, as in the classic printed maze. Fractions refer to the wall
# segment length: a hexagon's neighbour pitch is sqrt(3) times its edge length,
# and using that pitch can make the wall wider than the remaining portal gap.
BLOCK_WALL = (0.15, 0.25, 0.35, 0.45, 0.50)
EDGE_LATTICES = {
    "easy":   {"rect": ("rect", 9, 9, 40),   "hex": ("hex", 8, 8, 25)},
    "medium": {"rect": ("rect", 13, 13, 29), "hex": ("hex", 10, 10, 20)},
}


def edge_mode(split: str, tier: str, form: str, occ: int, kind: str):
    """-> (edge, exit): entrances on the rim; target an exit opening on the rim."""
    if tier not in EDGE_LATTICES or form not in EDGE_FORMS or kind not in EDGE_LATTICES[tier]:
        return False, False
    rng = random.Random(f"{RNG_NAMESPACE}|edge|{split}|{tier}|{form}|{occ}")
    edge = rng.random() < P_EDGE
    exit_ = edge and form != "tf" and rng.random() < P_EXIT
    return edge, exit_


def lattice_kinds(tier: str, form: str):
    return FORM_LATTICES["small" if tier in SMALL else tier][form]


# straight-segment wall-crossing windows on non-rect lattices are looser:
# the generic crossing count is approximate (nearest-centre sampling)
W_NONRECT = {"W": (0, 6), "cap": 8}
# the big tiers reuse detour_core for closer/kth/within; tf/mc need their own bands
BIG = {                                  # mc_D unused since the depth freeze at 18-20 (see _build_generic)
    "hard":   dict(X_tf=(4, 6),   mc_D=(18, 20)),
    "xhard":  dict(X_tf=(8, 12),  mc_D=(18, 20)),
    "xxhard": dict(X_tf=(14, 16), mc_D=(18, 20)),
}
MAX_ATT = 400


def k_law(tier: str):
    return SMALL[tier]["K"] if tier in SMALL else detour_core.K_LAW


# ---- lattice-neutral geometry.  "Straight-line distance in cells" is the
# lattice distance with NO walls (what the maze would cost if nothing
# blocked you): on rect that is exactly Manhattan L1, detour_core's definition;
# on hex it is the hex distance, on polar rings plus sectors.  X = corridor
# distance - that, on every lattice.
class Metric:
    def __init__(self, lat):
        self.lat = lat
        self.cache = {}
        self.centres = [lat.center(c) for c in lat.cells]
        self.cells = list(lat.cells)

    def straight(self, a, b):
        d = self.cache.get(b)
        if d is None:
            d = lattice.bfs_dist(b, self.lat.neighbors, set())
            self.cache[b] = d
        return d.get(a, 10 ** 6)

    def cell_at(self, x, y):
        best, bd = None, 1e18
        for c, (cx, cy) in zip(self.cells, self.centres):
            dd = (cx - x) ** 2 + (cy - y) ** 2
            if dd < bd:
                best, bd = c, dd
        return best

    def crossings(self, walls, a, b):
        """Closed walls crossed by the straight pixel segment centre(a)->centre(b),
        by nearest-centre sampling.  A jump between non-neighbour cells (a
        vertex) counts one."""
        (ax, ay), (bx, by) = self.lat.center(a), self.lat.center(b)
        n = max(4, int(math.hypot(bx - ax, by - ay) / 4))
        seq = []
        for i in range(n + 1):
            t = i / n
            c = self.cell_at(ax + (bx - ax) * t, ay + (by - ay) * t)
            if not seq or seq[-1] != c:
                seq.append(c)
        w = 0
        for p, q in zip(seq, seq[1:]):
            if q in self.lat.neighbors(p):
                w += lattice.wall_key(p, q) in walls
            else:
                w += 1
        return w


_METRIC = None
_GRID_SCALE = 1.0                   # 23x23 -> 1.10, 25x25 -> 1.19: detour_core's depths and X bands scale with it
_start_ok = lambda lat, c: True     # noqa: E731  edge boards: border cells only


_X_CAP = 10 ** 6                    # largest even X the grid's separation floor allows (set per big grid)


def _even(x):
    return 2 * int(round(x / 2))


def _even_floor(x):
    return 2 * int(math.floor(x / 2 + 1e-9))


def _even_ceil(x):
    return 2 * int(math.ceil(x / 2 - 1e-9))


def scale_band(band, s, cap=None):
    """An X band scaled by s: the low end rounds down to even, the high end up,
    both clamped to the grid's cap (a key must keep its straight distance
    above the pixel separation floor)."""
    cap = _X_CAP if cap is None else cap
    lo, hi = _even_floor(band[0] * s), _even_ceil(band[1] * s)
    return (min(lo, cap), max(min(hi, cap), min(lo, cap)))
_target_ok = lambda lat, c: True    # noqa: E731  edge boards: interior only


def l1(a, b):
    """Straight-line cells for the current lattice (see Metric)."""
    return _METRIC.straight(a, b)


@contextlib.contextmanager
def builder_config(tier: str, lat, edge: bool = False, exit_: bool = False):
    """detour_core and board_law keep their geometry at module level (rect L1,
    rect wall crossings, LATTICE = rect, depth windows).  Swap all of it for
    the duration of one build / render and restore after.  The rect 21x21
    tiers are left exactly as detour_core runs them."""
    global _METRIC, _start_ok, _target_ok, _GRID_SCALE, _X_CAP
    metric = Metric(lat)
    saved_metric = _METRIC
    _METRIC = metric
    saved = (detour_core.D_CLOSER, detour_core.D_KTH, detour_core.W_WIN, detour_core.W_CAP, detour_core.EFRAC_WIN, detour_core.l1, detour_core.crossings, board_law.LATTICE, board_law.GRID)
    saved_ok = (detour_core.START_OK, detour_core.TARGET_OK, _start_ok, _target_ok)
    saved_plan = {"R_WITHIN": detour_core.R_WITHIN, "D_FREEZE": detour_core.D_FREEZE, "BANDS_X": detour_core.BANDS_X, "PLAN_X": detour_core.PLAN_X,
                  "PLAN_CLOSER": detour_core.PLAN_CLOSER, "WMIN": (detour_core.CLOSER_LOSER_WMIN, detour_core.KTH_DECOY_WMIN)}
    saved_scale = (_GRID_SCALE, _X_CAP)
    try:
        if edge:
            detour_core.START_OK = _start_ok = lambda lat_, c: lat_.is_border(c)
            if exit_:
                detour_core.TARGET_OK = _target_ok = lambda lat_, c: lat_.is_border(c)
            else:
                detour_core.TARGET_OK = _target_ok = lambda lat_, c: not lat_.is_border(c)
        if tier in SMALL:
            c = SMALL[tier]
            detour_core.D_CLOSER = detour_core.D_KTH = c["D"]
            detour_core.EFRAC_WIN = c["ef"]
            detour_core.W_WIN = c["W"] if lat.name == "rect" else W_NONRECT["W"]
            detour_core.W_CAP = detour_core.W_CAP if lat.name == "rect" else W_NONRECT["cap"]
        if lat.name != "rect":
            detour_core.l1 = metric.straight
            detour_core.crossings = metric.crossings
            if tier not in SMALL:
                detour_core.W_WIN, detour_core.W_CAP = W_NONRECT["W"], W_NONRECT["cap"]
        if tier not in SMALL and lat.name == "rect" and lat.width != detour_core.GRID[0]:
            # 23x23 / 25x25: detour_core's depth structure scales with the side
            # (D 20 -> 22 / 24, R, the X bands and plan values, the tf / mc
            # freeze), rounded to even so the rect parity rule (X even) holds.
            # The pixel separation floor (12% of the diagonal) is 3.9 / 4.3
            # cells on these grids, so an unscaled xxhard key (straight
            # distance 4-6) can barely exist; scaled, it sits at 6-8.  E/N and
            # W keep detour_core's windows.
            _GRID_SCALE = s_ = lat.width / detour_core.GRID[0]
            detour_core.D_CLOSER = detour_core.D_KTH = _even(20 * s_)
            detour_core.R_WITHIN = tuple(_even(r * s_) for r in saved_plan["R_WITHIN"])
            detour_core.D_FREEZE = (_even(saved_plan["D_FREEZE"][0] * s_), detour_core.D_CLOSER)
            # the key's straight distance r - X must clear the pixel separation
            # floor (12% of the diagonal, in cells of this grid); X is even on rect
            l1_min = math.ceil(0.12 * math.hypot(*lat.image_size()) / lat.cell)
            _X_CAP = _even_floor(detour_core.D_CLOSER - detour_core.MARGIN - l1_min)
            detour_core.BANDS_X = {t: scale_band(b, s_, _X_CAP if t == OOD_TIER else _X_CAP - 2)      # xhard stays below xxhard
                         for t, b in saved_plan["BANDS_X"].items()}
            detour_core.PLAN_X = {t: {task: tuple(min(_even_floor(x * s_), _X_CAP) for x in xs) for task, xs in d.items()}
                        for t, d in saved_plan["PLAN_X"].items()}
            # longer straight segments cross more walls: the crossing windows widen by 2
            detour_core.W_WIN, detour_core.W_CAP = (detour_core.W_WIN[0], detour_core.W_WIN[1] + 2), detour_core.W_CAP + 2
            # xxhard's tie-heavy closer plan (cons / tie / tie / tie) needs two
            # cells at one exact pixel distance on the axis arms, which the
            # bigger grids rarely offer; they use xhard's cue rotation
            # (cons / tie / misl / tie) at the xxhard detour band instead
            detour_core.PLAN_CLOSER = dict(detour_core.PLAN_CLOSER, xxhard=saved_plan["PLAN_CLOSER"]["xhard"])
        if tier not in SMALL:
            # the closer loser (every hard tier) and the kth decoys (hard, xhard;
            # xxhard's tie mode has no slack for it) cross at least as many walls
            # on their straight segment as the key's floor
            detour_core.CLOSER_LOSER_WMIN = detour_core.W_WIN[0]
            detour_core.KTH_DECOY_WMIN = detour_core.W_WIN[0] if tier != OOD_TIER else 0
        board_law.LATTICE = lat.name
        board_law.GRID = (lat.width, lat.height, getattr(lat, "cell", getattr(lat, "size", 0)))
        yield metric
    finally:
        (detour_core.D_CLOSER, detour_core.D_KTH, detour_core.W_WIN, detour_core.W_CAP, detour_core.EFRAC_WIN, detour_core.l1, detour_core.crossings, board_law.LATTICE, board_law.GRID) = saved
        detour_core.START_OK, detour_core.TARGET_OK, _start_ok, _target_ok = saved_ok
        detour_core.R_WITHIN, detour_core.D_FREEZE, detour_core.BANDS_X, detour_core.PLAN_X = (saved_plan["R_WITHIN"], saved_plan["D_FREEZE"],
                                                      saved_plan["BANDS_X"], saved_plan["PLAN_X"])
        detour_core.PLAN_CLOSER = saved_plan["PLAN_CLOSER"]
        detour_core.CLOSER_LOSER_WMIN, detour_core.KTH_DECOY_WMIN = saved_plan["WMIN"]
        _GRID_SCALE, _X_CAP = saved_scale
        _METRIC = saved_metric


# ------------------------------------------------------------------ schedule
@dataclasses.dataclass(frozen=True)
class Spec:
    block: int
    tier: str
    form: str
    occurrence: int        # how many blocks of this (tier, form) came before
    split: str
    dataset_role: str
    lattice: tuple = ("rect",) + tuple(detour_core.GRID)
    edge: bool = False     # starts are openings in the outer wall
    exit: bool = False     # the target is an exit opening in the outer wall


def lattice_of(spec: "Spec"):
    kind, w, h, cell = spec.lattice
    return lattice.build_lattice(kind, w, h, cell)


def schedule(block: int, dataset_role: str, split: str) -> Spec:
    if dataset_role == "train_id":
        tier = TRAIN_TIERS[block % 4]
        tier_occ = block // 4
    elif dataset_role == "ood_eval":
        tier = OOD_TIER
        tier_occ = block
    else:
        raise ValueError(dataset_role)
    form = FORM_CYCLE[tier_occ % len(FORM_CYCLE)]
    cycle, pos = divmod(tier_occ, len(FORM_CYCLE))
    occ = cycle * FORM_CYCLE.count(form) + FORM_CYCLE[:pos].count(form)
    # the lattice is the fastest-varying digit of the (tier, form) occurrence;
    # the plan rotations (X_want, answer, label, ...) run on occ // n_kinds, so
    # every plan value meets every lattice equally and the lattice carries no
    # information about the answer
    kinds = lattice_kinds(tier, form)
    kind = kinds[occ % len(kinds)]
    edge, exit_ = edge_mode(split, tier, form, occ, kind)
    return Spec(block, tier, form, occ, split, dataset_role, (EDGE_LATTICES if edge else LATTICES)[tier][kind], edge, exit_)


def plan_occurrence(spec: "Spec") -> int:
    return spec.occurrence // len(lattice_kinds(spec.tier, spec.form))


# ------------------------------------------------------------------ tf / mc builders
def _two_comp_board(lat, rng, K):
    f = rng.uniform(0.25, 0.5)             # one law for True and False boards
    got = board_law.build_board(lat, rng, K, [f, 1.0 - f], focus_random=True)
    if got is None:
        return None
    walls, comps, carver = got
    if len(comps) != 2:
        return None
    return walls, comps, carver


def tf_x(lat, rng, K, want_true, X_band, D_band, stage):
    diag = math.hypot(*lat.image_size())
    got = _two_comp_board(lat, rng, K)
    if got is None:
        return None
    walls, comps, carver = got
    ex = board_law.excess(lat, walls)
    if want_true:
        host = rng.choice(comps)
        srcs = [c for c in host if board_law.marker_ok(lat, walls, c) and _start_ok(lat, c)]
        if len(srcs) < 2:
            return None
        rng.shuffle(srcs)
        for a in srcs[:12]:
            d = lattice.bfs_dist(a, lat.neighbors, walls)
            cands = []
            for b in srcs:
                dd = d.get(b)
                if dd is None or not (D_band[0] <= dd <= D_band[1]):
                    continue
                if not board_law.sep_ok(lat, diag, a, b):
                    continue
                X = dd - l1(a, b)
                if not (X_band[0] <= X <= X_band[1] or (stage >= 1 and X_band[0] <= X <= X_band[1] + 2)
                        or (stage >= 2 and X >= X_band[0])):
                    continue
                if not (1 <= detour_core.crossings(walls, a, b) <= 3):
                    continue          # a False pair always has a wall on its straight segment; so must a True pair
                cands.append((b, X, dd))
            if cands:
                b, X, dd = rng.choice(cands)
                E = sum(1 for _c, v in d.items() if v <= dd)
                return dict(walls=walls, a=a, b=b, solvable=True,
                            path=lattice.shortest_path(a, b, lat.neighbors, walls),
                            lam=board_law.lam(ex, [c for c in host if d.get(c, 1e9) <= dd]),
                            rstar=dd, E=E, carver=carver, ncomp=2, sub=0,
                            X=X, D=dd, L1=l1(a, b), comp_frac=round(len(host) / len(lat.cells), 3))
        return None
    # A False pair copies the straight distance AND the wall-crossing count of
    # a True pair that this very board could have asked (same D / X bands,
    # 1..3 walls on the straight segment): the two classes then share the law
    # of everything a glance measures, and only connectivity separates them.
    law = []
    for host in comps:
        srcs = [c for c in host if board_law.marker_ok(lat, walls, c) and _start_ok(lat, c)]
        rng.shuffle(srcs)
        for a0 in srcs[:8]:
            d0 = lattice.bfs_dist(a0, lat.neighbors, walls)
            for b0 in srcs:
                dd = d0.get(b0)
                if dd is None or not (D_band[0] <= dd <= D_band[1]) or not board_law.sep_ok(lat, diag, a0, b0):
                    continue
                X0 = dd - l1(a0, b0)
                if not (X_band[0] <= X0 <= X_band[1] or (stage >= 1 and X_band[0] <= X0 <= X_band[1] + 2)
                        or (stage >= 2 and X0 >= X_band[0])):
                    continue
                W0 = detour_core.crossings(walls, a0, b0)
                if 1 <= W0 <= 3:
                    law.append((l1(a0, b0), W0, int(lat.is_border(a0)) + int(lat.is_border(b0))))
    if not law:
        return None
    L_t, W_t, B_t = rng.choice(law)          # straight distance, walls crossed, markers on the rim
    ca, cb = comps if rng.random() < 0.5 else comps[::-1]
    av = [c for c in ca if board_law.marker_ok(lat, walls, c) and _start_ok(lat, c)]
    bv = [c for c in cb if board_law.marker_ok(lat, walls, c) and _start_ok(lat, c)]
    rng.shuffle(av)
    pairs = []
    for a in av[:40]:
        for b in bv:
            if (l1(a, b) == L_t and board_law.sep_ok(lat, diag, a, b) and detour_core.crossings(walls, a, b) == W_t
                    and int(lat.is_border(a)) + int(lat.is_border(b)) == B_t):
                pairs.append((a, b, L_t))
    if not pairs and stage >= 1:
        for a in av[:40]:
            for b in bv:
                L = l1(a, b)
                if abs(L - L_t) <= 1 and board_law.sep_ok(lat, diag, a, b) and 1 <= detour_core.crossings(walls, a, b) <= 3:
                    pairs.append((a, b, L))
    if not pairs:
        return None
    a, b, L = rng.choice(pairs)
    return dict(walls=walls, a=a, b=b, solvable=False, path=[],
                lam=board_law.lam(ex, ca), rstar=None, E=len(ca), carver=carver, ncomp=2, sub=0,
                X=None, D=None, L1=L, comp_frac=round(len(ca) / len(lat.cells), 3))


def mc_reach_x(lat, rng, K, D_band, want_rank, X_band, want_lab, stage, size_rank=None, l1_rank=None):
    """Four components, one start each; exactly one start reaches the target.
    The winner sits at straight-line rank `want_rank` among the four (1 =
    nearest looking): want_rank-1 unreachable decoys look nearer than it."""
    diag = math.hypot(*lat.image_size())
    got = board_law.build_board(lat, rng, K, board_law.rand_profile(rng, 4, lo=0.18), equal=True)
    if got is None:
        return None
    walls, comps, carver = got
    if len(comps) != 4:
        return None
    ex = board_law.excess(lat, walls)
    order = list(range(4))
    rng.shuffle(order)
    if size_rank is not None:
        by_size = sorted(range(4), key=lambda i: (-len(comps[i]), i))
        want = by_size[size_rank - 1]
        order = [want] if stage < 2 else [want] + [i for i in order if i != want]
    for ci in order:
        tc = comps[ci]
        ex_cells = [c for c in tc if board_law.marker_ok(lat, walls, c)]
        rng.shuffle(ex_cells)
        t_cells = [c for c in ex_cells if _target_ok(lat, c)]
        s_cells = [c for c in ex_cells if _start_ok(lat, c)]
        # decoys must be pixel- / L1-nearer than a winner whose straight
        # distance is D - X (2..8 cells at xxhard), so they must live in a
        # foreign region within a few cells of the target: try targets that
        # hug a region boundary first (on 25x25 / 29x29 boards the regions
        # are big blobs and a random target rarely has foreign cells nearby)
        foreign = [c for j, comp in enumerate(comps) if j != ci for c in comp if board_law.marker_ok(lat, walls, c)]
        reach = max(4, D_band[1] - X_band[0])
        t_cells = sorted(t_cells[:24], key=lambda t: -sum(1 for c in foreign if l1(c, t) <= reach))
        for target in t_cells[:8]:
            d = lattice.bfs_dist(target, lat.neighbors, walls)
            wins = []
            for c in s_cells:
                dd = d.get(c)
                if dd is None or not (D_band[0] <= dd <= D_band[1]) or not board_law.sep_ok(lat, diag, c, target):
                    continue
                X = dd - l1(c, target)
                if (X_band[0] <= X <= X_band[1] or (stage >= 1 and X_band[0] <= X <= X_band[1] + 2)
                        or (stage >= 2 and X >= X_band[0])):
                    wins.append((c, X, dd))
            rng.shuffle(wins)
            for win, X, dd in wins[:8]:
                ewin = lattice.distance_px(lat, win, target)
                others_idx = [j for j in range(4) if j != ci]
                rng.shuffle(others_idx)
                picks, bad = [], False
                l1win = l1(win, target)
                l1_slots = None
                if l1_rank is not None:
                    # which of the three decoy slots are L1-nearer than the winner:
                    # a random subset of size l1_rank-1, independent of the pixel roles
                    l1_slots = set(rng.sample(range(3), l1_rank - 1))
                for slot, cj in enumerate(others_idx):
                    nearer = slot < want_rank - 1
                    pool = [c for c in comps[cj] if board_law.marker_ok(lat, walls, c) and _start_ok(lat, c)
                            and board_law.sep_ok(lat, diag, c, target)
                            and all(board_law.sep_ok(lat, diag, c, p) for p in picks + [win])
                            and ((lattice.distance_px(lat, c, target) < ewin) == nearer)
                            and (l1_slots is None or ((l1(c, target) < l1win) == (slot in l1_slots)))]
                    if not pool:
                        bad = True
                        break
                    picks.append(rng.choice(pool))
                if bad:
                    continue
                labels = ["A", "B", "C", "D"]
                rest = [x for x in labels if x != want_lab]
                rng.shuffle(rest)
                starts = sorted([(want_lab, win)] + list(zip(rest, picks)))
                return dict(walls=walls, starts=starts, exit=target, answer=want_lab,
                            path=lattice.shortest_path(win, target, lat.neighbors, walls),
                            lam=board_law.lam(ex, [c for c in tc if d.get(c, 1e9) <= dd]),
                            rstar=dd, E=sum(1 for _c, v in d.items() if v <= dd),
                            carver=carver, ncomp=4, sub=0,
                            comp_sizes=sorted(len(c) for c in comps), target_comp=len(tc),
                            target_size_rank=sorted(range(4), key=lambda i: (-len(comps[i]), i)).index(ci) + 1,
                            euclid_rank=want_rank, X=X, D=dd, L1=l1(win, target),
                            target_is_largest=(len(tc) == max(len(c) for c in comps)))
    return None


# ------------------------------------------------------------------ small-tier plans
# Plan rotations.  Every visible cue must be at
# chance against the answer, so each rotation is balanced on its own and the
# rotations that share a board run on coprime periods (or on quotient digits)
# so they are independent of each other:
#   closer  cue class cons / misl / tie 2 : 2 : 1 (the pixel-nearer start is
#           the answer exactly half the time), label alternates A / B;
#   kth     answer's pixel rank uniform over {2,3,4} on easy (rank 1 is not
#           constructible at X <= 2 on rect: a shallower decoy is always
#           nearer) and over {1,2,3,4} on medium, independent of k and label;
#   mc      winner's pixel rank uniform over 1..4 on every tier (with a rank
#           fixed per tier, "the n-th nearest-looking start" would score 1.0);
#   tf      True / False alternate; the False pair's straight distance is
#           drawn from the True boards' support, and True boards need >= 1
#           wall on the straight segment (see tf_x).
SMALL_PLAN = {
    "closer": {"cls": ("cons", "misl", "tie", "misl", "cons")},
    "within": {"ans": (1, 4, 6, 2, 5, 7, 0, 3)},
    "tf":     {"true": (True, False)},
    "mc":     {"rank_lab": ("A", "B", "C", "D", "B", "D", "A", "C")},
}
# answer's pixel rank per tier.  xxhard: None = detour_core's tie mode -- at X = 14
# the answer sits at the separation floor, nothing can look nearer, so one
# decoy is placed at exactly its pixel distance and the tie label rotates
# over the other three letters (the "nearest, alphabetical tie-break" rule is
# then right half the time).
KTH_ERANKS = {"easy": (2, 3, 4), "medium": (1, 2, 3, 4), "hard": (1, 2, 3, 4), "xhard": (1, 2, 3, 4), "xxhard": None}
CLOSER_CLS = {"easy": SMALL_PLAN["closer"]["cls"], "medium": SMALL_PLAN["closer"]["cls"],
              "hard": SMALL_PLAN["closer"]["cls"], "xhard": SMALL_PLAN["closer"]["cls"],
              "xxhard": ("tie",) * 5}        # misl is not constructible at the floor; all ties keep the cue at 0.5


def kth_plan(tier, occ):
    """(k, label, answer's pixel rank, tie label) for the occ-th kth board of
    a tier; mutually independent digits.  The answer's rank by straight
    LATTICE distance is not planned: at X <= 2 every shallower decoy is
    L1-nearer than the answer, so that rank is >= k by construction (planning
    it gives no gain over the unplanned 0.43 and costs 80 min per 200
    boards)."""
    q, r = divmod(occ, 4)
    # xxhard asks for the third closest only (detour_core's choice): with k = 2 the
    # answer's corridor depth is 14-16 and at X = 14 its straight distance
    # would be 0-2 cells, under the separation floor
    k, lab = (3 if tier == OOD_TIER else (2, 3)[q % 2]), "ABCD"[r]
    eranks = KTH_ERANKS[tier]
    if eranks is None:
        others = [x for x in "ABCD" if x != lab]
        return k, lab, None, others[q % 3]
    return k, lab, eranks[(q + r) % len(eranks)], None


def mc_plan(occ):
    """(winner label, winner's pixel rank, size rank of the target's region)
    for the occ-th mc board; three independent digits.  The size rank is
    planned because the builder otherwise favours the biggest region (the
    winner needs depth D inside it), and "the start in the largest region" would
    then score 0.60 on hard."""
    lab = SMALL_PLAN["mc"]["rank_lab"][occ % 8]
    rank = 1 + ((occ // 8) + occ) % 4
    size_rank = 1 + ((occ // 32) + (occ // 8) + occ) % 4
    l1_rank = 1 + ((occ // 128) + (occ // 32) + (occ // 8) + occ) % 4
    return lab, rank, size_rank, l1_rank


WITHIN_FOOL_PLAN = (1, 2, 3, 0)      # non-key outside markers that look inside by straight line


def build_case(spec: Spec, seed: int):
    """-> (case, task, meta); task is the board_law / corridor_questions task key used by the renderer."""
    tier, form = spec.tier, spec.form
    lat = lattice_of(spec)
    task = {"closer": "closer", "kth": "mc", "within": "count", "tf": "tf", "mc": "mc"}[form]
    # hard and above: detour_core's builders and bands (closer_x / kth_x / within_x,
    # BANDS_X, D = 20, R_WITHIN, E window) driven by the balanced plan digits.
    # detour_core's own per-slot plans leak: the k-th nearest-LOOKING start is
    # the kth answer 0.67 of the time on xhard, within's answers are three values
    # per tier (best constant 0.335), xxhard's closer is 3/4 ties + 1/4 cons
    # (nearer-looking start 0.62).  The bands are unchanged.
    with builder_config(tier, lat, spec.edge, spec.exit):
        return _build_generic(spec, seed, lat, task)


def _build_generic(spec, seed, lat, task):
    tier, form, occ = spec.tier, spec.form, plan_occurrence(spec)
    boards = 0
    for att in range(MAX_ATT):
        rng = random.Random(f"{RNG_NAMESPACE}|{seed}|{spec.split}|{tier}|{form}|{occ}|{lat.name}|{'exit' if spec.exit else 'edge' if spec.edge else 'in'}|{att}")
        K = rng.choice(k_law(tier))
        stage = 0 if att < 150 else (1 if att < 300 else 2)
        if lat.name != "rect":
            stage = max(stage, 1)          # X parity is a rect property; accept the band
        boards += 1
        if form == "tf":
            want_true = SMALL_PLAN["tf"]["true"][occ % 2]
            cfg = SMALL[tier] if tier in SMALL else BIG[tier]
            # hard and above: the reachable path is as deep as the distance forms' key
            # (18-20 on 21x21, scaled on the bigger grids), never a short 14-step hop
            Db = (cfg["D"] - 2, cfg["D"] + 2) if tier in SMALL else (detour_core.D_FREEZE[1] - 2, detour_core.D_FREEZE[1])
            Xb = cfg["X"]["tf"] if tier in SMALL else scale_band(cfg["X_tf"], _GRID_SCALE, _X_CAP + Db[1] - detour_core.D_CLOSER + detour_core.MARGIN)
            case = tf_x(lat, rng, K, want_true, Xb, Db, stage)
        elif form == "mc":
            cfg = SMALL[tier] if tier in SMALL else BIG[tier]
            lab, rank, size_rank, l1_rank = mc_plan(occ)
            Db = cfg["mc_D"] if tier in SMALL else (detour_core.D_FREEZE[1] - 2, detour_core.D_FREEZE[1])
            Xb = cfg["X"]["closer"] if tier in SMALL else scale_band(BIG_X_MC[tier], _GRID_SCALE, _X_CAP + Db[1] - detour_core.D_CLOSER + detour_core.MARGIN)
            case = mc_reach_x(lat, rng, K, Db, rank, Xb, lab, stage, size_rank, l1_rank if stage < 2 else None)
        else:
            # closer / kth / within on a small tier, or on a non-rect hard board
            c = SMALL[tier] if tier in SMALL else None
            band = c["X"][form] if c else detour_core.BANDS_X[tier]
            ef = c["ef"] if c else detour_core.EFRAC_WIN
            X_want = band[0] + 2 * (occ % ((band[1] - band[0]) // 2 + 1))
            if form == "closer":
                cls = CLOSER_CLS[tier][occ % 5]
                detour_core.CLOSER_L1 = {"cons": "win", "misl": "lose", "tie": "tie"}[cls]
                try:
                    case = detour_core.closer_x(lat, rng, K, X_want, band, cls, "AB"[occ % 2], ef, stage)
                finally:
                    detour_core.CLOSER_L1 = None
            elif form == "kth":
                k, lab, erank, tie_lab = kth_plan(tier, occ)
                case = detour_core.kth_x(lat, rng, K, X_want, band, k, lab, erank, ef, stage, tie_lab=tie_lab)
            else:
                R = c["R"][occ % len(c["R"])] if c else detour_core.R_WITHIN[occ % 3]
                ans = SMALL_PLAN["within"]["ans"][occ % 8]
                saved_fool = detour_core.WITHIN_FOOL
                detour_core.WITHIN_FOOL = WITHIN_FOOL_PLAN[(occ // 8 + occ) % 4] if c else 0
                try:
                    case = detour_core.within_x(lat, rng, K, R, X_want, band, ans, ef, stage)
                finally:
                    detour_core.WITHIN_FOOL = saved_fool
        if case is not None:
            case["K_used"] = K
            return case, task, {"attempts": att + 1, "boards": boards, "stage": stage, "builder": "plan"}
    return None, task, {"attempts": MAX_ATT, "boards": boards, "stage": 2, "builder": "plan"}


BIG_X_MC = {"hard": (4, 6), "xhard": (8, 12), "xxhard": (12, 16)}


# ------------------------------------------------------------------ records
# ---- wall thickness.  board_law draws every board at 0.7 / 1.0 / 1.5 x a
# per-lattice base (rect and polar 3 px, hex 5-7 px), so rect and polar walls
# were 2-4 px on every tier.  Here the width is sampled in pixels per record from a
# wider menu, capped per lattice so passages never close: 30% of the cell, and
# on polar also 75% of the innermost wedge's arc (16 sectors at cell 20 give
# a 7.9 px wedge; a 6 px wall there is a black hub).
WALL_PX = {"rect": (2, 3, 4, 5, 6, 8, 10), "hex": (2, 3, 4, 5, 6, 8, 10), "polar": (3, 4, 5, 6, 8, 10)}


def wall_px_menu(lat):
    cap = round(lat.unit * 0.30)          # 21 px rect cells: 6 px
    if lat.name == "polar":
        cap = min(cap, round(0.75 * 2 * math.pi * lat.ring_step / lat.width))
    menu = [w for w in WALL_PX[lat.name] if w <= cap]
    return menu or [max(2, cap)]


def block_wall_px_menu(lat):
    """Keep thick rim walls at most half of the actual opening's edge length."""
    return [max(3, round(f * lat.edge_len)) for f in BLOCK_WALL]


DESC_RE = re.compile(r"\b(?:the unlabeled target marker|the square marker|the diamond|the star)\b")   # \b: not "the start"
_RENDER_CASE = lattice.render_case    # the original, looked up once: render_record swaps the module attribute


def outward_dir(lat, cell):
    """Index into lat.dir_pix of a boundary side of `cell` (the first one)."""
    for i in range(len(lat.dir_pix)):
        if lat.neighbor_in_dir(cell, i) is None:
            return i
    raise ValueError("not a border cell")


def _arrow(draw, tail, tip, color, width):
    """Shaft from tail to tip with a triangular head at the tip."""
    (tx, ty), (hx, hy) = tail, tip
    L = math.hypot(hx - tx, hy - ty)
    ux, uy = (hx - tx) / L, (hy - ty) / L
    head = min(0.45 * L, 3.2 * width)
    half = 0.62 * head
    bx, by = hx - ux * head, hy - uy * head          # base of the head
    draw.line(((tx, ty), (bx, by)), fill=color, width=width)
    draw.polygon([(hx, hy), (bx + -uy * half, by + ux * half), (bx - -uy * half, by - ux * half)], fill=color)


EDGE_MARGIN = 56       # px added around the canvas of an edge board: the arrows and letters live there


def enlarge_margin(lat, P=EDGE_MARGIN):
    """Shift every cell centre by P and grow the canvas by 2P (this instance
    only).  The lattice margin is 0.8 cell; the outside arrow and letter need
    ~50 px, and anything past the canvas is simply not drawn."""
    lat._centers = {k: (x + P, y + P) for k, (x, y) in lat._centers.items()}
    w, h = lat.image_size()
    lat.image_size = lambda: (w + 2 * P, h + 2 * P)


def render_case_edge(lat, walls, markers, style=None, marker_scale=1.0):
    """lattice.render_case for the classic printed-maze idiom: every lettered start on
    a border cell becomes an entrance -- its boundary wall segment is opened,
    an arrow in the start colour points in through the gap and the letter
    stands outside -- and an unlettered target on a border cell becomes the
    exit: an arrow in the target colour pointing out through its gap.
    Markers on interior cells are drawn as usual."""
    style = style or lattice.DEFAULT_STYLE
    inside, rim = [], []
    for mk in markers:
        (rim if lat.is_border(mk[0]) else inside).append(mk)
    lw = lat.styled_width(style.thickness)
    h = (lat.edge_len - lw) / 2
    if rim and h <= 2:
        # A silently omitted cut leaves an arrow pointing at a solid wall even
        # though the graph and ground truth say this cell is an entrance/exit.
        raise ValueError(f"Border opening too narrow: edge={lat.edge_len}, wall={lw}")
    img = _RENDER_CASE(lat, walls, inside, style, marker_scale)
    draw = ImageDraw.Draw(img)
    for mk in rim:
        cell, label, color = mk[0], mk[1], mk[2]
        i = outward_dir(lat, cell)
        dx, dy = lat.dir_pix[i]
        L = math.hypot(dx, dy)                       # pitch: one neighbour step
        ux, uy = dx / L, dy / L
        cx, cy = lat.center(cell)
        gx, gy = cx + dx / 2, cy + dy / 2            # midpoint of the rim segment
        px, py = -uy, ux
        # The joints and side walls keep their width; the opening is mandatory.
        draw.line(((gx + px * h, gy + py * h), (gx - px * h, gy - py * h)), fill=style.bg, width=lw + 4)
        clear = lw / 2 + 2                           # just outside the rim line
        alen = min(0.5 * L, 20.0)
        sw = max(3, round(0.09 * L))
        if label:                                    # entrance: arrow in, letter beside it (as in the printed idiom)
            _arrow(draw, (gx + ux * (clear + alen), gy + uy * (clear + alen)), (gx + ux * clear, gy + uy * clear), color, sw)
            fs = int(min(0.45 * L, 19))
            font = load_bundled_font(fs, "sans-bold")
            bb = draw.textbbox((0, 0), label, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            d = clear + alen / 2                     # level with the arrow's middle
            side = 0.62 * min(0.45 * alen, 3.2 * sw) + 4 + max(tw, th) / 2     # clear of the arrow head
            # beside the arrow, on whichever side is farther from the maze (a
            # hex rim zigzags: the row below an E-side cell sticks out under
            # the arrow); if both sides touch the walls, beyond the tail
            centres = list(lat._centers.values())
            def clearance(pt):
                return min(math.hypot(pt[0] - cx_, pt[1] - cy_) for cx_, cy_ in centres)
            cands = [(gx + ux * d + px * side, gy + uy * d + py * side),
                     (gx + ux * d - px * side, gy + uy * d - py * side)]
            x, y = max(cands, key=clearance)
            if clearance((x, y)) < 0.72 * L:
                dd = clear + alen + 3 + max(tw, th) / 2
                x, y = gx + ux * dd, gy + uy * dd
            draw.text((x - tw / 2 - bb[0], y - th / 2 - bb[1]), label, fill=style.wall, font=font)
        else:                                        # exit: arrow out
            _arrow(draw, (gx + ux * clear, gy + uy * clear), (gx + ux * (clear + alen), gy + uy * (clear + alen)), color, sw)
    return img


def _align_visual_primitives(rec, source_size, board_size, offset):
    """Map autocropped pixel centres to the final canvas.

    board_law.render already subtracts the autocrop origin.  Pillow resize maps a
    pixel centre x to (x + 0.5) * new_width / old_width - 0.5; use the actual
    integer resized dimensions, then apply the paste translation.  Keep the
    record builders' nearest-integer point convention and lattice cells.
    """
    primitives = rec.get("visual_primitives")
    if not primitives or primitives.get("coordinate_space") != "image_pixels":
        return
    sx, sy = board_size[0] / source_size[0], board_size[1] / source_size[1]
    dx, dy = offset
    for entry in primitives["points"]:
        x, y = entry["point"]
        entry["point"] = [round((x + 0.5) * sx - 0.5 + dx),
                          round((y + 0.5) * sy - 0.5 + dy)]


def render_record(spec: Spec, case, task, seed: int, palettes, out_size: int = 448):
    from PIL import Image, ImageChops
    rng = detour_core.StyleRng(f"{RNG_NAMESPACE}:render:{seed}:{spec.split}:{spec.block}")
    img_rel = f"images/{spec.split}_b{spec.block:06d}_maze.png"
    lat = lattice_of(spec)
    menu = wall_px_menu(lat)
    if spec.exit:                                    # nothing inside the maze: block walls allowed
        menu = block_wall_px_menu(lat)
    wall_px = random.Random(f"{RNG_NAMESPACE}:wall:{seed}:{spec.split}:{spec.block}").choice(menu)
    lat.styled_width = lambda _thickness, w=wall_px: w      # this instance only
    saved_render = lattice.render_case
    if spec.edge:
        enlarge_margin(lat)
        lattice.render_case = render_case_edge    # board_law.render looks it up on the module
    try:
        with builder_config(spec.tier, lat):
            if task in ("tf",) or (task == "mc" and spec.form == "mc"):
                base, rec = board_law.render(lat, case, task, rng, palettes, spec.block, img_rel, spec.tier, case["K_used"])
            else:
                base, rec = corridor_questions.render(lat, case, task, rng, palettes, spec.block, img_rel, spec.tier, case["K_used"])
    finally:
        lattice.render_case = saved_render
    source_size = base.size
    if max(base.size) > out_size:
        # autocrop leaves a 14 px pad; the centred paste below trims it (that
        # is exactly what detour_core did with its 444 px 21x21 boards).  Only a
        # board whose drawn content itself does not fit is scaled, and then
        # the scale is recorded.
        bb = ImageChops.difference(base, Image.new("RGB", base.size, base.getpixel((2, 2)))).getbbox()
        cw, ch = (bb[2] - bb[0], bb[3] - bb[1]) if bb else base.size
        if cw > out_size or ch > out_size:
            f = out_size / max(cw, ch)
            base = base.resize((int(base.width * f), int(base.height * f)), Image.Resampling.LANCZOS)
            rec.setdefault("axis", {})["render_scale"] = round(f, 4)
    canvas = Image.new("RGB", (out_size, out_size), base.getpixel((2, 2)))
    offset = ((out_size - base.width) // 2, (out_size - base.height) // 2)
    canvas.paste(base, offset)
    _align_visual_primitives(rec, source_size, base.size, offset)
    if task == "count":
        rec["options"] = [str(i) for i in range(detour_core.N_WITHIN + 1)]
        rec["qa"]["choices"] = rec["options"]
    rec["id"] = f"maze_{spec.split}_{spec.block:08d}"
    rec["source_id"] = rec["id"]
    rec["task"] = "maze"
    rec["tier"] = rec["difficulty"] = spec.tier
    rec["split"] = spec.split
    rec["question_form"] = FORM_NAME[spec.form]
    rec["rendered_image_size"] = [out_size, out_size]
    rec["image"] = img_rel
    if spec.lattice[0] != "rect":
        # the corridor_questions kth / within templates count "squares"; hex and polar cells are not squares
        for key in ("question", "qa_template"):
            if isinstance(rec.get(key), str):
                rec[key] = rec[key].replace("squares", "cells")
    ax = rec["axis"]
    ax["grid"] = list(spec.lattice[1:])
    ax["lattice"] = spec.lattice[0]
    ax["edge_entrances"] = spec.edge
    ax["exit_on_rim"] = spec.exit
    if spec.exit:
        rec["question"] = DESC_RE.sub("the exit", rec["question"])
        rec.setdefault("style", {})["target_shape"] = "exit"
    ax["board_px"] = list(base.size)
    rec.setdefault("style", {})["wall_px"] = wall_px
    ax["X"] = case.get("X")
    ax["D"] = case.get("D")
    ax["L1"] = case.get("L1")
    ax["form"] = spec.form
    ax["occurrence"] = spec.occurrence
    for k in ("cue_class", "kth", "R", "euclid_rank", "comp_frac", "target_is_largest", "key_cell", "key_W", "key_L1", "X_all", "W_all", "L1_all", "cand_dists"):
        if k in case:
            ax[k] = case[k]
    rec["meta"] = {"schema_version": RNG_NAMESPACE, "render_version": RENDER_VERSION,
                   "generation_seed": seed, "dataset_role": spec.dataset_role,
                   "is_ood": spec.dataset_role == "ood_eval", "split": spec.split, "block": spec.block,
                   "tier": spec.tier, "question_form": FORM_NAME[spec.form], "questions_per_image": 1,
                   "image_size": out_size, "image_pixel_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
                   "board_hash": hashlib.sha256((spec.lattice[0] + repr(sorted(case["walls"]))).encode()).hexdigest()}
    return canvas, rec


RETRY_SALTS = 24         # extra deterministic attempts (MAX_ATT boards each) before a fallback; a medium block needed 6 in a 500-board test


def block_record(seed: int, block: int, dataset_role: str, split: str, palettes):
    spec = schedule(block, dataset_role, split)
    case, task, meta = build_case(spec, seed)
    salt = 0
    # About 2% of easy blocks exhaust MAX_ATT boards (seen: an exit-on-rim kth,
    # a polar mc); a fresh board stream with the same plan digits almost always
    # builds.  Salts are a pure function of the block, so shards stay
    # reproducible, and the plan (answer, ranks, lattice) is unchanged.
    while case is None and salt < RETRY_SALTS:
        salt += 1
        case, task, meta = build_case(spec, seed + 7919 * salt)
    if case is None and (spec.edge or spec.exit):
        # last resort for a rim variant: the same block as an interior board
        spec = dataclasses.replace(spec, edge=False, exit=False, lattice=LATTICES[spec.tier][spec.lattice[0]])
        case, task, meta = build_case(spec, seed)
        salt = -1
    if case is None:
        raise RuntimeError(f"block {block} ({spec.tier}/{spec.form}) unsatisfiable: {meta}")
    seed = seed + 7919 * max(salt, 0)
    audit = validate_case(spec, case, task)
    img, rec = render_record(spec, case, task, seed, palettes)
    rec["meta"].update({"board_attempts": meta.get("boards"), "relax_stage": meta.get("stage"),
                        "builder": meta.get("builder"), "retry_salt": salt})
    return rec, img, audit


def validate_case(spec: Spec, case, task) -> dict:
    """Recompute the answer from the walls before anything is rendered; return the
    accepted graph (cells, open edges) and markers, from which the answer follows."""
    lat = lattice_of(spec)
    walls = case["walls"]
    if task == "tf":
        reach = lattice.reachable(case["a"], lat.neighbors, walls)
        if (case["b"] in reach) != case["solvable"]:
            raise ValueError("tf answer disagrees with the walls")
    else:
        target = case["exit"]
        d = lattice.bfs_dist(target, lat.neighbors, walls)
        starts = case["starts"]
        if spec.form == "mc":
            reach = [lab for lab, c in starts if c in d]
            if reach != [case["answer"]]:
                raise ValueError(f"mc: reaching starts {reach} != {case['answer']}")
        elif spec.form == "closer":
            dist = {lab: d[c] for lab, c in starts}
            if min(dist, key=dist.get) != case["answer"] or len(set(dist.values())) < 2:
                raise ValueError("closer answer disagrees with the walls")
        elif spec.form == "kth":
            ranked = sorted(starts, key=lambda t: d[t[1]])
            if len({d[c] for _l, c in starts}) != 4 or ranked[case["kth"] - 1][0] != case["answer"]:
                raise ValueError("kth answer disagrees with the walls")
        elif spec.form == "within":
            n = sum(1 for _l, c in starts if d.get(c, 10 ** 9) <= case["R"])
            if str(n) != str(case["answer"]):
                raise ValueError("within answer disagrees with the walls")
    edges = sorted({lattice.wall_key(c, n) for c in lat.cells for n in lat.neighbors(c)
                    if lattice.wall_key(c, n) not in walls})
    return {"form": spec.form, "cells": list(lat.cells), "edges": edges,
            "starts": case.get("starts"), "target": case.get("exit"), "a": case.get("a"),
            "b": case.get("b"), "k": case.get("kth"), "radius": case.get("R")}
