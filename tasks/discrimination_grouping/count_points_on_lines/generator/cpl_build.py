"""count_points_on_lines -- problem builder.

gen_tiers.py drives this module: it registers the five tiers in DIAL and writes
each tier's acuity constants into this module's globals before building.  This
is the only place a problem is built and rendered.

======================================================================
The question form (shared by every tier)
----------------------------------------------------------------------
The answer gives how many dots each segment passes through (descending) and the
deduplicated total; both halves must be right.
    answer string looks like   9,7,6,6,5;28
That does three things at once:
  * the guessing floor -- exact-matching an L-tuple plus a total is
    effectively impossible to hit by chance;
  * no error cancellation -- one segment over by 1 and another under by 1
    cannot cancel inside a sum;
  * every intermediate count has to be written down.

======================================================================
The difficulty axes (number of segments L; grid side N)
----------------------------------------------------------------------
Larger L => more numbers that must all be right (multiplicative loss);
larger N => smaller spacing, smaller dots, tighter clearance, and longer
lines (more dots to count per line).  N=20 is the ceiling that a 672x672
canvas plus the hard legibility floors (dot dia >= 6 / line width >= 2 /
clearance >= 3) can still support: at N=21 the margin band is empty, so past
N=20 only L can still grow.

======================================================================
Gates
----------------------------------------------------------------------
From cpl_core / cpl_gates:
  G0 pixel budget   w%28==0 && h%28==0 && w*h<=451584
  G1 legibility     dot dia >= 6px, line width >= 2px, OFF clearance >= 3px;
                    near-miss count met; full geometry recheck by assert
  G2 tier shape     grid_n / n_segs exactly equal to the tier's values
  G3 ON dots have line ink on both sides (pixels)
  G4 tightest OFF white gap >= 1.0px (pixels)
  G5 no false-positive OFF dot (pixels)
  G6 red rings in the solution overlay == deduplicated total (pixels)
  * no trivial slopes (0 / inf / +-1 all banned)
  * endpoint overshoot 0.35*spacing (removes the "does the endpoint count"
    ambiguity)
For the per-segment answer format -- L must be countable and each segment
traceable:
  G7 endpoint resolution: any two segments' overshot endpoints are at least
     0.80*spacing apart.  No shared-endpoint V shapes, no "almost joined"
     false continuations; L can be counted.
  G8 near-parallel separation: if two segments' directions differ by less
     than 15 degrees, their minimum distance is at least 1.0*spacing.  Two
     nearly coincident lines can never be read as one.
  G9 per-line uniqueness: the dot count of each segment = gcd(dx,dy)+1 from
     integer geometry, cross-checked by an independent second implementation
     (scan every lattice point against the segment).  The two must agree.

======================================================================
Where the answer comes from
----------------------------------------------------------------------
Per-segment counts = len(collinear_grid_points(a,b)) -- integer geometry,
equal to gcd+1.  The deduplicated total = size of the union of ON dots over
all segments.  Both come from the generator's own integer coordinates and are
then cross-checked by the pixel-side red-ring count (G6).  The answer is
NEVER read back out of pixels.
"""
from __future__ import annotations
import math
from collections import Counter
from PIL import Image, ImageDraw

from . import cpl_core
from .base.grid_geometry import BG, DOT_COLOR, LINE_COLORS, collinear_grid_points, pt_seg_dist
from .cpl_core import Illegible, overshot_px

SIZE = 672                 # 28*24 ; 672*672 = 451584 = MAX_PIXELS, exactly on the line
MARGIN_CANVAS = 0.08       # canvas margin fraction (0.12 -> 0.08 to buy spacing for N=20)
DOT_R_FRAC = 0.10
LW_FRAC = 0.050
ON_TOL = 0.13
MARGIN_LO, MARGIN_HI = 0.24, 0.26
NEARMISS_BAND = 1.7

# --- G7/G8 thresholds
EP_SEP_FRAC = 0.80         # G7 min pairwise endpoint distance / spacing
PARA_ANG_DEG = 15.0        # G8 angle (degrees) below which two lines count as parallel
PARA_SEP_FRAC = 1.00       # G8 min distance between near-parallel segments / spacing


# --------------------------------------------------------------- tiers
DIAL = {}                  # tier -> dict(dial, N, segs, nearmiss); filled by gen_tiers


# --------------------------------------------------------------- geometry helpers
def seg_seg_dist(a1, a2, b1, b2):
    """Minimum distance between two segments, in pixel coordinates."""
    def p2s(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))
    # intersection first
    (x1, y1), (x2, y2) = a1, a2
    (x3, y3), (x4, y4) = b1, b2
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) > 1e-12:
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
        u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            return 0.0
    return min(p2s(x1, y1, x3, y3, x4, y4), p2s(x2, y2, x3, y3, x4, y4),
               p2s(x3, y3, x1, y1, x2, y2), p2s(x4, y4, x1, y1, x2, y2))


def dir_angle_deg(u, v):
    """Angle between two direction vectors, folded into [0, 90]."""
    a = math.degrees(math.atan2(u[1], u[0])) % 180.0
    b = math.degrees(math.atan2(v[1], v[0])) % 180.0
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def gen_candidate(N, rng):
    p, q = rng.choice(cpl_core.DIRS)  # slope 0/inf/+-1 all banned
    if rng.random() < 0.5:
        p, q = -p, -q
    kmax = min((N - 1) // abs(p), (N - 1) // abs(q))
    if kmax < 3:
        return None
    k = rng.randint(max(3, kmax // 2), kmax)
    if k * math.hypot(p, q) < 0.55 * (N - 1):
        return None
    dx, dy = p * k, q * k
    ax = rng.randint(max(0, -dx), min(N - 1, N - 1 - dx))
    ay = rng.randint(max(0, -dy), min(N - 1, N - 1 - dy))
    return (ax, ay), (ax + dx, ay + dy)


def line_key(a, b):
    (ax, ay), (bx, by) = a, b
    A, B = by - ay, ax - bx
    C = -(A * ax + B * ay)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) or 1
    A, B, C = A // g, B // g, C // g
    if (A, B, C) < (-A, -B, -C):
        A, B, C = -A, -B, -C
    return (A, B, C)


# --------------------------------------------------------------- placement (with G7/G8)
def build_segments(N, rng, target, to_px, near_margin, spacing, tries=20000):
    segs, on_set, keys = [], set(), set()
    ends = []          # overshot endpoints of the segments already placed (pixels)
    drawn = []         # overshot segments already placed (pixels) + direction
    px_cache = {(gx, gy): to_px(gx, gy) for gx in range(N) for gy in range(N)}
    ep_sep = EP_SEP_FRAC * spacing
    para_sep = PARA_SEP_FRAC * spacing
    for _ in range(tries):
        if len(segs) >= target:
            break
        c = gen_candidate(N, rng)
        if c is None:
            continue
        a, b = c
        k = line_key(a, b)
        if k in keys:
            continue
        A, B = overshot_px(a, b, to_px, spacing)
        u = (B[0] - A[0], B[1] - A[1])
        # G7 endpoint resolution
        bad = False
        for e in ends:
            if math.hypot(A[0] - e[0], A[1] - e[1]) < ep_sep or \
               math.hypot(B[0] - e[0], B[1] - e[1]) < ep_sep:
                bad = True
                break
        if bad:
            continue
        # G8 near-parallel separation
        for (A2, B2, u2) in drawn:
            if dir_angle_deg(u, u2) < PARA_ANG_DEG and \
               seg_seg_dist(A, B, A2, B2) < para_sep:
                bad = True
                break
        if bad:
            continue
        # near-miss margin: the candidate may not pull any dot that is neither on
        # itself nor already ON to within near_margin
        s_on = set(collinear_grid_points(a, b))
        for gp, (px, py) in px_cache.items():
            if gp in s_on or gp in on_set:
                continue
            d, within = pt_seg_dist(px, py, A[0], A[1], B[0], B[1])
            if within and d <= near_margin:
                bad = True
                break
        if bad:
            continue
        segs.append((a, b))
        on_set |= s_on
        keys.add(k)
        ends += [A, B]
        drawn.append((A, B, u))
    return segs, on_set


# --------------------------------------------------------------- problem construction
def make_problem(rng, tier, size=SIZE, tries=60, force_color="blue"):
    cfg = DIAL[tier]
    if not cpl_core.budget_ok(size, size):
        raise Illegible(f"canvas {size}x{size} exceeds the Qwen budget")
    N = cfg["N"]
    target = cfg["segs"]

    margin_px = int(size * MARGIN_CANVAS)
    span = size - 2 * margin_px
    spacing = span / (N - 1)
    dot_r = max(cpl_core.MIN_DOT_DIA_PX / 2.0, DOT_R_FRAC * spacing)
    lw = max(cpl_core.MIN_LINE_W_PX, LW_FRAC * spacing)
    on_tol = ON_TOL * spacing
    lo = max(MARGIN_LO, (cpl_core.MIN_CLEAR_PX + dot_r + lw / 2.0) / spacing)
    if lo > MARGIN_HI:
        raise Illegible(f"margin band empty: need >= {lo:.3f} > {MARGIN_HI}")
    margin_frac = rng.uniform(lo, MARGIN_HI)
    near_margin = margin_frac * spacing
    leg = cpl_core.legibility_gate(spacing, dot_r, lw, near_margin)

    def to_px(gx, gy):
        return (margin_px + gx * spacing, margin_px + gy * spacing)

    want_nm = cfg["nearmiss"]

    def nearmiss_count(segs, on_set):
        c = 0
        pre = [overshot_px(a, b, to_px, spacing) for a, b in segs]
        for gx in range(N):
            for gy in range(N):
                if (gx, gy) in on_set:
                    continue
                px, py = to_px(gx, gy)
                dmin = 1e9
                for (A, B) in pre:
                    d, within = pt_seg_dist(px, py, A[0], A[1], B[0], B[1])
                    if within:
                        dmin = min(dmin, d)
                if near_margin < dmin <= NEARMISS_BAND * near_margin:
                    c += 1
        return c

    best = None
    for _ in range(tries):
        segs, on_set = build_segments(N, rng, target, to_px, near_margin, spacing)
        if len(segs) < target:
            continue
        n_on = len(on_set)
        if n_on < 8 or n_on > N * N - 4:
            continue
        nm = nearmiss_count(segs, on_set)
        score = (min(nm, want_nm), nm, n_on)
        if best is None or score > best[0]:
            best = (score, segs, on_set, nm)
        if nm >= want_nm:
            break
    if best is None or best[3] < want_nm:
        return None
    _, segs, on_set, n_nearmiss = best
    n_on = len(on_set)

    # ---- full geometry recheck, independent of the incremental logic
    pre = [overshot_px(a, b, to_px, spacing) for a, b in segs]
    off_min = 1e9
    for gx in range(N):
        for gy in range(N):
            px, py = to_px(gx, gy)
            dmin = 1e9
            for (A, B) in pre:
                d, within = pt_seg_dist(px, py, A[0], A[1], B[0], B[1])
                if within:
                    dmin = min(dmin, d)
            if (gx, gy) in on_set:
                assert dmin <= on_tol, "an ON dot is not on its line"
            else:
                assert dmin > near_margin, f"OFF dot too close {dmin:.2f}"
                off_min = min(off_min, dmin)

    # ---- G7/G8 recheck, run in full and independent of the incremental logic
    ep_sep = EP_SEP_FRAC * spacing
    para_sep = PARA_SEP_FRAC * spacing
    min_ep = 1e9
    min_para = None
    for i in range(len(pre)):
        Ai, Bi = pre[i]
        ui = (Bi[0] - Ai[0], Bi[1] - Ai[1])
        for j in range(i + 1, len(pre)):
            Aj, Bj = pre[j]
            uj = (Bj[0] - Aj[0], Bj[1] - Aj[1])
            for e1 in (Ai, Bi):
                for e2 in (Aj, Bj):
                    min_ep = min(min_ep, math.hypot(e1[0] - e2[0], e1[1] - e2[1]))
            ang = dir_angle_deg(ui, uj)
            if ang < PARA_ANG_DEG:
                d = seg_seg_dist(Ai, Bi, Aj, Bj)
                min_para = d if min_para is None else min(min_para, d)
    assert min_ep >= ep_sep - 1e-6, f"G7 endpoint resolution lost {min_ep:.2f} < {ep_sep:.2f}"
    assert min_para is None or min_para >= para_sep - 1e-6, \
        f"G8 near-parallel separation lost {min_para:.2f}"

    # ---- G9 per-line counts: integer geometry + a scan, two independent impls
    per_line = []
    for a, b in segs:
        n1 = len(collinear_grid_points(a, b))
        n2 = math.gcd(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1
        # second implementation: test every lattice point against this segment
        A, B = overshot_px(a, b, to_px, spacing)
        n3 = 0
        for gx in range(N):
            for gy in range(N):
                px, py = to_px(gx, gy)
                d, within = pt_seg_dist(px, py, A[0], A[1], B[0], B[1])
                if within and d <= on_tol:
                    n3 += 1
        assert n1 == n2 == n3, f"G9 per-line counts disagree {n1}/{n2}/{n3}"
        per_line.append(n1)
    assert set().union(*[set(collinear_grid_points(a, b)) for a, b in segs]) == on_set

    per_line_sorted = sorted(per_line, reverse=True)
    ans = ",".join(str(x) for x in per_line_sorted) + ";" + str(n_on)

    # shared-dot bookkeeping: sum(per_line) - total = number of double counts
    cnt = Counter()
    for a, b in segs:
        for p in collinear_grid_points(a, b):
            cnt[p] += 1
    shared = {f"{p[0]},{p[1]}": c for p, c in cnt.items() if c >= 2}
    assert sum(per_line) - n_on == sum(c - 1 for c in cnt.values() if c >= 2)

    # ---- render
    ss = 3
    big = Image.new("RGB", (size * ss, size * ss), BG)
    d = ImageDraw.Draw(big)
    cname, line_col = rng.choice(LINE_COLORS)
    if force_color is not None:
        cname, line_col = force_color, dict(LINE_COLORS)[force_color]
    for (A, B) in pre:
        d.line([(A[0] * ss, A[1] * ss), (B[0] * ss, B[1] * ss)], fill=line_col,
               width=max(1, int(round(lw * ss))))
    rr = dot_r * ss
    for gx in range(N):
        for gy in range(N):
            x, y = to_px(gx, gy)
            cx, cy = x * ss, y * ss
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=DOT_COLOR)
    img = big.resize((size, size), Image.LANCZOS)

    slopes = []
    for a, b in segs:
        dx, dy = b[0] - a[0], b[1] - a[1]
        g = math.gcd(abs(dx), abs(dy))
        slopes.append((dx // g, dy // g))

    meta = dict(tier=tier, dial=cfg["dial"], size=size, canvas=f"{size}x{size}",
                budget_ok=cpl_core.budget_ok(size, size), pixels=size * size,
                grid_n=N, n_dots=N * N, n_on=n_on, n_segs=len(segs),
                line_color=cname, spacing_px=round(spacing, 2),
                margin_frac=round(margin_frac, 3),
                near_margin_px=round(near_margin, 2),
                off_min_px=round(off_min, 2),
                n_nearmiss=n_nearmiss,
                nearmiss_band_px=round(NEARMISS_BAND * near_margin, 2),
                min_endpoint_sep_px=round(min_ep, 2),
                min_parallel_sep_px=(None if min_para is None else round(min_para, 2)),
                per_line=per_line, per_line_sorted=per_line_sorted,
                sum_per_line=sum(per_line), n_shared_pts=len(shared),
                shared_pts=shared, answer=ans, prim_dirs=slopes, **leg)
    return img, meta, segs, on_set, to_px, dot_r

