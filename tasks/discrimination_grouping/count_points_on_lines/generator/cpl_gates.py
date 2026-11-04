"""Pixel-level gates of count_points_on_lines:

  G0 pixel budget   w%28==0 and h%28==0 and w*h<=451584   (cpl_core.budget_ok)
  G1 build ok       make_problem raises no Illegible and returns non-None
                    (this already contains the hard legibility floors
                     dia>=6 / lw>=2 / clear>=3, the near-miss count floor,
                     and a full independent geometry recheck by assert)
  G2 tier shape     grid_n == want_n and n_segs == want_segs
  G3 ON visible     in the final PNG every ON dot has line ink on BOTH sides
  G4 OFF white gap  the tightest OFF-dot white gap >= MIN_WHITE_GAP px
  G5 no false pos   no OFF dot satisfies the ON criterion
  G6 answer sound   red rings in the solution overlay == n_on (an independent
                    pixel-side cross-check of the integer-geometry answer)

G10 (ON-dot clearance) is defined at the bottom and called separately by
gen_tiers.gen_one.
"""
import math
from collections import deque
import numpy as np

from . import cpl_core, cpl_ink
from .base.grid_geometry import LINE_COLORS, collinear_grid_points, pt_seg_dist

PAD = 2.5
MIN_WHITE_GAP = 1.0     # px; the two delivered sample images measured 3.0 and
                        # 1.5, so 1.0 is the hard floor


def _both_sides(lm, W, H, cx, cy, R):
    for k in range(180):
        th = math.pi * k / 180
        x1, y1 = int(round(cx + R*math.cos(th))), int(round(cy + R*math.sin(th)))
        x2, y2 = int(round(cx - R*math.cos(th))), int(round(cy - R*math.sin(th)))
        if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H \
           and lm[y1, x1] and lm[y2, x2]:
            return True
    return False


def _ring_count(sol):
    S = np.asarray(sol).astype(int)
    red = (S[:, :, 0] == 220) & (S[:, :, 1] == 0) & (S[:, :, 2] == 0)
    lab = np.zeros(red.shape, int); comp = 0
    ys, xs = np.nonzero(red)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        comp += 1; q = deque([(y0, x0)]); lab[y0, x0] = comp
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < red.shape[0] and 0 <= xx < red.shape[1] \
                       and red[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = comp; q.append((yy, xx))
    return comp


def pixel_gates(img, m, segs, on_set, to_px, dot_r):
    """G3/G4/G5/G6.  Returns (ok, failed_gate_name, diagnostics)."""
    A = np.asarray(img).astype(int)
    col = dict(LINE_COLORS)[m["line_color"]]
    lm, dark = cpl_ink.masks(A, np.array(col))
    H, W = A.shape[:2]
    N, sp = m["grid_n"], m["spacing_px"]
    R = dot_r + PAD

    # G3 -- every ON dot has line ink on both sides
    bad_on = [p for p in sorted(on_set) if not _both_sides(lm, W, H, *to_px(*p), R)]

    # G5 -- no OFF dot fires the same test
    off = [(gx, gy) for gx in range(N) for gy in range(N) if (gx, gy) not in on_set]
    fp = [p for p in off if _both_sides(lm, W, H, *to_px(*p), R)]

    # G4 -- tightest OFF-dot white gap
    worst = (1e9, None)
    for gx, gy in off:
        cx, cy = to_px(gx, gy); best = (1e9, None)
        for a, b in segs:
            (ax, ay), (bx, by) = cpl_core.overshot_px(a, b, to_px, sp)
            d, w = pt_seg_dist(cx, cy, ax, ay, bx, by)
            if w and d < best[0]:
                best = (d, (ax, ay, bx, by))
        d, seg = best
        if seg is None or d > 1.2 * sp:
            continue
        ax, ay, bx, by = seg
        t = ((cx-ax)*(bx-ax) + (cy-ay)*(by-ay)) / ((bx-ax)**2 + (by-ay)**2)
        t = max(0.0, min(1.0, t)); fx, fy = ax + t*(bx-ax), ay + t*(by-ay)
        ux, uy = (fx-cx)/d, (fy-cy)/d
        white = 0.0
        for s in np.arange(0.5, d, 0.5):
            x, y = int(round(cx+ux*s)), int(round(cy+uy*s))
            if not (0 <= x < W and 0 <= y < H):
                break
            if lm[y, x]:
                break
            if not dark[y, x]:
                white += 0.5
        if white < worst[0]:
            worst = (white, (gx, gy, round(d, 2)))

    # G6 -- red rings in the solution overlay
    sol = cpl_core.make_solution(img, segs, on_set, to_px, dot_r)
    rings = _ring_count(sol)

    diag = dict(on_visible=f"{len(on_set)-len(bad_on)}/{len(on_set)}",
                bad_on=bad_on, n_falsepos=len(fp), falsepos=fp[:5],
                min_white_gap=(None if worst[1] is None else worst[0]),
                min_gap_at=worst[1], rings=rings)
    if bad_on:
        return False, "G3_on_visible", diag
    if worst[1] is not None and worst[0] < MIN_WHITE_GAP:
        return False, "G4_white_gap", diag
    if fp:
        return False, "G5_falsepos", diag
    if rings != m["n_on"]:
        return False, "G6_ring_count", diag
    return True, None, diag


# ---------------------------------------------------------------- G10
MIN_ON_CLEAR_PX = 3.0

def on_clearance_gate(m, segs, on_set, to_px, dot_r, min_clear=MIN_ON_CLEAR_PX):
    """G10 -- carry the OFF dots' 3px clearance floor over to ON dots.

    The original gates only bound the distance from an OFF dot to its nearest
    segment (G1's near_margin).  For an ON dot they only guarantee that *some*
    segment runs through it; the SECOND nearest segment had no lower bound at
    all.  With many segments, line placement exempts dots already in on_set, so
    another line may shave past an ON dot's edge -- and then which segment that
    dot should be counted for becomes ambiguous.  Under the answer format
    "how many dots does each segment pass through" that ambiguity is fatal.

    This gate requires: for every ON dot P and every segment S that does NOT
    pass through P,
        dist(P, S) - dot_r - lw/2  >=  min_clear   (default 3.0 px)
    -- the same ruler and the same number the OFF dots already get.

    Returns (ok, min_clearance_px, worst_offender (dot, segment_index)).
    """
    sp, lw = m["spacing_px"], m["line_w_px"]
    pre = [(cpl_core.overshot_px(a, b, to_px, sp),
            set(collinear_grid_points(a, b))) for a, b in segs]
    worst_d, worst_at = 1e9, None
    for P in sorted(on_set):
        px, py = to_px(*P)
        for i, ((A, B), own) in enumerate(pre):
            if P in own:
                continue
            d, within = pt_seg_dist(px, py, A[0], A[1], B[0], B[1])
            if within and d < worst_d:
                worst_d, worst_at = d, (list(P), i)
    clear = worst_d - dot_r - lw / 2.0
    return (clear >= min_clear), round(clear, 2), worst_at
