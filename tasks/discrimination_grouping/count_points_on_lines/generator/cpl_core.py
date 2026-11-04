"""count_points_on_lines -- canvas budget, legibility floors, segment directions, overshoot.

The canvas is 672x672: 672 = 28*24 and 672*672 = 451,584 sits exactly on the
pixel budget of the Qwen smart_resize ceiling; it is the largest *square*
canvas inside the budget, and the dot lattice is square, so a wide canvas buys
nothing here.  A larger canvas would be scaled back down and every legibility
number measured on it would be void.

The three absolute pixel floors (dot diameter 6px / line width 2px /
clearance 3px) limit N; the difficulty that comes from structure is:

  1. No trivial slopes.  Directions are primitive vectors (p,q) with gcd 1,
     p != 0, q != 0, |p| != |q|.  Slope 0 / inf / +-1 are all banned, which
     kills the three "scan a row / a column / the diagonal" shortcuts.
  2. Forced near misses.  Every image must contain at least `nearmiss` true
     near-miss OFF dots (between near_margin and 1.7*near_margin from a
     segment), so each dot has to be judged individually.
  3. Endpoint overshoot.  Each segment is extended 0.35*spacing past both
     end dots, so at an endpoint the line visibly runs *through* the dot
     instead of stopping on it -- that removes the "does the endpoint count"
     ambiguity.
"""
from __future__ import annotations
import math
from PIL import ImageDraw

# ---------------------------------------------------------------- Qwen pixel budget
MAX_PIXELS = 451584          # Qwen2.5-VL max_pixels of the paper's training
PATCH = 28                   # patch14 * merge2

def budget_ok(w, h):
    return (w % PATCH == 0) and (h % PATCH == 0) and (w * h <= MAX_PIXELS)

# ------------------------------------------------- hard legibility floors (never lowered)
MIN_DOT_DIA_PX = 6.0    # black dot diameter in the final image
MIN_LINE_W_PX  = 2.0    # line width in the final image
MIN_CLEAR_PX   = 3.0    # clearance: OFF dot edge <-> line edge
OVERSHOOT      = 0.35   # each segment is extended 0.35*spacing past both ends

class Illegible(Exception):
    pass

def legibility_gate(spacing, dot_r, lw, near_margin):
    dia = 2 * dot_r
    clear = near_margin - dot_r - lw / 2.0
    if dia < MIN_DOT_DIA_PX:
        raise Illegible(f"dot dia {dia:.2f} < {MIN_DOT_DIA_PX}")
    if lw < MIN_LINE_W_PX:
        raise Illegible(f"line w {lw:.2f} < {MIN_LINE_W_PX}")
    if clear < MIN_CLEAR_PX:
        raise Illegible(f"clearance {clear:.2f} < {MIN_CLEAR_PX}")
    return dict(dot_dia_px=round(dia, 2), line_w_px=round(lw, 2),
                clearance_px=round(clear, 2))

# ---------------------------------------------------------------- non-trivial directions
def nontrivial_dirs(maxc=5):
    """Primitive direction vectors, excluding slope 0 / inf / +-1."""
    out = []
    for p in range(1, maxc + 1):
        for q in range(1, maxc + 1):
            if math.gcd(p, q) != 1:
                continue
            if p == q:            # slope +-1
                continue
            out.append((p, q)); out.append((p, -q))
    return out

DIRS = nontrivial_dirs(5)

# ---------------------------------------------------------------- segment geometry
def overshot_px(a, b, to_px, spacing):
    """Extend the segment by OVERSHOOT*spacing at each end.  The next lattice
    point along the line is at least sqrt(5)*spacing away, so the overshoot can
    never turn a new lattice point into an ON dot."""
    ax, ay = to_px(*a); bx, by = to_px(*b)
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    e = OVERSHOOT * spacing
    return (ax - ux * e, ay - uy * e), (bx + ux * e, by + uy * e)

def make_solution(img, segs, on_set, to_px, dot_r):
    sol = img.copy()
    d = ImageDraw.Draw(sol)
    r = dot_r * 2.4
    for gx, gy in sorted(on_set):
        cx, cy = to_px(gx, gy)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(220, 0, 0), width=2)
    return sol

