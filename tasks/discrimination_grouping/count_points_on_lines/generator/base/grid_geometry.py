"""Lattice geometry and colours of count_points_on_lines: point-to-segment distance, the lattice
points on a segment, and the background, dot and line colours.
"""
from __future__ import annotations

import math

BG = (255, 255, 255)
DOT_COLOR = (35, 35, 35)
# (name, rgb) -- all bright/saturated so the dark dots stay visible ON the line
LINE_COLORS = [
    ("blue", (32, 140, 220)),
    ("red", (220, 50, 45)),
    ("green", (35, 165, 75)),
    ("purple", (150, 70, 200)),
    ("orange", (240, 140, 30)),
    ("teal", (20, 165, 165)),
    ("magenta", (210, 50, 160)),
    ("pink", (235, 105, 150)),
]


# --------------------------------------------------------------------------- geometry
def pt_seg_dist(px, py, ax, ay, bx, by):
    """Distance from point P to segment AB, and whether the foot lies within AB."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return math.hypot(px - ax, py - ay), True
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    tc = max(0.0, min(1.0, t))
    fx, fy = ax + tc * dx, ay + tc * dy
    d = math.hypot(px - fx, py - fy)
    within = -1e-9 <= t <= 1 + 1e-9
    return d, within


def collinear_grid_points(a, b):
    """All integer grid points lying exactly on segment a->b (inclusive)."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return [(ax, ay)]
    sx, sy = dx // g, dy // g
    return [(ax + k * sx, ay + k * sy) for k in range(g + 1)]
