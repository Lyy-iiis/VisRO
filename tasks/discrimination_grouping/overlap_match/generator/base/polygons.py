"""Polygon helpers of overlap_match: star polygons, stroke point sets, diameters and Hausdorff distances."""
from __future__ import annotations

import math
import random

import numpy as np


def derive_rng(seed, idx):
    return random.Random(f"{seed}:{idx}")


# --------------------------------------------------------------------------- #
# Polygon machinery
# --------------------------------------------------------------------------- #
def sample_polygon(rng, n_vert=None, radius=80.0):
    """Star polygon around the origin: sorted angles + radii in [0.45,1]R.
    Star-shaped w.r.t. the centroid => always simple (non-self-intersecting)."""
    n = n_vert or rng.randint(4, 7)
    phase = rng.uniform(0, 2 * math.pi)
    for _ in range(80):
        # jittered-uniform angles: gaps >= 0.4 * (2pi/n) by construction
        angs = [phase + 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
                for i in range(n)]
        pts = []
        for a in angs:
            r = radius * rng.uniform(0.45, 1.0)
            pts.append((r * math.cos(a), r * math.sin(a)))
        if polygon_area(pts) >= 0.12 * radius * radius:
            return center_pts(pts)
    raise RuntimeError("polygon sampling failed")


def polygon_area(pts):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def center_pts(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(x - cx, y - cy) for x, y in pts]


def diameter(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def poly_stroke_points(pts, step=2.0):
    """Points sampled along the outline every `step` px (centroid frame)."""
    out = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        L = math.hypot(x2 - x1, y2 - y1)
        n = max(2, int(L / step))
        for t in range(n):
            f = t / n
            out.append((x1 + f * (x2 - x1), y1 + f * (y2 - y1)))
    return np.array(out)


# --------------------------------------------------------------------------- #
# Hausdorff gate (stroke point sets, centroid-aligned)
# --------------------------------------------------------------------------- #
def hausdorff(a, b, cap=3000):
    if len(a) > cap:
        a = a[np.linspace(0, len(a) - 1, cap).astype(int)]
    if len(b) > cap:
        b = b[np.linspace(0, len(b) - 1, cap).astype(int)]
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return max(np.sqrt(d2.min(axis=1)).max(), np.sqrt(d2.min(axis=0)).max())
