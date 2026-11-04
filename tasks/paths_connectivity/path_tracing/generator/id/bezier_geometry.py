"""Fixed Bezier geometry of the ``Bezier`` path generator.

Only the geometry-producing part of that generator is kept here.  Its numeric
constants and algorithms are intentionally fixed.  ``build_geometry_attempt``
adds endpoint-position indices to the return value; that bookkeeping does not
change any random draw or coordinate and lets the current renderer preserve its
number-label convention.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

Point = tuple[float, float]


_COLORS = (0.05, 0.05, 0.30, 0.60)
PT_LEVELS = {
    "easy": {
        "num_curves": (3, 4),
        "curve_complexity": "moderate",
        "length_scale": 1.2,
        "span_bias": 0.40,
        "crossings_per_curve": 0.5,
        "max_crossings_per_curve": 3.0,
        "max_knots": 4,
        "color_weights": _COLORS,
        "min_target_crossings": 0,
        "line_widths": (3, 4, 5),
    },
    "medium": {
        "num_curves": (5, 6),
        "curve_complexity": "moderate",
        "length_scale": 1.3,
        "span_bias": 0.60,
        "crossings_per_curve": 1.0,
        "max_crossings_per_curve": 6.0,
        "max_knots": 4,
        "color_weights": _COLORS,
        "min_target_crossings": 2,
        "line_widths": (3, 4, 4),
    },
    "hard": {
        "num_curves": (6, 7),
        "curve_complexity": "complex",
        "length_scale": 1.3,
        "span_bias": 0.80,
        "crossings_per_curve": 1.4,
        "max_crossings_per_curve": 9.0,
        "max_knots": 4,
        "color_weights": _COLORS,
        "min_target_crossings": 5,
        "line_widths": (3, 3, 4),
    },
    "xhard": {
        "num_curves": (7, 8),
        "curve_complexity": "complex",
        "length_scale": 1.4,
        "span_bias": 0.90,
        "crossings_per_curve": 1.8,
        "max_crossings_per_curve": 11.0,
        "max_knots": 4,
        "color_weights": _COLORS,
        "min_target_crossings": 8,
        "line_widths": (2, 3, 3),
    },
}

CURVE_COMPLEXITY = {
    "simple": {"knots": (1, 2), "swing": 0.13, "steps": 80, "trace_step": 22},
    "moderate": {"knots": (2, 3), "swing": 0.18, "steps": 100, "trace_step": 16},
    "complex": {"knots": (2, 4), "swing": 0.23, "steps": 120, "trace_step": 10},
}


def bezier(points: list[Point], t: float) -> Point:
    working = points[:]
    while len(working) > 1:
        working = [
            (
                (1 - t) * working[i][0] + t * working[i + 1][0],
                (1 - t) * working[i][1] + t * working[i + 1][1],
            )
            for i in range(len(working) - 1)
        ]
    return working[0]


def sample_bezier(points: list[Point], steps: int) -> list[Point]:
    return [bezier(points, i / steps) for i in range(steps + 1)]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return distance(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj = (ax + t * dx, ay + t * dy)
    return distance(p, proj)


def segment_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
    den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(den) < 1e-6:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / den
    u = ((ax - cx) * (ay - by) - (ay - cy) * (ax - bx)) / den
    if 0.03 <= t <= 0.97 and 0.03 <= u <= 0.97:
        return (ax + t * (bx - ax), ay + t * (by - ay))
    return None


def polyline_crossings(a: list[Point], b: list[Point]) -> list[tuple[Point, float]]:
    out: list[tuple[Point, float]] = []
    for p1, p2 in zip(a, a[1:]):
        for q1, q2 in zip(b, b[1:]):
            hit = segment_intersection(p1, p2, q1, q2)
            if hit is None:
                continue
            d1 = (p2[0] - p1[0], p2[1] - p1[1])
            d2 = (q2[0] - q1[0], q2[1] - q1[1])
            n1, n2 = math.hypot(*d1), math.hypot(*d2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosv = abs(d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
            out.append((hit, math.degrees(math.acos(max(0.0, min(1.0, cosv))))))
    return out


def clamp_point(p: Point, width: int, height: int, margin: int) -> Point:
    return (
        max(margin, min(width - margin, p[0])),
        max(margin, min(height - margin, p[1])),
    )


def layout_params(width: int, height: int) -> dict:
    scale = min(width, height) / 600.0
    return {
        "margin": max(28, round(64 * scale)),
        "band": max(20, round(48 * scale)),
        "min_gap": max(34, round(64 * scale)),
        "marker_r": max(12, round(24 * scale)),
        "knot_margin": max(30, round(76 * scale)),
        "ctrl_margin": max(20, round(44 * scale)),
        "font": max(13, round(23 * scale)),
        "grid": max(20, round(32 * scale)),
    }

def _perimeter_point(
    s: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    inward: float,
) -> Point:
    width, height = x1 - x0, y1 - y0
    if s < width:
        return (x0 + s, y0 + inward)
    s -= width
    if s < height:
        return (x1 - inward, y0 + s)
    s -= height
    if s < width:
        return (x1 - s, y1 - inward)
    s -= width
    return (x0 + inward, y1 - s)


def random_marker_points(n: int, width: int, height: int, rng: random.Random) -> list[Point]:
    lp = layout_params(width, height)
    margin, radius = lp["margin"], lp["marker_r"]
    x0, y0, x1, y1 = margin, margin, width - margin, height - margin
    perimeter = 2 * ((x1 - x0) + (y1 - y0))
    base = perimeter / n
    min_gap = 2 * radius + 8
    offset = rng.random() * perimeter

    def at(s: float) -> Point:
        return _perimeter_point(s % perimeter, x0, y0, x1, y1, 0.0)

    arcs = sorted(
        offset + i * base + rng.uniform(-base * 0.18, base * 0.18)
        for i in range(n)
    )
    for _ in range(120):
        arcs = sorted(s % perimeter for s in arcs)
        pts = [at(s) for s in arcs]
        moved = False
        for i in range(n):
            j = (i + 1) % n
            gap = distance(pts[i], pts[j])
            if gap < min_gap:
                arcs[j] = arcs[j] + (min_gap - gap) * 0.5
                moved = True
        if not moved:
            break

    points = [at(s) for s in sorted(s % perimeter for s in arcs)]
    rng.shuffle(points)
    return points


def interior_knots(
    start: Point,
    end: Point,
    width: int,
    height: int,
    curve_complexity: str,
    rng: random.Random,
    radius_scale: float = 1.0,
    length_scale: float = 1.0,
    max_knots: int = 4,
) -> list[Point]:
    profile = CURVE_COMPLEXITY[curve_complexity]
    kmin, kmax = profile["knots"]
    if length_scale != 1.0:
        kmin = max(1, round(kmin * length_scale))
        kmax = min(max_knots, max(kmin, round(kmax * length_scale)))
    n_knots = rng.randint(kmin, kmax)
    if n_knots == 0:
        return []

    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    pnx, pny = -dy / length, dx / length
    swing = length * profile["swing"] * radius_scale * 0.72
    knot_margin = layout_params(width, height)["knot_margin"]
    ts = sorted(rng.uniform(0.12, 0.88) for _ in range(n_knots))
    side = rng.choice([-1, 1])
    knots = []
    for k, t in enumerate(ts):
        bx, by = sx + dx * t, sy + dy * t
        off = side * (-1) ** k * rng.uniform(0.4, 1.0) * swing
        knots.append(
            clamp_point((bx + pnx * off, by + pny * off), width, height, knot_margin)
        )
    return knots


def catmull_rom_controls(anchors: list[Point], tension: float = 1.0) -> list[list[Point]]:
    n = len(anchors)

    def at(i: int) -> Point:
        return anchors[max(0, min(n - 1, i))]

    segments: list[list[Point]] = []
    for j in range(n - 1):
        p_prev, p0, p1, p_next = at(j - 1), at(j), at(j + 1), at(j + 2)
        b1 = (
            p0[0] + (p1[0] - p_prev[0]) / 6 * tension,
            p0[1] + (p1[1] - p_prev[1]) / 6 * tension,
        )
        b2 = (
            p1[0] - (p_next[0] - p0[0]) / 6 * tension,
            p1[1] - (p_next[1] - p0[1]) / 6 * tension,
        )
        segments.append([p0, b1, b2, p1])
    return segments


def build_curve(
    start: Point,
    end: Point,
    width: int,
    height: int,
    curve_complexity: str,
    rng: random.Random,
    length_scale: float = 1.0,
    max_knots: int = 4,
) -> tuple[list[Point], list[Point]]:
    profile = CURVE_COMPLEXITY[curve_complexity]
    radius_scale = rng.uniform(0.7, 1.3)
    tension = rng.uniform(0.8, 1.15)
    anchors = [
        start,
        *interior_knots(
            start,
            end,
            width,
            height,
            curve_complexity,
            rng,
            radius_scale,
            length_scale,
            max_knots,
        ),
        end,
    ]
    control_margin = layout_params(width, height)["ctrl_margin"]
    steps_per_segment = profile["steps"]
    controls: list[Point] = []
    dense: list[Point] = []
    for segment_index, segment in enumerate(catmull_rom_controls(anchors, tension)):
        segment = [
            segment[0],
            clamp_point(segment[1], width, height, control_margin),
            clamp_point(segment[2], width, height, control_margin),
            segment[3],
        ]
        controls.extend(segment if segment_index == 0 else segment[1:])
        sampled = sample_bezier(segment, steps_per_segment)
        dense.extend(sampled if segment_index == 0 else sampled[1:])
    return controls, dense


def endpoint_safe(
    curve: list[Point],
    markers: Iterable[Point],
    own: set[Point],
    radius: float,
) -> bool:
    check_curve = curve[::4]
    if check_curve[-1] != curve[-1]:
        check_curve.append(curve[-1])
    for marker in markers:
        if marker in own:
            continue
        for a, b in zip(check_curve, check_curve[1:]):
            if point_segment_distance(marker, a, b) < radius:
                return False
    return True
