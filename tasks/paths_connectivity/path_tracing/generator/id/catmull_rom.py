"""Bezier helpers of the path-tracing rope engine: curve sampling and Catmull-Rom control points."""

from __future__ import annotations

from shared.drawing.geometry import Point


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


def catmull_rom_controls(anchors: list[Point], tension: float = 1.0) -> list[list[Point]]:
    """Convert a polyline of anchors into a chain of cubic Bezier segments that
    share tangents at every anchor (C1-continuous). The spline passes smoothly
    through each waypoint, so there are no corners at the knots. Endpoint tangents
    are clamped (anchors duplicated at the ends)."""
    n = len(anchors)

    def at(i: int) -> Point:
        return anchors[max(0, min(n - 1, i))]

    segments: list[list[Point]] = []
    for j in range(n - 1):
        p_prev, p0, p1, p_next = at(j - 1), at(j), at(j + 1), at(j + 2)
        b1 = (p0[0] + (p1[0] - p_prev[0]) / 6 * tension,
              p0[1] + (p1[1] - p_prev[1]) / 6 * tension)
        b2 = (p1[0] - (p_next[0] - p0[0]) / 6 * tension,
              p1[1] - (p_next[1] - p0[1]) / 6 * tension)
        segments.append([p0, b1, b2, p1])
    return segments
