"""Path tracing with independent Bezier and rope engines.

Short, mostly monotone Bezier sweeps are visually clean but easy to trace, so
next to the Bezier branch this module has a fixed-endpoint rope branch in the
style of BabyVision's line-tracing tasks:

* shuffled left/right endpoint columns with clean horizontal terminal corridors;
* broad low-frequency C/S sweeps, loops, backtracking, and central re-entry;
* a four-to-five-rope scribble family and a larger permutation-rope family;
* difficulty selected from realized arc length, crossings, loops, reversals,
  distinct crossed ropes, and near-parallel runs rather than presets alone.

The default 25-degree crossing-angle filter applies between distinct queried
ropes. Self-crossings are measured separately because both branches retain the
same connectivity.

Ground truth is exact regardless of how hard the image looks: connectivity is the
known curve->endpoint assignment from generation, not anything read off pixels.
"""
from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

from .catmull_rom import catmull_rom_controls, sample_bezier

Point = tuple[float, float]


# -----------------------------------------------------------------------------
# geometry helpers
# -----------------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def polyline_length(pts: list[Point]) -> float:
    return sum(distance(p, q) for p, q in zip(pts, pts[1:]))


def clamp_point(p: Point, w: int, h: int, m: int) -> Point:
    return (max(m, min(w - m, p[0])), max(m, min(h - m, p[1])))


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return distance(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return distance(p, (ax + t * dx, ay + t * dy))


def _seg_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
    den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(den) < 1e-6:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / den
    u = ((ax - cx) * (ay - by) - (ay - cy) * (ax - bx)) / den
    if 0.03 <= t <= 0.97 and 0.03 <= u <= 0.97:
        return (ax + t * (bx - ax), ay + t * (by - ay))
    return None


def polyline_crossings(a: list[Point], b: list[Point]) -> list[tuple[Point, float]]:
    """Crossings between two polylines as (point, acute_angle_deg)."""
    out: list[tuple[Point, float]] = []
    for p1, p2 in zip(a, a[1:]):
        for q1, q2 in zip(b, b[1:]):
            hit = _seg_intersection(p1, p2, q1, q2)
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


def _coarse(pts: list[Point], step: int = 3) -> list[Point]:
    c = pts[::step]
    if c[-1] != pts[-1]:
        c.append(pts[-1])
    return c


# -----------------------------------------------------------------------------
# layout
# -----------------------------------------------------------------------------
def layout_params(w: int, h: int) -> dict:
    s = min(w, h) / 600.0
    return {
        "margin": max(28, round(64 * s)),
        "min_gap": max(34, round(64 * s)),
        "marker_r": max(12, round(24 * s)),
        "knot_margin": max(22, round(56 * s)),
        "font": max(13, round(23 * s)),
        "grid": max(20, round(32 * s)),
    }


def _perimeter_point(s: float, x0, y0, x1, y1) -> Point:
    w, h = x1 - x0, y1 - y0
    if s < w:
        return (x0 + s, y0)
    s -= w
    if s < h:
        return (x1, y0 + s)
    s -= h
    if s < w:
        return (x1 - s, y1)
    s -= w
    return (x0, y1 - s)


def perimeter_markers(n: int, w: int, h: int, rng: random.Random) -> list[Point]:
    lp = layout_params(w, h)
    m, r = lp["margin"], lp["marker_r"]
    x0, y0, x1, y1 = m, m, w - m, h - m
    perim = 2 * ((x1 - x0) + (y1 - y0))
    base = perim / n
    min_gap = 2 * r + 8
    offset = rng.random() * perim
    arcs = sorted(offset + i * base + rng.uniform(-base * 0.18, base * 0.18) for i in range(n))
    for _ in range(120):
        arcs = sorted(s % perim for s in arcs)
        pts = [_perimeter_point(s % perim, x0, y0, x1, y1) for s in arcs]
        moved = False
        for i in range(n):
            j = (i + 1) % n
            if distance(pts[i], pts[j]) < min_gap:
                arcs[j] += (min_gap - distance(pts[i], pts[j])) * 0.5
                moved = True
        if not moved:
            break
    pts = [_perimeter_point(s % perim, x0, y0, x1, y1) for s in sorted(s % perim for s in arcs)]
    rng.shuffle(pts)
    return pts


# -----------------------------------------------------------------------------
# arc-chain spine  ->  the heart of the rope engine
# -----------------------------------------------------------------------------
# Real "follow-the-line" puzzle curves are SMOOTH: long gentle sweeps and broad
# semicircle bends, with the RADIUS varying from one stretch to the next -- never
# sharp kinks or little loops. Gaussian steps on the *heading* would be a random
# walk: the heading accumulates without bound and the line spirals into 270deg
# hairpins. So CURVATURE is controlled directly and kept bounded: build the line as a chain of circular arcs, each with its own
# radius (=> heterogeneous curvature) and a bounded turn (<=~160deg, so a single
# arc is at most a broad semicircle and never a loop -- a loop needs 360deg). Arcs
# are chained tangentially (C1), their turn directions mostly alternate (a flowing
# S, like a river) with a guard that never lets the same-direction turn accumulate
# toward a full loop. Then a Brownian-bridge pins it to the two endpoints and the
# excursion is scaled to fit the canvas (endpoints fixed; bridge adds only a linear
# shear, which preserves smoothness).
def smooth_polyline(pts: list[Point], window: int = 2, passes: int = 1) -> list[Point]:
    """Light moving-average smoothing that keeps the endpoints fixed."""
    if len(pts) < 3:
        return pts
    out = pts
    for _ in range(passes):
        new = [out[0]]
        for i in range(1, len(out) - 1):
            lo, hi = max(0, i - window), min(len(out), i + window + 1)
            seg = out[lo:hi]
            new.append((sum(p[0] for p in seg) / len(seg), sum(p[1] for p in seg) / len(seg)))
        new.append(out[-1])
        out = new
    return out


def grow_snake(
    start: Point,
    heading0: float,
    w: int,
    h: int,
    rng: random.Random,
    target_len: float,
    r_lo: float,
    r_hi: float,
    turn_lo: float,
    turn_hi: float,
    obstacles: list[Point],
    step: float = 6.0,
    self_margin: float = 19.0,
    obstacle_margin: float = 30.0,
    flip_prob: float = 0.85,
    coil_drift: float = 0.0,
    repel: float = 1.0,
    radius_floor: float = 52.0,
    goal_x: float | None = None,
    drift_x: float = 0.0,
) -> list[Point]:
    """Grow ONE line as a free, SELF-AVOIDING, bounded-curvature snake from `start`
    (Pathfinder-style), heading `heading0`, until it reaches ~target_len OR can no
    longer advance without colliding. The endpoint is wherever it ends -- there is no
    target point to reach, so there is no bridge and therefore no endpoint hook/loop.

    Each step is a small arc; radius varies in [r_lo, r_hi] (heterogeneous curvature)
    and the per-arc turn is bounded and mostly alternating (a flowing S). A hard
    curvature cap (radius >= R_FLOOR) keeps every bend a broad sweep. Collision rules:
      * SELF: a new point may not come within `self_margin` of an earlier part of the
        same snake -> the snake can never cross or loop back on itself (no self-loops,
        and none at the endpoints in particular).
      * OBSTACLES: it also stays `obstacle_margin` from foreign MARKER points, so no
        line runs over another line's endpoint. (It MAY cross foreign line *bodies* --
        that's the intended difficulty -- those are not obstacles here.)
    On a would-be collision the snake simply ends, yielding a clean single strand."""
    lp = layout_params(w, h)
    margin = lp["knot_margin"]
    cxc, cyc = w / 2.0, h / 2.0
    max_dtheta = step / radius_floor    # lower radius_floor => tighter winding (denser tangle)
    sm2 = self_margin * self_margin
    om2 = obstacle_margin * obstacle_margin

    px, py = start
    heading = heading0
    pts: list[Point] = [(px, py)]
    direction = rng.choice([-1, 1])
    run_turn = 0.0
    grown = 0.0
    # recent points are naturally within self_margin; don't test against them
    skip_recent = max(6, int(2.2 * self_margin / step))

    while grown < target_len:
        R = rng.uniform(r_lo, r_hi)
        turn = math.radians(rng.uniform(turn_lo, turn_hi))
        # Higher flip_prob -> a flowing graceful S (good for easy/medium/hard). Lower
        # flip_prob -> longer same-direction runs so the snake SWEEPS across the canvas
        # instead of coiling (used only for xhard's very long lines). Cap accumulation
        # at 150deg so it never folds into a loop either way.
        if rng.random() < flip_prob or (run_turn + turn) > math.radians(150):
            direction = -direction
            run_turn = 0.0
        run_turn += turn
        nseg = max(2, int(R * turn / step))
        base = direction * turn / nseg
        for _ in range(nseg):
            steer = 0.0
            if drift_x > 0:                         # two-column mode: gentle rightward
                near_v = min(py - margin, (h - margin) - py)   # bias, vertical wall-steer only
                if near_v < margin:
                    steer += (0.5 if py < cyc else -0.5) * (1 - near_v / margin) * 0.5
                steer += max(-0.4, min(0.4, _wrap(0.0 - heading))) * drift_x
            else:
                near = min(px - margin, (w - margin) - px, py - margin, (h - margin) - py)
                if near < margin:                   # smoothly curve back from walls
                    steer += max(-0.3, min(0.3, _wrap(math.atan2(cyc - py, cxc - px) - heading))) * (1 - near / margin) * 0.5
            # ANTI-COIL (xhard only, coil_drift>0): drift away from the snake's own
            # recent centre, so a very long line travels into open canvas rather than
            # nesting arcs in one spot.
            if coil_drift > 0 and len(pts) > 24:
                recent = pts[-24:]
                mx = sum(q[0] for q in recent) / len(recent)
                my = sum(q[1] for q in recent) / len(recent)
                if (px - mx) ** 2 + (py - my) ** 2 > 1.0:
                    steer += max(-0.18, min(0.18, _wrap(math.atan2(py - my, px - mx) - heading))) * coil_drift
            # REPULSION (repel>0): steer away from a nearby own point / obstacle so the
            # snake glides AROUND it and keeps growing to full length instead of
            # stopping. Off for easy/medium/hard because it also straightens the line --
            # those levels want natural curving and don't need the extra length.
            if repel > 0:
                det = 2.4 * self_margin
                nrx = nry = 0.0
                if len(pts) > skip_recent:
                    for qx, qy in pts[:-skip_recent]:
                        dxp, dyp = px - qx, py - qy
                        d2 = dxp * dxp + dyp * dyp
                        if d2 < det * det and d2 > 1e-6:
                            wgt = (1.0 - math.sqrt(d2) / det) / d2 * 1000.0
                            nrx += dxp * wgt; nry += dyp * wgt
                for ox, oy in obstacles:
                    dxp, dyp = px - ox, py - oy
                    d2 = dxp * dxp + dyp * dyp
                    if d2 < (obstacle_margin * 2.2) ** 2 and d2 > 1e-6:
                        wgt = (1.0 - math.sqrt(d2) / (obstacle_margin * 2.2)) / d2 * 1400.0
                        nrx += dxp * wgt; nry += dyp * wgt
                if nrx or nry:
                    steer += max(-0.5, min(0.5, _wrap(math.atan2(nry, nrx) - heading))) * min(1.0, math.hypot(nrx, nry)) * repel
            dth = max(-max_dtheta, min(max_dtheta, base + steer))
            nh = heading + dth
            nx = px + step * math.cos(nh)
            ny = py + step * math.sin(nh)
            nx, ny = clamp_point((nx, ny), w, h, margin // 2)
            # collision: if repulsion couldn't avoid it, end the snake cleanly
            hit = False
            if len(pts) > skip_recent:
                for qx, qy in pts[:-skip_recent]:
                    if (nx - qx) ** 2 + (ny - qy) ** 2 < sm2:
                        hit = True
                        break
            if not hit:
                for ox, oy in obstacles:
                    if (nx - ox) ** 2 + (ny - oy) ** 2 < om2:
                        hit = True
                        break
            if hit:
                return smooth_polyline(pts, window=2, passes=1) if len(pts) >= 3 else pts
            px, py, heading = nx, ny, nh
            pts.append((px, py))
            grown += step
            if goal_x is not None and px >= goal_x:    # two-column: reached right edge
                return smooth_polyline(pts, window=2, passes=1) if len(pts) >= 3 else pts
            if goal_x is None and grown >= target_len:
                break
    return smooth_polyline(pts, window=2, passes=1) if len(pts) >= 3 else pts


def _catmull_dense(ctrl: list[Point], tension: float = 0.5, step_px: float = 6.0) -> list[Point]:
    """Smooth C1 Catmull-Rom curve through `ctrl` (passes through every control
    point, incl. ends), sampled ~every step_px."""
    n = len(ctrl)
    if n < 2:
        return list(ctrl)

    def at(i):
        return ctrl[max(0, min(n - 1, i))]

    dense = [ctrl[0]]
    for j in range(n - 1):
        p0, p1, p2, p3 = at(j - 1), at(j), at(j + 1), at(j + 2)
        b1 = (p1[0] + (p2[0] - p0[0]) / 6 * tension, p1[1] + (p2[1] - p0[1]) / 6 * tension)
        b2 = (p2[0] - (p3[0] - p1[0]) / 6 * tension, p2[1] - (p3[1] - p1[1]) / 6 * tension)
        steps = max(2, int(distance(p1, p2) / step_px))
        for s in range(1, steps + 1):
            t = s / steps
            mt = 1 - t
            x = mt**3 * p1[0] + 3 * mt**2 * t * b1[0] + 3 * mt * t**2 * b2[0] + t**3 * p2[0]
            y = mt**3 * p1[1] + 3 * mt**2 * t * b1[1] + 3 * mt * t**2 * b2[1] + t**3 * p2[1]
            dense.append((x, y))
    return dense


def bezier_line(
    start: Point, heading0: float, w: int, h: int, rng: random.Random, target_len: float,
    r_lo: float, r_hi: float, turn_lo: float, turn_hi: float, obstacles: list[Point],
    self_margin: float = 19.0, obstacle_margin: float = 30.0, step: float = 6.0,
) -> list[Point]:
    """An ELEGANT smooth curve (mixed in with snakes for variety on easy/medium):
    a coarse bounded-turn, collision-aware walk places control points spaced ~one
    radius apart with GENTLE turns, then a Catmull-Rom spline smooths through them.
    The result is a flowing bezier-like sweep with graceful inflections, rather than
    the more organic arc-by-arc snake. Same self/obstacle avoidance, so endpoints and
    markers stay clean."""
    lp = layout_params(w, h)
    margin = lp["knot_margin"]
    cxc, cyc = w / 2.0, h / 2.0
    sm2 = self_margin * self_margin
    om2 = obstacle_margin * obstacle_margin
    px, py = start
    heading = heading0
    ctrl: list[Point] = [(px, py)]
    direction = rng.choice([-1, 1])
    run_turn = 0.0
    grown = 0.0
    while grown < target_len:
        spacing = rng.uniform(r_lo, r_hi)
        # Broad C/S lobes need small heading changes and long tangent handles.
        # Large alternating turns make a Catmull-Rom technically smooth but
        # visually polygonal. Keep each change gentle and let its direction
        # persist for a few controls before reversing.
        turn = math.radians(rng.uniform(turn_lo, turn_hi)) * 0.43
        if rng.random() < 0.42 or run_turn + turn > math.radians(110):
            direction = -direction
            run_turn = 0.0
        run_turn += turn
        heading += direction * turn
        near = min(px - margin, (w - margin) - px, py - margin, (h - margin) - py)
        if near < margin:
            heading += max(-0.6, min(0.6, _wrap(math.atan2(cyc - py, cxc - px) - heading))) * (1 - near / margin)
        nx = px + spacing * math.cos(heading)
        ny = py + spacing * math.sin(heading)
        nx, ny = clamp_point((nx, ny), w, h, margin // 2)
        # coarse collision: new control point clear of obstacles and earlier controls
        blocked = any((nx - ox) ** 2 + (ny - oy) ** 2 < om2 for ox, oy in obstacles)
        if not blocked and len(ctrl) > 2:
            blocked = any((nx - q[0]) ** 2 + (ny - q[1]) ** 2 < sm2 for q in ctrl[:-2])
        if blocked:
            break
        ctrl.append((nx, ny))
        px, py = nx, ny
        grown += spacing
    return _catmull_dense(ctrl, tension=1.0, step_px=step) if len(ctrl) >= 2 else list(ctrl)


def trace_points(dense: list[Point], step: int = 12) -> list[Point]:
    out = [dense[0]]
    for i in range(1, len(dense) - 1):
        if i % step == 0:
            out.append(dense[i])
    out.append(dense[-1])
    ded = [out[0]]
    for p in out[1:]:
        if distance(p, ded[-1]) >= 3:
            ded.append(p)
    return ded


def endpoint_rope(
    start: Point,
    end: Point,
    w: int,
    h: int,
    rng: random.Random,
    arm_length_scale: tuple[float, float],
    radius: tuple[float, float],
    min_radius: float,
    step: float = 5.0,
) -> list[Point]:
    """Join two broad bounded-curvature arms with a tangent-aligned cubic.

    This construction cannot make tiny 180/270-degree hooks: every local turn is
    capped by ``min_radius``.
    """
    margin = layout_params(w, h)["knot_margin"] * 0.55

    def inward_heading(p: Point) -> float:
        edge = min(
            (p[0], 0.0),
            (w - p[0], math.pi),
            (p[1], math.pi / 2),
            (h - p[1], -math.pi / 2),
            key=lambda item: item[0],
        )
        if edge[0] <= 0.22 * min(w, h):
            return edge[1]
        return math.atan2(h / 2 - p[1], w / 2 - p[0])

    def grow_arm(origin: Point, heading: float, target_length: float) -> list[Point] | None:
        pts = [origin]
        px, py = origin
        direction = rng.choice((-1, 1))
        curvature = 0.0
        arc_left = 0.0
        grown = 0.0
        while grown < target_length:
            if arc_left <= 0.0:
                if rng.random() < 0.62:
                    direction *= -1
                r = rng.uniform(max(min_radius, radius[0]), max(min_radius + 1, radius[1]))
                curvature = direction / r
                arc_left = r * math.radians(rng.uniform(38.0, 118.0))

            # A bounded wall-steer bends the arm back into the canvas without
            # clamping points against an edge.
            wall = min(px - margin, w - margin - px, py - margin, h - margin - py)
            steer = 0.0
            if wall < 55.0:
                desired = math.atan2(h / 2 - py, w / 2 - px)
                steer = max(-1 / min_radius, min(1 / min_radius, _wrap(desired - heading) / 65.0))
            kappa = max(-1 / min_radius, min(1 / min_radius, curvature + steer))
            heading += kappa * step
            nx = px + step * math.cos(heading)
            ny = py + step * math.sin(heading)
            if not (margin <= nx <= w - margin and margin <= ny <= h - margin):
                return None
            pts.append((nx, ny))
            px, py = nx, ny
            grown += step
            arc_left -= step
        return pts

    arm_length = rng.uniform(*arm_length_scale) * min(w, h)
    arm_a = grow_arm(start, inward_heading(start) + rng.uniform(-0.16, 0.16), arm_length)
    arm_b = grow_arm(end, inward_heading(end) + rng.uniform(-0.16, 0.16), arm_length)
    if arm_a is None or arm_b is None or len(arm_a) < 3 or len(arm_b) < 3:
        return []

    a = arm_a[-1]
    b = arm_b[-1]
    ha = math.atan2(arm_a[-1][1] - arm_a[-2][1], arm_a[-1][0] - arm_a[-2][0])
    hb_in = math.atan2(arm_b[-1][1] - arm_b[-2][1], arm_b[-1][0] - arm_b[-2][0])
    join_dist = distance(a, b)
    if join_dist < 45.0:
        return []
    handle = min(0.42 * join_dist, 0.30 * min(w, h))
    c1 = (a[0] + handle * math.cos(ha), a[1] + handle * math.sin(ha))
    # The second arm is traversed backwards, so the connector arrives opposite
    # its growth direction.
    arrive = hb_in + math.pi
    c2 = (b[0] - handle * math.cos(arrive), b[1] - handle * math.sin(arrive))
    connector: list[Point] = []
    connector_steps = max(16, int(join_dist / step))
    for i in range(1, connector_steps):
        t = i / connector_steps
        mt = 1.0 - t
        connector.append((
            mt**3 * a[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * b[0],
            mt**3 * a[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * b[1],
        ))
    if any(not (margin <= x <= w - margin and margin <= y <= h - margin)
           for x, y in connector):
        return []
    return arm_a + connector + list(reversed(arm_b))


def fixed_bezier_curve(
    start: Point,
    end: Point,
    w: int,
    h: int,
    rng: random.Random,
    length_scale: float,
    max_knots: int,
    bend_waves: tuple[int, int],
    sway_scale: tuple[float, float],
    wave_decay: float,
    phase_jitter: float,
    endpoint_lead_px: float = 0.0,
    route_inset_px: float = 0.0,
) -> tuple[list[Point], list[Point]]:
    """Broad fixed-endpoint spline using engine.py's cubic Bezier utilities.

    The waypoints follow several broad chord-relative lobes, so increasing
    ``max_knots`` and ``bend_waves`` adds real direction changes rather than
    merely resampling the same C/S curve.
    Global offset scaling keeps those added bends inside the canvas without
    clamping individual controls into sharp hooks.
    """
    def inward_point(point: Point, amount: float) -> Point:
        x, y = point
        edge = min(
            ((x, (1.0, 0.0)), (w - x, (-1.0, 0.0)),
             (y, (0.0, 1.0)), (h - y, (0.0, -1.0))),
            key=lambda item: item[0],
        )
        return (x + amount * edge[1][0], y + amount * edge[1][1])

    route_start = inward_point(start, route_inset_px)
    route_end = inward_point(end, route_inset_px)
    route_chord = max(1.0, distance(route_start, route_end))
    ux = (route_end[0] - route_start[0]) / route_chord
    uy = (route_end[1] - route_start[1]) / route_chord
    vx, vy = -uy, ux
    n_knots = rng.randint(max(3, max_knots - 3), max_knots)
    wave_count = rng.randint(*bend_waves)
    # Higher-frequency lobes need proportionally less lateral displacement to
    # remain broad and smooth. Without this normalization, four bends become
    # four tight hairpins rather than a longer follow-the-line challenge.
    amplitude = (
        route_chord
        * rng.uniform(*sway_scale)
        * math.sqrt(length_scale)
        / (wave_count ** wave_decay)
    )
    side_bias = rng.choice((-1.0, 1.0)) * rng.uniform(0.18, 0.42)
    phase = rng.uniform(-phase_jitter, phase_jitter)
    secondary_phase = rng.uniform(-1.5 * phase_jitter, 1.5 * phase_jitter)
    spacing_phase = rng.uniform(-math.pi, math.pi)
    amplitude_phase = rng.uniform(-math.pi, math.pi)
    wave_warp = rng.uniform(0.08, 0.22)
    secondary_weight = rng.uniform(0.08, 0.28)
    secondary_frequency = wave_count + rng.choice((-1.0, 0.5, 1.0, 1.5))
    along_amp = (
        route_chord * rng.uniform(0.025, 0.055)
        * math.sqrt(length_scale / wave_count)
    )
    margin = layout_params(w, h)["knot_margin"] * 0.55

    # Uneven gaps keep different curves from placing every bend at the same
    # chord fractions. The bounded weights preserve ordering and endpoint room.
    gap_weights = [
        rng.uniform(0.65, 1.35)
        * (1.0 + 0.12 * math.sin(spacing_phase + 1.7 * i))
        for i in range(n_knots + 1)
    ]
    gap_total = sum(gap_weights)
    knot_positions: list[float] = []
    cumulative_gap = 0.0
    for gap in gap_weights[:-1]:
        cumulative_gap += gap
        knot_positions.append(cumulative_gap / gap_total)

    base_points: list[Point] = []
    offsets: list[Point] = []
    for t in knot_positions:
        base_x = route_start[0] + t * (route_end[0] - route_start[0])
        base_y = route_start[1] + t * (route_end[1] - route_start[1])
        envelope = math.sin(math.pi * t) ** 0.82
        local_amplitude = 1.0 + rng.uniform(0.16, 0.34) * math.sin(
            2.0 * math.pi * t + amplitude_phase
        )
        warped_t = t + (
            wave_warp
            * math.sin(math.pi * t)
            * math.sin(2.0 * math.pi * t + spacing_phase)
            / max(1.0, wave_count)
        )
        broad_lobes = (
            math.sin(wave_count * math.pi * warped_t + phase)
            + secondary_weight
            * math.sin(secondary_frequency * math.pi * t + secondary_phase)
        )
        across = amplitude * envelope * (side_bias + local_amplitude * broad_lobes)
        along = along_amp * (
            math.sin((wave_count + 0.5) * math.pi * warped_t + secondary_phase)
            + rng.uniform(0.15, 0.32)
            * math.sin((wave_count + 2.0) * math.pi * t + spacing_phase)
        )
        base_points.append((base_x, base_y))
        offsets.append((ux * along + vx * across, uy * along + vy * across))

    scale = 1.0
    for (bx, by), (ox, oy) in zip(base_points, offsets):
        if ox > 0:
            scale = min(scale, (w - margin - bx) / ox)
        elif ox < 0:
            scale = min(scale, (margin - bx) / ox)
        if oy > 0:
            scale = min(scale, (h - margin - by) / oy)
        elif oy < 0:
            scale = min(scale, (margin - by) / oy)
    scale = max(0.0, min(1.0, scale * 0.94))
    anchors = [start]
    anchors.extend(
        (bx + scale * ox, by + scale * oy)
        for (bx, by), (ox, oy) in zip(base_points, offsets)
    )
    anchors.append(end)

    controls: list[Point] = []
    dense: list[Point] = []
    tension = rng.uniform(0.76, 0.96)
    segments = catmull_rom_controls(anchors, tension)
    for segment_index, segment in enumerate(segments):
        if endpoint_lead_px > 0 and segment_index == 0:
            segment = [
                segment[0], inward_point(start, endpoint_lead_px),
                segment[2], segment[3],
            ]
        if endpoint_lead_px > 0 and segment_index == len(segments) - 1:
            segment = [
                segment[0], segment[1],
                inward_point(end, endpoint_lead_px), segment[3],
            ]
        controls.extend(segment if segment_index == 0 else segment[1:])
        sampled = sample_bezier(segment, 34)
        if any(not (margin * 0.5 <= p[0] <= w - margin * 0.5
                    and margin * 0.5 <= p[1] <= h - margin * 0.5)
               for p in sampled):
            return [], []
        dense.extend(sampled if segment_index == 0 else sampled[1:])
    return controls, dense


def _dedupe_crossing_events(events: list[dict], radius: float = 8.0) -> list[dict]:
    out: list[dict] = []
    r2 = radius * radius
    for event in sorted(events, key=lambda e: (e["pos_a"], e["pos_b"])):
        x, y = event["point"]
        if any((x - old["point"][0]) ** 2 + (y - old["point"][1]) ** 2 < r2
               for old in out):
            continue
        out.append(event)
    return out


def polyline_crossing_events(
    a: list[Point],
    b: list[Point],
    self_cross: bool = False,
) -> list[dict]:
    """Return de-duplicated crossings with arc positions on both polylines."""
    ca, cb = _coarse(a, 3), _coarse(b, 3)
    cuma = [0.0]
    for p, q in zip(ca, ca[1:]):
        cuma.append(cuma[-1] + distance(p, q))
    cumb = cuma if self_cross else [0.0]
    if not self_cross:
        for p, q in zip(cb, cb[1:]):
            cumb.append(cumb[-1] + distance(p, q))

    events: list[dict] = []
    for i, (p1, p2) in enumerate(zip(ca, ca[1:])):
        j0 = i + 3 if self_cross else 0
        for j in range(j0, len(cb) - 1):
            q1, q2 = cb[j], cb[j + 1]
            hit = _seg_intersection(p1, p2, q1, q2)
            if hit is None:
                continue
            d1 = (p2[0] - p1[0], p2[1] - p1[1])
            d2 = (q2[0] - q1[0], q2[1] - q1[1])
            n1, n2 = math.hypot(*d1), math.hypot(*d2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosv = abs(d1[0] * d2[0] + d1[1] * d2[1]) / (n1 * n2)
            angle = math.degrees(math.acos(max(0.0, min(1.0, cosv))))
            fa = distance(p1, hit) / max(1e-9, distance(p1, p2))
            fb = distance(q1, hit) / max(1e-9, distance(q1, q2))
            events.append({
                "point": hit,
                "angle": angle,
                "pos_a": cuma[i] + fa * distance(p1, p2),
                "pos_b": cumb[j] + fb * distance(q1, q2),
            })
    return _dedupe_crossing_events(events)


def _direction_reversals(pts: list[Point]) -> int:
    """Count substantial left/right direction reversals, ignoring tiny jitter."""
    signs: list[int] = []
    stride = max(3, len(pts) // 70)
    sampled = pts[::stride]
    if sampled[-1] != pts[-1]:
        sampled.append(pts[-1])
    for a, b in zip(sampled, sampled[1:]):
        dx = b[0] - a[0]
        if abs(dx) >= 5.0:
            signs.append(1 if dx > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _lateral_reversals(pts: list[Point]) -> int:
    """Count broad side-to-side bends relative to the endpoint chord."""
    if len(pts) < 3:
        return 0
    start, end = pts[0], pts[-1]
    chord = max(1.0, distance(start, end))
    vx = -(end[1] - start[1]) / chord
    vy = (end[0] - start[0]) / chord
    stride = max(3, len(pts) // 55)
    sampled = pts[::stride]
    if sampled[-1] != end:
        sampled.append(end)
    lateral = [(p[0] - start[0]) * vx + (p[1] - start[1]) * vy for p in sampled]
    signs: list[int] = []
    for a, b in zip(lateral, lateral[1:]):
        delta = b - a
        if abs(delta) >= 4.0:
            signs.append(1 if delta > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _central_reentries(pts: list[Point], w: int, h: int) -> int:
    """Count returns to the central ellipse after the rope has left it."""
    def inside(p: Point) -> bool:
        return ((p[0] - w / 2) / (0.32 * w)) ** 2 + ((p[1] - h / 2) / (0.34 * h)) ** 2 <= 1.0

    states = [inside(p) for p in pts[::3]]
    entries = sum((not a) and b for a, b in zip(states, states[1:]))
    return max(0, entries - (1 if states and not states[0] else 0))


def _near_parallel_length(curve: list[Point], others: list[list[Point]]) -> float:
    """Approximate length running close and nearly parallel to another rope."""
    c = _coarse(curve, 4)
    foreign = [_coarse(o, 4) for o in others]
    total = 0.0
    for a, b in zip(c, c[1:]):
        seg_len = distance(a, b)
        if seg_len < 1e-6:
            continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ux, uy = (b[0] - a[0]) / seg_len, (b[1] - a[1]) / seg_len
        matched = False
        for other in foreign:
            for p, q in zip(other, other[1:]):
                olen = distance(p, q)
                if olen < 1e-6:
                    continue
                if point_segment_distance((mx, my), p, q) > 15.0:
                    continue
                vx, vy = (q[0] - p[0]) / olen, (q[1] - p[1]) / olen
                angle = math.degrees(math.acos(max(0.0, min(1.0, abs(ux * vx + uy * vy)))))
                if angle <= 18.0:
                    matched = True
                    break
            if matched:
                break
        if matched:
            total += seg_len
    return total


# -----------------------------------------------------------------------------
# colors
# -----------------------------------------------------------------------------
PALETTE = [
    (215, 65, 75), (54, 135, 220), (51, 156, 92), (225, 151, 40), (142, 95, 204),
    (36, 170, 170), (198, 84, 160), (110, 125, 55), (90, 90, 90), (130, 86, 55),
    (60, 110, 200), (200, 110, 40),
]


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def curve_colors(n: int, mode: str, rng: random.Random) -> list[tuple[int, int, int]]:
    if mode == "distinct":
        off = rng.randrange(len(PALETTE))
        return [PALETTE[(i + off) % len(PALETTE)] for i in range(n)]
    if mode == "family":
        base = rng.random()
        return [_hsv(base + rng.uniform(-0.05, 0.05), rng.uniform(0.45, 0.72), rng.uniform(0.42, 0.74)) for _ in range(n)]
    if mode == "uniform":
        if rng.random() < 0.3:
            v = rng.randint(40, 70)
            shared = (v, v, v)
        else:
            shared = _hsv(rng.random(), rng.uniform(0.45, 0.8), rng.uniform(0.32, 0.5))
        return [shared for _ in range(n)]
    raise ValueError(mode)


def contrasting_outline(color: tuple[int, int, int], rng: random.Random) -> tuple[int, int, int]:
    """Choose a clearly separated but non-fixed rope exterior color."""
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    use_light = luminance < 105 or (luminance < 165 and rng.random() < 0.45)
    if use_light:
        mix = rng.uniform(0.58, 0.76)
        return tuple(round(c + (245 - c) * mix) for c in color)
    scale = rng.uniform(0.18, 0.38)
    return tuple(round(c * scale) for c in color)


def scheduled_appearance(seed: int) -> tuple[str, str, bool]:
    """Exact appearance ratios over every 50 consecutive seeds.

    Counts per block:
      color: 5 distinct, 10 family, 35 uniform
      stroke: 25 solid, 25 dashed
      outline: 10 yes, distributed 1/2/7 over the color modes
    """
    within_ten = seed % 10
    block = (seed // 10) % 5
    if within_ten == 0:
        color_mode = "distinct"
    elif within_ten in {2, 7}:
        color_mode = "family"
    else:
        color_mode = "uniform"
    line_style = "dashed" if seed % 2 else "solid"
    outline_positions = (
        {0, 5},  # one distinct, one uniform
        {2, 5},  # one family, one uniform
        {7, 5},  # one family, one uniform
        {3, 5},  # two uniform
        {4, 5},  # two uniform
    )
    outlined = within_ten in outline_positions[block]
    return color_mode, line_style, outlined


def two_dot_colors(rng: random.Random) -> list[tuple[int, int, int]]:
    h0 = rng.random()
    h1 = (h0 + rng.uniform(0.3, 0.7)) % 1.0
    return [_hsv(h0, rng.uniform(0.7, 0.98), rng.uniform(0.55, 0.88)),
            _hsv(h1, rng.uniform(0.7, 0.98), rng.uniform(0.55, 0.88))]


# -----------------------------------------------------------------------------
# difficulty ladder
# -----------------------------------------------------------------------------
# leg_angle is the tortuosity dial: arc/chord ~ 1/cos(leg_angle).
#   35deg~1.22  50deg~1.56  60deg~2.0  66deg~2.46  71deg~3.07  74deg~3.63
# span: how far apart paired endpoints are (lower => more canvas room for big
#       perpendicular swings, so high-tortuosity levels use moderate span).
# color: distinct (easy, color is a cue) -> uniform (hard, must trace continuity).
@dataclass
class Level:
    name: str
    num_curves: tuple[int, int]
    length: tuple[float, float]       # target snake arc length (px) -- the length dial
    radius: tuple[float, float]       # arc radius range (px); varies per arc => heterogeneous curvature
    turn: tuple[float, float]         # per-arc turn range (deg); bends stay broad sweeps
    color_weights: dict               # over (distinct, family, uniform)
    line_widths: tuple[int, ...]
    dashed_frac: float
    max_cross: int                    # soft cap on total scene crossings (scribble guard)
    bezier_frac: float = 0.0          # fraction of lines drawn as bezier sweeps vs snakes
    flip_prob: float = 0.85           # arc direction-flip prob (high=tight wiggle, low=big lobes)
    coil_drift: float = 0.0           # anti-coil drift strength (only needed for very long lines)
    repel: float = 0.0                # self/obstacle repulsion (straightens; only xhard needs it)
    radius_floor: float = 52.0        # min bend radius (lower => tighter winding => denser tangle)
    layout: str = "perimeter"         # "perimeter" (markers around edge) | "columns" (L/R, tangled)


@dataclass(frozen=True)
class PathfinderLevel:
    """Realized-geometry targets for the fixed-endpoint rope generator."""
    num_curves: tuple[int, int]
    arm_length_scale: tuple[float, float]
    radius: tuple[float, float]
    arc_ratio: tuple[float, float]
    target_crossings: tuple[int, int]
    target_distinct: tuple[int, int]
    target_loops: tuple[int, int]
    target_reentries: tuple[int, int]
    target_reversals: tuple[int, int]
    color_weights: dict
    line_widths: tuple[int, ...]
    outline_frac: float
    scribble_frac: float = 0.0
    scribble_curves: tuple[int, int] = (4, 5)
    min_turn_radius_px: float = 24.0
    min_self_loop_span_px: float = 150.0
    max_self_crossings: int = 5


@dataclass(frozen=True)
class BezierLevel:
    """Difficulty controls for the reused engine.py Bezier construction."""
    num_curves: tuple[int, int]
    complexity: str
    length_scale: float
    max_knots: int
    bend_waves: tuple[int, int]
    sway_scale: tuple[float, float]
    wave_decay: float
    phase_jitter: float
    min_arc_ratio: float
    min_lateral_reversals: int
    target_crossings: tuple[float, float]
    target_distinct: tuple[int, int]
    line_widths: tuple[int, ...]
    min_turn_radius_px: float
    max_local_turn_deg: float
    max_self_crossings: int


# This ladder is intentionally separate from LEVELS. The Bezier branch keeps its
# own construction and tuning; only curve_style="pathfinder" uses these
# fixed-endpoint rope targets.
PATHFINDER_LEVELS: dict[str, PathfinderLevel] = {
    "easy": PathfinderLevel(
        (2, 4), (0.22, 0.38), (80.0, 150.0),
        (1.20, 1.85), (1, 5), (1, 3), (0, 1), (0, 2), (0, 3),
        {"distinct": 0.10, "family": 0.20, "uniform": 0.70}, (3, 4), 0.20,
        min_turn_radius_px=34.0, min_self_loop_span_px=190.0, max_self_crossings=1,
    ),
    "medium": PathfinderLevel(
        (4, 6), (0.36, 0.58), (65.0, 135.0),
        (1.55, 2.45), (4, 10), (2, 4), (0, 2), (1, 3), (1, 5),
        {"distinct": 0.10, "family": 0.20, "uniform": 0.70}, (3, 4), 0.20,
        scribble_frac=0.10, min_turn_radius_px=30.0,
        min_self_loop_span_px=175.0, max_self_crossings=2,
    ),
    "hard": PathfinderLevel(
        (6, 8), (0.57, 0.88), (49.0, 114.0),
        (2.25, 3.75), (12, 25), (4, 7), (1, 4), (1, 5), (3, 8),
        {"distinct": 0.10, "family": 0.20, "uniform": 0.70}, (3, 4), 0.20,
        scribble_frac=0.25, scribble_curves=(5, 6), min_turn_radius_px=27.0,
        min_self_loop_span_px=160.0, max_self_crossings=4,
    ),
    "xhard": PathfinderLevel(
        (7, 10), (0.66, 1.02), (46.0, 108.0),
        (2.35, 4.10), (14, 28), (5, 7), (2, 5), (2, 5), (3, 9),
        {"distinct": 0.10, "family": 0.20, "uniform": 0.70}, (3, 4), 0.20,
        scribble_frac=0.22, scribble_curves=(7, 10),
        min_turn_radius_px=24.0, min_self_loop_span_px=150.0, max_self_crossings=5,
    ),
}


# Equivalent ladder for the fixed-endpoint Catmull-Rom/Bezier generator reused
# from engine.py. Bezier obtains difficulty from longer broad sweeps, more knots,
# more pairs, and crossing selection rather than Pathfinder's arc-chain length.
BEZIER_LEVELS: dict[str, BezierLevel] = {
    "easy": BezierLevel(
        (2, 4), "moderate", 1.10, 4, (1, 2), (0.20, 0.29), 0.85, 0.16, 1.08, 0,
        (0.5, 2.0), (1, 3), (3, 4), 14.0, 42.0, 0,
    ),
    "medium": BezierLevel(
        (4, 6), "moderate", 1.35, 6, (2, 3), (0.24, 0.36), 0.82, 0.24, 1.14, 1,
        (1.5, 3.5), (2, 5), (3, 4), 12.0, 42.0, 1,
    ),
    "hard": BezierLevel(
        (6, 8), "complex", 1.60, 8, (3, 4), (0.27, 0.40), 0.76, 0.34, 1.20, 2,
        (2.5, 5.0), (4, 7), (3, 4), 10.0, 42.0, 2,
    ),
    "xhard": BezierLevel(
        (7, 10), "complex", 2.15, 13, (4, 5), (0.25, 0.36), 0.68, 0.42, 1.38, 3,
        (4.5, 8.0), (5, 9), (3, 4), 7.0, 55.0, 3,
    ),
}


# The ladder:
#   easy   = short curvy lines, a few pairs.
#   medium = same flowing style but MUCH longer lines (length is the difficulty).
#   hard   = the dense perimeter tangle: many crossing lines.
#   xhard  = a two-COLUMN "follow the tangled line" puzzle:
#            8-12 pairs scrambled left->right, tight winding (low radius_floor) so the
#            lines knot densely in the middle with very many crossings.
# CURVINESS is set by flip_prob (moderate ~0.5 -> graceful C/S lobes). repel + coil_drift
# let long lines grow without coiling. radius_floor controls winding tightness.
LEVELS: dict[str, Level] = {
    "easy":   Level("easy",   (3, 6), (240, 440), (78, 135), (60, 120),
                    {"distinct": 0.55, "family": 0.3, "uniform": 0.15}, (3, 4, 5), 0.1, max_cross=18, flip_prob=0.5),
    "medium": Level("medium", (4, 7), (820, 1250), (64, 120), (62, 124),
                    {"distinct": 0.12, "family": 0.3, "uniform": 0.58}, (2, 3), 0.35, max_cross=44,
                    flip_prob=0.55, coil_drift=0.2, repel=1.0),
    "hard":   Level("hard",   (7, 9), (1000, 1450), (58, 120), (62, 124),
                    {"distinct": 0.05, "family": 0.12, "uniform": 0.83}, (2, 2), 0.5, max_cross=90,
                    flip_prob=0.55, coil_drift=0.2, repel=1.0),
    "xhard":  Level("xhard",  (8, 12), (1100, 1800), (34, 82), (70, 135),
                    {"distinct": 0.04, "family": 0.1, "uniform": 0.86}, (2, 2), 0.45, max_cross=240,
                    flip_prob=0.6, coil_drift=0.0, repel=1.0, radius_floor=30.0, layout="columns"),
}
LEVEL_ORDER = ["easy", "medium", "hard", "xhard"]
CURVE_STYLES = ("bezier", "pathfinder", "mixed")


# -----------------------------------------------------------------------------
# scene + curves
# -----------------------------------------------------------------------------
START_LABELS = list("ABCDEFGHIJKLMN")
END_LABELS = [str(i) for i in range(1, 15)]


@dataclass
class Curve:
    si: str            # start label
    ei: str            # end label
    start: Point
    end: Point
    controls: list[Point]
    dense: list[Point]
    trace: list[Point]
    color: tuple[int, int, int]
    outline_color: tuple[int, int, int] | None
    arc: float
    chord: float


@dataclass
class Scene:
    curves: list[Curve]
    crosses: list[set]            # which curves each curve crosses
    crossing_counts: list[int]
    min_angle: float              # global min acute crossing angle (any pair)
    total_crossings: int
    endpoint_clean: bool
    color_mode: str
    line_width: int
    line_style: str
    rope_outline: bool = False
    curve_min_angle: list[float] = field(default_factory=list)   # per-curve min crossing angle
    pair_angle: dict = field(default_factory=dict)               # (i,j)->min crossing angle
    angle_floor: float = 25.0
    curve_style: str = "bezier"
    pathfinder_family: str | None = None
    marker_layout: str | None = None
    curve_metrics: list[dict] = field(default_factory=list)
    crossing_grid_cells: int = 0
    central_crossings: int = 0
    crossing_bbox_area_frac: float = 0.0


def _grow_one(lvl, start, heading0, obstacles, w, h, rng, self_margin, obstacle_margin,
              foreign_dense, clearance, min_len, min_gap, curve_style="pathfinder",
              goal_x=None, drift_x=0.0, tries=10):
    """Grow one snake whose END marker is unambiguous: it (a) reaches at least
    `min_len`, (b) stays `clearance` from every foreign LINE body, and (c) lands
    `min_gap` from every existing MARKER (so two endpoints never crowd, which would
    put a line near a foreign marker). Returns the best dense polyline | None."""
    best = None
    for _ in range(tries):
        tgt = rng.uniform(*lvl.length)
        if curve_style == "bezier":
            dense = bezier_line(start, heading0 + rng.uniform(-0.5, 0.5), w, h, rng, tgt,
                                lvl.radius[0], lvl.radius[1], lvl.turn[0], lvl.turn[1],
                                obstacles, self_margin=self_margin, obstacle_margin=obstacle_margin)
        elif curve_style == "pathfinder":
            dense = grow_snake(start, heading0 + rng.uniform(-0.4, 0.4), w, h, rng, tgt,
                               lvl.radius[0], lvl.radius[1], lvl.turn[0], lvl.turn[1],
                               obstacles, self_margin=self_margin, obstacle_margin=obstacle_margin,
                               flip_prob=lvl.flip_prob, coil_drift=lvl.coil_drift, repel=lvl.repel,
                               radius_floor=lvl.radius_floor, goal_x=goal_x, drift_x=drift_x)
        else:
            raise ValueError(f"unknown curve_style: {curve_style}")
        if len(dense) < 3:
            continue
        end = dense[-1]
        end_clear = all(endpoint_safe(fd, [end], set(), clearance) for fd in foreign_dense)
        gap_ok = all(distance(end, m) >= min_gap for m in obstacles)
        long_enough = polyline_length(dense) >= min_len
        reached = goal_x is None or end[0] >= goal_x - 25   # columns: must reach right edge
        key = (end_clear and gap_ok and reached, long_enough)
        if best is None or key > best[0]:
            best = (key, dense)
        if end_clear and gap_ok and long_enough and reached:
            return dense
    return best[1] if best else None


def column_markers(n: int, w: int, h: int, rng: random.Random, side: str = "left") -> list[Point]:
    """n markers stacked in a vertical column on the left or right edge, evenly
    spaced with jitter and a Euclidean min-gap (for the two-column tangle layout)."""
    lp = layout_params(w, h)
    m, r = lp["margin"], lp["marker_r"]
    x = float(m if side == "left" else (w - m))
    y0, y1 = float(m), float(h - m)
    min_gap = 2 * r + 8
    base = (y1 - y0) / n
    ys = sorted(y0 + base * (i + 0.5) + rng.uniform(-base * 0.22, base * 0.22) for i in range(n))
    for _ in range(80):
        moved = False
        for i in range(n - 1):
            if ys[i + 1] - ys[i] < min_gap:
                d = (min_gap - (ys[i + 1] - ys[i])) / 2
                ys[i] -= d; ys[i + 1] += d
                moved = True
        ys = [max(y0, min(y1, y)) for y in ys]
        if not moved:
            break
    return [(x, y) for y in ys]


def side_markers(n: int, w: int, h: int, rng: random.Random, side: str) -> list[Point]:
    """Evenly spaced markers on any canvas side, with modest jitter."""
    if side in {"left", "right"}:
        return column_markers(n, w, h, rng, side)
    lp = layout_params(w, h)
    m, r = lp["margin"], lp["marker_r"]
    y = float(m if side == "top" else h - m)
    x0, x1 = float(m), float(w - m)
    min_gap = 2 * r + 8
    base = (x1 - x0) / n
    xs = sorted(x0 + base * (i + 0.5) + rng.uniform(-base * 0.22, base * 0.22)
                for i in range(n))
    for _ in range(80):
        moved = False
        for i in range(n - 1):
            if xs[i + 1] - xs[i] < min_gap:
                d = (min_gap - (xs[i + 1] - xs[i])) / 2
                xs[i] -= d
                xs[i + 1] += d
                moved = True
        xs = [max(x0, min(x1, x)) for x in xs]
        if not moved:
            break
    return [(x, y) for x in xs]


def _sample_spread_points(
    count: int,
    w: int,
    h: int,
    rng: random.Random,
    existing: list[Point] | None = None,
    peripheral: bool = False,
) -> list[Point]:
    lp = layout_params(w, h)
    margin = lp["margin"]
    gap = 2 * lp["marker_r"] + 12
    points = list(existing or [])
    made: list[Point] = []
    for _ in range(count):
        best = None
        best_clear = -1.0
        for _try in range(240):
            p = (
                rng.uniform(margin, w - margin),
                rng.uniform(margin, h - margin),
            )
            if peripheral:
                edge_distance = min(p[0], w - p[0], p[1], h - p[1])
                if edge_distance > 0.24 * min(w, h):
                    continue
            clear = min((distance(p, q) for q in points), default=1e9)
            if clear > best_clear:
                best, best_clear = p, clear
            if clear >= gap:
                break
        if best is None:
            raise RuntimeError("could not place endpoint marker")
        points.append(best)
        made.append(best)
    return made


def free_bezier_markers(
    n: int,
    w: int,
    h: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point], str]:
    """Place Bezier endpoints freely in the outer band, without side alignment."""
    points = _sample_spread_points(2 * n, w, h, rng, peripheral=True)
    rng.shuffle(points)
    return points[:n], points[n:], "free_spread"


def varied_marker_layout(
    n: int,
    w: int,
    h: int,
    rng: random.Random,
    curve_style: str,
    layout_override: str | None = None,
) -> tuple[list[Point], list[Point], str]:
    """Choose a style-specific endpoint layout."""
    layouts = ["left_right", "top_bottom", "adjacent_sides", "perimeter_random", "scattered"]
    if layout_override is not None and layout_override not in layouts:
        raise ValueError(f"unknown marker layout: {layout_override}")
    layout_weights = {
        # 15% aligned horizontally, 15% vertically, 15% adjacent; the
        # remaining 55% is split evenly between the two randomized layouts.
        "bezier": [0.15, 0.15, 0.15, 0.275, 0.275],
        # Pathfinder retains slightly more opposing-side structure.
        "pathfinder": [0.20, 0.20, 0.10, 0.25, 0.25],
    }
    if curve_style not in layout_weights:
        raise ValueError(f"unknown curve style for marker layout: {curve_style}")
    layout = (
        layout_override
        if layout_override is not None
        else rng.choices(layouts, weights=layout_weights[curve_style], k=1)[0]
    )
    if layout == "left_right":
        return side_markers(n, w, h, rng, "left"), side_markers(n, w, h, rng, "right"), layout
    if layout == "top_bottom":
        return side_markers(n, w, h, rng, "top"), side_markers(n, w, h, rng, "bottom"), layout
    if layout == "adjacent_sides":
        side_a, side_b = rng.choice([
            ("left", "top"), ("top", "right"), ("right", "bottom"), ("bottom", "left"),
        ])
        return side_markers(n, w, h, rng, side_a), side_markers(n, w, h, rng, side_b), layout
    if layout == "perimeter_random":
        points = perimeter_markers(2 * n, w, h, rng)
        starts, ends = points[:n], points[n:]
        return starts, ends, layout
    # Free, non-aligned positions in the outer band. Keeping badges out of the
    # central knot preserves endpoint readability while avoiding rows/columns.
    starts = _sample_spread_points(n, w, h, rng, peripheral=True)
    ends = _sample_spread_points(n, w, h, rng, existing=starts, peripheral=True)
    return starts, ends, layout


def resample_polyline(pts: list[Point], step: float = 8.0) -> list[Point]:
    if len(pts) < 2:
        return list(pts)
    cumulative = [0.0]
    for a, b in zip(pts, pts[1:]):
        cumulative.append(cumulative[-1] + distance(a, b))
    total = cumulative[-1]
    if total <= step:
        return [pts[0], pts[-1]]
    out = [pts[0]]
    target = step
    seg = 0
    while target < total:
        while seg + 1 < len(cumulative) and cumulative[seg + 1] < target:
            seg += 1
        span = max(1e-9, cumulative[seg + 1] - cumulative[seg])
        u = (target - cumulative[seg]) / span
        a, b = pts[seg], pts[seg + 1]
        out.append((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])))
        target += step
    if distance(out[-1], pts[-1]) >= 0.75 * step:
        out.append(pts[-1])
    return out


def _smoothness_metrics(
    pts: list[Point],
    self_events: list[dict] | None = None,
) -> dict:
    """Measure local curvature and reject the sharp hairpins seen in the bad pass."""
    sampled = resample_polyline(pts, 8.0)
    min_radius = float("inf")
    max_turn = 0.0
    for a, b, c in zip(sampled, sampled[1:], sampled[2:]):
        ab, bc, ac = distance(a, b), distance(b, c), distance(a, c)
        if min(ab, bc, ac) < 1e-6:
            continue
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > 1e-6:
            min_radius = min(min_radius, ab * bc * ac / (2.0 * cross))
        dot = ((b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])) / (ab * bc)
        max_turn = max(max_turn, math.degrees(math.acos(max(-1.0, min(1.0, dot)))))

    if self_events is None:
        self_events = polyline_crossing_events(pts, pts, self_cross=True)
    loop_spans = [abs(e["pos_b"] - e["pos_a"]) for e in self_events]
    return {
        "min_turn_radius_px": min_radius,
        "max_local_turn_deg": max_turn,
        "min_self_loop_span_px": min(loop_spans, default=float("inf")),
    }


def _range_score(value: float, target: tuple[float, float]) -> float:
    lo, hi = target
    if lo <= value <= hi:
        return 1.0
    span = max(1e-6, hi - lo)
    return max(0.0, 1.0 - (lo - value) / span) if value < lo else max(0.0, 1.0 - (value - hi) / span)


def basic_rope_metrics(dense: list[Point], w: int, h: int) -> dict:
    arc = polyline_length(dense)
    chord = max(1.0, distance(dense[0], dense[-1]))
    self_events = polyline_crossing_events(dense, dense, self_cross=True)
    large_loops = sum(abs(e["pos_b"] - e["pos_a"]) >= 120.0 for e in self_events)
    metrics = {
        "arc_px": arc,
        "arc_over_chord": arc / chord,
        "self_crossings": len(self_events),
        "large_loops": large_loops,
        "central_reentries": _central_reentries(dense, w, h),
        "x_reversals": _direction_reversals(dense),
        "lateral_reversals": _lateral_reversals(dense),
        "self_min_angle": min((e["angle"] for e in self_events), default=180.0),
    }
    metrics.update(_smoothness_metrics(dense, self_events=self_events))
    return metrics


def analyze_rope_scene(
    dense_list: list[list[Point]],
    w: int,
    h: int,
) -> tuple[list[set], list[int], list[float], dict, list[dict], int, float]:
    n = len(dense_list)
    crosses = [set() for _ in range(n)]
    crossing_counts = [0] * n
    curve_min_angle = [180.0] * n
    pair_angle: dict = {}
    crossing_positions: list[list[float]] = [[] for _ in range(n)]
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            events = polyline_crossing_events(dense_list[i], dense_list[j])
            if not events:
                continue
            crosses[i].add(j)
            crosses[j].add(i)
            crossing_counts[i] += len(events)
            crossing_counts[j] += len(events)
            total += len(events)
            mn = min(e["angle"] for e in events)
            pair_angle[(i, j)] = mn
            curve_min_angle[i] = min(curve_min_angle[i], mn)
            curve_min_angle[j] = min(curve_min_angle[j], mn)
            crossing_positions[i].extend(e["pos_a"] for e in events)
            crossing_positions[j].extend(e["pos_b"] for e in events)

    metrics: list[dict] = []
    for i, dense in enumerate(dense_list):
        m = basic_rope_metrics(dense, w, h)
        positions = sorted(crossing_positions[i])
        spacing = min((b - a for a, b in zip(positions, positions[1:])), default=float("inf"))
        others = [d for j, d in enumerate(dense_list) if j != i]
        m.update({
            "crossings": crossing_counts[i],
            "distinct_crossed_curves": len(crosses[i]),
            "min_cross_angle": curve_min_angle[i],
            "min_crossing_spacing_px": spacing,
            "near_parallel_px": _near_parallel_length(dense, others),
        })
        metrics.append(m)
    global_min = min(curve_min_angle, default=180.0)
    return crosses, crossing_counts, curve_min_angle, pair_angle, metrics, total, global_min


def crossing_distribution(dense_list: list[list[Point]], w: int, h: int) -> dict:
    """Measure whether crossings occupy the canvas instead of one small knot."""
    points: list[Point] = []
    for i in range(len(dense_list)):
        for j in range(i + 1, len(dense_list)):
            points.extend(e["point"] for e in polyline_crossing_events(dense_list[i], dense_list[j]))
    if not points:
        return {
            "crossing_grid_cells": 0,
            "central_crossings": 0,
            "crossing_bbox_area_frac": 0.0,
        }

    occupied = set()
    for x, y in points:
        gx = max(0, min(2, int(3 * x / w)))
        gy = max(0, min(2, int(3 * y / h)))
        occupied.add((gx, gy))
    central = sum(
        0.30 * w <= x <= 0.70 * w and 0.30 * h <= y <= 0.70 * h
        for x, y in points
    )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys)) / max(1.0, w * h)
    return {
        "crossing_grid_cells": len(occupied),
        "central_crossings": central,
        "crossing_bbox_area_frac": bbox_area,
    }


def rope_metric_score(metric: dict, spec: PathfinderLevel, min_cross_angle: float) -> float:
    # The crossing-angle constraint is between DISTINCT ropes. At a self-crossing
    # both branches have the same connectivity, so it cannot change the answer.
    angle_score = 1.0 if metric["min_cross_angle"] >= min_cross_angle else 0.0
    return (
        2.0 * _range_score(metric["arc_over_chord"], spec.arc_ratio)
        + 2.4 * _range_score(metric["crossings"], spec.target_crossings)
        + 1.2 * _range_score(metric["distinct_crossed_curves"], spec.target_distinct)
        + 1.4 * _range_score(metric["large_loops"], spec.target_loops)
        + 0.8 * _range_score(metric["central_reentries"], spec.target_reentries)
        + 1.0 * _range_score(metric["x_reversals"], spec.target_reversals)
        + 1.4 * angle_score
    )


def _endpoint_permutation(
    starts: list[Point],
    ends: list[Point],
    w: int,
    h: int,
    rng: random.Random,
) -> list[int]:
    """Choose a shuffled matching without trivial short connections."""
    n = len(starts)
    min_wanted = 0.36 * math.hypot(w, h)
    best = None
    best_key = None
    for _ in range(160):
        perm = list(range(n))
        rng.shuffle(perm)
        lengths = [distance(starts[i], ends[perm[i]]) for i in range(n)]
        key = (sum(d >= min_wanted for d in lengths), min(lengths), sum(lengths))
        if best is None or key > best_key:
            best, best_key = perm, key
        if key[0] == n:
            break
    return best


def _long_chord_endpoint_permutation(
    starts: list[Point],
    ends: list[Point],
    w: int,
    h: int,
) -> tuple[list[int], dict]:
    """Exact bottleneck matching for Bezier xhard endpoint pairs.

    Nearly every pair must span a large part of the canvas, but two shorter
    pairs are allowed so adjacent-side and scattered layouts remain viable.
    """
    n = len(starts)
    diagonal = math.hypot(w, h)
    long_threshold = 0.45 * diagonal
    dp: dict[int, tuple[tuple[int, float, float], tuple[int, ...]]] = {
        0: ((0, float("inf"), 0.0), ())
    }
    for i in range(n):
        next_dp: dict[int, tuple[tuple[int, float, float], tuple[int, ...]]] = {}
        for mask, (score, permutation) in dp.items():
            for j in range(n):
                if mask & (1 << j):
                    continue
                chord = distance(starts[i], ends[j])
                next_score = (
                    score[0] + int(chord >= long_threshold),
                    min(score[1], chord),
                    score[2] + chord,
                )
                next_mask = mask | (1 << j)
                if next_mask not in next_dp or next_score > next_dp[next_mask][0]:
                    next_dp[next_mask] = (next_score, permutation + (j,))
        dp = next_dp

    score, permutation = dp[(1 << n) - 1]
    stats = {
        "long_count": score[0],
        "min_chord_px": score[1],
        "mean_chord_px": score[2] / n,
        "long_threshold_px": long_threshold,
        "valid": (
            score[0] >= n - 2
            and score[1] >= 0.26 * diagonal
            and score[2] / n >= 0.43 * diagonal
        ),
    }
    return list(permutation), stats


def build_bezier_scene(
    level_name: str,
    w: int,
    h: int,
    rng: random.Random,
    min_cross_angle: float,
    max_attempts: int,
    color_mode_override: str | None,
    line_style_override: str | None,
    qa_type: str | None = None,
    marker_layout_override: str | None = None,
    rope_outline_override: bool | None = None,
    _layout_retry: int = 0,
) -> Scene:
    """Build a scene from engine.py's fixed-endpoint Bezier geometry."""
    spec = BEZIER_LEVELS[level_name]
    n = min(rng.randint(*spec.num_curves), len(START_LABELS))

    color_mode = color_mode_override or rng.choices(
        ["distinct", "family", "uniform"], weights=[0.10, 0.20, 0.70], k=1,
    )[0]
    force_outline = line_style_override == "outlined"
    line_style = (
        ("solid" if force_outline else line_style_override)
        if line_style_override is not None
        else ("dashed" if rng.random() < 0.5 else "solid")
    )
    if color_mode not in {"distinct", "family", "uniform"}:
        raise ValueError(f"unknown color mode: {color_mode}")
    if line_style not in {"solid", "dashed"}:
        raise ValueError(f"unknown line style: {line_style}")
    line_width = rng.choice(spec.line_widths)
    if line_style == "solid" and rng.random() < 0.5:
        line_width += 1
    rope_outline = (
        rope_outline_override
        if rope_outline_override is not None
        else (force_outline or rng.random() < 0.20)
    )

    # Bezier scenes are geometry-driven rather than layout-driven. Endpoint
    # positions vary freely; named layout quotas are reserved for Pathfinder.
    starts, end_positions, marker_layout = free_bezier_markers(n, w, h, rng)
    if level_name == "xhard":
        permutation, chord_stats = _long_chord_endpoint_permutation(
            starts, end_positions, w, h,
        )
        if not chord_stats["valid"]:
            if _layout_retry >= 14:
                raise RuntimeError(
                    "could not place long-chord xhard Bezier endpoints "
                    "after 15 layout attempts"
                )
            return build_bezier_scene(
                level_name, w, h, rng, min_cross_angle, max_attempts,
                color_mode_override, line_style_override, qa_type,
                None,
                rope_outline_override, _layout_retry + 1,
            )
    else:
        permutation = _endpoint_permutation(starts, end_positions, w, h, rng)
    ends = [end_positions[j] for j in permutation]
    all_markers = starts + end_positions

    best = None
    best_rank = None
    attempt_cap = 6 if level_name == "xhard" else 14
    attempts = max(4, min(max_attempts, attempt_cap))
    for _ in range(attempts):
        controls_list: list[list[Point]] = []
        dense_list: list[list[Point]] = []
        valid_geometry = True
        for start, end in zip(starts, ends):
            accepted = None
            candidate_cap = 80 if level_name == "xhard" else 60
            chord_fraction = distance(start, end) / math.hypot(w, h)
            short_xhard_chord = level_name == "xhard" and chord_fraction < 0.40
            curve_length_scale = 1.80 if short_xhard_chord else spec.length_scale
            curve_max_knots = 9 if short_xhard_chord else spec.max_knots
            curve_bend_waves = (2, 3) if short_xhard_chord else spec.bend_waves
            curve_sway_scale = (0.28, 0.40) if short_xhard_chord else spec.sway_scale
            curve_wave_decay = 0.72 if short_xhard_chord else spec.wave_decay
            curve_phase_jitter = 0.30 if short_xhard_chord else spec.phase_jitter
            curve_min_arc_ratio = 1.18 if short_xhard_chord else spec.min_arc_ratio
            curve_min_reversals = 1 if short_xhard_chord else spec.min_lateral_reversals
            for _candidate in range(candidate_cap):
                controls, dense = fixed_bezier_curve(
                    start, end, w, h, rng,
                    length_scale=curve_length_scale,
                    max_knots=curve_max_knots,
                    bend_waves=curve_bend_waves,
                    sway_scale=curve_sway_scale,
                    wave_decay=curve_wave_decay,
                    phase_jitter=curve_phase_jitter,
                )
                if len(dense) < 3:
                    continue
                own = basic_rope_metrics(dense, w, h)
                if (
                    own["min_turn_radius_px"] < spec.min_turn_radius_px
                    or own["max_local_turn_deg"] > spec.max_local_turn_deg
                    or own["self_crossings"] > spec.max_self_crossings
                    or own["arc_over_chord"] < curve_min_arc_ratio
                    or own["lateral_reversals"] < curve_min_reversals
                    or not endpoint_safe(
                        dense, all_markers, {start, end},
                        layout_params(w, h)["marker_r"] + 7,
                    )
                ):
                    continue
                accepted = (controls, dense)
                break
            if accepted is None:
                valid_geometry = False
                break
            controls, dense = accepted
            controls_list.append(controls)
            dense_list.append(dense)
        if not valid_geometry:
            continue

        crosses, counts, curve_min, pair_angle, metrics, total, global_min = (
            analyze_rope_scene(dense_list, w, h)
        )
        if qa_type == "all_pairs" and global_min < min_cross_angle:
            continue
        spatial = crossing_distribution(dense_list, w, h)
        clean_curves = sum(angle >= min_cross_angle for angle in curve_min)
        crossing_score = (
            2.5 * _range_score(total / max(1, n), spec.target_crossings)
            + 1.2 * _range_score(
                sum(len(x) for x in crosses) / max(1, n),
                spec.target_distinct,
            )
        )
        geometry_score = (
            1.8 * _range_score(
                sum(m["arc_over_chord"] for m in metrics) / max(1, n),
                (spec.min_arc_ratio, spec.min_arc_ratio + 0.55),
            )
            + 1.4 * _range_score(
                sum(m["lateral_reversals"] for m in metrics) / max(1, n),
                (spec.min_lateral_reversals, spec.min_lateral_reversals + 2),
            )
        )
        spatial_score = (
            min(1.0, spatial["crossing_grid_cells"] / 6.0)
            + min(1.0, spatial["central_crossings"] / 3.0)
            + min(1.0, spatial["crossing_bbox_area_frac"] / 0.25)
        )
        medium_ok = (
            level_name != "medium"
            or (
                spatial["crossing_grid_cells"] >= 4
                and spatial["central_crossings"] >= 2
                and spatial["crossing_bbox_area_frac"] >= 0.16
            )
        )
        rank = (
            1 if medium_ok else 0,
            clean_curves,
            crossing_score + geometry_score
            + (spatial_score if level_name in {"medium", "xhard"} else 0.0),
            total,
        )
        if best is None or rank > best_rank:
            best_rank = rank
            best = (
                controls_list, dense_list, crosses, counts, curve_min, pair_angle,
                metrics, total, global_min, spatial,
            )

    if best is None or (level_name == "medium" and best_rank[0] == 0):
        if _layout_retry >= 14:
            raise RuntimeError(
                f"could not build clean {level_name} Bezier scene after 15 layouts"
            )
        return build_bezier_scene(
            level_name, w, h, rng, min_cross_angle, max_attempts,
            color_mode_override, line_style_override, qa_type,
            None,
            rope_outline_override, _layout_retry + 1,
        )

    (
        controls_list, dense_list, crosses, counts, curve_min, pair_angle,
        metrics, total, global_min, spatial,
    ) = best
    colors = curve_colors(n, color_mode, rng)
    representative = tuple(round(sum(c[k] for c in colors) / len(colors)) for k in range(3))
    shared_outline = contrasting_outline(representative, rng)
    curves = [
        Curve(
            si=START_LABELS[i],
            ei=END_LABELS[permutation[i]],
            start=starts[i],
            end=ends[i],
            controls=controls_list[i],
            dense=dense_list[i],
            trace=trace_points(dense_list[i]),
            color=colors[i],
            outline_color=shared_outline,
            arc=metrics[i]["arc_px"],
            chord=distance(starts[i], ends[i]),
        )
        for i in range(n)
    ]
    return Scene(
        curves=curves,
        crosses=crosses,
        crossing_counts=counts,
        min_angle=global_min if global_min < 180 else None,
        total_crossings=total,
        endpoint_clean=True,
        color_mode=color_mode,
        line_width=line_width,
        line_style=line_style,
        rope_outline=rope_outline,
        curve_min_angle=curve_min,
        pair_angle=pair_angle,
        angle_floor=min_cross_angle,
        curve_style="bezier",
        pathfinder_family="bezier_spline",
        marker_layout=marker_layout,
        curve_metrics=metrics,
        crossing_grid_cells=spatial["crossing_grid_cells"],
        central_crossings=spatial["central_crossings"],
        crossing_bbox_area_frac=spatial["crossing_bbox_area_frac"],
    )


def build_pathfinder_scene(
    level_name: str,
    w: int,
    h: int,
    rng: random.Random,
    min_cross_angle: float,
    max_attempts: int,
    color_mode_override: str | None,
    line_style_override: str | None,
    qa_type: str | None = None,
    marker_layout_override: str | None = None,
    rope_outline_override: bool | None = None,
    _layout_retry: int = 0,
) -> Scene:
    """Build a fixed-endpoint rope puzzle with a varied marker layout.

    Each attempt generates a full endpoint permutation, then retains the scene
    with the strongest measured target candidates. This is deliberately
    target-aware: difficulty is based on the paths that can actually be queried,
    not just on nominal generation parameters.
    """
    spec = PATHFINDER_LEVELS[level_name]
    family = "scribble_rope" if rng.random() < spec.scribble_frac else "permutation_rope"
    if qa_type == "all_pairs":
        # Every rope is queried. Keep the 7-10 pair count, but use the cleaner
        # permutation family so all inter-rope crossings can satisfy 25 degrees.
        family = "permutation_rope"
    n_range = spec.scribble_curves if family == "scribble_rope" else spec.num_curves
    n = min(rng.randint(*n_range), len(START_LABELS))
    modes = list(spec.color_weights)
    color_mode = (
        color_mode_override
        if color_mode_override is not None
        else rng.choices(modes, weights=[spec.color_weights[m] for m in modes], k=1)[0]
    )
    if color_mode not in {"distinct", "family", "uniform"}:
        raise ValueError(f"unknown color_mode_override: {color_mode}")
    force_outline = line_style_override == "outlined"
    if line_style_override is None:
        line_style = "dashed" if rng.random() < 0.5 else "solid"
    else:
        line_style = "solid" if force_outline else line_style_override
    if line_style not in {"solid", "dashed"}:
        raise ValueError(f"unknown line_style_override: {line_style}")
    line_width = rng.choice(spec.line_widths)
    if line_style == "solid" and rng.random() < 0.5:
        line_width += 1
    rope_outline = (
        rope_outline_override
        if rope_outline_override is not None
        else (force_outline or rng.random() < spec.outline_frac)
    )

    starts, end_positions, marker_layout = varied_marker_layout(
        n, w, h, rng, curve_style="pathfinder",
        layout_override=marker_layout_override,
    )
    randomized_hard_layout = (
        level_name in {"hard", "xhard"}
        and marker_layout in {"perimeter_random", "scattered"}
    )
    if randomized_hard_layout:
        # Very long scribble arms are often impossible when both endpoint sets
        # occupy the full perimeter. Keep 7-10 pairs, but use the cleaner rope
        # family and slightly broader/shorter sweeps for these layouts.
        family = "permutation_rope"
    permutation = _endpoint_permutation(starts, end_positions, w, h, rng)
    ends = [end_positions[j] for j in permutation]

    best = None
    best_score = -float("inf")
    attempt_cap = 6 if level_name == "medium" else (4 if qa_type == "all_pairs" else 3)
    attempts = max(2, min(max_attempts, attempt_cap))
    all_markers = starts + end_positions
    for _ in range(attempts):
        dense_list: list[list[Point]] = []
        attempt_failed = False
        for i, (start, end) in enumerate(zip(starts, ends)):
            candidates = []
            candidate_tries = 0
            candidate_target = 4 if qa_type == "all_pairs" else 5
            candidate_cap = (
                160 if qa_type == "all_pairs" or randomized_hard_layout else 90
            )
            while len(candidates) < candidate_target and candidate_tries < candidate_cap:
                candidate_tries += 1
                length_bump = (
                    0.78 if qa_type == "all_pairs"
                    else (
                        (0.65 if level_name == "hard" else 0.50)
                        if randomized_hard_layout
                        else (1.12 if family == "scribble_rope" else 1.0)
                    )
                )
                radius_scale = (
                    1.12 if qa_type == "all_pairs"
                    else (
                        (1.12 if level_name == "hard" else 1.18)
                        if randomized_hard_layout else 1.0
                    )
                )
                dense = endpoint_rope(
                    start, end, w, h, rng,
                    arm_length_scale=(
                        spec.arm_length_scale[0] * length_bump,
                        spec.arm_length_scale[1] * length_bump,
                    ),
                    radius=(
                        spec.radius[0] * radius_scale,
                        spec.radius[1] * radius_scale,
                    ),
                    min_radius=spec.min_turn_radius_px,
                )
                if len(dense) < 3:
                    continue
                own = basic_rope_metrics(dense, w, h)
                smooth_enough = (
                    own["min_turn_radius_px"] >= spec.min_turn_radius_px
                    and own["min_self_loop_span_px"] >= spec.min_self_loop_span_px
                    and own["self_crossings"] <= spec.max_self_crossings
                )
                if not smooth_enough:
                    continue
                events_by_prior = [polyline_crossing_events(dense, other) for other in dense_list]
                prior_events = [event for events in events_by_prior for event in events]
                prior_cross = len(prior_events)
                prior_distinct = sum(bool(events) for events in events_by_prior)
                expected_cross = spec.target_crossings[0] * i / max(1, n - 1)
                endpoint_clear = endpoint_safe(
                    dense, all_markers, {start, end}, layout_params(w, h)["marker_r"] + 7
                )
                if not endpoint_clear:
                    continue
                score = (
                    2.4 * _range_score(own["arc_over_chord"], spec.arc_ratio)
                    + 1.2 * _range_score(own["large_loops"], spec.target_loops)
                    + 0.8 * _range_score(own["central_reentries"], spec.target_reentries)
                    + 1.0 * _range_score(own["x_reversals"], spec.target_reversals)
                    + 1.5 * min(1.0, prior_cross / max(1.0, expected_cross))
                    + 0.5 * min(1.0, prior_distinct / max(1, spec.target_distinct[0]))
                    + 2.0
                )
                min_candidate_angle = min((e["angle"] for e in prior_events), default=180.0)
                angle_clean = min_candidate_angle >= min_cross_angle
                if qa_type == "all_pairs" and not angle_clean:
                    continue
                if not angle_clean:
                    score -= 7.0 + 3.0 * (min_cross_angle - min_candidate_angle) / min_cross_angle
                candidates.append(((1 if angle_clean else 0, score), dense))
            if not candidates:
                attempt_failed = True
                break
            dense_list.append(max(candidates, key=lambda x: x[0])[1])
        if attempt_failed:
            continue

        analyzed = analyze_rope_scene(dense_list, w, h)
        crosses, counts, curve_min, pair_angle, metrics, total, global_min = analyzed
        if qa_type == "all_pairs" and global_min < min_cross_angle:
            continue
        spatial = crossing_distribution(dense_list, w, h)
        ranked = sorted(
            (rope_metric_score(m, spec, min_cross_angle) for m in metrics),
            reverse=True,
        )
        # Favor several viable query paths, rather than one pathological outlier.
        top_k = ranked[:max(2, n // 2)]
        scene_score = sum(top_k) / len(top_k)
        scene_score += 0.15 * min(total, n * spec.target_crossings[1]) / max(1, n)
        if level_name == "medium":
            scene_score += 2.0 * min(1.0, spatial["crossing_grid_cells"] / 6.0)
            scene_score += 1.5 * min(1.0, spatial["central_crossings"] / 3.0)
            scene_score += 1.0 * min(1.0, spatial["crossing_bbox_area_frac"] / 0.25)
        spatial_ok = (
            level_name != "medium"
            or (
                spatial["crossing_grid_cells"] >= 4
                and spatial["central_crossings"] >= 2
                and spatial["crossing_bbox_area_frac"] >= 0.16
            )
        )
        rank = (1 if spatial_ok else 0, scene_score)
        if best is None or rank > best_score:
            best_score = rank
            best = (
                dense_list, crosses, counts, curve_min, pair_angle, metrics,
                total, global_min, spatial,
            )

    if best is None:
        # Regenerate the marker layout rather than emitting a geometrically
        # invalid example. The caller already retries whole scenes.
        if _layout_retry >= 12:
            raise RuntimeError(
                f"could not build clean {level_name} rope scene after 13 marker layouts"
            )
        return build_pathfinder_scene(
            level_name, w, h, rng, min_cross_angle,
            max_attempts=max_attempts,
            color_mode_override=color_mode_override,
            line_style_override=line_style_override,
            qa_type=qa_type,
            marker_layout_override=marker_layout_override,
            rope_outline_override=rope_outline_override,
            _layout_retry=_layout_retry + 1,
        )

    dense_list, crosses, counts, curve_min, pair_angle, metrics, total, global_min, spatial = best
    medium_spatial_ok = (
        spatial["crossing_grid_cells"] >= 4
        and spatial["central_crossings"] >= 2
        and spatial["crossing_bbox_area_frac"] >= 0.16
    )
    if level_name == "medium" and not medium_spatial_ok:
        if _layout_retry >= 12:
            raise RuntimeError(
                "could not build spatially distributed medium scene after 13 marker layouts"
            )
        return build_pathfinder_scene(
            level_name, w, h, rng, min_cross_angle,
            max_attempts=max_attempts,
            color_mode_override=color_mode_override,
            line_style_override=line_style_override,
            qa_type=qa_type,
            marker_layout_override=marker_layout_override,
            rope_outline_override=rope_outline_override,
            _layout_retry=_layout_retry + 1,
        )
    colors = curve_colors(n, color_mode, rng)
    representative = tuple(round(sum(c[k] for c in colors) / len(colors)) for k in range(3))
    shared_outline = contrasting_outline(representative, rng)
    curves: list[Curve] = []
    for i in range(n):
        dense = dense_list[i]
        curves.append(Curve(
            si=START_LABELS[i],
            ei=END_LABELS[permutation[i]],
            start=starts[i],
            end=ends[i],
            controls=[starts[i], ends[i]],
            dense=dense,
            trace=trace_points(dense),
            color=colors[i], outline_color=shared_outline,
            arc=metrics[i]["arc_px"],
            chord=distance(starts[i], ends[i]),
        ))

    lp = layout_params(w, h)
    clearance = lp["marker_r"] + 7
    marker_clean = True
    for i, curve in enumerate(curves):
        for marker in (curve.start, curve.end):
            if not all(endpoint_safe(curves[j].dense, [marker], set(), clearance)
                       for j in range(n) if j != i):
                marker_clean = False
                break

    return Scene(
        curves=curves,
        crosses=crosses,
        crossing_counts=counts,
        min_angle=global_min if global_min < 180.0 else None,
        total_crossings=total,
        endpoint_clean=marker_clean,
        color_mode=color_mode,
        line_width=line_width,
        line_style=line_style,
        rope_outline=rope_outline,
        curve_min_angle=curve_min,
        pair_angle=pair_angle,
        angle_floor=min_cross_angle,
        curve_style="pathfinder",
        pathfinder_family=family,
        marker_layout=marker_layout,
        curve_metrics=metrics,
        crossing_grid_cells=spatial["crossing_grid_cells"],
        central_crossings=spatial["central_crossings"],
        crossing_bbox_area_frac=spatial["crossing_bbox_area_frac"],
    )


def build_scene(lvl: Level, w: int, h: int, rng: random.Random,
                min_cross_angle: float = 25.0, max_attempts: int = 16,
                curve_style: str = "pathfinder",
                color_mode_override: str | None = None,
                line_style_override: str | None = None,
                qa_type: str | None = None,
                marker_layout_override: str | None = None,
                rope_outline_override: bool | None = None) -> Scene:
    if curve_style not in CURVE_STYLES:
        raise ValueError(f"unknown curve_style: {curve_style}")
    if curve_style == "pathfinder":
        return build_pathfinder_scene(
            lvl.name, w, h, rng,
            min_cross_angle=min_cross_angle,
            max_attempts=max_attempts,
            color_mode_override=color_mode_override,
            line_style_override=line_style_override,
            qa_type=qa_type,
            marker_layout_override=marker_layout_override,
            rope_outline_override=rope_outline_override,
        )
    if curve_style == "bezier":
        return build_bezier_scene(
            lvl.name, w, h, rng,
            min_cross_angle=min_cross_angle,
            max_attempts=max_attempts,
            color_mode_override=color_mode_override,
            line_style_override=line_style_override,
            qa_type=qa_type,
            marker_layout_override=marker_layout_override,
            rope_outline_override=rope_outline_override,
        )
    n = rng.randint(*lvl.num_curves)
    n = min(n, len(START_LABELS))
    modes = list(lvl.color_weights)
    color_mode = (
        color_mode_override
        if color_mode_override is not None
        else rng.choices(modes, weights=[lvl.color_weights[m] for m in modes], k=1)[0]
    )
    if color_mode not in {"distinct", "family", "uniform"}:
        raise ValueError(f"unknown color_mode_override: {color_mode}")
    line_width = rng.choice(lvl.line_widths)
    line_style = (
        line_style_override
        if line_style_override is not None
        else ("dashed" if rng.random() < lvl.dashed_frac else "solid")
    )
    if line_style not in {"solid", "dashed"}:
        raise ValueError(f"unknown line_style_override: {line_style}")
    lp = layout_params(w, h)
    clearance = lp["marker_r"] + 10
    min_gap = lp["marker_r"] * 2 + 10
    self_margin = max(20.0, line_width * 3.5 + 10.0)
    cxc, cyc = w / 2.0, h / 2.0
    min_len = lvl.length[0] * 0.6

    best = None
    best_rank = None
    for _ in range(max_attempts):
        # all START markers known up front -> every snake avoids all of them; END
        # markers (snake ends) are added to the obstacle set as they are placed, so
        # later snakes avoid earlier endpoints too. No line ever runs over a marker.
        columns = lvl.layout == "columns"
        goal_x = (w - lp["margin"]) if columns else None      # two-column: grow to right edge
        drift_x = 0.065 if columns else 0.0                   # gentle -> lines wander -> dense tangle
        starts = column_markers(n, w, h, rng, "left") if columns else perimeter_markers(n, w, h, rng)
        markers = list(starts)
        dense_list, ends, ok = [], [], True
        for i in range(n):
            heading0 = 0.0 if columns else math.atan2(cyc - starts[i][1], cxc - starts[i][0])
            obstacles = [m for k, m in enumerate(markers) if m != starts[i]]
            dense = _grow_one(lvl, starts[i], heading0, obstacles, w, h, rng,
                              self_margin, clearance + 4, dense_list, clearance, min_len, min_gap,
                              curve_style=curve_style, goal_x=goal_x, drift_x=drift_x)
            if dense is None:
                ok = False
                dense = [starts[i], (cxc, cyc)]
            # TRIM the tail back until the END marker lands in clear space: away from
            # every foreign line body (>=clearance) and every existing marker
            # (>=min_gap). This deterministically guarantees an unambiguous endpoint
            # (later snakes also avoid it), instead of relying on re-rolls.
            def end_clean(p):
                return (all(distance(p, m) >= min_gap for m in obstacles)
                        and all(endpoint_safe(dl, [p], set(), clearance) for dl in dense_list))
            while len(dense) > 6 and not end_clean(dense[-1]):
                dense = dense[:-2]
            dense_list.append(dense)
            ends.append(dense[-1])
            markers.append(dense[-1])

        # verify every marker is clear of every NON-owning line (no ambiguous dots)
        markers_clear = 0
        for i in range(n):
            for mk in (starts[i], ends[i]):
                if all(endpoint_safe(dense_list[j], [mk], set(), clearance)
                       for j in range(n) if j != i):
                    markers_clear += 1
        # also penalise two markers landing on top of each other (overlapping badges)
        all_marks = list(starts) + list(ends)
        marker_pairs_ok = sum(
            1 for a in range(len(all_marks)) for b in range(a + 1, len(all_marks))
            if distance(all_marks[a], all_marks[b]) >= min_gap)

        coarse = [_coarse(d) for d in dense_list]
        crosses = [set() for _ in range(n)]
        counts = [0] * n
        curve_min = [180.0] * n
        pair_angle: dict = {}
        min_angle = 180.0
        for i in range(n):
            for j in range(i + 1, n):
                hits = polyline_crossings(coarse[i], coarse[j])
                if hits:
                    crosses[i].add(j); crosses[j].add(i)
                for _pt, ang in hits:
                    counts[i] += 1; counts[j] += 1
                    min_angle = min(min_angle, ang)
                    curve_min[i] = min(curve_min[i], ang)
                    curve_min[j] = min(curve_min[j], ang)
                    pair_angle[(i, j)] = min(pair_angle.get((i, j), 180.0), ang)
        total = sum(counts) // 2
        # how many curves could host a WELL-POSED query (all their crossings >=floor)
        n_clean_curves = sum(1 for i in range(n) if curve_min[i] >= min_cross_angle)
        under_cap = total <= lvl.max_cross
        # two-column: count end markers that actually reached the right edge (so they
        # form a clean right column instead of stranding in the middle).
        ends_reached = (sum(1 for e in ends if e[0] >= goal_x - 25) if columns else n)
        # rank: UNAMBIGUOUS MARKERS first (no dot on a foreign line + no overlapping
        # badges), then ends-reached (columns), all snakes grew, well-posed query hosts.
        max_pairs = len(all_marks) * (len(all_marks) - 1) // 2
        rank = (markers_clear, marker_pairs_ok, ends_reached, 1 if ok else 0, n_clean_curves,
                1 if under_cap else 0, min_angle, -total)
        if best is None or rank > best_rank:
            best_rank = rank
            best = dict(starts=starts, ends=ends, dense_list=dense_list,
                        crosses=crosses, counts=counts, curve_min=curve_min, pair_angle=pair_angle,
                        min_angle=min_angle, clean=ok, total=total)
        if (markers_clear == 2 * n and marker_pairs_ok == max_pairs and ok
                and under_cap and n_clean_curves >= max(1, n // 2)):
            break

    colors = curve_colors(n, color_mode, rng)
    end_labels = END_LABELS[:n]
    rng.shuffle(end_labels)
    curves = []
    for i in range(n):
        arc = polyline_length(best["dense_list"][i])
        ch = distance(best["starts"][i], best["ends"][i])
        curves.append(Curve(
            si=START_LABELS[i], ei=end_labels[i], start=best["starts"][i], end=best["ends"][i],
            controls=[best["starts"][i], best["ends"][i]], dense=best["dense_list"][i],
            trace=trace_points(best["dense_list"][i]), color=colors[i], outline_color=None,
            arc=arc, chord=ch))
    return Scene(curves, best["crosses"], best["counts"],
                 best["min_angle"] if best["min_angle"] < 180 else None,
                 best["total"], best["clean"], color_mode, line_width, line_style,
                 curve_min_angle=best["curve_min"], pair_angle=best["pair_angle"],
                 angle_floor=min_cross_angle)


def endpoint_safe(curve, markers, own, radius):
    chk = curve[::4]
    if chk[-1] != curve[-1]:
        chk.append(curve[-1])
    for m in markers:
        if m in own:
            continue
        for a, b in zip(chk, chk[1:]):
            if point_segment_distance(m, a, b) < radius:
                return False
    return True


# -----------------------------------------------------------------------------
# QA
# -----------------------------------------------------------------------------
def _ordered_indices(curves: list[Curve]) -> list[int]:
    return sorted(range(len(curves)), key=lambda i: curves[i].si)


def make_qa(scene: Scene, qa_type: str, rng: random.Random, min_target_crossings: int = 0) -> dict:
    curves = scene.curves
    n = len(curves)
    oi = _ordered_indices(curves)

    floor = scene.angle_floor
    well_posed = [
        i for i in range(n)
        if not scene.curve_min_angle or scene.curve_min_angle[i] >= floor
    ]

    def hard_pool(idxs):
        # Restrict the queried line to WELL-POSED curves (every crossing on them is
        # >=floor, so the trace is never ambiguous), preferring heavily-crossed ones
        # so the question still tests long-range tracking. If no curve is fully
        # well-posed (very dense scenes), fall back to the CLEANEST available curve
        # (max per-curve min angle) rather than a heavily-crossed shallow one.
        wp = [i for i in idxs if i in well_posed]
        if wp:
            base = wp
            if min_target_crossings > 0:
                pool = [i for i in base if scene.crossing_counts[i] >= min_target_crossings]
                base = pool or base
            if scene.curve_metrics and scene.curve_style in {"pathfinder", "bezier"}:
                # Query the genuinely difficult ropes. Crossing count alone misses
                # long-loop, re-entry, and repeated-direction complexity.
                ranked = sorted(
                    base,
                    key=lambda i: (
                        scene.curve_metrics[i]["crossings"],
                        scene.curve_metrics[i]["large_loops"],
                        scene.curve_metrics[i]["central_reentries"],
                        scene.curve_metrics[i]["arc_over_chord"],
                    ),
                    reverse=True,
                )
                base = ranked[:max(1, len(ranked) // 2)]
            return base
        if scene.curve_min_angle:                       # nothing fully clean -> cleanest
            mx = max(scene.curve_min_angle[i] for i in idxs)
            return [i for i in idxs if scene.curve_min_angle[i] == mx]
        return list(idxs)

    pairs = [{"start": curves[i].si, "end": curves[i].ei} for i in oi]

    if qa_type == "one_marker":
        i = rng.choice(hard_pool(oi))
        c = curves[i]
        if rng.random() < 0.5:
            q = f"What is {c.si} connected to? Answer with only the marker label."
            ans = c.ei
        else:
            q = f"What is {c.ei} connected to? Answer with only the marker label."
            ans = c.si
        return {"type": qa_type, "question": q, "answer": ans, "target_indices": [i], "pairs": pairs}

    if qa_type == "all_pairs":
        txt = ", ".join(f"{curves[i].si}-{curves[i].ei}" for i in oi)
        return {"type": qa_type, "question": "List all connected pairs. Use the format A-1, B-2.",
                "answer": txt, "target_indices": oi, "pairs": pairs}

    if qa_type == "ordered":
        candidates = [i for i in oi if i in well_posed]
        if len(candidates) < 2:
            candidates = sorted(
                oi,
                key=lambda i: scene.curve_min_angle[i] if scene.curve_min_angle else 180.0,
                reverse=True,
            )[:max(2, min(5, n))]
        k = rng.randint(2, min(5, len(candidates)))
        sel = sorted(rng.sample(candidates, k), key=oi.index)
        qs = "".join(curves[i].si for i in sel)
        ans = "".join(curves[i].ei for i in sel)
        return {"type": qa_type, "question": f"What is {qs} connected to, in that order? Answer with only the labels.",
                "answer": ans, "target_indices": sel, "pairs": pairs}

    if qa_type == "same_line":
        want_yes = n < 2 or rng.random() < 0.5
        if want_yes:
            i = rng.choice(hard_pool(list(range(n))))
            ends = [(i, curves[i].start), (i, curves[i].end)]
            tgt = [i]
        else:
            i = rng.choice(hard_pool(list(range(n))))
            # Prefer a partner j that (a) actually crosses i so the dots are
            # entangled (can't be told apart by region) and (b) crosses i at a
            # STEEP (>=floor) angle, so the two lines are clearly distinct rather
            # than grazing/merging -- otherwise "no" could be ambiguous.
            def pa(a, b):
                return scene.pair_angle.get((min(a, b), max(a, b)), 180.0)
            entangled = [k for k in scene.crosses[i] if k != i]
            steep = [k for k in entangled if pa(i, k) >= floor and k in well_posed]
            if steep:
                j = rng.choice(steep)
            elif entangled:
                safe = [k for k in well_posed if k != i]
                j = rng.choice(safe or entangled)
            else:
                j = rng.choice([k for k in range(n) if k != i])
            ei = rng.choice([curves[i].start, curves[i].end])
            ej = rng.choice([curves[j].start, curves[j].end])
            ends = [(i, ei), (j, ej)]
            tgt = [i, j]
        cols = two_dot_colors(rng)
        dot_markers = [{"point": [int(round(p[0])), int(round(p[1]))], "color": list(c)}
                       for (_, p), c in zip(ends, cols)]
        return {"type": qa_type, "question": "Are the two colored dots connected by the same line? Answer yes or no.",
                "answer": "yes" if want_yes else "no", "target_indices": tgt,
                "dot_markers": dot_markers, "pairs": pairs}

    raise ValueError(qa_type)


def draw_dashed(draw, pts, fill, width, dash, gap):
    period = dash + gap
    pos = 0.0
    cap = max(1, width // 2)
    for a, b in zip(pts, pts[1:]):
        seg = distance(a, b)
        if seg < 1e-9:
            continue
        dx, dy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
        s = 0.0
        while s < seg:
            phase = pos % period
            if phase < dash:
                step = min(dash - phase, seg - s)
                p0 = (a[0] + dx * s, a[1] + dy * s)
                p1 = (a[0] + dx * (s + step), a[1] + dy * (s + step))
                draw.line([p0, p1], fill=fill, width=width, joint="curve")
                draw.ellipse((p1[0] - cap, p1[1] - cap, p1[0] + cap, p1[1] + cap), fill=fill)
            else:
                step = min(period - phase, seg - s)
            s += step
            pos += step


def render(scene: Scene, qa: dict, w: int, h: int) -> Image.Image:
    ss = 3 if scene.curve_style in {"pathfinder", "bezier"} else 1
    lp0 = layout_params(w, h)
    lp = {key: value * ss for key, value in lp0.items()}
    rw, rh = w * ss, h * ss
    img = Image.new("RGB", (rw, rh), (252, 252, 248))
    d = ImageDraw.Draw(img)
    for x in range(0, rw, lp["grid"]):
        d.line((x, 0, x, rh), fill=(238, 238, 232), width=ss)
    for y in range(0, rh, lp["grid"]):
        d.line((0, y, rw, y), fill=(238, 238, 232), width=ss)

    def pt(p: Point) -> tuple[int, int]:
        return (int(round(p[0] * ss)), int(round(p[1] * ss)))

    lw = scene.line_width * ss
    dash, gap = max(8, 4 * lw), max(6, 3 * lw)
    for c in scene.curves:
        pts = [pt(p) for p in c.dense]
        outline = c.outline_color or (35, 35, 35)
        outer_w = lw + 3 * ss
        if scene.line_style == "dashed":
            if scene.rope_outline:
                draw_dashed(d, pts, outline, outer_w, dash, gap)
            draw_dashed(d, pts, c.color, lw, dash, gap)
        else:
            if scene.rope_outline:
                d.line(pts, fill=outline, width=outer_w, joint="curve")
            d.line(pts, fill=c.color, width=lw, joint="curve")

    r = lp["marker_r"]
    dots = qa.get("dot_markers")
    if dots is not None:
        # same_line: mark ONLY the two queried endpoints (small dots). Other
        # endpoints are left unmarked -- the task is just "are these two dots on the
        # same line", so unrelated endpoints don't need badges.
        dr = max(5 * ss, round(r * 0.42))
        for dm in dots:
            x, y = (round(dm["point"][0] * ss), round(dm["point"][1] * ss))
            d.ellipse((x - dr, y - dr, x + dr, y + dr),
                      fill=tuple(dm["color"]), outline=(25, 25, 25), width=ss)
        return img.resize((w, h), Image.Resampling.LANCZOS) if ss > 1 else img

    font = load_bundled_font(lp["font"])
    for c in scene.curves:
        for label, p, ring in [(c.si, c.start, (255, 246, 210)), (c.ei, c.end, (225, 245, 255))]:
            x, y = pt(p)
            d.ellipse((x - r, y - r, x + r, y + r),
                      fill=ring, outline=(35, 35, 35), width=2 * ss)
            bbox = d.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            d.text((x - tw / 2 - bbox[0], y - th / 2 - bbox[1]), label, font=font, fill=(25, 25, 25))
    return img.resize((w, h), Image.Resampling.LANCZOS) if ss > 1 else img


# -----------------------------------------------------------------------------
# one sample
# -----------------------------------------------------------------------------
def generate(level: str, qa_type: str, seed: int, size: int = 448,
             min_cross_angle: float = 25.0, scene_retries: int = 6,
             curve_style: str = "pathfinder",
             color_mode_override: str | None = None,
             line_style_override: str | None = None,
             marker_layout_override: str | None = None,
             rope_outline_override: bool | None = None,
             geometry_salt: int = 0) -> tuple[Image.Image, dict]:
    if curve_style not in CURVE_STYLES:
        raise ValueError(f"unknown curve_style: {curve_style}")
    requested_curve_style = curve_style
    if curve_style == "mixed":
        # Four-seed cycle: PF-solid, PF-dashed, Bezier-solid, Bezier-dashed.
        # This keeps curve family independent of the exact 1:1 stroke schedule.
        curve_style = "pathfinder" if (seed // 2) % 2 == 0 else "bezier"
    rng = random.Random(f"{seed}:{level}:{qa_type}:{curve_style}:geom{geometry_salt}")
    lvl = LEVELS[level]
    effective_color_mode = color_mode_override
    effective_line_style = line_style_override
    effective_rope_outline = rope_outline_override
    if line_style_override == "outlined" and effective_rope_outline is None:
        effective_rope_outline = True
    if curve_style in {"pathfinder", "bezier"}:
        scheduled_color, scheduled_style, scheduled_outline = scheduled_appearance(seed)
        if effective_color_mode is None:
            effective_color_mode = scheduled_color
        if effective_line_style is None:
            effective_line_style = scheduled_style
        if effective_rope_outline is None:
            effective_rope_outline = scheduled_outline
    # ask the queried line to be among the more-crossed ones at higher difficulty
    mtc = {"easy": 0, "medium": 2, "hard": 3, "xhard": 4}[level]
    # Build whole scenes until the line(s) we actually trace clear the angle floor;
    # keep the best-so-far. This is what makes the >=25deg guarantee hold on the
    # TRACED path even when background lines may graze each other in dense scenes.
    best = None
    best_tmin = -1.0
    best_quality = -float("inf")
    for _ in range(max(1, scene_retries)):
        scene = build_scene(
            lvl, size, size, rng,
            min_cross_angle=min_cross_angle,
            curve_style=curve_style,
            color_mode_override=effective_color_mode,
            line_style_override=effective_line_style,
            qa_type=qa_type,
            marker_layout_override=marker_layout_override,
            rope_outline_override=effective_rope_outline,
        )
        qa = make_qa(scene, qa_type, rng, min_target_crossings=mtc)
        # Traced-line well-posedness is the minimum angle where it crosses a
        # DIFFERENT rope. Self-crossings preserve connectivity and are reported
        # separately rather than treated as label-changing junctions.
        tmin = 180.0
        for i in qa["target_indices"]:
            if scene.curve_min_angle:
                tmin = min(tmin, scene.curve_min_angle[i])
        if curve_style == "pathfinder" and scene.curve_metrics:
            spec = PATHFINDER_LEVELS[level]
            target_metrics = [scene.curve_metrics[i] for i in qa["target_indices"]]
            quality = sum(rope_metric_score(m, spec, min_cross_angle) for m in target_metrics) / len(target_metrics)
            quality += 2.0 if tmin >= min_cross_angle else tmin / max(1.0, min_cross_angle)
            if quality > best_quality:
                best_quality, best_tmin, best = quality, tmin, (scene, qa)
            avg_cross = sum(m["crossings"] for m in target_metrics) / len(target_metrics)
            avg_ratio = sum(m["arc_over_chord"] for m in target_metrics) / len(target_metrics)
            avg_loops = sum(m["large_loops"] for m in target_metrics) / len(target_metrics)
            if (tmin >= min_cross_angle
                    and avg_cross >= 0.75 * spec.target_crossings[0]
                    and avg_ratio >= 0.85 * spec.arc_ratio[0]
                    and avg_loops >= max(0, 0.6 * spec.target_loops[0])):
                break
        elif curve_style == "bezier" and scene.curve_metrics:
            spec = BEZIER_LEVELS[level]
            target_metrics = [scene.curve_metrics[i] for i in qa["target_indices"]]
            avg_cross = sum(m["crossings"] for m in target_metrics) / len(target_metrics)
            avg_distinct = (
                sum(m["distinct_crossed_curves"] for m in target_metrics)
                / len(target_metrics)
            )
            avg_ratio = sum(m["arc_over_chord"] for m in target_metrics) / len(target_metrics)
            avg_reversals = (
                sum(m["lateral_reversals"] for m in target_metrics)
                / len(target_metrics)
            )
            quality = (
                2.4 * _range_score(avg_cross, spec.target_crossings)
                + 1.2 * _range_score(avg_distinct, spec.target_distinct)
                + 2.0 * _range_score(
                    avg_ratio,
                    (spec.min_arc_ratio, spec.min_arc_ratio + 0.55),
                )
                + 1.4 * _range_score(
                    avg_reversals,
                    (spec.min_lateral_reversals, spec.min_lateral_reversals + 2),
                )
                + (2.0 if tmin >= min_cross_angle
                   else tmin / max(1.0, min_cross_angle))
            )
            if quality > best_quality:
                best_quality, best_tmin, best = quality, tmin, (scene, qa)
            if (
                tmin >= min_cross_angle
                and avg_cross >= 0.75 * spec.target_crossings[0]
                and avg_ratio >= spec.min_arc_ratio
                and avg_reversals >= spec.min_lateral_reversals
            ):
                break
        else:
            if tmin > best_tmin:
                best_tmin, best = tmin, (scene, qa)
            if tmin >= min_cross_angle:
                break
    scene, qa = best
    img = render(scene, qa, size, size)
    arcs = [c.arc for c in scene.curves]
    chords = [c.chord for c in scene.curves if c.chord > 1]
    tort = sum(c.arc / c.chord for c in scene.curves if c.chord > 1) / max(1, len(chords))
    # min crossing angle ON THE TRACED line(s) -- the well-posedness guarantee that
    # actually matters (distractor-distractor grazes don't make the question hard).
    tgt = qa["target_indices"]
    tmin = 180.0
    for i in tgt:
        if scene.curve_min_angle:
            tmin = min(tmin, scene.curve_min_angle[i])
    target_metrics = [scene.curve_metrics[i] for i in tgt] if scene.curve_metrics else []

    def mean_metric(name: str) -> float | None:
        if not scene.curve_metrics:
            return None
        return sum(m[name] for m in scene.curve_metrics) / len(scene.curve_metrics)

    def target_mean(name: str) -> float | None:
        if not target_metrics:
            return None
        return sum(m[name] for m in target_metrics) / len(target_metrics)

    def finite_mean(name: str, rows: list[dict]) -> float | None:
        values = [m[name] for m in rows if math.isfinite(m[name])]
        return sum(values) / len(values) if values else None

    target_spacing = target_mean("min_crossing_spacing_px")
    if target_spacing is not None and math.isinf(target_spacing):
        target_spacing = None
    meta = {
        "level": level, "qa_type": qa_type, "seed": seed, "curve_style": curve_style,
        "requested_curve_style": requested_curve_style,
        "geometry_salt": geometry_salt,
        "pathfinder_family": scene.pathfinder_family,
        "marker_layout": scene.marker_layout,
        "crossing_grid_cells": scene.crossing_grid_cells,
        "central_crossings": scene.central_crossings,
        "crossing_bbox_area_frac": round(scene.crossing_bbox_area_frac, 3),
        "question": qa["question"], "answer": qa["answer"],
        "pairs": qa["pairs"],
        "target_indices": qa["target_indices"],
        "dot_markers": qa.get("dot_markers"),
        "num_curves": len(scene.curves), "color_mode": scene.color_mode,
        "line_style": scene.line_style, "line_width": scene.line_width,
        "rope_outline": scene.rope_outline,
        "min_cross_angle": round(scene.min_angle, 1) if scene.min_angle else None,
        "target_min_angle": round(tmin, 1) if tmin < 180 else None,
        "total_crossings": scene.total_crossings,
        "mean_arc_px": round(sum(arcs) / len(arcs), 1),
        "mean_arc_over_chord": round(tort, 2),
        "min_chord_px": round(min(chords), 1),
        "mean_chord_px": round(sum(chords) / len(chords), 1),
        "long_chord_fraction": round(
            sum(chord >= 0.45 * math.hypot(size, size) for chord in chords)
            / len(chords),
            3,
        ),
        "endpoint_clean": scene.endpoint_clean,
        "curves": [
            {
                "start_label": c.si,
                "end_label": c.ei,
                "start_point": [round(c.start[0]), round(c.start[1])],
                "end_point": [round(c.end[0]), round(c.end[1])],
                "trace": [[round(p[0]), round(p[1])] for p in c.trace],
                "color": list(c.color),
                "length_px": round(c.arc, 1),
                "crossing_count": scene.crossing_counts[i],
            }
            for i, c in enumerate(scene.curves)
        ],
    }
    if scene.curve_metrics:
        meta.update({
            "mean_self_crossings": round(mean_metric("self_crossings"), 2),
            "mean_large_loops": round(mean_metric("large_loops"), 2),
            "mean_central_reentries": round(mean_metric("central_reentries"), 2),
            "mean_x_reversals": round(mean_metric("x_reversals"), 2),
            "mean_lateral_reversals": round(mean_metric("lateral_reversals"), 2),
            "mean_distinct_crossed_curves": round(mean_metric("distinct_crossed_curves"), 2),
            "mean_near_parallel_px": round(mean_metric("near_parallel_px"), 1),
            "mean_min_turn_radius_px": round(mean_metric("min_turn_radius_px"), 1),
            "mean_max_local_turn_deg": round(mean_metric("max_local_turn_deg"), 1),
            "mean_min_self_loop_span_px": (
                round(finite_mean("min_self_loop_span_px", scene.curve_metrics), 1)
                if finite_mean("min_self_loop_span_px", scene.curve_metrics) is not None else None
            ),
            "target_arc_px": round(target_mean("arc_px"), 1),
            "target_arc_over_chord": round(target_mean("arc_over_chord"), 2),
            "target_crossings": round(target_mean("crossings"), 2),
            "target_distinct_crossed_curves": round(target_mean("distinct_crossed_curves"), 2),
            "target_self_crossings": round(target_mean("self_crossings"), 2),
            "target_self_min_angle": round(
                min(m["self_min_angle"] for m in target_metrics), 1
            ),
            "target_large_loops": round(target_mean("large_loops"), 2),
            "target_central_reentries": round(target_mean("central_reentries"), 2),
            "target_x_reversals": round(target_mean("x_reversals"), 2),
            "target_lateral_reversals": round(target_mean("lateral_reversals"), 2),
            "target_near_parallel_px": round(target_mean("near_parallel_px"), 1),
            "target_min_turn_radius_px": round(
                min(m["min_turn_radius_px"] for m in target_metrics), 1
            ),
            "target_max_local_turn_deg": round(
                max(m["max_local_turn_deg"] for m in target_metrics), 1
            ),
            "target_min_self_loop_span_px": (
                round(finite_mean("min_self_loop_span_px", target_metrics), 1)
                if finite_mean("min_self_loop_span_px", target_metrics) is not None else None
            ),
            "target_min_crossing_spacing_px": (
                round(target_spacing, 1) if target_spacing is not None else None
            ),
        })
    return img, meta
