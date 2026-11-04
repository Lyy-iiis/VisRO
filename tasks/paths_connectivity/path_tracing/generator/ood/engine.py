"""Scene construction and rendering of the path-tracing out-of-distribution (xxhard) items."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Any
import unicodedata

from PIL import Image, ImageDraw
from shared.drawing.fonts import load_bundled_font
from visro.seeding import seed_text

from ..id import bezier_geometry, curve_engine, rope_engine


Point = tuple[float, float]
# default rng_namespace of the scene builders; generate_eval passes the release namespace
DEFAULT_RNG_NAMESPACE = seed_text(2506172965200998457699280819584379792338548274)
QA_TYPES = ("one_marker", "same_line", "ordered", "all_pairs")
CONDITIONS = (
    "stroke_ood",
    "program_harmonic_flow",
    "program_rounded_orthogonal",
    "endpoint_layout_ood",
    "background_ood",
)
ANGLE_FLOOR_DEG = 30.0
PREFERRED_SINGLE_TARGET_CROSSINGS = 4
CUSTOM_SCENE_ATTEMPTS = 180
CURVE_ATTEMPTS = 72
PREVIEW_NUM_CURVES = 7
ICON_CONTENT_POLICY = "pictorial_no_text_math_ui_or_marker_glyphs"
# Curve program names.  Each is part of its program's RNG namespace (changing one
# changes every item of that program), so they are fixed integer-coded strings.
HARMONIC_FLOW = seed_text(138745685221959975019114097650916357681)
ROUNDED_ORTHOGONAL = seed_text(167247503434987069323957277445710180014706376275505)
CATMULL_ROM_LAYOUT = seed_text(145245030622168653707417416854418837628561825691185)
CATMULL_ROM = seed_text(2216263284640024623221090955422972274391003697)

ICON_PROFILE_RULES = {
    "openmoji_clutter": {
        "style": "color",
        "allowed_sets": frozenset({"openmoji"}),
    },
    "silhouette_clutter": {
        "style": "silhouette",
        "allowed_sets": frozenset({"gameicons"}),
    },
}
# Exact normalized identities supplement the broad semantic-term filter below.
# These assets visibly contain label-like text/digits, UI/navigation notation,
# simple dot/shape markers, or line diagrams that can masquerade as task cues.
ICON_EXCLUDED_NORMALIZED_IDENTITIES = frozenset(
    {
        "gameicons/caro-asercion/slot-machine",
        "gameicons/caro-asercion/boarding-pass",
        "gameicons/delapouite/bullet-impacts",
        "gameicons/delapouite/dice-target",
        "gameicons/delapouite/dice-six-faces-six",
        "gameicons/delapouite/domino-tiles",
        "gameicons/delapouite/encrypted-channel",
        "gameicons/delapouite/horizontal-flip",
        "gameicons/delapouite/i-brick",
        "gameicons/delapouite/logic-gate-nand",
        "gameicons/delapouite/logic-gate-xor",
        "gameicons/delapouite/measure-tape",
        "gameicons/delapouite/network-bars",
        "gameicons/delapouite/organigram",
        "gameicons/delapouite/perspective-dice-four",
        "gameicons/delapouite/perspective-dice-two",
        "gameicons/delapouite/private-first-class",
        "gameicons/delapouite/round-knob",
        "gameicons/delapouite/settings-knobs",
        "gameicons/delapouite/rule-book",
        "gameicons/delapouite/tv-remote",
        "gameicons/delapouite/temporary-shield",
        "gameicons/delapouite/ursa-major",
        "gameicons/delapouite/z-brick",
        "gameicons/lorc/cross-mark",
        "gameicons/lorc/crossed-slashes",
        "gameicons/lorc/hospital-cross",
        "gameicons/lorc/heavy-timer",
        "gameicons/lorc/pokecog",
        "gameicons/lorc/rune-stone",
        "gameicons/lorc/wavy-itinerary",
        "gameicons/sbed/level-four",
        "gameicons/sbed/level-three-advanced",
        "gameicons/sbed/nodular",
        "gameicons/sbed/pounce",
        "gameicons/sbed/trigger-hurt",
        "gameicons/skoll/clubs",
        "gameicons/skoll/combination-lock",
        "gameicons/skoll/diamonds",
        "gameicons/skoll/halt",
        "gameicons/skoll/inverted-dice-3",
        "gameicons/skoll/rank-1",
        "gameicons/skoll/rank-2",
        "gameicons/viscious-speed/abstract-005",
        "gameicons/viscious-speed/abstract-023",
        "gameicons/viscious-speed/abstract-029",
        "gameicons/viscious-speed/abstract-037",
        "gameicons/viscious-speed/abstract-041",
        "gameicons/viscious-speed/abstract-053",
        "gameicons/viscious-speed/abstract-064",
        "gameicons/viscious-speed/abstract-067",
        "gameicons/viscious-speed/abstract-074",
        "gameicons/viscious-speed/abstract-084",
        "gameicons/viscious-speed/abstract-090",
        "gameicons/viscious-speed/abstract-111",
        "gameicons/viscious-speed/abstract-115",
        "gameicons/viscious-speed/abstract-117",
        "openmoji/1F3B0",
        "openmoji/1F3B5",
        "openmoji/1F3F3-200D-1F7E6-200D-1F30C",
        "openmoji/1F3F3-200D-1F7E9-200D-2B50-200D-1F7E9",
        "openmoji/1F3F4-200D-1F7E9",
        "openmoji/1F4CA",
        "openmoji/1F4CB",
        "openmoji/1F4E5",
        "openmoji/1F50A",
        "openmoji/1F518",
        "openmoji/1F531",
        "openmoji/1F534",
        "openmoji/1F553",
        "openmoji/1F556",
        "openmoji/1F563",
        "openmoji/1F579",
        "openmoji/1F5B2",
        "openmoji/1F6B3",
        "openmoji/1F7E9",
        "openmoji/1F7EA",
        "openmoji/1F7EB",
        "openmoji/1FA9F",
        "openmoji/23F0",
        "openmoji/25A1",
        "openmoji/25FD",
        "openmoji/264E",
        "openmoji/265E",
        "openmoji/2695",
        "openmoji/26AA",
        "openmoji/26AB",
        "openmoji/2721",
        "openmoji/2BC4",
    }
)
ICON_EXCLUDED_TERMS = frozenset(
    {
        "ab-testing",
        "abjad",
        "abugida",
        "alert",
        "alpha",
        "alphabet",
        "arabic",
        "arrow",
        "asterisk",
        "back",
        "braille",
        "calculator",
        "card",
        "caret",
        "chain",
        "chevron",
        "close",
        "collapse",
        "copyright",
        "currency",
        "cyrillic",
        "devanagari",
        "digit",
        "direction",
        "division",
        "dollar",
        "down",
        "download",
        "enter",
        "equal",
        "euro",
        "exclamation",
        "expand",
        "export",
        "format",
        "forward",
        "greater-than",
        "greek",
        "hangul",
        "hash",
        "header",
        "help",
        "ideograph",
        "ideogram",
        "import",
        "information",
        "integral",
        "keycap",
        "keyboard",
        "kana",
        "language",
        "latin",
        "left",
        "less-than",
        "letter",
        "hebrew",
        "link",
        "login",
        "logout",
        "mailbox-up",
        "map-marker",
        "math",
        "menu",
        "minus",
        "motorway",
        "move",
        "multiplication",
        "navigation",
        "next",
        "number",
        "numeric",
        "percent",
        "pause",
        "play",
        "playback",
        "plus",
        "pointing",
        "pound",
        "previous",
        "private-use",
        "progress",
        "punctuation",
        "question-mark",
        "redo",
        "refresh",
        "registered",
        "repeat",
        "reply",
        "rewind",
        "right",
        "road",
        "roman-numeral",
        "route",
        "ruler",
        "rupee",
        "send",
        "share",
        "sign",
        "skip",
        "sort",
        "square-root",
        "squared",
        "symbol",
        "swap",
        "sync",
        "syllabary",
        "tab",
        "tarot",
        "text",
        "stop",
        "trademark",
        "transfer",
        "translate",
        "trending",
        "undo",
        "unicode",
        "unfold",
        "unknown-codepoint",
        "upwards",
        "upload",
        "yen",
    }
)


@dataclass
class BuiltCase:
    scene: rope_engine.Scene
    qa: dict[str, Any]
    provenance: dict[str, Any]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _make_qa(
    scene: rope_engine.Scene,
    qa_type: str,
    rng: random.Random,
) -> dict[str, Any]:
    if qa_type == "same_line":
        return curve_engine.make_same_line_qa(
            scene,
            rng,
            PREFERRED_SINGLE_TARGET_CROSSINGS,
        )
    return rope_engine.make_qa(
        scene,
        qa_type,
        rng,
        min_target_crossings=PREFERRED_SINGLE_TARGET_CROSSINGS,
    )


def _qa_matches_constraints(
    scene: rope_engine.Scene,
    qa: dict[str, Any],
    constraints: dict[str, Any],
) -> bool:
    """Return whether a sampled QA obeys release-level balance constraints."""

    supported = {
        "same_line_answer",
        "one_marker_query_side",
        "ordered_target_count",
    }
    unknown = set(constraints) - supported
    if unknown:
        raise ValueError(f"unknown QA constraints: {sorted(unknown)}")

    if "same_line_answer" in constraints:
        if qa["type"] != "same_line":
            raise ValueError("same_line_answer requires a same_line QA")
        if qa["answer"] != constraints["same_line_answer"]:
            return False

    if "one_marker_query_side" in constraints:
        if qa["type"] != "one_marker":
            raise ValueError("one_marker_query_side requires a one_marker QA")
        curve = scene.curves[qa["target_indices"][0]]
        if qa["answer"] == curve.ei:
            realized_side = "start"
        elif qa["answer"] == curve.si:
            realized_side = "end"
        else:
            raise RuntimeError("one_marker answer does not match its target curve")
        if realized_side != constraints["one_marker_query_side"]:
            return False

    if "ordered_target_count" in constraints:
        if qa["type"] != "ordered":
            raise ValueError("ordered_target_count requires an ordered QA")
        if len(qa["target_indices"]) != constraints["ordered_target_count"]:
            return False

    return True


def _make_constrained_qa(
    scene: rope_engine.Scene,
    qa_type: str,
    rng_namespace: str,
    constraints: dict[str, Any] | None,
    *,
    attempts: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample QA-only randomness until deterministic quotas are satisfied.

    With no constraints the QA is drawn from rng_namespace directly.
    Constraints never alter geometry or flip a sampled label after the fact.
    """

    if not constraints:
        return _make_qa(scene, qa_type, random.Random(rng_namespace)), {
            "qa_attempt_index": 0,
            "qa_attempts": 1,
            "qa_constraints": {},
            "qa_rng_namespace": rng_namespace,
        }

    last_error: RuntimeError | None = None
    for attempt in range(attempts):
        attempt_namespace = f"{rng_namespace}:constraint-attempt:{attempt}"
        try:
            qa = _make_qa(scene, qa_type, random.Random(attempt_namespace))
        except RuntimeError as error:
            last_error = error
            continue
        if _qa_matches_constraints(scene, qa, constraints):
            return qa, {
                "qa_attempt_index": attempt,
                "qa_attempts": attempt + 1,
                "qa_constraints": dict(constraints),
                "qa_rng_namespace": attempt_namespace,
            }
    detail = f"; last error: {last_error}" if last_error is not None else ""
    raise RuntimeError(
        f"could not satisfy QA constraints {constraints} after {attempts} "
        f"attempts{detail}"
    )


def build_current_xhard_case(
    qa_type: str,
    seed: int,
    size: int,
    *,
    geometry_retries: int = 12,
    rng_namespace: str = DEFAULT_RNG_NAMESPACE,
    qa_constraints: dict[str, Any] | None = None,
) -> BuiltCase:
    """Build a live current Bezier xhard case for render-only OOD variants."""

    last_error: Exception | None = None
    curve_count_rejected_salts: list[int] = []
    for geometry_salt in range(geometry_retries):
        try:
            scene, scene_attempts, validity_policy = curve_engine.build_bezier_scene(
                "xhard",
                qa_type,
                seed,
                size,
                ANGLE_FLOOR_DEG,
                3,
                geometry_salt,
                "uniform",
                None,
                None,
            )
            if len(scene.curves) != PREVIEW_NUM_CURVES:
                curve_count_rejected_salts.append(geometry_salt)
                last_error = RuntimeError(
                    f"geometry salt {geometry_salt} produced "
                    f"{len(scene.curves)} curves; required {PREVIEW_NUM_CURVES}"
                )
                continue
            qa, qa_provenance = _make_constrained_qa(
                scene,
                qa_type,
                f"{rng_namespace}:{seed}:{qa_type}:qa:{geometry_salt}",
                qa_constraints,
            )
            return BuiltCase(
                scene=scene,
                qa=qa,
                provenance={
                    "base_generator_version": "path-tracing",
                    "geometry_backend": "bezier_geometry",
                    "program_family": CATMULL_ROM,
                    "geometry_salt": geometry_salt,
                    "curve_count_policy": {
                        "required_num_curves": PREVIEW_NUM_CURVES,
                        "rejected_geometry_salts": curve_count_rejected_salts,
                    },
                    "scene_attempts": scene_attempts,
                    "validity_policy": validity_policy,
                    **qa_provenance,
                },
            )
        except (RuntimeError, ValueError) as error:
            last_error = error
    raise RuntimeError(
        f"could not build current xhard case for {qa_type} after "
        f"{geometry_retries} salts: {last_error}"
    )


def _inner_ring_markers(
    n: int,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    count = 2 * n
    center = (width / 2.0, height / 2.0)
    # A visibly inset circle, but still large enough for the unchanged current
    # long-chord admissibility constraints to retain >=100 possible matchings.
    radius = 0.37 * min(width, height)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    points = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count + rng.uniform(-0.025, 0.025)
        local_radius = radius + rng.uniform(-4.0, 4.0)
        points.append(
            (
                center[0] + local_radius * math.cos(angle),
                center[1] + local_radius * math.sin(angle),
            )
        )
    rng.shuffle(points)
    return points[:n], points[n:]


def _four_island_markers(
    n: int,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    centers = [
        (0.22 * width, 0.22 * height),
        (0.78 * width, 0.22 * height),
        (0.78 * width, 0.78 * height),
        (0.22 * width, 0.78 * height),
    ]
    rng.shuffle(centers)
    counts = [4, 4, 3, 3]
    groups: list[list[Point]] = []
    for center, count in zip(centers, counts):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        local = []
        for index in range(count):
            angle = phase + 2.0 * math.pi * index / count
            radius = rng.uniform(45.0, 50.0)
            local.append(
                (
                    center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle),
                )
            )
        groups.append(local)

    # Every island contains both marker roles.  Which island gets the 2/1 role
    # split is randomized, so label role is not tied to a fixed quadrant.
    start_counts = [2, 2, 2, 1]
    rng.shuffle(start_counts)
    starts: list[Point] = []
    ends: list[Point] = []
    for points, start_count in zip(groups, start_counts):
        rng.shuffle(points)
        starts.extend(points[:start_count])
        ends.extend(points[start_count:])
    if len(starts) != n or len(ends) != n:
        raise AssertionError("four-island role allocation is unbalanced")
    rng.shuffle(starts)
    rng.shuffle(ends)
    return starts, ends


def _marker_layout(
    layout: str,
    n: int,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    if layout == "perimeter_arclength":
        points = bezier_geometry.random_marker_points(2 * n, width, height, rng)
        return points[:n], points[n:]
    if layout == "inner_ring":
        return _inner_ring_markers(n, width, height, rng)
    if layout == "four_islands":
        return _four_island_markers(n, width, height, rng)
    raise ValueError(f"unknown OOD marker layout: {layout}")


def _fit_offsets_to_canvas(
    bases: list[Point],
    offsets: list[Point],
    width: int,
    height: int,
    margin: float,
) -> float:
    scale = 1.0
    for (base_x, base_y), (offset_x, offset_y) in zip(bases, offsets):
        if offset_x > 0:
            scale = min(scale, (width - margin - base_x) / offset_x)
        elif offset_x < 0:
            scale = min(scale, (margin - base_x) / offset_x)
        if offset_y > 0:
            scale = min(scale, (height - margin - base_y) / offset_y)
        elif offset_y < 0:
            scale = min(scale, (margin - base_y) / offset_y)
    return max(0.0, min(1.0, 0.94 * scale))


def _harmonic_flow_curve(
    start: Point,
    end: Point,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    """Endpoint-constrained Fourier flow, with no Bezier/Catmull construction."""

    chord = max(1.0, rope_engine.distance(start, end))
    ux = (end[0] - start[0]) / chord
    uy = (end[1] - start[1]) / chord
    vx, vy = -uy, ux
    frequencies = rng.sample([2, 3, 4, 5], 3)
    coefficients = [rng.uniform(0.45, 1.0) * rng.choice((-1.0, 1.0))]
    coefficients.extend(
        rng.uniform(0.16, 0.42) * rng.choice((-1.0, 1.0))
        for _ in range(2)
    )
    amplitude = chord * rng.uniform(0.19, 0.31)
    along_amplitude = chord * rng.uniform(0.025, 0.055)
    bases: list[Point] = []
    offsets: list[Point] = []
    samples = 360
    coefficient_norm = max(1.0, sum(abs(value) for value in coefficients))
    for index in range(samples + 1):
        t = index / samples
        base = (
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]),
        )
        transverse = sum(
            coefficient * math.sin(frequency * math.pi * t)
            for coefficient, frequency in zip(coefficients, frequencies)
        ) / coefficient_norm
        along = math.sin((frequencies[0] + 1) * math.pi * t)
        bases.append(base)
        offsets.append(
            (
                vx * amplitude * transverse + ux * along_amplitude * along,
                vy * amplitude * transverse + uy * along_amplitude * along,
            )
        )
    scale = _fit_offsets_to_canvas(bases, offsets, width, height, margin=34.0)
    dense = [
        (base[0] + scale * offset[0], base[1] + scale * offset[1])
        for base, offset in zip(bases, offsets)
    ]
    dense[0] = start
    dense[-1] = end
    return [], dense


def _nearest_edge_inward(point: Point, width: int, height: int, amount: float) -> Point:
    x, y = point
    _, direction = min(
        (
            (x, (1.0, 0.0)),
            (width - x, (-1.0, 0.0)),
            (y, (0.0, 1.0)),
            (height - y, (0.0, -1.0)),
        ),
        key=lambda item: item[0],
    )
    return (x + amount * direction[0], y + amount * direction[1])


def _dedupe_waypoints(points: list[Point]) -> list[Point]:
    out = [points[0]]
    for point in points[1:]:
        if rope_engine.distance(point, out[-1]) >= 3.0:
            out.append(point)
    return out


def _rounded_polyline(points: list[Point], radius: float) -> list[Point]:
    """Round a routed polyline's corners while retaining long straight runs."""

    points = _dedupe_waypoints(points)
    if len(points) < 3:
        return points
    dense: list[Point] = [points[0]]
    for index in range(1, len(points) - 1):
        previous, corner, following = points[index - 1 : index + 2]
        incoming = rope_engine.distance(previous, corner)
        outgoing = rope_engine.distance(corner, following)
        if incoming < 2.0 or outgoing < 2.0:
            continue
        trim = min(radius, 0.32 * incoming, 0.32 * outgoing)
        entry = (
            corner[0] + (previous[0] - corner[0]) * trim / incoming,
            corner[1] + (previous[1] - corner[1]) * trim / incoming,
        )
        exit_point = (
            corner[0] + (following[0] - corner[0]) * trim / outgoing,
            corner[1] + (following[1] - corner[1]) * trim / outgoing,
        )
        dense.append(entry)
        for step in range(1, 11):
            t = step / 10.0
            mt = 1.0 - t
            dense.append(
                (
                    mt * mt * entry[0]
                    + 2.0 * mt * t * corner[0]
                    + t * t * exit_point[0],
                    mt * mt * entry[1]
                    + 2.0 * mt * t * corner[1]
                    + t * t * exit_point[1],
                )
            )
    dense.append(points[-1])
    return rope_engine.resample_polyline(dense, step=4.0)


def _rounded_orthogonal_curve(
    start: Point,
    end: Point,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    """Route through one rectilinear lane, then round every corner."""

    lead = rng.uniform(30.0, 42.0)
    start_in = _nearest_edge_inward(start, width, height, lead)
    end_in = _nearest_edge_inward(end, width, height, lead)
    if rng.random() < 0.5:
        lane = rng.uniform(82.0, width - 82.0)
        raw = [
            start,
            start_in,
            (lane, start_in[1]),
            (lane, end_in[1]),
            end_in,
            end,
        ]
    else:
        lane = rng.uniform(82.0, height - 82.0)
        raw = [
            start,
            start_in,
            (start_in[0], lane),
            (end_in[0], lane),
            end_in,
            end,
        ]
    dense = _rounded_polyline(raw, radius=rng.uniform(15.0, 23.0))
    return raw, dense


def _bezier_curve(
    start: Point,
    end: Point,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    return bezier_geometry.build_curve(
        start,
        end,
        width,
        height,
        "complex",
        rng,
        length_scale=1.4,
        max_knots=4,
    )


def _layout_catmull_curve(
    start: Point,
    end: Point,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    """The Bezier Catmull grammar with explicit inward endpoint leads.

    Interior endpoint layouts need a short clean departure from their local
    marker island; the perimeter constructor gets that clearance from the canvas
    edge.
    """

    center = (width / 2.0, height / 2.0)

    def toward_center(point: Point, amount: float) -> Point:
        dx, dy = center[0] - point[0], center[1] - point[1]
        length = max(1.0, math.hypot(dx, dy))
        return (
            point[0] + amount * dx / length,
            point[1] + amount * dy / length,
        )

    lead_start = toward_center(start, rng.uniform(32.0, 42.0))
    lead_end = toward_center(end, rng.uniform(32.0, 42.0))
    anchors = [
        start,
        lead_start,
        *bezier_geometry.interior_knots(
            lead_start,
            lead_end,
            width,
            height,
            "complex",
            rng,
            radius_scale=rng.uniform(0.8, 1.2),
            length_scale=1.25,
            max_knots=3,
        ),
        lead_end,
        end,
    ]
    control_margin = bezier_geometry.layout_params(width, height)["ctrl_margin"]
    controls: list[Point] = []
    dense: list[Point] = []
    segments = bezier_geometry.catmull_rom_controls(anchors, rng.uniform(0.82, 1.05))
    for segment_index, segment in enumerate(segments):
        segment = [
            segment[0],
            bezier_geometry.clamp_point(segment[1], width, height, control_margin),
            bezier_geometry.clamp_point(segment[2], width, height, control_margin),
            segment[3],
        ]
        controls.extend(segment if segment_index == 0 else segment[1:])
        sampled = bezier_geometry.sample_bezier(segment, 80)
        dense.extend(sampled if segment_index == 0 else sampled[1:])
    return controls, dense


def _curve_candidate(
    program_family: str,
    start: Point,
    end: Point,
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[Point], list[Point]]:
    if program_family == HARMONIC_FLOW:
        return _harmonic_flow_curve(start, end, width, height, rng)
    if program_family == ROUNDED_ORTHOGONAL:
        return _rounded_orthogonal_curve(start, end, width, height, rng)
    if program_family == CATMULL_ROM:
        return _bezier_curve(start, end, width, height, rng)
    if program_family == CATMULL_ROM_LAYOUT:
        return _layout_catmull_curve(start, end, width, height, rng)
    raise ValueError(f"unknown curve program: {program_family}")


def _program_gates(
    program_family: str,
    size: int,
) -> dict[str, float | int]:
    if program_family == ROUNDED_ORTHOGONAL:
        return {
            "min_turn_radius_px": 4.5,
            "max_local_turn_deg": 72.0,
            "max_self_crossings": 3,
            "minimum_scene_crossings": 8,
            # Keep the foreign lane-routing grammar without allowing shared
            # lanes to turn into an ambiguous double-line cue.
            "maximum_mean_near_parallel_px": round(0.20 * size, 3),
            "maximum_curve_near_parallel_px": round(0.36 * size, 3),
            "minimum_crossing_spacing_px": round(0.045 * size, 3),
        }
    if program_family == HARMONIC_FLOW:
        return {
            "min_turn_radius_px": 5.5,
            "max_local_turn_deg": 62.0,
            "max_self_crossings": 3,
            "minimum_scene_crossings": 10,
        }
    return {
        "min_turn_radius_px": 7.0,
        "max_local_turn_deg": 55.0,
        "max_self_crossings": 3,
        "minimum_scene_crossings": 9,
    }


def build_custom_case(
    qa_type: str,
    seed: int,
    size: int,
    *,
    program_family: str,
    marker_layout: str,
    rng_namespace: str = DEFAULT_RNG_NAMESPACE,
    qa_constraints: dict[str, Any] | None = None,
) -> BuiltCase:
    """Build a truthful custom-program or custom-layout OOD scene."""

    n_curves = PREVIEW_NUM_CURVES
    diagonal = math.hypot(size, size)
    gates = _program_gates(program_family, size)
    marker_radius = bezier_geometry.layout_params(size, size)["marker_r"]
    endpoint_clearance = marker_radius + 10.0
    last_reason = "no attempts made"

    for scene_attempt in range(CUSTOM_SCENE_ATTEMPTS):
        namespace = (
            f"{rng_namespace}:{seed}:{qa_type}:{program_family}:"
            f"{marker_layout}:scene{scene_attempt}"
        )
        geometry_rng = random.Random(f"{namespace}:geometry")
        pairing_rng = random.Random(f"{namespace}:pairing")
        starts, end_positions = _marker_layout(
            marker_layout,
            n_curves,
            size,
            size,
            geometry_rng,
        )
        end_indices, pairing_policy = (
            curve_engine.uniform_admissible_endpoint_permutation(
                starts,
                end_positions,
                size,
                size,
                pairing_rng,
            )
        )
        pairing_policy["rng_namespace"] = f"{namespace}:pairing"
        pairing_policy["rng_isolated_from_geometry"] = True
        if end_indices is None:
            last_reason = "endpoint pairing policy rejected layout"
            continue

        ends = [end_positions[index] for index in end_indices]
        all_markers = starts + end_positions
        raw_curves: list[tuple[Point, Point, list[Point], list[Point]]] = []
        curve_attempt_counts: list[int] = []
        failed_curve = False
        for curve_index, (start, end) in enumerate(zip(starts, ends)):
            accepted = None
            for curve_attempt in range(CURVE_ATTEMPTS):
                curve_rng = random.Random(
                    f"{namespace}:curve{curve_index}:attempt{curve_attempt}"
                )
                controls, dense = _curve_candidate(
                    program_family,
                    start,
                    end,
                    size,
                    size,
                    curve_rng,
                )
                if len(dense) < 8:
                    continue
                if not bezier_geometry.endpoint_safe(
                    dense,
                    all_markers,
                    {start, end},
                    endpoint_clearance,
                ):
                    continue
                metrics = rope_engine.basic_rope_metrics(dense, size, size)
                if metrics["min_turn_radius_px"] < gates["min_turn_radius_px"]:
                    continue
                if metrics["max_local_turn_deg"] > gates["max_local_turn_deg"]:
                    continue
                if metrics["self_crossings"] > gates["max_self_crossings"]:
                    continue
                if metrics["arc_px"] < 1.04 * rope_engine.distance(start, end):
                    continue
                accepted = (start, end, controls, dense)
                curve_attempt_counts.append(curve_attempt + 1)
                break
            if accepted is None:
                last_reason = f"curve {curve_index} exhausted proposal budget"
                failed_curve = True
                break
            raw_curves.append(accepted)
        if failed_curve:
            continue

        dense_list = [row[3] for row in raw_curves]
        (
            crosses,
            crossing_counts,
            curve_min_angle,
            pair_angle,
            curve_metrics,
            total_crossings,
            global_min_angle,
        ) = rope_engine.analyze_rope_scene(dense_list, size, size)
        if global_min_angle < ANGLE_FLOOR_DEG:
            last_reason = f"shallow crossing: {global_min_angle:.2f}"
            continue
        if total_crossings < int(gates["minimum_scene_crossings"]):
            last_reason = f"too few crossings: {total_crossings}"
            continue
        if total_crossings > 42:
            last_reason = f"too many crossings: {total_crossings}"
            continue
        near_parallel = [
            float(metrics["near_parallel_px"]) for metrics in curve_metrics
        ]
        mean_near_parallel = sum(near_parallel) / len(near_parallel)
        max_near_parallel = max(near_parallel)
        finite_crossing_spacings = [
            float(metrics["min_crossing_spacing_px"])
            for metrics in curve_metrics
            if math.isfinite(float(metrics["min_crossing_spacing_px"]))
        ]
        min_crossing_spacing = min(finite_crossing_spacings, default=math.inf)
        if mean_near_parallel > float(
            gates.get("maximum_mean_near_parallel_px", math.inf)
        ):
            last_reason = (
                "too much mean near-parallel routing: "
                f"{mean_near_parallel:.2f}px"
            )
            continue
        if max_near_parallel > float(
            gates.get("maximum_curve_near_parallel_px", math.inf)
        ):
            last_reason = (
                "one curve has too much near-parallel routing: "
                f"{max_near_parallel:.2f}px"
            )
            continue
        if min_crossing_spacing < float(
            gates.get("minimum_crossing_spacing_px", -math.inf)
        ):
            last_reason = (
                "crossings are too tightly clustered on one curve: "
                f"{min_crossing_spacing:.2f}px"
            )
            continue

        appearance_rng = random.Random(f"{namespace}:appearance")
        color_mode = "uniform"
        colors = rope_engine.curve_colors(n_curves, color_mode, appearance_rng)
        representative = tuple(
            round(sum(color[channel] for color in colors) / len(colors))
            for channel in range(3)
        )
        outline = rope_engine.contrasting_outline(representative, appearance_rng)
        curves = []
        for index, (start, end, controls, dense) in enumerate(raw_curves):
            curves.append(
                rope_engine.Curve(
                    si=rope_engine.START_LABELS[index],
                    ei=rope_engine.END_LABELS[end_indices[index]],
                    start=start,
                    end=end,
                    controls=controls,
                    dense=dense,
                    trace=rope_engine.trace_points(dense),
                    color=colors[index],
                    outline_color=outline,
                    arc=curve_metrics[index]["arc_px"],
                    chord=rope_engine.distance(start, end),
                )
            )
        spatial = rope_engine.crossing_distribution(dense_list, size, size)
        scene = rope_engine.Scene(
            curves=curves,
            crosses=crosses,
            crossing_counts=crossing_counts,
            min_angle=global_min_angle if global_min_angle < 180.0 else None,
            total_crossings=total_crossings,
            endpoint_clean=True,
            color_mode=color_mode,
            line_width=3,
            line_style="dashed" if seed % 2 else "solid",
            rope_outline=bool(seed % 5 == 0),
            curve_min_angle=curve_min_angle,
            pair_angle=pair_angle,
            angle_floor=ANGLE_FLOOR_DEG,
            curve_style=program_family,
            pathfinder_family=program_family,
            marker_layout=marker_layout,
            curve_metrics=curve_metrics,
            crossing_grid_cells=spatial["crossing_grid_cells"],
            central_crossings=spatial["central_crossings"],
            crossing_bbox_area_frac=spatial["crossing_bbox_area_frac"],
        )
        try:
            qa, qa_provenance = _make_constrained_qa(
                scene,
                qa_type,
                f"{namespace}:qa",
                qa_constraints,
            )
        except RuntimeError as error:
            last_reason = f"QA construction failed: {error}"
            continue
        return BuiltCase(
            scene=scene,
            qa=qa,
            provenance={
                "base_generator_version": "path-tracing",
                "geometry_backend": f"ood_{program_family}",
                "program_family": program_family,
                "marker_layout": marker_layout,
                "scene_attempt_index": scene_attempt,
                "curve_attempt_counts": curve_attempt_counts,
                "endpoint_pairing_policy": pairing_policy,
                **qa_provenance,
                "validity_policy": {
                    "global_crossing_angle_floor_deg": ANGLE_FLOOR_DEG,
                    "minimum_scene_crossings": gates["minimum_scene_crossings"],
                    "maximum_scene_crossings": 42,
                    "endpoint_clearance_px": endpoint_clearance,
                    "minimum_chord_fraction": round(
                        min(curve.chord for curve in curves) / diagonal,
                        4,
                    ),
                    "realized_mean_near_parallel_px": round(
                        mean_near_parallel,
                        2,
                    ),
                    "realized_max_near_parallel_px": round(
                        max_near_parallel,
                        2,
                    ),
                    "realized_min_crossing_spacing_px": (
                        round(min_crossing_spacing, 2)
                        if math.isfinite(min_crossing_spacing)
                        else None
                    ),
                    **gates,
                },
            },
        )

    raise RuntimeError(
        f"could not build {program_family}/{marker_layout}/{qa_type} after "
        f"{CUSTOM_SCENE_ATTEMPTS} attempts; last rejection: {last_reason}"
    )


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def _icon_semantic_text(entry: dict[str, Any]) -> str:
    """Expose Unicode semantics hidden behind OpenMoji code-point IDs."""

    parts = [str(entry.get("name", "")), str(entry.get("id", ""))]
    if entry.get("set") == "openmoji":
        encoded = str(entry.get("id", "")).rsplit("/", 1)[-1]
        for token in encoded.split("-"):
            try:
                character = chr(int(token, 16))
                if unicodedata.category(character) == "Co":
                    parts.append("private-use")
                unicode_name = unicodedata.name(character, "")
                parts.append(unicode_name or "unknown-codepoint")
            except (ValueError, OverflowError):
                continue
    return " ".join(parts).lower()


def _icon_normalized_identity(entry: dict[str, Any]) -> str:
    """Collapse cosmetic OpenMoji tone/variation variants for placement."""

    icon_id = str(entry.get("id", ""))
    if entry.get("set") != "openmoji":
        return icon_id
    encoded = icon_id.rsplit("/", 1)[-1]
    normalized: list[str] = []
    for token in encoded.split("-"):
        try:
            codepoint = int(token, 16)
        except ValueError:
            normalized.append(token.upper())
            continue
        if 0x1F3FB <= codepoint <= 0x1F3FF:
            continue
        if codepoint in {0xFE0E, 0xFE0F}:
            continue
        normalized.append(f"{codepoint:X}")
    return "openmoji/" + "-".join(normalized)


def _icon_is_allowed(entry: dict[str, Any]) -> bool:
    semantic_text = _icon_semantic_text(entry)
    allowed_sets = {
        icon_set
        for rule in ICON_PROFILE_RULES.values()
        for icon_set in rule["allowed_sets"]
    }
    return bool(
        entry.get("set") in allowed_sets
        and _icon_normalized_identity(entry)
        not in ICON_EXCLUDED_NORMALIZED_IDENTITIES
        and not any(term in semantic_text for term in ICON_EXCLUDED_TERMS)
    )


def _normalized_icon_identities(
    entries: list[dict[str, Any]],
    split: str,
) -> set[str]:
    """Return every normalized identity in a manifest split, policy-independent."""

    return {
        _icon_normalized_identity(entry)
        for entry in entries
        if entry.get("split", "train") == split
    }


def _eligible_icon_entries(
    entries: list[dict[str, Any]],
    profile: str,
) -> list[dict[str, Any]]:
    """Return the complete, profile-specific held-out candidate pool."""

    if profile not in ICON_PROFILE_RULES:
        raise ValueError(f"unknown icon clutter profile: {profile}")
    rule = ICON_PROFILE_RULES[profile]
    train_identities = _normalized_icon_identities(entries, "train")
    return [
        entry
        for entry in entries
        if entry.get("split", "train") == "test"
        and entry.get("style") == rule["style"]
        and entry.get("set") in rule["allowed_sets"]
        and _icon_is_allowed(entry)
        and _icon_normalized_identity(entry) not in train_identities
    ]


def build_icon_clutter_layer(
    profile: str,
    seed: int,
    scene: rope_engine.Scene,
    qa: dict[str, Any],
    size: int,
    *,
    supersample: int = 3,
    rng_namespace: str = DEFAULT_RNG_NAMESPACE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Build one deterministic held-out-icon layer."""

    try:
        from shared.iconlib import IconLibrary, MANIFEST
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "icon backgrounds require the repository's vision environment "
            "with CairoSVG; run with `conda run -n vision python ...`"
        ) from error

    if profile not in ICON_PROFILE_RULES:
        raise ValueError(f"unknown icon clutter profile: {profile}")
    mode = ICON_PROFILE_RULES[profile]["style"]
    requested = 12 if mode == "color" else 14
    opacity = 0.28 if mode == "color" else 0.24
    background_rng_namespace = f"{rng_namespace}:background:{seed}:{profile}"
    rng = random.Random(background_rng_namespace)
    library = IconLibrary(default_split="test")
    train_normalized_identities = _normalized_icon_identities(library.icons, "train")
    eligible_entries = _eligible_icon_entries(library.icons, profile)
    eligible_identities = {
        _icon_normalized_identity(entry) for entry in eligible_entries
    }
    if len(eligible_identities) < requested:
        raise RuntimeError(
            f"only {len(eligible_identities)} eligible icon identities for {profile}"
        )
    layer = Image.new("RGBA", (size * supersample, size * supersample), (0, 0, 0, 0))
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_identities: set[str] = set()
    metadata: list[dict[str, Any]] = []
    marker_radius = bezier_geometry.layout_params(size, size)["marker_r"] + 10
    protected = [(curve.start[0], curve.start[1], marker_radius) for curve in scene.curves]
    protected.extend(
        (curve.end[0], curve.end[1], marker_radius) for curve in scene.curves
    )
    for dot in qa.get("dot_markers") or []:
        protected.append((dot["point"][0], dot["point"][1], marker_radius * 0.65))

    for _placement in range(1200):
        if len(metadata) >= requested:
            break
        entry = eligible_entries[rng.randrange(len(eligible_entries))]
        normalized_identity = _icon_normalized_identity(entry)
        if normalized_identity in placed_identities:
            continue
        tile_px = rng.randint(34, 56)
        tint = None
        if mode == "silhouette":
            shade = rng.randint(112, 172)
            tint = (shade, min(184, shade + rng.randint(-8, 12)), shade)
        tile = library.render(entry, S=tile_px * supersample, color=tint)
        alpha_box = tile.getbbox()
        if alpha_box is None:
            continue
        tile = tile.crop(alpha_box)
        rotation = rng.uniform(-15.0, 15.0)
        tile = tile.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        alpha = tile.getchannel("A").point(lambda value: round(value * opacity))
        tile.putalpha(alpha)
        width_px = max(1, round(tile.width / supersample))
        height_px = max(1, round(tile.height / supersample))
        if width_px >= size - 12 or height_px >= size - 12:
            continue
        x = rng.randint(6, size - width_px - 6)
        y = rng.randint(6, size - height_px - 6)
        box = (x, y, x + width_px, y + height_px)
        if any(_bbox_iou(box, other) > 0.10 for other in placed_boxes):
            continue
        if any(
            box[0] - radius <= px <= box[2] + radius
            and box[1] - radius <= py <= box[3] + radius
            for px, py, radius in protected
        ):
            continue
        layer.alpha_composite(tile, (x * supersample, y * supersample))
        asset_path = library.root / entry["path"]
        asset_sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        placed_boxes.append(box)
        placed_identities.add(normalized_identity)
        metadata.append(
            {
                "id": entry["id"],
                "name": entry.get("name", ""),
                "normalized_identity": normalized_identity,
                "set": entry["set"],
                "style": entry["style"],
                "split": entry.get("split", "test"),
                "asset_path": entry["path"],
                "asset_sha256": f"sha256:{asset_sha}",
                "bbox_xywh": [x, y, width_px, height_px],
                "rotation_deg": round(rotation, 2),
                "opacity": opacity,
            }
        )
    if len(metadata) != requested:
        raise RuntimeError(
            f"could place only {len(metadata)}/{requested} clutter icons for {profile}"
        )
    selected_train_overlap = placed_identities.intersection(
        train_normalized_identities
    )
    if selected_train_overlap:
        raise RuntimeError(
            "selected icon identities overlap the normalized train split: "
            f"{sorted(selected_train_overlap)}"
        )
    alpha_pixels = sum(value > 8 for value in layer.getchannel("A").getdata())
    manifest_sha = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
    return layer, {
        "profile": profile,
        "rng_namespace": background_rng_namespace,
        "icon_partition": "test",
        "content_policy": ICON_CONTENT_POLICY,
        "allowed_icon_sets": sorted(ICON_PROFILE_RULES[profile]["allowed_sets"]),
        "candidate_asset_count": len(eligible_entries),
        "candidate_normalized_identity_count": len(eligible_identities),
        "exact_denied_normalized_identity_count": len(
            ICON_EXCLUDED_NORMALIZED_IDENTITIES
        ),
        "semantic_identity_disjoint_from_icon_train": True,
        "selected_train_normalized_identity_overlap_count": 0,
        "excluded_train_normalized_identity_count": len(
            train_normalized_identities
        ),
        "manifest_sha256": f"sha256:{manifest_sha}",
        "requested_count": requested,
        "realized_count": len(metadata),
        "foreground_occupancy": round(alpha_pixels / (layer.width * layer.height), 5),
        "layer_order": "background<grid<clutter<paths<markers",
        "icons": metadata,
    }


def _draw_paddles(
    draw: ImageDraw.ImageDraw,
    points: list[Point],
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    width: int,
    step: float,
) -> None:
    anchors = rope_engine.resample_polyline(points, step=step)
    if len(anchors) < 2:
        return
    # A thin continuous guide preserves connectivity while the larger disks
    # make this visibly different from the ordinary solid/dashed renderer.
    draw.line(points, fill=fill, width=max(1, round(width * 0.34)))
    radius = max(2, round(width * 0.85))
    for point in anchors:
        draw.ellipse(
            (
                point[0] - radius,
                point[1] - radius,
                point[0] + radius,
                point[1] + radius,
            ),
            fill=fill,
        )


def render_case(
    scene: rope_engine.Scene,
    qa: dict[str, Any],
    size: int,
    *,
    stroke_profile: str = "current",
    clutter_layer: Image.Image | None = None,
) -> Image.Image:
    """Render an OOD variant without post-processing a completed Path raster."""

    supersample = 3
    width = height = size
    layout0 = rope_engine.layout_params(width, height)
    layout = {key: value * supersample for key, value in layout0.items()}
    raster_width = width * supersample
    raster_height = height * supersample
    image = Image.new("RGBA", (raster_width, raster_height), (252, 252, 248, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, raster_width, layout["grid"]):
        draw.line((x, 0, x, raster_height), fill=(238, 238, 232, 255), width=supersample)
    for y in range(0, raster_height, layout["grid"]):
        draw.line((0, y, raster_width, y), fill=(238, 238, 232, 255), width=supersample)
    if clutter_layer is not None:
        if clutter_layer.size != image.size:
            raise ValueError("clutter layer size does not match supersampled canvas")
        image.alpha_composite(clutter_layer)
        draw = ImageDraw.Draw(image)

    def point(value: Point) -> Point:
        return (round(value[0] * supersample), round(value[1] * supersample))

    line_width = scene.line_width * supersample
    dash = max(8, 4 * line_width)
    gap = max(6, 3 * line_width)
    scaled_curves = [[point(value) for value in curve.dense] for curve in scene.curves]

    if stroke_profile == "halo_core":
        halo_width = max(11 * supersample, line_width + 6 * supersample)
        halo_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo_layer)
        for points in scaled_curves:
            halo_draw.line(
                points,
                fill=(255, 255, 255, 150),
                width=halo_width,
                joint="curve",
            )
        image.alpha_composite(halo_layer)
        draw = ImageDraw.Draw(image)

    for curve, points in zip(scene.curves, scaled_curves):
        outline = curve.outline_color or (35, 35, 35)
        outer_width = line_width + 3 * supersample
        if stroke_profile == "paddle_beaded":
            if scene.rope_outline:
                _draw_paddles(draw, points, outline, outer_width, 13 * supersample)
            _draw_paddles(draw, points, curve.color, line_width, 13 * supersample)
        elif stroke_profile in {"current", "halo_core"}:
            if scene.line_style == "dashed" and stroke_profile == "current":
                if scene.rope_outline:
                    rope_engine.draw_dashed(draw, points, outline, outer_width, dash, gap)
                rope_engine.draw_dashed(
                    draw,
                    points,
                    curve.color,
                    line_width,
                    dash,
                    gap,
                )
            else:
                if scene.rope_outline and stroke_profile == "current":
                    draw.line(
                        points,
                        fill=outline,
                        width=outer_width,
                        joint="curve",
                    )
                draw.line(
                    points,
                    fill=curve.color,
                    width=line_width,
                    joint="curve",
                )
        else:
            raise ValueError(f"unknown stroke profile: {stroke_profile}")

    marker_radius = layout["marker_r"]
    dots = qa.get("dot_markers")
    if dots is not None:
        dot_radius = max(5 * supersample, round(marker_radius * 0.42))
        for dot in dots:
            x, y = point((dot["point"][0], dot["point"][1]))
            draw.ellipse(
                (
                    x - dot_radius,
                    y - dot_radius,
                    x + dot_radius,
                    y + dot_radius,
                ),
                fill=tuple(dot["color"]),
                outline=(25, 25, 25),
                width=supersample,
            )
        return image.convert("RGB").resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    font = load_bundled_font(layout["font"])
    for curve in scene.curves:
        badges = (
            (curve.si, curve.start, (255, 246, 210)),
            (curve.ei, curve.end, (225, 245, 255)),
        )
        for label, location, ring_color in badges:
            x, y = point(location)
            draw.ellipse(
                (
                    x - marker_radius,
                    y - marker_radius,
                    x + marker_radius,
                    y + marker_radius,
                ),
                fill=ring_color,
                outline=(35, 35, 35),
                width=2 * supersample,
            )
            box = draw.textbbox((0, 0), label, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
            draw.text(
                (
                    x - text_width / 2 - box[0],
                    y - text_height / 2 - box[1],
                ),
                label,
                font=font,
                fill=(25, 25, 25),
            )
    return image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def _qa_answer_is_consistent(scene: rope_engine.Scene, qa: dict[str, Any]) -> bool:
    qa_type = qa["type"]
    targets = qa["target_indices"]
    if qa_type == "one_marker":
        if len(targets) != 1:
            return False
        curve = scene.curves[targets[0]]
        question = qa["question"]
        if f"What is {curve.si} connected to?" in question:
            return qa["answer"] == curve.ei
        if f"What is {curve.ei} connected to?" in question:
            return qa["answer"] == curve.si
        return False
    if qa_type == "ordered":
        return qa["answer"] == "".join(
            scene.curves[index].ei for index in targets
        )
    if qa_type == "all_pairs":
        expected = ", ".join(
            f"{curve.si}-{curve.ei}"
            for curve in sorted(scene.curves, key=lambda curve: curve.si)
        )
        return qa["answer"] == expected
    if qa_type == "same_line":
        policy = qa.get("same_line_query_policy") or {}
        indices = policy.get("dot_curve_indices") or []
        if len(indices) != 2:
            return False
        return qa["answer"] == ("yes" if indices[0] == indices[1] else "no")
    return False


def case_validity(built: BuiltCase) -> dict[str, Any]:
    """Recompute every serialized geometry, pairing, and QA validity gate."""

    scene = built.scene
    qa = built.qa
    policy = built.provenance["validity_policy"]
    metrics = scene.curve_metrics
    minimum_crossings = int(
        policy.get(
            "required_scene_crossings",
            policy.get("minimum_scene_crossings", 0),
        )
    )
    maximum_crossings = int(policy.get("maximum_scene_crossings", 10**9))
    angle_floor = float(
        policy.get(
            "global_min_cross_angle_deg",
            policy.get("global_crossing_angle_floor_deg", ANGLE_FLOOR_DEG),
        )
    )
    minimum_turn = min(
        (float(row["min_turn_radius_px"]) for row in metrics),
        default=math.inf,
    )
    maximum_turn = max(
        (float(row["max_local_turn_deg"]) for row in metrics),
        default=0.0,
    )
    maximum_self_crossings = max(
        (int(row["self_crossings"]) for row in metrics),
        default=0,
    )
    near_parallel = [float(row["near_parallel_px"]) for row in metrics]
    mean_near_parallel = (
        sum(near_parallel) / len(near_parallel) if near_parallel else 0.0
    )
    max_near_parallel = max(near_parallel, default=0.0)
    finite_spacing = [
        float(row["min_crossing_spacing_px"])
        for row in metrics
        if math.isfinite(float(row["min_crossing_spacing_px"]))
    ]
    min_crossing_spacing = min(finite_spacing, default=math.inf)
    pairing = policy.get(
        "endpoint_pairing_policy",
        built.provenance.get("endpoint_pairing_policy", {}),
    )

    same_line_valid = True
    if qa["type"] == "same_line":
        same_line = qa.get("same_line_query_policy") or {}
        arc_fractions = same_line.get("dot_arc_fractions") or []
        other_clearance = same_line.get("realized_other_curve_clearance_px") or []
        self_clearance = (
            same_line.get("realized_nonlocal_self_clearance_px") or []
        )
        dot_indices = same_line.get("dot_curve_indices") or []
        same_line_valid = bool(
            same_line.get("dots_are_interior")
            and same_line.get("answer_sampled_after_distance_match")
            and len(qa.get("dot_markers") or []) == 2
            and len(dot_indices) == 2
            and len(arc_fractions) == 2
            and all(0.0 < float(value) < 1.0 for value in arc_fractions)
            and float(same_line.get("separation_match_error_px", math.inf))
            <= float(same_line.get("distance_match_tolerance_px", -math.inf))
            and len(other_clearance) == 2
            and all(
                float(value)
                >= float(same_line.get("minimum_other_curve_clearance_px", math.inf))
                for value in other_clearance
            )
            and len(self_clearance) == 2
            and all(
                float(value)
                >= float(
                    same_line.get("minimum_nonlocal_self_clearance_px", math.inf)
                )
                for value in self_clearance
            )
        )

    checks = {
        "endpoint_clean": bool(scene.endpoint_clean),
        "preview_curve_count": len(scene.curves) == PREVIEW_NUM_CURVES,
        "uniform_path_color": len({tuple(curve.color) for curve in scene.curves})
        == 1,
        "global_crossing_angle": bool(
            scene.total_crossings == 0
            or (scene.min_angle is not None and scene.min_angle >= angle_floor)
        ),
        "minimum_scene_crossings": scene.total_crossings >= minimum_crossings,
        "maximum_scene_crossings": scene.total_crossings <= maximum_crossings,
        "minimum_turn_radius": minimum_turn
        >= float(policy.get("min_turn_radius_px", -math.inf)),
        "maximum_local_turn": maximum_turn
        <= float(policy.get("max_local_turn_deg", math.inf)),
        "maximum_self_crossings": maximum_self_crossings
        <= int(policy.get("max_self_crossings", 10**9)),
        "endpoint_pairing": bool(
            pairing.get("valid")
            and pairing.get("uniform_sampling")
            and pairing.get("constraints_are_difficulty_independent")
        ),
        "mean_near_parallel": mean_near_parallel
        <= float(policy.get("maximum_mean_near_parallel_px", math.inf)),
        "curve_near_parallel": max_near_parallel
        <= float(policy.get("maximum_curve_near_parallel_px", math.inf)),
        "crossing_spacing": min_crossing_spacing
        >= float(policy.get("minimum_crossing_spacing_px", -math.inf)),
        "same_line_policy": same_line_valid,
        "qa_answer_consistency": _qa_answer_is_consistent(scene, qa),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "same_line_policy_preserved": same_line_valid,
        "thresholds": {
            "global_crossing_angle_floor_deg": angle_floor,
            "minimum_scene_crossings": minimum_crossings,
            "maximum_scene_crossings": maximum_crossings,
            "minimum_turn_radius_px": policy.get("min_turn_radius_px"),
            "maximum_local_turn_deg": policy.get("max_local_turn_deg"),
            "maximum_self_crossings": policy.get("max_self_crossings"),
            "maximum_mean_near_parallel_px": policy.get(
                "maximum_mean_near_parallel_px"
            ),
            "maximum_curve_near_parallel_px": policy.get(
                "maximum_curve_near_parallel_px"
            ),
            "minimum_crossing_spacing_px": policy.get(
                "minimum_crossing_spacing_px"
            ),
        },
        "realized": {
            "global_crossing_angle_deg": scene.min_angle,
            "scene_crossings": scene.total_crossings,
            "minimum_turn_radius_px": round(minimum_turn, 3),
            "maximum_local_turn_deg": round(maximum_turn, 3),
            "maximum_self_crossings": maximum_self_crossings,
            "mean_near_parallel_px": round(mean_near_parallel, 3),
            "maximum_curve_near_parallel_px": round(max_near_parallel, 3),
            "minimum_crossing_spacing_px": (
                round(min_crossing_spacing, 3)
                if math.isfinite(min_crossing_spacing)
                else None
            ),
        },
    }


def case_metadata(
    built: BuiltCase,
    *,
    condition: str,
    variant: str,
    seed: int,
    background: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = built.scene
    qa = built.qa
    target_metrics = [scene.curve_metrics[index] for index in qa["target_indices"]]
    target_min_angle = min(
        (scene.curve_min_angle[index] for index in qa["target_indices"]),
        default=180.0,
    )
    pairs = [
        {
            "curve_index": index,
            "start_label": curve.si,
            "end_label": curve.ei,
            "start_marker_text": curve.si,
            "end_marker_text": curve.ei,
        }
        for index, curve in enumerate(scene.curves)
    ]
    stroke_profile = variant if condition == "stroke_ood" else "current"
    if stroke_profile == "paddle_beaded":
        realized_line_style = "beaded_continuous_connector"
        realized_rope_outline = scene.rope_outline
        halo_enabled = False
    elif stroke_profile == "halo_core":
        realized_line_style = "solid"
        realized_rope_outline = False
        halo_enabled = True
    else:
        realized_line_style = scene.line_style
        realized_rope_outline = scene.rope_outline
        halo_enabled = False
    return {
        "difficulty": "xhard_base_ood",
        "base_difficulty": "xhard",
        "ood_axis": condition,
        "ood_variant": variant,
        "seed": seed,
        "qa_type": qa["type"],
        "question": qa["question"],
        "answer": qa["answer"],
        "pairs": pairs,
        "target_indices": qa["target_indices"],
        "dot_markers": qa.get("dot_markers"),
        "same_line_query_policy": qa.get("same_line_query_policy"),
        "num_curves": len(scene.curves),
        "geometry_backend": built.provenance["geometry_backend"],
        "program_family": built.provenance["program_family"],
        "marker_layout": scene.marker_layout,
        "stroke_profile": stroke_profile,
        "background_profile": (
            variant if condition == "background_ood" else "plain_grid"
        ),
        "background": background,
        "color_mode": scene.color_mode,
        "base_line_style": scene.line_style,
        "line_style": realized_line_style,
        "line_width": scene.line_width,
        "base_rope_outline": scene.rope_outline,
        "rope_outline": realized_rope_outline,
        "halo_enabled": halo_enabled,
        "endpoint_clean": scene.endpoint_clean,
        "min_cross_angle": (
            round(scene.min_angle, 2) if scene.min_angle is not None else None
        ),
        "target_min_angle": (
            round(target_min_angle, 2) if target_min_angle < 180.0 else None
        ),
        "total_crossings": scene.total_crossings,
        "target_crossings": round(_mean(target_metrics, "crossings") or 0.0, 2),
        "mean_near_parallel_px": round(
            _mean(scene.curve_metrics, "near_parallel_px") or 0.0,
            2,
        ),
        "max_near_parallel_px": round(
            max(
                float(metrics["near_parallel_px"])
                for metrics in scene.curve_metrics
            ),
            2,
        ),
        "mean_arc_px": round(
            sum(curve.arc for curve in scene.curves) / len(scene.curves),
            2,
        ),
        "mean_arc_over_chord": round(
            sum(curve.arc / curve.chord for curve in scene.curves) / len(scene.curves),
            3,
        ),
        "curves": [
            {
                "curve_index": index,
                "start_label": curve.si,
                "end_label": curve.ei,
                "start_point": [round(curve.start[0]), round(curve.start[1])],
                "end_point": [round(curve.end[0]), round(curve.end[1])],
                "color": list(curve.color),
                "length_px": round(curve.arc, 2),
                "crossing_count": scene.crossing_counts[index],
                "trace": [
                    [round(point[0]), round(point[1])] for point in curve.trace
                ],
            }
            for index, curve in enumerate(scene.curves)
        ],
        "validity": case_validity(built),
        "provenance": built.provenance,
    }
