"""Path Tracing curve engine: the Bezier and Pathfinder scenes.

Bezier scenes use the curve construction of bezier_geometry.py with the
rope-engine QA and rendering stack.  Pathfinder is delegated to
:mod:`rope_engine` with unchanged curve geometry; only its shortcut-prone
``same_line`` endpoint dots are replaced with distance-matched interior queries.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from visro.seeding import seed_text

from . import rope_engine
from .bezier_geometry import PT_LEVELS, build_curve as bezier_build_curve, distance as bezier_distance, endpoint_safe as bezier_endpoint_safe, layout_params as bezier_layout_params, polyline_crossings as bezier_polyline_crossings, random_marker_points as bezier_random_marker_points


# fixed RNG namespaces of the Bezier streams (changing them changes every item)
_BEZIER_STREAM = seed_text(7795765899059697209426276921905)
_SAME_LINE_STREAM = seed_text(12150992819972992234301721083963150078383771024505898740278997184865)
_PAIRING_STREAM = seed_text(148241894224219179686456585914047372650751066320433)
CURVE_STYLES = rope_engine.CURVE_STYLES
LEVEL_ORDER = rope_engine.LEVEL_ORDER
scheduled_appearance = rope_engine.scheduled_appearance
CURVE_RETRY_BUDGET = 80
ENDPOINT_PAIRING_POLICY = {
    "name": "uniform_admissible_long_chord",
    # These constraints are deliberately identical across difficulty levels.
    # Difficulty must not make the hidden pairing progressively easier to infer.
    "minimum_chord_diagonal_fraction": 0.22,
    "minimum_mean_chord_diagonal_fraction": 0.38,
    "long_chord_diagonal_fraction": 0.45,
    "minimum_long_chord_share": 0.50,
    # A qualifying marker layout must leave many physically distinct matchings.
    "minimum_admissible_permutations": {
        2: 2,
        3: 4,
        4: 8,
        5: 20,
        6: 50,
        7: 100,
        8: 500,
        9: 1000,
        10: 2000,
    },
    # Coordinate-only posterior concentration gates.  They are evaluated over
    # the uniform admissible set before one physical matching is sampled.
    "maximum_mean_row_probability_rule": "min(0.55, 2.2 / n)",
    "maximum_single_pair_probability_rule": "max(0.45, 2.5 / n)",
}
BEZIER_GATE_CONFIG = {
    "curve_retry_budget_per_curve": CURVE_RETRY_BUDGET,
    "scene_attempt_budget_rule": {
        "minimum": 8,
        "scene_retries_multiplier": 8,
    },
    "crossing_angle_floor_deg": 30.0,
    "require_endpoint_clearance": True,
    "require_bezier_all_crossings_angle_floor": True,
    "endpoint_pairing": ENDPOINT_PAIRING_POLICY,
    "levels": {
        level: {
            "min_turn_radius_px": spec.min_turn_radius_px,
            "max_local_turn_deg": spec.max_local_turn_deg,
            "max_self_crossings": spec.max_self_crossings,
        }
        for level, spec in rope_engine.BEZIER_LEVELS.items()
    },
}


def uniform_admissible_endpoint_permutation(
    starts: list[tuple[float, float]],
    end_positions: list[tuple[float, float]],
    width: int,
    height: int,
    rng: random.Random,
) -> tuple[list[int] | None, dict]:
    """Sample the physical endpoint matching from a large uniform set.

    A sampler that preferred far endpoints would make connectivity partly
    recoverable from endpoint coordinates without tracing a path.  Here the
    geometric constraints are level-independent, every admissible permutation
    is equiprobable, and concentrated marker layouts are rejected.
    """
    n = len(starts)
    if n != len(end_positions) or n < 2:
        raise ValueError("endpoint pairing requires equal endpoint sets of size >=2")
    diagonal = math.hypot(width, height)
    min_chord = ENDPOINT_PAIRING_POLICY["minimum_chord_diagonal_fraction"] * diagonal
    min_mean = (
        ENDPOINT_PAIRING_POLICY["minimum_mean_chord_diagonal_fraction"] * diagonal
    )
    long_threshold = ENDPOINT_PAIRING_POLICY["long_chord_diagonal_fraction"] * diagonal
    min_long_count = math.ceil(ENDPOINT_PAIRING_POLICY["minimum_long_chord_share"] * n)
    distances = [
        [bezier_distance(start, end) for end in end_positions] for start in starts
    ]
    admissible: list[tuple[int, ...]] = []
    for permutation in itertools.permutations(range(n)):
        chords = [
            distances[index][end_index] for index, end_index in enumerate(permutation)
        ]
        if min(chords) < min_chord:
            continue
        if sum(chords) / n < min_mean:
            continue
        if sum(chord >= long_threshold for chord in chords) < min_long_count:
            continue
        admissible.append(permutation)

    required = ENDPOINT_PAIRING_POLICY["minimum_admissible_permutations"].get(
        n, max(2, n * n)
    )
    stats = {
        "name": ENDPOINT_PAIRING_POLICY["name"],
        "constraints_are_difficulty_independent": True,
        "minimum_chord_diagonal_fraction": ENDPOINT_PAIRING_POLICY[
            "minimum_chord_diagonal_fraction"
        ],
        "minimum_mean_chord_diagonal_fraction": ENDPOINT_PAIRING_POLICY[
            "minimum_mean_chord_diagonal_fraction"
        ],
        "long_chord_diagonal_fraction": ENDPOINT_PAIRING_POLICY[
            "long_chord_diagonal_fraction"
        ],
        "minimum_long_chord_share": ENDPOINT_PAIRING_POLICY["minimum_long_chord_share"],
        "admissible_permutation_count": len(admissible),
        "minimum_admissible_permutation_count": required,
        "permutations_evaluated": math.factorial(n),
        "uniform_sampling": True,
        "selection_draws": 0,
        "full_mapping_coordinate_only_probability": (
            round(1.0 / len(admissible), 9) if admissible else None
        ),
        "valid": False,
    }
    if len(admissible) < required:
        stats.update(
            {
                "mean_row_max_probability": None,
                "maximum_single_pair_probability": None,
                "selected_pair_marginal_probabilities": None,
                "selected_admissible_rank": None,
            }
        )
        return None, stats

    counts = [[0 for _ in range(n)] for _ in range(n)]
    for permutation in admissible:
        for start_index, end_index in enumerate(permutation):
            counts[start_index][end_index] += 1
    row_maxima = [max(row) / len(admissible) for row in counts]
    mean_row_max = sum(row_maxima) / n
    maximum_single = max(row_maxima)
    max_mean_allowed = min(0.55, 2.2 / n)
    max_single_allowed = max(0.45, 2.5 / n)
    stats.update(
        {
            "mean_row_max_probability": round(mean_row_max, 6),
            "maximum_single_pair_probability": round(maximum_single, 6),
            "maximum_allowed_mean_row_probability": round(max_mean_allowed, 6),
            "maximum_allowed_single_pair_probability": round(max_single_allowed, 6),
        }
    )
    if mean_row_max > max_mean_allowed or maximum_single > max_single_allowed:
        stats["selected_pair_marginal_probabilities"] = None
        stats["selected_admissible_rank"] = None
        return None, stats

    selected_rank = rng.randrange(len(admissible))
    selected = list(admissible[selected_rank])
    chords = [distances[index][end_index] for index, end_index in enumerate(selected)]
    stats.update(
        {
            "selection_draws": 1,
            "selected_admissible_rank": selected_rank,
            "selected_pair_marginal_probabilities": [
                round(counts[index][end_index] / len(admissible), 6)
                for index, end_index in enumerate(selected)
            ],
            "selected_minimum_chord_px": round(min(chords), 3),
            "selected_mean_chord_px": round(sum(chords) / n, 3),
            "selected_long_chord_count": sum(
                chord >= long_threshold for chord in chords
            ),
            "valid": True,
        }
    )
    return selected, stats


def _scene_attempt_budget(scene_retries: int) -> int:
    rule = BEZIER_GATE_CONFIG["scene_attempt_budget_rule"]
    return max(
        int(rule["minimum"]),
        int(scene_retries) * int(rule["scene_retries_multiplier"]),
    )


def _resolved_style(seed: int, curve_style: str) -> str:
    if curve_style not in CURVE_STYLES:
        raise ValueError(f"unknown curve_style: {curve_style}")
    if curve_style == "mixed":
        return "pathfinder" if (seed // 2) % 2 == 0 else "bezier"
    return curve_style


SAME_LINE_QUERY_POLICY = {
    "name": "distance_matched_interior_points",
    "arc_fractions": [round(0.08 + 0.03 * index, 2) for index in range(29)],
    "minimum_same_curve_arc_fraction_gap": 0.24,
    "minimum_dot_separation_px": 55.0,
    "maximum_dot_separation_px": 320.0,
    # Looser matches reintroduce a useful distance cue at 448 px.  Reject the
    # scene and let the outer geometry-salt retry find a better support instead.
    "distance_match_tolerances_px": [4.0, 6.0, 8.0, 10.0],
    "minimum_other_curve_clearance_px": 16.0,
    "minimum_nonlocal_self_clearance_px": 16.0,
    "local_self_arc_neighborhood_px": 40.0,
}


def _polyline_point_at_fraction(
    points: list[tuple[float, float]], fraction: float
) -> tuple[tuple[float, float], int]:
    if not points:
        raise ValueError("cannot sample an empty polyline")
    if len(points) == 1:
        return points[0], 0
    lengths = [rope_engine.distance(left, right) for left, right in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 0:
        return points[0], 0
    target = min(1.0, max(0.0, fraction)) * total
    walked = 0.0
    for index, segment in enumerate(lengths):
        if walked + segment >= target:
            alpha = 0.0 if segment <= 0 else (target - walked) / segment
            left, right = points[index], points[index + 1]
            return (
                (
                    left[0] + alpha * (right[0] - left[0]),
                    left[1] + alpha * (right[1] - left[1]),
                ),
                index,
            )
        walked += segment
    return points[-1], len(points) - 2


def _point_render_clearances(
    scene: rope_engine.Scene,
    curve_index: int,
    point: tuple[float, float],
    own_arc_fraction: float,
) -> tuple[float, float]:
    other_minimum = float("inf")
    self_minimum = float("inf")
    for other_index, curve in enumerate(scene.curves):
        own_total_arc = (
            sum(
                rope_engine.distance(left, right)
                for left, right in zip(curve.dense, curve.dense[1:])
            )
            if other_index == curve_index
            else 0.0
        )
        own_target_arc = own_arc_fraction * own_total_arc
        walked = 0.0
        for segment_index, (left, right) in enumerate(
            zip(curve.dense, curve.dense[1:])
        ):
            segment_length = rope_engine.distance(left, right)
            if other_index == curve_index:
                segment_mid_arc = walked + 0.5 * segment_length
                walked += segment_length
                if (
                    abs(segment_mid_arc - own_target_arc)
                    <= SAME_LINE_QUERY_POLICY["local_self_arc_neighborhood_px"]
                ):
                    continue
            separation = rope_engine.point_segment_distance(point, left, right)
            if other_index == curve_index:
                self_minimum = min(self_minimum, separation)
            else:
                other_minimum = min(other_minimum, separation)
    return other_minimum, self_minimum


def _same_line_candidate_points(scene: rope_engine.Scene) -> list[list[dict]]:
    clearance = SAME_LINE_QUERY_POLICY["minimum_other_curve_clearance_px"]
    rows: list[list[dict]] = []
    for curve_index, curve in enumerate(scene.curves):
        candidates = []
        for fraction in SAME_LINE_QUERY_POLICY["arc_fractions"]:
            point, segment_index = _polyline_point_at_fraction(curve.dense, fraction)
            other_clearance, self_clearance = _point_render_clearances(
                scene, curve_index, point, fraction
            )
            if other_clearance < clearance or self_clearance < clearance:
                continue
            candidates.append(
                {
                    "curve_index": curve_index,
                    "arc_fraction": fraction,
                    "point": point,
                    "segment_index": segment_index,
                    "other_curve_clearance_px": other_clearance,
                    "nonlocal_self_clearance_px": self_clearance,
                }
            )
        rows.append(candidates)
    return rows


def make_same_line_qa(
    scene: rope_engine.Scene,
    rng: random.Random,
    min_target_crossings: int,
) -> dict:
    """Choose interior dot pairs from a shared, distance-matched support.

    A matched yes/no candidate pair is selected first; the answer coin is drawn
    only afterward.  Consequently dot separation cannot reveal the label, while
    every visible dot remains on exactly one curve and away from crossings.
    """
    n = len(scene.curves)
    if n < 2:
        raise RuntimeError("same_line requires at least two curves")
    candidates = _same_line_candidate_points(scene)
    eligible = [
        index
        for index in range(n)
        if len(candidates[index]) >= 2
        and (
            not scene.curve_min_angle
            or scene.curve_min_angle[index] >= scene.angle_floor
        )
    ]
    difficult = [
        index
        for index in eligible
        if scene.crossing_counts[index] >= min_target_crossings
    ]
    focus = difficult or eligible
    if scene.curve_metrics and focus:
        focus = sorted(
            focus,
            key=lambda index: (
                scene.curve_metrics[index]["crossings"],
                scene.curve_metrics[index]["large_loops"],
                scene.curve_metrics[index]["central_reentries"],
                scene.curve_metrics[index]["arc_over_chord"],
            ),
            reverse=True,
        )[: max(1, len(focus) // 2)]
    if not focus:
        raise RuntimeError("no unambiguous interior points for same_line")

    minimum_gap = SAME_LINE_QUERY_POLICY["minimum_same_curve_arc_fraction_gap"]
    minimum_distance = SAME_LINE_QUERY_POLICY["minimum_dot_separation_px"]
    maximum_distance = SAME_LINE_QUERY_POLICY["maximum_dot_separation_px"]
    yes_candidates = []
    for curve_index in focus:
        for left, right in itertools.combinations(candidates[curve_index], 2):
            if abs(left["arc_fraction"] - right["arc_fraction"]) < minimum_gap:
                continue
            separation = rope_engine.distance(left["point"], right["point"])
            if minimum_distance <= separation <= maximum_distance:
                yes_candidates.append((separation, left, right))

    no_candidates = []
    curve_pairs = [
        (left, right)
        for left in focus
        for right in range(left + 1, n)
        if right in eligible and right in scene.crosses[left]
    ]
    if not curve_pairs:
        curve_pairs = list(itertools.combinations(eligible, 2))
    for left_index, right_index in curve_pairs:
        for left in candidates[left_index]:
            for right in candidates[right_index]:
                separation = rope_engine.distance(left["point"], right["point"])
                if minimum_distance <= separation <= maximum_distance:
                    no_candidates.append((separation, left, right))
    if not yes_candidates or not no_candidates:
        raise RuntimeError("could not construct both same_line answer classes")

    no_candidates.sort(key=lambda row: row[0])
    no_distances = [row[0] for row in no_candidates]
    selected = None
    match_count = 0
    used_tolerance = None
    for tolerance in SAME_LINE_QUERY_POLICY["distance_match_tolerances_px"]:
        selected = None
        match_count = 0
        for yes_candidate in yes_candidates:
            separation = yes_candidate[0]
            low = bisect.bisect_left(no_distances, separation - tolerance)
            high = bisect.bisect_right(no_distances, separation + tolerance)
            for no_candidate in no_candidates[low:high]:
                match_count += 1
                if rng.randrange(match_count) == 0:
                    selected = (yes_candidate, no_candidate)
        if selected is not None:
            used_tolerance = tolerance
            break
    if selected is None or used_tolerance is None:
        raise RuntimeError("could not distance-match same_line answer classes")

    yes_candidate, no_candidate = selected
    want_yes = rng.random() < 0.5
    chosen = yes_candidate if want_yes else no_candidate
    counterfactual = no_candidate if want_yes else yes_candidate
    _, left, right = chosen
    target_indices = sorted({left["curve_index"], right["curve_index"]})
    colors = rope_engine.two_dot_colors(rng)
    dot_markers = [
        {
            "point": [
                int(round(candidate["point"][0])),
                int(round(candidate["point"][1])),
            ],
            "color": list(color),
        }
        for candidate, color in zip((left, right), colors)
    ]
    pairs = [
        {"start": curve.si, "end": curve.ei}
        for curve in sorted(scene.curves, key=lambda curve: curve.si)
    ]
    query_policy = {
        "name": SAME_LINE_QUERY_POLICY["name"],
        "dots_are_interior": True,
        "answer_sampled_after_distance_match": True,
        "minimum_other_curve_clearance_px": SAME_LINE_QUERY_POLICY[
            "minimum_other_curve_clearance_px"
        ],
        "minimum_nonlocal_self_clearance_px": SAME_LINE_QUERY_POLICY[
            "minimum_nonlocal_self_clearance_px"
        ],
        "local_self_arc_neighborhood_px": SAME_LINE_QUERY_POLICY[
            "local_self_arc_neighborhood_px"
        ],
        "distance_match_tolerance_px": used_tolerance,
        "matched_candidate_pair_count": match_count,
        "yes_candidate_count": len(yes_candidates),
        "no_candidate_count": len(no_candidates),
        "dot_curve_indices": [left["curve_index"], right["curve_index"]],
        "dot_arc_fractions": [left["arc_fraction"], right["arc_fraction"]],
        "realized_other_curve_clearance_px": [
            round(left["other_curve_clearance_px"], 3),
            round(right["other_curve_clearance_px"], 3),
        ],
        "realized_nonlocal_self_clearance_px": [
            round(left["nonlocal_self_clearance_px"], 3),
            round(right["nonlocal_self_clearance_px"], 3),
        ],
        "dot_separation_px": round(chosen[0], 3),
        "counterfactual_dot_separation_px": round(counterfactual[0], 3),
        "separation_match_error_px": round(abs(chosen[0] - counterfactual[0]), 3),
    }
    return {
        "type": "same_line",
        "question": (
            "Are the two colored dots connected by the same line? " "Answer yes or no."
        ),
        "answer": "yes" if want_yes else "no",
        "target_indices": target_indices,
        "dot_markers": dot_markers,
        "pairs": pairs,
        "same_line_query_policy": query_policy,
    }


def _appearance(
    level: str,
    qa_type: str,
    seed: int,
    color_mode_override: str | None,
    line_style_override: str | None,
    rope_outline_override: bool | None,
) -> tuple[str, str, int, bool, random.Random]:
    scheduled_color, scheduled_style, scheduled_outline = scheduled_appearance(seed)
    color_mode = color_mode_override or scheduled_color
    force_outline = line_style_override == "outlined"
    line_style = "solid" if force_outline else (line_style_override or scheduled_style)
    if color_mode not in {"distinct", "family", "uniform"}:
        raise ValueError(f"unknown color mode: {color_mode}")
    if line_style not in {"solid", "dashed"}:
        raise ValueError(f"unknown line style: {line_style}")
    appearance_rng = random.Random(f"{seed}:{level}:{qa_type}:{_BEZIER_STREAM}:appearance")
    line_width = appearance_rng.choice(rope_engine.BEZIER_LEVELS[level].line_widths)
    if line_style == "solid" and appearance_rng.random() < 0.5:
        line_width += 1
    rope_outline = (
        bool(rope_outline_override)
        if rope_outline_override is not None
        else (force_outline or scheduled_outline)
    )
    return color_mode, line_style, line_width, rope_outline, appearance_rng


def _endpoints_are_clean(geometry: dict, width: int, height: int) -> bool:
    if not geometry["endpoint_clean"]:
        return False
    markers = geometry["starts"] + geometry["end_positions"]
    clearance = bezier_layout_params(width, height)["marker_r"] + 10
    for start, end, _, dense in geometry["geos"]:
        if not rope_engine.endpoint_safe(dense, markers, {start, end}, clearance):
            return False
    return True


def _build_current_gated_geometry_attempt(
    n_curves: int,
    width: int,
    height: int,
    historical: dict,
    current_limits: rope_engine.BezierLevel,
    geometry_rng: random.Random,
    pairing_rng: random.Random,
    pairing_rng_namespace: str,
    endpoint_clearance: float,
    angle_floor: float,
    curve_retry_budget: int,
) -> tuple[dict | None, dict]:
    """Build with the Bezier constructor, rejecting only unreadable curves.

    Marker placement, endpoint pairing, and every curve proposal come from the
    fixed Bezier implementation.  The extra loop selects proposals that
    satisfy the rope-engine smoothness/self-crossing contract before composing a
    scene.  It does not post-process or reshape accepted curves.
    """
    marker_points = bezier_random_marker_points(
        n_curves * 2, width, height, geometry_rng
    )
    starts, end_positions = marker_points[:n_curves], marker_points[n_curves:]
    end_indices, pairing_stats = uniform_admissible_endpoint_permutation(
        starts, end_positions, width, height, pairing_rng
    )
    pairing_stats["rng_namespace"] = pairing_rng_namespace
    pairing_stats["rng_isolated_from_geometry"] = True
    stats = {
        "endpoint_pairing_policy": pairing_stats,
        "curve_retry_budget_per_curve": curve_retry_budget,
        "curve_candidate_attempts": 0,
        "curve_candidates_accepted": 0,
        "curve_candidates_rejected": 0,
        "accepted_curve_attempts": [],
        # A proposal can increment more than one reason, so this is explicitly
        # a non-exclusive diagnostic tally.
        "curve_rejection_reasons": {
            "endpoint_clearance": 0,
            "min_turn_radius": 0,
            "max_local_turn": 0,
            "max_self_crossings": 0,
        },
        "curve_budget_exhausted_at_index": None,
    }
    if end_indices is None:
        return None, stats
    ends = [end_positions[index] for index in end_indices]
    all_markers = starts + end_positions
    geos = []
    for curve_index, (start, end) in enumerate(zip(starts, ends)):
        accepted = None
        for curve_attempt in range(1, curve_retry_budget + 1):
            stats["curve_candidate_attempts"] += 1
            controls, dense = bezier_build_curve(
                start,
                end,
                width,
                height,
                historical["curve_complexity"],
                geometry_rng,
                historical["length_scale"],
                historical["max_knots"],
            )
            endpoint_clean = bezier_endpoint_safe(
                dense, all_markers, {start, end}, endpoint_clearance
            )
            if not endpoint_clean:
                stats["curve_candidates_rejected"] += 1
                stats["curve_rejection_reasons"]["endpoint_clearance"] += 1
                continue

            metrics = rope_engine.basic_rope_metrics(dense, width, height)
            reasons = []
            if metrics["min_turn_radius_px"] < current_limits.min_turn_radius_px:
                reasons.append("min_turn_radius")
            if metrics["max_local_turn_deg"] > current_limits.max_local_turn_deg:
                reasons.append("max_local_turn")
            if metrics["self_crossings"] > current_limits.max_self_crossings:
                reasons.append("max_self_crossings")
            if reasons:
                stats["curve_candidates_rejected"] += 1
                for reason in reasons:
                    stats["curve_rejection_reasons"][reason] += 1
                continue

            accepted = (start, end, controls, dense)
            stats["curve_candidates_accepted"] += 1
            stats["accepted_curve_attempts"].append(curve_attempt)
            break

        if accepted is None:
            stats["curve_budget_exhausted_at_index"] = curve_index
            return None, stats
        geos.append(accepted)

    coarse_curves = []
    for _, _, _, dense in geos:
        coarse = dense[::3]
        if coarse[-1] != dense[-1]:
            coarse.append(dense[-1])
        coarse_curves.append(coarse)
    crossings = [[] for _ in geos]
    crosses = [set() for _ in geos]
    shallow = 0
    bezier_min_angle = 180.0
    for left in range(len(geos)):
        for right in range(left + 1, len(geos)):
            hits = bezier_polyline_crossings(coarse_curves[left], coarse_curves[right])
            if hits:
                crosses[left].add(right)
                crosses[right].add(left)
            for point, angle in hits:
                crossings[left].append(point)
                crossings[right].append(point)
                bezier_min_angle = min(bezier_min_angle, angle)
                if angle < angle_floor:
                    shallow += 1

    geometry = {
        "starts": starts,
        "ends": ends,
        "end_positions": end_positions,
        "end_indices": end_indices,
        "geos": geos,
        "crossings": crossings,
        "crosses": crosses,
        "endpoint_clean": True,
        "shallow": shallow,
        "min_angle": bezier_min_angle,
        "total_crossings": (
            sum(len(curve_crossings) for curve_crossings in crossings) // 2
        ),
        "valid": shallow == 0,
    }
    return geometry, stats


def build_bezier_scene(
    level: str,
    qa_type: str,
    seed: int,
    size: int,
    angle_floor: float,
    scene_retries: int,
    geometry_salt: int,
    color_mode_override: str | None,
    line_style_override: str | None,
    rope_outline_override: bool | None,
) -> tuple[rope_engine.Scene, int, dict]:
    historical = PT_LEVELS[level]
    current_limits = rope_engine.BEZIER_LEVELS[level]
    color_mode, line_style, line_width, rope_outline, appearance_rng = _appearance(
        level,
        qa_type,
        seed,
        color_mode_override,
        line_style_override,
        rope_outline_override,
    )
    attempt_budget = _scene_attempt_budget(scene_retries)
    total_curve_attempts = 0
    total_curve_candidates_accepted = 0
    total_curve_candidates_rejected = 0
    total_curve_rejection_reasons = {
        "endpoint_clearance": 0,
        "min_turn_radius": 0,
        "max_local_turn": 0,
        "max_self_crossings": 0,
    }
    scene_rejection_counts = {
        "endpoint_pairing_ambiguity": 0,
        "curve_retry_budget": 0,
        "endpoint_clearance": 0,
        "bezier_shallow_crossing": 0,
        "current_global_crossing_angle": 0,
        "too_few_crossings": 0,
        "too_many_crossings": 0,
        "current_smoothness_assertion": 0,
    }
    for attempt in range(attempt_budget):
        geometry_namespace = (
            f"{seed}:{level}:{qa_type}:{_BEZIER_STREAM}:"
            f"geom{geometry_salt}:scene{attempt}"
        )
        pairing_namespace = f"{geometry_namespace}:{_PAIRING_STREAM}"
        geometry_rng = random.Random(geometry_namespace)
        pairing_rng = random.Random(pairing_namespace)
        n_curves = geometry_rng.randint(*historical["num_curves"])
        geometry, curve_retry_stats = _build_current_gated_geometry_attempt(
            n_curves,
            size,
            size,
            historical,
            current_limits,
            geometry_rng,
            pairing_rng,
            pairing_namespace,
            bezier_layout_params(size, size)["marker_r"] + 10,
            angle_floor,
            CURVE_RETRY_BUDGET,
        )
        total_curve_attempts += curve_retry_stats["curve_candidate_attempts"]
        total_curve_candidates_accepted += curve_retry_stats[
            "curve_candidates_accepted"
        ]
        total_curve_candidates_rejected += curve_retry_stats[
            "curve_candidates_rejected"
        ]
        for reason, count in curve_retry_stats["curve_rejection_reasons"].items():
            total_curve_rejection_reasons[reason] += count
        if geometry is None:
            if not curve_retry_stats["endpoint_pairing_policy"]["valid"]:
                scene_rejection_counts["endpoint_pairing_ambiguity"] += 1
            else:
                scene_rejection_counts["curve_retry_budget"] += 1
            continue

        dense_list = [row[3] for row in geometry["geos"]]
        (
            crosses,
            crossing_counts,
            curve_min_angle,
            pair_angle,
            metrics,
            total_crossings,
            global_min_angle,
        ) = rope_engine.analyze_rope_scene(dense_list, size, size)

        max_crossings = round(historical["max_crossings_per_curve"] * n_curves)
        required_crossings = round(historical["crossings_per_curve"] * n_curves)
        smooth = all(
            metric["min_turn_radius_px"] >= current_limits.min_turn_radius_px
            and metric["max_local_turn_deg"] <= current_limits.max_local_turn_deg
            and metric["self_crossings"] <= current_limits.max_self_crossings
            for metric in metrics
        )
        scene_reasons = []
        if not _endpoints_are_clean(geometry, size, size):
            scene_reasons.append("endpoint_clearance")
        if geometry["shallow"] != 0:
            scene_reasons.append("bezier_shallow_crossing")
        if global_min_angle < angle_floor:
            scene_reasons.append("current_global_crossing_angle")
        if total_crossings < required_crossings:
            scene_reasons.append("too_few_crossings")
        if total_crossings > max_crossings:
            scene_reasons.append("too_many_crossings")
        if not smooth:
            # This should never fire because each curve passed these same gates
            # before scene analysis. Keep it as a serialized invariant check.
            scene_reasons.append("current_smoothness_assertion")
        if scene_reasons:
            for reason in scene_reasons:
                scene_rejection_counts[reason] += 1
            continue

        colors = rope_engine.curve_colors(n_curves, color_mode, appearance_rng)
        representative = tuple(
            round(sum(color[channel] for color in colors) / len(colors))
            for channel in range(3)
        )
        shared_outline = rope_engine.contrasting_outline(representative, appearance_rng)
        curves = []
        for index, (start, end, controls, dense) in enumerate(geometry["geos"]):
            end_index = geometry["end_indices"][index]
            curves.append(
                rope_engine.Curve(
                    si=rope_engine.START_LABELS[index],
                    ei=rope_engine.END_LABELS[end_index],
                    start=start,
                    end=end,
                    controls=controls,
                    dense=dense,
                    trace=rope_engine.trace_points(dense),
                    color=colors[index],
                    outline_color=shared_outline,
                    arc=metrics[index]["arc_px"],
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
            line_width=line_width,
            line_style=line_style,
            rope_outline=rope_outline,
            curve_min_angle=curve_min_angle,
            pair_angle=pair_angle,
            angle_floor=angle_floor,
            curve_style="bezier",
            pathfinder_family="bezier",
            marker_layout="perimeter_arclength",
            curve_metrics=metrics,
            crossing_grid_cells=spatial["crossing_grid_cells"],
            central_crossings=spatial["central_crossings"],
            crossing_bbox_area_frac=spatial["crossing_bbox_area_frac"],
        )
        policy = {
            "require_endpoint_clean": True,
            "require_bezier_all_crossings_angle_floor": True,
            "bezier_shallow_crossing_count": geometry["shallow"],
            "bezier_min_cross_angle_deg": (
                round(geometry["min_angle"], 3)
                if geometry["min_angle"] < 180.0
                else None
            ),
            "global_min_cross_angle_deg": angle_floor,
            "min_cross_angle_deg": angle_floor,
            "min_turn_radius_px": current_limits.min_turn_radius_px,
            "max_local_turn_deg": current_limits.max_local_turn_deg,
            "max_self_crossings": current_limits.max_self_crossings,
            "required_scene_crossings": required_crossings,
            "maximum_scene_crossings": max_crossings,
            "scene_retry_budget": attempt_budget,
            "scene_attempts": attempt + 1,
            "scene_rejections": attempt,
            "scene_rejection_reasons": scene_rejection_counts,
            "endpoint_pairing_policy": curve_retry_stats["endpoint_pairing_policy"],
            "curve_retry_budget_per_curve": CURVE_RETRY_BUDGET,
            "accepted_scene_curve_attempts": curve_retry_stats[
                "accepted_curve_attempts"
            ],
            "accepted_scene_curve_candidate_attempts": curve_retry_stats[
                "curve_candidate_attempts"
            ],
            "curve_candidate_attempts_total": total_curve_attempts,
            "curve_candidates_accepted_total": total_curve_candidates_accepted,
            "curve_candidates_rejected_total": total_curve_candidates_rejected,
            "curve_rejection_reasons_total": total_curve_rejection_reasons,
            "realized_global_min_cross_angle_deg": (
                round(global_min_angle, 3) if global_min_angle < 180.0 else None
            ),
            "realized_min_turn_radius_px": round(
                min(metric["min_turn_radius_px"] for metric in metrics), 3
            ),
            "realized_max_local_turn_deg": round(
                max(metric["max_local_turn_deg"] for metric in metrics), 3
            ),
            "realized_max_self_crossings": max(
                metric["self_crossings"] for metric in metrics
            ),
        }
        return scene, attempt + 1, policy

    raise RuntimeError(
        f"could not build strict {level} Bezier scene after "
        f"{attempt_budget} deterministic attempts"
    )


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return sum(values) / len(values) if values else None


def _bezier_meta(
    scene: rope_engine.Scene,
    qa: dict,
    level: str,
    qa_type: str,
    seed: int,
    requested_curve_style: str,
    geometry_salt: int,
    geometry_attempts: int,
    validity_policy: dict,
    size: int,
) -> dict:
    arcs = [curve.arc for curve in scene.curves]
    chords = [curve.chord for curve in scene.curves if curve.chord > 1]
    target_indices = qa["target_indices"]
    target_metrics = [scene.curve_metrics[index] for index in target_indices]
    target_min_angle = min(
        (scene.curve_min_angle[index] for index in target_indices),
        default=180.0,
    )
    target_spacing = _mean(target_metrics, "min_crossing_spacing_px")
    metrics = scene.curve_metrics

    meta = {
        "level": level,
        "qa_type": qa_type,
        "seed": seed,
        "curve_style": "bezier",
        "requested_curve_style": requested_curve_style,
        "geometry_salt": geometry_salt,
        "geometry_backend": "bezier_geometry",
        "geometry_attempts": geometry_attempts,
        "validity_policy": validity_policy,
        "valid": True,
        "pathfinder_family": scene.pathfinder_family,
        "marker_layout": scene.marker_layout,
        "crossing_grid_cells": scene.crossing_grid_cells,
        "central_crossings": scene.central_crossings,
        "crossing_bbox_area_frac": round(scene.crossing_bbox_area_frac, 3),
        "question": qa["question"],
        "answer": qa["answer"],
        "pairs": qa["pairs"],
        "target_indices": target_indices,
        "dot_markers": qa.get("dot_markers"),
        "same_line_query_policy": qa.get("same_line_query_policy"),
        "num_curves": len(scene.curves),
        "color_mode": scene.color_mode,
        "line_style": scene.line_style,
        "line_width": scene.line_width,
        "rope_outline": scene.rope_outline,
        "min_cross_angle": (
            round(scene.min_angle, 1) if scene.min_angle is not None else None
        ),
        "target_min_angle": (
            round(target_min_angle, 1) if target_min_angle < 180.0 else None
        ),
        "total_crossings": scene.total_crossings,
        "mean_arc_px": round(sum(arcs) / len(arcs), 1),
        "mean_arc_over_chord": round(
            sum(curve.arc / curve.chord for curve in scene.curves if curve.chord > 1)
            / max(1, len(chords)),
            2,
        ),
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
                "start_label": curve.si,
                "end_label": curve.ei,
                "start_point": [round(curve.start[0]), round(curve.start[1])],
                "end_point": [round(curve.end[0]), round(curve.end[1])],
                "trace": [[round(point[0]), round(point[1])] for point in curve.trace],
                "color": list(curve.color),
                "length_px": round(curve.arc, 1),
                "crossing_count": scene.crossing_counts[index],
            }
            for index, curve in enumerate(scene.curves)
        ],
        "mean_self_crossings": round(_mean(metrics, "self_crossings") or 0.0, 2),
        "mean_large_loops": round(_mean(metrics, "large_loops") or 0.0, 2),
        "mean_central_reentries": round(_mean(metrics, "central_reentries") or 0.0, 2),
        "mean_x_reversals": round(_mean(metrics, "x_reversals") or 0.0, 2),
        "mean_lateral_reversals": round(_mean(metrics, "lateral_reversals") or 0.0, 2),
        "mean_distinct_crossed_curves": round(
            _mean(metrics, "distinct_crossed_curves") or 0.0, 2
        ),
        "mean_near_parallel_px": round(_mean(metrics, "near_parallel_px") or 0.0, 1),
        "mean_min_turn_radius_px": round(
            _mean(metrics, "min_turn_radius_px") or 0.0, 1
        ),
        "mean_max_local_turn_deg": round(
            _mean(metrics, "max_local_turn_deg") or 0.0, 1
        ),
        "target_arc_px": round(_mean(target_metrics, "arc_px") or 0.0, 1),
        "target_arc_over_chord": round(
            _mean(target_metrics, "arc_over_chord") or 0.0, 2
        ),
        "target_crossings": round(_mean(target_metrics, "crossings") or 0.0, 2),
        "target_distinct_crossed_curves": round(
            _mean(target_metrics, "distinct_crossed_curves") or 0.0, 2
        ),
        "target_self_crossings": round(
            _mean(target_metrics, "self_crossings") or 0.0, 2
        ),
        "target_large_loops": round(_mean(target_metrics, "large_loops") or 0.0, 2),
        "target_central_reentries": round(
            _mean(target_metrics, "central_reentries") or 0.0, 2
        ),
        "target_x_reversals": round(_mean(target_metrics, "x_reversals") or 0.0, 2),
        "target_lateral_reversals": round(
            _mean(target_metrics, "lateral_reversals") or 0.0, 2
        ),
        "target_near_parallel_px": round(
            _mean(target_metrics, "near_parallel_px") or 0.0, 1
        ),
        "target_min_turn_radius_px": round(
            min(row["min_turn_radius_px"] for row in target_metrics), 1
        ),
        "target_max_local_turn_deg": round(
            max(row["max_local_turn_deg"] for row in target_metrics), 1
        ),
        "target_min_crossing_spacing_px": (
            round(target_spacing, 1) if target_spacing is not None else None
        ),
    }
    return meta


def _pathfinder_same_line_scene(
    level: str,
    seed: int,
    size: int,
    min_cross_angle: float,
    scene_retries: int,
    requested_curve_style: str,
    color_mode_override: str | None,
    line_style_override: str | None,
    marker_layout_override: str | None,
    rope_outline_override: bool | None,
    geometry_salt: int,
) -> tuple[rope_engine.Scene, int, int]:
    """Replay rope-engine scene selection exactly, then return its Pathfinder scene.

    The fixed rope-engine API returns only rendered images and metadata.  Replaying its
    deterministic selection loop here lets the curve engine keep the exact underlying path
    geometry while replacing only the shortcut-prone same-line query dots.
    """
    curve_style = requested_curve_style
    if curve_style == "mixed":
        curve_style = "pathfinder" if (seed // 2) % 2 == 0 else "bezier"
    if curve_style != "pathfinder":
        raise ValueError("Pathfinder same_line replay requires Pathfinder style")
    rng = random.Random(f"{seed}:{level}:same_line:pathfinder:geom{geometry_salt}")
    level_config = rope_engine.LEVELS[level]
    effective_color = color_mode_override
    effective_line = line_style_override
    effective_outline = rope_outline_override
    if line_style_override == "outlined" and effective_outline is None:
        effective_outline = True
    scheduled_color, scheduled_style, scheduled_outline = scheduled_appearance(seed)
    if effective_color is None:
        effective_color = scheduled_color
    if effective_line is None:
        effective_line = scheduled_style
    if effective_outline is None:
        effective_outline = scheduled_outline
    minimum_target_crossings = {
        "easy": 0,
        "medium": 2,
        "hard": 3,
        "xhard": 4,
    }[level]
    best = None
    best_attempt = None
    best_quality = -float("inf")
    attempt_budget = max(1, scene_retries)
    for attempt in range(attempt_budget):
        scene = rope_engine.build_scene(
            level_config,
            size,
            size,
            rng,
            min_cross_angle=min_cross_angle,
            curve_style="pathfinder",
            color_mode_override=effective_color,
            line_style_override=effective_line,
            qa_type="same_line",
            marker_layout_override=marker_layout_override,
            rope_outline_override=effective_outline,
        )
        selection_qa = rope_engine.make_qa(
            scene,
            "same_line",
            rng,
            min_target_crossings=minimum_target_crossings,
        )
        target_min_angle = 180.0
        for index in selection_qa["target_indices"]:
            if scene.curve_min_angle:
                target_min_angle = min(target_min_angle, scene.curve_min_angle[index])
        spec = rope_engine.PATHFINDER_LEVELS[level]
        target_metrics = [
            scene.curve_metrics[index] for index in selection_qa["target_indices"]
        ]
        quality = sum(
            rope_engine.rope_metric_score(metric, spec, min_cross_angle)
            for metric in target_metrics
        ) / len(target_metrics)
        quality += (
            2.0
            if target_min_angle >= min_cross_angle
            else target_min_angle / max(1.0, min_cross_angle)
        )
        if quality > best_quality:
            best_quality, best, best_attempt = quality, scene, attempt + 1
        average_crossings = sum(metric["crossings"] for metric in target_metrics) / len(
            target_metrics
        )
        average_ratio = sum(
            metric["arc_over_chord"] for metric in target_metrics
        ) / len(target_metrics)
        average_loops = sum(metric["large_loops"] for metric in target_metrics) / len(
            target_metrics
        )
        if (
            target_min_angle >= min_cross_angle
            and average_crossings >= 0.75 * spec.target_crossings[0]
            and average_ratio >= 0.85 * spec.arc_ratio[0]
            and average_loops >= max(0, 0.6 * spec.target_loops[0])
        ):
            break
    if best is None:
        raise RuntimeError("Pathfinder scene replay produced no scene")
    return best, best_attempt, attempt + 1


def _pathfinder_same_line_generate(
    level: str,
    seed: int,
    size: int,
    min_cross_angle: float,
    scene_retries: int,
    curve_style: str,
    color_mode_override: str | None,
    line_style_override: str | None,
    marker_layout_override: str | None,
    rope_outline_override: bool | None,
    geometry_salt: int,
):
    scene, accepted_scene_attempt, scene_attempts_executed = (
        _pathfinder_same_line_scene(
            level,
            seed,
            size,
            min_cross_angle,
            scene_retries,
            curve_style,
            color_mode_override,
            line_style_override,
            marker_layout_override,
            rope_outline_override,
            geometry_salt,
        )
    )
    qa = make_same_line_qa(
        scene,
        random.Random(
            f"{seed}:{level}:{_SAME_LINE_STREAM}:geom{geometry_salt}"
        ),
        {"easy": 0, "medium": 2, "hard": 3, "xhard": 4}[level],
    )
    image = rope_engine.render(scene, qa, size, size)
    validity_policy = {
        "require_endpoint_clean": True,
        "target_min_cross_angle_deg": min_cross_angle,
        "scope": "queried_paths",
        "backend_attempts_observable": True,
        "scene_attempt_budget": max(1, scene_retries),
        "accepted_scene_attempt_index": accepted_scene_attempt,
        "scene_attempts_executed": scene_attempts_executed,
        "same_line_query": qa["same_line_query_policy"],
    }
    meta = _bezier_meta(
        scene,
        qa,
        level,
        "same_line",
        seed,
        curve_style,
        geometry_salt,
        scene_attempts_executed,
        validity_policy,
        size,
    )
    meta.update(
        {
            "curve_style": "pathfinder",
            "geometry_backend": "pathfinder",
        }
    )
    valid = bool(
        meta["endpoint_clean"]
        and (
            meta["target_min_angle"] is None
            or meta["target_min_angle"] >= min_cross_angle
        )
    )
    meta.update(
        {
            "valid": valid,
        }
    )
    return image, meta


def _pathfinder_generate(
    level: str,
    qa_type: str,
    seed: int,
    size: int,
    min_cross_angle: float,
    scene_retries: int,
    curve_style: str,
    color_mode_override: str | None,
    line_style_override: str | None,
    marker_layout_override: str | None,
    rope_outline_override: bool | None,
    geometry_salt: int,
):
    if qa_type == "same_line":
        return _pathfinder_same_line_generate(
            level,
            seed,
            size,
            min_cross_angle,
            scene_retries,
            curve_style,
            color_mode_override,
            line_style_override,
            marker_layout_override,
            rope_outline_override,
            geometry_salt,
        )
    image, original_meta = rope_engine.generate(
        level,
        qa_type,
        seed,
        size=size,
        min_cross_angle=min_cross_angle,
        scene_retries=scene_retries,
        curve_style=curve_style,
        color_mode_override=color_mode_override,
        line_style_override=line_style_override,
        marker_layout_override=marker_layout_override,
        rope_outline_override=rope_outline_override,
        geometry_salt=geometry_salt,
    )
    meta = dict(original_meta)
    valid = bool(
        meta["endpoint_clean"]
        and (
            meta["target_min_angle"] is None
            or meta["target_min_angle"] >= min_cross_angle
        )
    )
    meta.update(
        {
            "geometry_backend": "pathfinder",
            # The fixed Pathfinder engine does not expose its internal attempt count.
            # Do not fabricate a value; the wrapper records observable invocations.
            "geometry_attempts": None,
            "validity_policy": {
                "require_endpoint_clean": True,
                "target_min_cross_angle_deg": min_cross_angle,
                "scope": "queried_paths",
                "backend_attempts_observable": False,
            },
            "valid": valid,
        }
    )
    return image, meta


def generate(
    level: str,
    qa_type: str,
    seed: int,
    size: int = 448,
    min_cross_angle: float = 25.0,
    scene_retries: int = 6,
    curve_style: str = "pathfinder",
    color_mode_override: str | None = None,
    line_style_override: str | None = None,
    marker_layout_override: str | None = None,
    rope_outline_override: bool | None = None,
    geometry_salt: int = 0,
):
    """Generate one curve-engine sample while preserving Pathfinder byte behavior."""
    if level not in LEVEL_ORDER:
        raise ValueError(f"unknown level: {level}")
    if qa_type not in {"one_marker", "same_line", "ordered", "all_pairs"}:
        raise ValueError(f"unknown QA type: {qa_type}")
    resolved_style = _resolved_style(seed, curve_style)
    if resolved_style == "pathfinder":
        return _pathfinder_generate(
            level,
            qa_type,
            seed,
            size,
            min_cross_angle,
            scene_retries,
            curve_style,
            color_mode_override,
            line_style_override,
            marker_layout_override,
            rope_outline_override,
            geometry_salt,
        )

    if marker_layout_override not in {None, "perimeter_arclength"}:
        raise ValueError(
            "Bezier geometry supports only marker_layout=perimeter_arclength"
        )
    angle_floor = max(30.0, float(min_cross_angle))
    if angle_floor > 90.0:
        raise ValueError("Bezier crossing-angle floor cannot exceed 90 degrees")
    scene, attempts, policy = build_bezier_scene(
        level,
        qa_type,
        seed,
        size,
        angle_floor,
        scene_retries,
        geometry_salt,
        color_mode_override,
        line_style_override,
        rope_outline_override,
    )
    qa_rng = random.Random(f"{seed}:{level}:{qa_type}:{_BEZIER_STREAM}:qa")
    min_target_crossings = {"easy": 0, "medium": 2, "hard": 3, "xhard": 4}[level]
    qa = (
        make_same_line_qa(scene, qa_rng, min_target_crossings)
        if qa_type == "same_line"
        else rope_engine.make_qa(
            scene,
            qa_type,
            qa_rng,
            min_target_crossings=min_target_crossings,
        )
    )
    image = rope_engine.render(scene, qa, size, size)
    meta = _bezier_meta(
        scene,
        qa,
        level,
        qa_type,
        seed,
        curve_style,
        geometry_salt,
        attempts,
        policy,
        size,
    )
    return image, meta
