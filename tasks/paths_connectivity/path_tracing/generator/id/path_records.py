"""Path Tracing record and validity helpers.
"""

from __future__ import annotations

from .curve_engine import ENDPOINT_PAIRING_POLICY, SAME_LINE_QUERY_POLICY

GENERATOR_VERSION = "path-tracing"
SHORTCUT_CONTROL_PROFILE = {
    "bezier_endpoint_pairing": ENDPOINT_PAIRING_POLICY,
    "same_line_query": SAME_LINE_QUERY_POLICY,
}


def normalized_pairs(pairs: list[dict]) -> list[dict]:
    """One stable pair shape for both curve backends (Bezier and rope)."""
    normalized = []
    for pair_index, pair in enumerate(pairs):
        start = pair.get(
            "start_marker_text",
            pair.get("start_label", pair.get("start")),
        )
        end = pair.get(
            "end_marker_text",
            pair.get("end_label", pair.get("end")),
        )
        normalized.append(
            {
                "curve_index": pair.get("curve_index", pair_index),
                "start_label": start,
                "end_label": end,
                "start_marker_text": start,
                "end_marker_text": end,
            }
        )
    return normalized


def attempt_provenance(
    meta: dict,
    *,
    geometry_salt: int,
    geometry_retries: int,
    scene_retries: int,
    seed_bump: int = 0,
) -> dict:
    """Describe wrapper and backend retries without inventing hidden counts."""
    wrapper_invocations = seed_bump * geometry_retries + geometry_salt + 1
    if meta.get("geometry_backend") == "bezier_geometry":
        inner_budget = max(8, scene_retries * 8)
        accepted_scene_attempts = int(meta["geometry_attempts"])
        scene_attempts_total = (
            wrapper_invocations - 1
        ) * inner_budget + accepted_scene_attempts
        return {
            "seed_retry_index": seed_bump,
            "geometry_retry_index": geometry_salt,
            "wrapper_geometry_invocations": wrapper_invocations,
            "backend_scene_attempts_in_accepted_invocation": accepted_scene_attempts,
            "scene_attempts_total": scene_attempts_total,
            "curve_candidate_attempts_in_accepted_wrapper_invocation": (
                meta["validity_policy"]["curve_candidate_attempts_total"]
            ),
            "curve_candidate_attempts_across_failed_wrapper_invocations": None,
            "backend_attempts_observable": True,
        }
    validity_policy = meta.get("validity_policy", {})
    if validity_policy.get("backend_attempts_observable"):
        accepted_scene_attempts = int(meta["geometry_attempts"])
        return {
            "seed_retry_index": seed_bump,
            "geometry_retry_index": geometry_salt,
            "wrapper_geometry_invocations": wrapper_invocations,
            "backend_scene_attempts_in_accepted_invocation": (accepted_scene_attempts),
            # The accepted current Pathfinder same_line replay exposes its count,
            # but the fixed backend does not expose attempts spent in failed
            # wrapper invocations. Do not manufacture an aggregate.
            "scene_attempts_total": None,
            "curve_candidate_attempts_in_accepted_wrapper_invocation": None,
            "curve_candidate_attempts_across_failed_wrapper_invocations": None,
            "backend_attempts_observable": True,
        }
    return {
        "seed_retry_index": seed_bump,
        "geometry_retry_index": geometry_salt,
        "wrapper_geometry_invocations": wrapper_invocations,
        "backend_scene_attempts_in_accepted_invocation": None,
        "scene_attempts_total": None,
        "curve_candidate_attempts_in_accepted_wrapper_invocation": None,
        "curve_candidate_attempts_across_failed_wrapper_invocations": None,
        "backend_attempts_observable": False,
    }


def crossing_threshold(meta: dict) -> float:
    policy = meta.get("validity_policy", {})
    if meta.get("curve_style") == "bezier":
        return float(policy.get("global_min_cross_angle_deg", 30.0))
    return float(policy.get("target_min_cross_angle_deg", 25.0))


def is_valid(meta: dict) -> bool:
    if "valid" in meta:
        return bool(meta["valid"])
    threshold = crossing_threshold(meta)
    if meta.get("curve_style") == "bezier":
        angle = meta.get("min_cross_angle")
    else:
        angle = meta.get("target_min_angle")
    return bool(meta.get("endpoint_clean") and (angle is None or angle >= threshold))
