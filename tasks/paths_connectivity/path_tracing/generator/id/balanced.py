"""Balanced Path Tracing generation.
"""

from __future__ import annotations

import random

from PIL import Image
from visro.seeding import seed_text

from ..answers import structured_answer
from .curve_engine import BEZIER_GATE_CONFIG, SAME_LINE_QUERY_POLICY, generate
from .path_records import GENERATOR_VERSION, SHORTCUT_CONTROL_PROFILE, attempt_provenance, crossing_threshold, is_valid, normalized_pairs
from .schedule_helpers import CURVE_STYLES, DIFFICULTIES, QA_TYPES, balanced_schedule

# fixed RNG namespace of the appearance schedule (changing it changes every item)
_APPEARANCE_STREAM = seed_text(505933330083480808186756642163076657)
BALANCED_ENTRYPOINT = 'tasks.paths_connectivity.path_tracing.generator.id.balanced'
APPEARANCE_TARGETS = {
    "color_mode": {"distinct": 0.10, "family": 0.20, "uniform": 0.70},
    "line_style": {"solid": 0.50, "dashed": 0.50},
    "rope_outline": {"outlined": 0.20, "plain": 0.80},
}


def _apportioned_values(
    total: int,
    weights: dict[str, float],
    seed: str,
) -> list[str]:
    """Return deterministically shuffled values with largest-remainder quotas."""
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        weights,
        key=lambda name: (raw[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    values = [name for name, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(values)
    return values


def appearance_schedule(total: int, seed: int) -> list[tuple]:
    """Add per-style/difficulty appearance quotas to the balanced rope-engine schedule."""
    base = balanced_schedule(total, seed)
    appearances: dict[int, tuple[str, str, bool]] = {}
    for style in CURVE_STYLES:
        for difficulty in DIFFICULTIES:
            indices = [
                index
                for index, row in enumerate(base)
                if row[0] == style and row[1] == difficulty
            ]
            namespace = f"{seed}:{style}:{difficulty}:{_APPEARANCE_STREAM}"
            colors = _apportioned_values(
                len(indices),
                APPEARANCE_TARGETS["color_mode"],
                f"{namespace}:color",
            )
            strokes = _apportioned_values(
                len(indices),
                APPEARANCE_TARGETS["line_style"],
                f"{namespace}:stroke",
            )
            outlines = _apportioned_values(
                len(indices),
                APPEARANCE_TARGETS["rope_outline"],
                f"{namespace}:outline",
            )
            for index, color, stroke, outline in zip(
                indices, colors, strokes, outlines
            ):
                appearances[index] = (color, stroke, outline == "outlined")
    return [(*row, *appearances[index]) for index, row in enumerate(base)]


def resolved_config(
    *,
    count: int,
    size: int,
    scene_retries: int,
    geometry_retries: int,
    seed_retries: int,
) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "generator_entrypoint": BALANCED_ENTRYPOINT,
        "requested_count": count,
        "size": size,
        "scene_retries": scene_retries,
        "geometry_retries": geometry_retries,
        "seed_retries": seed_retries,
        "curve_styles": list(CURVE_STYLES),
        "difficulties": list(DIFFICULTIES),
        "qa_types": list(QA_TYPES),
        "schedule": "seeded_balanced_style_x_difficulty_x_qa_layout",
        "appearance_schedule": {
            "scope": "curve_style_x_difficulty",
            "assignment": "seeded_largest_remainder_marginals",
            "targets": APPEARANCE_TARGETS,
        },
        "crossing_angle_floor_deg": {
            "bezier": 30.0,
            "pathfinder": 25.0,
        },
        "bezier_gate_config": BEZIER_GATE_CONFIG,
        "same_line_query_policy": SAME_LINE_QUERY_POLICY,
        "shortcut_control_profile": SHORTCUT_CONTROL_PROFILE,
        "source_profile": f"source_rgb{size}_square_path",
        "model_input_profile": "rgb448_bicubic",
        "text_profile": "short_answer_256",
    }


def _record_from_meta(
    idx: int,
    style: str,
    difficulty: str,
    qa_type: str,
    marker_layout: str | None,
    scheduled_color: str,
    scheduled_style: str,
    scheduled_outline: bool,
    seed: int,
    image,
    meta: dict,
    config: dict,
    config_sha256: str,
    retry_provenance: dict,
) -> dict:
    sample_id = f"ptv3_{idx:08d}"
    image_rel = f"images/{sample_id}.png"
    target_curves = [meta["curves"][i] for i in meta["target_indices"]]
    answer_sequence = (
        [str(curve["end_label"]) for curve in target_curves]
        if qa_type == "ordered"
        else None
    )
    pairs = normalized_pairs(meta["pairs"])
    return {
        "id": sample_id,
        "task_type": "visual_tracking",
        "subtype": "path_tracing",
        "image": image_rel,
        "question": meta["question"],
        "answer": {
            "text": meta["answer"],
            "boxed": meta["answer"],
            "qa_type": qa_type,
            "pairs": pairs,
        },
        "answer_text": meta["answer"],
        "qa": {
            "type": qa_type,
            "answer_type": "blank",
            "answer_text": meta["answer"],
            "options": [],
            "structured_answer": structured_answer(
                qa_type, meta["answer"], answer_sequence
            ),
            "grading": {
                "method": f"path_tracing_{qa_type}",
                "version": "1",
                "parameters": {},
            },
            "pairs": pairs,
            **(
                {"answer_sequence": answer_sequence}
                if answer_sequence is not None
                else {}
            ),
            "target_indices": meta["target_indices"],
            "dot_markers": meta["dot_markers"],
        },
        "render": {
            "source": {
                "profile": config["source_profile"],
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            },
            "model_input_profile": config["model_input_profile"],
        },
        "generation": {
            "generator_version": GENERATOR_VERSION,
            "generator_entrypoint": BALANCED_ENTRYPOINT,
            "config_sha256": f"sha256:{config_sha256}",
            "resolved_config": config,
            "seed": seed,
            "sample_seed": seed + idx,
            "index": idx,
            "retry": retry_provenance,
        },
        "visual_primitives": {
            "type": "point_trace",
            "coordinate_space": "image_pixels",
            "traces": [
                {
                    "curve_index": meta["target_indices"][i],
                    "start_marker_text": curve["start_label"],
                    "end_marker_text": curve["end_label"],
                    "points": [{"point": point} for point in curve["trace"]],
                }
                for i, curve in enumerate(target_curves)
            ],
        },
        "path_tracing": {
            **meta,
            "difficulty": difficulty,
            "valid": is_valid(meta),
            "min_crossing_angle_threshold": crossing_threshold(meta),
            "generator_version": GENERATOR_VERSION,
            "generator_entrypoint": BALANCED_ENTRYPOINT,
            "retry": retry_provenance,
        },
        "eval_balance": {
            "seed": seed,
            "scheduled_curve_style": style,
            "scheduled_difficulty": difficulty,
            "scheduled_qa_type": qa_type,
            "scheduled_marker_layout": marker_layout,
            "scheduled_color_mode": scheduled_color,
            "scheduled_line_style": scheduled_style,
            "scheduled_rope_outline": scheduled_outline,
        },
    }


def generate_one(
    *,
    idx: int,
    row: tuple,
    seed: int,
    size: int,
    scene_retries: int,
    geometry_retries: int,
    seed_retries: int,
    config: dict,
    config_sha256: str,
) -> tuple[dict, Image.Image]:
    """One appearance_schedule row at global index idx -> (record, image)."""
    (
        style,
        difficulty,
        qa_type,
        marker_layout,
        scheduled_color,
        scheduled_style,
        scheduled_outline,
    ) = row
    sample_seed = seed + idx
    generated = None
    last_error = None
    accepted_seed_bump = None
    for seed_bump in range(seed_retries):
        geometry_seed = sample_seed + seed_bump * 1_000_003
        for geometry_salt in range(geometry_retries):
            try:
                candidate = generate(
                    difficulty,
                    qa_type,
                    geometry_seed,
                    size=size,
                    min_cross_angle=30.0 if style == "bezier" else 25.0,
                    scene_retries=scene_retries,
                    curve_style=style,
                    color_mode_override=scheduled_color,
                    line_style_override=scheduled_style,
                    marker_layout_override=marker_layout,
                    rope_outline_override=scheduled_outline,
                    geometry_salt=geometry_salt,
                )
                if not is_valid(candidate[1]):
                    last_error = RuntimeError(
                        "generated scene failed its serialized validity policy"
                    )
                    continue
                generated = candidate
                accepted_seed_bump = seed_bump
                break
            except RuntimeError as error:
                last_error = error
        if generated is not None:
            break
    if generated is None:
        raise RuntimeError(
            f"idx={idx} {style}/{difficulty}/{qa_type} failed after "
            f"{seed_retries} seeds x {geometry_retries} geometry retries: "
            f"{last_error}"
        )

    image, meta = generated
    record = _record_from_meta(
        idx,
        style,
        difficulty,
        qa_type,
        marker_layout,
        scheduled_color,
        scheduled_style,
        scheduled_outline,
        seed,
        image,
        meta,
        config,
        config_sha256,
        attempt_provenance(
            meta,
            geometry_salt=geometry_salt,
            geometry_retries=geometry_retries,
            scene_retries=scene_retries,
            seed_bump=accepted_seed_bump,
        ),
    )
    record["eval_balance"].update(
        {
            "geometry_salt": meta["geometry_salt"],
            "geometry_seed_bump": accepted_seed_bump,
        }
    )
    return record, image
