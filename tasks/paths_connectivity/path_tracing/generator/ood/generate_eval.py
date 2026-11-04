"""Out-of-distribution Path Tracing schedule and per-item producer."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from PIL import Image
from visro.seeding import canonical

from ..answers import structured_answer
from . import BASE_GENERATOR_VERSION, DATASET_ROLE, DATASET_SPLIT, GENERATOR_ENTRYPOINT, GENERATOR_VERSION, ID_PREFIX, RNG_NAMESPACE
from .engine import CATMULL_ROM_LAYOUT, CONDITIONS, HARMONIC_FLOW, ROUNDED_ORTHOGONAL, QA_TYPES, build_current_xhard_case, build_custom_case, build_icon_clutter_layer, case_metadata, render_case

DEFAULT_COUNT = 600
BALANCE_BLOCK_SIZE = 40
DEFAULT_MASTER_SEED = 20260909
DEFAULT_SEED_RETRIES = 8
SEED_HIGH_PREFIX = 0x6000_0000_0000_0000
SEED_LOW_MASK = 0x0FFF_FFFF_FFFF_FFFF
VARIANTS = {
    "stroke_ood": ("paddle_beaded", "halo_core"),
    "program_harmonic_flow": (HARMONIC_FLOW,),
    "program_rounded_orthogonal": (ROUNDED_ORTHOGONAL,),
    "endpoint_layout_ood": ("inner_ring", "four_islands"),
    "background_ood": ("openmoji_clutter", "silhouette_clutter"),
}


def _qa_constraints(
    qa_type: str,
    *,
    block: int,
    half: int,
    condition_index: int,
) -> dict[str, Any]:
    if qa_type == "same_line":
        want_yes = (half + condition_index) % 2 == 0
        return {"same_line_answer": "yes" if want_yes else "no"}
    if qa_type == "one_marker":
        query_start = (half + condition_index) % 2 == 1
        return {"one_marker_query_side": "start" if query_start else "end"}
    if qa_type == "ordered":
        return {
            "ordered_target_count": (
                2 + ((block // 2 + 2 * half + condition_index) % 4)
            )
        }
    return {}


def release_schedule(
    count: int = DEFAULT_COUNT,
    seed: int = DEFAULT_MASTER_SEED,
) -> list[dict[str, Any]]:
    """Return shuffled 20-row pages grouped into exact 40-row balance blocks."""

    if count <= 0 or count % BALANCE_BLOCK_SIZE:
        raise ValueError(
            f"count must be positive and divisible by {BALANCE_BLOCK_SIZE}; "
            f"received {count}"
        )
    rows: list[dict[str, Any]] = []
    for block in range(count // BALANCE_BLOCK_SIZE):
        for half in range(2):
            page_rows = []
            for condition_index, condition in enumerate(CONDITIONS):
                variants = VARIANTS[condition]
                for qa_index, qa_type in enumerate(QA_TYPES):
                    variant_index = (
                        (half + block + condition_index) % len(variants)
                        if len(variants) > 1
                        else 0
                    )
                    variant = variants[variant_index]
                    semantic_key = (
                        f"block={block}|half={half}|condition={condition}|"
                        f"qa={qa_type}|variant={variant}"
                    )
                    page_rows.append(
                        {
                            "block": block,
                            "half": half,
                            "review_page": 2 * block + half,
                            "condition_index": condition_index,
                            "qa_index": qa_index,
                            "condition": condition,
                            "variant": variant,
                            "qa_type": qa_type,
                            "qa_constraints": _qa_constraints(
                                qa_type,
                                block=block,
                                half=half,
                                condition_index=condition_index,
                            ),
                            "semantic_key": semantic_key,
                        }
                    )
            random.Random(
                f"{RNG_NAMESPACE}:{seed}:schedule:block={block}:half={half}"
            ).shuffle(page_rows)
            for page_position, row in enumerate(page_rows):
                row["page_position"] = page_position
            rows.extend(page_rows)
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def _semantic_seed(
    master_seed: int,
    row: dict[str, Any],
    domain: str,
    attempt: int = 0,
) -> int:
    payload = (
        f"{RNG_NAMESPACE}|master={master_seed}|{row['semantic_key']}|"
        f"domain={domain}|attempt={attempt}"
    ).encode()
    low = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & SEED_LOW_MASK
    return SEED_HIGH_PREFIX | low


def _build_case(row: dict[str, Any], sample_seed: int, size: int):
    condition = row["condition"]
    qa_type = row["qa_type"]
    variant = row["variant"]
    kwargs = {
        "rng_namespace": RNG_NAMESPACE,
        "qa_constraints": row["qa_constraints"],
    }
    if condition in {"stroke_ood", "background_ood"}:
        return build_current_xhard_case(qa_type, sample_seed, size, **kwargs)
    if condition == "program_harmonic_flow":
        return build_custom_case(
            qa_type,
            sample_seed,
            size,
            program_family=HARMONIC_FLOW,
            marker_layout="perimeter_arclength",
            **kwargs,
        )
    if condition == "program_rounded_orthogonal":
        return build_custom_case(
            qa_type,
            sample_seed,
            size,
            program_family=ROUNDED_ORTHOGONAL,
            marker_layout="perimeter_arclength",
            **kwargs,
        )
    if condition == "endpoint_layout_ood":
        return build_custom_case(
            qa_type,
            sample_seed,
            size,
            program_family=CATMULL_ROM_LAYOUT,
            marker_layout=variant,
            **kwargs,
        )
    raise ValueError(f"unknown OOD condition: {condition}")


def _transform_point(point: list[int], size: int, transform: int) -> tuple[int, int]:
    x, y = int(point[0]), int(point[1])
    edge = size - 1
    transforms = (
        (x, y),
        (edge - y, x),
        (edge - x, edge - y),
        (y, edge - x),
        (edge - x, y),
        (x, edge - y),
        (y, x),
        (edge - y, edge - x),
    )
    return transforms[transform]


def _geometry_view_sha256(
    meta: dict[str, Any],
    size: int,
    transform: int,
) -> str:
    curves = []
    for curve in meta["curves"]:
        trace = tuple(
            _transform_point(point, size, transform) for point in curve["trace"]
        )
        curves.append(min(trace, tuple(reversed(trace))))
    payload = canonical(sorted(curves)).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _geometry_fingerprint(meta: dict[str, Any], size: int) -> str:
    """Canonicalize unlabeled traces under reversal, permutation, and D4."""

    return min(_geometry_view_sha256(meta, size, transform) for transform in range(8))


def _record(
    *,
    row: dict[str, Any],
    built,
    meta: dict[str, Any],
    master_seed: int,
    sample_seed: int,
    background_seed: int | None,
    attempted_seeds: list[dict[str, Any]],
    accepted_retry: int,
    config: dict[str, Any],
    config_sha256: str,
    size: int,
) -> dict[str, Any]:
    sample_id = f"{ID_PREFIX}{row['index']:08d}"
    qa_type = row["qa_type"]
    answer_sequence = None
    if qa_type == "ordered":
        answer_sequence = [
            built.scene.curves[index].ei for index in built.qa["target_indices"]
        ]
    structured = structured_answer(
        qa_type,
        built.qa["answer"],
        answer_sequence,
    )
    target_curves = [
        built.scene.curves[index] for index in built.qa["target_indices"]
    ]
    return {
        "id": sample_id,
        "task_type": "visual_tracking",
        "subtype": "path_tracing_ood",
        "eval_task": "path_tracing_ood",
        "dataset_role": DATASET_ROLE,
        "split": DATASET_SPLIT,
        "difficulty": "xxhard",
        "image": f"images/{sample_id}.png",
        "question": built.qa["question"],
        "answer_text": built.qa["answer"],
        "answer": {
            "text": built.qa["answer"],
            "boxed": built.qa["answer"],
            "qa_type": qa_type,
            "pairs": meta["pairs"],
        },
        "qa": {
            "type": qa_type,
            "answer_type": "blank",
            "answer_text": built.qa["answer"],
            "options": [],
            "structured_answer": structured,
            "grading": {
                "method": f"path_tracing_{qa_type}",
                "version": "1",
                "parameters": {},
            },
            "pairs": meta["pairs"],
            "target_indices": built.qa["target_indices"],
            "dot_markers": built.qa.get("dot_markers"),
            **(
                {"answer_sequence": answer_sequence}
                if answer_sequence is not None
                else {}
            ),
        },
        "render": {
            "source": {
                "profile": f"source_rgb{size}_square_path_ood",
                "width": size,
                "height": size,
                "mode": "RGB",
            },
            "model_input_profile": "rgb448_bicubic",
        },
        "generation": {
            "generator_version": GENERATOR_VERSION,
            "generator_entrypoint": GENERATOR_ENTRYPOINT,
            "base_generator_version": BASE_GENERATOR_VERSION,
            "config_sha256": f"sha256:{config_sha256}",
            "resolved_config": config,
            "master_seed": master_seed,
            "sample_seed": sample_seed,
            "background_seed": background_seed,
            "index": row["index"],
            "semantic_key": row["semantic_key"],
            "seed_retry": {
                "budget": config["seed_retries"],
                "accepted_retry_index": accepted_retry,
                "attempts": attempted_seeds,
            },
        },
        "ood_schedule": row,
        "visual_primitives": {
            "type": "point_trace",
            "coordinate_space": "image_pixels",
            "traces": [
                {
                    "curve_index": built.qa["target_indices"][position],
                    "start_marker_text": curve.si,
                    "end_marker_text": curve.ei,
                    "points": [
                        {"point": [round(point[0]), round(point[1])]}
                        for point in curve.trace
                    ],
                }
                for position, curve in enumerate(target_curves)
            ],
        },
        "path_tracing_ood": meta,
    }


def generate_one(
    *,
    row: dict[str, Any],
    master_seed: int,
    size: int,
    seed_retries: int,
    config: dict[str, Any],
    config_sha256: str,
) -> tuple[dict[str, Any], Image.Image]:
    """One release_schedule row -> (record, image)."""
    attempted_seeds: list[dict[str, Any]] = []
    accepted = None
    last_error: Exception | None = None
    for retry in range(seed_retries):
        sample_seed = _semantic_seed(master_seed, row, "sample", retry)
        try:
            built = _build_case(row, sample_seed, size)
            background_seed = None
            clutter_layer = None
            background_meta = None
            if row["condition"] == "background_ood":
                background_seed = _semantic_seed(
                    master_seed,
                    row,
                    "background",
                    retry,
                )
                clutter_layer, background_meta = build_icon_clutter_layer(
                    row["variant"],
                    background_seed,
                    built.scene,
                    built.qa,
                    size,
                    rng_namespace=RNG_NAMESPACE,
                )
            stroke_profile = (
                row["variant"] if row["condition"] == "stroke_ood" else "current"
            )
            image = render_case(
                built.scene,
                built.qa,
                size,
                stroke_profile=stroke_profile,
                clutter_layer=clutter_layer,
            )
            meta = case_metadata(
                built,
                condition=row["condition"],
                variant=row["variant"],
                seed=sample_seed,
                background=background_meta,
            )
            meta["difficulty"] = "xxhard"
            meta["ood_tier"] = "xxhard"
            meta["base_difficulty"] = "xhard"
            meta["generator_version"] = GENERATOR_VERSION
            meta["base_generator_version"] = BASE_GENERATOR_VERSION
            meta["geometry_fingerprint_d4_sha256"] = _geometry_fingerprint(meta, size)
            if not meta["validity"]["valid"]:
                raise RuntimeError("serialized validity failed")
            attempted_seeds.append(
                {"retry_index": retry, "sample_seed": sample_seed, "status": "accepted"}
            )
            accepted = (
                retry,
                sample_seed,
                background_seed,
                built,
                meta,
                image,
            )
            break
        except (RuntimeError, ValueError) as error:
            last_error = error
            attempted_seeds.append(
                {
                    "retry_index": retry,
                    "sample_seed": sample_seed,
                    "status": "rejected",
                    "reason": str(error),
                }
            )
    if accepted is None:
        raise RuntimeError(
            f"index {row['index']} ({row['semantic_key']}) exhausted "
            f"{seed_retries} seed attempts; last error: {last_error}"
        )

    (
        accepted_retry,
        sample_seed,
        background_seed,
        built,
        meta,
        image,
    ) = accepted
    record = _record(
        row=row,
        built=built,
        meta=meta,
        master_seed=master_seed,
        sample_seed=sample_seed,
        background_seed=background_seed,
        attempted_seeds=attempted_seeds,
        accepted_retry=accepted_retry,
        config=config,
        config_sha256=config_sha256,
        size=size,
    )
    return record, image
