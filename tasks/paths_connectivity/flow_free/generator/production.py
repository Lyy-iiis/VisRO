"""Split-aware Flow Free production generator.

ID splits use the easy/medium/hard/xhard semantic generator and a 50/50
board-shared versus per-colour material schedule.  The 600-record OOD split
contains five equally weighted, genuinely shifted strata:

* an alternative Prim path program with the ID renderer;
* Backbite geometry with rounded/sine pipes, patchwork, and pictograms;
* the Prim program with that same curved appearance bundle;
* held-out width/alpha/texture compositions sampled once per board;
* the same held-out compositions sampled independently per colour group.

The OOD split contains no unshifted control records.
"""
from __future__ import annotations
from visro.seeding import seed_text

import copy
import dataclasses
import hashlib
import pathlib
from collections import Counter
from typing import Any, Sequence

from PIL import Image

from . import (
    backbite, board_core, board_render, length_ladder, ood_core, pipe_colors,
    pipe_styles, style_schedule,
)

RNG_NAMESPACE = seed_text(642923205654485854333923836032318266850434673777603037257521)  # fixed RNG namespace (_stable_int)
SIZE = 448
SPLITS = ("train", "eval", "val", "ood")
TOTALS = {"train": 100_000, "eval": 600, "val": 100, "ood": 600}
ROLE_BY_SPLIT = {
    "train": "train_id",
    "eval": "eval_id",
    "val": "val_id",
}
OOD_STRATA = (
    "program_only",
    "curved_appearance_backbite",
    "curved_appearance_prim",
    "material_board_shared",
    "material_per_pipe",
)
OOD_SOURCE_ARM = {
    "program_only": "program_only",
    "curved_appearance_backbite": "appearance_only",
    "curved_appearance_prim": "combined",
    "material_board_shared": "length_dose_control",
    "material_per_pipe": "length_dose_control",
}
MATERIAL_OOD_STRATA = {"material_board_shared", "material_per_pipe"}
CURVED_OOD_STRATA = {
    "curved_appearance_backbite", "curved_appearance_prim"
}
DIFFICULTY_RANK = {
    "easy": 1, "medium": 2, "hard": 3, "xhard": 4, "xxhard": 5
}
TRAIN_TIERS = board_core.TRAIN_TIERS
QUESTION_FORMS = board_core.QUESTION_FORMS
OOD_TIER = board_core.OOD_TIER


@dataclasses.dataclass(frozen=True)
class OODCondition:
    index: int
    stratum: str
    question_form: str
    form_index: int
    repeat: int
    source_case_index: int
    source_arm: str
    pair_seed: int


def _stable_int(*parts: Any) -> int:
    raw = "|".join(map(str, (RNG_NAMESPACE,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def ood_condition(seed: int, index: int, total: int = 600) -> OODCondition:
    """Cross five OOD strata with four forms over a 20-record supercell."""
    if total <= 0 or total % (len(OOD_STRATA) * len(board_core.QUESTION_FORMS)):
        raise ValueError("OOD total must be a positive multiple of 20")
    if not 0 <= index < total:
        raise ValueError("OOD index is outside the declared split total")
    stratum = OOD_STRATA[index % len(OOD_STRATA)]
    form_index = (index // len(OOD_STRATA)) % len(board_core.QUESTION_FORMS)
    question_form = board_core.QUESTION_FORMS[form_index]
    occurrence = index // (len(OOD_STRATA) * len(board_core.QUESTION_FORMS))
    repeat = occurrence % 4
    source_case_index = form_index * 4 + repeat
    return OODCondition(
        index=index,
        stratum=stratum,
        question_form=question_form,
        form_index=form_index,
        repeat=repeat,
        source_case_index=source_case_index,
        source_arm=OOD_SOURCE_ARM[stratum],
        # Every official OOD record gets an independent semantic pair.  The
        # small 0..15 source case is only its balanced wording/nuisance cell.
        pair_seed=_stable_int(seed, "ood_pair", index),
    )


def condition(split: str, seed: int, index: int, total: int | None = None) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"unknown production split: {split!r}")
    total = TOTALS[split] if total is None else total
    if not 0 <= index < total:
        raise ValueError("index is outside the declared split total")
    if split == "ood":
        value = ood_condition(seed, index, total)
        return {
            "split": split,
            "tier": "xxhard",
            "question_form": value.question_form,
            "ood_stratum": value.stratum,
            "source_case_index": value.source_case_index,
            "source_arm": value.source_arm,
        }
    role = ROLE_BY_SPLIT[split]
    spec = board_core.schedule(index, role)
    return {
        "split": split,
        "tier": spec.tier,
        "question_form": spec.question_form,
        "layout_family": spec.layout_family,
        "style_assignment_mode": style_schedule.id_assignment_mode(spec),
    }


def _rebuild(
    record: dict[str, Any],
) -> tuple[
    board_core.BaseBoard,
    list[tuple[str, tuple[int, int, int]]],
    board_core.Labeling | None,
    board_render.Appearance,
]:
    meta = record["meta"]
    rows = sorted(meta["segments"], key=lambda row: int(row["pipe_index"]))
    segments = [
        [tuple(map(int, cell)) for cell in row["cells"]]
        for row in rows
    ]
    base = board_core.BaseBoard(
        n=int(meta["grid_n"]),
        segments=segments,
        target_index=int(meta["target_index"]),
        distractor_index=int(meta["distractor_index"]),
        tortuosity=float(meta.get("tortuosity", length_ladder.TORT)),
        path_turn_fraction=float(meta["path_turn_fraction"]),
        attempts=int(meta.get("generation_attempts", meta.get("pair_build_attempts", 1))),
        base_board_hash=str(meta["base_board_hash"]),
        recipe_origin=str(meta.get("recipe_origin", meta.get("layout_family", "production"))),
    )
    board_core.validate_base(base.n, base.segments)
    if board_core.canonical_board_hash(base.n, base.segments) != base.base_board_hash:
        raise ValueError("source board hash does not recompute")
    colors = [
        (str(row["color_name"]), tuple(map(int, row["color_rgb"])))
        for row in rows
    ]
    labeling = None
    if record["scope"] == "all":
        labeling = board_core.Labeling(
            [int(row["display_label"]) for row in rows],
            [0 if row["label_endpoint_side"] == "start" else 1 for row in rows],
        )
    appearance = board_render.Appearance(
        family=str(meta["appearance_family"]),
        background_rgb=tuple(map(int, meta["background_rgb"])),
        grid_rgb=tuple(map(int, meta["grid_rgb"])),
        texture_style=str(meta["texture_style"]),
        texture_amplitude=int(meta.get("texture_amplitude_rgb", 0)),
        texture_scale_px=int(meta.get("texture_scale_px", 0)),
        texture_seed=int(meta.get("texture_seed", 0)),
        pipe_style="orthogonal_solid",
    )
    return base, colors, labeling, appearance


def _true_port_pixels(base: board_core.BaseBoard, size: int) -> list[tuple[int, int]]:
    points = []
    for segment in base.segments:
        for first, second in zip(segment, segment[1:]):
            ax, ay = board_core.cell_center(first, base.n, size)  # noqa: SLF001
            bx, by = board_core.cell_center(second, base.n, size)  # noqa: SLF001
            points.append((int(round((ax + bx) / 2)), int(round((ay + by) / 2))))
    return points


def _add_annotations(
    image: Image.Image,
    record: dict[str, Any],
    base: board_core.BaseBoard,
    labeling: board_core.Labeling | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    before = image.copy()
    labels: list[dict[str, Any]] = []
    ring = None
    if record["scope"] == "all":
        if labeling is None:
            raise ValueError("all-pipe record lost its endpoint labels")
        labels = board_core.draw_labels(image, base, labeling, image.width)
    elif record["reference"] == "circle":
        ring = board_core.draw_ring(
            image,
            base.segments[base.target_index][0],
            base.n,
            image.width,
            dense_safe=True,
        )
    changed = sum(
        before.getpixel(point) != image.getpixel(point)
        for point in _true_port_pixels(base, image.width)
    )
    if changed:
        raise ValueError("endpoint annotations changed a true-port pixel")
    return labels, ring, changed


def _render_material(
    source: dict[str, Any],
    *,
    seed: int,
    split: str,
    index: int,
    spec: board_core.Spec | None,
    ood_mode: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    base, colors, labeling, appearance = _rebuild(source)
    if ood_mode is None:
        if spec is None:
            raise ValueError("ID material rendering requires its semantic spec")
        mode, styles, sampling = style_schedule.assign_id_styles(
            seed, split, index, spec, colors
        )
        domain = "id"
        support_hash = style_schedule.ID_SUPPORT_SHA256
    else:
        mode = ood_mode
        styles, sampling = style_schedule.assign_ood_styles(seed, index, mode, colors)
        domain = "ood_compositional"
        support_hash = style_schedule.OOD_SUPPORT_SHA256
    image, raw_qa = pipe_styles.render_pipe_styles(base, colors, appearance, styles, source["meta"]["image_size"])
    labels, ring, changed = _add_annotations(image, source, base, labeling)
    if labels != source["meta"]["label_annotations"]:
        raise ValueError("material renderer changed label geometry")
    if ring != source["meta"]["ring"]:
        raise ValueError("material renderer changed ring geometry")
    qa = {
        key: value
        for key, value in raw_qa.items()
        if key not in {"schema_version", "pipe_styles", "style"}
    }
    if (
        qa["route_mask_true_port_recall"] != 1.0
        or qa["route_mask_false_port_clearance"] != 1.0
        or qa["route_mask_pairwise_overlap_pixels"] != 0
        or qa["low_alpha_requires_source_color_core"]
    ):
        raise ValueError("material renderer failed topology or low-alpha QA")
    assignment = style_schedule.assignment_rows(
        colors, styles, raw_qa["pipe_width_px_by_pipe"]
    )
    return image.convert("RGB"), {
        "pipe_style": "orthogonal_material",
        "style_domain": domain,
        "style_assignment_mode": mode,
        "material_schedule_version": style_schedule.RNG_NAMESPACE,
        "material_renderer_version": pipe_styles.VERSION,
        "material_support_sha256": support_hash,
        "pipe_style_assignments": assignment,
        "material_sampling": sampling,
        "material_qa": qa,
        "annotation_true_port_pixels_changed": changed,
        "final_port_midpoint_recall": round(
            board_core.port_midpoint_recall(
                image, base.n, base.segments, [rgb for _, rgb in colors], image.width
            ),
            6,
        ),
        "compositing_order": [
            "background", "texture", "grid", "pipe_materials", "endpoint_annotations"
        ],
    }


def _normalize(
    source: dict[str, Any],
    image: Image.Image,
    *,
    seed: int,
    split: str,
    index: int,
    semantic_generator: str,
    source_pixel_sha256: str,
    extra_meta: dict[str, Any],
) -> tuple[dict[str, Any], Image.Image]:
    record = copy.deepcopy(source)
    image = image.convert("RGB")
    identifier = f"flow_free/{split}/s{seed}/{index:08d}"
    image_path = f"images/{index:08d}.png"
    record.update({
        "id": identifier,
        "source_id": identifier,
        "index": index,
        "split": split,
        "difficulty_rank": DIFFICULTY_RANK[record["tier"]],
        "image": image_path,
        "solution_image": None,
    })
    meta = record["meta"]
    meta.update({
        "schema_version": RNG_NAMESPACE,
        "semantic_generator_version": semantic_generator,
        "production_rng_namespace_version": RNG_NAMESPACE,
        "production_seed": seed,
        "global_index": index,
        "split": split,
        "dataset_role": ROLE_BY_SPLIT.get(split, "ood_eval"),
        "is_ood": split == "ood",
        "allowed_tiers": ["xxhard"] if split == "ood" else list(board_core.TRAIN_TIERS),
        "source_image_pixel_sha256": source_pixel_sha256,
        "image_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "image_size": image.width,
        **extra_meta,
    })
    return record, image


def _generate_id(seed: int, index: int, split: str, size: int) -> tuple[dict[str, Any], Image.Image]:
    role = ROLE_BY_SPLIT[split]
    source, canonical = board_core.generate_one(
        seed, index, size=size, dataset_role=role
    )
    board_core.validate_record(source, canonical)
    spec = board_core.schedule(index, role)
    image, material_meta = _render_material(
        source, seed=seed, split=split, index=index, spec=spec
    )
    record, image = _normalize(
        source,
        image,
        seed=seed,
        split=split,
        index=index,
        semantic_generator=board_core.VERSION,
        source_pixel_sha256=source["meta"]["image_pixel_sha256"],
        extra_meta={
            "ood_stratum": None,
            "ood_axes": [],
            **material_meta,
        },
    )
    return record, image


def _generate_ood(seed: int, index: int, total: int, size: int) -> tuple[dict[str, Any], Image.Image]:
    planned = ood_condition(seed, index, total)
    pair = ood_core.build_semantic_pair(planned.pair_seed, planned.source_case_index)
    source, source_image = ood_core.build_record(
        planned.pair_seed,
        planned.source_case_index,
        planned.source_arm,
        pair,
        size=size,
    )
    ood_core.validate_candidate(source, source_image)
    source_pixel_sha = source["meta"]["image_pixel_sha256"]
    if planned.stratum in MATERIAL_OOD_STRATA:
        mode = (
            "board_shared"
            if planned.stratum == "material_board_shared"
            else "per_pipe_independent"
        )
        image, render_meta = _render_material(
            source,
            seed=seed,
            split="ood",
            index=index,
            spec=None,
            ood_mode=mode,
        )
        axes = ["heldout_width_alpha_texture_composition"]
    else:
        image = source_image.convert("RGB")
        render_meta = {
            "style_domain": (
                "ood_curved_bundle"
                if planned.stratum in CURVED_OOD_STRATA
                else "id_renderer_program_shift"
            ),
            "style_assignment_mode": (
                "per_pipe_curve_grammar"
                if planned.stratum in CURVED_OOD_STRATA
                else "board_shared"
            ),
            "material_schedule_version": None,
            "material_support_sha256": None,
            "pipe_style_assignments": [],
            "material_sampling": None,
            "material_qa": None,
            "annotation_true_port_pixels_changed": 0,
        }
        axes = []
        if source["meta"]["path_program"] == "prim_macrocycle_odd_annex":
            axes.append("alternative_path_program")
        if planned.stratum in CURVED_OOD_STRATA:
            axes.extend(["rounded_sine_pipe_grammar", "patchwork_surface", "pictogram_underlay"])
    record, image = _normalize(
        source,
        image,
        seed=seed,
        split="ood",
        index=index,
        semantic_generator=ood_core.RNG_NAMESPACE,
        source_pixel_sha256=source_pixel_sha,
        extra_meta={
            "ood_stratum": planned.stratum,
            "ood_axes": axes,
            "ood_schedule_version": "flow_free_ood_5x4_supercell",
            "source_candidate_case_index": planned.source_case_index,
            "source_candidate_arm": planned.source_arm,
            "source_candidate_seed": planned.pair_seed,
            "source_candidate_id": source["id"],
            "official_ood_contains_unshifted_control": False,
            **render_meta,
        },
    )
    return record, image


def generate_one(
    seed: int,
    index: int,
    split: str,
    size: int = SIZE,
    total: int | None = None,
) -> tuple[dict[str, Any], Image.Image]:
    """Generate one formal record; output is a pure function of its coordinates."""
    if split not in SPLITS:
        raise ValueError(f"unknown production split: {split!r}")
    total = TOTALS[split] if total is None else total
    if size != SIZE:
        raise ValueError("Flow Free production is fixed at 448x448")
    if not 0 <= index < total:
        raise ValueError("index is outside the declared split total")
    if split == "ood":
        record, image = _generate_ood(seed, index, total, size)
    else:
        record, image = _generate_id(seed, index, split, size)
    validate_record(record, image, total=total)
    return record, image


def _record_spec(record: dict[str, Any]) -> board_core.Spec:
    meta = record["meta"]
    return board_core.Spec(
        index=int(record["index"]),
        block=int(meta.get("factor_block", meta.get("source_candidate_case_index", 0))),
        question_form=record["question_form"],
        scope=record["scope"],
        statistic=record["statistic"],
        layout_family=record["layout_family"],
        tier=record["tier"],
        order=record["order"],
        reference=record["reference"],
        dataset_role=str(meta["dataset_role"]),
        nuisance_row=int(meta.get("nuisance_row", 0)),
        n_variant=int(meta.get("n_variant", 0)),
    )


def _validate_semantics(record: dict[str, Any]) -> tuple[board_core.BaseBoard, list[tuple[str, tuple[int, int, int]]]]:
    meta = record["meta"]
    base, colors, labeling, _ = _rebuild(record)
    rows = sorted(meta["segments"], key=lambda row: int(row["pipe_index"]))
    lengths = [len(segment) for segment in base.segments]
    turn_counts = [board_core.turns(segment) for segment in base.segments]
    if lengths != meta["all_lengths"] or turn_counts != meta["all_turns"]:
        raise ValueError("stored length/turn vectors disagree with geometry")
    if sum(lengths) != base.n * base.n:
        raise ValueError("pipe lengths do not cover the board")
    for pipe_index, (row, length, turn_count) in enumerate(zip(rows, lengths, turn_counts)):
        if row.get("pipe_index") != pipe_index or row.get("pipe_id") != f"pipe{pipe_index}":
            raise ValueError("pipe row identity mismatch")
        if row.get("length") != length or row.get("turn_count") != turn_count:
            raise ValueError("pipe row statistic mismatch")
        if dict(pipe_colors.PIPE_COLORS).get(row.get("color_name")) != tuple(row.get("color_rgb", ())):
            raise ValueError("pipe colour name/RGB mismatch")
    full_path = [cell for segment in base.segments for cell in segment]
    if any(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
        for a, b in zip(full_path, full_path[1:])
    ):
        raise ValueError("segments no longer follow one Hamiltonian path")
    if round(backbite.turn_frac(full_path), 6) != float(meta["path_turn_fraction"]):
        raise ValueError("path turn fraction mismatch")
    spec = _record_spec(record)
    if board_core.answer_for(base, spec, labeling) != str(record["answer"]):
        raise ValueError("answer does not recompute from stored geometry")
    target, distractor = base.target_index, base.distractor_index
    names = [name for name, _ in colors]
    if not 0 <= target < len(names) or not 0 <= distractor < len(names):
        raise ValueError("target/distractor index outside pipe vector")
    if record["scope"] == "one" and record["reference"] == "color_name":
        if len(set(names)) != len(names) or names[target] not in record["question"]:
            raise ValueError("named-colour question has an ambiguous/missing target")
    elif names[target] != names[distractor]:
        raise ValueError("non-colour-name record lost its repeated-colour distractor")
    if record["scope"] == "all":
        if not meta.get("labels_present") or len(meta.get("label_to_segment", {})) != len(rows):
            raise ValueError("all-pipe record lost its endpoint labels")
    elif record["reference"] == "circle" and not meta.get("ring_present"):
        raise ValueError("circle-reference record lost its white ring")
    return base, colors


def _validate_material(record: dict[str, Any], colors: Sequence[tuple[str, tuple[int, int, int]]]) -> None:
    meta = record["meta"]
    assignments = meta.get("pipe_style_assignments")
    if not isinstance(assignments, list) or len(assignments) != len(colors):
        raise ValueError("material record has the wrong assignment count")
    for pipe_index, (row, (color_name, color_rgb)) in enumerate(
        zip(assignments, colors)
    ):
        if (
            row.get("pipe_index") != pipe_index
            or row.get("color_name") != color_name
            or tuple(row.get("color_rgb", ())) != color_rgb
        ):
            raise ValueError("material assignment identity disagrees with its pipe")
    styles = [pipe_styles.PipeStyle(**row["style"]) for row in assignments]
    signatures = [style_schedule.style_signature(style) for style in styles]
    for row, signature in zip(assignments, signatures):
        if tuple(row.get("knob_signature", ())) != signature:
            raise ValueError("stored material signature disagrees with style")
    by_color: dict[tuple[str, tuple[int, int, int]], set[tuple[Any, ...]]] = {}
    for color, signature in zip(colors, signatures):
        by_color.setdefault(color, set()).add(signature)
    if any(len(group) != 1 for group in by_color.values()):
        raise ValueError("same-colour pipes received different materials")
    domain = meta.get("style_domain")
    expected_domain = "ood_compositional" if record["split"] == "ood" else "id"
    if domain != expected_domain:
        raise ValueError("material style domain disagrees with split")
    expected_support_hash = (
        style_schedule.OOD_SUPPORT_SHA256
        if domain == "ood_compositional"
        else style_schedule.ID_SUPPORT_SHA256
    )
    if (
        meta.get("material_schedule_version") != style_schedule.RNG_NAMESPACE
        or meta.get("material_renderer_version") != pipe_styles.VERSION
        or meta.get("material_support_sha256") != expected_support_hash
    ):
        raise ValueError("material renderer/support provenance mismatch")
    allowed = {
        style_schedule.style_signature(style)
        for style in (
            style_schedule.OOD_STYLE_BANK if domain == "ood_compositional" else style_schedule.ID_STYLE_BANK
        )
    }
    if not set(signatures) <= allowed:
        raise ValueError("material assignment is outside its declared support")
    if domain == "ood_compositional" and not all(
        style_schedule.is_compositional_ood_values(
            style.width_fraction, style.opacity_percent, style.texture
        )
        for style in styles
    ):
        raise ValueError("OOD material contains an ID combination")
    qa = meta.get("material_qa") or {}
    if (
        qa.get("route_mask_true_port_recall") != 1.0
        or qa.get("route_mask_false_port_clearance") != 1.0
        or qa.get("route_mask_pairwise_overlap_pixels") != 0
        or qa.get("low_alpha_requires_source_color_core") is not False
        or meta.get("annotation_true_port_pixels_changed") != 0
    ):
        raise ValueError("material topology QA failed")


def validate_record(
    record: dict[str, Any],
    image: Image.Image | None = None,
    *,
    total: int | None = None,
) -> None:
    """Validate formal provenance, schedule, semantics, and renderer contract."""
    meta = record.get("meta") or {}
    split = record.get("split")
    if split not in SPLITS or meta.get("split") != split:
        raise ValueError("production split mismatch")
    total = TOTALS[split] if total is None else total
    index = int(record.get("index", -1))
    seed = int(meta.get("production_seed", -1))
    if not 0 <= index < total:
        raise ValueError("production index outside split")
    expected_id = f"flow_free/{split}/s{seed}/{index:08d}"
    if record.get("id") != expected_id or record.get("source_id") != expected_id:
        raise ValueError("production ID namespace mismatch")
    if record.get("image") != f"images/{index:08d}.png":
        raise ValueError("production image namespace mismatch")
    if pathlib.PurePosixPath(record["image"]).is_absolute() or ".." in pathlib.PurePosixPath(record["image"]).parts:
        raise ValueError("production image path is not portable")
    if meta.get("schema_version") != RNG_NAMESPACE:
        raise ValueError("production schema version mismatch")
    if meta.get("production_rng_namespace_version") != RNG_NAMESPACE:
        raise ValueError("production RNG namespace mismatch")
    for field in (
        "difficulty", "tier", "question_form", "scope", "statistic",
        "layout_family", "order", "reference",
    ):
        if record.get(field) != meta.get(field):
            raise ValueError(f"top-level/meta field mismatch: {field}")
    expected = condition(split, seed, index, total)
    if record["tier"] != expected["tier"] or record["question_form"] != expected["question_form"]:
        raise ValueError("record disagrees with the production schedule")
    if split != "ood":
        if meta.get("dataset_role") != ROLE_BY_SPLIT[split] or meta.get("is_ood"):
            raise ValueError("ID role/is_ood mismatch")
        if (
            meta.get("semantic_generator_version") != board_core.VERSION
            or meta.get("path_program") != "backbite_hamiltonian"
        ):
            raise ValueError("ID semantic generator provenance mismatch")
        if record["layout_family"] != expected["layout_family"]:
            raise ValueError("ID layout disagrees with schedule")
        if meta.get("style_assignment_mode") != expected["style_assignment_mode"]:
            raise ValueError("ID style scope disagrees with schedule")
        if meta.get("ood_stratum") is not None or meta.get("ood_axes") != []:
            raise ValueError("ID record carries OOD provenance")
    else:
        if record["tier"] != "xxhard" or not meta.get("is_ood"):
            raise ValueError("OOD role/tier mismatch")
        if meta.get("ood_stratum") != expected["ood_stratum"]:
            raise ValueError("OOD stratum disagrees with schedule")
        if meta.get("official_ood_contains_unshifted_control") is not False:
            raise ValueError("official OOD claims an unshifted control")
        planned = ood_condition(seed, index, total)
        if (
            meta.get("semantic_generator_version") != ood_core.RNG_NAMESPACE
            or meta.get("source_candidate_case_index") != planned.source_case_index
            or meta.get("source_candidate_arm") != planned.source_arm
            or meta.get("source_candidate_seed") != planned.pair_seed
            or meta.get("ood_schedule_version")
            != "flow_free_ood_5x4_supercell"
        ):
            raise ValueError("OOD source/schedule provenance mismatch")
    base, colors = _validate_semantics(record)
    if meta.get("pipe_style") == "orthogonal_material":
        _validate_material(record, colors)
        expected_mode = (
            expected["style_assignment_mode"]
            if split != "ood"
            else (
                "board_shared"
                if meta["ood_stratum"] == "material_board_shared"
                else "per_pipe_independent"
            )
        )
        if meta.get("style_assignment_mode") != expected_mode:
            raise ValueError("material assignment scope disagrees with stratum")
        if split == "ood" and (
            meta.get("path_program") != "backbite_hamiltonian"
            or meta.get("ood_axes")
            != ["heldout_width_alpha_texture_composition"]
        ):
            raise ValueError("material OOD stratum changed its declared axes")
    elif split != "ood":
        raise ValueError("ID production record lacks the material renderer")
    elif meta["ood_stratum"] in CURVED_OOD_STRATA:
        if meta.get("pipe_style") != ood_core.OOD_PIPE_STYLE:
            raise ValueError("curved OOD stratum lacks its curve grammar")
        if Counter(meta.get("per_pipe_styles", [])) != {
            style: 2 for style in ood_core.OOD_COMPONENT_STYLES
        }:
            raise ValueError("curved OOD grammar is not a 2/2 mix")
        if meta.get("same_color_curve_style_tied") is not True:
            raise ValueError("curved OOD grammar leaks same-colour identity")
        if meta.get("distractor_object_count") != 6 or not meta.get("distractor_objects_are_underlay"):
            raise ValueError("curved OOD stratum lost its pictogram underlay")
        expected_program = (
            "backbite_hamiltonian"
            if meta["ood_stratum"] == "curved_appearance_backbite"
            else "prim_macrocycle_odd_annex"
        )
        expected_axes = (
            []
            if expected_program == "backbite_hamiltonian"
            else ["alternative_path_program"]
        ) + [
            "rounded_sine_pipe_grammar", "patchwork_surface", "pictogram_underlay"
        ]
        if (
            meta.get("path_program") != expected_program
            or meta.get("style_domain") != "ood_curved_bundle"
            or meta.get("style_assignment_mode") != "per_pipe_curve_grammar"
            or meta.get("ood_axes") != expected_axes
        ):
            raise ValueError("curved OOD stratum changed its declared axes")
    elif meta["ood_stratum"] == "program_only":
        if meta.get("path_program") != "prim_macrocycle_odd_annex":
            raise ValueError("program-only stratum lacks the alternative program")
        if meta.get("pipe_style") != "orthogonal_solid":
            raise ValueError("program-only stratum changed the ID renderer")
        if (
            meta.get("style_domain") != "id_renderer_program_shift"
            or meta.get("style_assignment_mode") != "board_shared"
            or meta.get("ood_axes") != ["alternative_path_program"]
        ):
            raise ValueError("program-only stratum changed its declared axes")
    else:
        raise ValueError("unknown OOD renderer contract")
    if meta.get("pipe_style") == "orthogonal_material":
        # A deliberately translucent 85% sleeve need not fall within the
        # source-RGB tolerance used by the opaque renderer.  Continuity
        # is certified above on the exact route masks (including every true
        # and false port), rather than by pretending translucent pixels equal
        # their source colour.
        if float(meta.get("final_port_midpoint_recall", -1.0)) < 0.0:
            raise ValueError("material record lacks its diagnostic port recall")
    elif float(meta.get("final_port_midpoint_recall", 0.0)) != 1.0:
        raise ValueError("final image lost a true pipe port")
    if image is not None:
        if image.mode != "RGB" or image.size != (SIZE, SIZE):
            raise ValueError("formal image is not RGB 448x448")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest != meta.get("image_pixel_sha256"):
            raise ValueError("decoded image hash disagrees with metadata")
    if not str(record.get("question", "")).strip() or not str(record.get("answer", "")).strip():
        raise ValueError("question or answer is blank")


__all__ = [
    "OOD_TIER",
    "OOD_STRATA",
    "QUESTION_FORMS",
    "ROLE_BY_SPLIT",
    "SIZE",
    "SPLITS",
    "TOTALS",
    "TRAIN_TIERS",
    "RNG_NAMESPACE",
    "condition",
    "generate_one",
    "ood_condition",
    "validate_record",
]
