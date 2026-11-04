"""Five equally weighted Count Clusters question pools using base scenes and Fluent."""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

from functools import lru_cache
import random

from shared.iconlib.fluent import FluentLibrary

from . import fluent_scenes, overlap_mixed, scene_ink_budget
from .base import scenes
from .difficulty import COUNT_RANGES, OVERLAP_TARGET_RANGES, TARGET_RANGES, TIERS

VERSION = "count_clusters_mixed"
POOLS = ("total", "attribute", "region", "overlap", "category")
ATTRIBUTES = ("shape", "color", "conjunction", "striped")
DEFAULT_MANIFEST = (_ROOT / 'shared/iconlib/fluent_manifest.json')


def _cycle(values, index, seed, label):
    block, offset = divmod(index, len(values))
    order = list(values)
    random.Random(f"{seed}:count-mixed:{label}:{block}").shuffle(order)
    return order[offset]


def spec_for_index(index, *, seed, difficulty):
    """Every aligned 20-per-tier block covers every pool and attribute equally."""
    if index < 0:
        raise ValueError("index must be nonnegative")
    if difficulty not in TIERS:
        raise ValueError("difficulty must be easy, medium, hard, xhard or xxhard")
    tier = difficulty
    selected = _cycle(POOLS, index, seed, "pool")
    occurrence = index // len(POOLS)
    spec = {"pool": selected, "difficulty": tier, "occurrence": occurrence,
            "count_range": list(COUNT_RANGES[tier]),
            "target_range": list(TARGET_RANGES[tier]), "is_ood": False}
    if selected == "attribute":
        kind = _cycle(ATTRIBUTES, occurrence, seed, f"attribute:{tier}")
        question_type = {"shape": "shape_subset", "color": "color_subset",
                         "conjunction": "shape_color_subset", "striped": "striped_subset"}[kind]
        spec.update(attribute_kind=kind, question_type=question_type,
                    material="sticks" if kind == "striped" else "geometry")
    elif selected == "region":
        spec.update(question_type="section_count", material="sticks")
    elif selected == "overlap":
        spec.update(question_type="overlap_any", material="mixed")
    elif selected == "category":
        spec.update(question_type="category_subset", material="fluent")
    else:
        spec.update(question_type="total",
                    material=_cycle(("fluent", "geometry", "sticks"), occurrence, seed, f"total-material:{tier}"))
    method = "standard"
    if tier == "xxhard":
        if selected == "attribute":
            # Two methods per four-attribute block; reverse their assignment
            # in the next block so each predicate also sees both methods.
            order = [_cycle(ATTRIBUTES, i, seed, "xxhard-attribute-method") for i in range(4)]
            method = ("standard", "extended")[(order.index(spec["attribute_kind"]) // 2 + occurrence // 4) % 2]
        else:
            method = _cycle(("standard", "extended"), occurrence, seed, f"xxhard-method:{selected}:all")
    profile = "ood" if method == "extended" else tier
    spec.update(profile=profile, generation_method=method,
                generation_profile=f"xxhard_{method}" if tier == "xxhard" else tier,
                count_range=list(COUNT_RANGES[profile]), target_range=list(TARGET_RANGES[profile]))
    if selected == "overlap":
        lo, hi = OVERLAP_TARGET_RANGES[profile]
        spec.update(target_range=[lo, hi],
                    answer_target=_cycle(tuple(range(lo, hi + 1)), occurrence // 2 if tier == "xxhard" else occurrence,
                                         seed, f"overlap-answer:{profile}"))
    return spec


@lru_cache(maxsize=4)
def _library(manifest_path):
    return FluentLibrary(manifest_path)


def _scene_problem(rng, size, spec):
    problem = scene_ink_budget.make_scene_problem(rng, size, {**spec, "difficulty": spec["profile"]})
    image, question, answer, cot, meta, elements = problem
    if spec["pool"] == "region":
        meta.setdefault("grid_rows", 3)
        meta.setdefault("grid_cols", 3)
        if not meta.get("target_sections"):
            meta["target_sections"] = [meta.get("target_section", scenes.section_of(elements[0]["cx"], elements[0]["cy"], size))]
        if len(meta["target_sections"]) == 1:
            meta["target_section"] = meta["target_sections"][0]
    instances = []
    raw = meta.get("sticks", meta.get("placements", []))
    def section_of_recorded_center(item):
        rows, cols = meta["grid_rows"], meta["grid_cols"]
        cx, cy = item["centroid"]
        return min(rows - 1, int(cy * rows / size)) * cols + min(cols - 1, int(cx * cols / size)) + 1
    if spec["pool"] == "region" and any(section_of_recorded_center(item) != item["section"] for item in raw):
        raise RuntimeError("main centroid rounding crossed a region boundary")
    for number, item in enumerate(raw):
        info = item.get("info", {})
        instances.append({
            "instance_id": number, "centroid": item["centroid"],
            "class_id": info.get("shape", "line"), "material": spec["material"],
            "shape": info.get("shape"), "color": info.get("color", item.get("color")),
            "style": item.get("style", "geometry"), "section": item.get("section"),
            "info": info,
            "foreground_area_px": item.get("foreground_area_px", info.get("foreground_area_px")),
            "foreground_size_px": item.get("foreground_size_px", info.get("foreground_size_px")),
            "foreground_bbox_px": item.get("foreground_bbox_px"),
            "width_px": item.get("width_px"), "length_px": item.get("length_px"),
        })
    target = meta.get("subset_target")
    for item in instances:
        if spec["question_type"] == "shape_subset":
            flag = item["shape"] == target
        elif spec["question_type"] == "color_subset":
            flag = item["color"] == target
        elif spec["question_type"] == "shape_color_subset":
            flag = item["shape"] == target["shape"] and item["color"] == target["color"]
        elif spec["question_type"] == "striped_subset":
            flag = item["style"] == "candy"
        elif spec["question_type"] == "section_count":
            flag = item["section"] in meta["target_sections"]
        else:
            flag = True
        item["is_target"] = bool(flag)
    meta["instances"] = instances
    meta["source_implementation"] = "scenes_with_size_adapter"
    return image, question, answer, cot, meta, elements


def _normalize(problem, spec, split, index, seed, attempt, size):
    image, question, answer, cot, meta, elements = problem
    if "instances" not in meta:
        raise ValueError("generator did not return instance annotations")
    for index_in_scene, instance in enumerate(meta["instances"]):
        instance.setdefault("instance_id", index_in_scene)
        if "centroid" not in instance:
            if "center" in instance:
                instance["centroid"] = list(instance["center"])
            else:
                instance["centroid"] = [instance["cx"], instance["cy"]]
        if "is_target" not in instance:
            instance["is_target"] = bool(instance.get("counted", instance.get("target", False)))
        if instance.get("identity_id", "").startswith("fluent/"):
            if instance.get("split") != split:
                raise ValueError("Fluent target or distractor came from the wrong split")
    targets = [item["instance_id"] for item in meta["instances"] if item["is_target"]]
    if len(targets) != int(answer):
        raise ValueError(f"instance target count {len(targets)} differs from answer {answer}")
    if len(elements) != int(answer):
        raise ValueError("solution annotations differ from the answer")
    n = len(meta["instances"])
    if not spec["count_range"][0] <= n <= spec["count_range"][1]:
        raise ValueError(f"realized count {n} is outside the requested {spec['difficulty']} tier")
    if spec["pool"] != "total" and not spec["target_range"][0] <= int(answer) <= spec["target_range"][1]:
        raise ValueError(f"answer {answer} is outside requested {spec['difficulty']} target range")
    if spec["generation_method"] == "extended" and not meta.get("ood_axes"):
        raise ValueError("extended xxhard scene must declare its construction axes")
    axes = meta.pop("ood_axes", [])
    meta["extended_axes"] = ["higher_count_range" if axis in ("heldout_count_range", "count_extrapolation") else axis
                             for axis in axes]
    if meta.get("source_tier") == "ood":
        meta["source_tier"] = "xxhard_extended"
    controls = meta.get("difficulty_controls", {})
    if "ood_holdout" in controls:
        controls["extended_topology"] = controls.pop("ood_holdout")
    if isinstance(controls.get("layout"), str):
        controls["layout"] = controls["layout"].removeprefix("heldout_")
    if image.size != (size, size):
        raise ValueError("unexpected image dimensions")
    if meta.get("target_class") is not None:
        meta["target_class_id"] = meta["target_class"]
    meta.update(question_pool=spec["pool"], question_type=spec["question_type"],
                difficulty=spec["difficulty"], asset_split=split, target_ids=targets,
                scene_seed=f"{seed}:count-mixed:{split}:{index}:{attempt}",
                generation_attempt=attempt + 1, canvas_size=[size, size],
                n_total=n, generator_version=VERSION, is_ood=spec["is_ood"],
                generation_method=spec["generation_method"], generation_profile=spec["generation_profile"],
                difficulty_count_range=spec["count_range"], difficulty_target_range=spec["target_range"])
    meta.setdefault("source_implementation", "fluent_main_scatter" if spec["material"] == "fluent" else "branch_overlap")
    normalized_elements = []
    for item in elements:
        center = item.get("centroid", item.get("center"))
        normalized_elements.append({"cx": item["cx"] if "cx" in item else center[0],
                                    "cy": item["cy"] if "cy" in item else center[1]})
    return image, question, str(answer), cot, meta, normalized_elements


def make_problem(index, *, seed, split, size, difficulty):
    if split not in ("train", "eval"):
        raise ValueError("this experiment uses train or the combined eval pool")
    if size < 320:
        raise ValueError("mixed hard scenes require size >= 320")
    spec = spec_for_index(index, seed=seed, difficulty=difficulty)
    lib = _library(str(DEFAULT_MANIFEST)) if spec["material"] in ("fluent", "mixed") else None
    target_class = None
    if spec["pool"] == "category":
        target_rng = random.Random(f"{seed}:count-mixed:{split}:{index}:target")
        target_class = target_rng.choice(lib.classes(split))
    last_error = None
    for attempt in range(6):
        rng = random.Random(f"{seed}:count-mixed:{split}:{index}:{attempt}")
        try:
            if spec["pool"] == "overlap":
                problem = overlap_mixed.make_overlap_problem(rng, size, spec["profile"], split, lib,
                                                             answer_target=spec["answer_target"])
            elif spec["material"] == "fluent":
                problem = fluent_scenes.make_fluent_problem(rng, size, spec["profile"], split, lib,
                                                            query=spec["pool"], target_class=target_class)
            else:
                problem = _scene_problem(rng, size, spec)
        except RuntimeError as exc:
            last_error = exc
            continue
        return _normalize(problem, spec, split, index, seed, attempt, size)
    raise RuntimeError(f"could not generate index {index}, fixed spec {spec}: {last_error}")

